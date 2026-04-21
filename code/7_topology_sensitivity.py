"""
7_topology_sensitivity.py
Topology Analysis + NoveltyScore Sensitivity.

This script validates the robustness of NoveltyScore weights α/β/γ and
threshold θ=0.7, and uses full PageRank for the IP component.  It:
  1. Rebuilds the KG graph with entity merging (d<0.03, 3 rounds)
  2. Computes full topology metrics (betweenness, PageRank, cross-layer matrix)
  3. Mines cross-layer paths (2-5 hops, ~8000 paths)
  4. Computes NoveltyScore with corrected IP (full PageRank, not 0.5)
  5. Sweeps α/β/γ weight combinations to prove robustness

Pipeline position:
    Input:  data/processed/kg/*.json                (script 1, 5738 files)
            data/processed/entity_merge_eval/       (script 6 cached embeddings)
    Output: data/processed/topology_sensitivity/

Usage:
    cd workspace/code
    python 7_topology_sensitivity.py                  # full run, ~1.5-2.5 h
    python 7_topology_sensitivity.py --phase 2        # single phase only
    python 7_topology_sensitivity.py --phase 5        # sensitivity scan only
"""

import argparse
import csv
import json
import logging
import os
import pickle
import random
import time
import multiprocessing as mp
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import faiss
import networkx as nx
import numpy as np
from scipy.stats import spearmanr, kendalltau, chi2

# ============================================================
# Paths
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent
KG_DIR = BASE_DIR / "data" / "processed" / "kg"
EMBED_DIR = BASE_DIR / "data" / "processed" / "entity_merge_eval"
OUT_DIR = BASE_DIR / "data" / "processed" / "topology_sensitivity"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Logging
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(OUT_DIR / "7_topology_sensitivity.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ============================================================
# Constants
# ============================================================
LAYERS = ["physical", "biological", "social", "economic"]
CONFIDENCE_MAP = {"high": 1.0, "medium": 0.6, "low": 0.2}
MERGE_THRESHOLD = 0.03
MERGE_ROUNDS = 3
TOP_K = 500
BC_WORKERS = 60          # parallel workers for betweenness centrality
PATH_WORKERS = 16        # parallel workers for path mining
TARGET_PATHS = 10000     # target number of cross-layer paths
DEFAULT_ALPHA = 0.5
DEFAULT_BETA = 0.3
DEFAULT_GAMMA = 0.2
NOVELTY_THETA = 0.7
N_PERMUTATIONS = 10000   # permutation test iterations
FDR_ALPHA = 0.05         # Benjamini-Hochberg FDR threshold
LAYER_SEVERITY = {
    "economic": 1.0,
    "social": 0.8,
    "biological": 0.7,
    "physical": 0.6,
    "unknown": 0.5,
}


# ============================================================
# Utility — greedy clustering (from script 6)
# ============================================================
def _greedy_cluster(entities: List[str], vectors: np.ndarray,
                    distances: np.ndarray, indices: np.ndarray,
                    threshold: float) -> Dict[str, str]:
    """Single round of greedy clustering.  Returns entity -> canonical mapping."""
    n = len(entities)
    cosine_dist = 1.0 - distances  # inner-product -> cosine distance
    processed: Set[int] = set()
    groups: List[List[int]] = []

    for i in range(n):
        if i in processed:
            continue
        group = [i]
        processed.add(i)
        for j_idx in range(len(indices[i])):
            j = int(indices[i][j_idx])
            if j == i or j in processed or j >= n:
                continue
            if cosine_dist[i][j_idx] < threshold:
                group.append(j)
                processed.add(j)
        if len(group) >= 2:
            groups.append(group)

    mapping = {}
    for group in groups:
        canonical = entities[group[0]]
        for idx in group:
            mapping[entities[idx]] = canonical
    return mapping


def _compute_group_centroids(entities: List[str], vectors: np.ndarray,
                             mapping: Dict[str, str]) -> Tuple[List[str], np.ndarray]:
    """Compute L2-normalised centroids for each canonical group."""
    groups: Dict[str, List[int]] = {}
    entity_to_idx = {e: i for i, e in enumerate(entities)}
    for e, canon in mapping.items():
        if e in entity_to_idx:
            groups.setdefault(canon, []).append(entity_to_idx[e])
    for i, e in enumerate(entities):
        if e not in mapping:
            groups.setdefault(e, []).append(i)

    canon_names = sorted(groups.keys())
    centroids = []
    for name in canon_names:
        idxs = groups[name]
        centroid = vectors[idxs].mean(axis=0)
        centroids.append(centroid)
    centroids = np.array(centroids, dtype="float32")
    faiss.normalize_L2(centroids)
    return canon_names, centroids


# ============================================================
# Utility — parallel betweenness (from script 6)
# ============================================================
def _bc_worker(edges, nodes, subset, directed):
    """Compute betweenness centrality from a subset of source nodes."""
    G = nx.DiGraph() if directed else nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    bc = nx.betweenness_centrality_subset(G, sources=subset, targets=nodes, normalized=True)
    return bc


def _parallel_betweenness(G: nx.DiGraph, n_workers: int) -> Dict[str, float]:
    """Exact betweenness centrality using multiprocessing."""
    nodes = list(G.nodes())
    n = len(nodes)

    if n < 500 or n_workers <= 1:
        return nx.betweenness_centrality(G)

    chunk_size = max(1, n // n_workers)
    chunks = [nodes[i:i + chunk_size] for i in range(0, n, chunk_size)]

    edges = list(G.edges())
    directed = G.is_directed()

    bc_total = dict.fromkeys(nodes, 0.0)
    with ProcessPoolExecutor(max_workers=min(n_workers, len(chunks))) as executor:
        futures = {executor.submit(_bc_worker, edges, nodes, chunk, directed): i
                   for i, chunk in enumerate(chunks)}
        done_count = 0
        for f in as_completed(futures):
            partial_bc = f.result()
            for node, val in partial_bc.items():
                bc_total[node] += val
            done_count += 1
            if done_count % 10 == 0 or done_count == len(futures):
                logger.info("    betweenness: %d/%d chunks done", done_count, len(futures))

    return bc_total


# ============================================================
# Phase 1 — Load KG + entity merge + build graph (no API)
# ============================================================
def phase1() -> Tuple[nx.DiGraph, Dict[str, str], List[dict], Dict[str, str]]:
    """
    Load KG triples, apply entity merging (d<0.03, 3 rounds), build DiGraph.

    Returns: (G, entity_layer, all_triples, merge_mapping)
    """
    ckpt_graph = OUT_DIR / "phase1_graph.pkl"
    ckpt_merge = OUT_DIR / "phase1_merge_mapping.json"

    if ckpt_graph.exists() and ckpt_merge.exists():
        logger.info("Phase 1: loading from checkpoint ...")
        with open(ckpt_graph, "rb") as f:
            saved = pickle.load(f)
        G = saved["G"]
        entity_layer = saved["entity_layer"]
        all_triples = saved["all_triples"]
        with open(ckpt_merge, "r", encoding="utf-8") as f:
            merge_mapping = json.load(f)
        logger.info("Phase 1: restored — %d nodes, %d edges, %d triples",
                     G.number_of_nodes(), G.number_of_edges(), len(all_triples))
        return G, entity_layer, all_triples, merge_mapping

    # --- 1a: Load triples ---
    logger.info("Phase 1a: loading KG triples from %s ...", KG_DIR)
    entity_layer_counter: Dict[str, Counter] = {}
    all_triples: List[dict] = []
    file_count = 0

    for jf in sorted(KG_DIR.glob("*.json")):
        try:
            with open(jf, "r", encoding="utf-8") as f:
                triples = json.load(f)
        except Exception:
            continue
        if not isinstance(triples, list):
            continue
        for t in triples:
            sn = t.get("start_node", "").strip()
            en = t.get("end_node", "").strip()
            sl = t.get("start_layer", "").strip().lower()
            el = t.get("end_layer", "").strip().lower()
            if not sn or not en:
                continue
            # confidence string → float
            conf_str = t.get("confidence", "medium")
            if isinstance(conf_str, str):
                t["confidence_float"] = CONFIDENCE_MAP.get(conf_str.lower(), 0.5)
            else:
                t["confidence_float"] = float(conf_str) if conf_str else 0.5
            all_triples.append(t)
            for node, layer in [(sn, sl), (en, el)]:
                if node not in entity_layer_counter:
                    entity_layer_counter[node] = Counter()
                if layer in LAYERS:
                    entity_layer_counter[node][layer] += 1
        file_count += 1

    entities_all = sorted(entity_layer_counter.keys())
    entity_layer: Dict[str, str] = {}
    for e, cnt in entity_layer_counter.items():
        entity_layer[e] = cnt.most_common(1)[0][0] if cnt else "unknown"

    logger.info("Phase 1a: %d files, %d triples, %d unique entities",
                file_count, len(all_triples), len(entities_all))

    # --- 1b: Load cached embeddings + FAISS results ---
    embed_path = EMBED_DIR / "embeddings.npy"
    faiss_path = EMBED_DIR / "faiss_results.npz"
    entities_path = EMBED_DIR / "entities.json"

    if not embed_path.exists() or not faiss_path.exists() or not entities_path.exists():
        logger.error("Missing entity_merge_eval files — run script 6 first!")
        raise FileNotFoundError("entity_merge_eval files not found")

    with open(entities_path, "r", encoding="utf-8") as f:
        ent_data = json.load(f)
    embed_entities = ent_data["entities"]  # ordered entity list matching embeddings.npy

    vectors = np.load(embed_path).astype("float32")
    faiss_data = np.load(faiss_path)
    distances = faiss_data["distances"]
    indices = faiss_data["indices"]
    logger.info("Phase 1b: loaded %d embeddings (%d dims), FAISS top-%d",
                vectors.shape[0], vectors.shape[1], indices.shape[1])

    # --- 1c: Greedy clustering 3 rounds at d<0.03 ---
    logger.info("Phase 1c: running greedy merge (d<%.2f, %d rounds) ...",
                MERGE_THRESHOLD, MERGE_ROUNDS)
    cumulative_mapping: Dict[str, str] = {}
    cur_entities = embed_entities
    cur_vectors = vectors.copy()
    faiss.normalize_L2(cur_vectors)
    cur_distances = distances
    cur_indices = indices

    for rd in range(1, MERGE_ROUNDS + 1):
        if rd > 1:
            canon_names, centroids = _compute_group_centroids(embed_entities, vectors, cumulative_mapping)
            cur_entities = canon_names
            cur_vectors = centroids
            idx = faiss.IndexFlatIP(centroids.shape[1])
            idx.add(centroids)
            k = min(TOP_K, len(canon_names))
            cur_distances, cur_indices = idx.search(centroids, k)

        rd_mapping = _greedy_cluster(cur_entities, cur_vectors,
                                     cur_distances, cur_indices, MERGE_THRESHOLD)
        new_merges = len(rd_mapping)

        if rd == 1:
            cumulative_mapping.update(rd_mapping)
        else:
            for e in list(cumulative_mapping.keys()):
                old_canon = cumulative_mapping[e]
                if old_canon in rd_mapping:
                    cumulative_mapping[e] = rd_mapping[old_canon]
            for e, canon in rd_mapping.items():
                if e not in cumulative_mapping:
                    cumulative_mapping[e] = canon
        logger.info("  Round %d: %d new merges", rd, new_merges)

    merge_mapping = cumulative_mapping
    logger.info("Phase 1c: total merged entities = %d", len(merge_mapping))

    # --- 1d: Build graph ---
    logger.info("Phase 1d: building nx.DiGraph ...")

    G = nx.DiGraph()
    edge_confidence: Dict[Tuple[str, str], float] = {}

    for t in all_triples:
        sn = merge_mapping.get(t.get("start_node", "").strip(),
                                t.get("start_node", "").strip())
        en = merge_mapping.get(t.get("end_node", "").strip(),
                                t.get("end_node", "").strip())
        sl = t.get("start_layer", "").strip().lower()
        el = t.get("end_layer", "").strip().lower()
        rel = t.get("relationship", "")
        conf = t.get("confidence_float", 0.5)

        if not sn or not en or sn == en:
            continue

        edge_key = (sn, en)
        # Multi-edge: keep max confidence
        if edge_key in edge_confidence:
            if conf > edge_confidence[edge_key]:
                edge_confidence[edge_key] = conf
                G[sn][en]["confidence"] = conf
                G[sn][en]["relationship"] = rel
                G[sn][en]["start_layer"] = sl
                G[sn][en]["end_layer"] = el
        else:
            edge_confidence[edge_key] = conf
            G.add_edge(sn, en, relationship=rel, start_layer=sl,
                       end_layer=el, confidence=conf)

    # Update entity_layer for merged entities
    merged_entity_layer: Dict[str, str] = {}
    for node in G.nodes():
        if node in entity_layer:
            merged_entity_layer[node] = entity_layer[node]
        else:
            # Try to find via reverse mapping
            merged_entity_layer[node] = "unknown"
    # Also check edges for layer info
    for u, v, data in G.edges(data=True):
        sl = data.get("start_layer", "")
        el = data.get("end_layer", "")
        if sl in LAYERS and merged_entity_layer.get(u) == "unknown":
            merged_entity_layer[u] = sl
        if el in LAYERS and merged_entity_layer.get(v) == "unknown":
            merged_entity_layer[v] = el

    entity_layer = merged_entity_layer

    logger.info("Phase 1d: graph built — %d nodes, %d edges",
                G.number_of_nodes(), G.number_of_edges())

    # --- Save checkpoint ---
    with open(ckpt_graph, "wb") as f:
        pickle.dump({"G": G, "entity_layer": entity_layer, "all_triples": all_triples}, f)
    with open(ckpt_merge, "w", encoding="utf-8") as f:
        json.dump(merge_mapping, f, ensure_ascii=False)
    logger.info("Phase 1: checkpoints saved")

    return G, entity_layer, all_triples, merge_mapping


# ============================================================
# Phase 2 — Topology analysis (pure local computation)
# ============================================================
def phase2(G: nx.DiGraph, entity_layer: Dict[str, str]) -> Dict:
    """Compute comprehensive topology metrics."""
    ckpt_bc = OUT_DIR / "phase2_bc.pkl"
    out_path = OUT_DIR / "topology_results.json"

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    # --- Degree distribution ---
    degrees = [d for _, d in G.degree()]
    in_degrees = [d for _, d in G.in_degree()]
    out_degrees = [d for _, d in G.out_degree()]
    degree_stats = {
        "mean": float(np.mean(degrees)) if degrees else 0,
        "median": float(np.median(degrees)) if degrees else 0,
        "max": int(np.max(degrees)) if degrees else 0,
        "std": float(np.std(degrees)) if degrees else 0,
        "in_degree_mean": float(np.mean(in_degrees)) if in_degrees else 0,
        "out_degree_mean": float(np.mean(out_degrees)) if out_degrees else 0,
    }

    # --- Betweenness centrality Top-20 ---
    if ckpt_bc.exists():
        logger.info("Phase 2: loading betweenness from checkpoint ...")
        with open(ckpt_bc, "rb") as f:
            bc = pickle.load(f)
    else:
        logger.info("Phase 2: computing betweenness centrality (%d nodes, %d workers) ...",
                     n_nodes, BC_WORKERS)
        bc = _parallel_betweenness(G, BC_WORKERS)
        with open(ckpt_bc, "wb") as f:
            pickle.dump(bc, f)
        logger.info("Phase 2: betweenness checkpoint saved")

    bc_sorted = sorted(bc.items(), key=lambda x: x[1], reverse=True)
    bc_top20 = [{"entity": node, "bc": score, "layer": entity_layer.get(node, "unknown")}
                for node, score in bc_sorted[:20]]

    # --- PageRank Top-20 ---
    logger.info("Phase 2: computing PageRank ...")
    pagerank = nx.pagerank(G, alpha=0.85)
    pr_sorted = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
    pr_top20 = [{"entity": node, "pagerank": score, "layer": entity_layer.get(node, "unknown")}
                for node, score in pr_sorted[:20]]

    # --- 4x4 cross-layer transition matrix ---
    layer_matrix = {l1: {l2: 0 for l2 in LAYERS} for l1 in LAYERS}
    total_cross = 0
    for u, v, data in G.edges(data=True):
        sl = data.get("start_layer", "")
        el = data.get("end_layer", "")
        if sl in LAYERS and el in LAYERS:
            layer_matrix[sl][el] += 1
            if sl != el:
                total_cross += 1

    # --- Bio-mediation ratio ---
    phys_bio = layer_matrix.get("physical", {}).get("biological", 0)
    bio_econ = layer_matrix.get("biological", {}).get("economic", 0)
    phys_econ = layer_matrix.get("physical", {}).get("economic", 0)
    bio_med_ratio = (phys_bio + bio_econ) / phys_econ if phys_econ > 0 else float("inf")
    cross_layer_pct = total_cross / n_edges if n_edges > 0 else 0

    # --- Community detection ---
    logger.info("Phase 2: community detection ...")
    G_undirected = G.to_undirected()
    try:
        communities = list(nx.community.greedy_modularity_communities(G_undirected))
        n_communities = len(communities)
        modularity = nx.community.modularity(G_undirected, communities)
        # Layer composition of top 5 communities
        community_layers = []
        for i, comm in enumerate(sorted(communities, key=len, reverse=True)[:5]):
            layer_count = Counter(entity_layer.get(n, "unknown") for n in comm)
            community_layers.append({
                "community_id": i,
                "size": len(comm),
                "layers": dict(layer_count),
            })
    except Exception as e:
        logger.warning("Community detection failed: %s", e)
        n_communities = 0
        modularity = 0.0
        community_layers = []

    # --- Bridge nodes (connect >=3 layers) ---
    bridge_nodes = []
    for node in G.nodes():
        connected_layers = set()
        for _, v, data in G.out_edges(node, data=True):
            el = data.get("end_layer", "")
            if el in LAYERS:
                connected_layers.add(el)
        for u, _, data in G.in_edges(node, data=True):
            sl = data.get("start_layer", "")
            if sl in LAYERS:
                connected_layers.add(sl)
        node_layer = entity_layer.get(node, "unknown")
        if node_layer in LAYERS:
            connected_layers.add(node_layer)
        if len(connected_layers) >= 3:
            bridge_nodes.append({
                "entity": node,
                "connected_layers": sorted(connected_layers),
                "degree": G.degree(node),
                "layer": node_layer,
            })
    bridge_nodes.sort(key=lambda x: x["degree"], reverse=True)

    # --- Layer distribution of nodes ---
    layer_node_count = Counter(entity_layer.get(n, "unknown") for n in G.nodes())

    # --- Per-layer betweenness top-5 (for inset chart in Bridge_Nodes figure) ---
    bc_by_layer = defaultdict(list)
    for node, score in bc.items():
        layer = entity_layer.get(node, "unknown")
        bc_by_layer[layer].append({"entity": node, "bc": score, "layer": layer})
    betweenness_top5_per_layer = {}
    for layer in LAYERS:
        layer_list = sorted(bc_by_layer.get(layer, []),
                            key=lambda x: x["bc"], reverse=True)
        betweenness_top5_per_layer[layer] = layer_list[:5]

    results = {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "degree_stats": degree_stats,
        "betweenness_top20": bc_top20,
        "pagerank_top20": pr_top20,
        "layer_matrix": layer_matrix,
        "phys_bio": phys_bio,
        "bio_econ": bio_econ,
        "phys_econ": phys_econ,
        "bio_mediation_ratio": bio_med_ratio,
        "cross_layer_pct": cross_layer_pct,
        "n_communities": n_communities,
        "modularity": modularity,
        "top5_communities": community_layers,
        "bridge_nodes_count": len(bridge_nodes),
        "bridge_nodes_top20": bridge_nodes[:20],
        "betweenness_top5_per_layer": betweenness_top5_per_layer,
        "layer_node_distribution": dict(layer_node_count),
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Phase 2 done: bio-mediation=%.3f, communities=%d, bridges=%d",
                bio_med_ratio, n_communities, len(bridge_nodes))

    # Return pagerank + bc for later phases
    results["_pagerank"] = pagerank
    results["_bc"] = bc
    return results


# ============================================================
# Phase 3 — Path mining (pure local, multi-process)
# ============================================================
MAX_PATHS_PER_SOURCE = 50   # cap per source node to prevent combinatorial explosion
MAX_PATHS_PER_WORKER = 5000  # hard cap per worker


def _bfs_paths_worker(args):
    """BFS worker: find cross-layer paths from a set of source nodes."""
    edges_data, nodes_data, entity_layer_local, sources, min_hops, max_hops = args

    # Rebuild local graph
    G = nx.DiGraph()
    for u, v, d in edges_data:
        G.add_edge(u, v, **d)

    found_paths = []
    for src in sources:
        if len(found_paths) >= MAX_PATHS_PER_WORKER:
            break

        src_layer = entity_layer_local.get(src, "unknown")
        if src_layer not in LAYERS:
            continue

        # BFS with per-source cap
        queue = [(src, [src])]
        src_count = 0

        while queue and src_count < MAX_PATHS_PER_SOURCE:
            current, path = queue.pop(0)
            if len(path) > max_hops + 1:
                continue

            if len(path) >= min_hops + 1:
                # Check if cross-layer
                path_layers = [entity_layer_local.get(n, "unknown") for n in path]
                unique_layers = set(l for l in path_layers if l in LAYERS)
                if len(unique_layers) >= 2:
                    # Collect edge info
                    relationships = []
                    confidences = []
                    for i in range(len(path) - 1):
                        ed = G.get_edge_data(path[i], path[i + 1])
                        if ed:
                            relationships.append(ed.get("relationship", ""))
                            confidences.append(ed.get("confidence", 0.5))

                    found_paths.append({
                        "nodes": path,
                        "layers": path_layers,
                        "relationships": relationships,
                        "confidences": confidences,
                        "hops": len(path) - 1,
                        "n_layers": len(unique_layers),
                    })
                    src_count += 1

            if len(path) < max_hops + 1 and src_count < MAX_PATHS_PER_SOURCE:
                for neighbor in G.successors(current):
                    if neighbor not in path:  # avoid cycles
                        queue.append((neighbor, path + [neighbor]))

    return found_paths


def phase3(G: nx.DiGraph, entity_layer: Dict[str, str]) -> List[dict]:
    """Mine cross-layer paths (2-5 hops) using parallel BFS."""
    out_path = OUT_DIR / "phase3_paths.json"

    target = TARGET_PATHS
    min_hops = 2
    max_hops = 5

    logger.info("Phase 3: mining cross-layer paths (%d-%d hops, target=%d) ...",
                min_hops, max_hops, target)

    # Prepare edge data for serialization to workers
    edges_data = [(u, v, dict(d)) for u, v, d in G.edges(data=True)]
    nodes_list = list(G.nodes())

    # Stratified source selection by layer pair
    layer_pairs = []
    for l1 in LAYERS:
        for l2 in LAYERS:
            if l1 != l2:
                layer_pairs.append((l1, l2))

    # Collect source nodes per layer
    layer_nodes: Dict[str, List[str]] = defaultdict(list)
    for node in nodes_list:
        nl = entity_layer.get(node, "unknown")
        if nl in LAYERS:
            layer_nodes[nl].append(node)

    # Distribute sources across workers
    all_sources = []
    for layer_name, layer_node_list in layer_nodes.items():
        all_sources.extend(layer_node_list)
    random.seed(42)
    random.shuffle(all_sources)

    n_workers = min(PATH_WORKERS, max(1, len(all_sources) // 10))
    chunk_size = max(1, len(all_sources) // n_workers)
    source_chunks = [all_sources[i:i + chunk_size]
                     for i in range(0, len(all_sources), chunk_size)]

    logger.info("  %d source nodes, %d workers, chunk_size=%d",
                len(all_sources), n_workers, chunk_size)

    all_paths = []
    seen_paths: Set[tuple] = set()

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        for chunk in source_chunks:
            args = (edges_data, nodes_list, entity_layer, chunk, min_hops, max_hops)
            futures.append(executor.submit(_bfs_paths_worker, args))

        done_count = 0
        for f in as_completed(futures):
            try:
                paths = f.result()
                for p in paths:
                    path_key = tuple(p["nodes"])
                    if path_key not in seen_paths:
                        seen_paths.add(path_key)
                        all_paths.append(p)
            except Exception as e:
                logger.warning("Path worker failed: %s", e)
            done_count += 1
            if done_count % 4 == 0 or done_count == len(futures):
                logger.info("    path mining: %d/%d workers done, %d paths found",
                            done_count, len(futures), len(all_paths))

    logger.info("Phase 3: found %d unique cross-layer paths", len(all_paths))

    # If too many, sample stratified by layer pairs
    if len(all_paths) > target:
        # Categorize by source-target layer pair
        pair_paths: Dict[str, List[dict]] = defaultdict(list)
        for p in all_paths:
            layers = p["layers"]
            valid_layers = [l for l in layers if l in LAYERS]
            if len(valid_layers) >= 2:
                pair_key = f"{valid_layers[0]}->{valid_layers[-1]}"
                pair_paths[pair_key].append(p)

        # Sample proportionally
        sampled = []
        per_pair = max(1, target // max(1, len(pair_paths)))
        random.seed(42)
        for pair_key, plist in sorted(pair_paths.items()):
            n_sample = min(per_pair, len(plist))
            sampled.extend(random.sample(plist, n_sample))
        # Fill remaining from all
        remaining = target - len(sampled)
        if remaining > 0:
            leftover = [p for p in all_paths if p not in sampled]
            sampled.extend(random.sample(leftover, min(remaining, len(leftover))))
        all_paths = sampled[:target]
        logger.info("Phase 3: sampled to %d paths", len(all_paths))
    elif len(all_paths) < target:
        logger.warning("Phase 3: only found %d paths (target was %d)", len(all_paths), target)

    # Save
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_paths, f, indent=1, ensure_ascii=False)
    logger.info("Phase 3 done: %d paths saved", len(all_paths))

    # Save checkpoint every 1000 for safety (already done above)
    return all_paths


# ============================================================
# Phase 4 — NoveltyScore (corrected IP with full PageRank)
# ============================================================
def phase4(G: nx.DiGraph, entity_layer: Dict[str, str],
           paths: List[dict], topo_results: Dict) -> List[dict]:
    """
    Compute NoveltyScore with corrected IP (full PageRank, not hardcoded 0.5).
    Caches LF/CLC/IP components for each path for Phase 5 sensitivity sweep.
    """
    out_path = OUT_DIR / "phase4_paths_scored.json"

    pagerank = topo_results.get("_pagerank", {})
    max_pr = max(pagerank.values()) if pagerank else 1.0

    # --- Pre-compute entity/relation frequencies ---
    entity_freq: Counter = Counter()
    relation_freq: Counter = Counter()
    for u, v, data in G.edges(data=True):
        entity_freq[u] += 1
        entity_freq[v] += 1
        rel = data.get("relationship", "unknown")
        relation_freq[rel] += 1

    max_freq = max(entity_freq.values()) if entity_freq else 1

    # --- Pre-compute severity scores (from script 66) ---
    out_degrees_dict = dict(G.out_degree())
    max_out_degree = max(out_degrees_dict.values()) if out_degrees_dict else 1
    severity_scores: Dict[str, float] = {}
    for node in G.nodes():
        node_layer = entity_layer.get(node, "unknown")
        layer_weight = LAYER_SEVERITY.get(node_layer, 0.5)
        out_deg = out_degrees_dict.get(node, 0)
        degree_factor = out_deg / max_out_degree if max_out_degree > 0 else 0
        # avg confidence of out-edges
        edge_confs = [data.get("confidence", 0.5) for _, _, data in G.out_edges(node, data=True)]
        avg_conf = np.mean(edge_confs) if edge_confs else 0.5
        severity_scores[node] = layer_weight * 0.4 + degree_factor * 0.3 + avg_conf * 0.3

    logger.info("Phase 4: scoring %d paths (PageRank nodes=%d, max_PR=%.6f) ...",
                len(paths), len(pagerank), max_pr)

    for i, p in enumerate(paths):
        nodes = p["nodes"]
        n = len(nodes)
        if n < 2:
            p["LF"] = 0.0
            p["CLC"] = 0.0
            p["IP"] = 0.0
            p["novelty_score"] = 0.0
            continue

        # --- LF(P): Literature Frequency ---
        e_freqs = [entity_freq.get(node, 1) for node in nodes]
        # Include relation frequencies
        r_freqs = []
        for j in range(n - 1):
            ed = G.get_edge_data(nodes[j], nodes[j + 1])
            if ed:
                rel = ed.get("relationship", "unknown")
                r_freqs.append(relation_freq.get(rel, 1))
        all_freqs = e_freqs + r_freqs
        f_p = np.exp(np.mean(np.log(np.array(all_freqs, dtype=float) + 1)))
        lf = 1.0 - f_p / (max_freq + 1)
        lf = max(0.0, min(1.0, lf))

        # --- CLC(P): Cross-Layer Connectivity ---
        cross_count = 0
        for j in range(n - 1):
            l1 = entity_layer.get(nodes[j], "unknown")
            l2 = entity_layer.get(nodes[j + 1], "unknown")
            if l1 != l2 and l1 != "unknown" and l2 != "unknown":
                cross_count += 1
        clc = cross_count / (n - 1)

        # --- IP(P): Impact Potential (CORRECTED — full PageRank) ---
        impact_sum = 0.0
        for node in nodes:
            pr_val = pagerank.get(node, 0.0)
            sev_val = severity_scores.get(node, 0.5)
            impact_sum += pr_val * sev_val
        ip = (impact_sum / n) / max_pr if max_pr > 0 else 0.0
        ip = min(1.0, ip)

        # --- NoveltyScore = α·LF + β·CLC + γ·IP ---
        novelty = DEFAULT_ALPHA * lf + DEFAULT_BETA * clc + DEFAULT_GAMMA * ip

        p["LF"] = round(lf, 6)
        p["CLC"] = round(clc, 6)
        p["IP"] = round(ip, 6)
        p["novelty_score"] = round(novelty, 6)

    # Sort by novelty score
    paths.sort(key=lambda x: x.get("novelty_score", 0), reverse=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(paths, f, indent=1, ensure_ascii=False)
    logger.info("Phase 4 done: scored %d paths, top score=%.4f, median=%.4f",
                len(paths),
                paths[0]["novelty_score"] if paths else 0,
                paths[len(paths) // 2]["novelty_score"] if paths else 0)

    return paths


# ============================================================
# Phase 5 — Sensitivity scan (pure numerical, very fast)
# ============================================================
def phase5(paths: List[dict]) -> Dict:
    """
    Sweep α/β/γ weight combinations and θ thresholds.
    LF/CLC/IP are pre-computed — only linear combination needed.

    Statistical tests:
      - Spearman rank correlation (full ranking) + p-value per combo
      - Kendall's W concordance coefficient + χ² test across all combos
      - Grey Rhino rank CV (coefficient of variation)
    """
    out_sensitivity = OUT_DIR / "sensitivity_results.json"
    out_summary = OUT_DIR / "sensitivity_summary.csv"
    out_grey_rhino = OUT_DIR / "grey_rhino_paths.json"

    logger.info("Phase 5: sensitivity scan ...")

    # --- Extract pre-computed components ---
    path_components = []
    for p in paths:
        path_components.append({
            "idx": len(path_components),
            "nodes": p["nodes"],
            "LF": p.get("LF", 0),
            "CLC": p.get("CLC", 0),
            "IP": p.get("IP", 0),
            "hops": p.get("hops", 0),
        })
    n_paths = len(path_components)
    logger.info("  %d paths with pre-computed LF/CLC/IP", n_paths)

    if n_paths == 0:
        logger.warning("Phase 5: no paths to analyse")
        return {}

    # Vectorised arrays for fast computation
    lf_arr = np.array([p["LF"] for p in path_components])
    clc_arr = np.array([p["CLC"] for p in path_components])
    ip_arr = np.array([p["IP"] for p in path_components])

    # --- Generate weight combinations ---
    alpha_values = [0.3, 0.4, 0.5, 0.6, 0.7]
    step = 0.05
    weight_combos = []
    for alpha in alpha_values:
        remaining = round(1.0 - alpha, 10)
        beta = 0.0
        while beta <= remaining + 1e-9:
            gamma = round(remaining - beta, 10)
            if gamma >= 0:
                weight_combos.append((round(alpha, 2), round(beta, 2), round(gamma, 2)))
            beta = round(beta + step, 10)
    logger.info("  %d weight combinations", len(weight_combos))

    # --- Default scores and ranking ---
    default_scores = DEFAULT_ALPHA * lf_arr + DEFAULT_BETA * clc_arr + DEFAULT_GAMMA * ip_arr
    default_ranking = np.argsort(-default_scores)
    default_top20_set = set(default_ranking[:20].tolist())
    # Build default rank array (0-indexed position → rank 1..n)
    default_ranks = np.empty(n_paths, dtype=int)
    default_ranks[default_ranking] = np.arange(1, n_paths + 1)

    # --- θ thresholds ---
    theta_values = [0.5, 0.6, 0.7, 0.8, 0.9]

    # --- Scan: compute scores, rankings, Jaccard, Spearman per combo ---
    sensitivity_rows = []
    all_rank_matrices = []  # for Kendall's W

    for alpha, beta, gamma in weight_combos:
        scores = alpha * lf_arr + beta * clc_arr + gamma * ip_arr
        ranking = np.argsort(-scores)
        top20_set = set(ranking[:20].tolist())

        # Build rank array for this combo
        combo_ranks = np.empty(n_paths, dtype=int)
        combo_ranks[ranking] = np.arange(1, n_paths + 1)
        all_rank_matrices.append(combo_ranks)

        # Jaccard with default Top-20
        intersection = len(default_top20_set & top20_set)
        union = len(default_top20_set | top20_set)
        jaccard = intersection / union if union > 0 else 0.0

        # Spearman rank correlation (full ranking vs default)
        rho, p_val = spearmanr(default_ranks, combo_ranks)

        # Novel counts at different θ
        novel_counts = {}
        for theta in theta_values:
            novel_counts[str(theta)] = int(np.sum(scores >= theta))

        # Top-20 entries
        top20_entries = []
        for idx in ranking[:20]:
            top20_entries.append({
                "idx": int(idx),
                "nodes": path_components[idx]["nodes"],
                "score": round(float(scores[idx]), 6),
            })

        row = {
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "jaccard_vs_default": round(jaccard, 4),
            "spearman_rho": round(float(rho), 6),
            "spearman_p": float(p_val),
            "novel_counts": novel_counts,
            "top20": top20_entries,
        }
        sensitivity_rows.append(row)

    # --- Kendall's W concordance coefficient ---
    # W = 12 * S / (k² * (n³ - n))  where k = number of judges, n = number of items
    # S = sum of squared deviations of column rank-sums from mean rank-sum
    logger.info("  Computing Kendall's W concordance ...")
    rank_matrix = np.array(all_rank_matrices)  # shape (k, n)
    k = rank_matrix.shape[0]  # number of weight combos
    n = rank_matrix.shape[1]  # number of paths
    rank_sums = rank_matrix.sum(axis=0)  # sum of ranks per path
    mean_rank_sum = k * (n + 1) / 2.0
    S = np.sum((rank_sums - mean_rank_sum) ** 2)
    kendalls_w = 12.0 * S / (k ** 2 * (n ** 3 - n))
    # χ² approximation: χ² = k*(n-1)*W, df = n-1
    chi2_stat = k * (n - 1) * kendalls_w
    chi2_df = n - 1
    chi2_p = 1.0 - chi2.cdf(chi2_stat, chi2_df)  # p-value

    logger.info("  Kendall's W = %.6f, χ²(df=%d) = %.2f, p = %.2e",
                kendalls_w, chi2_df, chi2_stat, chi2_p)

    # --- Spearman summary ---
    spearman_rhos = [r["spearman_rho"] for r in sensitivity_rows]
    spearman_ps = [r["spearman_p"] for r in sensitivity_rows]
    # Exclude self-comparison (rho=1.0)
    non_self_rhos = [r for r in spearman_rhos if r < 1.0 - 1e-9]
    significant_count = sum(1 for p in spearman_ps if p < 0.05)

    logger.info("  Spearman ρ: mean=%.4f, median=%.4f, min=%.4f (all %d combos)",
                np.mean(spearman_rhos), np.median(spearman_rhos),
                np.min(spearman_rhos), len(spearman_rhos))
    logger.info("  Spearman p<0.05: %d/%d (%.1f%%)",
                significant_count, len(spearman_ps),
                100 * significant_count / len(spearman_ps))

    # --- Grey Rhino analysis ---
    grey_rhino_indices = []
    for i in range(n_paths):
        if default_scores[i] > NOVELTY_THETA and lf_arr[i] > 0.6:
            grey_rhino_indices.append(i)
    logger.info("  Grey Rhino paths (novelty>%.1f & LF>0.6): %d",
                NOVELTY_THETA, len(grey_rhino_indices))

    grey_rhino_paths = []
    for idx in grey_rhino_indices:
        # Collect rank positions from pre-computed rank matrix
        rank_positions = rank_matrix[:, idx].tolist()
        in_top20_count = sum(1 for r in rank_positions if r <= 20)
        in_top50_count = sum(1 for r in rank_positions if r <= 50)
        rank_mean = float(np.mean(rank_positions))
        rank_std = float(np.std(rank_positions))
        rank_cv = rank_std / rank_mean if rank_mean > 0 else 0.0

        grey_rhino_paths.append({
            "idx": int(idx),
            "nodes": path_components[idx]["nodes"],
            "default_score": round(float(default_scores[idx]), 6),
            "LF": round(float(lf_arr[idx]), 6),
            "CLC": round(float(clc_arr[idx]), 6),
            "IP": round(float(ip_arr[idx]), 6),
            "rank_range": [int(min(rank_positions)), int(max(rank_positions))],
            "rank_mean": round(rank_mean, 1),
            "rank_std": round(rank_std, 1),
            "rank_cv": round(rank_cv, 4),
            "in_top20_ratio": round(in_top20_count / len(weight_combos), 4),
            "in_top50_ratio": round(in_top50_count / len(weight_combos), 4),
        })

    grey_rhino_paths.sort(key=lambda x: x["default_score"], reverse=True)

    # --- Grey Rhino stability summary ---
    if grey_rhino_paths:
        gr_top50_gt80 = sum(1 for g in grey_rhino_paths if g["in_top50_ratio"] > 0.8)
        gr_top50_gt50 = sum(1 for g in grey_rhino_paths if g["in_top50_ratio"] > 0.5)
        gr_cvs = [g["rank_cv"] for g in grey_rhino_paths]
        gr_mean_cv = float(np.mean(gr_cvs))
        gr_median_cv = float(np.median(gr_cvs))
    else:
        gr_top50_gt80 = 0
        gr_top50_gt50 = 0
        gr_mean_cv = 0.0
        gr_median_cv = 0.0

    # --- Summary ---
    jaccard_values = [r["jaccard_vs_default"] for r in sensitivity_rows]

    summary = {
        "n_weight_combos": len(weight_combos),
        "n_paths": n_paths,
        # Jaccard
        "jaccard_mean": round(float(np.mean(jaccard_values)), 4),
        "jaccard_median": round(float(np.median(jaccard_values)), 4),
        "jaccard_min": round(float(np.min(jaccard_values)), 4),
        "jaccard_max": round(float(np.max(jaccard_values)), 4),
        "jaccard_gte_0.5_pct": round(sum(1 for j in jaccard_values if j >= 0.5) / len(jaccard_values), 4),
        # Spearman (full-ranking correlation)
        "spearman_rho_mean": round(float(np.mean(spearman_rhos)), 6),
        "spearman_rho_median": round(float(np.median(spearman_rhos)), 6),
        "spearman_rho_min": round(float(np.min(spearman_rhos)), 6),
        "spearman_significant_pct": round(significant_count / len(spearman_ps), 4),
        # Kendall's W
        "kendalls_w": round(kendalls_w, 6),
        "kendalls_w_chi2": round(chi2_stat, 2),
        "kendalls_w_df": chi2_df,
        "kendalls_w_p": chi2_p,
        # Grey Rhino
        "n_grey_rhino": len(grey_rhino_paths),
        "grey_rhino_top50_gt80pct": gr_top50_gt80,
        "grey_rhino_top50_gt50pct": gr_top50_gt50,
        "grey_rhino_rank_cv_mean": round(gr_mean_cv, 4),
        "grey_rhino_rank_cv_median": round(gr_median_cv, 4),
    }

    logger.info("Phase 5: Jaccard mean=%.4f, Spearman ρ mean=%.4f, Kendall W=%.6f (p=%.2e)",
                summary["jaccard_mean"], summary["spearman_rho_mean"],
                kendalls_w, chi2_p)
    logger.info("  Grey Rhino: %d total, %d in Top50 >80%% combos, rank CV mean=%.4f",
                len(grey_rhino_paths), gr_top50_gt80, gr_mean_cv)

    # --- Save results ---
    with open(out_sensitivity, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": sensitivity_rows}, f,
                  indent=1, ensure_ascii=False)

    with open(out_grey_rhino, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "paths": grey_rhino_paths}, f,
                  indent=2, ensure_ascii=False)

    # --- CSV for paper ---
    with open(out_summary, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["alpha", "beta", "gamma", "jaccard_vs_default",
                          "spearman_rho", "spearman_p",
                          "novel_theta_0.5", "novel_theta_0.6", "novel_theta_0.7",
                          "novel_theta_0.8", "novel_theta_0.9"])
        for r in sensitivity_rows:
            writer.writerow([
                r["alpha"], r["beta"], r["gamma"],
                r["jaccard_vs_default"],
                r["spearman_rho"], f"{r['spearman_p']:.2e}",
                r["novel_counts"].get("0.5", 0),
                r["novel_counts"].get("0.6", 0),
                r["novel_counts"].get("0.7", 0),
                r["novel_counts"].get("0.8", 0),
                r["novel_counts"].get("0.9", 0),
            ])
    logger.info("Phase 5 done: saved sensitivity_results.json, sensitivity_summary.csv, grey_rhino_paths.json")

    return {
        "summary": summary,
        "sensitivity_rows": sensitivity_rows,
        "grey_rhino_paths": grey_rhino_paths,
    }


# ============================================================
# Phase 6 — Permutation test for multiple comparisons
# ============================================================
def phase6(G: nx.DiGraph, entity_layer: Dict[str, str],
           paths: List[dict], topo_results: Dict) -> Dict:
    """
    Permutation test for multiple-comparison control.

    Strategy: compare actual NoveltyScore distribution against a null
    distribution from random walks on the same graph.

    Three complementary analyses:
    1. Mann-Whitney U: actual scores vs null scores (distribution-level)
    2. Per-path empirical p-values with BH-FDR (path-level, 1000 perms)
    3. Quantile exceedance: fraction of null permutations whose top-K max
       exceeds the actual top-K threshold (controls for cherry-picking)

    Runs 1000 permutations (N_PERMUTATIONS); ~5 min for 10k paths.
    """
    out_path = OUT_DIR / "permutation_test.json"

    n_paths = len(paths)
    if n_paths == 0:
        logger.warning("Phase 6: no paths")
        return {}

    n_perm = N_PERMUTATIONS  # 1000
    logger.info("Phase 6: permutation test (%d iterations × %d paths) ...",
                n_perm, n_paths)

    # --- Actual scores ---
    actual_scores = np.array([p.get("novelty_score", 0) for p in paths])
    path_lengths = [len(p["nodes"]) for p in paths]

    # --- Graph properties for scoring ---
    pagerank = topo_results.get("_pagerank", {})
    max_pr = max(pagerank.values()) if pagerank else 1.0

    entity_freq: Counter = Counter()
    relation_freq: Counter = Counter()
    for u, v, data in G.edges(data=True):
        entity_freq[u] += 1
        entity_freq[v] += 1
        relation_freq[data.get("relationship", "unknown")] += 1
    max_freq = max(entity_freq.values()) if entity_freq else 1

    out_degrees_dict = dict(G.out_degree())
    max_out_degree = max(out_degrees_dict.values()) if out_degrees_dict else 1
    severity_scores: Dict[str, float] = {}
    for node in G.nodes():
        nl = entity_layer.get(node, "unknown")
        lw = LAYER_SEVERITY.get(nl, 0.5)
        od = out_degrees_dict.get(node, 0)
        df = od / max_out_degree if max_out_degree > 0 else 0
        ec = [d.get("confidence", 0.5) for _, _, d in G.out_edges(node, data=True)]
        ac = float(np.mean(ec)) if ec else 0.5
        severity_scores[node] = lw * 0.4 + df * 0.3 + ac * 0.3

    # Adjacency for random walks
    adj = {node: list(G.successors(node)) for node in G.nodes()}
    all_nodes = list(G.nodes())
    n_nodes = len(all_nodes)

    rng = np.random.RandomState(42)

    def _random_walk(length: int) -> List[str]:
        start = all_nodes[rng.randint(n_nodes)]
        walk = [start]
        for _ in range(length - 1):
            nbrs = adj.get(walk[-1])
            if nbrs:
                walk.append(nbrs[rng.randint(len(nbrs))])
            else:
                walk.append(all_nodes[rng.randint(n_nodes)])
        return walk

    def _score_path(nodes: List[str]) -> float:
        n = len(nodes)
        if n < 2:
            return 0.0
        # LF
        e_f = [entity_freq.get(nd, 1) for nd in nodes]
        r_f = []
        for j in range(n - 1):
            ed = G.get_edge_data(nodes[j], nodes[j + 1])
            if ed:
                r_f.append(relation_freq.get(ed.get("relationship", "unknown"), 1))
            else:
                r_f.append(1)
        all_f = np.array(e_f + r_f, dtype=float)
        f_p = np.exp(np.mean(np.log(all_f + 1)))
        lf = max(0.0, min(1.0, 1.0 - f_p / (max_freq + 1)))
        # CLC
        cross = 0
        for j in range(n - 1):
            l1 = entity_layer.get(nodes[j], "unknown")
            l2 = entity_layer.get(nodes[j + 1], "unknown")
            if l1 != l2 and l1 != "unknown" and l2 != "unknown":
                cross += 1
        clc = cross / (n - 1)
        # IP
        impact = sum(pagerank.get(nd, 0.0) * severity_scores.get(nd, 0.5) for nd in nodes)
        ip = min(1.0, (impact / n) / max_pr) if max_pr > 0 else 0.0
        return DEFAULT_ALPHA * lf + DEFAULT_BETA * clc + DEFAULT_GAMMA * ip

    # --- Permutation loop ---
    exceedance_count = np.zeros(n_paths, dtype=int)
    null_score_samples = []  # collect all null scores for distribution comparison
    null_maxima = []         # max score per permutation (for quantile exceedance)

    for perm_i in range(n_perm):
        perm_scores = []
        for p_idx in range(n_paths):
            rw = _random_walk(path_lengths[p_idx])
            ns = _score_path(rw)
            perm_scores.append(ns)
            if ns >= actual_scores[p_idx]:
                exceedance_count[p_idx] += 1
        null_score_samples.append(perm_scores)
        null_maxima.append(float(max(perm_scores)))
        if (perm_i + 1) % 100 == 0:
            logger.info("    permutation %d/%d done", perm_i + 1, n_perm)

    # --- Empirical p-values ---
    empirical_p = (exceedance_count + 1) / (n_perm + 1)

    # --- BH-FDR ---
    sorted_indices = np.argsort(empirical_p)
    sorted_p = empirical_p[sorted_indices]
    bh_threshold = np.arange(1, n_paths + 1) / n_paths * FDR_ALPHA
    reject = sorted_p <= bh_threshold
    if np.any(reject):
        max_k = np.max(np.where(reject)[0])
        significant_mask = np.zeros(n_paths, dtype=bool)
        significant_mask[sorted_indices[:max_k + 1]] = True
    else:
        significant_mask = np.zeros(n_paths, dtype=bool)

    n_significant = int(np.sum(significant_mask))
    n_novel = int(np.sum(actual_scores > NOVELTY_THETA))
    n_novel_significant = int(np.sum(significant_mask & (actual_scores > NOVELTY_THETA)))

    # --- Analysis 1: Mann-Whitney U (distribution-level) ---
    all_null_scores = np.array(null_score_samples).flatten()
    from scipy.stats import mannwhitneyu
    u_stat, mw_p = mannwhitneyu(actual_scores, all_null_scores, alternative='greater')

    # Effect size: rank-biserial correlation r = 1 - 2U/(n1*n2)
    n1, n2 = len(actual_scores), len(all_null_scores)
    effect_size_r = 1.0 - 2.0 * u_stat / (n1 * n2)
    # Cohen's d
    pooled_std = np.sqrt((np.var(actual_scores) + np.var(all_null_scores)) / 2)
    cohens_d = (np.mean(actual_scores) - np.mean(all_null_scores)) / pooled_std if pooled_std > 0 else 0

    # --- Analysis 2: Quantile exceedance test ---
    # For each quantile threshold, what fraction of null permutations
    # produced a max score exceeding the actual score at that quantile?
    actual_sorted = np.sort(actual_scores)[::-1]  # descending
    null_maxima_arr = np.array(null_maxima)
    quantile_tests = {}
    for k_label, k_val in [("top1", 0), ("top10", 9), ("top20", 19),
                            ("top50", 49), ("top100", 99)]:
        if k_val < n_paths:
            actual_threshold = float(actual_sorted[k_val])
            # How many null permutations have at least k_val+1 paths above threshold?
            # Simpler proxy: fraction of null permutations whose max > actual threshold
            null_above = int(np.sum(null_maxima_arr >= actual_threshold))
            quantile_tests[k_label] = {
                "actual_threshold": round(actual_threshold, 6),
                "null_exceed_count": null_above,
                "null_exceed_pct": round(100.0 * null_above / n_perm, 2),
            }

    # --- Analysis 3: Score-binned FDR estimate ---
    # Instead of per-path FDR, estimate global false discovery rate by score bin
    bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    null_flat = all_null_scores
    score_bin_fdr = []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        actual_in_bin = int(np.sum((actual_scores >= lo) & (actual_scores < hi)))
        null_in_bin = int(np.sum((null_flat >= lo) & (null_flat < hi)))
        # Expected null per permutation in this bin
        null_per_perm = null_in_bin / n_perm if n_perm > 0 else 0
        # FDR estimate = expected false / observed
        fdr_est = null_per_perm / actual_in_bin if actual_in_bin > 0 else 0.0
        score_bin_fdr.append({
            "bin": f"[{lo},{hi})",
            "actual_count": actual_in_bin,
            "null_expected_per_perm": round(null_per_perm, 2),
            "fdr_estimate": round(min(fdr_est, 1.0), 4),
        })
    # Last bin: >= 0.9
    actual_ge09 = int(np.sum(actual_scores >= 0.9))
    null_ge09 = int(np.sum(null_flat >= 0.9))
    null_per_perm_09 = null_ge09 / n_perm if n_perm > 0 else 0
    fdr_09 = null_per_perm_09 / actual_ge09 if actual_ge09 > 0 else 0.0

    logger.info("Phase 6: permutation test results:")
    logger.info("  Total paths: %d, permutations: %d", n_paths, n_perm)
    logger.info("  Significant (BH-FDR<%.2f): %d (%.1f%%)",
                FDR_ALPHA, n_significant, 100 * n_significant / n_paths)
    logger.info("  Novel (>%.1f): %d, of which BH-significant: %d",
                NOVELTY_THETA, n_novel, n_novel_significant)
    logger.info("  Mann-Whitney U=%.0f, p=%.4e, effect size r=%.4f, Cohen's d=%.4f",
                u_stat, mw_p, effect_size_r, cohens_d)
    logger.info("  Actual scores: mean=%.4f, std=%.4f", np.mean(actual_scores), np.std(actual_scores))
    logger.info("  Null scores: mean=%.4f, std=%.4f", np.mean(all_null_scores), np.std(all_null_scores))
    for k, v in quantile_tests.items():
        logger.info("  Quantile exceedance [%s]: threshold=%.4f, null exceed=%.1f%%",
                    k, v["actual_threshold"], v["null_exceed_pct"])
    for sb in score_bin_fdr:
        logger.info("  Score bin %s: actual=%d, null/perm=%.1f, FDR=%.4f",
                    sb["bin"], sb["actual_count"], sb["null_expected_per_perm"], sb["fdr_estimate"])

    # --- p-value percentiles ---
    p_percentiles = {
        "min": round(float(np.min(empirical_p)), 6),
        "p5": round(float(np.percentile(empirical_p, 5)), 6),
        "p25": round(float(np.percentile(empirical_p, 25)), 6),
        "median": round(float(np.median(empirical_p)), 6),
        "p75": round(float(np.percentile(empirical_p, 75)), 6),
        "max": round(float(np.max(empirical_p)), 6),
    }

    result = {
        "method": "Permutation test: random walks as null, 3-level statistical control",
        "n_permutations": n_perm,
        "fdr_alpha": FDR_ALPHA,
        "n_paths": n_paths,
        "analysis_1_distribution": {
            "test": "Mann-Whitney U (one-sided: actual > null)",
            "U_statistic": float(u_stat),
            "p_value": float(mw_p),
            "effect_size_r": round(float(effect_size_r), 6),
            "cohens_d": round(float(cohens_d), 4),
            "actual_mean": round(float(np.mean(actual_scores)), 4),
            "actual_std": round(float(np.std(actual_scores)), 4),
            "null_mean": round(float(np.mean(all_null_scores)), 4),
            "null_std": round(float(np.std(all_null_scores)), 4),
        },
        "analysis_2_bh_fdr": {
            "n_significant": n_significant,
            "significant_pct": round(100 * n_significant / n_paths, 2),
            "n_novel_paths": n_novel,
            "n_novel_significant": n_novel_significant,
            "novel_significant_pct": round(100 * n_novel_significant / n_novel if n_novel > 0 else 0, 2),
            "empirical_p_distribution": p_percentiles,
            "note": (f"BH-FDR requires p <= rank/m × α; with {n_perm} permutations "
                     f"the minimum achievable p is 1/{n_perm+1} ≈ {1/(n_perm+1):.1e}, which limits "
                     "rejection when m=10000. See score-bin FDR for practical estimate."),
        },
        "analysis_3_quantile_exceedance": quantile_tests,
        "analysis_4_score_bin_fdr": score_bin_fdr,
        "top20_p_values": [
            {
                "rank": i + 1,
                "nodes": paths[idx]["nodes"],
                "novelty_score": round(float(actual_scores[idx]), 6),
                "empirical_p": round(float(empirical_p[idx]), 6),
                "significant_bh": bool(significant_mask[idx]),
            }
            for i, idx in enumerate(np.argsort(-actual_scores)[:20])
        ],
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info("Phase 6 done: saved permutation_test.json")

    return result


# ============================================================
# Phase 7 — Final report
# ============================================================
def phase7(topo_results: Dict, paths: List[dict],
           sensitivity: Dict, permutation: Dict) -> None:
    """Merge all results into a single report JSON."""
    out_path = OUT_DIR / "topology_sensitivity_report.json"

    # Remove internal keys from topo_results for serialisation
    topo_clean = {k: v for k, v in topo_results.items() if not k.startswith("_")}

    report = {
        "meta": {
            "script": "7_topology_sensitivity.py",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "description": "Topology analysis + NoveltyScore sensitivity scan",
        },
        "topology": topo_clean,
        "path_mining": {
            "total_paths": len(paths),
            "hop_distribution": dict(Counter(p.get("hops", 0) for p in paths)),
            "layer_pair_distribution": dict(Counter(
                f"{p['layers'][0]}->{p['layers'][-1]}" for p in paths
                if len(p.get("layers", [])) >= 2
            )),
            "top10_paths": [
                {k: v for k, v in p.items() if k != "_internal"}
                for p in paths[:10]
            ],
        },
        "novelty_score": {
            "default_weights": {"alpha": DEFAULT_ALPHA, "beta": DEFAULT_BETA, "gamma": DEFAULT_GAMMA},
            "default_theta": NOVELTY_THETA,
            "ip_method": "full PageRank (alpha=0.85) × severity",
            "score_distribution": {
                "min": round(min(p.get("novelty_score", 0) for p in paths), 4) if paths else 0,
                "max": round(max(p.get("novelty_score", 0) for p in paths), 4) if paths else 0,
                "mean": round(np.mean([p.get("novelty_score", 0) for p in paths]), 4) if paths else 0,
                "median": round(np.median([p.get("novelty_score", 0) for p in paths]), 4) if paths else 0,
                "above_0.7": sum(1 for p in paths if p.get("novelty_score", 0) > 0.7),
            },
        },
        "sensitivity": sensitivity.get("summary", {}),
        "permutation_test": {
            "method": permutation.get("method", ""),
            "n_permutations": permutation.get("n_permutations", 0),
            "distribution_test": permutation.get("analysis_1_distribution", {}),
            "bh_fdr": permutation.get("analysis_2_bh_fdr", {}),
            "quantile_exceedance": permutation.get("analysis_3_quantile_exceedance", {}),
            "score_bin_fdr": permutation.get("analysis_4_score_bin_fdr", []),
        },
        "grey_rhino": {
            "count": len(sensitivity.get("grey_rhino_paths", [])),
            "top5": sensitivity.get("grey_rhino_paths", [])[:5],
        },
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("Phase 6: final report saved to %s", out_path)

    # --- Print key metrics ---
    logger.info("\n" + "=" * 80)
    logger.info("TOPOLOGY & SENSITIVITY REPORT SUMMARY")
    logger.info("=" * 80)
    logger.info("  Nodes: %d  |  Edges: %d", topo_clean.get("n_nodes", 0), topo_clean.get("n_edges", 0))
    logger.info("  Bio-mediation ratio: %.3f", topo_clean.get("bio_mediation_ratio", 0))
    logger.info("  Communities: %d  |  Modularity: %.3f",
                topo_clean.get("n_communities", 0), topo_clean.get("modularity", 0))
    logger.info("  Bridge nodes: %d", topo_clean.get("bridge_nodes_count", 0))
    logger.info("  Paths mined: %d", len(paths))
    ns = report["novelty_score"]["score_distribution"]
    logger.info("  NoveltyScore: mean=%.4f, median=%.4f, >0.7=%d",
                ns["mean"], ns["median"], ns["above_0.7"])
    ss = sensitivity.get("summary", {})
    logger.info("  Weight combos: %d  |  Avg Jaccard: %.4f", ss.get("n_weight_combos", 0), ss.get("jaccard_mean", 0))
    logger.info("  Spearman ρ mean: %.4f  |  Kendall W: %.6f (p=%.2e)",
                ss.get("spearman_rho_mean", 0), ss.get("kendalls_w", 0), ss.get("kendalls_w_p", 1))
    logger.info("  Grey Rhino: %d paths, Top50>80%%: %d, rank CV mean: %.4f",
                ss.get("n_grey_rhino", 0), ss.get("grey_rhino_top50_gt80pct", 0),
                ss.get("grey_rhino_rank_cv_mean", 0))
    pm = permutation if permutation else {}
    dist_test = pm.get("analysis_1_distribution", {})
    bh_fdr = pm.get("analysis_2_bh_fdr", {})
    logger.info("  Permutation test (%d perms):", pm.get("n_permutations", 0))
    logger.info("    Mann-Whitney U=%.0f, p=%.4e, Cohen's d=%.4f",
                dist_test.get("U_statistic", 0), dist_test.get("p_value", 1),
                dist_test.get("cohens_d", 0))
    logger.info("    BH-FDR significant: %d/%d", bh_fdr.get("n_significant", 0), pm.get("n_paths", 0))
    for sb in pm.get("analysis_4_score_bin_fdr", []):
        logger.info("    Score bin %s: actual=%d, FDR=%.4f",
                    sb.get("bin", "?"), sb.get("actual_count", 0), sb.get("fdr_estimate", 0))
    logger.info("=" * 80)


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Topology Analysis + NoveltyScore Sensitivity")
    parser.add_argument("--phase", type=int, default=None,
                        help="Run only a specific phase (1-7)")
    args = parser.parse_args()

    run_phase = args.phase

    start_time = time.time()

    # ---- Phase 1 ----
    if run_phase is None or run_phase == 1:
        logger.info("=" * 60 + " PHASE 1: Load KG + Merge + Graph " + "=" * 60)
        G, entity_layer, all_triples, merge_mapping = phase1()
    else:
        logger.info("Loading Phase 1 from checkpoint ...")
        ckpt_graph = OUT_DIR / "phase1_graph.pkl"
        ckpt_merge = OUT_DIR / "phase1_merge_mapping.json"
        with open(ckpt_graph, "rb") as f:
            saved = pickle.load(f)
        G = saved["G"]
        entity_layer = saved["entity_layer"]
        all_triples = saved["all_triples"]
        with open(ckpt_merge, "r", encoding="utf-8") as f:
            merge_mapping = json.load(f)

    t1 = time.time()
    logger.info("Phase 1 elapsed: %.1f s", t1 - start_time)

    # ---- Phase 2 ----
    if run_phase is None or run_phase == 2:
        logger.info("=" * 60 + " PHASE 2: Topology Analysis " + "=" * 60)
        topo_results = phase2(G, entity_layer)
    else:
        topo_path = OUT_DIR / "topology_results.json"
        if topo_path.exists():
            with open(topo_path, "r", encoding="utf-8") as f:
                topo_results = json.load(f)
            # Reload pagerank + bc for Phase 4
            ckpt_bc = OUT_DIR / "phase2_bc.pkl"
            if ckpt_bc.exists():
                with open(ckpt_bc, "rb") as f:
                    topo_results["_bc"] = pickle.load(f)
            pagerank = nx.pagerank(G, alpha=0.85)
            topo_results["_pagerank"] = pagerank
        else:
            logger.info("No topology checkpoint — running Phase 2 ...")
            topo_results = phase2(G, entity_layer)

    t2 = time.time()
    logger.info("Phase 2 elapsed: %.1f s", t2 - t1)

    # ---- Phase 3 ----
    if run_phase is None or run_phase == 3:
        logger.info("=" * 60 + " PHASE 3: Path Mining " + "=" * 60)
        paths = phase3(G, entity_layer)
    else:
        paths_path = OUT_DIR / "phase3_paths.json"
        if paths_path.exists():
            with open(paths_path, "r", encoding="utf-8") as f:
                paths = json.load(f)
        else:
            logger.info("No paths checkpoint — running Phase 3 ...")
            paths = phase3(G, entity_layer)

    t3 = time.time()
    logger.info("Phase 3 elapsed: %.1f s", t3 - t2)

    # ---- Phase 4 ----
    if run_phase is None or run_phase == 4:
        logger.info("=" * 60 + " PHASE 4: NoveltyScore (corrected IP) " + "=" * 60)
        paths = phase4(G, entity_layer, paths, topo_results)
    else:
        scored_path = OUT_DIR / "phase4_paths_scored.json"
        if scored_path.exists():
            with open(scored_path, "r", encoding="utf-8") as f:
                paths = json.load(f)
        else:
            logger.info("No scored paths — running Phase 4 ...")
            paths = phase4(G, entity_layer, paths, topo_results)

    t4 = time.time()
    logger.info("Phase 4 elapsed: %.1f s", t4 - t3)

    # ---- Phase 5 ----
    if run_phase is None or run_phase == 5:
        logger.info("=" * 60 + " PHASE 5: Sensitivity Scan " + "=" * 60)
        sensitivity = phase5(paths)
    else:
        sens_path = OUT_DIR / "sensitivity_results.json"
        if sens_path.exists():
            with open(sens_path, "r", encoding="utf-8") as f:
                sens_data = json.load(f)
            gr_path = OUT_DIR / "grey_rhino_paths.json"
            with open(gr_path, "r", encoding="utf-8") as f:
                gr_data = json.load(f)
            sensitivity = {"summary": sens_data["summary"],
                           "sensitivity_rows": sens_data["rows"],
                           "grey_rhino_paths": gr_data.get("paths", [])}
        else:
            logger.info("No sensitivity checkpoint — running Phase 5 ...")
            sensitivity = phase5(paths)

    t5 = time.time()
    logger.info("Phase 5 elapsed: %.1f s", t5 - t4)

    # ---- Phase 6 ----
    if run_phase is None or run_phase == 6:
        logger.info("=" * 60 + " PHASE 6: Permutation Test " + "=" * 60)
        permutation = phase6(G, entity_layer, paths, topo_results)
    else:
        perm_path = OUT_DIR / "permutation_test.json"
        if perm_path.exists():
            with open(perm_path, "r", encoding="utf-8") as f:
                permutation = json.load(f)
        else:
            permutation = {}

    t6 = time.time()
    logger.info("Phase 6 elapsed: %.1f s", t6 - t5)

    # ---- Phase 7 ----
    if run_phase is None or run_phase == 7:
        logger.info("=" * 60 + " PHASE 7: Final Report " + "=" * 60)
        phase7(topo_results, paths, sensitivity, permutation)

    total_elapsed = time.time() - start_time
    logger.info("Total elapsed: %.1f seconds (%.1f minutes)", total_elapsed, total_elapsed / 60)


if __name__ == "__main__":
    main()
