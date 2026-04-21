"""
6_entity_merge_eval.py
Entity Merge Evaluation.

Evaluates over-merging (Precision) and under-merging (Recall) at three cosine
distance thresholds, plus topology stability (bio-mediation ratio, betweenness
ranking, cross-layer matrix) across baseline + 3 merge configurations.

Pipeline position:
    Input:  data/processed/kg/*.json   (script 1 output, 5738 files)
    Output: data/processed/entity_merge_eval/

Usage:
    cd workspace/code
    python 6_entity_merge_eval.py               # full run (checkpoint/resume)
    python 6_entity_merge_eval.py --phase B     # embedding only
    python 6_entity_merge_eval.py --phase D     # Precision only
    python 6_entity_merge_eval.py --phase E     # Recall only
"""

import argparse
import csv
import json
import logging
import os
import random
import time
import threading
import multiprocessing as mp
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import faiss
import networkx as nx
import numpy as np
import requests
import yaml
from openai import OpenAI
from scipy.stats import kendalltau

# ============================================================
# Paths
# ============================================================
BASE_DIR = Path(__file__).parent.parent
CONFIG_DIR = BASE_DIR / "config"
KG_DIR = BASE_DIR / "data" / "processed" / "kg"
OUT_DIR = BASE_DIR / "data" / "processed" / "entity_merge_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# API config
# ============================================================
_key_path = CONFIG_DIR / "api_keys.yaml"
with open(_key_path, "r", encoding="utf-8") as _f:
    _api_keys = yaml.safe_load(_f)

# Embedding API (DashScope text-embedding-v4)
_embed_cfg = _api_keys.get("embedding", {})
EMBED_API_KEY = _embed_cfg.get("api_key", "")
EMBED_BASE_URL = _embed_cfg.get("base_url", "").rstrip("/")
EMBED_MODEL = _embed_cfg.get("model", "text-embedding-v4")
EMBED_DIM = 1024
EMBED_BATCH = 10  # DashScope hard limit

# LLM API (primary, Claude Sonnet)
_primary = _api_keys.get("primary", {})
LLM_API_KEY = _primary.get("api_key", "")
LLM_BASE_URL = _primary.get("base_url", "")
LLM_MODEL = _primary.get("model", "claude-sonnet-4-5")
_llm_client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

# ============================================================
# Logging
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(OUT_DIR / "merge_eval.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ============================================================
# Constants
# ============================================================
THRESHOLDS = [0.02, 0.03, 0.05, 0.08, 0.10]
MERGE_ROUNDS = 3
TOP_K = 500
SAMPLE_SIZE = 300
RECALL_MARGIN = 0.02  # narrower margin for tighter thresholds
LLM_RPM = 60
BC_WORKERS = 40  # parallel workers for betweenness centrality
LAYERS = ["physical", "biological", "social", "economic"]


# ============================================================
# RPM controller (reused pattern from 6_hrrb_benchmark.py)
# ============================================================
class RPMController:
    """Sliding-window rate limiter."""

    def __init__(self, target_rpm: int = 60):
        self.target_rpm = target_rpm
        self.min_interval = 60.0 / target_rpm
        self.request_times: list = []
        self.lock = threading.Lock()

    def wait_if_needed(self):
        with self.lock:
            now = time.time()
            self.request_times = [t for t in self.request_times if now - t < 60]
            wait_time = 0.0
            if len(self.request_times) >= self.target_rpm:
                wait_time = 60 - (now - self.request_times[0]) + 0.1
            elif self.request_times:
                elapsed = now - self.request_times[-1]
                if elapsed < self.min_interval:
                    wait_time = self.min_interval - elapsed
            self.request_times.append(time.time() + wait_time)
        if wait_time > 0:
            time.sleep(wait_time)


llm_rpm = RPMController(LLM_RPM)

# Embedding throttle (simple interval, ~300 RPM)
_embed_lock = threading.Lock()
_embed_last_time = 0.0
_EMBED_INTERVAL = 60.0 / 300  # ~0.2s


def _embed_throttle():
    global _embed_last_time
    with _embed_lock:
        elapsed = time.time() - _embed_last_time
        if elapsed < _EMBED_INTERVAL:
            time.sleep(_EMBED_INTERVAL - elapsed)
        _embed_last_time = time.time()


# ============================================================
# Phase A: Data preparation
# ============================================================
def phase_a(test_limit: int = 0) -> Tuple[List[str], Dict[str, str], List[dict]]:
    """Load KG, collect unique entities + layer mapping + raw triples."""
    ckpt = OUT_DIR / "entities.json"

    # Always reload triples (needed for Phase F)
    logger.info("Phase A: loading KG triples from %s ...", KG_DIR)
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
            all_triples.append(t)
            for node, layer in [(sn, sl), (en, el)]:
                if node not in entity_layer_counter:
                    entity_layer_counter[node] = Counter()
                if layer in LAYERS:
                    entity_layer_counter[node][layer] += 1
        file_count += 1

    entities = sorted(entity_layer_counter.keys())
    entity_layer = {}
    for e, cnt in entity_layer_counter.items():
        if cnt:
            entity_layer[e] = cnt.most_common(1)[0][0]
        else:
            entity_layer[e] = "unknown"

    if test_limit > 0:
        random.seed(42)
        entities = random.sample(entities, min(test_limit, len(entities)))
        entity_layer = {e: entity_layer[e] for e in entities}
        # Filter triples to only those involving sampled entities
        entity_set = set(entities)
        all_triples = [t for t in all_triples
                       if t.get("start_node", "").strip() in entity_set
                       or t.get("end_node", "").strip() in entity_set]
        logger.info("  Test mode: filtered triples to %d (involving %d sampled entities)",
                     len(all_triples), len(entities))

    # Save checkpoint
    with open(ckpt, "w", encoding="utf-8") as f:
        json.dump({"entities": entities, "entity_layer": entity_layer}, f)

    logger.info("Phase A done: %d files, %d triples, %d unique entities",
                file_count, len(all_triples), len(entities))
    return entities, entity_layer, all_triples


# ============================================================
# Phase B: Embedding
# ============================================================
def _request_embeddings(texts: List[str], max_retries: int = 5) -> List[List[float]]:
    """Request embeddings from DashScope API."""
    url = f"{EMBED_BASE_URL}/embeddings"
    headers = {"Authorization": f"Bearer {EMBED_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": EMBED_MODEL, "input": texts, "encoding_format": "float"}

    for attempt in range(1, max_retries + 1):
        try:
            _embed_throttle()
            resp = requests.post(url, json=payload, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json().get("data", [])
            if len(data) != len(texts):
                raise ValueError(f"Embedding count mismatch: got {len(data)}, expected {len(texts)}")
            return [item["embedding"] for item in data]
        except Exception as exc:
            wait_s = min(2 ** attempt, 10)
            logger.warning("Embedding attempt %d/%d failed: %s", attempt, max_retries, exc)
            time.sleep(wait_s)

    logger.error("All %d embedding attempts failed, returning zero vectors", max_retries)
    return [np.zeros(EMBED_DIM).tolist() for _ in texts]


def phase_b(entities: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Embed entities, build FAISS index, search Top-K neighbours."""
    embed_path = OUT_DIR / "embeddings.npy"
    faiss_path = OUT_DIR / "faiss_results.npz"

    # --- embeddings ---
    if embed_path.exists():
        vectors = np.load(embed_path).astype("float32")
        if vectors.shape[0] == len(entities):
            logger.info("Phase B: loaded cached embeddings (%d, %d)", *vectors.shape)
        else:
            logger.warning("Embedding size mismatch (%d vs %d), rebuilding", vectors.shape[0], len(entities))
            vectors = None
    else:
        vectors = None

    if vectors is None:
        logger.info("Phase B: embedding %d entities (batch=%d) ...", len(entities), EMBED_BATCH)
        all_vecs = []
        total = len(entities)
        for i in range(0, total, EMBED_BATCH):
            if i % (EMBED_BATCH * 50) == 0:
                logger.info("  embedding %d/%d (%.1f%%)", i, total, 100 * i / total)
            batch = entities[i:i + EMBED_BATCH]
            all_vecs.extend(_request_embeddings(batch))
        vectors = np.asarray(all_vecs, dtype="float32")
        np.save(embed_path, vectors)
        logger.info("Phase B: embeddings saved (%d, %d)", *vectors.shape)

    # --- FAISS search ---
    if faiss_path.exists():
        data = np.load(faiss_path)
        distances = data["distances"]
        indices = data["indices"]
        if distances.shape[0] == len(entities):
            logger.info("Phase B: loaded cached FAISS results (%d, %d)", *distances.shape)
            return vectors, distances, indices
        else:
            logger.warning("FAISS result size mismatch, rebuilding")

    logger.info("Phase B: building FAISS index and searching Top-%d ...", TOP_K)
    vecs_norm = vectors.copy()
    faiss.normalize_L2(vecs_norm)
    index = faiss.IndexFlatIP(vecs_norm.shape[1])
    index.add(vecs_norm)

    k = min(TOP_K, len(entities))
    distances, indices = index.search(vecs_norm, k)
    np.savez(faiss_path, distances=distances, indices=indices)
    logger.info("Phase B done: FAISS search complete")
    return vectors, distances, indices


# ============================================================
# Phase C: 3-threshold x 3-round iterative merge
# ============================================================
def _greedy_cluster(entities: List[str], vectors: np.ndarray,
                    distances: np.ndarray, indices: np.ndarray,
                    threshold: float) -> Dict[str, str]:
    """Single round of greedy clustering. Returns entity -> canonical_name mapping."""
    n = len(entities)
    cosine_dist = 1.0 - distances  # inner product → cosine distance
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

    # Build mapping: all members → first member (canonical)
    mapping = {}
    for group in groups:
        canonical = entities[group[0]]
        for idx in group:
            mapping[entities[idx]] = canonical

    return mapping


def _compute_group_centroids(entities: List[str], vectors: np.ndarray,
                             mapping: Dict[str, str]) -> Tuple[List[str], np.ndarray]:
    """Compute L2-normalized centroids for each canonical group."""
    # Group entities by canonical name
    groups: Dict[str, List[int]] = {}
    entity_to_idx = {e: i for i, e in enumerate(entities)}
    for e, canon in mapping.items():
        if e in entity_to_idx:
            groups.setdefault(canon, []).append(entity_to_idx[e])
    # Add unmapped entities as singletons
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


def phase_c(entities: List[str], vectors: np.ndarray,
            distances: np.ndarray, indices: np.ndarray) -> Dict[float, Dict]:
    """Run 3-round iterative merge at each threshold."""
    results = {}

    for thresh in THRESHOLDS:
        logger.info("Phase C: threshold=%.2f, starting 3-round merge ...", thresh)
        cumulative_mapping: Dict[str, str] = {}
        round_stats = []

        cur_entities = entities
        cur_vectors = vectors.copy()
        faiss.normalize_L2(cur_vectors)
        cur_distances = distances
        cur_indices = indices

        for rd in range(1, MERGE_ROUNDS + 1):
            if rd > 1:
                # Recompute centroids and FAISS search for merged groups
                canon_names, centroids = _compute_group_centroids(entities, vectors, cumulative_mapping)
                cur_entities = canon_names
                cur_vectors = centroids
                idx = faiss.IndexFlatIP(centroids.shape[1])
                idx.add(centroids)
                k = min(TOP_K, len(canon_names))
                cur_distances, cur_indices = idx.search(centroids, k)

            rd_mapping = _greedy_cluster(cur_entities, cur_vectors,
                                         cur_distances, cur_indices, thresh)
            new_merges = len(rd_mapping)

            # Update cumulative mapping
            if rd == 1:
                cumulative_mapping.update(rd_mapping)
            else:
                # Chain: if original entity A → old_canon B, and B → new_canon C, then A → C
                for e in list(cumulative_mapping.keys()):
                    old_canon = cumulative_mapping[e]
                    if old_canon in rd_mapping:
                        cumulative_mapping[e] = rd_mapping[old_canon]
                # Add new direct mappings
                for e, canon in rd_mapping.items():
                    if e not in cumulative_mapping:
                        cumulative_mapping[e] = canon

            round_stats.append({"round": rd, "new_merges": new_merges})
            logger.info("  Round %d: %d new merges", rd, new_merges)

        # Compute final stats
        canonical_groups: Dict[str, List[str]] = {}
        for e, canon in cumulative_mapping.items():
            canonical_groups.setdefault(canon, []).append(e)
        # Add canonical itself if not already there
        for canon, members in list(canonical_groups.items()):
            if canon not in members:
                members.insert(0, canon)

        num_groups = len(canonical_groups)
        merged_entities = sum(len(m) for m in canonical_groups.values())
        merge_rate = merged_entities / len(entities) if entities else 0

        results[thresh] = {
            "mapping": cumulative_mapping,
            "groups": canonical_groups,
            "num_groups": num_groups,
            "merged_entities": merged_entities,
            "merge_rate": merge_rate,
            "round_stats": round_stats,
        }
        logger.info("Phase C: thresh=%.2f done — %d groups, %d merged (%.1f%%)",
                     thresh, num_groups, merged_entities, merge_rate * 100)

    return results


# ============================================================
# Phase D: Precision (over-merging)
# ============================================================
def _llm_judge_pair(entity_a: str, entity_b: str) -> str:
    """Ask LLM whether two entities are synonymous. Returns 'YES', 'NO', or 'INVALID'."""
    prompt = (
        "In heatwave cascading risk research, are these two entities "
        "synonymous or referring to the same concept?\n"
        f"Entity A: {entity_a}\n"
        f"Entity B: {entity_b}\n"
        "Answer YES or NO only."
    )
    try:
        llm_rpm.wait_if_needed()
        resp = _llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.0,
        )
        answer = resp.choices[0].message.content.strip().upper()
        if "YES" in answer:
            return "YES"
        elif "NO" in answer:
            return "NO"
        else:
            logger.warning("LLM ambiguous answer: %s (for %s vs %s)", answer, entity_a, entity_b)
            return "INVALID"
    except Exception as exc:
        logger.warning("LLM call failed: %s", exc)
        return "INVALID"


def _load_llm_cache(path: Path) -> Dict[str, str]:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_llm_cache(path: Path, cache: Dict[str, str]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def phase_d(merge_results: Dict[float, Dict], sample_size: int = SAMPLE_SIZE) -> Dict[float, Dict]:
    """Precision evaluation: sample merged pairs, LLM judges synonymy."""
    precision_results = {}

    for thresh in THRESHOLDS:
        cache_path = OUT_DIR / f"precision_cache_{thresh:.2f}.json"
        cache = _load_llm_cache(cache_path)
        groups = merge_results[thresh]["groups"]

        # Collect all merged pairs
        all_pairs = []
        for canon, members in groups.items():
            for i, a in enumerate(members):
                for b in members[i + 1:]:
                    all_pairs.append((a, b))

        random.seed(42)
        sampled = random.sample(all_pairs, min(sample_size, len(all_pairs)))
        logger.info("Phase D: thresh=%.2f — %d total pairs, sampling %d",
                     thresh, len(all_pairs), len(sampled))

        yes_count = 0
        no_count = 0
        invalid_count = 0

        for i, (a, b) in enumerate(sampled):
            key = f"{a}|||{b}"
            if key in cache:
                answer = cache[key]
            else:
                answer = _llm_judge_pair(a, b)
                cache[key] = answer
                if (i + 1) % 50 == 0:
                    _save_llm_cache(cache_path, cache)
                    logger.info("  Precision %.2f: %d/%d done", thresh, i + 1, len(sampled))

            if answer == "YES":
                yes_count += 1
            elif answer == "NO":
                no_count += 1
            else:
                invalid_count += 1

        _save_llm_cache(cache_path, cache)
        total_valid = yes_count + no_count
        precision = yes_count / total_valid if total_valid > 0 else 0.0

        precision_results[thresh] = {
            "total_pairs": len(all_pairs),
            "sampled": len(sampled),
            "yes": yes_count,
            "no": no_count,
            "invalid": invalid_count,
            "precision": precision,
        }
        logger.info("Phase D: thresh=%.2f — Precision=%.3f (%d YES / %d valid, %d invalid)",
                     thresh, precision, yes_count, total_valid, invalid_count)

    return precision_results


# ============================================================
# Phase E: Recall / under-merging (margin-based sampling)
# ============================================================
def phase_e(entities: List[str], merge_results: Dict[float, Dict],
            distances: np.ndarray, indices: np.ndarray,
            sample_size: int = SAMPLE_SIZE) -> Dict[float, Dict]:
    """Recall evaluation: sample close-but-unmerged pairs near threshold boundary."""
    recall_results = {}
    entity_idx = {e: i for i, e in enumerate(entities)}

    for thresh in THRESHOLDS:
        cache_path = OUT_DIR / f"recall_cache_{thresh:.2f}.json"
        cache = _load_llm_cache(cache_path)
        mapping = merge_results[thresh]["mapping"]

        # Find canonical for each entity
        def get_canon(e):
            return mapping.get(e, e)

        # Collect close-but-unmerged pairs in [threshold, threshold + margin]
        margin_upper = thresh + RECALL_MARGIN
        cosine_dist = 1.0 - distances
        candidate_pairs = []

        for e in entities:
            if e not in entity_idx:
                continue
            i = entity_idx[e]
            for j_idx in range(len(indices[i])):
                j = int(indices[i][j_idx])
                if j == i or j >= len(entities):
                    continue
                d = cosine_dist[i][j_idx]
                if thresh <= d < margin_upper:
                    other = entities[j]
                    # Only include if they are NOT in the same merge group
                    if get_canon(e) != get_canon(other):
                        pair = tuple(sorted([e, other]))
                        candidate_pairs.append(pair)

        # Deduplicate
        candidate_pairs = list(set(candidate_pairs))
        random.seed(43)
        sampled = random.sample(candidate_pairs, min(sample_size, len(candidate_pairs)))
        logger.info("Phase E: thresh=%.2f — %d candidate margin pairs, sampling %d",
                     thresh, len(candidate_pairs), len(sampled))

        yes_count = 0
        no_count = 0
        invalid_count = 0

        for i, (a, b) in enumerate(sampled):
            key = f"{a}|||{b}"
            if key in cache:
                answer = cache[key]
            else:
                answer = _llm_judge_pair(a, b)
                cache[key] = answer
                if (i + 1) % 50 == 0:
                    _save_llm_cache(cache_path, cache)
                    logger.info("  Recall %.2f: %d/%d done", thresh, i + 1, len(sampled))

            if answer == "YES":
                yes_count += 1
            elif answer == "NO":
                no_count += 1
            else:
                invalid_count += 1

        _save_llm_cache(cache_path, cache)
        total_valid = yes_count + no_count
        under_merge_rate = yes_count / total_valid if total_valid > 0 else 0.0

        recall_results[thresh] = {
            "candidate_pairs": len(candidate_pairs),
            "sampled": len(sampled),
            "yes": yes_count,
            "no": no_count,
            "invalid": invalid_count,
            "under_merge_rate": under_merge_rate,
        }
        logger.info("Phase E: thresh=%.2f — Under-merge rate=%.3f (%d YES / %d valid)",
                     thresh, under_merge_rate, yes_count, total_valid)

    return recall_results


# ============================================================
# Phase F: Topology stability
# ============================================================
def _build_graph(triples: List[dict], mapping: Dict[str, str]) -> nx.DiGraph:
    """Build nx.DiGraph from triples, applying entity merge mapping."""
    G = nx.DiGraph()
    for t in triples:
        sn = mapping.get(t.get("start_node", ""), t.get("start_node", ""))
        en = mapping.get(t.get("end_node", ""), t.get("end_node", ""))
        sl = t.get("start_layer", "").strip().lower()
        el = t.get("end_layer", "").strip().lower()
        rel = t.get("relationship", "")
        if not sn or not en or sn == en:
            continue
        G.add_edge(sn, en, relationship=rel, start_layer=sl, end_layer=el)
    return G


def _bc_worker(edges, nodes, subset, directed):
    """Compute betweenness centrality contribution from a subset of source nodes."""
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

    # Split source nodes into chunks
    chunk_size = max(1, n // n_workers)
    chunks = [nodes[i:i + chunk_size] for i in range(0, n, chunk_size)]

    edges = list(G.edges())
    directed = G.is_directed()

    bc_total = dict.fromkeys(nodes, 0.0)
    with ProcessPoolExecutor(max_workers=min(n_workers, len(chunks))) as executor:
        futures = {executor.submit(_bc_worker, edges, nodes, chunk, directed): i
                   for i, chunk in enumerate(chunks)}
        done = 0
        for f in as_completed(futures):
            partial_bc = f.result()
            for node, val in partial_bc.items():
                bc_total[node] += val
            done += 1
            if done % 10 == 0 or done == len(futures):
                logger.info("    betweenness: %d/%d chunks done", done, len(futures))

    return bc_total


def _compute_topology(G: nx.DiGraph, entity_layer: Dict[str, str],
                      mapping: Dict[str, str]) -> Dict:
    """Compute topology metrics for a graph."""
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    # Degree stats
    degrees = [d for _, d in G.degree()]
    degree_stats = {
        "mean": float(np.mean(degrees)) if degrees else 0,
        "median": float(np.median(degrees)) if degrees else 0,
        "max": int(np.max(degrees)) if degrees else 0,
        "std": float(np.std(degrees)) if degrees else 0,
    }

    # Betweenness centrality Top-20 (parallel exact computation)
    logger.info("  Computing betweenness centrality (%d nodes, %d edges, %d workers)...",
                n_nodes, n_edges, BC_WORKERS)
    bc = _parallel_betweenness(G, BC_WORKERS)
    top20 = sorted(bc.items(), key=lambda x: x[1], reverse=True)[:20]
    top20_nodes = [node for node, _ in top20]
    top20_scores = {node: score for node, score in top20}
    # Full ranking for Kendall tau
    all_bc_ranked = sorted(bc.items(), key=lambda x: x[1], reverse=True)
    node_rank = {node: rank + 1 for rank, (node, _) in enumerate(all_bc_ranked)}

    # Cross-layer matrix
    layer_matrix = {l1: {l2: 0 for l2 in LAYERS} for l1 in LAYERS}
    total_cross = 0
    for u, v, data in G.edges(data=True):
        sl = data.get("start_layer", "")
        el = data.get("end_layer", "")
        if sl in LAYERS and el in LAYERS:
            layer_matrix[sl][el] += 1
            if sl != el:
                total_cross += 1

    # Bio-mediation ratio
    phys_bio = layer_matrix.get("physical", {}).get("biological", 0)
    bio_econ = layer_matrix.get("biological", {}).get("economic", 0)
    phys_econ = layer_matrix.get("physical", {}).get("economic", 0)
    bio_med_ratio = (phys_bio + bio_econ) / phys_econ if phys_econ > 0 else float("inf")

    cross_layer_pct = total_cross / n_edges if n_edges > 0 else 0

    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "degree_stats": degree_stats,
        "top20_nodes": top20_nodes,
        "top20_scores": top20_scores,
        "node_rank": node_rank,
        "layer_matrix": layer_matrix,
        "phys_bio": phys_bio,
        "bio_econ": bio_econ,
        "phys_econ": phys_econ,
        "bio_mediation_ratio": bio_med_ratio,
        "cross_layer_pct": cross_layer_pct,
    }


def phase_f(triples: List[dict], entity_layer: Dict[str, str],
            merge_results: Dict[float, Dict]) -> Dict:
    """Topology stability analysis: baseline + 3 thresholds."""
    topo_results = {}

    # Baseline (no merge)
    logger.info("Phase F: building baseline graph (no merge)...")
    G_base = _build_graph(triples, {})
    topo_base = _compute_topology(G_base, entity_layer, {})
    topo_results["baseline"] = topo_base
    logger.info("  Baseline: %d nodes, %d edges, bio-med ratio=%.3f",
                topo_base["n_nodes"], topo_base["n_edges"], topo_base["bio_mediation_ratio"])

    # Each threshold
    for thresh in THRESHOLDS:
        logger.info("Phase F: building graph for thresh=%.2f ...", thresh)
        mapping = merge_results[thresh]["mapping"]
        G = _build_graph(triples, mapping)
        topo = _compute_topology(G, entity_layer, mapping)

        # Kendall tau vs baseline (union of Top-20 sets)
        base_top20 = set(topo_base["top20_nodes"])
        this_top20 = set(topo["top20_nodes"])
        union_nodes = list(base_top20 | this_top20)

        base_ranks = []
        this_ranks = []
        for node in union_nodes:
            base_ranks.append(topo_base["node_rank"].get(node, topo_base["n_nodes"] + 1))
            this_ranks.append(topo["node_rank"].get(node, topo["n_nodes"] + 1))

        if len(union_nodes) >= 2:
            tau, p_value = kendalltau(base_ranks, this_ranks)
        else:
            tau, p_value = 1.0, 1.0
        topo["kendall_tau"] = float(tau)
        topo["kendall_p"] = float(p_value)
        topo["tau_union_size"] = len(union_nodes)

        topo_results[f"d<{thresh:.2f}"] = topo
        logger.info("  thresh=%.2f: %d nodes, %d edges, bio-med=%.3f, tau=%.3f",
                     thresh, topo["n_nodes"], topo["n_edges"],
                     topo["bio_mediation_ratio"], tau)

    return topo_results


# ============================================================
# Phase G: Output
# ============================================================
def phase_g(merge_results: Dict[float, Dict], precision_results: Dict[float, Dict],
            recall_results: Dict[float, Dict], topo_results: Dict) -> None:
    """Write final outputs: merge_comparison.json + merge_summary.csv + llm_details.json."""

    # --- merge_comparison.json ---
    comparison = {"thresholds": {}, "topology": {}}
    for thresh in THRESHOLDS:
        key = f"d<{thresh:.2f}"
        mr = merge_results[thresh]
        comparison["thresholds"][key] = {
            "num_groups": mr["num_groups"],
            "merged_entities": mr["merged_entities"],
            "merge_rate": mr["merge_rate"],
            "round_stats": mr["round_stats"],
            "precision": precision_results.get(thresh, {}),
            "recall": recall_results.get(thresh, {}),
        }

    # Topology (remove node_rank from output — too large)
    for label, topo in topo_results.items():
        topo_out = {k: v for k, v in topo.items() if k != "node_rank"}
        comparison["topology"][label] = topo_out

    comp_path = OUT_DIR / "merge_comparison.json"
    with open(comp_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    logger.info("Phase G: wrote %s", comp_path)

    # --- merge_summary.csv ---
    csv_path = OUT_DIR / "merge_summary.csv"
    rows = []

    def _get_topo(label):
        return topo_results.get(label, {})

    labels = ["baseline"] + [f"d<{t:.2f}" for t in THRESHOLDS]
    metrics = [
        "merge_rounds",
        "num_groups", "merged_entities", "merge_rate",
        "precision_llm", "under_merge_rate_llm",
        "n_nodes", "n_edges", "degree_mean",
        "kendall_tau", "bio_mediation_ratio", "cross_layer_pct",
    ]

    header = ["metric"] + labels
    rows.append(header)

    for metric in metrics:
        row = [metric]
        for label in labels:
            if label == "baseline":
                if metric in ("merge_rounds", "num_groups", "merged_entities",
                               "precision_llm", "under_merge_rate_llm"):
                    row.append("-")
                elif metric == "merge_rate":
                    row.append("0%")
                elif metric == "kendall_tau":
                    row.append("1.000")
                else:
                    topo = _get_topo("baseline")
                    if metric == "n_nodes":
                        row.append(str(topo.get("n_nodes", "")))
                    elif metric == "n_edges":
                        row.append(str(topo.get("n_edges", "")))
                    elif metric == "degree_mean":
                        row.append(f"{topo.get('degree_stats', {}).get('mean', 0):.2f}")
                    elif metric == "bio_mediation_ratio":
                        row.append(f"{topo.get('bio_mediation_ratio', 0):.3f}")
                    elif metric == "cross_layer_pct":
                        row.append(f"{topo.get('cross_layer_pct', 0):.3f}")
            else:
                thresh = float(label.split("<")[1])
                mr = merge_results.get(thresh, {})
                pr = precision_results.get(thresh, {})
                rr = recall_results.get(thresh, {})
                topo = _get_topo(label)

                if metric == "merge_rounds":
                    row.append(str(MERGE_ROUNDS))
                elif metric == "num_groups":
                    row.append(str(mr.get("num_groups", "")))
                elif metric == "merged_entities":
                    row.append(str(mr.get("merged_entities", "")))
                elif metric == "merge_rate":
                    row.append(f"{mr.get('merge_rate', 0) * 100:.1f}%")
                elif metric == "precision_llm":
                    row.append(f"{pr.get('precision', 0):.3f}")
                elif metric == "under_merge_rate_llm":
                    row.append(f"{rr.get('under_merge_rate', 0):.3f}")
                elif metric == "n_nodes":
                    row.append(str(topo.get("n_nodes", "")))
                elif metric == "n_edges":
                    row.append(str(topo.get("n_edges", "")))
                elif metric == "degree_mean":
                    row.append(f"{topo.get('degree_stats', {}).get('mean', 0):.2f}")
                elif metric == "kendall_tau":
                    row.append(f"{topo.get('kendall_tau', 0):.3f}")
                elif metric == "bio_mediation_ratio":
                    row.append(f"{topo.get('bio_mediation_ratio', 0):.3f}")
                elif metric == "cross_layer_pct":
                    row.append(f"{topo.get('cross_layer_pct', 0):.3f}")
        rows.append(row)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    logger.info("Phase G: wrote %s", csv_path)

    # --- llm_details.json ---
    details = {"precision": {}, "recall": {}}
    for thresh in THRESHOLDS:
        key = f"d<{thresh:.2f}"
        p_cache_path = OUT_DIR / f"precision_cache_{thresh:.2f}.json"
        r_cache_path = OUT_DIR / f"recall_cache_{thresh:.2f}.json"
        if p_cache_path.exists():
            with open(p_cache_path) as f:
                details["precision"][key] = json.load(f)
        if r_cache_path.exists():
            with open(r_cache_path) as f:
                details["recall"][key] = json.load(f)

    det_path = OUT_DIR / "llm_details.json"
    with open(det_path, "w", encoding="utf-8") as f:
        json.dump(details, f, indent=2, ensure_ascii=False)
    logger.info("Phase G: wrote %s", det_path)

    # Print summary table
    logger.info("\n" + "=" * 80)
    logger.info("ENTITY MERGE EVALUATION SUMMARY")
    logger.info("=" * 80)
    col_widths = [25] + [12] * len(labels)
    for row in rows:
        line = row[0].ljust(col_widths[0])
        for j, val in enumerate(row[1:], 1):
            line += str(val).rjust(col_widths[j])
        logger.info(line)
    logger.info("=" * 80)


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Entity Merge Evaluation")
    parser.add_argument("--phase", type=str, default=None,
                        help="Run only a specific phase: A/B/C/D/E/F")
    parser.add_argument("--test", type=int, default=0,
                        help="Test mode: limit to N entities (e.g. --test 100)")
    parser.add_argument("--sample", type=int, default=SAMPLE_SIZE,
                        help="LLM sample size per threshold (default 300)")
    args = parser.parse_args()

    run_phase = args.phase.upper() if args.phase else None
    test_limit = args.test
    sample_size = args.sample

    if test_limit > 0:
        logger.info("=== TEST MODE: %d entities, %d samples ===", test_limit, sample_size)

    start_time = time.time()

    # Phase A
    if run_phase is None or run_phase == "A":
        entities, entity_layer, all_triples = phase_a(test_limit)
    else:
        # Load from checkpoint
        ckpt = OUT_DIR / "entities.json"
        with open(ckpt) as f:
            data = json.load(f)
        entities = data["entities"]
        entity_layer = data["entity_layer"]
        all_triples = []  # Lazy load in Phase F
        if run_phase in ("F",):
            _, _, all_triples = phase_a(test_limit)

    # Phase B
    if run_phase is None or run_phase == "B":
        vectors, distances, indices = phase_b(entities)
    elif run_phase in ("C", "D", "E", "F"):
        vectors, distances, indices = phase_b(entities)

    # Phase C
    if run_phase is None or run_phase == "C":
        merge_results = phase_c(entities, vectors, distances, indices)
    elif run_phase in ("D", "E", "F"):
        merge_results = phase_c(entities, vectors, distances, indices)

    # Phase D
    if run_phase is None or run_phase == "D":
        precision_results = phase_d(merge_results, sample_size)
    elif run_phase in ("E", "F"):
        precision_results = {}  # Skip if not needed

    # Phase E
    if run_phase is None or run_phase == "E":
        recall_results = phase_e(entities, merge_results, distances, indices, sample_size)
    elif run_phase == "F":
        recall_results = {}

    # Phase F
    if run_phase is None or run_phase == "F":
        if not all_triples:
            _, entity_layer, all_triples = phase_a(test_limit)
        topo_results = phase_f(all_triples, entity_layer, merge_results)
    else:
        topo_results = {}

    # Phase G
    if run_phase is None:
        phase_g(merge_results, precision_results, recall_results, topo_results)

    elapsed = time.time() - start_time
    logger.info("Total elapsed: %.1f seconds (%.1f minutes)", elapsed, elapsed / 60)


if __name__ == "__main__":
    main()
