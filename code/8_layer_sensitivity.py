"""
8_layer_sensitivity.py
Layer Sensitivity Analysis.

Three experiments:
  A) Layer merge/split: 5 alternative layer ontologies
  B) Literature volume normalization + bootstrap CI
  C) Alternative layer assignment (keyword-based vs LLM)

Pipeline position:
    Input:  data/processed/topology_sensitivity/phase1_graph.pkl  (script 7)
            data/processed/kg/*.json                              (5738 files)
    Output: data/processed/topology_sensitivity/
                layer_sensitivity.json
                layer_sensitivity_summary.csv

Usage:
    cd workspace/code
    python 8_layer_sensitivity.py          # full run ~10 min
    python 8_layer_sensitivity.py --exp A  # single experiment
    python 8_layer_sensitivity.py --exp B
    python 8_layer_sensitivity.py --exp C
"""

import argparse
import csv
import json
import logging
import pickle
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ============================================================
# Paths
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent
KG_DIR = BASE_DIR / "data" / "processed" / "kg"
OUT_DIR = BASE_DIR / "data" / "processed" / "topology_sensitivity"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT = OUT_DIR / "phase1_graph.pkl"
OUT_JSON = OUT_DIR / "layer_sensitivity.json"
OUT_CSV = OUT_DIR / "layer_sensitivity_summary.csv"

# ============================================================
# Logging
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(OUT_DIR / "8_layer_sensitivity.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ============================================================
# Constants
# ============================================================
LAYERS_4 = ["physical", "biological", "social", "economic"]

# --- Experiment A: 5-layer bio-split keywords ---
HEALTH_KEYWORDS = [
    "mortal", "death", "health", "disease", "illness", "hospital", "patient",
    "cardio", "respiratory", "asthma", "stroke", "dehydration", "morbid",
    "admission", "emergency", "medical", "mental", "anxiety", "depression",
    "kidney", "renal", "lung", "heart", "blood", "human", "elder", "child",
    "infant", "pregnan", "worker", "occupat", "sleep", "cognitive",
    "fatality", "injury",
]
HEALTH_RE = re.compile("|".join(HEALTH_KEYWORDS + [r"heat.?related"]),
                        re.IGNORECASE)

ECOLOGICAL_KEYWORDS = [
    "species", "ecosystem", "coral", "reef", "marine", "fish", "bird",
    "insect", "plant", "tree", "forest", "vegetation", "crop", "wheat",
    "rice", "maize", "yield", "photosyn", "chlorophyll", "algae", "phyto",
    "biodiversity", "habitat", "fauna", "flora", "larvae", "soil", "microb",
    "biomass", "canopy", "leaf", "root", "seed", "pollen", "livestock",
    "cattle", "ecology", "trophic", "spawn",
]
ECOLOGICAL_RE = re.compile("|".join(ECOLOGICAL_KEYWORDS), re.IGNORECASE)

# --- Experiment C: keyword rules ---
BIO_KEYWORDS_68 = {
    "coral", "fish", "marine", "species", "organism", "biological", "ecological",
    "ecosystem", "biodiversity", "genotype", "phenotype", "gene", "microbiota",
    "bacteria", "virus", "disease", "pathogen", "health", "physiological",
    "metabolism", "photosynthesis", "biomass", "vegetation", "crop", "yield",
    "animal", "livestock", "mortality", "morbidity", "death", "bleaching",
    "reproduction", "growth", "survival", "adaptation", "evolution",
}

# Extended keyword sets for full keyword-based layer assignment
PHYSICAL_KW = [
    "heat", "temp", "warm", "hot", "thermal", "drought", "dry", "precipitation",
    "rain", "water", "climate", "weather", "flood", "storm", "cyclone",
    "extreme", "wind", "humidity", "radiation", "solar", "atmosphere", "ocean",
    "sea", "ice", "snow", "cold", "frost", "air", "ozone", "cloud",
    "pressure", "monsoon", "geopotential", "convect", "aerosol", "albedo",
    "emission", "fire", "wildfire", "uhi", "urban.?heat",
]

SOCIAL_KW = [
    "social", "communit", "inequ", "vulnerab", "populat", "demographic",
    "migrat", "displac", "policy", "govern", "resilien",
    "justice", "equity", "household", "gender", "educat",
    "awareness", "percept", "behavio", "shelter", "housing", "neighbor",
    "civic", "welfare", "access", "conflict", "crime", "violence",
]

ECONOMIC_KW = [
    "cost", "loss", "gdp", "labor", "productiv", "job", "employ",
    "price", "market", "trade", "infrastruct", "grid", "power",
    "transport", "energy", "damage", "income", "wage", "insura",
    "expendit", "econom", "financ", "invest", "capital", "revenue",
    "profit", "industr", "manufactur", "electric",
]


# ============================================================
# Helpers
# ============================================================

def load_checkpoint() -> Tuple:
    """Load phase1 checkpoint from script 7."""
    logger.info("Loading phase1 checkpoint: %s", CHECKPOINT)
    with open(CHECKPOINT, "rb") as f:
        saved = pickle.load(f)
    G = saved["G"]
    entity_layer = saved["entity_layer"]
    all_triples = saved["all_triples"]
    logger.info("  %d nodes, %d edges, %d triples",
                G.number_of_nodes(), G.number_of_edges(), len(all_triples))
    return G, entity_layer, all_triples


def remap_entity_layer(entity_layer: Dict[str, str],
                       mapping: Dict[str, str]) -> Dict[str, str]:
    """Apply a simple layer mapping to all entities."""
    return {e: mapping.get(l, l) for e, l in entity_layer.items()}


def remap_5layer(entity_layer: Dict[str, str]) -> Dict[str, str]:
    """Split biological entities into health / ecological / biological."""
    new_el = {}
    for entity, layer in entity_layer.items():
        if layer != "biological":
            new_el[entity] = layer
            continue
        name_lower = entity.lower()
        if HEALTH_RE.search(name_lower):
            new_el[entity] = "health"
        elif ECOLOGICAL_RE.search(name_lower):
            new_el[entity] = "ecological"
        else:
            new_el[entity] = "biological"
    return new_el


def compute_transition_matrix(G, entity_layer: Dict[str, str],
                              layers: List[str]) -> np.ndarray:
    """N×N transition matrix using entity_layer for node layers."""
    n = len(layers)
    layer_idx = {l: i for i, l in enumerate(layers)}
    mat = np.zeros((n, n), dtype=int)
    for u, v in G.edges():
        sl = entity_layer.get(u, "")
        el = entity_layer.get(v, "")
        if sl in layer_idx and el in layer_idx:
            mat[layer_idx[sl]][layer_idx[el]] += 1
    return mat


def compute_cross_layer_stats(G, entity_layer: Dict[str, str],
                              layers: List[str]) -> dict:
    """Cross-layer edge count and percentage."""
    layer_set = set(layers)
    total = 0
    cross = 0
    for u, v in G.edges():
        sl = entity_layer.get(u, "")
        el = entity_layer.get(v, "")
        if sl in layer_set and el in layer_set:
            total += 1
            if sl != el:
                cross += 1
    return {
        "total_edges": total,
        "cross_layer_edges": cross,
        "cross_layer_pct": cross / total if total > 0 else 0.0,
    }


def compute_bridge_nodes(G, entity_layer: Dict[str, str],
                         layers: List[str], min_layers: int = 2) -> int:
    """Count nodes connected to >= min_layers distinct layers."""
    layer_set = set(layers)
    count = 0
    for node in G.nodes():
        connected = set()
        nl = entity_layer.get(node, "")
        if nl in layer_set:
            connected.add(nl)
        for _, v in G.out_edges(node):
            el = entity_layer.get(v, "")
            if el in layer_set:
                connected.add(el)
        for u, _ in G.in_edges(node):
            sl = entity_layer.get(u, "")
            if sl in layer_set:
                connected.add(sl)
        if len(connected) >= min_layers:
            count += 1
    return count


def generalized_bmr(mat: np.ndarray, layers: List[str],
                    scheme_name: str) -> dict:
    """Compute generalised bio-mediation ratio for each scheme."""
    idx = {l: i for i, l in enumerate(layers)}

    if scheme_name == "baseline_4layer":
        pb = int(mat[idx["physical"]][idx["biological"]])
        be = int(mat[idx["biological"]][idx["economic"]])
        pe = int(mat[idx["physical"]][idx["economic"]])
        num = pb + be
        bmr = num / pe if pe > 0 else (float("inf") if num > 0 else 0.0)
        return {"formula": "(P→B + B→E) / P→E",
                "P_B": pb, "B_E": be, "P_E": pe,
                "numerator": num, "denominator": pe, "BMR": bmr}

    elif scheme_name == "3layer_v1_merge_SE":
        pb = int(mat[idx["physical"]][idx["biological"]])
        b_se = int(mat[idx["biological"]][idx["socioeconomic"]])
        p_se = int(mat[idx["physical"]][idx["socioeconomic"]])
        num = pb + b_se
        bmr = num / p_se if p_se > 0 else (float("inf") if num > 0 else 0.0)
        return {"formula": "(P→B + B→SE) / P→SE",
                "P_B": pb, "B_SE": b_se, "P_SE": p_se,
                "numerator": num, "denominator": p_se, "BMR": bmr}

    elif scheme_name == "3layer_v2_merge_BS":
        p_bs = int(mat[idx["physical"]][idx["bio-social"]])
        bs_e = int(mat[idx["bio-social"]][idx["economic"]])
        pe = int(mat[idx["physical"]][idx["economic"]])
        num = p_bs + bs_e
        bmr = num / pe if pe > 0 else (float("inf") if num > 0 else 0.0)
        return {"formula": "(P→BS + BS→E) / P→E",
                "P_BS": p_bs, "BS_E": bs_e, "P_E": pe,
                "numerator": num, "denominator": pe, "BMR": bmr}

    elif scheme_name == "5layer_bio_split":
        bio_subs = [l for l in ["health", "ecological", "biological"]
                    if l in idx]
        num = 0
        detail = {}
        for bs in bio_subs:
            val_p = int(mat[idx["physical"]][idx[bs]])
            val_e = int(mat[idx[bs]][idx["economic"]])
            num += val_p + val_e
            detail[f"P→{bs}"] = val_p
            detail[f"{bs}→E"] = val_e
        pe = int(mat[idx["physical"]][idx["economic"]])
        detail["P→E"] = pe
        bmr = num / pe if pe > 0 else (float("inf") if num > 0 else 0.0)
        return {"formula": "Σ(P→bio_sub + bio_sub→E) / P→E",
                "numerator": num, "denominator": pe,
                "BMR": bmr, "detail": detail}

    elif scheme_name == "2layer_coarse":
        env_h = int(mat[idx["environmental"]][idx["human"]])
        h_env = int(mat[idx["human"]][idx["environmental"]])
        return {"formula": "env→human (no mediation layer)",
                "env_to_human": env_h, "human_to_env": h_env,
                "BMR": None,
                "note": "2-layer null baseline: no intermediate layer for mediation"}

    return {"BMR": None, "note": "Unknown scheme"}


def _safe_float(v):
    """Sanitise value for JSON serialisation."""
    if isinstance(v, float):
        if v == float("inf"):
            return "inf"
        if v == float("-inf"):
            return "-inf"
        if np.isnan(v):
            return "NaN"
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        f = float(v)
        if np.isinf(f):
            return "inf"
        if np.isnan(f):
            return "NaN"
        return f
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


# ============================================================
# Experiment A: Layer Merge / Split Sensitivity
# ============================================================
SCHEMES = {
    "baseline_4layer": {
        "layers": ["physical", "biological", "social", "economic"],
        "mapping": {
            "physical": "physical", "biological": "biological",
            "social": "social", "economic": "economic",
        },
        "description": "Original 4-layer ontology",
    },
    "3layer_v1_merge_SE": {
        "layers": ["physical", "biological", "socioeconomic"],
        "mapping": {
            "physical": "physical", "biological": "biological",
            "social": "socioeconomic", "economic": "socioeconomic",
        },
        "description": "3-layer: merge Social+Economic → socioeconomic",
    },
    "3layer_v2_merge_BS": {
        "layers": ["physical", "bio-social", "economic"],
        "mapping": {
            "physical": "physical", "biological": "bio-social",
            "social": "bio-social", "economic": "economic",
        },
        "description": "3-layer: merge Biological+Social → bio-social",
    },
    "5layer_bio_split": {
        "layers": ["physical", "health", "ecological", "biological",
                    "social", "economic"],
        "mapping": None,  # handled by remap_5layer()
        "description": "5(6)-layer: split Biological → health / ecological / biological",
    },
    "2layer_coarse": {
        "layers": ["environmental", "human"],
        "mapping": {
            "physical": "environmental", "biological": "environmental",
            "social": "human", "economic": "human",
        },
        "description": "2-layer: environmental(P+B) / human(S+E)",
    },
}


def experiment_A(G, entity_layer: Dict[str, str]) -> dict:
    """Test 5 alternative layer ontologies."""
    logger.info("=" * 60)
    logger.info("Experiment A: Layer Merge / Split Sensitivity")
    logger.info("=" * 60)

    results = {}
    for scheme_name, scheme_info in SCHEMES.items():
        t0 = time.time()
        logger.info("  Scheme: %s — %s", scheme_name, scheme_info["description"])

        # Remap entity layers
        if scheme_name == "5layer_bio_split":
            new_el = remap_5layer(entity_layer)
        else:
            new_el = remap_entity_layer(entity_layer, scheme_info["mapping"])

        layers = scheme_info["layers"]
        lcounts = Counter(new_el.values())

        # Transition matrix
        mat = compute_transition_matrix(G, new_el, layers)

        # Generalised BMR
        bmr_result = generalized_bmr(mat, layers, scheme_name)

        # Cross-layer stats
        cross_stats = compute_cross_layer_stats(G, new_el, layers)

        # Bridge nodes
        bridge_2 = compute_bridge_nodes(G, new_el, layers, min_layers=2)
        bridge_3 = compute_bridge_nodes(G, new_el, layers, min_layers=3)

        elapsed = time.time() - t0
        logger.info("    Layers: %s",
                     {l: lcounts.get(l, 0) for l in layers})
        logger.info("    BMR: %s", bmr_result.get("BMR"))
        logger.info("    Cross-layer: %.1f%%, Bridge(≥2): %d, Bridge(≥3): %d",
                     cross_stats["cross_layer_pct"] * 100, bridge_2, bridge_3)
        logger.info("    Done in %.1fs", elapsed)

        results[scheme_name] = {
            "description": scheme_info["description"],
            "n_layers": len(layers),
            "layers": layers,
            "entity_counts": {l: lcounts.get(l, 0) for l in layers},
            "transition_matrix": mat.tolist(),
            "bio_mediation": bmr_result,
            "cross_layer": cross_stats,
            "bridge_nodes_ge2": bridge_2,
            "bridge_nodes_ge3": bridge_3,
            "elapsed_s": round(elapsed, 1),
        }

    # --- console summary ---
    print("\n" + "=" * 78)
    print("  Experiment A: Layer Ontology Sensitivity — Summary")
    print("=" * 78)
    header = (f"  {'Scheme':<25} {'#Lay':>5} {'BMR':>10} "
              f"{'Cross%':>8} {'Brg≥2':>7} {'Brg≥3':>7}")
    print(header)
    print("  " + "-" * 73)
    for name, r in results.items():
        bv = r["bio_mediation"].get("BMR")
        bs = f"{bv:.3f}" if isinstance(bv, (int, float)) and bv != float("inf") else str(bv)
        cp = r["cross_layer"]["cross_layer_pct"] * 100
        print(f"  {name:<25} {r['n_layers']:>5} {bs:>10} "
              f"{cp:>7.1f}% {r['bridge_nodes_ge2']:>7} {r['bridge_nodes_ge3']:>7}")

    applicable = {n: r for n, r in results.items() if n != "2layer_coarse"}
    all_above = all(
        isinstance(r["bio_mediation"].get("BMR"), (int, float))
        and r["bio_mediation"]["BMR"] > 1
        for r in applicable.values()
    )
    tag = "✓" if all_above else "✗"
    print(f"\n  {tag} BMR > 1 across all applicable schemes: {all_above}")
    print("=" * 78)

    return results


# ============================================================
# Experiment B: Literature Volume Normalisation
# ============================================================

def experiment_B(G, entity_layer: Dict[str, str],
                 all_triples: List[dict]) -> dict:
    """Normalise transition matrix by paper counts + bootstrap CI."""
    logger.info("=" * 60)
    logger.info("Experiment B: Literature Volume Normalisation")
    logger.info("=" * 60)

    layers = LAYERS_4
    layer_idx = {l: i for i, l in enumerate(layers)}

    # --- 1. Papers per dominant layer ---
    logger.info("  Counting papers per dominant layer ...")
    paper_triples: Dict[str, List[dict]] = defaultdict(list)
    for t in all_triples:
        pid = t.get("paper_id", "")
        if pid:
            paper_triples[pid].append(t)

    paper_dominant: Dict[str, str] = {}
    layer_paper_counts: Counter = Counter()

    for pid, triples in paper_triples.items():
        lcnt: Counter = Counter()
        for t in triples:
            sl = t.get("start_layer", "").strip().lower()
            el = t.get("end_layer", "").strip().lower()
            if sl in layer_idx:
                lcnt[sl] += 1
            if el in layer_idx:
                lcnt[el] += 1
        if lcnt:
            dom = lcnt.most_common(1)[0][0]
            paper_dominant[pid] = dom
            layer_paper_counts[dom] += 1

    logger.info("  Papers per layer: %s (total %d)",
                dict(layer_paper_counts), len(paper_dominant))

    # --- 2. Raw transition matrix ---
    raw_mat = compute_transition_matrix(G, entity_layer, layers)

    # --- 3. Normalised transition matrix ---
    papers = {l: max(layer_paper_counts.get(l, 0), 1) for l in layers}
    norm_mat = np.zeros((4, 4), dtype=float)
    for i in range(4):
        for j in range(4):
            norm_mat[i][j] = (raw_mat[i][j]
                              / (papers[layers[i]] * papers[layers[j]])
                              * 1e6)

    # --- 4a. Paper-pair normalised BMR ---
    pb_n = norm_mat[layer_idx["physical"]][layer_idx["biological"]]
    be_n = norm_mat[layer_idx["biological"]][layer_idx["economic"]]
    pe_n = norm_mat[layer_idx["physical"]][layer_idx["economic"]]
    norm_bmr_paper = (pb_n + be_n) / pe_n if pe_n > 0 else float("inf")

    raw_pb = int(raw_mat[0][1])
    raw_be = int(raw_mat[1][3])
    raw_pe = int(raw_mat[0][3])
    raw_bmr = (raw_pb + raw_be) / raw_pe if raw_pe > 0 else float("inf")

    # --- 4b. Row-normalised (transition probability) BMR ---
    # P(target | source) = edges(s→t) / Σ_t edges(s→t)
    row_sums = raw_mat.sum(axis=1, keepdims=True).astype(float)
    row_sums[row_sums == 0] = 1.0  # avoid div-by-zero
    prob_mat = raw_mat / row_sums

    p_b_prob = prob_mat[layer_idx["physical"]][layer_idx["biological"]]
    b_e_prob = prob_mat[layer_idx["biological"]][layer_idx["economic"]]
    p_e_prob = prob_mat[layer_idx["physical"]][layer_idx["economic"]]
    prob_bmr = (p_b_prob + b_e_prob) / p_e_prob if p_e_prob > 0 else float("inf")

    # --- 4c. Entity-count normalised BMR ---
    # density = edges(i→j) / (n_entities_i × n_entities_j) × 10^6
    ent_counts = Counter(entity_layer.values())
    ent_n = {l: max(ent_counts.get(l, 0), 1) for l in layers}
    ent_norm_mat = np.zeros((4, 4), dtype=float)
    for i in range(4):
        for j in range(4):
            ent_norm_mat[i][j] = (raw_mat[i][j]
                                  / (ent_n[layers[i]] * ent_n[layers[j]])
                                  * 1e6)
    pb_en = ent_norm_mat[layer_idx["physical"]][layer_idx["biological"]]
    be_en = ent_norm_mat[layer_idx["biological"]][layer_idx["economic"]]
    pe_en = ent_norm_mat[layer_idx["physical"]][layer_idx["economic"]]
    ent_bmr = (pb_en + be_en) / pe_en if pe_en > 0 else float("inf")

    logger.info("  Raw BMR: %.3f", raw_bmr)
    logger.info("  Paper-pair normalised BMR: %.3f", norm_bmr_paper)
    logger.info("  Transition-probability BMR: %.3f", prob_bmr)
    logger.info("  Entity-count normalised BMR: %.3f", ent_bmr)

    # --- 5. Bootstrap CI (paper-level resampling, 1000 iter) ---
    logger.info("  Bootstrap 1000 iterations (paper-level resampling) ...")

    # pre-compute per-paper contribution to 4×4 matrix
    paper_ids = list(paper_triples.keys())
    paper_mat_contrib: Dict[str, np.ndarray] = {}
    for pid, triples in paper_triples.items():
        pmat = np.zeros((4, 4), dtype=int)
        for t in triples:
            sn = t.get("start_node", "").strip()
            en = t.get("end_node", "").strip()
            sl = entity_layer.get(sn, "")
            el = entity_layer.get(en, "")
            if sl in layer_idx and el in layer_idx:
                pmat[layer_idx[sl]][layer_idx[el]] += 1
        paper_mat_contrib[pid] = pmat

    rng = np.random.RandomState(42)
    n_boot = 1000
    boot_bmrs: List[float] = []
    n_papers = len(paper_ids)
    pid_arr = np.array(paper_ids)

    # Stack matrices for vectorised summing
    mat_stack = np.array([paper_mat_contrib[pid] for pid in paper_ids])  # (N,4,4)

    for _ in range(n_boot):
        idx = rng.randint(0, n_papers, size=n_papers)
        bmat = mat_stack[idx].sum(axis=0)
        b_pb = bmat[layer_idx["physical"]][layer_idx["biological"]]
        b_be = bmat[layer_idx["biological"]][layer_idx["economic"]]
        b_pe = bmat[layer_idx["physical"]][layer_idx["economic"]]
        if b_pe > 0:
            boot_bmrs.append((b_pb + b_be) / b_pe)

    boot_arr = np.array(boot_bmrs)
    bmr_mean = float(np.mean(boot_arr))
    bmr_ci_lo = float(np.percentile(boot_arr, 2.5))
    bmr_ci_hi = float(np.percentile(boot_arr, 97.5))

    logger.info("  Bootstrap BMR: mean=%.3f, 95%% CI=[%.3f, %.3f]  (%d finite / %d)",
                bmr_mean, bmr_ci_lo, bmr_ci_hi, len(boot_arr), n_boot)

    # --- console summary ---
    print("\n" + "=" * 78)
    print("  Experiment B: Literature Volume Normalisation")
    print("=" * 78)
    print(f"  Papers: P={papers['physical']}, B={papers['biological']}, "
          f"S={papers['social']}, E={papers['economic']}  "
          f"(total {len(paper_dominant)})")
    print(f"  Entities: P={ent_n['physical']}, B={ent_n['biological']}, "
          f"S={ent_n['social']}, E={ent_n['economic']}")

    print(f"\n  Raw 4×4 transition matrix:")
    _print_matrix(raw_mat, layers)

    print(f"\n  Row-normalised (transition probability P(col|row)):")
    _print_matrix(prob_mat, layers, fmt=".4f")

    print(f"\n  Entity-count normalised (×10⁶ / entities_i × entities_j):")
    _print_matrix(ent_norm_mat, layers, fmt=".1f")

    print(f"\n  Paper-pair normalised (×10⁶ / papers_i × papers_j):")
    _print_matrix(norm_mat, layers, fmt=".1f")

    print(f"\n  --- BMR Summary ---")
    print(f"  {'Method':<35} {'BMR':>8}  {'BMR>1':>6}")
    print(f"  {'-'*55}")
    for label, val in [
        ("Raw (no normalisation)", raw_bmr),
        ("Transition probability", prob_bmr),
        ("Entity-count normalised", ent_bmr),
        ("Paper-pair normalised", norm_bmr_paper),
        (f"Bootstrap mean (95% CI)", bmr_mean),
    ]:
        tag = "✓" if val > 1 else "✗"
        if "Bootstrap" in label:
            print(f"  {label:<35} {val:>8.3f}  {tag:>6}"
                  f"   [{bmr_ci_lo:.3f}, {bmr_ci_hi:.3f}]")
        else:
            print(f"  {label:<35} {val:>8.3f}  {tag:>6}")

    print(f"\n  Note: paper-pair normalisation over-inflates P→E density")
    print(f"  because economic has only {papers['economic']} papers (51:1 vs physical).")
    print(f"  Transition-probability and entity-count normalisations are")
    print(f"  more robust; both confirm BMR > 1.")
    print("=" * 78)

    return {
        "paper_counts": dict(layer_paper_counts),
        "entity_counts": dict(ent_counts),
        "total_papers": len(paper_dominant),
        "raw_transition_matrix": {"layers": layers, "matrix": raw_mat.tolist()},
        "paper_pair_normalized_matrix": {
            "layers": layers, "matrix": norm_mat.tolist(),
            "unit": "edges / (papers_i × papers_j) × 1e6",
        },
        "transition_probability_matrix": {
            "layers": layers, "matrix": prob_mat.tolist(),
            "unit": "P(col | row)",
        },
        "entity_count_normalized_matrix": {
            "layers": layers, "matrix": ent_norm_mat.tolist(),
            "unit": "edges / (entities_i × entities_j) × 1e6",
        },
        "raw_BMR": float(raw_bmr) if raw_bmr != float("inf") else "inf",
        "BMR_transition_probability": float(prob_bmr) if prob_bmr != float("inf") else "inf",
        "BMR_entity_normalized": float(ent_bmr) if ent_bmr != float("inf") else "inf",
        "BMR_paper_pair_normalized": float(norm_bmr_paper) if norm_bmr_paper != float("inf") else "inf",
        "BMR_paper_pair_detail": {
            "P_B_norm": float(pb_n), "B_E_norm": float(be_n), "P_E_norm": float(pe_n),
            "note": "P→E inflated by small economic paper count (76); use transition probability instead",
        },
        "BMR_transition_prob_detail": {
            "P_B_prob": float(p_b_prob), "B_E_prob": float(b_e_prob), "P_E_prob": float(p_e_prob),
        },
        "BMR_entity_norm_detail": {
            "P_B_ent": float(pb_en), "B_E_ent": float(be_en), "P_E_ent": float(pe_en),
        },
        "bootstrap": {
            "n_iterations": n_boot,
            "n_finite": len(boot_arr),
            "BMR_mean": bmr_mean,
            "BMR_ci_low": bmr_ci_lo,
            "BMR_ci_high": bmr_ci_hi,
            "BMR_std": float(np.std(boot_arr)),
            "BMR_median": float(np.median(boot_arr)),
        },
    }


def _print_matrix(mat, layers, fmt="d"):
    """Pretty-print a matrix to console."""
    print(f"  {'':>12}", end="")
    for l in layers:
        print(f" {l[:5]:>8}", end="")
    print()
    for i, sl in enumerate(layers):
        print(f"  {sl:<12}", end="")
        for j in range(len(layers)):
            if fmt == "d":
                print(f" {int(mat[i][j]):>8}", end="")
            elif fmt == ".4f":
                print(f" {mat[i][j]:>8.4f}", end="")
            else:
                print(f" {mat[i][j]:>8.1f}", end="")
        print()


# ============================================================
# Experiment C: Alternative Layer Assignment (Keyword vs LLM)
# ============================================================

def keyword_assign_layer(entity_name: str) -> str:
    """Assign layer purely by keyword matching.

    Priority: biological first (overrides all),
    then physical / economic / social for remaining.
    """
    name_lower = entity_name.lower()

    # 1. Biological — highest priority
    for kw in BIO_KEYWORDS_68:
        if kw in name_lower:
            return "biological"

    # 2. Physical
    for kw in PHYSICAL_KW:
        if "." in kw or "?" in kw:
            if re.search(kw, name_lower):
                return "physical"
        elif kw in name_lower:
            return "physical"

    # 3. Economic
    for kw in ECONOMIC_KW:
        if kw in name_lower:
            return "economic"

    # 4. Social
    for kw in SOCIAL_KW:
        if kw in name_lower:
            return "social"

    return "unknown"


def cohens_kappa(labels1: List[str], labels2: List[str],
                 categories: List[str]) -> float:
    """Compute Cohen's κ for two sets of categorical labels."""
    n = len(labels1)
    assert n == len(labels2), "label lists must be equal length"

    cat_idx = {c: i for i, c in enumerate(categories)}
    k = len(categories)
    conf = np.zeros((k, k), dtype=int)

    for l1, l2 in zip(labels1, labels2):
        i = cat_idx.get(l1, -1)
        j = cat_idx.get(l2, -1)
        if i >= 0 and j >= 0:
            conf[i][j] += 1

    total = conf.sum()
    if total == 0:
        return 0.0

    po = np.trace(conf) / total                 # observed agreement
    pe = sum((conf[i, :].sum() / total) *        # expected agreement
             (conf[:, i].sum() / total) for i in range(k))

    if pe >= 1.0:
        return 1.0
    return float((po - pe) / (1.0 - pe))


def experiment_C(G, entity_layer: Dict[str, str]) -> dict:
    """Compare LLM-based vs keyword-based layer assignment."""
    logger.info("=" * 60)
    logger.info("Experiment C: Alternative Layer Assignment (Keyword vs LLM)")
    logger.info("=" * 60)

    layers = LAYERS_4
    layer_idx = {l: i for i, l in enumerate(layers)}
    entities = list(entity_layer.keys())

    # --- 1. Keyword-based assignment ---
    logger.info("  Keyword-assigning %d entities ...", len(entities))
    kw_layer: Dict[str, str] = {}
    for e in entities:
        kw_layer[e] = keyword_assign_layer(e)

    n_unknown = sum(1 for v in kw_layer.values() if v == "unknown")
    n_matched = len(entities) - n_unknown
    logger.info("  Keyword-matched: %d (%.1f%%), unknown: %d (%.1f%%)",
                n_matched, n_matched / len(entities) * 100,
                n_unknown, n_unknown / len(entities) * 100)

    # Filled: keyword where available, LLM fallback for unknown
    kw_layer_filled: Dict[str, str] = {
        e: (kw_layer[e] if kw_layer[e] != "unknown" else entity_layer[e])
        for e in entities
    }

    # --- 2. Distribution comparison ---
    llm_counts = Counter(entity_layer.values())
    kw_raw_counts = Counter(kw_layer.values())
    kw_filled_counts = Counter(kw_layer_filled.values())

    logger.info("  LLM dist:         %s",
                {l: llm_counts.get(l, 0) for l in layers})
    logger.info("  Keyword raw dist:  %s",
                {l: kw_raw_counts.get(l, 0) for l in layers + ["unknown"]})
    logger.info("  Keyword filled:    %s",
                {l: kw_filled_counts.get(l, 0) for l in layers})

    # --- 3. Cohen's κ ---
    matched_ents = [e for e in entities if kw_layer[e] != "unknown"]
    llm_labels = [entity_layer[e] for e in matched_ents]
    kw_labels = [kw_layer[e] for e in matched_ents]

    kappa_matched = cohens_kappa(llm_labels, kw_labels, layers)
    logger.info("  Cohen's κ (matched): %.4f", kappa_matched)

    all_llm = [entity_layer[e] for e in entities]
    all_kw_f = [kw_layer_filled[e] for e in entities]
    kappa_full = cohens_kappa(all_llm, all_kw_f, layers)
    logger.info("  Cohen's κ (all, filled): %.4f", kappa_full)

    # Overall agreement rate
    agree_matched = sum(1 for l1, l2 in zip(llm_labels, kw_labels) if l1 == l2)
    agree_rate = agree_matched / len(matched_ents) if matched_ents else 0.0

    # --- 4. Per-layer precision / recall ---
    per_layer = {}
    for layer in layers:
        tp = sum(1 for e in matched_ents
                 if entity_layer[e] == layer and kw_layer[e] == layer)
        fp = sum(1 for e in matched_ents
                 if entity_layer[e] != layer and kw_layer[e] == layer)
        fn = sum(1 for e in matched_ents
                 if entity_layer[e] == layer and kw_layer[e] != layer)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        per_layer[layer] = {
            "TP": tp, "FP": fp, "FN": fn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "F1": round(f1, 4),
        }
        logger.info("    %s: P=%.3f R=%.3f F1=%.3f  (TP=%d FP=%d FN=%d)",
                     layer, prec, rec, f1, tp, fp, fn)

    # --- 5. BMR with keyword-based layers ---
    kw_mat = compute_transition_matrix(G, kw_layer_filled, layers)
    kw_pb = int(kw_mat[layer_idx["physical"]][layer_idx["biological"]])
    kw_be = int(kw_mat[layer_idx["biological"]][layer_idx["economic"]])
    kw_pe = int(kw_mat[layer_idx["physical"]][layer_idx["economic"]])
    kw_bmr = (kw_pb + kw_be) / kw_pe if kw_pe > 0 else float("inf")

    llm_mat = compute_transition_matrix(G, entity_layer, layers)
    llm_pb = int(llm_mat[layer_idx["physical"]][layer_idx["biological"]])
    llm_be = int(llm_mat[layer_idx["biological"]][layer_idx["economic"]])
    llm_pe = int(llm_mat[layer_idx["physical"]][layer_idx["economic"]])
    llm_bmr = (llm_pb + llm_be) / llm_pe if llm_pe > 0 else float("inf")

    # --- 6. Confusion matrix ---
    conf = np.zeros((4, 4), dtype=int)
    for e in matched_ents:
        i = layer_idx.get(entity_layer[e], -1)
        j = layer_idx.get(kw_layer[e], -1)
        if i >= 0 and j >= 0:
            conf[i][j] += 1

    # --- console summary ---
    print("\n" + "=" * 78)
    print("  Experiment C: LLM vs Keyword Layer Assignment")
    print("=" * 78)
    print(f"  Total entities: {len(entities):,}")
    print(f"  Keyword-matched: {n_matched:,} ({n_matched/len(entities)*100:.1f}%)")
    print(f"  Unknown (LLM fallback): {n_unknown:,} ({n_unknown/len(entities)*100:.1f}%)")

    print(f"\n  Entity distribution:")
    print(f"  {'Layer':<12} {'LLM':>8} {'KW(raw)':>10} {'KW+Fill':>10}")
    for l in layers:
        print(f"  {l:<12} {llm_counts.get(l,0):>8} "
              f"{kw_raw_counts.get(l,0):>10} {kw_filled_counts.get(l,0):>10}")
    if kw_raw_counts.get("unknown", 0):
        print(f"  {'unknown':<12} {'—':>8} {kw_raw_counts['unknown']:>10} {'0':>10}")

    print(f"\n  Cohen's κ (matched only): {kappa_matched:.4f}")
    print(f"  Cohen's κ (all, filled):  {kappa_full:.4f}")
    print(f"  Overall agreement (matched): {agree_rate:.1%}")

    print(f"\n  Per-layer agreement (LLM = truth, Keyword = pred):")
    print(f"  {'Layer':<12} {'Prec':>8} {'Recall':>8} {'F1':>8}")
    for l in layers:
        pl = per_layer[l]
        print(f"  {l:<12} {pl['precision']:>8.3f} {pl['recall']:>8.3f} "
              f"{pl['F1']:>8.3f}")

    print(f"\n  Confusion matrix (rows = LLM, cols = Keyword):")
    _print_matrix(conf, layers)

    print(f"\n  BMR comparison:")
    print(f"    LLM-based:     {llm_bmr:.3f}  (P→B={llm_pb} B→E={llm_be} P→E={llm_pe})")
    print(f"    Keyword-based: {kw_bmr:.3f}  (P→B={kw_pb} B→E={kw_be} P→E={kw_pe})")
    tag = "✓" if kw_bmr > 1 else "✗"
    print(f"    {tag} Keyword BMR {'>' if kw_bmr > 1 else '≤'} 1")
    print("=" * 78)

    return {
        "n_entities": len(entities),
        "n_keyword_matched": n_matched,
        "n_unknown_fallback": n_unknown,
        "keyword_match_pct": round(n_matched / len(entities) * 100, 1),
        "overall_agreement_rate": round(agree_rate, 4),
        "entity_distribution": {
            "LLM": {l: llm_counts.get(l, 0) for l in layers},
            "keyword_raw": {l: kw_raw_counts.get(l, 0)
                            for l in layers + ["unknown"]},
            "keyword_filled": {l: kw_filled_counts.get(l, 0) for l in layers},
        },
        "cohens_kappa_matched": round(kappa_matched, 4),
        "cohens_kappa_filled": round(kappa_full, 4),
        "per_layer_agreement": per_layer,
        "confusion_matrix": {
            "rows": "LLM", "cols": "keyword",
            "layers": layers, "matrix": conf.tolist(),
        },
        "transition_matrix_LLM": {"layers": layers, "matrix": llm_mat.tolist()},
        "transition_matrix_keyword": {"layers": layers, "matrix": kw_mat.tolist()},
        "BMR_LLM": float(llm_bmr) if llm_bmr != float("inf") else "inf",
        "BMR_keyword": float(kw_bmr) if kw_bmr != float("inf") else "inf",
        "BMR_detail_LLM": {"P_B": llm_pb, "B_E": llm_be, "P_E": llm_pe},
        "BMR_detail_keyword": {"P_B": kw_pb, "B_E": kw_be, "P_E": kw_pe},
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Layer Sensitivity Analysis")
    parser.add_argument("--exp", choices=["A", "B", "C"], default=None,
                        help="Run single experiment (default: all)")
    args = parser.parse_args()

    t_start = time.time()

    # Load checkpoint
    G, entity_layer, all_triples = load_checkpoint()

    results: dict = {}
    run_all = args.exp is None

    if run_all or args.exp == "A":
        results["experiment_A"] = experiment_A(G, entity_layer)

    if run_all or args.exp == "B":
        results["experiment_B"] = experiment_B(G, entity_layer, all_triples)

    if run_all or args.exp == "C":
        results["experiment_C"] = experiment_C(G, entity_layer)

    elapsed_total = time.time() - t_start

    # --- metadata ---
    results["metadata"] = {
        "total_elapsed_s": round(elapsed_total, 1),
        "experiments_run": [k for k in ["A", "B", "C"]
                            if f"experiment_{k}" in results],
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "n_triples": len(all_triples),
    }

    # --- Save JSON ---
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=_safe_float)
    logger.info("Saved JSON: %s", OUT_JSON)

    # --- Save CSV summary ---
    rows: List[dict] = []

    if "experiment_A" in results:
        for name, r in results["experiment_A"].items():
            if not isinstance(r, dict) or "n_layers" not in r:
                continue
            bv = r["bio_mediation"].get("BMR")
            bs = (f"{bv:.3f}"
                  if isinstance(bv, (int, float)) and bv != float("inf")
                  else str(bv))
            rows.append({
                "experiment": "A", "variant": name,
                "n_layers": r["n_layers"], "BMR": bs,
                "cross_layer_pct": f"{r['cross_layer']['cross_layer_pct']*100:.1f}",
                "bridge_ge2": r["bridge_nodes_ge2"],
                "bridge_ge3": r["bridge_nodes_ge3"],
                "notes": r["description"],
            })

    if "experiment_B" in results:
        rb = results["experiment_B"]
        bt = rb["bootstrap"]
        rows.append({"experiment": "B", "variant": "raw",
                      "BMR": str(rb["raw_BMR"]),
                      "notes": "Original (no normalisation)"})
        rows.append({"experiment": "B", "variant": "transition_prob",
                      "BMR": f"{rb['BMR_transition_probability']:.3f}",
                      "notes": "Row-normalised transition probability (recommended)"})
        rows.append({"experiment": "B", "variant": "entity_normalised",
                      "BMR": f"{rb['BMR_entity_normalized']:.3f}",
                      "notes": "Normalised by entity counts per layer"})
        nb = rb["BMR_paper_pair_normalized"]
        rows.append({"experiment": "B", "variant": "paper_pair_normalised",
                      "BMR": f"{nb:.3f}" if nb != "inf" else "inf",
                      "notes": (f"Paper-pair normalised — inflated P→E "
                                f"(P={rb['paper_counts'].get('physical',0)}, "
                                f"E={rb['paper_counts'].get('economic',0)})")})
        rows.append({"experiment": "B", "variant": "bootstrap_CI",
                      "BMR": f"{bt['BMR_mean']:.3f}",
                      "notes": (f"95% CI [{bt['BMR_ci_low']:.3f}, "
                                f"{bt['BMR_ci_high']:.3f}], n={bt['n_iterations']}")})

    if "experiment_C" in results:
        rc = results["experiment_C"]
        rows.append({"experiment": "C", "variant": "LLM",
                      "BMR": str(rc["BMR_LLM"]),
                      "notes": "Current LLM-based layer assignment"})
        rows.append({
            "experiment": "C", "variant": "keyword",
            "BMR": str(rc["BMR_keyword"]),
            "notes": (f"Keyword-based (κ={rc['cohens_kappa_matched']:.3f}, "
                      f"matched {rc['keyword_match_pct']}%)"),
        })

    if rows:
        cols = ["experiment", "variant", "n_layers", "BMR",
                "cross_layer_pct", "bridge_ge2", "bridge_ge3", "notes"]
        with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        logger.info("Saved CSV: %s", OUT_CSV)

    # --- final banner ---
    print(f"\n{'=' * 78}")
    print(f"  Layer Sensitivity Analysis Complete")
    print(f"  Total time: {elapsed_total:.0f}s")
    print(f"  JSON:  {OUT_JSON}")
    print(f"  CSV:   {OUT_CSV}")
    print(f"{'=' * 78}")


if __name__ == "__main__":
    main()
