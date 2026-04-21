"""
5_fulltext_compare.py
Compare Abstract-only vs Full-text triple extraction (TODO-4a, Step 5).

Metrics:
    - Per-paper: relation count, layer distribution, cross-layer edges
    - Global: 4x4 layer transition matrix, bio-mediation ratio
    - Statistical tests: Wilcoxon signed-rank, chi-square, bootstrap CI

Usage:
    cd workspace/code
    python 5_fulltext_compare.py
"""

import csv
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

# ---------- paths ----------
BASE_DIR = Path(__file__).parent.parent
FULLTEXT_DIR = BASE_DIR / "data" / "fulltext"
KG_DIR = BASE_DIR / "data" / "processed" / "kg"
EXTRACTED_DIR = FULLTEXT_DIR / "extracted"
SAMPLE_CSV = FULLTEXT_DIR / "sample_list.csv"
OUT_JSON = FULLTEXT_DIR / "comparison_results.json"
OUT_CSV = FULLTEXT_DIR / "comparison_summary.csv"

LAYERS = ["physical", "biological", "social", "economic"]
LAYER_SET = set(LAYERS)

# ---------- logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(FULLTEXT_DIR / "comparison.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ============================================================
# Helpers
# ============================================================

def load_rels(json_path: Path) -> list[dict]:
    """Load relations from a JSON file. Returns [] if missing."""
    if not json_path.exists():
        return []
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return []


def layer_counts(rels: list[dict]) -> Counter:
    """Count how many times each layer appears (as start or end)."""
    c = Counter()
    for r in rels:
        sl = r.get("start_layer", "")
        el = r.get("end_layer", "")
        if sl in LAYER_SET:
            c[sl] += 1
        if el in LAYER_SET:
            c[el] += 1
    return c


def transition_matrix(rels: list[dict]) -> np.ndarray:
    """Build 4x4 layer transition matrix (start_layer -> end_layer)."""
    mat = np.zeros((4, 4), dtype=int)
    layer_idx = {l: i for i, l in enumerate(LAYERS)}
    for r in rels:
        sl = r.get("start_layer", "")
        el = r.get("end_layer", "")
        if sl in layer_idx and el in layer_idx:
            mat[layer_idx[sl]][layer_idx[el]] += 1
    return mat


def cross_layer_count(rels: list[dict]) -> int:
    """Count relations where start_layer != end_layer."""
    return sum(
        1 for r in rels
        if r.get("start_layer", "") != r.get("end_layer", "")
        and r.get("start_layer", "") in LAYER_SET
        and r.get("end_layer", "") in LAYER_SET
    )


def directional_count(rels: list[dict], from_layer: str, to_layer: str) -> int:
    """Count relations from from_layer to to_layer."""
    return sum(
        1 for r in rels
        if r.get("start_layer", "") == from_layer
        and r.get("end_layer", "") == to_layer
    )


def bio_mediation_ratio(rels: list[dict]) -> float:
    """Compute (phys->bio + bio->econ) / phys->econ.

    Returns float; inf if phys->econ == 0 but numerator > 0; 0 if both 0.
    """
    pb = directional_count(rels, "physical", "biological")
    be = directional_count(rels, "biological", "economic")
    pe = directional_count(rels, "physical", "economic")
    numerator = pb + be
    if pe == 0:
        return float("inf") if numerator > 0 else 0.0
    return numerator / pe


def bootstrap_ci(values: list[float], n_boot: int = 10000,
                 ci: float = 0.95, seed: int = 42) -> tuple[float, float, float]:
    """Bootstrap mean and CI. Returns (mean, ci_low, ci_high)."""
    rng = np.random.RandomState(seed)
    arr = np.array(values)
    boot_means = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boot_means.append(np.mean(sample))
    boot_means = np.array(boot_means)
    alpha = (1 - ci) / 2
    lo = np.percentile(boot_means, alpha * 100)
    hi = np.percentile(boot_means, (1 - alpha) * 100)
    return float(np.mean(arr)), float(lo), float(hi)


# ============================================================
# Main
# ============================================================

def main():
    # Load sample list
    with open(SAMPLE_CSV, "r", encoding="utf-8-sig") as f:
        samples = list(csv.DictReader(f))
    logger.info("Loaded %d samples", len(samples))

    # Per-paper results
    paper_results = []
    abs_all_rels = []
    ft_all_rels = []

    missing_abs = 0
    missing_ft = 0

    for sample in samples:
        paper_id = sample.get("paper_id", "").strip()
        safe_id = paper_id.replace(":", "_")
        dominant_layer = sample.get("dominant_layer", "")

        abs_path = KG_DIR / f"{safe_id}.json"
        ft_path = EXTRACTED_DIR / f"{safe_id}.json"

        abs_rels = load_rels(abs_path)
        ft_rels = load_rels(ft_path)

        if not abs_path.exists():
            missing_abs += 1
        if not ft_path.exists():
            missing_ft += 1

        abs_lc = layer_counts(abs_rels)
        ft_lc = layer_counts(ft_rels)

        result = {
            "paper_id": paper_id,
            "dominant_layer": dominant_layer,
            "abs_rel_count": len(abs_rels),
            "ft_rel_count": len(ft_rels),
            "abs_cross_layer": cross_layer_count(abs_rels),
            "ft_cross_layer": cross_layer_count(ft_rels),
            "abs_phys_econ": directional_count(abs_rels, "physical", "economic"),
            "ft_phys_econ": directional_count(ft_rels, "physical", "economic"),
            "abs_bio_med_ratio": bio_mediation_ratio(abs_rels),
            "ft_bio_med_ratio": bio_mediation_ratio(ft_rels),
            "abs_layer_dist": {l: abs_lc.get(l, 0) for l in LAYERS},
            "ft_layer_dist": {l: ft_lc.get(l, 0) for l in LAYERS},
        }
        paper_results.append(result)
        abs_all_rels.extend(abs_rels)
        ft_all_rels.extend(ft_rels)

    if missing_abs:
        logger.warning("Missing abstract JSONs: %d", missing_abs)
    if missing_ft:
        logger.warning("Missing full-text JSONs: %d", missing_ft)

    # ---- Global metrics ----
    n = len(paper_results)
    abs_counts = [r["abs_rel_count"] for r in paper_results]
    ft_counts = [r["ft_rel_count"] for r in paper_results]

    # 1. Relation count comparison
    abs_mean = np.mean(abs_counts)
    ft_mean = np.mean(ft_counts)
    abs_median = np.median(abs_counts)
    ft_median = np.median(ft_counts)

    # Wilcoxon signed-rank test (paired)
    try:
        wilcoxon_stat, wilcoxon_p = stats.wilcoxon(abs_counts, ft_counts)
    except ValueError:
        # All differences are zero
        wilcoxon_stat, wilcoxon_p = 0, 1.0

    # 2. Layer transition matrices
    abs_tm = transition_matrix(abs_all_rels)
    ft_tm = transition_matrix(ft_all_rels)

    # 3. Layer distribution chi-square
    abs_layer_totals = [sum(r["abs_layer_dist"].get(l, 0) for r in paper_results) for l in LAYERS]
    ft_layer_totals = [sum(r["ft_layer_dist"].get(l, 0) for r in paper_results) for l in LAYERS]

    # Chi-square test on 2x4 contingency table
    contingency = np.array([abs_layer_totals, ft_layer_totals])
    # Remove zero columns to avoid chi2 issues
    nonzero_cols = contingency.sum(axis=0) > 0
    if nonzero_cols.sum() >= 2:
        chi2_stat, chi2_p, chi2_dof, _ = stats.chi2_contingency(contingency[:, nonzero_cols])
    else:
        chi2_stat, chi2_p, chi2_dof = 0, 1.0, 0

    # 4. Bio-mediation ratio (global)
    abs_bmr_global = bio_mediation_ratio(abs_all_rels)
    ft_bmr_global = bio_mediation_ratio(ft_all_rels)

    # Per-paper bio-mediation ratios (exclude inf for bootstrap)
    abs_bmrs = [r["abs_bio_med_ratio"] for r in paper_results if r["abs_bio_med_ratio"] != float("inf")]
    ft_bmrs = [r["ft_bio_med_ratio"] for r in paper_results if r["ft_bio_med_ratio"] != float("inf")]

    abs_bmr_boot = bootstrap_ci(abs_bmrs) if abs_bmrs else (0, 0, 0)
    ft_bmr_boot = bootstrap_ci(ft_bmrs) if ft_bmrs else (0, 0, 0)

    # 5. Cross-layer edge comparison
    abs_cross = [r["abs_cross_layer"] for r in paper_results]
    ft_cross = [r["ft_cross_layer"] for r in paper_results]
    try:
        cross_wilcox_stat, cross_wilcox_p = stats.wilcoxon(abs_cross, ft_cross)
    except ValueError:
        cross_wilcox_stat, cross_wilcox_p = 0, 1.0

    # ---- Print summary ----
    print("\n" + "=" * 70)
    print("  Abstract-only vs Full-text Extraction Comparison")
    print("=" * 70)

    print(f"\nPapers analyzed: {n}")
    print(f"Missing abstract JSONs: {missing_abs}, Missing full-text JSONs: {missing_ft}")

    print(f"\n--- Relation Counts ---")
    print(f"  Abstract:  mean={abs_mean:.1f}, median={abs_median:.1f}, total={sum(abs_counts)}")
    print(f"  Full-text: mean={ft_mean:.1f}, median={ft_median:.1f}, total={sum(ft_counts)}")
    print(f"  Ratio (FT/Abs): {ft_mean/abs_mean:.2f}x" if abs_mean > 0 else "  Ratio: N/A")
    print(f"  Wilcoxon signed-rank: W={wilcoxon_stat:.1f}, p={wilcoxon_p:.2e}")

    print(f"\n--- Layer Distribution ---")
    print(f"  {'Layer':<12} {'Abstract':>10} {'Full-text':>10} {'Abs%':>8} {'FT%':>8}")
    abs_total = sum(abs_layer_totals)
    ft_total = sum(ft_layer_totals)
    for i, layer in enumerate(LAYERS):
        abs_pct = abs_layer_totals[i] / abs_total * 100 if abs_total else 0
        ft_pct = ft_layer_totals[i] / ft_total * 100 if ft_total else 0
        print(f"  {layer:<12} {abs_layer_totals[i]:>10} {ft_layer_totals[i]:>10} {abs_pct:>7.1f}% {ft_pct:>7.1f}%")
    print(f"  Chi-square: χ²={chi2_stat:.2f}, df={chi2_dof}, p={chi2_p:.2e}")

    print(f"\n--- 4×4 Transition Matrix (start→end) ---")
    print(f"  Abstract:")
    print(f"  {'':>12}", end="")
    for l in LAYERS:
        print(f" {l[:4]:>6}", end="")
    print()
    for i, sl in enumerate(LAYERS):
        print(f"  {sl:<12}", end="")
        for j in range(4):
            print(f" {abs_tm[i][j]:>6}", end="")
        print()

    print(f"\n  Full-text:")
    print(f"  {'':>12}", end="")
    for l in LAYERS:
        print(f" {l[:4]:>6}", end="")
    print()
    for i, sl in enumerate(LAYERS):
        print(f"  {sl:<12}", end="")
        for j in range(4):
            print(f" {ft_tm[i][j]:>6}", end="")
        print()

    print(f"\n--- Cross-layer Edges ---")
    print(f"  Abstract:  mean={np.mean(abs_cross):.1f}, total={sum(abs_cross)}")
    print(f"  Full-text: mean={np.mean(ft_cross):.1f}, total={sum(ft_cross)}")
    print(f"  Wilcoxon: W={cross_wilcox_stat:.1f}, p={cross_wilcox_p:.2e}")

    print(f"\n--- Bio-mediation Ratio: (phys→bio + bio→econ) / phys→econ ---")
    print(f"  Abstract (global):  {abs_bmr_global:.2f}")
    print(f"  Full-text (global): {ft_bmr_global:.2f}")
    print(f"  Abstract (per-paper bootstrap):  mean={abs_bmr_boot[0]:.2f}, "
          f"95% CI=[{abs_bmr_boot[1]:.2f}, {abs_bmr_boot[2]:.2f}]")
    print(f"  Full-text (per-paper bootstrap): mean={ft_bmr_boot[0]:.2f}, "
          f"95% CI=[{ft_bmr_boot[1]:.2f}, {ft_bmr_boot[2]:.2f}]")

    # Key conclusion
    print(f"\n{'=' * 70}")
    if ft_bmr_global > 1:
        print("  ✓ Bio-ecological mediation persists in full-text extraction")
        print(f"    (ratio={ft_bmr_global:.2f} > 1, phys→bio→econ pathway dominant)")
    else:
        print("  ✗ Bio-ecological mediation NOT confirmed in full-text")
        print(f"    (ratio={ft_bmr_global:.2f} <= 1)")

    if wilcoxon_p < 0.05:
        print(f"  ✓ Relation counts differ significantly (p={wilcoxon_p:.2e})")
        print(f"    Full-text yields {ft_mean/abs_mean:.1f}x more relations" if abs_mean > 0 else "")
    else:
        print(f"  ○ No significant difference in relation counts (p={wilcoxon_p:.2e})")

    if chi2_p < 0.05:
        print(f"  ⚠ Layer distribution differs significantly (χ²={chi2_stat:.1f}, p={chi2_p:.2e})")
    else:
        print(f"  ✓ Layer distribution consistent (χ²={chi2_stat:.1f}, p={chi2_p:.2e})")
    print("=" * 70)

    # ---- Save results ----
    # JSON (full results)
    output = {
        "n_papers": n,
        "missing_abstract": missing_abs,
        "missing_fulltext": missing_ft,
        "relation_counts": {
            "abstract": {"mean": abs_mean, "median": abs_median, "total": sum(abs_counts)},
            "fulltext": {"mean": ft_mean, "median": ft_median, "total": sum(ft_counts)},
            "wilcoxon_W": float(wilcoxon_stat),
            "wilcoxon_p": float(wilcoxon_p),
        },
        "layer_distribution": {
            "abstract": {l: abs_layer_totals[i] for i, l in enumerate(LAYERS)},
            "fulltext": {l: ft_layer_totals[i] for i, l in enumerate(LAYERS)},
            "chi2": float(chi2_stat),
            "chi2_p": float(chi2_p),
            "chi2_dof": int(chi2_dof),
        },
        "transition_matrix": {
            "layers": LAYERS,
            "abstract": abs_tm.tolist(),
            "fulltext": ft_tm.tolist(),
        },
        "bio_mediation": {
            "abstract_global": abs_bmr_global if abs_bmr_global != float("inf") else "inf",
            "fulltext_global": ft_bmr_global if ft_bmr_global != float("inf") else "inf",
            "abstract_bootstrap": {"mean": abs_bmr_boot[0], "ci_low": abs_bmr_boot[1], "ci_high": abs_bmr_boot[2]},
            "fulltext_bootstrap": {"mean": ft_bmr_boot[0], "ci_low": ft_bmr_boot[1], "ci_high": ft_bmr_boot[2]},
        },
        "cross_layer": {
            "abstract": {"mean": float(np.mean(abs_cross)), "total": sum(abs_cross)},
            "fulltext": {"mean": float(np.mean(ft_cross)), "total": sum(ft_cross)},
            "wilcoxon_W": float(cross_wilcox_stat),
            "wilcoxon_p": float(cross_wilcox_p),
        },
        "per_paper": paper_results,
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    logger.info("Saved: %s", OUT_JSON)

    # CSV summary (one row per paper)
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "paper_id", "dominant_layer",
            "abs_rels", "ft_rels", "ratio",
            "abs_cross_layer", "ft_cross_layer",
            "abs_phys_econ", "ft_phys_econ",
            "abs_bio_med_ratio", "ft_bio_med_ratio",
            "abs_physical", "abs_biological", "abs_social", "abs_economic",
            "ft_physical", "ft_biological", "ft_social", "ft_economic",
        ])
        for r in paper_results:
            ratio = r["ft_rel_count"] / r["abs_rel_count"] if r["abs_rel_count"] > 0 else "N/A"
            abs_bmr = r["abs_bio_med_ratio"] if r["abs_bio_med_ratio"] != float("inf") else "inf"
            ft_bmr = r["ft_bio_med_ratio"] if r["ft_bio_med_ratio"] != float("inf") else "inf"
            writer.writerow([
                r["paper_id"], r["dominant_layer"],
                r["abs_rel_count"], r["ft_rel_count"],
                f"{ratio:.2f}" if isinstance(ratio, float) else ratio,
                r["abs_cross_layer"], r["ft_cross_layer"],
                r["abs_phys_econ"], r["ft_phys_econ"],
                abs_bmr, ft_bmr,
                r["abs_layer_dist"]["physical"], r["abs_layer_dist"]["biological"],
                r["abs_layer_dist"]["social"], r["abs_layer_dist"]["economic"],
                r["ft_layer_dist"]["physical"], r["ft_layer_dist"]["biological"],
                r["ft_layer_dist"]["social"], r["ft_layer_dist"]["economic"],
            ])
    logger.info("Saved: %s", OUT_CSV)


if __name__ == "__main__":
    main()
