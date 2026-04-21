"""
19_sample_fulltext.py
Stratified sampling of 80 papers (4 layers x 20) from KG set for full-text
comparison experiment (TODO-4a).

Usage:
    cd workspace/code
    python 19_sample_fulltext.py          # default: 80 papers (20 per layer)
    python 19_sample_fulltext.py --n 30   # 120 papers (30 per layer)

Output:
    ../data/fulltext/sample_list.csv   — sampling list with paper_id, title, year,
                                          dominant_layer, num_abstract_rels
    ../data/fulltext/sample_list.json  — same data in JSON format
"""

import json
import os
import re
import csv
import random
import argparse
import logging
from pathlib import Path
from collections import Counter

# ---------- config ----------
RANDOM_SEED = 42
KG_DIR = Path(__file__).parent.parent / "data" / "processed" / "kg"
RAW_FILE = Path(__file__).parent.parent / "data" / "raw" / "paper.txt"
OUT_DIR = Path(__file__).parent.parent / "data" / "fulltext"
LAYERS = ["physical", "biological", "social", "economic"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_wos_records(raw_path: Path) -> dict:
    """Parse paper.txt, return {paper_id: {title, year}} for KG papers only."""
    with open(raw_path, "r", encoding="utf-8") as f:
        raw = f.read()

    docs = re.split(r'\n(?=PT )', raw)
    records = {}

    for doc in docs:
        if not doc.strip().startswith("PT "):
            continue

        # Paper ID
        ut_match = re.search(r'\nUT (WOS:\S+)', doc)
        if not ut_match:
            continue
        paper_id = ut_match.group(1)

        # Year
        py_match = re.search(r'\nPY (\d{4})', doc)
        year = int(py_match.group(1)) if py_match else None

        # Only KG set (<=2024)
        if year is None or year > 2024:
            continue

        # Title
        ti_match = re.search(r'TI (.*?)(?=\n[A-Z]{2} )', doc, re.DOTALL)
        title = ""
        if ti_match:
            title = re.sub(r'\s+', ' ', ti_match.group(1).replace('\n', ' ')).strip()

        # DOI (for PDF download)
        doi_match = re.search(r'\nDI (.*?)(?=\n)', doc)
        doi = doi_match.group(1).strip() if doi_match else ""

        # WoS categories (for reference)
        sc_match = re.search(r'\nSC (.*?)(?=\n[A-Z]{2} )', doc, re.DOTALL)
        subject = ""
        if sc_match:
            subject = re.sub(r'\s+', ' ', sc_match.group(1).replace('\n', ' ')).strip()

        records[paper_id] = {
            "title": title,
            "year": year,
            "doi": doi,
            "subject": subject,
        }

    return records


def classify_papers(kg_dir: Path) -> dict:
    """Read KG JSONs, determine dominant layer for each paper.

    Returns {paper_id: {dominant_layer, num_rels, layer_dist}}
    """
    results = {}

    for fname in os.listdir(kg_dir):
        if not fname.endswith(".json"):
            continue

        fpath = kg_dir / fname
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not data:
            continue

        paper_id = data[0].get("paper_id", fname.replace(".json", "").replace("_", ":"))

        # Count layer mentions
        layer_counter = Counter()
        for rel in data:
            sl = rel.get("start_layer", "")
            el = rel.get("end_layer", "")
            if sl in LAYERS:
                layer_counter[sl] += 1
            if el in LAYERS:
                layer_counter[el] += 1

        if not layer_counter:
            continue

        dominant = layer_counter.most_common(1)[0][0]
        results[paper_id] = {
            "dominant_layer": dominant,
            "num_rels": len(data),
            "layer_dist": dict(layer_counter),
        }

    return results


def stratified_sample(classified: dict, per_layer: int, seed: int) -> list:
    """Stratified random sampling: per_layer papers from each of 4 layers.

    Returns list of paper_ids.
    """
    rng = random.Random(seed)

    by_layer = {layer: [] for layer in LAYERS}
    for pid, info in classified.items():
        layer = info["dominant_layer"]
        if layer in by_layer:
            by_layer[layer].append(pid)

    sampled = []
    for layer in LAYERS:
        pool = by_layer[layer]
        rng.shuffle(pool)
        n = min(per_layer, len(pool))
        chosen = pool[:n]
        sampled.extend(chosen)
        logger.info("Layer %-12s: pool=%d, sampled=%d", layer, len(pool), n)
        if n < per_layer:
            logger.warning("Layer %s: only %d available, requested %d", layer, n, per_layer)

    return sampled


def main():
    parser = argparse.ArgumentParser(description="Stratified sampling for full-text experiment")
    parser.add_argument("--n", type=int, default=20,
                        help="Number of papers per layer (default: 20, total = n*4)")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED,
                        help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()

    per_layer = args.n
    seed = args.seed
    total = per_layer * len(LAYERS)

    logger.info("=== Stratified Sampling for Full-text Experiment ===")
    logger.info("Random seed: %d", seed)
    logger.info("Per-layer: %d, Total: %d", per_layer, total)

    # Step 1: Parse WoS records
    logger.info("Parsing WoS records from %s ...", RAW_FILE)
    wos_records = parse_wos_records(RAW_FILE)
    logger.info("Parsed %d KG-set records (<=2024)", len(wos_records))

    # Step 2: Classify papers by dominant layer
    logger.info("Classifying papers by dominant layer from %s ...", KG_DIR)
    classified = classify_papers(KG_DIR)
    logger.info("Classified %d non-empty papers", len(classified))

    layer_dist = Counter(v["dominant_layer"] for v in classified.values())
    for layer in LAYERS:
        logger.info("  %-12s: %d papers", layer, layer_dist.get(layer, 0))

    # Step 3: Stratified sampling
    logger.info("Sampling %d per layer (seed=%d) ...", per_layer, seed)
    sampled_ids = stratified_sample(classified, per_layer, seed)

    # Step 4: Build output table
    rows = []
    for pid in sampled_ids:
        wos = wos_records.get(pid, {})
        cls = classified.get(pid, {})
        rows.append({
            "paper_id": pid,
            "title": wos.get("title", ""),
            "year": wos.get("year", ""),
            "doi": wos.get("doi", ""),
            "subject": wos.get("subject", ""),
            "dominant_layer": cls.get("dominant_layer", ""),
            "num_abstract_rels": cls.get("num_rels", 0),
            "layer_dist": json.dumps(cls.get("layer_dist", {})),
            "pdf_pages": "",        # user fills: e.g. "1-12"
            "pdf_downloaded": "",   # user fills: yes/no
        })

    # Step 5: Write outputs
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = OUT_DIR / "sample_list.csv"
    fieldnames = ["paper_id", "title", "year", "doi", "subject",
                   "dominant_layer", "num_abstract_rels", "layer_dist",
                   "pdf_pages", "pdf_downloaded"]
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("CSV written: %s", csv_path)

    # JSON
    json_path = OUT_DIR / "sample_list.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    logger.info("JSON written: %s", json_path)

    # Summary
    logger.info("=== Sampling Summary ===")
    logger.info("Total sampled: %d papers", len(rows))
    for layer in LAYERS:
        count = sum(1 for r in rows if r["dominant_layer"] == layer)
        logger.info("  %-12s: %d", layer, count)
    logger.info("Seed: %d (reproducible)", seed)
    logger.info("Next steps:")
    logger.info("  1. Open %s", csv_path)
    logger.info("  2. Download PDFs (use DOI column)")
    logger.info("  3. Fill 'pdf_pages' column (e.g. '1-12', skip refs/appendix)")
    logger.info("  4. Fill 'pdf_downloaded' column (yes/no)")


if __name__ == "__main__":
    main()
