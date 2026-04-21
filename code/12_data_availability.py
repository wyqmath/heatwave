#!/usr/bin/env python3
"""
12_data_availability.py
-----------------------
Generate Supplementary Material CSV for Data Availability statement:
Extract bibliographic metadata from WoS-exported paper.txt, output Dataset_S1_WOS_Records.csv.

Search strategy:
  - Database: Web of Science Core Collection
  - Query: heatwave (Topic field)
  - Search date: 2026-04-08
  - Results: 8,365 records
  - Export format: Plain text, Full Record
  - Temporal split: <=2024 (KG_Construction) / >=2025 (Holdout_QA)
"""

import csv
import re
import sys
from pathlib import Path
from collections import Counter

# ── paths ──────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_FILE = SCRIPT_DIR.parent / "data" / "raw" / "paper.txt"
OUTPUT_CSV = SCRIPT_DIR.parent / "results" / "Dataset_S1_WOS_Records.csv"


def parse_field(doc: str, tag: str) -> str | None:
    """Extract a WOS field value. Handles multi-line continuation (leading spaces)."""
    m = re.search(rf'\n{tag} (.*?)(?=\n[A-Z]{{2}} )', doc, re.DOTALL)
    if not m:
        return None
    # collapse multi-line: replace newline + leading spaces with single space
    return re.sub(r'\s+', ' ', m.group(1)).strip()


def extract_year(doc: str) -> int | None:
    """Extract publication year: PY > EA > PD."""
    py = re.search(r'\nPY (\d{4})', doc)
    if py:
        return int(py.group(1))
    ea = re.search(r'\nEA .*?(\d{4})', doc)
    if ea:
        return int(ea.group(1))
    pd = re.search(r'\nPD .*?(\d{4})', doc)
    if pd:
        return int(pd.group(1))
    return None


def parse_records(raw: str) -> list[dict]:
    """Parse all WOS records from raw text."""
    docs = re.split(r'\n(?=PT )', raw)
    records = []
    for doc in docs:
        if not doc.strip().startswith("PT "):
            continue

        # UT (WOS ID)
        ut_m = re.search(r'\nUT (WOS:\S+)', doc)
        wos_id = ut_m.group(1) if ut_m else ""

        # TI (Title)
        title = parse_field(doc, "TI") or ""

        # Year
        year = extract_year(doc)

        # SO (Journal)
        journal = parse_field(doc, "SO") or ""

        # DI (DOI)
        doi = parse_field(doc, "DI") or ""

        # Split
        if year is not None and year <= 2024:
            split = "KG_Construction"
        else:
            split = "Holdout_QA"

        records.append({
            "WOS_ID": wos_id,
            "Title": title,
            "Year": year if year is not None else "",
            "Journal": journal,
            "DOI": doi,
            "Split": split,
        })
    return records


def write_csv(records: list[dict], path: Path) -> None:
    """Write CSV sorted by Year desc, then WOS_ID."""
    records.sort(key=lambda r: (-int(r["Year"]) if r["Year"] != "" else 0, r["WOS_ID"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["WOS_ID", "Title", "Year", "Journal", "DOI", "Split"])
        writer.writeheader()
        writer.writerows(records)


def print_stats(records: list[dict]) -> None:
    """Print summary statistics for paper / figure captions."""
    n = len(records)
    split_counts = Counter(r["Split"] for r in records)
    years = [r["Year"] for r in records if r["Year"] != ""]
    doi_count = sum(1 for r in records if r["DOI"])
    journals = set(r["Journal"] for r in records if r["Journal"])

    print("=" * 60)
    print("Dataset_S1_WOS_Records  —  Summary Statistics")
    print("=" * 60)
    print(f"Total records:       {n}")
    for s in ["KG_Construction", "Holdout_QA"]:
        print(f"  {s:20s} {split_counts.get(s, 0)}")
    print(f"Year range:          {min(years)}–{max(years)}")
    print(f"DOI coverage:        {doi_count}/{n} ({doi_count/n*100:.1f}%)")
    print(f"Unique journals:     {len(journals)}")

    # WOS_ID checks
    ids = [r["WOS_ID"] for r in records]
    empty_ids = sum(1 for i in ids if not i)
    dup_ids = n - len(set(ids))
    print(f"Empty WOS_ID:        {empty_ids}")
    print(f"Duplicate WOS_ID:    {dup_ids}")

    # Year distribution (top 10)
    year_counts = Counter(years)
    print("\nYear distribution (top 10):")
    for yr, cnt in year_counts.most_common(10):
        print(f"  {yr}  {cnt:>5d}")

    print("\n" + "=" * 60)
    print("Search strategy (copy to Data Availability):")
    print("-" * 60)
    print("Database: Web of Science Core Collection")
    print('Query: TS = "heatwave"')
    print("Search date: 2026-04-08")
    print(f"Total results: {n}")
    print("Export: Plain text, Full Record")
    print(f"Split: ≤2024 → KG_Construction ({split_counts.get('KG_Construction',0)} papers)")
    print(f"       ≥2025 → Holdout_QA      ({split_counts.get('Holdout_QA',0)} papers)")
    print("=" * 60)


def main():
    if not INPUT_FILE.exists():
        print(f"ERROR: Input file not found: {INPUT_FILE}", file=sys.stderr)
        sys.exit(1)

    raw = INPUT_FILE.read_text(encoding="utf-8")
    records = parse_records(raw)
    write_csv(records, OUTPUT_CSV)
    print(f"Wrote {len(records)} records → {OUTPUT_CSV}")
    print_stats(records)


if __name__ == "__main__":
    main()
