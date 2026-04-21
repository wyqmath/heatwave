"""
3_pdf_to_markdown.py
Convert sampled PDFs to Markdown via Doc2X API (TODO-4a).

Pipeline:
    1. Read paper_range.csv for page ranges (skip appendix/references)
    2. Split each PDF to specified pages (PyPDF2)
    3. Upload split PDF to Doc2X API -> poll status -> get per-page markdown
    4. Concatenate pages and save as {paper_id}.md

Usage:
    cd workspace/code
    python 3_pdf_to_markdown.py                # process all 80
    python 3_pdf_to_markdown.py --test 1       # test with 1 file only
    python 3_pdf_to_markdown.py --start 10 --end 20  # process files 10-20
"""

import csv
import io
import json
import logging
import os
import time
from pathlib import Path

import requests
import yaml
from PyPDF2 import PdfReader, PdfWriter

# ---------- paths ----------
BASE_DIR = Path(__file__).parent.parent
CONFIG_DIR = BASE_DIR / "config"
FULLTEXT_DIR = BASE_DIR / "data" / "fulltext"
PDF_DIR = FULLTEXT_DIR / "paper"
MD_OUT_DIR = FULLTEXT_DIR / "markdown"
SAMPLE_CSV = FULLTEXT_DIR / "sample_list.csv"
RANGE_CSV = FULLTEXT_DIR / "paper_range.csv"

# ---------- API config ----------
_key_path = CONFIG_DIR / "api_keys.yaml"
with open(_key_path, "r", encoding="utf-8") as f:
    _api_keys = yaml.safe_load(f)
_doc2x = _api_keys.get("doc2x", {})
API_KEY = _doc2x.get("api_key", "")
BASE_URL = _doc2x.get("base_url", "")

# ---------- logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(FULLTEXT_DIR / "pdf_convert.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ============================================================
# PDF splitting
# ============================================================

def split_pdf(pdf_path: Path, page_range: str) -> bytes:
    """Split PDF to specified page range, return bytes of the new PDF.

    page_range: e.g. "1-10" means pages 1 through 10 (1-indexed, inclusive).
    """
    start_str, end_str = page_range.strip().split("-")
    start_page = int(start_str) - 1   # convert to 0-indexed
    end_page = int(end_str)            # exclusive upper bound for PyPDF2

    reader = PdfReader(str(pdf_path))
    writer = PdfWriter()

    total_pages = len(reader.pages)
    end_page = min(end_page, total_pages)

    for i in range(start_page, end_page):
        writer.add_page(reader.pages[i])

    buf = io.BytesIO()
    writer.write(buf)
    buf.seek(0)
    return buf.read()


# ============================================================
# Doc2X API
# ============================================================

def _headers():
    return {"Authorization": f"Bearer {API_KEY}"}


def preupload() -> dict:
    """Step 1: Get presigned upload URL."""
    url = f"{BASE_URL}/api/v2/parse/preupload"
    resp = requests.post(url, headers=_headers(), timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if data.get("code") != "success":
        raise RuntimeError(f"preupload failed: {data}")
    return data["data"]  # {uid, url}


def upload_pdf(upload_url: str, pdf_bytes: bytes):
    """Step 2: PUT pdf bytes to presigned URL."""
    resp = requests.put(upload_url, data=pdf_bytes, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(f"PUT upload failed: {resp.status_code} {resp.text}")


def poll_status(uid: str, max_wait: int = 300) -> dict:
    """Step 3: Poll until success/failed. Returns result dict."""
    url = f"{BASE_URL}/api/v2/parse/status?uid={uid}"
    start = time.time()

    while time.time() - start < max_wait:
        resp = requests.get(url, headers=_headers(), timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if data.get("code") != "success":
            raise RuntimeError(f"status query error: {data}")

        status = data["data"]["status"]
        if status == "success":
            return data["data"]["result"]
        elif status == "failed":
            detail = data["data"].get("detail", "unknown")
            raise RuntimeError(f"parse failed: {detail}")
        else:
            progress = data["data"].get("progress", 0)
            logger.debug("  uid=%s progress=%d%%", uid, progress)
            time.sleep(3)

    raise TimeoutError(f"Polling timed out after {max_wait}s for uid={uid}")


def convert_pdf_to_markdown(pdf_bytes: bytes) -> list[dict]:
    """Full pipeline: upload -> poll -> return list of page dicts.

    Each page dict has: page_idx, md, score
    """
    # Step 1
    upload_data = preupload()
    uid = upload_data["uid"]
    upload_url = upload_data["url"]

    # Step 2
    upload_pdf(upload_url, pdf_bytes)

    # Step 3: wait for server to pick up (~20s max)
    time.sleep(5)
    result = poll_status(uid)

    return result.get("pages", [])


# ============================================================
# Data loading
# ============================================================

def load_page_ranges(csv_path: Path) -> dict[int, str]:
    """Load paper_range.csv -> {1: '1-10', 2: '1-5', ...}"""
    ranges = {}
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        for idx, line in enumerate(f, 1):
            rng = line.strip().replace("\r", "")
            if rng:
                ranges[idx] = rng
    return ranges


def load_sample_list(csv_path: Path) -> list[dict]:
    """Load sample_list.csv, return list of dicts (1-indexed by position)."""
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader)


# ============================================================
# Main
# ============================================================

def process_one(idx: int, paper_id: str, page_range: str,
                pdf_dir: Path, out_dir: Path) -> bool:
    """Process a single PDF: split -> upload -> save markdown.

    Returns True on success, False on failure.
    """
    pdf_path = pdf_dir / f"{idx}.pdf"
    safe_id = paper_id.replace(":", "_")
    md_path = out_dir / f"{safe_id}.md"

    # Checkpoint: skip if already done
    if md_path.exists() and md_path.stat().st_size > 0:
        logger.info("[%d] Skip (already exists): %s", idx, safe_id)
        return True

    if not pdf_path.exists():
        logger.error("[%d] PDF not found: %s", idx, pdf_path)
        return False

    try:
        start_time = time.time()

        # Split PDF
        pdf_bytes = split_pdf(pdf_path, page_range)
        split_pages = int(page_range.split("-")[1]) - int(page_range.split("-")[0]) + 1
        logger.info("[%d] Split %s -> %d pages (%s)", idx, pdf_path.name, split_pages, page_range)

        # Upload & convert
        pages = convert_pdf_to_markdown(pdf_bytes)

        # Concatenate markdown
        md_parts = []
        for page in sorted(pages, key=lambda p: p.get("page_idx", 0)):
            page_md = page.get("md", "").strip()
            score = page.get("score", 0)
            if page_md:
                md_parts.append(f"<!-- page {page.get('page_idx', 0) + 1}, score: {score} -->\n\n{page_md}")

        full_md = "\n\n---\n\n".join(md_parts)

        # Save
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(full_md)

        elapsed = time.time() - start_time
        logger.info("[%d] Done: %s (%d pages, %.1fs)", idx, safe_id, len(pages), elapsed)
        return True

    except Exception as e:
        logger.error("[%d] Error processing %s: %s", idx, paper_id, e)
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Convert sampled PDFs to Markdown via Doc2X")
    parser.add_argument("--test", type=int, default=0, help="Only process first N files for testing")
    parser.add_argument("--start", type=int, default=1, help="Start index (1-based, inclusive)")
    parser.add_argument("--end", type=int, default=80, help="End index (1-based, inclusive)")
    args = parser.parse_args()

    if not API_KEY or API_KEY == "sk-xxx":
        logger.error("Doc2X API key not configured! Edit config/api_keys.yaml")
        return

    MD_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    page_ranges = load_page_ranges(RANGE_CSV)
    samples = load_sample_list(SAMPLE_CSV)

    logger.info("=== PDF to Markdown Conversion ===")
    logger.info("Total samples: %d, Page ranges: %d", len(samples), len(page_ranges))
    logger.info("Output dir: %s", MD_OUT_DIR)

    # Determine range
    start_idx = args.start
    end_idx = args.end
    if args.test > 0:
        end_idx = min(args.test, len(samples))
        start_idx = 1

    success = 0
    fail = 0

    for idx in range(start_idx, end_idx + 1):
        if idx > len(samples) or idx not in page_ranges:
            logger.warning("[%d] No sample or page range, skipping", idx)
            continue

        sample = samples[idx - 1]  # 0-indexed list
        paper_id = sample.get("paper_id", "").strip().replace("\r", "")
        page_range = page_ranges[idx]

        ok = process_one(idx, paper_id, page_range, PDF_DIR, MD_OUT_DIR)
        if ok:
            success += 1
        else:
            fail += 1

    logger.info("=== Summary: %d success, %d failed ===", success, fail)


if __name__ == "__main__":
    main()
