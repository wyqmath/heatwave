"""
4_fulltext_extract.py
Extract causal/relational triples from full-text Markdown (TODO-4a, Step 4).

Pipeline per paper:
    1. Read {safe_id}.md from fulltext/markdown/
    2. Split into pages by <!-- page --> markers
    3. Clean noise (<!-- Meanless: ... -->, <!-- Media -->, etc.)
    4. Merge short pages (<50 words) into next page
    5. For each page-chunk: prepend last 2 sentences of previous page as context
    6. Call LLM (same prompt as abstract extraction, except "research paper text")
    7. Merge all chunk results, deduplicate by (start, rel, end)
    8. Save to fulltext/extracted/{safe_id}.json

Usage:
    cd workspace/code
    python 4_fulltext_extract.py              # process all 80
    python 4_fulltext_extract.py --test 1     # test with 1 file
    python 4_fulltext_extract.py --start 10 --end 20
"""

import csv
import json
import logging
import os
import re
import time
import threading
from pathlib import Path

import yaml
from openai import OpenAI

# ---------- paths ----------
BASE_DIR = Path(__file__).parent.parent
CONFIG_DIR = BASE_DIR / "config"
FULLTEXT_DIR = BASE_DIR / "data" / "fulltext"
MD_DIR = FULLTEXT_DIR / "markdown"
EXTRACTED_DIR = FULLTEXT_DIR / "extracted"
SAMPLE_CSV = FULLTEXT_DIR / "sample_list.csv"

# ---------- API config ----------
_key_path = CONFIG_DIR / "api_keys.yaml"
with open(_key_path, "r", encoding="utf-8") as _f:
    _api_keys = yaml.safe_load(_f)
_primary = _api_keys.get("primary", {})
API_KEY = _primary.get("api_key", "")
BASE_URL = _primary.get("base_url", "")
MODEL = _primary.get("model", "claude-sonnet-4-5")

_client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ---------- logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(FULLTEXT_DIR / "fulltext_extract.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ---------- RPM controller ----------
class RPMController:
    """Sliding-window rate limiter."""

    def __init__(self, target_rpm: int = 60):
        self.target_rpm = target_rpm
        self.min_interval = 60.0 / target_rpm
        self.request_times: list[float] = []
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


rpm = RPMController(60)


# ---------- Prompt (minimal change from abstract version) ----------
SYSTEM_PROMPT = """\
You are a domain expert in heatwave disaster research. Your task is to extract
entity-level causal and relational knowledge from the provided research paper text.

# Extraction Rules
1. Extract ONLY relationships that are **explicitly stated or directly supported**
   by the text. Do NOT infer, speculate, or fabricate relationships.
2. There is NO minimum or maximum quota. If the text supports zero relationships,
   return an empty JSON array `[]`.
3. Each relationship must be traceable to a specific sentence in the input.

# Ontology – Four-Layer Classification
Assign every entity (start_node, end_node) to exactly one layer:

| Layer        | Definition & Examples |
|-------------|----------------------|
| physical    | Climate / meteorological / environmental phenomena. Examples: HeatwaveDuration, UrbanHeatIsland, AirTemperature, Humidity |
| biological  | Human / animal / plant physiological and health effects. Examples: HeatStroke, CardiovascularMortality, CropYieldLoss, ThermalDiscomfort |
| social      | Human behaviour, governance, policy, demographics. Examples: PublicHealthPolicy, ElderlyPopulation, OutdoorWorkers, EarlyWarningSystem |
| economic    | Financial, infrastructural, productivity impacts. Examples: ElectricityCost, LaborProductivityLoss, HealthcareBurden, AgriculturalRevenue |

# Output Schema
Return a JSON **array** of objects. Each object has these fields:
{
  "start_node": "PascalCase entity name",
  "relationship": "activeVerbPhrase (e.g. increases, causes, mitigates)",
  "end_node": "PascalCase entity name",
  "evidence_sentence": "Copy the original sentence from the text that supports this relationship",
  "confidence": "high | medium | low",
  "start_layer": "physical | biological | social | economic",
  "end_layer": "physical | biological | social | economic"
}

# Important Constraints
- Use PascalCase for all node names (e.g. HeatRelatedMortality, not heat-related mortality).
- Use an active verb phrase for `relationship` (e.g. "increases", "mitigates", "correlatesWith").
- `evidence_sentence` must be copied verbatim from the input text.
- `confidence`: high = explicit causal/statistical claim; medium = stated association;
  low = weakly implied but still text-supported.
- If no relationships can be extracted, return an empty JSON array `[]`.

Output ONLY valid JSON array. Do not include markdown formatting.\
"""


# ============================================================
# Markdown parsing
# ============================================================

def _clean_page(text: str) -> str:
    """Remove noise comments and non-text elements from a single page."""
    # Remove <!-- Meanless: ... -->
    text = re.sub(r'<!--\s*Meanless:.*?-->', '', text, flags=re.DOTALL)
    # Remove <!-- Media --> and <img .../> tags
    text = re.sub(r'<!--\s*Media\s*-->', '', text)
    text = re.sub(r'<img\s[^>]*/>', '', text)
    # Remove <!-- figureText: ... -->
    text = re.sub(r'<!--\s*figureText:.*?-->', '', text, flags=re.DOTALL)
    # Remove <!-- Footnote -->
    text = re.sub(r'<!--\s*Footnote\s*-->', '', text)
    # Remove remaining HTML tags (tables etc.) but keep text content
    text = re.sub(r'<[^>]+>', ' ', text)
    # Collapse whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def split_into_pages(md_text: str) -> list[str]:
    """Split markdown by <!-- page N, score: S --> markers.

    Returns list of cleaned page texts (may include empty strings).
    """
    # Split on page markers; keep content after each marker
    parts = re.split(r'<!--\s*page\s+\d+,\s*score:\s*\d+\s*-->', md_text)

    # First part is before any page marker (usually empty or metadata); skip it
    pages = []
    for part in parts[1:]:
        # Remove horizontal rule separators between pages
        cleaned = part.strip().rstrip('-').strip()
        cleaned = _clean_page(cleaned)
        pages.append(cleaned)

    return pages


def _last_n_sentences(text: str, n: int = 2) -> str:
    """Extract last N sentences from text for cross-page context."""
    # Split on sentence-ending punctuation followed by space or end
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s for s in sentences if len(s) > 10]  # filter fragments
    if not sentences:
        return ""
    return " ".join(sentences[-n:])


def prepare_chunks(pages: list[str], min_words: int = 50) -> list[dict]:
    """Merge short pages and prepare chunks with cross-page context.

    Returns list of dicts:
        {"text": str, "page_nums": list[int]}
    """
    # Step 1: merge short pages into next page
    merged: list[dict] = []
    buffer_text = ""
    buffer_pages: list[int] = []

    for i, page_text in enumerate(pages):
        page_num = i + 1
        word_count = len(page_text.split())

        if word_count < min_words and i < len(pages) - 1:
            # Accumulate into buffer
            buffer_text += ("\n\n" + page_text) if buffer_text else page_text
            buffer_pages.append(page_num)
        else:
            combined = (buffer_text + "\n\n" + page_text).strip() if buffer_text else page_text
            merged.append({
                "text": combined,
                "page_nums": buffer_pages + [page_num],
            })
            buffer_text = ""
            buffer_pages = []

    # Flush remaining buffer
    if buffer_text:
        if merged:
            merged[-1]["text"] += "\n\n" + buffer_text
            merged[-1]["page_nums"].extend(buffer_pages)
        else:
            merged.append({"text": buffer_text, "page_nums": buffer_pages})

    # Step 2: add cross-page context
    chunks = []
    for idx, item in enumerate(merged):
        if not item["text"].strip():
            continue  # skip completely empty chunks

        context = ""
        if idx > 0:
            prev_text = merged[idx - 1]["text"]
            context = _last_n_sentences(prev_text, 2)

        chunks.append({
            "text": item["text"],
            "context": context,
            "page_nums": item["page_nums"],
        })

    return chunks


# ============================================================
# LLM call
# ============================================================

def _parse_json_response(content_str: str) -> list | dict:
    """Parse LLM JSON response with multiple fallback strategies."""
    cleaned = content_str.strip()
    # Strip markdown code fences (greedy: handle missing closing fence)
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```\s*$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Attempt repairs
    repaired = cleaned
    repaired = re.sub(r"(?<!\\)'", '"', repaired)
    repaired = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', repaired)
    repaired = re.sub(r'/\*.*?\*/', '', repaired, flags=re.DOTALL)

    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    from ast import literal_eval
    try:
        return literal_eval(repaired)
    except Exception:
        pass

    # Last resort: salvage complete JSON objects from truncated array
    # Find all complete {...} blocks and wrap in []
    objects = []
    depth = 0
    start = None
    for i, ch in enumerate(repaired):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    obj = json.loads(repaired[start:i + 1])
                    objects.append(obj)
                except json.JSONDecodeError:
                    pass
                start = None

    if objects:
        logger.warning("Salvaged %d complete objects from truncated JSON", len(objects))
        return objects

    raise ValueError(f"Cannot parse JSON response: {content_str[:200]}...")


def call_llm_for_chunk(chunk_text: str, context: str, paper_id: str,
                       chunk_idx: int, total_chunks: int) -> list[dict]:
    """Call LLM to extract relationships from one chunk.

    Returns list of relationship dicts (already with paper_id injected).
    """
    # Build user message
    user_parts = []
    if context:
        user_parts.append(
            f"[Context from previous page, for continuity only — do NOT extract "
            f"relationships from this section]\n{context}\n\n---\n"
        )
    user_parts.append(chunk_text)
    user_message = "\n".join(user_parts)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    max_retries = 3
    for attempt in range(max_retries + 1):
        rpm.wait_if_needed()
        try:
            completion = _client.chat.completions.create(
                model=MODEL,
                messages=messages,
                max_tokens=16384,
                response_format={"type": "json_object"},
            )
            content_str = completion.choices[0].message.content
            break
        except Exception as e:
            status_code = getattr(getattr(e, "response", None), "status_code", None)
            retryable = status_code in (429, 500, 502, 503) if status_code else True

            if not retryable or attempt == max_retries:
                logger.error("%s chunk %d/%d: API failed after %d attempts: %s",
                             paper_id, chunk_idx + 1, total_chunks, attempt + 1, e)
                raise

            wait = min(2 ** attempt, 30)
            logger.warning("%s chunk %d/%d: retry %d/%d in %ds (%s)",
                           paper_id, chunk_idx + 1, total_chunks,
                           attempt + 1, max_retries, wait, e)
            time.sleep(wait)

    # Parse
    parsed = _parse_json_response(content_str)

    # Normalise to list
    if isinstance(parsed, dict):
        for key in ("relationships", "results", "data"):
            if key in parsed and isinstance(parsed[key], list):
                parsed = parsed[key]
                break
        else:
            parsed = [parsed]

    if not isinstance(parsed, list):
        parsed = []

    # Filter valid dicts
    parsed = [
        item for item in parsed
        if isinstance(item, dict) and "start_node" in item and "end_node" in item
    ]

    # Inject paper_id
    for rel in parsed:
        rel["paper_id"] = paper_id

    return parsed


# ============================================================
# Deduplication
# ============================================================

_CONF_ORDER = {"high": 3, "medium": 2, "low": 1}


def deduplicate(relations: list[dict]) -> list[dict]:
    """Deduplicate by (start_node, relationship, end_node) case-insensitive.

    Keep the one with highest confidence.
    """
    best: dict[tuple, dict] = {}
    for rel in relations:
        key = (
            rel.get("start_node", "").lower(),
            rel.get("relationship", "").lower(),
            rel.get("end_node", "").lower(),
        )
        existing = best.get(key)
        if existing is None:
            best[key] = rel
        else:
            old_conf = _CONF_ORDER.get(existing.get("confidence", "low"), 0)
            new_conf = _CONF_ORDER.get(rel.get("confidence", "low"), 0)
            if new_conf > old_conf:
                best[key] = rel

    return list(best.values())


# ============================================================
# Per-paper processing
# ============================================================

def process_paper(paper_id: str, md_path: Path, out_path: Path) -> bool:
    """Process a single paper's full-text markdown.

    Returns True on success, False on failure.
    """
    # Checkpoint
    if out_path.exists() and out_path.stat().st_size > 0:
        logger.info("Skip (exists): %s", paper_id)
        return True

    try:
        start_time = time.time()

        # Read markdown
        md_text = md_path.read_text(encoding="utf-8")

        # Split into pages
        pages = split_into_pages(md_text)
        if not pages:
            logger.warning("%s: no pages found in markdown", paper_id)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump([], f)
            return True

        # Prepare chunks (merge short pages, add context)
        chunks = prepare_chunks(pages, min_words=50)
        logger.info("%s: %d pages -> %d chunks", paper_id, len(pages), len(chunks))

        # Extract from each chunk (per-chunk errors don't kill the paper)
        all_rels: list[dict] = []
        chunk_errors = 0
        for idx, chunk in enumerate(chunks):
            try:
                rels = call_llm_for_chunk(
                    chunk_text=chunk["text"],
                    context=chunk["context"],
                    paper_id=paper_id,
                    chunk_idx=idx,
                    total_chunks=len(chunks),
                )
                all_rels.extend(rels)
                logger.debug("%s chunk %d/%d (pages %s): %d rels",
                             paper_id, idx + 1, len(chunks),
                             chunk["page_nums"], len(rels))
            except Exception as e:
                chunk_errors += 1
                logger.warning("%s chunk %d/%d (pages %s): SKIPPED — %s",
                               paper_id, idx + 1, len(chunks),
                               chunk["page_nums"], e)

        # Deduplicate
        deduped = deduplicate(all_rels)

        # Save
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(deduped, f, ensure_ascii=False, indent=4)

        elapsed = time.time() - start_time
        err_note = f", {chunk_errors} chunk errors" if chunk_errors else ""
        logger.info("%s: done — %d raw -> %d deduped rels (%.1fs, %d chunks%s)",
                    paper_id, len(all_rels), len(deduped), elapsed, len(chunks), err_note)
        return True

    except Exception as e:
        logger.error("%s: error — %s", paper_id, e)
        return False


# ============================================================
# Main
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Full-text triple extraction (per-page)")
    parser.add_argument("--test", type=int, default=0, help="Only process first N papers")
    parser.add_argument("--start", type=int, default=1, help="Start index (1-based)")
    parser.add_argument("--end", type=int, default=80, help="End index (1-based)")
    args = parser.parse_args()

    if not API_KEY:
        logger.error("Primary API key not configured!")
        return

    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)

    # Load sample list
    with open(SAMPLE_CSV, "r", encoding="utf-8-sig") as f:
        samples = list(csv.DictReader(f))

    logger.info("=== Full-text Triple Extraction ===")
    logger.info("Samples: %d, Output: %s", len(samples), EXTRACTED_DIR)

    # Determine range
    start_idx = args.start
    end_idx = args.end
    if args.test > 0:
        start_idx = 1
        end_idx = min(args.test, len(samples))

    success = 0
    fail = 0
    total_start = time.time()

    for idx in range(start_idx, end_idx + 1):
        if idx > len(samples):
            break

        sample = samples[idx - 1]
        paper_id = sample.get("paper_id", "").strip()
        safe_id = paper_id.replace(":", "_")

        md_path = MD_DIR / f"{safe_id}.md"
        out_path = EXTRACTED_DIR / f"{safe_id}.json"

        if not md_path.exists():
            logger.error("[%d] Markdown not found: %s", idx, md_path)
            fail += 1
            continue

        logger.info("[%d/%d] Processing %s ...", idx, end_idx, paper_id)
        ok = process_paper(paper_id, md_path, out_path)
        if ok:
            success += 1
        else:
            fail += 1

    total_elapsed = time.time() - total_start
    logger.info("=== Summary: %d success, %d failed (%.1fs total) ===",
                success, fail, total_elapsed)


if __name__ == "__main__":
    main()
