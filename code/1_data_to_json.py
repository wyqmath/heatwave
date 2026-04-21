import json
import os
import re
import time
import logging
from pathlib import Path
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
import threading

import yaml

# ---------- API keys (load once) ----------
_key_path = Path(__file__).parent.parent / "config" / "api_keys.yaml"
if _key_path.exists():
    with open(_key_path, 'r', encoding='utf-8') as _f:
        _api_keys = yaml.safe_load(_f)
else:
    _api_keys = {}
_primary = _api_keys.get('primary', {})
API_KEY = _primary.get('api_key', '')
BASE_URL = _primary.get('base_url', '')
MODEL = _primary.get('model', 'claude-sonnet-4-5')

# ---------- shared OpenAI client ----------
_client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ---------- logging setup ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("extraction.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ---------- counters (thread-safe) ----------
_counter_lock = threading.Lock()
_empty_count = 0
_total_count = 0


# ---------- RPM controller (thread-safe) ----------
class RPMController:
    """Global RPM controller to limit API call frequency."""

    def __init__(self, target_rpm: int = 60):
        self.target_rpm = target_rpm
        self.min_interval = 60.0 / target_rpm
        self.request_times = []
        self.lock = threading.Lock()

    def wait_if_needed(self):
        with self.lock:
            current_time = time.time()
            self.request_times = [t for t in self.request_times if current_time - t < 60]
            current_rpm = len(self.request_times)

            wait_time = 0
            if current_rpm >= self.target_rpm:
                oldest_time = self.request_times[0]
                wait_time = 60 - (current_time - oldest_time) + 0.1
            elif self.request_times:
                last_time = self.request_times[-1]
                elapsed = current_time - last_time
                if elapsed < self.min_interval:
                    wait_time = self.min_interval - elapsed

            self.request_times.append(time.time() + wait_time)

        if wait_time > 0:
            time.sleep(wait_time)


rpm_controller = RPMController(60)


SYSTEM_PROMPT = """\
You are a domain expert in heatwave disaster research. Your task is to extract
entity-level causal and relational knowledge from the provided title and abstract.

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


def call_with_messages(i, content, paper_id):
    """Call LLM to extract relationships from a single paper."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]

    max_retries = 3
    last_exc = None

    for attempt in range(max_retries + 1):
        # Rate limiting
        rpm_controller.wait_if_needed()

        try:
            completion = _client.chat.completions.create(
                model=MODEL,
                messages=messages,
                max_tokens=4096,
                response_format={"type": "json_object"},
            )
            content_str = completion.choices[0].message.content
            break  # success
        except Exception as e:
            last_exc = e
            status_code = getattr(getattr(e, 'response', None), 'status_code', None)
            retryable = status_code in (429, 500, 502, 503) if status_code else True

            if not retryable or attempt == max_retries:
                logger.error("Paper %d (%s): API failed after %d attempts: %s",
                             i, paper_id, attempt + 1, e)
                raise

            wait = min(2 ** attempt, 30)
            logger.warning("Paper %d (%s): retry %d/%d in %ds (%s)",
                           i, paper_id, attempt + 1, max_retries, wait, e)
            time.sleep(wait)

    # ---------- JSON parsing with fallback ----------
    parsed_content = _parse_json_response(content_str, i)

    # Normalise to list
    if isinstance(parsed_content, dict):
        # Some models wrap array in {"relationships": [...]}
        for key in ("relationships", "results", "data"):
            if key in parsed_content and isinstance(parsed_content[key], list):
                parsed_content = parsed_content[key]
                break
        else:
            parsed_content = [parsed_content]

    if not isinstance(parsed_content, list):
        parsed_content = []

    # Filter: keep only dicts with required keys
    parsed_content = [
        item for item in parsed_content
        if isinstance(item, dict) and "start_node" in item and "end_node" in item
    ]

    # Inject paper_id into every relationship
    for rel in parsed_content:
        rel["paper_id"] = paper_id

    # Track empty results
    global _empty_count, _total_count
    with _counter_lock:
        _total_count += 1
        if len(parsed_content) == 0:
            _empty_count += 1
            logger.warning("Paper %s (%d): extracted 0 relationships", paper_id, i)

    return parsed_content


def _parse_json_response(content_str, i):
    """Parse LLM JSON response with multiple fallback strategies."""
    # Strip markdown code fences if present
    cleaned = content_str.strip()
    if cleaned.startswith("```"):
        # Remove opening fence (```json or ```)
        cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
        # Remove closing fence
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

    # Last resort: ast.literal_eval
    from ast import literal_eval
    try:
        return literal_eval(repaired)
    except Exception:
        # Save raw response for debugging
        os.makedirs("errors", exist_ok=True)
        with open(f"errors/error_{i}_raw.txt", "w") as f:
            f.write(content_str)
        raise ValueError(
            f"Cannot parse JSON (paper #{i}). Raw saved to errors/error_{i}_raw.txt"
        )


def _extract_year(doc):
    """Extract publication year from WoS record. Returns int or None.

    Priority: PY field > EA field > PD field.
    """
    # PY field (explicit year)
    py_match = re.search(r'\nPY (\d{4})', doc)
    if py_match:
        return int(py_match.group(1))

    # EA field (early access date, e.g. "MAY 2024")
    ea_match = re.search(r'\nEA .*?(\d{4})', doc)
    if ea_match:
        return int(ea_match.group(1))

    # PD field (publication date, e.g. "JAN 2024" or "2024")
    pd_match = re.search(r'\nPD .*?(\d{4})', doc)
    if pd_match:
        return int(pd_match.group(1))

    return None


def _safe_filename(paper_id):
    """Convert paper_id to a safe filename. Fallback to None if unusable."""
    if not paper_id:
        return None
    # Replace characters not allowed in filenames
    safe = re.sub(r'[<>:"/\\|?*]', '_', paper_id)
    return safe if safe else None


def process_document(args):
    """Process a single WoS document: extract relationships, route to kg/ or holdout/."""
    i, doc, kg_dir, holdout_dir = args
    try:
        # Extract UT (paper_id) early for checkpoint check
        paper_id = ""
        ut_match = re.search(r'\nUT (WOS:\S+)', doc)
        if ut_match:
            paper_id = ut_match.group(1)
        elif "UT " in doc:
            # fallback for non-WOS UT formats
            ut_part = doc.split("UT ")[1]
            paper_id = ut_part.split("\n")[0].strip().split()[-1]

        # ---------- checkpoint: skip already processed ----------
        safe_name = _safe_filename(paper_id)
        if safe_name:
            for d in (kg_dir, holdout_dir):
                check_path = os.path.join(d, f"{safe_name}.json")
                if os.path.exists(check_path):
                    logger.debug("Skipping already processed: %s", paper_id)
                    return True
        # ---------------------------------------------------

        start_time = time.time()
        content = []

        # Extract title (TI)
        ti_match = re.search(r'TI (.*?)(?=\n[A-Z]{2} )', doc, re.DOTALL)
        if ti_match:
            ti_content = re.sub(r'\s+', ' ', ti_match.group(1).replace('\n', ' ')).strip()
            content.append(ti_content)

        # Extract abstract (AB)
        ab_match = re.search(r'AB (.*?)(?=\n[A-Z]{2} )', doc, re.DOTALL)
        if ab_match:
            ab_content = ab_match.group(1).replace('\n', ' ').strip()
            content.append(ab_content)

        final_content = " ".join(content)
        if not final_content:
            raise ValueError("No valid content (missing TI and AB fields)")

        # Call LLM
        parsed_content = call_with_messages(i=i, content=final_content, paper_id=paper_id)

        # Determine output directory based on year
        year = _extract_year(doc)
        if year is not None and year <= 2024:
            out_dir = kg_dir
        else:
            # >= 2025 or unknown year -> holdout (conservative)
            out_dir = holdout_dir
            if year is None:
                logger.info("Paper %s (%d): no year found, routing to holdout", paper_id, i)

        # Determine filename (safe_name already computed above)
        if safe_name:
            filename = f"{safe_name}.json"
        else:
            filename = f"{i}.json"

        filepath = os.path.join(out_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(parsed_content, f, ensure_ascii=False, indent=4)

        elapsed = time.time() - start_time
        logger.info(
            "Paper %d done (%s, year=%s, rels=%d, %.2fs) -> %s",
            i, paper_id, year, len(parsed_content), elapsed,
            "kg" if out_dir == kg_dir else "holdout",
        )
        return True
    except Exception as e:
        with lock:
            error_file.write(f"{i} Error: {str(e)}\n")
            logger.error("Paper %d error: %s", i, str(e))
        return False


if __name__ == "__main__":
    import sys
    sys.path.append("..")

    # Load config
    try:
        from agents.base_agent import ConfigManager
        config = ConfigManager.load_config()
        input_file = config.get("data_processing", {}).get("input_file", "../data/raw/paper.txt")
        kg_dir = config.get("data_processing", {}).get("kg_output_dir", "../data/processed/kg/")
        holdout_dir = config.get("data_processing", {}).get("holdout_output_dir", "../data/processed/holdout/")
        max_workers = config.get("data_processing", {}).get("max_workers", 15)
    except Exception:
        input_file = "../data/raw/paper.txt"
        kg_dir = "../data/processed/kg/"
        holdout_dir = "../data/processed/holdout/"
        max_workers = 15

    # Create output directories
    os.makedirs(kg_dir, exist_ok=True)
    os.makedirs(holdout_dir, exist_ok=True)

    error_file = open("extraction_errors.txt", mode="a", newline="\n")
    lock = threading.Lock()

    logger.info("Reading input: %s", input_file)
    with open(input_file, "r", encoding="utf-8") as file:
        raw_content = file.read()

        # Split by WoS record boundary
        documents = re.split(r'\n(?=PT )', raw_content)
        valid_documents = [doc for doc in documents if doc.strip().startswith("PT ")]
        valid_doc_count = len(valid_documents)
        logger.info("Detected %d valid documents", valid_doc_count)

        # Validation: show first 3 records
        for idx, doc in enumerate(valid_documents[:3], 1):
            header = doc.strip().split("\n")[0]
            logger.info("Doc %d header: %s", idx, header[:40])

        total_start = time.time()

        logger.info("Processing with %d workers", max_workers)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            tasks = [(i + 1, doc, kg_dir, holdout_dir) for i, doc in enumerate(valid_documents)]
            logger.info("Total papers to process: %d", len(tasks))
            list(executor.map(process_document, tasks))

        total_time = time.time() - total_start
        logger.info(
            "Done! %d papers in %.2fs (avg %.2fs/paper)",
            valid_doc_count, total_time, total_time / max(valid_doc_count, 1),
        )

        # Report empty-result statistics
        with _counter_lock:
            empty_rate = _empty_count / max(_total_count, 1)
            logger.info(
                "Empty results: %d / %d (%.1f%%)",
                _empty_count, _total_count, empty_rate * 100,
            )
            if empty_rate > 0.3:
                logger.warning(
                    "Empty rate %.1f%% is abnormally high -- review the extraction prompt!",
                    empty_rate * 100,
                )

    error_file.close()
