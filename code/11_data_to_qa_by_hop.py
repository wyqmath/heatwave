import argparse
import csv
import json
import logging
import os
import random
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml
except ImportError:
    yaml = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

def load_config(path: str = "config/default.yaml") -> Dict[str, Any]:
    cfg_path = Path(path)
    if yaml is None or not cfg_path.exists():
        return {}
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

class RateLimiter:
    def __init__(self, min_interval: float):
        self.min_interval = max(min_interval, 0.0)
        self._lock = threading.Lock()
        self._last_call = 0.0

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            wait_for = self.min_interval - (now - self._last_call)
            if wait_for > 0:
                time.sleep(wait_for)
            self._last_call = time.monotonic()

def load_adjacency(path: Path) -> Tuple[Dict[str, List[Tuple[str, str]]], List[str]]:
    adjacency: Dict[str, List[Tuple[str, str]]] = {}
    nodes = set()

    if not path.exists():
        return adjacency, []

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    for k, v in raw.items():
        adjacency[k] = [(item[0], item[1]) for item in v]
        nodes.add(k)
        for item in v:
            nodes.add(item[1])

    return adjacency, sorted(nodes)

def sample_path_with_exact_hops(
    adjacency: Dict[str, List[Tuple[str, str]]],
    nodes: List[str],
    target_hops: int,
    rng: random.Random
) -> Optional[List[Dict[str, str]]]:
    if not nodes or target_hops < 1:
        return None

    start = rng.choice(nodes)
    current = start
    visited = {start}
    path: List[Dict[str, str]] = []

    for _ in range(target_hops):
        options = adjacency.get(current)
        if not options:
            return None
        candidates = [opt for opt in options if opt[1] not in visited]
        if not candidates:
            return None
        rel, nxt = rng.choice(candidates)
        path.append({"from": current, "relationship": rel, "to": nxt})
        visited.add(nxt)
        current = nxt

    return path if len(path) == target_hops else None

def iter_paths_with_exact_hops(
    adjacency: Dict[str, List[Tuple[str, str]]],
    nodes: List[str],
    target_hops: int,
    seed: int,
    max_attempts: int
):
    rng = random.Random(seed)
    seen = set()
    attempts = 0

    while attempts < max_attempts:
        attempts += 1
        path = sample_path_with_exact_hops(adjacency, nodes, target_hops, rng)
        if not path:
            continue
        key = tuple((step["from"], step["relationship"], step["to"]) for step in path)
        if key in seen:
            continue
        seen.add(key)
        yield path

def format_path(path_steps: List[Dict[str, str]]) -> str:
    return "\n".join([f"{step['from']} --[{step['relationship']}]--> {step['to']}" for step in path_steps])

def normalize_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = []
        for item in value:
            if item is None:
                continue
            text = item.strip() if isinstance(item, str) else str(item).strip()
            if text:
                parts.append(text)
        return " ".join(parts)
    if value is None:
        return ""
    return str(value).strip()

NON_HEATWAVE_MARKER = "not heatwave content"

def contains_marker(text: Any) -> bool:
    return NON_HEATWAVE_MARKER in normalize_text(text).lower()

def normalize_answer_letter(value: Any) -> str:
    text = normalize_text(value).upper()
    if not text:
        return ""
    for char in text:
        if char in "ABCD":
            return char
    return ""

def normalize_options(value: Any) -> List[str]:
    raw: List[str] = []
    if isinstance(value, list):
        raw = [normalize_text(item) for item in value]
    elif isinstance(value, str):
        parts = re.split(r"[\n\r]+|\s*\|\s*", value)
        raw = [normalize_text(part) for part in parts]
    else:
        return []

    cleaned: List[str] = []
    for text in raw:
        if not text:
            continue
        text = re.sub(r"^[A-Da-d][\).:、\-\s]+", "", text).strip()
        if text and text not in cleaned:
            cleaned.append(text)

    if len(cleaned) < 4:
        return []
    cleaned = cleaned[:4]
    return [f"{chr(65 + i)}. {text}" for i, text in enumerate(cleaned)]

def format_options(options: List[str]) -> str:
    return " | ".join(options)

def validate_mcq_item(question: str, options: List[str], answer: str) -> bool:
    if not question or not options or not answer:
        return False
    if len(options) != 4:
        return False
    if answer not in {"A", "B", "C", "D"}:
        return False
    if contains_marker(question) or any(contains_marker(opt) for opt in options):
        return False
    return True

def fuzzy_match(node: str, text: str) -> bool:
    node_lower = node.lower()
    if node_lower in text:
        return True
    words = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)', node)
    if len(words) >= 3:
        matched = sum(1 for w in words if w.lower() in text)
        return matched >= min(3, len(words) - 1)
    return False

def validate_multihop_response(data: Any, path_nodes: List[str]) -> bool:
    if not isinstance(data, dict):
        return False
    question = normalize_text(data.get("question"))
    answer_letter = normalize_text(data.get("answer")).upper()
    options_raw = data.get("options", [])

    if not question or not answer_letter:
        return False
    if question.lower() == "nan" or answer_letter.lower() == "nan":
        return False
    if contains_marker(question):
        return False

    q_lower = question.lower()

    if not fuzzy_match(path_nodes[0], q_lower):
        return False
    if not fuzzy_match(path_nodes[-1], q_lower):
        return False

    if not isinstance(options_raw, list) or len(options_raw) < 4:
        return False

    answer_index = ord(answer_letter) - ord('A')
    if answer_index < 0 or answer_index >= len(options_raw):
        return False

    correct_option = normalize_text(options_raw[answer_index])
    option_lower = correct_option.lower()

    for node in path_nodes[1:-1]:
        if not fuzzy_match(node, option_lower):
            return False

    return True

def generate_qa_for_path(
    path_steps: List[Dict[str, str]],
    client,
    model: str,
    max_retries: int,
    timeout: int,
    rate_limiter: RateLimiter,
    hop_count: int
) -> Optional[Dict[str, Any]]:
    if not path_steps:
        return None

    path_nodes = [path_steps[0]["from"]] + [step["to"] for step in path_steps]

    if hop_count == 1:
        system_prompt = (
            "You are given a single-hop knowledge graph relation. Create a multiple-choice QA pair.\n"
            "Rules:\n"
            "1) Question must mention the start and end entities and ask about their relationship.\n"
            "2) Correct option describes the relationship correctly.\n"
            "3) Provide exactly 4 options labeled A/B/C/D.\n"
            "4) Only one option is correct.\n"
            "5) If not heatwave-related, output 'Not Heatwave Content'.\n"
            "6) Output JSON: {\"question\":...,\"options\":[\"A. ...\",\"B. ...\",\"C. ...\",\"D. ...\"],\"answer\":\"A\"}."
        )
    else:
        system_prompt = (
            f"You are given a {hop_count}-hop knowledge graph path. Create a multiple-choice QA pair.\n"
            "Rules:\n"
            "1) Question must mention the start and end entities.\n"
            "2) Correct option must list all intermediate entities in order.\n"
            "3) Provide exactly 4 options labeled A/B/C/D.\n"
            "4) Only one option is correct.\n"
            "5) If not heatwave-related, output 'Not Heatwave Content'.\n"
            "6) Output JSON: {\"question\":...,\"options\":[\"A. ...\",\"B. ...\",\"C. ...\",\"D. ...\"],\"answer\":\"A\"}."
        )

    user_prompt = f"Path:\n{format_path(path_steps)}"

    for attempt in range(max_retries):
        try:
            rate_limiter.wait()
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                timeout=timeout,
            )
            raw_response = completion.choices[0].message.content
            parsed = json.loads(raw_response)

            if validate_multihop_response(parsed, path_nodes):
                options = normalize_options(parsed.get("options"))
                answer_letter = normalize_answer_letter(parsed.get("answer"))
                item = {
                    "question": normalize_text(parsed.get("question")),
                    "options": options,
                    "answer": answer_letter,
                    "hop_count": hop_count,
                }
                if validate_mcq_item(item["question"], options, answer_letter):
                    return item
        except Exception:
            time.sleep(1 * (attempt + 1))

    return None

def generate_qa_for_hop(
    adjacency: Dict[str, List[Tuple[str, str]]],
    nodes: List[str],
    hop_count: int,
    target_count: int,
    seed: int,
    max_attempts: int,
    client,
    model: str,
    max_workers: int,
    max_retries: int,
    timeout: int,
    rate_limiter: RateLimiter
) -> List[Dict[str, Any]]:

    if target_count <= 0 or not adjacency or not nodes:
        return []

    path_iter = iter_paths_with_exact_hops(adjacency, nodes, hop_count, seed, max_attempts)
    results: List[Dict[str, Any]] = []
    stop_event = threading.Event()

    stats = {"processed": 0, "success": 0, "failed": 0}

    pbar = tqdm(total=target_count, desc=f"{hop_count}-hop QA", unit="qa") if tqdm else None

    def handle_path(path_steps: List[Dict[str, str]]):
        if stop_event.is_set():
            return None
        stats["processed"] += 1
        result = generate_qa_for_path(
            path_steps, client, model, max_retries, timeout, rate_limiter, hop_count
        )
        if result:
            stats["success"] += 1
        else:
            stats["failed"] += 1
        return result

    def submit_next(executor, iterator, pending):
        try:
            path_steps = next(iterator)
        except StopIteration:
            return False
        future = executor.submit(handle_path, path_steps)
        pending[future] = path_steps
        return True

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        pending = {}
        for _ in range(max_workers):
            if not submit_next(executor, path_iter, pending):
                break

        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                pending.pop(future, None)
                item = future.result()
                if item:
                    results.append(item)
                    if pbar:
                        pbar.update(1)
                    if len(results) >= target_count:
                        stop_event.set()
                        for fut in pending:
                            fut.cancel()
                        pending.clear()
                        break
                if stop_event.is_set():
                    continue
                submit_next(executor, path_iter, pending)

    if pbar:
        pbar.close()

    print(f"\n[{hop_count}-hop Stats] Processed: {stats['processed']}, Success: {stats['success']}, "
          f"Failed: {stats['failed']}, Success Rate: {stats['success']/max(stats['processed'],1)*100:.1f}%")

    return results

def export_csv(qa_list: List[Dict[str, Any]], path: Path, include_hop: bool = True) -> None:
    if not qa_list:
        return
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["Question", "Options", "Answer"]
    if include_hop:
        fieldnames.append("HopCount")

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for item in qa_list:
            row = {
                "Question": item.get("question", ""),
                "Options": format_options(item.get("options", [])),
                "Answer": item.get("answer", ""),
            }
            if include_hop:
                row["HopCount"] = item.get("hop_count", "")
            writer.writerow(row)

def export_json(qa_list: List[Dict[str, Any]], path: Path) -> None:
    if not qa_list:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(qa_list, ensure_ascii=False, indent=2), encoding="utf-8")

def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("data_to_qa_by_hop")

    cfg = load_config()
    llm_cfg = cfg.get("llm", {})

    parser = argparse.ArgumentParser(description="Generate QA datasets by hop count")
    parser.add_argument("--adjacency", default="adjacency.json", help="Path to adjacency list file")
    parser.add_argument("--output-dir", default=".", help="Output directory")
    parser.add_argument("--target-per-hop", type=int, default=1000, help="Number of questions per hop")
    parser.add_argument("--max-workers", type=int, default=20, help="Concurrent threads")
    parser.add_argument("--max-retries", type=int, default=3, help="LLM retry attempts")
    parser.add_argument("--timeout", type=int, default=60, help="LLM timeout seconds")
    parser.add_argument("--request-interval", type=float, default=0.1, help="Request interval")
    parser.add_argument("--global-rpm", type=int, default=500, help="Global RPM limit")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-attempts", type=int, default=50000, help="Max attempts per hop")
    parser.add_argument("--hops", type=str, default="1,2,3,4", help="Hops to generate, comma separated")
    args = parser.parse_args()

    api_key = llm_cfg.get("api_key") or os.getenv("LLM_API_KEY", "")
    base_url = llm_cfg.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    model = llm_cfg.get("model", "qwen3-max")

    if not api_key:
        logger.error("Missing API Key. Please set llm.api_key in config/default.yaml or use LLM_API_KEY environment variable.")
        return

    try:
        from openai import OpenAI
    except ImportError:
        logger.error("Missing openai package. Run: pip install openai")
        return

    adjacency_path = Path(args.adjacency)
    if not adjacency_path.exists():
        logger.error(f"Adjacency file not found: {adjacency_path}")
        return

    print(f"📂 Loading adjacency list: {adjacency_path}")
    adjacency, nodes = load_adjacency(adjacency_path)
    print(f"✅ Loaded: {len(nodes)} nodes, {sum(len(v) for v in adjacency.values())} edges")

    client = OpenAI(api_key=api_key, base_url=base_url)
    min_interval = args.request_interval
    if args.global_rpm > 0:
        min_interval = max(min_interval, 60.0 / args.global_rpm)
    rate_limiter = RateLimiter(min_interval)

    hops_to_generate = [int(h.strip()) for h in args.hops.split(",")]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    print(f"\n{'='*70}")
    print(f"Starting QA Dataset Generation")
    print(f"Target: {args.target_per_hop} questions per hop")
    print(f"Hops: {hops_to_generate}")
    print(f"{'='*70}\n")

    for hop in hops_to_generate:
        print(f"\n{'='*70}")
        print(f"🚀 Generating {hop}-hop QA Dataset (Target: {args.target_per_hop})")
        print(f"{'='*70}")

        results = generate_qa_for_hop(
            adjacency=adjacency,
            nodes=nodes,
            hop_count=hop,
            target_count=args.target_per_hop,
            seed=args.seed + hop,
            max_attempts=args.max_attempts,
            client=client,
            model=model,
            max_workers=args.max_workers,
            max_retries=args.max_retries,
            timeout=args.timeout,
            rate_limiter=rate_limiter
        )

        csv_path = output_dir / f"qa_{hop}hop.csv"
        json_path = output_dir / f"qa_{hop}hop.json"

        export_csv(results, csv_path)
        export_json(results, json_path)

        print(f"✅ {hop}-hop dataset saved:")
        print(f"   CSV: {csv_path} ({len(results)} items)")
        print(f"   JSON: {json_path}")

        all_results.extend(results)

    combined_csv = output_dir / "qa_all_hops.csv"
    combined_json = output_dir / "qa_all_hops.json"

    export_csv(all_results, combined_csv)
    export_json(all_results, combined_json)

    print(f"\n{'='*70}")
    print(f"🎉 All completed!")
    print(f"{'='*70}")
    print(f"Combined Dataset: {combined_csv} ({len(all_results)} items)")
    print(f"\nStats by Hop:")
    for hop in hops_to_generate:
        count = sum(1 for r in all_results if r.get("hop_count") == hop)
        print(f"  {hop}-hop: {count} items")

if __name__ == "__main__":
    main()