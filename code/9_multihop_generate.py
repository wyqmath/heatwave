"""
9_multihop_generate.py — Multi-Hop LBD Benchmark Generator

Design:
  - 1-hop: negative control (edge exists in both KG and holdout; KG has no discriminative power → Δ ≈ 0)
  - 2-hop: A→B→C, ≥1 novel + ≥1 in_kg, answer is 1 intermediate node
  - 3-hop: A→B→C→D, ≥1 novel + ≥1 in_kg, answer is 2-node sequence
  - 4-hop: A→B→C→D→E, ≥1 novel + ≥1 in_kg, answer is 3-node sequence

  All nodes must exist in KG (FAISS-matchable); no repeated nodes in path.
  No additional constraints (e.g. start/end edges must be in KG) to keep the benchmark fair.

  Distractors: (hop-1)-step random walk from start in KG.

Usage:
    python 9_multihop_generate.py                       # default 200/hop × 4 = 800 questions (pilot)
    python 9_multihop_generate.py --per-hop 1000        # full 4000 questions
    python 9_multihop_generate.py --hops 2 3            # generate 2/3-hop only
"""

import argparse
import json
import os
import random
import re
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
from tqdm import tqdm

# ── Paths ──
WORKSPACE = Path(__file__).parent.parent
KG_JSON_DIR = WORKSPACE / "data" / "processed" / "kg"
HOLDOUT_DIR = WORKSPACE / "data" / "processed" / "holdout"
BENCHMARK_DIR = WORKSPACE / "data" / "benchmark"
MERGE_MAPPING_JSON = WORKSPACE / "data" / "processed" / "topology_sensitivity" / "phase1_merge_mapping.json"


# ============================================================
# Shared Utilities
# ============================================================

class RPMController:
    def __init__(self, rpm: int = 200):
        self.rpm = rpm
        self._times: deque = deque()
        self._lock = threading.Lock()

    def acquire(self):
        with self._lock:
            now = time.time()
            while self._times and self._times[0] < now - 60:
                self._times.popleft()
            if len(self._times) >= self.rpm:
                wait = 60 - (now - self._times[0]) + 0.05
                if wait > 0:
                    time.sleep(wait)
            self._times.append(time.time())


def make_entity_readable(name: str) -> str:
    """CamelCase → 'Camel Case'."""
    readable = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', name)
    readable = re.sub(r'(?<=[A-Z])(?=[A-Z][a-z])', ' ', readable)
    return readable


# ============================================================
# Data Loading (reuse from 12_lbd_benchmark.py)
# ============================================================

def load_merge_mapping() -> Dict[str, str]:
    """Load entity merge mapping (d<0.03, 3 rounds)."""
    if MERGE_MAPPING_JSON.exists():
        with open(MERGE_MAPPING_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    print("  WARNING: merge mapping not found, using unmerged entities")
    return {}


def load_kg_graph(merge_map: Dict[str, str]) -> Tuple[nx.DiGraph, Set[str]]:
    """Load 2024 KG as DiGraph with entity merging."""
    g = nx.DiGraph()
    for jf in KG_JSON_DIR.glob("*.json"):
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                continue
            for t in data:
                sn = merge_map.get(t.get("start_node", "").strip(),
                                   t.get("start_node", "").strip())
                en = merge_map.get(t.get("end_node", "").strip(),
                                   t.get("end_node", "").strip())
                rel = t.get("relationship", "")
                if not (sn and en and rel) or sn == en:
                    continue
                g.add_edge(sn, en, relationship=rel,
                           start_layer=t.get("start_layer", ""),
                           end_layer=t.get("end_layer", ""),
                           evidence_sentence=t.get("evidence_sentence", ""))
        except Exception:
            pass
    return g, set(g.nodes())


def load_holdout_graph(merge_map: Dict[str, str]) -> nx.DiGraph:
    """Load 2025 holdout as DiGraph with entity merging."""
    g = nx.DiGraph()
    for jf in HOLDOUT_DIR.glob("*.json"):
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                continue
            for t in data:
                sn = merge_map.get(t.get("start_node", "").strip(),
                                   t.get("start_node", "").strip())
                en = merge_map.get(t.get("end_node", "").strip(),
                                   t.get("end_node", "").strip())
                rel = t.get("relationship", "")
                if not (sn and en and rel) or sn == en:
                    continue
                paper_id = jf.stem  # WOS:XXXX
                g.add_edge(sn, en, relationship=rel,
                           start_layer=t.get("start_layer", ""),
                           end_layer=t.get("end_layer", ""),
                           paper_id=paper_id)
        except Exception:
            pass
    return g


# ============================================================
# Multi-Hop Path Mining
# ============================================================

def mine_1hop_paths(
    holdout_G: nx.DiGraph,
    kg_G: nx.DiGraph,
    kg_nodes: Set[str],
    max_paths: int = 5000,
    rng: random.Random = None,
) -> List[dict]:
    """1-hop negative control: A→B edge exists in both holdout and KG.

    KG retriever finds connections for all options → Δ ≈ 0, confirming no data leakage.
    """
    kg_edges = set(kg_G.edges())
    paths = []
    seen = set()

    start_nodes = list(holdout_G.nodes())
    if rng:
        rng.shuffle(start_nodes)

    for A in start_nodes:
        if A not in kg_nodes:
            continue
        for B in holdout_G.successors(A):
            if B not in kg_nodes or B == A:
                continue
            # Key: edge must exist in both holdout and KG
            if (A, B) not in kg_edges:
                continue
            key = (A, B)
            if key in seen:
                continue
            seen.add(key)
            paths.append({
                "hop_count": 1,
                "full_path": [A, B],
                "start_entity": A,
                "end_entity": B,
                "correct_sequence": [],  # 1-hop has no intermediate nodes
                "edges_in_kg": [True],
            })
            if len(paths) >= max_paths:
                break
        if len(paths) >= max_paths:
            break

    return paths


def mine_nhop_paths(
    holdout_G: nx.DiGraph,
    kg_G: nx.DiGraph,
    kg_nodes: Set[str],
    hop_count: int,
    max_paths: int = 5000,
    rng: random.Random = None,
) -> List[dict]:
    """Generic n-hop mining (n = number of edges).

    hop_count=2: A→B→C (3 nodes, 2 edges), sequence=[B]
    hop_count=3: A→B→C→D (4 nodes, 3 edges), sequence=[B,C]
    hop_count=4: A→B→C→D→E (5 nodes, 4 edges), sequence=[B,C,D]

    Requires:
      - All nodes in kg_nodes (FAISS can match)
      - No duplicate nodes
      - ≥1 edge in KG (so KG can help)
      - ≥1 edge novel (so it's a discovery)
    """
    kg_edges = set(kg_G.edges())
    paths = []
    seen = set()

    start_nodes = list(holdout_G.nodes())
    if rng:
        rng.shuffle(start_nodes)

    def _dfs(current_path: list, depth: int):
        """depth counts edges traversed so far."""
        if len(paths) >= max_paths:
            return
        if depth == hop_count:
            # Check constraints
            edges_in_kg = []
            for i in range(len(current_path) - 1):
                edges_in_kg.append((current_path[i], current_path[i + 1]) in kg_edges)
            n_in_kg = sum(edges_in_kg)
            # ≥1 in KG, ≥1 novel (no additional constraints — fair benchmark)
            if n_in_kg == 0 or n_in_kg == len(edges_in_kg):
                return
            key = tuple(current_path)
            if key in seen:
                return
            seen.add(key)
            paths.append({
                "hop_count": hop_count,
                "full_path": list(current_path),
                "start_entity": current_path[0],
                "end_entity": current_path[-1],
                "correct_sequence": list(current_path[1:-1]),
                "edges_in_kg": edges_in_kg,
            })
            return

        last = current_path[-1]
        neighbors = list(holdout_G.successors(last))
        if rng:
            rng.shuffle(neighbors)
        for nxt in neighbors[:50]:  # limit branching
            if nxt not in kg_nodes:
                continue
            if nxt in current_path:  # no cycles
                continue
            current_path.append(nxt)
            _dfs(current_path, depth + 1)
            current_path.pop()
            if len(paths) >= max_paths:
                return

    for A in start_nodes:
        if A not in kg_nodes:
            continue
        _dfs([A], 0)
        if len(paths) >= max_paths:
            break

    return paths


# ============================================================
# Distractor Generation
# ============================================================

def generate_distractors(
    kg_G: nx.DiGraph,
    start: str,
    end_entity: str,
    correct_seq: List[str],
    hop_count: int,
    rng: random.Random,
    n: int = 3,
) -> List[dict]:
    """Generate distractor option sequences via random walk in KG.

    1-hop: single KG neighbor of start (excluding end)
    2-hop: single KG neighbor of start (excluding correct B and end C)
    3-hop: 2-step random walk from start → [X, Y]
    4-hop: 3-step random walk from start → [X, Y, Z]

    Returns list of dicts: {"entities": [...], "text": "X → Y → Z"}
    """
    seq_len = len(correct_seq) if correct_seq else 1  # 1-hop → 1 node, 2-hop → 1 node, 3-hop → 2 nodes, 4-hop → 3 nodes
    excluded = set(correct_seq) | {start, end_entity}
    distractors = []
    attempts = 0
    max_attempts = n * 50

    kg_undirected = kg_G.to_undirected()

    while len(distractors) < n and attempts < max_attempts:
        attempts += 1
        # Random walk from start in KG
        path = [start]
        valid = True
        for step in range(seq_len):
            current = path[-1]
            if current not in kg_undirected:
                valid = False
                break
            neighbors = list(kg_undirected.neighbors(current))
            candidates = [nb for nb in neighbors if nb not in set(path) and nb != end_entity]
            if not candidates:
                valid = False
                break
            path.append(rng.choice(candidates))

        if not valid:
            continue

        seq = path[1:]  # exclude start
        assert len(seq) == seq_len

        # Constraint 1: last node must not connect to end_entity (prevents accidental correct answers)
        if end_entity and kg_undirected.has_edge(seq[-1], end_entity):
            continue

        # Constraint 2: no overlap with the correct answer
        overlap = len(set(seq) & excluded)
        if overlap > 0:
            continue

        # Constraint 3: no more than 50% node overlap with existing distractors
        too_similar = False
        for existing in distractors:
            common = len(set(seq) & set(existing["entities"]))
            if common > len(seq) * 0.5:
                too_similar = True
                break
        if too_similar:
            continue

        text = " → ".join(make_entity_readable(e) for e in seq)
        distractors.append({"entities": seq, "text": text})

    return distractors


# ============================================================
# Question Templates
# ============================================================

TEMPLATES = {
    1: [
        "Which phenomenon is a direct consequence of {start} in the context of heatwave cascading risks?",
        "In heatwave risk research, what is the most immediate downstream effect of {start}?",
        "When studying cascading heatwave impacts, which factor is most directly driven by {start}?",
        "Among the following, which is most plausibly a direct result of {start} according to climate risk literature?",
    ],
    2: [
        "In the causal chain from {start} to {end}, what is the intermediate mechanism?",
        "Research shows that {start} leads to impacts on {end} via a cascading pathway. What serves as the primary bridging mechanism?",
        "When examining how {start} contributes to {end} in heat-related risks, what intermediate factor connects them?",
        "Scientists found that the impact of {start} on {end} is mediated by an intermediate process. Which mechanism best explains this?",
    ],
    3: [
        "Research shows {start} ultimately leads to {end} through a two-step cascade. What is the intermediate pathway?",
        "In the multi-step causal chain from {start} to {end}, what two-step intermediate process connects them?",
        "Climate risk analysis reveals that {start} affects {end} via two sequential mechanisms. What is this pathway?",
        "What two-step bridging pathway most plausibly links {start} to {end} in heatwave risk cascades?",
    ],
    4: [
        "In a complex cascading chain from {start} to {end}, researchers identified a three-step intermediate process. What is it?",
        "The pathway from {start} to {end} involves three intermediate steps. What sequence of mechanisms connects them?",
        "Climate scientists traced a four-hop causal chain from {start} to {end}. What three-step pathway bridges them?",
        "What three-step cascading process most plausibly connects {start} to {end} in heatwave risk propagation?",
    ],
}


def build_question(
    path_info: dict,
    distractors: List[dict],
    answer_pos: str,
    qid: str,
    rng: random.Random,
) -> Optional[dict]:
    """Build one MCQ from path + distractors."""
    hop = path_info["hop_count"]
    start = path_info["start_entity"]
    end = path_info["end_entity"]
    correct_seq = path_info["correct_sequence"]

    start_r = make_entity_readable(start)
    end_r = make_entity_readable(end)

    # Question text
    templates = TEMPLATES[hop]
    if hop == 1:
        question = rng.choice(templates).format(start=start_r)
    else:
        question = rng.choice(templates).format(start=start_r, end=end_r)

    # Correct option
    if hop == 1:
        correct_text = make_entity_readable(end)
        correct_entities = [end]
    else:
        correct_text = " → ".join(make_entity_readable(e) for e in correct_seq)
        correct_entities = list(correct_seq)

    # Build options list
    all_opts = [{"text": correct_text, "entities": correct_entities}]
    for d in distractors[:3]:
        all_opts.append(d)
    if len(all_opts) < 4:
        return None

    # Place correct answer at answer_pos
    target_idx = ord(answer_pos) - ord("A")
    correct_opt = all_opts.pop(0)
    all_opts.insert(target_idx, correct_opt)

    options = []
    for i, opt in enumerate(all_opts):
        label = chr(65 + i)
        options.append({
            "label": label,
            "text": opt["text"],
            "entities": opt["entities"],
        })

    return {
        "id": qid,
        "hop_count": hop,
        "start_entity": start,
        "end_entity": end,
        "correct_sequence": correct_seq,
        "full_path": path_info["full_path"],
        "edges_in_kg": path_info["edges_in_kg"],
        "question": question,
        "options": options,
        "answer": answer_pos,
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Multi-hop LBD Benchmark Generator")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSONL path (default: data/benchmark/multihop_lbd.jsonl)")
    parser.add_argument("--per-hop", type=int, default=200,
                        help="Questions per hop (default 200 for pilot; use 1000 for full)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hops", type=int, nargs="+", default=[1, 2, 3, 4],
                        help="Which hops to generate (default: 1 2 3 4)")
    parser.add_argument("--max-mine", type=int, default=5000,
                        help="Max paths to mine per hop")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    output_path = Path(args.output) if args.output else BENCHMARK_DIR / "multihop_lbd.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Multi-Hop LBD Benchmark Generator")
    print(f"  Hops: {args.hops}")
    print(f"  Per-hop: {args.per_hop}")
    print(f"  Output: {output_path}")

    # ── Load data ──
    print("\n[1/4] Loading data ...")
    merge_map = load_merge_mapping()
    print(f"  Merge mapping: {len(merge_map)} → {len(set(merge_map.values()))} canonical")

    kg_G, kg_nodes = load_kg_graph(merge_map)
    print(f"  KG: {kg_G.number_of_nodes()} nodes, {kg_G.number_of_edges()} edges")

    holdout_G = load_holdout_graph(merge_map)
    print(f"  Holdout: {holdout_G.number_of_nodes()} nodes, {holdout_G.number_of_edges()} edges")

    overlap = kg_nodes & set(holdout_G.nodes())
    print(f"  Overlap: {len(overlap)} entities ({len(overlap)/len(set(holdout_G.nodes()))*100:.1f}%)")

    # ── Mine paths ──
    print("\n[2/4] Mining paths ...")
    all_mined = {}
    for hop in args.hops:
        if hop == 1:
            paths = mine_1hop_paths(holdout_G, kg_G, kg_nodes, args.max_mine, rng)
        else:
            paths = mine_nhop_paths(holdout_G, kg_G, kg_nodes, hop, args.max_mine, rng)
        all_mined[hop] = paths
        print(f"  {hop}-hop: {len(paths)} paths mined")

    # ── Generate questions ──
    print("\n[3/4] Generating questions ...")
    all_questions = []
    answer_cycle = ["A", "B", "C", "D"]

    for hop in args.hops:
        paths = all_mined[hop]
        rng.shuffle(paths)
        generated = 0
        skipped = 0

        for path_info in paths:
            if generated >= args.per_hop:
                break

            distractors = generate_distractors(
                kg_G, path_info["start_entity"], path_info["end_entity"],
                path_info["correct_sequence"], hop, rng, n=3,
            )
            if len(distractors) < 3:
                skipped += 1
                continue

            qid = f"hop{hop}-{generated+1:04d}"
            ans_pos = answer_cycle[(len(all_questions)) % 4]

            q = build_question(path_info, distractors, ans_pos, qid, rng)
            if q:
                all_questions.append(q)
                generated += 1

        print(f"  {hop}-hop: {generated} generated, {skipped} skipped (insufficient distractors)")

    # ── Save ──
    print(f"\n[4/4] Saving {len(all_questions)} questions ...")

    # JSONL (primary format)
    with open(output_path, "w", encoding="utf-8") as f:
        for q in all_questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"  JSONL: {output_path}")

    # CSV (simplified for quick viewing)
    csv_path = output_path.with_suffix(".csv")
    import csv
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "HopCount", "Question", "Options", "Answer",
                         "StartEntity", "EndEntity", "CorrectSequence"])
        for q in all_questions:
            opts_str = " | ".join(f"{o['label']}. {o['text']}" for o in q["options"])
            writer.writerow([
                q["id"], q["hop_count"], q["question"], opts_str, q["answer"],
                q["start_entity"], q["end_entity"],
                " → ".join(q["correct_sequence"]) if q["correct_sequence"] else q["end_entity"],
            ])
    print(f"  CSV:  {csv_path}")

    # Stats
    stats = {
        "total": len(all_questions),
        "per_hop": {},
        "answer_distribution": {},
    }
    for hop in args.hops:
        hop_qs = [q for q in all_questions if q["hop_count"] == hop]
        stats["per_hop"][str(hop)] = {
            "count": len(hop_qs),
            "paths_mined": len(all_mined.get(hop, [])),
        }
    for letter in "ABCD":
        stats["answer_distribution"][letter] = sum(
            1 for q in all_questions if q["answer"] == letter
        )

    stats_path = output_path.with_name(output_path.stem + "_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  Stats: {stats_path}")

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  Total: {len(all_questions)} questions")
    for hop in args.hops:
        hop_qs = [q for q in all_questions if q["hop_count"] == hop]
        print(f"  {hop}-hop: {len(hop_qs)}")
    print(f"  Answer dist: {stats['answer_distribution']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
