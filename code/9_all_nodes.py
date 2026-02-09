#!/usr/bin/env python3
import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

import pandas as pd


def resolve_json_dir() -> Path | None:
    enhanced_dir = Path("enhanced_json")
    original_dir = Path("json")

    if enhanced_dir.exists() and list(enhanced_dir.glob("*.json")):
        return enhanced_dir
    if original_dir.exists() and list(original_dir.glob("*.json")):
        return original_dir
    return None


def normalize_fields(relation: dict) -> dict:
    return {
        str(key).lower().replace(" ", "").replace("_", ""): value
        for key, value in relation.items()
    }


def collect_nodes_and_adjacency(json_dir: Path) -> Tuple[Set[str], Dict[str, List[List[str]]], int, int, int]:
    unique_nodes: Set[str] = set()
    adjacency: Dict[str, List[List[str]]] = {}
    total_relations = 0
    valid_relations = 0
    invalid_relations = 0

    json_files = sorted(json_dir.glob("*.json"))
    logging.info("Found %d JSON files", len(json_files))

    iterator = tqdm(json_files, desc="Processing JSON files", unit="file") if tqdm else json_files

    for json_file in iterator:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        if isinstance(data, dict):
            if isinstance(data.get("data"), list):
                data = data["data"]
            elif isinstance(data.get("triples"), list):
                data = data["triples"]
            else:
                continue

        if not isinstance(data, list):
            continue

        for relation in data:
            if not isinstance(relation, dict):
                invalid_relations += 1
                continue

            total_relations += 1
            fields = normalize_fields(relation)
            start = fields.get("startnode")
            end = fields.get("endnode")
            rel = fields.get("relationship", "")

            if isinstance(start, str) and isinstance(end, str):
                start = start.strip()
                end = end.strip()
                if start and end:
                    unique_nodes.add(start)
                    unique_nodes.add(end)
                    adjacency.setdefault(start, []).append([rel, end])
                    valid_relations += 1
                else:
                    invalid_relations += 1
            else:
                invalid_relations += 1

    return unique_nodes, adjacency, total_relations, valid_relations, invalid_relations


def write_csv(nodes: Set[str], output_path: Path) -> None:
    df = pd.DataFrame(sorted(nodes), columns=["Node"])
    df.to_csv(output_path, index=False, encoding="utf-8")


def write_adjacency(adjacency: Dict[str, List[List[str]]], output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(adjacency, f, ensure_ascii=False)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    json_dir = resolve_json_dir()
    if not json_dir:
        logging.error("Neither enhanced_json nor json has .json files.")
        return 1

    logging.info("Using JSON dir: %s", json_dir)
    nodes, adjacency, total_relations, valid_relations, invalid_relations = collect_nodes_and_adjacency(json_dir)

    nodes_file = Path("all_nodes.csv")
    adjacency_file = Path("adjacency.json")

    write_csv(nodes, nodes_file)
    write_adjacency(adjacency, adjacency_file)

    logging.info(
        "Done. nodes=%d edges=%d adjacency_keys=%d output=%s, %s",
        len(nodes),
        valid_relations,
        len(adjacency),
        nodes_file,
        adjacency_file,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

