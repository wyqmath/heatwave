#!/usr/bin/env python3

import argparse
import json
import logging
import os
import re
from collections import defaultdict
from pathlib import Path

import yaml
from neo4j import GraphDatabase
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_config(config_path: Path) -> dict:
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as exc:
        logging.warning("Failed to load configuration file %s: %s", config_path, exc)
        return {}


def get_neo4j_config(config: dict) -> dict:
    neo_cfg = (config.get("knowledge_graph") or {}).get("neo4j") or {}
    return {
        "uri": os.getenv("NEO4J_URI", neo_cfg.get("uri", "bolt://localhost:7687")),
        "user": os.getenv("NEO4J_USER", neo_cfg.get("user", "neo4j")),
        "password": os.getenv("NEO4J_PASSWORD", neo_cfg.get("password", "")),
    }


def clean_relationship_name(relationship: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_\u4e00-\u9fff]", "_", relationship or "")
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        return "REL"
    if cleaned[0].isdigit():
        return f"REL_{cleaned}"
    return cleaned


def choose_json_dir(root_dir: Path) -> Path:
    enhanced_dir = root_dir / "enhanced_json"
    json_dir = root_dir / "json"
    if enhanced_dir.exists():
        return enhanced_dir
    if json_dir.exists():
        return json_dir
    raise FileNotFoundError(f"JSON directory not found: {enhanced_dir} or {json_dir}")


def clear_database(driver) -> None:
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
        try:
            session.run("DROP INDEX node_name_index IF EXISTS")
        except Exception:
            pass


def create_indexes(driver) -> None:
    with driver.session() as session:
        session.run("CREATE INDEX node_name_index IF NOT EXISTS FOR (n:Node) ON (n.name)")


def load_triples(json_dir: Path):
    data_by_rel = defaultdict(list)
    total = 0
    for file_path in sorted(json_dir.glob("*.json")):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                file_data = json.load(f)
        except Exception as exc:
            logging.error("Read failed %s: %s", file_path.name, exc)
            continue
        for record in file_data:
            if not all(k in record for k in ("start_node", "relationship", "end_node")):
                logging.warning("Invalid record %s: %s", file_path.name, record)
                continue
            rel_type = clean_relationship_name(record.get("relationship"))
            data_by_rel[rel_type].append({
                "start": str(record.get("start_node", "")).strip(),
                "end": str(record.get("end_node", "")).strip(),
                "relationship_raw": record.get("relationship"),
                "layer": record.get("layer"),
                "relation_type": record.get("relation_type"),
                "confidence": record.get("confidence"),
                "enhanced_timestamp": record.get("enhanced_timestamp"),
            })
            total += 1
    return data_by_rel, total


def _write_batch(tx, rel_type: str, records: list) -> None:
    query = f"""
    UNWIND $records AS row
    MERGE (a:Node {{name: row.start}})
    MERGE (b:Node {{name: row.end}})
    MERGE (a)-[r:`{rel_type}`]->(b)
    ON CREATE SET r.relationship_raw = row.relationship_raw,
                  r.layer = row.layer,
                  r.relation_type = row.relation_type,
                  r.confidence = row.confidence,
                  r.enhanced_timestamp = row.enhanced_timestamp
    """
    tx.run(query, records=records)


def import_grouped(driver, data_by_rel: dict, total: int, batch_size: int) -> int:
    processed = 0
    with driver.session() as session, tqdm(total=total, desc="Importing data", unit="records") as pbar:
        for rel_type, records in data_by_rel.items():
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                try:
                    session.execute_write(_write_batch, rel_type, batch)
                except Exception as exc:
                    logging.error("Batch failed (%s): %s", rel_type, exc)
                processed += len(batch)
                pbar.update(len(batch))
    return processed


def parse_args():
    parser = argparse.ArgumentParser(description="Step 8: Upload Knowledge Graph to Neo4j")
    parser.add_argument("--no-clear", action="store_true", help="Do not clear the database")
    parser.add_argument("--batch-size", type=int, default=2000, help="Batch size")
    return parser.parse_args()


def main():
    args = parse_args()
    root_dir = Path(__file__).resolve().parent.parent
    config = load_config(root_dir / "config" / "default.yaml")
    neo_cfg = get_neo4j_config(config)

    json_dir = choose_json_dir(root_dir)
    logging.info("Using JSON directory: %s", json_dir)

    data_by_rel, total = load_triples(json_dir)
    logging.info("Loaded records: %s, Relationship types: %s", total, len(data_by_rel))
    if not total:
        logging.error("No valid data found, terminating")
        return

    with GraphDatabase.driver(neo_cfg["uri"], auth=(neo_cfg["user"], neo_cfg["password"])) as driver:
        if not args.no_clear:
            logging.info("Clearing database...")
            clear_database(driver)
        logging.info("Creating indexes...")
        create_indexes(driver)
        processed = import_grouped(driver, data_by_rel, total, args.batch_size)

    logging.info("Import completed, processed %s records", processed)


if __name__ == "__main__":
    main()