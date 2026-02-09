#!/usr/bin/env python3

import json
import logging
import yaml
import networkx as nx
from pathlib import Path
from typing import List, Dict, Any
import openai
from collections import defaultdict, deque
import argparse
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import threading
import random
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MultiHopReasoningEngine:

    def __init__(self, config_path: str = "config/default.yaml", max_threads: int = 16):
        self.config = self._load_config(config_path)
        self.client = self._init_llm_client()
        self.graph = nx.DiGraph()
        self.entity_to_layer = {}
        self.max_hops = 5
        self.min_hops = 3
        self.confidence_threshold = 0.3
        self.max_threads = max_threads
        self.lock = threading.Lock()
        
        self.adjacency_list = defaultdict(list)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        try:
            config_file = Path(config_path)
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                logger.info(f"Successfully loaded configuration file: {config_path}")
                return config
            else:
                logger.warning(f"Configuration file does not exist: {config_path}, using default configuration")
                return self._get_default_config()
        except Exception as e:
            logger.warning(f"Failed to load configuration file {config_path}: {e}, using default configuration")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        return {
            "llm": {
                "api_key": os.environ.get('HEDA_API_KEY', 'YOUR_API_KEY_HERE'),
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "model": "qwen3-max"
            }
        }

    def _init_llm_client(self):
        llm_config = self.config.get("llm", {})
        return openai.OpenAI(
            api_key=llm_config.get("api_key"),
            base_url=llm_config.get("base_url")
        )

    def load_enhanced_triples(self, enhanced_json_dir: str = "enhanced_json") -> bool:
        try:
            enhanced_dir = Path(enhanced_json_dir)
            if not enhanced_dir.exists():
                logger.error(f"Enhanced triples directory does not exist: {enhanced_dir}")
                return False

            json_files = list(enhanced_dir.glob("*.json"))
            if not json_files:
                logger.error(f"No enhanced triple files found")
                return False

            logger.info(f"Start loading {len(json_files)} enhanced triple files...")

            total_triples = 0
            layer_counts = defaultdict(int)

            for idx, json_file in enumerate(json_files):
                if (idx + 1) % 1000 == 0:
                    logger.info(f"Processed {idx + 1}/{len(json_files)} files...")

                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        triples = json.load(f)

                    for triple in triples:
                        start_node = triple.get("start_node", "")
                        end_node = triple.get("end_node", "")
                        relationship = triple.get("relationship", "")

                        if not start_node or not end_node or not relationship:
                            continue

                        layer = triple.get("layer", "unknown")
                        self.entity_to_layer[start_node] = layer
                        self.entity_to_layer[end_node] = layer
                        layer_counts[layer] += 1

                        edge_data = {
                            "relationship": relationship,
                            "layer": layer,
                            "relation_type": triple.get("relation_type", "other"),
                            "confidence": triple.get("confidence", 0.5),
                            "source_file": json_file.name
                        }

                        self.graph.add_edge(start_node, end_node, **edge_data)

                        self.adjacency_list[start_node].append({
                            "to": end_node,
                            **edge_data
                        })

                        total_triples += 1

                except Exception as e:
                    logger.warning(f"Failed to process file {json_file.name}: {e}")
                    continue

            logger.info(f"\n{'='*60}")
            logger.info(f"Graph construction complete:")
            logger.info(f"  Nodes: {self.graph.number_of_nodes():,}")
            logger.info(f"  Edges: {self.graph.number_of_edges():,}")
            logger.info(f"  Total Triples: {total_triples:,}")
            logger.info(f"\nLayer Distribution:")
            for layer, count in sorted(layer_counts.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"    {layer}: {count:,}")
            logger.info(f"{'='*60}\n")

            return True

        except Exception as e:
            logger.error(f"Failed to load enhanced triples: {e}")
            return False

    def find_paths_with_length_priority(
        self,
        start_entity: str,
        end_entity: str,
        min_hops: int = 3,
        max_hops: int = 5,
        max_paths_per_length: int = 5
    ) -> Dict[int, List[List[Dict]]]:
        try:
            if start_entity not in self.graph.nodes() or end_entity not in self.graph.nodes():
                return {}

            paths_by_length = defaultdict(list)

            queue = deque([(start_entity, [start_entity])])
            visited_paths = set()
            max_queue_size = 20000

            while queue:
                if len(queue) > max_queue_size:
                    break

                current_node, path = queue.popleft()

                path_length = len(path) - 1

                if path_length > max_hops:
                    continue

                if current_node == end_entity and path_length >= min_hops:
                    path_details = []
                    for i in range(len(path) - 1):
                        edge_data = self.graph.get_edge_data(path[i], path[i + 1])
                        if edge_data:
                            path_details.append({
                                "from": path[i],
                                "to": path[i + 1],
                                "relationship": edge_data.get("relationship", "unknown"),
                                "layer": edge_data.get("layer", "unknown"),
                                "relation_type": edge_data.get("relation_type", "other"),
                                "confidence": edge_data.get("confidence", 0.5)
                            })

                    if path_details:
                        hops = len(path_details)
                        if len(paths_by_length[hops]) < max_paths_per_length:
                            paths_by_length[hops].append(path_details)
                    continue

                if path_length < max_hops:
                    for neighbor in self.graph.successors(current_node):
                        if neighbor not in path:
                            new_path = path + [neighbor]
                            path_key = tuple(new_path)
                            if path_key not in visited_paths:
                                visited_paths.add(path_key)
                                queue.append((neighbor, new_path))

            return dict(paths_by_length)

        except Exception as e:
            logger.error(f"Path search failed ({start_entity} -> {end_entity}): {e}")
            return {}

    def _search_paths_for_node_pair(self, args):
        start_node, end_node, min_hops, max_hops, causal_keywords, search_type = args

        try:
            paths_by_length = self.find_paths_with_length_priority(
                start_node, end_node, min_hops=min_hops, max_hops=max_hops, max_paths_per_length=3
            )

            results = []

            for hops in sorted(paths_by_length.keys(), reverse=True):
                for path in paths_by_length[hops]:
                    if search_type == "causal":
                        has_causal = any(
                            any(keyword in step["relationship"].lower() for keyword in causal_keywords)
                            for step in path
                        )
                        if not has_causal:
                            continue

                    results.append({
                        "type": search_type,
                        "start": start_node,
                        "end": end_node,
                        "path": path,
                        "hops": len(path),
                        "strength": self._calculate_path_strength(path),
                        "layers": self._get_path_layers(path)
                    })

            return results

        except Exception as e:
            return []

    def find_causal_chains(
        self,
        min_hops: int = 3,
        max_hops: int = 5,
        top_k: int = 50
    ) -> List[Dict]:
        logger.info(f"Start searching for causal chains (Hops: {min_hops}-{max_hops})...")
        logger.info(f"Using multi-threaded parallel search, max threads: {self.max_threads}")

        causal_keywords = [
            "cause", "lead", "result", "trigger", "induce",
            "contribute", "influence", "affect", "impact", "drive"
        ]

        all_nodes = list(self.graph.nodes())
        logger.info(f"Total nodes in graph: {len(all_nodes):,}")

        random.seed(42)
        node_degrees = [(node, dict(self.graph.degree())[node]) for node in all_nodes]
        node_degrees.sort(key=lambda x: x[1], reverse=True)

        high_degree_nodes = [node for node, _ in node_degrees[:800]]
        logger.info(f"Searching causal chains from {len(high_degree_nodes)} high-degree nodes...")

        tasks = []
        for i, start_node in enumerate(high_degree_nodes[:400]):
            sample_size = min(40, len(high_degree_nodes))
            target_nodes = random.sample(high_degree_nodes, sample_size)

            for end_node in target_nodes:
                if start_node != end_node:
                    tasks.append((start_node, end_node, min_hops, max_hops, causal_keywords, "causal_chain"))

        logger.info(f"Prepared {len(tasks)} node pair search tasks...")

        chains = []
        completed = 0

        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            futures = {executor.submit(self._search_paths_for_node_pair, task): task for task in tasks}

            for future in as_completed(futures):
                completed += 1
                if completed % 500 == 0:
                    logger.info(f"  Completed {completed}/{len(tasks)} tasks, found {len(chains)} causal chains")

                try:
                    results = future.result()
                    if results:
                        chains.extend(results)

                        if len(chains) >= top_k * 5:
                            logger.info(f"Found enough causal chains ({len(chains)}), stopping search early")
                            for f in futures:
                                f.cancel()
                            break

                except Exception as e:
                    pass

        logger.info(f"Search complete: Found {len(chains)} causal chains")

        def sort_key(chain):
            hops = chain.get("hops", 0)
            strength = chain.get("strength", 0.0)
            if hops >= 5:
                return (3, strength)
            elif hops == 4:
                return (2, strength)
            else:
                return (1, strength)

        chains.sort(key=sort_key, reverse=True)

        gc.collect()

        return chains[:top_k]

    def find_cross_layer_paths(
        self,
        min_hops: int = 3,
        max_hops: int = 5,
        top_k: int = 50
    ) -> List[Dict]:
        logger.info(f"Start searching for cross-layer paths (Hops: {min_hops}-{max_hops})...")
        logger.info(f"Using multi-threaded parallel search, max threads: {self.max_threads}")

        all_nodes = list(self.graph.nodes())
        layer_nodes = defaultdict(list)
        for node in all_nodes:
            layer = self.entity_to_layer.get(node, "unknown")
            if layer != "unknown":
                layer_nodes[layer].append(node)

        logger.info(f"Layer Distribution:")
        for layer, nodes in layer_nodes.items():
            logger.info(f"  {layer}: {len(nodes):,} nodes")

        random.seed(42)
        tasks = []
        layers = list(layer_nodes.keys())

        for i, from_layer in enumerate(layers):
            for to_layer in layers[i+1:]:
                from_nodes = random.sample(layer_nodes[from_layer], min(150, len(layer_nodes[from_layer])))
                to_nodes = random.sample(layer_nodes[to_layer], min(150, len(layer_nodes[to_layer])))

                for start_node in from_nodes:
                    for end_node in to_nodes:
                        tasks.append((start_node, end_node, min_hops, max_hops, [], "cross_layer"))

        logger.info(f"Prepared {len(tasks)} node pair search tasks...")

        cross_layer_paths = []
        completed = 0

        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            futures = {executor.submit(self._search_paths_for_node_pair, task): task for task in tasks}

            for future in as_completed(futures):
                completed += 1
                if completed % 500 == 0:
                    logger.info(f"  Completed {completed}/{len(tasks)} tasks, found {len(cross_layer_paths)} cross-layer paths")

                try:
                    results = future.result()
                    if results:
                        for result in results:
                            path_layers = result.get("layers", [])
                            if len(set(path_layers)) > 1:
                                cross_layer_paths.append(result)

                        if len(cross_layer_paths) >= top_k * 5:
                            logger.info(f"Found enough cross-layer paths ({len(cross_layer_paths)}), stopping search early")
                            for f in futures:
                                f.cancel()
                            break

                except Exception as e:
                    pass

        logger.info(f"Search complete: Found {len(cross_layer_paths)} cross-layer paths")

        def sort_key(chain):
            hops = chain.get("hops", 0)
            strength = chain.get("strength", 0.0)
            if hops >= 5:
                return (3, strength)
            elif hops == 4:
                return (2, strength)
            else:
                return (1, strength)

        cross_layer_paths.sort(key=sort_key, reverse=True)

        gc.collect()

        return cross_layer_paths[:top_k]

    def _calculate_path_strength(self, path: List[Dict]) -> float:
        if not path:
            return 0.0

        confidences = [step.get("confidence", 0.5) for step in path]
        avg_confidence = sum(confidences) / len(confidences)

        length_bonus = 1.0
        path_len = len(path)
        if path_len >= 5:
            length_bonus = 1.3
        elif path_len == 4:
            length_bonus = 1.15
        elif path_len == 3:
            length_bonus = 1.0

        return avg_confidence * length_bonus

    def _get_path_layers(self, path: List[Dict]) -> List[str]:
        layers = []
        for step in path:
            layer = step.get("layer", "unknown")
            if layer not in layers:
                layers.append(layer)
        return layers

    def analyze_risk_chains(self) -> Dict[str, Any]:
        logger.info("\n" + "="*60)
        logger.info("Starting Risk Chain Analysis (Optimized Version)")
        logger.info("="*60 + "\n")

        causal_chains = self.find_causal_chains(min_hops=3, max_hops=5, top_k=50)

        cross_layer_paths = self.find_cross_layer_paths(min_hops=3, max_hops=5, top_k=50)

        all_chains = causal_chains + cross_layer_paths

        analysis = {
            "total_chains": len(all_chains),
            "causal_chains": len(causal_chains),
            "cross_layer_paths": len(cross_layer_paths),
            "top_causal_chains": causal_chains[:10],
            "top_cross_layer_paths": cross_layer_paths[:10],
            "hop_distribution": self._analyze_hop_distribution(all_chains),
            "layer_distribution": self._analyze_layer_distribution(all_chains),
            "strength_statistics": self._analyze_strength_statistics(all_chains),
            "timestamp": datetime.now().isoformat()
        }

        return analysis

    def _analyze_hop_distribution(self, chains: List[Dict]) -> Dict[int, int]:
        hop_dist = defaultdict(int)
        for chain in chains:
            hops = chain.get("hops", 0)
            hop_dist[hops] += 1
        return dict(sorted(hop_dist.items()))

    def _analyze_layer_distribution(self, chains: List[Dict]) -> Dict[str, int]:
        layer_dist = defaultdict(int)
        for chain in chains:
            layers = chain.get("layers", [])
            for layer in layers:
                layer_dist[layer] += 1
        return dict(sorted(layer_dist.items(), key=lambda x: x[1], reverse=True))

    def _analyze_strength_statistics(self, chains: List[Dict]) -> Dict[str, float]:
        if not chains:
            return {"mean": 0.0, "max": 0.0, "min": 0.0, "median": 0.0}

        strengths = [chain.get("strength", 0.0) for chain in chains]
        return {
            "mean": sum(strengths) / len(strengths),
            "max": max(strengths),
            "min": min(strengths),
            "median": sorted(strengths)[len(strengths) // 2]
        }

    def save_reports(self, analysis: Dict[str, Any], output_dir: str = "reports") -> None:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        json_file = output_path / "risk_chains.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)
        logger.info(f"✅ Full report saved: {json_file}")

        multi_hop_file = output_path / "multi_hop_analysis.json"
        multi_hop_data = {
            "hop_distribution": analysis["hop_distribution"],
            "layer_distribution": analysis["layer_distribution"],
            "strength_statistics": analysis["strength_statistics"],
            "total_chains": analysis["total_chains"],
            "timestamp": analysis["timestamp"]
        }
        with open(multi_hop_file, 'w', encoding='utf-8') as f:
            json.dump(multi_hop_data, f, ensure_ascii=False, indent=2)
        logger.info(f"✅ Multi-hop analysis saved: {multi_hop_file}")

        summary_file = output_path / "risk_chains_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("Risk Chain Analysis Report (Optimized - 3-5 hops, 4+ hops priority)\n")
            f.write("="*80 + "\n\n")

            f.write(f"Generation Time: {analysis['timestamp']}\n")
            f.write(f"Total Chains: {analysis['total_chains']}\n")
            f.write(f"  - Causal Chains: {analysis['causal_chains']}\n")
            f.write(f"  - Cross-layer Paths: {analysis['cross_layer_paths']}\n\n")

            f.write("-"*80 + "\n")
            f.write("Hop Distribution:\n")
            f.write("-"*80 + "\n")
            for hops, count in analysis["hop_distribution"].items():
                f.write(f"  {hops} hops: {count} chains\n")
            f.write("\n")

            f.write("-"*80 + "\n")
            f.write("Layer Distribution:\n")
            f.write("-"*80 + "\n")
            for layer, count in analysis["layer_distribution"].items():
                f.write(f"  {layer}: {count} occurrences\n")
            f.write("\n")

            f.write("-"*80 + "\n")
            f.write("Path Strength Statistics:\n")
            f.write("-"*80 + "\n")
            stats = analysis["strength_statistics"]
            f.write(f"  Mean: {stats['mean']:.4f}\n")
            f.write(f"  Max: {stats['max']:.4f}\n")
            f.write(f"  Min: {stats['min']:.4f}\n")
            f.write(f"  Median: {stats['median']:.4f}\n\n")

            f.write("="*80 + "\n")
            f.write("Top 10 Causal Chains (4+ hops priority)\n")
            f.write("="*80 + "\n\n")
            for i, chain in enumerate(analysis["top_causal_chains"], 1):
                f.write(f"[{i}] {chain['start']}→ {chain['end']}\n")
                f.write(f"    Hops: {chain['hops']}, Strength: {chain['strength']:.4f}\n")
                f.write(f"    Path:\n")
                for step in chain["path"]:
                    f.write(f"      {step['from']}--[{step['relationship']}]--> {step['to']}\n")
                f.write("\n")

            f.write("="*80 + "\n")
            f.write("Top 10 Cross-layer Paths (4+ hops priority)\n")
            f.write("="*80 + "\n\n")
            for i, chain in enumerate(analysis["top_cross_layer_paths"], 1):
                f.write(f"[{i}] {chain.get('from_layer', 'N/A')} → {chain.get('to_layer', 'N/A')}\n")
                f.write(f"    Start: {chain['start']}, End: {chain['end']}\n")
                f.write(f"    Hops: {chain['hops']}, Strength: {chain['strength']:.4f}\n")
                f.write(f"    Path:\n")
                for step in chain["path"]:
                    f.write(f"      {step['from']}--[{step['relationship']}]--> {step['to']}\n")
                f.write("\n")

        logger.info(f"✅ Readable report saved: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="Multi-hop Reasoning Engine - Risk Propagation Chain Discovery (Optimized)")
    parser.add_argument("--enhanced-json", default="enhanced_json", help="Enhanced triples directory")
    parser.add_argument("--output-dir", default="reports", help="Output directory")
    parser.add_argument("--min-hops", type=int, default=3, help="Minimum hops (default 3, exclude 2 hops)")
    parser.add_argument("--max-hops", type=int, default=5, help="Maximum hops (default 5)")
    parser.add_argument("--top-k", type=int, default=50, help="Number of top chains to keep per category")
    parser.add_argument("--max-threads", type=int, default=16, help="Maximum threads (default 16)")
    parser.add_argument("--config", default="config/default.yaml", help="Configuration file path")

    args = parser.parse_args()

    logger.info("="*80)
    logger.info("Multi-hop Reasoning Engine - Risk Propagation Chain Discovery (Optimized)")
    logger.info("="*80)
    logger.info(f"Configuration:")
    logger.info(f"  Enhanced Triples Directory: {args.enhanced_json}")
    logger.info(f"  Output Directory: {args.output_dir}")
    logger.info(f"  Hop Range: {args.min_hops}-{args.max_hops} (excluding 2 hops)")
    logger.info(f"  Top-K: {args.top_k}")
    logger.info(f"  Max Threads: {args.max_threads}")
    logger.info(f"  Priority: 4+ hops > 4 hops > 3 hops")
    logger.info("="*80 + "\n")

    engine = MultiHopReasoningEngine(config_path=args.config, max_threads=args.max_threads)

    if not engine.load_enhanced_triples(args.enhanced_json):
        logger.error("❌ Failed to load knowledge graph")
        return

    analysis = engine.analyze_risk_chains()

    engine.save_reports(analysis, args.output_dir)

    logger.info("\n" + "="*80)
    logger.info("Analysis Complete Summary")
    logger.info("="*80)
    logger.info(f"✅ Total Chains: {analysis['total_chains']}")
    logger.info(f"   - Causal Chains: {analysis['causal_chains']}")
    logger.info(f"   - Cross-layer Paths: {analysis['cross_layer_paths']}")
    logger.info(f"\n📊 Hop Distribution: {analysis['hop_distribution']}")
    logger.info(f"\n📈 Strength Statistics:")
    for key, value in analysis['strength_statistics'].items():
        logger.info(f"   {key}: {value:.4f}")
    logger.info(f"\n📁 Reports saved to: {args.output_dir}/")
    logger.info("="*80)


if __name__ == "__main__":
    main()