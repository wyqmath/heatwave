#!/usr/bin/env python3

import json
import logging
import yaml
import networkx as nx
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple
from collections import defaultdict, Counter, deque
import numpy as np
import argparse
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import gc
import random
import time
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LargeScaleMiningEngine:

    def __init__(self, config_path: str = "config/default.yaml"):
        self.config = self._load_config(config_path)
        self.graph = nx.DiGraph()
        self.entity_to_layer = {}
        self.layer_to_entities = defaultdict(list)

        self.min_hops = 3
        self.max_hops = 6
        self.strength_threshold = 0.1
        self.novelty_threshold = 0.3
        self.max_threads = 16
        self.target_paths = 8000

        self.entity_frequency = Counter()
        self.relation_frequency = Counter()
        self.max_frequency = 1

        self.all_paths = []
        self.lock = threading.Lock()

        self.output_dir = Path("reports/large_scale")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except:
            return {}

    def load_graph(self, enhanced_json_dir: str = "enhanced_json") -> bool:
        logger.info("=" * 60)
        logger.info("Start loading knowledge graph...")
        logger.info("=" * 60)

        json_dir = Path(enhanced_json_dir)
        if not json_dir.exists():
            logger.error(f"Directory does not exist: {enhanced_json_dir}")
            return False

        json_files = list(json_dir.glob("*.json"))
        logger.info(f"Found {len(json_files)} JSON files")

        triplet_count = 0

        for json_file in tqdm(json_files, desc="Loading triplets", unit="file"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if not isinstance(data, list):
                    continue

                for triplet in data:
                    if not isinstance(triplet, dict):
                        continue

                    start_node = triplet.get('start_node')
                    end_node = triplet.get('end_node')
                    relationship = triplet.get('relationship')
                    layer = triplet.get('layer', 'unknown')

                    if not all([start_node, end_node, relationship]):
                        continue

                    self.graph.add_edge(
                        start_node, end_node,
                        relationship=relationship,
                        layer=layer,
                        relation_type=triplet.get('relation_type', 'unknown'),
                        confidence=triplet.get('confidence', 0.5)
                    )

                    self.entity_to_layer[start_node] = layer
                    self.entity_to_layer[end_node] = layer
                    self.layer_to_entities[layer].append(start_node)

                    self.entity_frequency[start_node] += 1
                    self.entity_frequency[end_node] += 1
                    self.relation_frequency[relationship] += 1

                    triplet_count += 1

            except Exception as e:
                continue

        if self.entity_frequency:
            self.max_frequency = max(self.entity_frequency.values())

        logger.info(f"\nGraph loading complete:")
        logger.info(f"  Nodes: {self.graph.number_of_nodes():,}")
        logger.info(f"  Edges: {self.graph.number_of_edges():,}")
        logger.info(f"  Triplets: {triplet_count:,}")
        logger.info(f"\nLayer distribution:")
        for layer, entities in self.layer_to_entities.items():
            logger.info(f"  {layer}: {len(set(entities)):,} nodes")

        return True

    def _find_paths_bfs(self, start: str, end: str, max_length: int = 6, max_paths: int = 20) -> List[List[str]]:
        if start == end or start not in self.graph or end not in self.graph:
            return []

        queue = deque([(start, [start])])
        found_paths = []
        visited_paths = set()
        max_queue_size = 50000

        while queue and len(found_paths) < max_paths:
            if len(queue) > max_queue_size:
                break

            current, path = queue.popleft()

            if len(path) > max_length + 1:
                continue

            if current == end and len(path) >= self.min_hops + 1:
                found_paths.append(path)
                continue

            if len(path) <= max_length:
                for neighbor in self.graph.successors(current):
                    if neighbor not in path:
                        new_path = path + [neighbor]
                        path_key = tuple(new_path)
                        if path_key not in visited_paths:
                            visited_paths.add(path_key)
                            queue.append((neighbor, new_path))

        return found_paths

    def _calculate_path_strength(self, path: List[str]) -> float:
        if len(path) < 2:
            return 0.0

        confidences = []
        for i in range(len(path) - 1):
            edge_data = self.graph.get_edge_data(path[i], path[i + 1])
            if edge_data:
                confidences.append(edge_data.get('confidence', 0.5))

        if not confidences:
            return 0.0

        avg_conf = sum(confidences) / len(confidences)

        length_bonus = 1.0 + (len(path) - 3) * 0.1
        return avg_conf * length_bonus

    def _calculate_novelty_score(self, path: List[str]) -> float:
        if len(path) < 2:
            return 0.0

        entity_freqs = [self.entity_frequency.get(n, 1) for n in path]
        f_p = np.exp(np.mean(np.log(np.array(entity_freqs) + 1)))
        lf_score = 1.0 - (f_p / (self.max_frequency + 1))

        cross_layer_count = 0
        for i in range(len(path) - 1):
            l1 = self.entity_to_layer.get(path[i], 'unknown')
            l2 = self.entity_to_layer.get(path[i + 1], 'unknown')
            if l1 != l2 and l1 != 'unknown' and l2 != 'unknown':
                cross_layer_count += 1
        clc_score = cross_layer_count / (len(path) - 1)

        ip_score = 0.5

        return 0.5 * lf_score + 0.3 * clc_score + 0.2 * ip_score

    def _get_path_layers(self, path: List[str]) -> List[str]:
        return [self.entity_to_layer.get(n, 'unknown') for n in path]

    def _build_path_info(self, path: List[str]) -> Dict:
        path_details = []
        for i in range(len(path) - 1):
            edge_data = self.graph.get_edge_data(path[i], path[i + 1]) or {}
            path_details.append({
                "from": path[i],
                "to": path[i + 1],
                "relationship": edge_data.get("relationship", "unknown"),
                "layer": edge_data.get("layer", "unknown"),
                "confidence": edge_data.get("confidence", 0.5)
            })

        layers = self._get_path_layers(path)
        unique_layers = list(set(layers) - {'unknown'})

        return {
            "path": path,
            "path_details": path_details,
            "hops": len(path) - 1,
            "layers": layers,
            "unique_layers": unique_layers,
            "strength": self._calculate_path_strength(path),
            "novelty_score": self._calculate_novelty_score(path),
            "is_cross_layer": len(unique_layers) > 1
        }

    def _search_node_pair(self, args) -> List[Dict]:
        start, end = args
        results = []

        try:
            paths = self._find_paths_bfs(start, end, max_length=self.max_hops, max_paths=20)

            for path in paths:
                info = self._build_path_info(path)
                if info['strength'] >= 0.05:
                    results.append(info)

        except Exception:
            pass

        return results

    def mine_large_scale_paths(self) -> List[Dict]:
        logger.info("\n" + "=" * 60)
        logger.info("Start large-scale path mining")
        logger.info(f"Target: {self.target_paths} paths")
        logger.info(f"Parameters: min_hops={self.min_hops}, max_hops={self.max_hops}")
        logger.info(f"Thresholds: strength>={self.strength_threshold}, novelty>={self.novelty_threshold}")
        logger.info("=" * 60 + "\n")

        logger.info("Analyzing node degrees...")
        all_nodes = list(self.graph.nodes())
        node_degrees = [(n, self.graph.degree(n)) for n in tqdm(all_nodes, desc="Calculating degrees", unit="node")]
        node_degrees.sort(key=lambda x: x[1], reverse=True)

        high_degree_nodes = [n for n, _ in node_degrees[:1500]]

        layer_nodes = defaultdict(list)
        for node in high_degree_nodes:
            layer = self.entity_to_layer.get(node, 'unknown')
            if layer != 'unknown':
                layer_nodes[layer].append(node)

        logger.info(f"High-degree node distribution:")
        for layer, nodes in layer_nodes.items():
            logger.info(f"  {layer}: {len(nodes)} nodes")

        logger.info("\nGenerating search tasks...")
        random.seed(42)
        tasks = []

        layers = list(layer_nodes.keys())
        for from_layer in tqdm(layers, desc="Generating cross-layer tasks", unit="layer"):
            for to_layer in layers:
                if from_layer != to_layer:
                    from_sample = random.sample(layer_nodes[from_layer],
                                               min(200, len(layer_nodes[from_layer])))
                    to_sample = random.sample(layer_nodes[to_layer],
                                             min(200, len(layer_nodes[to_layer])))
                    for s in from_sample:
                        for e in to_sample[:50]:
                            tasks.append((s, e))

        top_nodes = [n for n, _ in node_degrees[:500]]
        logger.info("Generating high-degree node tasks...")
        for s in top_nodes[:200]:
            targets = random.sample(top_nodes, min(30, len(top_nodes)))
            for e in targets:
                if s != e:
                    tasks.append((s, e))

        tasks = list(set(tasks))
        random.shuffle(tasks)

        logger.info(f"\nTotal search tasks: {len(tasks):,} node pairs")
        logger.info(f"Using {self.max_threads} threads for parallel search")

        all_paths = []
        save_interval = 50
        last_save_count = 0

        logger.info("\nStarting parallel search (this is the main time-consuming step)...")

        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            futures = {}
            logger.info("Submitting search tasks...")
            for task in tqdm(tasks, desc="Submitting tasks", unit="task"):
                future = executor.submit(self._search_node_pair, task)
                futures[future] = task

            logger.info(f"Submitted {len(futures)} tasks, collecting results...")

            with tqdm(total=len(futures), desc="Search progress", unit="pair") as pbar:
                for future in as_completed(futures):
                    try:
                        results = future.result(timeout=30)
                        if results:
                            with self.lock:
                                all_paths.extend(results)

                                if len(all_paths) - last_save_count >= save_interval:
                                    self._save_checkpoint(all_paths)
                                    last_save_count = len(all_paths)

                    except Exception:
                        pass
                    finally:
                        pbar.update(1)
                        pbar.set_postfix({"Found": len(all_paths)})

                    if pbar.n % 2000 == 0:
                        gc.collect()

        logger.info(f"\nMining complete! Found {len(all_paths)} paths")

        unique_paths = self._deduplicate_paths(all_paths)
        unique_paths.sort(key=lambda x: (x['hops'], x['strength']), reverse=True)

        self.all_paths = unique_paths
        return unique_paths

    def _deduplicate_paths(self, paths: List[Dict]) -> List[Dict]:
        seen = set()
        unique = []
        for p in paths:
            key = tuple(p['path'])
            if key not in seen:
                seen.add(key)
                unique.append(p)
        return unique

    def _save_checkpoint(self, paths: List[Dict]):
        checkpoint_file = self.output_dir / "checkpoint_paths.json"
        try:
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(paths[-1000:], f, ensure_ascii=False)
            logger.info(f"  [Checkpoint] Saved {len(paths)} paths")
        except:
            pass

    def generate_sankey_data(self) -> Dict:
        logger.info("\nGenerating Sankey diagram data...")

        layer_transitions = defaultdict(int)
        layer_order = ['physical', 'social', 'economic', 'cross_layer']

        for path_info in tqdm(self.all_paths, desc="Analyzing layer flow", unit="path"):
            layers = path_info['layers']
            for i in range(len(layers) - 1):
                from_layer = layers[i]
                to_layer = layers[i + 1]
                if from_layer != 'unknown' and to_layer != 'unknown':
                    layer_transitions[(from_layer, to_layer)] += 1

        nodes = []
        node_index = {}

        for i, layer in enumerate(layer_order):
            if layer in self.layer_to_entities:
                node_index[layer] = len(nodes)
                nodes.append({"name": layer, "layer_index": i})

        links = []
        for (from_layer, to_layer), count in layer_transitions.items():
            if from_layer in node_index and to_layer in node_index:
                links.append({
                    "source": node_index[from_layer],
                    "target": node_index[to_layer],
                    "value": count,
                    "source_name": from_layer,
                    "target_name": to_layer
                })

        links.sort(key=lambda x: x['value'], reverse=True)

        sankey_data = {
            "nodes": nodes,
            "links": links,
            "total_transitions": sum(layer_transitions.values()),
            "layer_transition_matrix": {f"{k[0]}->{k[1]}": v for k, v in layer_transitions.items()},
            "timestamp": datetime.now().isoformat()
        }

        output_file = self.output_dir / "sankey_data.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(sankey_data, f, ensure_ascii=False, indent=2)
        logger.info(f"✅ Sankey data saved: {output_file}")

        logger.info(f"  Layer transition statistics:")
        for link in links[:10]:
            logger.info(f"    {link['source_name']} -> {link['target_name']}: {link['value']} times")

        return sankey_data

    def calculate_betweenness_centrality(self) -> Dict:
        logger.info("\nCalculating betweenness centrality...")

        path_graph = nx.DiGraph()
        node_path_count = Counter()

        for path_info in tqdm(self.all_paths, desc="Building path subgraph", unit="path"):
            path = path_info['path']
            for i in range(len(path) - 1):
                path_graph.add_edge(path[i], path[i + 1])
            for node in path[1:-1]:
                node_path_count[node] += 1

        logger.info(f"  Path subgraph: {path_graph.number_of_nodes()} nodes, {path_graph.number_of_edges()} edges")

        logger.info("  Calculating betweenness centrality (may take a few minutes)...")
        try:
            betweenness = nx.betweenness_centrality(path_graph, k=min(500, path_graph.number_of_nodes()))
        except:
            betweenness = {}
            for node in path_graph.nodes():
                betweenness[node] = path_graph.degree(node) / path_graph.number_of_nodes()

        ranked_nodes = []
        for node, bc in sorted(betweenness.items(), key=lambda x: x[1], reverse=True):
            ranked_nodes.append({
                "node": node,
                "betweenness_centrality": bc,
                "layer": self.entity_to_layer.get(node, 'unknown'),
                "path_count": node_path_count.get(node, 0),
                "degree": path_graph.degree(node)
            })

        result = {
            "ranked_nodes": ranked_nodes[:200],
            "total_nodes": len(ranked_nodes),
            "statistics": {
                "mean_bc": np.mean(list(betweenness.values())) if betweenness else 0,
                "max_bc": max(betweenness.values()) if betweenness else 0,
                "top_10_nodes": [n['node'] for n in ranked_nodes[:10]]
            },
            "timestamp": datetime.now().isoformat()
        }

        output_file = self.output_dir / "betweenness_centrality.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f"✅ Betweenness centrality saved: {output_file}")

        logger.info(f"  Top 10 Bridge Nodes:")
        for i, node_info in enumerate(ranked_nodes[:10], 1):
            logger.info(f"    {i}. {node_info['node'][:50]}...")
            logger.info(f"       BC={node_info['betweenness_centrality']:.6f}, Layer={node_info['layer']}")

        return result

    def analyze_motifs(self) -> Dict:
        logger.info("\nAnalyzing network Motifs...")

        motif_3_counter = Counter()
        motif_4_counter = Counter()
        layer_pattern_counter = Counter()

        for path_info in tqdm(self.all_paths, desc="Analyzing Motifs", unit="path"):
            layers = path_info['layers']

            layer_pattern = "->".join([l for l in layers if l != 'unknown'])
            if layer_pattern:
                layer_pattern_counter[layer_pattern] += 1

            for i in range(len(layers) - 2):
                motif = f"{layers[i]}->{layers[i+1]}->{layers[i+2]}"
                motif_3_counter[motif] += 1

            for i in range(len(layers) - 3):
                motif = f"{layers[i]}->{layers[i+1]}->{layers[i+2]}->{layers[i+3]}"
                motif_4_counter[motif] += 1

        feedback_loops = []
        compound_events = []

        for pattern, count in layer_pattern_counter.items():
            parts = pattern.split('->')
            if len(parts) >= 3:
                for i in range(len(parts) - 2):
                    if parts[i] == parts[i + 2] and parts[i] != parts[i + 1]:
                        feedback_loops.append({
                            "pattern": pattern,
                            "loop_type": f"{parts[i]}->{parts[i+1]}->{parts[i+2]}",
                            "count": count
                        })
                        break

            if parts.count('physical') >= 2 and 'social' in parts:
                compound_events.append({
                    "pattern": pattern,
                    "count": count
                })

        result = {
            "motif_3_node": dict(motif_3_counter.most_common(50)),
            "motif_4_node": dict(motif_4_counter.most_common(50)),
            "layer_patterns": dict(layer_pattern_counter.most_common(100)),
            "feedback_loops": sorted(feedback_loops, key=lambda x: x['count'], reverse=True)[:20],
            "compound_events": sorted(compound_events, key=lambda x: x['count'], reverse=True)[:20],
            "statistics": {
                "total_3_motifs": sum(motif_3_counter.values()),
                "unique_3_motifs": len(motif_3_counter),
                "total_4_motifs": sum(motif_4_counter.values()),
                "unique_4_motifs": len(motif_4_counter),
                "total_layer_patterns": len(layer_pattern_counter)
            },
            "timestamp": datetime.now().isoformat()
        }

        output_file = self.output_dir / "motif_analysis.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f"✅ Motif analysis saved: {output_file}")

        logger.info(f"  Top 5 Three-Node Motifs:")
        for motif, count in motif_3_counter.most_common(5):
            logger.info(f"    {motif}: {count} times")

        logger.info(f"  Found {len(feedback_loops)} feedback loop patterns")
        logger.info(f"  Found {len(compound_events)} compound event patterns")

        return result

    def generate_novelty_vs_frequency_data(self) -> Dict:
        logger.info("\nGenerating Novelty vs Frequency data...")

        scatter_data = []

        for path_info in tqdm(self.all_paths, desc="Calculating frequency", unit="path"):
            path = path_info['path']

            entity_freqs = [self.entity_frequency.get(n, 1) for n in path]
            avg_frequency = np.mean(entity_freqs)

            scatter_data.append({
                "path_id": hash(tuple(path)) % 100000,
                "path_summary": f"{path[0][:20]}...{path[-1][:20]}",
                "frequency": float(avg_frequency),
                "novelty_score": path_info['novelty_score'],
                "strength": path_info['strength'],
                "hops": path_info['hops'],
                "is_cross_layer": path_info['is_cross_layer'],
                "layers": path_info['unique_layers']
            })

        grey_rhinos = [
            p for p in scatter_data
            if p['frequency'] < np.percentile([x['frequency'] for x in scatter_data], 30)
            and p['novelty_score'] > np.percentile([x['novelty_score'] for x in scatter_data], 70)
        ]

        result = {
            "scatter_points": scatter_data,
            "grey_rhino_risks": sorted(grey_rhinos, key=lambda x: x['novelty_score'], reverse=True)[:50],
            "statistics": {
                "total_points": len(scatter_data),
                "grey_rhino_count": len(grey_rhinos),
                "frequency_range": [min(p['frequency'] for p in scatter_data),
                                   max(p['frequency'] for p in scatter_data)],
                "novelty_range": [min(p['novelty_score'] for p in scatter_data),
                                 max(p['novelty_score'] for p in scatter_data)]
            },
            "timestamp": datetime.now().isoformat()
        }

        output_file = self.output_dir / "novelty_vs_frequency.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f"✅ Novelty vs Frequency data saved: {output_file}")

        logger.info(f"  Found {len(grey_rhinos)} 'Grey Rhino' risk paths")

        return result

    def save_all_results(self) -> Dict:
        logger.info("\nSaving all results...")

        all_paths_file = self.output_dir / "all_paths.json"
        with open(all_paths_file, 'w', encoding='utf-8') as f:
            json.dump(self.all_paths, f, ensure_ascii=False, indent=2)
        logger.info(f"✅ All paths saved: {all_paths_file}")

        hop_distribution = Counter(p['hops'] for p in self.all_paths)
        layer_distribution = Counter()
        for p in self.all_paths:
            for l in p['unique_layers']:
                layer_distribution[l] += 1

        cross_layer_count = sum(1 for p in self.all_paths if p['is_cross_layer'])

        summary = {
            "mining_parameters": {
                "min_hops": self.min_hops,
                "max_hops": self.max_hops,
                "strength_threshold": self.strength_threshold,
                "novelty_threshold": self.novelty_threshold,
                "target_paths": self.target_paths
            },
            "results": {
                "total_paths": len(self.all_paths),
                "cross_layer_paths": cross_layer_count,
                "single_layer_paths": len(self.all_paths) - cross_layer_count,
                "hop_distribution": dict(hop_distribution),
                "layer_distribution": dict(layer_distribution)
            },
            "strength_statistics": {
                "mean": float(np.mean([p['strength'] for p in self.all_paths])) if self.all_paths else 0,
                "max": float(max(p['strength'] for p in self.all_paths)) if self.all_paths else 0,
                "min": float(min(p['strength'] for p in self.all_paths)) if self.all_paths else 0
            },
            "novelty_statistics": {
                "mean": float(np.mean([p['novelty_score'] for p in self.all_paths])) if self.all_paths else 0,
                "above_threshold": sum(1 for p in self.all_paths if p['novelty_score'] >= self.novelty_threshold)
            },
            "graph_info": {
                "total_nodes": self.graph.number_of_nodes(),
                "total_edges": self.graph.number_of_edges()
            },
            "timestamp": datetime.now().isoformat()
        }

        summary_file = self.output_dir / "mining_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info(f"✅ Mining summary saved: {summary_file}")

        report_file = self.output_dir / "mining_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("Large-scale Path Mining Report\n")
            f.write(f"Generated at: {summary['timestamp']}\n")
            f.write("=" * 80 + "\n\n")

            f.write("[Mining Parameters]\n")
            f.write(f"  Hop Range: {self.min_hops}-{self.max_hops}\n")
            f.write(f"  Strength Threshold: {self.strength_threshold}\n")
            f.write(f"  Novelty Threshold: {self.novelty_threshold}\n\n")

            f.write("[Mining Results]\n")
            f.write(f"  Total Paths: {len(self.all_paths):,}\n")
            f.write(f"  Cross-layer Paths: {cross_layer_count:,}\n")
            f.write(f"  Single-layer Paths: {len(self.all_paths) - cross_layer_count:,}\n\n")

            f.write("[Hop Distribution]\n")
            for hops, count in sorted(hop_distribution.items()):
                f.write(f"  {hops} hops: {count:,} paths\n")
            f.write("\n")

            f.write("[Layer Distribution]\n")
            for layer, count in layer_distribution.most_common():
                f.write(f"  {layer}: {count:,} times\n")
            f.write("\n")

            f.write("[Top 20 High Strength Paths]\n")
            f.write("-" * 80 + "\n")
            top_paths = sorted(self.all_paths, key=lambda x: x['strength'], reverse=True)[:20]
            for i, p in enumerate(top_paths, 1):
                f.write(f"{i}. {' -> '.join(p['path'][:3])}... ({p['hops']} hops)\n")
                f.write(f"   Strength: {p['strength']:.4f}, Novelty: {p['novelty_score']:.4f}\n")
                f.write(f"   Layers: {' -> '.join(p['layers'])}\n\n")

            f.write("=" * 80 + "\n")
            f.write("End of Report\n")
            f.write("=" * 80 + "\n")

        logger.info(f"✅ Readable report saved: {report_file}")

        return summary

    def run_full_analysis(self):
        start_time = time.time()

        logger.info("\n" + "=" * 80)
        logger.info("Large-scale Path Mining and Deep Topological Analysis")
        logger.info("=" * 80)
        logger.info(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Output Directory: {self.output_dir}")
        logger.info("=" * 80 + "\n")

        self.mine_large_scale_paths()

        if not self.all_paths:
            logger.error("No paths mined, exiting.")
            return

        self.generate_sankey_data()

        self.calculate_betweenness_centrality()

        self.analyze_motifs()

        self.generate_novelty_vs_frequency_data()

        summary = self.save_all_results()

        elapsed_time = time.time() - start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)

        logger.info("\n" + "=" * 80)
        logger.info("Analysis Complete!")
        logger.info(f"Total Time: {hours}h {minutes}m {seconds}s")
        logger.info(f"Paths Mined: {len(self.all_paths):,}")
        logger.info(f"Output Directory: {self.output_dir}")
        logger.info("=" * 80)

        logger.info("\nGenerated Files:")
        for f in self.output_dir.glob("*.json"):
            size = f.stat().st_size / 1024
            logger.info(f"  📄 {f.name}({size:.1f} KB)")
        for f in self.output_dir.glob("*.txt"):
            size = f.stat().st_size / 1024
            logger.info(f"  📄 {f.name} ({size:.1f} KB)")


def main():
    parser = argparse.ArgumentParser(
        description="Large Scale Path Mining and Deep Topological Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output files (saved to reports/large_scale/):
  - all_paths.json: All mined paths (core data)
  - sankey_data.json: Sankey diagram data
  - betweenness_centrality.json: Betweenness centrality ranking
  - motif_analysis.json: Motif analysis
  - novelty_vs_frequency.json: Novelty vs Frequency data
  - mining_summary.json: Mining statistics summary
  - mining_report.txt: Readable report

Example:
  python 17_large_scale_mining.py
  python 17_large_scale_mining.py --target-paths 5000 --max-threads 32
        """
    )

    parser.add_argument("--enhanced-json-dir", type=str, default="enhanced_json",
                        help="Enhanced JSON directory (default: enhanced_json)")
    parser.add_argument("--config", type=str, default="config/default.yaml",
                        help="Config file path (default: config/default.yaml)")
    parser.add_argument("--target-paths", type=int, default=5000,
                        help="Target number of paths (default: 5000)")
    parser.add_argument("--min-hops", type=int, default=3,
                        help="Minimum hops (default: 3)")
    parser.add_argument("--max-hops", type=int, default=6,
                        help="Maximum hops (default: 6)")
    parser.add_argument("--strength-threshold", type=float, default=0.1,
                        help="Strength threshold (default: 0.1)")
    parser.add_argument("--novelty-threshold", type=float, default=0.3,
                        help="Novelty threshold (default: 0.3)")
    parser.add_argument("--max-threads", type=int, default=16,
                        help="Maximum threads (default: 16)")

    args = parser.parse_args()

    engine = LargeScaleMiningEngine(config_path=args.config)

    engine.target_paths = args.target_paths
    engine.min_hops = args.min_hops
    engine.max_hops = args.max_hops
    engine.strength_threshold = args.strength_threshold
    engine.novelty_threshold = args.novelty_threshold
    engine.max_threads = args.max_threads

    if not engine.load_graph(args.enhanced_json_dir):
        logger.error("Graph loading failed, exiting.")
        return

    engine.run_full_analysis()

    logger.info("\n✅ Large-scale path mining complete!")
    logger.info(f"📁 Results saved to: reports/large_scale/")


if __name__ == "__main__":
    main()