#!/usr/bin/env python3

import json
import logging
import yaml
import networkx as nx
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple
from collections import defaultdict, Counter
import numpy as np
import argparse
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import gc
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CrossLayerAnalyzer:
    
    def __init__(self, graph: nx.DiGraph):
        self.graph = graph
        self.layer_mapping = {}
        self.layer_stats = defaultdict(int)
        
    def build_layer_mapping(self):
        logger.info("Building layer mapping...")
        
        for u, v, data in self.graph.edges(data=True):
            layer = data.get('layer', 'unknown')
            
            if u not in self.layer_mapping:
                self.layer_mapping[u] = layer
            if v not in self.layer_mapping:
                self.layer_mapping[v] = layer
                
            self.layer_stats[layer] += 1
        
        logger.info(f"Layer mapping completed: {len(self.layer_mapping)} nodes")
        logger.info(f"Layer distribution: {dict(self.layer_stats)}")
        
    def find_cross_layer_bridges(self, top_k: int = 50) -> List[Dict]:
        logger.info("Finding cross-layer bridge nodes...")
        
        bridges = []

        for node in tqdm(self.graph.nodes(), desc="Analyzing bridge nodes", unit="node"):
            neighbor_layers = set()
            in_neighbors = list(self.graph.predecessors(node))
            out_neighbors = list(self.graph.successors(node))
            
            for neighbor in in_neighbors + out_neighbors:
                layer = self.layer_mapping.get(neighbor, 'unknown')
                if layer != 'unknown':
                    neighbor_layers.add(layer)
            
            if len(neighbor_layers) >= 2:
                node_layer = self.layer_mapping.get(node, 'unknown')
                degree = self.graph.degree(node)
                
                bridges.append({
                    'node': node,
                    'node_layer': node_layer,
                    'connected_layers': sorted(list(neighbor_layers)),
                    'bridge_strength': len(neighbor_layers),
                    'degree': degree,
                    'in_degree': len(in_neighbors),
                    'out_degree': len(out_neighbors)
                })
        
        bridges.sort(key=lambda x: (x['bridge_strength'], x['degree']), reverse=True)
        
        logger.info(f"Found {len(bridges)} cross-layer bridge nodes")
        return bridges[:top_k]
    
    def analyze_layer_interactions(self) -> Dict[str, Any]:
        logger.info("Analyzing layer interactions...")
        
        interaction_matrix = defaultdict(lambda: defaultdict(int))
        
        for u, v, data in self.graph.edges(data=True):
            u_layer = self.layer_mapping.get(u, 'unknown')
            v_layer = self.layer_mapping.get(v, 'unknown')
            
            if u_layer != 'unknown' and v_layer != 'unknown':
                interaction_matrix[u_layer][v_layer] += 1
        
        cross_layer_connections = {}
        total_cross_layer = 0
        
        for source_layer in interaction_matrix:
            for target_layer in interaction_matrix[source_layer]:
                if source_layer != target_layer:
                    key = f"{source_layer} -> {target_layer}"
                    count = interaction_matrix[source_layer][target_layer]
                    cross_layer_connections[key] = count
                    total_cross_layer += count
        
        within_layer_connections = {}
        total_within_layer = 0

        for layer in interaction_matrix:
            if layer in interaction_matrix[layer]:
                count = interaction_matrix[layer][layer]
                within_layer_connections[layer] = count
                total_within_layer += count

        sorted_cross_layer = sorted(
            cross_layer_connections.items(),
            key=lambda x: x[1],
            reverse=True
        )

        result = {
            'interaction_matrix': {k: dict(v) for k, v in interaction_matrix.items()},
            'cross_layer_connections': dict(sorted_cross_layer),
            'within_layer_connections': within_layer_connections,
            'total_cross_layer': total_cross_layer,
            'total_within_layer': total_within_layer,
            'cross_layer_ratio': total_cross_layer / (total_cross_layer + total_within_layer) if (total_cross_layer + total_within_layer) > 0 else 0
        }

        logger.info(f"Cross-layer connections: {total_cross_layer}, Within-layer connections: {total_within_layer}")
        logger.info(f"Cross-layer ratio: {result['cross_layer_ratio']:.2%}")

        return result

    def _find_paths_bfs_sampling(self, source, target, max_length=5, max_paths=10):
        if source == target:
            return []

        from collections import deque
        queue = deque([(source, [source])])
        found_paths = []
        visited_paths = set()
        max_queue_size = 20000

        while queue:
            if len(queue) > max_queue_size:
                break

            if len(found_paths) >= max_paths:
                break

            current, path = queue.popleft()

            if len(path) > max_length:
                continue

            if current == target:
                if len(path) >= 3:
                    found_paths.append(path)
                continue

            if len(path) < max_length:
                for neighbor in self.graph.successors(current):
                    if neighbor not in path:
                        new_path = path + [neighbor]
                        path_key = tuple(new_path)
                        if path_key not in visited_paths:
                            visited_paths.add(path_key)
                            queue.append((neighbor, new_path))

        return found_paths

    def find_risk_propagation_paths(self, max_length: int = 5, sample_size: int = 100, max_workers: int = 16) -> List[Dict]:
        logger.info("Finding risk propagation paths...")

        physical_nodes = [n for n, l in self.layer_mapping.items() if l == 'physical']
        economic_nodes = [n for n, l in self.layer_mapping.items() if l == 'economic']

        logger.info(f"Physical nodes: {len(physical_nodes)}, Economic nodes: {len(economic_nodes)}")

        import random
        if len(physical_nodes) > sample_size:
            physical_nodes = random.sample(physical_nodes, sample_size)
            logger.info(f"Sampled Physical nodes: {len(physical_nodes)}")
        if len(economic_nodes) > sample_size:
            economic_nodes = random.sample(economic_nodes, sample_size)
            logger.info(f"Sampled Economic nodes: {len(economic_nodes)}")

        node_pairs = [(p, e) for p in physical_nodes for e in economic_nodes]
        total_pairs = len(node_pairs)

        logger.info(f"Start searching {total_pairs:,} pairs of nodes (using {max_workers} threads)...")

        risk_paths = []
        lock = threading.Lock()

        def search_single_pair(pair):
            p_node, e_node = pair
            local_paths = []

            try:
                paths = self._find_paths_bfs_sampling(p_node, e_node, max_length=max_length, max_paths=10)

                for path in paths:
                    if len(path) >= 3:
                        path_layers = [self.layer_mapping.get(n, 'unknown') for n in path]
                        unique_layers = set(path_layers)

                        if len(unique_layers) >= 2:
                            risk_score = self._calculate_path_risk_score(path)

                            local_paths.append({
                                'path': path,
                                'path_layers': path_layers,
                                'unique_layers': sorted(list(unique_layers)),
                                'length': len(path),
                                'hops': len(path) - 1,
                                'risk_score': risk_score
                            })
            except Exception:
                pass

            return local_paths

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(search_single_pair, pair): pair for pair in node_pairs}

            with tqdm(total=total_pairs, desc="Searching risk paths", unit="pair") as pbar:
                for future in as_completed(futures):
                    try:
                        local_paths = future.result()
                        if local_paths:
                            with lock:
                                risk_paths.extend(local_paths)
                    except Exception:
                        pass
                    finally:
                        pbar.update(1)

                        if pbar.n % 1000 == 0:
                            gc.collect()

        risk_paths.sort(key=lambda x: x['risk_score'], reverse=True)

        logger.info(f"Found {len(risk_paths)} risk propagation paths")

        gc.collect()

        return risk_paths[:50]

    def _calculate_path_risk_score(self, path: List[str]) -> float:
        if len(path) < 2:
            return 0.0

        total_confidence = 0.0
        edge_count = 0

        for i in range(len(path) - 1):
            edge_data = self.graph.get_edge_data(path[i], path[i + 1])
            if edge_data:
                confidence = edge_data.get('confidence', 0.5)
                total_confidence += confidence
                edge_count += 1

        if edge_count == 0:
            return 0.0

        avg_confidence = total_confidence / edge_count

        length_bonus = 1.0 + (len(path) - 3) * 0.1

        return avg_confidence * length_bonus


class NoveltyAnalyzer:

    def __init__(self, graph: nx.DiGraph, layer_mapping: Dict[str, str] = None):
        self.graph = graph
        self.layer_mapping = layer_mapping or {}
        self.entity_frequency = Counter()
        self.relation_frequency = Counter()
        self.pagerank_scores = {}
        self.severity_scores = {}
        self.max_frequency = 1

        self.alpha = 0.5
        self.beta = 0.3
        self.gamma = 0.2

        self.novelty_threshold = 0.7

    def build_frequency_stats(self):
        logger.info("Building frequency stats...")

        for u, v, data in self.graph.edges(data=True):
            self.entity_frequency[u] += 1
            self.entity_frequency[v] += 1

            relation = data.get('relationship', 'unknown')
            self.relation_frequency[relation] += 1

        if self.entity_frequency:
            self.max_frequency = max(self.entity_frequency.values())

        logger.info(f"Entity frequency stats: {len(self.entity_frequency)} entities")
        logger.info(f"Relation frequency stats: {len(self.relation_frequency)} relations")
        logger.info(f"Max frequency F_max: {self.max_frequency}")

        self._compute_centrality_scores()

        self._compute_severity_scores()

    def _compute_centrality_scores(self):
        logger.info("Computing PageRank centrality scores...")
        try:
            self.pagerank_scores = nx.pagerank(self.graph, alpha=0.85)
            logger.info(f"PageRank computation completed: {len(self.pagerank_scores)} nodes")
        except Exception as e:
            logger.warning(f"PageRank computation failed: {e}, using degree centrality instead")
            total_nodes = self.graph.number_of_nodes()
            if total_nodes > 0:
                for node in self.graph.nodes():
                    self.pagerank_scores[node] = self.graph.degree(node) / total_nodes
            else:
                self.pagerank_scores = {}

    def _compute_severity_scores(self):
        logger.info("Computing severity scores...")

        layer_severity = {
            'economic': 1.0,
            'social': 0.8,
            'physical': 0.6,
            'cross_layer': 0.9,
            'unknown': 0.5
        }

        out_degrees = dict(self.graph.out_degree())
        max_out_degree = max(out_degrees.values()) if out_degrees else 1

        nodes = list(self.graph.nodes())
        for node in tqdm(nodes, desc="Computing severity scores", unit="node"):
            node_layer = self.layer_mapping.get(node, 'unknown')
            layer_weight = layer_severity.get(node_layer, 0.5)

            out_degree = out_degrees.get(node, 0)
            degree_factor = out_degree / max_out_degree if max_out_degree > 0 else 0

            edge_confidences = [data.get('confidence', 0.5) for _, _, data in self.graph.out_edges(node, data=True)]
            avg_confidence = np.mean(edge_confidences) if edge_confidences else 0.5

            self.severity_scores[node] = (layer_weight * 0.4 + degree_factor * 0.3 + avg_confidence * 0.3)

        logger.info(f"Severity score calculation completed: {len(self.severity_scores)} nodes")

    def calculate_literature_frequency(self, path: List[str]) -> float:
        if len(path) < 2:
            return 0.0

        entity_freqs = [self.entity_frequency.get(node, 1) for node in path]

        relation_freqs = []
        for i in range(len(path) - 1):
            edge_data = self.graph.get_edge_data(path[i], path[i + 1])
            if edge_data:
                relation = edge_data.get('relationship', 'unknown')
                relation_freqs.append(self.relation_frequency.get(relation, 1))

        all_freqs = entity_freqs + relation_freqs
        if not all_freqs:
            return 1.0

        f_p = np.exp(np.mean(np.log(np.array(all_freqs) + 1)))

        lf_score = 1.0 - (f_p / (self.max_frequency + 1))

        return max(0.0, min(1.0, lf_score))

    def calculate_cross_layer_connectivity(self, path: List[str]) -> float:
        if len(path) < 2:
            return 0.0

        n = len(path)
        cross_layer_count = 0

        for i in range(n - 1):
            layer_i = self.layer_mapping.get(path[i], 'unknown')
            layer_i_plus_1 = self.layer_mapping.get(path[i + 1], 'unknown')

            if layer_i != layer_i_plus_1 and layer_i != 'unknown' and layer_i_plus_1 != 'unknown':
                cross_layer_count += 1

        clc_score = cross_layer_count / (n - 1)

        return clc_score

    def calculate_impact_potential(self, path: List[str]) -> float:
        if len(path) < 1:
            return 0.0

        n = len(path)
        impact_sum = 0.0

        for node in path:
            centrality = self.pagerank_scores.get(node, 0.0)
            severity = self.severity_scores.get(node, 0.5)

            impact_sum += centrality * severity

        ip_score = impact_sum / n

        max_possible = max(self.pagerank_scores.values()) if self.pagerank_scores else 1.0
        ip_score = ip_score / max_possible if max_possible > 0 else 0.0

        return min(1.0, ip_score)

    def calculate_novelty_score(self, path: List[str]) -> float:
        if len(path) < 2:
            return 0.0

        lf_score = self.calculate_literature_frequency(path)
        clc_score = self.calculate_cross_layer_connectivity(path)
        ip_score = self.calculate_impact_potential(path)

        novelty_score = (
            self.alpha * lf_score +
            self.beta * clc_score +
            self.gamma * ip_score
        )

        return novelty_score

    def calculate_novelty_score_detailed(self, path: List[str]) -> Dict[str, float]:
        if len(path) < 2:
            return {
                'novelty_score': 0.0,
                'lf_score': 0.0,
                'clc_score': 0.0,
                'ip_score': 0.0,
                'alpha': self.alpha,
                'beta': self.beta,
                'gamma': self.gamma
            }

        lf_score = self.calculate_literature_frequency(path)
        clc_score = self.calculate_cross_layer_connectivity(path)
        ip_score = self.calculate_impact_potential(path)

        novelty_score = (
            self.alpha * lf_score +
            self.beta * clc_score +
            self.gamma * ip_score
        )

        return {
            'novelty_score': novelty_score,
            'lf_score': lf_score,
            'clc_score': clc_score,
            'ip_score': ip_score,
            'alpha': self.alpha,
            'beta': self.beta,
            'gamma': self.gamma
        }

    def find_novel_patterns(self, paths: List[Dict], top_k: int = 20) -> List[Dict]:
        logger.info("Analyzing path novelty (using paper formula)...")
        logger.info(f"Weight parameters: alpha={self.alpha}, beta={self.beta}, gamma={self.gamma}")
        logger.info(f"Novelty threshold: theta_novelty={self.novelty_threshold}")

        novel_paths = []

        for path_info in tqdm(paths, desc="Calculating novelty scores", unit="path"):
            path = path_info.get('path', [])

            scores = self.calculate_novelty_score_detailed(path)

            path_info['novelty_score'] = scores['novelty_score']
            path_info['lf_score'] = scores['lf_score']
            path_info['clc_score'] = scores['clc_score']
            path_info['ip_score'] = scores['ip_score']

            novel_paths.append(path_info)

        novel_paths.sort(key=lambda x: x['novelty_score'], reverse=True)

        above_threshold = sum(1 for p in novel_paths if p['novelty_score'] > self.novelty_threshold)
        logger.info(f"Novelty analysis completed:")
        logger.info(f"  - Total paths: {len(novel_paths)}")
        logger.info(f"  - Above threshold (>{self.novelty_threshold}): {above_threshold}")
        logger.info(f"  - Returning top {top_k} paths")

        return novel_paths[:top_k]


class AdvancedReasoningEngine:

    def __init__(self, config_path: str = "config/default.yaml"):
        self.config = self._load_config(config_path)
        self.graph = nx.DiGraph()
        self.cross_layer_analyzer = None
        self.novelty_analyzer = None

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        try:
            config_file = Path(config_path)
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                logger.info(f"Successfully loaded configuration: {config_path}")
                return config
            else:
                logger.warning(f"Config file not found: {config_path}, using default configuration")
                return self._get_default_config()
        except Exception as e:
            logger.warning(f"Failed to load config: {e}, using default configuration")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        return {
            "llm": {
                "api_key": os.environ.get('HEDA_API_KEY', 'YOUR_API_KEY_HERE'),
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "model": "qwen3-max"
            }
        }

    def load_enhanced_graph(self, enhanced_json_dir: str = "enhanced_json") -> bool:
        logger.info(f"Loading enhanced graph from {enhanced_json_dir}...")

        json_dir = Path(enhanced_json_dir)
        if not json_dir.exists():
            logger.error(f"Directory not found: {enhanced_json_dir}")
            return False

        json_files = list(json_dir.glob("*.json"))
        logger.info(f"Found {len(json_files)} JSON files")

        triplet_count = 0

        for json_file in json_files:
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

                    if not all([start_node, end_node, relationship]):
                        continue

                    self.graph.add_edge(
                        start_node,
                        end_node,
                        relationship=relationship,
                        layer=triplet.get('layer', 'unknown'),
                        relation_type=triplet.get('relation_type', 'unknown'),
                        confidence=triplet.get('confidence', 0.5)
                    )

                    triplet_count += 1

            except Exception as e:
                logger.warning(f"Processing file {json_file.name} failed: {e}")
                continue

        logger.info(f"Graph loading completed:")
        logger.info(f"  Nodes: {self.graph.number_of_nodes():,}")
        logger.info(f"  Edges: {self.graph.number_of_edges():,}")
        logger.info(f"  Total triplets: {triplet_count:,}")

        self.cross_layer_analyzer = CrossLayerAnalyzer(self.graph)
        self.cross_layer_analyzer.build_layer_mapping()

        self.novelty_analyzer = NoveltyAnalyzer(
            self.graph,
            layer_mapping=self.cross_layer_analyzer.layer_mapping
        )
        self.novelty_analyzer.build_frequency_stats()

        return True

    def run_full_analysis(self, output_dir: str = "reports") -> Dict[str, Any]:
        logger.info("=" * 60)
        logger.info("Start Advanced Reasoning Analysis")
        logger.info("=" * 60)

        if self.cross_layer_analyzer is None or self.novelty_analyzer is None:
            logger.error("Analyzers not initialized, please load the graph first")
            return {}

        results = {}

        logger.info("\n[1/4] Cross-layer bridge analysis...")
        bridges = self.cross_layer_analyzer.find_cross_layer_bridges(top_k=50)
        results['cross_layer_bridges'] = bridges

        logger.info("\n[2/4] Layer interaction analysis...")
        layer_interactions = self.cross_layer_analyzer.analyze_layer_interactions()
        results['layer_interactions'] = layer_interactions

        logger.info("\n[3/4] Risk propagation path discovery...")
        risk_paths = self.cross_layer_analyzer.find_risk_propagation_paths(
            max_length=5,
            sample_size=150,
            max_workers=16
        )
        results['risk_propagation_paths'] = risk_paths

        logger.info("\n[4/4] Novelty analysis...")
        novel_paths = self.novelty_analyzer.find_novel_patterns(risk_paths, top_k=20)
        results['novel_patterns'] = novel_paths

        self._save_results(results, output_dir)

        self._print_summary(results)

        return results

    def _save_results(self, results: Dict[str, Any], output_dir: str):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        analysis_file = output_path / "advanced_analysis.json"
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"✅ Full analysis results saved: {analysis_file}")

        bridges_file = output_path / "cross_layer_bridges.json"
        with open(bridges_file, 'w', encoding='utf-8') as f:
            json.dump(results.get('cross_layer_bridges', []), f, indent=2, ensure_ascii=False)
        logger.info(f"✅ Cross-layer bridges saved: {bridges_file}")

        interactions_file = output_path / "layer_interactions.json"
        with open(interactions_file, 'w', encoding='utf-8') as f:
            json.dump(results.get('layer_interactions', {}), f, indent=2, ensure_ascii=False)
        logger.info(f"✅ Layer interaction matrix saved: {interactions_file}")

        summary_file = output_path / "advanced_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(self._generate_text_summary(results))
        logger.info(f"✅ Readable report saved: {summary_file}")

    def _print_summary(self, results: Dict[str, Any]):
        logger.info("\n" + "=" * 60)
        logger.info("Advanced Reasoning Analysis Summary")
        logger.info("=" * 60)

        bridges = results.get('cross_layer_bridges', [])
        logger.info(f"\n📊 Cross-layer bridge nodes: {len(bridges)}")
        if bridges:
            logger.info("  Top 5 Bridge Nodes:")
            for i, bridge in enumerate(bridges[:5], 1):
                logger.info(f"    {i}. {bridge['node']}")
                logger.info(f"       - Connected layers: {', '.join(bridge['connected_layers'])}")
                logger.info(f"       - Bridge strength: {bridge['bridge_strength']}, Degree: {bridge['degree']}")

        interactions = results.get('layer_interactions', {})
        logger.info(f"\n🔗 Layer Interaction Statistics:")
        logger.info(f"  Cross-layer connections: {interactions.get('total_cross_layer', 0):,}")
        logger.info(f"  Within-layer connections: {interactions.get('total_within_layer', 0):,}")
        logger.info(f"  Cross-layer ratio: {interactions.get('cross_layer_ratio', 0):.2%}")

        cross_layer_conns = interactions.get('cross_layer_connections', {})
        if cross_layer_conns:
            logger.info("  Top 5 Cross-layer connections:")
            for i, (conn, count) in enumerate(list(cross_layer_conns.items())[:5], 1):
                logger.info(f"    {i}. {conn}: {count:,} edges")

        risk_paths = results.get('risk_propagation_paths', [])
        logger.info(f"\n⚠️  Risk propagation paths: {len(risk_paths)}")
        if risk_paths:
            logger.info("  Top 3 Risk paths:")
            for i, path_info in enumerate(risk_paths[:3], 1):
                path = path_info['path']
                logger.info(f"    {i}. {path[0]} -> ... -> {path[-1]}")
                logger.info(f"       - Hops: {path_info['hops']}, Risk score: {path_info['risk_score']:.4f}")
                logger.info(f"       - Layers spanned: {', '.join(path_info['unique_layers'])}")

        novel_patterns = results.get('novel_patterns', [])
        logger.info(f"\n✨ Novel Patterns: {len(novel_patterns)}")
        logger.info(f"  Formula: NoveltyScore(P) = alpha*LF(P) + beta*CLC(P) + gamma*IP(P)")
        logger.info(f"  Weights: alpha=0.5, beta=0.3, gamma=0.2")
        if novel_patterns:
            logger.info("  Top 3 Novel paths:")
            for i, pattern in enumerate(novel_patterns[:3], 1):
                path = pattern['path']
                logger.info(f"    {i}. {path[0]} -> ... -> {path[-1]}")
                logger.info(f"       - Novelty score: {pattern['novelty_score']:.4f}")
                logger.info(f"       - LF={pattern.get('lf_score', 0):.4f}, CLC={pattern.get('clc_score', 0):.4f}, IP={pattern.get('ip_score', 0):.4f}")
                logger.info(f"       - Risk score: {pattern['risk_score']:.4f}")

        logger.info("\n" + "=" * 60)

    def _generate_text_summary(self, results: Dict[str, Any]) -> str:
        lines = []
        lines.append("=" * 80)
        lines.append("Advanced Reasoning and Cross-Layer Analysis Report")
        lines.append(f"Generation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 80)
        lines.append("")

        bridges = results.get('cross_layer_bridges', [])
        lines.append(f"1. Cross-layer Bridge Node Analysis (Total {len(bridges)})")
        lines.append("-" * 80)
        lines.append("")
        lines.append("Top 10 Bridge Nodes:")
        for i, bridge in enumerate(bridges[:10], 1):
            lines.append(f"{i:2d}. {bridge['node']}")
            lines.append(f"    Node Layer: {bridge['node_layer']}")
            lines.append(f"    Connected Layers: {', '.join(bridge['connected_layers'])}")
            lines.append(f"    Bridge Strength: {bridge['bridge_strength']}, Total Degree: {bridge['degree']}(In: {bridge['in_degree']}, Out: {bridge['out_degree']})")
            lines.append("")

        interactions = results.get('layer_interactions', {})
        lines.append("2. Layer Interaction Analysis")
        lines.append("-" * 80)
        lines.append("")
        lines.append(f"Total Cross-layer Connections: {interactions.get('total_cross_layer', 0):,}")
        lines.append(f"Total Within-layer Connections: {interactions.get('total_within_layer', 0):,}")
        lines.append(f"Cross-layer Connection Ratio: {interactions.get('cross_layer_ratio', 0):.2%}")
        lines.append("")
        lines.append("Cross-layer Connection Details:")
        cross_layer_conns = interactions.get('cross_layer_connections', {})
        for conn, count in cross_layer_conns.items():
            lines.append(f"  {conn}: {count:,} edges")
        lines.append("")

        risk_paths = results.get('risk_propagation_paths', [])
        lines.append(f"3. Risk Propagation Paths (Total {len(risk_paths)})")
        lines.append("-" * 80)
        lines.append("")
        lines.append("Top 10 Risk Paths:")
        for i, path_info in enumerate(risk_paths[:10], 1):
            path = path_info['path']
            lines.append(f"{i:2d}. Path: {' -> '.join(path)}")
            lines.append(f"    Hops: {path_info['hops']}, Risk Score: {path_info['risk_score']:.4f}")
            lines.append(f"    Layers Spanned: {', '.join(path_info['unique_layers'])}")
            lines.append("")

        novel_patterns = results.get('novel_patterns', [])
        lines.append(f"4. Novel Pattern Discovery (Total {len(novel_patterns)})")
        lines.append("-" * 80)
        lines.append("")
        lines.append("Formula (Section 4.2 - Cross-layer Pathway Discovery Algorithm):")
        lines.append("  NoveltyScore(P) = alpha*LF(P) + beta*CLC(P) + gamma*IP(P)")
        lines.append("")
        lines.append("  Where:")
        lines.append("  - LF(P): Literature Frequency (Information Theoretic Novelty)")
        lines.append("  - CLC(P): Cross-Layer Connectivity (Structural Diversity)")
        lines.append("  - IP(P): Impact Potential (PageRank Centrality * Severity)")
        lines.append("")
        lines.append("  Weight Parameters: alpha=0.5, beta=0.3, gamma=0.2")
        lines.append("  Novelty Threshold: theta_novelty=0.7")
        lines.append("")
        lines.append("Top 10 Novel Paths:")
        for i, pattern in enumerate(novel_patterns[:10], 1):
            path = pattern['path']
            lines.append(f"{i:2d}. Path: {' -> '.join(path)}")
            lines.append(f"    Novelty Score: {pattern['novelty_score']:.4f}")
            lines.append(f"    - LF(P) = {pattern.get('lf_score', 0):.4f} (Literature Frequency)")
            lines.append(f"    - CLC(P) = {pattern.get('clc_score', 0):.4f} (Cross-Layer Connectivity)")
            lines.append(f"    - IP(P) = {pattern.get('ip_score', 0):.4f} (Impact Potential)")
            lines.append(f"    Risk Score: {pattern['risk_score']:.4f}")
            lines.append(f"    Layers Spanned: {', '.join(pattern['unique_layers'])}")
            lines.append("")

        lines.append("=" * 80)
        lines.append("End of Report")
        lines.append("=" * 80)

        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Advanced Reasoning and Cross-layer Analysis")
    parser.add_argument("--enhanced-json-dir", type=str, default="enhanced_json",
                        help="Directory for enhanced JSON files")
    parser.add_argument("--output-dir", type=str, default="reports",
                        help="Output directory")
    parser.add_argument("--config", type=str, default="config/default.yaml",
                        help="Path to configuration file")

    args = parser.parse_args()

    engine = AdvancedReasoningEngine(config_path=args.config)

    if not engine.load_enhanced_graph(args.enhanced_json_dir):
        logger.error("Graph load failed, exiting")
        return

    results = engine.run_full_analysis(output_dir=args.output_dir)

    logger.info("\n✅ Advanced reasoning analysis completed!")
    logger.info(f"📁 Results saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()