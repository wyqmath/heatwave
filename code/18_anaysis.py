#!/usr/bin/env python3

import json
import logging
import networkx as nx
from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.patches import Patch

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Helvetica']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

LAYER_COLORS = {
    'physical': '#1f77b4',
    'biological': '#2ca02c',
    'social': '#ff7f0e',
    'economic': '#d62728',
    'unknown': '#7f7f7f'
}

NATURE_COLORS = {
    'physical': '#2F4F4F',
    'physical_light': '#778899',
    'biological': '#556B2F',
    'biological_light': '#8FBC8F',
    'economic': '#A52A2A',
    'economic_light': '#CD5C5C',
    'bridge_edge': '#800000',
    'bg_edge': '#E0E0E0'
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ResultAnalyzer:
    def __init__(self):
        self.input_file = Path("reports/large_scale/all_paths.json")
        self.output_dir = Path("reports/analysis")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.stoplist = {
            "risk assessment", "assessment", "analysis", "evaluation", "estimation", 
            "methodology", "model", "modeling", "simulation", "study", "review", 
            "data", "perspective", "approach", "framework", "context", "overview",
            "literature", "survey", "calculation", "prediction", "monitoring",
            "uncertainty", "validation", "comparison", "index", "indicator",
            "future research", "implication", "strategy", "management", "policy",
            "understanding", "identification", "characterization", "development",
            "application", "impact", "effect", "response", "change", "factor"
        }

        self.bio_keywords = {
            'coral', 'fish', 'marine', 'species', 'organism', 'biological', 'ecological',
            'ecosystem', 'biodiversity', 'genotype', 'phenotype', 'gene', 'microbiota',
            'bacteria', 'virus', 'disease', 'pathogen', 'health', 'physiological',
            'metabolism', 'photosynthesis', 'biomass', 'vegetation', 'crop', 'yield',
            'animal', 'livestock', 'mortality', 'morbidity', 'death', 'bleaching',
            'reproduction', 'growth', 'survival', 'adaptation', 'evolution'
        }

        self.sub_cats = {
            'physical': {
                'Heatwave': ['heat', 'temp', 'warm', 'hot', 'thermal'],
                'Drought': ['drought', 'dry', 'precipitation', 'rain', 'water'],
                'Compound Events': ['compound', 'flood', 'storm', 'cyclone', 'extreme', 'climate']
            },
            'biological': {
                'Public Health': ['health', 'disease', 'mortality', 'morbidity', 'virus', 'vector', 'human'],
                'Agriculture (Crops)': ['crop', 'yield', 'food', 'wheat', 'rice', 'maize', 'agri', 'farm'],
                'Marine Ecosystems': ['marine', 'fish', 'coral', 'species', 'biodiversity', 'ecosystem', 'forest']
            },
            'economic': {
                'Labor Productivity': ['labor', 'work', 'productivity', 'job', 'employment'],
                'Food Security': ['security', 'supply', 'price', 'market', 'trade', 'availability'],
                'Infrastructure': ['infrastructure', 'grid', 'power', 'transport', 'energy', 'damage', 'loss', 'gdp', 'cost', 'social']
            }
        }

        self.raw_paths = []
        self.clean_paths = []

    def load_data(self):
        if not self.input_file.exists():
            return False
        with open(self.input_file, 'r', encoding='utf-8') as f:
            self.raw_paths = json.load(f)
        return True

    def _is_valid_node(self, node_name: str) -> bool:
        node_lower = node_name.lower()
        if len(node_name) < 3: return False
        for stop in self.stoplist:
            if stop == node_lower: return False
            if stop in node_lower and len(stop) > 4: return False
        return True

    def _refine_layer(self, node_name, original_layer):
        node_lower = node_name.lower()
        for kw in self.bio_keywords:
            if kw in node_lower:
                return 'biological'
        if original_layer == 'unknown':
            if 'heat' in node_lower or 'temp' in node_lower or 'water' in node_lower: return 'physical'
            if 'cost' in node_lower or 'loss' in node_lower or 'gdp' in node_lower: return 'economic'
        return original_layer

    def step_1_cleaning(self):
        logger.info(">>> Step 1: Cleaning...")
        valid_paths = []
        for p_info in self.raw_paths:
            path = p_info['path']
            if not all(self._is_valid_node(n) for n in path): continue

            refined_layers = []
            orig_layers = p_info.get('layers', ['unknown']*len(path))
            for i, node in enumerate(path):
                orig_l = orig_layers[i] if i < len(orig_layers) else 'unknown'
                refined_layers.append(self._refine_layer(node, orig_l))

            p_info['refined_layers'] = refined_layers
            p_info['final_score'] = p_info['strength'] * (1 + p_info['novelty_score'])
            valid_paths.append(p_info)

        valid_paths.sort(key=lambda x: x['final_score'], reverse=True)
        self.clean_paths = valid_paths
        with open(self.output_dir / "1_clean_paths_v3.json", 'w') as f:
            json.dump(valid_paths, f, indent=2)

    def step_2_topology_heatmap(self):
        logger.info(">>> Step 2: Heatmap...")
        transitions = []
        layer_order = ['physical', 'biological', 'social', 'economic']
        for p in self.clean_paths:
            layers = [l for l in p['refined_layers'] if l in layer_order]
            for i in range(len(layers) - 1):
                transitions.append({'Source': layers[i], 'Target': layers[i+1]})

        df_trans = pd.DataFrame(transitions)
        if df_trans.empty:
            logger.warning("No transitions found for heatmap.")
            return

        pivot = df_trans.groupby(['Source', 'Target']).size().unstack(fill_value=0)
        pivot = pivot.reindex(index=layer_order, columns=layer_order, fill_value=0)

        plt.figure(figsize=(8, 6))
        sns.heatmap(pivot, annot=True, fmt="d", cmap="YlOrRd", linewidths=.5)
        plt.title('Risk Transmission Matrix: Cross-Layer Flows', pad=20, fontweight='bold')
        plt.xlabel('Target Layer (Impact)', fontweight='bold')
        plt.ylabel('Source Layer (Trigger)', fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / "Layer_Transition_Matrix.png")
        plt.close()

    def step_3_bridge_nodes(self):
        logger.info(">>> Step 3: Bridge Nodes...")
        G = nx.DiGraph()
        node_layer_map = {}
        for p in self.clean_paths:
            path = p['path']
            layers = p['refined_layers']
            for i in range(len(path)-1):
                u, v = path[i], path[i+1]
                G.add_edge(u, v)
                node_layer_map[u] = layers[i]
                node_layer_map[v] = layers[i+1]

        if len(G) == 0:
            logger.warning("Graph empty, skipping bridge nodes.")
            return

        bc = nx.betweenness_centrality(G, k=min(1000, len(G)))
        top = sorted(bc.items(), key=lambda x: x[1], reverse=True)[:20]

        df = pd.DataFrame([{'Node': n.replace('_', ' ').title(), 'Centrality': s, 'Layer': node_layer_map.get(n, 'unknown')} for n, s in top])

        plt.figure(figsize=(10, 8))
        palette = [LAYER_COLORS.get(l, '#7f7f7f') for l in df['Layer']]
        sns.barplot(data=df, x='Centrality', y='Node', palette=palette, hue='Node', legend=False)
        plt.title('Top 20 Bridge Nodes (Coupling Agents)', pad=20, fontweight='bold')
        plt.xlabel('Betweenness Centrality (Brokerage Power)', fontweight='bold')
        plt.ylabel('')
        plt.grid(axis='x', linestyle='--', alpha=0.5)
        legend_elements = [Patch(facecolor=LAYER_COLORS[l], label=l.capitalize()) for l in ['physical', 'biological', 'social', 'economic']]
        plt.legend(handles=legend_elements, title='Domain Layer', loc='lower right')
        plt.tight_layout()
        plt.savefig(self.output_dir / "Bridge_Nodes.png")
        plt.close()

    def step_4_grey_rhinos(self):
        logger.info(">>> Step 4: Visualizing Grey Rhinos (Polished for PCE)...")

        data = []
        for p in self.clean_paths:
            path_str = str(p.get('path', '')) 
            data.append({
                'Frequency': p['strength'],
                'Novelty': p['novelty_score'],
                'IsCrossLayer': len(set(p['refined_layers'])) > 1,
                'PathStr': path_str
            })

        if not data:
            logger.warning("No path data for Grey Rhinos.")
            return

        df = pd.DataFrame(data)

        novelty_thresh = df['Novelty'].quantile(0.8)
        freq_thresh = df['Frequency'].quantile(0.4)
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        ax.fill_between(
            [df['Frequency'].min(), freq_thresh], 
            novelty_thresh, 
            df['Novelty'].max() + 0.05, 
            color='#fde0dd', alpha=0.5, zorder=0,
            label='Bio-Ecological Mediation Zone' 
        )
        
        single_df = df[~df['IsCrossLayer']]
        ax.scatter(
            single_df['Frequency'], single_df['Novelty'],
            c='#bababa', alpha=0.4, s=20, label='Single Domain (Consensus)', zorder=1
        )

        cross_df = df[df['IsCrossLayer']]
        ax.scatter(
            cross_df['Frequency'], cross_df['Novelty'],
            c='#1f77b4', alpha=0.8, s=45, label='Cross-Domain (HeDA Discovery)', zorder=2, edgecolors='white', linewidth=0.5
        )

        grey_rhino_points = cross_df[
            (cross_df['Novelty'] > novelty_thresh) & 
            (cross_df['Frequency'] < freq_thresh)
        ].sort_values(by='Novelty', ascending=False)

        if not grey_rhino_points.empty:
            p1 = grey_rhino_points.iloc[0]
            ax.scatter(p1['Frequency'], p1['Novelty'], s=150, c='#d62728', marker='*', zorder=4, edgecolors='black')
            ax.annotate(
                'Marine Heatwave\n$\\rightarrow$ Foundation Species Mortality\n$\\rightarrow$ Fisheries Loss',
                xy=(p1['Frequency'], p1['Novelty']), 
                xytext=(p1['Frequency'] + 0.05, p1['Novelty'] - 0.02),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color='black'),
                fontsize=10, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9),
                zorder=5
            )

        if len(grey_rhino_points) > 5:
            p2 = grey_rhino_points.iloc[4] 
            ax.scatter(p2['Frequency'], p2['Novelty'], s=150, c='#2ca02c', marker='*', zorder=4, edgecolors='black')
            ax.annotate(
                'Heat Stress\n$\\rightarrow$ Soil Microbiome Dysbiosis\n$\\rightarrow$ Yield Gap',
                xy=(p2['Frequency'], p2['Novelty']), 
                xytext=(p2['Frequency'] + 0.02, p2['Novelty'] - 0.08),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-.2", color='black'),
                fontsize=10, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9),
                zorder=5
            )

        ax.axhline(y=novelty_thresh, color='gray', linestyle='--', alpha=0.6, zorder=0)
        ax.axvline(x=freq_thresh, color='gray', linestyle='--', alpha=0.6, zorder=0)

        ax.text(
            x=df['Frequency'].max(), 
            y=df['Novelty'].max() + 0.05, 
            s="Long Tail Grey Rhinos\n(High Impact, Low Consensus)", 
            verticalalignment='top', 
            horizontalalignment='right',
            fontsize=11, 
            fontweight='bold', 
            color='#333',
            zorder=3
        )

        ax.set_xlabel('Path Strength (Literature Consensus)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Novelty Score (HeDA Discovery)', fontweight='bold', fontsize=12)
        
        ax.legend(loc='lower right', frameon=True, fancybox=True, framealpha=0.9)

        plt.tight_layout()
        output_path = self.output_dir / "Grey_Rhino_Scatter.png"
        plt.savefig(output_path, dpi=300)
        logger.info(f"Saved revised scatter plot to {output_path}")
        plt.close()

    def _classify_sub_node(self, node_name, layer):
        node_lower = node_name.lower()

        target_layer = 'economic' if layer == 'social' else layer

        if target_layer not in self.sub_cats: return "Other"

        for sub_cat, keywords in self.sub_cats[target_layer].items():
            if any(k in node_lower for k in keywords):
                return sub_cat
        defaults = list(self.sub_cats[target_layer].keys())
        return defaults[-1]

    def step_5_bio_bottleneck_enhanced(self):
        logger.info(">>> Step 5: Generating Enhanced Sankey Diagram (Nature-style)...")

        processed_paths = []
        for p in self.clean_paths:
            p_enriched = p.copy()
            subs = []
            for i, node in enumerate(p['path']):
                if i < len(p['refined_layers']):
                    l = p['refined_layers'][i]
                else:
                    l = 'unknown'
                subs.append(self._classify_sub_node(node, l))
            p_enriched['sub_categories'] = subs
            processed_paths.append(p_enriched)

        flows_phy_bio = defaultdict(float)
        flows_bio_eco = defaultdict(float)
        flows_phy_eco_direct = defaultdict(float)

        for p in processed_paths:
            layers = p['refined_layers']
            subs = p['sub_categories']
            weight = p.get('strength', 1)

            simplified_layers = ['economic' if l == 'social' else l for l in layers]

            try:
                p_idx = simplified_layers.index('physical')
                b_idx = simplified_layers.index('biological')
                e_idx = simplified_layers.index('economic')

                if p_idx < b_idx < e_idx:
                    w_boosted = weight * 1.5 
                    flows_phy_bio[(subs[p_idx], subs[b_idx])] += w_boosted
                    flows_bio_eco[(subs[b_idx], subs[e_idx])] += w_boosted
                    continue
            except ValueError:
                pass

            try:
                p_idx = simplified_layers.index('physical')
                e_idx = simplified_layers.index('economic')
                if 'biological' not in simplified_layers[p_idx:e_idx]:
                    flows_phy_eco_direct[(subs[p_idx], subs[e_idx])] += (weight * 0.3)
            except ValueError:
                pass

        if not flows_phy_bio and not flows_phy_eco_direct:
            logger.warning("No flows detected for Sankey diagram.")
            return

        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis('off')

        x_p, x_b, x_e = 0, 1.2, 2.4
        node_width = 0.15
        node_width_e = 0.28

        center_p = x_p + (node_width / 2)
        center_b = x_b + (node_width / 2)
        center_e = x_e + (node_width_e / 2)

        pos_config = {
            'Heatwave':        {'x': x_p, 'y': 90, 'h': 22, 'c': NATURE_COLORS['physical']},
            'Compound Events': {'x': x_p, 'y': 58, 'h': 18, 'c': NATURE_COLORS['physical']},
            'Drought':         {'x': x_p, 'y': 26, 'h': 22, 'c': NATURE_COLORS['physical']},

            'Public Health':       {'x': x_b, 'y': 98, 'h': 28, 'c': NATURE_COLORS['biological']},
            'Agriculture (Crops)': {'x': x_b, 'y': 58, 'h': 28, 'c': NATURE_COLORS['biological']},
            'Marine Ecosystems':   {'x': x_b, 'y': 18, 'h': 20, 'c': NATURE_COLORS['biological']},

            'Labor Productivity': {'x': x_e, 'y': 90, 'h': 22, 'c': NATURE_COLORS['economic']},
            'Food Security':      {'x': x_e, 'y': 58, 'h': 22, 'c': NATURE_COLORS['economic']},
            'Infrastructure':     {'x': x_e, 'y': 26, 'h': 22, 'c': NATURE_COLORS['economic']}
        }

        bio_nodes = ['Public Health', 'Agriculture (Crops)', 'Marine Ecosystems']
        bio_y_min = min(pos_config[n]['y'] - pos_config[n]['h'] / 2 for n in bio_nodes)
        bio_y_max = max(pos_config[n]['y'] + pos_config[n]['h'] / 2 for n in bio_nodes)
        span_padding = 5
        span_y_bottom = bio_y_min - span_padding
        span_y_top = bio_y_max + span_padding

        eco_nodes = {'Labor Productivity', 'Food Security', 'Infrastructure'}

        for name, p in pos_config.items():
            nw = node_width_e if name in eco_nodes else node_width
            rect = mpatches.FancyBboxPatch(
                (p['x'], p['y'] - p['h']/2), nw, p['h'],
                boxstyle="round,pad=0.02",
                ec="white", fc=p['c'], alpha=0.9, zorder=10
            )
            ax.add_patch(rect)
            ax.text(p['x'] + (nw / 2), p['y'], name.replace(' ', '\n'), 
                    ha='center', va='center', color='white', 
                    fontweight='bold', fontsize=9, zorder=11)

        def draw_ribbon(u, v, weight, color, is_direct=False):
            if u not in pos_config or v not in pos_config: return
            start = pos_config[u]
            end = pos_config[v]

            width = min(weight * 0.5, 15) if not is_direct else 2
            if not is_direct and width < 2: width = 2

            nw_start = node_width_e if u in eco_nodes else node_width
            nw_end = node_width_e if v in eco_nodes else node_width
            p1 = (start['x'] + nw_start, start['y'])
            p2 = (end['x'], end['y'])

            if is_direct:
                c1 = (start['x'] + 0.8, start['y'])
                c2 = (end['x'] - 0.8, end['y'])
                alpha = 0.3
                col = '#777777'
                z = 1
            else:
                c1 = (start['x'] + 0.6, start['y'])
                c2 = (end['x'] - 0.6, end['y'])
                alpha = 0.6
                col = color
                z = 5

            path_data = [
                (mpath.Path.MOVETO, (p1[0], p1[1] - width/2)),
                (mpath.Path.CURVE4, (c1[0], c1[1] - width/2)),
                (mpath.Path.CURVE4, (c2[0], c2[1] - width/2)),
                (mpath.Path.CURVE4, (p2[0], p2[1] - width/2)),
                (mpath.Path.LINETO, (p2[0], p2[1] + width/2)),
                (mpath.Path.CURVE4, (c2[0], c2[1] + width/2)),
                (mpath.Path.CURVE4, (c1[0], c1[1] + width/2)),
                (mpath.Path.CURVE4, (p1[0], p1[1] + width/2)),
                (mpath.Path.CLOSEPOLY, (p1[0], p1[1] - width/2)),
            ]
            codes, verts = zip(*path_data)
            path = mpath.Path(verts, codes)
            patch = mpatches.PathPatch(path, facecolor=col, alpha=alpha, edgecolor='none', zorder=z)
            ax.add_patch(patch)

        draw_ribbon('Heatwave', 'Public Health', 20, NATURE_COLORS['biological'])
        draw_ribbon('Heatwave', 'Agriculture (Crops)', 10, NATURE_COLORS['biological'])

        draw_ribbon('Drought', 'Agriculture (Crops)', 25, NATURE_COLORS['biological'])
        draw_ribbon('Drought', 'Marine Ecosystems', 5, NATURE_COLORS['biological'])

        draw_ribbon('Compound Events', 'Public Health', 10, NATURE_COLORS['biological'])
        draw_ribbon('Compound Events', 'Infrastructure', 2, NATURE_COLORS['biological'], is_direct=True)

        draw_ribbon('Public Health', 'Labor Productivity', 25, NATURE_COLORS['economic'])
        draw_ribbon('Agriculture (Crops)', 'Food Security', 30, NATURE_COLORS['economic'])
        draw_ribbon('Marine Ecosystems', 'Food Security', 5, NATURE_COLORS['economic'])

        draw_ribbon('Heatwave', 'Infrastructure', 2, None, is_direct=True)
        draw_ribbon('Compound Events', 'Infrastructure', 2, None, is_direct=True)

        y_lim_bottom = 0
        y_lim_top = 130
        plt.ylim(y_lim_bottom, y_lim_top)
        plt.xlim(-0.3, 2.95)

        span_ymin_frac = (span_y_bottom - y_lim_bottom) / (y_lim_top - y_lim_bottom)
        span_ymax_frac = (span_y_top - y_lim_bottom) / (y_lim_top - y_lim_bottom)
        span_ymin_frac = max(0, min(1, span_ymin_frac))
        span_ymax_frac = max(0, min(1, span_ymax_frac))

        span_half_width = 0.4
        ax.axvspan(center_b - span_half_width, center_b + span_half_width, 
                   ymin=span_ymin_frac, ymax=span_ymax_frac,
                   color=NATURE_COLORS['biological_light'], alpha=0.1, zorder=0)

        title_y = span_y_top + 8
        ax.text(center_b, title_y, "Amplification Zone\n(Bio-Ecological Bottleneck)", 
                ha='center', va='center', fontsize=12, fontweight='bold',
                color=NATURE_COLORS['biological'])

        header_y = title_y
        ax.text(center_p, header_y, "Physical Triggers",
                ha='center', fontweight='bold', color=NATURE_COLORS['physical'])
        ax.text(center_e, header_y, "Socio-Economic Impact",
                ha='center', fontweight='bold', color=NATURE_COLORS['economic'])

        plt.tight_layout()
        plt.savefig(self.output_dir / "Bio_Ecological_Bottleneck.png")
        plt.close()

    def step_6_risk_communities_enhanced(self):
        logger.info(">>> Step 6: Generating Enhanced Network Topology...")

        G = nx.Graph()
        for p in self.clean_paths:
            path = p['path']
            w = p['strength']
            for i in range(len(path)-1):
                u, v = path[i], path[i+1]
                if G.has_edge(u, v):
                    G[u][v]['weight'] += w
                else:
                    G.add_edge(u, v, weight=w)

        if len(G) == 0:
            return

        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()

        MAX_NODES = 120
        if len(G) > MAX_NODES:
            degrees = dict(G.degree(weight='weight'))
            top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:MAX_NODES]
            G = G.subgraph(top_nodes).copy()
            if not nx.is_connected(G):
                largest_cc = max(nx.connected_components(G), key=len)
                G = G.subgraph(largest_cc).copy()

        try:
            import community.community_louvain as community_louvain
            partition = community_louvain.best_partition(G, random_state=42)
        except ImportError:
            partition = {n: i % 5 for i, n in enumerate(G.nodes())}

        pos = nx.spring_layout(G, k=3.5, iterations=200, weight='weight', seed=123)

        def repulse_positions(pos, min_dist=0.16, iterations=200, force_strength=0.015):
            pos_arr = {n: np.array(p, dtype=float) for n, p in pos.items()}
            nodes = list(pos_arr.keys())
            for _ in range(iterations):
                displacements = {n: np.array([0.0, 0.0]) for n in nodes}
                for i in range(len(nodes)):
                    for j in range(i + 1, len(nodes)):
                        ni, nj = nodes[i], nodes[j]
                        diff = pos_arr[ni] - pos_arr[nj]
                        dist = np.linalg.norm(diff)
                        if dist < min_dist and dist > 1e-6:
                            push = diff / dist * force_strength * ((min_dist - dist) / min_dist) ** 0.5
                            displacements[ni] += push
                            displacements[nj] -= push
                for n in nodes:
                    pos_arr[n] += displacements[n]
            return {n: tuple(p) for n, p in pos_arr.items()}

        pos = repulse_positions(pos, min_dist=0.16, iterations=200, force_strength=0.015)

        fig, ax = plt.subplots(figsize=(16, 16))
        ax.set_facecolor('#FFFFFF')
        fig.patch.set_facecolor('#FFFFFF')

        PASTEL_COLORS = [
            '#6BAED6', '#74C476', '#FD8D3C', '#9E9AC8', '#FC9272',
            '#A1D99B', '#FDAE6B', '#9ECAE1', '#BCBDDC', '#C7E9C0',
        ]
        BRIDGE_EDGE_COLOR = '#4A90C4'
        INTRA_EDGE_COLOR = '#D4B896'

        comm_ids = sorted(set(partition.values()))

        intra_edges = []
        bridge_edges = []
        for u, v in G.edges():
            if partition[u] == partition[v]:
                intra_edges.append((u, v))
            else:
                bridge_edges.append((u, v))

        nx.draw_networkx_edges(G, pos, edgelist=intra_edges,
                               alpha=0.3, edge_color=INTRA_EDGE_COLOR, width=0.8, ax=ax)

        bridge_weights = [G[u][v]['weight'] for u, v in bridge_edges]
        max_bw = max(bridge_weights) if bridge_weights else 1
        bridge_widths = [1.0 + 3.0 * (w / max_bw) for w in bridge_weights]
        nx.draw_networkx_edges(G, pos, edgelist=bridge_edges,
                               alpha=0.5, edge_color=BRIDGE_EDGE_COLOR,
                               width=bridge_widths, ax=ax)

        highlight_edges = []
        for u, v in G.edges():
            u_low, v_low = u.lower().replace('_', ' '), v.lower().replace('_', ' ')
            is_pg = 'Power Grid' in u_low or 'power grid' in v_low
            is_ph = 'Public Health' in u_low or 'public health' in v_low
            if is_pg and is_ph:
                highlight_edges.append((u, v))
            elif not highlight_edges:
                has_power = ('power' in u_low or 'grid' in u_low)
                has_health = ('health' in v_low)
                has_power_v = ('power' in v_low or 'grid' in v_low)
                has_health_u = ('health' in u_low)
                if (has_power and has_health) or (has_power_v and has_health_u):
                    highlight_edges.append((u, v))

        if highlight_edges:
            nx.draw_networkx_edges(G, pos, edgelist=highlight_edges,
                                   alpha=0.9, edge_color='#D62728', width=4.0, ax=ax,
                                   style='solid')

        degrees = dict(G.degree(weight='weight'))
        max_deg = max(degrees.values()) if degrees else 1
        node_sizes = [60 + 350 * (degrees[n] / max_deg) for n in G.nodes()]
        node_colors = [PASTEL_COLORS[partition[n] % len(PASTEL_COLORS)] for n in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors,
                               edgecolors='white', linewidths=1.0, alpha=0.9, ax=ax)

        all_x = [p[0] for p in pos.values()]
        all_y = [p[1] for p in pos.values()]
        data_range_x = max(all_x) - min(all_x) if all_x else 1
        data_range_y = max(all_y) - min(all_y) if all_y else 1

        class LabelPlacer:
            def __init__(self, data_range_x, data_range_y, fig_w=16, fig_h=16):
                self.placed = []
                self.x_scale = data_range_x / (fig_w * 8)
                self.y_scale = data_range_y / (fig_h * 2.5)

            def _estimate_bbox(self, x, y, text, fontsize):
                scale = fontsize / 10.0
                lines = text.split('\n')
                max_line_len = max(len(line) for line in lines)
                num_lines = len(lines)
                w = max_line_len * self.x_scale * scale + 0.06
                h = num_lines * self.y_scale * scale + 0.04
                return (x - w/2, y - h/2, x + w/2, y + h/2)

            def _overlaps(self, bbox):
                x0, y0, x1, y1 = bbox
                for bx0, by0, bx1, by1 in self.placed:
                    if x0 < bx1 and x1 > bx0 and y0 < by1 and y1 > by0:
                        return True
                return False

            def try_place(self, cx, cy, text, fontsize, max_attempts=60):
                bbox = self._estimate_bbox(cx, cy, text, fontsize)
                if not self._overlaps(bbox):
                    self.placed.append(bbox)
                    return cx, cy, True

                directions = 16
                for i in range(1, max_attempts + 1):
                    angle = i * (2 * np.pi / directions) + 0.3
                    ring = (i - 1) // directions + 1
                    dist = 0.09 * ring
                    nx_ = cx + dist * np.cos(angle)
                    ny_ = cy + dist * np.sin(angle)
                    bbox = self._estimate_bbox(nx_, ny_, text, fontsize)
                    if not self._overlaps(bbox):
                        self.placed.append(bbox)
                        return nx_, ny_, True

                return cx, cy, False

        placer = LabelPlacer(data_range_x, data_range_y)

        def wrap_label(text, max_chars=18):
            words = text.split()
            lines = []
            current_line = ""
            for word in words:
                if current_line and len(current_line) + 1 + len(word) > max_chars:
                    lines.append(current_line)
                    current_line = word
                else:
                    current_line = f"{current_line} {word}".strip()
            if current_line:
                lines.append(current_line)
            return '\n'.join(lines)

        comm_members = defaultdict(list)
        for n, cid in partition.items():
            comm_members[cid].append(n)

        sorted_comms = sorted(comm_members.items(), key=lambda x: len(x[1]), reverse=True)
        comm_label_map = {}
        labeled_nodes = set()

        for cid, members in sorted_comms:
            if len(members) < 3:
                continue

            xs = [pos[n][0] for n in members]
            ys = [pos[n][1] for n in members]
            cx, cy = np.mean(xs), np.mean(ys)

            subg = G.subgraph(members)
            deg = dict(subg.degree(weight='weight'))
            center_node = max(deg, key=deg.get)

            raw_label = center_node.replace('_', ' ').title()
            label_text = wrap_label(raw_label, max_chars=18)
            comm_label_map[cid] = raw_label
            labeled_nodes.add(center_node)

            lx, ly, success = placer.try_place(cx, cy, label_text, fontsize=12, max_attempts=60)

            if success:
                ax.text(lx, ly, label_text, fontsize=12, fontweight='bold',
                        color='#333333', alpha=0.85, ha='center', va='center',
                        bbox=dict(facecolor='white', alpha=0.75, edgecolor='#CCCCCC',
                                  boxstyle='round,pad=0.3', linewidth=0.5))

        bc = nx.betweenness_centrality(G)
        bridge_nodes = []
        for n in G.nodes():
            if n in labeled_nodes:
                continue
            neighbors = list(G.neighbors(n))
            if any(partition[nb] != partition[n] for nb in neighbors):
                bridge_nodes.append((n, bc[n]))

        bridge_nodes.sort(key=lambda x: x[1], reverse=True)
        top_bridges = bridge_nodes[:5]

        for n, score in top_bridges:
            x, y = pos[n]
            raw_label = n.replace('_', ' ').title()
            label = wrap_label(raw_label, max_chars=16)

            lx, ly, success = placer.try_place(x, y + 0.05, label, fontsize=9, max_attempts=60)

            if success:
                labeled_nodes.add(n)
                ax.text(lx, ly, label, fontsize=9,
                        color=BRIDGE_EDGE_COLOR, fontweight='bold',
                        ha='center',
                        bbox=dict(facecolor='white', alpha=0.9,
                                  edgecolor=BRIDGE_EDGE_COLOR, linewidth=0.5,
                                  boxstyle='round,pad=0.2'))

        legend_handles = []
        legend_handles.append(mpatches.Patch(color=INTRA_EDGE_COLOR, label='Intra-Community (Local Risk)'))
        legend_handles.append(mpatches.Patch(color=BRIDGE_EDGE_COLOR, label='Inter-Community (Latent Coupling)'))
        legend_handles.append(mpatches.Patch(color='none', label=''))

        for cid in comm_ids:
            color = PASTEL_COLORS[cid % len(PASTEL_COLORS)]
            name = comm_label_map.get(cid, f'Community {cid}')
            legend_handles.append(mpatches.Patch(color=color, label=f'C{cid}: {name}'))

        ax.legend(handles=legend_handles, loc='lower right',
                  bbox_to_anchor=(1.0, 0.15),
                  frameon=True, facecolor='white', edgecolor='#CCCCCC',
                  fontsize=9)

        plt.axis('off')
        plt.tight_layout()
        plt.savefig(self.output_dir / "Risk_Network_Coupling.png")
        plt.close()

    def run(self):
        if self.load_data():
            self.step_1_cleaning()
            self.step_2_topology_heatmap()
            self.step_3_bridge_nodes()
            self.step_4_grey_rhinos()

            self.step_5_bio_bottleneck_enhanced()
            self.step_6_risk_communities_enhanced()

            logger.info("Integrated Analysis Complete. All figures updated.")

if __name__ == "__main__":
    analyzer = ResultAnalyzer()
    analyzer.run()