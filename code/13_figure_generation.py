#!/usr/bin/env python3
"""
13_figure_generation.py
Generate 5 data-driven figures for JPCE paper using new KG data.

Outputs (workspace/results/figures/):
  1. Layer_Transition_Matrix.png  — 4×4 heatmap of cross-layer risk flows
  2. Bridge_Nodes.png             — Betweenness top-20 bar chart
  3. Grey_Rhino_Scatter.png       — Novelty vs Strength scatter
  4. Bio_Ecological_Bottleneck.png— 4-column Sankey diagram
  5. Risk_Network_Coupling.png    — Community network with layer coloring
"""

import json
import pickle
import re
import logging
from pathlib import Path
from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
    'font.size': 12,
    'axes.unicode_minus': False,
    'figure.dpi': 600,
    'savefig.dpi': 600,
})

LAYER_COLORS = {
    'physical':   '#1f77b4',
    'biological': '#2ca02c',
    'social':     '#ff7f0e',
    'economic':   '#d62728',
    'unknown':    '#7f7f7f',
}

NATURE_COLORS = {
    'physical':         '#2F4F4F',
    'physical_light':   '#778899',
    'biological':       '#556B2F',
    'biological_light': '#8FBC8F',
    'social':           '#B8860B',
    'social_light':     '#DAA520',
    'economic':         '#A52A2A',
    'economic_light':   '#CD5C5C',
}

LAYER_ORDER = ['physical', 'biological', 'social', 'economic']

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths — resolved relative to repo root (script is run from repo root)
DATA_DIR = Path("workspace/data/processed/topology_sensitivity")
OUTPUT_DIR = Path("workspace/results/figures")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_entity(name: str) -> str:
    """CamelCase / snake_case → Title Case with spaces."""
    s = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
    s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', s)
    return s.replace('_', ' ').title()


def compute_strength(confidences, hops):
    """Replicate old strength formula: mean(conf) × (1 + (hops-3)×0.1)."""
    if not confidences:
        return 0.5
    return float(np.mean(confidences)) * (1.0 + (hops - 3) * 0.1)

# ---------------------------------------------------------------------------
# Sub-category classification for Sankey
# ---------------------------------------------------------------------------
SUB_CATS = {
    'physical': {
        'Heatwave':         ['heat', 'temp', 'warm', 'hot', 'thermal'],
        'Drought':          ['drought', 'dry', 'precipitation', 'rain', 'water'],
        'Compound Events':  ['compound', 'flood', 'storm', 'cyclone', 'extreme',
                             'climate', 'wind', 'fire'],
    },
    'biological': {
        'Public Health':    ['health', 'disease', 'mortality', 'morbidity',
                            'virus', 'vector', 'human', 'death'],
        'Agriculture':      ['crop', 'yield', 'food', 'wheat', 'rice', 'maize',
                            'agri', 'farm', 'soil', 'plant', 'vegetat'],
        'Marine Ecosystems':['marine', 'fish', 'coral', 'species', 'biodiversity',
                            'ecosystem', 'forest', 'ocean', 'reef'],
    },
    'social': {
        'Policy':           ['policy', 'govern', 'regulat', 'law', 'institution',
                            'plan', 'manag'],
        'Community':        ['community', 'social', 'public', 'population',
                            'urban', 'city', 'resident', 'people'],
        'Adaptation':       ['adapt', 'resilien', 'mitigat', 'response',
                            'capacity', 'vulnerab', 'aware'],
    },
    'economic': {
        'Labor Productivity': ['labor', 'work', 'productivity', 'job',
                              'employment', 'occupat'],
        'Food Security':      ['security', 'supply', 'price', 'market',
                              'trade', 'availab'],
        'Infrastructure':     ['infrastructure', 'grid', 'power', 'transport',
                              'energy', 'damage', 'loss', 'gdp', 'cost',
                              'econom'],
    },
}


def classify_sub(node_name: str, layer: str) -> str:
    """Map an entity to a sub-category within its layer."""
    low = node_name.lower()
    cats = SUB_CATS.get(layer)
    if not cats:
        return 'Other'
    for sub, kws in cats.items():
        if any(k in low for k in kws):
            return sub
    return list(cats.keys())[-1]  # default to last sub-category


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
class FigureGenerator:
    def __init__(self):
        self.paths = None
        self.topology = None
        self.G = None
        self.entity_layer = None

    # ---- data loading -----------------------------------------------------
    def load_data(self):
        logger.info("Loading data …")

        with open(DATA_DIR / "phase4_paths_scored.json") as f:
            self.paths = json.load(f)
        logger.info(f"  paths: {len(self.paths)}")

        with open(DATA_DIR / "topology_results.json") as f:
            self.topology = json.load(f)
        logger.info(f"  topology: {self.topology['n_nodes']} nodes, "
                     f"{self.topology['n_edges']} edges")

        with open(DATA_DIR / "phase1_graph.pkl", "rb") as f:
            gdata = pickle.load(f)
        self.G = gdata['G']
        self.entity_layer = gdata['entity_layer']
        logger.info(f"  graph: {self.G.number_of_nodes()} nodes, "
                     f"{self.G.number_of_edges()} edges")

        # Pre-compute strength for every path
        for p in self.paths:
            p['strength'] = compute_strength(p['confidences'], p['hops'])

        return True

    # ---- Figure 1: Layer Transition Matrix --------------------------------
    def fig_layer_transition_matrix(self):
        logger.info(">>> Layer_Transition_Matrix.png")

        mat = self.topology['layer_matrix']
        data = np.array([[mat[s][t] for t in LAYER_ORDER] for s in LAYER_ORDER])
        labels = [l.capitalize() for l in LAYER_ORDER]

        df = pd.DataFrame(data, index=labels, columns=labels)

        # ---- stats for figure caption ----
        total_edges = int(data.sum())
        diag_sum = int(np.trace(data))
        cross_sum = total_edges - diag_sum
        logger.info("  [Caption Stats] Layer Transition Matrix")
        logger.info(f"    Total directed edges: {total_edges}")
        logger.info(f"    Intra-layer (diagonal): {diag_sum} ({diag_sum/total_edges*100:.1f}%)")
        logger.info(f"    Cross-layer (off-diagonal): {cross_sum} ({cross_sum/total_edges*100:.1f}%)")
        # Top-3 cells
        flat = []
        for i, s in enumerate(LAYER_ORDER):
            for j, t in enumerate(LAYER_ORDER):
                flat.append((s, t, data[i, j]))
        flat.sort(key=lambda x: x[2], reverse=True)
        logger.info("    Top-5 flows:")
        for s, t, v in flat[:5]:
            logger.info(f"      {s} → {t}: {int(v)}")
        # Asymmetry
        pb = int(mat['physical']['biological'])
        bp = int(mat['biological']['physical'])
        logger.info(f"    Asymmetry: physical→biological={pb} vs biological→physical={bp} (ratio={pb/max(bp,1):.1f}x)")

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df, annot=True, fmt="d", cmap="YlOrRd",
                    linewidths=.5, ax=ax)
        ax.set_title('Risk Transmission Matrix: Cross-Layer Flows',
                     pad=20, fontweight='bold')
        ax.set_xlabel('Target Layer (Impact)', fontweight='bold')
        ax.set_ylabel('Source Layer (Trigger)', fontweight='bold')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "Layer_Transition_Matrix.png")
        plt.close()

    # ---- Figure 2: Bridge Nodes -------------------------------------------
    def fig_bridge_nodes(self):
        logger.info(">>> Bridge_Nodes.png")

        top20 = self.topology['betweenness_top20']
        df = pd.DataFrame([{
            'Node':       format_entity(x['entity']),
            'Centrality': x['bc'],
            'Layer':      x['layer'],
        } for x in top20])

        # ---- stats for figure caption ----
        from collections import Counter
        layer_dist = Counter(x['layer'] for x in top20)
        logger.info("  [Caption Stats] Bridge Nodes (Top-20 Betweenness)")
        logger.info(f"    Layer distribution: {dict(layer_dist)}")
        logger.info(f"    #1: {top20[0]['entity']} (bc={top20[0]['bc']:.6f}, {top20[0]['layer']})")
        logger.info(f"    #2: {top20[1]['entity']} (bc={top20[1]['bc']:.6f}, {top20[1]['layer']})")
        logger.info(f"    #3: {top20[2]['entity']} (bc={top20[2]['bc']:.6f}, {top20[2]['layer']})")
        logger.info(f"    BC range: {top20[-1]['bc']:.6f} – {top20[0]['bc']:.6f}")
        logger.info(f"    Top-1 / Top-20 ratio: {top20[0]['bc']/top20[-1]['bc']:.1f}x")

        fig, ax = plt.subplots(figsize=(13, 9))
        fig.subplots_adjust(left=0.24, right=0.95, bottom=0.10)
        palette = [LAYER_COLORS.get(l, '#7f7f7f') for l in df['Layer']]
        sns.barplot(data=df, x='Centrality', y='Node',
                    palette=palette, hue='Node', legend=False, ax=ax)
        ax.set_title('Top 20 Bridge Nodes (Coupling Agents)',
                     pad=20, fontweight='bold')
        ax.set_xlabel('Betweenness Centrality (Brokerage Power)',
                      fontweight='bold')
        ax.set_ylabel('')
        ax.grid(axis='x', linestyle='--', alpha=0.5)

        # ---- Inset: per-layer Top-5 for non-physical layers ---------------
        per_layer_data = self.topology.get('betweenness_top5_per_layer', {})
        inset_layers = ['biological', 'social', 'economic']
        inset_rows = []
        for layer in inset_layers:
            for x in per_layer_data.get(layer, []):
                inset_rows.append({
                    'Node':       format_entity(x['entity']),
                    'Centrality': x['bc'],
                    'Layer':      x['layer'],
                })

        if inset_rows:
            df_inset = pd.DataFrame(inset_rows)
            # Log per-layer top-5
            for layer in inset_layers:
                layer_df = df_inset[df_inset['Layer'] == layer]
                logger.info(f"    Per-layer Top-5 [{layer}]:")
                for _, row in layer_df.iterrows():
                    logger.info(f"      {row['Node']}: bc={row['Centrality']:.6f}")
            # Inset positioning: right-bottom corner INCLUDING axis labels
            # must align with main chart's right-bottom corner.
            # Main axes box: right=0.95, bottom=0.10
            # Reserve ~0.04 below inset for x-axis labels,
            # ~0.03 right of inset for tick labels.
            inset_w, inset_h = 0.32, 0.24
            inset_x = 0.95 - inset_w - 0.03   # tick labels on right
            inset_y = 0.10 + 0.04              # x-axis labels below
            ax_in = fig.add_axes([inset_x, inset_y, inset_w, inset_h])
            colors_in = [LAYER_COLORS[l] for l in df_inset['Layer']]
            sns.barplot(data=df_inset, x='Centrality', y='Node',
                        palette=colors_in, hue='Node', legend=False,
                        ax=ax_in)
            ax_in.set_title('Per-Layer Top 5 (Bio / Social / Econ)',
                            fontsize=9, fontweight='bold')
            ax_in.set_xlabel('Betweenness Centrality', fontsize=8)
            ax_in.set_ylabel('')
            ax_in.tick_params(labelsize=7)
            ax_in.grid(axis='x', linestyle='--', alpha=0.4)
            for spine in ax_in.spines.values():
                spine.set_edgecolor('#CCCCCC')

            # Domain Layer legend — directly above inset title
            legend_els = [Patch(facecolor=LAYER_COLORS[l], label=l.capitalize())
                          for l in LAYER_ORDER]
            legend_y = inset_y + inset_h + 0.04
            fig.legend(handles=legend_els, title='Domain Layer',
                       loc='lower right',
                       bbox_to_anchor=(inset_x + inset_w, legend_y),
                       fontsize=9, frameon=True, fancybox=True,
                       ncol=4)

        plt.savefig(OUTPUT_DIR / "Bridge_Nodes.png")
        plt.close()

    # ---- Figure 3: Grey Rhino Scatter -------------------------------------
    def fig_grey_rhino_scatter(self):
        logger.info(">>> Grey_Rhino_Scatter.png")

        rows = []
        for p in self.paths:
            rows.append({
                'Strength':  p['strength'],
                'Novelty':   p['novelty_score'],
                'n_layers':  p['n_layers'],
                'nodes':     p['nodes'],
            })
        df = pd.DataFrame(rows)

        novelty_thresh = 0.7
        strength_thresh = df['Strength'].quantile(0.4)

        # ---- stats for figure caption ----
        n_total = len(df)
        n_2l = (df['n_layers'] == 2).sum()
        n_3l = (df['n_layers'] == 3).sum()
        n_4l = (df['n_layers'] == 4).sum()
        n_high_novelty = (df['Novelty'] > novelty_thresh).sum()
        gr_zone = df[(df['Novelty'] > novelty_thresh) & (df['Strength'] < strength_thresh)]
        n_grey_rhino = len(gr_zone)
        logger.info("  [Caption Stats] Grey Rhino Scatter")
        logger.info(f"    Total paths: {n_total}")
        logger.info(f"    By n_layers: 2-layer={n_2l}, 3-layer={n_3l}, 4-layer={n_4l}")
        logger.info(f"    Novelty > {novelty_thresh}: {n_high_novelty} ({n_high_novelty/n_total*100:.1f}%)")
        logger.info(f"    Strength threshold (40th pctile): {strength_thresh:.4f}")
        logger.info(f"    Grey Rhino Zone (high novelty + low strength): {n_grey_rhino} paths")
        logger.info(f"    Strength range: {df['Strength'].min():.4f} – {df['Strength'].max():.4f}")
        logger.info(f"    Novelty range: {df['Novelty'].min():.4f} – {df['Novelty'].max():.4f}")
        # Log annotated paths
        gr_sorted = gr_zone.sort_values('Novelty', ascending=False)
        if len(gr_sorted) >= 1:
            p1 = gr_sorted.iloc[0]
            logger.info(f"    Annotated #1: {' → '.join(p1['nodes'])} "
                        f"(strength={p1['Strength']:.4f}, novelty={p1['Novelty']:.4f})")
        if len(gr_sorted) >= 6:
            p2 = gr_sorted.iloc[5]
            logger.info(f"    Annotated #2: {' → '.join(p2['nodes'])} "
                        f"(strength={p2['Strength']:.4f}, novelty={p2['Novelty']:.4f})")

        fig, ax = plt.subplots(figsize=(10, 7))

        # Grey-Rhino zone shading
        ax.fill_between(
            [df['Strength'].min() - 0.02, strength_thresh],
            novelty_thresh,
            df['Novelty'].max() + 0.05,
            color='#fde0dd', alpha=0.5, zorder=0,
            label='Grey Rhino Zone',
        )

        # Scatter by n_layers
        for nl, color, alpha, size, label in [
            (2, '#bababa', 0.4, 20, '2-Layer Paths'),
            (3, '#1f77b4', 0.7, 40, '3-Layer Paths'),
            (4, '#d62728', 0.8, 60, '4-Layer Paths'),
        ]:
            mask = df['n_layers'] == nl
            ax.scatter(df.loc[mask, 'Strength'], df.loc[mask, 'Novelty'],
                       c=color, alpha=alpha, s=size, label=label,
                       zorder=nl, edgecolors='white', linewidth=0.3)

        # Auto-annotate top grey rhinos
        gr = df[(df['Novelty'] > novelty_thresh) &
                (df['Strength'] < strength_thresh)].copy()
        gr = gr.sort_values('Novelty', ascending=False)

        def _annotate(row, color, text_offset):
            nodes_str = '\n$\\rightarrow$ '.join(
                format_entity(n) for n in row['nodes'])
            ax.scatter(row['Strength'], row['Novelty'],
                       s=150, c=color, marker='*', zorder=6,
                       edgecolors='black')
            ax.annotate(
                nodes_str,
                xy=(row['Strength'], row['Novelty']),
                xytext=text_offset,
                textcoords='offset points',
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="arc3,rad=.2",
                                color='black'),
                fontsize=9, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec="gray", alpha=0.9),
                zorder=7,
            )

        if len(gr) >= 1:
            _annotate(gr.iloc[0], '#d62728', (30, -10))
        if len(gr) >= 6:
            _annotate(gr.iloc[5], '#2ca02c', (-160, -50))

        # Threshold lines
        ax.axhline(y=novelty_thresh, color='gray', ls='--', alpha=0.6)
        ax.axvline(x=strength_thresh, color='gray', ls='--', alpha=0.6)

        ax.set_xlabel('Path Strength (Literature Consensus)',
                      fontweight='bold', fontsize=12)
        ax.set_ylabel('Novelty Score (HeDA Discovery)',
                      fontweight='bold', fontsize=12)
        ax.legend(loc='lower right', frameon=True, fancybox=True,
                  framealpha=0.9)

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "Grey_Rhino_Scatter.png")
        plt.close()

    # ---- Figure 4: Bio-Ecological Bottleneck (Sankey) ---------------------
    def fig_bio_ecological_bottleneck(self):
        logger.info(">>> Bio_Ecological_Bottleneck.png")

        # Compute sub-category level cross-layer flows from paths
        flow_counts = defaultdict(float)
        for p in self.paths:
            nodes, layers = p['nodes'], p['layers']
            w = p['strength']
            for i in range(len(nodes) - 1):
                sl, tl = layers[i], layers[i + 1]
                if sl == tl:
                    continue
                ss = classify_sub(nodes[i], sl)
                ts = classify_sub(nodes[i + 1], tl)
                flow_counts[(ss, ts)] += w

        # ---- layout -------------------------------------------------------
        fig, ax = plt.subplots(figsize=(18, 9))
        ax.axis('off')

        x_pos = {'physical': 0, 'biological': 1.1, 'social': 2.2, 'economic': 3.3}
        NW = 0.16  # node width

        # Build node position config (3 sub-cats per column, uniform y)
        y_slots = [90, 58, 26]
        h_slots = [22, 20, 20]
        pos_cfg = {}
        for layer in LAYER_ORDER:
            subs = list(SUB_CATS[layer].keys())
            for j, sub in enumerate(subs):
                pos_cfg[sub] = {
                    'x': x_pos[layer], 'y': y_slots[j], 'h': h_slots[j],
                    'c': NATURE_COLORS[layer], 'layer': layer,
                }

        # ---- draw nodes ---------------------------------------------------
        for name, cfg in pos_cfg.items():
            rect = mpatches.FancyBboxPatch(
                (cfg['x'], cfg['y'] - cfg['h'] / 2), NW, cfg['h'],
                boxstyle="round,pad=0.02",
                ec="white", fc=cfg['c'], alpha=0.9, zorder=10,
            )
            ax.add_patch(rect)
            ax.text(cfg['x'] + NW / 2, cfg['y'],
                    name.replace(' ', '\n'),
                    ha='center', va='center', color='white',
                    fontweight='bold', fontsize=8, zorder=11)

        # ---- draw ribbons -------------------------------------------------
        def _ribbon(u, v, width, color, alpha=0.5, zorder=5):
            if u not in pos_cfg or v not in pos_cfg:
                return
            s, e = pos_cfg[u], pos_cfg[v]
            p1 = (s['x'] + NW, s['y'])
            p2 = (e['x'],      e['y'])
            gap = p2[0] - p1[0]
            mx = (p1[0] + p2[0]) / 2
            c1, c2 = (mx, p1[1]), (mx, p2[1])
            hw = width / 2
            verts = [
                (p1[0], p1[1] - hw),
                (c1[0], c1[1] - hw), (c2[0], c2[1] - hw), (p2[0], p2[1] - hw),
                (p2[0], p2[1] + hw),
                (c2[0], c2[1] + hw), (c1[0], c1[1] + hw), (p1[0], p1[1] + hw),
                (p1[0], p1[1] - hw),
            ]
            codes = [mpath.Path.MOVETO,
                     mpath.Path.CURVE4, mpath.Path.CURVE4, mpath.Path.CURVE4,
                     mpath.Path.LINETO,
                     mpath.Path.CURVE4, mpath.Path.CURVE4, mpath.Path.CURVE4,
                     mpath.Path.CLOSEPOLY]
            patch = mpatches.PathPatch(
                mpath.Path(verts, codes),
                facecolor=color, alpha=alpha, edgecolor='none', zorder=zorder)
            ax.add_patch(patch)

        # Only draw left-to-right flows (trigger → impact direction)
        layer_idx = {l: i for i, l in enumerate(LAYER_ORDER)}
        lr_flows = {}
        for (src, tgt), w in flow_counts.items():
            if src not in pos_cfg or tgt not in pos_cfg:
                continue
            sl = pos_cfg[src]['layer']
            tl = pos_cfg[tgt]['layer']
            if layer_idx[sl] < layer_idx[tl]:
                lr_flows[(src, tgt)] = w

        # ---- stats for figure caption ----
        logger.info("  [Caption Stats] Bio-Ecological Bottleneck (Sankey)")
        logger.info(f"    Total left→right sub-category flows: {len(lr_flows)}")
        # Aggregate by layer pair
        layer_pair_totals = defaultdict(float)
        for (src, tgt), w in lr_flows.items():
            sl = pos_cfg[src]['layer']
            tl = pos_cfg[tgt]['layer']
            layer_pair_totals[(sl, tl)] += w
        logger.info("    Aggregated layer-pair flows (weighted):")
        for (sl, tl), w in sorted(layer_pair_totals.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"      {sl} → {tl}: {w:.1f}")
        # Top-10 sub-category flows
        logger.info("    Top-10 sub-category flows:")
        for i, ((src, tgt), w) in enumerate(sorted(lr_flows.items(), key=lambda x: x[1], reverse=True)[:10]):
            logger.info(f"      {i+1}. {src} → {tgt}: {w:.1f}")
        # Mediation ratio
        phy_bio = layer_pair_totals.get(('physical', 'biological'), 0)
        phy_eco = layer_pair_totals.get(('physical', 'economic'), 0)
        bio_eco = layer_pair_totals.get(('biological', 'economic'), 0)
        logger.info(f"    Physical→Biological: {phy_bio:.1f}")
        logger.info(f"    Physical→Economic (direct): {phy_eco:.1f}")
        logger.info(f"    Biological→Economic: {bio_eco:.1f}")
        if phy_eco > 0:
            logger.info(f"    Bio-mediation ratio (P→B / P→E): {phy_bio/phy_eco:.2f}x")

        if lr_flows:
            max_w = max(lr_flows.values())
            for (src, tgt), w in sorted(lr_flows.items(),
                                        key=lambda x: x[1], reverse=True):
                scaled = 2 + 20 * (w / max_w)
                tl = pos_cfg[tgt]['layer']
                _ribbon(src, tgt, scaled, NATURE_COLORS[tl])

        # ---- decorations --------------------------------------------------
        ax.set_xlim(-0.3, 3.8)
        ax.set_ylim(0, 135)

        # Bio amplification zone
        bio_subs = list(SUB_CATS['biological'].keys())
        bio_ymin = min(pos_cfg[n]['y'] - pos_cfg[n]['h'] / 2 for n in bio_subs)
        bio_ymax = max(pos_cfg[n]['y'] + pos_cfg[n]['h'] / 2 for n in bio_subs)
        cb = x_pos['biological'] + NW / 2
        frac_lo = max(0, (bio_ymin - 5) / 135)
        frac_hi = min(1, (bio_ymax + 5) / 135)
        ax.axvspan(cb - 0.38, cb + 0.38,
                   ymin=frac_lo, ymax=frac_hi,
                   color=NATURE_COLORS['biological_light'], alpha=0.1, zorder=0)

        # Column headers
        header_y = 120
        header_labels = {
            'physical':   'Physical Triggers',
            'biological': 'Biological Systems',
            'social':     'Social Systems',
            'economic':   'Economic Impacts',
        }
        for layer in LAYER_ORDER:
            cx = x_pos[layer] + NW / 2
            ax.text(cx, header_y, header_labels[layer],
                    ha='center', fontweight='bold', fontsize=11,
                    color=NATURE_COLORS[layer])

        # Bio bottleneck title (concise)
        ax.text(cb, 126,
                "Bio-Ecological\nBottleneck",
                ha='center', va='center', fontsize=12, fontweight='bold',
                color=NATURE_COLORS['biological'])

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "Bio_Ecological_Bottleneck.png")
        plt.close()

    # ---- Figure 5: Risk Network Coupling ----------------------------------
    def fig_risk_network_coupling(self):
        logger.info(">>> Risk_Network_Coupling.png")

        G_und = self.G.to_undirected()
        cc = max(nx.connected_components(G_und), key=len)
        G = G_und.subgraph(cc).copy()

        MAX_NODES = 150
        if len(G) > MAX_NODES:
            # Layer-balanced selection: top nodes per layer by degree
            deg = dict(G.degree())
            layer_nodes = defaultdict(list)
            for n in G.nodes():
                l = self.entity_layer.get(n, 'unknown')
                layer_nodes[l].append((n, deg[n]))
            for l in layer_nodes:
                layer_nodes[l].sort(key=lambda x: x[1], reverse=True)

            # Allocate slots proportionally but with minimum per layer
            present_layers = [l for l in LAYER_ORDER if layer_nodes.get(l)]
            min_per_layer = 15
            remaining = MAX_NODES - min_per_layer * len(present_layers)
            total_deg = sum(sum(d for _, d in layer_nodes[l])
                            for l in present_layers)

            selected = set()
            for l in present_layers:
                layer_total = sum(d for _, d in layer_nodes[l])
                quota = min_per_layer + int(
                    remaining * layer_total / total_deg) if total_deg else min_per_layer
                for n, _ in layer_nodes[l][:quota]:
                    selected.add(n)

            G = G.subgraph(selected).copy()
            if not nx.is_connected(G):
                cc = max(nx.connected_components(G), key=len)
                G = G.subgraph(cc).copy()

        # Community detection
        try:
            import community.community_louvain as community_louvain
            partition = community_louvain.best_partition(G, random_state=42)
        except ImportError:
            logger.warning("python-louvain not installed; falling back")
            partition = {n: i % 5 for i, n in enumerate(G.nodes())}

        # ---- stats for figure caption ----
        from collections import Counter
        n_communities = len(set(partition.values()))
        layer_counts = Counter(self.entity_layer.get(n, 'unknown') for n in G.nodes())
        intra_e = sum(1 for u, v in G.edges() if partition[u] == partition[v])
        bridge_e = G.number_of_edges() - intra_e
        comm_sizes = Counter(partition.values())
        top_comms = comm_sizes.most_common(5)
        logger.info("  [Caption Stats] Risk Network Coupling")
        logger.info(f"    Displayed nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")
        logger.info(f"    Full graph: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")
        logger.info(f"    Node layer distribution: {dict(layer_counts)}")
        logger.info(f"    Louvain communities: {n_communities}")
        logger.info(f"    Top-5 community sizes: {[s for _, s in top_comms]}")
        logger.info(f"    Intra-community edges: {intra_e} ({intra_e/G.number_of_edges()*100:.1f}%)")
        logger.info(f"    Inter-community (bridge) edges: {bridge_e} ({bridge_e/G.number_of_edges()*100:.1f}%)")

        # Layout
        pos = nx.spring_layout(G, k=3.5, iterations=200, seed=123)
        pos = self._repulse(pos)

        fig, ax = plt.subplots(figsize=(16, 16))
        ax.set_facecolor('#FFFFFF')
        fig.patch.set_facecolor('#FFFFFF')

        # Edges
        intra = [(u, v) for u, v in G.edges() if partition[u] == partition[v]]
        bridge = [(u, v) for u, v in G.edges() if partition[u] != partition[v]]

        INTRA_C = '#D4B896'
        BRIDGE_C = '#4A90C4'

        nx.draw_networkx_edges(G, pos, edgelist=intra,
                               alpha=0.3, edge_color=INTRA_C, width=0.8, ax=ax)
        if bridge:
            nx.draw_networkx_edges(G, pos, edgelist=bridge,
                                   alpha=0.5, edge_color=BRIDGE_C,
                                   width=1.5, ax=ax)

        # Nodes — colored by entity layer
        node_colors = [LAYER_COLORS.get(self.entity_layer.get(n, 'unknown'),
                                        '#7f7f7f')
                       for n in G.nodes()]
        deg = dict(G.degree())
        max_d = max(deg.values()) if deg else 1
        sizes = [60 + 350 * (deg[n] / max_d) for n in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color=node_colors,
                               edgecolors='white', linewidths=1.0, alpha=0.9,
                               ax=ax)

        # ---- Labels -------------------------------------------------------
        all_x = [p[0] for p in pos.values()]
        all_y = [p[1] for p in pos.values()]
        drx = (max(all_x) - min(all_x)) if all_x else 1
        dry = (max(all_y) - min(all_y)) if all_y else 1

        placed_bboxes = []

        def _est_bbox(x, y, text, fs):
            sc = fs / 10.0
            lines = text.split('\n')
            w = max(len(l) for l in lines) * (drx / 128) * sc + 0.06
            h = len(lines) * (dry / 40) * sc + 0.04
            return (x - w / 2, y - h / 2, x + w / 2, y + h / 2)

        def _overlaps(bb):
            for ob in placed_bboxes:
                if bb[0] < ob[2] and bb[2] > ob[0] and bb[1] < ob[3] and bb[3] > ob[1]:
                    return True
            return False

        def _try_place(cx, cy, text, fs, attempts=60):
            bb = _est_bbox(cx, cy, text, fs)
            if not _overlaps(bb):
                placed_bboxes.append(bb)
                return cx, cy, True
            for i in range(1, attempts + 1):
                angle = i * (2 * np.pi / 16) + 0.3
                ring = (i - 1) // 16 + 1
                d = 0.09 * ring
                nx_, ny_ = cx + d * np.cos(angle), cy + d * np.sin(angle)
                bb = _est_bbox(nx_, ny_, text, fs)
                if not _overlaps(bb):
                    placed_bboxes.append(bb)
                    return nx_, ny_, True
            return cx, cy, False

        def _wrap(text, mc=18):
            words, lines, cur = text.split(), [], ""
            for w in words:
                if cur and len(cur) + 1 + len(w) > mc:
                    lines.append(cur); cur = w
                else:
                    cur = f"{cur} {w}".strip()
            if cur:
                lines.append(cur)
            return '\n'.join(lines)

        # Community center labels
        comm_members = defaultdict(list)
        for n, cid in partition.items():
            comm_members[cid].append(n)

        labeled = set()
        for cid, members in sorted(comm_members.items(),
                                    key=lambda x: len(x[1]), reverse=True):
            if len(members) < 3:
                continue
            cx = np.mean([pos[n][0] for n in members])
            cy = np.mean([pos[n][1] for n in members])
            subg = G.subgraph(members)
            center = max(dict(subg.degree()).items(), key=lambda x: x[1])[0]
            label = _wrap(format_entity(center))
            labeled.add(center)
            lx, ly, ok = _try_place(cx, cy, label, 16)
            if ok:
                ax.text(lx, ly, label, fontsize=16, fontweight='bold',
                        color='#333333', alpha=0.85, ha='center', va='center',
                        bbox=dict(facecolor='white', alpha=0.75,
                                  edgecolor='#CCCCCC',
                                  boxstyle='round,pad=0.3', linewidth=0.5))

        # Bridge node labels (top-5 not already labeled)
        bc = nx.betweenness_centrality(G)
        br_candidates = [(n, bc[n]) for n in G.nodes()
                         if n not in labeled
                         and any(partition[nb] != partition[n]
                                 for nb in G.neighbors(n))]
        br_candidates.sort(key=lambda x: x[1], reverse=True)
        for n, _ in br_candidates[:5]:
            x, y = pos[n]
            label = _wrap(format_entity(n), 16)
            lx, ly, ok = _try_place(x, y + 0.05, label, 12)
            if ok:
                labeled.add(n)
                ax.text(lx, ly, label, fontsize=12, color=BRIDGE_C,
                        fontweight='bold', ha='center',
                        bbox=dict(facecolor='white', alpha=0.9,
                                  edgecolor=BRIDGE_C, linewidth=0.5,
                                  boxstyle='round,pad=0.2'))

        # Legend (by layer)
        legend_h = [
            Patch(facecolor=LAYER_COLORS['physical'],   label='Physical'),
            Patch(facecolor=LAYER_COLORS['biological'],  label='Biological'),
            Patch(facecolor=LAYER_COLORS['social'],      label='Social'),
            Patch(facecolor=LAYER_COLORS['economic'],    label='Economic'),
            Patch(facecolor='none', label=''),
            Patch(facecolor=INTRA_C,  label='Intra-Community'),
            Patch(facecolor=BRIDGE_C, label='Inter-Community (Coupling)'),
        ]
        ax.legend(handles=legend_h, loc='lower right',
                  bbox_to_anchor=(1.0, 0.15), frameon=True,
                  facecolor='white', edgecolor='#CCCCCC', fontsize=13)

        plt.axis('off')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "Risk_Network_Coupling.png")
        plt.close()

    # ---- helpers ----------------------------------------------------------
    @staticmethod
    def _repulse(pos, min_dist=0.16, iters=200, fs=0.015):
        arr = {n: np.array(p, dtype=float) for n, p in pos.items()}
        nodes = list(arr.keys())
        for _ in range(iters):
            disp = {n: np.zeros(2) for n in nodes}
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    ni, nj = nodes[i], nodes[j]
                    diff = arr[ni] - arr[nj]
                    d = np.linalg.norm(diff)
                    if 1e-6 < d < min_dist:
                        push = diff / d * fs * ((min_dist - d) / min_dist) ** 0.5
                        disp[ni] += push
                        disp[nj] -= push
            for n in nodes:
                arr[n] += disp[n]
        return {n: tuple(p) for n, p in arr.items()}

    # ---- entry point ------------------------------------------------------
    def run(self):
        if not self.load_data():
            logger.error("Failed to load data.")
            return

        self.fig_layer_transition_matrix()
        self.fig_bridge_nodes()
        self.fig_grey_rhino_scatter()
        self.fig_bio_ecological_bottleneck()
        self.fig_risk_network_coupling()

        logger.info("All 5 figures generated in %s", OUTPUT_DIR)


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FigureGenerator().run()
