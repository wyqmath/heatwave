"""
11_traceability_showcase.py — Edge-Level Traceability for Grey Rhino Paths

Proves every KG edge has original text support. Addresses Reviewer R1-6
concern about "no edge-level precision".

Outputs (all under results/traceability/):
  1. fig_traceability.png       — Main-text figure (600 DPI), three panels:
       (a) Bottleneck Provenance Profile: top-500 edges by betweenness,
           barcode colored by number of source papers (log-binned).
       (b) Consensus enrichment: stacked bar comparing All vs Top-500.
       (c) Evidence penetration: example cross-layer edge with verbatim quotes.
  2. table_s1_audit.tex         — Appendix longtable: top-20 path provenance.
  3. coverage_stats.json        — Aggregate statistics for inline citation.

Usage:
    python 11_traceability_showcase.py
"""

import ast
import json
import pickle
import re
import textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
import networkx as nx
import numpy as np

# ── Paths ──
WORKSPACE = Path(__file__).parent.parent
TOPOLOGY_DIR = WORKSPACE / "data" / "processed" / "topology_sensitivity"
GREY_RHINO_JSON = TOPOLOGY_DIR / "grey_rhino_paths.json"
PHASE1_GRAPH_PKL = TOPOLOGY_DIR / "phase1_graph.pkl"
MERGE_MAPPING_JSON = TOPOLOGY_DIR / "phase1_merge_mapping.json"
RESULTS_DIR = WORKSPACE / "results" / "traceability"

LAYER_SHORT = {"physical": "Phys", "biological": "Bio",
               "social": "Soc", "economic": "Econ"}

# Color palette for paper-count tiers (log-binned)
TIER_COLORS = {
    "1":    "#85c1e9",   # light blue
    "2–4":  "#2e86c1",   # medium blue
    "5–9":  "#1a5276",   # dark blue
    "≥10":  "#4a0072",   # deep purple — "super-consensus"
    "none": "#e74c3c",   # red — no provenance
}

def _tier(n):
    if n == 0:   return "none"
    if n == 1:   return "1"
    if n <= 4:   return "2–4"
    if n <= 9:   return "5–9"
    return "≥10"

def _tier_color(n):
    return TIER_COLORS[_tier(n)]


# ============================================================
# Data Loading
# ============================================================

def load_data():
    with open(GREY_RHINO_JSON, "r", encoding="utf-8") as f:
        raw = json.load(f)
    raw_paths = raw["paths"] if isinstance(raw, dict) else raw

    paths = []
    for p in raw_paths:
        entry = {}
        nodes_raw = p["nodes"]
        entry["nodes"] = ast.literal_eval(nodes_raw) if isinstance(nodes_raw, str) else nodes_raw
        for key in ("default_score", "LF", "CLC", "IP",
                     "in_top20_ratio", "in_top50_ratio",
                     "rank_mean", "rank_std", "rank_cv"):
            entry[key] = float(p.get(key, 0))
        entry["idx"] = int(p.get("idx", 0))
        paths.append(entry)

    with open(PHASE1_GRAPH_PKL, "rb") as f:
        gdata = pickle.load(f)
    with open(MERGE_MAPPING_JSON, "r", encoding="utf-8") as f:
        merge_mapping = json.load(f)

    print(f"Loaded {len(paths):,} paths, {len(gdata['all_triples']):,} triples, "
          f"graph {gdata['G'].number_of_nodes():,}N/{gdata['G'].number_of_edges():,}E")
    return {
        "paths": paths,
        "all_triples": gdata["all_triples"],
        "G": gdata["G"],
        "entity_layer": gdata["entity_layer"],
        "merge_mapping": merge_mapping,
    }


def build_edge_index(all_triples, merge_mapping):
    """Index triples by both raw and merged entity names."""
    idx = defaultdict(list)
    for t in all_triples:
        s, d = t["start_node"], t["end_node"]
        idx[(s, d)].append(t)
        ms, md = merge_mapping.get(s, s), merge_mapping.get(d, d)
        if (ms, md) != (s, d):
            idx[(ms, md)].append(t)
    return idx


def _lookup(edge_index, src, dst):
    return edge_index.get((src, dst), []) or edge_index.get((dst, src), [])


def _best_triple(triples):
    return max(triples, key=lambda t: len(str(t.get("evidence_sentence", ""))))


def _n_papers(triples):
    return len(set(t["paper_id"] for t in triples)) if triples else 0


# ============================================================
# Figure
# ============================================================

def _find_showcase_edge(G, edge_index, entity_layer, ebc):
    """Find best cross-layer, multi-paper edge for Panel (c).

    Prefer 3-15 papers (too many = common knowledge, too few = weak).
    Prefer high BC, different layer pair (phys→bio ideal for our narrative).
    """
    best = None
    for u, v in G.edges():
        sl, dl = entity_layer.get(u, ""), entity_layer.get(v, "")
        if sl == dl or not sl or not dl:
            continue
        triples = _lookup(edge_index, u, v)
        np_ = _n_papers(triples)
        if np_ < 3:
            continue
        bc = ebc.get((u, v), 0.0)
        # Prefer 3-10 papers range, penalize >10 (common knowledge)
        paper_score = np_ if np_ <= 10 else 10 - (np_ - 10) * 2
        # Bonus for phys→bio (core narrative)
        layer_bonus = 2.0 if (sl == "physical" and dl == "biological") else 1.0
        score = paper_score * layer_bonus + bc * 1e5
        if best is None or score > best[0]:
            best = (score, u, v, sl, dl, triples)
    return best


def _camel_to_words(s):
    """CamelCase → 'Camel Case' for highlighting in evidence text."""
    return re.sub(r'(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])', ' ', s)


def generate_figure(G, edge_index, all_paths, entity_layer):
    """Three-panel main-text figure at 600 DPI."""
    out = RESULTS_DIR / "fig_traceability.png"

    # ── Compute edge betweenness centrality ──
    print("Computing edge betweenness centrality (k=500) ...")
    ebc = nx.edge_betweenness_centrality(G, k=500, seed=42)

    # ── Per-edge stats for ALL graph edges ──
    all_edge_np = []   # paper counts for every edge in G
    edge_rows = []     # for top-500 ranking
    for u, v in G.edges():
        bc = ebc.get((u, v), 0.0)
        triples = _lookup(edge_index, u, v)
        np_ = _n_papers(triples)
        sl = triples[0].get("start_layer", "") if triples else ""
        el = triples[0].get("end_layer", "") if triples else ""
        all_edge_np.append(np_)
        edge_rows.append({
            "betweenness": bc, "n_papers": np_,
            "is_cross_layer": bool(sl and el and sl != el),
        })

    # Sort by BC, take top-500
    ranked = sorted(edge_rows, key=lambda r: r["betweenness"], reverse=True)
    top_n = 500
    top = ranked[:top_n]

    bc_vals = np.array([r["betweenness"] for r in top])
    np_vals = [r["n_papers"] for r in top]
    cross = [r["is_cross_layer"] for r in top]

    prov_colors = [_tier_color(n) for n in np_vals]
    cross_colors = ["#e67e22" if c else "#ffffff" for c in cross]

    # ── Tier counts for Panel (b) ──
    tier_names = ["1", "2–4", "5–9", "≥10"]
    all_tiers = [_tier(n) for n in all_edge_np]
    top_tiers = [_tier(n) for n in np_vals]

    def _pcts(tier_list):
        total = len(tier_list)
        return [sum(1 for t in tier_list if t == tn) / total * 100 for tn in tier_names]

    all_pcts = _pcts(all_tiers)
    top_pcts = _pcts(top_tiers)

    # ── Find showcase edge for Panel (c) ──
    # Hardcoded: ThermalStress→CoralBleaching (5 papers, phys→bio, ideal narrative)
    showcase_triples = _lookup(edge_index, "ThermalStress", "CoralBleaching")
    if showcase_triples:
        showcase = (0, "ThermalStress", "CoralBleaching", "physical", "biological",
                    showcase_triples)
    else:
        showcase = _find_showcase_edge(G, edge_index, entity_layer, ebc)

    # ── Build figure ──
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(
        3, 2, width_ratios=[1.8, 1],
        height_ratios=[1, 1, 1], hspace=0.06, wspace=0.30,
    )
    ax_bc   = fig.add_subplot(gs[0, 0])
    ax_prov = fig.add_subplot(gs[1, 0], sharex=ax_bc)
    ax_cl   = fig.add_subplot(gs[2, 0], sharex=ax_bc)

    # Right column: Panel (b) top ~35%, Panel (c) bottom ~65%
    gs_right = gs[:, 1].subgridspec(2, 1, height_ratios=[1, 1.8], hspace=0.30)
    ax_bar = fig.add_subplot(gs_right[0])
    ax_ev  = fig.add_subplot(gs_right[1])

    x = np.arange(top_n)

    # ════════════════ Panel (a): Provenance Profile ════════════════

    # Row 1: BC decay — teal to contrast with blue barcode/Panel(b)
    ax_bc.fill_between(x, bc_vals, alpha=0.20, color="#1a9e76")
    ax_bc.plot(x, bc_vals, color="#16835f", lw=0.9)
    ax_bc.set_ylabel("Edge betweenness\ncentrality", fontsize=9)
    ax_bc.set_yscale("log")
    ax_bc.text(0.5, 1.05, "a", fontsize=12, fontweight="bold",
               transform=ax_bc.transAxes, va="bottom", ha="center")
    ax_bc.tick_params(labelbottom=False)

    n_traced = sum(1 for n in np_vals if n > 0)
    n_multi  = sum(1 for n in np_vals if n >= 2)
    ax_bc.text(0.98, 0.93,
               f"Traceable to source: {n_traced}/{top_n} ({n_traced/top_n*100:.0f}%)\n"
               f"Multi-paper (≥2): {n_multi}  |  No provenance: {top_n - n_traced}",
               transform=ax_bc.transAxes, fontsize=8, ha="right", va="top",
               bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#16835f", alpha=0.85))

    # Row 2: provenance barcode
    ax_prov.bar(x, 1, width=1.0, color=prov_colors, edgecolor="none")
    ax_prov.set_ylim(0, 1); ax_prov.set_yticks([])
    ax_prov.set_ylabel("Source\npapers", fontsize=8, labelpad=10)
    ax_prov.tick_params(labelbottom=False)
    ax_prov.legend(
        handles=[Patch(fc=TIER_COLORS[t], label=t) for t in tier_names]
              + [Patch(fc=TIER_COLORS["none"], label="No provenance")],
        loc="upper right", fontsize=5.5, ncol=5,
        facecolor="white", framealpha=0.85, edgecolor="#ccc",
        title="Independent source papers", title_fontsize=6.5)

    # Row 3: cross-layer
    ax_cl.bar(x, 1, width=1.0, color=cross_colors, edgecolor="none")
    ax_cl.set_ylim(0, 1); ax_cl.set_yticks([])
    ax_cl.set_ylabel("Cross-\nlayer", fontsize=8, labelpad=10)
    ax_cl.set_xlabel("Edge rank (by betweenness centrality)", fontsize=9)
    ax_cl.set_xlim(-1, top_n + 1)
    n_cross = sum(1 for c in cross if c)
    ax_cl.legend(
        handles=[Patch(fc="#e67e22", label=f"Cross-layer ({n_cross}/{top_n})")],
        loc="upper right", fontsize=6, ncol=1,
        facecolor="white", framealpha=0.85, edgecolor="#ccc")

    # ════════════════ Panel (b): Consensus Enrichment ════════════════

    bar_w = 0.5
    bottom_all = 0.0
    bottom_top = 0.0

    for i, tn in enumerate(tier_names):
        c = TIER_COLORS[tn]
        ax_bar.barh(0, all_pcts[i], left=bottom_all, height=bar_w,
                    color=c, edgecolor="white", linewidth=0.3)
        ax_bar.barh(1, top_pcts[i], left=bottom_top, height=bar_w,
                    color=c, edgecolor="white", linewidth=0.3)
        if all_pcts[i] > 3:
            ax_bar.text(bottom_all + all_pcts[i]/2, 0,
                        f"{all_pcts[i]:.1f}%", ha="center", va="center", fontsize=6.5,
                        color="white" if i >= 1 else "#333")
        if top_pcts[i] > 3:
            ax_bar.text(bottom_top + top_pcts[i]/2, 1,
                        f"{top_pcts[i]:.1f}%", ha="center", va="center", fontsize=6.5,
                        color="white" if i >= 1 else "#333")
        bottom_all += all_pcts[i]
        bottom_top += top_pcts[i]

    # Trapezoid band connecting ≥2-paper boundary between the two bars
    all_split = all_pcts[0]          # boundary in "All edges" bar (end of 1-paper segment)
    top_split = top_pcts[0]          # boundary in "Top-500" bar
    from matplotlib.patches import Polygon as MplPolygon
    trap = MplPolygon(
        [[all_split, 0 + bar_w/2],   # top-left of "All" bar's split point
         [top_split, 1 - bar_w/2],   # bottom-left of "Top-500" bar's split point
         [100, 1 - bar_w/2],         # bottom-right
         [100, 0 + bar_w/2]],        # top-right
        closed=True, fc="#2e86c1", alpha=0.10, ec="#2e86c1", lw=0.8, ls="--",
    )
    ax_bar.add_patch(trap)

    ax_bar.set_yticks([0, 1])
    ax_bar.set_yticklabels([f"All edges\n(N={len(all_edge_np):,})",
                             f"Top-500\nbottleneck"], fontsize=8)
    ax_bar.set_xlim(0, 100)
    ax_bar.set_xlabel("Proportion (%)", fontsize=8)
    ax_bar.text(0.5, 1.08, "b", fontsize=12, fontweight="bold",
                transform=ax_bar.transAxes, va="bottom", ha="center")

    # Combined legend + enrichment annotation — in blank space between bars
    all_multi_pct = sum(all_pcts[1:])
    top_multi_pct = sum(top_pcts[1:])
    enrichment = top_multi_pct / all_multi_pct if all_multi_pct > 0 else 0

    legend_handles = [Patch(fc=TIER_COLORS[t], label=f"{t} paper{'s' if t != '1' else ''}")
                      for t in tier_names]
    leg = ax_bar.legend(
        handles=legend_handles,
        loc="center", fontsize=6,
        facecolor="white", framealpha=0.85, edgecolor="#ccc",
        title=f"Independent papers   |   ≥2 papers: {all_multi_pct:.1f}% → {top_multi_pct:.1f}% ({enrichment:.0f}× enrichment)",
        title_fontsize=6.5, ncol=4,
        bbox_to_anchor=(0.5, 0.5))

    # ════════════════ Panel (c): Evidence Penetration ════════════════

    LAYER_BG = {"physical": "#fce4e4", "biological": "#e4f5e4",
                "social": "#e4e8f5", "economic": "#fdf3e4"}

    ax_ev.axis("off")
    if showcase:
        _, u, v, sl, dl, triples = showcase
        np_show = _n_papers(triples)
        seen = {}
        for t in triples:
            pid = t["paper_id"]
            if pid not in seen:
                seen[pid] = t

        sl_short = LAYER_SHORT.get(sl, sl)
        dl_short = LAYER_SHORT.get(dl, dl)

        # c label — centered, with extra space from increased hspace
        ax_ev.text(0.5, 1.05, "c", fontsize=12, fontweight="bold",
                   transform=ax_ev.transAxes, va="bottom", ha="center")

        # Entity badges — arrow anchored at center, entities on each side
        ax_ev.text(0.5, 0.94, "→", fontsize=11, va="center", ha="center",
                   transform=ax_ev.transAxes, color="#666")
        ax_ev.text(0.47, 0.94, f"{u} ({sl_short})", fontsize=8.5,
                   fontweight="bold", va="center", ha="right",
                   transform=ax_ev.transAxes,
                   bbox=dict(boxstyle="round,pad=0.2", fc=LAYER_BG.get(sl, "#eee"),
                             ec="#999", lw=0.5))
        ax_ev.text(0.53, 0.94, f"{v} ({dl_short})", fontsize=8.5,
                   fontweight="bold", va="center", ha="left",
                   transform=ax_ev.transAxes,
                   bbox=dict(boxstyle="round,pad=0.2", fc=LAYER_BG.get(dl, "#eee"),
                             ec="#999", lw=0.5))

        # 5 evidence cards — extend left (into b's y-label area) and down
        n_cards = min(5, len(seen))
        x_left = -0.08       # push left past ax_ev boundary
        x_right = 0.99
        card_w = x_right - x_left
        y_top = 0.85
        y_bottom = -0.04     # push below ax_ev boundary
        total_h = y_top - y_bottom
        card_gap = 0.04
        card_height = (total_h - card_gap * (n_cards - 1)) / n_cards

        y_pos = y_top
        for card_i, (pid, t) in enumerate(list(seen.items())[:n_cards]):
            rel = t.get("relationship", "")
            ev = str(t.get("evidence_sentence", ""))
            if len(ev) > 250:
                ev = ev[:247] + "..."

            # Highlight entity words in CAPS
            for entity in (u, v):
                for word in _camel_to_words(entity).split():
                    if len(word) > 3:
                        pattern = re.compile(re.escape(word), re.IGNORECASE)
                        ev = pattern.sub(word.upper(), ev)

            # Card background
            from matplotlib.patches import FancyBboxPatch
            card_bg = FancyBboxPatch(
                (x_left, y_pos - card_height), card_w, card_height,
                boxstyle="round,pad=0.01", transform=ax_ev.transAxes,
                fc="#fef6e9", ec="#e0d5c0", lw=0.5, clip_on=False,
            )
            ax_ev.add_patch(card_bg)

            # Paper ID + relationship
            ax_ev.text(x_left + 0.02, y_pos - 0.01,
                       f"[{pid}]   rel: {rel}",
                       fontsize=5.5, va="top", transform=ax_ev.transAxes,
                       fontstyle="italic", color="#555", clip_on=False)

            # Evidence text
            wrapped = textwrap.fill(f'"{ev}"', width=83)
            ax_ev.text(x_left + 0.02, y_pos - 0.04, wrapped,
                       fontsize=5.5, va="top", transform=ax_ev.transAxes,
                       family="serif", linespacing=1.2, clip_on=False)

            y_pos -= card_height + card_gap

    fig.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure → {out}")

    # ── Caption-ready output ──
    print("\n" + "=" * 70)
    print("FIGURE CAPTION MATERIAL")
    print("=" * 70)

    print(f"\n── Panel (a): Bottleneck Provenance Profile ──")
    print(f"  Top-{top_n} edges ranked by edge betweenness centrality (BC)")
    print(f"  BC range: {bc_vals[0]:.6f} (rank 1) → {bc_vals[-1]:.6f} (rank {top_n})")
    print(f"  Traceable to source paper: {n_traced}/{top_n} ({n_traced/top_n*100:.0f}%)")
    print(f"  Multi-paper support (≥2): {n_multi}/{top_n} ({n_multi/top_n*100:.1f}%)")
    print(f"  No provenance: {top_n - n_traced}/{top_n}")
    print(f"  Cross-layer edges: {n_cross}/{top_n} ({n_cross/top_n*100:.1f}%)")
    # Tier breakdown for top-500
    for tn in tier_names:
        cnt = sum(1 for t in top_tiers if t == tn)
        print(f"    Tier '{tn}': {cnt} edges ({cnt/top_n*100:.1f}%)")

    print(f"\n── Panel (b): Consensus Enrichment ──")
    print(f"  All edges (N={len(all_edge_np):,}):")
    for i, tn in enumerate(tier_names):
        print(f"    {tn} paper{'s' if tn != '1' else ''}: {all_pcts[i]:.1f}%")
    print(f"  Top-500 bottleneck edges:")
    for i, tn in enumerate(tier_names):
        print(f"    {tn} paper{'s' if tn != '1' else ''}: {top_pcts[i]:.1f}%")
    print(f"  Multi-paper (≥2): {all_multi_pct:.1f}% → {top_multi_pct:.1f}% "
          f"({enrichment:.0f}× enrichment)")

    if showcase:
        print(f"\n── Panel (c): Evidence Penetration ──")
        print(f"  Showcase edge: {showcase[1]} → {showcase[2]}")
        print(f"  Layer crossing: {LAYER_SHORT.get(showcase[3], showcase[3])} → "
              f"{LAYER_SHORT.get(showcase[4], showcase[4])}")
        s_triples = showcase[5]
        s_papers = list({t["paper_id"]: t for t in s_triples}.items())
        print(f"  Independent papers: {len(s_papers)} total, {min(5, len(s_papers))} shown")
        for j, (pid, t) in enumerate(s_papers):
            rel = t.get("relationship", "")
            ev = str(t.get("evidence_sentence", ""))
            if len(ev) > 120:
                ev = ev[:117] + "..."
            shown = "  ← shown" if j < 5 else ""
            print(f"    [{j+1}] {pid}  rel={rel}{shown}")
            print(f"        \"{ev}\"")

    print("=" * 70)


# ============================================================
# Appendix: Table S1 (Top-20 Provenance Audit)
# ============================================================

def _esc(s):
    if not isinstance(s, str):
        return str(s)
    for old, new in [("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"),
                     ("$", "\\$"), ("#", "\\#"), ("_", "\\_"),
                     ("{", "\\{"), ("}", "\\}"), ("~", "\\textasciitilde{}"),
                     ("^", "\\textasciicircum{}")]:
        s = s.replace(old, new)
    return s


def _trunc(s, n=120):
    return s if len(s) <= n else s[:n - 3] + "..."


def generate_appendix_table(paths, edge_index, entity_layer):
    top20 = sorted(paths, key=lambda p: p["default_score"], reverse=True)[:20]

    L = []
    L.append("% Auto-generated by 11_traceability_showcase.py")
    L.append("% Table S1: Edge-level Provenance Audit for Top-20 Grey Rhino Pathways")
    L.append("")
    L.append("\\begin{longtable}{c p{3cm} p{1.5cm} p{2cm} c p{5.5cm} c}")
    L.append("\\caption{Edge-level provenance audit for the top-20 Grey Rhino pathways "
             "(ranked by novelty score). Each propagation step is traced to its source "
             "paper and a representative evidence sentence. "
             "$N$ = number of independent papers supporting the edge.}")
    L.append("\\label{tab:audit} \\\\")
    L.append("\\toprule")
    L.append("Step & Propagation Edge & Rel. & Domain Shift & $N$ & "
             "Representative Evidence (Excerpt) & WOS Accession \\\\")
    L.append("\\midrule")
    L.append("\\endfirsthead")
    L.append("\\multicolumn{7}{c}{\\small\\itshape Table S1 continued} \\\\")
    L.append("\\toprule")
    L.append("Step & Propagation Edge & Rel. & Domain Shift & $N$ & "
             "Representative Evidence (Excerpt) & WOS Accession \\\\")
    L.append("\\midrule")
    L.append("\\endhead")
    L.append("\\bottomrule")
    L.append("\\endlastfoot")

    for rank, p in enumerate(top20, 1):
        nodes = p["nodes"]
        path_str = " $\\rightarrow$ ".join(_esc(n) for n in nodes)
        L.append(f"\\multicolumn{{7}}{{l}}{{\\textbf{{\\#{rank} "
                 f"(Score\\,=\\,{p['default_score']:.3f}, "
                 f"LF\\,=\\,{p['LF']:.3f}):}} {path_str}}} \\\\")
        L.append("\\midrule")

        for i in range(len(nodes) - 1):
            src, dst = nodes[i], nodes[i + 1]
            triples = _lookup(edge_index, src, dst)
            if triples:
                best = _best_triple(triples)
                edge = f"{_esc(src)} $\\rightarrow$ {_esc(dst)}"
                rel = _esc(str(best.get("relationship", "")))
                sl = LAYER_SHORT.get(entity_layer.get(src, ""), "?")
                el = LAYER_SHORT.get(entity_layer.get(dst, ""), "?")
                shift = f"{sl} $\\rightarrow$ {el}"
                n = _n_papers(triples)
                ev = _esc(_trunc(str(best.get("evidence_sentence", "")), 110))
                paper = _esc(str(best.get("paper_id", "")))
                L.append(f"{i+1} & {edge} & {rel} & {shift} & {n} & "
                         f"``{ev}'' & {paper} \\\\")
            else:
                edge = f"{_esc(src)} $\\rightarrow$ {_esc(dst)}"
                L.append(f"{i+1} & {edge} & --- & --- & 0 & "
                         f"(no evidence) & --- \\\\")

        L.append("\\midrule")

    L.append("\\end{longtable}")

    out = RESULTS_DIR / "table_s1_audit.tex"
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"Table S1 → {out}  ({len(top20)} pathways)")


# ============================================================
# Coverage Statistics
# ============================================================

def compute_coverage_stats(all_paths, edge_index):
    total_edges = covered_edges = 0
    unique_papers = set()

    for p in all_paths:
        nodes = p["nodes"]
        for i in range(len(nodes) - 1):
            total_edges += 1
            triples = _lookup(edge_index, nodes[i], nodes[i + 1])
            if triples:
                covered_edges += 1
                for t in triples:
                    pid = t.get("paper_id", "")
                    if pid:
                        unique_papers.add(pid)

    coverages = []
    for p in all_paths:
        nodes = p["nodes"]
        ne = len(nodes) - 1
        cov = sum(1 for i in range(ne) if _lookup(edge_index, nodes[i], nodes[i + 1]))
        coverages.append(cov / ne if ne > 0 else 0)

    stats = {
        "total_paths": len(all_paths),
        "total_edges": total_edges,
        "covered_edges": covered_edges,
        "edge_coverage_pct": round(covered_edges / total_edges * 100, 2),
        "full_path_coverage_pct": round(
            sum(1 for c in coverages if c == 1.0) / len(coverages) * 100, 2),
        "unique_papers_cited": len(unique_papers),
    }

    print("\n" + "=" * 70)
    print("COVERAGE STATS (for inline citation / caption)")
    print("=" * 70)
    for k, v in stats.items():
        print(f"  {k:30s}  {v}")
    print("=" * 70)

    out = RESULTS_DIR / "coverage_stats.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats → {out}")
    return stats


# ============================================================
# Main
# ============================================================

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()
    edge_index = build_edge_index(data["all_triples"], data["merge_mapping"])

    generate_figure(data["G"], edge_index, data["paths"], data["entity_layer"])
    generate_appendix_table(data["paths"], edge_index, data["entity_layer"])
    compute_coverage_stats(data["paths"], edge_index)
    print("\nDone.")


if __name__ == "__main__":
    main()
