#!/usr/bin/env python3
"""Plot per-model modality contributions across benchmarks.

For each benchmark and model:
- Compute pairwise SHAPE analysis from ``*_results_vlm.json`` files.
- Aggregate each modality's contribution by averaging over all pairs containing it.

Produces figures using Plotly (exported as static PNG via kaleido).
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = EVAL_DIR / "shape_metric_output" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Local imports
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(EVAL_DIR))

from longdocurl.eval import eval_score as ldu_eval_score  # noqa: E402
from mmlongbench.eval.eval_score import eval_score as mmlong_eval_score  # noqa: E402
from shape_metric import (  # noqa: E402
    analyze_model_results,
    MODEL_DISPLAY_NAMES,
    MODEL_DISPLAY_ORDER,
    MODALITY_PAIR_ORDER,
)


BENCHMARKS = {
    "LongDocURL": (REPO_ROOT / "results_longdocurl", ldu_eval_score),
    "MMLongBench-Doc": (REPO_ROOT / "results_mmlongbench", mmlong_eval_score),
}

MODALITIES = ["image", "layout", "plain_text", "table"]
MODALITY_LABELS = {
    "image": "Image",
    "layout": "Layout",
    "plain_text": "Text",
    "table": "Table",
}
MODALITY_COLORS = {
    "image": "#4A7DC5",
    "layout": "#6AB08A",
    "plain_text": "#D4726A",
    "table": "#E8A854",
}

PAIR_LABELS_FULL = {
    ("image", "layout"): "Image + Layout",
    ("image", "plain_text"): "Image + Text",
    ("layout", "plain_text"): "Layout + Text",
    ("plain_text", "table"): "Text + Table",
    ("image", "table"): "Image + Table",
    ("layout", "table"): "Layout + Table",
}

PAIR_COLORS = {
    ("image", "layout"): "#4A7DC5",
    ("image", "plain_text"): "#6AB08A",
    ("layout", "plain_text"): "#D4726A",
    ("plain_text", "table"): "#E8A854",
    ("image", "table"): "#B8A9C9",
    ("layout", "table"): "#7CC7CF",
}

MODEL_COLORS = {
    "gemma3_4b": "#4A7DC5",
    "gemma3_27b": "#6AB08A",
    "gpt-4o-mini": "#D4726A",
    "qwen3-vl_8b": "#E8A854",
    "qwen3-vl_30b": "#B8A9C9",
}

COMMON_LAYOUT = dict(
    font=dict(family="Helvetica, Arial, sans-serif", size=13),
    plot_bgcolor="#FAFAFA",
    paper_bgcolor="white",
    margin=dict(l=60, r=20, t=30, b=80),
    bargap=0.25,
    bargroupgap=0.06,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5,
        bgcolor="rgba(0,0,0,0)",
        font=dict(size=12),
    ),
    yaxis=dict(
        zeroline=True,
        zerolinecolor="#BBBBBB",
        zerolinewidth=1,
        gridcolor="#E0E0E0",
        gridwidth=0.5,
    ),
    xaxis=dict(
        tickangle=-25,
    ),
)


def get_ordered_models(model_data: dict) -> list[str]:
    preferred = [m for m in MODEL_DISPLAY_ORDER if m in model_data]
    extras = sorted([m for m in model_data if m not in preferred])
    return preferred + extras


def collect_model_modality_contributions(results_dir: Path, scorer_fn: Callable) -> dict:
    """Return {model: {modality: {S_mean, phi_mean, num_pairs}}}."""
    contributions = {}
    result_files = sorted(results_dir.glob("*_results_vlm.json"))

    for result_file in result_files:
        analysis = analyze_model_results(str(result_file), scorer_fn=scorer_fn)

        s_values = defaultdict(list)
        phi_values = defaultdict(list)

        for group in analysis["modality_groups"]:
            m1, m2 = group["modalities"]
            s1 = group.get(f"S_{m1}")
            s2 = group.get(f"S_{m2}")
            p1 = group.get(f"phi_{m1}")
            p2 = group.get(f"phi_{m2}")

            if s1 is not None:
                s_values[m1].append(float(s1))
            if s2 is not None:
                s_values[m2].append(float(s2))
            if p1 is not None:
                phi_values[m1].append(float(p1))
            if p2 is not None:
                phi_values[m2].append(float(p2))

        model_summary = {}
        for modality in MODALITIES:
            modality_s = s_values.get(modality, [])
            modality_phi = phi_values.get(modality, [])
            model_summary[modality] = {
                "S_mean": float(np.mean(modality_s)) if modality_s else np.nan,
                "phi_mean": float(np.mean(modality_phi)) if modality_phi else np.nan,
                "num_pairs": len(modality_s),
            }

        contributions[analysis["model"]] = model_summary

    return contributions


def collect_model_pair_c12(results_dir: Path, scorer_fn: Callable) -> dict:
    """Return {model: {(m1,m2): C12_value, ...}}."""
    data = {}
    result_files = sorted(results_dir.glob("*_results_vlm.json"))

    for result_file in result_files:
        analysis = analyze_model_results(str(result_file), scorer_fn=scorer_fn)
        pairs = {}
        for group in analysis["modality_groups"]:
            m1, m2 = group["modalities"]
            c12 = group.get("C12_cooperation")
            if c12 is not None:
                pairs[(m1, m2)] = float(c12)
        data[analysis["model"]] = pairs

    return data


def aggregate_modality_scores(all_bench: dict) -> dict:
    """Average per-modality S scores across benchmarks.

    Input:  {bench: {model: {modality: {S_mean: ...}}}}
    Output: {model: {modality: S_mean_across_benchmarks}}
    """
    combined: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for bench_data in all_bench.values():
        for model, mod_dict in bench_data.items():
            for modality, vals in mod_dict.items():
                v = vals["S_mean"]
                if not np.isnan(v):
                    combined[model][modality].append(v)

    result = {}
    for model, mod_dict in combined.items():
        result[model] = {
            modality: float(np.mean(vals)) if vals else np.nan
            for modality, vals in mod_dict.items()
        }
    return result


def aggregate_c12_scores(all_bench: dict) -> dict:
    """Average per-pair C12 across benchmarks.

    Input:  {bench: {model: {(m1,m2): C12}}}
    Output: {model: {(m1,m2): C12_mean}}
    """
    combined: dict[str, dict[tuple, list[float]]] = defaultdict(lambda: defaultdict(list))
    for bench_data in all_bench.values():
        for model, pair_dict in bench_data.items():
            for pair, val in pair_dict.items():
                combined[model][pair].append(val)

    result = {}
    for model, pair_dict in combined.items():
        result[model] = {
            pair: float(np.mean(vals))
            for pair, vals in pair_dict.items()
        }
    return result


def _save(fig: go.Figure, filename: str) -> None:
    """Write a Plotly figure to PNG."""
    out = OUTPUT_DIR / filename
    fig.write_image(str(out), scale=3)
    print(f"Saved: {out}")


def plot_aggregated_s(agg_data: dict) -> None:
    """Single-panel grouped bar chart: S per modality, averaged across benchmarks."""
    models = [m for m in MODEL_DISPLAY_ORDER if m in agg_data]
    model_labels = [MODEL_DISPLAY_NAMES.get(m, m) for m in models]

    fig = go.Figure()
    for modality in MODALITIES:
        values = [agg_data[m].get(modality, None) for m in models]
        fig.add_trace(go.Bar(
            x=model_labels,
            y=values,
            name=MODALITY_LABELS[modality],
            marker_color=MODALITY_COLORS[modality],
            text=[f"{v:.2f}" if v is not None and not np.isnan(v) else "" for v in values],
            textposition="outside",
            textfont=dict(size=12),
        ))

    fig.update_layout(
        font=dict(family="Helvetica, Arial, sans-serif", size=18),
        plot_bgcolor="#FAFAFA",
        paper_bgcolor="white",
        margin=dict(l=80, r=20, t=40, b=100),
        bargap=0.25,
        bargroupgap=0.06,
        barmode="group",
        yaxis_title="Mean SHAPE score (S)",
        width=900,
        height=540,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=14),
            itemwidth=40,
        ),
        yaxis=dict(
            title=dict(font=dict(size=19)),
            tickfont=dict(size=16),
            zeroline=True,
            zerolinecolor="#BBBBBB",
            zerolinewidth=1,
            gridcolor="#E0E0E0",
            gridwidth=0.5,
        ),
        xaxis=dict(
            tickangle=-25,
            tickfont=dict(size=16),
        ),
    )
    _save(fig, "aggregated_S_across_benchmarks.png")


def plot_aggregated_c12(agg_data: dict) -> None:
    """Single-panel grouped bar chart: C12 per modality pair, averaged across benchmarks."""
    models = [m for m in MODEL_DISPLAY_ORDER if m in agg_data]
    model_labels = [MODEL_DISPLAY_NAMES.get(m, m) for m in models]
    pairs = [p for p in MODALITY_PAIR_ORDER
             if any(p in agg_data[m] for m in models)]

    fig = go.Figure()
    for pair in pairs:
        label = PAIR_LABELS_FULL.get(pair, f"{pair[0]}+{pair[1]}")
        color = PAIR_COLORS.get(pair, "#888888")
        values = [agg_data[m].get(pair, None) for m in models]
        fig.add_trace(go.Bar(
            x=model_labels,
            y=values,
            name=label,
            marker_color=color,
            text=[f"{v:.2f}" if v is not None and not np.isnan(v) else "" for v in values],
            textposition="outside",
            textfont=dict(size=10),
        ))

    fig.update_layout(
        font=dict(family="Helvetica, Arial, sans-serif", size=18),
        plot_bgcolor="#FAFAFA",
        paper_bgcolor="white",
        margin=dict(l=80, r=20, t=40, b=100),
        bargap=0.25,
        bargroupgap=0.06,
        barmode="group",
        yaxis_title="Mean Cooperation score (C<sub>12</sub>)",
        width=950,
        height=540,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=16),
            itemwidth=40,
        ),
        yaxis=dict(
            title=dict(font=dict(size=19)),
            tickfont=dict(size=16),
            zeroline=True,
            zerolinecolor="#BBBBBB",
            zerolinewidth=1,
            gridcolor="#E0E0E0",
            gridwidth=0.5,
        ),
        xaxis=dict(
            tickangle=-25,
            tickfont=dict(size=16),
        ),
    )
    _save(fig, "aggregated_C12_across_benchmarks.png")


def plot_metric_across_benchmarks(
    benchmark_data: dict,
    metric_key: str,
    y_label: str,
    title: str,
    output_filename: str,
) -> None:
    """Side-by-side subplots, one per benchmark."""
    bench_names = list(benchmark_data.keys())
    n = len(bench_names)

    fig = make_subplots(
        rows=1, cols=n,
        shared_yaxes=True,
        horizontal_spacing=0.06,
        subplot_titles=bench_names,
    )

    for col_idx, bench_name in enumerate(bench_names, start=1):
        model_data = benchmark_data[bench_name]
        models = get_ordered_models(model_data)
        model_labels = [MODEL_DISPLAY_NAMES.get(m, m) for m in models]

        for modality in MODALITIES:
            values = [model_data[m][modality][metric_key] for m in models]
            values = [v if not np.isnan(v) else None for v in values]
            show_legend = col_idx == 1
            fig.add_trace(
                go.Bar(
                    x=model_labels,
                    y=values,
                    name=MODALITY_LABELS[modality],
                    marker_color=MODALITY_COLORS[modality],
                    showlegend=show_legend,
                    legendgroup=modality,
                ),
                row=1, col=col_idx,
            )

    fig.update_layout(
        font=dict(family="Helvetica, Arial, sans-serif", size=18),
        plot_bgcolor="#FAFAFA",
        paper_bgcolor="white",
        margin=dict(l=80, r=20, t=60, b=100),
        bargap=0.25,
        bargroupgap=0.06,
        barmode="group",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=16),
        ),
        width=650 * n,
        height=540,
    )

    for i in range(1, n + 1):
        fig.update_xaxes(tickangle=-25, row=1, col=i)
        fig.update_yaxes(
            zeroline=True, zerolinecolor="#BBBBBB", zerolinewidth=1,
            gridcolor="#E0E0E0", gridwidth=0.5,
            row=1, col=i,
        )
    fig.update_yaxes(title_text=y_label, row=1, col=1)

    _save(fig, output_filename)


def _safe_bench_slug(bench_name: str) -> str:
    """Turn a benchmark name into a filename-safe slug."""
    return bench_name.replace(" ", "_").replace("/", "_").replace("-", "_")


def plot_per_benchmark_s_horizontal(
    benchmark_data: dict,
) -> None:
    """One standard (vertical bars) S plot per benchmark, matching aggregated S style."""
    for bench_name, model_data in benchmark_data.items():
        models = [m for m in MODEL_DISPLAY_ORDER if m in model_data]
        model_labels = [MODEL_DISPLAY_NAMES.get(m, m) for m in models]

        fig = go.Figure()
        for modality in MODALITIES:
            values = [
                model_data[m][modality]["S_mean"]
                if not np.isnan(model_data[m][modality]["S_mean"]) else None
                for m in models
            ]
            fig.add_trace(go.Bar(
                x=model_labels,
                y=values,
                name=MODALITY_LABELS[modality],
                marker_color=MODALITY_COLORS[modality],
                text=[f"{v:.2f}" if v is not None else "" for v in values],
                textposition="outside",
                textfont=dict(size=12),
            ))

        fig.update_layout(
            font=dict(family="Helvetica, Arial, sans-serif", size=18),
            plot_bgcolor="#FAFAFA",
            paper_bgcolor="white",
            margin=dict(l=80, r=20, t=40, b=100),
            bargap=0.25,
            bargroupgap=0.06,
            barmode="group",
            yaxis_title="Mean SHAPE score (S)",
            width=900,
            height=540,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(0,0,0,0)",
                font=dict(size=14),
                itemwidth=40,
            ),
            yaxis=dict(
                title=dict(font=dict(size=19)),
                tickfont=dict(size=16),
                zeroline=True,
                zerolinecolor="#BBBBBB",
                zerolinewidth=1,
                gridcolor="#E0E0E0",
                gridwidth=0.5,
            ),
            xaxis=dict(
                tickangle=-25,
                tickfont=dict(size=16),
            ),
        )
        slug = _safe_bench_slug(bench_name)
        _save(fig, f"{slug}_S_horizontal.png")


def plot_per_benchmark_c12_horizontal(
    benchmark_c12: dict,
) -> None:
    """One standard (vertical bars) C12 plot per benchmark, matching aggregated C12 style."""
    for bench_name, model_data in benchmark_c12.items():
        models = [m for m in MODEL_DISPLAY_ORDER if m in model_data]
        model_labels = [MODEL_DISPLAY_NAMES.get(m, m) for m in models]
        pairs = [p for p in MODALITY_PAIR_ORDER
                 if any(p in model_data[m] for m in models)]

        fig = go.Figure()
        for pair in pairs:
            label = PAIR_LABELS_FULL.get(pair, f"{pair[0]}+{pair[1]}")
            color = PAIR_COLORS.get(pair, "#888888")
            values = [model_data[m].get(pair, None) for m in models]
            fig.add_trace(go.Bar(
                x=model_labels,
                y=values,
                name=label,
                marker_color=color,
                text=[f"{v:.2f}" if v is not None else "" for v in values],
                textposition="outside",
                textfont=dict(size=10),
            ))

        fig.update_layout(
            font=dict(family="Helvetica, Arial, sans-serif", size=18),
            plot_bgcolor="#FAFAFA",
            paper_bgcolor="white",
            margin=dict(l=80, r=20, t=40, b=100),
            bargap=0.25,
            bargroupgap=0.06,
            barmode="group",
            yaxis_title="Mean Cooperation score (C<sub>12</sub>)",
            width=950,
            height=540,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(0,0,0,0)",
                font=dict(size=16),
                itemwidth=40,
            ),
            yaxis=dict(
                title=dict(font=dict(size=19)),
                tickfont=dict(size=16),
                zeroline=True,
                zerolinecolor="#BBBBBB",
                zerolinewidth=1,
                gridcolor="#E0E0E0",
                gridwidth=0.5,
            ),
            xaxis=dict(
                tickangle=-25,
                tickfont=dict(size=16),
            ),
        )
        slug = _safe_bench_slug(bench_name)
        _save(fig, f"{slug}_C12_horizontal.png")


def _save_inverted(fig: go.Figure, filename: str) -> None:
    """Write a Plotly figure to PNG."""
    out = OUTPUT_DIR / filename
    fig.write_image(str(out), scale=3)
    print(f"Saved: {out}")


# ── Inverted plots (x-axis = modality, legend = model) ───────────────


def plot_aggregated_s_by_modality(agg_data: dict) -> None:
    """Horizontal: x-axis = modalities, bars = models."""
    models = [m for m in MODEL_DISPLAY_ORDER if m in agg_data]
    modality_labels = [MODALITY_LABELS[mod] for mod in MODALITIES]

    fig = go.Figure()
    for model in models:
        label = MODEL_DISPLAY_NAMES.get(model, model)
        color = MODEL_COLORS.get(model, "#888888")
        values = [agg_data[model].get(mod, None) for mod in MODALITIES]
        fig.add_trace(go.Bar(
            x=modality_labels,
            y=values,
            name=label,
            marker_color=color,
            text=[f"{v:.2f}" if v is not None and not np.isnan(v) else "" for v in values],
            textposition="outside",
            textfont=dict(size=12),
        ))

    fig.update_layout(
        font=dict(family="Helvetica, Arial, sans-serif", size=18),
        plot_bgcolor="#FAFAFA",
        paper_bgcolor="white",
        margin=dict(l=80, r=20, t=40, b=20),
        bargap=0.25,
        bargroupgap=0.06,
        barmode="group",
        yaxis_title="S<sub>m<sub>1</sub></sub>",
        width=800,
        height=440,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=14),
            itemwidth=40,
        ),
        yaxis=dict(
            title=dict(font=dict(size=22)),
            tickfont=dict(size=16),
            zeroline=True,
            zerolinecolor="#BBBBBB",
            zerolinewidth=1,
            gridcolor="#E0E0E0",
            gridwidth=0.5,
        ),
        xaxis=dict(tickfont=dict(size=16)),
    )
    _save_inverted(fig, "aggregated_S_across_benchmarks.png")


def plot_aggregated_c12_by_modality(agg_data: dict) -> None:
    """Horizontal: x-axis = modality pairs, bars = models."""
    models = [m for m in MODEL_DISPLAY_ORDER if m in agg_data]
    pairs = [p for p in MODALITY_PAIR_ORDER
             if any(p in agg_data[m] for m in models)]
    pair_labels = [PAIR_LABELS_FULL.get(p, f"{p[0]}+{p[1]}") for p in pairs]

    fig = go.Figure()
    for model in models:
        label = MODEL_DISPLAY_NAMES.get(model, model)
        color = MODEL_COLORS.get(model, "#888888")
        values = [agg_data[model].get(p, None) for p in pairs]
        fig.add_trace(go.Bar(
            x=pair_labels,
            y=values,
            name=label,
            marker_color=color,
            text=[f"{v:.2f}" if v is not None and not np.isnan(v) else "" for v in values],
            textposition="outside",
            textfont=dict(size=10),
        ))

    fig.update_layout(
        font=dict(family="Helvetica, Arial, sans-serif", size=18),
        plot_bgcolor="#FAFAFA",
        paper_bgcolor="white",
        margin=dict(l=80, r=20, t=40, b=100),
        bargap=0.25,
        bargroupgap=0.06,
        barmode="group",
        yaxis_title="C<sub>{m<sub>1</sub>,m<sub>2</sub>}</sub>",
        width=950,
        height=540,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=16),
            itemwidth=40,
        ),
        yaxis=dict(
            title=dict(font=dict(size=22)),
            tickfont=dict(size=16),
            zeroline=True,
            zerolinecolor="#BBBBBB",
            zerolinewidth=1,
            gridcolor="#E0E0E0",
            gridwidth=0.5,
        ),
        xaxis=dict(
            tickangle=-25,
            tickfont=dict(size=16),
        ),
    )
    _save_inverted(fig, "aggregated_C12_across_benchmarks.png")


def plot_per_benchmark_s_by_modality(benchmark_data: dict) -> None:
    """Per-benchmark horizontal: x-axis = modalities, bars = models."""
    for bench_name, model_data in benchmark_data.items():
        models = [m for m in MODEL_DISPLAY_ORDER if m in model_data]
        modality_labels = [MODALITY_LABELS[mod] for mod in MODALITIES]

        fig = go.Figure()
        for model in models:
            label = MODEL_DISPLAY_NAMES.get(model, model)
            color = MODEL_COLORS.get(model, "#888888")
            values = [
                model_data[model][mod]["S_mean"]
                if not np.isnan(model_data[model][mod]["S_mean"]) else None
                for mod in MODALITIES
            ]
            fig.add_trace(go.Bar(
                x=modality_labels,
                y=values,
                name=label,
                marker_color=color,
                text=[f"{v:.2f}" if v is not None else "" for v in values],
                textposition="outside",
                textfont=dict(size=12),
            ))

        fig.update_layout(
            font=dict(family="Helvetica, Arial, sans-serif", size=18),
            plot_bgcolor="#FAFAFA",
            paper_bgcolor="white",
            margin=dict(l=80, r=20, t=40, b=70),
            bargap=0.25,
            bargroupgap=0.06,
            barmode="group",
            yaxis_title="S<sub>m<sub>1</sub></sub>",
            width=800,
            height=440,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(0,0,0,0)",
                font=dict(size=14),
                itemwidth=40,
            ),
            yaxis=dict(
                title=dict(font=dict(size=22)),
                tickfont=dict(size=16),
                zeroline=True,
                zerolinecolor="#BBBBBB",
                zerolinewidth=1,
                gridcolor="#E0E0E0",
                gridwidth=0.5,
            ),
            xaxis=dict(tickfont=dict(size=16)),
        )
        slug = _safe_bench_slug(bench_name)
        _save_inverted(fig, f"{slug}_S_by_modality.png")


def plot_per_benchmark_c12_by_modality(benchmark_c12: dict) -> None:
    """Per-benchmark horizontal: x-axis = modality pairs, bars = models."""
    for bench_name, model_data in benchmark_c12.items():
        models = [m for m in MODEL_DISPLAY_ORDER if m in model_data]
        pairs = [p for p in MODALITY_PAIR_ORDER
                 if any(p in model_data[m] for m in models)]
        pair_labels = [PAIR_LABELS_FULL.get(p, f"{p[0]}+{p[1]}") for p in pairs]

        fig = go.Figure()
        for model in models:
            label = MODEL_DISPLAY_NAMES.get(model, model)
            color = MODEL_COLORS.get(model, "#888888")
            values = [model_data[model].get(p, None) for p in pairs]
            fig.add_trace(go.Bar(
                x=pair_labels,
                y=values,
                name=label,
                marker_color=color,
                text=[f"{v:.2f}" if v is not None else "" for v in values],
                textposition="outside",
                textfont=dict(size=10),
            ))

        fig.update_layout(
            font=dict(family="Helvetica, Arial, sans-serif", size=18),
            plot_bgcolor="#FAFAFA",
            paper_bgcolor="white",
            margin=dict(l=80, r=20, t=40, b=100),
            bargap=0.25,
            bargroupgap=0.06,
            barmode="group",
            yaxis_title="C<sub>{m<sub>1</sub>,m<sub>2</sub>}</sub>",
            width=950,
            height=540,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(0,0,0,0)",
                font=dict(size=16),
                itemwidth=40,
            ),
            yaxis=dict(
                title=dict(font=dict(size=22)),
                tickfont=dict(size=16),
                zeroline=True,
                zerolinecolor="#BBBBBB",
                zerolinewidth=1,
                gridcolor="#E0E0E0",
                gridwidth=0.5,
            ),
            xaxis=dict(
                tickangle=-25,
                tickfont=dict(size=16),
            ),
        )
        slug = _safe_bench_slug(bench_name)
        _save_inverted(fig, f"{slug}_C12_by_modality.png")


def main() -> None:
    benchmark_data = {}   # {bench: {model: {modality: {...}}}}
    benchmark_c12 = {}    # {bench: {model: {(m1,m2): C12}}}

    for bench_name, (results_dir, scorer_fn) in BENCHMARKS.items():
        if not results_dir.exists():
            print(f"Skipping {bench_name}: directory not found: {results_dir}")
            continue

        data = collect_model_modality_contributions(results_dir, scorer_fn)
        c12_data = collect_model_pair_c12(results_dir, scorer_fn)
        if not data:
            print(f"Skipping {bench_name}: no *_results_vlm.json files found")
            continue

        benchmark_data[bench_name] = data
        benchmark_c12[bench_name] = c12_data

    if not benchmark_data:
        raise RuntimeError("No benchmark data available to plot.")

    # ── Aggregated (cross-benchmark) plots ────────────────────────────────
    agg_s = aggregate_modality_scores(benchmark_data)
    agg_c12 = aggregate_c12_scores(benchmark_c12)

    # ── By-modality plots (x = modality/pair, legend = model) ─────────
    plot_aggregated_s_by_modality(agg_s)
    plot_aggregated_c12_by_modality(agg_c12)
    plot_per_benchmark_s_by_modality(benchmark_data)
    plot_per_benchmark_c12_by_modality(benchmark_c12)


if __name__ == "__main__":
    main()
