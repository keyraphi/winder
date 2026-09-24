#!/usr/bin/env python3
"""Plot metrics from a Winder batch-evaluation JSONL file.

The script inspects the JSONL file and automatically produces every plot
that makes sense for the data it finds:

  * Forward winding-number metrics   -> speedup, timing, accuracy,
                                        misclassification, trade-off
  * Backward gradient metrics        -> gradient quality, per-primitive
                                        angular and magnitude distributions
  * Mixed files (multiple modes)     -> the union, one set per mode

Modes present in a nested-schema JSONL (records with a "modes" key) are
detected automatically. Legacy flat-schema records are treated as
forward_triangle.

Usage:
    python plot_results.py --input results.jsonl --mode display
    python plot_results.py --input results.jsonl --mode paper --output ./figs
    python plot_results.py --input results.jsonl --modes backward_triangle
    python plot_results.py --input results.jsonl --figures speedup,accuracy
"""

import argparse
import json
import math
import textwrap
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ============================================================================
# Constants
# ============================================================================

# All numeric keys that may appear in a record. Missing keys become NaN.
NUMERIC_KEYS = [
    # Forward metrics
    "signal_max",
    "mean_abs",
    "p50_abs",
    "p95_abs",
    "p99_abs",
    "max_abs",
    "rms_rel_masked",
    "rms_abs",
    "p99_rel_masked",
    "misclass_count",
    "misclass_frac",
    "time_brute_ms",
    "time_fast_ms",
    "speedup",
    # Backward scalar
    "global_cosine",
    "global_rel_norm",
    # Backward per-vector (cosine)
    "per_vec_cos_mean",
    "per_vec_cos_std",
    "per_vec_cos_min",
    "per_vec_cos_p10",
    "per_vec_cos_p25",
    "per_vec_cos_p50",
    "per_vec_cos_p75",
    "per_vec_cos_p90",
    "per_vec_cos_p95",
    "per_vec_cos_p99",
    "per_vec_cos_p99_9",
    # Backward per-vector (angular error, degrees)
    "per_vec_ang_mean",
    "per_vec_ang_std",
    "per_vec_ang_max",
    "per_vec_ang_p10",
    "per_vec_ang_p25",
    "per_vec_ang_p50",
    "per_vec_ang_p75",
    "per_vec_ang_p90",
    "per_vec_ang_p95",
    "per_vec_ang_p99",
    "per_vec_ang_p99_9",
    # Backward per-vector (magnitude ratio)
    "mag_ratio_mean",
    "mag_ratio_std",
    "mag_ratio_p10",
    "mag_ratio_p25",
    "mag_ratio_p50",
    "mag_ratio_p75",
    "mag_ratio_p90",
    "mag_ratio_p95",
    "mag_ratio_p99",
    "mag_ratio_p99_9",
]

PALETTE = {
    "primary": "#1f77b4",
    "secondary": "#d62728",
    "tertiary": "#2ca02c",
    "quaternary": "#9467bd",
    "accent": "#ff7f0e",
    "neutral": "#7f7f7f",
    "grid": "#cccccc",
    "highlight": "#e41a1c",
    "dark": "#333333",
}


# ============================================================================
# Data loading
# ============================================================================


def load_records_by_mode(path: Path) -> dict:
    """Return {mode_name: [flat_records]}.

    Handles both the legacy flat schema (all metrics at top level, assumed
    to be forward_triangle) and the extended schema (metrics nested under
    ``record["modes"][mode_name]``).
    """
    by_mode: dict = defaultdict(list)
    bad = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if "error" in rec:
                continue
            if "modes" in rec and isinstance(rec["modes"], dict):
                common = {k: v for k, v in rec.items() if k != "modes"}
                for mode_name, mode_metrics in rec["modes"].items():
                    if not isinstance(mode_metrics, dict):
                        continue
                    if "error" in mode_metrics:
                        continue
                    # Split optional "@eps=<value>" suffix from the mode name.
                    if "@eps=" in mode_name:
                        base_mode, eps_str = mode_name.split("@eps=", 1)
                        try:
                            eps_val = float(eps_str)
                        except ValueError:
                            eps_val = float("nan")
                    else:
                        base_mode = mode_name
                        eps_val = float(mode_metrics.get("epsilon", float("nan")))
                    flat = {
                        **common,
                        **mode_metrics,
                        "_mode_name": mode_name,
                        "_base_mode": base_mode,
                        "_epsilon": eps_val,
                    }
                    by_mode[mode_name].append(flat)
            else:
                by_mode["forward_triangle"].append(rec)

    if bad:
        print(f"  Note: {bad} malformed lines skipped")
    return dict(by_mode)


def to_arrays(records: list) -> dict:
    keys = ["n_tri", "n_queries"] + NUMERIC_KEYS
    data = {k: [] for k in keys}
    for rec in records:
        for k in keys:
            v = rec.get(k, float("nan"))
            try:
                v = float(v)
            except (TypeError, ValueError):
                v = float("nan")
            data[k].append(v)
    return {k: np.asarray(v, dtype=np.float64) for k, v in data.items()}


# ============================================================================
# Bucket data (distance-to-surface attribution)
# ============================================================================


def load_bucket_data(path: Path) -> dict:
    """Return {mode_name: {bucket_label: [metrics_dict, ...]}}.

    Buckets are stored per mode as ``record['modes'][m]['buckets']``. Missing
    or disabled bucketing yields an empty dict.
    """
    out: dict = defaultdict(lambda: defaultdict(list))
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" in rec or "modes" not in rec:
                continue
            for mode_name, mm in rec["modes"].items():
                if not isinstance(mm, dict) or "error" in mm:
                    continue
                buckets = mm.get("buckets")
                if not isinstance(buckets, dict):
                    continue
                for label, bm in buckets.items():
                    if not isinstance(bm, dict) or "error" in bm:
                        continue
                    if bm.get("empty"):
                        continue
                    out[mode_name][label].append(bm)
    return {m: dict(v) for m, v in out.items()}


def _bucket_sort_key(lbl: str) -> float:
    """Sortable key for bucket labels produced by eval_pipeline._bucket_label.

    Formats handled:
      d_lt_<hi>        -> 0.0
      d_<lo>_<hi>      -> float(<lo>)
      d_ge_<lo>        -> float(<lo>)
    """
    body = lbl[2:] if lbl.startswith("d_") else lbl
    if body.startswith("lt_"):
        return 0.0
    if body.startswith("ge_"):
        return float(body[3:])
    # d_<lo>_<hi>: split on the underscore separating the two floats.
    # Both halves are plain Python float literals like "1e-03".
    head = body.split("_", 1)[0]
    try:
        return float(head)
    except ValueError:
        return 0.0


# ============================================================================
# Style
# ============================================================================


def setup_style(mode: str):
    if mode == "paper":
        mpl.rcParams.update(
            {
                "font.family": "serif",
                "font.serif": ["DejaVu Serif"],
                "font.size": 9,
                "axes.labelsize": 10,
                "axes.titlesize": 10,
                "xtick.labelsize": 8,
                "ytick.labelsize": 8,
                "legend.fontsize": 8,
                "figure.dpi": 150,
                "savefig.dpi": 300,
                "savefig.bbox": "tight",
                "savefig.pad_inches": 0.02,
                "axes.grid": True,
                "axes.axisbelow": True,
                "grid.alpha": 0.3,
                "grid.linewidth": 0.4,
                "grid.color": PALETTE["grid"],
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.linewidth": 0.6,
                "lines.linewidth": 1.0,
                "lines.markersize": 2.5,
                "legend.frameon": False,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
                "text.usetex": False,
            }
        )
    else:
        mpl.rcParams.update(
            {
                "font.family": "sans-serif",
                "font.size": 10,
                "axes.labelsize": 11,
                "axes.titlesize": 12,
                "legend.fontsize": 9,
                "figure.dpi": 110,
                "savefig.dpi": 130,
                "axes.grid": True,
                "axes.axisbelow": True,
                "grid.alpha": 0.25,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "lines.linewidth": 1.4,
                "lines.markersize": 5,
                "legend.frameon": True,
                "legend.framealpha": 0.9,
                "text.usetex": False,
            }
        )


def figsize(mode: str, kind: str):
    paper = {
        "wide": (7.0, 2.3),
        "tall": (3.4, 6.0),
        "grid22": (7.0, 5.0),
        "grid23": (7.0, 4.5),
        "grid32": (7.0, 6.5),
        "square": (3.6, 3.6),
    }
    display = {
        "wide": (14, 4.5),
        "tall": (6, 10),
        "grid22": (12, 9),
        "grid23": (15, 8.5),
        "grid32": (12, 13),
        "square": (7, 6.5),
    }
    return (paper if mode == "paper" else display)[kind]


def save_or_show(fig, name, mode, output_dir, prefix: str = ""):
    output_dir = Path(output_dir or ".")
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{prefix}_{name}" if prefix else name
    if mode == "paper":
        path = output_dir / f"{stem}.pdf"
        fig.savefig(path, format="pdf")
    else:
        path = output_dir / f"{stem}.png"
        fig.savefig(path, format="png", bbox_inches="tight")
    print(f"    wrote {path}")
    if mode == "display" and mpl.get_backend().lower() not in (
        "agg",
        "pdf",
        "ps",
        "svg",
        "cairo",
    ):
        plt.show()
    plt.close(fig)


# ============================================================================
# Style helpers
# ============================================================================


def _finite_positive(*arrays):
    mask = np.ones_like(arrays[0], dtype=bool)
    for a in arrays:
        mask &= np.isfinite(a) & (a > 0)
    return mask


def _finite(*arrays):
    mask = np.ones_like(arrays[0], dtype=bool)
    for a in arrays:
        mask &= np.isfinite(a)
    return mask


def finalize_axes(
    ax, *, xlabel=None, ylabel=None, title=None, legend_loc=None, legend_ncol=1
):
    """Apply consistent professional styling to an axis."""
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    ax.set_axisbelow(True)
    ax.tick_params(which="both", direction="out", length=3, width=0.6, pad=2)
    if legend_loc is not None:
        leg = ax.legend(loc=legend_loc, ncol=legend_ncol)
        for h in leg.legend_handles:
            # Enlarge markers in the legend so they are visible
            if hasattr(h, "set_markersize"):
                try:
                    ms = h.get_markersize()
                    h.set_markersize(max(ms, 6.5))
                except Exception:
                    pass
            if hasattr(h, "set_markerscale"):
                try:
                    h.set_markerscale(2.0)
                except Exception:
                    pass
            # Make legend markers more opaque
            if hasattr(h, "set_alpha"):
                try:
                    h.set_alpha(0.95)
                except Exception:
                    pass
        return leg
    return None


def scatter_legend_handle(color, label, alpha=0.85, size=7, marker="o"):
    """Return a proxy handle for scatter points with visible legend appearance."""
    return Line2D(
        [],
        [],
        linestyle="none",
        marker=marker,
        markersize=size,
        markerfacecolor=color,
        markeredgecolor="none",
        alpha=alpha,
        label=label,
    )


def line_legend_handle(color, label, lw=1.6, ls="-"):
    return Line2D([], [], color=color, linewidth=lw, linestyle=ls, label=label)


def format_log_axis(ax, which="both"):
    for a in (
        [ax.xaxis, ax.yaxis]
        if which == "both"
        else [ax.xaxis]
        if which == "x"
        else [ax.yaxis]
    ):
        a.set_major_locator(mpl.ticker.LogLocator(base=10, numticks=10))
        a.set_minor_locator(
            mpl.ticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=10)
        )
        a.set_minor_formatter(mpl.ticker.NullFormatter())


def log_log_fit(x, y):
    m = _finite_positive(x, y)
    if m.sum() < 3:
        return None
    lx = np.log10(x[m])
    ly = np.log10(y[m])
    a, b = np.polyfit(lx, ly, 1)
    return float(a), float(b)


def binned_median(x, y, n_bins=24, log_x=True):
    m = _finite(x, y) & (x > 0 if log_x else np.ones_like(x, dtype=bool))
    x, y = x[m], y[m]
    if x.size < 4:
        return (np.array([]),) * 4 + (np.array([], dtype=int),)
    lo, hi = x.min(), x.max()
    if hi <= lo:
        return (np.array([]),) * 4 + (np.array([], dtype=int),)
    edges = (
        np.logspace(np.log10(lo), np.log10(hi), n_bins + 1)
        if log_x
        else np.linspace(lo, hi, n_bins + 1)
    )
    c, med, p25, p75, cnt = [], [], [], [], []
    for i in range(n_bins):
        in_bin = (x >= edges[i]) & (x < edges[i + 1])
        if in_bin.sum() >= 2:
            c.append(
                math.sqrt(edges[i] * edges[i + 1])
                if log_x
                else 0.5 * (edges[i] + edges[i + 1])
            )
            med.append(np.median(y[in_bin]))
            p25.append(np.percentile(y[in_bin], 25))
            p75.append(np.percentile(y[in_bin], 75))
            cnt.append(int(in_bin.sum()))
    return (
        np.array(c),
        np.array(med),
        np.array(p25),
        np.array(p75),
        np.array(cnt, dtype=int),
    )


def ecdf(x):
    m = _finite(x)
    x = np.sort(x[m])
    y = np.arange(1, x.size + 1) / x.size
    return x, y


def pareto_front(x, y, x_maximize=True, y_minimize=True):
    order = np.argsort(-x if x_maximize else x)
    best = np.inf if y_minimize else -np.inf
    keep = []
    for i in order:
        v = y[i]
        better = (v < best) if y_minimize else (v > best)
        if better:
            best = v
            keep.append(i)
    return np.array(keep)


def add_identity_line(ax, x, y):
    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    ax.plot(
        [lo, hi],
        [lo, hi],
        color=PALETTE["neutral"],
        linewidth=0.9,
        linestyle="--",
        zorder=1,
    )


# ============================================================================
# Hints — printed for every plot
# ============================================================================

HINTS = {
    "overview": "Six-panel summary for forward winding-number evaluation. "
    "(a) Speedup distribution; median is the typical user experience, "
    "tail is best case. (b) RMS absolute error distribution; lower is "
    "better, and the median is comparable to the FWN paper's reported "
    "8e-3. (c) Misclassifications per mesh; 0 is the common case. "
    "(d) Speedup vs mesh complexity; larger meshes should see more "
    "speedup. (e) Brute-force vs fast timing; points below the diagonal "
    "mean the fast engine wins. (f) Joint trade-off between speedup "
    "and accuracy.",
    "timing": "(a,b) Scaling of brute-force and fast-engine runtimes with mesh "
    "complexity, with log-log slope fits. Slope near 1 indicates the "
    "runtime is linear in mesh size; slope < 1 indicates sublinear "
    "growth. (c) Speedup vs mesh complexity: the fast engine's "
    "advantage grows with the mesh, so the trend should rise.",
    "speedup": "(a) Distribution of per-mesh speedup values. (b) ECDF with "
    "quantile markers: read off the fraction of meshes that beat any "
    "given speedup threshold. (c) Fraction of meshes faster than a "
    "given speedup; useful for picking an operating point.",
    "accuracy": "(a) Overlaid histograms of per-mesh error percentiles (p50, p95, "
    "p99, max). The left shift from p50 to max shows the tail "
    "behaviour. (b) ECDFs: what fraction of meshes stay below a given "
    "error. (c) RMS distribution with mean and median; a large gap "
    "indicates a heavy right tail. (d) Masked relative error; only "
    "queries with |winding| > 1e-3 contribute, avoiding the "
    "near-zero-division artifact.",
    "gradient_quality": "(a) ECDFs of the per-primitive angular error distribution: each "
    "curve is an item-wise percentile (p50/p90/p99/max) across meshes. "
    "The median item's median primitive is off by the leftmost curve's "
    "median. (b) Per-mesh median angular error: how far off the "
    "typical primitive gradient points. (c) Magnitude ratio "
    "||fast|| / ||ref||; 1.0 is perfect, systematic deviations from "
    "1.0 indicate a scaling issue. (d) Sanity check: if global "
    "cosine is close to 1.0 while per-primitive p99 is lower, the "
    "global metric is being carried by the largest entries.",
    "error_scaling": "(a) Absolute error percentiles vs mesh complexity. Flat curves "
    "mean the error does not grow with mesh size; upward curves mean "
    "it does. (b) Error vs the brute-force cost: shows whether more "
    "compute is needed to reach a given accuracy. (c) Joint density "
    "of error over (complexity, runtime); look for vertical bands.",
    "misclassification": "(a) Histogram of misclassified voxels per mesh. The '0' bar is "
    "the primary quality metric: the fraction of meshes with zero "
    "misclassifications should match or exceed the FWN paper's "
    "reported >50%. (b) ECDF of the misclassification fraction: "
    "fraction of meshes below any threshold. (c) Misclassifications "
    "vs mesh complexity; a rising trend indicates the algorithm "
    "struggles on more complex inputs.",
    "tradeoff": "(a) Scatter of speedup vs error, colored by mesh complexity, "
    "with the Pareto frontier highlighted. Points on the frontier "
    "cannot be improved on either axis without regressing the other. "
    "(b) Distribution of the joint quality metric speedup / error. "
    "Higher is better; the median is a summary statistic.",
    "paper_dashboard": "Compact 2x2 summary suitable for a paper's results section. "
    "Panels: (a) speedup ECDF with median marker, (b) RMS error ECDF, "
    "(c) speedup vs complexity with IQR band, (d) misclassification "
    "bin counts with percentages.",
    "buckets": (
        "Per-distance-bucket attribution. Each backward mode is re-run with a "
        "grad_output masked to queries in one distance-to-surface bucket, so "
        "the resulting gradient is exactly that bucket's contribution. Panel "
        "(a): direction agreement by bucket — should be near 1 for far buckets "
        "and degrade monotonically as distance shrinks. Panel (b): per-vector "
        "angular error p99 by bucket, in degrees. Panel (c): magnitude ratio "
        "by bucket; systematic < 1 for far buckets indicates Barnes-Hut "
        "truncation bias, which grows with the effective β. Panel (d): "
        "fraction of the total reference gradient norm carried by each bucket; "
        "typically the near-surface bucket dominates, which is why global "
        "norm-based metrics are dominated by near-surface queries."
    ),
    "epsilon_sweep": (
        "Fast-vs-brute gradient error, direction agreement, and speedup as "
        "functions of the regularization strength epsilon. One line per "
        "mode. Read the shape: (a) usually grows roughly linearly with "
        "epsilon up to the point where the smoothing radius matches the "
        "mesh feature scale, after which it plateaus. (b) global cosine "
        "should stay above ~0.999 for moderate epsilon. (c) angular error "
        "p99 quantifies the worst-case direction disagreement at each "
        "epsilon. (d) speedup is largely independent of epsilon, since "
        "the traversal cost is set by beta, not by the kernel width."
    ),
}


def print_hint(name: str, mode_name: str):
    hint = HINTS.get(name)
    if hint is None:
        return
    width = 72
    print(f"\n  {'┌' + '─' * width}")
    header = f"[{name}]  mode={mode_name}"
    print(f"  │ {header}")
    print(f"  │")
    for line in textwrap.wrap(hint, width=width - 4):
        print(f"  │   {line}")
    print(f"  {'└' + '─' * width}")


# ============================================================================
# Summary
# ============================================================================


def print_summary(data, dataset_name, mode_name):
    def _stat(name, a, fmt=".3e"):
        m = _finite(a)
        if m.sum() == 0:
            return f"    {name:<22} (no data)"
        s = a[m]
        return (
            f"    {name:<22} n={len(s):<5d}  "
            f"min={s.min():{fmt}}  med={np.median(s):{fmt}}  "
            f"mean={s.mean():{fmt}}  max={s.max():{fmt}}"
        )

    n = len(data["n_tri"])
    print(f"\n=== {dataset_name}  [{mode_name}] ===")
    print(f"  Items            : {n}")
    print(f"  Total primitives : {int(data['n_tri'].sum()):,}")
    print(f"  Total queries    : {int(data['n_queries'].sum()):,}")
    print()

    common = [
        ("n_tri", "n_tri"),
        ("brute_ms", "time_brute_ms"),
        ("fast_ms", "time_fast_ms"),
        ("speedup", "speedup"),
        ("rms_abs", "rms_abs"),
        ("p50_abs", "p50_abs"),
        ("p99_abs", "p99_abs"),
        ("max_abs", "max_abs"),
    ]
    for label, key in common:
        print(_stat(label, data[key]))

    if _has_data(data, "misclass_count"):
        print(_stat("misclass_count", data["misclass_count"]))
        print(_stat("misclass_frac", data["misclass_frac"]))

    if _has_data(data, "global_cosine"):
        print(_stat("global_cosine", data["global_cosine"], ".6f"))
        print(_stat("global_rel_norm", data["global_rel_norm"]))
        print(_stat("per_vec_ang_p50", data["per_vec_ang_p50"], ".4f"))
        print(_stat("per_vec_ang_p99", data["per_vec_ang_p99"], ".4f"))
        print(_stat("mag_ratio_p50", data["mag_ratio_p50"], ".6f"))
    print()


# ============================================================================
# Capability detection
# ============================================================================


def _has_data(data, key, min_count: int = 3) -> bool:
    """True if ``data[key]`` exists with at least ``min_count`` finite values."""
    v = data.get(key)
    if v is None:
        return False
    return int(np.isfinite(v).sum()) >= min_count


def compute_capabilities(data: dict) -> dict:
    return {
        "timing": _has_data(data, "time_brute_ms")
        and _has_data(data, "time_fast_ms")
        and _has_data(data, "n_tri"),
        "speedup": _has_data(data, "speedup"),
        "accuracy": _has_data(data, "rms_abs"),
        "misclassification": _has_data(data, "misclass_count"),
        "gradient": (
            _has_data(data, "per_vec_ang_p50") and _has_data(data, "mag_ratio_p50")
        ),
        "global_gradient": _has_data(data, "global_cosine"),
    }


# ============================================================================
# Figures
# ============================================================================


def fig_overview(data, style_mode, label, out, prefix=""):
    fig, axes = plt.subplots(2, 3, figsize=figsize(style_mode, "grid23"))

    # (a) Speedup histogram
    ax = axes[0, 0]
    sp = data["speedup"]
    sp = sp[np.isfinite(sp) & (sp > 0)]
    if sp.size:
        bins = np.logspace(np.log10(sp.min()), np.log10(sp.max()), 40)
        ax.hist(sp, bins=bins, color=PALETTE["primary"], alpha=0.85, label="_hist")
        med = float(np.median(sp))
        ax.axvline(
            med,
            color=PALETTE["highlight"],
            lw=1.4,
            linestyle="--",
            label=f"median = {med:.2f}×",
        )
        ax.set_xscale("log")
        format_log_axis(ax, which="x")
        finalize_axes(
            ax,
            xlabel="Speedup (brute / fast)",
            ylabel="Item count",
            title="(a) Speedup distribution",
            legend_loc="upper right",
        )
    else:
        finalize_axes(
            ax, xlabel="Speedup", ylabel="Item count", title="(a) Speedup distribution"
        )

    # (b) RMS histogram
    ax = axes[0, 1]
    e = data["rms_abs"]
    e = e[np.isfinite(e) & (e > 0)]
    if e.size:
        bins = np.logspace(np.log10(e.min()), np.log10(e.max()), 40)
        ax.hist(e, bins=bins, color=PALETTE["secondary"], alpha=0.85)
        med = float(np.median(e))
        ax.axvline(
            med,
            color=PALETTE["highlight"],
            lw=1.4,
            linestyle="--",
            label=f"median = {med:.2e}",
        )
        ax.set_xscale("log")
        format_log_axis(ax, which="x")
        finalize_axes(
            ax,
            xlabel="RMS absolute error",
            ylabel="Item count",
            title="(b) RMS error distribution",
            legend_loc="upper right",
        )
    else:
        finalize_axes(
            ax,
            xlabel="RMS absolute error",
            ylabel="Item count",
            title="(b) RMS error distribution",
        )

    # (c) Misclassification bins
    ax = axes[0, 2]
    mc = data["misclass_count"]
    mc = mc[np.isfinite(mc)]
    if mc.size:
        edges = [0, 0.5, 5, 50, 500, np.inf]
        labels = ["0", "1–5", "6–50", "51–500", ">500"]
        colors = [
            PALETTE["tertiary"],
            PALETTE["primary"],
            PALETTE["accent"],
            PALETTE["quaternary"],
            PALETTE["secondary"],
        ]
        vals = [int(((mc >= edges[i]) & (mc < edges[i + 1])).sum()) for i in range(5)]
        ax.bar(range(5), vals, color=colors, alpha=0.9)
        ax.set_xticks(range(5))
        ax.set_xticklabels(labels)
        for i, v in enumerate(vals):
            if v:
                ax.text(i, v, str(v), ha="center", va="bottom", fontsize=8)
        ax.set_ylim(0, max(vals) * 1.15 if max(vals) else 1)
    finalize_axes(
        ax,
        xlabel="Misclassifications per item",
        ylabel="Item count",
        title="(c) Misclassification bins",
    )

    # (d) n_tri vs speedup
    ax = axes[1, 0]
    nt = data["n_tri"]
    m = _finite_positive(nt, data["speedup"])
    if m.sum():
        ax.scatter(
            nt[m],
            data["speedup"][m],
            s=4,
            alpha=0.25,
            color=PALETTE["primary"],
            edgecolors="none",
            label="_scatter",
        )
        c, med, _, _, _ = binned_median(nt, data["speedup"])
        handles = [scatter_legend_handle(PALETTE["primary"], "data")]
        if c.size:
            ax.plot(c, med, color=PALETTE["highlight"], lw=1.6)
            handles.append(line_legend_handle(PALETTE["highlight"], "binned median"))
        ax.axhline(1.0, color=PALETTE["neutral"], lw=0.9, linestyle=":")
        handles.append(line_legend_handle(PALETTE["neutral"], "parity", lw=0.9, ls=":"))
        ax.set_xscale("log")
        ax.set_yscale("log")
        format_log_axis(ax)
        finalize_axes(
            ax,
            xlabel="Triangle count",
            ylabel="Speedup",
            title="(d) Speedup vs complexity",
        )
        ax.legend(handles=handles, loc="upper left")
    else:
        finalize_axes(
            ax,
            xlabel="Triangle count",
            ylabel="Speedup",
            title="(d) Speedup vs complexity",
        )

    # (e) brute vs fast
    ax = axes[1, 1]
    b = data["time_brute_ms"]
    f = data["time_fast_ms"]
    m = _finite_positive(b, f)
    if m.sum():
        ax.scatter(
            b[m], f[m], s=4, alpha=0.25, color=PALETTE["primary"], edgecolors="none"
        )
        lo = float(min(b[m].min(), f[m].min()))
        hi = float(max(b[m].max(), f[m].max()))
        ax.plot([lo, hi], [lo, hi], color=PALETTE["neutral"], lw=0.9, linestyle="--")
        ax.set_xscale("log")
        ax.set_yscale("log")
        format_log_axis(ax)
        handles = [
            scatter_legend_handle(PALETTE["primary"], "per item"),
            line_legend_handle(PALETTE["neutral"], "y = x", lw=0.9, ls="--"),
        ]
        finalize_axes(
            ax,
            xlabel="Brute force time (ms)",
            ylabel="Fast engine time (ms)",
            title="(e) Brute force vs fast engine",
        )
        ax.legend(handles=handles, loc="upper left")
    else:
        finalize_axes(
            ax,
            xlabel="Brute force time (ms)",
            ylabel="Fast engine time (ms)",
            title="(e) Brute force vs fast engine",
        )

    # (f) speedup vs rms
    ax = axes[1, 2]
    m = _finite_positive(data["speedup"], data["rms_abs"])
    if m.sum():
        ax.scatter(
            data["speedup"][m],
            data["rms_abs"][m],
            s=4,
            alpha=0.25,
            color=PALETTE["primary"],
            edgecolors="none",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        format_log_axis(ax)
    finalize_axes(
        ax,
        xlabel="Speedup",
        ylabel="RMS absolute error",
        title="(f) Accuracy–performance trade-off",
    )

    fig.suptitle(label, y=1.005, fontsize=mpl.rcParams["font.size"] + 2)
    fig.tight_layout()
    save_or_show(fig, "01_overview", style_mode, out, prefix)


def fig_timing(data, style_mode, label, out, prefix=""):
    fig, axes = plt.subplots(1, 3, figsize=figsize(style_mode, "wide"))
    nt = data["n_tri"]

    # (a) brute force vs n_tri
    ax = axes[0]
    b = data["time_brute_ms"]
    m = _finite_positive(nt, b)
    if m.sum():
        ax.scatter(
            nt[m],
            b[m],
            s=3,
            alpha=0.2,
            color=PALETTE["secondary"],
            edgecolors="none",
            label="_scatter",
        )
        handles = [scatter_legend_handle(PALETTE["secondary"], "per item")]
        c, med, _, _, _ = binned_median(nt, b)
        if c.size:
            ax.plot(c, med, color=PALETTE["highlight"], lw=1.6)
            handles.append(line_legend_handle(PALETTE["highlight"], "binned median"))
        fit = log_log_fit(nt, b)
        if fit:
            a, lb = fit
            xfit = np.array([nt[m].min(), nt[m].max()])
            ax.plot(
                xfit,
                10 ** (a * np.log10(xfit) + lb),
                color=PALETTE["secondary"],
                lw=1.4,
                linestyle="--",
            )
            handles.append(
                line_legend_handle(
                    PALETTE["secondary"], f"fit slope = {a:.2f}", lw=1.4, ls="--"
                )
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        format_log_axis(ax)
        finalize_axes(
            ax,
            xlabel="Triangle count",
            ylabel="Brute force time (ms)",
            title="(a) Brute force scaling",
        )
        ax.legend(handles=handles, loc="upper left")
    else:
        finalize_axes(
            ax,
            xlabel="Triangle count",
            ylabel="Brute force time (ms)",
            title="(a) Brute force scaling",
        )

    # (b) fast engine vs n_tri
    ax = axes[1]
    f = data["time_fast_ms"]
    m = _finite_positive(nt, f)
    if m.sum():
        ax.scatter(
            nt[m], f[m], s=3, alpha=0.2, color=PALETTE["primary"], edgecolors="none"
        )
        handles = [scatter_legend_handle(PALETTE["primary"], "per item")]
        c, med, _, _, _ = binned_median(nt, f)
        if c.size:
            ax.plot(c, med, color=PALETTE["highlight"], lw=1.6)
            handles.append(line_legend_handle(PALETTE["highlight"], "binned median"))
        fit = log_log_fit(nt, f)
        if fit:
            a, lb = fit
            xfit = np.array([nt[m].min(), nt[m].max()])
            ax.plot(
                xfit,
                10 ** (a * np.log10(xfit) + lb),
                color=PALETTE["primary"],
                lw=1.4,
                linestyle="--",
            )
            handles.append(
                line_legend_handle(
                    PALETTE["primary"], f"fit slope = {a:.2f}", lw=1.4, ls="--"
                )
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        format_log_axis(ax)
        finalize_axes(
            ax,
            xlabel="Triangle count",
            ylabel="Fast engine time (ms)",
            title="(b) Fast engine scaling",
        )
        ax.legend(handles=handles, loc="upper left")
    else:
        finalize_axes(
            ax,
            xlabel="Triangle count",
            ylabel="Fast engine time (ms)",
            title="(b) Fast engine scaling",
        )

    # (c) speedup vs n_tri
    ax = axes[2]
    sp = data["speedup"]
    m = _finite_positive(nt, sp)
    if m.sum():
        ax.scatter(
            nt[m], sp[m], s=3, alpha=0.2, color=PALETTE["primary"], edgecolors="none"
        )
        handles = [scatter_legend_handle(PALETTE["primary"], "per item")]
        c, med, _, _, _ = binned_median(nt, sp)
        if c.size:
            ax.plot(c, med, color=PALETTE["highlight"], lw=1.6)
            handles.append(line_legend_handle(PALETTE["highlight"], "binned median"))
        fit = log_log_fit(nt, sp)
        if fit:
            a, lb = fit
            xfit = np.array([nt[m].min(), nt[m].max()])
            ax.plot(
                xfit,
                10 ** (a * np.log10(xfit) + lb),
                color=PALETTE["primary"],
                lw=1.4,
                linestyle="--",
            )
            handles.append(
                line_legend_handle(
                    PALETTE["primary"], f"fit slope = {a:.2f}", lw=1.4, ls="--"
                )
            )
        ax.axhline(1.0, color=PALETTE["neutral"], lw=0.9, linestyle=":")
        handles.append(line_legend_handle(PALETTE["neutral"], "parity", lw=0.9, ls=":"))
        ax.set_xscale("log")
        ax.set_yscale("log")
        format_log_axis(ax)
        finalize_axes(
            ax,
            xlabel="Triangle count",
            ylabel="Speedup",
            title="(c) Speedup vs complexity",
        )
        ax.legend(handles=handles, loc="upper left")
    else:
        finalize_axes(
            ax,
            xlabel="Triangle count",
            ylabel="Speedup",
            title="(c) Speedup vs complexity",
        )

    fig.suptitle(label, y=1.02, fontsize=mpl.rcParams["font.size"] + 2)
    fig.tight_layout()
    save_or_show(fig, "02_timing", style_mode, out, prefix)


def fig_speedup(data, style_mode, label, out, prefix=""):
    fig, axes = plt.subplots(1, 3, figsize=figsize(style_mode, "wide"))
    sp = data["speedup"]
    sp_pos = sp[np.isfinite(sp) & (sp > 0)]

    # (a) histogram
    ax = axes[0]
    if sp_pos.size:
        bins = np.logspace(np.log10(sp_pos.min()), np.log10(sp_pos.max()), 40)
        ax.hist(sp_pos, bins=bins, color=PALETTE["primary"], alpha=0.9)
        med = float(np.median(sp_pos))
        ax.axvline(
            med,
            color=PALETTE["highlight"],
            lw=1.4,
            linestyle="--",
            label=f"median = {med:.2f}×",
        )
        ax.axvline(1.0, color=PALETTE["neutral"], lw=0.9, linestyle=":", label="parity")
        ax.set_xscale("log")
        format_log_axis(ax, which="x")
        finalize_axes(
            ax,
            xlabel="Speedup",
            ylabel="Item count",
            title="(a) Speedup histogram",
            legend_loc="upper right",
        )
    else:
        finalize_axes(
            ax, xlabel="Speedup", ylabel="Item count", title="(a) Speedup histogram"
        )

    # (b) ECDF
    ax = axes[1]
    if sp_pos.size:
        x, y = ecdf(sp_pos)
        ax.plot(x, y, color=PALETTE["primary"], lw=1.6)
        ax.axvline(1.0, color=PALETTE["neutral"], lw=0.9, linestyle=":")
        for q, c in [
            (0.25, PALETTE["tertiary"]),
            (0.5, PALETTE["primary"]),
            (0.75, PALETTE["secondary"]),
            (0.95, PALETTE["quaternary"]),
        ]:
            v = float(np.quantile(sp_pos, q))
            ax.axvline(v, color=c, lw=0.9, linestyle="--", alpha=0.7)
            ax.text(
                v,
                0.02,
                f"p{int(q * 100)}={v:.2f}×",
                rotation=90,
                fontsize=7,
                color=c,
                va="bottom",
                ha="right",
            )
        ax.set_xscale("log")
        format_log_axis(ax, which="x")
    finalize_axes(
        ax, xlabel="Speedup", ylabel="Fraction of items ≤ x", title="(b) Speedup ECDF"
    )
    ax.set_ylim(0, 1.02)

    # (c) fraction above threshold
    ax = axes[2]
    if sp_pos.size:
        xs = np.logspace(np.log10(sp_pos.min()), np.log10(sp_pos.max()), 100)
        n_faster = np.array([(sp_pos >= x).mean() for x in xs])
        ax.plot(xs, n_faster, color=PALETTE["primary"], lw=1.6)
        ax.set_xscale("log")
        format_log_axis(ax, which="x")
        for thresh in [1, 2, 5, 10, 50]:
            frac = float((sp_pos >= thresh).mean())
            ax.axvline(thresh, color=PALETTE["neutral"], lw=0.7, linestyle=":")
            ax.text(
                thresh,
                frac,
                f"{100 * frac:.0f}%",
                fontsize=7,
                ha="left",
                va="bottom",
                color=PALETTE["dark"],
            )
    finalize_axes(
        ax,
        xlabel="Speedup threshold",
        ylabel="Fraction of items ≥ threshold",
        title="(c) Fraction faster than threshold",
    )
    ax.set_ylim(0, 1.02)

    fig.suptitle(label, y=1.02, fontsize=mpl.rcParams["font.size"] + 2)
    fig.tight_layout()
    save_or_show(fig, "03_speedup", style_mode, out, prefix)


def fig_accuracy(data, style_mode, label, out, prefix=""):
    fig, axes = plt.subplots(2, 2, figsize=figsize(style_mode, "grid22"))

    err_keys = [
        ("p50_abs", PALETTE["tertiary"], "p50"),
        ("p95_abs", PALETTE["primary"], "p95"),
        ("p99_abs", PALETTE["accent"], "p99"),
        ("max_abs", PALETTE["secondary"], "max"),
    ]

    # Determine common bins so the panels are comparable
    all_vals = np.concatenate(
        [
            data[k][np.isfinite(data[k]) & (data[k] > 0)]
            for k, _, _ in err_keys
            if np.isfinite(data[k]).any()
        ]
    )
    common_bins = (
        np.logspace(np.log10(all_vals.min()), np.log10(all_vals.max()), 30)
        if all_vals.size
        else None
    )

    # (a) histograms
    ax = axes[0, 0]
    handles = []
    for key, color, lbl in err_keys:
        v = data[key]
        v = v[np.isfinite(v) & (v > 0)]
        if v.size == 0:
            continue
        bins = common_bins if common_bins is not None else 30
        ax.hist(v, bins=bins, histtype="step", linewidth=1.3, color=color, density=True)
        handles.append(line_legend_handle(color, lbl, lw=1.6, ls="-"))
    if handles:
        ax.set_xscale("log")
        format_log_axis(ax, which="x")
    finalize_axes(
        ax,
        xlabel="Absolute error",
        ylabel="Density",
        title="(a) Error percentile distributions",
    )
    if handles:
        ax.legend(handles=handles, loc="upper right", ncol=2)

    # (b) ECDFs
    ax = axes[0, 1]
    handles = []
    for key, color, lbl in err_keys:
        v = data[key]
        v = v[np.isfinite(v) & (v > 0)]
        if v.size == 0:
            continue
        x, y = ecdf(v)
        ax.plot(x, y, color=color, lw=1.5)
        handles.append(line_legend_handle(color, lbl, lw=1.6, ls="-"))
    if handles:
        ax.set_xscale("log")
        format_log_axis(ax, which="x")
    finalize_axes(
        ax,
        xlabel="Absolute error",
        ylabel="Fraction of items ≤ x",
        title="(b) Error percentile ECDFs",
    )
    ax.set_ylim(0, 1.02)
    if handles:
        ax.legend(handles=handles, loc="lower right")

    # (c) RMS distribution
    ax = axes[1, 0]
    e = data["rms_abs"]
    e = e[np.isfinite(e) & (e > 0)]
    if e.size:
        bins = np.logspace(np.log10(e.min()), np.log10(e.max()), 40)
        ax.hist(e, bins=bins, color=PALETTE["primary"], alpha=0.9)
        med = float(np.median(e))
        mean = float(e.mean())
        ax.axvline(
            med,
            color=PALETTE["highlight"],
            lw=1.4,
            linestyle="--",
            label=f"median = {med:.2e}",
        )
        ax.axvline(
            mean,
            color=PALETTE["secondary"],
            lw=1.4,
            linestyle=":",
            label=f"mean = {mean:.2e}",
        )
        ax.set_xscale("log")
        format_log_axis(ax, which="x")
        finalize_axes(
            ax,
            xlabel="RMS absolute error",
            ylabel="Item count",
            title="(c) RMS error distribution",
            legend_loc="upper right",
        )
    else:
        finalize_axes(
            ax,
            xlabel="RMS absolute error",
            ylabel="Item count",
            title="(c) RMS error distribution",
        )

    # (d) relative error
    ax = axes[1, 1]
    r = data["rms_rel_masked"]
    r = r[np.isfinite(r) & (r > 0)]
    if r.size:
        bins = np.logspace(np.log10(r.min()), np.log10(r.max()), 40)
        ax.hist(r, bins=bins, color=PALETTE["quaternary"], alpha=0.9)
        med = float(np.median(r))
        ax.axvline(
            med,
            color=PALETTE["highlight"],
            lw=1.4,
            linestyle="--",
            label=f"median = {med:.2e}",
        )
        ax.set_xscale("log")
        format_log_axis(ax, which="x")
        finalize_axes(
            ax,
            xlabel="Masked RMS relative error",
            ylabel="Item count",
            title="(d) Relative error distribution",
            legend_loc="upper right",
        )
    else:
        finalize_axes(
            ax,
            xlabel="Masked RMS relative error",
            ylabel="Item count",
            title="(d) Relative error distribution",
        )

    fig.suptitle(label, y=1.005, fontsize=mpl.rcParams["font.size"] + 2)
    fig.tight_layout()
    save_or_show(fig, "04_accuracy", style_mode, out, prefix)


def fig_gradient_quality(data, style_mode, label, out, prefix=""):
    fig, axes = plt.subplots(2, 2, figsize=figsize(style_mode, "grid22"))

    # (a) angular error ECDFs
    ax = axes[0, 0]
    ang_keys = [
        ("per_vec_ang_p50", PALETTE["tertiary"], "p50"),
        ("per_vec_ang_p90", PALETTE["primary"], "p90"),
        ("per_vec_ang_p99", PALETTE["accent"], "p99"),
        ("per_vec_ang_max", PALETTE["secondary"], "max"),
    ]
    handles = []
    for key, color, lbl in ang_keys:
        v = data.get(key, np.array([]))
        v = v[np.isfinite(v) & (v >= 0)]
        if v.size == 0:
            continue
        x, y = ecdf(v + 1e-6)  # small floor to allow log-scale
        ax.plot(x, y, color=color, lw=1.5)
        handles.append(line_legend_handle(color, lbl, lw=1.6))
    if handles:
        ax.set_xscale("log")
        format_log_axis(ax, which="x")
    finalize_axes(
        ax,
        xlabel="Per-primitive angular error (degrees)",
        ylabel="Fraction of items ≤ x",
        title="(a) Angular error, within-item percentiles",
    )
    ax.set_ylim(0, 1.02)
    if handles:
        ax.legend(handles=handles, loc="lower right")

    # (b) per-item median angular error
    ax = axes[0, 1]
    v = data.get("per_vec_ang_p50", np.array([]))
    v = v[np.isfinite(v) & (v >= 0)]
    if v.size:
        vmin = max(v.min(), 1e-3)
        vmax = max(v.max(), 1e-2)
        bins = np.logspace(np.log10(vmin), np.log10(vmax), 40)
        ax.hist(v, bins=bins, color=PALETTE["primary"], alpha=0.9)
        med = float(np.median(v))
        p95 = float(np.percentile(v, 95))
        ax.axvline(
            med,
            color=PALETTE["highlight"],
            lw=1.4,
            linestyle="--",
            label=f"median = {med:.3f}°",
        )
        ax.axvline(
            p95,
            color=PALETTE["secondary"],
            lw=1.4,
            linestyle=":",
            label=f"p95 = {p95:.3f}°",
        )
        ax.set_xscale("log")
        format_log_axis(ax, which="x")
        finalize_axes(
            ax,
            xlabel="Median per-primitive angular error (deg)",
            ylabel="Item count",
            title="(b) Item-wise median angular error",
            legend_loc="upper right",
        )
    else:
        finalize_axes(
            ax,
            xlabel="Median per-primitive angular error (deg)",
            ylabel="Item count",
            title="(b) Item-wise median angular error",
        )

    # (c) magnitude ratio
    ax = axes[1, 0]
    mag_keys = [
        ("mag_ratio_p50", PALETTE["tertiary"], "p50"),
        ("mag_ratio_p90", PALETTE["primary"], "p90"),
        ("mag_ratio_p99", PALETTE["accent"], "p99"),
    ]
    handles = []
    for key, color, lbl in mag_keys:
        v = data.get(key, np.array([]))
        v = v[np.isfinite(v) & (v > 0)]
        if v.size == 0:
            continue
        x, y = ecdf(v)
        ax.plot(x, y, color=color, lw=1.5)
        handles.append(line_legend_handle(color, lbl, lw=1.6))
    ax.axvline(1.0, color=PALETTE["neutral"], lw=0.9, linestyle=":")
    handles.append(line_legend_handle(PALETTE["neutral"], "perfect", lw=0.9, ls=":"))
    if handles:
        ax.set_xscale("log")
        format_log_axis(ax, which="x")
    finalize_axes(
        ax,
        xlabel="Magnitude ratio ‖fast‖ / ‖ref‖",
        ylabel="Fraction of items ≤ x",
        title="(c) Magnitude ratio, within-item percentiles",
    )
    ax.set_ylim(0, 1.02)
    if handles:
        ax.legend(handles=handles, loc="upper left")

    # (d) global vs per-vector
    ax = axes[1, 1]
    gc = data.get("global_cosine", np.array([]))
    pv99 = data.get("per_vec_cos_p99", np.array([]))
    m = _finite(gc, pv99)
    if m.sum():
        ax.scatter(
            pv99[m], gc[m], s=5, alpha=0.5, color=PALETTE["primary"], edgecolors="none"
        )
    ax.plot(
        [0, 1],
        [0, 1],
        color=PALETTE["neutral"],
        lw=0.9,
        linestyle=":",
        label="equality",
    )
    finalize_axes(
        ax,
        xlabel="Per-primitive cosine, p99",
        ylabel="Global cosine similarity",
        title="(d) Global vs per-primitive agreement",
        legend_loc="lower right",
    )

    fig.suptitle(label, y=1.005, fontsize=mpl.rcParams["font.size"] + 2)
    fig.tight_layout()
    save_or_show(fig, "09_gradient_quality", style_mode, out, prefix)


def fig_error_scaling(data, style_mode, label, out, prefix=""):
    fig, axes = plt.subplots(1, 3, figsize=figsize(style_mode, "wide"))
    nt = data["n_tri"]

    # (a) error vs n_tri
    ax = axes[0]
    handles = []
    for key, color, lbl in [
        ("rms_abs", PALETTE["primary"], "RMS"),
        ("p99_abs", PALETTE["accent"], "p99"),
        ("max_abs", PALETTE["secondary"], "max"),
    ]:
        y = data[key]
        m = _finite_positive(nt, y)
        if m.sum() == 0:
            continue
        ax.scatter(nt[m], y[m], s=3, alpha=0.15, color=color, edgecolors="none")
        handles.append(scatter_legend_handle(color, lbl))
        c, med, _, _, _ = binned_median(nt, y)
        if c.size:
            ax.plot(c, med, color=color, lw=1.4)
    if handles:
        ax.set_xscale("log")
        ax.set_yscale("log")
        format_log_axis(ax)
        finalize_axes(
            ax,
            xlabel="Triangle count",
            ylabel="Absolute error",
            title="(a) Error vs complexity",
        )
        ax.legend(handles=handles, loc="upper left")
    else:
        finalize_axes(
            ax,
            xlabel="Triangle count",
            ylabel="Absolute error",
            title="(a) Error vs complexity",
        )

    # (b) error vs brute-force cost
    ax = axes[1]
    b = data["time_brute_ms"]
    m = _finite_positive(b, data["rms_abs"])
    if m.sum():
        ax.scatter(
            b[m],
            data["rms_abs"][m],
            s=3,
            alpha=0.2,
            color=PALETTE["primary"],
            edgecolors="none",
        )
        c, med, _, _, _ = binned_median(b, data["rms_abs"])
        if c.size:
            ax.plot(c, med, color=PALETTE["highlight"], lw=1.6)
        ax.set_xscale("log")
        ax.set_yscale("log")
        format_log_axis(ax)
    finalize_axes(
        ax,
        xlabel="Brute force time (ms)",
        ylabel="RMS absolute error",
        title="(b) Error vs brute force cost",
    )

    # (c) hexbin
    ax = axes[2]
    f = data["time_fast_ms"]
    m = _finite_positive(nt, f, data["rms_abs"])
    if m.sum() > 20:
        h = ax.hexbin(
            np.log10(nt[m]),
            np.log10(f[m]),
            C=np.log10(data["rms_abs"][m]),
            reduce_C_function=np.median,
            gridsize=30,
            cmap="viridis",
            mincnt=2,
        )
        cb = plt.colorbar(h, ax=ax)
        cb.set_label("log₁₀ RMS error", fontsize=mpl.rcParams["font.size"] - 1)
        ax.set_xlabel("log₁₀ triangle count")
        ax.set_ylabel("log₁₀ fast engine time (ms)")
    finalize_axes(ax, title="(c) Error across (complexity, runtime)")

    fig.suptitle(label, y=1.02, fontsize=mpl.rcParams["font.size"] + 2)
    fig.tight_layout()
    save_or_show(fig, "05_error_scaling", style_mode, out, prefix)


def fig_misclassification(data, style_mode, label, out, prefix=""):
    fig, axes = plt.subplots(1, 3, figsize=figsize(style_mode, "wide"))
    mc = data["misclass_count"]
    mf = data["misclass_frac"]
    nt = data["n_tri"]

    # (a) bar bins
    ax = axes[0]
    mc_v = mc[np.isfinite(mc)]
    if mc_v.size:
        edges = [0, 0.5, 5, 50, 500, np.inf]
        labels = ["0", "1–5", "6–50", "51–500", ">500"]
        colors = [
            PALETTE["tertiary"],
            PALETTE["primary"],
            PALETTE["accent"],
            PALETTE["quaternary"],
            PALETTE["secondary"],
        ]
        vals = [
            int(((mc_v >= edges[i]) & (mc_v < edges[i + 1])).sum()) for i in range(5)
        ]
        ax.bar(range(5), vals, color=colors, alpha=0.9)
        ax.set_xticks(range(5))
        ax.set_xticklabels(labels)
        for i, v in enumerate(vals):
            if v:
                ax.text(
                    i,
                    v,
                    f"{v}\n({100 * v / len(mc_v):.0f}%)",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
        ax.set_ylim(0, max(vals) * 1.25 if max(vals) else 1)
    finalize_axes(
        ax,
        xlabel="Misclassifications per item",
        ylabel="Item count",
        title="(a) Misclassification bins",
    )

    # (b) ECDF of fraction
    ax = axes[1]
    mf_v = mf[np.isfinite(mf) & (mf >= 0)]
    if mf_v.size:
        floor = 0.5 / max(1, int(data["n_queries"][0]))
        x, y = ecdf(mf_v + floor * (mf_v == 0))
        ax.plot(x, y, color=PALETTE["primary"], lw=1.6)
        ax.set_xscale("log")
        format_log_axis(ax, which="x")
    finalize_axes(
        ax,
        xlabel="Misclassification fraction",
        ylabel="Fraction of items ≤ x",
        title="(b) Misclassification fraction ECDF",
    )
    ax.set_ylim(0, 1.02)

    # (c) misclass vs complexity
    ax = axes[2]
    m = _finite(nt, mc) & (nt > 0)
    if m.sum():
        jitter = 10 ** (0.01 * np.random.default_rng(0).standard_normal(m.sum()))
        ax.scatter(
            nt[m],
            mc[m] + jitter,
            s=4,
            alpha=0.35,
            color=PALETTE["primary"],
            edgecolors="none",
        )
        ax.set_xscale("log")
        format_log_axis(ax, which="x")
        if (mc[m] > 0).any():
            ax.set_yscale("symlog", linthresh=1)
    finalize_axes(
        ax,
        xlabel="Triangle count",
        ylabel="Misclassifications",
        title="(c) Misclass vs complexity",
    )

    fig.suptitle(label, y=1.02, fontsize=mpl.rcParams["font.size"] + 2)
    fig.tight_layout()
    save_or_show(fig, "06_misclassification", style_mode, out, prefix)


def fig_tradeoff(data, style_mode, label, out, prefix=""):
    w, h = figsize(style_mode, "wide")
    fig, axes = plt.subplots(1, 2, figsize=(2 * w, 1.6 * h))
    sp = data["speedup"]
    e = data["rms_abs"]
    m = _finite_positive(sp, e)

    # (a) trade-off frontier
    ax = axes[0]
    if m.sum():
        sc = ax.scatter(
            sp[m],
            e[m],
            c=np.log10(data["n_tri"][m]),
            s=6,
            alpha=0.65,
            cmap="viridis",
            edgecolors="none",
        )
        cb = plt.colorbar(sc, ax=ax)
        cb.set_label("log₁₀ triangle count", fontsize=mpl.rcParams["font.size"] - 1)
        idx = pareto_front(sp[m], e[m])
        ss = sp[m][idx]
        ee = e[m][idx]
        order = np.argsort(ss)
        ax.plot(ss[order], ee[order], color=PALETTE["highlight"], lw=1.6)
        handles = [line_legend_handle(PALETTE["highlight"], "Pareto frontier", lw=1.8)]
        ax.set_xscale("log")
        ax.set_yscale("log")
        format_log_axis(ax)
        finalize_axes(
            ax,
            xlabel="Speedup",
            ylabel="RMS absolute error",
            title="(a) Trade-off frontier",
        )
        ax.legend(handles=handles, loc="lower left")
    else:
        finalize_axes(
            ax,
            xlabel="Speedup",
            ylabel="RMS absolute error",
            title="(a) Trade-off frontier",
        )

    # (b) joint quality
    ax = axes[1]
    if m.sum():
        q = sp[m] / e[m]
        q = q[np.isfinite(q) & (q > 0)]
        if q.size:
            bins = np.logspace(np.log10(q.min()), np.log10(q.max()), 40)
            ax.hist(q, bins=bins, color=PALETTE["quaternary"], alpha=0.9)
            med = float(np.median(q))
            ax.axvline(
                med,
                color=PALETTE["highlight"],
                lw=1.4,
                linestyle="--",
                label=f"median = {med:.2e}",
            )
            ax.set_xscale("log")
            format_log_axis(ax, which="x")
            finalize_axes(
                ax,
                xlabel="Speedup / RMS error",
                ylabel="Item count",
                title="(b) Joint quality distribution",
                legend_loc="upper right",
            )
        else:
            finalize_axes(
                ax,
                xlabel="Speedup / RMS error",
                ylabel="Item count",
                title="(b) Joint quality distribution",
            )
    else:
        finalize_axes(
            ax,
            xlabel="Speedup / RMS error",
            ylabel="Item count",
            title="(b) Joint quality distribution",
        )

    fig.suptitle(label, y=1.02, fontsize=mpl.rcParams["font.size"] + 2)
    fig.tight_layout()
    save_or_show(fig, "07_tradeoff", style_mode, out, prefix)


def fig_paper_dashboard(data, style_mode, label, out, prefix="", caps=None):
    """Compact 2x2 summary suitable for a paper."""
    fig, axes = plt.subplots(2, 2, figsize=figsize(style_mode, "grid22"))
    sp = data["speedup"]
    e = data["rms_abs"]
    nt = data["n_tri"]
    mc = data["misclass_count"]

    # (a) Speedup ECDF
    ax = axes[0, 0]
    sp_v = sp[np.isfinite(sp) & (sp > 0)]
    if sp_v.size:
        x, y = ecdf(sp_v)
        ax.plot(x, y, color=PALETTE["primary"], lw=1.6)
        ax.axvline(1.0, color=PALETTE["neutral"], lw=0.8, linestyle=":")
        med = float(np.median(sp_v))
        ax.axvline(
            med,
            color=PALETTE["highlight"],
            lw=1.2,
            linestyle="--",
            label=f"median {med:.2f}×",
        )
        ax.set_xscale("log")
        format_log_axis(ax, which="x")
        finalize_axes(
            ax,
            xlabel="Speedup",
            ylabel="Fraction of items",
            title="(a) Speedup distribution",
            legend_loc="lower right",
        )
    else:
        finalize_axes(
            ax,
            xlabel="Speedup",
            ylabel="Fraction of items",
            title="(a) Speedup distribution",
        )
    ax.set_ylim(0, 1.02)

    # (b) RMS ECDF
    ax = axes[0, 1]
    e_v = e[np.isfinite(e) & (e > 0)]
    if e_v.size:
        x, y = ecdf(e_v)
        ax.plot(x, y, color=PALETTE["secondary"], lw=1.6)
        med = float(np.median(e_v))
        ax.axvline(
            med,
            color=PALETTE["highlight"],
            lw=1.2,
            linestyle="--",
            label=f"median {med:.2e}",
        )
        ax.set_xscale("log")
        format_log_axis(ax, which="x")
        finalize_axes(
            ax,
            xlabel="RMS absolute error",
            ylabel="Fraction of items",
            title="(b) Error distribution",
            legend_loc="lower right",
        )
    else:
        finalize_axes(
            ax,
            xlabel="RMS absolute error",
            ylabel="Fraction of items",
            title="(b) Error distribution",
        )
    ax.set_ylim(0, 1.02)

    # (c) Speedup vs complexity
    ax = axes[1, 0]
    m = _finite_positive(nt, sp)
    if m.sum():
        ax.scatter(
            nt[m], sp[m], s=3, alpha=0.2, color=PALETTE["primary"], edgecolors="none"
        )
        c, med, p25, p75, _ = binned_median(nt, sp)
        handles = [scatter_legend_handle(PALETTE["primary"], "per item")]
        if c.size:
            ax.fill_between(c, p25, p75, color=PALETTE["primary"], alpha=0.2)
            ax.plot(c, med, color=PALETTE["primary"], lw=1.6)
            handles.append(line_legend_handle(PALETTE["primary"], "median (IQR band)"))
        ax.axhline(1.0, color=PALETTE["neutral"], lw=0.8, linestyle=":")
        handles.append(line_legend_handle(PALETTE["neutral"], "parity", lw=0.9, ls=":"))
        ax.set_xscale("log")
        ax.set_yscale("log")
        format_log_axis(ax)
        finalize_axes(
            ax,
            xlabel="Triangle count",
            ylabel="Speedup",
            title="(c) Speedup vs complexity",
        )
        ax.legend(handles=handles, loc="upper left")
    else:
        finalize_axes(
            ax,
            xlabel="Triangle count",
            ylabel="Speedup",
            title="(c) Speedup vs complexity",
        )

    # (d) Misclassification summary
    ax = axes[1, 1]
    mc_v = mc[np.isfinite(mc)]
    if mc_v.size:
        edges = [0, 0.5, 5, 50, 500, np.inf]
        labels = ["0", "1–5", "6–50", "51–500", ">500"]
        colors = [
            PALETTE["tertiary"],
            PALETTE["primary"],
            PALETTE["accent"],
            PALETTE["quaternary"],
            PALETTE["secondary"],
        ]
        vals = [
            int(((mc_v >= edges[i]) & (mc_v < edges[i + 1])).sum()) for i in range(5)
        ]
        ax.bar(range(5), vals, color=colors, alpha=0.9)
        ax.set_xticks(range(5))
        ax.set_xticklabels(labels)
        total = len(mc_v)
        for i, v in enumerate(vals):
            if v:
                ax.text(
                    i,
                    v,
                    f"{100 * v / total:.0f}%",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
        ax.set_ylim(0, max(vals) * 1.15 if max(vals) else 1)
    finalize_axes(
        ax,
        xlabel="Misclassifications per item",
        ylabel="Item count",
        title="(d) Misclassification summary",
    )

    fig.suptitle(label, y=1.005, fontsize=mpl.rcParams["font.size"] + 2)
    fig.tight_layout()
    save_or_show(fig, "08_paper_dashboard", style_mode, out, prefix)


def fig_paper_dashboard_backward(data, style_mode, label, out, prefix=""):
    """Compact 2x2 summary for backward-mode results."""
    fig, axes = plt.subplots(2, 2, figsize=figsize(style_mode, "grid22"))
    sp = data["speedup"]
    e = data["rms_abs"]
    nt = data["n_tri"]

    # (a) Speedup ECDF
    ax = axes[0, 0]
    sp_v = sp[np.isfinite(sp) & (sp > 0)]
    if sp_v.size:
        x, y = ecdf(sp_v)
        ax.plot(x, y, color=PALETTE["primary"], lw=1.6)
        ax.axvline(1.0, color=PALETTE["neutral"], lw=0.8, linestyle=":")
        med = float(np.median(sp_v))
        ax.axvline(
            med,
            color=PALETTE["highlight"],
            lw=1.2,
            linestyle="--",
            label=f"median {med:.2f}×",
        )
        ax.set_xscale("log")
        format_log_axis(ax, which="x")
        finalize_axes(
            ax,
            xlabel="Speedup",
            ylabel="Fraction of items",
            title="(a) Speedup distribution",
            legend_loc="lower right",
        )
    else:
        finalize_axes(
            ax,
            xlabel="Speedup",
            ylabel="Fraction of items",
            title="(a) Speedup distribution",
        )
    ax.set_ylim(0, 1.02)

    # (b) Per-primitive angular error (median within item)
    ax = axes[0, 1]
    v = data.get("per_vec_ang_p50", np.array([]))
    v = v[np.isfinite(v) & (v >= 0)]
    if v.size:
        vmin = max(v.min(), 1e-3)
        vmax = max(v.max(), 1e-2)
        bins = np.logspace(np.log10(vmin), np.log10(vmax), 40)
        ax.hist(v, bins=bins, color=PALETTE["secondary"], alpha=0.9)
        med = float(np.median(v))
        ax.axvline(
            med,
            color=PALETTE["highlight"],
            lw=1.4,
            linestyle="--",
            label=f"median = {med:.3f}°",
        )
        ax.set_xscale("log")
        format_log_axis(ax, which="x")
        finalize_axes(
            ax,
            xlabel="Median per-primitive angular error (deg)",
            ylabel="Item count",
            title="(b) Angular error distribution",
            legend_loc="upper right",
        )
    else:
        finalize_axes(
            ax,
            xlabel="Median per-primitive angular error (deg)",
            ylabel="Item count",
            title="(b) Angular error distribution",
        )

    # (c) Speedup vs complexity
    ax = axes[1, 0]
    m = _finite_positive(nt, sp)
    if m.sum():
        ax.scatter(
            nt[m], sp[m], s=3, alpha=0.2, color=PALETTE["primary"], edgecolors="none"
        )
        c, med, p25, p75, _ = binned_median(nt, sp)
        handles = [scatter_legend_handle(PALETTE["primary"], "per item")]
        if c.size:
            ax.fill_between(c, p25, p75, color=PALETTE["primary"], alpha=0.2)
            ax.plot(c, med, color=PALETTE["primary"], lw=1.6)
            handles.append(line_legend_handle(PALETTE["primary"], "median (IQR band)"))
        ax.axhline(1.0, color=PALETTE["neutral"], lw=0.8, linestyle=":")
        handles.append(line_legend_handle(PALETTE["neutral"], "parity", lw=0.9, ls=":"))
        ax.set_xscale("log")
        ax.set_yscale("log")
        format_log_axis(ax)
        finalize_axes(
            ax,
            xlabel="Triangle count",
            ylabel="Speedup",
            title="(c) Speedup vs complexity",
        )
        ax.legend(handles=handles, loc="upper left")
    else:
        finalize_axes(
            ax,
            xlabel="Triangle count",
            ylabel="Speedup",
            title="(c) Speedup vs complexity",
        )

    # (d) Magnitude ratio distribution
    ax = axes[1, 1]
    mr = data.get("mag_ratio_p50", np.array([]))
    mr = mr[np.isfinite(mr) & (mr > 0)]
    if mr.size:
        bins = np.linspace(max(0.5, mr.min() - 0.05), min(1.5, mr.max() + 0.05), 40)
        ax.hist(mr, bins=bins, color=PALETTE["quaternary"], alpha=0.9)
        med = float(np.median(mr))
        ax.axvline(
            1.0, color=PALETTE["neutral"], lw=0.9, linestyle=":", label="perfect"
        )
        ax.axvline(
            med,
            color=PALETTE["highlight"],
            lw=1.4,
            linestyle="--",
            label=f"median = {med:.4f}",
        )
        finalize_axes(
            ax,
            xlabel="Magnitude ratio ‖fast‖ / ‖ref‖ (p50)",
            ylabel="Item count",
            title="(d) Magnitude ratio distribution",
            legend_loc="upper right",
        )
    else:
        finalize_axes(
            ax,
            xlabel="Magnitude ratio ‖fast‖ / ‖ref‖ (p50)",
            ylabel="Item count",
            title="(d) Magnitude ratio distribution",
        )

    fig.suptitle(label, y=1.005, fontsize=mpl.rcParams["font.size"] + 2)
    fig.tight_layout()
    save_or_show(fig, "08_paper_dashboard", style_mode, out, prefix)


def fig_buckets(bucket_data, style_mode, label, out, prefix=""):
    """Per-bucket metrics for backward modes (near / mid / far from surface).

    Each backward mode is re-run with a grad_output masked to queries in one
    distance-to-surface bucket, so the resulting gradient is exactly that
    bucket's contribution. Panels:

      (a) direction agreement by bucket (global cosine, per-bucket)
      (b) per-vector angular error p99 by bucket, in degrees
      (c) magnitude ratio ‖fast‖ / ‖ref‖ (p50) by bucket
      (d) fraction of the total reference gradient norm carried by each bucket
    """
    modes = [m for m in bucket_data.keys() if bucket_data[m]]
    if not modes:
        return

    fig, axes = plt.subplots(2, 2, figsize=figsize(style_mode, "grid22"))
    ax_gc, ax_ang, ax_mag, ax_frac = axes.flat

    palette = [
        PALETTE["secondary"],
        PALETTE["accent"],
        PALETTE["primary"],
        PALETTE["tertiary"],
        PALETTE["quaternary"],
    ]

    # Union of bucket labels across modes, sorted by lower edge.
    all_labels = sorted(
        {lbl for mode in modes for lbl in bucket_data[mode].keys()},
        key=_bucket_sort_key,
    )
    xs = np.arange(len(all_labels))

    for mi, mode_name in enumerate(modes):
        buckets = bucket_data[mode_name]
        color = palette[mi % len(palette)]

        def _collect(metric_key):
            med, p25, p75, cnt = [], [], [], []
            for lbl in all_labels:
                vals = [
                    float(r[metric_key])
                    for r in buckets.get(lbl, [])
                    if metric_key in r and np.isfinite(float(r[metric_key]))
                ]
                if not vals:
                    med.append(np.nan)
                    p25.append(np.nan)
                    p75.append(np.nan)
                    cnt.append(0)
                    continue
                med.append(float(np.median(vals)))
                p25.append(float(np.percentile(vals, 25)))
                p75.append(float(np.percentile(vals, 75)))
                cnt.append(len(vals))
            return (
                np.array(med),
                np.array(p25),
                np.array(p75),
                np.array(cnt, dtype=int),
            )

        for ax, key in [
            (ax_gc, "global_cosine"),
            (ax_ang, "per_vec_ang_p99"),
            (ax_mag, "mag_ratio_p50"),
        ]:
            med, p25, p75, _ = _collect(key)
            ax.plot(xs, med, marker="o", color=color, lw=1.6, label=mode_name)
            ax.fill_between(xs, p25, p75, color=color, alpha=0.15)

        # Signal fraction: ‖ref_bucket‖ / Σ_k ‖ref_k‖
        med_norm, _, _, _ = _collect("ref_norm")
        finite = np.isfinite(med_norm)
        if finite.any() and med_norm[finite].sum() > 0:
            ax_frac.plot(
                xs,
                med_norm / np.nansum(med_norm),
                marker="o",
                color=color,
                lw=1.6,
                label=mode_name,
            )

    panel_defs = [
        (ax_gc, "(a) Global cosine per distance bucket", "Global cosine", (0, 1.02)),
        (ax_ang, "(b) Per-vector angular error p99", "Degrees", None),
        (ax_mag, "(c) Magnitude ratio p50", "‖fast‖ / ‖ref‖", None),
        (ax_frac, "(d) Signal fraction per bucket", "‖ref_k‖ / Σ‖ref_k‖", None),
    ]
    for ax, title, ylabel, ylim in panel_defs:
        ax.set_xticks(xs)
        ax.set_xticklabels(all_labels, rotation=30, ha="right", fontsize=7)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_axisbelow(True)
        ax.grid(True, which="both", alpha=0.3)
        if ylim is not None:
            ax.set_ylim(*ylim)
        if ax is ax_mag:
            ax.axhline(1.0, color=PALETTE["neutral"], lw=0.8, linestyle=":")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="best", fontsize=7)

    fig.suptitle(
        f"{label} — per-distance-bucket attribution",
        y=1.005,
        fontsize=mpl.rcParams["font.size"] + 2,
    )
    fig.tight_layout()
    save_or_show(fig, "10_buckets", style_mode, out, prefix)


def fig_epsilon_sweep(by_mode, style_mode, label, out, prefix=""):
    """Fast-vs-brute error, direction agreement, and speedup as a function
    of the regularization strength epsilon. One line per base mode.

    Expects each mode key to be either a plain mode name or a name of the
    form ``<mode>@eps=<value>``. Groups by base mode and sorts the points by
    epsilon.
    """
    # Group: base_mode -> [(eps, record), ...]
    groups: dict = defaultdict(list)
    for key, records in by_mode.items():
        for rec in records:
            base = rec.get("_base_mode", key)
            eps = rec.get("_epsilon", float("nan"))
            if not math.isfinite(eps):
                continue
            groups[base].append((eps, rec))

    # Only produce the figure if there is at least one base mode with 2+
    # distinct epsilon values.
    has_sweep = any(len({e for e, _ in pts}) >= 2 for pts in groups.values())
    if not has_sweep:
        return

    palette = [
        PALETTE["secondary"],
        PALETTE["accent"],
        PALETTE["primary"],
        PALETTE["tertiary"],
        PALETTE["quaternary"],
    ]

    def _collect(pts, key, agg="median"):
        """Return (eps_sorted, value_sorted) using per-eps median."""
        per_eps = defaultdict(list)
        for e, r in pts:
            v = r.get(key, float("nan"))
            try:
                v = float(v)
            except (TypeError, ValueError):
                v = float("nan")
            if math.isfinite(v):
                per_eps[e].append(v)
        eps_sorted = sorted(per_eps.keys())
        if agg == "median":
            vals = [float(np.median(per_eps[e])) for e in eps_sorted]
        else:
            vals = [float(np.max(per_eps[e])) for e in eps_sorted]
        return np.array(eps_sorted), np.array(vals)

    panel_defs = [
        ("global_rel_norm", "Global relative-norm error", True, "(a)"),
        ("global_cosine", "Global cosine similarity", True, "(b)"),
        ("per_vec_ang_p99", "Per-primitive angular error p99 (deg)", False, "(c)"),
        ("speedup", "Speedup (brute / fast)", False, "(d)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=figsize(style_mode, "grid22"))
    for ax, (key, ylabel, logy, tag) in zip(axes.flat, panel_defs):
        handles = []
        for mi, (base, pts) in enumerate(sorted(groups.items())):
            eps, val = _collect(pts, key)
            if eps.size < 2:
                continue
            color = palette[mi % len(palette)]
            # A small offset makes the lowest-eps point visible on log axes.
            eps_plot = np.where(
                eps > 0,
                eps,
                max(eps[eps > 0].min() / 10, 1e-6) if (eps > 0).any() else 1e-6,
            )
            ax.plot(eps_plot, val, marker="o", color=color, lw=1.6, label=base)
            handles.append(line_legend_handle(color, base, lw=1.6))
        ax.set_xscale("log")
        format_log_axis(ax, which="x")
        if logy:
            ax.set_yscale("log")
            format_log_axis(ax, which="y")
        if key == "global_cosine":
            ax.set_ylim(0, 1.02)
        if key == "speedup":
            ax.axhline(1.0, color=PALETTE["neutral"], lw=0.8, linestyle=":")
        finalize_axes(
            ax, xlabel=r"$\varepsilon$", ylabel=ylabel, title=f"{tag} {ylabel}"
        )
        if handles:
            ax.legend(handles=handles, loc="best", fontsize=8)

    fig.suptitle(
        f"{label} — regularization sweep",
        y=1.005,
        fontsize=mpl.rcParams["font.size"] + 2,
    )
    fig.tight_layout()
    save_or_show(fig, "11_epsilon_sweep", style_mode, out, prefix)


# ============================================================================
# Plot registry
# ============================================================================


@dataclass
class PlotSpec:
    name: str
    fn: Callable
    requires: Callable[[dict, dict], bool]
    kind: str  # "forward", "backward", or "both"


def _req(caps_key):
    def check(caps, data):
        return caps.get(caps_key, False)

    return check


PLOT_SPECS = [
    PlotSpec(
        "overview",
        fig_overview,
        requires=lambda caps, d: (
            caps["speedup"]
            and caps["accuracy"]
            and caps["misclassification"]
            and caps["timing"]
        ),
        kind="forward",
    ),
    PlotSpec(
        "timing",
        fig_timing,
        requires=_req("timing"),
        kind="both",
    ),
    PlotSpec(
        "speedup",
        fig_speedup,
        requires=_req("speedup"),
        kind="both",
    ),
    PlotSpec(
        "accuracy",
        fig_accuracy,
        requires=_req("accuracy"),
        kind="both",
    ),
    PlotSpec(
        "gradient_quality",
        fig_gradient_quality,
        requires=_req("gradient"),
        kind="backward",
    ),
    PlotSpec(
        "error_scaling",
        fig_error_scaling,
        requires=lambda caps, d: caps["timing"] and caps["accuracy"],
        kind="both",
    ),
    PlotSpec(
        "misclassification",
        fig_misclassification,
        requires=_req("misclassification"),
        kind="forward",
    ),
    PlotSpec(
        "tradeoff",
        fig_tradeoff,
        requires=lambda caps, d: caps["speedup"] and caps["accuracy"],
        kind="both",
    ),
]


def dispatch_paper_dashboard(caps, data, style_mode, label, out, prefix):
    if caps["gradient"]:
        print_hint("paper_dashboard", prefix or "backward")
        fig_paper_dashboard_backward(data, style_mode, label, out, prefix)
    elif caps["misclassification"]:
        print_hint("paper_dashboard", prefix or "forward")
        fig_paper_dashboard(data, style_mode, label, out, prefix)


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Plot metrics from a Winder batch-evaluation JSONL file."
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Path to results.jsonl"
    )
    parser.add_argument(
        "--mode",
        choices=["display", "paper"],
        default="display",
        help="Style preset. 'paper' saves PDFs at 300 DPI, 'display' saves PNGs.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./plots",
        help="Directory where figures are written.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Dataset name in figure titles. Defaults to the JSONL file stem.",
    )
    parser.add_argument(
        "--figures",
        type=str,
        default="all",
        help="Comma-separated list of figure names, or 'all'.",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for jitter.")
    parser.add_argument(
        "--modes",
        type=str,
        default="auto",
        help="Comma-separated list of evaluation modes to "
        "plot, or 'auto' to plot every mode found in "
        "the file.",
    )
    parser.add_argument(
        "--prefix",
        action="store_true",
        help="Prefix every output filename with the mode "
        "name (useful when plotting multiple modes).",
    )

    args = parser.parse_args()
    np.random.seed(args.seed)
    setup_style(args.mode)

    input_path = Path(args.input)
    dataset_name = args.title or input_path.stem

    print(f"Loading {input_path} ...")
    by_mode = load_records_by_mode(input_path)
    if not by_mode:
        raise SystemExit("No valid records in file.")

    available = sorted(by_mode.keys())
    print(f"  Modes found: {', '.join(available)}")
    for m, recs in by_mode.items():
        print(f"    {m:<24} {len(recs):>6d} records")

    if args.modes == "auto":
        selected = available
    else:
        selected = [m.strip() for m in args.modes.split(",") if m.strip()]
        missing = [m for m in selected if m not in by_mode]
        if missing:
            raise SystemExit(
                f"Requested modes not found in file: {missing}\nAvailable: {available}"
            )

    # Requested figure subset
    if args.figures == "all":
        requested_figures = None  # all
    else:
        requested_figures = {k.strip() for k in args.figures.split(",")}

    bucket_data_loaded = False
    bucket_data_all = None

    for mode_name in selected:
        records = by_mode[mode_name]
        if not records:
            print(f"\n  [{mode_name}] no records; skipping")
            continue

        data = to_arrays(records)
        caps = compute_capabilities(data)

        mode_label = f"{dataset_name} [{mode_name}]"
        print_summary(data, dataset_name, mode_name)
        print(f"  Detected capabilities in [{mode_name}]:")
        for k, v in caps.items():
            print(f"    {k:<20} {v}")

        prefix = mode_name if args.prefix else ""
        # Make output paths unique per mode even without --prefix
        if len(selected) > 1 and not args.prefix:
            prefix = mode_name

        print(f"\n  Writing figures for [{mode_name}] to {args.output} ...")

        for spec in PLOT_SPECS:
            if requested_figures is not None and spec.name not in requested_figures:
                continue
            if spec.kind == "forward" and caps["gradient"]:
                continue
            if spec.kind == "backward" and not caps["gradient"]:
                continue
            if not spec.requires(caps, data):
                continue
            print_hint(spec.name, mode_name)
            try:
                spec.fn(data, args.mode, mode_label, args.output, prefix)
            except Exception as exc:
                print(
                    f"    ERROR generating '{spec.name}': {type(exc).__name__}: {exc}"
                )
                # Per-bucket figure — needs its own data structure.
        if (
            requested_figures is None or "buckets" in requested_figures
        ) and mode_name.startswith("backward"):
            if not bucket_data_loaded:
                bucket_data_all = load_bucket_data(input_path)
                bucket_data_loaded = True
            if mode_name in bucket_data_all:
                print_hint("buckets", mode_name)
                try:
                    fig_buckets(
                        {mode_name: bucket_data_all[mode_name]},
                        args.mode,
                        mode_label,
                        args.output,
                        prefix,
                    )
                except Exception as exc:
                    print(
                        f"    ERROR generating 'buckets': {type(exc).__name__}: {exc}"
                    )

        # Paper dashboard: choose forward or backward variant
        if requested_figures is None or "paper_dashboard" in requested_figures:
            try:
                dispatch_paper_dashboard(
                    caps, data, args.mode, mode_label, args.output, prefix
                )
            except Exception as exc:
                print(
                    f"    ERROR generating 'paper_dashboard': "
                    f"{type(exc).__name__}: {exc}"
                )

        # Epsilon-sweep figure: only runs when at least one mode key has
        # the "@eps=" suffix with multiple distinct values.
        if requested_figures is None or "epsilon_sweep" in requested_figures:
            try:
                # The label is the dataset name without any mode suffix.
                fig_epsilon_sweep(by_mode, args.mode, dataset_name, args.output)
            except Exception as exc:
                print(
                    f"    ERROR generating 'epsilon_sweep': {type(exc).__name__}: {exc}"
                )

    print("\nDone.")


if __name__ == "__main__":
    main()
