#!/usr/bin/env python3
"""Plot metrics from a Winder batch-evaluation JSONL file.

Reads the JSONL produced by evaluate_thingi10K.py (or any script that emits
records with the same schema) and produces a family of plots covering
timing, accuracy, misclassification, and trade-offs.

Usage:
    python plot_results.py --input results.jsonl --mode paper --output ./figs
    python plot_results.py --input results.jsonl --mode display
"""

import argparse
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# ============================================================================
# Data loading
# ============================================================================

REQUIRED_KEYS = {"n_tri", "n_queries", "rms_abs"}
NUMERIC_KEYS = [
    # ... existing forward keys ...
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


def load_records(path: Path, mode_name: str = "forward_triangle"):
    """Load valid records from a JSONL file.

    Supports both the legacy flat schema (all metrics at top level) and
    the extended schema (metrics nested under ``record["modes"][mode_name]``).
    For nested records, the requested mode is flattened to the top level so
    the rest of the plotting code can consume it unchanged.
    """
    records = []
    skipped = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            if "error" in rec:
                continue
            if "modes" in rec and isinstance(rec["modes"], dict):
                mode_metrics = rec["modes"].get(mode_name)
                if mode_metrics is None or "error" in mode_metrics:
                    skipped += 1
                    continue
                # Flatten the selected mode onto the top level
                for k, v in mode_metrics.items():
                    rec.setdefault(k, v)
                rec["_mode_name"] = mode_name
            if not REQUIRED_KEYS.issubset(rec.keys()):
                skipped += 1
                continue
            records.append(rec)
    return records, skipped


def to_arrays(records):
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
# Style
# ============================================================================

PALETTE = {
    "primary": "#1f77b4",
    "secondary": "#d62728",
    "tertiary": "#2ca02c",
    "quaternary": "#9467bd",
    "accent": "#ff7f0e",
    "neutral": "#7f7f7f",
    "grid": "#cccccc",
    "highlight": "#e41a1c",
}


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
                "grid.alpha": 0.3,
                "grid.linewidth": 0.4,
                "grid.color": PALETTE["grid"],
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.linewidth": 0.6,
                "lines.linewidth": 1.0,
                "lines.markersize": 2.5,
                "legend.frameon": False,
                "pdf.fonttype": 42,  # embed as Type 42 (TrueType)
                "ps.fonttype": 42,
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
                "grid.alpha": 0.25,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "lines.linewidth": 1.4,
                "lines.markersize": 5,
                "legend.frameon": True,
                "legend.framealpha": 0.9,
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


def save_or_show(fig, name, mode, output_dir):
    output_dir = Path(output_dir or ".")
    output_dir.mkdir(parents=True, exist_ok=True)
    if mode == "paper":
        path = output_dir / f"{name}.pdf"
        fig.savefig(path, format="pdf")
    else:
        path = output_dir / f"{name}.png"
        fig.savefig(path, format="png", bbox_inches="tight")
    print(f"  wrote {path}")
    if mode == "display":
        plt.show()
    plt.close(fig)


# ============================================================================
# Plot helpers
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


def log_log_fit(x, y):
    """Fit log10(y) = a * log10(x) + b. Returns (a, b) or None."""
    m = _finite_positive(x, y)
    if m.sum() < 3:
        return None
    lx = np.log10(x[m])
    ly = np.log10(y[m])
    a, b = np.polyfit(lx, ly, 1)
    return float(a), float(b)


def binned_median(x, y, n_bins=24, log_x=True):
    """Median y in bins of x. Returns (centers, medians, p25, p75, counts)."""
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
    """Indices of Pareto-optimal points (want high x, low y by default)."""
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
        linewidth=0.8,
        linestyle="--",
        zorder=1,
        label="y = x",
    )


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


def annotate_stats(ax, text_lines, loc="upper left", fontsize=None):
    if fontsize is None:
        fontsize = mpl.rcParams["font.size"] - 1
    txt = "\n".join(text_lines)
    ax.text(
        0.02,
        0.98,
        txt,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=fontsize,
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            alpha=0.85,
            edgecolor=PALETTE["neutral"],
            linewidth=0.4,
        ),
    )


# ============================================================================
# Print summary
# ============================================================================


def print_summary(data, dataset_name, mode_name="forward_triangle"):
    def _stat(name, a, fmt=".3e"):
        m = _finite(a)
        if m.sum() == 0:
            return f"  {name:<24} (no data)"
        s = a[m]
        return (
            f"  {name:<24} n={len(s):<6d}  "
            f"min={s.min():{fmt}}  med={np.median(s):{fmt}}  "
            f"mean={s.mean():{fmt}}  max={s.max():{fmt}}"
        )

    n = len(data["n_tri"])
    print(f"\n=== {dataset_name}  [{mode_name}] ===")
    print(f"  Items                  : {n}")
    print(f"  Total primitives       : {int(data['n_tri'].sum()):,}")
    print(f"  Total queries          : {int(data['n_queries'].sum()):,}")
    print()

    # Common metrics
    for name, key in [
        ("n_tri", "n_tri"),
        ("brute_force_ms", "time_brute_ms"),
        ("fast_ms", "time_fast_ms"),
        ("speedup", "speedup"),
        ("rms_abs", "rms_abs"),
        ("p50_abs", "p50_abs"),
        ("p95_abs", "p95_abs"),
        ("p99_abs", "p99_abs"),
        ("max_abs", "max_abs"),
    ]:
        print(_stat(name, data[key]))

    # Forward-specific
    if "misclass_count" in data:
        print(_stat("misclass_count", data["misclass_count"]))
        print(_stat("misclass_frac", data["misclass_frac"]))

    # Backward-specific
    if "global_cosine" in data:
        print(_stat("global_cosine", data["global_cosine"], ".6f"))
        print(_stat("global_rel_norm", data["global_rel_norm"]))
        print(_stat("per_vec_cos_p01", data["per_vec_cos_p01"], ".6f"))
        print(_stat("per_vec_cos_min", data["per_vec_cos_min"], ".6f"))
    print()


# ============================================================================
# Figures
# ============================================================================


def fig_overview(data, mode, dataset_name, out):
    fig, axes = plt.subplots(2, 3, figsize=figsize(mode, "grid23"))

    # 1. Speedup histogram (log-x)
    ax = axes[0, 0]
    sp = data["speedup"]
    sp = sp[np.isfinite(sp) & (sp > 0)]
    if sp.size:
        bins = np.logspace(np.log10(sp.min()), np.log10(sp.max()), 40)
        ax.hist(sp, bins=bins, color=PALETTE["primary"], alpha=0.85)
        ax.axvline(
            np.median(sp),
            color=PALETTE["highlight"],
            lw=1.2,
            linestyle="--",
            label=f"median = {np.median(sp):.2f}×",
        )
        ax.set_xscale("log")
        ax.legend(loc="upper right")
    ax.set_xlabel("Speedup (brute / fast)")
    ax.set_ylabel("Mesh count")
    ax.set_title("(a) Speedup distribution")

    # 2. RMS histogram (log-x)
    ax = axes[0, 1]
    e = data["rms_abs"]
    e = e[np.isfinite(e) & (e > 0)]
    if e.size:
        bins = np.logspace(np.log10(e.min()), np.log10(e.max()), 40)
        ax.hist(e, bins=bins, color=PALETTE["secondary"], alpha=0.85)
        ax.axvline(
            np.median(e),
            color=PALETTE["highlight"],
            lw=1.2,
            linestyle="--",
            label=f"median = {np.median(e):.2e}",
        )
        ax.set_xscale("log")
        ax.legend(loc="upper right")
    ax.set_xlabel("RMS absolute error")
    ax.set_ylabel("Mesh count")
    ax.set_title("(b) RMS error distribution")

    # 3. Misclass bar
    ax = axes[0, 2]
    mc = data["misclass_count"]
    mc = mc[np.isfinite(mc)]
    if mc.size:
        bins = [0, 0.5, 5, 50, 500, np.inf]
        labels = ["0", "1–5", "6–50", "51–500", ">500"]
        counts = [(mc >= bins[i]) & (mc < bins[i + 1]) for i in range(5)]
        vals = [int(c.sum()) for c in counts]
        colors = [
            PALETTE["tertiary"],
            PALETTE["primary"],
            PALETTE["accent"],
            PALETTE["quaternary"],
            PALETTE["secondary"],
        ]
        ax.bar(range(5), vals, color=colors, alpha=0.85)
        ax.set_xticks(range(5))
        ax.set_xticklabels(labels)
        for i, v in enumerate(vals):
            if v:
                ax.text(i, v, str(v), ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Mesh count")
    ax.set_xlabel("Misclassifications per mesh")
    ax.set_title("(c) Misclassification bins")

    # 4. n_tri vs speedup
    ax = axes[1, 0]
    nt = data["n_tri"]
    if sp.size and nt.size:
        m = _finite_positive(nt, data["speedup"])
        ax.scatter(
            nt[m],
            data["speedup"][m],
            s=4,
            alpha=0.25,
            color=PALETTE["primary"],
            edgecolors="none",
        )
        c, med, _, _, _ = binned_median(nt, data["speedup"])
        if c.size:
            ax.plot(c, med, color=PALETTE["highlight"], lw=1.2, label="binned median")
        ax.set_xscale("log")
        ax.set_yscale("log")
        format_log_axis(ax)
        ax.legend(loc="upper left")
    ax.axhline(1.0, color=PALETTE["neutral"], lw=0.8, linestyle=":")
    ax.set_xlabel("Triangle count")
    ax.set_ylabel("Speedup")
    ax.set_title("(d) Speedup vs complexity")

    # 5. brute vs fast
    ax = axes[1, 1]
    b = data["time_brute_ms"]
    f = data["time_fast_ms"]
    m = _finite_positive(b, f)
    if m.sum():
        ax.scatter(
            b[m], f[m], s=4, alpha=0.25, color=PALETTE["primary"], edgecolors="none"
        )
        add_identity_line(ax, b[m], f[m])
        ax.set_xscale("log")
        ax.set_yscale("log")
        format_log_axis(ax)
        ax.legend(loc="upper left")
    ax.set_xlabel("Brute force time (ms)")
    ax.set_ylabel("Fast engine time (ms)")
    ax.set_title("(e) Brute force vs fast engine")

    # 6. Speedup vs error
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
    ax.set_xlabel("Speedup")
    ax.set_ylabel("RMS absolute error")
    ax.set_title("(f) Accuracy–performance trade-off")

    fig.suptitle(
        f"{dataset_name}: overview", y=1.00, fontsize=mpl.rcParams["font.size"] + 2
    )
    fig.tight_layout()
    save_or_show(fig, "01_overview", mode, out)


def fig_timing(data, mode, dataset_name, out):
    fig, axes = plt.subplots(1, 3, figsize=figsize(mode, "wide"))

    nt = data["n_tri"]

    # Panel A: brute force vs n_tri
    ax = axes[0]
    b = data["time_brute_ms"]
    m = _finite_positive(nt, b)
    if m.sum():
        ax.scatter(
            nt[m], b[m], s=3, alpha=0.2, color=PALETTE["secondary"], edgecolors="none"
        )
        c, med, _, _, _ = binned_median(nt, b)
        if c.size:
            ax.plot(c, med, color=PALETTE["highlight"], lw=1.2)
        fit = log_log_fit(nt, b)
        if fit:
            a, lb = fit
            xfit = np.array([nt[m].min(), nt[m].max()])
            ax.plot(
                xfit,
                10 ** (a * np.log10(xfit) + lb),
                color=PALETTE["secondary"],
                lw=1.0,
                linestyle="--",
                label=f"slope = {a:.2f}",
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        format_log_axis(ax)
        ax.legend(loc="upper left")
    ax.set_xlabel("Triangle count")
    ax.set_ylabel("Brute force time (ms)")
    ax.set_title("(a) Brute force scaling")

    # Panel B: fast engine vs n_tri
    ax = axes[1]
    f = data["time_fast_ms"]
    m = _finite_positive(nt, f)
    if m.sum():
        ax.scatter(
            nt[m], f[m], s=3, alpha=0.2, color=PALETTE["primary"], edgecolors="none"
        )
        c, med, _, _, _ = binned_median(nt, f)
        if c.size:
            ax.plot(c, med, color=PALETTE["highlight"], lw=1.2)
        fit = log_log_fit(nt, f)
        if fit:
            a, lb = fit
            xfit = np.array([nt[m].min(), nt[m].max()])
            ax.plot(
                xfit,
                10 ** (a * np.log10(xfit) + lb),
                color=PALETTE["primary"],
                lw=1.0,
                linestyle="--",
                label=f"slope = {a:.2f}",
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        format_log_axis(ax)
        ax.legend(loc="upper left")
    ax.set_xlabel("Triangle count")
    ax.set_ylabel("Fast engine time (ms)")
    ax.set_title("(b) Fast engine scaling")

    # Panel C: speedup vs n_tri with regression
    ax = axes[2]
    sp = data["speedup"]
    m = _finite_positive(nt, sp)
    if m.sum():
        ax.scatter(
            nt[m], sp[m], s=3, alpha=0.2, color=PALETTE["primary"], edgecolors="none"
        )
        c, med, _, _, _ = binned_median(nt, sp)
        if c.size:
            ax.plot(c, med, color=PALETTE["highlight"], lw=1.4, label="binned median")
        fit = log_log_fit(nt, sp)
        if fit:
            a, lb = fit
            xfit = np.array([nt[m].min(), nt[m].max()])
            ax.plot(
                xfit,
                10 ** (a * np.log10(xfit) + lb),
                color=PALETTE["primary"],
                lw=1.0,
                linestyle="--",
                label=f"slope = {a:.2f}",
            )
        ax.axhline(1.0, color=PALETTE["neutral"], lw=0.8, linestyle=":")
        ax.set_xscale("log")
        ax.set_yscale("log")
        format_log_axis(ax)
        ax.legend(loc="upper left")
    ax.set_xlabel("Triangle count")
    ax.set_ylabel("Speedup")
    ax.set_title("(c) Speedup vs complexity")

    fig.suptitle(
        f"{dataset_name}: timing", y=1.02, fontsize=mpl.rcParams["font.size"] + 2
    )
    fig.tight_layout()
    save_or_show(fig, "02_timing", mode, out)


def fig_speedup(data, mode, dataset_name, out):
    fig, axes = plt.subplots(1, 3, figsize=figsize(mode, "wide"))

    sp = data["speedup"]
    sp_pos = sp[np.isfinite(sp) & (sp > 0)]

    # Panel A: histogram with median
    ax = axes[0]
    if sp_pos.size:
        bins = np.logspace(np.log10(sp_pos.min()), np.log10(sp_pos.max()), 40)
        ax.hist(sp_pos, bins=bins, color=PALETTE["primary"], alpha=0.85)
        ax.axvline(
            np.median(sp_pos),
            color=PALETTE["highlight"],
            lw=1.2,
            linestyle="--",
            label=f"median = {np.median(sp_pos):.2f}×",
        )
        ax.axvline(1.0, color=PALETTE["neutral"], lw=0.8, linestyle=":", label="parity")
        ax.set_xscale("log")
        ax.legend(loc="upper right")
    ax.set_xlabel("Speedup")
    ax.set_ylabel("Mesh count")
    ax.set_title("(a) Speedup histogram")

    # Panel B: ECDF
    ax = axes[1]
    if sp_pos.size:
        x, y = ecdf(sp_pos)
        ax.plot(x, y, color=PALETTE["primary"], lw=1.4)
        ax.axvline(1.0, color=PALETTE["neutral"], lw=0.8, linestyle=":")
        for q, c in [
            (0.25, PALETTE["tertiary"]),
            (0.5, PALETTE["primary"]),
            (0.75, PALETTE["secondary"]),
            (0.95, PALETTE["quaternary"]),
        ]:
            v = np.quantile(sp_pos, q)
            ax.axvline(v, color=c, lw=0.8, linestyle="--", alpha=0.7)
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
    ax.set_xlabel("Speedup")
    ax.set_ylabel("Fraction of meshes ≤ x")
    ax.set_title("(b) Speedup ECDF")
    ax.set_ylim(0, 1.02)

    # Panel C: fraction of meshes faster than x
    ax = axes[2]
    if sp_pos.size:
        n_faster = np.array(
            [
                (sp_pos >= x).mean()
                for x in np.logspace(
                    np.log10(sp_pos.min()), np.log10(sp_pos.max()), 100
                )
            ]
        )
        xs = np.logspace(np.log10(sp_pos.min()), np.log10(sp_pos.max()), 100)
        ax.plot(xs, n_faster, color=PALETTE["primary"], lw=1.4)
        ax.set_xscale("log")
        format_log_axis(ax, which="x")
        for thresh in [1, 2, 5, 10, 50]:
            frac = (sp_pos >= thresh).mean()
            ax.axvline(thresh, color=PALETTE["neutral"], lw=0.6, linestyle=":")
            ax.text(
                thresh,
                frac,
                f"{100 * frac:.0f}%",
                fontsize=7,
                ha="left",
                va="bottom",
                color=PALETTE["neutral"],
            )
    ax.set_xlabel("Speedup threshold")
    ax.set_ylabel("Fraction of meshes ≥ threshold")
    ax.set_title("(c) Fraction faster than threshold")
    ax.set_ylim(0, 1.02)

    fig.suptitle(
        f"{dataset_name}: speedup", y=1.02, fontsize=mpl.rcParams["font.size"] + 2
    )
    fig.tight_layout()
    save_or_show(fig, "03_speedup", mode, out)


def fig_accuracy(data, mode, dataset_name, out):
    fig, axes = plt.subplots(2, 2, figsize=figsize(mode, "grid22"))

    err_keys = [
        ("p50_abs", PALETTE["tertiary"], "p50"),
        ("p95_abs", PALETTE["primary"], "p95"),
        ("p99_abs", PALETTE["accent"], "p99"),
        ("max_abs", PALETTE["secondary"], "max"),
    ]

    # Panel A: histograms of each percentile
    ax = axes[0, 0]
    for key, color, label in err_keys:
        v = data[key]
        v = v[np.isfinite(v) & (v > 0)]
        if v.size == 0:
            continue
        bins = np.logspace(np.log10(v.min()), np.log10(v.max()), 30)
        ax.hist(
            v,
            bins=bins,
            histtype="step",
            linewidth=1.1,
            color=color,
            label=label,
            density=True,
        )
    ax.set_xscale("log")
    ax.legend(loc="upper right", ncol=2)
    ax.set_xlabel("Absolute error")
    ax.set_ylabel("Density")
    ax.set_title("(a) Error percentile distributions")

    # Panel B: ECDF of each percentile
    ax = axes[0, 1]
    for key, color, label in err_keys:
        v = data[key]
        v = v[np.isfinite(v) & (v > 0)]
        if v.size == 0:
            continue
        x, y = ecdf(v)
        ax.plot(x, y, color=color, lw=1.3, label=label)
    ax.set_xscale("log")
    format_log_axis(ax, which="x")
    ax.legend(loc="lower right")
    ax.set_xlabel("Absolute error")
    ax.set_ylabel("Fraction of meshes ≤ x")
    ax.set_title("(b) Error percentile ECDFs")
    ax.set_ylim(0, 1.02)

    # Panel C: RMS distribution with fit
    ax = axes[1, 0]
    e = data["rms_abs"]
    e = e[np.isfinite(e) & (e > 0)]
    if e.size:
        bins = np.logspace(np.log10(e.min()), np.log10(e.max()), 40)
        ax.hist(e, bins=bins, color=PALETTE["primary"], alpha=0.85)
        med = np.median(e)
        mean = e.mean()
        ax.axvline(
            med,
            color=PALETTE["highlight"],
            lw=1.2,
            linestyle="--",
            label=f"median = {med:.2e}",
        )
        ax.axvline(
            mean,
            color=PALETTE["secondary"],
            lw=1.2,
            linestyle=":",
            label=f"mean = {mean:.2e}",
        )
        ax.set_xscale("log")
        ax.legend(loc="upper right")
    ax.set_xlabel("RMS absolute error")
    ax.set_ylabel("Mesh count")
    ax.set_title("(c) RMS error distribution")

    # Panel D: relative error
    ax = axes[1, 1]
    r = data["rms_rel_masked"]
    r = r[np.isfinite(r) & (r > 0)]
    if r.size:
        bins = np.logspace(np.log10(r.min()), np.log10(r.max()), 40)
        ax.hist(r, bins=bins, color=PALETTE["quaternary"], alpha=0.85)
        ax.axvline(
            np.median(r),
            color=PALETTE["highlight"],
            lw=1.2,
            linestyle="--",
            label=f"median = {np.median(r):.2e}",
        )
        ax.set_xscale("log")
        ax.legend(loc="upper right")
    ax.set_xlabel("Masked RMS relative error")
    ax.set_ylabel("Mesh count")
    ax.set_title("(d) Relative error distribution")

    fig.suptitle(
        f"{dataset_name}: accuracy", y=1.00, fontsize=mpl.rcParams["font.size"] + 2
    )
    fig.tight_layout()
    save_or_show(fig, "04_accuracy", mode, out)


def fig_error_scaling(data, mode, dataset_name, out):
    fig, axes = plt.subplots(1, 3, figsize=figsize(mode, "wide"))
    nt = data["n_tri"]

    # Panel A: RMS vs n_tri
    ax = axes[0]
    for key, color, label in [
        ("rms_abs", PALETTE["primary"], "RMS"),
        ("p99_abs", PALETTE["accent"], "p99"),
        ("max_abs", PALETTE["secondary"], "max"),
    ]:
        y = data[key]
        m = _finite_positive(nt, y)
        if m.sum() == 0:
            continue
        ax.scatter(
            nt[m], y[m], s=3, alpha=0.15, color=color, edgecolors="none", label=label
        )
        c, med, _, _, _ = binned_median(nt, y)
        if c.size:
            ax.plot(c, med, color=color, lw=1.2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    format_log_axis(ax)
    ax.legend(loc="upper left")
    ax.set_xlabel("Triangle count")
    ax.set_ylabel("Absolute error")
    ax.set_title("(a) Error vs complexity")

    # Panel B: RMS vs brute_force time (error vs compute)
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
            ax.plot(c, med, color=PALETTE["highlight"], lw=1.4)
        ax.set_xscale("log")
        ax.set_yscale("log")
        format_log_axis(ax)
    ax.set_xlabel("Brute force time (ms)")
    ax.set_ylabel("RMS absolute error")
    ax.set_title("(b) Error vs brute force cost")

    # Panel C: heatmap of error vs (n_tri, fast_ms)
    ax = axes[2]
    f = data["time_fast_ms"]
    m = _finite_positive(nt, f, data["rms_abs"])
    if m.sum() > 20:
        x = np.log10(nt[m])
        y = np.log10(f[m])
        c = np.log10(data["rms_abs"][m])
        h = ax.hexbin(
            x,
            y,
            C=c,
            reduce_C_function=np.median,
            gridsize=30,
            cmap="viridis",
            mincnt=2,
        )
        plt.colorbar(h, ax=ax, label="log₁₀ RMS error")
        ax.set_xlabel("log₁₀ triangle count")
        ax.set_ylabel("log₁₀ fast engine time (ms)")
    ax.set_title("(c) Error across (complexity, runtime)")

    fig.suptitle(
        f"{dataset_name}: error scaling", y=1.02, fontsize=mpl.rcParams["font.size"] + 2
    )
    fig.tight_layout()
    save_or_show(fig, "05_error_scaling", mode, out)


def fig_misclassification(data, mode, dataset_name, out):
    fig, axes = plt.subplots(1, 3, figsize=figsize(mode, "wide"))

    mc = data["misclass_count"]
    mf = data["misclass_frac"]
    nt = data["n_tri"]

    # Panel A: bar bins
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
        ax.bar(range(5), vals, color=colors, alpha=0.85)
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
        ax.set_ylim(0, max(vals) * 1.25)
    ax.set_xlabel("Misclassifications per mesh")
    ax.set_ylabel("Mesh count")
    ax.set_title("(a) Misclassification bins")

    # Panel B: ECDF of misclass fraction
    ax = axes[1]
    mf_v = mf[np.isfinite(mf) & (mf >= 0)]
    if mf_v.size:
        # Handle zeros in the ECDF on a log axis by adding a floor
        floor = 0.5 / max(1, int(data["n_queries"][0]))
        x, y = ecdf(mf_v + floor * (mf_v == 0))
        ax.plot(x, y, color=PALETTE["primary"], lw=1.4)
        ax.set_xscale("log")
        format_log_axis(ax, which="x")
    ax.set_xlabel("Misclassification fraction")
    ax.set_ylabel("Fraction of meshes ≤ x")
    ax.set_title("(b) Misclassification fraction ECDF")
    ax.set_ylim(0, 1.02)

    # Panel C: misclass vs n_tri
    ax = axes[2]
    m = _finite(nt, mc) & (nt > 0)
    if m.sum():
        jitter = 10 ** (0.01 * np.random.default_rng(0).standard_normal(m.sum()))
        ax.scatter(
            nt[m],
            mc[m] + jitter,
            s=4,
            alpha=0.3,
            color=PALETTE["primary"],
            edgecolors="none",
        )
        ax.set_xscale("log")
        format_log_axis(ax, which="x")
        if (mc[m] > 0).any():
            ax.set_yscale("symlog", linthresh=1)
    ax.set_xlabel("Triangle count")
    ax.set_ylabel("Misclassifications")
    ax.set_title("(c) Misclass vs complexity")

    fig.suptitle(
        f"{dataset_name}: misclassification",
        y=1.02,
        fontsize=mpl.rcParams["font.size"] + 2,
    )
    fig.tight_layout()
    save_or_show(fig, "06_misclassification", mode, out)


def fig_tradeoff(data, mode, dataset_name, out):
    w, h = figsize(mode, "wide")
    fig, axes = plt.subplots(1, 2, figsize=(2 * w, 1.6 * h))

    sp = data["speedup"]
    e = data["rms_abs"]
    m = _finite_positive(sp, e)

    # Panel A: scatter, colored by n_tri
    ax = axes[0]
    if m.sum():
        sc = ax.scatter(
            sp[m],
            e[m],
            c=np.log10(data["n_tri"][m]),
            s=5,
            alpha=0.6,
            cmap="viridis",
            edgecolors="none",
        )
        plt.colorbar(sc, ax=ax, label="log₁₀ triangle count")
        idx = pareto_front(sp[m], e[m])
        ss = sp[m][idx]
        ee = e[m][idx]
        order = np.argsort(ss)
        ax.plot(
            ss[order],
            ee[order],
            color=PALETTE["highlight"],
            lw=1.4,
            label="Pareto frontier",
        )
        ax.legend(loc="lower left")
        ax.set_xscale("log")
        ax.set_yscale("log")
        format_log_axis(ax)
    ax.set_xlabel("Speedup")
    ax.set_ylabel("RMS absolute error")
    ax.set_title("(a) Trade-off frontier")

    # Panel B: normalized joint quality (speedup / error)
    ax = axes[1]
    if m.sum():
        q = sp[m] / e[m]
        q = q[np.isfinite(q) & (q > 0)]
        if q.size:
            bins = np.logspace(np.log10(q.min()), np.log10(q.max()), 40)
            ax.hist(q, bins=bins, color=PALETTE["quaternary"], alpha=0.85)
            med = np.median(q)
            ax.axvline(
                med,
                color=PALETTE["highlight"],
                lw=1.2,
                linestyle="--",
                label=f"median = {med:.2e}",
            )
            ax.set_xscale("log")
            ax.legend(loc="upper right")
    ax.set_xlabel("Speedup / RMS error")
    ax.set_ylabel("Mesh count")
    ax.set_title("(b) Joint quality distribution")

    fig.suptitle(
        f"{dataset_name}: accuracy–performance trade-off",
        y=1.02,
        fontsize=mpl.rcParams["font.size"] + 2,
    )
    fig.tight_layout()
    save_or_show(fig, "07_tradeoff", mode, out)


def fig_paper_dashboard(data, mode, dataset_name, out):
    """Compact single-figure summary suitable for a paper."""
    fig, axes = plt.subplots(2, 2, figsize=figsize(mode, "grid22"))

    sp = data["speedup"]
    e = data["rms_abs"]
    nt = data["n_tri"]
    mc = data["misclass_count"]

    # Top-left: speedup ECDF
    ax = axes[0, 0]
    sp_v = sp[np.isfinite(sp) & (sp > 0)]
    if sp_v.size:
        x, y = ecdf(sp_v)
        ax.plot(x, y, color=PALETTE["primary"], lw=1.3)
        ax.axvline(1.0, color=PALETTE["neutral"], lw=0.7, linestyle=":")
        med = np.median(sp_v)
        ax.axvline(
            med,
            color=PALETTE["highlight"],
            lw=1.0,
            linestyle="--",
            label=f"median {med:.2f}×",
        )
        ax.legend(loc="lower right")
        ax.set_xscale("log")
        format_log_axis(ax, which="x")
    ax.set_xlabel("Speedup")
    ax.set_ylabel("Fraction of meshes")
    ax.set_title("(a) Speedup distribution")
    ax.set_ylim(0, 1.02)

    # Top-right: RMS ECDF
    ax = axes[0, 1]
    e_v = e[np.isfinite(e) & (e > 0)]
    if e_v.size:
        x, y = ecdf(e_v)
        ax.plot(x, y, color=PALETTE["secondary"], lw=1.3)
        med = np.median(e_v)
        ax.axvline(
            med,
            color=PALETTE["highlight"],
            lw=1.0,
            linestyle="--",
            label=f"median {med:.2e}",
        )
        ax.legend(loc="lower right")
        ax.set_xscale("log")
        format_log_axis(ax, which="x")
    ax.set_xlabel("RMS absolute error")
    ax.set_ylabel("Fraction of meshes")
    ax.set_title("(b) Error distribution")
    ax.set_ylim(0, 1.02)

    # Bottom-left: speedup vs n_tri
    ax = axes[1, 0]
    m = _finite_positive(nt, sp)
    if m.sum():
        ax.scatter(
            nt[m], sp[m], s=3, alpha=0.2, color=PALETTE["primary"], edgecolors="none"
        )
        c, med, p25, p75, _ = binned_median(nt, sp)
        if c.size:
            ax.fill_between(c, p25, p75, color=PALETTE["primary"], alpha=0.2)
            ax.plot(c, med, color=PALETTE["primary"], lw=1.2)
        ax.axhline(1.0, color=PALETTE["neutral"], lw=0.7, linestyle=":")
        ax.set_xscale("log")
        ax.set_yscale("log")
        format_log_axis(ax)
    ax.set_xlabel("Triangle count")
    ax.set_ylabel("Speedup")
    ax.set_title("(c) Speedup vs complexity")

    # Bottom-right: misclassification bin
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
        bars = ax.bar(range(5), vals, color=colors, alpha=0.85)
        ax.set_xticks(range(5))
        ax.set_xticklabels(labels)
        # annotate percentages
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
    ax.set_xlabel("Misclassifications per mesh")
    ax.set_ylabel("Mesh count")
    ax.set_title("(d) Misclassification summary")

    fig.suptitle(dataset_name, y=1.00, fontsize=mpl.rcParams["font.size"] + 2)
    fig.tight_layout()
    save_or_show(fig, "08_paper_dashboard", mode, out)


def fig_gradient_quality(data, mode, dataset_name, out):
    """Distribution of per-primitive directional and magnitude errors."""
    fig, axes = plt.subplots(2, 2, figsize=figsize(mode, "grid22"))

    # ---- Panel A: per-primitive angular error, ECDF over the distribution
    # of *within-item* percentiles. Each x value is one item (mesh); each
    # y value is the fraction of items whose pXX angular error is <= x.
    ax = axes[0, 0]
    ang_keys = [
        ("per_vec_ang_p50", PALETTE["tertiary"], "p50"),
        ("per_vec_ang_p90", PALETTE["primary"], "p90"),
        ("per_vec_ang_p99", PALETTE["accent"], "p99"),
        ("per_vec_ang_max", PALETTE["secondary"], "max"),
    ]
    for key, color, label in ang_keys:
        v = data.get(key, np.array([]))
        v = v[np.isfinite(v) & (v >= 0)]
        if v.size:
            x, y = ecdf(v)
            ax.plot(x, y, color=color, lw=1.3, label=label)
    ax.set_xscale("log")
    format_log_axis(ax, which="x")
    ax.legend(loc="lower right")
    ax.set_xlabel("Per-primitive angular error (degrees)")
    ax.set_ylabel("Fraction of items ≤ x")
    ax.set_title("(a) Angular error, within-item percentiles")
    ax.set_ylim(0, 1.02)

    # ---- Panel B: distribution of the *median* per-primitive angular
    # error across items. Reads as: "the median item's median primitive
    # gradient is off by X degrees."
    ax = axes[0, 1]
    v = data.get("per_vec_ang_p50", np.array([]))
    v = v[np.isfinite(v) & (v >= 0)]
    if v.size:
        # Histogram with a floor to show the mass at ~0.1°
        bins = np.logspace(
            np.log10(max(v.min(), 1e-3)), np.log10(max(v.max(), 1e-2)), 40
        )
        ax.hist(v, bins=bins, color=PALETTE["primary"], alpha=0.85)
        med = float(np.median(v))
        ax.axvline(
            med,
            color=PALETTE["highlight"],
            lw=1.2,
            linestyle="--",
            label=f"median = {med:.3f}°",
        )
        p95 = float(np.percentile(v, 95))
        ax.axvline(
            p95,
            color=PALETTE["secondary"],
            lw=1.2,
            linestyle=":",
            label=f"p95 = {p95:.3f}°",
        )
        ax.set_xscale("log")
        ax.legend(loc="upper right")
    ax.set_xlabel("Median per-primitive angular error (deg)")
    ax.set_ylabel("Item count")
    ax.set_title("(b) Item-wise median angular error")

    # ---- Panel C: magnitude ratio distribution. Values near 1.0 mean
    # fast and ref agree in magnitude.
    ax = axes[1, 0]
    for key, color, label in [
        ("mag_ratio_p50", PALETTE["tertiary"], "p50"),
        ("mag_ratio_p90", PALETTE["primary"], "p90"),
        ("mag_ratio_p99", PALETTE["accent"], "p99"),
    ]:
        v = data.get(key, np.array([]))
        v = v[np.isfinite(v) & (v > 0)]
        if v.size:
            x, y = ecdf(v)
            ax.plot(x, y, color=color, lw=1.3, label=label)
    ax.axvline(1.0, color=PALETTE["neutral"], lw=0.8, linestyle=":", label="perfect")
    ax.set_xscale("log")
    format_log_axis(ax, which="x")
    ax.legend(loc="upper left")
    ax.set_xlabel("Magnitude ratio ‖fast‖ / ‖ref‖")
    ax.set_ylabel("Fraction of items ≤ x")
    ax.set_title("(c) Magnitude ratio, within-item percentiles")
    ax.set_ylim(0, 1.02)

    # ---- Panel D: sanity check — global cosine vs per-vector p99. If
    # global cosine is high but per-vector p99 is low, the global metric
    # is being dominated by large-magnitude entries.
    ax = axes[1, 1]
    gc = data.get("global_cosine", np.array([]))
    pv99 = data.get("per_vec_cos_p99", np.array([]))
    m = _finite(gc, pv99)
    if m.sum():
        ax.scatter(
            pv99[m], gc[m], s=5, alpha=0.4, color=PALETTE["primary"], edgecolors="none"
        )
        ax.set_xlabel("Per-primitive cosine, p99")
        ax.set_ylabel("Global cosine similarity")
    ax.set_title("(d) Global vs per-primitive agreement")
    # Diagonal reference
    ax.plot([0, 1], [0, 1], color=PALETTE["neutral"], lw=0.7, linestyle=":")

    fig.suptitle(
        f"{dataset_name}: gradient quality",
        y=1.00,
        fontsize=mpl.rcParams["font.size"] + 2,
    )
    fig.tight_layout()
    save_or_show(fig, "09_gradient_quality", mode, out)


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
        help="Style preset. 'paper' saves PDFs at 300 DPI, "
        "'display' saves PNGs and shows interactively.",
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
        help="Dataset name shown in figure titles. Defaults to the JSONL file stem.",
    )
    parser.add_argument(
        "--figures",
        type=str,
        default="all",
        help="Comma-separated list of figure names, or 'all'. "
        "Options: overview, timing, speedup, accuracy, "
        "error_scaling, misclassification, tradeoff, "
        "paper_dashboard",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="RNG seed for reproducibility (jitter)."
    )
    parser.add_argument(
        "--mode-name",
        type=str,
        default="forward_triangle",
        help="Which evaluation mode to plot when the JSONL uses the nested "
        "schema. Options: forward_triangle, forward_point_normal, "
        "backward_triangle, backward_mesh, backward_point_normal.",
    )

    args = parser.parse_args()

    np.random.seed(args.seed)
    setup_style(args.mode)

    input_path = Path(args.input)
    dataset_name = args.title or input_path.stem

    print(f"Loading {input_path} ...")
    records, skipped = load_records(input_path)
    print(f"  Loaded {len(records)} valid records ({skipped} skipped malformed/error)")
    if not records:
        raise SystemExit("No valid records to plot.")

    data = to_arrays(records)
    dataset_label = f"{dataset_name} [{args.mode_name}]"
    print_summary(data, dataset_name, mode_name=args.mode_name)

    figures = (
        [k.strip() for k in args.figures.split(",")]
        if args.figures != "all"
        else [
            "overview",
            "timing",
            "speedup",
            "accuracy",
            "gradient_quality",
            "error_scaling",
            "misclassification",
            "tradeoff",
            "paper_dashboard",
        ]
    )
    dispatch = {
        "overview": fig_overview,
        "timing": fig_timing,
        "speedup": fig_speedup,
        "accuracy": fig_accuracy,
        "gradient_quality": fig_gradient_quality,
        "error_scaling": fig_error_scaling,
        "misclassification": fig_misclassification,
        "tradeoff": fig_tradeoff,
        "paper_dashboard": fig_paper_dashboard,
    }

    print(f"\nWriting figures to {args.output} ...")
    for name in figures:
        if name not in dispatch:
            print(f"  skipping unknown figure '{name}'")
            continue
        print(f"  [{name}]")
        dispatch[name](data, args.mode, dataset_name, args.output)

    print("\nDone.")


if __name__ == "__main__":
    main()
