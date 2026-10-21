"""Matplotlib figures shared by the report scripts and the Streamlit dashboard.

Design rules (a validated, colour-blind-safe palette; see README "Figures"):

* failures are drawn in the reserved "critical" red against a muted grey for ordinary
  records (emphasis form: one series is the point, the rest is context);
* model series use fixed accent hues in a fixed order (blue, orange, aqua) and always
  come with a legend; single-series bar charts use one hue and no legend;
* thin marks, solid hairline grids, no number on every point, text in ink colours;
* sequential data use one blue ramp, signed data a blue-grey-red diverging ramp.

All functions return a ``matplotlib.figure.Figure`` built without ``pyplot`` so they are
thread-safe inside Streamlit and need no GUI backend.
"""

from __future__ import annotations

import functools
import io
import textwrap
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, PercentFormatter
from sklearn.metrics import precision_recall_curve, roc_curve

from . import config as cfg
from .metrics import classification_metrics, threshold_table

# --------------------------------------------------------------------------- #
# Tokens
# --------------------------------------------------------------------------- #
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
CRITICAL = "#d03b3b"
DASHED = (0, (4, 3))
DPI = 160

BLUE_RAMP = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
SEQUENTIAL = LinearSegmentedColormap.from_list("blue_sequential", BLUE_RAMP)
DIVERGING = LinearSegmentedColormap.from_list("blue_red_diverging", [BLUE, "#f0efec", "#e34948"])

# Entity -> colour, fixed across every chart (colour follows the entity).
MODEL_COLORS = {"rf_engineered": BLUE, "rf_raw": ORANGE, "logreg_raw": AQUA}
MODEL_SHORT = {
    "baseline_majority": "Baseline (majority class)",
    "logreg_raw": "Logistic regression, raw",
    "rf_raw": "Random forest, raw",
    "rf_engineered": "Random forest, raw + engineered",
}

STYLE: dict[str, Any] = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK_2,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "grid.linestyle": "-",
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "text.color": INK,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "legend.frameon": False,
    "legend.fontsize": 8.5,
    "legend.labelcolor": INK_2,
    "lines.linewidth": 1.5,
    "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",
}


def styled(function: Callable[..., Figure]) -> Callable[..., Figure]:
    """Build the figure inside the project's matplotlib style context."""

    @functools.wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Figure:
        with mpl.rc_context(STYLE):
            return function(*args, **kwargs)

    return wrapper


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def save_figure(fig: Figure, path: Path | str, dpi: int = DPI) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.15)
    return path


def figure_png(fig: Figure, dpi: int = 130) -> bytes:
    """Render a figure to PNG bytes (what the dashboard shows; no pyplot, no files)."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.15)
    return buffer.getvalue()


def _heading(fig: Figure, title: str, subtitle: str | None = None) -> float:
    """Left-aligned title + wrapped subtitle; returns the top margin in inches."""
    width, height = fig.get_size_inches()
    fig.text(0.0, 1 - 0.10 / height, title, ha="left", va="top", fontsize=11.5, fontweight="bold", color=INK)
    margin = 0.50
    if subtitle:
        wrapped = textwrap.fill(subtitle, width=max(40, int(width * 15.5)))
        n_lines = wrapped.count("\n") + 1
        fig.text(0.0, 1 - 0.36 / height, wrapped, ha="left", va="top", fontsize=8.8, color=INK_2, linespacing=1.35)
        margin = 0.42 + 0.17 * n_lines + 0.12
    return margin


def _top(fig: Figure, margin_in: float) -> float:
    return 1 - margin_in / fig.get_size_inches()[1]


def _grid(ax: Axes, axis: str = "y") -> None:
    ax.grid(True, axis=axis, color=GRID, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)


def _thousands(axis: Any) -> None:
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))


def _luminance(color: Any) -> float:
    r, g, b = to_rgb(color)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _ink_on(color: Any) -> str:
    return INK if _luminance(color) > 0.55 else "#ffffff"


def _hbar(
    ax: Axes,
    labels: Sequence[str],
    values: Sequence[float],
    texts: Sequence[str],
    color: Any = BLUE,
    colors: Sequence[Any] | None = None,
    xmax: float | None = None,
    errors: Sequence[float] | None = None,
) -> None:
    """Thin horizontal bars from a single baseline with the value written at the tip."""
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, height=0.5, color=colors if colors is not None else color, linewidth=0)
    if errors is not None:
        ax.errorbar(values, y, xerr=errors, fmt="none", ecolor=INK_2, elinewidth=1.0, capsize=2.5, capthick=1.0)
    top = xmax or max(max(values) * 1.28, 1e-9)
    ax.set_xlim(0, top)
    for yi, value, text, error in zip(y, values, texts, errors if errors is not None else [0] * len(values)):
        ax.annotate(
            text, (value + error, yi), xytext=(5, 0), textcoords="offset points", va="center", ha="left",
            fontsize=8.5, color=INK_2,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.tick_params(axis="y", length=0)
    _grid(ax, "x")
    ax.spines["left"].set_visible(False)


def _line_label(ax: Axes, text: str, xy: tuple[float, float], offset: tuple[float, float], ha: str = "right", va: str = "bottom") -> None:
    """Direct label for a reference line, with a surface-coloured backing for legibility."""
    ax.annotate(
        text, xy, xytext=offset, textcoords="offset points", ha=ha, va=va, fontsize=8, color=INK_2,
        bbox={"boxstyle": "square,pad=0.15", "facecolor": SURFACE, "edgecolor": "none", "alpha": 0.9}, zorder=6,
    )


def _legend_markers(ax: Axes | Figure, **kwargs: Any) -> Any:
    handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=5, markerfacecolor=MUTED, markeredgecolor=MUTED, alpha=0.7, label="No failure"),
        Line2D([], [], marker="D", linestyle="none", markersize=6, markerfacecolor=CRITICAL, markeredgecolor=SURFACE, label="Machine failure"),
    ]
    return ax.legend(handles=handles, **kwargs)


def _draw_scatter(ax: Axes, df: pd.DataFrame, x: str, y: str) -> None:
    normal = df[df[cfg.TARGET] == 0]
    failed = df[df[cfg.TARGET] == 1]
    ax.scatter(normal[x], normal[y], s=4, color=MUTED, alpha=0.20, linewidths=0, rasterized=True)
    ax.scatter(
        failed[x], failed[y], s=24, marker="D", color=CRITICAL, edgecolors=SURFACE,
        linewidths=0.6, zorder=3,
    )
    ax.set_xlabel(cfg.FEATURE_LABELS.get(x, x))
    ax.set_ylabel(cfg.FEATURE_LABELS.get(y, y))
    _grid(ax, "both")


def _draw_distribution(ax: Axes, df: pd.DataFrame, column: str, bins: int = 30) -> None:
    values = df[column].to_numpy(dtype=float)
    edges = np.histogram_bin_edges(values, bins=bins)
    normal = df.loc[df[cfg.TARGET] == 0, column].to_numpy(dtype=float)
    failed = df.loc[df[cfg.TARGET] == 1, column].to_numpy(dtype=float)
    ax.hist(normal, bins=edges, density=True, histtype="stepfilled", color=MUTED, alpha=0.38, linewidth=0)
    ax.hist(normal, bins=edges, density=True, histtype="step", color=MUTED, linewidth=1.0)
    ax.hist(failed, bins=edges, density=True, histtype="stepfilled", color=CRITICAL, alpha=0.16, linewidth=0)
    ax.hist(failed, bins=edges, density=True, histtype="step", color=CRITICAL, linewidth=1.5)
    ax.set_xlabel(cfg.FEATURE_LABELS.get(column, column))
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    _grid(ax, "x")


def _distribution_legend(fig_or_ax: Any, **kwargs: Any) -> Any:
    handles = [
        Patch(facecolor=MUTED, alpha=0.5, edgecolor=MUTED, linewidth=1.0, label="No failure"),
        Patch(facecolor=CRITICAL, alpha=0.35, edgecolor=CRITICAL, linewidth=1.5, label="Machine failure"),
    ]
    return fig_or_ax.legend(handles=handles, **kwargs)


def with_engineered(df: pd.DataFrame) -> pd.DataFrame:
    """Copy of ``df`` with the three engineered features added (for plotting only)."""
    from .physics import mechanical_power_w, strain_index_min_nm, temperature_difference_k

    out = df.copy()
    out["temp_diff_k"] = temperature_difference_k(out["process_temp_k"], out["air_temp_k"])
    out["power_w"] = mechanical_power_w(out["torque_nm"], out["rotational_speed_rpm"])
    out["strain_min_nm"] = strain_index_min_nm(out["tool_wear_min"], out["torque_nm"])
    return out


# --------------------------------------------------------------------------- #
# Exploratory data analysis
# --------------------------------------------------------------------------- #
@styled
def plot_failure_modes(counts: Mapping[str, int], n_rows: int, n_failures: int, n_rnf_without_failure: int) -> Figure:
    order = sorted(cfg.FAILURE_MODE_COLUMNS, key=lambda m: -counts[m])
    fig = Figure(figsize=(7.2, 3.4))
    margin = _heading(
        fig,
        "Recorded failure modes",
        f"{n_failures} of {n_rows:,} records ({n_failures / n_rows:.2%}) have Machine failure = 1. "
        f"Mode flags can overlap; {n_rnf_without_failure} RNF flags carry no recorded machine failure.",
    )
    ax = fig.add_axes((0.30, 0.14, 0.62, _top(fig, margin) - 0.14 - 0.04))
    _hbar(
        ax,
        [cfg.FAILURE_MODE_LABELS[m] for m in order],
        [counts[m] for m in order],
        [f"{counts[m]}" for m in order],
        BLUE,
    )
    ax.set_xlabel("Records with the flag set")
    return fig


@styled
def plot_distributions(df: pd.DataFrame) -> Figure:
    data = with_engineered(df)
    columns = cfg.NUMERIC_SENSOR_FEATURES + ["power_w"]
    fig = Figure(figsize=(10.0, 5.6))
    n_fail = int(data[cfg.TARGET].sum())
    margin = _heading(
        fig,
        "Sensor distributions by outcome",
        f"Each class is normalised to equal area ({n_fail} failures vs {len(data) - n_fail:,} other records), "
        "so shapes are comparable. Mechanical power is derived from torque and speed.",
    )
    fig.subplots_adjust(top=_top(fig, margin) - 0.02, bottom=0.10, left=0.03, right=0.99, hspace=0.50, wspace=0.12)
    axes = fig.subplots(2, 3)
    for ax, column in zip(axes.ravel(), columns):
        _draw_distribution(ax, data, column)
        if column in ("power_w", "rotational_speed_rpm", "tool_wear_min"):
            _thousands(ax.xaxis)
    _distribution_legend(fig, loc="upper right", ncol=2, bbox_to_anchor=(0.99, 1 - 0.10 / fig.get_size_inches()[1]))
    return fig


@styled
def plot_distribution(df: pd.DataFrame, column: str) -> Figure:
    """Single distribution panel (dashboard explorer)."""
    data = with_engineered(df)
    fig = Figure(figsize=(6.4, 3.4))
    margin = _heading(fig, cfg.FEATURE_LABELS.get(column, column), "Each class normalised to equal area.")
    ax = fig.add_axes((0.04, 0.17, 0.94, _top(fig, margin) - 0.17 - 0.03))
    _draw_distribution(ax, data, column)
    _distribution_legend(ax, loc="upper right", ncol=1)
    return fig


@styled
def plot_scatter(df: pd.DataFrame, x: str, y: str) -> Figure:
    """Scatter of two features with failures emphasised (dashboard explorer)."""
    data = with_engineered(df)
    fig = Figure(figsize=(6.4, 4.4))
    margin = _heading(
        fig,
        f"{cfg.SHORT_LABELS.get(y, y)} vs {cfg.SHORT_LABELS.get(x, x)}",
        f"{int(data[cfg.TARGET].sum())} failures (red) among {len(data):,} records shown.",
    )
    ax = fig.add_axes((0.11, 0.12, 0.86, _top(fig, margin) - 0.12 - 0.02))
    _draw_scatter(ax, data, x, y)
    _legend_markers(ax, loc="upper right", ncol=2)
    return fig


@styled
def plot_failure_mechanisms(df: pd.DataFrame) -> Figure:
    """Three panels linking the sensor data to the documented failure criteria."""
    data = with_engineered(df)
    fig = Figure(figsize=(13.0, 4.7))
    margin = _heading(
        fig,
        "Failures against the documented failure criteria",
        "Grey: records without failure. Red diamonds: recorded machine failures. Thin lines mark the criteria "
        "documented by the dataset authors (power limits, overstrain limits per product variant, heat-dissipation region).",
    )
    fig.subplots_adjust(top=_top(fig, margin) - 0.05, bottom=0.14, left=0.055, right=0.99, wspace=0.22)
    ax_power, ax_strain, ax_heat = fig.subplots(1, 3)

    # (a) power envelope
    _draw_scatter(ax_power, data, "rotational_speed_rpm", "torque_nm")
    speeds = np.linspace(1150, 2900, 200)
    for power, side in ((cfg.POWER_MAX_W, "upper"), (cfg.POWER_MIN_W, "lower")):
        torque = power * 60.0 / (2 * np.pi * speeds)
        ax_power.plot(speeds, torque, color=INK_2, linewidth=1.0, zorder=2)
        _line_label(ax_power, f"{power:,.0f} W", (speeds[-1], torque[-1]), (-4, 8))
    ax_power.set_ylim(0, 85)
    ax_power.set_xlim(1100, 2950)
    _thousands(ax_power.xaxis)
    ax_power.set_title("Power limits", loc="left", fontsize=9.5, color=INK, pad=6)

    # (b) overstrain
    _draw_scatter(ax_strain, data, "tool_wear_min", "torque_nm")
    wear = np.linspace(110, 253, 200)
    for variant, limit in cfg.OSF_LIMITS_MIN_NM.items():
        torque = limit / wear
        ax_strain.plot(wear, torque, color=INK_2, linewidth=1.0, zorder=2)
        _line_label(ax_strain, f"Type {variant}", (wear[-1], torque[-1]), (4, 0), ha="left", va="center")
    ax_strain.set_ylim(0, 85)
    ax_strain.set_xlim(-5, 275)
    ax_strain.set_title("Overstrain limits (tool wear x torque)", loc="left", fontsize=9.5, color=INK, pad=6)

    # (c) heat dissipation
    _draw_scatter(ax_heat, data, "rotational_speed_rpm", "temp_diff_k")
    ax_heat.add_patch(
        mpl.patches.Rectangle(
            (1100, 7.0), cfg.HDF_SPEED_MAX_RPM - 1100, cfg.HDF_TEMP_DIFF_MAX_K - 7.0,
            facecolor=CRITICAL, alpha=0.08, edgecolor=INK_2, linewidth=1.0, zorder=1,
        )
    )
    ax_heat.annotate("heat-dissipation region", (1120, 7.1), xytext=(0, 3), textcoords="offset points", fontsize=8, color=INK_2)
    ax_heat.set_xlim(1100, 2950)
    ax_heat.set_ylim(7.0, 13.2)
    _thousands(ax_heat.xaxis)
    ax_heat.set_title("Heat dissipation criterion", loc="left", fontsize=9.5, color=INK, pad=6)

    _legend_markers(fig, loc="upper right", ncol=2, bbox_to_anchor=(0.99, 1 - 0.10 / fig.get_size_inches()[1]))
    return fig


@styled
def plot_failure_rates(df: pd.DataFrame) -> Figure:
    fig = Figure(figsize=(10.0, 3.9))
    margin = _heading(
        fig,
        "Failure rate by product variant and by tool wear",
        "Share of records with Machine failure = 1. Tool wear is grouped in 25-minute bins (225 min and above merged); n is the number of records.",
    )
    fig.subplots_adjust(top=_top(fig, margin) - 0.05, bottom=0.17, left=0.07, right=0.99, wspace=0.18)
    ax_type, ax_wear = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1, 2.4]})

    by_type = df.groupby("type")[cfg.TARGET].agg(["mean", "size"]).reindex(cfg.TYPE_VALUES)
    x = np.arange(len(by_type))
    ax_type.bar(x, by_type["mean"], width=0.5, color=BLUE, linewidth=0)
    for xi, rate in enumerate(by_type["mean"]):
        ax_type.annotate(f"{rate:.1%}", (xi, rate), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, color=INK_2)
    ax_type.set_xticks(x)
    ax_type.set_xticklabels([f"Type {t}\nn={n:,}" for t, n in zip(by_type.index, by_type["size"])])
    ax_type.set_ylim(0, min(1.0, by_type["mean"].max() * 1.3))
    ax_type.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax_type.set_ylabel("Failure rate")
    _grid(ax_type, "y")

    # 25-minute bins; the sparse tail (225 min and above) is merged into one bin.
    edges = list(range(0, 226, 25)) + [int(df["tool_wear_min"].max()) + 1]
    bins = pd.cut(df["tool_wear_min"], edges, right=False)
    by_wear = df.groupby(bins, observed=False)[cfg.TARGET].agg(["mean", "size"])
    by_wear = by_wear[by_wear["size"] > 0]
    xw = np.arange(len(by_wear))
    ax_wear.bar(xw, by_wear["mean"], width=0.6, color=BLUE, linewidth=0)
    for xi, rate in enumerate(by_wear["mean"]):
        ax_wear.annotate(f"{rate:.1%}", (xi, rate), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, color=INK_2)
    ax_wear.set_xticks(xw)
    ax_wear.set_xticklabels(
        [f"{int(i.left)}-{int(i.right) - 1}\nn={n:,}" for i, n in zip(by_wear.index, by_wear["size"])], fontsize=8
    )
    ax_wear.set_ylim(0, min(1.0, by_wear["mean"].max() * 1.3))
    ax_wear.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax_wear.set_xlabel("Tool wear [min]")
    _grid(ax_wear, "y")
    return fig


@styled
def plot_correlation(df: pd.DataFrame) -> Figure:
    data = with_engineered(df)
    columns = cfg.NUMERIC_SENSOR_FEATURES + cfg.ENGINEERED_FEATURES + [cfg.TARGET]
    corr = data[columns].corr().to_numpy()
    labels = [cfg.SHORT_LABELS.get(c, "Machine failure") for c in columns]
    fig = Figure(figsize=(7.4, 6.4))
    margin = _heading(
        fig,
        "Correlation of sensor readings, engineered features and failure",
        "Pearson correlation; failure is the 0/1 label. Blue negative, red positive, grey near zero.",
    )
    ax = fig.add_axes((0.20, 0.16, 0.64, _top(fig, margin) - 0.16 - 0.03))
    image = ax.imshow(corr, cmap=DIVERGING, vmin=-1, vmax=1, aspect="auto")
    n = len(columns)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8.5)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2.0)
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(n):
        for j in range(n):
            text = f"{corr[i, j]:.2f}".replace("-0.00", "0.00")
            ax.text(j, i, text, ha="center", va="center", fontsize=7.5, color=_ink_on(image.cmap(image.norm(corr[i, j]))))
    colorbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.03, ticks=[-1, 0, 1])
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(length=0, labelsize=8)
    return fig


# --------------------------------------------------------------------------- #
# Model evaluation
# --------------------------------------------------------------------------- #
@styled
def plot_confusion_matrix(counts: Mapping[str, int], threshold: float, title: str = "Hold-out confusion matrix") -> Figure:
    matrix = np.array([[counts["tn"], counts["fp"]], [counts["fn"], counts["tp"]]], dtype=float)
    shares = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    fig = Figure(figsize=(5.4, 4.5))
    margin = _heading(
        fig,
        title,
        f"Flag when score >= {threshold:.3f}. Colour and percentage show the share of each actual class.",
    )
    ax = fig.add_axes((0.24, 0.15, 0.70, _top(fig, margin) - 0.15 - 0.03))
    image = ax.imshow(shares, cmap=SEQUENTIAL, vmin=0, vmax=1, aspect="auto")
    names = ["No failure", "Failure"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels([f"Predicted\n{n.lower()}" for n in names])
    ax.set_yticklabels([f"Actual\n{n.lower()}" for n in names])
    ax.set_xticks([-0.5, 0.5, 1.5], minor=True)
    ax.set_yticks([-0.5, 0.5, 1.5], minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=3.0)
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    labels = [["True negatives", "False alarms"], ["Missed failures", "True positives"]]
    for i in range(2):
        for j in range(2):
            ink = _ink_on(image.cmap(image.norm(shares[i, j])))
            ax.text(j, i - 0.12, f"{int(matrix[i, j]):,}", ha="center", va="center", fontsize=17, fontweight="bold", color=ink)
            ax.text(j, i + 0.14, f"{labels[i][j]}  {shares[i, j]:.1%}", ha="center", va="center", fontsize=8.5, color=ink)
    return fig


@styled
def plot_roc_pr(
    y_true: Sequence[int],
    scores_by_model: Mapping[str, Sequence[float]],
    thresholds: Mapping[str, float],
    final_key: str,
) -> Figure:
    y = np.asarray(y_true).astype(int)
    prevalence = float(y.mean())
    fig = Figure(figsize=(10.0, 5.6))
    margin = _heading(
        fig,
        "ROC and precision-recall curves on the hold-out set",
        f"{int(y.sum())} failures among {len(y):,} records. The marker shows the operating threshold of the final model. "
        "The precision-recall view is the more informative one for a 3.4% failure rate.",
    )
    fig.subplots_adjust(top=_top(fig, margin) - 0.04, bottom=0.31, left=0.06, right=0.99, wspace=0.18)
    ax_roc, ax_pr = fig.subplots(1, 2)

    ax_roc.plot([0, 1], [0, 1], color=MUTED, linewidth=1.0, zorder=1)
    ax_roc.annotate("no skill", (0.55, 0.50), xytext=(0, -11), textcoords="offset points", fontsize=8, color=MUTED, rotation=33, ha="center")
    ax_pr.plot([0, 1], [prevalence, prevalence], color=MUTED, linewidth=1.0, zorder=1)
    ax_pr.annotate(f"no skill (prevalence {prevalence:.1%})", (0.02, prevalence), xytext=(0, 4), textcoords="offset points", fontsize=8, color=MUTED, ha="left")

    order = [k for k in ("logreg_raw", "rf_raw", "rf_engineered") if k in scores_by_model]
    for key in order:
        scores = np.asarray(scores_by_model[key], dtype=float)
        metrics = classification_metrics(y, scores, thresholds[key])
        fpr, tpr, _ = roc_curve(y, scores)
        precision, recall, _ = precision_recall_curve(y, scores)
        color = MODEL_COLORS[key]
        ax_roc.plot(fpr, tpr, color=color, label=f"{MODEL_SHORT[key]} (AUC {metrics['roc_auc']:.3f})", zorder=3 if key == final_key else 2)
        ax_pr.plot(recall, precision, color=color, label=f"{MODEL_SHORT[key]} (AP {metrics['pr_auc']:.3f})", zorder=3 if key == final_key else 2)
        if key == final_key:
            marker = {"marker": "o", "markersize": 7, "markerfacecolor": color, "markeredgecolor": SURFACE, "markeredgewidth": 1.5, "linestyle": "none", "zorder": 5}
            ax_roc.plot([metrics["false_alarm_rate"]], [metrics["recall"]], **marker)
            ax_pr.plot([metrics["recall"]], [metrics["precision"]], **marker)

    ax_roc.set_xlabel("False alarm rate (1 - specificity)")
    ax_roc.set_ylabel("Recall (true positive rate)")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    for ax in (ax_roc, ax_pr):
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.03)
        _grid(ax, "both")
        handles, labels = ax.get_legend_handles_labels()
        if final_key in thresholds:
            handles.append(
                Line2D([], [], marker="o", markersize=7, markerfacecolor=MODEL_COLORS[final_key], markeredgecolor=SURFACE, linestyle="none")
            )
            labels.append(f"Final model at its operating threshold ({thresholds[final_key]:.2f})")
        ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.0, -0.16), ncol=1)
    return fig


@styled
def plot_threshold_tradeoff(y_true: Sequence[int], scores: Sequence[float], threshold: float, beta: float = 2.0) -> Figure:
    grid = np.linspace(0.01, 0.99, 99)
    table = threshold_table(y_true, scores, grid)
    fig = Figure(figsize=(7.2, 4.2))
    margin = _heading(
        fig,
        "Precision, recall and F2 against the decision threshold",
        f"Hold-out set, final model. A machine is flagged when its score is at or above the threshold. "
        f"F{beta:g} weights recall {beta ** 2:g} times as much as precision.",
    )
    ax = fig.add_axes((0.09, 0.14, 0.88, _top(fig, margin) - 0.14 - 0.03))
    ax.plot(table["threshold"], table["precision"], color=BLUE, label="Precision")
    ax.plot(table["threshold"], table["recall"], color=ORANGE, label="Recall")
    ax.plot(table["threshold"], table["f2"], color=AQUA, label="F2")
    ax.axvline(threshold, color=INK_2, linewidth=1.0, zorder=1)
    ax.annotate(f"operating threshold {threshold:.2f}", (threshold, 1.02), xytext=(5, 0), textcoords="offset points", fontsize=8.5, color=INK_2, va="top")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.04)
    ax.set_xlabel("Decision threshold on the model score")
    ax.set_ylabel("Metric value")
    _grid(ax, "y")
    ax.legend(loc="lower left", ncol=3)
    return fig


@styled
def plot_importance(importance: pd.DataFrame) -> Figure:
    permutation = importance[importance["kind"] == "permutation_raw_inputs_holdout"].sort_values("importance_mean", ascending=False)
    impurity = importance[importance["kind"] == "impurity_model_features_train"].sort_values("importance_mean", ascending=False)
    fig = Figure(figsize=(11.0, 4.6))
    engineered_share = float(impurity.loc[impurity["feature"].isin(cfg.ENGINEERED_FEATURES), "importance_mean"].sum()) if len(impurity) else 0.0
    margin = _heading(
        fig,
        "Which inputs drive the model",
        "Left: drop in hold-out PR-AUC when one raw input is shuffled (mean of 10 repeats, bars show one standard deviation); "
        "temperatures matter through their difference. "
        + (f"Right: impurity importance inside the forest; engineered features hold {engineered_share:.0%} of it." if len(impurity) else ""),
    )
    fig.subplots_adjust(top=_top(fig, margin) - 0.03, bottom=0.12, left=0.14, right=0.985, wspace=0.62)
    ax_perm, ax_imp = fig.subplots(1, 2)

    _hbar(
        ax_perm,
        [cfg.SHORT_LABELS.get(f, f) for f in permutation["feature"]],
        permutation["importance_mean"].tolist(),
        [f"{v:.2f}" for v in permutation["importance_mean"]],
        BLUE,
        errors=permutation["importance_std"].tolist(),
        xmax=float((permutation["importance_mean"] + permutation["importance_std"]).max() * 1.22),
    )
    ax_perm.set_xlabel("Drop in PR-AUC")
    ax_perm.set_title("Permutation importance (raw inputs)", loc="left", fontsize=9.5, color=INK, pad=6)

    if len(impurity):
        names = []
        for feature in impurity["feature"]:
            names.append(cfg.SHORT_LABELS.get(feature, feature.replace("type_", "Type ")))
        colors = [BLUE if f in cfg.ENGINEERED_FEATURES else MUTED for f in impurity["feature"]]
        _hbar(
            ax_imp,
            names,
            impurity["importance_mean"].tolist(),
            [f"{v:.2f}" for v in impurity["importance_mean"]],
            colors=colors,
        )
        ax_imp.set_xlabel("Share of impurity reduction")
        ax_imp.set_title("Impurity importance (model features)", loc="left", fontsize=9.5, color=INK, pad=6)
        ax_imp.legend(
            handles=[Patch(facecolor=BLUE, label="Engineered"), Patch(facecolor=MUTED, label="Raw / one-hot")],
            loc="lower right",
        )
    else:
        ax_imp.set_axis_off()
    return fig


def _score_grid(model: Any, x: np.ndarray, y: np.ndarray, build: Callable[[np.ndarray, np.ndarray], pd.DataFrame]) -> np.ndarray:
    gx, gy = np.meshgrid(x, y)
    frame = build(gx.ravel(), gy.ravel())
    return model.pipeline.predict_proba(frame[cfg.RAW_FEATURES])[:, 1].reshape(gx.shape)


@styled
def plot_risk_surfaces(model: Any, df: pd.DataFrame, osf_type: str = "M", resolution: int = 80) -> Figure:
    """Model score over three two-input slices, with the documented limits overlaid."""
    medians = model.training_medians
    air = float(medians["air_temp_k"])
    base = {
        "type": "L",
        "air_temp_k": air,
        "process_temp_k": float(medians["process_temp_k"]),
        "rotational_speed_rpm": float(medians["rotational_speed_rpm"]),
        "torque_nm": float(medians["torque_nm"]),
        "tool_wear_min": 50.0,
    }
    ranges = model.training_ranges

    def frame(**columns: Any) -> pd.DataFrame:
        n = len(next(iter(columns.values())))
        out = pd.DataFrame({k: np.full(n, v) for k, v in base.items()})
        for key, values in columns.items():
            out[key] = values
        return out

    fig = Figure(figsize=(13.0, 4.8))
    margin = _heading(
        fig,
        "What the model has learned: failure score over three input slices",
        "Colour is the model failure score (0 to 1); other inputs are held at training medians (tool wear 50 min unless varied). "
        "Red line: model decision boundary (score = operating threshold). Dashed black lines: limits documented by the dataset authors.",
    )
    fig.subplots_adjust(top=_top(fig, margin) - 0.05, bottom=0.14, left=0.055, right=0.92, wspace=0.24)
    axes = fig.subplots(1, 3)

    # (a) power: speed x torque
    speed = np.linspace(ranges["rotational_speed_rpm"][0], ranges["rotational_speed_rpm"][1], resolution)
    torque = np.linspace(ranges["torque_nm"][0], ranges["torque_nm"][1], resolution)
    z = _score_grid(model, speed, torque, lambda gx, gy: frame(rotational_speed_rpm=gx, torque_nm=gy))
    ax = axes[0]
    mesh = ax.pcolormesh(speed, torque, z, cmap=SEQUENTIAL, vmin=0, vmax=1, shading="auto", rasterized=True)
    ax.contour(speed, torque, z, levels=[model.threshold], colors=[CRITICAL], linewidths=2.2, zorder=3)
    for power, dy, va in ((cfg.POWER_MAX_W, -7, "top"), (cfg.POWER_MIN_W, 7, "bottom")):
        curve = power * 60.0 / (2 * np.pi * speed)
        ax.plot(speed, curve, color=INK, linewidth=0.9, linestyle=DASHED, zorder=4)
        _line_label(ax, f"{power:,.0f} W", (speed[-1], curve[-1]), (-4, dy), va=va)
    ax.set_xlabel(cfg.FEATURE_LABELS["rotational_speed_rpm"])
    ax.set_ylabel(cfg.FEATURE_LABELS["torque_nm"])
    ax.set_ylim(torque[0], torque[-1])
    _thousands(ax.xaxis)
    ax.set_title("Power: speed vs torque", loc="left", fontsize=9.5, color=INK, pad=6)

    # (b) overstrain: tool wear x torque (speed fixed so power stays inside its limits)
    wear = np.linspace(0, ranges["tool_wear_min"][1], resolution)
    torque_b = np.linspace(27.0, 64.0, resolution)
    z = _score_grid(
        model, wear, torque_b,
        lambda gx, gy: frame(tool_wear_min=gx, torque_nm=gy, rotational_speed_rpm=np.full(len(gx), 1300.0), type=np.full(len(gx), osf_type)),
    )
    ax = axes[1]
    ax.pcolormesh(wear, torque_b, z, cmap=SEQUENTIAL, vmin=0, vmax=1, shading="auto", rasterized=True)
    limit = cfg.OSF_LIMITS_MIN_NM[osf_type]
    w = np.linspace(limit / torque_b[-1], wear[-1], 100)
    ax.contour(wear, torque_b, z, levels=[model.threshold], colors=[CRITICAL], linewidths=2.2, zorder=3)
    ax.plot(w, limit / w, color=INK, linewidth=0.9, linestyle=DASHED, zorder=4)
    anchor = float(w[int(len(w) * 0.55)])
    ax.annotate(
        f"documented limit\n{limit:,.0f} min*Nm", (anchor, limit / anchor), xytext=(8.0, 58.5), textcoords="data",
        fontsize=8, color=INK_2, va="center", arrowprops={"arrowstyle": "-", "color": INK_2, "linewidth": 0.8},
        bbox={"boxstyle": "square,pad=0.2", "facecolor": SURFACE, "edgecolor": "none", "alpha": 0.9}, zorder=6,
    )
    ax.set_xlabel(cfg.FEATURE_LABELS["tool_wear_min"])
    ax.set_ylabel(cfg.FEATURE_LABELS["torque_nm"])
    ax.set_title(f"Overstrain: tool wear vs torque (Type {osf_type}, 1,300 rpm)", loc="left", fontsize=9.5, color=INK, pad=6)

    # (c) heat dissipation: speed x temperature difference (process = air + difference)
    diff = cfg.HDF_TEMP_DIFF_MAX_K
    dt_all = (df["process_temp_k"] - df["air_temp_k"]).to_numpy()
    speed_c = np.linspace(ranges["rotational_speed_rpm"][0], 2000.0, resolution)
    dt = np.linspace(float(np.floor(dt_all.min())), float(np.ceil(dt_all.max())), resolution)
    z = _score_grid(
        model, speed_c, dt,
        lambda gx, gy: frame(rotational_speed_rpm=gx, process_temp_k=air + gy, torque_nm=np.full(len(gx), 40.0)),
    )
    ax = axes[2]
    ax.pcolormesh(speed_c, dt, z, cmap=SEQUENTIAL, vmin=0, vmax=1, shading="auto", rasterized=True)
    ax.contour(speed_c, dt, z, levels=[model.threshold], colors=[CRITICAL], linewidths=2.2, zorder=3)
    ax.plot([cfg.HDF_SPEED_MAX_RPM, cfg.HDF_SPEED_MAX_RPM, speed_c[0]], [dt[0], diff, diff], color=INK, linewidth=0.9, linestyle=DASHED, zorder=4)
    _line_label(ax, "documented region", (speed_c[0], diff), (4, 6), ha="left", va="bottom")
    ax.set_xlabel(cfg.FEATURE_LABELS["rotational_speed_rpm"])
    ax.set_ylabel(cfg.FEATURE_LABELS["temp_diff_k"].replace(", process - air", ""))
    _thousands(ax.xaxis)
    ax.set_title("Heat dissipation: speed vs temperature difference (40 Nm)", loc="left", fontsize=9.5, color=INK, pad=6)

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(False)
    colorbar_ax = fig.add_axes((0.935, 0.14, 0.012, _top(fig, margin) - 0.05 - 0.14))
    colorbar = fig.colorbar(mesh, cax=colorbar_ax, ticks=[0, 0.25, 0.5, 0.75, 1])
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(length=0, labelsize=8)
    colorbar.set_label("Model failure score", fontsize=8.5, color=INK_2)
    return fig


@styled
def plot_per_mode_recall(per_mode: pd.DataFrame) -> Figure:
    rows = per_mode[(per_mode["n_failures"] > 0) & (per_mode["mode"] != "all_failures")]
    total = per_mode[per_mode["mode"] == "all_failures"]
    rows = pd.concat([rows, total], ignore_index=True)
    fig = Figure(figsize=(7.6, 3.8))
    margin = _heading(
        fig,
        "Share of recorded failures detected, by failure mode",
        "Hold-out set, final model at its operating threshold. A failure can have several modes. "
        "Tool-wear and random failures are assigned partly at random in the data, so the sensor readings can only hint at them.",
    )
    ax = fig.add_axes((0.33, 0.14, 0.62, _top(fig, margin) - 0.14 - 0.03))
    labels = [f"{r.label}  (n={r.n_failures})" for r in rows.itertuples()]
    _hbar(
        ax,
        labels,
        rows["recall"].tolist(),
        [f"{r.detected}/{r.n_failures}  ({r.recall:.0%})" for r in rows.itertuples()],
        BLUE,
        xmax=1.28,
    )
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Recall (share of failures flagged)")
    return fig


@styled
def plot_cumulative_gains(y_true: Sequence[int], scores: Sequence[float], marks: Sequence[float] = (0.02, 0.05, 0.10)) -> Figure:
    from .metrics import cumulative_gains, recall_at_fraction

    gains = cumulative_gains(y_true, scores)
    y = np.asarray(y_true).astype(int)
    limit = 0.30
    fig = Figure(figsize=(7.2, 4.4))
    margin = _heading(
        fig,
        "Inspection budget: failures found when inspecting top-ranked machines",
        f"Hold-out set ({len(y):,} machines, {int(y.sum())} failures), machines ranked by model score. "
        f"The horizontal axis is cut at {limit:.0%} of machines.",
    )
    ax = fig.add_axes((0.10, 0.14, 0.86, _top(fig, margin) - 0.14 - 0.03))
    mask = gains["fraction_inspected"] <= limit
    ideal = np.minimum(gains["fraction_inspected"] * len(y) / max(int(y.sum()), 1), 1.0)
    ax.plot(gains["fraction_inspected"][mask], ideal[mask], color=AXIS, linewidth=1.5, label="Perfect ranking")
    ax.plot(gains["fraction_inspected"][mask], gains["fraction_failures_found"][mask], color=BLUE, label="Model ranking")
    ax.plot([0, limit], [0, limit], color=MUTED, linewidth=1.0, label="Random inspection")
    for fraction in marks:
        found = recall_at_fraction(y, scores, fraction)
        ax.plot([fraction], [found], marker="o", markersize=7, markerfacecolor=BLUE, markeredgecolor=SURFACE, markeredgewidth=1.5, linestyle="none", zorder=5)
        ax.annotate(f"top {fraction:.0%}: {found:.0%} of failures", (fraction, found), xytext=(10, -14), textcoords="offset points", fontsize=8.5, color=INK_2)
    ax.set_xlim(0, limit)
    ax.set_ylim(0, 1.04)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Share of machines inspected (highest score first)")
    ax.set_ylabel("Share of all failures found")
    _grid(ax, "both")
    ax.legend(loc="lower right")
    return fig


@styled
def plot_calibration(y_true: Sequence[int], scores: Sequence[float], small_bin: int = 25) -> Figure:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    edges = np.array([0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 0.90, 1.0001])
    index = np.digitize(s, edges) - 1
    xs, ys, ns = [], [], []
    for b in range(len(edges) - 1):
        members = index == b
        if members.any():
            xs.append(s[members].mean())
            ys.append(y[members].mean())
            ns.append(int(members.sum()))
    fig = Figure(figsize=(5.8, 4.8))
    margin = _heading(
        fig,
        "Is the score a probability? Reliability on the hold-out set",
        "Mean score vs observed failure share in score bins. Points off the diagonal show the score is not a calibrated "
        f"probability. Bins with fewer than {small_bin} records are labelled with their size and are noisy.",
    )
    ax = fig.add_axes((0.14, 0.13, 0.82, _top(fig, margin) - 0.13 - 0.03))
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1.0, label="Perfectly calibrated")
    ax.plot(xs, ys, color=BLUE, marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5, label="Final model")
    for x, y_obs, n in zip(xs, ys, ns):
        if n < small_bin:
            ax.annotate(f"n={n}", (x, y_obs), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8, color=INK_2)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.08)
    ax.set_xlabel("Mean model score in bin")
    ax.set_ylabel("Observed share of failures")
    _grid(ax, "both")
    ax.legend(loc="upper left")
    return fig


# --------------------------------------------------------------------------- #
# Machine assessment
# --------------------------------------------------------------------------- #
@styled
def plot_score_strip(score: float, threshold: float, elevated_fraction: float = cfg.ELEVATED_SCORE_FRACTION) -> Figure:
    """Where one failure score sits relative to the model levels (Low / Elevated / High)."""
    low_edge = threshold * elevated_fraction
    fig = Figure(figsize=(7.2, 1.7))
    ax = fig.add_axes((0.02, 0.36, 0.96, 0.30))
    zones = ((0.0, low_edge, BLUE_RAMP[0], "Low"), (low_edge, threshold, BLUE_RAMP[4], "Elevated"), (threshold, 1.0, BLUE_RAMP[9], "High"))
    for start, end, color, name in zones:
        ax.barh([0], [end - start], left=[start], height=1.0, color=color, linewidth=0)
        if end - start > 0.07:
            ax.text((start + end) / 2, 0, name, ha="center", va="center", fontsize=9, color=_ink_on(color))
    ax.axvline(score, ymin=0.0, ymax=1.3, color=CRITICAL, linewidth=2.2, clip_on=False, zorder=5)
    ax.plot([score], [0.62], marker="v", markersize=8, color=CRITICAL, clip_on=False, zorder=6, linestyle="none")
    side = "right" if score > 0.80 else "left"
    ax.annotate(
        f"score {score:.3f}", (score, 0.62), xytext=(-7 if side == "right" else 7, 4), textcoords="offset points",
        ha=side, va="bottom", fontsize=9.5, fontweight="bold", color=INK, annotation_clip=False,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 0.5)
    ticks = sorted({0.0, round(low_edge, 3), round(threshold, 3), 1.0})
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{v:g}" for v in ticks])
    ax.set_yticks([])
    for spine in ("left", "right", "top"):
        ax.spines[spine].set_visible(False)
    ax.set_xlabel("Model failure score (levels: Low below half the threshold, High at or above the threshold)")
    return fig


@styled
def plot_sensitivity(table: pd.DataFrame) -> Figure:
    """Occlusion-style effect of each input on one score (see ``predict.local_sensitivity``)."""
    rows = table.sort_values("effect_on_score", key=lambda s: s.abs(), ascending=True)
    fig = Figure(figsize=(7.2, 3.6))
    margin = _heading(
        fig,
        "How each input moves this score",
        "Change in score when one input is replaced by its training median (most common Type), all else unchanged. "
        "Red raises the score, blue lowers it. Effects need not add up: inputs act together (for example wear x torque).",
    )
    ax = fig.add_axes((0.30, 0.15, 0.64, _top(fig, margin) - 0.15 - 0.03))
    effects = rows["effect_on_score"].to_numpy(dtype=float)
    labels = [f"{cfg.SHORT_LABELS.get(k, k)} = {v:g}" if not isinstance(v, str) else f"Type {v}"
              for k, v in zip(_input_keys(rows), rows["value"])]
    y = np.arange(len(rows))
    ax.barh(y, effects, height=0.5, color=[CRITICAL if e > 0 else BLUE for e in effects], linewidth=0)
    span = max(float(np.abs(effects).max()), 0.05)
    ax.set_xlim(-span * 1.35, span * 1.35)
    for yi, effect in zip(y, effects):
        ax.annotate(
            f"{effect:+.3f}", (effect, yi), xytext=(5 if effect >= 0 else -5, 0), textcoords="offset points",
            ha="left" if effect >= 0 else "right", va="center", fontsize=8.5, color=INK_2,
        )
    ax.axvline(0, color=INK_2, linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Effect on the failure score")
    _grid(ax, "x")
    return fig


def _input_keys(table: pd.DataFrame) -> list[str]:
    """Map the human-readable ``input`` column back to canonical feature names."""
    reverse = {label: key for key, label in cfg.FEATURE_LABELS.items()}
    return [reverse.get(name, name) for name in table["input"]]

