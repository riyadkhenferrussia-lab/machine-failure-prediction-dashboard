"""Exploratory data analysis: figures and a machine-readable summary.

    python -m src.eda
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import config as cfg
from .data import load_and_validate
from .plots import (
    plot_correlation,
    plot_distributions,
    plot_failure_mechanisms,
    plot_failure_modes,
    plot_failure_rates,
    save_figure,
    with_engineered,
)
from .utils import write_json


def summarise(df: pd.DataFrame, validation_stats: dict[str, Any]) -> dict[str, Any]:
    """Numbers quoted in the README / dashboard, computed from the data."""
    data = with_engineered(df)
    by_type = df.groupby("type")[cfg.TARGET].agg(["mean", "sum", "size"]).reindex(cfg.TYPE_VALUES)
    edges = [0, 100, 150, 180, 200, 254]
    bins = pd.cut(df["tool_wear_min"], edges, right=False)
    by_wear = df.groupby(bins, observed=False)[cfg.TARGET].agg(["mean", "sum", "size"])
    failed = data[data[cfg.TARGET] == 1]
    normal = data[data[cfg.TARGET] == 0]
    features = cfg.NUMERIC_SENSOR_FEATURES + cfg.ENGINEERED_FEATURES
    correlation = data[features + [cfg.TARGET]].corr()[cfg.TARGET].drop(cfg.TARGET)
    return {
        "n_rows": int(len(df)),
        "n_failures": int(df[cfg.TARGET].sum()),
        "failure_rate": float(df[cfg.TARGET].mean()),
        "failure_rate_by_type": {
            t: {"rate": float(r["mean"]), "failures": int(r["sum"]), "records": int(r["size"])}
            for t, r in by_type.iterrows()
        },
        "failure_rate_by_tool_wear_bin": {
            f"{int(i.left)}-{int(i.right) - 1} min": {
                "rate": float(r["mean"]),
                "failures": int(r["sum"]),
                "records": int(r["size"]),
            }
            for i, r in by_wear.iterrows()
        },
        "feature_means_by_outcome": {
            f: {"no_failure": float(normal[f].mean()), "failure": float(failed[f].mean())}
            for f in features
        },
        "correlation_with_failure": {k: float(v) for k, v in correlation.items()},
        "documented_rule_agreement": validation_stats.get("documented_rule_agreement", {}),
        "label_inconsistencies": validation_stats.get("label_inconsistencies", {}),
        "describe": df[cfg.NUMERIC_SENSOR_FEATURES].describe().round(4).to_dict(),
    }


def run_eda(figures_dir: Path | str = cfg.FIGURES_DIR, summary_path: Path | str = cfg.EDA_SUMMARY_PATH) -> dict[str, Any]:
    frame, report = load_and_validate()
    report.raise_if_errors()
    figures_dir = Path(figures_dir)
    stats = report.stats
    inconsistent = stats["label_inconsistencies"]

    save_figure(
        plot_failure_modes(
            stats["failure_mode_counts"], stats["n_rows"], stats["n_failures"], inconsistent["rnf_flag_without_failure"]
        ),
        figures_dir / "eda_failure_modes.png",
    )
    save_figure(plot_distributions(frame), figures_dir / "eda_distributions.png")
    save_figure(plot_failure_mechanisms(frame), figures_dir / "eda_failure_mechanisms.png")
    save_figure(plot_failure_rates(frame), figures_dir / "eda_failure_rates.png")
    save_figure(plot_correlation(frame), figures_dir / "eda_correlation.png")

    summary = summarise(frame, stats)
    write_json(summary_path, summary)
    return summary


def main() -> int:
    summary = run_eda()
    print(f"EDA complete: {summary['n_rows']:,} rows, failure rate {summary['failure_rate']:.2%}")
    print(f"Figures written to {cfg.FIGURES_DIR}")
    print(f"Summary written to {cfg.EDA_SUMMARY_PATH}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
