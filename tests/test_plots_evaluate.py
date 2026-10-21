"""Figures build without pyplot and render to PNG; the model card renders from the artefacts."""

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from src import config as cfg
from src import evaluate as E
from src import plots
from src import predict as P
from src.utils import read_json

PNG = b"\x89PNG\r\n\x1a\n"


def check(fig):
    assert isinstance(fig, Figure) and fig.axes
    png = plots.figure_png(fig, dpi=60)
    assert png[:8] == PNG and len(png) > 2000


def test_data_figures(dataset):
    counts = {m: int(dataset[m].sum()) for m in cfg.FAILURE_MODE_COLUMNS}
    check(plots.plot_failure_modes(counts, len(dataset), int(dataset[cfg.TARGET].sum()), 18))
    check(plots.plot_distributions(dataset))
    check(plots.plot_distribution(dataset, "torque_nm"))
    check(plots.plot_scatter(dataset, "rotational_speed_rpm", "torque_nm"))
    check(plots.plot_failure_mechanisms(dataset))
    check(plots.plot_correlation(dataset))


def test_failure_rate_axis_is_capped_at_100_percent(dataset):
    fig = plots.plot_failure_rates(dataset)
    assert all(ax.get_ylim()[1] <= 1.0 + 1e-9 for ax in fig.axes)


def test_evaluation_figures(shipped_model, holdout, dataset):
    metrics = read_json(cfg.METRICS_PATH)
    importance = pd.read_csv(cfg.IMPORTANCE_PATH)
    figures = E.build_figures(shipped_model, metrics, holdout, importance, dataset)
    assert set(figures) == set(E.FIGURE_FILES)
    for fig in figures.values():
        check(fig)


def test_single_assessment_figures(shipped_model, good_reading):
    a = P.assess_reading(shipped_model, good_reading)
    check(plots.plot_score_strip(a.score, a.threshold))
    check(plots.plot_sensitivity(a.sensitivity))


def test_no_pyplot_state_is_left_behind(dataset):
    import matplotlib.pyplot as plt

    plots.plot_distribution(dataset, "torque_nm")
    assert plt.get_fignums() == []


def test_level_table():
    table = E.level_table(np.array([2, 2, 1, 0, 0, 0]), np.array([1, 0, 1, 0, 0, 1]))
    assert table.iloc[:, 0].tolist() == ["High", "Elevated", "Low"]
    assert table.iloc[:, 1].tolist() == [2, 1, 3]


def test_model_card_renders_from_artefacts():
    metrics = read_json(cfg.METRICS_PATH)
    card = read_json(cfg.MODEL_CARD_PATH)
    text = E.render_model_card(metrics, card, pd.read_csv(cfg.IMPORTANCE_PATH))
    for heading in ("## Intended use", "## Hold-out results", "## Interpretation and limitations"):
        assert heading in text
    assert "not a confirmed fault" in text.lower() or "never a confirmed fault" in text
