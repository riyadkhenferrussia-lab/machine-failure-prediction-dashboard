"""The committed artefacts agree with each other and with the data (no hand-edited numbers)."""

import numpy as np
import pandas as pd
import pytest

from src import config as cfg
from src.metrics import classification_metrics
from src.utils import read_json, sha256_of_file

PNG = b"\x89PNG\r\n\x1a\n"


@pytest.fixture(scope="module")
def metrics():
    if not cfg.METRICS_PATH.exists():
        pytest.skip("reports/metrics.json is missing; run `python -m src.train`")
    return read_json(cfg.METRICS_PATH)


def test_metrics_recompute_from_stored_predictions(metrics, holdout):
    for key, entry in metrics["models"].items():
        live = classification_metrics(holdout[cfg.TARGET], holdout[f"score_{key}"], entry["threshold"])
        for name in ("tp", "fp", "fn", "tn"):
            assert live[name] == entry["test"][name], (key, name)
        for name in ("precision", "recall", "f1", "pr_auc", "roc_auc"):
            assert live[name] == pytest.approx(entry["test"][name], abs=1e-4), (key, name)


def test_final_model_was_selected_by_cross_validation_not_holdout(metrics):
    candidates = {k: v for k, v in metrics["models"].items() if k != "baseline_majority"}
    assert metrics["final"]["key"] == max(candidates, key=lambda k: candidates[k]["cv_pr_auc_mean"])


def test_threshold_and_metadata_are_consistent(metrics, shipped_model):
    card = read_json(cfg.MODEL_CARD_PATH)
    assert card["threshold"] == metrics["final"]["threshold"] == pytest.approx(shipped_model.threshold)
    meta = metrics["meta"]
    assert meta["n_train"] + meta["n_test"] == meta["n_rows"] == 10_000
    assert meta["data_sha256"] == sha256_of_file(cfg.RAW_DATA_PATH)
    assert set(card["excluded_columns_leakage_guard"]) == set(cfg.LEAKAGE_COLUMNS)
    assert not set(card["model_input_names"]) & set(cfg.LEAKAGE_COLUMNS)


def test_comparison_csv_matches_metrics(metrics):
    table = pd.read_csv(cfg.MODEL_COMPARISON_PATH).set_index("model_key")
    for key, entry in metrics["models"].items():
        assert table.loc[key, "tp"] == entry["test"]["tp"]
        assert table.loc[key, "test_pr_auc"] == pytest.approx(entry["test"]["pr_auc"], rel=1e-5)


def test_importance_covers_raw_inputs_only_for_permutation():
    table = pd.read_csv(cfg.IMPORTANCE_PATH)
    perm = table[table["kind"] == "permutation_raw_inputs_holdout"]
    assert set(perm["feature"]) == set(cfg.RAW_FEATURES)
    assert not set(table["feature"]) & set(cfg.LEAKAGE_COLUMNS)


def test_holdout_level_table_matches_recomputation(metrics, holdout):
    from src.evaluate import holdout_level_table

    stored = pd.read_csv(cfg.RECOMMENDATION_LEVELS_PATH)
    live = holdout_level_table(holdout, metrics)
    assert stored.iloc[:, :3].astype(str).values.tolist() == live.iloc[:, :3].astype(str).values.tolist()
    assert live.iloc[:, 1].sum() == len(holdout)


def test_model_card_quotes_the_stored_numbers(metrics):
    text = cfg.MODEL_CARD_MD_PATH.read_text(encoding="utf-8")
    t = metrics["final"]["test"]
    for value in (t["precision"], t["recall"], t["f1"], t["pr_auc"], t["roc_auc"]):
        assert f"{value:.3f}" in text
    import re

    assert not re.search(r"\bnan%?\b", text, flags=re.IGNORECASE)   # no undefined values printed as numbers


@pytest.mark.parametrize("name", [
    "eda_correlation", "eda_distributions", "eda_failure_mechanisms", "eda_failure_modes", "eda_failure_rates",
    "eval_calibration", "eval_confusion_matrix", "eval_cumulative_gains", "eval_feature_importance",
    "eval_per_mode_recall", "eval_risk_surfaces", "eval_roc_pr", "eval_threshold_tradeoff",
])
def test_report_figures_exist_and_are_png(name):
    path = cfg.FIGURES_DIR / f"{name}.png"
    assert path.exists() and path.read_bytes()[:8] == PNG and path.stat().st_size > 20_000


def test_data_validation_report(metrics):
    report = read_json(cfg.DATA_VALIDATION_PATH)
    assert report["ok"] and report["n_errors"] == 0


def test_readme_quotes_the_stored_numbers(metrics):
    text = cfg.README_PATH.read_text(encoding="utf-8")
    t = metrics["final"]["test"]
    for value in (t["precision"], t["recall"], t["f1"], t["pr_auc"], t["roc_auc"]):
        assert f"{value:.3f}" in text
    assert f"{t['tp']} true alarms, {t['fp']} false alarms, {t['fn']} missed" in text
