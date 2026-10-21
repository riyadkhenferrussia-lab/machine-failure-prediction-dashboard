"""End-to-end training on a 2,000-row subsample, written to a temporary folder.

The shipped artefacts are never touched. Forest size is reduced to keep the run short.
"""

import numpy as np
import pandas as pd
import pytest

from src import config as cfg
from src import train
from src.predict import load_model, score_frame
from src.utils import read_json

pytestmark = pytest.mark.slow


@pytest.fixture()
def small_forest(monkeypatch):
    monkeypatch.setattr(cfg, "DEFAULT_RF_PARAMS", {**cfg.DEFAULT_RF_PARAMS, "n_estimators": 40, "max_depth": 12})


def run(tmp_path, csv, seed=42):
    paths = train.ArtifactPaths.under(tmp_path / f"run{seed}")
    options = train.TrainingOptions(search=False, cv_folds=3, bootstrap=20, n_jobs=1, data_path=csv, seed=seed)
    return train.run_training(options, paths), paths


def test_training_writes_consistent_artefacts(tmp_path, tiny_dataset_csv, small_forest):
    payload, paths = run(tmp_path, tiny_dataset_csv)
    for path in (paths.model, paths.model_card, paths.metrics, paths.comparison, paths.predictions, paths.importance, paths.data_validation):
        assert path.exists(), path
    assert payload["meta"]["n_rows"] == 2000 and payload["meta"]["n_test"] == 400
    assert payload["final"]["key"] != "baseline_majority"
    assert read_json(paths.metrics)["final"]["threshold"] == payload["final"]["threshold"]

    model = load_model(paths.model)
    predictions = pd.read_csv(paths.predictions)
    np.testing.assert_allclose(score_frame(model, predictions), predictions[f"score_{model.model_key}"], atol=1e-5)
    assert not set(model.pipeline.named_steps["prep"].get_feature_names_out()) & set(cfg.LEAKAGE_COLUMNS)
    # The hold-out rows are not part of the training ranges' source split.
    assert payload["meta"]["n_train"] + payload["meta"]["n_test"] == 2000


def test_training_is_reproducible_for_a_seed(tmp_path, tiny_dataset_csv, small_forest):
    first, _ = run(tmp_path / "a", tiny_dataset_csv)
    second, _ = run(tmp_path / "b", tiny_dataset_csv)
    assert first["final"]["test"] == second["final"]["test"]
    assert first["final"]["threshold"] == second["final"]["threshold"]
