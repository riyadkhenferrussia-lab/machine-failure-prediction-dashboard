"""Shared fixtures.

Fixtures that need the shipped artefacts (``models/``, ``reports/``) skip with a clear reason
when those files are missing; run ``python -m src.train`` and ``python -m src.evaluate`` to
create them. The dataset itself is part of the repository (``data/raw/ai4i2020.csv``).
"""

from __future__ import annotations

import pandas as pd
import pytest

from src import config as cfg
from src import data as data_module
from src.predict import ModelLoadError, ModelNotFoundError, load_holdout_predictions, load_model


@pytest.fixture(scope="session")
def dataset() -> pd.DataFrame:
    """The real AI4I 2020 data with canonical column names (validated)."""
    return data_module.load_dataset()


@pytest.fixture(scope="session")
def validation_report():
    return data_module.load_and_validate()[1]


@pytest.fixture(scope="session")
def split(dataset):
    return data_module.split_dataset(dataset)


@pytest.fixture(scope="session")
def shipped_model():
    try:
        return load_model()
    except ModelNotFoundError:
        pytest.skip("models/failure_model.joblib is missing; run `python -m src.train`")
    except ModelLoadError as exc:
        pytest.skip(f"the saved model cannot be loaded with these library versions ({exc}); run `python -m src.train`")


@pytest.fixture(scope="session")
def holdout():
    frame = load_holdout_predictions()
    if frame is None:
        pytest.skip("reports/test_predictions.csv is missing; run `python -m src.train`")
    return frame


@pytest.fixture()
def good_reading() -> dict:
    """A plausible reading inside the training ranges, far from every failure criterion."""
    return {
        "type": "M",
        "air_temp_k": 300.0,
        "process_temp_k": 310.0,
        "rotational_speed_rpm": 1500.0,
        "torque_nm": 40.0,
        "tool_wear_min": 100.0,
    }


@pytest.fixture()
def tiny_dataset_csv(dataset, tmp_path):
    """A stratified 2,000-row subsample in the published CSV layout (for fast pipeline tests)."""
    sample = (
        dataset.groupby(cfg.TARGET, group_keys=False)
        .sample(frac=0.2, random_state=0)
        .sort_values("udi")
    )
    inverse = {canonical: raw for raw, canonical in cfg.RAW_TO_CANONICAL.items()}
    path = tmp_path / "tiny_ai4i.csv"
    sample.rename(columns=inverse).to_csv(path, index=False)
    return path
