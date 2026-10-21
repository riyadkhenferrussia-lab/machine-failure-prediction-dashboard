"""Feature engineering as a scikit-learn transformer.

Keeping the engineering inside the fitted pipeline means the saved model accepts the
raw sensor columns and computes the derived features itself, so training and serving
cannot drift apart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from . import config as cfg
from .physics import mechanical_power_w, strain_index_min_nm, temperature_difference_k


class SensorFeatureEngineer(TransformerMixin, BaseEstimator):
    """Select the raw model inputs and optionally append physically motivated features.

    Only ``cfg.RAW_FEATURES`` are ever read from the input frame. Identifiers, the
    failure-mode flags and the label are ignored even when the caller passes them, so
    a leaky column cannot reach the model by accident.

    Engineered features (computed from the same row only, never from labels):

    * ``temp_diff_k``    process temperature - air temperature
    * ``power_w``        torque x angular speed
    * ``strain_min_nm``  tool wear x torque
    """

    def __init__(self, add_engineered: bool = True):
        self.add_engineered = add_engineered

    @staticmethod
    def _check_input(X: object) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"SensorFeatureEngineer expects a pandas DataFrame with columns {cfg.RAW_FEATURES}"
            )
        missing = [c for c in cfg.RAW_FEATURES if c not in X.columns]
        if missing:
            raise ValueError(f"Missing input columns: {missing}")

    def fit(self, X: pd.DataFrame, y: object = None) -> "SensorFeatureEngineer":
        self._check_input(X)
        self.n_features_in_ = len(cfg.RAW_FEATURES)
        self.feature_names_in_ = np.array(cfg.RAW_FEATURES, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "n_features_in_")
        self._check_input(X)
        out = X[cfg.RAW_FEATURES].copy()
        for column in cfg.NUMERIC_SENSOR_FEATURES:
            out[column] = out[column].astype(float)
        out[cfg.CATEGORICAL_FEATURE] = out[cfg.CATEGORICAL_FEATURE].astype(str)
        if self.add_engineered:
            out["temp_diff_k"] = temperature_difference_k(out["process_temp_k"], out["air_temp_k"])
            out["power_w"] = mechanical_power_w(out["torque_nm"], out["rotational_speed_rpm"])
            out["strain_min_nm"] = strain_index_min_nm(out["tool_wear_min"], out["torque_nm"])
        return out

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        names = list(cfg.RAW_FEATURES)
        if self.add_engineered:
            names += cfg.ENGINEERED_FEATURES
        return np.array(names, dtype=object)
