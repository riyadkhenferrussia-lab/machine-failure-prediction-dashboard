"""Feature engineering and the guarantees against target leakage."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from src import config as cfg
from src import data as data_module
from src.features import SensorFeatureEngineer
from src.modeling import build_pipeline
from src.predict import score_frame


def raw_frame(dataset, n=200):
    return dataset[cfg.RAW_FEATURES].head(n)


class TestSensorFeatureEngineer:
    def test_adds_exactly_the_three_documented_features(self, dataset):
        out = SensorFeatureEngineer().fit_transform(raw_frame(dataset))
        assert list(out.columns) == cfg.RAW_FEATURES + cfg.ENGINEERED_FEATURES
        row = dataset.iloc[0]
        assert out.loc[0, "temp_diff_k"] == pytest.approx(row["process_temp_k"] - row["air_temp_k"])
        assert out.loc[0, "power_w"] == pytest.approx(row["torque_nm"] * row["rotational_speed_rpm"] * 2 * np.pi / 60)
        assert out.loc[0, "strain_min_nm"] == pytest.approx(row["tool_wear_min"] * row["torque_nm"])

    def test_can_be_switched_off(self, dataset):
        out = SensorFeatureEngineer(add_engineered=False).fit_transform(raw_frame(dataset))
        assert list(out.columns) == cfg.RAW_FEATURES

    def test_ignores_label_mode_and_identifier_columns_even_when_passed(self, dataset):
        out = SensorFeatureEngineer().fit_transform(dataset.head(50))  # full frame incl. labels and ids
        assert not set(out.columns) & set(cfg.LEAKAGE_COLUMNS)
        assert list(out.columns) == cfg.RAW_FEATURES + cfg.ENGINEERED_FEATURES

    def test_does_not_modify_its_input(self, dataset):
        frame = raw_frame(dataset).copy()
        before = frame.copy(deep=True)
        SensorFeatureEngineer().fit_transform(frame)
        pd.testing.assert_frame_equal(frame, before)

    def test_clear_errors_for_bad_input(self, dataset):
        with pytest.raises(TypeError, match="DataFrame"):
            SensorFeatureEngineer().fit(np.zeros((3, 6)))
        with pytest.raises(ValueError, match="Missing input columns"):
            SensorFeatureEngineer().fit(raw_frame(dataset).drop(columns=["torque_nm"]))

    def test_transform_requires_fit(self, dataset):
        from sklearn.exceptions import NotFittedError

        with pytest.raises(NotFittedError):
            SensorFeatureEngineer().transform(raw_frame(dataset))


class TestPipelineLeakage:
    def test_fitted_pipeline_never_sees_a_leakage_column(self, dataset):
        X, y = data_module.features_and_target(dataset.head(1500))
        pipeline = build_pipeline("rf_engineered", {"n_estimators": 10}).fit(X, y)
        names = set(pipeline.named_steps["prep"].get_feature_names_out())
        assert not names & set(cfg.LEAKAGE_COLUMNS)
        assert names == {"air_temp_k", "process_temp_k", "rotational_speed_rpm", "torque_nm", "tool_wear_min",
                         "temp_diff_k", "power_w", "strain_min_nm", "type_H", "type_L", "type_M"}

    def test_shipped_model_inputs_are_clean(self, shipped_model):
        names = shipped_model.metadata and shipped_model.pipeline.named_steps["prep"].get_feature_names_out()
        assert not set(names) & set(cfg.LEAKAGE_COLUMNS)

    def test_scores_do_not_change_when_label_columns_are_tampered_with(self, shipped_model, holdout):
        frame = holdout.copy()
        baseline = score_frame(shipped_model, frame)
        tampered = frame.copy()
        for column in cfg.FAILURE_MODE_COLUMNS + [cfg.TARGET]:
            tampered[column] = 1 - tampered[column]
        tampered["udi"] = tampered["udi"].iloc[::-1].to_numpy()
        np.testing.assert_array_equal(baseline, score_frame(shipped_model, tampered))

    def test_a_model_cannot_learn_shuffled_labels(self, split):
        """Permutation sanity check: with labels shuffled the pipeline must fall to chance level.

        Any hidden path from the label (or an id / mode flag) into the features would keep the
        hold-out PR-AUC far above the failure rate even though the training labels are noise.
        """
        train, test = split
        X_train, y_train = data_module.features_and_target(train)
        X_test, y_test = data_module.features_and_target(test)
        rng = np.random.default_rng(0)
        shuffled = pd.Series(rng.permutation(y_train.to_numpy()), index=y_train.index)
        pipeline = build_pipeline("rf_engineered", {"n_estimators": 60, "max_depth": 12}, n_jobs=1).fit(X_train, shuffled)
        pr_auc = average_precision_score(y_test, pipeline.predict_proba(X_test)[:, 1])
        assert pr_auc < 0.10, f"PR-AUC {pr_auc:.3f} on shuffled labels (failure rate {y_test.mean():.3f}): possible leakage"

    def test_real_labels_are_learnable_by_the_same_setup(self, split):
        """Counterpart of the test above: the same small model does learn the real labels."""
        train, test = split
        X_train, y_train = data_module.features_and_target(train)
        X_test, y_test = data_module.features_and_target(test)
        pipeline = build_pipeline("rf_engineered", {"n_estimators": 60, "max_depth": 12}, n_jobs=1).fit(X_train, y_train)
        assert average_precision_score(y_test, pipeline.predict_proba(X_test)[:, 1]) > 0.6
