"""Loading the saved model and assessing single readings and uploaded tables."""

import joblib
import numpy as np
import pandas as pd
import pytest

from src import config as cfg
from src import predict as P
from src.validation import InputValidationError


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
class TestLoadModel:
    def test_missing_file_has_actionable_message(self, tmp_path):
        with pytest.raises(P.ModelNotFoundError, match="python -m src.train"):
            P.load_model(tmp_path / "missing.joblib")

    def test_corrupt_file(self, tmp_path):
        path = tmp_path / "broken.joblib"
        path.write_bytes(b"this is not a joblib file")
        with pytest.raises(P.ModelLoadError, match="could not be loaded"):
            P.load_model(path)

    def test_foreign_object(self, tmp_path):
        path = tmp_path / "other.joblib"
        joblib.dump({"hello": "world"}, path)
        with pytest.raises(P.ModelLoadError, match="not a model bundle"):
            P.load_model(path)

    def test_unsupported_format_version(self, tmp_path, shipped_model):
        bundle = {
            "format_version": 999, "pipeline": shipped_model.pipeline, "threshold": 0.5, "model_key": "x",
            "training_ranges": {}, "training_medians": {}, "metadata": {},
        }
        path = tmp_path / "future.joblib"
        joblib.dump(bundle, path)
        with pytest.raises(P.ModelLoadError, match="Unsupported model format"):
            P.load_model(path)

    def test_errors_are_distinct_types(self):
        assert issubclass(P.ModelNotFoundError, FileNotFoundError)
        assert issubclass(P.ModelLoadError, RuntimeError)

    def test_shipped_model_contents(self, shipped_model):
        assert shipped_model.model_key == "rf_engineered"
        assert 0.0 < shipped_model.threshold < 1.0
        assert set(shipped_model.training_ranges) == set(cfg.NUMERIC_SENSOR_FEATURES)
        assert set(shipped_model.training_medians) == set(cfg.RAW_FEATURES)
        assert shipped_model.label == "Random forest (raw + engineered features)"

    def test_training_ranges_come_from_the_training_split_only(self, shipped_model, split):
        from src.validation import ranges_from_dataframe

        assert shipped_model.training_ranges == ranges_from_dataframe(split[0])

    def test_saved_model_reproduces_the_stored_holdout_scores(self, shipped_model, holdout):
        """The artefact on disk is the model that produced reports/metrics.json."""
        scores = P.score_frame(shipped_model, holdout)
        stored = holdout[f"score_{shipped_model.model_key}"].to_numpy()
        np.testing.assert_allclose(scores, stored, atol=1e-5)


# --------------------------------------------------------------------------- #
# Single readings
# --------------------------------------------------------------------------- #
class TestAssessReading:
    def test_quiet_machine(self, shipped_model, good_reading):
        result = P.assess_reading(shipped_model, good_reading)
        assert 0.0 <= result.score <= 1.0 and result.score < shipped_model.threshold
        assert result.recommendation.level.label == "Low"
        assert result.validation_warnings == [] and result.context is None
        assert result.model_label == shipped_model.label

    def test_documented_overstrain_case_is_high(self, shipped_model):
        reading = {"type": "L", "air_temp_k": 298.0, "process_temp_k": 308.0, "rotational_speed_rpm": 1348.0,
                   "torque_nm": 58.8, "tool_wear_min": 202.0}
        result = P.assess_reading(shipped_model, reading)
        assert result.score >= shipped_model.threshold and result.recommendation.flagged
        assert result.recommendation.level.label == "High"

    def test_context_describes_holdout_position(self, shipped_model, holdout, good_reading):
        result = P.assess_reading(shipped_model, good_reading, holdout)
        ctx = result.context
        assert ctx["holdout_records"] == len(holdout)
        assert 0.0 <= ctx["percentile"] <= 100.0
        assert ctx["recorded_failures_at_or_above"] <= ctx["records_at_or_above"]

    def test_sensitivity_table(self, shipped_model, good_reading):
        table = P.assess_reading(shipped_model, good_reading).sensitivity
        assert sorted(table["input"]) == sorted(cfg.FEATURE_LABELS[f] for f in cfg.RAW_FEATURES)
        assert (table["effect_on_score"].abs().diff().dropna() <= 1e-12).all()  # sorted by absolute effect
        assert table["score_with_reference"].between(0, 1).all()

    def test_invalid_input_raises_with_all_messages(self, shipped_model, good_reading):
        good_reading.update(air_temp_k=27.0, torque_nm=float("nan"))
        with pytest.raises(InputValidationError) as excinfo:
            P.assess_reading(shipped_model, good_reading)
        assert len(excinfo.value.messages) == 2

    def test_out_of_range_input_still_scores_with_a_warning(self, shipped_model, good_reading):
        good_reading["tool_wear_min"] = 400.0
        result = P.assess_reading(shipped_model, good_reading)
        assert len(result.validation_warnings) == 1 and 0.0 <= result.score <= 1.0

    def test_to_dict_is_json_serialisable_and_labelled_as_a_prediction(self, shipped_model, good_reading, holdout):
        import json

        from src.utils import to_jsonable

        payload = P.assess_reading(shipped_model, good_reading, holdout).to_dict()
        text = json.dumps(to_jsonable(payload))
        assert json.loads(text)["kind"] == "screening_result_not_a_confirmed_fault"
        assert "not a confirmed fault" in payload["disclaimer"]
        assert set(payload["inputs"]) == set(cfg.RAW_FEATURES)

    def test_a_label_in_the_reading_is_ignored(self, shipped_model, good_reading):
        base = P.assess_reading(shipped_model, good_reading).score
        good_reading["machine_failure"] = 1
        assert P.assess_reading(shipped_model, good_reading).score == pytest.approx(base)

    def test_score_context_without_holdout(self):
        assert P.score_context(0.5, None, "rf_engineered") is None


# --------------------------------------------------------------------------- #
# Uploaded tables
# --------------------------------------------------------------------------- #
def table_from(holdout, n=30):
    columns = ["udi", "product_id"] + cfg.RAW_FEATURES
    inverse = {v: k for k, v in cfg.RAW_TO_CANONICAL.items()}
    return holdout[columns].head(n).rename(columns=inverse)


class TestAssessBatch:
    def test_batch_scores_equal_single_scores(self, shipped_model, holdout):
        table = table_from(holdout, 25)
        batch = P.assess_batch(shipped_model, table)
        for _, row in batch.results.iterrows():
            reading = {c: row[c] for c in cfg.RAW_FEATURES}
            single = P.assess_reading(shipped_model, reading)
            assert row["failure_score"] == pytest.approx(single.score, abs=1e-9)
            assert row["risk_level"] == single.recommendation.level.label

    def test_results_are_sorted_by_priority_and_counted(self, shipped_model, holdout):
        batch = P.assess_batch(shipped_model, table_from(holdout, 500))
        ranks = batch.results["risk_rank"].tolist()
        assert ranks == sorted(ranks, reverse=True)
        assert sum(batch.counts.values()) == len(batch.results) == 500
        top = batch.results[batch.results["risk_rank"] == batch.results["risk_rank"].max()]
        assert top["failure_score"].is_monotonic_decreasing
        assert {"udi", "product_id", "row"} <= set(batch.results.columns)

    def test_bad_rows_are_reported_not_fatal(self, shipped_model, holdout):
        table = table_from(holdout, 12)
        table.loc[table.index[0], "Air temperature [K]"] = 27.0
        table.loc[table.index[1], "Type"] = "?"
        batch = P.assess_batch(shipped_model, table)
        assert batch.validation.n_invalid == 2 and len(batch.results) == 10
        assert batch.validation.problems["row"].tolist() == [1, 2]

    def test_nothing_valid_gives_empty_results(self, shipped_model, holdout):
        table = table_from(holdout, 5)
        table["Torque [Nm]"] = -1.0
        batch = P.assess_batch(shipped_model, table)
        assert batch.results.empty and batch.counts == {"High": 0, "Elevated": 0, "Low": 0}

    def test_structural_problems_raise(self, shipped_model, holdout):
        with pytest.raises(InputValidationError):
            P.assess_batch(shipped_model, table_from(holdout, 5).drop(columns=["Type"]))

    def test_labels_in_the_upload_are_not_used(self, shipped_model, holdout):
        table = table_from(holdout, 20)
        plain = P.assess_batch(shipped_model, table).results["failure_score"].to_numpy()
        with_labels = table.assign(**{"Machine failure": 1, "TWF": 1, "HDF": 1})
        np.testing.assert_array_equal(plain, P.assess_batch(shipped_model, with_labels).results["failure_score"].to_numpy())
