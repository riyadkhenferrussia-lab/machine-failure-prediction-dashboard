"""Dataset acquisition/validation, the hold-out split, and validation of user inputs."""

import json

import numpy as np
import pandas as pd
import pytest

from src import config as cfg
from src import data as data_module
from src import validation as V
from src.utils import sha256_of_file


# --------------------------------------------------------------------------- #
# The shipped dataset
# --------------------------------------------------------------------------- #
class TestShippedDataset:
    def test_matches_published_statistics(self, dataset):
        assert len(dataset) == 10_000
        assert int(dataset[cfg.TARGET].sum()) == 339
        assert dataset["type"].value_counts().to_dict() == {"L": 6000, "M": 2997, "H": 1003}
        assert {m: int(dataset[m].sum()) for m in cfg.FAILURE_MODE_COLUMNS} == {
            "twf": 46, "hdf": 115, "pwf": 95, "osf": 98, "rnf": 19,
        }

    def test_file_fingerprint_is_stable(self):
        assert sha256_of_file(cfg.RAW_DATA_PATH) == "59db4f1d9c34c58136d89e5a006ec190dcea19e9dbea74f6b3b0c6f22a44d183"

    def test_report_has_no_errors_and_records_the_published_label_quirks(self, validation_report):
        assert validation_report.ok
        quirks = validation_report.stats["label_inconsistencies"]
        assert quirks == {
            "failure_without_mode_flag": 9,
            "mode_flag_without_failure": 18,
            "rnf_flag_without_failure": 18,
        }
        codes = {issue.code for issue in validation_report.warnings}
        assert {"failure_without_mode", "mode_without_failure"} <= codes

    def test_column_order_and_names_are_canonical(self, dataset):
        assert list(dataset.columns)[: len(cfg.REQUIRED_COLUMNS)] == cfg.REQUIRED_COLUMNS

    def test_missing_file_has_actionable_message(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="data --download"):
            data_module.read_raw_csv(tmp_path / "nope.csv")

    def test_unreadable_file_raises_validation_error(self, tmp_path):
        path = tmp_path / "empty.csv"
        path.write_text("")
        with pytest.raises(V.DataValidationError, match="not a readable CSV"):
            data_module.read_raw_csv(path)


# --------------------------------------------------------------------------- #
# validate_dataset on corrupted copies
# --------------------------------------------------------------------------- #
class TestValidateDataset:
    def codes(self, frame):
        return {issue.code for issue in V.validate_dataset(frame).errors}

    def test_missing_column(self, dataset):
        assert self.codes(dataset.drop(columns=["torque_nm"])) == {"missing_columns"}

    def test_empty_frame(self, dataset):
        assert self.codes(dataset.iloc[0:0]) == {"empty"}

    def test_missing_values(self, dataset):
        broken = dataset.copy()
        broken.loc[3, "torque_nm"] = np.nan
        assert "missing_values" in self.codes(broken)

    def test_non_numeric_sensor(self, dataset):
        broken = dataset.copy()
        broken["torque_nm"] = broken["torque_nm"].astype(str)
        assert "non_numeric" in self.codes(broken)

    def test_invalid_type_value(self, dataset):
        broken = dataset.copy()
        broken.loc[0, "type"] = "X"
        assert "invalid_type" in self.codes(broken)

    def test_non_binary_label(self, dataset):
        broken = dataset.copy()
        broken.loc[0, "machine_failure"] = 2
        assert "non_binary_label" in self.codes(broken)

    def test_celsius_instead_of_kelvin_is_caught(self, dataset):
        broken = dataset.copy()
        broken["air_temp_k"] = broken["air_temp_k"] - 273.15
        assert "out_of_range" in self.codes(broken)

    def test_duplicate_identifier(self, dataset):
        broken = dataset.copy()
        broken.loc[1, "udi"] = broken.loc[0, "udi"]
        assert "duplicate_id" in self.codes(broken)

    def test_row_count_warning_not_error(self, dataset):
        report = V.validate_dataset(dataset.iloc[:100], expected_rows=10_000)
        assert report.ok
        assert "row_count" in {issue.code for issue in report.warnings}

    def test_rule_mismatch_is_a_warning(self, dataset):
        broken = dataset.copy()
        flipped = broken.index[:30]
        broken.loc[flipped, "osf"] = 1 - broken.loc[flipped, "osf"]
        report = V.validate_dataset(broken)
        assert report.ok
        assert "documented_rule_mismatch" in {issue.code for issue in report.warnings}

    def test_raise_if_errors_lists_every_error(self, dataset):
        broken = dataset.copy()
        broken.loc[0, "type"] = "X"
        broken.loc[1, "machine_failure"] = 5
        with pytest.raises(V.DataValidationError) as excinfo:
            V.validate_dataset(broken).raise_if_errors()
        assert "invalid_type" in str(excinfo.value) and "non_binary_label" in str(excinfo.value)

    def test_report_serialises_to_json(self, validation_report):
        payload = json.loads(json.dumps(validation_report.to_dict()))
        assert payload["ok"] is True and payload["n_errors"] == 0


# --------------------------------------------------------------------------- #
# Split and model inputs
# --------------------------------------------------------------------------- #
class TestSplit:
    def test_sizes_and_stratification(self, split):
        train, test = split
        assert (len(train), len(test)) == (8000, 2000)
        assert (int(train[cfg.TARGET].sum()), int(test[cfg.TARGET].sum())) == (271, 68)

    def test_no_overlap_and_full_coverage(self, split, dataset):
        train, test = split
        assert set(train["udi"]).isdisjoint(test["udi"])
        assert set(train["udi"]) | set(test["udi"]) == set(dataset["udi"])

    def test_deterministic_for_a_given_seed(self, dataset):
        a = data_module.split_dataset(dataset, seed=7)
        b = data_module.split_dataset(dataset, seed=7)
        c = data_module.split_dataset(dataset, seed=8)
        assert a[1]["udi"].tolist() == b[1]["udi"].tolist()
        assert a[1]["udi"].tolist() != c[1]["udi"].tolist()

    def test_model_inputs_exclude_every_leakage_column(self, split):
        X, y = data_module.features_and_target(split[0])
        assert list(X.columns) == cfg.RAW_FEATURES
        assert not set(X.columns) & set(cfg.LEAKAGE_COLUMNS)
        assert y.name == cfg.TARGET and set(y.unique()) <= {0, 1}

    def test_leakage_list_covers_label_modes_and_identifiers(self):
        assert set(cfg.LEAKAGE_COLUMNS) == {"machine_failure", "twf", "hdf", "pwf", "osf", "rnf", "udi", "product_id"}
        assert not set(cfg.LEAKAGE_COLUMNS) & set(cfg.RAW_FEATURES + cfg.ENGINEERED_FEATURES)


# --------------------------------------------------------------------------- #
# A single reading typed into the dashboard
# --------------------------------------------------------------------------- #
class TestValidateReading:
    RANGES = {
        "air_temp_k": (295.3, 304.5), "process_temp_k": (305.7, 313.8), "rotational_speed_rpm": (1168.0, 2886.0),
        "torque_nm": (3.8, 76.6), "tool_wear_min": (0.0, 253.0),
    }

    def test_valid_reading_is_normalised(self, good_reading):
        good_reading["type"] = " m "
        good_reading["torque_nm"] = "40"
        result = V.validate_reading(good_reading, self.RANGES)
        assert result.reading["type"] == "M" and result.reading["torque_nm"] == 40.0
        assert list(result.reading) == cfg.RAW_FEATURES
        assert result.warnings == []

    def test_published_column_names_are_accepted(self):
        reading = {"Type": "L", "Air temperature [K]": 300, "Process temperature [K]": 310,
                   "Rotational speed [rpm]": 1500, "Torque [Nm]": 40, "Tool wear [min]": 10}
        assert V.validate_reading(reading).reading["air_temp_k"] == 300.0

    def test_missing_inputs_are_all_listed(self):
        with pytest.raises(V.InputValidationError) as excinfo:
            V.validate_reading({"type": "L"})
        assert "Air temperature [K]" in excinfo.value.messages[0] and "Tool wear [min]" in excinfo.value.messages[0]

    @pytest.mark.parametrize("bad", [None, "", "  ", "abc", float("nan"), float("inf"), True])
    def test_non_numeric_values_are_rejected(self, good_reading, bad):
        good_reading["torque_nm"] = bad
        with pytest.raises(V.InputValidationError, match="Torque"):
            V.validate_reading(good_reading)

    @pytest.mark.parametrize("bad", ["X", "", None, 3])
    def test_bad_type_is_rejected(self, good_reading, bad):
        good_reading["type"] = bad
        with pytest.raises(V.InputValidationError, match="Type must be one of"):
            V.validate_reading(good_reading)

    def test_celsius_input_gets_a_unit_hint(self, good_reading):
        good_reading["air_temp_k"] = 27.0
        with pytest.raises(V.InputValidationError, match="degrees Celsius") as excinfo:
            V.validate_reading(good_reading)
        assert "K = degC + 273.15" in excinfo.value.messages[0]

    def test_all_problems_are_reported_together(self, good_reading):
        good_reading.update(air_temp_k=27.0, torque_nm=-5.0, tool_wear_min="many")
        with pytest.raises(V.InputValidationError) as excinfo:
            V.validate_reading(good_reading)
        assert len(excinfo.value.messages) == 3

    @pytest.mark.parametrize("column", list(cfg.HARD_LIMITS))
    def test_hard_limits_are_inclusive_and_enforced(self, good_reading, column):
        low, high = cfg.HARD_LIMITS[column]
        good_reading["air_temp_k"], good_reading["process_temp_k"] = 300.0, 310.0
        good_reading[column] = high
        V.validate_reading(good_reading)  # exactly on the limit is accepted
        good_reading[column] = high * 1.0001 + 1e-6
        with pytest.raises(V.InputValidationError, match="outside the plausible range"):
            V.validate_reading(good_reading)

    def test_extrapolation_is_a_warning_not_an_error(self, good_reading):
        good_reading["tool_wear_min"] = 400.0
        result = V.validate_reading(good_reading, self.RANGES)
        assert len(result.warnings) == 1 and "extrapolation" in result.warnings[0]

    def test_inverted_temperatures_warn(self, good_reading):
        good_reading.update(air_temp_k=305.0, process_temp_k=300.0)
        warnings = V.validate_reading(good_reading).warnings
        assert any("not above air temperature" in w for w in warnings)

    def test_extra_keys_are_ignored(self, good_reading):
        good_reading["machine_failure"] = 1  # a label must never reach the model
        assert "machine_failure" not in V.validate_reading(good_reading).reading


# --------------------------------------------------------------------------- #
# Uploaded table
# --------------------------------------------------------------------------- #
def sample_table(n=6):
    return pd.DataFrame(
        {
            "UDI": range(1, n + 1),
            "Product ID": [f"M{i}" for i in range(n)],
            "Type": ["L", "M", "H"] * (n // 3),
            "Air temperature [K]": 300.0,
            "Process temperature [K]": 310.0,
            "Rotational speed [rpm]": 1500.0,
            "Torque [Nm]": 40.0,
            "Tool wear [min]": 50.0,
        }
    )


class TestValidateBatch:
    def test_clean_table(self):
        result = V.validate_batch(sample_table())
        assert (result.n_input_rows, result.n_valid, result.n_invalid) == (6, 6, 0)
        assert result.problems.empty
        assert list(result.valid.columns) == ["row", "udi", "product_id"] + cfg.RAW_FEATURES

    def test_row_problems_are_reported_with_row_numbers(self):
        table = sample_table()
        table.loc[0, "Air temperature [K]"] = 25.0          # Celsius
        table.loc[1, "Type"] = "Z"
        table.loc[2, "Torque [Nm]"] = np.nan
        table["Tool wear [min]"] = table["Tool wear [min]"].astype(object)  # as read from a CSV with text in it
        table.loc[3, "Tool wear [min]"] = "n/a"
        result = V.validate_batch(table)
        assert (result.n_valid, result.n_invalid) == (2, 4)
        rows = {(int(r.row), r.column) for r in result.problems.itertuples()}
        assert rows == {(1, "air_temp_k"), (2, "type"), (3, "torque_nm"), (4, "tool_wear_min")}
        assert set(result.valid["row"]) == {5, 6}

    def test_blank_strings_count_as_missing(self):
        table = sample_table()
        table["Torque [Nm]"] = table["Torque [Nm]"].astype(object)
        table.loc[0, "Torque [Nm]"] = "  "
        problems = V.validate_batch(table).problems
        assert problems.loc[0, "problem"] == "missing value"

    def test_infinite_values_are_rejected(self):
        table = sample_table()
        table.loc[0, "Rotational speed [rpm]"] = np.inf
        assert V.validate_batch(table).problems.loc[0, "problem"] == "not finite"

    def test_missing_required_column_is_structural(self):
        with pytest.raises(V.InputValidationError, match=r"Missing required column\(s\): torque_nm"):
            V.validate_batch(sample_table().drop(columns=["Torque [Nm]"]))

    def test_empty_and_oversized_tables_are_refused(self):
        with pytest.raises(V.InputValidationError, match="no data rows"):
            V.validate_batch(sample_table().iloc[0:0])
        with pytest.raises(V.InputValidationError, match="the limit is 3"):
            V.validate_batch(sample_table(), max_rows=3)

    def test_duplicate_spellings_of_a_column_are_refused(self):
        table = sample_table()
        table["torque_nm"] = 1.0
        with pytest.raises(V.InputValidationError, match="both map to"):
            V.validate_batch(table)

    def test_snake_case_headers_and_label_columns(self):
        table = sample_table().rename(columns={v: k for k, v in cfg.RAW_TO_CANONICAL.items()})
        table = table.rename(columns=cfg.RAW_TO_CANONICAL)
        table["machine_failure"] = 1  # present in the upload, must not be used
        result = V.validate_batch(table)
        assert result.n_valid == 6 and "machine_failure" not in result.valid.columns

    def test_extrapolation_summary_is_one_warning_per_column(self):
        table = sample_table()
        table.loc[:1, "Tool wear [min]"] = 500.0
        result = V.validate_batch(table, {"tool_wear_min": (0.0, 253.0)})
        assert result.warnings == ["2 row(s) have Tool wear [min] outside the training range (0 to 253); their scores are extrapolations."]

    def test_all_rows_invalid_gives_empty_valid_frame(self):
        table = sample_table()
        table["Air temperature [K]"] = 20.0
        result = V.validate_batch(table)
        assert result.n_valid == 0 and len(result.problems) == 6
