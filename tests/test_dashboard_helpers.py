"""UI-independent dashboard logic (everything except the widget layout in app.py)."""

import io
import json

import numpy as np
import pandas as pd
import pytest

from src import config as cfg
from src import dashboard as D
from src import predict as P
from src.recommend import WatchRules
from src.utils import read_json
from src.validation import InputValidationError


# --------------------------------------------------------------------------- #
# Upload parsing
# --------------------------------------------------------------------------- #
CSV = (
    "UDI,Product ID,Type,Air temperature [K],Process temperature [K],Rotational speed [rpm],Torque [Nm],Tool wear [min]\n"
    "1,M14860,M,298.1,308.6,1551,42.8,0\n"
    "2,L47181,L,298.2,308.7,1408,46.3,3\n"
)


class TestReadUploadedCsv:
    def test_comma_separated(self):
        frame = D.read_uploaded_csv(CSV.encode())
        assert frame.shape == (2, 8) and frame["Product ID"].tolist() == ["M14860", "L47181"]

    @pytest.mark.parametrize("sep", [";", "\t"])
    def test_other_separators_are_detected(self, sep):
        frame = D.read_uploaded_csv(CSV.replace(",", sep).encode())
        assert frame.shape == (2, 8)

    def test_utf8_bom_and_crlf(self):
        raw = ("﻿" + CSV.replace("\n", "\r\n")).encode("utf-8")
        frame = D.read_uploaded_csv(raw)
        assert frame.columns[0] == "UDI" and len(frame) == 2

    def test_latin1_bytes_fall_back_instead_of_failing(self):
        raw = CSV.replace("M14860", "M14860é").encode("latin-1")
        frame = D.read_uploaded_csv(raw)
        assert frame["Product ID"].iloc[0] == "M14860é"

    @pytest.mark.parametrize(
        "raw, message",
        [(b"", "empty"), (b"UDI,Type\n", "no data rows")],
    )
    def test_empty_inputs(self, raw, message):
        with pytest.raises(InputValidationError, match=message):
            frame = D.read_uploaded_csv(raw)
            from src.validation import validate_batch

            validate_batch(frame)  # a header-only file parses; validation rejects it

    def test_size_limit(self):
        with pytest.raises(InputValidationError, match="larger than the upload limit of 1.0 KB"):
            D.read_uploaded_csv(b"a" * 2000, max_bytes=1024)

    def test_binary_garbage_is_a_validation_error_not_a_crash(self):
        raw = bytes(np.random.default_rng(0).integers(0, 256, 4000, dtype=np.uint8))
        try:
            frame = D.read_uploaded_csv(raw)
        except InputValidationError as exc:
            assert "could not be read as CSV" in exc.messages[0]
        else:  # if pandas manages to parse it, validation must reject it
            from src.validation import validate_batch

            with pytest.raises(InputValidationError):
                validate_batch(frame)

    def test_too_many_columns(self):
        header = ",".join(f"c{i}" for i in range(D.MAX_UPLOAD_COLUMNS + 1))
        with pytest.raises(InputValidationError, match="columns; at most"):
            D.read_uploaded_csv((header + "\n" + ",".join("1" for _ in range(D.MAX_UPLOAD_COLUMNS + 1)) + "\n").encode())

    def test_round_trip_through_the_real_validator(self, shipped_model):
        batch = P.assess_batch(shipped_model, D.read_uploaded_csv(CSV.encode()))
        assert batch.validation.n_valid == 2


# --------------------------------------------------------------------------- #
# CSV export safety
# --------------------------------------------------------------------------- #
class TestFormulaNeutralisation:
    @pytest.mark.parametrize("value", ["=1+1", "+SUM(A1)", "-2+3", "@cmd", '=HYPERLINK("http://x","y")', "\t=1", "\r=1"])
    def test_formula_like_text_is_prefixed(self, value):
        out = D.neutralise_formulas(pd.DataFrame({"Product ID": [value, "M1"]}))
        assert out["Product ID"].iloc[0] == "'" + value and out["Product ID"].iloc[1] == "M1"

    def test_numbers_and_ordinary_text_are_untouched(self):
        frame = pd.DataFrame({"a": [-1.5, 2.0], "b": ["abc", "x=y"], "c": [-3, 4]})
        pd.testing.assert_frame_equal(D.neutralise_formulas(frame), frame)

    def test_input_frame_is_not_modified(self):
        frame = pd.DataFrame({"id": ["=1"]})
        D.neutralise_formulas(frame)
        assert frame["id"].iloc[0] == "=1"

    def test_csv_bytes_are_utf8_without_index_and_neutralised(self):
        raw = D.results_to_csv(pd.DataFrame({"id": ["=evil()", "ü"], "score": [0.5, 0.25]}))
        assert raw.decode("utf-8").splitlines() == ["id,score", "'=evil(),0.5", "ü,0.25"]

    def test_uploaded_identifier_is_neutralised_in_the_downloadable_results(self, shipped_model):
        upload = pd.read_csv(io.StringIO(CSV))
        upload["Product ID"] = ['=HYPERLINK("http://x","y")', "L47181"]
        table = D.read_uploaded_csv(upload.to_csv(index=False).encode())   # properly quoted CSV
        assert table["Product ID"].iloc[0].startswith("=HYPERLINK")
        results = P.assess_batch(shipped_model, table).results
        text = D.results_to_csv(results).decode("utf-8")
        assert "\"'=HYPERLINK" in text
        parsed = pd.read_csv(io.StringIO(text))
        assert not parsed["product_id"].astype(str).str.startswith("=").any()


# --------------------------------------------------------------------------- #
# Examples and sample file
# --------------------------------------------------------------------------- #
class TestExamples:
    def test_typical_machine_uses_training_medians(self, shipped_model):
        reading = D.typical_reading(shipped_model)
        assert set(reading) == set(cfg.RAW_FEATURES) and reading["type"] in cfg.TYPE_VALUES
        assert reading["torque_nm"] == shipped_model.training_medians["torque_nm"]

    def test_examples_are_real_holdout_records_copied_unchanged(self, shipped_model, holdout):
        examples = D.build_examples(shipped_model, holdout)
        assert examples[0].key == "typical"
        assert {e.key for e in examples} >= {"overstrain", "heat", "power", "missed", "false_alarm"}
        by_udi = holdout.set_index("udi")
        for example in examples[1:]:
            udi = int(example.label.rsplit("UDI ", 1)[1].rstrip(")"))
            row = by_udi.loc[udi]
            assert example.reading == {"type": row["type"], **{c: float(row[c]) for c in cfg.NUMERIC_SENSOR_FEATURES}}
            assert f"UDI {udi}" in example.description

    def test_example_outcomes_match_the_recorded_labels_and_scores(self, shipped_model, holdout):
        column = f"score_{shipped_model.model_key}"
        by_udi = holdout.set_index("udi")
        for example in D.build_examples(shipped_model, holdout)[1:]:
            row = by_udi.loc[int(example.label.rsplit("UDI ", 1)[1].rstrip(")"))]
            flagged = row[column] >= shipped_model.threshold
            if example.key in ("overstrain", "heat", "power"):
                assert row[cfg.TARGET] == 1 and flagged
            elif example.key == "missed":
                assert row[cfg.TARGET] == 1 and not flagged and "no failure" not in example.description
            elif example.key == "false_alarm":
                assert row[cfg.TARGET] == 0 and flagged and "no failure recorded" in example.description

    def test_without_holdout_only_the_typical_machine_is_offered(self, shipped_model):
        assert [e.key for e in D.build_examples(shipped_model, None)] == ["typical"]

    def test_every_example_can_be_assessed(self, shipped_model, holdout):
        for example in D.build_examples(shipped_model, holdout):
            assert 0.0 <= P.assess_reading(shipped_model, example.reading, holdout).score <= 1.0

    def test_sample_file_is_loadable_unlabelled_and_in_the_published_layout(self, shipped_model, holdout):
        sample = D.sample_batch_frame(shipped_model, holdout)
        assert list(sample.columns) == ["UDI", "Product ID", "Type", "Air temperature [K]", "Process temperature [K]",
                                         "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
        assert len(sample) == 10
        csv_bytes = D.results_to_csv(sample)
        batch = P.assess_batch(shipped_model, D.read_uploaded_csv(csv_bytes))
        assert batch.validation.n_valid == len(sample)
        assert set(batch.results["udi"]) <= set(holdout["udi"])

    def test_sample_file_needs_the_holdout_predictions(self, shipped_model):
        with pytest.raises(ValueError, match="not available"):
            D.sample_batch_frame(shipped_model, None)


# --------------------------------------------------------------------------- #
# Tables, filters, numbers
# --------------------------------------------------------------------------- #
class TestFilters:
    def test_filter_combinations(self, dataset):
        both = D.filter_records(dataset, ["L", "M"], "Failures only", (100, 200))
        assert set(both["type"]) <= {"L", "M"} and (both[cfg.TARGET] == 1).all()
        assert both["tool_wear_min"].between(100, 200).all()
        assert len(D.filter_records(dataset, cfg.TYPE_VALUES)) == len(dataset)
        assert len(D.filter_records(dataset, ["H"], "No failure only")) == int(((dataset["type"] == "H") & (dataset[cfg.TARGET] == 0)).sum())
        assert D.filter_records(dataset, []).empty

    def test_unknown_outcome_is_rejected(self, dataset):
        with pytest.raises(ValueError, match="Unknown outcome"):
            D.filter_records(dataset, ["L"], "Everything")


class TestTables:
    def test_threshold_metrics_equals_stored_metrics_at_the_operating_point(self, holdout, shipped_model):
        stored = read_json(cfg.METRICS_PATH)["final"]["test"]
        live = D.threshold_metrics(holdout, shipped_model.model_key, shipped_model.threshold)
        for name in ("precision", "recall", "f1", "tp", "fp", "fn", "tn"):
            assert live[name] == pytest.approx(stored[name], abs=1e-9), name

    def test_threshold_metrics_move_in_the_expected_direction(self, holdout, shipped_model):
        low = D.threshold_metrics(holdout, shipped_model.model_key, 0.05)
        high = D.threshold_metrics(holdout, shipped_model.model_key, 0.9)
        assert low["recall"] >= high["recall"] and low["alert_rate"] > high["alert_rate"]

    def test_overview_numbers_come_from_metrics_json(self):
        metrics = read_json(cfg.METRICS_PATH)
        n = D.overview_numbers(metrics)
        assert (n["records"], n["failures"], n["holdout_records"], n["holdout_failures"]) == (10000, 339, 2000, 68)
        assert n["recall"] == metrics["final"]["test"]["recall"] and set(n["ci95"]) >= {"precision", "recall", "f1"}

    def test_comparison_frame(self):
        frame = D.comparison_frame(read_json(cfg.METRICS_PATH))
        assert len(frame) == 4 and frame["Model"].str.endswith(" - final").sum() == 1
        assert frame.loc[0, "CV PR-AUC (train)"] is None or pd.isna(frame.loc[0, "CV PR-AUC (train)"])
        assert (frame[["TP", "FP", "FN", "TN"]].sum(axis=1) == 2000).all()

    def test_rules_frame_states_the_configured_limits(self):
        text = D.rules_frame().to_string()
        for expected in ("11,000 / 12,000 / 13,000", "3,500", "9,000", "8.6", "1,380", "90%", "200 min"):
            assert expected in text
        custom = D.rules_frame(WatchRules(overstrain_margin=0.25, tool_wear_watch_min=150.0)).to_string()
        assert "75%" in custom and "150 min" in custom

    def test_indicators_frame(self, shipped_model, good_reading):
        frame = D.indicators_frame(P.assess_reading(shipped_model, good_reading))
        assert list(frame.columns) == ["Indicator", "Value", "Limit", "Status", "Meaning", "What to check"]
        assert frame["Status"].tolist() == ["OK"] * 4

    def test_eda_findings_are_computed_from_the_summary(self):
        findings = D.eda_findings(read_json(cfg.EDA_SUMMARY_PATH))
        text = " ".join(findings)
        assert "Failure rate by product variant" in text and "Type L" in text
        assert "consistent with the dataset being synthetic" in text


class TestFormatting:
    @pytest.mark.parametrize("n, text", [(0, "0 B"), (812, "812 B"), (1024, "1.0 KB"), (14541, "14.2 KB"), (10 * 1024**2, "10.0 MB")])
    def test_format_bytes(self, n, text):
        assert D.format_bytes(n) == text

    def test_escape_markdown(self):
        assert D.escape_markdown("a*b_c [d] `e` <f>") == r"a\*b\_c \[d\] \`e\` \<f\>"
        assert D.escape_markdown(5) == "5"

    def test_pages_are_the_four_required_sections(self):
        assert D.PAGES == ("Overview", "Data exploration", "Model evaluation", "Machine assessment")
