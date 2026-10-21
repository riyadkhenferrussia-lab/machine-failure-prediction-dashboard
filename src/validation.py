"""Validation of the training dataset and of user-supplied inputs.

Two separate concerns:

* ``validate_dataset``  - blocking *errors* and informative *warnings* about the
  training data (schema, types, ranges, label consistency, documented physics).
* ``validate_reading`` / ``validate_batch`` - validation of what a user types into
  the dashboard or uploads as CSV before it reaches the model. Hard errors mean
  "cannot be scored"; warnings mean "can be scored, but treat with care".
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from . import config as cfg
from .physics import documented_rule_agreement
from .schema import accepted_names, column_mapping

Ranges = Mapping[str, tuple[float, float]]


class DataValidationError(ValueError):
    """The dataset failed a blocking validation check."""


class InputValidationError(ValueError):
    """A user-supplied reading or file is invalid; ``messages`` lists every problem."""

    def __init__(self, messages: Iterable[str]):
        self.messages = list(messages)
        super().__init__("; ".join(self.messages))


# --------------------------------------------------------------------------- #
# Dataset validation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ValidationIssue:
    level: str  # "error" | "warning" | "info"
    code: str
    message: str


@dataclass
class ValidationReport:
    issues: list[ValidationIssue] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def add(self, level: str, code: str, message: str) -> None:
        self.issues.append(ValidationIssue(level, code, message))

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.level == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_errors(self) -> None:
        if self.errors:
            raise DataValidationError("; ".join(f"[{i.code}] {i.message}" for i in self.errors))

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "n_errors": len(self.errors),
            "n_warnings": len(self.warnings),
            "issues": [asdict(i) for i in self.issues],
            "stats": self.stats,
        }


def validate_dataset(df: pd.DataFrame, expected_rows: int | None = None) -> ValidationReport:
    """Validate a dataset that uses canonical column names."""
    report = ValidationReport()

    missing = [c for c in cfg.REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        report.add("error", "missing_columns", f"Missing required columns: {missing}")
        return report
    if df.empty:
        report.add("error", "empty", "The dataset contains no rows")
        return report

    required = df[cfg.REQUIRED_COLUMNS]
    null_counts = required.isna().sum()
    null_counts = null_counts[null_counts > 0]
    if len(null_counts):
        report.add(
            "error",
            "missing_values",
            "Missing values in: " + ", ".join(f"{c} ({n})" for c, n in null_counts.items()),
        )

    numeric_cols = cfg.NUMERIC_SENSOR_FEATURES + [cfg.TARGET] + cfg.FAILURE_MODE_COLUMNS + ["udi"]
    non_numeric = [c for c in numeric_cols if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        report.add("error", "non_numeric", f"Columns that must be numeric are not: {non_numeric}")

    bad_type = ~df["type"].isin(cfg.TYPE_VALUES)
    if bad_type.any():
        values = sorted(map(str, df.loc[bad_type, "type"].unique()))[:5]
        report.add(
            "error",
            "invalid_type",
            f"{int(bad_type.sum())} rows have a Type outside {cfg.TYPE_VALUES}: {values}",
        )

    for col in [cfg.TARGET] + cfg.FAILURE_MODE_COLUMNS:
        if col in non_numeric:
            continue
        bad = ~df[col].isin([0, 1])
        if bad.any():
            report.add("error", "non_binary_label", f"{col}: {int(bad.sum())} values are not 0/1")

    for col, (low, high) in cfg.HARD_LIMITS.items():
        if col in non_numeric:
            continue
        outside = (df[col] < low) | (df[col] > high)
        if outside.any():
            report.add(
                "error",
                "out_of_range",
                f"{col}: {int(outside.sum())} values outside the physical sanity range "
                f"[{low:g}, {high:g}]",
            )

    for col in cfg.IDENTIFIER_COLUMNS:
        duplicated = int(df[col].duplicated().sum())
        if duplicated:
            report.add("error", "duplicate_id", f"{col}: {duplicated} duplicated identifiers")

    if report.errors:
        return report  # the checks below assume a structurally sound frame

    n_dup_rows = int(df.duplicated().sum())
    if n_dup_rows:
        report.add("warning", "duplicate_rows", f"{n_dup_rows} fully duplicated rows")

    if expected_rows is not None and len(df) != expected_rows:
        report.add(
            "warning",
            "row_count",
            f"Expected {expected_rows} rows (published AI4I 2020 size), found {len(df)}",
        )

    failed = df[cfg.TARGET] == 1
    any_mode = df[cfg.FAILURE_MODE_COLUMNS].sum(axis=1) > 0
    n_fail_no_mode = int((failed & ~any_mode).sum())
    n_mode_no_fail = int((~failed & any_mode).sum())
    n_rnf_no_fail = int(((df["rnf"] == 1) & ~failed).sum())
    if n_fail_no_mode:
        report.add(
            "warning",
            "failure_without_mode",
            f"{n_fail_no_mode} rows have Machine failure = 1 but no failure-mode flag set",
        )
    if n_mode_no_fail:
        report.add(
            "warning",
            "mode_without_failure",
            f"{n_mode_no_fail} rows have a failure-mode flag set but Machine failure = 0 "
            f"({n_rnf_no_fail} of them flagged RNF). Labels are kept exactly as published.",
        )

    n_inverted = int((df["process_temp_k"] <= df["air_temp_k"]).sum())
    if n_inverted:
        report.add(
            "warning", "temperature_order", f"{n_inverted} rows have process temp <= air temp"
        )

    agreement = documented_rule_agreement(df)
    for mode, share in agreement.items():
        if share < 1.0:
            report.add(
                "warning",
                "documented_rule_mismatch",
                f"The documented {mode.upper()} criterion reproduces only {share:.2%} of the "
                f"{mode.upper()} flags; the file may differ from the published dataset",
            )

    report.stats = {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "n_failures": int(df[cfg.TARGET].sum()),
        "failure_rate": float(df[cfg.TARGET].mean()),
        "type_counts": {str(k): int(v) for k, v in df["type"].value_counts().items()},
        "failure_mode_counts": {m: int(df[m].sum()) for m in cfg.FAILURE_MODE_COLUMNS},
        "label_inconsistencies": {
            "failure_without_mode_flag": n_fail_no_mode,
            "mode_flag_without_failure": n_mode_no_fail,
            "rnf_flag_without_failure": n_rnf_no_fail,
        },
        "documented_rule_agreement": agreement,
        "ranges": {
            c: {
                "min": float(df[c].min()),
                "max": float(df[c].max()),
                "mean": float(df[c].mean()),
                "std": float(df[c].std()),
            }
            for c in cfg.NUMERIC_SENSOR_FEATURES
        },
    }
    return report


def ranges_from_dataframe(df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Min/max of every numeric sensor column (used to flag extrapolation)."""
    return {c: (float(df[c].min()), float(df[c].max())) for c in cfg.NUMERIC_SENSOR_FEATURES}


# --------------------------------------------------------------------------- #
# Single reading
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ReadingValidation:
    reading: dict[str, Any]  # canonical keys, numeric values as float, type upper-case
    warnings: list[str]


def _label(column: str) -> str:
    return cfg.FEATURE_LABELS.get(column, column)


def _unit_hint(column: str, value: float) -> str:
    if column in ("air_temp_k", "process_temp_k") and -100.0 <= value <= 150.0:
        return " This looks like degrees Celsius; enter kelvin (K = degC + 273.15)."
    return ""


def _to_float(value: Any) -> float | None:
    """Convert to float, rejecting booleans, blanks and non-numeric text."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number


def validate_reading(reading: Mapping[str, Any], training_ranges: Ranges | None = None) -> ReadingValidation:
    """Validate one machine reading; raise ``InputValidationError`` listing all problems."""
    problems: list[str] = []
    warnings: list[str] = []

    try:
        mapping = column_mapping(reading.keys())
    except ValueError as exc:
        raise InputValidationError([str(exc)]) from exc
    canonical = {mapping[k]: v for k, v in reading.items() if k in mapping}

    missing = [c for c in cfg.RAW_FEATURES if c not in canonical]
    if missing:
        problems.append("Missing inputs: " + ", ".join(_label(c) for c in missing))

    clean: dict[str, Any] = {}
    type_value = canonical.get("type")
    if "type" in canonical:
        text = str(type_value).strip().upper() if type_value is not None else ""
        if text in cfg.TYPE_VALUES:
            clean["type"] = text
        else:
            problems.append(f"Type must be one of {cfg.TYPE_VALUES}, got {type_value!r}")

    for column in cfg.NUMERIC_SENSOR_FEATURES:
        if column not in canonical:
            continue
        number = _to_float(canonical[column])
        if number is None or not math.isfinite(number):
            problems.append(f"{_label(column)} must be a finite number, got {canonical[column]!r}")
            continue
        low, high = cfg.HARD_LIMITS[column]
        if not low <= number <= high:
            problems.append(
                f"{_label(column)} = {number:g} is outside the plausible range "
                f"[{low:g}, {high:g}]." + _unit_hint(column, number)
            )
            continue
        clean[column] = number

    if problems:
        raise InputValidationError(problems)

    if training_ranges:
        for column in cfg.NUMERIC_SENSOR_FEATURES:
            if column in training_ranges:
                low, high = training_ranges[column]
                if not low <= clean[column] <= high:
                    warnings.append(
                        f"{_label(column)} = {clean[column]:g} is outside the range seen in "
                        f"training ({low:g} to {high:g}); the model has not seen such values, "
                        "so the score is an extrapolation."
                    )
    if clean["process_temp_k"] <= clean["air_temp_k"]:
        warnings.append(
            "Process temperature is not above air temperature, which never occurs in the "
            "training data; please re-check both temperatures."
        )
    return ReadingValidation(reading={k: clean[k] for k in cfg.RAW_FEATURES}, warnings=warnings)


# --------------------------------------------------------------------------- #
# Batch (CSV) validation
# --------------------------------------------------------------------------- #
@dataclass
class BatchValidation:
    valid: pd.DataFrame  # row (1-based data row), id columns if present, then RAW_FEATURES
    problems: pd.DataFrame  # columns: row, column, problem   (row = data row, 1-based)
    warnings: list[str]
    n_input_rows: int

    @property
    def n_valid(self) -> int:
        return int(len(self.valid))

    @property
    def n_invalid(self) -> int:
        return self.n_input_rows - self.n_valid


def validate_batch(
    df: pd.DataFrame,
    training_ranges: Ranges | None = None,
    max_rows: int = cfg.MAX_BATCH_ROWS,
) -> BatchValidation:
    """Validate an uploaded table. Structural problems raise; row problems are reported."""
    if df is None or df.empty:
        raise InputValidationError(["The file contains no data rows."])
    if len(df) > max_rows:
        raise InputValidationError(
            [f"The file has {len(df):,} rows; the limit is {max_rows:,}. Split the file."]
        )

    try:
        mapping = column_mapping(df.columns)
    except ValueError as exc:
        raise InputValidationError([str(exc)]) from exc
    frame = df.rename(columns=mapping)

    missing = [c for c in cfg.RAW_FEATURES if c not in frame.columns]
    if missing:
        raise InputValidationError(
            [
                "Missing required column(s): "
                + "; ".join(f"{c} (accepted: {', '.join(accepted_names(c))})" for c in missing)
            ]
        )

    n = len(frame)
    row_numbers = np.arange(1, n + 1)
    problem_rows: list[pd.DataFrame] = []
    bad = np.zeros(n, dtype=bool)

    def flag(mask: np.ndarray, column: str, text: Any) -> None:
        nonlocal bad
        mask = np.asarray(mask, dtype=bool)
        if mask.any():
            texts = text if isinstance(text, (list, np.ndarray, pd.Series)) else [text] * int(mask.sum())
            problem_rows.append(
                pd.DataFrame(
                    {"row": row_numbers[mask], "column": column, "problem": list(texts)}
                )
            )
            bad |= mask

    type_raw = frame["type"]
    type_clean = type_raw.astype(str).str.strip().str.upper()
    type_missing = type_raw.isna().to_numpy(dtype=bool) | type_clean.eq("").to_numpy(dtype=bool)
    type_invalid = ~type_clean.isin(cfg.TYPE_VALUES).to_numpy(dtype=bool) & ~type_missing
    flag(type_missing, "type", "missing value")
    flag(type_invalid, "type", f"must be one of {cfg.TYPE_VALUES}")

    numeric = pd.DataFrame(index=frame.index)
    for column in cfg.NUMERIC_SENSOR_FEATURES:
        original = frame[column]
        converted = pd.to_numeric(original, errors="coerce").astype(float)
        numeric[column] = converted
        was_blank = original.isna().to_numpy(dtype=bool)
        if original.dtype == object or isinstance(original.dtype, pd.StringDtype):
            was_blank = was_blank | original.astype(str).str.strip().eq("").to_numpy(dtype=bool)
        flag(was_blank, column, "missing value")
        not_a_number = converted.isna().to_numpy() & ~was_blank
        flag(not_a_number, column, "not a number")
        infinite = np.isinf(converted.to_numpy())
        flag(infinite, column, "not finite")
        low, high = cfg.HARD_LIMITS[column]
        values = converted.to_numpy()
        with np.errstate(invalid="ignore"):
            outside = (values < low) | (values > high)
        outside &= ~np.isnan(values) & ~infinite
        flag(
            outside,
            column,
            [f"{v:g} outside plausible range [{low:g}, {high:g}]" for v in values[outside]],
        )

    keep_ids = [c for c in cfg.IDENTIFIER_COLUMNS if c in frame.columns]
    valid = pd.concat(
        [frame[keep_ids], type_clean.rename("type"), numeric], axis=1
    )[keep_ids + cfg.RAW_FEATURES]
    valid["type"] = valid["type"].astype(str)
    valid.insert(0, "row", row_numbers)
    valid = valid.loc[~bad].copy()

    warnings: list[str] = []
    if training_ranges and len(valid):
        for column in cfg.NUMERIC_SENSOR_FEATURES:
            if column in training_ranges:
                low, high = training_ranges[column]
                n_out = int(((valid[column] < low) | (valid[column] > high)).sum())
                if n_out:
                    warnings.append(
                        f"{n_out} row(s) have {_label(column)} outside the training range "
                        f"({low:g} to {high:g}); their scores are extrapolations."
                    )
    if len(valid):
        n_inverted = int((valid["process_temp_k"] <= valid["air_temp_k"]).sum())
        if n_inverted:
            warnings.append(f"{n_inverted} row(s) have process temperature <= air temperature.")

    problems = (
        pd.concat(problem_rows, ignore_index=True).sort_values(["row", "column"], ignore_index=True)
        if problem_rows
        else pd.DataFrame(columns=["row", "column", "problem"])
    )
    return BatchValidation(valid=valid, problems=problems, warnings=warnings, n_input_rows=n)
