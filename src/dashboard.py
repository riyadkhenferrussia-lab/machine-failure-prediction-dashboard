"""UI-independent helpers for the Streamlit dashboard.

Everything that can be tested without a browser lives here; ``app.py`` only arranges
widgets around these functions.
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from typing import Any, Sequence

import pandas as pd

from . import config as cfg
from .metrics import classification_metrics
from .modeling import MODEL_SPECS
from .predict import Assessment, LoadedModel
from .recommend import DEFAULT_RULES, WatchRules
from .validation import InputValidationError

PAGES = ("Overview", "Data exploration", "Model evaluation", "Machine assessment")

# A cell that starts with one of these is interpreted as a formula by spreadsheet programs.
SPREADSHEET_TRIGGERS = ("=", "+", "-", "@", "\t", "\r")
MAX_UPLOAD_COLUMNS = 200


# --------------------------------------------------------------------------- #
# Starting points for the single-machine form
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Example:
    key: str
    label: str
    description: str
    reading: dict[str, Any]


def typical_reading(model: LoadedModel) -> dict[str, Any]:
    """Training medians (most common Type): a neutral starting point, not a real machine."""
    medians = model.training_medians
    reading: dict[str, Any] = {"type": str(medians["type"])}
    reading.update({column: float(medians[column]) for column in cfg.NUMERIC_SENSOR_FEATURES})
    return reading


def build_examples(model: LoadedModel, holdout: pd.DataFrame | None) -> list[Example]:
    """Starting points: the typical machine plus real hold-out records with their recorded outcome.

    The records are copied unchanged from ``reports/test_predictions.csv`` (never
    generated); the recorded outcome comes from the dataset's own label.
    """
    examples = [
        Example(
            "typical",
            "Typical machine (training medians)",
            "Training-set medians and the most common Type. Not a real machine.",
            typical_reading(model),
        )
    ]
    column = f"score_{model.model_key}"
    needed = {column, cfg.TARGET, "udi", *cfg.RAW_FEATURES, *cfg.FAILURE_MODE_COLUMNS}
    if holdout is None or not needed <= set(holdout.columns):
        return examples

    ordered = holdout.sort_values("udi")
    flagged = ordered[column] >= model.threshold
    failed = ordered[cfg.TARGET] == 1
    only = lambda mode: ordered[mode] == 1  # noqa: E731 - tiny local helper

    def mode_only(mode: str) -> pd.Series:
        others = [m for m in ("hdf", "pwf", "osf", "twf", "rnf") if m != mode]
        mask = failed & flagged & only(mode)
        for other in others:
            mask &= ordered[other] == 0
        return mask

    candidates = [
        ("overstrain", "Overstrain failure, flagged by the model", mode_only("osf")),
        ("heat", "Heat-dissipation failure, flagged by the model", mode_only("hdf")),
        ("power", "Power failure, flagged by the model", mode_only("pwf")),
        ("missed", "Recorded failure the model did not flag", failed & ~flagged),
        ("false_alarm", "Flagged by the model, no recorded failure", ~failed & flagged),
    ]
    for key, label, mask in candidates:
        rows = ordered[mask]
        if rows.empty:
            continue
        row = rows.iloc[0]
        outcome = "failure recorded" if int(row[cfg.TARGET]) == 1 else "no failure recorded"
        examples.append(
            Example(
                key,
                f"{label} (hold-out UDI {int(row['udi'])})",
                f"Real record from the hold-out set, copied unchanged: UDI {int(row['udi'])}, "
                f"Type {row['type']}, {outcome}, model score {float(row[column]):.3f}.",
                {"type": str(row["type"]), **{c: float(row[c]) for c in cfg.NUMERIC_SENSOR_FEATURES}},
            )
        )
    return examples


# --------------------------------------------------------------------------- #
# Batch upload / download
# --------------------------------------------------------------------------- #
def read_uploaded_csv(raw: bytes, max_bytes: int = cfg.MAX_UPLOAD_BYTES) -> pd.DataFrame:
    """Parse an uploaded CSV (comma, semicolon or tab separated). Raises ``InputValidationError``."""
    if not raw:
        raise InputValidationError(["The uploaded file is empty."])
    if len(raw) > max_bytes:
        raise InputValidationError(
            [f"The file is larger than the upload limit of {format_bytes(max_bytes)}. Split it into smaller files."]
        )

    def parse(encoding: str) -> pd.DataFrame:
        return pd.read_csv(io.BytesIO(raw), sep=None, engine="python", encoding=encoding)

    try:
        try:
            frame = parse("utf-8-sig")
        except UnicodeDecodeError:
            frame = parse("latin-1")  # never fails to decode; bad content shows up in validation
    except (pd.errors.ParserError, pd.errors.EmptyDataError, csv.Error, ValueError) as exc:
        raise InputValidationError([f"The file could not be read as CSV: {exc}"]) from exc
    if frame.shape[1] > MAX_UPLOAD_COLUMNS:
        raise InputValidationError(
            [f"The file has {frame.shape[1]} columns; at most {MAX_UPLOAD_COLUMNS} are accepted."]
        )
    return frame


def neutralise_formulas(frame: pd.DataFrame) -> pd.DataFrame:
    """Prefix text cells that start like a spreadsheet formula with an apostrophe.

    Uploaded identifiers end up in the result file; without this a cell such as
    ``=HYPERLINK(...)`` would run when the downloaded CSV is opened in a spreadsheet.
    """
    safe = frame.copy()
    for column in safe.columns:
        if safe[column].dtype == object or isinstance(safe[column].dtype, pd.StringDtype):
            safe[column] = safe[column].map(
                lambda value: "'" + value if isinstance(value, str) and value.startswith(SPREADSHEET_TRIGGERS) else value
            )
    return safe


def results_to_csv(frame: pd.DataFrame) -> bytes:
    """CSV bytes for download (UTF-8, no index, formula-neutralised)."""
    return neutralise_formulas(frame).to_csv(index=False).encode("utf-8")


def sample_batch_frame(model: LoadedModel, holdout: pd.DataFrame | None, n_quiet: int = 5) -> pd.DataFrame:
    """A small upload template of real hold-out records, in the dataset's own CSV layout.

    It mixes machines flagged by the model, a recorded failure that was missed, a false
    alarm and a few quiet machines. Labels and failure-mode columns are not included.
    """
    column = f"score_{model.model_key}"
    needed = {column, cfg.TARGET, "udi", "product_id", *cfg.RAW_FEATURES}
    if holdout is None or not needed <= set(holdout.columns):
        raise ValueError("The hold-out predictions are not available, so no sample file can be built.")
    ordered = holdout.sort_values("udi")
    flagged = ordered[column] >= model.threshold
    failed = ordered[cfg.TARGET] == 1
    parts = [
        ordered[failed & flagged].head(3),
        ordered[failed & ~flagged].head(1),
        ordered[~failed & flagged].head(1),
        ordered[~failed & ~flagged].head(n_quiet),
    ]
    sample = pd.concat(parts).sort_values("udi")
    to_original = {canonical: raw for raw, canonical in cfg.RAW_TO_CANONICAL.items()}
    columns = ["udi", "product_id", *cfg.RAW_FEATURES]
    return sample[columns].rename(columns=to_original).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Tables and numbers shown on the pages
# --------------------------------------------------------------------------- #
def indicators_frame(assessment: Assessment) -> pd.DataFrame:
    """The engineering indicators of one assessment as a display table."""
    table = pd.DataFrame([indicator.to_dict() for indicator in assessment.recommendation.indicators])
    return table.rename(
        columns={
            "indicator": "Indicator",
            "value": "Value",
            "limit": "Limit",
            "status": "Status",
            "explanation": "Meaning",
            "what_to_check": "What to check",
        }
    )


def threshold_metrics(holdout: pd.DataFrame, model_key: str, threshold: float) -> dict[str, float]:
    """Hold-out metrics of a model at an arbitrary threshold (exploration only)."""
    return classification_metrics(holdout[cfg.TARGET], holdout[f"score_{model_key}"], threshold)


def filter_records(
    frame: pd.DataFrame,
    types: Sequence[str],
    outcome: str = "All records",
    wear_range: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """Row filter for the data explorer. ``outcome``: All records / Failures only / No failure only."""
    mask = frame["type"].isin(list(types))
    if outcome == "Failures only":
        mask &= frame[cfg.TARGET] == 1
    elif outcome == "No failure only":
        mask &= frame[cfg.TARGET] == 0
    elif outcome != "All records":
        raise ValueError(f"Unknown outcome filter {outcome!r}")
    if wear_range is not None:
        low, high = wear_range
        mask &= frame["tool_wear_min"].between(low, high)
    return frame[mask]


def overview_numbers(metrics: dict[str, Any]) -> dict[str, Any]:
    """Headline numbers for the overview page, read from ``reports/metrics.json``."""
    meta, final = metrics["meta"], metrics["final"]
    test = final["test"]
    failures = meta["failures_train"] + meta["failures_test"]
    return {
        "records": meta["n_rows"],
        "failures": failures,
        "failure_rate": failures / meta["n_rows"],
        "model": final["label"],
        "threshold": final["threshold"],
        "holdout_records": meta["n_test"],
        "holdout_failures": meta["failures_test"],
        "precision": test["precision"],
        "recall": test["recall"],
        "f1": test["f1"],
        "pr_auc": test["pr_auc"],
        "roc_auc": test["roc_auc"],
        "created_utc": meta["created_utc"],
        "environment": meta["environment"],
        "ci95": final["test_ci95"],
    }


def format_bytes(n: int) -> str:
    """Human-readable size: 812 B, 14.2 KB, 3.5 MB."""
    if n < 1024:
        return f"{n} B"
    if n < 1024**2:
        return f"{n / 1024:.1f} KB"
    return f"{n / 1024**2:.1f} MB"


_MARKDOWN_SPECIALS = re.compile(r"([\\`*_\[\]<>~$|])")


def escape_markdown(text: str) -> str:
    """Backslash-escape characters that Markdown would interpret (``*``, ``_``, backticks, ...)."""
    return _MARKDOWN_SPECIALS.sub(r"\\\1", str(text))


def comparison_frame(metrics: dict[str, Any]) -> pd.DataFrame:
    """Hold-out comparison of all candidate models (values straight from ``metrics.json``)."""
    rows = []
    final_key = metrics["final"]["key"]
    for key in ("baseline_majority", "logreg_raw", "rf_raw", "rf_engineered"):
        entry = metrics["models"][key]
        test = entry["test"]
        rows.append(
            {
                "Model": MODEL_SPECS[key].label + (" - final" if key == final_key else ""),
                "CV PR-AUC (train)": None if key == "baseline_majority" else round(entry["cv_pr_auc_mean"], 3),
                "Threshold": round(entry["threshold"], 3),
                "Precision": round(test["precision"], 3),
                "Recall": round(test["recall"], 3),
                "F1": round(test["f1"], 3),
                "PR-AUC": round(test["pr_auc"], 3),
                "ROC-AUC": round(test["roc_auc"], 3),
                "TP": test["tp"],
                "FP": test["fp"],
                "FN": test["fn"],
                "TN": test["tn"],
            }
        )
    return pd.DataFrame(rows)


def rules_frame(rules: WatchRules = DEFAULT_RULES) -> pd.DataFrame:
    """The recommendation rules as a table (limits come from ``src/config.py``)."""
    limits = " / ".join(f"{v:,.0f}" for v in cfg.OSF_LIMITS_MIN_NM.values())
    return pd.DataFrame(
        [
            {
                "Indicator": "Overstrain (tool wear x torque)",
                "ALERT when": f"above {limits} min*Nm for Type L / M / H",
                "WATCH when": f"at least {1 - rules.overstrain_margin:.0%} of that limit",
                "Basis": "Documented OSF criterion of the AI4I 2020 data",
            },
            {
                "Indicator": "Mechanical power (torque x speed)",
                "ALERT when": f"below {cfg.POWER_MIN_W:,.0f} W or above {cfg.POWER_MAX_W:,.0f} W",
                "WATCH when": f"within {rules.power_margin:.0%} of either limit",
                "Basis": "Documented PWF criterion",
            },
            {
                "Indicator": "Heat dissipation",
                "ALERT when": f"temperature difference below {cfg.HDF_TEMP_DIFF_MAX_K} K and speed below {cfg.HDF_SPEED_MAX_RPM:,.0f} rpm",
                "WATCH when": f"both within {rules.heat_margin:.0%} of those limits",
                "Basis": "Documented HDF criterion",
            },
            {
                "Indicator": "Tool wear window",
                "ALERT when": "never (the window is probabilistic)",
                "WATCH when": f"tool wear at least {rules.tool_wear_watch_min:,.0f} min",
                "Basis": "Documented TWF window of 200-240 min",
            },
            {
                "Indicator": "Model score",
                "ALERT when": "High: score at or above the operating threshold",
                "WATCH when": f"Elevated: score from {rules.elevated_score_fraction:.0%} of the threshold",
                "Basis": "Threshold chosen on training data (maximum F2)",
            },
        ]
    )


def eda_findings(summary: dict[str, Any]) -> list[str]:
    """Plain-language findings, each computed from ``reports/eda_summary.json``."""
    findings = []
    by_type = summary["failure_rate_by_type"]
    parts = ", ".join(f"Type {t} {by_type[t]['rate']:.1%} (n={by_type[t]['records']:,})" for t in cfg.TYPE_VALUES if t in by_type)
    findings.append(f"Failure rate by product variant: {parts}.")

    bins = list(summary["failure_rate_by_tool_wear_bin"].items())
    if len(bins) >= 2:
        last_name, last = bins[-1]
        rest_failures = sum(b["failures"] for _, b in bins[:-1])
        rest_records = sum(b["records"] for _, b in bins[:-1])
        findings.append(
            f"Tool wear of {last_name.replace(' min', '')} min goes with a failure rate of {last['rate']:.1%} "
            f"({last['failures']} of {last['records']:,} records), against {rest_failures / rest_records:.1%} below it."
        )

    ranked = sorted(summary["correlation_with_failure"].items(), key=lambda kv: -abs(kv[1]))[:3]
    names = ", ".join(f"{cfg.SHORT_LABELS.get(k, k)} ({v:+.2f})" for k, v in ranked)
    findings.append(
        f"Strongest single-feature correlations with failure: {names}. Correlations are weak individually; "
        "failures arise from combinations (power, wear x torque, temperature difference at low speed)."
    )
    agreement = summary.get("documented_rule_agreement", {})
    if agreement and all(v == 1.0 for v in agreement.values()):
        findings.append(
            "The documented criteria for power, heat-dissipation and overstrain failures reproduce those labels "
            "exactly in this file, consistent with the dataset being synthetic (generated from rules plus random components)."
        )
    return findings

