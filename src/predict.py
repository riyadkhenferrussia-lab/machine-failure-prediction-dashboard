"""Loading the saved model and assessing machine readings (single and batch).

All assessments are *screening results*: a model score plus transparent engineering
indicators. They are predictions, never confirmed faults.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning

from . import config as cfg
from .recommend import DISCLAIMER, Recommendation, batch_recommendations, recommend
from .validation import BatchValidation, validate_batch, validate_reading


class ModelNotFoundError(FileNotFoundError):
    """No saved model exists at the expected location."""


class ModelLoadError(RuntimeError):
    """The saved model exists but cannot be used (corrupt, wrong format or library version)."""


@dataclass
class LoadedModel:
    pipeline: Any
    threshold: float
    model_key: str
    training_ranges: dict[str, tuple[float, float]]
    training_medians: dict[str, Any]
    metadata: dict[str, Any]
    path: Path | None = None

    @property
    def label(self) -> str:
        from .modeling import MODEL_SPECS

        spec = MODEL_SPECS.get(self.model_key)
        return spec.label if spec else self.model_key


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_model(path: Path | str | None = None) -> LoadedModel:
    """Load the saved model bundle; raise a clear error when it is missing or unusable."""
    path = Path(path) if path is not None else cfg.MODEL_PATH
    if not path.exists():
        raise ModelNotFoundError(
            f"No saved model at {path}. Train one with `python -m src.train` "
            "(add --no-search for a fast run)."
        )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)
            bundle = joblib.load(path)
    except Exception as exc:  # noqa: BLE001 - any unpickling failure means "unusable model"
        raise ModelLoadError(
            f"The saved model at {path} could not be loaded ({type(exc).__name__}: {exc}). "
            "It was probably created with different library versions; retrain it with "
            "`python -m src.train`."
        ) from exc
    required = {"format_version", "pipeline", "threshold", "model_key", "training_ranges", "training_medians", "metadata"}
    if not isinstance(bundle, dict) or not required <= set(bundle):
        raise ModelLoadError(f"{path} is not a model bundle created by this project.")
    if bundle["format_version"] != cfg.MODEL_BUNDLE_FORMAT:
        raise ModelLoadError(
            f"Unsupported model format {bundle['format_version']!r}; retrain with `python -m src.train`."
        )
    return LoadedModel(
        pipeline=bundle["pipeline"],
        threshold=float(bundle["threshold"]),
        model_key=str(bundle["model_key"]),
        training_ranges={k: (float(v[0]), float(v[1])) for k, v in bundle["training_ranges"].items()},
        training_medians=dict(bundle["training_medians"]),
        metadata=dict(bundle["metadata"]),
        path=path,
    )


def ensure_model(log: Callable[[str], None] = lambda message: None) -> LoadedModel:
    """Load the default model; if it is missing or incompatible, retrain it quickly.

    The quick retrain skips the hyper-parameter search (default parameters, see
    ``src/config.py``) and takes well under a minute.
    """
    try:
        return load_model()
    except (ModelNotFoundError, ModelLoadError) as exc:
        log(f"{exc}\nRetraining a model with default hyper-parameters ...")
    from .train import TrainingOptions, run_training

    run_training(TrainingOptions(search=False, bootstrap=200, n_jobs=1), log=log)
    return load_model()


def load_holdout_predictions(path: Path | str | None = None) -> pd.DataFrame | None:
    """Hold-out records with the scores of every model (``None`` if not generated yet)."""
    path = Path(path) if path is not None else cfg.PREDICTIONS_PATH
    return pd.read_csv(path) if path.exists() else None


# --------------------------------------------------------------------------- #
# Scoring helpers
# --------------------------------------------------------------------------- #
def score_frame(model: LoadedModel, frame: pd.DataFrame) -> np.ndarray:
    """Failure scores for rows with canonical raw-feature columns."""
    return model.pipeline.predict_proba(frame[cfg.RAW_FEATURES])[:, 1]


def local_sensitivity(model: LoadedModel, reading: Mapping[str, Any], score: float | None = None) -> pd.DataFrame:
    """Occlusion-style explanation of one score.

    Each input is replaced, one at a time, by its training-set median (the most
    frequent Type for the categorical input) and the model is re-scored.
    ``effect_on_score`` = score - score_with_reference: positive values mean the input
    pushes the score up relative to a typical machine. Inputs are correlated, so this is
    a transparent approximation, not an exact attribution.
    """
    base = pd.DataFrame([dict(reading)])[cfg.RAW_FEATURES]
    variants = []
    for feature in cfg.RAW_FEATURES:
        variant = base.copy()
        variant.loc[0, feature] = model.training_medians[feature]
        variants.append(variant)
    scores_ref = score_frame(model, pd.concat(variants, ignore_index=True))
    base_score = float(score if score is not None else score_frame(model, base)[0])
    table = pd.DataFrame(
        {
            "input": [cfg.FEATURE_LABELS[f] for f in cfg.RAW_FEATURES],
            "value": [reading[f] for f in cfg.RAW_FEATURES],
            "reference (training median)": [model.training_medians[f] for f in cfg.RAW_FEATURES],
            "score_with_reference": scores_ref,
            "effect_on_score": base_score - scores_ref,
        }
    )
    order = table["effect_on_score"].abs().sort_values(ascending=False).index
    return table.loc[order].reset_index(drop=True)


def score_context(score: float, holdout: pd.DataFrame | None, model_key: str) -> dict[str, Any] | None:
    """Where a score sits relative to the hold-out records the model never trained on."""
    column = f"score_{model_key}"
    if holdout is None or column not in holdout or cfg.TARGET not in holdout:
        return None
    scores = holdout[column].to_numpy(dtype=float)
    at_or_above = holdout[scores >= score]
    return {
        "holdout_records": int(len(holdout)),
        "percentile": float((scores < score).mean() * 100.0),
        "records_at_or_above": int(len(at_or_above)),
        "recorded_failures_at_or_above": int(at_or_above[cfg.TARGET].sum()),
        "failure_share_at_or_above": (
            float(at_or_above[cfg.TARGET].mean()) if len(at_or_above) else None
        ),
    }


# --------------------------------------------------------------------------- #
# Single machine
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Assessment:
    reading: dict[str, Any]
    validation_warnings: list[str]
    score: float
    threshold: float
    recommendation: Recommendation
    sensitivity: pd.DataFrame
    context: dict[str, Any] | None
    model_label: str

    def to_dict(self) -> dict[str, Any]:
        rec = self.recommendation
        return {
            "kind": "screening_result_not_a_confirmed_fault",
            "inputs": self.reading,
            "model": self.model_label,
            "failure_score": self.score,
            "operating_threshold": self.threshold,
            "flagged_by_model": rec.flagged,
            "risk_level": rec.level.label,
            "model_level": rec.model_level.label,
            "indicator_level": rec.indicator_level.label,
            "headline": rec.headline,
            "actions": rec.actions,
            "reasoning": rec.reasoning,
            "indicators": [i.to_dict() for i in rec.indicators],
            "input_warnings": self.validation_warnings,
            "holdout_context": self.context,
            "disclaimer": DISCLAIMER,
        }


def assess_reading(
    model: LoadedModel,
    reading: Mapping[str, Any],
    holdout: pd.DataFrame | None = None,
) -> Assessment:
    """Validate, score and explain one reading. Raises ``InputValidationError`` if invalid."""
    validated = validate_reading(reading, model.training_ranges)
    row = pd.DataFrame([validated.reading])
    score = float(score_frame(model, row)[0])
    return Assessment(
        reading=validated.reading,
        validation_warnings=validated.warnings,
        score=score,
        threshold=model.threshold,
        recommendation=recommend(score, model.threshold, validated.reading),
        sensitivity=local_sensitivity(model, validated.reading, score),
        context=score_context(score, holdout, model.model_key),
        model_label=model.label,
    )


# --------------------------------------------------------------------------- #
# Batch
# --------------------------------------------------------------------------- #
@dataclass
class BatchAssessment:
    validation: BatchValidation
    results: pd.DataFrame  # valid rows, highest priority first
    counts: dict[str, int]  # rows per risk level


def assess_batch(model: LoadedModel, table: pd.DataFrame) -> BatchAssessment:
    """Validate and score an uploaded table. Row problems are reported, not fatal."""
    validation = validate_batch(table, model.training_ranges)
    valid = validation.valid
    if valid.empty:
        return BatchAssessment(validation, valid.copy(), {"High": 0, "Elevated": 0, "Low": 0})
    scores = score_frame(model, valid)
    recs = batch_recommendations(valid, scores, model.threshold)
    results = pd.concat([valid, recs], axis=1).sort_values(
        ["risk_rank", "failure_score"], ascending=[False, False], kind="stable"
    )
    counts = {level: int((results["risk_level"] == level).sum()) for level in ("High", "Elevated", "Low")}
    return BatchAssessment(validation, results.reset_index(drop=True), counts)
