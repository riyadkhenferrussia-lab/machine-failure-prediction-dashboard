"""Training pipeline.

Order of operations (no step looks at data it is not allowed to see):

1. load + validate the dataset, write the validation report;
2. stratified train / hold-out split (the hold-out set is used once, at the end);
3. for every candidate model: hyper-parameter search by CV on the **training** set,
   out-of-fold training scores, decision threshold chosen on those scores;
4. the final model is picked by **cross-validated** PR-AUC (never by hold-out results);
5. one evaluation of every model on the hold-out set; bootstrap intervals, per-failure-mode
   recall and inspection-budget analysis for the final model;
6. permutation importance on the hold-out set; artifacts are written to disk.

Command line::

    python -m src.train               # full run with hyper-parameter search
    python -m src.train --no-search   # fast run with the configured default parameters
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.inspection import permutation_importance

from . import __version__
from . import config as cfg
from .data import features_and_target, load_and_validate, split_dataset
from .metrics import (
    bootstrap_ci,
    classification_metrics,
    per_mode_recall,
    recall_at_fraction,
)
from .modeling import (
    MODEL_SPECS,
    ModelSpec,
    build_pipeline,
    out_of_fold_scores,
    select_threshold,
    tune,
)
from .utils import sha256_of_file, write_json
from .validation import ranges_from_dataframe

BASELINE_THRESHOLD = 0.5
TOP_FRACTIONS = (0.01, 0.02, 0.05, 0.10, 0.20)
SELECTION_RULE = (
    "Final model = highest mean cross-validated PR-AUC on the training set among the "
    "non-baseline candidates. Hold-out results are never used for selection."
)


@dataclass(frozen=True)
class ArtifactPaths:
    """Where training writes its outputs (tests redirect these to a temp folder)."""

    model: Path = cfg.MODEL_PATH
    model_card: Path = cfg.MODEL_CARD_PATH
    metrics: Path = cfg.METRICS_PATH
    comparison: Path = cfg.MODEL_COMPARISON_PATH
    predictions: Path = cfg.PREDICTIONS_PATH
    importance: Path = cfg.IMPORTANCE_PATH
    data_validation: Path = cfg.DATA_VALIDATION_PATH

    @classmethod
    def under(cls, root: Path | str) -> "ArtifactPaths":
        root = Path(root)
        return cls(
            model=root / "models" / "failure_model.joblib",
            model_card=root / "models" / "model_card.json",
            metrics=root / "reports" / "metrics.json",
            comparison=root / "reports" / "model_comparison.csv",
            predictions=root / "reports" / "test_predictions.csv",
            importance=root / "reports" / "feature_importance.csv",
            data_validation=root / "reports" / "data_validation.json",
        )


@dataclass
class TrainingOptions:
    search: bool = True
    n_iter: int = cfg.SEARCH_ITERATIONS
    cv_folds: int = cfg.CV_FOLDS
    seed: int = cfg.RANDOM_STATE
    test_size: float = cfg.TEST_SIZE
    n_jobs: int = -1
    bootstrap: int = 1000
    data_path: Path | None = None


@dataclass
class ModelResult:
    spec: ModelSpec
    estimator: Any
    best_params: dict[str, Any]
    cv_pr_auc_mean: float
    cv_pr_auc_std: float
    n_candidates: int
    threshold: float
    threshold_source: str
    oof_metrics: dict[str, float] | None
    test_scores: np.ndarray
    test_metrics: dict[str, float]
    test_metrics_at_half: dict[str, float]
    extras: dict[str, Any] = field(default_factory=dict)


def fit_and_evaluate(
    spec: ModelSpec,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    options: TrainingOptions,
    log: Callable[[str], None] = lambda message: None,
) -> ModelResult:
    """Tune on the training data, choose the threshold out-of-fold, evaluate on hold-out."""
    started = time.time()
    tuned = tune(
        spec,
        X_train,
        y_train,
        search=options.search,
        n_iter=options.n_iter,
        cv_folds=options.cv_folds,
        random_state=options.seed,
        n_jobs=options.n_jobs,
    )
    estimator = tuned.estimator

    if spec.estimator == "dummy":
        threshold, source, oof_metrics = BASELINE_THRESHOLD, "fixed 0.5 (majority-class baseline)", None
    else:
        oof = out_of_fold_scores(
            estimator,
            X_train,
            y_train,
            cv_folds=options.cv_folds,
            random_state=options.seed + 1,
            n_jobs=options.n_jobs,
        )
        choice = select_threshold(y_train, oof, cfg.THRESHOLD_BETA)
        threshold = choice.threshold
        source = (
            f"maximises F{cfg.THRESHOLD_BETA:g} on out-of-fold training scores "
            f"({choice.n_samples} rows, {choice.n_positive} failures)"
        )
        oof_metrics = classification_metrics(y_train, oof, threshold)

    test_scores = estimator.predict_proba(X_test)[:, 1]
    result = ModelResult(
        spec=spec,
        estimator=estimator,
        best_params=tuned.best_params,
        cv_pr_auc_mean=tuned.cv_pr_auc_mean,
        cv_pr_auc_std=tuned.cv_pr_auc_std,
        n_candidates=tuned.n_candidates,
        threshold=float(threshold),
        threshold_source=source,
        oof_metrics=oof_metrics,
        test_scores=test_scores,
        test_metrics=classification_metrics(y_test, test_scores, threshold),
        test_metrics_at_half=classification_metrics(y_test, test_scores, 0.5),
    )
    log(
        f"  {spec.label:<45s} CV PR-AUC {tuned.cv_pr_auc_mean:.3f} +/- {tuned.cv_pr_auc_std:.3f} | "
        f"hold-out recall {result.test_metrics['recall']:.3f}, precision "
        f"{result.test_metrics['precision']:.3f} ({time.time() - started:.0f}s)"
    )
    return result


def select_final_model(results: dict[str, ModelResult]) -> str:
    candidates = {k: r for k, r in results.items() if r.spec.estimator != "dummy"}
    return max(candidates, key=lambda key: candidates[key].cv_pr_auc_mean)


def importance_tables(
    final: ModelResult, X_test: pd.DataFrame, y_test: pd.Series, seed: int
) -> pd.DataFrame:
    """Permutation importance on raw inputs (hold-out) + impurity importance (if available)."""
    permutation = permutation_importance(
        final.estimator,
        X_test,
        y_test,
        scoring="average_precision",
        n_repeats=10,
        random_state=seed,
        n_jobs=1,
    )
    frames = [
        pd.DataFrame(
            {
                "kind": "permutation_raw_inputs_holdout",
                "feature": list(X_test.columns),
                "importance_mean": permutation.importances_mean,
                "importance_std": permutation.importances_std,
            }
        )
    ]
    classifier = final.estimator.named_steps["clf"]
    if hasattr(classifier, "feature_importances_"):
        names = final.estimator.named_steps["prep"].get_feature_names_out()
        frames.append(
            pd.DataFrame(
                {
                    "kind": "impurity_model_features_train",
                    "feature": list(names),
                    "importance_mean": classifier.feature_importances_,
                    "importance_std": np.std(
                        [tree.feature_importances_ for tree in classifier.estimators_], axis=0
                    ),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _comparison_frame(results: dict[str, ModelResult]) -> pd.DataFrame:
    rows = []
    for key, r in results.items():
        t = r.test_metrics
        rows.append(
            {
                "model_key": key,
                "model": r.spec.label,
                "features": "raw + engineered" if r.spec.engineered else "raw",
                "cv_pr_auc_mean": r.cv_pr_auc_mean,
                "cv_pr_auc_std": r.cv_pr_auc_std,
                "threshold": r.threshold,
                "test_pr_auc": t["pr_auc"],
                "test_roc_auc": t["roc_auc"],
                "test_precision": t["precision"],
                "test_recall": t["recall"],
                "test_f1": t["f1"],
                "test_f2": t["f2"],
                "test_mcc": t["mcc"],
                "test_balanced_accuracy": t["balanced_accuracy"],
                "test_accuracy": t["accuracy"],
                "tp": t["tp"],
                "fp": t["fp"],
                "fn": t["fn"],
                "tn": t["tn"],
            }
        )
    return pd.DataFrame(rows)


def _environment() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "scikit_learn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "joblib": joblib.__version__,
        "project_version": __version__,
    }


def run_training(
    options: TrainingOptions | None = None,
    paths: ArtifactPaths | None = None,
    log: Callable[[str], None] = lambda message: None,
) -> dict[str, Any]:
    """Run the whole pipeline and write all artifacts. Returns the metrics payload."""
    options = options or TrainingOptions()
    paths = paths or ArtifactPaths()
    started = time.time()

    frame, report = load_and_validate(options.data_path)
    report.raise_if_errors()
    write_json(paths.data_validation, report.to_dict())
    data_path = Path(options.data_path) if options.data_path else cfg.RAW_DATA_PATH

    train_df, test_df = split_dataset(frame, options.test_size, options.seed)
    X_train, y_train = features_and_target(train_df)
    X_test, y_test = features_and_target(test_df)
    leaked = set(cfg.LEAKAGE_COLUMNS) & set(X_train.columns)
    if leaked:  # pragma: no cover - guarded by features_and_target
        raise RuntimeError(f"Leakage columns present in the model inputs: {sorted(leaked)}")
    log(
        f"Split: {len(train_df):,} train rows ({int(y_train.sum())} failures) / "
        f"{len(test_df):,} hold-out rows ({int(y_test.sum())} failures)"
    )

    log("Models (hyper-parameters tuned by CV on the training set only):")
    results = {
        key: fit_and_evaluate(spec, X_train, y_train, X_test, y_test, options, log)
        for key, spec in MODEL_SPECS.items()
    }
    final_key = select_final_model(results)
    final = results[final_key]
    log(f"Selected by cross-validated PR-AUC: {final.spec.label}")

    if "clf__n_jobs" in final.estimator.get_params():
        final.estimator.set_params(clf__n_jobs=1)  # single-row scoring is faster without threads
    model_inputs = list(final.estimator.named_steps["prep"].get_feature_names_out())
    if set(model_inputs) & set(cfg.LEAKAGE_COLUMNS):  # pragma: no cover
        raise RuntimeError("A leakage column reached the fitted model")

    importance = importance_tables(final, X_test, y_test, options.seed)
    ci = (
        bootstrap_ci(y_test, final.test_scores, final.threshold, n_boot=options.bootstrap, seed=options.seed)
        if options.bootstrap
        else {}
    )
    modes = per_mode_recall(test_df, final.test_scores, final.threshold)
    top_fraction = {
        f"{fraction:g}": recall_at_fraction(y_test, final.test_scores, fraction, seed=options.seed)
        for fraction in TOP_FRACTIONS
    }

    predictions = test_df[
        ["udi", "product_id"] + cfg.RAW_FEATURES + [cfg.TARGET] + cfg.FAILURE_MODE_COLUMNS
    ].copy()
    for key, r in results.items():
        predictions[f"score_{key}"] = r.test_scores
    paths.predictions.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(paths.predictions, index=False, float_format="%.6g")
    importance.to_csv(paths.importance, index=False, float_format="%.6g")

    comparison = _comparison_frame(results)
    comparison.to_csv(paths.comparison, index=False, float_format="%.6g")

    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    meta = {
        "created_utc": created,
        "seed": options.seed,
        "search": options.search,
        "search_iterations": options.n_iter if options.search else 0,
        "cv_folds": options.cv_folds,
        "test_size": options.test_size,
        "data_file": data_path.name,
        "data_sha256": sha256_of_file(data_path),
        "n_rows": int(len(frame)),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "failures_train": int(y_train.sum()),
        "failures_test": int(y_test.sum()),
        "environment": _environment(),
    }
    payload: dict[str, Any] = {
        "meta": meta,
        "selection": {
            "rule": SELECTION_RULE,
            "final_model": final_key,
            "final_label": final.spec.label,
            "threshold_rule": (
                f"Operating threshold maximises F{cfg.THRESHOLD_BETA:g} on out-of-fold training "
                "scores (recall weighted more than precision)."
            ),
        },
        "models": {
            key: {
                "label": r.spec.label,
                "description": r.spec.description,
                "features": "raw + engineered" if r.spec.engineered else "raw",
                "best_params": r.best_params,
                "n_search_candidates": r.n_candidates,
                "cv_pr_auc_mean": r.cv_pr_auc_mean,
                "cv_pr_auc_std": r.cv_pr_auc_std,
                "threshold": r.threshold,
                "threshold_source": r.threshold_source,
                "out_of_fold_train": r.oof_metrics,
                "test": r.test_metrics,
                "test_at_threshold_0_5": r.test_metrics_at_half,
            }
            for key, r in results.items()
        },
        "final": {
            "key": final_key,
            "label": final.spec.label,
            "threshold": final.threshold,
            "test": final.test_metrics,
            "test_ci95": {name: list(bounds) for name, bounds in ci.items()},
            "bootstrap_replicates": options.bootstrap,
            "per_mode_recall": modes.to_dict(orient="records"),
            "recall_at_top_fraction": top_fraction,
        },
    }
    write_json(paths.metrics, payload)

    bundle = {
        "format_version": cfg.MODEL_BUNDLE_FORMAT,
        "pipeline": final.estimator,
        "threshold": final.threshold,
        "model_key": final_key,
        "feature_columns": list(cfg.RAW_FEATURES),
        "model_input_names": model_inputs,
        "training_ranges": ranges_from_dataframe(train_df),
        "training_medians": {
            **{c: float(train_df[c].median()) for c in cfg.NUMERIC_SENSOR_FEATURES},
            cfg.CATEGORICAL_FEATURE: str(train_df[cfg.CATEGORICAL_FEATURE].mode().iloc[0]),
        },
        "metadata": meta,
    }
    paths.model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, paths.model, compress=3)

    write_json(
        paths.model_card,
        {
            "model": final.spec.label,
            "model_key": final_key,
            "hyperparameters": final.best_params,
            "threshold": final.threshold,
            "threshold_source": final.threshold_source,
            "score_interpretation": (
                "Model score in [0, 1]; a row is flagged when score >= threshold. The score is the "
                "ensemble's averaged vote, not a calibrated probability (see the reliability figure)."
            ),
            "inputs": list(cfg.RAW_FEATURES),
            "engineered_features": cfg.ENGINEERED_FEATURES if final.spec.engineered else [],
            "model_input_names": model_inputs,
            "excluded_columns_leakage_guard": cfg.LEAKAGE_COLUMNS,
            "training_ranges": bundle["training_ranges"],
            "training_medians": bundle["training_medians"],
            "holdout_metrics": final.test_metrics,
            "holdout_metrics_ci95": {name: list(bounds) for name, bounds in ci.items()},
            "metadata": meta,
        },
    )
    log(f"Artifacts written. Total time {time.time() - started:.0f}s")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.train", description=__doc__.split("\n\n")[0])
    parser.add_argument("--no-search", action="store_true", help="skip hyper-parameter search")
    parser.add_argument("--n-iter", type=int, default=cfg.SEARCH_ITERATIONS, help="random-search candidates")
    parser.add_argument("--seed", type=int, default=cfg.RANDOM_STATE)
    parser.add_argument("--bootstrap", type=int, default=1000, help="bootstrap replicates (0 = skip)")
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args(argv)
    options = TrainingOptions(
        search=not args.no_search,
        n_iter=args.n_iter,
        seed=args.seed,
        bootstrap=args.bootstrap,
        n_jobs=args.n_jobs,
    )
    payload = run_training(options, log=lambda message: print(message, flush=True))
    final = payload["final"]
    t = final["test"]
    print(
        f"\nFinal model: {final['label']}\n"
        f"  threshold {final['threshold']:.4f} | hold-out precision {t['precision']:.3f} "
        f"recall {t['recall']:.3f} F1 {t['f1']:.3f} PR-AUC {t['pr_auc']:.3f} ROC-AUC {t['roc_auc']:.3f}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
