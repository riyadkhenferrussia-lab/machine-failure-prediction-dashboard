"""Model definitions, hyper-parameter search and operating-point selection.

Everything that learns from data (scaling, one-hot encoding, the classifier and the
decision threshold) is fitted on training data only; the hold-out set is touched once,
for the final evaluation (see ``src/train.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as cfg
from .features import SensorFeatureEngineer


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    estimator: str  # "dummy" | "logistic_regression" | "random_forest"
    engineered: bool
    description: str


MODEL_SPECS: dict[str, ModelSpec] = {
    "baseline_majority": ModelSpec(
        "baseline_majority",
        "Baseline: majority class",
        "dummy",
        False,
        "Always predicts 'no failure' (no skill; shows why accuracy alone is misleading).",
    ),
    "logreg_raw": ModelSpec(
        "logreg_raw",
        "Logistic regression (raw features)",
        "logistic_regression",
        False,
        "Linear baseline on the five sensor readings and Type.",
    ),
    "rf_raw": ModelSpec(
        "rf_raw",
        "Random forest (raw features)",
        "random_forest",
        False,
        "Non-linear model on the raw readings only (ablation without engineered features).",
    ),
    "rf_engineered": ModelSpec(
        "rf_engineered",
        "Random forest (raw + engineered features)",
        "random_forest",
        True,
        "Random forest with temperature difference, mechanical power and wear x torque added.",
    ),
}

RF_SEARCH_SPACE: dict[str, list[Any]] = {
    "clf__n_estimators": [200, 300, 400],
    "clf__max_depth": [None, 8, 12, 16, 24],
    "clf__min_samples_leaf": [1, 2, 3, 5, 10],
    "clf__max_features": ["sqrt", "log2", 0.5],
    "clf__class_weight": ["balanced_subsample", "balanced", None],
}
LR_SEARCH_SPACE: dict[str, list[Any]] = {"clf__C": [0.01, 0.1, 1.0, 10.0, 100.0]}


def _spec(spec: ModelSpec | str) -> ModelSpec:
    return MODEL_SPECS[spec] if isinstance(spec, str) else spec


def build_pipeline(
    spec: ModelSpec | str,
    params: Mapping[str, Any] | None = None,
    random_state: int = cfg.RANDOM_STATE,
    n_jobs: int | None = 1,
) -> Pipeline:
    """Create an unfitted ``features -> preprocessing -> classifier`` pipeline.

    ``params`` use plain estimator names (``n_estimators``, ``C`` ...), not the
    ``clf__`` prefix used inside the search.
    """
    spec = _spec(spec)
    params = dict(params or {})

    numeric_step: Any = StandardScaler() if spec.estimator == "logistic_regression" else "passthrough"
    prep = ColumnTransformer(
        [
            ("num", numeric_step, make_column_selector(dtype_include=np.number)),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                [cfg.CATEGORICAL_FEATURE],
            ),
        ],
        verbose_feature_names_out=False,
    )

    if spec.estimator == "dummy":
        classifier: Any = DummyClassifier(strategy="prior")
    elif spec.estimator == "logistic_regression":
        classifier = LogisticRegression(
            class_weight="balanced",
            max_iter=5000,
            C=float(params.get("C", cfg.DEFAULT_LR_PARAMS["C"])),
            random_state=random_state,
        )
    elif spec.estimator == "random_forest":
        rf_params = {**cfg.DEFAULT_RF_PARAMS, **params}
        classifier = RandomForestClassifier(random_state=random_state, n_jobs=n_jobs, **rf_params)
    else:  # pragma: no cover
        raise ValueError(f"Unknown estimator {spec.estimator!r}")

    return Pipeline(
        [
            ("features", SensorFeatureEngineer(add_engineered=spec.engineered)),
            ("prep", prep),
            ("clf", classifier),
        ]
    )


def strip_prefix(params: Mapping[str, Any]) -> dict[str, Any]:
    """``{'clf__C': 1}`` -> ``{'C': 1}``."""
    return {key.split("__", 1)[1] if "__" in key else key: value for key, value in params.items()}


@dataclass
class TuneResult:
    estimator: Pipeline  # refitted on all the data passed to ``tune``
    best_params: dict[str, Any]
    cv_pr_auc_mean: float
    cv_pr_auc_std: float
    n_candidates: int


def cv_splitter(n_splits: int = cfg.CV_FOLDS, random_state: int = cfg.RANDOM_STATE) -> StratifiedKFold:
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def tune(
    spec: ModelSpec | str,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    search: bool = True,
    n_iter: int = cfg.SEARCH_ITERATIONS,
    cv_folds: int = cfg.CV_FOLDS,
    random_state: int = cfg.RANDOM_STATE,
    n_jobs: int | None = -1,
) -> TuneResult:
    """Select hyper-parameters by stratified CV on ``X, y`` (training data only).

    The criterion is average precision (area under the precision-recall curve), which
    is robust to the 3.4 % failure rate. With ``search=False`` the configured default
    parameters are used and only the CV score is computed.
    """
    spec = _spec(spec)
    cv = cv_splitter(cv_folds, random_state)
    pipeline = build_pipeline(spec, random_state=random_state, n_jobs=1)

    if search and spec.estimator == "random_forest":
        searcher: Any = RandomizedSearchCV(
            pipeline,
            RF_SEARCH_SPACE,
            n_iter=n_iter,
            scoring="average_precision",
            cv=cv,
            n_jobs=n_jobs,
            random_state=random_state,
            refit=True,
        )
    elif search and spec.estimator == "logistic_regression":
        searcher = GridSearchCV(
            pipeline, LR_SEARCH_SPACE, scoring="average_precision", cv=cv, n_jobs=n_jobs, refit=True
        )
    else:
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring="average_precision", n_jobs=n_jobs)
        fitted = clone(pipeline).fit(X, y)
        defaults = (
            dict(cfg.DEFAULT_RF_PARAMS)
            if spec.estimator == "random_forest"
            else dict(cfg.DEFAULT_LR_PARAMS)
            if spec.estimator == "logistic_regression"
            else {}
        )
        return TuneResult(fitted, defaults, float(scores.mean()), float(scores.std()), 1)

    searcher.fit(X, y)
    best = searcher.best_index_
    return TuneResult(
        estimator=searcher.best_estimator_,
        best_params=strip_prefix(searcher.best_params_),
        cv_pr_auc_mean=float(searcher.cv_results_["mean_test_score"][best]),
        cv_pr_auc_std=float(searcher.cv_results_["std_test_score"][best]),
        n_candidates=len(searcher.cv_results_["params"]),
    )


def out_of_fold_scores(
    estimator: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    cv_folds: int = cfg.CV_FOLDS,
    random_state: int = cfg.RANDOM_STATE,
    n_jobs: int | None = 1,
) -> np.ndarray:
    """Out-of-fold failure scores: every row is scored by a model that never saw it."""
    proba = cross_val_predict(
        clone(estimator),
        X,
        y,
        cv=cv_splitter(cv_folds, random_state),
        method="predict_proba",
        n_jobs=n_jobs,
    )
    return proba[:, 1]


# --------------------------------------------------------------------------- #
# Operating point
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ThresholdChoice:
    threshold: float
    beta: float
    precision: float
    recall: float
    f_beta: float
    n_positive: int
    n_samples: int


def select_threshold(
    y_true: np.ndarray | pd.Series,
    scores: np.ndarray | pd.Series,
    beta: float = cfg.THRESHOLD_BETA,
) -> ThresholdChoice:
    """Pick the decision threshold that maximises F-beta on the given scores.

    Call this with *out-of-fold training* scores only. A row is flagged when
    ``score >= threshold``. The returned threshold is the midpoint between the best
    observed score and the next lower distinct score, so it is not a knife-edge value
    (the flagged set on the scores passed in is identical).
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    if y.size == 0 or y.sum() == 0 or y.sum() == y.size:
        raise ValueError("select_threshold needs both classes to be present")
    precision, recall, thresholds = precision_recall_curve(y, s)
    p, r = precision[:-1], recall[:-1]
    denominator = beta**2 * p + r
    f_beta = np.divide(
        (1.0 + beta**2) * p * r, denominator, out=np.zeros_like(denominator), where=denominator > 0
    )
    best = int(np.argmax(f_beta))
    threshold = float(thresholds[best])
    if best > 0:
        threshold = (threshold + float(thresholds[best - 1])) / 2.0
    return ThresholdChoice(
        threshold=threshold,
        beta=beta,
        precision=float(p[best]),
        recall=float(r[best]),
        f_beta=float(f_beta[best]),
        n_positive=int(y.sum()),
        n_samples=int(y.size),
    )
