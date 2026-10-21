"""Evaluation report: figures and ``MODEL_CARD.md`` built from the saved training outputs.

    python -m src.evaluate

Nothing is re-fitted and no number is typed in by hand: every figure and every value in
the model card is computed from ``reports/metrics.json``, ``reports/test_predictions.csv``,
``reports/feature_importance.csv`` and the saved model.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import config as cfg
from .data import features_and_target, load_dataset, split_dataset
from .modeling import build_pipeline, out_of_fold_scores
from .plots import (
    MODEL_SHORT,
    plot_calibration,
    plot_confusion_matrix,
    plot_cumulative_gains,
    plot_importance,
    plot_per_mode_recall,
    plot_risk_surfaces,
    plot_roc_pr,
    plot_threshold_tradeoff,
    save_figure,
)
from .predict import LoadedModel, load_model
from .recommend import DEFAULT_RULES, WatchRules, level_codes
from .utils import read_json

FIGURE_FILES = {
    "confusion_matrix": "eval_confusion_matrix.png",
    "roc_pr": "eval_roc_pr.png",
    "threshold_tradeoff": "eval_threshold_tradeoff.png",
    "feature_importance": "eval_feature_importance.png",
    "risk_surfaces": "eval_risk_surfaces.png",
    "per_mode_recall": "eval_per_mode_recall.png",
    "cumulative_gains": "eval_cumulative_gains.png",
    "calibration": "eval_calibration.png",
}


@dataclass(frozen=True)
class EvaluationPaths:
    model: Path = cfg.MODEL_PATH
    metrics: Path = cfg.METRICS_PATH
    predictions: Path = cfg.PREDICTIONS_PATH
    importance: Path = cfg.IMPORTANCE_PATH
    model_card_json: Path = cfg.MODEL_CARD_PATH
    eda_summary: Path = cfg.EDA_SUMMARY_PATH
    figures: Path = cfg.FIGURES_DIR
    model_card_md: Path = cfg.MODEL_CARD_MD_PATH
    levels: Path = cfg.RECOMMENDATION_LEVELS_PATH
    design: Path = cfg.RECOMMENDATION_DESIGN_PATH
    data: Path | None = None


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def build_figures(
    model: LoadedModel,
    metrics: dict[str, Any],
    predictions: pd.DataFrame,
    importance: pd.DataFrame,
    data: pd.DataFrame,
) -> dict[str, Any]:
    """Every evaluation figure for the final model, keyed by ``FIGURE_FILES``."""
    key = metrics["final"]["key"]
    threshold = float(metrics["final"]["threshold"])
    y = predictions[cfg.TARGET].to_numpy()
    scores = {k: predictions[f"score_{k}"].to_numpy() for k in ("logreg_raw", "rf_raw", "rf_engineered")}
    thresholds = {k: float(metrics["models"][k]["threshold"]) for k in scores}
    per_mode = pd.DataFrame(metrics["final"]["per_mode_recall"])
    final_scores = predictions[f"score_{key}"].to_numpy()
    return {
        "confusion_matrix": plot_confusion_matrix(metrics["final"]["test"], threshold),
        "roc_pr": plot_roc_pr(y, scores, thresholds, key),
        "threshold_tradeoff": plot_threshold_tradeoff(y, final_scores, threshold, beta=cfg.THRESHOLD_BETA),
        "feature_importance": plot_importance(importance),
        "risk_surfaces": plot_risk_surfaces(model, data),
        "per_mode_recall": plot_per_mode_recall(per_mode),
        "cumulative_gains": plot_cumulative_gains(y, final_scores),
        "calibration": plot_calibration(y, final_scores),
    }


# --------------------------------------------------------------------------- #
# Recommendation levels
# --------------------------------------------------------------------------- #
RULE_VARIANTS: dict[str, WatchRules] = {
    "Wide margins: 10% everywhere, tool-wear watch from 180 min": WatchRules(heat_margin=0.10, tool_wear_watch_min=180.0),
    "Tool-wear watch from the 200 min window start": WatchRules(heat_margin=0.10),
    "Heat-dissipation margin 5%": DEFAULT_RULES,
    "No heat-dissipation watch": WatchRules(heat_margin=0.0),
}


def level_table(final_codes: np.ndarray, y_true: np.ndarray) -> pd.DataFrame:
    """Machines, recorded failures and failure rate per final risk level (High, Elevated, Low)."""
    y = np.asarray(y_true).astype(int)
    total = int(y.sum())
    rows = []
    for code, label in ((2, "High"), (1, "Elevated"), (0, "Low")):
        mask = np.asarray(final_codes) == code
        n, failures = int(mask.sum()), int(y[mask].sum())
        rows.append(
            {
                "risk_level": label,
                "machines": n,
                "share_of_machines": n / len(y),
                "recorded_failures": failures,
                "failure_rate": failures / n if n else float("nan"),
                "share_of_all_failures": failures / total if total else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def holdout_level_table(predictions: pd.DataFrame, metrics: dict[str, Any]) -> pd.DataFrame:
    """Risk levels of the dashboard rules on the hold-out set, using the stored final-model scores."""
    scores = predictions[f"score_{metrics['final']['key']}"].to_numpy()
    final, _, _ = level_codes(predictions, scores, float(metrics["final"]["threshold"]))
    return level_table(final, predictions[cfg.TARGET].to_numpy())


def design_table(card: dict[str, Any], metrics: dict[str, Any], data: pd.DataFrame) -> pd.DataFrame:
    """Watch-rule variants compared on the TRAINING split with out-of-fold model scores.

    This is how the watch margins were chosen: the hold-out set plays no part. Every
    training row is scored by a model that never saw it, so the comparison is not
    flattered by in-sample scores.
    """
    meta = metrics["meta"]
    train, _ = split_dataset(data, test_size=meta["test_size"], seed=meta["seed"])
    X, y = features_and_target(train)
    pipeline = build_pipeline(metrics["final"]["key"], card["hyperparameters"], random_state=meta["seed"], n_jobs=-1)
    oof = out_of_fold_scores(pipeline, X, y, cv_folds=meta["cv_folds"], random_state=meta["seed"])
    threshold = float(metrics["final"]["threshold"])
    frames = []
    for name, rules in RULE_VARIANTS.items():
        final, _, _ = level_codes(train, oof, threshold, rules)
        table = level_table(final, y.to_numpy())
        table.insert(0, "rules", name)
        table.insert(1, "is_default", rules == DEFAULT_RULES)
        frames.append(table)
    result = pd.concat(frames, ignore_index=True)
    result.insert(0, "data", "training split, out-of-fold scores")
    return result


# --------------------------------------------------------------------------- #
# Model card
# --------------------------------------------------------------------------- #
def _pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def _f3(value: float) -> str:
    return f"{value:.3f}"


def _interval(interval: list[float] | tuple[float, float]) -> str:
    return f"[{interval[0]:.3f}, {interval[1]:.3f}]"


def render_model_card(
    metrics: dict[str, Any],
    card: dict[str, Any],
    importance: pd.DataFrame,
    eda_summary: dict[str, Any] | None = None,
    levels: pd.DataFrame | None = None,
    design: pd.DataFrame | None = None,
) -> str:
    """Markdown model card. Values come from the saved artefacts; prose states design facts."""
    meta = metrics["meta"]
    final = metrics["final"]
    test = final["test"]
    ci = final["test_ci95"]
    env = meta["environment"]
    models = metrics["models"]
    params = ", ".join(f"`{k}={v}`" for k, v in sorted(card["hyperparameters"].items()))

    comparison_rows = []
    for key in ("baseline_majority", "logreg_raw", "rf_raw", "rf_engineered"):
        m = models[key]
        t = m["test"]
        cv = f"{m['cv_pr_auc_mean']:.3f} ± {m['cv_pr_auc_std']:.3f}" if key != "baseline_majority" else "n/a"
        comparison_rows.append(
            f"| {MODEL_SHORT[key]}{' (final)' if key == final['key'] else ''} | {cv} | {m['threshold']:.3f} | "
            f"{_f3(t['precision'])} | {_f3(t['recall'])} | {_f3(t['f1'])} | {_f3(t['pr_auc'])} | {_f3(t['roc_auc'])} | "
            f"{t['tp']} / {t['fp']} / {t['fn']} / {t['tn']} |"
        )

    mode_rows = [
        f"| {r['label']} | {r['n_failures']} | {r['detected']} | {_pct(r['recall'], 0) if r['n_failures'] else 'n/a'} |"
        for r in final["per_mode_recall"]
    ]
    top_rows = [
        f"| top {_pct(float(fraction), 0)} of machines | {_pct(share, 1)} |"
        for fraction, share in final["recall_at_top_fraction"].items()
    ]

    permutation = importance[importance["kind"] == "permutation_raw_inputs_holdout"].sort_values(
        "importance_mean", ascending=False
    )
    permutation_rows = [
        f"| {cfg.FEATURE_LABELS.get(r.feature, r.feature)} | {r.importance_mean:.3f} ± {r.importance_std:.3f} |"
        for r in permutation.itertuples()
    ]

    level_rows = []
    if levels is not None:
        level_rows = [
            f"| {r.risk_level} | {r.machines:,} | {_pct(r.share_of_machines)} | {r.recorded_failures} | "
            f"{_pct(r.failure_rate, 2) if r.machines else 'n/a'} | {_pct(r.share_of_all_failures, 0)} |"
            for r in levels.itertuples()
        ]
    design_rows = []
    if design is not None:
        for name, group in design.groupby("rules", sort=False):
            elevated = group[group["risk_level"] == "Elevated"].iloc[0]
            low = group[group["risk_level"] == "Low"].iloc[0]
            used = " (used)" if bool(group["is_default"].iloc[0]) else ""
            design_rows.append(
                f"| {name}{used} | {_pct(elevated.share_of_machines)} | {_pct(elevated.failure_rate, 1)} | "
                f"{low.recorded_failures} of {int(group['recorded_failures'].sum())} |"
            )

    quirks = ""
    if eda_summary and eda_summary.get("label_inconsistencies"):
        q = eda_summary["label_inconsistencies"]
        quirks = (
            f"The published labels contain {q['failure_without_mode_flag']} records with `Machine failure = 1` but no "
            f"failure-mode flag, and {q['mode_flag_without_failure']} records with a mode flag (all random-failure "
            "flags) but `Machine failure = 0`. They are kept exactly as published and reported as validation warnings."
        )

    lines = [
        "# Model card: machine failure screening model",
        "",
        "> Generated by `python -m src.evaluate` from `reports/metrics.json` and `models/model_card.json`. "
        "Re-run the command instead of editing this file.",
        "",
        "## Summary",
        "",
        f"- **Model:** {final['label']}, a scikit-learn `Pipeline` (feature engineering, preprocessing, classifier).",
        f"- **Task:** binary classification of the `Machine failure` label of the AI4I 2020 Predictive Maintenance "
        f"dataset ({meta['n_rows']:,} records, {meta['failures_train'] + meta['failures_test']} failures).",
        f"- **Output:** a failure score in [0, 1]; a record is flagged when score ≥ **{final['threshold']:.3f}** "
        "(the operating threshold). The score is the forest's averaged vote, not a calibrated probability.",
        f"- **Trained:** {meta['created_utc']} with Python {env['python']}, scikit-learn {env['scikit_learn']}, "
        f"pandas {env['pandas']}, numpy {env['numpy']}; seed {meta['seed']}.",
        f"- **Data file SHA-256:** `{meta['data_sha256']}`.",
        "",
        "## Intended use",
        "",
        "Portfolio demonstration of a predictive-maintenance workflow: ranking machine records by failure score so that "
        "inspections can be prioritised, with transparent engineering indicators next to the score. "
        "A flagged record is a **prediction to be checked by a person, never a confirmed fault**, and the model does "
        "not estimate remaining useful life or the cause of a failure.",
        "",
        "## Out of scope",
        "",
        "- Real-time monitoring, control or any safety-related decision (the project has no live data connection).",
        "- Use on real equipment without re-training and validation on that equipment's own data, units and product mix.",
        "- Treating the score as a calibrated probability (see the reliability figure).",
        "",
        "## Data",
        "",
        "AI4I 2020 Predictive Maintenance Dataset (S. Matzka, UCI Machine Learning Repository, "
        f"DOI {cfg.DATASET_DOI}, CC BY 4.0). The dataset is **synthetic**: the authors generated it to resemble real "
        "milling-machine maintenance data, and its failure labels follow documented rules plus random components. "
        f"Stratified split: {meta['n_train']:,} training records ({meta['failures_train']} failures) and "
        f"{meta['n_test']:,} hold-out records ({meta['failures_test']} failures). {quirks}",
        "",
        "## Inputs and leakage controls",
        "",
        f"- Model inputs: {', '.join(f'`{c}`' for c in card['inputs'])}.",
        f"- Features derived inside the pipeline from those inputs: {', '.join(f'`{c}`' for c in card['engineered_features'])} "
        "(temperature difference, mechanical power, wear × torque).",
        f"- Never given to the model: {', '.join(f'`{c}`' for c in card['excluded_columns_leakage_guard'])}. "
        "Identifiers and failure-mode flags define or identify the label, so they are excluded by an explicit guard "
        "that is covered by automated tests. Mode flags are used only for the after-the-fact per-mode analysis below.",
        "- Scaling, one-hot encoding, the classifier, hyper-parameters and the decision threshold are all fitted on "
        "training data only. The hold-out set is evaluated once, after every choice has been made.",
        "",
        "## Training procedure",
        "",
        f"- Candidates: majority-class baseline, logistic regression (raw inputs), random forest (raw inputs), random "
        f"forest (raw + engineered features).",
        f"- Hyper-parameters: {'randomised search with ' + str(meta['search_iterations']) + ' candidates' if meta['search'] else 'fixed defaults'} "
        f"scored by {meta['cv_folds']}-fold stratified cross-validated PR-AUC on the training set.",
        f"- Final model: {selection_text(metrics)} Chosen hyper-parameters: {params}.",
        f"- Operating threshold {final['threshold']:.3f}: {card['threshold_source']}. With beta = {cfg.THRESHOLD_BETA:g}, "
        "recall counts four times as much as precision, which assumes that a missed failure costs more than an "
        "unnecessary inspection. That cost assumption is a design choice, not a measured fact.",
        "",
        "## Hold-out results",
        "",
        f"All values are on the {meta['n_test']:,} hold-out records ({meta['failures_test']} failures). "
        "Precision, recall and F1 refer to the failure class. Counts are TP / FP / FN / TN.",
        "",
        "| Model | CV PR-AUC (train) | Threshold | Precision | Recall | F1 | PR-AUC | ROC-AUC | Counts |",
        "|---|---|---|---|---|---|---|---|---|",
        *comparison_rows,
        "",
        f"Final model, 95% bootstrap intervals (stratified, {final['bootstrap_replicates']} replicates): "
        f"precision {_interval(ci['precision'])}, recall {_interval(ci['recall'])}, F1 {_interval(ci['f1'])}, "
        f"PR-AUC {_interval(ci['pr_auc'])}, ROC-AUC {_interval(ci['roc_auc'])}. "
        f"Other hold-out values: accuracy {_f3(test['accuracy'])}, balanced accuracy {_f3(test['balanced_accuracy'])}, "
        f"specificity {_f3(test['specificity'])}, MCC {_f3(test['mcc'])}, F2 {_f3(test['f2'])}, "
        f"Brier score {test['brier']:.4f}, alert rate {_pct(test['alert_rate'], 1)} of machines.",
        "",
        f"Accuracy is not informative here: always predicting \"no failure\" already scores "
        f"{_pct(models['baseline_majority']['test']['accuracy'], 1)}.",
        "",
        "### Recall by recorded failure mode (final model, hold-out)",
        "",
        "| Failure mode | Failures | Detected | Recall |",
        "|---|---|---|---|",
        *mode_rows,
        "",
        "A failure can carry several modes. Tool-wear (TWF) and random (RNF) failures are assigned partly at random "
        "in the dataset, so the sensor readings can only hint at them; the model is expected to find the "
        "rule-based modes (power, heat dissipation, overstrain) and to miss most of the random ones.",
        "",
        "### Inspection budget (final model, hold-out)",
        "",
        "Share of all hold-out failures found when only the highest-scoring machines are inspected.",
        "",
        "| Inspected | Failures found |",
        "|---|---|",
        *top_rows,
        "",
        *(
            [
                "### Recommendation levels (model score + engineering indicators, hold-out)",
                "",
                "Final level = the higher of the model level and the indicator level "
                "(`src/recommend.py`; the dashboard applies exactly these rules). "
                "Failure rate = share of the machines at that level with a recorded failure.",
                "",
                "| Level | Machines | Share of machines | Recorded failures | Failure rate | Share of all failures |",
                "|---|---|---|---|---|---|",
                *level_rows,
                "",
            ]
            if level_rows
            else []
        ),
        *(
            [
                "How the watch margins were chosen: four rule variants compared on the **training** split with "
                "out-of-fold model scores (the hold-out set played no part). Wider margins put a large share of "
                "machines on the watch list for little gain in failure rate.",
                "",
                "| Rule variant | Elevated: share of machines | Elevated: failure rate | Failures left in Low |",
                "|---|---|---|---|",
                *design_rows,
                "",
            ]
            if design_rows
            else []
        ),
        "### Input importance",
        "",
        "Drop in hold-out PR-AUC when one raw input is shuffled (10 repeats, mean ± standard deviation).",
        "",
        "| Input | PR-AUC drop |",
        "|---|---|",
        *permutation_rows,
        "",
        "## Interpretation and limitations",
        "",
        "- **Synthetic data.** The labels follow documented rules (power outside 3,500–9,000 W, small temperature "
        "difference at low speed, wear × torque above a limit per product variant) plus random components. The "
        "model largely re-learns those rules; the results say how well a pipeline can recover them, not how well "
        "it would predict failures on real equipment.",
        f"- **Small number of failures.** The hold-out set contains {meta['failures_test']} failures, so every "
        "metric has a wide interval (see above). Differences between the random-forest variants and the "
        "operating point should be read with that uncertainty in mind.",
        "- **Threshold depends on assumed costs.** A different cost ratio between missed failures and unnecessary "
        "inspections would change the best threshold (see the threshold trade-off figure).",
        "- **Score is not a probability.** See the reliability figure; do not read 0.35 as a 35% chance.",
        "- **Extrapolation.** Inputs outside the training range "
        f"({', '.join(f'{cfg.SHORT_LABELS[k]} {v[0]:g} to {v[1]:g}' for k, v in card['training_ranges'].items())}) "
        "get a warning in the dashboard; the model has not seen such conditions.",
        "- **No temporal structure.** Records are independent snapshots; there is no sensor history, so trends and "
        "degradation over time are not modelled.",
        "- **Not a confirmed fault.** A flag or a high score means \"inspect\", not \"this machine is broken\".",
        "",
        "## Reproducing this report",
        "",
        "```bash",
        "python -m src.train      # trains, evaluates, writes models/ and reports/",
        "python -m src.evaluate   # regenerates the figures and this file",
        "```",
        "",
    ]
    return "\n".join(lines)


def selection_text(metrics: dict[str, Any]) -> str:
    final = metrics["final"]
    cv = metrics["models"][final["key"]]
    return (
        f"{final['label']}, selected for the highest cross-validated PR-AUC on the training set "
        f"({cv['cv_pr_auc_mean']:.3f} ± {cv['cv_pr_auc_std']:.3f}); hold-out results played no part in the selection."
    )


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_evaluation(paths: EvaluationPaths | None = None) -> dict[str, Path]:
    paths = paths or EvaluationPaths()
    for required in (paths.metrics, paths.predictions, paths.importance, paths.model, paths.model_card_json):
        if not Path(required).exists():
            raise FileNotFoundError(f"{required} not found. Run `python -m src.train` first.")
    metrics = read_json(paths.metrics)
    card = read_json(paths.model_card_json)
    predictions = pd.read_csv(paths.predictions)
    importance = pd.read_csv(paths.importance)
    eda = read_json(paths.eda_summary) if Path(paths.eda_summary).exists() else None
    model = load_model(paths.model)
    data = load_dataset(paths.data)

    written: dict[str, Path] = {}
    for name, figure in build_figures(model, metrics, predictions, importance, data).items():
        written[name] = save_figure(figure, Path(paths.figures) / FIGURE_FILES[name])

    levels = holdout_level_table(predictions, metrics)
    design = design_table(card, metrics, data)
    for name, table, target in (("recommendation_levels", levels, paths.levels), ("recommendation_design", design, paths.design)):
        target.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(target, index=False)
        written[name] = target

    paths.model_card_md.parent.mkdir(parents=True, exist_ok=True)
    paths.model_card_md.write_text(render_model_card(metrics, card, importance, eda, levels, design), encoding="utf-8")
    written["model_card"] = paths.model_card_md
    return written


def main() -> int:
    written = run_evaluation()
    for name, path in written.items():
        print(f"{name:20s} {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
