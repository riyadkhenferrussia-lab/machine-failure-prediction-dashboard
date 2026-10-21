"""Model construction, threshold selection and evaluation metrics."""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src import config as cfg
from src import data as data_module
from src import metrics as M
from src import modeling


# --------------------------------------------------------------------------- #
# Model construction
# --------------------------------------------------------------------------- #
class TestPipelines:
    def test_all_candidates_share_the_same_structure(self):
        assert list(modeling.MODEL_SPECS) == ["baseline_majority", "logreg_raw", "rf_raw", "rf_engineered"]
        for key in modeling.MODEL_SPECS:
            pipe = modeling.build_pipeline(key)
            assert isinstance(pipe, Pipeline)
            assert [name for name, _ in pipe.steps] == ["features", "prep", "clf"]

    def test_unknown_candidate_is_an_error(self):
        with pytest.raises(KeyError):
            modeling.build_pipeline("gradient_boosting")

    def test_parameters_are_forwarded_and_defaults_come_from_config(self):
        default = modeling.build_pipeline("rf_engineered").named_steps["clf"]
        assert default.n_estimators == cfg.DEFAULT_RF_PARAMS["n_estimators"]
        custom = modeling.build_pipeline("rf_engineered", {"n_estimators": 7, "max_depth": 3}).named_steps["clf"]
        assert (custom.n_estimators, custom.max_depth) == (7, 3)
        assert modeling.build_pipeline("logreg_raw", {"C": 0.5}).named_steps["clf"].C == 0.5

    def test_strip_prefix(self):
        assert modeling.strip_prefix({"clf__C": 1, "plain": 2}) == {"C": 1, "plain": 2}

    def test_pipelines_accept_the_raw_columns_and_return_probabilities(self, split):
        train, test = split
        X, y = data_module.features_and_target(train.head(1200))
        Xt, _ = data_module.features_and_target(test.head(100))
        for key in modeling.MODEL_SPECS:
            pipe = modeling.build_pipeline(key, {"n_estimators": 10} if key.startswith("rf") else None).fit(X, y)
            proba = pipe.predict_proba(Xt)
            assert proba.shape == (100, 2)
            assert np.all((proba >= 0) & (proba <= 1)) and np.allclose(proba.sum(axis=1), 1)

    def test_baseline_predicts_the_training_prior(self, split):
        X, y = data_module.features_and_target(split[0])
        score = modeling.build_pipeline("baseline_majority").fit(X, y).predict_proba(X.head(3))[:, 1]
        assert score == pytest.approx([y.mean()] * 3)

    def test_unseen_type_does_not_crash_the_encoder(self, split):
        X, y = data_module.features_and_target(split[0].head(800))
        pipe = modeling.build_pipeline("logreg_raw").fit(X, y)
        odd = X.head(2).copy()
        odd["type"] = "Q"
        assert pipe.predict_proba(odd).shape == (2, 2)

    def test_tune_without_search_reports_cv_score_and_defaults(self, split):
        X, y = data_module.features_and_target(split[0].head(1500))
        result = modeling.tune("logreg_raw", X, y, search=False, cv_folds=3, n_jobs=1)
        assert 0.0 <= result.cv_pr_auc_mean <= 1.0 and result.n_candidates == 1
        assert result.best_params == cfg.DEFAULT_LR_PARAMS

    def test_tune_with_search_picks_parameters_from_the_grid(self, split):
        X, y = data_module.features_and_target(split[0].head(1500))
        result = modeling.tune("logreg_raw", X, y, search=True, cv_folds=3, n_jobs=1)
        assert result.n_candidates == len(modeling.LR_SEARCH_SPACE["clf__C"])
        assert result.best_params["C"] in modeling.LR_SEARCH_SPACE["clf__C"]

    def test_out_of_fold_scores_are_not_in_sample_scores(self, split):
        X, y = data_module.features_and_target(split[0].head(2000))
        pipe = modeling.build_pipeline("rf_raw", {"n_estimators": 30, "max_depth": None})
        oof = modeling.out_of_fold_scores(pipe, X, y, cv_folds=3)
        in_sample = pipe.fit(X, y).predict_proba(X)[:, 1]
        assert oof.shape == (len(X),) and np.all((oof >= 0) & (oof <= 1))
        from sklearn.metrics import average_precision_score as ap

        # A fully grown forest memorises its training rows; out-of-fold scores must be clearly worse.
        assert ap(y, in_sample) > 0.95 > ap(y, oof) + 0.05


# --------------------------------------------------------------------------- #
# Operating threshold
# --------------------------------------------------------------------------- #
class TestSelectThreshold:
    def test_separable_scores_give_a_perfect_operating_point(self):
        y = np.array([0, 0, 0, 0, 1, 1])
        s = np.array([0.1, 0.2, 0.15, 0.3, 0.7, 0.9])
        choice = modeling.select_threshold(y, s)
        assert 0.3 < choice.threshold <= 0.7
        assert (choice.precision, choice.recall, choice.f_beta) == (1.0, 1.0, 1.0)
        assert (choice.n_positive, choice.n_samples) == (2, 6)

    def test_matches_brute_force_search_on_random_scores(self):
        rng = np.random.default_rng(1)
        y = (rng.random(400) < 0.1).astype(int)
        s = np.clip(0.3 * y + rng.normal(0.2, 0.15, 400), 0, 1)
        choice = modeling.select_threshold(y, s, beta=2.0)

        def f2(t):
            pred = s >= t
            tp, fp, fn = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum()), int((~pred & (y == 1)).sum())
            return M.f_beta_from_counts(tp, fp, fn, 2.0)

        best = max(f2(t) for t in np.unique(s))
        assert choice.f_beta == pytest.approx(best)
        assert f2(choice.threshold) == pytest.approx(best)  # the returned midpoint flags the same rows

    def test_more_weight_on_recall_never_raises_the_threshold(self):
        rng = np.random.default_rng(2)
        y = (rng.random(600) < 0.08).astype(int)
        s = np.clip(0.35 * y + rng.normal(0.25, 0.15, 600), 0, 1)
        t_precision = modeling.select_threshold(y, s, beta=0.5).threshold
        t_recall = modeling.select_threshold(y, s, beta=4.0).threshold
        assert t_recall <= t_precision

    @pytest.mark.parametrize("y", [[0, 0, 0], [1, 1, 1], []])
    def test_requires_both_classes(self, y):
        with pytest.raises(ValueError, match="both classes"):
            modeling.select_threshold(np.array(y), np.array([0.1] * len(y)))


# --------------------------------------------------------------------------- #
# Metrics against hand calculations
# --------------------------------------------------------------------------- #
Y = np.array([0, 0, 0, 0, 1, 1, 0, 1, 0, 0])
S = np.array([0.1, 0.2, 0.3, 0.6, 0.9, 0.4, 0.05, 0.8, 0.15, 0.55])


class TestMetrics:
    def test_threshold_is_inclusive(self):
        assert M.predict_labels([0.2, 0.5, 0.8], 0.5).tolist() == [0, 1, 1]

    def test_confusion_counts(self):
        assert M.confusion_counts(Y, M.predict_labels(S, 0.5)) == {"tn": 5, "fp": 2, "fn": 1, "tp": 2}

    def test_classification_metrics_match_hand_calculation(self):
        m = M.classification_metrics(Y, S, 0.5)
        assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 2, 1, 5)
        assert m["precision"] == pytest.approx(2 / 4)
        assert m["recall"] == pytest.approx(2 / 3)
        assert m["specificity"] == pytest.approx(5 / 7)
        assert m["accuracy"] == pytest.approx(7 / 10)
        assert m["balanced_accuracy"] == pytest.approx((2 / 3 + 5 / 7) / 2)
        assert m["f1"] == pytest.approx(2 * 0.5 * (2 / 3) / (0.5 + 2 / 3))
        assert m["f2"] == pytest.approx(5 * 0.5 * (2 / 3) / (4 * 0.5 + 2 / 3))
        assert m["alert_rate"] == pytest.approx(0.4) and m["false_alarm_rate"] == pytest.approx(2 / 7)
        expected_mcc = (2 * 5 - 2 * 1) / np.sqrt((2 + 2) * (2 + 1) * (5 + 2) * (5 + 1))
        assert m["mcc"] == pytest.approx(expected_mcc)

    def test_ranking_metrics_are_threshold_free(self):
        low, high = M.classification_metrics(Y, S, 0.1), M.classification_metrics(Y, S, 0.9)
        assert low["roc_auc"] == pytest.approx(high["roc_auc"]) and low["pr_auc"] == pytest.approx(high["pr_auc"])
        # ROC-AUC: share of (failure, normal) pairs ranked correctly. 3 x 7 = 21 pairs, 19 correct.
        assert low["roc_auc"] == pytest.approx(19 / 21)

    def test_degenerate_cases_do_not_divide_by_zero(self):
        m = M.classification_metrics(Y, np.zeros(len(Y)), 0.5)  # nothing flagged
        assert m["precision"] == 0.0 and m["recall"] == 0.0 and m["f1"] == 0.0 and m["tp"] == 0
        assert M.f_beta_from_counts(0, 0, 0, 2.0) == 0.0

    def test_threshold_table_agrees_with_classification_metrics(self):
        table = M.threshold_table(Y, S, [0.3, 0.5, 0.7]).set_index("threshold")
        for t in (0.3, 0.5, 0.7):
            reference = M.classification_metrics(Y, S, t)
            for column in ("precision", "recall", "f1", "f2", "tp", "fp", "fn", "tn"):
                assert table.loc[t, column] == pytest.approx(reference[column]), (t, column)
        assert table["recall"].is_monotonic_decreasing

    def test_cumulative_gains(self):
        gains = M.cumulative_gains(Y, S)
        assert gains["fraction_inspected"].iloc[0] == 0 and gains["fraction_failures_found"].iloc[0] == 0
        assert gains["fraction_failures_found"].iloc[-1] == pytest.approx(1.0)
        assert gains["fraction_failures_found"].is_monotonic_increasing
        assert M.recall_at_fraction(Y, S, 0.3) == pytest.approx(2 / 3)  # top 3 scores hold 2 of 3 failures
        assert M.recall_at_fraction(Y, S, 1.0) == pytest.approx(1.0)

    def test_per_mode_recall(self):
        frame = pd.DataFrame(
            {
                cfg.TARGET: [1, 1, 1, 1, 0],
                "twf": [1, 0, 0, 0, 0], "hdf": [0, 1, 0, 0, 0], "pwf": [0, 1, 1, 0, 0],
                "osf": [0, 0, 0, 0, 0], "rnf": [0, 0, 0, 0, 1],
            }
        )
        scores = [0.1, 0.9, 0.2, 0.9, 0.9]
        table = M.per_mode_recall(frame, scores, 0.5).set_index("mode")
        assert table.loc["hdf", ["n_failures", "detected"]].tolist() == [1, 1]
        assert table.loc["pwf", ["n_failures", "detected"]].tolist() == [2, 1]   # a failure may carry several modes
        assert table.loc["twf", "detected"] == 0 and table.loc["osf", "n_failures"] == 0
        assert table.loc["rnf", "n_failures"] == 0                               # the RNF flag without a failure is not a failure
        assert table.loc["no_mode_flag", ["n_failures", "detected"]].tolist() == [1, 1]
        assert table.loc["all_failures", ["n_failures", "detected"]].tolist() == [4, 2]
        assert np.isnan(table.loc["osf", "recall"])

    def test_bootstrap_intervals_are_ordered_deterministic_and_bracket_the_estimate(self, holdout, shipped_model):
        scores = holdout[f"score_{shipped_model.model_key}"].to_numpy()
        y = holdout[cfg.TARGET].to_numpy()
        a = M.bootstrap_ci(y, scores, shipped_model.threshold, n_boot=100, seed=3)
        b = M.bootstrap_ci(y, scores, shipped_model.threshold, n_boot=100, seed=3)
        assert a == b
        point = M.classification_metrics(y, scores, shipped_model.threshold)
        for name, (low, high) in a.items():
            assert 0.0 <= low <= high <= 1.0
            assert low - 0.05 <= point[name] <= high + 0.05, name
