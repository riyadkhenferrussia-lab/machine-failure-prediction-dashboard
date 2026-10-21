"""The transparent recommendation layer: status rules, levels, reasoning and batch equivalence."""

import numpy as np
import pandas as pd
import pytest

from src import config as cfg
from src import recommend as R
from src.recommend import IndicatorStatus as S
from src.recommend import RiskLevel as L

OK, WATCH, ALERT = int(S.OK), int(S.WATCH), int(S.ALERT)


# --------------------------------------------------------------------------- #
# Vectorised status rules at their boundaries
# --------------------------------------------------------------------------- #
class TestStatusRules:
    @pytest.mark.parametrize(
        "strain, expected",
        [(11000.0, WATCH), (11000.1, ALERT), (9900.0, WATCH), (9899.9, OK), (0.0, OK)],
    )
    def test_overstrain_type_l(self, strain, expected):
        assert int(R.overstrain_status(strain, 11000.0, 0.10)) == expected

    @pytest.mark.parametrize(
        "power, expected",
        [(3499.9, ALERT), (3500.0, WATCH), (3849.0, WATCH), (3851.0, OK), (8101.0, WATCH), (8099.0, OK), (9000.0, WATCH), (9000.1, ALERT)],
    )
    def test_power(self, power, expected):
        assert int(R.power_status(power, 0.10)) == expected

    @pytest.mark.parametrize(
        "diff, rpm, expected",
        [
            (8.5, 1379.0, ALERT),
            (8.6, 1379.0, WATCH),    # no longer below the limit but within 5%
            (9.0, 1400.0, WATCH),
            (9.05, 1400.0, OK),      # temperature difference 5% above the limit or more
            (8.5, 1450.0, OK),       # speed above the margin: not both conditions
            (12.0, 1000.0, OK),
        ],
    )
    def test_heat_dissipation_needs_both_quantities_close(self, diff, rpm, expected):
        assert int(R.heat_dissipation_status(diff, rpm, 0.05)) == expected

    @pytest.mark.parametrize("wear, expected", [(199.9, OK), (200.0, WATCH), (253.0, WATCH)])
    def test_tool_wear_is_a_watch_never_an_alert(self, wear, expected):
        assert int(R.tool_wear_status(wear)) == expected
        assert int(R.tool_wear_status(10_000.0)) == WATCH

    def test_rules_work_on_arrays(self):
        out = R.power_status(np.array([1000.0, 5000.0, 9500.0]))
        assert out.tolist() == [ALERT, OK, ALERT]


# --------------------------------------------------------------------------- #
# Levels
# --------------------------------------------------------------------------- #
class TestLevels:
    @pytest.mark.parametrize(
        "score, expected",
        [(0.0, L.LOW), (0.1749, L.LOW), (0.175, L.ELEVATED), (0.3499, L.ELEVATED), (0.35, L.HIGH), (1.0, L.HIGH)],
    )
    def test_model_level_bands(self, score, expected):
        assert R.model_level(score, 0.35) is expected

    def test_indicator_level_is_the_worst_status(self):
        assert R.indicator_level([OK, OK]) is L.LOW
        assert R.indicator_level([OK, WATCH, OK]) is L.ELEVATED
        assert R.indicator_level([WATCH, ALERT]) is L.HIGH
        assert R.indicator_level([]) is L.LOW

    def test_levels_are_ordered(self):
        assert L.LOW < L.ELEVATED < L.HIGH and max(L.LOW, L.HIGH) is L.HIGH


class TestRecommend:
    def test_quiet_machine(self, good_reading):
        rec = R.recommend(0.01, 0.35, good_reading)
        assert (rec.level, rec.model_level, rec.indicator_level) == (L.LOW, L.LOW, L.LOW)
        assert not rec.flagged and rec.headline == "No inspection triggered"
        assert rec.actions == ["Continue the routine maintenance schedule."]
        assert len(rec.indicators) == 4 and all(i.status is S.OK for i in rec.indicators)

    def test_model_alone_can_raise_the_level(self, good_reading):
        rec = R.recommend(0.6, 0.35, good_reading)
        assert rec.level is L.HIGH and rec.flagged
        assert any("No documented criterion is met" in line for line in rec.reasoning)

    def test_documented_criterion_overrides_a_low_model_score(self, good_reading):
        good_reading.update(type="L", torque_nm=70.0, tool_wear_min=170.0)   # 11,900 min*Nm > 11,000 (Type L limit)
        rec = R.recommend(0.01, 0.35, good_reading)
        assert rec.indicator_level is L.HIGH and rec.level is L.HIGH and not rec.flagged
        assert any("treat the criterion as authoritative" in line for line in rec.reasoning)
        assert any("Overstrain" in line and "ALERT" in line for line in rec.reasoning)

    def test_watch_indicator_gives_elevated(self, good_reading):
        good_reading["tool_wear_min"] = 215.0
        rec = R.recommend(0.01, 0.35, good_reading)
        assert rec.level is L.ELEVATED and rec.headline == "Add to watch list"
        assert rec.actions[0].startswith("Re-check the readings")
        assert rec.actions[-1].startswith("Record the readings")

    def test_score_between_half_threshold_and_threshold_is_elevated(self, good_reading):
        assert R.recommend(0.2, 0.35, good_reading).level is L.ELEVATED

    def test_every_step_is_explained(self, good_reading):
        rec = R.recommend(0.4, 0.35, good_reading)
        text = " ".join(rec.reasoning)
        assert "0.400" in text and "0.350" in text and "Final level is the higher of" in text
        assert "prediction, not a confirmed fault" in rec.disclaimer

    def test_triggered_indicators_come_with_inspection_hints(self, good_reading):
        good_reading.update(rotational_speed_rpm=800.0, torque_nm=10.0)   # power 837 W: alert
        rec = R.recommend(0.9, 0.35, good_reading)
        assert any(a.startswith("Drive and motor load") for a in rec.actions)

    @pytest.mark.parametrize("score", [-0.1, 1.1, float("nan"), float("inf")])
    def test_invalid_scores_are_rejected(self, good_reading, score):
        with pytest.raises(ValueError, match="score must be"):
            R.recommend(score, 0.35, good_reading)

    @pytest.mark.parametrize("threshold", [0.0, -0.2, 1.5])
    def test_invalid_thresholds_are_rejected(self, good_reading, threshold):
        with pytest.raises(ValueError, match="threshold must be"):
            R.recommend(0.5, threshold, good_reading)

    def test_watch_rules_are_adjustable(self, good_reading):
        good_reading.update(torque_nm=40.0, tool_wear_min=190.0)         # strain 7,600: 63% of the Type L limit
        good_reading["type"] = "L"
        assert R.recommend(0.01, 0.35, good_reading).level is L.LOW
        strict = R.WatchRules(tool_wear_watch_min=180.0)
        assert R.recommend(0.01, 0.35, good_reading, strict).level is L.ELEVATED
        wide = R.WatchRules(overstrain_margin=0.5)
        assert R.recommend(0.01, 0.35, good_reading, wide).level is L.ELEVATED
        quiet = R.WatchRules(elevated_score_fraction=1.0)
        assert R.recommend(0.2, 0.35, good_reading, quiet).level is L.LOW

    def test_default_rules_follow_the_configuration(self):
        rules = R.DEFAULT_RULES
        assert (rules.overstrain_margin, rules.power_margin, rules.heat_margin) == (
            cfg.APPROACH_MARGIN, cfg.APPROACH_MARGIN, cfg.HEAT_APPROACH_MARGIN)
        assert rules.tool_wear_watch_min == cfg.TOOL_WEAR_WATCH_MIN == cfg.TOOL_WEAR_WINDOW_START_MIN
        assert rules.elevated_score_fraction == cfg.ELEVATED_SCORE_FRACTION

    def test_indicator_values_use_the_documented_limits(self, good_reading):
        indicators = {i.key: i for i in R.evaluate_indicators(good_reading)}
        assert set(indicators) == {"overstrain", "power", "heat_dissipation", "tool_wear"}
        assert "12,000 min*Nm (Type M)" in indicators["overstrain"].limit
        assert "3,500 W" in indicators["power"].limit and "9,000 W" in indicators["power"].limit
        assert "8.6 K" in indicators["heat_dissipation"].limit and "1,380 rpm" in indicators["heat_dissipation"].limit
        assert indicators["power"].value == "6,283 W"
        assert indicators["overstrain"].to_dict()["status"] == "OK"


# --------------------------------------------------------------------------- #
# Batch path == single path
# --------------------------------------------------------------------------- #
class TestBatchEquivalence:
    def test_batch_levels_equal_single_levels_on_real_records(self, holdout, shipped_model):
        sample = holdout.sample(400, random_state=0).reset_index(drop=True)
        scores = sample[f"score_{shipped_model.model_key}"].to_numpy()
        batch = R.batch_recommendations(sample, scores, shipped_model.threshold)
        for i, row in sample.iterrows():
            reading = {c: row[c] for c in cfg.RAW_FEATURES}
            single = R.recommend(float(scores[i]), shipped_model.threshold, reading)
            assert batch.loc[i, "risk_level"] == single.level.label, i
            assert batch.loc[i, "model_level"] == single.model_level.label, i
            assert batch.loc[i, "indicator_level"] == single.indicator_level.label, i
            assert bool(batch.loc[i, "flagged_by_model"]) == single.flagged, i

    def test_equivalence_holds_for_custom_rules(self, holdout, shipped_model):
        rules = R.WatchRules(overstrain_margin=0.25, power_margin=0.2, heat_margin=0.15, tool_wear_watch_min=150.0, elevated_score_fraction=0.3)
        sample = holdout.sample(150, random_state=1).reset_index(drop=True)
        scores = sample[f"score_{shipped_model.model_key}"].to_numpy()
        batch = R.batch_recommendations(sample, scores, shipped_model.threshold, rules)
        for i, row in sample.iterrows():
            single = R.recommend(float(scores[i]), shipped_model.threshold, {c: row[c] for c in cfg.RAW_FEATURES}, rules)
            assert batch.loc[i, "risk_level"] == single.level.label

    def test_batch_columns_and_ranks(self, holdout, shipped_model):
        scores = holdout[f"score_{shipped_model.model_key}"].to_numpy()
        batch = R.batch_recommendations(holdout, scores, shipped_model.threshold)
        assert list(batch.columns) == [
            "failure_score", "flagged_by_model", "model_level", "indicator_level", "risk_level",
            "risk_rank", "recommended_action", "triggered_indicators",
        ]
        assert set(batch["risk_level"]) <= {"High", "Elevated", "Low"}
        assert (batch["risk_rank"] == batch["risk_level"].map({"High": 2, "Elevated": 1, "Low": 0})).all()
        assert (batch.loc[batch["risk_level"] == "Low", "triggered_indicators"] == "").all()

    def test_status_frame_matches_scalar_functions(self):
        frame = pd.DataFrame(
            {"type": ["L", "M", "H"], "tool_wear_min": [210.0, 100.0, 20.0], "torque_nm": [60.0, 20.0, 5.0],
             "rotational_speed_rpm": [1300.0, 1500.0, 1400.0], "process_temp_k": [308.0, 310.0, 309.0],
             "air_temp_k": [299.8, 300.0, 300.0]}
        )
        statuses = R.indicator_status_frame(frame)
        assert list(statuses.columns) == ["overstrain", "power", "heat_dissipation", "tool_wear"]
        for i, row in frame.iterrows():
            by_key = {x.key: int(x.status) for x in R.evaluate_indicators(row.to_dict())}
            assert statuses.loc[i].to_dict() == by_key

    def test_empty_batch(self, shipped_model):
        empty = pd.DataFrame({c: [] for c in cfg.RAW_FEATURES})
        out = R.batch_recommendations(empty, np.array([]), shipped_model.threshold)
        assert out.empty


class TestRecommendationQualityOnHoldout:
    """The rules are meaningful: higher levels have much higher recorded failure rates."""

    def test_failure_rate_increases_with_level(self, holdout, shipped_model):
        scores = holdout[f"score_{shipped_model.model_key}"].to_numpy()
        final, _, _ = R.level_codes(holdout, scores, shipped_model.threshold)
        y = holdout[cfg.TARGET].to_numpy()
        rates = [y[final == code].mean() for code in (0, 1, 2)]
        assert rates[0] < rates[1] < rates[2]
        assert rates[2] > 0.8 and rates[0] < 0.01
        assert (final == 0).sum() > 0.7 * len(final)   # the watch list stays a small minority
