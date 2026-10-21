"""Transparent maintenance-inspection recommendations.

The recommendation combines two independent, inspectable sources:

1. the model's failure *score* relative to the operating threshold chosen on the
   training data, and
2. explicit engineering *indicators* whose limits are the failure criteria documented
   for the AI4I 2020 process (see ``src/config.py``).

Nothing is hidden inside the model: every threshold, rule and the way the final level is
derived is returned alongside the result. The output is a *screening result*, never a
confirmed fault.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Mapping

import numpy as np
import pandas as pd

from . import config as cfg
from .physics import mechanical_power_w, osf_limit, strain_index_min_nm, temperature_difference_k

DISCLAIMER = (
    "This is a model- and rule-based screening result for decision support. It is a "
    "prediction, not a confirmed fault - confirm with a physical inspection."
)


class RiskLevel(IntEnum):
    """Ordered so that ``max()`` gives the more severe level."""

    LOW = 0
    ELEVATED = 1
    HIGH = 2

    @property
    def label(self) -> str:
        return self.name.capitalize()


class IndicatorStatus(IntEnum):
    OK = 0
    WATCH = 1  # approaching a limit, or inside the probabilistic tool-wear window
    ALERT = 2  # a documented failure criterion is met


@dataclass(frozen=True)
class WatchRules:
    """The adjustable part of the rule layer: when does an indicator move from OK to WATCH?

    The ALERT criteria are the documented failure criteria and are not adjustable here
    (they live in ``src/config.py``). Defaults are the values used by the dashboard.
    """

    overstrain_margin: float = cfg.APPROACH_MARGIN
    power_margin: float = cfg.APPROACH_MARGIN
    heat_margin: float = cfg.HEAT_APPROACH_MARGIN
    tool_wear_watch_min: float = cfg.TOOL_WEAR_WATCH_MIN
    elevated_score_fraction: float = cfg.ELEVATED_SCORE_FRACTION


DEFAULT_RULES = WatchRules()


# --------------------------------------------------------------------------- #
# Vectorised status rules (shared by the single-reading and the batch paths)
# --------------------------------------------------------------------------- #
def _as_array(values: Any) -> np.ndarray:
    """Float ndarray (0-d for scalars) so ``~`` and ``&`` always act on numpy booleans."""
    return np.asarray(values, dtype=float)


def _status(alert: np.ndarray, watch: np.ndarray) -> np.ndarray:
    return np.where(alert, int(IndicatorStatus.ALERT), np.where(watch, int(IndicatorStatus.WATCH), 0))


def overstrain_status(strain: Any, limit: Any, margin: float = cfg.APPROACH_MARGIN) -> np.ndarray:
    strain, limit = _as_array(strain), _as_array(limit)
    alert = strain > limit
    watch = ~alert & (strain >= (1.0 - margin) * limit)
    return _status(alert, watch)


def power_status(power_w: Any, margin: float = cfg.APPROACH_MARGIN) -> np.ndarray:
    power_w = _as_array(power_w)
    low, high = cfg.POWER_MIN_W, cfg.POWER_MAX_W
    alert = (power_w < low) | (power_w > high)
    watch = ~alert & ((power_w < low * (1.0 + margin)) | (power_w > high * (1.0 - margin)))
    return _status(alert, watch)


def heat_dissipation_status(temp_diff_k: Any, speed_rpm: Any, margin: float = cfg.HEAT_APPROACH_MARGIN) -> np.ndarray:
    temp_diff_k, speed_rpm = _as_array(temp_diff_k), _as_array(speed_rpm)
    alert = (temp_diff_k < cfg.HDF_TEMP_DIFF_MAX_K) & (speed_rpm < cfg.HDF_SPEED_MAX_RPM)
    watch = (
        ~alert
        & (temp_diff_k < cfg.HDF_TEMP_DIFF_MAX_K * (1.0 + margin))
        & (speed_rpm < cfg.HDF_SPEED_MAX_RPM * (1.0 + margin))
    )
    return _status(alert, watch)


def tool_wear_status(tool_wear_min: Any, watch_from_min: float = cfg.TOOL_WEAR_WATCH_MIN) -> np.ndarray:
    tool_wear_min = _as_array(tool_wear_min)
    return _status(np.zeros_like(tool_wear_min, dtype=bool), tool_wear_min >= watch_from_min)


# --------------------------------------------------------------------------- #
# Single reading
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Indicator:
    key: str
    name: str
    value: str
    limit: str
    status: IndicatorStatus
    explanation: str
    inspect: str

    def to_dict(self) -> dict[str, str]:
        return {
            "indicator": self.name,
            "value": self.value,
            "limit": self.limit,
            "status": self.status.name,
            "explanation": self.explanation,
            "what_to_check": self.inspect,
        }


def evaluate_indicators(reading: Mapping[str, Any], rules: WatchRules = DEFAULT_RULES) -> list[Indicator]:
    """Evaluate the four engineering indicators for one validated reading."""
    wear = float(reading["tool_wear_min"])
    torque = float(reading["torque_nm"])
    speed = float(reading["rotational_speed_rpm"])
    air = float(reading["air_temp_k"])
    process = float(reading["process_temp_k"])
    variant = str(reading["type"])

    strain = float(strain_index_min_nm(wear, torque))
    limit = float(osf_limit(variant))
    power = float(mechanical_power_w(torque, speed))
    diff = float(temperature_difference_k(process, air))

    s_strain = IndicatorStatus(int(overstrain_status(strain, limit, rules.overstrain_margin)))
    s_power = IndicatorStatus(int(power_status(power, rules.power_margin)))
    s_heat = IndicatorStatus(int(heat_dissipation_status(diff, speed, rules.heat_margin)))
    s_wear = IndicatorStatus(int(tool_wear_status(wear, rules.tool_wear_watch_min)))

    if s_strain is IndicatorStatus.ALERT:
        strain_text = f"Tool wear x torque exceeds the Type {variant} limit ({strain / limit:.0%} of limit)."
    else:
        strain_text = f"Tool wear x torque is at {strain / limit:.0%} of the Type {variant} limit."

    if power < cfg.POWER_MIN_W:
        power_text = "Mechanical power is below the lower limit."
    elif power > cfg.POWER_MAX_W:
        power_text = "Mechanical power is above the upper limit."
    elif s_power is IndicatorStatus.WATCH:
        power_text = f"Mechanical power is within {rules.power_margin:.0%} of one of its limits."
    else:
        power_text = "Mechanical power is inside its limits."

    if s_heat is IndicatorStatus.ALERT:
        heat_text = "Small process-air temperature difference at low speed: heat is not being dissipated."
    elif s_heat is IndicatorStatus.WATCH:
        heat_text = (
            f"Temperature difference and speed are both within {rules.heat_margin:.0%} of the "
            "heat-dissipation limits."
        )
    else:
        heat_text = "Heat-dissipation criterion not close to its limits."

    if s_wear is IndicatorStatus.WATCH:
        wear_text = (
            "Tool wear has reached the 200-240 min range in which the dataset documents tool "
            "replacement or failure (a probabilistic window, not a fixed criterion)."
        )
    else:
        wear_text = "Tool wear is below the documented replacement/failure window."

    return [
        Indicator(
            "overstrain",
            "Overstrain (tool wear x torque)",
            f"{strain:,.0f} min*Nm",
            f"alert above {limit:,.0f} min*Nm (Type {variant})",
            s_strain,
            strain_text,
            "Tool and spindle load: check cutting parameters, torque loading, tool condition and clamping; replace worn tooling.",
        ),
        Indicator(
            "power",
            "Mechanical power",
            f"{power:,.0f} W",
            f"alert below {cfg.POWER_MIN_W:,.0f} W or above {cfg.POWER_MAX_W:,.0f} W",
            s_power,
            power_text,
            "Drive and motor load: check motor current, drive settings, couplings/belts and that the operating point is intended.",
        ),
        Indicator(
            "heat_dissipation",
            "Heat dissipation",
            f"dT {diff:.1f} K at {speed:,.0f} rpm",
            f"alert if dT < {cfg.HDF_TEMP_DIFF_MAX_K} K and speed < {cfg.HDF_SPEED_MAX_RPM:,.0f} rpm",
            s_heat,
            heat_text,
            "Cooling path: check coolant flow and level, heat exchanger, fans and filters, and ambient ventilation.",
        ),
        Indicator(
            "tool_wear",
            "Tool wear window",
            f"{wear:,.0f} min",
            f"watch from {rules.tool_wear_watch_min:,.0f} min (documented window: 200-240 min)",
            s_wear,
            wear_text,
            "Tool condition: inspect the tool and schedule a tool change.",
        ),
    ]


def model_level(score: float, threshold: float, elevated_fraction: float = cfg.ELEVATED_SCORE_FRACTION) -> RiskLevel:
    """HIGH at/above the operating threshold, ELEVATED from half of it, else LOW."""
    if score >= threshold:
        return RiskLevel.HIGH
    if score >= threshold * elevated_fraction:
        return RiskLevel.ELEVATED
    return RiskLevel.LOW


def indicator_level(statuses: Any) -> RiskLevel:
    """Map the worst indicator status to a risk level (ALERT->HIGH, WATCH->ELEVATED)."""
    worst = int(np.max(np.asarray(statuses, dtype=int))) if len(statuses) else 0
    return RiskLevel(worst)


@dataclass(frozen=True)
class Recommendation:
    level: RiskLevel
    model_level: RiskLevel
    indicator_level: RiskLevel
    score: float
    threshold: float
    flagged: bool
    headline: str
    actions: list[str]
    reasoning: list[str]
    indicators: list[Indicator]
    disclaimer: str = DISCLAIMER


_HEADLINES = {
    RiskLevel.HIGH: "Flag for inspection",
    RiskLevel.ELEVATED: "Add to watch list",
    RiskLevel.LOW: "No inspection triggered",
}


def recommend(
    score: float, threshold: float, reading: Mapping[str, Any], rules: WatchRules = DEFAULT_RULES
) -> Recommendation:
    """Build the recommendation for one machine reading."""
    if not np.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"score must be a finite value in [0, 1], got {score!r}")
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"threshold must be in (0, 1], got {threshold!r}")

    fraction = rules.elevated_score_fraction
    indicators = evaluate_indicators(reading, rules)
    by_model = model_level(score, threshold, fraction)
    by_rules = indicator_level([int(i.status) for i in indicators])
    final = max(by_model, by_rules)
    triggered = sorted(
        (i for i in indicators if i.status is not IndicatorStatus.OK),
        key=lambda i: (-int(i.status), i.name),
    )

    if score >= threshold:
        position = "at or above the operating threshold"
    elif score >= threshold * fraction:
        position = f"between {fraction:.0%} and 100% of the threshold"
    else:
        position = f"below {fraction:.0%} of the threshold"
    reasoning = [
        f"Model score {score:.3f} vs operating threshold {threshold:.3f}: {position} "
        f"-> model level {by_model.label}."
    ]
    if triggered:
        reasoning += [f"{i.name}: {i.status.name} - {i.value} ({i.limit})." for i in triggered]
    else:
        reasoning.append("No engineering indicator is in its watch or alert range.")
    reasoning.append(
        f"Final level is the higher of the model level ({by_model.label}) and the indicator "
        f"level ({by_rules.label}) -> {final.label}."
    )
    if by_model is RiskLevel.LOW and by_rules is RiskLevel.HIGH:
        reasoning.append(
            "The model score is low although a documented failure criterion is met: treat the "
            "criterion as authoritative and re-check the input values."
        )
    if by_model is RiskLevel.HIGH and by_rules is RiskLevel.LOW:
        reasoning.append(
            "No documented criterion is met; the model reacts to a combination of readings "
            "(see the sensitivity table). Inspect general condition and recent history."
        )

    if final is RiskLevel.HIGH:
        actions = ["Prioritise a manual inspection before continued operation under these conditions."]
    elif final is RiskLevel.ELEVATED:
        actions = [
            "Re-check the readings at the next planned stop and consider a shorter inspection interval."
        ]
    else:
        actions = ["Continue the routine maintenance schedule."]
    if final is not RiskLevel.LOW:
        actions += [i.inspect for i in triggered]
        actions.append("Record the readings and the inspection findings to confirm or reject this prediction.")

    return Recommendation(
        level=final,
        model_level=by_model,
        indicator_level=by_rules,
        score=float(score),
        threshold=float(threshold),
        flagged=bool(score >= threshold),
        headline=_HEADLINES[final],
        actions=actions,
        reasoning=reasoning,
        indicators=indicators,
    )


# --------------------------------------------------------------------------- #
# Batch (vectorised)
# --------------------------------------------------------------------------- #
_INDICATOR_NAMES = {
    "overstrain": "Overstrain",
    "power": "Power",
    "heat_dissipation": "Heat dissipation",
    "tool_wear": "Tool wear window",
}


def indicator_status_frame(df: pd.DataFrame, rules: WatchRules = DEFAULT_RULES) -> pd.DataFrame:
    """Vectorised indicator statuses (0 OK, 1 WATCH, 2 ALERT) for canonical-named rows."""
    strain = strain_index_min_nm(df["tool_wear_min"], df["torque_nm"])
    power = mechanical_power_w(df["torque_nm"], df["rotational_speed_rpm"])
    diff = temperature_difference_k(df["process_temp_k"], df["air_temp_k"])
    return pd.DataFrame(
        {
            "overstrain": overstrain_status(strain, osf_limit(df["type"]), rules.overstrain_margin),
            "power": power_status(power, rules.power_margin),
            "heat_dissipation": heat_dissipation_status(diff, df["rotational_speed_rpm"], rules.heat_margin),
            "tool_wear": tool_wear_status(df["tool_wear_min"], rules.tool_wear_watch_min),
        },
        index=df.index,
    ).astype(int)


def level_codes(
    df: pd.DataFrame, scores: Any, threshold: float, rules: WatchRules = DEFAULT_RULES
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Final, model and indicator risk levels as integer codes (2 High, 1 Elevated, 0 Low)."""
    scores = np.asarray(scores, dtype=float)
    statuses = indicator_status_frame(df, rules)
    worst = statuses.max(axis=1).to_numpy(dtype=int) if len(statuses) else np.zeros(0, dtype=int)
    by_model = np.where(
        scores >= threshold,
        int(RiskLevel.HIGH),
        np.where(scores >= threshold * rules.elevated_score_fraction, int(RiskLevel.ELEVATED), int(RiskLevel.LOW)),
    )
    return np.maximum(by_model, worst), by_model, worst


def batch_recommendations(
    df: pd.DataFrame, scores: np.ndarray, threshold: float, rules: WatchRules = DEFAULT_RULES
) -> pd.DataFrame:
    """Model level, indicator level and final risk level for many rows at once.

    Uses exactly the same rules as ``recommend`` (a test asserts the equivalence).
    ``risk_rank`` is 2/1/0 for High/Elevated/Low, handy for sorting.
    """
    scores = np.asarray(scores, dtype=float)
    statuses = indicator_status_frame(df, rules)
    final, by_model, worst = level_codes(df, scores, threshold, rules)

    keys = list(statuses.columns)
    values = statuses.to_numpy(dtype=int)
    triggered = [
        ", ".join(
            f"{_INDICATOR_NAMES[key]} ({IndicatorStatus(int(v)).name})"
            for key, v in zip(keys, row)
            if v
        )
        for row in values
    ]
    return pd.DataFrame(
        {
            "failure_score": scores,
            "flagged_by_model": scores >= threshold,
            "model_level": [RiskLevel(int(v)).label for v in by_model],
            "indicator_level": [RiskLevel(int(v)).label for v in worst],
            "risk_level": [RiskLevel(int(v)).label for v in final],
            "risk_rank": final.astype(int),
            "recommended_action": [_HEADLINES[RiskLevel(int(v))] for v in final],
            "triggered_indicators": triggered,
        },
        index=df.index,
    )
