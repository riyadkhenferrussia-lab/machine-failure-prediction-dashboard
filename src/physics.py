"""Physical quantities and the documented failure criteria of the AI4I 2020 process.

These are plain vectorised functions (they accept floats, numpy arrays or pandas
Series) shared by feature engineering, data validation and the recommendation
rules, so a formula is defined exactly once.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from . import config as cfg


def mechanical_power_w(torque_nm: Any, speed_rpm: Any) -> Any:
    """Mechanical power in watts: torque [Nm] x angular speed [rad/s]."""
    return torque_nm * speed_rpm * (2.0 * math.pi / 60.0)


def temperature_difference_k(process_temp_k: Any, air_temp_k: Any) -> Any:
    """Process temperature minus air temperature, in kelvin."""
    return process_temp_k - air_temp_k


def strain_index_min_nm(tool_wear_min: Any, torque_nm: Any) -> Any:
    """Overstrain index: tool wear [min] x torque [Nm]."""
    return tool_wear_min * torque_nm


def osf_limit(type_value: Any) -> Any:
    """Overstrain limit [min*Nm] for the product variant (scalar or Series)."""
    if isinstance(type_value, pd.Series):
        return type_value.map(cfg.OSF_LIMITS_MIN_NM).astype(float)
    return cfg.OSF_LIMITS_MIN_NM[str(type_value)]


# ----- documented failure criteria (deterministic modes) ------------------- #
def pwf_criterion(power_w: Any) -> Any:
    """Power failure: power outside [3500 W, 9000 W]."""
    return (power_w < cfg.POWER_MIN_W) | (power_w > cfg.POWER_MAX_W)


def hdf_criterion(temp_diff_k: Any, speed_rpm: Any) -> Any:
    """Heat dissipation failure: small temperature difference AND low speed."""
    return (temp_diff_k < cfg.HDF_TEMP_DIFF_MAX_K) & (speed_rpm < cfg.HDF_SPEED_MAX_RPM)


def osf_criterion(strain: Any, type_value: Any) -> Any:
    """Overstrain failure: wear x torque above the limit of the product variant."""
    return strain > osf_limit(type_value)


def documented_rule_agreement(df: pd.DataFrame) -> dict[str, float]:
    """Fraction of rows where each documented criterion reproduces the dataset flag.

    ``df`` must use canonical column names. A value of 1.0 means the labels in the
    file follow the documented physics exactly, which is a strong integrity check.
    """
    power = mechanical_power_w(df["torque_nm"], df["rotational_speed_rpm"])
    dt = temperature_difference_k(df["process_temp_k"], df["air_temp_k"])
    strain = strain_index_min_nm(df["tool_wear_min"], df["torque_nm"])
    rules = {
        "pwf": pwf_criterion(power),
        "hdf": hdf_criterion(dt, df["rotational_speed_rpm"]),
        "osf": osf_criterion(strain, df["type"]),
    }
    return {
        mode: float(np.mean(np.asarray(rule).astype(bool) == df[mode].to_numpy().astype(bool)))
        for mode, rule in rules.items()
    }
