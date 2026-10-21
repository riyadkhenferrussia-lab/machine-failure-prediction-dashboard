"""Column naming and the physical formulas / documented failure criteria."""

import math

import numpy as np
import pandas as pd
import pytest

from src import config as cfg
from src import physics, schema


class TestSchema:
    @pytest.mark.parametrize(
        "name, expected",
        [
            ("Air temperature [K]", "air_temp_k"),
            ("air_temp_k", "air_temp_k"),
            ("  AIR   TEMPERATURE [k] ", "air_temp_k"),
            ("Machine failure", "machine_failure"),
            ("UID", "udi"),  # the UCI web page spells the first column this way
            ("Product ID", "product_id"),
            ("Torque [Nm]", "torque_nm"),
        ],
    )
    def test_known_names_resolve(self, name, expected):
        assert schema.canonical_column_name(name) == expected

    def test_unknown_name_is_none(self):
        assert schema.canonical_column_name("Vibration [mm/s]") is None

    def test_mapping_skips_unknown_columns(self):
        mapping = schema.column_mapping(["UDI", "Notes", "Type"])
        assert mapping == {"UDI": "udi", "Type": "type"}

    def test_two_spellings_of_one_column_are_rejected(self):
        with pytest.raises(ValueError, match="both map to"):
            schema.column_mapping(["Torque [Nm]", "torque_nm"])

    def test_standardize_keeps_unknown_columns(self):
        frame = pd.DataFrame({"Torque [Nm]": [1.0], "Notes": ["x"]})
        assert list(schema.standardize_columns(frame).columns) == ["torque_nm", "Notes"]

    def test_accepted_names_lists_both_spellings(self):
        assert schema.accepted_names("torque_nm") == ["torque_nm", "Torque [Nm]"]

    def test_every_required_column_has_a_published_spelling(self):
        assert set(cfg.RAW_TO_CANONICAL.values()) == set(cfg.REQUIRED_COLUMNS)


class TestFormulas:
    def test_power_matches_hand_calculation(self):
        # 40 Nm at 1500 rpm: 40 * 1500 * 2*pi/60 = 6283.19 W
        assert physics.mechanical_power_w(40.0, 1500.0) == pytest.approx(40 * 1500 * 2 * math.pi / 60)
        assert physics.mechanical_power_w(40.0, 1500.0) == pytest.approx(6283.185, abs=1e-3)

    def test_temperature_difference_and_strain(self):
        assert physics.temperature_difference_k(310.0, 300.0) == 10.0
        assert physics.strain_index_min_nm(100.0, 40.0) == 4000.0

    def test_osf_limit_per_variant_scalar_and_series(self):
        assert [physics.osf_limit(v) for v in "LMH"] == [11000.0, 12000.0, 13000.0]
        out = physics.osf_limit(pd.Series(["H", "L"]))
        assert out.tolist() == [13000.0, 11000.0]

    @pytest.mark.parametrize("power, expected", [(3499.9, True), (3500.0, False), (9000.0, False), (9000.1, True)])
    def test_power_criterion_boundaries(self, power, expected):
        assert bool(physics.pwf_criterion(power)) is expected

    @pytest.mark.parametrize(
        "dt, rpm, expected",
        [(8.5, 1379.0, True), (8.6, 1379.0, False), (8.5, 1380.0, False), (12.0, 1000.0, False)],
    )
    def test_heat_dissipation_needs_both_conditions(self, dt, rpm, expected):
        assert bool(physics.hdf_criterion(dt, rpm)) is expected

    @pytest.mark.parametrize("variant, limit", [("L", 11000.0), ("M", 12000.0), ("H", 13000.0)])
    def test_overstrain_boundary_is_strict(self, variant, limit):
        assert bool(physics.osf_criterion(limit, variant)) is False
        assert bool(physics.osf_criterion(limit + 1, variant)) is True

    def test_vectorised_functions_accept_series(self):
        series = pd.Series([3000.0, 5000.0, 9500.0])
        assert physics.pwf_criterion(series).tolist() == [True, False, True]


class TestDocumentedRulesOnRealData:
    """The published labels follow the documented criteria exactly (integrity check on the file)."""

    def test_deterministic_modes_are_reproduced_exactly(self, dataset):
        agreement = physics.documented_rule_agreement(dataset)
        assert agreement == {"pwf": 1.0, "hdf": 1.0, "osf": 1.0}

    def test_agreement_drops_when_labels_are_corrupted(self, dataset):
        broken = dataset.copy()
        broken.loc[broken.index[:50], "pwf"] = 1 - broken.loc[broken.index[:50], "pwf"]
        assert physics.documented_rule_agreement(broken)["pwf"] == pytest.approx(1 - 50 / len(broken))
