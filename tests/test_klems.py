"""Tests for the KLEMS production function branch of InputOutputModel.

The fixture is a 3-sector economy of electricity (energy), steel (materials), and services (ND).
"""

from __future__ import annotations

import numpy as np
import pytest

from pyMacroIO.model import InputOutputModel
from pyMacroIO.config import ModelConfig


# Fixtures
@pytest.fixture(scope="module")
def klems_model(klems_data_dict) -> InputOutputModel:
    """Return a 3-sector KLEMS InputOutputModel with one energy, one material, and one ND sector."""
    config = ModelConfig(
        n_periods=10,
        time_frequency="quarterly",
        prod_function="klems",
    )
    return InputOutputModel(
        n_periods=10,
        time_frequency="quarterly",
        config=config,
        _data_dict=klems_data_dict,
    )


# Shared helpers
def _base_inventories_klems(m: InputOutputModel) -> np.ndarray:
    """Return base-year target inventory S_tar = A * x0 * n where n is the coverage factor."""
    return m.A * m.x0[np.newaxis, :] * m.n[np.newaxis, :]


def _base_l_array(m: InputOutputModel) -> np.ndarray:
    """Return an (N, 2) labour array with l0 in both columns."""
    l = np.zeros((m.N, 2))
    l[:, 0] = m.l0
    l[:, 1] = m.l0
    return l


# Tests for _ces_output_constraint
class TestCesOutputConstraint:
    """Tests for the _ces_output_constraint helper method."""

    def test_cobb_douglas_limit(self, klems_model: InputOutputModel) -> None:
        """Verifies that sigma=1 recovers the Cobb-Douglas geometric-mean formula."""
        result = klems_model._ces_output_constraint(
            np.array([4.0, 9.0]),
            np.array([0.5, 0.5]),
            sigma=1.0,
        )
        assert np.isclose(result, 6.0, rtol=1e-10)

    def test_general_ces_sigma_two(self, klems_model: InputOutputModel) -> None:
        """Checks the CES formula at sigma=2 against a hand-computed value."""
        result = klems_model._ces_output_constraint(
            np.array([4.0, 9.0]),
            np.array([0.5, 0.5]),
            sigma=2.0,
        )
        assert np.isclose(result, 6.25, rtol=1e-10)

    def test_low_sigma_approaches_leontief(self, klems_model: InputOutputModel) -> None:
        """Very low sigma (near-Leontief) pulls output close to the minimum input."""
        result = klems_model._ces_output_constraint(
            np.array([2.0, 8.0]),
            np.array([0.5, 0.5]),
            sigma=0.01,
        )
        assert result > 1.9   # should be close to min(2, 8) = 2
        assert result < 2.5   # must not be close to the Cobb-Douglas value of 4

    def test_single_input_trivial(self, klems_model: InputOutputModel) -> None:
        """Checks that a single-input CES with unit weight returns the input itself."""
        result = klems_model._ces_output_constraint(
            np.array([5.0]),
            np.array([1.0]),
            sigma=2.0,
        )
        assert np.isclose(result, 5.0, rtol=1e-10)

    def test_no_valid_inputs_returns_inf(self, klems_model: InputOutputModel) -> None:
        """Asserts that the guard on valid inputs returns inf when no weight is positive."""
        result = klems_model._ces_output_constraint(
            np.array([0.0]),
            np.array([0.0]),
            sigma=1.0,
        )
        assert result == np.inf


# Tests for KLEMS initialisation
class TestKlemsInitialisation:
    """Tests for the sector masks and weights set by _init_klems_weights."""

    def test_model_constructs_with_klems_masks(self, klems_model: InputOutputModel) -> None:
        """Checks that the model constructs without error and populates klems_masks."""
        assert klems_model.klems_masks is not None
        assert klems_model.klems_masks["energy"][0]
        assert not klems_model.klems_masks["materials"][0]
        assert not klems_model.klems_masks["other"][0]

        assert not klems_model.klems_masks["energy"][1]
        assert klems_model.klems_masks["materials"][1]
        assert not klems_model.klems_masks["other"][1]

        assert not klems_model.klems_masks["energy"][2]
        assert not klems_model.klems_masks["materials"][2]
        assert klems_model.klems_masks["other"][2]

    def test_energy_input_presence_flags(self, klems_model: InputOutputModel) -> None:
        """Checks that _klems_has_e reflects which sectors receive electricity as an input."""
        assert not klems_model._klems_has_e[0]
        assert klems_model._klems_has_e[1]
        assert klems_model._klems_has_e[2]

    def test_material_input_presence_flags(self, klems_model: InputOutputModel) -> None:
        """Checks that _klems_has_m reflects which sectors receive steel as an input."""
        assert klems_model._klems_has_m[0]
        assert not klems_model._klems_has_m[1]
        assert klems_model._klems_has_m[2]

    def test_kl_weights_sum_to_one(self, klems_model: InputOutputModel) -> None:
        """Labour and capital weights in the KL sub-aggregate sum to one per sector."""
        np.testing.assert_allclose(
            klems_model.klems_w_L + klems_model.klems_w_K,
            np.ones(klems_model.N),
            rtol=1e-10,
        )


# Tests for the KLEMS producing_x branch
class TestKlemsProducingX:
    """Tests for producing_x called with prod_f='klems'."""

    def test_base_year_full_inventories_output_equals_demand(
        self, klems_model: InputOutputModel
    ) -> None:
        """Output equals demand when inventories are at base level and demand equals x0."""
        m = klems_model
        S = _base_inventories_klems(m)
        l_ = _base_l_array(m)
        result = m.producing_x("klems", None, m.x0, l_, S, m.A, m.x0, t=1)
        np.testing.assert_allclose(result["output"], m.x0, rtol=1e-6)

    def test_return_dict_has_required_keys(self, klems_model: InputOutputModel) -> None:
        """Asserts that producing_x returns a dict with the three expected keys."""
        m = klems_model
        S = _base_inventories_klems(m)
        l_ = _base_l_array(m)
        result = m.producing_x("klems", None, m.x0, l_, S, m.A, m.x0, t=1)
        assert "output" in result
        assert "output.constraints" in result
        assert "import_supplement_matrix" in result

    def test_output_is_non_negative(self, klems_model: InputOutputModel) -> None:
        """Checks that KLEMS output is non-negative at base-year conditions."""
        m = klems_model
        S = _base_inventories_klems(m)
        l_ = _base_l_array(m)
        result = m.producing_x("klems", None, m.x0, l_, S, m.A, m.x0, t=1)
        assert np.all(result["output"] >= 0)

    def test_tight_energy_reduces_steel_output(self, klems_model: InputOutputModel) -> None:
        """Cutting the electricity inventory lowers steel output but not electricity output."""
        m = klems_model
        S = _base_inventories_klems(m)
        l_ = _base_l_array(m)
        high_demand = 3.0 * m.x0

        S_tight = S.copy()
        S_tight[0, :] *= 0.4

        result_tight = m.producing_x("klems", None, m.x0, l_, S_tight, m.A, high_demand, t=1)
        result_full  = m.producing_x("klems", None, m.x0, l_, S,       m.A, high_demand, t=1)

        assert result_tight["output"][1] < result_full["output"][1]
        assert np.isclose(result_tight["output"][0], result_full["output"][0], rtol=1e-10)

    def test_supplement_charged_when_kle_capacity_binds(self, klems_data_dict: dict) -> None:
        """A supplement raising output through the KLE nest survives the benefit test."""
        config = ModelConfig(
            n_periods=10, time_frequency="quarterly", prod_function="klems",
        )
        m = InputOutputModel(
            n_periods=10, time_frequency="quarterly",
            config=config, _data_dict=klems_data_dict,
        )
        m.import_flexibility_vector = np.zeros(m.N)
        m.import_flexibility_vector[0] = 1.0
        S_tar = _base_inventories_klems(m)
        l_ = _base_l_array(m)
        S_tight = S_tar.copy()
        S_tight[0, :] *= 0.4
        high_demand = 3.0 * m.x0

        with_flex = m.producing_x(
            "klems", None, m.x0, l_, S_tight, m.A, high_demand, t=1, S_tar=S_tar
        )
        m.import_flexibility_vector[0] = 0.0
        no_flex = m.producing_x(
            "klems", None, m.x0, l_, S_tight, m.A, high_demand, t=1, S_tar=S_tar
        )

        assert with_flex["output"][1] > no_flex["output"][1]
        assert with_flex["import_supplement_matrix"][0, 1] > 0

    def test_output_constraints_shape(self, klems_model: InputOutputModel) -> None:
        """Asserts that output.constraints has shape (N, 4)."""
        m = klems_model
        S = _base_inventories_klems(m)
        l_ = _base_l_array(m)
        result = m.producing_x("klems", None, m.x0, l_, S, m.A, m.x0, t=1)
        assert result["output.constraints"].shape == (m.N, 4)


# Tests for KLEMS skill-tier weights
class TestKlemsSkillTiers:
    """Tests for the klems_w_skill attribute."""

    def test_klems_w_skill_shape(self, klems_model: InputOutputModel) -> None:
        """Checks that klems_w_skill has shape (3, N)."""
        m = klems_model
        assert m.klems_w_skill.shape == (3, m.N)

    def test_klems_w_skill_column_sums_to_one(self, klems_model: InputOutputModel) -> None:
        """Verifies that each column of klems_w_skill sums to one."""
        np.testing.assert_allclose(
            klems_model.klems_w_skill.sum(axis=0),
            np.ones(klems_model.N),
            rtol=1e-10,
        )

    def test_klems_w_skill_matches_l0_by_skill_ratio(self, klems_model: InputOutputModel) -> None:
        """Asserts that klems_w_skill equals l0_by_skill divided by l0 element-wise."""
        m = klems_model
        expected = m.l0_by_skill / np.where(m.l0 > 0, m.l0, 1.0)[np.newaxis, :]
        np.testing.assert_allclose(m.klems_w_skill, expected, rtol=1e-10)

    def test_klems_missing_l0_by_skill_raises(self, klems_data_dict: dict) -> None:
        """Constructing a KLEMS model without l0_by_skill raises ValueError."""
        data_without_skill = {k: v for k, v in klems_data_dict.items() if k != "l0_by_skill"}
        config = ModelConfig(
            n_periods=10,
            time_frequency="quarterly",
            prod_function="klems",
        )
        with pytest.raises(ValueError, match="l0_by_skill"):
            InputOutputModel(
                n_periods=10,
                time_frequency="quarterly",
                config=config,
                _data_dict=data_without_skill,
            )


class TestKlemsMScope:
    def test_folded_masks(self, klems_data_dict: dict) -> None:
        """klems_m_scope='intermediates' folds every non-energy input into M."""
        config = ModelConfig(n_periods=10, time_frequency="quarterly",
                             prod_function="klems",
                             klems_m_scope="intermediates")
        m = InputOutputModel(n_periods=10, time_frequency="quarterly",
                             config=config, _data_dict=klems_data_dict)
        assert m.klems_masks["energy"].tolist() == [True, False, False]
        assert m.klems_masks["materials"].tolist() == [False, True, True]
        assert not m.klems_masks["other"].any()

    def test_folded_model_runs(self, klems_data_dict: dict) -> None:
        config = ModelConfig(n_periods=6, time_frequency="quarterly",
                             prod_function="klems",
                             klems_m_scope="intermediates")
        m = InputOutputModel(n_periods=6, time_frequency="quarterly",
                             config=config, _data_dict=klems_data_dict)
        res = m.run_model()
        assert np.all(np.isfinite(res["gross_output"]))

    def test_invalid_scope_raises(self) -> None:
        with pytest.raises(ValueError, match="klems_m_scope"):
            ModelConfig(n_periods=5, klems_m_scope="everything")
