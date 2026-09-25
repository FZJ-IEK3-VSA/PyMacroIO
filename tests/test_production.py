"""Tests for the producing_x and hire_fire methods of InputOutputModel.

Fixtures shared across the suite are defined in conftest.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyMacroIO.model import InputOutputModel
from pyMacroIO.config import ModelConfig
from pyMacroIO.constants import (
    CAPACITY_MIN_SCALE,
    CAPACITY_MAX_SCALE,
    FIRING_SPEED_DAMPING,
)


# Fixtures
@pytest.fixture(scope="module")
def model(minimal_data_dict) -> InputOutputModel:
    """Return a 3-sector InputOutputModel with hiringfiring enabled (default)."""
    config = ModelConfig(n_periods=10, time_frequency="quarterly")
    return InputOutputModel(
        n_periods=10,
        time_frequency="quarterly",
        config=config,
        _data_dict=minimal_data_dict,
    )


@pytest.fixture(scope="module")
def model_hiring(minimal_data_dict) -> InputOutputModel:
    """Return a model with hiringfiring explicitly enabled."""
    config = ModelConfig(
        n_periods=10, time_frequency="quarterly", hiringfiring=True
    )
    return InputOutputModel(
        n_periods=10,
        time_frequency="quarterly",
        config=config,
        _data_dict=minimal_data_dict,
    )


@pytest.fixture(scope="module")
def model_no_hiring(minimal_data_dict) -> InputOutputModel:
    """Return a model with hiringfiring disabled."""
    config = ModelConfig(
        n_periods=10, time_frequency="quarterly", hiringfiring=False
    )
    return InputOutputModel(
        n_periods=10,
        time_frequency="quarterly",
        config=config,
        _data_dict=minimal_data_dict,
    )


@pytest.fixture(scope="module")
def model_wage_curve(minimal_data_dict) -> InputOutputModel:
    """Return a model with the wage curve active (beta=0.1 per sector)."""
    config = ModelConfig(
        n_periods=10,
        time_frequency="quarterly",
        hiringfiring=True,
        wage_curve=True,
        wage_curve_beta=0.1,
    )
    return InputOutputModel(
        n_periods=10,
        time_frequency="quarterly",
        config=config,
        _data_dict=minimal_data_dict,
    )


# Shared helpers
def _base_inventories(m: InputOutputModel) -> np.ndarray:
    """Return the base-year inventory matrix S = A * x0 * n (purchaser coverage)."""
    return m.A * m.x0[np.newaxis, :] * m.n[np.newaxis, :]


def _base_labour(m: InputOutputModel) -> np.ndarray:
    """Return an (N, TT) labour array with every column equal to l0."""
    return np.tile(m.l0[:, np.newaxis], (1, m.TT))


def _base_prod_constraints(m: InputOutputModel, scale: float = 1.0) -> np.ndarray:
    """Return prod_constraints (N, 4) with columns 1-3 set to scale * x0."""
    pc = np.zeros((m.N, 4))
    pc[:, 1:4] = (scale * m.x0)[:, np.newaxis]
    return pc


# Tests for producing_x
class TestProducingX:
    """Tests for the producing_x method across Leontief, linear, CES, and adapted branches."""

    # Leontief branch

    def test_leontief_base_year_output_equals_demand(self, model: InputOutputModel) -> None:
        """With full inventories and matching capacity, Leontief output equals demand."""
        m = model
        xcap0 = m.x0.copy()
        l_ = _base_labour(m)
        S = _base_inventories(m)
        d = m.x0.copy()
        result = m.producing_x("leontief", None, xcap0, l_, S, m.A, d, t=1)
        np.testing.assert_allclose(result["output"], m.x0, rtol=1e-10)

    def test_leontief_zero_inventory_zeroes_sector_output(self, model: InputOutputModel) -> None:
        """Setting a single inventory cell to zero forces the receiving sector's output to zero."""
        m = model
        xcap0 = m.x0.copy()
        l_ = _base_labour(m)
        S = _base_inventories(m)
        S[0, 1] = 0.0  # deplete sector A's stock held by sector B
        d = m.x0.copy()
        result = m.producing_x("leontief", None, xcap0, l_, S, m.A, d, t=1)
        assert result["output"][1] == pytest.approx(0.0, abs=1e-10)

    def test_leontief_output_non_negative(self, model: InputOutputModel) -> None:
        """Leontief output is never negative regardless of inventory levels."""
        m = model
        xcap0 = m.x0.copy()
        l_ = _base_labour(m)
        S = np.zeros_like(m.A)  # no inventories
        d = m.x0.copy()
        result = m.producing_x("leontief", None, xcap0, l_, S, m.A, d, t=1)
        assert np.all(result["output"] >= 0.0)

    def test_leontief_output_does_not_exceed_demand(self, model: InputOutputModel) -> None:
        """Leontief output never exceeds sector demand."""
        m = model
        xcap0 = m.x0.copy()
        l_ = _base_labour(m)
        S = _base_inventories(m)
        d = m.x0.copy()
        result = m.producing_x("leontief", None, xcap0, l_, S, m.A, d, t=1)
        assert np.all(result["output"] <= d + 1e-10)

    def test_leontief_output_does_not_exceed_capacity(self, model: InputOutputModel) -> None:
        """Leontief output never exceeds labour capacity."""
        m = model
        xcap0 = m.x0.copy()
        l_ = _base_labour(m)
        S = _base_inventories(m)
        d = m.x0.copy()
        result = m.producing_x("leontief", None, xcap0, l_, S, m.A, d, t=1)
        xcap_col = result["output.constraints"][:, 0]
        assert np.all(result["output"] <= xcap_col + 1e-10)

    # Linear branch

    def test_linear_exceeds_leontief_when_input_scarce(self, model: InputOutputModel) -> None:
        """Linear output for a sector with one scarce input exceeds Leontief output."""
        m = model
        xcap0 = m.x0.copy()
        l_ = _base_labour(m)
        S = _base_inventories(m)
        S[0, 1] = _base_inventories(m)[0, 1] * 0.10  # deplete row 0 col 1 to 10 %
        d = m.x0.copy()
        res_leontief = m.producing_x("leontief", None, xcap0, l_, S, m.A, d, t=1)
        res_linear = m.producing_x("linear", None, xcap0, l_, S, m.A, d, t=1)
        assert res_linear["output"][1] > res_leontief["output"][1]

    def test_linear_output_does_not_exceed_demand(self, model: InputOutputModel) -> None:
        """Linear output never exceeds sector demand."""
        m = model
        xcap0 = m.x0.copy()
        l_ = _base_labour(m)
        S = _base_inventories(m)
        S[0, 1] = _base_inventories(m)[0, 1] * 0.10
        d = m.x0.copy()
        result = m.producing_x("linear", None, xcap0, l_, S, m.A, d, t=1)
        assert np.all(result["output"] <= d + 1e-10)

    def test_linear_output_does_not_exceed_capacity(self, model: InputOutputModel) -> None:
        """Linear output never exceeds labour capacity."""
        m = model
        xcap0 = m.x0.copy()
        l_ = _base_labour(m)
        S = _base_inventories(m)
        S[0, 1] = _base_inventories(m)[0, 1] * 0.10
        d = m.x0.copy()
        result = m.producing_x("linear", None, xcap0, l_, S, m.A, d, t=1)
        xcap_col = result["output.constraints"][:, 0]
        assert np.all(result["output"] <= xcap_col + 1e-10)

    # CES branch

    def test_ces_output_above_leontief_below_linear_when_scarce(
        self, model: InputOutputModel
    ) -> None:
        """CES output with sigma > 1 lies between Leontief and linear when one input is scarce."""
        m = model
        xcap0 = m.x0.copy()
        l_ = _base_labour(m)
        S = _base_inventories(m)
        S[0, 1] = _base_inventories(m)[0, 1] * 0.10
        d = m.x0.copy()
        res_leontief = m.producing_x("leontief", None, xcap0, l_, S, m.A, d, t=1)
        res_linear = m.producing_x("linear", None, xcap0, l_, S, m.A, d, t=1)
        res_ces = m.producing_x("ces", None, xcap0, l_, S, m.A, d, t=1)
        assert res_ces["output"][1] > res_leontief["output"][1]
        assert res_ces["output"][1] <= res_linear["output"][1] + 1e-10

    def test_ces_output_non_negative(self, model: InputOutputModel) -> None:
        """CES output is never negative under any inventory level."""
        m = model
        xcap0 = m.x0.copy()
        l_ = _base_labour(m)
        S = np.zeros_like(m.A)
        d = m.x0.copy()
        result = m.producing_x("ces", None, xcap0, l_, S, m.A, d, t=1)
        assert np.all(result["output"] >= 0.0)

    def test_ces_output_does_not_exceed_demand(self, model: InputOutputModel) -> None:
        """CES output never exceeds sector demand."""
        m = model
        xcap0 = m.x0.copy()
        l_ = _base_labour(m)
        S = _base_inventories(m)
        S[0, 1] = _base_inventories(m)[0, 1] * 0.10
        d = m.x0.copy()
        result = m.producing_x("ces", None, xcap0, l_, S, m.A, d, t=1)
        assert np.all(result["output"] <= d + 1e-10)

    def test_ces_output_does_not_exceed_capacity(self, model: InputOutputModel) -> None:
        """CES output never exceeds labour capacity."""
        m = model
        xcap0 = m.x0.copy()
        l_ = _base_labour(m)
        S = _base_inventories(m)
        S[0, 1] = _base_inventories(m)[0, 1] * 0.10
        d = m.x0.copy()
        result = m.producing_x("ces", None, xcap0, l_, S, m.A, d, t=1)
        xcap_col = result["output.constraints"][:, 0]
        assert np.all(result["output"] <= xcap_col + 1e-10)

    # Adapted Leontief branch

    def test_adapted_leontief_all_essential_matches_plain(
        self, model: InputOutputModel
    ) -> None:
        """Adapted Leontief with all inputs essential produces the same output as plain Leontief."""
        m = model
        xcap0 = m.x0.copy()
        l_ = _base_labour(m)
        S = _base_inventories(m)
        S[0, 1] = _base_inventories(m)[0, 1] * 0.10
        d = m.x0.copy()
        A_ess_ones = np.ones((m.N, m.N))
        res_plain = m.producing_x("leontief", None, xcap0, l_, S, m.A, d, t=1)
        res_adapted = m.producing_x(
            "leontief.adapted", A_ess_ones, xcap0, l_, S, m.A, d, t=1
        )
        np.testing.assert_allclose(res_adapted["output"], res_plain["output"], rtol=1e-10)

    def test_adapted_leontief_all_nonessential_exceeds_plain(
        self, model: InputOutputModel
    ) -> None:
        """With no essential inputs, adapted exceeds plain Leontief when an input is scarce."""
        m = model
        xcap0 = m.x0.copy()
        l_ = _base_labour(m)
        S = _base_inventories(m)
        S[0, 1] = _base_inventories(m)[0, 1] * 0.10
        d = m.x0.copy()
        A_ess_zeros = np.zeros((m.N, m.N))
        res_plain = m.producing_x("leontief", None, xcap0, l_, S, m.A, d, t=1)
        res_adapted = m.producing_x(
            "leontief.adapted", A_ess_zeros, xcap0, l_, S, m.A, d, t=1
        )
        assert res_adapted["output"][1] > res_plain["output"][1]

    def test_adapted_leontief_mixed_essential_strict_improvement(
        self, model: InputOutputModel
    ) -> None:
        """Mixed A_essential exceeds plain Leontief where the scarce input is non-essential."""
        m = model
        xcap0 = m.x0.copy()
        l_ = _base_labour(m)
        S = _base_inventories(m)
        # Deplete sector A's output (row 0) to 1 % of base-year stocks.
        S[0, :] *= 0.01
        d = m.x0.copy()
        # Row 0 (A) non-essential; row 1 (B) essential; row 2 (C) non-essential.
        A_ess_mixed = np.zeros((m.N, m.N))
        A_ess_mixed[1, :] = 1.0  # B is essential to all sectors

        res_plain = m.producing_x("leontief", None, xcap0, l_, S, m.A, d, t=1)
        res_adapted = m.producing_x(
            "leontief.adapted", A_ess_mixed, xcap0, l_, S, m.A, d, t=1
        )
        # Sector 1 (B): scarce input A is non-essential; adapted should strictly exceed plain.
        assert res_adapted["output"][1] > res_plain["output"][1] + 1e-10

    def test_empty_essential_matrix_raises(self, model: InputOutputModel) -> None:
        """An empty essential estimate raises instead of degenerating to the non-essential nest."""
        with pytest.raises(ValueError, match="no input was classified as essential"):
            model._estimate_essential_inputs(np.diag(np.full(model.N, 0.1)))

    def test_broken_io_identity_reports_the_identity_not_the_essential_set(
        self, minimal_data_dict
    ) -> None:
        """Malformed data names the identity that failed, whichever production function is set."""
        broken = dict(minimal_data_dict, Z0=minimal_data_dict["Z0"] * 1e-4)
        with pytest.raises(ValueError, match="Value added inconsistency"):
            InputOutputModel(
                n_periods=4,
                time_frequency="quarterly",
                config=ModelConfig(n_periods=4, time_frequency="quarterly"),
                _data_dict=broken,
            )

    def test_zero_output_sector_does_not_empty_the_essential_matrix(
        self, inactive_sector_data_dict
    ) -> None:
        """An inactive sector still leaves essential inputs classified among the active ones."""
        m = InputOutputModel(
            n_periods=4,
            time_frequency="quarterly",
            config=ModelConfig(n_periods=4, time_frequency="quarterly"),
            _data_dict=inactive_sector_data_dict,
        )
        assert m.x0[3] == 0.0
        assert m.A_essential.sum() > 0
        assert m.A_essential[3, :].sum() == 0 and m.A_essential[:, 3].sum() == 0

    def test_production_function_switched_after_construction_gets_an_estimate(
        self, minimal_data_dict
    ) -> None:
        """Switching to leontief.adapted post-construction still estimates A_essential."""
        m = InputOutputModel(
            n_periods=4,
            time_frequency="quarterly",
            config=ModelConfig(
                n_periods=4, time_frequency="quarterly", prod_function="leontief"
            ),
            _data_dict=minimal_data_dict,
        )
        assert m.A_essential is None
        m.prod_function = "leontief.adapted"
        m.run_model()
        assert m.A_essential is not None and m.A_essential.sum() > 0

    # Return-dict structure

    def test_producing_x_returns_expected_keys(self, model: InputOutputModel) -> None:
        """producing_x returns a dict with the three expected keys."""
        m = model
        xcap0 = m.x0.copy()
        l_ = _base_labour(m)
        S = _base_inventories(m)
        d = m.x0.copy()
        result = m.producing_x("leontief", None, xcap0, l_, S, m.A, d, t=1)
        assert set(result.keys()) == {"output", "output.constraints", "import_supplement_matrix"}

    def test_output_constraints_has_four_columns(self, model: InputOutputModel) -> None:
        """output.constraints is (N, 4) with columns xcap, xinp, d, output_constraint."""
        m = model
        xcap0 = m.x0.copy()
        l_ = _base_labour(m)
        S = _base_inventories(m)
        d = m.x0.copy()
        result = m.producing_x("leontief", None, xcap0, l_, S, m.A, d, t=1)
        assert result["output.constraints"].shape == (m.N, 4)

    # fprod_l scaling

    def test_fprod_l_doubling_doubles_labour_bound_output(
        self, model: InputOutputModel
    ) -> None:
        """Doubling fprod_l doubles output when the labour capacity bound is binding."""
        m = model
        xcap0 = m.x0.copy()
        l_ = _base_labour(m)
        S = _base_inventories(m) * 100.0
        d = m.x0 * 10.0
        fprod_l_base = np.ones(m.N)
        fprod_l_double = np.full(m.N, 2.0)
        res_base = m.producing_x(
            "leontief", None, xcap0, l_, S, m.A, d, t=1, fprod_l=fprod_l_base
        )
        res_double = m.producing_x(
            "leontief", None, xcap0, l_, S, m.A, d, t=1, fprod_l=fprod_l_double
        )
        np.testing.assert_allclose(
            res_double["output"], 2.0 * res_base["output"], rtol=1e-10
        )


# Tests for hire_fire
class TestHireFire:
    """Tests for the hire_fire labour-adjustment method."""

    def _build_arrays(
        self, m: InputOutputModel
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (l_, x_, delta_) initialised to base-year values."""
        N, TT = m.N, m.TT
        l_ = np.zeros((N, TT))
        l_[:, 0] = m.l0
        x_ = np.zeros((N, TT))
        x_[:, 0] = m.x0
        delta_ = np.zeros((N, TT))
        return l_, x_, delta_

    def test_no_hiringfiring_returns_previous_labour(
        self, model_no_hiring: InputOutputModel
    ) -> None:
        """When hiringfiring is False, hire_fire returns l_[:, t-1] without modification."""
        m = model_no_hiring
        assert not m.hiringfiring
        l_, x_, delta_ = self._build_arrays(m)
        pc = _base_prod_constraints(m, scale=1.0)
        result = m.hire_fire(t=1, l_=l_, x_=x_, delta_=delta_, prod_constraints=pc)
        np.testing.assert_allclose(result, l_[:, 0])

    def test_excess_demand_increases_labour(
        self, model_hiring: InputOutputModel
    ) -> None:
        """Desired output above capacity raises labour by gamma_hire times the gap."""
        m = model_hiring
        l_, x_, delta_ = self._build_arrays(m)
        pc = _base_prod_constraints(m, scale=2.0)
        result = m.hire_fire(t=1, l_=l_, x_=x_, delta_=delta_, prod_constraints=pc)
        expected_gap = m.l0 * (CAPACITY_MAX_SCALE - 1.0)
        expected = np.clip(
            m.l0 + m.gamma_hire * expected_gap,
            m.l0 * CAPACITY_MIN_SCALE,
            m.l0 * CAPACITY_MAX_SCALE,
        )
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_insufficient_demand_reduces_labour(
        self, model_hiring: InputOutputModel
    ) -> None:
        """Desired output below capacity sheds labour at gamma_fire * FIRING_SPEED_DAMPING."""
        m = model_hiring
        l_, x_, delta_ = self._build_arrays(m)
        pc = _base_prod_constraints(m, scale=0.1)
        result = m.hire_fire(t=1, l_=l_, x_=x_, delta_=delta_, prod_constraints=pc)
        min_cap = m.l0 * CAPACITY_MIN_SCALE
        max_cap = m.l0 * CAPACITY_MAX_SCALE
        desired_l = np.clip(0.1 * m.l0, min_cap, max_cap)
        gap = desired_l - m.l0
        gam = m.gamma_fire * FIRING_SPEED_DAMPING
        expected = np.clip(m.l0 + gam * gap, min_cap, max_cap)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_result_never_below_min_capacity(
        self, model_hiring: InputOutputModel
    ) -> None:
        """hire_fire output is never below CAPACITY_MIN_SCALE times initial labour."""
        m = model_hiring
        l_, x_, delta_ = self._build_arrays(m)
        pc = _base_prod_constraints(m, scale=0.0)  # zero demand -> maximum firing pressure
        result = m.hire_fire(t=1, l_=l_, x_=x_, delta_=delta_, prod_constraints=pc)
        assert np.all(result >= m.l0 * CAPACITY_MIN_SCALE - 1e-10)

    def test_result_never_above_max_capacity(
        self, model_hiring: InputOutputModel
    ) -> None:
        """With no reference path, hire_fire never exceeds CAPACITY_MAX_SCALE times l0."""
        m = model_hiring
        l_, x_, delta_ = self._build_arrays(m)
        pc = _base_prod_constraints(m, scale=10.0)  # extreme demand -> maximum hiring pressure
        result = m.hire_fire(t=1, l_=l_, x_=x_, delta_=delta_, prod_constraints=pc)
        assert np.all(result <= m.l0 * CAPACITY_MAX_SCALE + 1e-10)

    def test_balanced_demand_leaves_labour_unchanged(
        self, model_hiring: InputOutputModel
    ) -> None:
        """When desired labour equals current labour, hire_fire returns the same labour level."""
        m = model_hiring
        l_, x_, delta_ = self._build_arrays(m)
        pc = _base_prod_constraints(m, scale=1.0)
        result = m.hire_fire(t=1, l_=l_, x_=x_, delta_=delta_, prod_constraints=pc)
        np.testing.assert_allclose(result, m.l0, rtol=1e-10)


# Tests for hire_fire wage curve
class TestHireFireWageCurve:
    """Tests for the Blanchflower-Oswald wage-curve adjustment inside hire_fire."""

    def _build_arrays(
        self, m: InputOutputModel
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (l_, x_, delta_) initialised to base-year values."""
        N, TT = m.N, m.TT
        l_ = np.zeros((N, TT))
        l_[:, 0] = m.l0
        x_ = np.zeros((N, TT))
        x_[:, 0] = m.x0
        delta_ = np.zeros((N, TT))
        return l_, x_, delta_

    def test_high_unemployment_lowers_desired_labour(
        self, model_wage_curve: InputOutputModel
    ) -> None:
        """Unemployment above the base rate lowers hire_fire labour below the unadjusted level."""
        m = model_wage_curve
        l_, x_, delta_ = self._build_arrays(m)
        pc = _base_prod_constraints(m, scale=1.0)
        r_no_adj = m.hire_fire(t=1, l_=l_, x_=x_, delta_=delta_, prod_constraints=pc)
        U_r_prev = np.full(m.N, 0.10)
        U_r_0 = np.full(m.N, 0.05)
        r_high_u = m.hire_fire(
            t=1, l_=l_, x_=x_, delta_=delta_, prod_constraints=pc,
            U_r_prev=U_r_prev, U_r_0=U_r_0,
        )
        assert np.all(r_high_u <= r_no_adj)

    def test_low_unemployment_raises_desired_labour(
        self, model_wage_curve: InputOutputModel
    ) -> None:
        """Unemployment below the base rate raises hire_fire labour above the unadjusted level."""
        m = model_wage_curve
        l_, x_, delta_ = self._build_arrays(m)
        pc = _base_prod_constraints(m, scale=1.0)
        r_no_adj = m.hire_fire(t=1, l_=l_, x_=x_, delta_=delta_, prod_constraints=pc)
        # Low unemployment: U_prev < U_0 -> w_adj > 1 -> labour share rises.
        U_r_prev = np.full(m.N, 0.02)
        U_r_0 = np.full(m.N, 0.05)
        r_low_u = m.hire_fire(
            t=1, l_=l_, x_=x_, delta_=delta_, prod_constraints=pc,
            U_r_prev=U_r_prev, U_r_0=U_r_0,
        )
        assert np.all(r_low_u >= r_no_adj)

    def test_high_and_low_unemployment_are_ordered(
        self, model_wage_curve: InputOutputModel
    ) -> None:
        """Labour from high-unemployment call is strictly below that from low-unemployment call."""
        m = model_wage_curve
        l_, x_, delta_ = self._build_arrays(m)
        pc = _base_prod_constraints(m, scale=1.0)
        U_r_0 = np.full(m.N, 0.05)
        r_high = m.hire_fire(
            t=1, l_=l_, x_=x_, delta_=delta_, prod_constraints=pc,
            U_r_prev=np.full(m.N, 0.10), U_r_0=U_r_0,
        )
        r_low = m.hire_fire(
            t=1, l_=l_, x_=x_, delta_=delta_, prod_constraints=pc,
            U_r_prev=np.full(m.N, 0.02), U_r_0=U_r_0,
        )
        assert np.all(r_high < r_low)
