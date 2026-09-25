"""Tests for the benefits, wage_curve, wage_floor_ratio, and hiringfiring parameters.

Fixtures shared across the suite are defined in conftest.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyMacroIO.model import InputOutputModel
from pyMacroIO.config import ModelConfig


# Fixtures

@pytest.fixture(scope="module")
def model_benefits_zero(minimal_data_dict) -> InputOutputModel:
    """Return a 3-sector model with benefits=0 (no baseline blending)."""
    config = ModelConfig(n_periods=10, time_frequency="quarterly", benefits=0.0)
    return InputOutputModel(
        n_periods=10,
        time_frequency="quarterly",
        config=config,
        _data_dict=minimal_data_dict,
    )


@pytest.fixture(scope="module")
def model_benefits_one(minimal_data_dict) -> InputOutputModel:
    """Return a 3-sector model with benefits=1 (full baseline blending)."""
    config = ModelConfig(n_periods=10, time_frequency="quarterly", benefits=1.0)
    return InputOutputModel(
        n_periods=10,
        time_frequency="quarterly",
        config=config,
        _data_dict=minimal_data_dict,
    )


@pytest.fixture(scope="module")
def model_benefits_half(minimal_data_dict) -> InputOutputModel:
    """Return a 3-sector model with benefits=0.5 (symmetric blending)."""
    config = ModelConfig(n_periods=10, time_frequency="quarterly", benefits=0.5)
    return InputOutputModel(
        n_periods=10,
        time_frequency="quarterly",
        config=config,
        _data_dict=minimal_data_dict,
    )


@pytest.fixture(scope="module")
def model_wage_curve_full(minimal_data_dict) -> InputOutputModel:
    """Return a model with wage_curve=True and beta=0.1 for full-run tests."""
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


@pytest.fixture(scope="module")
def model_no_wage_curve(minimal_data_dict) -> InputOutputModel:
    """Return a model with wage_curve=False for comparison in wage-curve tests."""
    config = ModelConfig(
        n_periods=10,
        time_frequency="quarterly",
        hiringfiring=True,
        wage_curve=False,
    )
    return InputOutputModel(
        n_periods=10,
        time_frequency="quarterly",
        config=config,
        _data_dict=minimal_data_dict,
    )


@pytest.fixture(scope="module")
def model_hiringfiring_false(minimal_data_dict) -> InputOutputModel:
    """Return a model with hiringfiring=False for full-run tests."""
    config = ModelConfig(
        n_periods=10,
        time_frequency="quarterly",
        hiringfiring=False,
    )
    return InputOutputModel(
        n_periods=10,
        time_frequency="quarterly",
        config=config,
        _data_dict=minimal_data_dict,
    )


@pytest.fixture(scope="module")
def model_hiringfiring_true(minimal_data_dict) -> InputOutputModel:
    """Return a model with hiringfiring=True for comparison in hiringfiring tests."""
    config = ModelConfig(
        n_periods=10,
        time_frequency="quarterly",
        hiringfiring=True,
    )
    return InputOutputModel(
        n_periods=10,
        time_frequency="quarterly",
        config=config,
        _data_dict=minimal_data_dict,
    )


# Shared helpers

def _base_prod_constraints(m: InputOutputModel, scale: float = 1.0) -> np.ndarray:
    """Return prod_constraints (N, 4) with columns 1-3 set to scale * x0."""
    pc = np.zeros((m.N, 4))
    pc[:, 1:4] = (scale * m.x0)[:, np.newaxis]
    return pc


# TestBenefitsParameter

class TestBenefitsParameter:
    """Tests for the benefits scalar and its effect on household income."""

    def test_benefits_zero_income_equals_actual_labour_income(
        self, model_benefits_zero: InputOutputModel
    ) -> None:
        """Verifies that _household_income returns actual labour income when benefits=0."""
        m = model_benefits_zero
        assert m.benefits == 0.0
        actual_labour = 25.0
        profits = 5.0
        result = m._household_income(actual_labour, profits)
        # benefits=0: adjusted = 0 * baseline + 1 * actual = actual
        np.testing.assert_allclose(result, actual_labour + profits, rtol=1e-10)

    def test_benefits_one_income_equals_baseline_labour_income(
        self, model_benefits_one: InputOutputModel
    ) -> None:
        """Verifies that _household_income returns baseline labour income when benefits=1."""
        m = model_benefits_one
        assert m.benefits == 1.0
        actual_labour = 5.0  # well below baseline
        profits = 3.0
        baseline_labour = float(np.sum(m.l0))
        result = m._household_income(actual_labour, profits)
        # benefits=1: adjusted = 1 * baseline + 0 * actual = baseline
        np.testing.assert_allclose(result, baseline_labour + profits, rtol=1e-10)

    def test_benefits_half_blends_symmetrically(
        self, model_benefits_half: InputOutputModel
    ) -> None:
        """_household_income returns the mean of baseline and actual when benefits=0.5."""
        m = model_benefits_half
        assert m.benefits == 0.5
        actual_labour = 10.0
        profits = 2.0
        baseline_labour = float(np.sum(m.l0))
        expected_labour = 0.5 * baseline_labour + 0.5 * actual_labour
        result = m._household_income(actual_labour, profits)
        np.testing.assert_allclose(result, expected_labour + profits, rtol=1e-10)

    def test_benefits_affects_run_model_consumption(self, minimal_data_dict) -> None:
        """Higher benefits yields higher consumption when output falls below baseline."""
        n_periods = 5

        def _make_model(benefits_val: float) -> InputOutputModel:
            # scarred closure makes income signal feed directly into consumption
            cfg = ModelConfig(
                n_periods=n_periods,
                time_frequency="quarterly",
                benefits=benefits_val,
                hiringfiring=True,
                household_closure_mode="scarred",
            )
            return InputOutputModel(
                n_periods=n_periods,
                time_frequency="quarterly",
                config=cfg,
                _data_dict=minimal_data_dict,
            )

        m_low = _make_model(0.0)
        m_high = _make_model(0.8)

        # Apply the same output shock to both models: sector A drops by 40 % at t=1 to t=4.
        for t in range(1, n_periods):
            m_low.apply_output_constraint_shock("A", t, 0.4)
            m_high.apply_output_constraint_shock("A", t, 0.4)

        res_low = m_low.run_model()
        res_high = m_high.run_model()

        # higher benefits should sustain higher total consumption in at least one shocked period
        cons_low = res_low["realised_consumption"].sum(axis=0)
        cons_high = res_high["realised_consumption"].sum(axis=0)
        assert any(cons_high[t] > cons_low[t] for t in range(1, n_periods)), (
            "Expected higher benefits to produce higher consumption in at least one shocked period"
        )


# TestWageCurveFullRun

class TestWageCurveFullRun:
    """Tests for wage_curve=True and wage_floor_ratio in a full run_model call."""

    def test_wage_curve_run_completes_without_error(
        self, model_wage_curve_full: InputOutputModel
    ) -> None:
        """Verifies that a model with wage_curve=True runs to completion without raising."""
        result = model_wage_curve_full.run_model()
        assert "labour_compensation" in result

    def test_wage_curve_higher_unemployment_lowers_wages(
        self, minimal_data_dict
    ) -> None:
        """The wage curve lowers labour relative to a no-curve model as unemployment rises."""
        n_periods = 8

        def _make(wage_curve: bool) -> InputOutputModel:
            cfg = ModelConfig(
                n_periods=n_periods,
                time_frequency="quarterly",
                hiringfiring=True,
                wage_curve=wage_curve,
                wage_curve_beta=0.1,
            )
            m = InputOutputModel(
                n_periods=n_periods,
                time_frequency="quarterly",
                config=cfg,
                _data_dict=minimal_data_dict,
            )
            return m

        m_curve = _make(True)
        m_flat = _make(False)

        # Apply a labour-reducing output shock to raise unemployment.
        for t in range(1, n_periods):
            m_curve.apply_output_constraint_shock("A", t, 0.5)
            m_flat.apply_output_constraint_shock("A", t, 0.5)

        # unemployment_schedule is required for the wage curve to activate
        N, TT = m_curve.N, m_curve.TT
        u_sched = np.zeros((N, TT))
        u_sched[:, 0] = 0.05
        for t in range(1, TT):
            u_sched[:, t] = 0.05 + 0.05 * t
        m_curve.unemployment_schedule = u_sched
        m_flat.unemployment_schedule = u_sched.copy()

        res_curve = m_curve.run_model()
        res_flat = m_flat.run_model()

        # The wage curve model should show lower labour in at least one period.
        lc_curve = res_curve["labour_compensation"].sum(axis=0)
        lc_flat = res_flat["labour_compensation"].sum(axis=0)
        assert any(
            lc_curve[t] < lc_flat[t] for t in range(1, n_periods)
        ), "Expected wage curve to lower labour compensation relative to no-curve model"

    def test_wage_floor_ratio_prevents_wage_falling_below_floor(
        self, minimal_data_dict
    ) -> None:
        """Wages never fall below wage_floor_ratio * w0 after a severe labour shock."""
        floor = 0.9

        cfg = ModelConfig(
            n_periods=10,
            time_frequency="quarterly",
            hiringfiring=True,
            wage_curve=True,
            wage_curve_beta=0.1,
            wage_floor_ratio=floor,
        )
        m = InputOutputModel(
            n_periods=10,
            time_frequency="quarterly",
            config=cfg,
            _data_dict=minimal_data_dict,
        )

        N, TT = m.N, m.TT
        l_ = np.zeros((N, TT)); l_[:, 0] = m.l0
        x_ = np.zeros((N, TT)); x_[:, 0] = m.x0
        delta_ = np.zeros((N, TT))
        pc = _base_prod_constraints(m, scale=1.0)

        # Very high unemployment to trigger the floor.
        U_r_prev = np.full(N, 0.80)
        U_r_0 = np.full(N, 0.05)

        result = m.hire_fire(
            t=1,
            l_=l_,
            x_=x_,
            delta_=delta_,
            prod_constraints=pc,
            U_r_prev=U_r_prev,
            U_r_0=U_r_0,
        )

        # Baseline desired labour ignoring the wage curve.
        pc_no_adj = _base_prod_constraints(m, scale=1.0)
        cfg_no_floor = ModelConfig(
            n_periods=10,
            time_frequency="quarterly",
            hiringfiring=True,
            wage_curve=False,
        )
        m_no_floor = InputOutputModel(
            n_periods=10,
            time_frequency="quarterly",
            config=cfg_no_floor,
            _data_dict=minimal_data_dict,
        )
        baseline = m_no_floor.hire_fire(
            t=1, l_=l_.copy(), x_=x_.copy(), delta_=delta_.copy(), prod_constraints=pc_no_adj
        )

        # wage-adjusted labour should be no less than floor * baseline
        assert np.all(result >= floor * baseline - 1e-10), (
            f"Wages fell below floor={floor}: result={result}, floor*baseline={floor * baseline}"
        )


# TestHiringfiringFalseFullRun

class TestHiringfiringFalseFullRun:
    """Tests for hiringfiring=False through a full run_model call."""

    def test_hiringfiring_false_run_completes(
        self, model_hiringfiring_false: InputOutputModel
    ) -> None:
        """Verifies that run_model returns a results dict without error when hiringfiring=False."""
        result = model_hiringfiring_false.run_model()
        assert isinstance(result, dict)
        assert "labour_compensation" in result

    def test_hiringfiring_false_labour_stays_constant(
        self, model_hiringfiring_false: InputOutputModel
    ) -> None:
        """The labour array stays at l0 in every period when hiringfiring=False."""
        m = model_hiringfiring_false
        result = m.run_model()
        l_ = result["labour_compensation"]  # shape (N, TT)
        for t in range(m.TT):
            np.testing.assert_allclose(
                l_[:, t], m.l0, rtol=1e-10,
                err_msg=f"Labour deviated from l0 at period t={t}",
            )

    def test_hiringfiring_false_vs_true_gdp_differs_under_shock(
        self, minimal_data_dict
    ) -> None:
        """GDP differs with and without hiringfiring under a positive demand shock."""
        n_periods = 8

        def _make(hiringfiring: bool) -> InputOutputModel:
            cfg = ModelConfig(
                n_periods=n_periods,
                time_frequency="quarterly",
                hiringfiring=hiringfiring,
            )
            return InputOutputModel(
                n_periods=n_periods,
                time_frequency="quarterly",
                config=cfg,
                _data_dict=minimal_data_dict,
            )

        m_true = _make(True)
        m_false = _make(False)

        # apply a consumption shock to lift household demand
        m_true.apply_consumption_shock(start=1, duration=n_periods - 1, intensity=-0.3)
        m_false.apply_consumption_shock(start=1, duration=n_periods - 1, intensity=-0.3)

        res_true = m_true.run_model()
        res_false = m_false.run_model()

        gdp_true = res_true["gdp"]
        gdp_false = res_false["gdp"]

        # GDP paths should differ because one model can adjust labour and the other cannot
        assert any(
            abs(gdp_true[t] - gdp_false[t]) > 1e-8 for t in range(1, n_periods)
        ), "Expected GDP to differ between hiringfiring=True and hiringfiring=False under a demand shock"


class TestFiringSpeedDamping:
    def _fire_step(self, data_dict, damping):
        config = ModelConfig(n_periods=5, time_frequency="quarterly",
                             firing_speed_damping=damping)
        m = InputOutputModel(n_periods=5, time_frequency="quarterly",
                             config=config, _data_dict=data_dict)
        N = m.N
        l_ = np.zeros((N, 3)); l_[:, 0] = m.l0; l_[:, 1] = m.l0
        x_ = np.zeros((N, 3)); x_[:, 0] = m.x0
        delta_ = np.zeros((N, 3))
        constraints = np.column_stack(
            [np.full(N, np.inf), np.zeros(N), np.zeros(N), np.full(N, np.inf)])
        return m, m.hire_fire(2, l_, x_, delta_, constraints)

    def test_damping_scales_shedding(self, minimal_data_dict) -> None:
        """Firing at damping 1.0 sheds twice the labour of the 0.5 default."""
        m_half, l_half = self._fire_step(minimal_data_dict, 0.5)
        m_full, l_full = self._fire_step(minimal_data_dict, 1.0)
        shed_half = m_half.l0 - l_half
        shed_full = m_full.l0 - l_full
        active = shed_half > 1e-12
        assert active.any()
        np.testing.assert_allclose(shed_full[active], 2.0 * shed_half[active],
                                   rtol=1e-10)

    def test_invalid_damping_raises(self) -> None:
        with pytest.raises(ValueError, match="firing_speed_damping"):
            ModelConfig(n_periods=5, firing_speed_damping=0.0)
