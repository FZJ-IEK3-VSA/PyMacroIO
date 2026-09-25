"""Tests for time-frequency parameters, accounting identities, and firm-priority allocation."""

from __future__ import annotations

import numpy as np
import pytest

from pyMacroIO.model import InputOutputModel
from pyMacroIO.config import ModelConfig


# Helpers

def _build_model(
    data_dict: dict,
    *,
    n_periods: int = 15,
    time_frequency: str = "quarterly",
    firm_priority: str = "no",
    household_closure_mode: str | list[str] = "return_to_base",
    n_regions: int = 1,
) -> InputOutputModel:
    """Return an InputOutputModel built from the supplied data dictionary and keyword options."""
    config = ModelConfig(
        n_periods=n_periods,
        time_frequency=time_frequency,
        firm_priority=firm_priority,
        household_closure_mode=household_closure_mode,
        n_regions=n_regions,
    )
    return InputOutputModel(
        n_periods=n_periods,
        time_frequency=time_frequency,
        config=config,
        _data_dict=data_dict,
    )


# Module-scoped fixtures

@pytest.fixture(scope="module")
def two_region_validated_result(two_region_data_dict: dict) -> dict:
    """Return the result dict from a validated two-region run used by TestValidateRunIdentities."""
    config = ModelConfig(
        n_periods=20,
        time_frequency="quarterly",
        n_regions=2,
        household_closure_mode=["return_to_base", "return_to_base"],
    )
    m = InputOutputModel(
        n_periods=20,
        time_frequency="quarterly",
        config=config,
        _data_dict=two_region_data_dict,
    )
    return m.run_model(validate=True)


@pytest.fixture(scope="module")
def frozen_region_result(two_region_data_dict: dict) -> tuple[dict, InputOutputModel]:
    """Return the result dict and model for a two-region run where region 1 is frozen."""
    config = ModelConfig(
        n_periods=20,
        time_frequency="quarterly",
        n_regions=2,
        household_closure_mode=["return_to_base", "frozen"],
    )
    m = InputOutputModel(
        n_periods=20,
        time_frequency="quarterly",
        config=config,
        _data_dict=two_region_data_dict,
    )
    # Apply the shock only to region 0 so the frozen region 1 is not directly perturbed.
    m.apply_consumption_shock(start=3, duration=5, intensity=0.4, region=0)
    result = m.run_model(validate=True)
    return result, m


# TestTimeFrequency

class TestTimeFrequency:
    """Tests for time_frequency and the consumption-persistence parameters rho0 and rho1."""

    def test_quarterly_rho0_less_than_one(self, two_region_data_dict: dict) -> None:
        """Verifies that a quarterly model has rho0 strictly less than one."""
        m = _build_model(two_region_data_dict, time_frequency="quarterly")
        assert m.rho0 < 1.0

    def test_daily_rho1_greater_than_quarterly(self, two_region_data_dict: dict) -> None:
        """A daily model has a higher rho1 (stronger persistence) than a quarterly model."""
        m_q = _build_model(two_region_data_dict, time_frequency="quarterly")
        m_d = _build_model(two_region_data_dict, time_frequency="daily")
        assert m_d.rho1 > m_q.rho1

    def test_time_frequency_quarterly_vs_daily_rho_differ(self, two_region_data_dict: dict) -> None:
        """Verifies that rho0 and rho1 differ between quarterly and daily models."""
        m_q = _build_model(two_region_data_dict, time_frequency="quarterly")
        m_d = _build_model(two_region_data_dict, time_frequency="daily")
        assert m_q.rho0 != m_d.rho0
        assert m_q.rho1 != m_d.rho1

    def test_quarterly_rho_exact_values(self, two_region_data_dict: dict) -> None:
        """A quarterly model sets rho0=0.4 and rho1=0.6 per _calculate_time_step_parameters."""
        m = _build_model(two_region_data_dict, time_frequency="quarterly")
        np.testing.assert_allclose(m.rho0, 0.4, rtol=1e-12)
        np.testing.assert_allclose(m.rho1, 0.6, rtol=1e-12)

    def test_daily_rho_exact_values(self, two_region_data_dict: dict) -> None:
        """Verifies that a daily model sets rho1 = 0.6**(1/90) (compounded) and rho0 = 1 - rho1."""
        rho_bar = 0.6
        dt = 1.0 / 90.0
        expected_rho1 = rho_bar ** (1.0 / 90.0)
        expected_rho0 = 1.0 - expected_rho1
        m = _build_model(two_region_data_dict, time_frequency="daily")
        np.testing.assert_allclose(m.rho1, expected_rho1, rtol=1e-12)
        np.testing.assert_allclose(m.rho0, expected_rho0, rtol=1e-12)

    def test_time_frequency_affects_consumption_dynamics(self, two_region_data_dict: dict) -> None:
        """Consumption differs across time frequencies under a scarred-closure supply shock."""
        # scarred closure is needed for rho1 to affect the consumption path
        config_q = ModelConfig(
            n_periods=30,
            time_frequency="quarterly",
            firm_priority="no",
            n_regions=2,
            household_closure_mode=["scarred", "scarred"],
        )
        m_q = InputOutputModel(
            n_periods=30,
            time_frequency="quarterly",
            config=config_q,
            _data_dict=two_region_data_dict,
        )
        config_d = ModelConfig(
            n_periods=30,
            time_frequency="daily",
            firm_priority="no",
            n_regions=2,
            household_closure_mode=["scarred", "scarred"],
        )
        m_d = InputOutputModel(
            n_periods=30,
            time_frequency="daily",
            config=config_d,
            _data_dict=two_region_data_dict,
        )
        for m in (m_q, m_d):
            for t in range(2, 7):
                m.apply_output_constraint_shock("A", time_period=t, reduction_pct=0.5)
        res_q = m_q.run_model()
        res_d = m_d.run_model()
        cons_q = res_q["realised_consumption"].sum(axis=0)
        cons_d = res_d["realised_consumption"].sum(axis=0)
        assert not np.allclose(cons_q, cons_d, rtol=1e-6), (
            "Quarterly and daily consumption paths should differ when the scarred persistence parameter differs."
        )


# TestValidateRunIdentities

class TestValidateRunIdentities:
    """Tests for the accounting identities enforced by _validate_run."""

    def test_regional_gdp_additivity(self, two_region_validated_result: dict) -> None:
        """Verifies that gdp_regional.sum(axis=0) equals gdp at every period within 1e-6."""
        result = two_region_validated_result
        np.testing.assert_allclose(
            result["gdp_regional"].sum(axis=0),
            result["gdp"],
            atol=1e-6,
        )

    def test_regional_consumption_additivity(self, two_region_validated_result: dict) -> None:
        """consumption_by_hh_region and realised_consumption agree in total within 1e-6."""
        result = two_region_validated_result
        np.testing.assert_allclose(
            result["consumption_by_hh_region"].sum(axis=0),
            result["realised_consumption"].sum(axis=0),
            atol=1e-6,
        )

    def test_savings_additivity(self, two_region_validated_result: dict) -> None:
        """Verifies that savings_regional.sum(axis=0) equals savings at every period within 1e-6."""
        result = two_region_validated_result
        np.testing.assert_allclose(
            result["savings_regional"].sum(axis=0),
            result["savings"],
            atol=1e-6,
        )

    def test_trade_balance_sums_to_zero(self, two_region_validated_result: dict) -> None:
        """Verifies that the trade balance sums to zero across regions at every period."""
        result = two_region_validated_result
        tb_sum = result["trade_balance"].sum(axis=0)
        total_flow = np.abs(result["trade_balance"]).sum(axis=0) + 1e-9
        relative_deviation = np.abs(tb_sum) / total_flow
        np.testing.assert_array_less(
            relative_deviation,
            np.full_like(relative_deviation, 1e-6),
        )

    def test_frozen_region_consumption_invariance(
        self,
        frozen_region_result: tuple[dict, InputOutputModel],
    ) -> None:
        """Region 1 (frozen) keeps consumption near its base-year total at every period."""
        result, m = frozen_region_result
        frozen_r = 1
        expected = float(m.base_consumption_total_r[frozen_r])
        actual = result["consumption_by_hh_region"][frozen_r, :]
        max_dev = float(np.max(np.abs(actual - expected)))
        assert max_dev < expected * 0.01 + 1.0, (
            f"Frozen region {frozen_r} consumption drifted by {max_dev:.4f} from base {expected:.4f}."
        )


# TestFirmPriorityFullRun

class TestFirmPriorityFullRun:
    """Tests for firm_priority='no' through a full run_model call."""

    def test_firm_priority_no_run_completes(self, two_region_data_dict: dict) -> None:
        """Verifies that run_model returns a non-empty results dict when firm_priority='no'."""
        m = _build_model(two_region_data_dict, firm_priority="no", n_periods=10)
        result = m.run_model()
        assert isinstance(result, dict)
        assert len(result) > 0
        assert "realised_consumption" in result

    def test_firm_priority_no_vs_yes_consumption_differs_under_supply_shock(
        self, two_region_data_dict: dict
    ) -> None:
        """firm_priority='no' and 'yes' yield different consumption under a supply shock."""
        m_no  = _build_model(two_region_data_dict, firm_priority="no",  n_periods=15)
        m_yes = _build_model(two_region_data_dict, firm_priority="yes", n_periods=15)
        # Apply a moderately severe output constraint to sector A at periods 3..7.
        for t in range(3, 8):
            for m in (m_no, m_yes):
                m.apply_output_constraint_shock("A", time_period=t, reduction_pct=0.5)
        res_no  = m_no.run_model()
        res_yes = m_yes.run_model()
        cons_no  = res_no["realised_consumption"].sum(axis=0)
        cons_yes = res_yes["realised_consumption"].sum(axis=0)
        # The two allocation rules must diverge in at least one period.
        assert not np.allclose(cons_no, cons_yes, rtol=1e-8), (
            "firm_priority='no' and 'yes' should produce different consumption paths under a supply shock."
        )

    def test_firm_priority_no_never_raises_consumption_above_unconstrained(
        self, two_region_data_dict: dict
    ) -> None:
        """firm_priority='no' never raises consumption above the unconstrained baseline."""
        m_baseline   = _build_model(two_region_data_dict, firm_priority="no", n_periods=15)
        m_constrained = _build_model(two_region_data_dict, firm_priority="no", n_periods=15)
        for t in range(3, 8):
            m_constrained.apply_output_constraint_shock("A", time_period=t, reduction_pct=0.4)
        res_base = m_baseline.run_model()
        res_cons = m_constrained.run_model()
        total_baseline    = res_base["realised_consumption"].sum(axis=0)
        total_constrained = res_cons["realised_consumption"].sum(axis=0)
        # Allow a tiny numerical tolerance of 1e-9.
        assert np.all(total_constrained <= total_baseline + 1e-9), (
            "firm_priority='no' should never produce more consumption than the unconstrained baseline."
        )


class TestBaselineTrend:
    """Verifies the baseline_trend_r multiplier on the household base anchors."""

    def test_neutral_overrides_reproduce_the_baseline_exactly(self, minimal_data_dict):
        """A set-but-unit xi_r takes the override branch and must change nothing."""
        from pyMacroIO.model import InputOutputModel
        from pyMacroIO.config import ModelConfig
        import numpy as np

        def fresh():
            config = ModelConfig(n_periods=8, time_frequency="quarterly")
            return InputOutputModel(n_periods=8, time_frequency="quarterly",
                                    config=config, _data_dict=minimal_data_dict)

        ref = fresh().run_model()
        m = fresh()
        m.xi_r = np.ones((m.n_regions, m.TT))
        m.baseline_trend_r[:, :] = 1.0
        result = m.run_model()
        for key in ("gdp", "realised_consumption", "gross_output"):
            np.testing.assert_array_equal(result[key], ref[key])

    def test_invalid_schedules_are_rejected_at_run(self, minimal_data_dict):
        from pyMacroIO.model import InputOutputModel
        from pyMacroIO.config import ModelConfig
        import numpy as np
        import pytest

        config = ModelConfig(n_periods=8, time_frequency="quarterly")
        m = InputOutputModel(n_periods=8, time_frequency="quarterly",
                             config=config, _data_dict=minimal_data_dict)
        m.baseline_trend_r[0, 3] = np.nan
        with pytest.raises(ValueError):
            m.run_model()
        m.baseline_trend_r[0, 3] = 1.0
        m.xi_r = np.ones((m.n_regions, m.TT))
        m.xi_r[0, 2] = -0.5
        with pytest.raises(ValueError):
            m.run_model()

    def test_old_pickles_are_backfilled(self, minimal_data_dict):
        import pickle
        import numpy as np
        from pyMacroIO.model import InputOutputModel
        from pyMacroIO.config import ModelConfig

        config = ModelConfig(n_periods=8, time_frequency="quarterly")
        m = InputOutputModel(n_periods=8, time_frequency="quarterly",
                             config=config, _data_dict=minimal_data_dict)
        for attr in ("import_supplement_caps_", "baseline_trend_r", "xi_r"):
            del m.__dict__[attr]
        restored = pickle.loads(pickle.dumps(m))
        assert restored.xi_r is None
        assert restored.import_supplement_caps_ == {}
        np.testing.assert_array_equal(
            restored.baseline_trend_r, np.ones((m.n_regions, m.TT))
        )
        restored.run_model()

    def test_trending_anchor_with_expectations_raises_consumption(self, minimal_data_dict):
        """The anchor sets the target and xi_r carries the income signal."""
        from pyMacroIO.model import InputOutputModel
        from pyMacroIO.config import ModelConfig
        import numpy as np

        def fresh():
            config = ModelConfig(n_periods=8, time_frequency="quarterly")
            return InputOutputModel(n_periods=8, time_frequency="quarterly",
                                    config=config, _data_dict=minimal_data_dict)

        ref = fresh().run_model()
        m = fresh()
        trend = np.linspace(1.0, 1.05, m.TT)
        m.baseline_trend_r[:, :] = trend[None, :]
        m.xi_r = np.broadcast_to(trend, (m.n_regions, m.TT)).copy()
        result = m.run_model()
        assert (
            result["realised_consumption"][:, -1].sum()
            > ref["realised_consumption"][:, -1].sum()
        ), "A trending base anchor with trending expectations must raise consumption."

    def test_regional_xi_moves_only_its_region(self, row_data_dict):
        """xi_r on one region leaves the other region's consumption untouched."""
        from pyMacroIO.model import InputOutputModel
        from pyMacroIO.config import ModelConfig
        import numpy as np

        def fresh():
            config = ModelConfig(n_periods=8, time_frequency="quarterly")
            return InputOutputModel(n_periods=8, time_frequency="quarterly",
                                    config=config, _data_dict=row_data_dict)

        ref = fresh().run_model()
        m = fresh()
        trend = np.linspace(1.0, 1.05, m.TT)
        m.xi_r = np.ones((m.n_regions, m.TT))
        m.xi_r[0, :] = trend
        m.baseline_trend_r[0, :] = trend
        result = m.run_model()
        np.testing.assert_allclose(
            result["consumption_by_hh_region"][1],
            ref["consumption_by_hh_region"][1], rtol=1e-6,
        )
        assert (
            result["consumption_by_hh_region"][0, -1]
            > ref["consumption_by_hh_region"][0, -1]
        )
