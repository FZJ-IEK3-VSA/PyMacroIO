"""Tests for multi-region behaviour, Keynesian closure, price pass-through and household income."""

from __future__ import annotations

import numpy as np
import pytest

from pyMacroIO.model import InputOutputModel
from pyMacroIO.config import ModelConfig

from conftest import _make_two_region_data_dict


def _keynesian_model(data_dict: dict, closure: str = "keynesian") -> InputOutputModel:
    """Return an InputOutputModel with the given investment closure mode."""
    config = ModelConfig(
        n_periods=20,
        time_frequency="quarterly",
        investment_closure=closure,
        investment_adj_speed=1.0,
        investment_savings_ema=1.0,
    )
    return InputOutputModel(
        n_periods=20,
        time_frequency="quarterly",
        config=config,
        _data_dict=data_dict,
    )


@pytest.fixture(scope="module")
def two_region_model(two_region_data_dict: dict) -> InputOutputModel:
    """Return a 3-sector, 2-region ``InputOutputModel``."""
    config = ModelConfig(n_periods=10, time_frequency="quarterly")
    return InputOutputModel(
        n_periods=10,
        time_frequency="quarterly",
        config=config,
        _data_dict=two_region_data_dict,
    )


# TestTwoRegionSetup
class TestTwoRegionSetup:
    """Verifies that region assignment is parsed correctly from a region_map."""

    def test_model_has_two_regions(self, two_region_model: InputOutputModel) -> None:
        """Checks that n_regions is 2 when the region_map spans two distinct values."""
        assert two_region_model.n_regions == 2

    def test_region_sector_indices_correct(self, two_region_model: InputOutputModel) -> None:
        """Checks that sectors A and B are in region 0 and sector C is in region 1."""
        assert list(two_region_model.region_sector_indices[0]) == [0, 1]
        assert list(two_region_model.region_sector_indices[1]) == [2]

    def test_x0_unchanged_by_region_assignment(self, two_region_model: InputOutputModel) -> None:
        """Asserts that gross output at base year is independent of the region_map."""
        np.testing.assert_allclose(
            two_region_model.x0,
            [60., 40., 40.],
            rtol=1e-10,
        )


# TestHouseholdIncomeR
class TestHouseholdIncomeR:
    """Verifies the ``_household_income_r`` formula and its floor behaviour."""

    def test_region_0_income_at_base_year(self, two_region_model: InputOutputModel) -> None:
        """Checks that region 0 income equals 34.0 at base-year labour and profit inputs."""
        m = two_region_model
        # Region 0 sectors A and B: l0 = [20, 7], profits0 = [5, 2]
        labour_r0 = 20. + 7.   # 27
        profit_r0 =  5. + 2.   # 7
        baseline_r0 = 27.       # sum of l0 for region 0 sectors
        beta = float(m.benefits_r[0])  # 0.1 by default
        expected = beta * baseline_r0 + (1.0 - beta) * labour_r0 + profit_r0
        result = m._household_income_r(0, labour_r0, profit_r0)
        np.testing.assert_allclose(result, expected, rtol=1e-10)
        np.testing.assert_allclose(result, 34.0, rtol=1e-10)

    def test_region_1_income_at_base_year(self, two_region_model: InputOutputModel) -> None:
        """Checks that region 1 income equals 8.0 at base-year labour and profit inputs."""
        m = two_region_model
        # Region 1 sector C: l0 = 6, profits0 = 2
        labour_r1 = 6.
        profit_r1 = 2.
        baseline_r1 = 6.  # l0[2]
        beta = float(m.benefits_r[1])  # 0.1
        expected = beta * baseline_r1 + (1.0 - beta) * labour_r1 + profit_r1
        result = m._household_income_r(1, labour_r1, profit_r1)
        np.testing.assert_allclose(result, expected, rtol=1e-10)
        np.testing.assert_allclose(result, 8.0, rtol=1e-10)

    def test_income_floor_prevents_zero(self, two_region_model: InputOutputModel) -> None:
        """Verifies that _household_income_r stays positive when variable income is zero."""
        result = two_region_model._household_income_r(0, 0.0, 0.0)
        assert result >= 1e-9
        np.testing.assert_allclose(result, 0.1 * 27.0, rtol=1e-10)


# TestTwoRegionRunModel
class TestTwoRegionRunModel:
    """Verifies regional aggregates and a non-trivial trade balance when two regions trade."""

    def test_regional_gdp_sums_to_total_gdp(self, two_region_model: InputOutputModel) -> None:
        """Checks that gdp_regional sums to gdp at every period."""
        result = two_region_model.run_model()
        np.testing.assert_allclose(
            result["gdp_regional"].sum(axis=0),
            result["gdp"],
            rtol=1e-6,
        )

    def test_trade_balance_sums_to_zero_and_is_non_trivial(
        self, two_region_model: InputOutputModel
    ) -> None:
        """Checks that trade balances sum to zero across regions and are non-zero at t=0."""
        result = two_region_model.run_model()
        tb = result["trade_balance"]  # shape (2, TT)
        np.testing.assert_allclose(tb.sum(axis=0), 0., atol=1e-8)
        assert abs(tb[0, 0]) > 1.0

    def test_validate_passes_for_two_region_model(
        self, two_region_model: InputOutputModel
    ) -> None:
        """Checks that ``_validate_run`` completes without error on a two-region result."""
        result = two_region_model.run_model(validate=True)
        assert len(result) > 0


# TestKeynesianInvestment
class TestKeynesianInvestment:
    """Verifies the Keynesian closure responds to savings changes and converges without shocks."""

    def test_keynesian_closure_amplifies_shock_through_investment(
        self,
    ) -> None:
        """Keynesian closure gives strictly lower GDP than fixed during a consumption shock."""
        m_fixed     = _keynesian_model(_make_two_region_data_dict(), "fixed")
        m_keynesian = _keynesian_model(_make_two_region_data_dict(), "keynesian")

        for m in (m_fixed, m_keynesian):
            m.apply_consumption_shock(start=3, duration=3, intensity=0.3)

        res_fixed     = m_fixed.run_model()
        res_keynesian = m_keynesian.run_model()

        # GDP loss at t=5 (last shocked period)
        assert res_keynesian["gdp"][5] < res_fixed["gdp"][5]

    def test_keynesian_closure_carries_exogenous_investment_deviation(
        self,
    ) -> None:
        """An exogenous fd_investment_ deviation adds in full under the Keynesian rule."""
        m_base = _keynesian_model(_make_two_region_data_dict())
        m_dev  = _keynesian_model(_make_two_region_data_dict())
        delta = 0.02 * m_dev.fd_investment_[:, 5]
        m_dev.fd_investment_[:, 5] += delta

        res_base = m_base.run_model()
        res_dev  = m_dev.run_model()

        gained = (res_dev["realised_investment"][:, 5]
                  - res_base["realised_investment"][:, 5])
        assert np.allclose(gained, delta, rtol=1e-6, atol=1e-9)

    def test_keynesian_closure_without_shock_does_not_diverge(
        self,
    ) -> None:
        """Without shocks, GDP stays within 20 per cent of base year at every period."""
        m = _keynesian_model(_make_two_region_data_dict())
        result = m.run_model()
        gdp_base = result["gdp"][0]
        for t in range(1, m.TT):
            assert result["gdp"][t] > gdp_base * 0.8
            assert result["gdp"][t] < gdp_base * 1.2


# TestBuildPriceInverses
class TestBuildPriceInverses:
    """Verifies ``_build_price_inverses`` output shapes and its ``ValueError`` on singular input."""

    def test_returns_correct_shape_and_diagonal(self) -> None:
        """Checks that both Leontief inverses have shape (N, N) with diagonal entries >= 1."""
        config = ModelConfig(n_periods=5, time_frequency="quarterly")
        m = InputOutputModel(
            n_periods=5,
            time_frequency="quarterly",
            config=config,
            _data_dict=_make_two_region_data_dict(),
        )
        m.price_passthrough_pos = 0.5
        m.price_passthrough_neg = 0.5

        L_pos, L_neg = m._build_price_inverses(m.A)

        assert L_pos.shape == (m.N, m.N)
        assert L_neg.shape == (m.N, m.N)
        # Network cost-push ensures diagonal >= 1.
        assert np.all(np.diag(L_pos) >= 1.0 - 1e-10)

    def test_singular_matrix_raises_value_error(self) -> None:
        """Checks that a singular (I - A) raises ValueError with a match on 'invertible'."""
        config = ModelConfig(n_periods=5, time_frequency="quarterly")
        m = InputOutputModel(
            n_periods=5,
            time_frequency="quarterly",
            config=config,
            _data_dict=_make_two_region_data_dict(),
        )
        m.price_passthrough_pos = 1.0
        m.price_passthrough_neg = 1.0
        # Column sums of A_bad are all 1.0, so (I - A_bad) is singular.
        A_bad = np.array(
            [[0.5, 0.5, 0.5],
             [0.5, 0.5, 0.5],
             [0.,  0.,  0.]],
            dtype=np.float64,
        )
        with pytest.raises(ValueError, match="invertible"):
            m._build_price_inverses(A_bad)


# TestFixedInvestmentClosure
class TestFixedInvestmentClosure:
    """Verifies fixed-closure investment is stable and bounds Keynesian GDP from below."""

    def test_fixed_closure_has_higher_gdp_than_keynesian_at_all_shocked_periods(
        self,
    ) -> None:
        """Fixed-closure GDP is at least Keynesian GDP at every period in the shocked window."""
        m_fixed     = _keynesian_model(_make_two_region_data_dict(), "fixed")
        m_keynesian = _keynesian_model(_make_two_region_data_dict(), "keynesian")

        for m in (m_fixed, m_keynesian):
            m.apply_consumption_shock(start=3, duration=3, intensity=0.3)

        res_fixed     = m_fixed.run_model()
        res_keynesian = m_keynesian.run_model()

        for t in range(3, 6):
            assert res_fixed["gdp"][t] >= res_keynesian["gdp"][t], (
                f"Fixed >= Keynesian GDP at t={t}."
            )

    def test_fixed_closure_stable_without_shocks(self) -> None:
        """With no shock, GDP stays within 5 per cent of base year at every period."""
        m = _keynesian_model(_make_two_region_data_dict(), "fixed")
        result = m.run_model()
        gdp_base = result["gdp"][0]
        for t in range(1, m.TT):
            assert result["gdp"][t] >= gdp_base * 0.95, (
                f"GDP at t={t} below 95 per cent of base."
            )
            assert result["gdp"][t] <= gdp_base * 1.05, (
                f"GDP at t={t} above 105 per cent of base."
            )


# TestKeynesianInvestmentParameters
class TestKeynesianInvestmentParameters:
    """Verifies each investment parameter moves GDP as the savings-investment link predicts."""

    def _build_keynesian(
        self,
        adj_speed: float = 1.0,
        ema: float = 1.0,
        growth_cap: float | None = None,
        n_periods: int = 20,
    ) -> InputOutputModel:
        """Return a Keynesian InputOutputModel with the given adjustment parameters."""
        config = ModelConfig(
            n_periods=n_periods,
            time_frequency="quarterly",
            investment_closure="keynesian",
            investment_adj_speed=adj_speed,
            investment_savings_ema=ema,
            investment_scale_growth_cap=growth_cap,
        )
        return InputOutputModel(
            n_periods=n_periods,
            time_frequency="quarterly",
            config=config,
            _data_dict=_make_two_region_data_dict(),
        )

    def test_faster_adj_speed_amplifies_shock_at_first_response_period(self) -> None:
        """Faster adjustment sharpens the GDP fall at t=4, the first period lagged savings reach."""
        m_fast = self._build_keynesian(adj_speed=1.0)
        m_slow = self._build_keynesian(adj_speed=0.5)

        for m in (m_fast, m_slow):
            m.apply_consumption_shock(start=3, duration=5, intensity=0.3)

        res_fast = m_fast.run_model()
        res_slow = m_slow.run_model()

        assert res_fast["gdp"][4] < res_slow["gdp"][4], (
            "Faster adjustment should produce a deeper GDP fall at t=4."
        )

    def test_ema_smoothing_reduces_immediate_investment_response(self) -> None:
        """Checks that EMA smoothing dampens the investment fall at t=1 relative to no smoothing."""
        m_no_smooth = self._build_keynesian(ema=1.0)
        m_smooth    = self._build_keynesian(ema=0.5)

        for m in (m_no_smooth, m_smooth):
            m.apply_consumption_shock(start=0, duration=8, intensity=0.3)

        res_no_smooth = m_no_smooth.run_model()
        res_smooth    = m_smooth.run_model()

        assert res_no_smooth["gdp"][1] < res_smooth["gdp"][1], (
            "No-smoothing model should show a deeper GDP fall at t=1."
        )

    def test_growth_cap_slows_investment_recovery(self) -> None:
        """A binding growth cap keeps GDP lower during the post-shock recovery window."""
        m_capped   = self._build_keynesian(growth_cap=0.02, n_periods=25)
        m_uncapped = self._build_keynesian(growth_cap=None,  n_periods=25)

        for m in (m_capped, m_uncapped):
            m.apply_consumption_shock(start=2, duration=5, intensity=0.4)

        res_capped   = m_capped.run_model()
        res_uncapped = m_uncapped.run_model()

        assert res_capped["gdp"][7:12].mean() < res_uncapped["gdp"][7:12].mean(), (
            "Capped model should show lower GDP over the post-shock recovery window."
        )

    def test_growth_cap_binds_the_endogenous_scale_only(self) -> None:
        """A period on the growth cap still delivers an fd_investment_ deviation in full."""
        m_base = self._build_keynesian(growth_cap=0.02, n_periods=25)
        m_dev  = self._build_keynesian(growth_cap=0.02, n_periods=25)

        for m in (m_base, m_dev):
            m.apply_consumption_shock(start=2, duration=5, intensity=0.4)
        delta = 0.02 * m_dev.fd_investment_[:, 5]
        m_dev.fd_investment_[:, 5] += delta

        res_base = m_base.run_model()
        res_dev  = m_dev.run_model()

        inv_base = np.sum(m_base.inv_agg_base)
        scale = res_base["realised_investment"].sum(axis=0) / inv_base
        assert np.isclose(scale[5] / scale[4], 1.02, rtol=1e-9), (
            "Growth cap should bind at t=5 in the recovery window."
        )

        gained = (res_dev["realised_investment"][:, 5]
                  - res_base["realised_investment"][:, 5])
        assert np.allclose(gained, delta, rtol=1e-6, atol=1e-9), (
            "The cap should bind the endogenous scale alone, leaving the "
            "exogenous deviation whole."
        )


# TestPerRegionClosureMode
class TestPerRegionClosureMode:
    """Verifies per-region household closure modes act on each region independently."""

    def _build_mixed_closure_model(self) -> InputOutputModel:
        """Return a 2-region InputOutputModel with region 0 frozen and region 1 return_to_base."""
        config = ModelConfig(
            n_periods=15,
            time_frequency="quarterly",
            n_regions=2,
            household_closure_mode=["frozen", "return_to_base"],
        )
        return InputOutputModel(
            n_periods=15,
            time_frequency="quarterly",
            config=config,
            _data_dict=_make_two_region_data_dict(),
        )

    def test_model_accepts_per_region_closure_mode_list(self) -> None:
        """Checks that household_closure_mode_r matches the list supplied to ModelConfig."""
        m = self._build_mixed_closure_model()
        assert m.household_closure_mode_r == ["frozen", "return_to_base"]

    def test_frozen_region_consumption_stays_flat(self) -> None:
        """Region 0 (frozen) keeps consumption at base-year level in the shock window t=3..6."""
        m = self._build_mixed_closure_model()
        m.apply_consumption_shock(start=3, duration=4, intensity=0.4, region=1)
        result = m.run_model()
        c_r0_shock = result["consumption_by_hh_region"][0, 3:7]
        np.testing.assert_allclose(c_r0_shock, result["consumption_by_hh_region"][0, 0], rtol=1e-6)

    def test_return_to_base_region_responds_to_shock(self) -> None:
        """Region 1 (return_to_base) shows lower consumption at t=4 than at t=0."""
        m = self._build_mixed_closure_model()
        m.apply_consumption_shock(start=3, duration=4, intensity=0.4, region=1)
        result = m.run_model()
        assert result["consumption_by_hh_region"][1, 4] < result["consumption_by_hh_region"][1, 0], (
            "Region 1 consumption at t=4 should be below t=0."
        )


# TestGovernmentElasticity
class TestGovernmentElasticity:
    """Verifies gov_income_elasticity amplifies shocks through government spending."""

    def test_positive_elasticity_amplifies_consumption_shock(self) -> None:
        """Positive government income elasticity lowers GDP during a consumption shock."""
        data_dict = _make_two_region_data_dict()

        config_elastic = ModelConfig(
            n_periods=20,
            time_frequency="quarterly",
            gov_income_elasticity=0.5,
        )
        m_elastic = InputOutputModel(
            n_periods=20,
            time_frequency="quarterly",
            config=config_elastic,
            _data_dict=data_dict,
        )

        config_inelastic = ModelConfig(
            n_periods=20,
            time_frequency="quarterly",
            gov_income_elasticity=0.0,
        )
        m_inelastic = InputOutputModel(
            n_periods=20,
            time_frequency="quarterly",
            config=config_inelastic,
            _data_dict=data_dict,
        )

        for m in (m_elastic, m_inelastic):
            m.apply_consumption_shock(start=3, duration=4, intensity=0.35)

        res_elastic   = m_elastic.run_model()
        res_inelastic = m_inelastic.run_model()

        assert res_elastic["gdp"][5] < res_inelastic["gdp"][5], (
            "Elastic government spending should amplify the GDP loss at t=5."
        )

    def test_zero_elasticity_holds_government_spending_flat(self) -> None:
        """Zero-elasticity GDP is at least elastic-model GDP at every shocked period."""
        data_dict = _make_two_region_data_dict()

        config_elastic = ModelConfig(
            n_periods=20,
            time_frequency="quarterly",
            gov_income_elasticity=0.5,
        )
        m_elastic = InputOutputModel(
            n_periods=20,
            time_frequency="quarterly",
            config=config_elastic,
            _data_dict=data_dict,
        )

        config_inelastic = ModelConfig(
            n_periods=20,
            time_frequency="quarterly",
            gov_income_elasticity=0.0,
        )
        m_inelastic = InputOutputModel(
            n_periods=20,
            time_frequency="quarterly",
            config=config_inelastic,
            _data_dict=data_dict,
        )

        for m in (m_elastic, m_inelastic):
            m.apply_consumption_shock(start=3, duration=4, intensity=0.35)

        res_elastic   = m_elastic.run_model()
        res_inelastic = m_inelastic.run_model()

        for t in range(3, 7):
            assert res_inelastic["gdp"][t] >= res_elastic["gdp"][t], (
                f"Zero-elasticity GDP should be >= elastic GDP at shocked period t={t}."
            )


# TestGovernmentElasticityExpansion
class TestGovernmentElasticityExpansion:
    """Verifies gov_income_elasticity raises spending when income sits above the base level."""

    def test_positive_elasticity_boosts_gdp_in_expansion(self) -> None:
        """With base_gov_income_total cut 20 per cent, the elastic model has higher GDP at t=5."""
        data_dict = _make_two_region_data_dict()

        config_elastic = ModelConfig(
            n_periods=15,
            time_frequency="quarterly",
            gov_income_elasticity=0.5,
        )
        m_elastic = InputOutputModel(
            n_periods=15,
            time_frequency="quarterly",
            config=config_elastic,
            _data_dict=data_dict,
        )

        config_inelastic = ModelConfig(
            n_periods=15,
            time_frequency="quarterly",
            gov_income_elasticity=0.0,
        )
        m_inelastic = InputOutputModel(
            n_periods=15,
            time_frequency="quarterly",
            config=config_inelastic,
            _data_dict=data_dict,
        )

        m_elastic.base_gov_income_total *= 0.8

        res_elastic   = m_elastic.run_model()
        res_inelastic = m_inelastic.run_model()

        assert res_elastic["gdp"][5] > res_inelastic["gdp"][5], (
            "Elastic model should record higher GDP than zero-elasticity at t=5."
        )

    def test_government_elasticity_symmetric_around_base(self) -> None:
        """With no shock and unpatched base, GDP stays within 5 per cent at every t in 1..9."""
        data_dict = _make_two_region_data_dict()

        config = ModelConfig(
            n_periods=10,
            time_frequency="quarterly",
            gov_income_elasticity=0.5,
        )
        m_elastic = InputOutputModel(
            n_periods=10,
            time_frequency="quarterly",
            config=config,
            _data_dict=data_dict,
        )

        result = m_elastic.run_model()

        gdp_base = result["gdp"][0]
        for t in range(1, 10):
            ratio = abs(result["gdp"][t] - gdp_base) / gdp_base
            assert ratio < 0.05, (
                f"GDP at t={t} deviates by {ratio:.4f} from base year, exceeding 5 per cent."
            )


# TestPricePassThrough
class TestPricePassThrough:
    """Verifies cost shocks raise the price index and deflation reduces consumption."""

    def test_cost_shock_raises_price_index(self) -> None:
        """A cost shock on sector A raises some price index above one in the shock window."""
        data_dict = _make_two_region_data_dict()

        config = ModelConfig(
            n_periods=15,
            time_frequency="quarterly",
            price_passthrough_enabled=True,
            price_passthrough_pos=0.8,
            price_passthrough_neg=0.4,
        )
        m = InputOutputModel(
            n_periods=15,
            time_frequency="quarterly",
            config=config,
            _data_dict=data_dict,
        )
        m.apply_price_cost_shock("A", time_period=3, delta_cost=0.5, duration=6)

        result = m.run_model()

        assert result["price_index"][:, 5].max() > 1.0, (
            "A cost shock on sector A should raise at least one price index above 1.0."
        )

    def test_higher_passthrough_pos_produces_larger_price_response(self) -> None:
        """Higher price_passthrough_pos amplifies the price response to a positive cost shock."""
        data_dict = _make_two_region_data_dict()

        config_high = ModelConfig(
            n_periods=15,
            time_frequency="quarterly",
            price_passthrough_enabled=True,
            price_passthrough_pos=0.9,
            price_passthrough_neg=0.3,
        )
        m_high = InputOutputModel(
            n_periods=15,
            time_frequency="quarterly",
            config=config_high,
            _data_dict=data_dict,
        )

        config_low = ModelConfig(
            n_periods=15,
            time_frequency="quarterly",
            price_passthrough_enabled=True,
            price_passthrough_pos=0.3,
            price_passthrough_neg=0.3,
        )
        m_low = InputOutputModel(
            n_periods=15,
            time_frequency="quarterly",
            config=config_low,
            _data_dict=data_dict,
        )

        for m in (m_high, m_low):
            m.apply_price_cost_shock("A", time_period=3, delta_cost=0.5, duration=6)

        result_high = m_high.run_model()
        result_low  = m_low.run_model()

        assert result_high["price_index"][:, 5].sum() > result_low["price_index"][:, 5].sum(), (
            "Higher passthrough_pos should produce a larger total price index at t=5."
        )

    def test_price_deflation_reduces_household_consumption(self) -> None:
        """Deflating household income by the price index reduces consumption under a cost shock."""
        data_dict = _make_two_region_data_dict()

        config_deflate = ModelConfig(
            n_periods=15,
            time_frequency="quarterly",
            price_passthrough_enabled=True,
            price_passthrough_pos=0.8,
            price_passthrough_neg=0.4,
            price_deflate_household_income=True,
        )
        m_deflate = InputOutputModel(
            n_periods=15,
            time_frequency="quarterly",
            config=config_deflate,
            _data_dict=data_dict,
        )

        config_nodeflate = ModelConfig(
            n_periods=15,
            time_frequency="quarterly",
            price_passthrough_enabled=True,
            price_passthrough_pos=0.8,
            price_passthrough_neg=0.4,
            price_deflate_household_income=False,
        )
        m_nodeflate = InputOutputModel(
            n_periods=15,
            time_frequency="quarterly",
            config=config_nodeflate,
            _data_dict=data_dict,
        )

        for m in (m_deflate, m_nodeflate):
            m.apply_price_cost_shock("A", time_period=3, delta_cost=0.5, duration=6)

        res_deflate   = m_deflate.run_model()
        res_nodeflate = m_nodeflate.run_model()

        assert res_deflate["consumption_by_hh_region"].sum() < res_nodeflate["consumption_by_hh_region"].sum(), (
            "Price deflation should reduce total consumption relative to the non-deflated path."
        )


# TestPerRegionIncomeTax
class TestPerRegionIncomeTax:
    """Verifies that per-region income tax rates are stored and applied correctly."""

    def test_per_region_income_tax_rates_are_stored_correctly(self) -> None:
        """Checks that income_tax_rate_r stores each region's tax rate as supplied."""
        config = ModelConfig(
            n_periods=10,
            time_frequency="quarterly",
            n_regions=2,
            income_tax_rate=[0.0, 0.2],
        )
        m = InputOutputModel(
            n_periods=10,
            time_frequency="quarterly",
            config=config,
            _data_dict=_make_two_region_data_dict(),
        )

        assert float(m.income_tax_rate_r[0]) == 0.0
        assert abs(float(m.income_tax_rate_r[1]) - 0.2) < 1e-12

    def test_taxed_region_consumption_falls_below_base_year(self) -> None:
        """Region 1, taxed at 20 per cent, records lower consumption at t=3 than at t=0."""
        config = ModelConfig(
            n_periods=15,
            time_frequency="quarterly",
            n_regions=2,
            income_tax_rate=[0.0, 0.2],
        )
        m = InputOutputModel(
            n_periods=15,
            time_frequency="quarterly",
            config=config,
            _data_dict=_make_two_region_data_dict(),
        )

        result = m.run_model()

        assert result["consumption_by_hh_region"][1, 3] < result["consumption_by_hh_region"][1, 0], (
            "Region 1 after-tax income should reduce consumption below base year at t=3."
        )

    def test_zero_tax_region_remains_near_base_year(self) -> None:
        """Region 0, untaxed and unshocked, keeps consumption within 5 per cent of base at t=3."""
        config = ModelConfig(
            n_periods=15,
            time_frequency="quarterly",
            n_regions=2,
            income_tax_rate=[0.0, 0.2],
        )
        m = InputOutputModel(
            n_periods=15,
            time_frequency="quarterly",
            config=config,
            _data_dict=_make_two_region_data_dict(),
        )

        result = m.run_model()

        base = result["consumption_by_hh_region"][0, 0]
        deviation = abs(result["consumption_by_hh_region"][0, 3] - base) / base
        assert deviation < 0.05, (
            f"Region 0 consumption at t=3 deviates by {deviation:.4f} from base year."
        )
