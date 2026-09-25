"""Tests for household consumption demand methods in the pyMacroIO model.

Covers findemand_cd closures (frozen, return-to-base, scarred), LES, the floor clamp, and capacity.
"""

import numpy as np
import numpy.testing as npt
import pytest

from pyMacroIO.constants import CONSUMPTION_FLOOR_RATIO

from pyMacroIO import InputOutputModel, ModelConfig

from conftest import _make_skill_data_dict


# TestFindemandCdFrozen
class TestFindemandCdFrozen:
    """Tests for findemand_cd in frozen closure mode."""

    def test_frozen_cdt_new_equals_base_consumption_total(self, minimal_model):
        """Frozen closure returns base_consumption_total regardless of income signal."""
        m = minimal_model
        theta = m.household_consumption_shares
        bct = m.base_consumption_total
        Cdt_new, _ = m.findemand_cd(
            theta, bct, 1.0, 0.0, 0.0, household_closure_mode="frozen"
        )
        npt.assert_allclose(Cdt_new, bct, rtol=1e-12)

    def test_frozen_cdt_new_unaffected_by_low_income_signal(self, minimal_model):
        """Frozen closure ignores a near-zero income signal when computing Cdt_new."""
        m = minimal_model
        theta = m.household_consumption_shares
        bct = m.base_consumption_total
        Cdt_new, _ = m.findemand_cd(
            theta, bct, 1.0, 1e-6, 0.0, household_closure_mode="frozen"
        )
        npt.assert_allclose(Cdt_new, bct, rtol=1e-12)

    def test_frozen_sectoral_demand_equals_theta_times_bct_no_eps(self, minimal_model):
        """With eps=0 and frozen closure, sectoral demand equals theta * base_consumption_total."""
        m = minimal_model
        theta = m.household_consumption_shares
        bct = m.base_consumption_total
        _, cd = m.findemand_cd(
            theta, bct, 1.0, bct, 0.0, household_closure_mode="frozen"
        )
        npt.assert_allclose(cd, theta * bct, rtol=1e-12)

    def test_frozen_sum_cd_equals_bct_when_eps_zero(self, minimal_model):
        """With eps=0, sectoral demand sums to base_consumption_total in frozen closure."""
        m = minimal_model
        theta = m.household_consumption_shares
        bct = m.base_consumption_total
        _, cd = m.findemand_cd(
            theta, bct, 1.0, bct, 0.0, household_closure_mode="frozen"
        )
        npt.assert_allclose(cd.sum(), bct, rtol=1e-12)

    def test_frozen_sum_cd_halved_when_eps_half(self, minimal_model):
        """With eps=0.5, the sum of sectoral demand equals 0.5 * base_consumption_total."""
        m = minimal_model
        theta = m.household_consumption_shares
        bct = m.base_consumption_total
        _, cd = m.findemand_cd(
            theta, bct, 1.0, bct, 0.5, household_closure_mode="frozen"
        )
        npt.assert_allclose(cd.sum(), 0.5 * bct, rtol=1e-12)

    def test_frozen_income_independence(self, minimal_model):
        """Frozen closure gives base_consumption_total even with an extreme income signal."""
        m = minimal_model
        theta = m.household_consumption_shares
        bct = m.base_consumption_total
        bhi = m.base_household_income
        Cdt_new, _ = m.findemand_cd(
            theta, bct, 1.0, bhi * 100, 0.0, household_closure_mode="frozen"
        )
        npt.assert_allclose(Cdt_new, bct, rtol=1e-12)


# TestFindemandCdReturnToBase
class TestFindemandCdReturnToBase:
    """Tests for findemand_cd in return-to-base closure mode."""

    def test_return_to_base_cdt_new_equals_bct_at_base_income(self, minimal_model):
        """With income equal to base and xit=1.0, return-to-base closure gives Cdt_new = bct."""
        m = minimal_model
        theta = m.household_consumption_shares
        bct = m.base_consumption_total
        bhi = m.base_household_income
        sr_eff = 1.0 - bct / bhi   # so that (1 - sr_eff) * bhi = bct with coc=0

        Cdt_new, _ = m.findemand_cd(
            theta, bct, 1.0, bhi, 0.0,
            household_closure_mode="return_to_base",
            base_household_income=bhi,
            income_tax_rate=0.0,
            c_other_coef=0.0,
            savings_rate=sr_eff,
        )
        npt.assert_allclose(Cdt_new, bct, rtol=1e-6)

    def test_return_to_base_sum_cd_less_than_cdt_new_with_eps(self, minimal_model):
        """With eps > 0, sectoral demand sums strictly below Cdt_new in return-to-base closure."""
        m = minimal_model
        theta = m.household_consumption_shares
        bct = m.base_consumption_total
        bhi = m.base_household_income
        sr_eff = 1.0 - bct / bhi

        Cdt_new, cd = m.findemand_cd(
            theta, bct, 1.0, bhi, 0.3,
            household_closure_mode="return_to_base",
            base_household_income=bhi,
            income_tax_rate=0.0,
            c_other_coef=0.0,
            savings_rate=sr_eff,
        )
        assert cd.sum() < Cdt_new

    def test_return_to_base_sum_cd_equals_cdt_new_times_one_minus_eps(self, minimal_model):
        """sum(cd) equals Cdt_new * (1 - eps) in return-to-base closure."""
        m = minimal_model
        theta = m.household_consumption_shares
        bct = m.base_consumption_total
        bhi = m.base_household_income
        sr_eff = 1.0 - bct / bhi
        eps = 0.3

        Cdt_new, cd = m.findemand_cd(
            theta, bct, 1.0, bhi, eps,
            household_closure_mode="return_to_base",
            base_household_income=bhi,
            income_tax_rate=0.0,
            c_other_coef=0.0,
            savings_rate=sr_eff,
        )
        npt.assert_allclose(cd.sum(), Cdt_new * (1.0 - eps), rtol=1e-12)

    def test_return_to_base_off_equilibrium_halved_income(self, minimal_model):
        """Return-to-base with halved income gives Cdt_new equal to current consumption capacity."""
        m = minimal_model
        theta = m.household_consumption_shares
        bct = m.base_consumption_total
        bhi = m.base_household_income
        sr_eff = 1.0 - bct / bhi

        Cdt_new, _ = m.findemand_cd(
            theta, bct, 1.0, bhi / 2, 0.0,
            household_closure_mode="return_to_base",
            base_household_income=bhi,
            income_tax_rate=0.0,
            c_other_coef=0.0,
            savings_rate=sr_eff,
        )

        current_capacity = m._household_consumption_capacity(
            bhi / 2, savings_rate=sr_eff, c_other_coef=0.0
        )
        npt.assert_allclose(Cdt_new, current_capacity, rtol=1e-10)
        assert Cdt_new < bct


# TestFindemandCdScarred
class TestFindemandCdScarred:
    """Tests for findemand_cd in scarred closure mode."""

    def test_scarred_lower_cdt_pulls_new_below_frozen_baseline(
        self, minimal_model
    ) -> None:
        """A scarred target (Cdt below bct) produces a lower Cdt_new than the unscarred case."""
        m = minimal_model
        bct = m.base_consumption_total
        bhi = m.base_household_income

        Cdt_scarred   = bct * 0.7
        Cdt_unscarred = bct

        Cdt_new_scarred, _ = m.findemand_cd(
            theta=m.household_consumption_shares,
            Cdt=Cdt_scarred,
            xit=Cdt_scarred / bct,
            household_income_signal=bhi,
            eps=0.0,
            household_closure_mode="scarred",
            base_household_income=bhi,
        )
        Cdt_new_unscarred, _ = m.findemand_cd(
            theta=m.household_consumption_shares,
            Cdt=Cdt_unscarred,
            xit=Cdt_unscarred / bct,
            household_income_signal=bhi,
            eps=0.0,
            household_closure_mode="scarred",
            base_household_income=bhi,
        )
        assert Cdt_new_scarred < Cdt_new_unscarred, (
            "A lower Cdt (scarred) should produce a lower Cdt_new than the unscarred case."
        )

    def test_scarred_cdt_new_lower_than_return_to_base_for_same_inputs(self, minimal_model):
        """Scarred closure with reduced Cdt gives a lower Cdt_new than return-to-base."""
        m = minimal_model
        theta = m.household_consumption_shares
        bct = m.base_consumption_total
        bhi = m.base_household_income
        sr_eff = 1.0 - bct / bhi

        Cdt_scarred, _ = m.findemand_cd(
            theta, bct * 0.7, 1.0, bhi, 0.0,
            household_closure_mode="scarred",
            base_household_income=bhi,
            income_tax_rate=0.0,
            c_other_coef=0.0,
            savings_rate=sr_eff,
        )
        Cdt_rtb, _ = m.findemand_cd(
            theta, bct, 1.0, bhi, 0.0,
            household_closure_mode="return_to_base",
            base_household_income=bhi,
            income_tax_rate=0.0,
            c_other_coef=0.0,
            savings_rate=sr_eff,
        )
        assert Cdt_scarred < Cdt_rtb

    def test_scarred_magnitude_hand_computed(self, minimal_model):
        """Scarred closure with Cdt = 0.7 * bct gives the hand-computed value bct * 0.7^rho1."""
        m = minimal_model
        theta = m.household_consumption_shares
        bct = m.base_consumption_total
        bhi = m.base_household_income
        sr_eff = 1.0 - bct / bhi

        Cdt_new, _ = m.findemand_cd(
            theta, bct * 0.7, 1.0, bhi, 0.0,
            household_closure_mode="scarred",
            base_household_income=bhi,
            income_tax_rate=0.0,
            c_other_coef=0.0,
            savings_rate=sr_eff,
        )

        expected_scarred = bct * (0.7 ** m.rho1)
        npt.assert_allclose(Cdt_new, expected_scarred, rtol=1e-6)
        assert Cdt_new < bct


# TestFindemandCdLES
class TestFindemandCdLES:
    """Tests for the linear expenditure system (LES) branch of findemand_cd."""

    def test_les_frozen_cd_matches_hand_computed_values(self, minimal_model):
        """In frozen mode with LES, cd matches (gamma + beta * supernumerary) * (1-eps)."""
        m = minimal_model
        theta = m.household_consumption_shares
        bct = m.base_consumption_total
        gamma = np.array([2.0, 4.0, 6.0], dtype=np.float64)
        beta  = np.array([0.3, 0.4, 0.3], dtype=np.float64)
        eps   = 0.2

        _, cd = m.findemand_cd(
            theta, bct, 1.0, bct, eps,
            household_closure_mode="frozen",
            gamma=gamma, beta=beta,
        )

        Gamma = gamma.sum()
        supernumerary = bct - Gamma
        expected = (gamma + beta * supernumerary) * (1.0 - eps)
        npt.assert_allclose(cd, expected, rtol=1e-12)

    def test_les_non_trivial_hand_computation(self, minimal_model):
        """LES in frozen mode with non-trivial gamma and beta matches the hand-computed result."""
        m = minimal_model
        theta = m.household_consumption_shares
        gamma = np.array([2.0, 3.0, 1.0], dtype=np.float64)
        bct_test = 20.0
        beta = np.array([0.5, 0.3, 0.2], dtype=np.float64)
        eps = 0.0
        # supernumerary = bct_test - Gamma = 14.0
        expected_cd = np.array([9.0, 7.2, 3.8])

        _, cd = m.findemand_cd(
            theta, m.base_consumption_total, 1.0, m.base_consumption_total, eps,
            household_closure_mode="frozen",
            base_consumption_total=bct_test,
            gamma=gamma,
            beta=beta,
        )
        npt.assert_allclose(cd, expected_cd, rtol=1e-12)

    def test_les_zero_gamma_matches_cobb_douglas_guard(self, minimal_model):
        """With all-zero gamma, findemand_cd takes the Cobb-Douglas path, not the LES path."""
        m = minimal_model
        theta = m.household_consumption_shares
        bct = m.base_consumption_total
        N = m.N
        eps = 0.0

        _, cd_cd = m.findemand_cd(
            theta, bct, 1.0, bct, eps,
            household_closure_mode="frozen",
        )
        _, cd_les_zero = m.findemand_cd(
            theta, bct, 1.0, bct, eps,
            household_closure_mode="frozen",
            gamma=np.zeros(N, dtype=np.float64),
            beta=theta.copy(),
        )

        npt.assert_allclose(cd_les_zero, cd_cd, rtol=1e-12)


# TestConsumptionFloorClamp
class TestConsumptionFloorClamp:
    """Tests for the consumption floor clamp inside findemand_cd."""

    def test_floor_clamp_binds_under_deep_scarring(self, minimal_model):
        """With extreme scarring and very low income, Cdt_new stays at or above the floor."""
        m = minimal_model
        theta = m.household_consumption_shares
        bct = m.base_consumption_total
        bhi = m.base_household_income

        Cdt_new, _ = m.findemand_cd(
            theta, bct * 1e-10, 1.0, bhi * 1e-4, 0.0,
            household_closure_mode="scarred",
            base_household_income=bhi,
        )

        current_capacity = m._household_consumption_capacity(bhi * 1e-4)
        floor_consumption = min(bct * CONSUMPTION_FLOOR_RATIO, current_capacity)
        assert Cdt_new >= floor_consumption - 1e-12


# TestHouseholdCapacity
class TestHouseholdCapacity:
    """Tests for _household_consumption_capacity."""

    def test_capacity_equals_income_when_savings_and_coc_zero(self, minimal_model):
        """With savings_rate=0 and c_other_coef=0, consumption capacity equals household income."""
        cap = minimal_model._household_consumption_capacity(
            100.0, savings_rate=0.0, c_other_coef=0.0
        )
        npt.assert_allclose(cap, 100.0, rtol=1e-12)

    def test_capacity_equals_eight_tenths_income_at_twenty_percent_savings(self, minimal_model):
        """With savings_rate=0.2 and c_other_coef=0, capacity equals 0.8 times household income."""
        cap = minimal_model._household_consumption_capacity(
            100.0, savings_rate=0.2, c_other_coef=0.0
        )
        npt.assert_allclose(cap, 80.0, rtol=1e-12)

    def test_capacity_scales_linearly_with_income(self, minimal_model):
        """Doubling income doubles capacity when savings and other-coef are fixed."""
        cap_1 = minimal_model._household_consumption_capacity(
            50.0, savings_rate=0.1, c_other_coef=0.0
        )
        cap_2 = minimal_model._household_consumption_capacity(
            100.0, savings_rate=0.1, c_other_coef=0.0
        )
        npt.assert_allclose(cap_2, 2.0 * cap_1, rtol=1e-12)

    def test_capacity_reduced_by_c_other_coef(self, minimal_model):
        """A positive c_other_coef reduces capacity below (1 - savings_rate) * income."""
        cap_no_coc = minimal_model._household_consumption_capacity(
            100.0, savings_rate=0.0, c_other_coef=0.0
        )
        cap_with_coc = minimal_model._household_consumption_capacity(
            100.0, savings_rate=0.0, c_other_coef=0.2
        )
        assert cap_with_coc < cap_no_coc
        npt.assert_allclose(cap_with_coc, 80.0, rtol=1e-12)


# TestIncomeTaxRate
class TestIncomeTaxRate:
    """Tests for the income_tax_rate parameter in findemand_cd."""

    def test_income_tax_reduces_cdt_new_in_return_to_base(self, minimal_model):
        """A positive income_tax_rate reduces Cdt_new in return-to-base closure."""
        m = minimal_model
        theta = m.household_consumption_shares
        bct = m.base_consumption_total
        bhi = m.base_household_income
        sr_eff = 1.0 - bct / bhi

        Cdt_new_no_tax, _ = m.findemand_cd(
            theta, bct, 1.0, bhi, 0.0,
            household_closure_mode="return_to_base",
            base_household_income=bhi,
            income_tax_rate=0.0,
            c_other_coef=0.0,
            savings_rate=sr_eff,
        )
        Cdt_new_taxed, _ = m.findemand_cd(
            theta, bct, 1.0, bhi, 0.0,
            household_closure_mode="return_to_base",
            base_household_income=bhi,
            income_tax_rate=0.2,
            c_other_coef=0.0,
            savings_rate=sr_eff,
        )
        assert Cdt_new_taxed < Cdt_new_no_tax

    def test_income_tax_proportionally_reduces_capacity(self, minimal_model):
        """A 20 per cent income tax reduces consumption capacity by exactly 20 per cent."""
        m = minimal_model
        bct = m.base_consumption_total
        bhi = m.base_household_income
        sr_eff = 1.0 - bct / bhi

        untaxed_capacity = m._household_consumption_capacity(
            bhi, savings_rate=sr_eff, c_other_coef=0.0
        )
        taxed_capacity = m._household_consumption_capacity(
            bhi * (1 - 0.2), savings_rate=sr_eff, c_other_coef=0.0
        )
        npt.assert_allclose(taxed_capacity, 0.8 * untaxed_capacity, rtol=1e-12)

    def test_income_tax_has_no_effect_in_frozen_closure(self, minimal_model):
        """In frozen closure, Cdt_new equals base_consumption_total at any income_tax_rate."""
        m = minimal_model
        theta = m.household_consumption_shares
        bct = m.base_consumption_total
        bhi = m.base_household_income

        Cdt_new_no_tax, _ = m.findemand_cd(
            theta, bct, 1.0, bhi, 0.0,
            household_closure_mode="frozen",
            income_tax_rate=0.0,
        )
        Cdt_new_taxed, _ = m.findemand_cd(
            theta, bct, 1.0, bhi, 0.0,
            household_closure_mode="frozen",
            income_tax_rate=0.4,
        )
        npt.assert_allclose(Cdt_new_no_tax, bct, rtol=1e-12)
        npt.assert_allclose(Cdt_new_taxed, Cdt_new_no_tax, rtol=1e-12)

    def test_income_tax_with_zero_income_signal_hits_floor(self, minimal_model):
        """With income_tax_rate = 0.5, the capacity clamp limits Cdt_new to at most 0.5 * bct."""
        m = minimal_model
        theta = m.household_consumption_shares
        bct = m.base_consumption_total
        bhi = m.base_household_income
        sr_eff = 1.0 - bct / bhi

        Cdt_new, _ = m.findemand_cd(
            theta, bct, 1.0, bhi, 0.0,
            household_closure_mode="return_to_base",
            base_household_income=bhi,
            income_tax_rate=0.5,
            c_other_coef=0.0,
            savings_rate=sr_eff,
        )
        assert Cdt_new <= 0.5 * bct + 1e-9



# TestSavingsRateBySkill
class TestSavingsRateBySkill:
    """Tests for the savings_rate_by_skill feature in ModelConfig."""

    def test_skill_weighted_savings_rate_matches_formula(self):
        """Checks that savings_rate_r[0] equals the skill-share-weighted formula."""
        config = ModelConfig(
            n_periods=10,
            time_frequency="quarterly",
            savings_rate_by_skill=np.array([0.05, 0.15, 0.30]),
        )
        m = InputOutputModel(
            n_periods=10,
            time_frequency="quarterly",
            config=config,
            _data_dict=_make_skill_data_dict(),
        )

        l0_by_skill = np.array([[10., 3., 3.], [7., 2., 2.], [3., 2., 1.]], dtype=float)
        l0_r = l0_by_skill.sum(axis=1)
        phi_r = l0_r / l0_r.sum()
        sr_sk = np.array([0.05, 0.15, 0.30])
        expected_sr = float(np.clip(1.0 - np.dot(1.0 - sr_sk, phi_r), 0.0, 0.999))

        assert abs(m.savings_rate_r[0] - expected_sr) < 1e-10

    def test_higher_skill_saving_share_raises_aggregate_savings_rate(self):
        """A workforce weighted to high savers has a higher aggregate savings rate."""
        sr_sk = np.array([0.05, 0.15, 0.30])

        data_high = _make_skill_data_dict()
        data_high["l0_by_skill"] = np.array(
            [
                [ 2., 1., 1.],
                [ 3., 2., 2.],
                [15., 4., 3.],
            ],
            dtype=np.float64,
        )

        data_low = _make_skill_data_dict()
        data_low["l0_by_skill"] = np.array(
            [
                [15., 4., 3.],
                [ 3., 2., 2.],
                [ 2., 1., 1.],
            ],
            dtype=np.float64,
        )

        config = ModelConfig(
            n_periods=10,
            time_frequency="quarterly",
            savings_rate_by_skill=sr_sk,
        )

        m_high_skill = InputOutputModel(
            n_periods=10,
            time_frequency="quarterly",
            config=config,
            _data_dict=data_high,
        )
        m_low_skill = InputOutputModel(
            n_periods=10,
            time_frequency="quarterly",
            config=config,
            _data_dict=data_low,
        )

        assert m_high_skill.savings_rate_r[0] > m_low_skill.savings_rate_r[0]

    def test_skill_savings_rate_model_runs_without_error(self):
        """A savings_rate_by_skill model runs and produces a finite, positive GDP series."""
        config = ModelConfig(
            n_periods=10,
            time_frequency="quarterly",
            savings_rate_by_skill=np.array([0.05, 0.15, 0.30]),
        )
        m = InputOutputModel(
            n_periods=10,
            time_frequency="quarterly",
            config=config,
            _data_dict=_make_skill_data_dict(),
        )

        result = m.run_model()

        assert "gdp" in result
        gdp = np.asarray(result["gdp"])
        assert np.all(np.isfinite(gdp))
        assert np.all(gdp > 0)
