"""Tests for the rationing, inventory-update, profit, and savings methods of InputOutputModel."""

from __future__ import annotations

import numpy as np
import pytest

from pyMacroIO.model import InputOutputModel
from pyMacroIO.config import ModelConfig


# Fixtures
@pytest.fixture(scope="module")
def model(minimal_data_dict) -> InputOutputModel:
    """Return a 3-sector InputOutputModel built from the minimal conftest data dictionary."""
    config = ModelConfig(n_periods=10, time_frequency="quarterly")
    return InputOutputModel(
        n_periods=10,
        time_frequency="quarterly",
        config=config,
        _data_dict=minimal_data_dict,
    )


@pytest.fixture(scope="module")
def bare_model(minimal_data_dict) -> InputOutputModel:
    """Return a model instance used by TestInventoryS for self-contained array tests."""
    config = ModelConfig(n_periods=10, time_frequency="quarterly")
    return InputOutputModel(
        n_periods=10,
        time_frequency="quarterly",
        config=config,
        _data_dict=minimal_data_dict,
    )


# Tests for allocate_deliveries
class TestAllocateDeliveriesZ:
    """Tests for the intermediate-delivery split of allocate_deliveries."""

    def test_full_supply_proportional_returns_orders(self, model: InputOutputModel) -> None:
        """Verifies that Z equals Z0 when supply equals demand under the proportional rule."""
        m = model
        Z = m.allocate_deliveries(x=m.x0, O=m.Z0, cd=m.c0, fd={}, d=m.x0, rule="no")["Z"]
        np.testing.assert_allclose(Z, m.Z0, rtol=1e-10)

    def test_half_supply_proportional_scales_rows(self, model: InputOutputModel) -> None:
        """Verifies that constraining two sectors independently scales their delivery rows."""
        m = model
        x = np.array([30.0, 20.0, 40.0])
        Z = m.allocate_deliveries(x=x, O=m.Z0, cd=m.c0, fd={}, d=m.x0, rule="no")["Z"]
        np.testing.assert_allclose(Z[0, :], np.array([0.0, 10.0, 2.5]),  rtol=1e-10)
        np.testing.assert_allclose(Z[1, :], np.array([1.5,  0.0, 7.5]),  rtol=1e-10)
        np.testing.assert_allclose(Z[2, :], m.Z0[2, :],                  rtol=1e-10)

    def test_supplier_priority_tight_constraint(self, model: InputOutputModel) -> None:
        """Verifies that supplier priority applies the fill ratio against total orders."""
        m = model
        x = np.array([15.0, 40.0, 40.0])
        Z = m.allocate_deliveries(x=x, O=m.Z0, cd=m.c0, fd={}, d=m.x0, rule="yes")["Z"]
        expected_row0 = np.array([0.0, 12.0, 3.0])
        np.testing.assert_allclose(Z[0, :], expected_row0, rtol=1e-10)
        np.testing.assert_allclose(Z[1, :], m.Z0[1, :], rtol=1e-10)
        np.testing.assert_allclose(Z[2, :], m.Z0[2, :], rtol=1e-10)

    def test_supplier_priority_more_generous_to_intermediate_users(
        self, model: InputOutputModel
    ) -> None:
        """Under constrained output, supplier priority delivers more to intermediate users."""
        m = model
        x = np.array([15.0, 40.0, 40.0])
        Z_supplier = m.allocate_deliveries(x=x, O=m.Z0, cd=m.c0, fd={}, d=m.x0, rule="yes")["Z"]
        Z_no = m.allocate_deliveries(x=x, O=m.Z0, cd=m.c0, fd={}, d=m.x0, rule="no")["Z"]
        assert Z_supplier[0, 1] > Z_no[0, 1]

    def test_zero_demand_sector_yields_no_nan(self, model: InputOutputModel) -> None:
        """A sector with zero demand and zero output propagates no NaN into its deliveries."""
        m = model
        # Sector 1: x=0, d=0; safe-divide guard prevents NaN.
        d = m.x0.copy()
        d[1] = 0.0
        x = m.x0.copy()
        x[1] = 0.0
        Z = m.allocate_deliveries(x=x, O=m.Z0, cd=m.c0, fd={}, d=d, rule="no")["Z"]
        assert not np.any(np.isnan(Z))
        # Sector 1 cannot deliver anything when its output is zero.
        np.testing.assert_allclose(Z[1, :], 0.0, atol=1e-10)

    def test_supplier_priority_zero_orders_yields_no_nan(self, model: InputOutputModel) -> None:
        """Verifies the safe-divide guard when a supplier receives no intermediate orders."""
        m = model
        O = m.Z0.copy()
        O[0, :] = 0.0
        out = m.allocate_deliveries(x=m.x0, O=O, cd=m.c0, fd={}, d=m.x0, rule="yes")
        assert not np.any(np.isnan(out["Z"]))
        assert not np.any(np.isnan(out["consumption"]))
        np.testing.assert_allclose(out["Z"][0, :], 0.0, atol=1e-10)

    def test_unknown_rule_raises(self, model: InputOutputModel) -> None:
        """Verifies that an unknown allocation rule raises ValueError."""
        m = model
        with pytest.raises(ValueError, match="rule must be one of"):
            m.allocate_deliveries(x=m.x0, O=m.Z0, cd=m.c0, fd={}, d=m.x0, rule="supplier")


class TestAllocateDeliveriesC:
    """Tests for the realised-consumption split of allocate_deliveries."""

    def test_full_supply_proportional_returns_desired(self, model: InputOutputModel) -> None:
        """Realised consumption equals desired at full supply under the proportional rule."""
        m = model
        c = m.allocate_deliveries(x=m.x0, O=m.Z0, cd=m.c0, fd={}, d=m.x0, rule="no")["consumption"]
        np.testing.assert_allclose(c, m.c0, rtol=1e-10)

    def test_half_supply_proportional_scales_constrained_sector(
        self, model: InputOutputModel
    ) -> None:
        """The half-output sector has halved consumption under the proportional rule."""
        m = model
        x = np.array([30.0, 40.0, 40.0])
        c = m.allocate_deliveries(x=x, O=m.Z0, cd=m.c0, fd={}, d=m.x0, rule="no")["consumption"]
        expected = np.array([m.c0[0] * 0.5, m.c0[1], m.c0[2]])
        np.testing.assert_allclose(c, expected, rtol=1e-10)

    def test_supplier_priority_fully_constrained_sector_zero_consumption(
        self, model: InputOutputModel
    ) -> None:
        """Consumption is zero when intermediate users absorb a sector's entire output."""
        m = model
        x = np.array([15.0, 40.0, 40.0])
        # Z[0,:] = [0,12,3]; sum = 15 = x[0], so nothing remains for households
        c = m.allocate_deliveries(x=x, O=m.Z0, cd=m.c0, fd={}, d=m.x0, rule="yes")["consumption"]
        assert np.isclose(c[0], 0.0, atol=1e-10)
        np.testing.assert_allclose(c[1:], m.c0[1:], rtol=1e-10)

    def test_supplier_priority_partial_remainder_matches_hand_computed(
        self, model: InputOutputModel
    ) -> None:
        """Verifies supplier-priority residual share when output covers orders but not demand."""
        m = model
        x = np.array([40.0, 40.0, 40.0])
        c = m.allocate_deliveries(x=x, O=m.Z0, cd=m.c0, fd={}, d=m.x0, rule="yes")["consumption"]
        expected_c0 = m.c0[0] * 15.0 / 35.0   # = 225/35
        np.testing.assert_allclose(c[0], expected_c0, rtol=1e-10)
        # Sectors 1 and 2 are at full supply; their consumption is unaffected.
        np.testing.assert_allclose(c[1:], m.c0[1:], rtol=1e-10)

    def test_supplier_priority_full_supply_returns_desired(
        self, model: InputOutputModel
    ) -> None:
        """Supplier priority at full supply matches the proportional rule."""
        m = model
        c = m.allocate_deliveries(x=m.x0, O=m.Z0, cd=m.c0, fd={}, d=m.x0, rule="yes")["consumption"]
        np.testing.assert_allclose(c, m.c0, rtol=1e-10)

    def test_consumption_non_negative_at_half_capacity(
        self, model: InputOutputModel
    ) -> None:
        """Verifies that realised consumption is non-negative under half-capacity output."""
        m = model
        x = m.x0 * 0.5
        c = m.allocate_deliveries(x=x, O=m.Z0 * 0.5, cd=m.c0, fd={}, d=m.x0, rule="no")["consumption"]
        assert np.all(c >= 0.0)


class TestAllocationIdentities:
    """Regression tests for the full delivery split of allocate_deliveries."""

    @staticmethod
    def _base_fd(m: InputOutputModel) -> dict:
        return {
            "government":  m.fd_government_[:, 0],
            "investment":  m.fd_investment_[:, 0],
            "inventories": m.fd_inventories_[:, 0],
            "exports":     m.fd_exports_[:, 0],
        }

    def _delivery_total(self, out: dict, fd: dict) -> np.ndarray:
        return (
            np.sum(out["Z"], axis=1)
            + out["consumption"]
            + sum(out[name] for name in fd)
        )

    def test_deliveries_sum_to_output_proportional(self, model: InputOutputModel) -> None:
        """All deliveries sum to feasible output per sector under the proportional rule."""
        m = model
        fd = self._base_fd(m)
        x = np.array([30.0, 20.0, 40.0])
        out = m.allocate_deliveries(x=x, O=m.Z0, cd=m.c0, fd=fd, d=m.x0, rule="no")
        np.testing.assert_allclose(self._delivery_total(out, fd), x, rtol=1e-10)

    def test_deliveries_sum_to_output_supplier(self, model: InputOutputModel) -> None:
        """All deliveries sum to feasible output per sector under supplier priority."""
        m = model
        fd = self._base_fd(m)
        for x in (np.array([15.0, 40.0, 40.0]), np.array([40.0, 30.0, 40.0]), m.x0):
            out = m.allocate_deliveries(x=x, O=m.Z0, cd=m.c0, fd=fd, d=m.x0, rule="yes")
            np.testing.assert_allclose(self._delivery_total(out, fd), x, rtol=1e-10)

    def test_proportional_rule_matches_closed_form(self, model: InputOutputModel) -> None:
        """Verifies the split against the pre-refactor form O*s, cd*s, fd*s, s = clip(x/d, 0, 1)."""
        m = model
        fd = self._base_fd(m)
        x = np.array([30.0, 20.0, 40.0])
        out = m.allocate_deliveries(x=x, O=m.Z0, cd=m.c0, fd=fd, d=m.x0, rule="no")
        s = np.clip(x / m.x0, 0, 1)
        np.testing.assert_allclose(out["Z"], m.Z0 * s[:, np.newaxis], rtol=1e-10)
        np.testing.assert_allclose(out["consumption"], m.c0 * s, rtol=1e-10)
        for name, comp in fd.items():
            np.testing.assert_allclose(out[name], comp * s, rtol=1e-10)


class TestRunLevelIdentities:
    """Run-level adding-up and GDP identity tests on the minimal model."""

    def _run(self, minimal_data_dict: dict, **cfg) -> tuple:
        config = ModelConfig(n_periods=6, time_frequency="quarterly", **cfg)
        m = InputOutputModel(
            n_periods=6, time_frequency="quarterly",
            config=config, _data_dict=minimal_data_dict,
        )
        m.apply_output_constraint_shock("A", 2, 0.4)
        m.apply_output_constraint_shock("A", 3, 0.4)
        r = m.run_model(store_full_matrices=True)
        return m, r

    def _assert_identities(self, m: InputOutputModel, r: dict) -> None:
        x = r["gross_output"]
        finals = (
            r["realised_consumption"]
            + r["realised_government"]
            + r["realised_investment"]
            + r["realised_inventories"]
            + r["realised_exports"]
            + r["realised_other"]
        )
        for t in range(1, m.TT):
            Z_t = r["intermediate_deliveries"][:, :, t]
            np.testing.assert_allclose(
                np.sum(Z_t, axis=1) + finals[:, t], x[:, t], atol=1e-8,
                err_msg=f"deliveries do not sum to feasible output at t={t}",
            )
            dS_t = np.sum(r["inventories"][:, :, t] - r["inventories"][:, :, t - 1])
            np.testing.assert_allclose(
                r["gdp"][t],
                np.sum(finals[:, t]) + dS_t - np.sum(m.imp_share * x[:, t]),
                atol=1e-6,
                err_msg=f"expenditure-production GDP identity fails at t={t}",
            )

    def test_identities_default_closures(self, minimal_data_dict: dict) -> None:
        """Verifies adding-up and the GDP identity under default closures and proportional rule."""
        m, r = self._run(minimal_data_dict)
        self._assert_identities(m, r)

    def test_identities_endogenous_closures(self, minimal_data_dict: dict) -> None:
        """Verifies adding-up and the GDP identity with endogenous government and investment."""
        m, r = self._run(
            minimal_data_dict,
            investment_closure="keynesian",
            gov_income_elasticity=0.5,
        )
        self._assert_identities(m, r)


# Tests for inventory_S
class TestInventoryS:
    """Tests for the inventory_S end-of-period update, on self-contained hand-computed arrays."""

    def test_delivery_covers_use_exactly_leaves_S_unchanged(
        self, bare_model: InputOutputModel
    ) -> None:
        """Inventories are unchanged when deliveries exactly cover production requirements."""
        m = bare_model
        x = np.array([10.0, 10.0])
        S = np.array([[5.0, 4.0], [3.0, 2.0]])
        Z = np.array([[1.0, 0.0], [0.0, 1.0]])
        A = np.array([[0.1, 0.0], [0.0, 0.1]])
        S_new = m.inventory_S(x=x, S=S, Z=Z, A=A)
        np.testing.assert_allclose(S_new, S, rtol=1e-10)

    def test_no_deliveries_depletes_stocks(
        self, bare_model: InputOutputModel
    ) -> None:
        """Inventories fall by production requirements when no deliveries arrive."""
        m = bare_model
        x = np.array([10.0, 10.0])
        S = np.array([[1.0, 2.0], [3.0, 1.0]])
        Z = np.zeros((2, 2))
        A = np.array([[0.2, 0.1], [0.1, 0.2]])
        S_new = m.inventory_S(x=x, S=S, Z=Z, A=A)
        expected = np.array([[0.0, 1.0], [2.0, 0.0]])
        np.testing.assert_allclose(S_new, expected, rtol=1e-10)

    def test_excess_deliveries_accumulate_stocks(
        self, bare_model: InputOutputModel
    ) -> None:
        """Verifies that inventories accumulate when deliveries exceed production requirements."""
        m = bare_model
        x = np.array([5.0, 5.0])
        S = np.array([[1.0, 0.0], [0.0, 1.0]])
        Z = np.array([[2.0, 1.0], [1.0, 2.0]])
        A = np.array([[0.1, 0.1], [0.1, 0.1]])
        S_new = m.inventory_S(x=x, S=S, Z=Z, A=A)
        expected = np.array([[2.5, 0.5], [0.5, 2.5]])
        np.testing.assert_allclose(S_new, expected, rtol=1e-10)

    def test_floor_binds_giving_non_negative_inventories(
        self, bare_model: InputOutputModel
    ) -> None:
        """Verifies that the zero floor is binding and no inventory entry turns negative."""
        m = bare_model
        x = np.array([10.0, 10.0])
        S = np.array([[1.0, 2.0], [3.0, 1.0]])
        Z = np.zeros((2, 2))
        A = np.array([[0.2, 0.1], [0.1, 0.2]])
        S_new = m.inventory_S(x=x, S=S, Z=Z, A=A)
        assert np.all(S_new >= 0.0)


# Tests for profit_pi
class TestProfitPi:
    """Tests for the profit_pi accounting identity method."""

    def test_doubling_inputs_doubles_profits(self, model: InputOutputModel) -> None:
        """Verifies that proportional doubling of output and all inputs doubles profits."""
        m = model
        x2 = 2.0 * m.x0
        Z2 = 2.0 * m.Z0
        l2 = 2.0 * m.l0
        pi2 = m.profit_pi(x=x2, Z=Z2, l=l2)
        np.testing.assert_allclose(pi2, 2.0 * m.profits0, rtol=1e-10)

    def test_halved_capital_productivity_reduces_profits(
        self, model: InputOutputModel
    ) -> None:
        """Halving sector 0 capital productivity cuts its profit below the base-year level."""
        m = model
        fprod_k = np.array([0.5, 1.0, 1.0])
        pi_base = m.profit_pi(x=m.x0, Z=m.Z0, l=m.l0)
        pi_reduced = m.profit_pi(x=m.x0, Z=m.Z0, l=m.l0, fprod_k=fprod_k)
        assert pi_reduced[0] < pi_base[0]
        assert np.isclose(pi_reduced[0], -10.0, rtol=1e-10)
        np.testing.assert_allclose(pi_reduced[1:], pi_base[1:], rtol=1e-10)

    def test_zero_output_gives_zero_profit(self, model: InputOutputModel) -> None:
        """Profit is zero when output, intermediate inputs, and labour are all zero."""
        m = model
        x = np.zeros(m.N)
        Z = np.zeros((m.N, m.N))
        l = np.zeros(m.N)
        pi = m.profit_pi(x=x, Z=Z, l=l)
        np.testing.assert_allclose(pi, np.zeros(m.N), atol=1e-10)


# Tests for savings_s_regional
class TestSavingsRegional:
    """Tests for the savings_s_regional per-region household savings method."""

    def test_known_income_and_consumption(self, model: InputOutputModel) -> None:
        """Verifies the savings formula for a known income and consumption pair."""
        m = model
        income_r = np.array([100.0])
        c_r = np.array([45.0])
        result = m.savings_s_regional(household_income_r_t=income_r, c_r_t=c_r)
        assert np.isclose(result[0], 50.0, rtol=1e-10)

    def test_higher_consumption_reduces_savings(self, model: InputOutputModel) -> None:
        """Verifies that savings fall when consumption rises, holding income fixed."""
        m = model
        income = np.array([100.0])
        s_low = m.savings_s_regional(
            household_income_r_t=income, c_r_t=np.array([45.0])
        )
        s_high = m.savings_s_regional(
            household_income_r_t=income, c_r_t=np.array([81.0])
        )
        assert s_high[0] < s_low[0]

    def test_zero_consumption_savings_equals_income(self, model: InputOutputModel) -> None:
        """Verifies that savings equal income when consumption is zero."""
        m = model
        income = np.array([100.0])
        c = np.array([0.0])
        result = m.savings_s_regional(household_income_r_t=income, c_r_t=c)
        assert np.isclose(result[0], 100.0, rtol=1e-10)

    def test_output_shape_is_n_regions(self, model: InputOutputModel) -> None:
        """The return array has shape (n_regions,), here (1,) for the single-region model."""
        m = model
        income = np.array([100.0])
        c = np.array([45.0])
        result = m.savings_s_regional(household_income_r_t=income, c_r_t=c)
        assert result.shape == (m.n_regions,)
        assert result.shape == (1,)
