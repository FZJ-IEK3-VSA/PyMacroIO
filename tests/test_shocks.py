"""Tests for shock application methods of InputOutputModel."""

from __future__ import annotations

import numpy as np
import pytest

from pyMacroIO.model import InputOutputModel
from pyMacroIO.config import ModelConfig


# Fresh-model helper
def _fresh_model(data_dict: dict) -> InputOutputModel:
    """Return a freshly constructed InputOutputModel from the given data dictionary."""
    config = ModelConfig(n_periods=10, time_frequency="quarterly")
    return InputOutputModel(
        n_periods=10,
        time_frequency="quarterly",
        config=config,
        _data_dict=data_dict,
    )


# Rationing branch
class TestRationingBranch:
    """Rationing branch of allocate_deliveries, active when t is in rationing_shocks_."""

    def test_rationing_overrides_intermediate_deliveries(
        self, minimal_data_dict: dict
    ) -> None:
        """Z[0,:] equals Z0[0,:]*capacity_pct when a rationing shock is registered for t."""
        m = _fresh_model(minimal_data_dict)
        m.apply_rationing_shock("A", time_period=5, capacity_pct=0.5)
        Z = m.allocate_deliveries(
            x=m.x0, O=m.Z0, cd=m.c0, fd={}, d=m.x0, rule="no", t=5
        )["Z"]
        np.testing.assert_allclose(Z[0, :], m.Z0[0, :] * 0.5, rtol=1e-10)

    def test_rationing_leaves_other_sectors_unaffected_in_Z(
        self, minimal_data_dict: dict
    ) -> None:
        """Verifies that sectors not covered by the rationing shock are unaffected in Z."""
        m = _fresh_model(minimal_data_dict)
        m.apply_rationing_shock("A", time_period=5, capacity_pct=0.5)
        Z = m.allocate_deliveries(
            x=m.x0, O=m.Z0, cd=m.c0, fd={}, d=m.x0, rule="no", t=5
        )["Z"]
        # With x=x0 and d=x0, s=1 and the normal formula gives Z=Z0 for unshocked rows.
        np.testing.assert_allclose(Z[1, :], m.Z0[1, :], rtol=1e-10)
        np.testing.assert_allclose(Z[2, :], m.Z0[2, :], rtol=1e-10)

    def test_no_rationing_at_unregistered_period_leaves_Z_unchanged(
        self, minimal_data_dict: dict
    ) -> None:
        """An unregistered period t leaves Z equal to the no-rationing result."""
        m = _fresh_model(minimal_data_dict)
        m.apply_rationing_shock("A", time_period=5, capacity_pct=0.5)
        # t=3 has no registered shock, so the branch is not entered.
        Z = m.allocate_deliveries(
            x=m.x0, O=m.Z0, cd=m.c0, fd={}, d=m.x0, rule="no", t=3
        )["Z"]
        np.testing.assert_allclose(Z, m.Z0, rtol=1e-10)

    def test_rationing_overrides_household_consumption(
        self, minimal_data_dict: dict
    ) -> None:
        """Verifies that c[0] equals c0[0]*capacity_pct when include_households=True."""
        m = _fresh_model(minimal_data_dict)
        m.apply_rationing_shock("A", time_period=5, capacity_pct=0.5, include_households=True)
        c = m.allocate_deliveries(
            x=m.x0, O=m.Z0, cd=m.c0, fd={}, d=m.x0, rule="no", t=5
        )["consumption"]
        assert np.isclose(c[0], m.c0[0] * 0.5, rtol=1e-10)
        np.testing.assert_allclose(c[1:], m.c0[1:], rtol=1e-10)

    def test_rationing_without_households_does_not_override_c(
        self, minimal_data_dict: dict
    ) -> None:
        """Household consumption stays at its normal level when include_households=False."""
        m = _fresh_model(minimal_data_dict)
        m.apply_rationing_shock("A", time_period=5, capacity_pct=0.5, include_households=False)
        out = m.allocate_deliveries(
            x=m.x0, O=m.Z0, cd=m.c0, fd={}, d=m.x0, rule="no", t=5
        )
        np.testing.assert_allclose(out["Z"][0, :], m.Z0[0, :] * 0.5, rtol=1e-10)
        assert np.isclose(out["consumption"][0], m.c0[0], rtol=1e-5)

    def test_override_is_reconciled_against_realised_output(
        self, minimal_data_dict: dict
    ) -> None:
        """Verifies that orders above baseline cannot lift deliveries above feasible output."""
        m = _fresh_model(minimal_data_dict)
        m.apply_rationing_shock("A", time_period=5, capacity_pct=1.0)
        x  = m.x0 * 0.1
        fd = {"government": m.x0 * 0.05}
        out = m.allocate_deliveries(
            x=x, O=m.Z0 * 5, cd=m.c0, fd=fd, d=m.x0, rule="no", t=5
        )
        delivered = (
            out["Z"][0, :].sum() + out["consumption"][0] + out["government"][0]
        )
        assert delivered <= x[0] + 1e-9, (
            "Deliveries from the rationed sector exceed its realised output."
        )
        # Unrationed rows keep the normal proportional split.
        np.testing.assert_allclose(out["Z"][1, :], m.Z0[1, :] * 5 * 0.1, rtol=1e-10)


# Consumption shock
class TestConsumptionShock:
    """Verifies that apply_consumption_shock reduces demand and GDP during the shock window."""

    def test_consumption_shock_reduces_gdp_in_shocked_window(
        self, minimal_data_dict: dict
    ) -> None:
        """Checks that GDP falls in the shock window relative to the pre-shock period."""
        m = _fresh_model(minimal_data_dict)
        m.apply_consumption_shock(start=3, duration=3, intensity=0.5)
        result = m.run_model()
        assert result["gdp"][3] < result["gdp"][0]

    def test_consumption_shock_gdp_recovers_after_window(
        self, minimal_data_dict: dict
    ) -> None:
        """Checks that GDP rises after the shock window ends relative to the shocked period."""
        m = _fresh_model(minimal_data_dict)
        m.apply_consumption_shock(start=3, duration=3, intensity=0.5)
        result = m.run_model()
        assert result["gdp"][9] > result["gdp"][3]

    def test_consumption_shock_reduces_realised_consumption(
        self, minimal_data_dict: dict
    ) -> None:
        """Checks that total realised consumption is lower in the shocked period than at t=0."""
        m = _fresh_model(minimal_data_dict)
        m.apply_consumption_shock(start=3, duration=3, intensity=0.5)
        result = m.run_model()
        assert (
            np.sum(result["realised_consumption"][:, 3])
            < np.sum(result["realised_consumption"][:, 0])
        )


# Input availability shock
class TestInputAvailabilityShock:
    """Verifies that apply_input_availability_shock constrains the affected sector's output."""

    def test_input_availability_shock_caps_output_in_shocked_period(
        self, minimal_data_dict: dict
    ) -> None:
        """Affected sector output stays within (1-reduction_pct)*x0 at the shocked period."""
        m = _fresh_model(minimal_data_dict)
        m.apply_input_availability_shock("A", time_period=4, reduction_pct=0.5)
        result = m.run_model()
        assert result["gross_output"][0, 4] <= m.x0[0] * 0.5 + 1e-6

    def test_input_availability_shock_unaffected_period_has_normal_output(
        self, minimal_data_dict: dict
    ) -> None:
        """Checks that gross output for the affected sector is near x0 in an unshocked period."""
        m = _fresh_model(minimal_data_dict)
        m.apply_input_availability_shock("A", time_period=4, reduction_pct=0.5)
        result = m.run_model()
        assert result["gross_output"][0, 3] > m.x0[0] * 0.95


# Technical change
class TestTechnicalChange:
    """Verifies that apply_technical_change alters the steady state from the changed period."""

    def test_halving_A_reduces_intermediate_costs_from_changed_period(
        self, minimal_data_dict: dict
    ) -> None:
        """Halving the technical coefficients from t=5 reduces intermediate deliveries."""
        m = _fresh_model(minimal_data_dict)
        new_A = m.A * 0.5
        m.apply_technical_change(t=5, new_A=new_A)
        result = m.run_model()
        # Z column sums at t=9 should fall below t=4, just before the A change.
        assert result["Z_colsums"][:, 9].sum() < result["Z_colsums"][:, 4].sum() * 0.9, (
            "Total intermediate inputs should be substantially lower after halving A."
        )

    def test_invalid_A_column_sum_raises_value_error(
        self, minimal_data_dict: dict
    ) -> None:
        """Verifies that a new_A with column sums >= 1 raises ValueError."""
        m = _fresh_model(minimal_data_dict)
        with pytest.raises(ValueError):
            m.apply_technical_change(t=3, new_A=np.ones((3, 3)))


# Factor productivity shock
class TestFactorProductivityShock:
    """Verifies that apply_factor_productivity_shock alters the labour capacity constraint."""

    def test_labour_productivity_reduction_lowers_output(
        self, minimal_data_dict: dict
    ) -> None:
        """Halving labour productivity for sector A reduces its output at the shocked period."""
        m = _fresh_model(minimal_data_dict)
        m.apply_factor_productivity_shock("A", t=4, prod_L=0.5)
        result = m.run_model()
        # With productivity halved, sector A produces at most 0.5 * x0[0] from labour alone.
        assert result["gross_output"][0, 4] < m.x0[0] - 1e-3, (
            "Sector A output should fall when labour productivity is halved."
        )
        # Finite hiring speed means t=3 output may differ from x0, so compare an unshocked run.
        m_ref = _fresh_model(minimal_data_dict)
        ref = m_ref.run_model()
        np.testing.assert_allclose(
            result["gross_output"][0, 3],
            ref["gross_output"][0, 3],
            rtol=1e-10,
            err_msg="Sector A output at t=3 should be unaffected by a shock at t=4.",
        )

    def test_invalid_sector_label_raises_value_error(
        self, minimal_data_dict: dict
    ) -> None:
        """Verifies that an unrecognised sector label raises ValueError."""
        m = _fresh_model(minimal_data_dict)
        with pytest.raises(ValueError):
            m.apply_factor_productivity_shock("Z", t=3, prod_L=1.5)



# Bilateral delivery shock
def _bilateral_model(data_dict: dict, **config_kwargs) -> InputOutputModel:
    """Return a freshly constructed two-region model for bilateral-shock tests."""
    config = ModelConfig(
        n_periods=10, time_frequency="quarterly", n_regions=2,
        region_labels=["R0", "R1"], **config_kwargs,
    )
    return InputOutputModel(
        n_periods=10, time_frequency="quarterly", config=config, _data_dict=data_dict,
    )


class TestBilateralDeliveryShock:
    """Verifies apply_bilateral_delivery_shock: registration, binding, and equivalence."""

    def test_uncut_registration_reproduces_the_baseline_exactly(
        self, bilateral_data_dict: dict
    ) -> None:
        """Verifies that a cap at the baseline level leaves every series untouched."""
        ref = _bilateral_model(bilateral_data_dict).run_model(store_full_matrices=True)
        m   = _bilateral_model(bilateral_data_dict)
        for t in range(1, m.TT):
            m.apply_bilateral_delivery_shock("R0:gas", "R1", t, capacity_pct=1.0)
        result = m.run_model(store_full_matrices=True)
        J_R1 = m.region_sector_indices[1]
        # Exact equality below only follows if the unshocked run never orders above base.
        assert np.all(ref["orders"][0, J_R1, :].sum(axis=0) <= m.Z0[0, J_R1].sum())
        for key in ("gross_output", "gdp", "realised_consumption", "Z_colsums"):
            np.testing.assert_array_equal(result[key], ref[key])

    def test_cut_binds_intermediate_and_household_flows(
        self, bilateral_data_dict: dict
    ) -> None:
        """Verifies that both buyer classes in the named region are held at the capped level."""
        m = _bilateral_model(bilateral_data_dict)
        J_R0, J_R1 = m.region_sector_indices
        for t in range(3, m.TT):
            m.apply_bilateral_delivery_shock("R0:gas", "R1", t, capacity_pct=0.25)
        result = m.run_model(store_full_matrices=True)
        ref    = _bilateral_model(bilateral_data_dict).run_model(store_full_matrices=True)

        cap_int = 0.25 * m.Z0[0, J_R1].sum()
        for t in range(3, m.TT):
            assert result["intermediate_deliveries"][0, J_R1, t].sum() <= cap_int + 1e-9
        assert (
            result["realised_consumption"][0, 5]
            <= m.cons_vec_r[0, 0] + 0.25 * m.cons_vec_r[0, 1] + 1e-9
        )
        np.testing.assert_allclose(
            result["intermediate_deliveries"][0, J_R0, 3],
            ref["intermediate_deliveries"][0, J_R0, 3],
            rtol=1e-10,
            err_msg="A shock aimed at region 1 must leave region 0 deliveries alone.",
        )

    def test_cut_to_every_buyer_region_matches_the_rationing_shock(
        self, bilateral_data_dict: dict
    ) -> None:
        """Verifies equivalence with apply_rationing_shock at the first shocked period."""
        # They diverge afterwards, as rationing also flags hire_fire and caps off baseline output.
        modes = ["frozen", "frozen"]
        bil = _bilateral_model(bilateral_data_dict, household_closure_mode=modes)
        for r in range(bil.n_regions):
            bil.apply_bilateral_delivery_shock("R0:gas", r, 1, capacity_pct=0.5)
        rat = _bilateral_model(bilateral_data_dict, household_closure_mode=modes)
        rat.apply_rationing_shock("R0:gas", time_period=1, capacity_pct=0.5)

        b = bil.run_model(store_full_matrices=True)
        r = rat.run_model(store_full_matrices=True)
        np.testing.assert_allclose(
            b["gross_output"][0, 1], r["gross_output"][0, 1], rtol=1e-10
        )
        np.testing.assert_allclose(
            b["intermediate_deliveries"][0, :, 1],
            r["intermediate_deliveries"][0, :, 1], rtol=1e-10,
        )
        np.testing.assert_allclose(
            b["realised_consumption"][0, 1], r["realised_consumption"][0, 1], rtol=1e-10
        )

    def test_stress_run_respects_the_cap_and_the_accounting_identities(
        self, bilateral_data_dict: dict
    ) -> None:
        """Verifies that a full cut holds deliveries, output and inventories consistent."""
        m = _bilateral_model(bilateral_data_dict)
        J_R1 = m.region_sector_indices[1]
        for t in range(3, m.TT):
            m.apply_bilateral_delivery_shock("R0:gas", "R1", t, capacity_pct=0.0)
        result = m.run_model(store_full_matrices=True, validate=True)

        Z, S, x = (result[k] for k in
                   ("intermediate_deliveries", "inventories", "gross_output"))
        for t in range(1, m.TT):
            delivered = Z[:, :, t].sum(axis=1) + sum(
                result[k][:, t] for k in (
                    "realised_consumption", "realised_government", "realised_investment",
                    "realised_inventories", "realised_exports", "realised_other",
                )
            )
            assert np.all(delivered <= x[:, t] + 1e-6), f"Deliveries exceed output at t={t}."
            assert np.all(S[:, :, t] >= 0.0), f"Negative inventories at t={t}."
        for t in range(3, m.TT):
            assert Z[0, J_R1, t].sum() <= 1e-12, f"Cut flow still delivered at t={t}."
            assert np.all(S[0, J_R1, t] <= S[0, J_R1, t - 1] + 1e-12), (
                f"Inventories stocked from a severed flow at t={t}."
            )

    def test_replenishment_orders_are_bound_by_the_cap(
        self, bilateral_data_dict: dict
    ) -> None:
        """Verifies that the cap binds the delivery level, not the use component alone."""
        m = _bilateral_model(bilateral_data_dict)
        J_R1 = m.region_sector_indices[1]
        for t in range(3, 6):
            m.apply_bilateral_delivery_shock("R0:gas", "R1", t, capacity_pct=0.0)
        for t in range(6, m.TT):
            m.apply_bilateral_delivery_shock("R0:gas", "R1", t, capacity_pct=0.4)
        result = m.run_model(store_full_matrices=True)

        S_tar = m.A * m.x0[np.newaxis, :] * m.n[np.newaxis, :]
        assert result["inventories"][0, J_R1, 5].sum() < S_tar[0, J_R1].sum(), (
            "The drawdown should leave a replenishment gap for the capped periods."
        )
        cap = 0.4 * m.Z0[0, J_R1].sum()
        np.testing.assert_allclose(
            result["orders"][0, J_R1, 6].sum(), cap, rtol=1e-10,
            err_msg="Replenishment demand should be bound to the cap, not pass it.",
        )
        assert result["intermediate_deliveries"][0, J_R1, 6].sum() <= cap + 1e-9

    def test_household_rate_can_differ_from_the_industry_rate(
        self, bilateral_data_dict: dict
    ) -> None:
        """Verifies that the household buyer class takes its own rate when one is given."""
        consumed = {}
        for hh_pct in (0.25, 1.0):
            m = _bilateral_model(bilateral_data_dict)
            J_R1 = m.region_sector_indices[1]
            for t in range(3, m.TT):
                m.apply_bilateral_delivery_shock(
                    "R0:gas", "R1", t, capacity_pct=0.25, household_capacity_pct=hh_pct
                )
            result = m.run_model(store_full_matrices=True)
            assert (
                result["intermediate_deliveries"][0, J_R1, 5].sum()
                <= 0.25 * m.Z0[0, J_R1].sum() + 1e-9
            )
            consumed[hh_pct] = result["realised_consumption"][0, 5]
        assert consumed[1.0] > consumed[0.25] + 1e-6, (
            "Sparing households should leave more of the good consumed."
        )

    def test_registration_overwrites_the_same_period_and_pair(
        self, bilateral_data_dict: dict
    ) -> None:
        """Verifies that a second registration replaces the first for the same pair."""
        m = _bilateral_model(bilateral_data_dict)
        m.apply_bilateral_delivery_shock("R0:gas", "R1", 4, capacity_pct=0.5)
        m.apply_bilateral_delivery_shock("R0:gas", "R1", 4, capacity_pct=0.1)
        assert m.bilateral_delivery_shocks_[4] == {(0, 1): (0.1, 0.1)}

    def test_region_label_and_index_are_equivalent(
        self, bilateral_data_dict: dict
    ) -> None:
        """Verifies that a buyer region may be named by label or by index."""
        m = _bilateral_model(bilateral_data_dict)
        assert (
            m.apply_bilateral_delivery_shock("R0:gas", "R1", 4, capacity_pct=0.5)
            == m.apply_bilateral_delivery_shock("R0:gas", 1, 4, capacity_pct=0.5)
        )

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"supplier_sector_label": "nope", "buyer_region": "R1", "time_period": 4,
             "capacity_pct": 0.5},
            {"supplier_sector_label": "R0:gas", "buyer_region": "nope", "time_period": 4,
             "capacity_pct": 0.5},
            {"supplier_sector_label": "R0:gas", "buyer_region": 7, "time_period": 4,
             "capacity_pct": 0.5},
            {"supplier_sector_label": "R0:gas", "buyer_region": "R1", "time_period": 99,
             "capacity_pct": 0.5},
            {"supplier_sector_label": "R0:gas", "buyer_region": "R1", "time_period": 4,
             "capacity_pct": 1.5},
            {"supplier_sector_label": "R0:gas", "buyer_region": "R1", "time_period": 4,
             "capacity_pct": 0.5, "household_capacity_pct": -0.1},
        ],
    )
    def test_invalid_arguments_raise_value_error(
        self, bilateral_data_dict: dict, kwargs: dict
    ) -> None:
        """Verifies that out-of-range or unknown arguments are rejected."""
        m = _bilateral_model(bilateral_data_dict)
        with pytest.raises(ValueError):
            m.apply_bilateral_delivery_shock(**kwargs)


# Import supplement cap
def _supplement_model(data_dict: dict, import_flexibility: float = 0.8) -> InputOutputModel:
    """Return a 2-region RoW model with the import supplement channel open."""
    config = ModelConfig(
        n_periods=12, time_frequency="quarterly",
        import_flexibility=import_flexibility,
    )
    return InputOutputModel(
        n_periods=12, time_frequency="quarterly", config=config, _data_dict=data_dict,
    )


class TestImportSupplementCap:
    """Verifies apply_import_supplement_cap: registration, binding, and the frozen channel."""

    @staticmethod
    def _shocked(m: InputOutputModel) -> InputOutputModel:
        """Ration sector A so its buyers run an inventory shortfall on row A."""
        for t in range(2, m.TT):
            m.apply_rationing_shock("A", time_period=t, capacity_pct=0.05)
        return m

    @staticmethod
    def _sourced_from(m: InputOutputModel, result: dict, row_label: str) -> np.ndarray:
        """Per-period supplement drawn from one RoW row, summed over the inputs it serves."""
        row_idx = m.label_to_index[row_label]
        rows = np.where(m.good_to_row_idx == row_idx)[0]
        return result["import_supplement_by_input"][rows, :].sum(axis=0)

    def test_absent_and_unbinding_caps_reproduce_the_uncapped_run_exactly(
        self, row_data_dict: dict
    ) -> None:
        """Verifies that no registration and a never-binding cap both leave the run untouched."""
        ref = self._shocked(_supplement_model(row_data_dict)).run_model()
        m = self._shocked(_supplement_model(row_data_dict))
        for t in range(m.TT):
            m.apply_import_supplement_cap("RoW:A", t, cap_level=1e9)
            m.apply_import_supplement_cap("RoW:B", t, cap_level=1e9)
        result = m.run_model()
        for key in ("gross_output", "gdp", "import_supplement",
                    "import_supplement_by_input", "realised_consumption"):
            np.testing.assert_array_equal(result[key], ref[key])

    def test_zero_cap_reproduces_a_frozen_channel(self, row_data_dict: dict) -> None:
        """Verifies that a zero cap on every RoW row equals import_flexibility of zero."""
        frozen = self._shocked(
            _supplement_model(row_data_dict, import_flexibility=0.0)
        ).run_model()
        m = self._shocked(_supplement_model(row_data_dict))
        for t in range(m.TT):
            m.apply_import_supplement_cap("RoW:A", t, cap_level=0.0)
            m.apply_import_supplement_cap("RoW:B", t, cap_level=0.0)
        result = m.run_model()
        for key in ("gross_output", "gdp", "import_supplement",
                    "import_supplement_by_input", "realised_consumption"):
            np.testing.assert_allclose(result[key], frozen[key], rtol=0, atol=1e-12)

    def test_binding_cap_holds_the_supplement_at_the_capacity_path(
        self, row_data_dict: dict
    ) -> None:
        """Verifies that the supplement from a capped row never exceeds its capacity path."""
        loose_m = self._shocked(_supplement_model(row_data_dict))
        loose = loose_m.run_model()
        sourced_loose = self._sourced_from(loose_m, loose, "RoW:A")
        cap = 0.5 * float(sourced_loose.max())
        assert cap > 0.0, "The uncapped run must draw a supplement for the test to bind."

        m = self._shocked(_supplement_model(row_data_dict))
        for t in range(m.TT):
            m.apply_import_supplement_cap("RoW:A", t, cap_level=cap)
        result = m.run_model()
        sourced = self._sourced_from(m, result, "RoW:A")
        assert np.all(sourced <= cap + 1e-9), "The supplement exceeds the capacity path."
        assert np.any(np.isclose(sourced, cap, rtol=1e-6)), (
            "The cap never binds, so the test is vacuous."
        )

    def test_registration_overwrites_the_same_row_and_period(
        self, row_data_dict: dict
    ) -> None:
        """Verifies the overwrite semantics shared with the other shock setters."""
        m = _supplement_model(row_data_dict)
        m.apply_import_supplement_cap("RoW:A", 4, cap_level=3.0)
        row_idx, level = m.apply_import_supplement_cap("RoW:A", 4, cap_level=1.0)
        assert m.import_supplement_caps_[4] == {row_idx: 1.0}
        assert level == 1.0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"supplier_sector_label": "nope", "time_period": 4, "cap_level": 1.0},
            {"supplier_sector_label": "RoW:A", "time_period": 99, "cap_level": 1.0},
            {"supplier_sector_label": "RoW:A", "time_period": 4, "cap_level": -1.0},
            {"supplier_sector_label": "RoW:A", "time_period": 4, "cap_level": float("nan")},
        ],
    )
    def test_invalid_arguments_raise_value_error(
        self, row_data_dict: dict, kwargs: dict
    ) -> None:
        """Unknown labels and bad values are rejected. Any source row may carry a cap."""
        m = _supplement_model(row_data_dict)
        with pytest.raises(ValueError):
            m.apply_import_supplement_cap(**kwargs)


class TestHouseholdSupplement:
    """Verifies the household supplement bound, budgets, and source credit."""

    def _capped(self, data_dict: dict, cover: float | None = None) -> InputOutputModel:
        m = _supplement_model(data_dict, import_flexibility=0.5)
        for t in range(3, 12):
            m.apply_bilateral_delivery_shock("A", "domestic", t, capacity_pct=0.0)
        if cover is not None:
            m.hh_supplement_cover = np.full(m.N, cover)
        return m

    def test_supplement_bounded_by_shortfall(self, row_data_dict: dict) -> None:
        m = self._capped(row_data_dict, cover=50.0)
        res = m.run_model()
        short = float(m.cons_vec_r[m.label_to_index["A"], 0])
        assert res["household_supplement"].sum(axis=0).max() <= short + 1e-9

    def test_zero_cap_leaves_baseline(self, row_data_dict: dict) -> None:
        base = _supplement_model(row_data_dict, import_flexibility=0.5).run_model()
        run = self._capped(row_data_dict)
        for t in range(3, 12):
            run.apply_import_supplement_cap("RoW:A", t, 0.0)
        res = run.run_model()
        assert res["household_supplement"].sum() == 0.0
        assert np.allclose(base["gdp"][:3], res["gdp"][:3])

    def test_multi_source_spill_respects_caps(self, row_data_dict: dict) -> None:
        m = self._capped(row_data_dict, cover=50.0)
        m.set_supplement_sources("A", ["RoW:B", "RoW:A"])
        for t in range(3, 12):
            m.apply_import_supplement_cap("RoW:B", t, 1.0)
        res = m.run_model()
        taken = res["household_supplement"] + res["supplement_by_source"]
        assert taken[m.label_to_index["RoW:B"], 3:].max() <= 1.0 + 1e-9
        assert taken[m.label_to_index["RoW:A"], 3:].sum() > 0.0


class TestInitialInventoryFill:
    def _leontief_model(self, data_dict, fill=None):
        config = ModelConfig(n_periods=6, time_frequency="quarterly",
                             prod_function="leontief")
        m = InputOutputModel(n_periods=6, time_frequency="quarterly",
                             config=config, _data_dict=data_dict)
        if fill is not None:
            m.set_initial_inventory_fill("A", "B", fill)
        return m

    def test_empty_stock_binds_leontief(self, minimal_data_dict) -> None:
        """Starting one input stock empty caps the buyer at that stock."""
        base = self._leontief_model(minimal_data_dict).run_model()
        starved = self._leontief_model(minimal_data_dict, 0.0).run_model()
        b = 1
        assert starved["gross_output"][b, 1] < 0.1 * base["gross_output"][b, 1]

    def test_full_fill_is_identity(self, minimal_data_dict) -> None:
        base = self._leontief_model(minimal_data_dict).run_model()
        filled = self._leontief_model(minimal_data_dict, 1.0).run_model()
        np.testing.assert_allclose(filled["gross_output"],
                                   base["gross_output"], rtol=1e-12)

    def test_validation(self, minimal_data_dict) -> None:
        m = self._leontief_model(minimal_data_dict)
        with pytest.raises(ValueError, match="not found"):
            m.set_initial_inventory_fill("Z", "B", 0.5)
        with pytest.raises(ValueError, match="fraction"):
            m.set_initial_inventory_fill("A", "B", 1.5)


class TestEssentialInputsOverride:
    def test_override_respected(self, minimal_data_dict) -> None:
        config = ModelConfig(n_periods=5, time_frequency="quarterly",
                             prod_function="leontief.adapted")
        m = InputOutputModel(n_periods=5, time_frequency="quarterly",
                             config=config, _data_dict=minimal_data_dict)
        override = np.zeros((m.N, m.N)); override[0, 1] = 1.0
        m.set_essential_inputs(override)
        assert m.A_essential[0, 1] == 1 and m.A_essential.sum() == 1
        res = m.run_model()
        assert np.all(np.isfinite(res["gross_output"]))

    def test_shape_and_empty_raise(self, minimal_data_dict) -> None:
        config = ModelConfig(n_periods=5, time_frequency="quarterly",
                             prod_function="leontief.adapted")
        m = InputOutputModel(n_periods=5, time_frequency="quarterly",
                             config=config, _data_dict=minimal_data_dict)
        with pytest.raises(ValueError, match="N, N"):
            m.set_essential_inputs(np.ones((2, 2)))
        with pytest.raises(ValueError, match="essential"):
            m.set_essential_inputs(np.zeros((m.N, m.N)))
