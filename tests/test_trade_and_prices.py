"""Tests for export pull, RoW supply cap, and asymmetric price pass-through in run_model."""

from __future__ import annotations

import numpy as np
import pytest

from pyMacroIO.model import InputOutputModel
from pyMacroIO.config import ModelConfig

# Model factory helpers

def _row_model(
    data_dict: dict,
    n_periods: int = 15,
    export_pull: float | list[float] = 0.0,
    row_supply_cap: float | list[float] = 0.0,
    import_flexibility: float | list[float] = 0.0,
) -> InputOutputModel:
    """Return a 2-region InputOutputModel built from the supplied RoW data dict."""
    config = ModelConfig(
        n_periods=n_periods,
        time_frequency="quarterly",
        export_pull=export_pull,
        row_supply_cap=row_supply_cap,
        import_flexibility=import_flexibility,
    )
    return InputOutputModel(
        n_periods=n_periods,
        time_frequency="quarterly",
        config=config,
        _data_dict=data_dict,
    )


def _price_model(
    data_dict: dict,
    n_periods: int = 15,
    price_passthrough_enabled: bool = True,
    price_passthrough_pos: float = 0.8,
    price_passthrough_neg: float = 0.4,
) -> InputOutputModel:
    """Return a 1-region InputOutputModel with price pass-through configured."""
    config = ModelConfig(
        n_periods=n_periods,
        time_frequency="quarterly",
        price_passthrough_enabled=price_passthrough_enabled,
        price_passthrough_pos=price_passthrough_pos,
        price_passthrough_neg=price_passthrough_neg,
    )
    return InputOutputModel(
        n_periods=n_periods,
        time_frequency="quarterly",
        config=config,
        _data_dict=data_dict,
    )


# TestExportPullVector

class TestExportPullVector:
    """Tests for the export_pull_vector codepath in run_model."""

    def test_export_pull_zero_no_extra_demand(self, row_data_dict: dict) -> None:
        """A zero export_pull_vector produces no export_pull_supplement at any period."""
        m = _row_model(row_data_dict, export_pull=0.0)
        result = m.run_model()
        np.testing.assert_allclose(
            result["export_pull_supplement"],
            0.0,
            atol=1e-12,
        )

    def test_export_pull_positive_raises_output_when_slack_exists(self, row_data_dict: dict) -> None:
        """Export_pull on sector A yields a supplement when a consumption shock creates slack."""
        m_base = _row_model(row_data_dict, export_pull=0.0)
        m_pull = _row_model(row_data_dict, export_pull=[0.5, 0.0, 0.0, 0.0])

        # A consumption shock in region 0 reduces demand and creates labour slack.
        for m in (m_base, m_pull):
            m.apply_consumption_shock(start=2, duration=5, intensity=0.3, region=0)

        res_base = m_base.run_model()
        res_pull = m_pull.run_model()

        # Export pull should generate a positive supplement for sector A (index 0).
        supplement_total = res_pull["export_pull_supplement"][0, :].sum()
        assert supplement_total > 0.0, (
            "Positive export_pull on sector A should generate a non-zero supplement during the slack window."
        )
        # The zero-pull model should have no supplement.
        np.testing.assert_allclose(
            res_base["export_pull_supplement"], 0.0, atol=1e-12,
        )

    def test_export_pull_no_slack_no_extra_output(self, row_data_dict: dict) -> None:
        """A binding output constraint stops export pull raising output above it."""
        m_unconstrained = _row_model(row_data_dict, export_pull=[0.5, 0.0, 0.0, 0.0])
        m_constrained   = _row_model(row_data_dict, export_pull=[0.5, 0.0, 0.0, 0.0])

        # Create demand slack via consumption shock (same for both models).
        for m in (m_unconstrained, m_constrained):
            m.apply_consumption_shock(start=2, duration=8, intensity=0.3, region=0)

        # Tightly constrain sector A in the constrained model so no slack can be used.
        for t in range(2, 15):
            m_constrained.apply_output_constraint_shock("A", t, reduction_pct=0.99)

        res_unconstrained = m_unconstrained.run_model()
        res_constrained   = m_constrained.run_model()

        # Constrained model should produce far less sector A output.
        assert (
            res_constrained["gross_output"][0, 4:].sum()
            < res_unconstrained["gross_output"][0, 4:].sum()
        ), (
            "Binding output constraint should suppress sector A output well below the unconstrained level."
        )


# TestRowSupplyCapVector

class TestRowSupplyCapVector:
    """Tests for the row_supply_cap_vector cap on the RoW import supplement."""

    def test_row_supply_cap_zero_suppresses_supplement(self, row_data_dict: dict) -> None:
        """row_export_supplement is zero at every period when row_supply_cap is zero."""
        m = _row_model(row_data_dict, row_supply_cap=0.0, import_flexibility=0.8)
        for t in range(2, 12):
            m.apply_rationing_shock("A", time_period=t, capacity_pct=0.05)
        result = m.run_model()
        np.testing.assert_allclose(
            result["row_export_supplement"],
            0.0,
            atol=1e-12,
        )

    def test_row_supply_cap_large_does_not_bind(self, row_data_dict: dict) -> None:
        """Checks that a cap of 1.0 allows the full import supplement to be credited to RoW."""
        m_tight   = _row_model(row_data_dict, row_supply_cap=1e-9, import_flexibility=0.8)
        m_generous = _row_model(row_data_dict, row_supply_cap=1.0,  import_flexibility=0.8)

        for m in (m_tight, m_generous):
            for t in range(2, 12):
                m.apply_rationing_shock("A", time_period=t, capacity_pct=0.05)

        res_tight   = m_tight.run_model()
        res_generous = m_generous.run_model()

        # A near-zero cap allows almost no RoW export supplement, a cap of 1.0 the full amount.
        assert (
            res_generous["row_export_supplement"].sum()
            > res_tight["row_export_supplement"].sum() + 1e-6
        ), (
            "A generous row_supply_cap should allow substantially more RoW export supplement than a near-zero cap."
        )

    def test_row_supply_cap_binding_reduces_output_relative_to_uncapped(self, row_data_dict: dict) -> None:
        """A tight row_supply_cap lowers GDP below a generous cap under a supply shock."""
        m_tight   = _row_model(row_data_dict, row_supply_cap=1e-9, import_flexibility=0.8)
        m_generous = _row_model(row_data_dict, row_supply_cap=1.0,  import_flexibility=0.8)

        for m in (m_tight, m_generous):
            for t in range(2, 12):
                m.apply_rationing_shock("A", time_period=t, capacity_pct=0.05)

        res_tight   = m_tight.run_model()
        res_generous = m_generous.run_model()

        # Without RoW supplement the economy has less import relief, so GDP is lower.
        assert res_tight["gdp"].sum() < res_generous["gdp"].sum(), (
            "Tight row_supply_cap should produce lower cumulative GDP than a generous cap."
        )


# TestNegativeCostShocks

class TestNegativeCostShocks:
    """Tests for negative cost shocks and the asymmetric price pass-through via L_price_neg."""

    def test_L_price_neg_constructed_when_price_passthrough_enabled(self, minimal_data_dict: dict) -> None:
        """Asserts that L_price_neg is not None when price_passthrough_enabled is True."""
        m = _price_model(minimal_data_dict, price_passthrough_enabled=True)
        assert m.L_price_neg is not None

    def test_L_price_neg_none_when_price_passthrough_disabled(self, minimal_data_dict: dict) -> None:
        """Checks that L_price_neg is None when price pass-through is disabled."""
        m = _price_model(minimal_data_dict, price_passthrough_enabled=False)
        assert m.L_price_neg is None

    def test_negative_cost_shock_lowers_price_index(self, minimal_data_dict: dict) -> None:
        """Checks that a negative cost shock reduces at least one sector price index below 1.0."""
        m_neg = _price_model(
            minimal_data_dict,
            price_passthrough_enabled=True,
            price_passthrough_pos=0.8,
            price_passthrough_neg=0.4,
        )
        m_pos = _price_model(
            minimal_data_dict,
            price_passthrough_enabled=True,
            price_passthrough_pos=0.8,
            price_passthrough_neg=0.4,
        )

        m_neg.apply_price_cost_shock("A", time_period=3, delta_cost=-0.3, duration=6)
        m_pos.apply_price_cost_shock("A", time_period=3, delta_cost= 0.3, duration=6)

        res_neg = m_neg.run_model()
        res_pos = m_pos.run_model()

        assert res_neg["price_index"][:, 5].min() < 1.0, (
            "A negative cost shock should lower at least one sector price index below 1.0."
        )
        assert res_pos["price_index"][:, 5].max() > 1.0, (
            "A positive cost shock should raise at least one sector price index above 1.0."
        )

    def test_symmetric_positive_negative_shocks_cancel(self, minimal_data_dict: dict) -> None:
        """Opposed cost shocks of equal size put the A price index either side of 1.0."""
        m_pos = _price_model(
            minimal_data_dict,
            price_passthrough_enabled=True,
            price_passthrough_pos=0.8,
            price_passthrough_neg=0.8,
        )
        m_neg = _price_model(
            minimal_data_dict,
            price_passthrough_enabled=True,
            price_passthrough_pos=0.8,
            price_passthrough_neg=0.8,
        )

        delta = 0.4
        m_pos.apply_price_cost_shock("A", time_period=2, delta_cost= delta, duration=5)
        m_neg.apply_price_cost_shock("A", time_period=2, delta_cost=-delta, duration=5)

        res_pos = m_pos.run_model()
        res_neg = m_neg.run_model()

        # At period 4, well inside the shock window, prices should bracket 1.0.
        p_pos_A = res_pos["price_index"][0, 4]
        p_neg_A = res_neg["price_index"][0, 4]

        assert p_pos_A > 1.0, (
            "Positive cost shock should raise sector A price index above 1.0 at t=4."
        )
        assert p_neg_A < 1.0, (
            "Negative cost shock should lower sector A price index below 1.0 at t=4."
        )
        # Both deviations from 1.0 are strictly positive.
        assert (p_pos_A - 1.0) > 0.0
        assert (1.0 - p_neg_A) > 0.0
