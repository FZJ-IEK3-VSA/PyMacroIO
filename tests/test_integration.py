"""Integration tests for the run_model simulation loop of InputOutputModel.

Verifies the five equation-block methods compose and base-year calibration is a steady state.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyMacroIO.model import InputOutputModel
from pyMacroIO.config import ModelConfig


# Steady-state tests
class TestRunModelSteadyState:
    """Verifies that the unshocked model reproduces base-year values at every period."""

    @pytest.fixture(scope="class")
    def result(self, minimal_model: InputOutputModel) -> dict:
        """Return the result dict from a single unshocked run."""
        return minimal_model.run_model(store_full_matrices=False, validate=False)

    def test_output_reproduces_base_year(
        self, minimal_model: InputOutputModel, result: dict
    ) -> None:
        """Verifies that gross output equals x0 at every period after the initial one."""
        x = result["gross_output"]  # (3, 11)
        for t in range(1, minimal_model.TT):
            np.testing.assert_allclose(
                x[:, t],
                minimal_model.x0,
                rtol=1e-6,
                err_msg=f"Output deviates from base year at t={t}",
            )

    def test_gdp_constant_across_periods(
        self, minimal_model: InputOutputModel, result: dict
    ) -> None:
        """GDP does not drift between consecutive periods under no-shock conditions."""
        gdp = result["gdp"]
        np.testing.assert_allclose(gdp[1:], gdp[:-1], rtol=1e-6)

    def test_labour_unchanged_from_base_year(
        self, minimal_model: InputOutputModel, result: dict
    ) -> None:
        """Verifies that labour compensation equals l0 at every period after the initial one."""
        l = result["labour_compensation"]
        for t in range(1, minimal_model.TT):
            np.testing.assert_allclose(
                l[:, t],
                minimal_model.l0,
                rtol=1e-6,
                err_msg=f"Labour deviates at t={t}",
            )

    def test_realised_consumption_equals_base(
        self, minimal_model: InputOutputModel, result: dict
    ) -> None:
        """Realised consumption equals the base-year household vector at every period."""
        c = result["realised_consumption"]
        for t in range(1, minimal_model.TT):
            np.testing.assert_allclose(
                c[:, t],
                minimal_model.c0,
                rtol=1e-6,
                err_msg=f"Consumption deviates at t={t}",
            )

    def test_gdp_strictly_positive(
        self, minimal_model: InputOutputModel, result: dict
    ) -> None:
        """Verifies that GDP is strictly positive at all periods."""
        assert np.all(result["gdp"] > 0)

    def test_output_non_negative(
        self, minimal_model: InputOutputModel, result: dict
    ) -> None:
        """Verifies that gross output is non-negative at all periods."""
        assert np.all(result["gross_output"] >= 0)


# Shock propagation tests
class TestRunModelShockPropagation:
    """Verifies output constraint shocks cut output and GDP, with a fresh model per test."""

    @pytest.fixture()
    def fresh_model(self, minimal_data_dict: dict) -> InputOutputModel:
        """Return a freshly constructed model that has not been shocked."""
        config = ModelConfig(n_periods=10, time_frequency="quarterly")
        return InputOutputModel(
            n_periods=10,
            time_frequency="quarterly",
            config=config,
            _data_dict=minimal_data_dict,
        )

    def test_output_constraint_shock_reduces_output_and_gdp(
        self, fresh_model: InputOutputModel
    ) -> None:
        """A 50 per cent cap on sector A at t=5 cuts output and GDP, leaving t=4 unaffected."""
        m = fresh_model
        # sector label "A" maps to index 0 in the 3-sector minimal economy.
        m.apply_output_constraint_shock(
            sector_label="A", time_period=5, reduction_pct=0.5
        )
        result = m.run_model()

        # Sector A output falls at t=5.
        assert result["gross_output"][0, 5] < m.x0[0], (
            "Sector A output should be below base-year level at the shocked period."
        )

        # Aggregate GDP falls at t=5.
        assert result["gdp"][5] < result["gdp"][0], (
            "GDP should be lower at the shocked period than at the base period."
        )

        # Period before the shock is unaffected.
        np.testing.assert_allclose(
            result["gross_output"][0, 4],
            m.x0[0],
            rtol=1e-5,
            err_msg="Sector A output should equal base year at t=4 (period before shock).",
        )


# Full-matrix storage tests
class TestRunModelStoreFull:
    """Verifies that store_full_matrices=True adds the expected (N, N, TT) arrays to the result."""

    @pytest.fixture(scope="class")
    def result(self, minimal_model: InputOutputModel) -> dict:
        """Return the result dict from a run with full matrix storage enabled."""
        return minimal_model.run_model(store_full_matrices=True)

    def test_inventories_present_with_correct_shape(
        self, minimal_model: InputOutputModel, result: dict
    ) -> None:
        """Verifies that the inventories array is in the result and has shape (N, N, TT)."""
        assert "inventories" in result
        assert result["inventories"].shape == (minimal_model.N, minimal_model.N, minimal_model.TT)

    def test_intermediate_deliveries_present_with_correct_shape(
        self, minimal_model: InputOutputModel, result: dict
    ) -> None:
        """The intermediate_deliveries array is in the result with shape (N, N, TT)."""
        assert "intermediate_deliveries" in result
        assert result["intermediate_deliveries"].shape == (
            minimal_model.N, minimal_model.N, minimal_model.TT
        )

    def test_orders_present_with_correct_shape(
        self, minimal_model: InputOutputModel, result: dict
    ) -> None:
        """Verifies that the orders array is in the result and has shape (N, N, TT)."""
        assert "orders" in result
        assert result["orders"].shape == (minimal_model.N, minimal_model.N, minimal_model.TT)

    def test_full_matrices_non_negative(self, result: dict) -> None:
        """Verifies that inventories, intermediate_deliveries, and orders are all non-negative."""
        assert np.all(result["inventories"] >= 0)
        assert np.all(result["intermediate_deliveries"] >= 0)
        assert np.all(result["orders"] >= 0)


def _make_les_model(data_dict: dict, subsistence_shares, n_periods: int = 20) -> "InputOutputModel":
    """Return a 3-sector InputOutputModel with the given subsistence_shares."""
    config = ModelConfig(
        subsistence_shares=subsistence_shares,
        n_periods=n_periods,
        time_frequency="quarterly",
    )
    return InputOutputModel(
        n_periods=n_periods,
        time_frequency="quarterly",
        config=config,
        _data_dict=data_dict,
    )


class TestSubsistenceInFullRun:
    """Verifies that the LES subsistence floor operates correctly in a full run_model simulation."""

    def test_les_model_reproduces_base_year_without_shocks(self, minimal_data_dict: dict) -> None:
        """LES introduces no drift from the base-year calibration without shocks."""
        model = _make_les_model(minimal_data_dict, subsistence_shares=0.3)
        result = model.run_model(store_full_matrices=False, validate=False)
        c = result["consumption_by_hh_region"]
        base = c[0, 0]
        for t in range(1, model.TT):
            np.testing.assert_allclose(
                c[0, t],
                base,
                rtol=1e-4,
                err_msg=f"LES consumption drifts from base year at t={t}.",
            )

    def test_les_consumption_higher_than_cd_under_income_shock(self, minimal_data_dict: dict) -> None:
        """LES consumption is at least Cobb-Douglas at the shocked period with active floors."""
        model_les = _make_les_model(minimal_data_dict, subsistence_shares=0.4)
        model_cd  = _make_les_model(minimal_data_dict, subsistence_shares=0.0)

        for t in range(3, 8):
            model_les.apply_output_constraint_shock("A", t, 0.5)
            model_cd.apply_output_constraint_shock("A", t, 0.5)

        result_les = model_les.run_model(store_full_matrices=False, validate=False)
        result_cd  = model_cd.run_model(store_full_matrices=False, validate=False)

        assert result_les["consumption_by_hh_region"][0, 5] >= (
            result_cd["consumption_by_hh_region"][0, 5]
        )

    def test_les_all_periods_consumption_positive(self, minimal_data_dict: dict) -> None:
        """Verifies that the subsistence floor prevents consumption from collapsing to zero."""
        model = _make_les_model(minimal_data_dict, subsistence_shares=0.5)
        for t in range(2, 10):
            model.apply_output_constraint_shock("A", t, 0.7)
        result = model.run_model(store_full_matrices=False, validate=False)
        c = result["consumption_by_hh_region"]
        for t in range(model.TT):
            assert c[0, t] > 0, (
                f"Consumption collapsed to zero or below at t={t} despite the subsistence floor."
            )


# Import flexibility helper and tests
def _make_flex_model(data_dict: dict, import_flexibility, n_periods: int = 20) -> "InputOutputModel":
    """Return a 3-sector InputOutputModel with the given import_flexibility."""
    config = ModelConfig(
        import_flexibility=import_flexibility,
        n_periods=n_periods,
        time_frequency="quarterly",
    )
    return InputOutputModel(
        n_periods=n_periods,
        time_frequency="quarterly",
        config=config,
        _data_dict=data_dict,
    )


class TestImportFlexibility:
    """Verifies that the import flexibility mechanism supplements inventories during shortfalls."""

    def test_import_supplement_activated_by_output_constraint(self, minimal_data_dict: dict) -> None:
        """Import supplements activate at t=6 once inventory becomes the binding constraint."""
        m = _make_flex_model(minimal_data_dict, import_flexibility=0.5)
        for t in range(3, 8):
            m.apply_output_constraint_shock("A", t, 0.5)
        result = m.run_model(store_full_matrices=False, validate=False)
        assert result["import_supplement_by_input"][0, 6] > 0

    def test_import_flexibility_reduces_gdp_loss_from_shortage(self, minimal_data_dict: dict) -> None:
        """Import flexibility raises GDP above the no-flexibility path during a shock."""
        m_flex   = _make_flex_model(minimal_data_dict, import_flexibility=0.5)
        m_noflex = _make_flex_model(minimal_data_dict, import_flexibility=0.0)
        for t in range(3, 8):
            m_flex.apply_output_constraint_shock("A", t, 0.5)
            m_noflex.apply_output_constraint_shock("A", t, 0.5)
        result_flex   = m_flex.run_model(store_full_matrices=False, validate=False)
        result_noflex = m_noflex.run_model(store_full_matrices=False, validate=False)
        assert result_flex["gdp"][7] > result_noflex["gdp"][7]

    def test_no_supplement_without_shortfall(self, minimal_data_dict: dict) -> None:
        """No import supplement is drawn when inventories remain at or above target."""
        m = _make_flex_model(minimal_data_dict, import_flexibility=0.5)
        result = m.run_model(store_full_matrices=False, validate=False)
        assert result["import_supplement_by_input"].sum() < 1e-10


# LES sector-specific subsistence helper and tests
def _make_les_sector_model(data_dict: dict, subsistence_shares, n_periods: int = 15) -> "InputOutputModel":
    """Return a 3-sector InputOutputModel via _make_les_model with the given subsistence_shares."""
    return _make_les_model(data_dict=data_dict, subsistence_shares=subsistence_shares, n_periods=n_periods)


class TestLESSectorSpecificVector:
    """Verifies that sector-specific subsistence shares are stored and applied correctly."""

    def test_sector_specific_subsistence_shares_stored_correctly(self, minimal_data_dict: dict) -> None:
        """Checks that gamma_r[0] holds the subsistence levels implied by the shares."""
        shares = [0.3, 0.1, 0.2]
        m = _make_les_sector_model(minimal_data_dict, subsistence_shares=shares)
        assert m.gamma_r[0].shape == (3,), (
            "gamma_r[0] should have shape (3,) for a 3-sector model."
        )
        for i, s in enumerate(shares):
            expected = s * m.c0_r[0][i]
            np.testing.assert_allclose(
                m.gamma_r[0][i],
                expected,
                rtol=1e-10,
                err_msg=f"gamma_r[0][{i}] does not match {s} * c0_r[0][{i}].",
            )

    def test_sector_vector_les_model_runs_and_consumption_positive(self, minimal_data_dict: dict) -> None:
        """Heterogeneous subsistence floors keep consumption positive under a shock."""
        m = _make_les_sector_model(minimal_data_dict, subsistence_shares=[0.4, 0.2, 0.3])
        for t in range(3, 8):
            m.apply_output_constraint_shock("A", t, 0.5)
        result = m.run_model(store_full_matrices=False, validate=False)
        c = result["consumption_by_hh_region"]
        assert np.all(c > 0)

    def test_vector_subsistence_higher_for_high_share_sector(self, minimal_data_dict: dict) -> None:
        """A high floor on sector A protects its consumption more than a high floor on C."""
        m_high_a = _make_les_sector_model(minimal_data_dict, subsistence_shares=[0.4, 0.1, 0.1])
        m_high_c = _make_les_sector_model(minimal_data_dict, subsistence_shares=[0.1, 0.1, 0.4])
        for t in range(3, 8):
            m_high_a.apply_output_constraint_shock("A", t, 0.5)
            m_high_c.apply_output_constraint_shock("A", t, 0.5)
        result_high_a = m_high_a.run_model(store_full_matrices=False, validate=False)
        result_high_c = m_high_c.run_model(store_full_matrices=False, validate=False)
        assert result_high_a["consumption_by_hh_region"][0, 5] >= result_high_c["consumption_by_hh_region"][0, 5]
