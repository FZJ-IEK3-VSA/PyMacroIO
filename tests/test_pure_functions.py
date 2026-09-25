"""Tests for pure or near-pure functions in pyMacroIO.model."""

import warnings

import numpy as np
import numpy.testing as npt
import pytest

from types import SimpleNamespace

from pyMacroIO.model import estimate_essential_inputs_from_io_data
from pyMacroIO.config import build_subsistence_shares_vector
from pyMacroIO.constants import NUMERIC_LARGE


# Small A matrix for multiple tests, derived from conftest Z0/x0 with x0=[60,40,40].
_A3 = np.array(
    [
        [0.0,     0.5,   0.125],
        [0.05,    0.0,   0.375],
        [0.13333333, 0.05, 0.0],
    ],
    dtype=np.float64,
)

# Gross output vector matching _A3 (conftest Z0/x0).
_X3 = np.array([60.0, 40.0, 40.0], dtype=np.float64)

# Backward-linkage chain; sector 0 backward linkage exceeds 1.5× the mean.
_A4_BACK_CHAIN = np.array(
    [
        [0.0,  0.0,  0.0,  0.0],
        [0.95, 0.0,  0.0,  0.0],
        [0.0,  0.95, 0.0,  0.0],
        [0.0,  0.0,  0.95, 0.0],
    ],
    dtype=np.float64,
)

# Forward-linkage chain; sector 0 Ghosh forward linkage exceeds 1.5× the mean.
_A4_FWD_CHAIN = np.array(
    [
        [0.0, 0.95, 0.0,  0.0 ],
        [0.0, 0.0,  0.95, 0.0 ],
        [0.0, 0.0,  0.0,  0.95],
        [0.0, 0.0,  0.0,  0.0 ],
    ],
    dtype=np.float64,
)


def _pad_inactive(A: np.ndarray, x0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Append a sector with no flows and no gross output, as inactive MRIO cells appear."""
    N = A.shape[0]
    A_pad = np.zeros((N + 1, N + 1), dtype=np.float64)
    A_pad[:N, :N] = A
    return A_pad, np.append(np.asarray(x0, dtype=np.float64), 0.0)


# Method with an A matrix and gross output for which it marks at least one cell.
_LINKAGE_CASES = [
    ("combined_linkage", _A3, _X3),
    ("combined", _A3, _X3),
    ("forward_linkage", _A4_FWD_CHAIN, np.ones(4)),
    ("linkage", _A4_BACK_CHAIN, np.ones(4)),
]

# A helper A matrix for orders_O tests.
_A_ORDERS = np.array(
    [[0.0, 0.2, 0.1],
     [0.3, 0.0, 0.2],
     [0.1, 0.1, 0.0]],
    dtype=np.float64,
)


def _make_orders_arrays():
    """Return consistent arrays for orders_O tests with inventory at target (steady state)."""
    A     = _A_ORDERS.copy()
    d     = np.array([10.0, 20.0, 15.0], dtype=np.float64)
    tau   = np.array([2.0,   3.0,  4.0], dtype=np.float64)
    S_tar = A * d[np.newaxis, :]
    S     = S_tar.copy()
    return A, d, tau, S_tar, S


# TestEstimateEssentialInputs
class TestEstimateEssentialInputs:
    """Tests for estimate_essential_inputs_from_io_data."""

    def test_value_method_flags_entries_above_share_threshold(self):
        """The value method marks entries whose column share meets value_threshold."""
        result = estimate_essential_inputs_from_io_data(
            _A3, method="value", value_threshold=0.05
        )
        expected = np.array(
            [[0, 1, 1],
             [1, 0, 1],
             [1, 1, 0]],
            dtype=np.int64,
        )
        npt.assert_array_equal(result, expected)

    def test_value_method_respects_non_default_threshold(self):
        """The value method with threshold 0.8 flags only entries with share at least 0.8."""
        result = estimate_essential_inputs_from_io_data(
            _A3, method="value", value_threshold=0.8
        )
        expected = np.array(
            [[0, 1, 0],
             [0, 0, 0],
             [0, 0, 0]],
            dtype=np.int64,
        )
        npt.assert_array_equal(result, expected)

    def test_value_method_output_is_binary(self):
        """The value method produces only 0 and 1 values."""
        result = estimate_essential_inputs_from_io_data(_A3, method="value")
        assert set(np.unique(result)).issubset({0, 1})

    def test_top_n_method_marks_exactly_top_n_inputs_per_column(self):
        """The top_n method marks exactly top_n entries per column."""
        result = estimate_essential_inputs_from_io_data(_A3, method="top_n", top_n=2)
        assert result.shape == _A3.shape
        for j in range(_A3.shape[1]):
            assert result[:, j].sum() == 2

    def test_top_n_method_selects_largest_coefficients(self):
        """The top_n method selects the largest coefficient entry in each column."""
        result = estimate_essential_inputs_from_io_data(_A3, method="top_n", top_n=1)
        expected_top1 = np.array(
            [[0, 1, 0],
             [0, 0, 1],
             [1, 0, 0]],
            dtype=np.int64,
        )
        npt.assert_array_equal(result, expected_top1)
        # Non-selected entries must be zero.
        assert result.sum() == _A3.shape[1]

    def test_combined_linkage_produces_non_trivial_result_for_heterogeneous_a(self):
        """The combined_linkage method marks at least one entry for a heterogeneous A matrix."""
        result = estimate_essential_inputs_from_io_data(
            _A3, method="combined_linkage", x0=_X3
        )
        assert result.sum() >= 1, (
            "combined_linkage should mark at least one entry."
        )

    def test_combined_linkage_uniform_diagonal_a_gives_all_zeros(self):
        """A diagonal A matrix gives all-zero output from combined_linkage."""
        A_diag = np.diag([0.1, 0.1, 0.1])
        result = estimate_essential_inputs_from_io_data(
            A_diag, method="combined_linkage", x0=np.ones(3)
        )
        npt.assert_array_equal(result, np.zeros_like(A_diag, dtype=int))

    def test_combined_linkage_output_shape_and_dtype(self):
        """The combined_linkage method returns an array of the same shape as the input."""
        result = estimate_essential_inputs_from_io_data(
            _A3, method="combined_linkage", x0=_X3
        )
        assert result.shape == _A3.shape
        assert result.dtype in (np.int32, np.int64, np.int_, int, np.intp)

    def test_linkage_method_output_shape_and_dtype(self):
        """The linkage method returns a binary array with at least one marked entry."""
        result = estimate_essential_inputs_from_io_data(_A4_BACK_CHAIN, method="linkage")
        assert result.shape == _A4_BACK_CHAIN.shape
        assert set(np.unique(result)).issubset({0, 1})
        assert result.sum() > 0, (
            "linkage method should mark at least one cell."
        )

    def test_forward_linkage_method_output_shape_and_dtype(self):
        """The forward_linkage method returns a binary array with at least one marked entry."""
        result = estimate_essential_inputs_from_io_data(
            _A4_FWD_CHAIN, method="forward_linkage", x0=np.ones(4)
        )
        assert result.shape == _A4_FWD_CHAIN.shape
        assert set(np.unique(result)).issubset({0, 1})
        assert result.sum() > 0, (
            "forward_linkage method should mark at least one cell."
        )

    def test_elasticity_method_output_shape_and_dtype(self):
        """The elasticity method returns an integer array with the same shape as the input."""
        result = estimate_essential_inputs_from_io_data(_A3, method="elasticity")
        assert result.shape == _A3.shape
        assert set(np.unique(result)).issubset({0, 1})

    def test_combined_method_output_shape_and_dtype(self):
        """The combined method returns an integer array with the same shape as the input."""
        result = estimate_essential_inputs_from_io_data(_A3, method="combined")
        assert result.shape == _A3.shape
        assert set(np.unique(result)).issubset({0, 1})

    @pytest.mark.parametrize("method,A,x0", _LINKAGE_CASES)
    def test_inactive_sector_leaves_the_active_block_unchanged(self, method, A, x0):
        """A zero-output sector is excluded from the linkage means, not propagated as NaN."""
        A_pad, x0_pad = _pad_inactive(A, x0)
        padded = estimate_essential_inputs_from_io_data(A_pad, method=method, x0=x0_pad)
        active = estimate_essential_inputs_from_io_data(A, method=method, x0=x0)
        N = A.shape[0]
        assert active.sum() > 0
        npt.assert_array_equal(padded[:N, :N], active)
        assert padded[N, :].sum() == 0 and padded[:, N].sum() == 0

    def test_inactive_sector_does_not_raise_a_division_warning(self):
        """The Ghosh ratio is not evaluated where gross output is absent."""
        A_pad, x0_pad = _pad_inactive(_A3, _X3)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            estimate_essential_inputs_from_io_data(
                A_pad, method="combined_linkage", x0=x0_pad
            )
        # numpy raises spurious FPE flags inside matmul, so only the elementwise divide counts.
        assert not [w for w in caught if "in divide" in str(w.message)]

    @pytest.mark.parametrize("method,A,x0", _LINKAGE_CASES)
    def test_all_sectors_inactive_raises_value_error(self, method, A, x0):
        """A gross-output vector with no active sector raises rather than returning zeros."""
        with pytest.raises(ValueError, match="positive gross output"):
            estimate_essential_inputs_from_io_data(
                A, method=method, x0=np.zeros(A.shape[0])
            )

    @pytest.mark.parametrize("bad", [np.array([60.0, 40.0, -1.0]), np.array([60.0, np.nan, 40.0])])
    def test_negative_or_non_finite_gross_output_raises(self, bad):
        """A negative or non-finite gross output raises rather than being read as inactive."""
        with pytest.raises(ValueError, match="finite and non-negative"):
            estimate_essential_inputs_from_io_data(_A3, method="combined_linkage", x0=bad)

    def test_gross_output_of_the_wrong_length_raises(self):
        """A gross-output vector that does not match the matrix raises on entry."""
        with pytest.raises(ValueError, match=r"shape \(3,\)"):
            estimate_essential_inputs_from_io_data(
                _A3, method="combined_linkage", x0=np.array([60.0, 40.0])
            )

    def test_supplier_without_gross_output_raises(self):
        """A sector with no gross output but non-zero supply rows is an inconsistent pair."""
        A_pad, x0_pad = _pad_inactive(_A3, _X3)
        A_pad[3, 0] = 0.2
        with pytest.raises(ValueError, match="supplies intermediate flows"):
            estimate_essential_inputs_from_io_data(
                A_pad, method="combined_linkage", x0=x0_pad
            )

    @pytest.mark.parametrize("method,A,x0", _LINKAGE_CASES)
    def test_linkage_indices_match_the_active_economy(self, method, A, x0):
        """Both indices are those of the economy formed by the active sectors alone."""
        A_pad, x0_pad = _pad_inactive(A, x0)
        A_pad[: A.shape[0], -1] = 0.3
        padded = estimate_essential_inputs_from_io_data(A_pad, method=method, x0=x0_pad)
        active = estimate_essential_inputs_from_io_data(A, method=method, x0=x0)
        npt.assert_array_equal(padded[: A.shape[0], : A.shape[0]], active)

    def test_unknown_method_raises_value_error(self):
        """An unrecognised method name raises a ValueError."""
        with pytest.raises(ValueError, match="Unknown method"):
            estimate_essential_inputs_from_io_data(_A3, method="not_a_real_method")

    def test_zero_matrix_returns_all_zeros_for_value(self):
        """A zero coefficient matrix produces all-zero output for the value method."""
        A_zero = np.zeros((4, 4), dtype=np.float64)
        result = estimate_essential_inputs_from_io_data(A_zero, method="value")
        npt.assert_array_equal(result, np.zeros((4, 4), dtype=int))

    def test_zero_matrix_returns_all_zeros_for_top_n(self):
        """A zero coefficient matrix produces all-zero output for the top_n method."""
        A_zero = np.zeros((3, 3), dtype=np.float64)
        result = estimate_essential_inputs_from_io_data(A_zero, method="top_n", top_n=2)
        npt.assert_array_equal(result, np.zeros((3, 3), dtype=int))

    def test_linalg_error_fallback_to_value(self):
        """Methods that invert (I - A) fall back to 'value' when the matrix is singular."""
        A_singular = np.eye(3, dtype=np.float64)
        expected = estimate_essential_inputs_from_io_data(A_singular, "value")
        for method in ("linkage", "forward_linkage", "elasticity", "combined_linkage"):
            result = estimate_essential_inputs_from_io_data(A_singular, method, x0=np.ones(3))
            npt.assert_array_equal(
                result,
                expected,
                err_msg=f"Method '{method}' should match 'value' after LinAlgError fallback.",
            )

    def test_combined_method_zero_max_score_falls_back_to_value(self):
        """The combined method returns all-ones when the A matrix is zero."""
        A_zero = np.zeros((3, 3), dtype=np.float64)
        result = estimate_essential_inputs_from_io_data(A_zero, method="combined")
        # Zero flows give an identity Leontief inverse, so uniform linkage exceeds the threshold.
        npt.assert_array_equal(result, np.ones((3, 3), dtype=int))


# TestOrdersO
class TestOrdersO:
    """Tests for InputOutputModel.orders_O."""

    def test_steady_state_orders_equal_use_demand(self, minimal_model):
        """When current inventory equals target, orders equal the intermediate use demand A*d."""
        A, d, tau, S_tar, S = _make_orders_arrays()
        result = minimal_model.orders_O(A, d, tau, S_tar, S)
        expected = A * d[np.newaxis, :]
        npt.assert_allclose(result, expected, rtol=1e-12)

    def test_below_target_orders_include_restock_term(self, minimal_model):
        """When S < S_tar, orders include the restock contribution (S_tar - S) / tau."""
        A, d, tau, S_tar, S = _make_orders_arrays()
        S_low = S_tar - 1.0
        result = minimal_model.orders_O(A, d, tau, S_tar, S_low)
        restock = 1.0 / tau[:, np.newaxis]
        expected = A * d[np.newaxis, :] + restock
        npt.assert_allclose(result, expected, rtol=1e-12)

    def test_below_target_orders_with_non_uniform_shortfall(self, minimal_model):
        """Per-cell broadcasting is correct when the shortfall differs across inputs and sectors."""
        A, d, tau, S_tar, S = _make_orders_arrays()
        shortfall = np.array(
            [[0.5, 1.0, 0.0],
             [0.0, 2.0, 1.5],
             [0.3, 0.0, 1.0]],
            dtype=np.float64,
        )
        S_low = S_tar - shortfall
        result = minimal_model.orders_O(A, d, tau, S_tar, S_low)
        restock_gap = np.maximum(shortfall, 0.0)
        expected = A * d[np.newaxis, :] + restock_gap / tau[:, np.newaxis]
        npt.assert_allclose(result, expected, rtol=1e-12)

    def test_above_target_stock_damps_orders(self, minimal_model):
        """When S > S_tar the signed gap reduces orders below use demand, floored at zero."""
        A, d, tau, S_tar, S = _make_orders_arrays()
        S_high = S_tar + 2.0
        result = minimal_model.orders_O(A, d, tau, S_tar, S_high)
        expected = np.maximum(A * d[np.newaxis, :] - 2.0 / tau[:, np.newaxis], 0.0)
        npt.assert_allclose(result, expected, rtol=1e-12)

    def test_nan_in_d_does_not_propagate(self, minimal_model):
        """NaN in demand element d[0] is treated as zero; other columns are unaffected."""
        A, d, tau, S_tar, S = _make_orders_arrays()
        d_nan = d.copy()
        d_nan[0] = np.nan
        result = minimal_model.orders_O(A, d_nan, tau, S_tar, S)
        assert np.all(np.isfinite(result))
        # Column 0: d[0] mapped to 0; restock_gap = 0 at steady state -> zero orders.
        npt.assert_allclose(result[:, 0], np.zeros(A.shape[0]), atol=1e-12)
        # Columns 1 and 2: unaffected by the NaN replacement.
        d_clean = d.copy()
        d_clean[0] = 0.0
        expected = A * d_clean[np.newaxis, :]
        npt.assert_allclose(result[:, 1], expected[:, 1], rtol=1e-12)
        npt.assert_allclose(result[:, 2], expected[:, 2], rtol=1e-12)

    def test_nan_in_S_does_not_propagate(self, minimal_model):
        """NaN values in the current inventory matrix are treated as zero."""
        A, d, tau, S_tar, S = _make_orders_arrays()
        S_nan = S.copy()
        S_nan[0, 1] = np.nan
        result = minimal_model.orders_O(A, d, tau, S_tar, S_nan)
        assert np.all(np.isfinite(result))

    def test_neginf_in_d_treated_as_zero(self, minimal_model):
        """d[1] = -inf is mapped to 0; with S at target, column 1 orders are zero."""
        A, d, tau, S_tar, S = _make_orders_arrays()
        d_neginf = d.copy()
        d_neginf[1] = -np.inf
        result = minimal_model.orders_O(A, d_neginf, tau, S_tar, S)
        assert np.all(np.isfinite(result))
        npt.assert_allclose(result[:, 1], np.zeros(A.shape[0]), atol=1e-12)

    def test_posinf_in_d_treated_as_numeric_large(self, minimal_model):
        """d[1] = +inf is mapped to NUMERIC_LARGE; orders in column 1 are very large."""
        A, d, tau, S_tar, S = _make_orders_arrays()
        d_posinf = d.copy()
        d_posinf[1] = np.inf
        result = minimal_model.orders_O(A, d_posinf, tau, S_tar, S)
        assert np.all(np.isfinite(result))
        expected_col1 = A[:, 1] * NUMERIC_LARGE
        npt.assert_allclose(result[:, 1], expected_col1, rtol=1e-12)
        # Columns 0 and 2 are not affected by the replacement.
        expected_normal = A * d[np.newaxis, :]
        npt.assert_allclose(result[:, 0], expected_normal[:, 0], rtol=1e-12)
        npt.assert_allclose(result[:, 2], expected_normal[:, 2], rtol=1e-12)

    def test_zero_demand_at_target_gives_zero_orders(self, minimal_model):
        """Zero demand with inventory at target produces a zero order matrix."""
        N = 3
        A     = np.full((N, N), 0.1, dtype=np.float64)
        np.fill_diagonal(A, 0.0)
        d     = np.zeros(N, dtype=np.float64)
        tau   = np.ones(N, dtype=np.float64)
        S_tar = np.zeros((N, N), dtype=np.float64)
        S     = np.zeros((N, N), dtype=np.float64)
        result = minimal_model.orders_O(A, d, tau, S_tar, S)
        npt.assert_array_equal(result, np.zeros((N, N)))

    def test_output_shape_matches_a(self, minimal_model):
        """The returned order matrix has the same shape as the coefficient matrix."""
        A, d, tau, S_tar, S = _make_orders_arrays()
        result = minimal_model.orders_O(A, d, tau, S_tar, S)
        assert result.shape == A.shape


# TestCombinedLinkageConvention
class TestCombinedLinkageConvention:
    """Pin combined_linkage to Ghosh forward x Rasmussen backward."""

    def test_combined_linkage_uses_ghosh_forward_and_rasmussen_backward(self):
        result = estimate_essential_inputs_from_io_data(
            _A3, method="combined_linkage", x0=_X3
        )
        expected = np.array(
            [[0, 1, 1],
             [0, 0, 1],
             [0, 0, 0]],
            dtype=np.int64,
        )
        npt.assert_array_equal(result, expected)

    def test_differs_from_old_symmetric_variants(self):
        """Both degenerate variants (backward- and forward-squared) must differ."""
        result = estimate_essential_inputs_from_io_data(
            _A3, method="combined_linkage", x0=_X3
        )
        old_backward_squared = np.array(
            [[0, 0, 0],
             [0, 0, 1],
             [0, 1, 0]],
            dtype=np.int64,
        )
        old_forward_squared = np.array(
            [[0, 1, 0],
             [1, 0, 0],
             [0, 0, 0]],
            dtype=np.int64,
        )
        assert not np.array_equal(result, old_backward_squared)
        assert not np.array_equal(result, old_forward_squared)

    def test_linkage_methods_require_x0(self):
        """combined_linkage and forward_linkage raise ValueError without x0."""
        for method in ("combined_linkage", "forward_linkage"):
            with pytest.raises(ValueError, match="x0"):
                estimate_essential_inputs_from_io_data(_A3, method=method)


# TestBuildSubsistenceShares
class TestBuildSubsistenceShares:
    """Tests for config.build_subsistence_shares_vector."""

    @staticmethod
    def _stub(labels):
        return SimpleNamespace(N=len(labels),
                               label_to_index={s: i for i, s in enumerate(labels)})

    def test_table_matching_fallback_and_region_prefix(self):
        model = self._stub(["DE:Fishing", "DE:Air transport",
                            "DE:Real estate activities", "DE:Widget frobnication"])
        shares = build_subsistence_shares_vector(model, frisch=-3.0)
        npt.assert_allclose(
            shares,
            [1 - 0.60 / 3, 1 - 1.40 / 3, 1 - 0.35 / 3, 1 - 1.0 / 3],
            rtol=1e-12,
        )

    def test_longest_substring_match_wins(self):
        model = self._stub(["Air transport by drone"])
        shares = build_subsistence_shares_vector(
            model, frisch=-3.0,
            sector_elasticities={"transport": 2.0, "air transport": 1.4},
        )
        npt.assert_allclose(shares, [1 - 1.4 / 3], rtol=1e-12)

    def test_shares_clipped_at_095(self):
        model = self._stub(["Fishing"])
        shares = build_subsistence_shares_vector(
            model, frisch=-3.0, sector_elasticities={"Fishing": 0.05}
        )
        npt.assert_allclose(shares, [0.95], rtol=1e-12)

    def test_nonnegative_frisch_raises(self):
        from pyMacroIO.config import build_subsistence_shares_vector
        with pytest.raises(ValueError, match="frisch"):
            build_subsistence_shares_vector(self._stub(["Fishing"]), frisch=0.5)
