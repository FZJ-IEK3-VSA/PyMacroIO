"""
Core dynamic disequilibrium input-output model, single- or multi-region on the same code paths.
estimate_essential_inputs_from_io_data is standalone as run_model re-calls it on technical change.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from .config import ModelConfig, classify_klems
from .constants import (
    COMBINED_COEF_THRESHOLD,
    DEFAULT_TAU,
    DEFAULT_GAMMA_HIRE,
    DEFAULT_GAMMA_FIRE,
    DELTA_FLOOR,
    DELTA_CAP,
    CAPACITY_MIN_SCALE,
    CAPACITY_MAX_SCALE,
    CONSUMPTION_FLOOR_RATIO,
    CONSUMPTION_FLOOR_LABOUR_RATIO,
    ESSENTIAL_INPUT_THRESHOLD,
    ROW_IDENTITY_ATOL,
    VA_IDENTITY_TOLERANCE,
    GAMMA_HIRE_MIN,
    GAMMA_HIRE_MAX,
    GAMMA_FIRE_MIN,
    GAMMA_FIRE_MAX,
    TAU_MIN,
    TAU_MAX,
    NUMERIC_LARGE,
    CES_ELASTICITY_DEFAULT,
)

logger = logging.getLogger(__name__)


# Essential-input estimation
def _active_sectors(A_matrix: np.ndarray, x0: np.ndarray | None) -> np.ndarray:
    """Return the mask of sectors carrying gross output, over which linkages are defined."""
    N = A_matrix.shape[0]
    if x0 is None:
        return np.ones(N, dtype=bool)
    x = np.asarray(x0, dtype=np.float64)
    if x.shape != (N,):
        raise ValueError(f"Gross output must have shape ({N},); got {x.shape}.")
    if not np.isfinite(x).all() or (x < 0.0).any():
        raise ValueError("Gross output must be finite and non-negative.")
    active = x > 0.0
    if not active.any():
        raise ValueError("No sector has positive gross output.")
    if (A_matrix[~active] != 0.0).any():
        raise ValueError("A sector without gross output supplies intermediate flows.")
    return active


def _norm_linkage(linkage: np.ndarray, active: np.ndarray) -> np.ndarray:
    """Normalise a linkage vector by its mean over active sectors, zeroing the rest."""
    return np.where(active, linkage, 0.0) / (linkage[active].mean() + 1e-10)


def _ghosh_forward(L: np.ndarray, x0: np.ndarray, active: np.ndarray) -> np.ndarray:
    """Ghosh inverse row sums (L @ x0) / x0, zero where gross output is absent."""
    x = np.asarray(x0, dtype=np.float64)
    forward = np.zeros(L.shape[0], dtype=np.float64)
    # numpy raises spurious FPE flags in the vectorised matmul.
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        forward[active] = (L @ x)[active] / x[active]
    if not np.isfinite(forward).all():
        raise ValueError("The Ghosh forward linkage is not finite.")
    return forward


def estimate_essential_inputs_from_io_data(
    A_matrix: np.ndarray,
    method: str = "combined_linkage",
    value_threshold: float = 0.05,
    top_n: int = 3,
    x0: np.ndarray | None = None,
) -> np.ndarray:
    """Estimate a 0/1 essential-input matrix from A for leontief.adapted. The forward and combined
    linkage methods require x0, which excludes zero-output sectors from the means and the set."""
    N = A_matrix.shape[0]
    A_essential = np.zeros_like(A_matrix)

    if method == "value":
        input_shares = A_matrix / (A_matrix.sum(axis=0) + 1e-10)
        A_essential = (input_shares >= value_threshold).astype(int)

    elif method == "top_n":
        for j in range(N):
            if A_matrix[:, j].sum() > 0:
                top_inputs = np.argsort(A_matrix[:, j])[-top_n:]
                A_essential[top_inputs, j] = 1

    elif method == "linkage":
        try:
            active = _active_sectors(A_matrix, x0)
            I = np.eye(N)
            L = np.linalg.inv(I - A_matrix)
            backward_linkage = L[active].sum(axis=0)
            norm = _norm_linkage(backward_linkage, active)
            A_essential[:, active] = (norm > 1.5).astype(int)[:, np.newaxis]
        except np.linalg.LinAlgError:
            return estimate_essential_inputs_from_io_data(A_matrix, "value")

    elif method == "forward_linkage":
        if x0 is None:
            raise ValueError("forward_linkage requires the gross output vector x0")
        try:
            active = _active_sectors(A_matrix, x0)
            I = np.eye(N)
            L = np.linalg.inv(I - A_matrix)
            # Ghosh inverse row sums: G = x_hat^-1 L x_hat, so G.sum(1) = (L @ x0) / x0.
            forward_linkage = _ghosh_forward(L, x0, active)
            norm = _norm_linkage(forward_linkage, active)
            A_essential[:, active] = (norm > 1.5).astype(int)[:, np.newaxis]
        except np.linalg.LinAlgError:
            return estimate_essential_inputs_from_io_data(A_matrix, "value")

    elif method == "combined_linkage":
        if x0 is None:
            raise ValueError("combined_linkage requires the gross output vector x0")
        try:
            active = _active_sectors(A_matrix, x0)
            I = np.eye(N)
            L = np.linalg.inv(I - A_matrix)
            norm_backward = _norm_linkage(L[active].sum(axis=0), active)
            forward       = _ghosh_forward(L, x0, active)
            norm_forward  = _norm_linkage(forward, active)
            combined      = np.outer(norm_forward, norm_backward)
            # Gate on coefficient share too, else trace coefficients turn essential in dense MRIO.
            A_essential   = ((combined > 1.0) & (A_matrix > COMBINED_COEF_THRESHOLD)).astype(int)
        except np.linalg.LinAlgError:
            return estimate_essential_inputs_from_io_data(A_matrix, "value")

    elif method == "elasticity":
        try:
            I = np.eye(N)
            L = np.linalg.inv(I - A_matrix)
            for i in range(N):
                for j in range(N):
                    if A_matrix[i, j] > 0:
                        elasticity = A_matrix[i, j] * L[j, j] / (L[j, j] + 1e-10)
                        if elasticity > 0.1:
                            A_essential[i, j] = 1
        except np.linalg.LinAlgError:
            return estimate_essential_inputs_from_io_data(A_matrix, "value")

    elif method == "combined":
        try:
            active = _active_sectors(A_matrix, x0)
            I = np.eye(N)
            L = np.linalg.inv(I - A_matrix)
            value_imp    = A_matrix / (A_matrix.sum(axis=0) + 1e-10)
            norm_backward = _norm_linkage(L[active].sum(axis=0), active)
            forward       = (_ghosh_forward(L, x0, active) if x0 is not None
                             else L.sum(axis=1))
            norm_forward  = _norm_linkage(forward, active)
            linkage_imp  = np.outer(norm_forward, norm_backward)
            elast_imp    = A_matrix * L.T
            score = (
                0.3  * value_imp
                + 0.25 * linkage_imp
                + 0.25 * elast_imp
                + 0.2  * A_matrix
            )
            max_s = score.max()
            if max_s > 0:
                A_essential = (score / max_s > 0.3).astype(int)
            else:
                return estimate_essential_inputs_from_io_data(A_matrix, "value")
        except np.linalg.LinAlgError:
            return estimate_essential_inputs_from_io_data(A_matrix, "value")

    else:
        raise ValueError(f"Unknown method: {method}")

    return A_essential


# Delivery allocation rules, keyed by the firm_priority config value.
def _supply_share(x: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Share of demand d servable from supply x, safe-divided and clipped to [0, 1]."""
    s = np.divide(x, d, out=np.zeros_like(x), where=d != 0)
    return np.clip(s, 0, 1)


def _int_share_proportional(x: np.ndarray, O: np.ndarray, d: np.ndarray) -> np.ndarray:
    return _supply_share(x, d)


def _fin_share_proportional(x: np.ndarray, d: np.ndarray, Z: np.ndarray) -> np.ndarray:
    return _supply_share(x, d)


def _int_share_supplier(x: np.ndarray, O: np.ndarray, d: np.ndarray) -> np.ndarray:
    return _supply_share(x, np.sum(O, axis=1))


def _fin_share_supplier(x: np.ndarray, d: np.ndarray, Z: np.ndarray) -> np.ndarray:
    delivered = np.sum(Z, axis=1)
    return _supply_share(x - delivered, d - delivered)


ALLOCATION_RULES = {
    "no":  (_int_share_proportional, _fin_share_proportional),
    "yes": (_int_share_supplier, _fin_share_supplier),
}


class ConvergenceAbort(Exception):
    """Raised by a step_callback inside run_model() to signal divergence in a draw."""
    def __init__(self, t_step: int, region_idx: int, value: float):
        self.t_step     = t_step
        self.region_idx = region_idx
        self.value      = value
        super().__init__(
            f"Divergence at t={t_step}: region {region_idx} value={value:.1f}"
        )


# Core model
class InputOutputModel:
    """Dynamic disequilibrium input-output model, single- or multi-region."""

    # Initialisation
    def __init__(
        self,
        n_periods: int = 40,
        time_frequency: str = "daily",
        config: ModelConfig | None = None,
        _data_dict: dict | None = None,
    ):
        """Initialise from config, with _data_dict supplying a pre-loaded IO dict for MC loops."""
        if config is None:
            config = ModelConfig(n_periods=n_periods, time_frequency=time_frequency)
        else:
            config = config.clone()
            if config.n_periods != n_periods:
                logger.debug(
                    "Overriding config n_periods=%s with explicit argument=%s",
                    config.n_periods, n_periods,
                )
                config.n_periods = n_periods
            if config.time_frequency != time_frequency:
                logger.debug(
                    "Overriding config time_frequency=%s with explicit argument=%s",
                    config.time_frequency, time_frequency,
                )
                config.time_frequency = time_frequency

        self.config = config
        self.TT = config.n_periods
        self.N = 3
        self.time_frequency = config.time_frequency

        self._calculate_time_step_parameters()

        def _resolve_array(value, fallback):
            if value is None:
                return np.array([fallback], dtype=np.float64)
            return np.atleast_1d(np.asarray(value, dtype=np.float64))

        self.tau        = _resolve_array(config.tau,        DEFAULT_TAU)
        self.gamma_hire = _resolve_array(config.gamma_hire, DEFAULT_GAMMA_HIRE)
        self.gamma_fire = _resolve_array(config.gamma_fire, DEFAULT_GAMMA_FIRE)
        self.benefits       = float(np.atleast_1d(config.benefits)[0])
        self.c_other_coef   = float(np.atleast_1d(config.c_other_coef)[0])
        self.ces_elasticity = config.ces_elasticity
        self.income_tax_rate = float(
            np.atleast_1d(np.asarray(config.income_tax_rate, dtype=float))[0]
        )
        self.household_closure_mode = (
            config.household_closure_mode
            if isinstance(config.household_closure_mode, str)
            else config.household_closure_mode[0]
        )
        self.gov_income_elasticity = config.gov_income_elasticity
        self.investment_closure = config.investment_closure
        self.investment_adj_speed = float(config.investment_adj_speed)
        self.investment_savings_ema = float(config.investment_savings_ema)
        self.investment_scale_growth_cap = config.investment_scale_growth_cap
        self.price_passthrough_enabled = bool(config.price_passthrough_enabled)
        self.price_passthrough_pos     = float(config.price_passthrough_pos)
        self.price_passthrough_neg     = float(config.price_passthrough_neg)
        # Size 1 or n_regions, broadcast in run_model where the realised region count is known.
        self.price_deflate_household_income_r = np.atleast_1d(
            np.asarray(config.price_deflate_household_income, dtype=bool)
        ).copy()
        self.n_regions = config.n_regions

        self.prod_function      = config.prod_function
        self.hiringfiring       = config.hiringfiring
        self.firing_speed_damping = float(config.firing_speed_damping)
        self.firm_priority      = config.firm_priority
        self.inventory_days_config  = config.inventory_days
        self.inventory_days_daily   = config.inventory_days_daily
        self.inventory_days_other   = config.inventory_days_other
        self.wage_curve         = config.wage_curve
        self.wage_curve_beta    = config.wage_curve_beta
        self.wage_floor_ratio   = config.wage_floor_ratio

        self._raw_data: dict | None = _data_dict
        self._initialize_data()
        self._validate_parameters()

    # Pickle compatibility
    def __setstate__(self, state: dict) -> None:
        """Restore state from a pickle; backfill attributes absent from older pickles."""
        self.__dict__.update(state)
        if not hasattr(self, "_reg_perm") and hasattr(self, "region_sector_indices"):
            self._reg_perm = np.concatenate(self.region_sector_indices).astype(np.int64)
            _reg_sizes = np.array(
                [len(j) for j in self.region_sector_indices], dtype=np.int64
            )
            self._reg_starts = np.concatenate(
                [[0], np.cumsum(_reg_sizes)[:-1]]
            ).astype(np.int64)
        if not hasattr(self, "import_supplement_caps_"):
            self.import_supplement_caps_ = {}
        if not hasattr(self, "baseline_trend_r"):
            self.baseline_trend_r = np.ones((self.n_regions, self.TT))
        if not hasattr(self, "xi_r"):
            self.xi_r = None
        if not hasattr(self, "firing_speed_damping"):
            self.firing_speed_damping = 0.5
        if not hasattr(self, "initial_inventory_fill_"):
            self.initial_inventory_fill_ = {}
        if not hasattr(self, "inventory_days_pair_"):
            self.inventory_days_pair_ = {}

    # Data loading
    def _load_data_dict(self, path: Path) -> dict:
        """Load and return the data dict from a Python pickle file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Data file not found: {path}. "
                "The data file may be created by running the export script "
                "(e.g. from Archive/data/) from the CSV files."
            )
        with open(path, "rb") as f:
            return pickle.load(f)

    def _estimate_essential_inputs(self, A_matrix: np.ndarray) -> np.ndarray:
        """Estimate A_essential for leontief.adapted, refusing an empty result."""
        A_essential = estimate_essential_inputs_from_io_data(
            A_matrix, method="combined_linkage", x0=self.x0
        )
        if not A_essential.any():
            raise ValueError("leontief.adapted: no input was classified as essential.")
        return A_essential

    def _initialize_essential_inputs(self) -> None:
        """Initialise A_essential from the IO matrix when prod_function is leontief.adapted."""
        if self.prod_function == "leontief.adapted":
            self.A_essential = self._estimate_essential_inputs(self.A)
        else:
            self.A_essential = None

    def _resolve_inventory_days_vector(self) -> np.ndarray:
        """Resolve the per-sector inventory coverage vector (days) from config."""
        if self.inventory_days_config is not None:
            days = np.asarray(self.inventory_days_config, dtype=np.float64)
            if days.size == 1:
                days = np.full(self.N, float(days.squeeze()), dtype=np.float64)
            elif days.size != self.N:
                raise ValueError(
                    f"inventory_days length ({days.size}) does not match N ({self.N})"
                )
        else:
            default_days = (
                self.inventory_days_daily
                if self.time_frequency == "daily"
                else self.inventory_days_other
            )
            days = np.full(self.N, default_days, dtype=np.float64)
        return days

    def _expand_sector_parameter(
        self, value: np.ndarray, default_value: float, name: str
    ) -> np.ndarray:
        """Expand a scalar or sector-length array to length N."""
        arr = np.atleast_1d(np.asarray(value, dtype=np.float64))
        if arr.size == 0:
            return np.full(self.N, default_value, dtype=np.float64)
        if arr.size == 1:
            return np.full(self.N, float(arr.squeeze()), dtype=np.float64)
        if arr.size == self.N:
            return arr.astype(np.float64)
        raise ValueError(f"{name} length ({arr.size}) does not match N ({self.N})")

    # Household helpers
    def _validate_savings_rate_value(self, savings_rate: float) -> float:
        rate = float(savings_rate)
        if not 0 <= rate < 1:
            raise ValueError(f"savings_rate must be in [0, 1); got {savings_rate}")
        return rate

    def _extra_household_expenditure(
        self, consumption_total: float, c_other_coef: float | None = None
    ) -> float:
        """Return other household outlays implied by c_other_coef."""
        coc = self.c_other_coef if c_other_coef is None else float(c_other_coef)
        return coc / (1 - coc) * consumption_total

    def _infer_base_savings_rate(
        self,
        consumption_total: float,
        household_income: float,
        c_other_coef: float | None = None,
    ) -> float:
        """Return the savings rate implied by base-year household income and spending."""
        total_spending = float(consumption_total) + self._extra_household_expenditure(
            consumption_total, c_other_coef=c_other_coef
        )
        disposable_income = max(float(household_income), 1e-9)
        implied_rate = 1 - total_spending / disposable_income
        return self._validate_savings_rate_value(np.clip(implied_rate, 0.0, 1.0 - 1e-9))

    def _household_income(self, labour_income_total: float, profit_income_total: float) -> float:
        baseline_labour_income = float(np.sum(self.l0))
        adjusted_labour_income = (
            self.benefits * baseline_labour_income
            + (1 - self.benefits) * float(labour_income_total)
        )
        return max(adjusted_labour_income + float(profit_income_total), 1e-9)

    def _household_income_r(
        self, r: int, labour_income_total: float, profit_income_total: float
    ) -> float:
        beta     = self.benefits_r[r]
        baseline = float(np.sum(self.l0[self.region_sector_indices[r]]))
        adjusted = beta * baseline + (1 - beta) * float(labour_income_total)
        return max(adjusted + float(profit_income_total), 1e-9)

    def _household_income_signal_for_period_r(
        self, r: int, household_income_prev: float, xit: float
    ) -> float:
        hcm = self.household_closure_mode_r[r]
        if hcm == "scarred":
            # Wedge offsets the calibration-floor gap so scarring fires only on real income drops.
            wedge = float(self._baseline_income_wedge_r[r])
            return max(float(household_income_prev) + wedge, 1e-9)
        if hcm == "frozen":
            return max(self.base_household_income_r[r], 1e-9)
        return max(self.base_household_income_r[r] * float(xit), 1e-9)  # return_to_base
    def _household_consumption_capacity(
        self,
        household_income: float,
        household_wealth: float = 0.0,
        savings_rate: float | None = None,
        c_other_coef: float | None = None,
    ) -> float:
        rate = self.savings_rate if savings_rate is None else self._validate_savings_rate_value(savings_rate)
        coc  = self.c_other_coef if c_other_coef is None else float(c_other_coef)
        available = max(float(household_income) + float(household_wealth), 1e-9)
        total_spending = (1 - rate) * available
        return max((1 - coc) * total_spending, 1e-9)

    def _household_income_signal_for_period(
        self, household_income_prev: float, xit: float
    ) -> float:
        if self.household_closure_mode == "scarred":
            return max(float(household_income_prev), 1e-9)
        return max(self.base_household_income * float(xit), 1e-9)

    def _set_household_baseline(self, cons_vec_template: np.ndarray) -> None:
        """Initialise household shares, baseline income, and baseline consumption."""
        household_consumption_template = np.asarray(cons_vec_template, dtype=np.float64)
        template_total = float(np.sum(household_consumption_template))
        if template_total > 0:
            self.household_consumption_shares = household_consumption_template / template_total
        else:
            self.household_consumption_shares = np.full(self.N, 1 / self.N, dtype=np.float64)

        self.c0 = household_consumption_template.copy()
        self.base_consumption_total = float(np.sum(self.c0))
        self.base_household_income  = self._household_income(
            np.sum(self.l0), np.sum(self.profits0)
        )
        if self.config.savings_rate is None:
            self.savings_rate = self._infer_base_savings_rate(
                self.base_consumption_total, self.base_household_income
            )
        else:
            self.savings_rate = self._validate_savings_rate_value(self.config.savings_rate)
        self.base_consumption_total = float(np.sum(self.c0))
        self.theta_ = np.repeat(
            self.household_consumption_shares[:, np.newaxis], self.TT, axis=1
        )
        self.mpc = self.base_consumption_total / max(self.base_household_income, 1e-9)
        self.base_household_savings = (
            self.base_household_income
            - self.base_consumption_total
            - self._extra_household_expenditure(self.base_consumption_total)
        )

        # LES subsistence - aggregate (inspection attributes; simulation uses gamma_r / beta_r)
        sub_shares = np.atleast_1d(np.asarray(self.config.subsistence_shares, dtype=float))
        if sub_shares.size == 1:
            sub_shares = np.full(self.N, sub_shares.item(), dtype=float)
        sub_shares = np.clip(sub_shares, 0.0, 1.0 - 1e-9)
        self.subsistence_shares = sub_shares
        self.gamma_      = sub_shares * self.c0
        self.Gamma_total = float(self.gamma_.sum())
        supernumerary    = max(self.base_consumption_total - self.Gamma_total, 1e-9)
        self.beta_       = (self.c0 - self.gamma_) / supernumerary
        self.les_active  = bool(np.any(self.gamma_ > 0))

    def _set_per_region_baselines(self, cons_vec_nr: np.ndarray) -> None:
        """Compute per-region household baseline quantities from (N, R) consumption matrix."""
        R = self.n_regions
        self.c0_r                          = [None] * R
        self.base_consumption_total_r      = np.zeros(R)
        self.household_consumption_shares_r = [None] * R
        self.base_household_income_r       = np.zeros(R)
        # bhi - bhi_production; added to actual income in the scarred signal.
        self._baseline_income_wedge_r      = np.zeros(R)
        self.savings_rate_r                = np.zeros(R)
        self.theta_r                       = [None] * R
        self.mpc_r                         = np.zeros(R)
        self.gamma_r                       = [None] * R
        self.beta_r                        = [None] * R

        for r in range(R):
            J_r = self.region_sector_indices[r]
            c_r = cons_vec_nr[:, r]
            bct = float(np.sum(c_r))
            self.c0_r[r] = c_r.copy()
            self.base_consumption_total_r[r] = bct

            if bct > 0:
                self.household_consumption_shares_r[r] = c_r / bct
            else:
                self.household_consumption_shares_r[r] = np.full(self.N, 1.0 / self.N)

            labour_r  = float(np.sum(self.l0[J_r]))
            profits_r = float(np.sum(self.profits0[J_r]))
            bhi_production = self._household_income_r(r, labour_r, profits_r)
            coc_r  = float(self.c_other_coef_r[r])
            bhi_min = bct / max(1.0 - coc_r, 1e-9)
            bhi     = max(bhi_production, bhi_min)
            if bhi > bhi_production and bhi_production > 0:
                logger.debug(
                    "Region %d (%s): production income %.1f < consumption %.1f / "
                    "(1-%.2f) = %.1f; base_household_income_r floored to %.1f.",
                    r,
                    self.region_labels[r] if hasattr(self, "region_labels") else r,
                    bhi_production, bct, coc_r, bhi_min, bhi,
                )
            self.base_household_income_r[r] = bhi
            self._baseline_income_wedge_r[r] = max(bhi - bhi_production, 0.0)

            if self.config.savings_rate is None:
                sr = self._infer_base_savings_rate(bct, bhi, c_other_coef=coc_r)
            else:
                sr = self._validate_savings_rate_value(
                    float(np.atleast_1d(self.config.savings_rate)[0])
                )
            if self.config.savings_rate_by_skill is not None and self.l0_by_skill is not None:
                sr_sk = np.asarray(self.config.savings_rate_by_skill, dtype=float)
                l0_r  = self.l0_by_skill[:, J_r].sum(axis=1)           # (3,)
                phi_r = l0_r / l0_r.sum() if l0_r.sum() > 0 else np.full(3, 1.0 / 3.0)
                sr    = float(np.clip(1.0 - np.dot(1.0 - sr_sk, phi_r), 0.0, 0.999))
            self.savings_rate_r[r] = sr
            shares_r = self.household_consumption_shares_r[r]
            self.theta_r[r] = np.tile(shares_r[:, np.newaxis], (1, self.TT))
            self.mpc_r[r]   = bct / max(bhi, 1e-9)

            gamma_r_vec     = self.subsistence_shares * c_r
            supernumerary_r = max(bct - float(gamma_r_vec.sum()), 1e-9)
            self.gamma_r[r] = gamma_r_vec
            self.beta_r[r]  = (c_r - gamma_r_vec) / supernumerary_r

    def _refresh_household_scalars(self) -> None:
        """Recompute derived household scalars after savings_rate is patched externally."""
        implied = (
            (1.0 - self.savings_rate)
            * (1.0 - self.c_other_coef)
            * self.base_household_income
        )
        self.mpc = implied / max(self.base_household_income, 1e-9)
        self.base_household_savings = (
            self.base_household_income
            - implied
            - self._extra_household_expenditure(implied)
        )

    def _refresh_household_scalars_r(self, r: int) -> None:
        """Recompute region r's derived scalars after savings_rate_r or c_other_coef_r changes."""
        sr  = float(self.savings_rate_r[r])
        coc = float(self.c_other_coef_r[r])
        bhi = self.base_household_income_r[r]
        implied = (1.0 - sr) * (1.0 - coc) * bhi
        self.mpc_r[r] = implied / max(bhi, 1e-9)
        if r == 0:
            self.savings_rate = sr
            self._refresh_household_scalars()

    # Production helpers
    def _ces_output_constraint(
        self,
        input_capacity: np.ndarray,
        weights: np.ndarray,
        sigma: float | None = None,
    ) -> float:
        """Return a CES-style output bound from per-input capacities and weights."""
        valid = np.isfinite(input_capacity) & (weights > 0)
        if not np.any(valid):
            return np.inf
        q = np.maximum(input_capacity[valid], 0.0)
        w = weights[valid]
        w = w / np.sum(w)
        if sigma is None:
            sigma = float(self.ces_elasticity_vector[0])
        if np.isclose(sigma, 1.0):
            safe_q = np.maximum(q, 1e-12)
            return float(np.exp(np.sum(w * np.log(safe_q))))
        rho = (sigma - 1.0) / sigma
        aggregate = np.sum(w * np.power(q, rho))
        return float(np.power(max(aggregate, 0.0), 1.0 / rho))

    def _period_output_constraints(self, t: int) -> np.ndarray:
        """Return effective output constraints for period t including supplier-side shocks."""
        output_constraint = np.array(self.output_constraint_[:, t], copy=True, dtype=np.float64)
        if t in self.input_availability_shocks_:
            for supplier_idx, reduction_pct in self.input_availability_shocks_[t].items():
                shocked = (1 - reduction_pct) * self.x0[supplier_idx]
                output_constraint[supplier_idx] = min(output_constraint[supplier_idx], shocked)
        return output_constraint

    # Timing and parameter validation
    def _calculate_time_step_parameters(self) -> None:
        """Set dt, rho0, rho1 from time_frequency."""
        if self.time_frequency == "quarterly":
            self.dt   = 0.25
            self.rho1 = 0.6
            self.rho0 = 0.4
        elif self.time_frequency == "daily":
            self.dt   = 1 / 90
            # Habit persistence 0.6 from Havranek et al., doi:10.1016/j.euroecorev.2017.03.009.
            rho_bar   = 0.6
            # Compounded so persistence over 90 days equals the quarterly rho_bar.
            self.rho1 = rho_bar ** (1.0 / 90.0)
            self.rho0 = 1 - self.rho1
        else:
            self.dt   = 1.0
            self.rho1 = 0.6
            self.rho0 = 0.4

    def _validate_parameters(self) -> None:
        """Validate gamma_hire, gamma_fire, tau, region_map, and column viability."""
        if not np.all((self.gamma_hire >= GAMMA_HIRE_MIN) & (self.gamma_hire <= GAMMA_HIRE_MAX)):
            raise ValueError(
                f"gamma_hire must be in [{GAMMA_HIRE_MIN}, {GAMMA_HIRE_MAX}] for all sectors"
            )
        if not np.all((self.gamma_fire >= GAMMA_FIRE_MIN) & (self.gamma_fire <= GAMMA_FIRE_MAX)):
            raise ValueError(
                f"gamma_fire must be in [{GAMMA_FIRE_MIN}, {GAMMA_FIRE_MAX}] for all sectors"
            )
        if not np.all((self.tau >= TAU_MIN) & (self.tau <= TAU_MAX)):
            raise ValueError(f"tau must be in [{TAU_MIN}, {TAU_MAX}] for all sectors")
        if (
            len(self.gamma_hire) != self.N
            or len(self.gamma_fire) != self.N
            or len(self.tau) != self.N
        ):
            raise ValueError(
                "All sector-specific parameter arrays must have length N"
            )

        if len(self.region_map) != self.N:
            raise ValueError(
                f"region_map length ({len(self.region_map)}) must equal N ({self.N})"
            )
        expected = set(range(self.n_regions))
        actual   = set(int(r) for r in self.region_map)
        if actual != expected:
            raise ValueError(
                f"region_map values {actual} do not cover exactly "
                f"{{0, ..., {self.n_regions - 1}}}"
            )

        col_sums = np.sum(self.A, axis=0)
        if np.any(col_sums >= 1.0 + 1e-6):
            worst = int(np.argmax(col_sums))
            raise ValueError(
                f"Column viability violated: A column sums must be < 1. "
                f"Max col sum = {col_sums[worst]:.6f} at sector {worst} "
                f"({self.sector_labels[worst] if worst < len(self.sector_labels) else worst})"
            )

    # Data initialisation
    def _initialize_data(self) -> None:
        """Load IO data; initialise matrices, shares, and time-series buffers."""
        if self._raw_data is not None:
            data = self._raw_data
        else:
            data = self._load_data_dict(Path(self.config.data_path))
            self._raw_data = data   # cache for downstream reuse (e.g. MC loops)
        self.sector_labels  = data["sector_labels"]
        self.label_to_index = {label: idx for idx, label in enumerate(self.sector_labels)}
        self.Z0 = np.asarray(data["Z0"], dtype=np.float64)
        if self.Z0.ndim != 2 or self.Z0.shape[0] != self.Z0.shape[1]:
            raise ValueError(f"Z matrix must be square. Got shape {self.Z0.shape}")
        self.N = self.Z0.shape[0]
        self.tau        = self._expand_sector_parameter(self.tau,        DEFAULT_TAU,        "tau")
        self.gamma_hire = self._expand_sector_parameter(self.gamma_hire, DEFAULT_GAMMA_HIRE, "gamma_hire")
        self.gamma_fire = self._expand_sector_parameter(self.gamma_fire, DEFAULT_GAMMA_FIRE, "gamma_fire")
        self.ces_elasticity_vector = self._expand_sector_parameter(
            np.atleast_1d(np.asarray(self.config.ces_elasticity, dtype=float)),
            CES_ELASTICITY_DEFAULT, "ces_elasticity"
        )
        self.import_flexibility_vector = self._expand_sector_parameter(
            np.atleast_1d(np.asarray(self.config.import_flexibility, dtype=float)),
            0.0, "import_flexibility"
        )
        self.row_supply_cap_vector = self._expand_sector_parameter(
            np.atleast_1d(np.asarray(self.config.row_supply_cap, dtype=float)),
            0.0, "row_supply_cap"
        )
        self.export_pull_vector = self._expand_sector_parameter(
            np.atleast_1d(np.asarray(self.config.export_pull, dtype=float)),
            0.0, "export_pull"
        )
        self.fprod_l_: np.ndarray = np.ones(self.N, dtype=np.float64)
        self.fprod_k_: np.ndarray = np.ones(self.N, dtype=np.float64)
        # Registered productivity shocks: {t: {sector_idx: (prod_L, prod_K)}}
        self.fprod_changes: dict[int, dict[int, tuple[float, float]]] = {}

        # (N, TT) labour supply reference path; None → l0 used for all periods.
        self.labour_supply_schedule: np.ndarray | None = None
        # (N, TT) sector-level unemployment path for wage curve; None → wage curve inactive.
        self.unemployment_schedule: np.ndarray | None = None

        _region_map_raw = data.get("region_map", None)
        if _region_map_raw is not None:
            region_map = np.asarray(_region_map_raw, dtype=np.int32)
        elif self.config.region_map is not None:
            region_map = np.asarray(self.config.region_map, dtype=np.int32)
        else:
            region_map = np.zeros(self.Z0.shape[0], dtype=np.int32)

        n_regions_from_map = int(region_map.max()) + 1 if len(region_map) > 0 else 1
        if self.config.n_regions != 1 and n_regions_from_map != self.config.n_regions:
            logger.warning(
                "config.n_regions=%d but region_map in data implies %d regions; using %d.",
                self.config.n_regions, n_regions_from_map, n_regions_from_map,
            )
        self.n_regions = n_regions_from_map

        _rl_raw = data.get("region_labels", None)
        if _rl_raw is not None:
            self.region_labels = list(_rl_raw)
        elif self.config.region_labels is not None:
            self.region_labels = list(self.config.region_labels)
        else:
            self.region_labels = [f"Region{r}" for r in range(self.n_regions)]

        self.region_map = region_map
        self.region_sector_indices = [
            np.where(region_map == r)[0] for r in range(self.n_regions)
        ]

        # Region-block permutation and starts for reduceat-based aggregations.
        self._reg_perm = np.concatenate(self.region_sector_indices).astype(np.int64)
        _reg_sizes = np.array([len(j) for j in self.region_sector_indices], dtype=np.int64)
        self._reg_starts = np.concatenate([[0], np.cumsum(_reg_sizes)[:-1]]).astype(np.int64)

        # RoW region: label matching 'RoW' (case-insensitive), else last region.
        self.row_region_idx: int | None = None
        if self.n_regions > 1:
            for _idx, _lbl in enumerate(self.region_labels):
                if str(_lbl).lower() == "row":
                    self.row_region_idx = _idx
                    break
            if self.row_region_idx is None:
                self.row_region_idx = self.n_regions - 1

        # Map each sector to its RoW counterpart by bare label; -1 if no match.
        self.good_to_row_idx: np.ndarray = np.full(self.N, -1, dtype=np.int64)
        if self.row_region_idx is not None:
            row_J = self.region_sector_indices[self.row_region_idx]
            row_good_map: dict[str, int] = {}
            for _ri in row_J:
                _lbl = self.sector_labels[_ri]
                _bare = _lbl.split(":", 1)[1] if ":" in _lbl else _lbl
                row_good_map[_bare] = int(_ri)
            for _j in range(self.N):
                _lbl_j = self.sector_labels[_j]
                _bare_j = _lbl_j.split(":", 1)[1] if ":" in _lbl_j else _lbl_j
                self.good_to_row_idx[_j] = row_good_map.get(_bare_j, -1)

        # Per-sector hiring ceiling scalar; override from project code after construction.
        self.capacity_max_scale_: np.ndarray = np.full(self.N, CAPACITY_MAX_SCALE)

        self.benefits_r = np.broadcast_to(
            np.atleast_1d(np.asarray(self.config.benefits, dtype=np.float64)),
            (self.n_regions,),
        ).copy()
        self.c_other_coef_r = np.broadcast_to(
            np.atleast_1d(np.asarray(self.config.c_other_coef, dtype=np.float64)),
            (self.n_regions,),
        ).copy()
        self.income_tax_rate_r = np.broadcast_to(
            np.atleast_1d(np.asarray(self.config.income_tax_rate, dtype=np.float64)),
            (self.n_regions,),
        ).copy()
        if isinstance(self.config.household_closure_mode, str):
            self.household_closure_mode_r = [self.config.household_closure_mode] * self.n_regions
        else:
            self.household_closure_mode_r = list(self.config.household_closure_mode)

        def _ensure_2d(v: np.ndarray) -> np.ndarray:
            return v[:, np.newaxis] if v.ndim == 1 else v

        cons_vec  = _ensure_2d(np.asarray(data["cons_vec"],  dtype=np.float64))
        gov_vec   = _ensure_2d(np.asarray(data["gov_vec"],   dtype=np.float64))
        inv_vec   = _ensure_2d(np.asarray(data["inv_vec"],   dtype=np.float64))
        invnt_vec = _ensure_2d(np.asarray(data["invnt_vec"], dtype=np.float64))
        exp_vec   = _ensure_2d(np.asarray(data["exp_vec"],   dtype=np.float64))

        self.cons_vec_r  = cons_vec
        self.gov_vec_r   = gov_vec
        self.inv_vec_r   = inv_vec
        self.invnt_vec_r = invnt_vec
        self.exp_vec_r   = exp_vec

        cons_agg  = np.sum(cons_vec,  axis=1)
        gov_agg   = np.sum(gov_vec,   axis=1)
        inv_agg   = np.sum(inv_vec,   axis=1)
        invnt_agg = np.sum(invnt_vec, axis=1)
        exp_agg   = np.sum(exp_vec,   axis=1)

        f_total              = cons_agg + gov_agg + inv_agg + invnt_agg + exp_agg
        intermediate_outputs = np.sum(self.Z0, axis=1)
        self.x0 = intermediate_outputs + f_total

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            self.A = np.divide(
                self.Z0,
                self.x0[np.newaxis, :],
                out=np.zeros_like(self.Z0),
                where=self.x0[np.newaxis, :] != 0,
            )

        row_identity_check = np.sum(self.Z0, axis=1) + f_total
        if not np.allclose(self.x0, row_identity_check, atol=ROW_IDENTITY_ATOL):
            max_diff = np.max(np.abs(self.x0 - row_identity_check))
            raise ValueError(
                f"Row IO identity violation. Max diff: {max_diff:.10f}"
            )

        self.l0   = np.asarray(data["l0"],   dtype=np.float64)
        self.cap0 = np.asarray(data["cap0"], dtype=np.float64)
        self.tax0 = np.asarray(data["tax0"], dtype=np.float64)
        self.imp0 = np.asarray(data["imp0"], dtype=np.float64)
        if len(self.l0) != self.N or len(self.cap0) != self.N:
            raise ValueError(
                f"Data sector count mismatch: l0/cap0/tax0/imp0 must have length N ({self.N})"
            )

        self.other0   = np.zeros(self.N)
        profits0_data = np.asarray(data.get("profits0", np.zeros(self.N)), dtype=np.float64)

        intermediate_inputs_domestic = np.sum(self.Z0, axis=0)
        intermediate_inputs_total    = intermediate_inputs_domestic + self.imp0
        value_added_components = self.l0 + self.cap0 + self.tax0 + self.other0 + profits0_data
        value_added_from_identity = self.x0 - intermediate_inputs_total

        if not np.allclose(value_added_from_identity, value_added_components, atol=VA_IDENTITY_TOLERANCE):
            max_diff = np.max(np.abs(value_added_from_identity - value_added_components))
            raise ValueError(
                f"Value added inconsistency: max difference = {max_diff}. "
                f"VA from column identity: {value_added_from_identity[:5]}, "
                f"VA from components: {value_added_components[:5]}"
            )

        column_identity_check = intermediate_inputs_total + value_added_components
        max_col_diff = np.max(np.abs(self.x0 - column_identity_check))
        if max_col_diff > VA_IDENTITY_TOLERANCE:
            raise ValueError(
                f"Column IO identity violation. Max diff: {max_col_diff:.10f}."
            )

        # After the identity checks, so malformed data reports the identity.
        self._initialize_essential_inputs()

        value_added = value_added_from_identity
        self.profits0 = value_added - self.l0 - self.cap0 - self.tax0 - self.other0

        # Skill-disaggregated wage bill (3, N); None when pkl pre-dates D1.
        self.l0_by_skill: np.ndarray | None = (
            np.asarray(data["l0_by_skill"], dtype=np.float64)
            if "l0_by_skill" in data else None
        )

        # KLEMS: validate, classify sectors, pre-compute CES weights.
        if self.prod_function == "klems":
            if self.l0_by_skill is None:
                raise ValueError(
                    "prod_function='klems' requires 'l0_by_skill' in the data pickle. "
                    "Regenerate the pkl after adding l0_by_skill to prepare_exiobase.py."
                )
            self.klems_masks: dict | None = classify_klems(
                self.sector_labels, self.config.klems_m_scope)
            self._init_klems_weights()
        else:
            self.klems_masks = None

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            self.cap_share = np.divide(
                self.cap0, self.x0, out=np.zeros_like(self.cap0), where=self.x0 != 0
            )
            self.tax_share = np.divide(
                self.tax0, self.x0, out=np.zeros_like(self.tax0), where=self.x0 != 0
            )
            self.imp_share = np.divide(
                self.imp0, self.x0, out=np.zeros_like(self.imp0), where=self.x0 != 0
            )

        self.n = self._resolve_inventory_days_vector()
        self.delta_ = np.zeros((self.N, self.TT))

        # epsilon_r is (R, TT); epsilon_ is a convenience view into row 0.
        self.epsilon_r = np.zeros((self.n_regions, self.TT))
        self.epsilon_  = self.epsilon_r[0]
        # Exogenous real-income stream (R, TT) added to household income each period, default zero.
        self.income_spillover_r = np.zeros((self.n_regions, self.TT))
        # Baseline-trend multiplier (R, TT), so the closures return to a trending counterfactual.
        self.baseline_trend_r = np.ones((self.n_regions, self.TT))
        # Optional per-region expectation schedule (R, TT), replacing the global xi_ when set.
        self.xi_r: np.ndarray | None = None
        self.xi_       = np.ones(self.TT)
        self.output_constraint_        = np.full((self.N, self.TT), np.inf)
        self.input_availability_shocks_: dict[int, dict] = {}
        self.rationing_shocks_:          dict[int, dict] = {}
        # Bilateral delivery caps: {t: {(supplier_idx, buyer_region): (pct, hh_pct)}}.
        self.bilateral_delivery_shocks_: dict[int, dict] = {}
        self._hh_shortfall_: dict = {}  # {(row, region): capped household demand}
        # Stock term matching the intermediate channel's flexibility x cover.
        self.hh_supplement_cover: np.ndarray | None = None
        # Per-period capacity caps on the import supplement: {t: {row_sector_idx: level}}.
        self.import_supplement_caps_: dict[int, dict[int, float]] = {}
        self.supplement_sources_: dict[int, tuple[int, ...]] = {}
        # Initial stock fractions of target: {(input_idx, buyer_idx): fraction}.
        self.initial_inventory_fill_: dict[tuple[int, int], float] = {}
        # Per-pair inventory-target days overriding the buyer's uniform n.
        self.inventory_days_pair_: dict[tuple[int, int], float] = {}

        self._supp_by_source_: dict[int, float] = {}
        self._supp_alloc_: dict[int, tuple] = {}
        # Technical-change events: {t: new_A (N,N)}.
        self.A_changes: dict[int, np.ndarray] = {}

        # Direct unit-cost shocks for pass-through, each a dict active over its window.
        self.price_cost_shocks_: list[dict] = []
        self.L_price_pos = None
        self.L_price_neg = None
        if self.price_passthrough_enabled:
            self.L_price_pos, self.L_price_neg = self._build_price_inverses(self.A)

        self._set_household_baseline(cons_agg)
        self._set_per_region_baselines(cons_vec)

        # Government elasticity denominator: production-side income so gov_scale=1 at t=0.
        self.base_gov_income_total = max(
            float(np.sum(self.base_household_income_r)), 1e-9
        )

        # Keynesian investment denominator: same savings_s_regional formula keeps scale=1 at t=0.
        self.base_savings_total = max(
            float(np.sum(self.savings_s_regional(
                self.base_household_income_r,
                self.base_consumption_total_r,
            ))),
            1e-9,
        )

        self.consumer_taxes_total = float(data["consumer_taxes_total"])
        self.fd_imports_totals    = dict(data["fd_imports_totals"])

        self.fd_government_  = np.zeros((self.N, self.TT))
        self.fd_investment_  = np.zeros((self.N, self.TT))
        self.fd_inventories_ = np.zeros((self.N, self.TT))
        self.fd_exports_     = np.zeros((self.N, self.TT))
        for t in range(self.TT):
            self.fd_government_[:, t]  = gov_agg
            self.fd_investment_[:, t]  = inv_agg
            self.fd_inventories_[:, t] = invnt_agg
            self.fd_exports_[:, t]     = exp_agg
        self.fd_other_ = np.zeros((self.N, self.TT))
        # Scaling anchors for endogenous government and investment closures.
        self.gov_agg_base = gov_agg.copy()
        self.inv_agg_base = inv_agg.copy()

        # Non-RoW sector indices for export-pull; empty without RoW.
        self.J_named: np.ndarray = np.array([], dtype=np.int64)
        if self.row_region_idx is not None and self.n_regions > 1:
            self.J_named = np.concatenate([
                self.region_sector_indices[r]
                for r in range(self.n_regions) if r != self.row_region_idx
            ])

    def _init_klems_weights(self) -> None:
        """Pre-compute base-year CES weights and masks for the KLEMS nest."""
        e_mask  = self.klems_masks["energy"]
        m_mask  = self.klems_masks["materials"]
        nd_mask = self.klems_masks["other"]   # Leontief residual

        self._klems_e_idx  = np.where(e_mask)[0]
        self._klems_m_idx  = np.where(m_mask)[0]
        self._klems_nd_idx = np.where(nd_mask)[0]

        # E sub-aggregate weights (n_E, N), A[e_i, j] normalised within column.
        A_E      = self.A[e_mask, :]                   # (n_E, N)
        e_colsum = A_E.sum(axis=0)                     # (N,)
        e_safe   = np.where(e_colsum > 0, e_colsum, 1.0)
        self.klems_w_E   = A_E / e_safe[np.newaxis, :] # (n_E, N)
        self._klems_has_e = e_colsum > 0               # (N,) bool

        # M sub-aggregate weights (n_M, N)
        A_M      = self.A[m_mask, :]                   # (n_M, N)
        m_colsum = A_M.sum(axis=0)                     # (N,)
        m_safe   = np.where(m_colsum > 0, m_colsum, 1.0)
        self.klems_w_M   = A_M / m_safe[np.newaxis, :] # (n_M, N)
        self._klems_has_m = m_colsum > 0               # (N,) bool

        # ND flag
        nd_colsum         = self.A[nd_mask, :].sum(axis=0) if nd_mask.any() else np.zeros(self.N)
        self._klems_has_nd = nd_colsum > 0             # (N,) bool

        # KL weights use non-negative parts, as negative factor income puts weights outside [0, 1].
        _l0_pos  = np.maximum(self.l0,  0.0)
        _cap_pos = np.maximum(self.cap0, 0.0)
        kl_total = _l0_pos + _cap_pos                  # (N,)
        kl_safe  = np.where(kl_total > 0, kl_total, 1.0)
        self.klems_w_L = _l0_pos  / kl_safe            # (N,)
        self.klems_w_K = _cap_pos / kl_safe            # (N,)

        # KLE weights are value shares in gross output, KL from factor income, E and M from A.
        x0_safe  = np.where(self.x0 > 0, self.x0, 1.0)
        kl_share = kl_total / x0_safe                  # (N,)
        kle_total = kl_share + e_colsum + m_colsum     # (N,)
        kle_safe  = np.where(kle_total > 0, kle_total, 1.0)
        self.klems_w_KL_top = kl_share  / kle_safe     # (N,)
        self.klems_w_E_top  = e_colsum  / kle_safe     # (N,)
        self.klems_w_M_top  = m_colsum  / kle_safe     # (N,)

        # Two-level top nest weights: KL against E inside, the bundle against M.
        _in_total = self.klems_w_KL_top + self.klems_w_E_top
        _in_safe  = np.where(_in_total > 0, _in_total, 1.0)
        self._klems_w_KL_in  = self.klems_w_KL_top / _in_safe
        self._klems_w_E_in   = self.klems_w_E_top  / _in_safe
        self._klems_w_IN_top = _in_total

        # Skill-tier weights for L sub-aggregate (3, N)
        l0_safe = np.where(self.l0 > 0, self.l0, 1.0)
        self.klems_w_skill = self.l0_by_skill / l0_safe[np.newaxis, :]  # (3, N)

        # Per-sector sigma vectors
        self.klems_sigma_e_vec = self._expand_sector_parameter(
            np.atleast_1d(np.asarray(self.config.klems_sigma_e,   dtype=float)),
            0.5, "klems_sigma_e",
        )
        self.klems_sigma_m_vec = self._expand_sector_parameter(
            np.atleast_1d(np.asarray(self.config.klems_sigma_m,   dtype=float)),
            0.3, "klems_sigma_m",
        )
        self.klems_sigma_kle_vec = self._expand_sector_parameter(
            np.atleast_1d(np.asarray(self.config.klems_sigma_kle, dtype=float)),
            0.3, "klems_sigma_kle",
        )
        self.klems_sigma_klem_vec = self._expand_sector_parameter(
            np.atleast_1d(np.asarray(self.config.klems_sigma_klem, dtype=float)),
            0.3, "klems_sigma_klem",
        )
        self.klems_sigma_kl_vec = self._expand_sector_parameter(
            np.atleast_1d(np.asarray(self.config.klems_sigma_kl,  dtype=float)),
            0.7, "klems_sigma_kl",
        )
        self.klems_sigma_l_vec = self._expand_sector_parameter(
            np.atleast_1d(np.asarray(self.config.klems_sigma_l,   dtype=float)),
            1.5, "klems_sigma_l",
        )

        # CES branch data fixed for the model lifetime, keeping np.isclose out of the hot path.
        def _cd_rho(sigma_v):
            cd  = np.isclose(sigma_v, 1.0)
            rho = np.where(cd, 1.0, (sigma_v - 1.0) / sigma_v)
            return cd, rho, bool(cd.any()), bool(cd.all())

        self._klems_kl_cd,  self._klems_kl_rho,  self._klems_kl_any_cd,  self._klems_kl_all_cd  = _cd_rho(self.klems_sigma_kl_vec)
        self._klems_e_cd,   self._klems_e_rho,   self._klems_e_any_cd,   self._klems_e_all_cd   = _cd_rho(self.klems_sigma_e_vec)
        self._klems_m_cd,   self._klems_m_rho,   self._klems_m_any_cd,   self._klems_m_all_cd   = _cd_rho(self.klems_sigma_m_vec)
        self._klems_kle_cd, self._klems_kle_rho, self._klems_kle_any_cd, self._klems_kle_all_cd = _cd_rho(self.klems_sigma_kle_vec)
        self._klems_klem_cd, self._klems_klem_rho, self._klems_klem_any_cd, self._klems_klem_all_cd = _cd_rho(self.klems_sigma_klem_vec)

        # Pre-computed weight presence masks
        self._klems_w_L_pos    = self.klems_w_L      > 0  # (N,)
        self._klems_w_K_pos    = self.klems_w_K      > 0  # (N,)
        self._klems_kl_zero    = ~self._klems_w_L_pos & ~self._klems_w_K_pos
        self._klems_w_KL_pos   = self.klems_w_KL_top > 0  # (N,)
        self._klems_w_E_tp_pos = self.klems_w_E_top  > 0  # (N,)
        self._klems_w_M_tp_pos = self.klems_w_M_top  > 0  # (N,)
        self._klems_w_KL_in_pos = self._klems_w_KL_in  > 0  # (N,)
        self._klems_w_E_in_pos  = self._klems_w_E_in   > 0  # (N,)
        self._klems_w_IN_pos    = self._klems_w_IN_top > 0  # (N,)

        # Pre-extracted base-year A slices for E, M, ND, keeping np.where off the hot path.
        def _a_slice(idx):
            if len(idx) == 0:
                return np.zeros((0, self.N), dtype=bool), np.ones((0, self.N))
            A_g = self.A[idx, :]
            nz  = A_g > 0
            return nz, np.where(nz, A_g, 1.0)

        self._klems_e_nz,  self._klems_A_e_safe  = _a_slice(self._klems_e_idx)
        self._klems_m_nz,  self._klems_A_m_safe  = _a_slice(self._klems_m_idx)
        self._klems_nd_nz, self._klems_A_nd_safe = _a_slice(self._klems_nd_idx)

    # Single-period economic building blocks
    def hire_fire(
        self,
        t: int,
        l_: np.ndarray,
        x_: np.ndarray,
        delta_: np.ndarray,
        prod_constraints: np.ndarray,
        l_ref_t: np.ndarray | None = None,
        fprod_l_t: np.ndarray | None = None,
        U_r_prev: np.ndarray | None = None,
        U_r_0: np.ndarray | None = None,
        desired_l_out: np.ndarray | None = None,
    ) -> np.ndarray:
        """Update labour for period t via hiring/firing speeds and slack. fprod_l_t deflates the
        labour-output ratio, and U_r_prev/U_r_0 are required only for wage_curve=True."""
        if not self.hiringfiring:
            if desired_l_out is not None:
                desired_l_out[:] = l_[:, t - 1]
            return l_[:, t - 1]

        lbase = l_ref_t if l_ref_t is not None else l_[:, 0]

        base_capacity = lbase * np.clip(1 - delta_[:, t], DELTA_FLOOR, DELTA_CAP)
        max_capacity  = lbase * self.capacity_max_scale_
        min_capacity  = l_[:, 0] * CAPACITY_MIN_SCALE
        disruption_active = delta_[:, t] > 0
        upper_capacity = np.where(
            disruption_active,
            np.minimum(max_capacity, np.maximum(base_capacity, l_[:, t - 1])),
            max_capacity,
        )
        # Effective x0 scales up with fprod_l_t so the labour-output ratio stays correct.
        if fprod_l_t is not None:
            eff_x0 = x_[:, 0] * fprod_l_t
        else:
            eff_x0 = x_[:, 0]
        labor_share = np.divide(
            l_[:, 0], eff_x0, out=np.zeros_like(l_[:, 0]), where=eff_x0 != 0
        )
        # Blanchflower-Oswald wage curve: tighter labour markets raise effective labour cost.
        if self.wage_curve and U_r_prev is not None and U_r_0 is not None:
            # Floor at 1e-4, since a negative U (labour shortage) to a fractional power is NaN.
            _uf = 1e-4
            U0_safe  = np.maximum(U_r_0,   _uf)
            U_p_safe = np.maximum(U_r_prev, _uf)
            w_adj = (U_p_safe / U0_safe) ** (-np.asarray(self.wage_curve_beta))
            # Downward nominal wage rigidity: wage cannot fall below wage_floor_ratio * w0.
            if self.wage_floor_ratio is not None:
                w_adj = np.maximum(w_adj, self.wage_floor_ratio)
            labor_share = labor_share * w_adj
        desired_output = np.min(prod_constraints[:, 1:4], axis=1)
        desired_l      = np.clip(labor_share * desired_output, min_capacity, upper_capacity)
        if desired_l_out is not None:
            desired_l_out[:] = desired_l
        gap            = desired_l - l_[:, t - 1]
        hire_ix        = gap > 0
        gam            = np.where(hire_ix, self.gamma_hire,
                                  self.gamma_fire * self.firing_speed_damping)
        new_l          = np.clip(l_[:, t - 1] + gam * gap, min_capacity, upper_capacity)
        return new_l

    def findemand_cd(
        self,
        theta: np.ndarray,
        Cdt: float,
        xit: float,
        household_income_signal: float,
        eps: float,
        *,
        base_consumption_total: float | None = None,
        savings_rate: float | None = None,
        household_closure_mode: str | None = None,
        base_household_income: float | None = None,
        c_other_coef: float | None = None,
        income_tax_rate: float | None = None,
        gamma: np.ndarray | None = None,
        beta: np.ndarray | None = None,
    ) -> tuple[float, np.ndarray]:
        """Update consumption demand from household closure, persistence, and expectations. Income
        tax reduces capacity only (savings unaffected), and non-zero gamma/beta activate LES."""
        bct = self.base_consumption_total if base_consumption_total is None else float(base_consumption_total)
        sr  = self.savings_rate           if savings_rate           is None else savings_rate
        hcm = self.household_closure_mode if household_closure_mode is None else household_closure_mode
        ty  = self.income_tax_rate        if income_tax_rate        is None else float(income_tax_rate)
        bhi = self.base_household_income  if base_household_income  is None else float(base_household_income)
        coc = self.c_other_coef           if c_other_coef           is None else float(c_other_coef)

        les = (gamma is not None) and bool(np.any(gamma > 0))

        if hcm == "frozen":
            if les:
                Gamma   = float(gamma.sum())
                supernu = max(bct - Gamma, 1e-9)
                cd      = gamma * (1 - eps) + beta * supernu * (1 - eps)
            else:
                cd = theta * bct * (1 - eps)
            return bct, cd

        current_capacity  = self._household_consumption_capacity(
            max(float(household_income_signal) * (1.0 - ty), 1e-9), savings_rate=sr, c_other_coef=coc
        )
        expected_capacity = self._household_consumption_capacity(
            max(bhi * xit * (1.0 - ty), 1e-9), savings_rate=sr, c_other_coef=coc
        )
        baseline_ct = max(bct, 1.0)
        log_prev_ratio = (
            np.log(max(Cdt / baseline_ct, CONSUMPTION_FLOOR_RATIO))
            if hcm == "scarred"
            else 0.0
        )
        log_current_ratio  = np.log(max(current_capacity  / baseline_ct, CONSUMPTION_FLOOR_LABOUR_RATIO, 1e-9))
        log_expected_ratio = np.log(max(expected_capacity / baseline_ct, CONSUMPTION_FLOOR_RATIO, 1e-9))
        Cdt_new = baseline_ct * np.exp(
            self.rho1 * log_prev_ratio
            + self.rho0 / 2 * log_current_ratio
            + self.rho0 / 2 * log_expected_ratio
        )
        floor_consumption = min(baseline_ct * CONSUMPTION_FLOOR_RATIO, current_capacity)
        Cdt_new = min(max(Cdt_new, floor_consumption), current_capacity)

        if les:
            Gamma   = float(gamma.sum())
            supernu = max(Cdt_new - Gamma, 1e-9)
            cd      = gamma * (1 - eps) + beta * supernu * (1 - eps)
        else:
            cd = theta * Cdt_new * (1 - eps)
        return Cdt_new, cd

    def _findemand_cd_regional(
        self, r: int, t: int, Cdt_r_prev: float, household_income_signal: float
    ) -> tuple[float, np.ndarray]:
        """Convenience wrapper: call findemand_cd with region r's parameters."""
        trend = float(self.baseline_trend_r[r, t])
        xit = float(self.xi_r[r, t]) if self.xi_r is not None else self.xi_[t]
        return self.findemand_cd(
            self.theta_r[r][:, t],
            Cdt_r_prev,
            xit,
            household_income_signal,
            self.epsilon_r[r, t],
            base_consumption_total=self.base_consumption_total_r[r] * trend,
            savings_rate=float(self.savings_rate_r[r]),
            household_closure_mode=self.household_closure_mode_r[r],
            base_household_income=self.base_household_income_r[r],
            c_other_coef=float(self.c_other_coef_r[r]),
            income_tax_rate=float(self.income_tax_rate_r[r]),
            gamma=self.gamma_r[r] if self.les_active else None,
            beta=self.beta_r[r]   if self.les_active else None,
        )

    def orders_O(
        self,
        A: np.ndarray,
        d: np.ndarray,
        tau: np.ndarray,
        S_tar: np.ndarray,
        S: np.ndarray,
    ) -> np.ndarray:
        """Return intermediate orders combining current-use demand and inventory replenishment."""
        d     = np.nan_to_num(d,     nan=0.0, posinf=NUMERIC_LARGE, neginf=0.0)
        S_tar = np.nan_to_num(S_tar, nan=0.0, posinf=NUMERIC_LARGE, neginf=0.0)
        S     = np.nan_to_num(S,     nan=0.0, posinf=NUMERIC_LARGE, neginf=0.0)
        use_orders  = A * d[np.newaxis, :]
        # Excess stock damps orders through the signed gap, per the appendix.
        restock_gap = S_tar - S
        O = use_orders + restock_gap / tau[:, np.newaxis]
        return np.maximum(O, 0)

    def producing_x(
        self,
        prod_f: str,
        A_essential: np.ndarray | None,
        xcap0: np.ndarray,
        l_: np.ndarray,
        S: np.ndarray,
        A: np.ndarray,
        d: np.ndarray,
        t: int,
        fprod_l: np.ndarray | None = None,
        fprod_k: np.ndarray | None = None,
        S_tar: np.ndarray | None = None,
    ) -> dict:
        """Compute feasible output per sector from labour, inventories, and supplier constraints.
        fprod_k enters the klems branch only, and S_tar is needed when import flexibility is on."""
        fprod_l_eff = self.fprod_l_ if fprod_l is None else fprod_l
        fprod_k_eff = self.fprod_k_ if fprod_k is None else fprod_k

        # Tracked separately so profits deduct the supplement and the stock restore returns it.
        if S_tar is not None and np.any(self.import_flexibility_vector > 0):
            shortfall = np.maximum(S_tar - S, 0.0)
            import_supplement = self.import_flexibility_vector[:, np.newaxis] * shortfall
            self._cap_import_supplement(import_supplement, t)
            S_eff = S + import_supplement
        else:
            import_supplement = np.zeros_like(S)
            S_eff = S

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            xcap = np.divide(
                l_[:, t], l_[:, 0],
                out=np.full_like(l_[:, t], np.inf),
                where=l_[:, 0] != 0,
            ) * xcap0 * fprod_l_eff
            xcap[l_[:, 0] == 0] = np.inf

        has_supplement = np.any(import_supplement > 0)

        # cap[i,k] = S[i,k]/A[i,k]: shared input-capacity primitive for Leontief/CES branches.
        if prod_f in ("leontief", "leontief.adapted", "ces"):
            nz       = A > 0
            A_safe   = np.where(nz, A, 1.0)
            cap_eff  = np.where(nz, S_eff / A_safe, 0.0)
            cap_orig = np.where(nz, S     / A_safe, 0.0)

        if prod_f in ("leontief", "leontief.adapted"):
            if prod_f == "leontief.adapted" and A_essential is not None:
                ess_mask = (A_essential > ESSENTIAL_INPUT_THRESHOLD) & nz
                nes_mask = (A_essential <= ESSENTIAL_INPUT_THRESHOLD) & nz

                # Essential branch: per-column min S/A; columns with no essentials give +inf.
                ess_eff = np.where(ess_mask, cap_eff,  np.inf).min(axis=0)
                ess_ns  = np.where(ess_mask, cap_orig, np.inf).min(axis=0)

                # Non-essential branch, A-weighted mean of S/A, +inf for columns with none.
                A_nes        = np.where(nes_mask, A, 0.0)
                A_nes_colsum = A_nes.sum(axis=0)
                has_nes      = A_nes_colsum > 0
                num_eff      = (A_nes * cap_eff ).sum(axis=0)
                num_ns       = (A_nes * cap_orig).sum(axis=0)
                den          = np.where(has_nes, A_nes_colsum, 1.0)
                nes_eff      = np.where(has_nes, num_eff / den, np.inf)
                nes_ns       = np.where(has_nes, num_ns  / den, np.inf)

                xinp    = np.minimum(ess_eff, nes_eff)
                xinp_ns = np.minimum(ess_ns,  nes_ns)
            else:
                # Plain Leontief: per-column min over rows with A[i,k] > 0.
                xinp    = np.where(nz, cap_eff,  np.inf).min(axis=0)
                xinp_ns = np.where(nz, cap_orig, np.inf).min(axis=0)

        elif prod_f == "linear":
            inpshare = A.sum(axis=0)
            mask     = inpshare > 0
            xinp     = np.where(mask, np.divide(S_eff.sum(axis=0), inpshare, where=mask,
                                                out=np.full(self.N, np.inf)), np.inf)
            xinp_ns  = np.where(mask, np.divide(S    .sum(axis=0), inpshare, where=mask,
                                                out=np.full(self.N, np.inf)), np.inf)

        elif prod_f == "ces":
            # CES per column with q=S/A, w=A/colsum(A), rho=(sigma-1)/sigma, and CD when sigma=1.
            A_colsum    = A.sum(axis=0)
            has_inp     = A_colsum > 0
            colsum_safe = np.where(has_inp, A_colsum, 1.0)
            w           = np.where(nz, A / colsum_safe[np.newaxis, :], 0.0)

            sigma   = self.ces_elasticity_vector
            cd_mask = np.isclose(sigma, 1.0)
            any_cd  = bool(cd_mask.any())
            all_cd  = bool(cd_mask.all())

            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                if any_cd:
                    # Outside input mask: 1.0 so log()=0 and masked entries don't contribute.
                    safe_eff = np.where(nz, np.maximum(cap_eff,  1e-12), 1.0)
                    safe_ns  = np.where(nz, np.maximum(cap_orig, 1e-12), 1.0)
                    cd_eff   = np.exp((w * np.log(safe_eff)).sum(axis=0))
                    cd_ns    = np.exp((w * np.log(safe_ns )).sum(axis=0))

                if not all_cd:
                    rho     = np.where(cd_mask, 1.0, (sigma - 1.0) / sigma)  # CD cols use rho=1 as placeholder
                    q_pow_eff = np.where(nz, np.power(np.maximum(cap_eff,  0.0), rho[np.newaxis, :]), 0.0)
                    q_pow_ns  = np.where(nz, np.power(np.maximum(cap_orig, 0.0), rho[np.newaxis, :]), 0.0)
                    agg_eff   = (w * q_pow_eff).sum(axis=0)
                    agg_ns    = (w * q_pow_ns ).sum(axis=0)
                    inv_rho   = 1.0 / rho
                    ces_eff   = np.power(np.maximum(agg_eff, 0.0), inv_rho)
                    ces_ns    = np.power(np.maximum(agg_ns,  0.0), inv_rho)

            if all_cd:
                xinp, xinp_ns = cd_eff, cd_ns
            elif not any_cd:
                xinp, xinp_ns = ces_eff, ces_ns
            else:
                xinp    = np.where(cd_mask, cd_eff, ces_eff)
                xinp_ns = np.where(cd_mask, cd_ns,  ces_ns )
            xinp    = np.where(has_inp, xinp,    np.inf)
            xinp_ns = np.where(has_inp, xinp_ns, np.inf)
        elif prod_f == "klems":
            # KLEMS nest, x_j = min(KLE_j, ND_j, d_j, output_constraint_j), KLE a two-level CES.
            if self.klems_masks is None or self.l0_by_skill is None:
                raise ValueError(
                    "prod_function='klems' requires KLEMS masks and l0_by_skill. "
                    "Ensure prod_function='klems' was set before _initialize_data()."
                )

            # All _klems_* quantities below are pre-computed at initialisation.

            # KL bundle, with xcap (= xcap_L) already computed above from l_[:,t] / l_[:,0].
            xcap_K = xcap0 * fprod_k_eff
            rho_kl = self._klems_kl_rho        # pre-computed (N,)
            w_L    = self.klems_w_L
            w_K    = self.klems_w_K
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                if self._klems_kl_all_cd:
                    KL_j = np.exp(
                        np.where(self._klems_w_L_pos, w_L * np.log(np.maximum(xcap,   1e-12)), 0.0)
                        + np.where(self._klems_w_K_pos, w_K * np.log(np.maximum(xcap_K, 1e-12)), 0.0)
                    )
                else:
                    # Normalised by the per-column maximum so level^rho never under- or overflows.
                    _xl = np.minimum(np.maximum(xcap,   1e-30), 1e30)
                    _xk = np.minimum(np.maximum(xcap_K, 1e-30), 1e30)
                    _ref_kl = np.maximum(
                        np.maximum(np.where(self._klems_w_L_pos, _xl, 0.0),
                                   np.where(self._klems_w_K_pos, _xk, 0.0)),
                        1e-30,
                    )
                    KL_j = _ref_kl * np.power(
                        np.maximum(
                            np.where(self._klems_w_L_pos, w_L * np.power(np.maximum(_xl / _ref_kl, 1e-12), rho_kl), 0.0)
                            + np.where(self._klems_w_K_pos, w_K * np.power(np.maximum(_xk / _ref_kl, 1e-12), rho_kl), 0.0),
                            1e-300,
                        ),
                        1.0 / rho_kl,
                    )
                    if self._klems_kl_any_cd:
                        KL_cd = np.exp(
                            np.where(self._klems_w_L_pos, w_L * np.log(np.maximum(xcap,   1e-12)), 0.0)
                            + np.where(self._klems_w_K_pos, w_K * np.log(np.maximum(xcap_K, 1e-12)), 0.0)
                        )
                        KL_j = np.where(self._klems_kl_cd, KL_cd, KL_j)
            KL_j = np.where(self._klems_kl_zero, np.maximum(xcap, xcap_K), KL_j)
            kl_f = np.where(np.isfinite(KL_j), KL_j, 0.0)  # pre-clamp inf for KLE

            # E and M CES sub-aggregates, _ces processing both eff and ns arms in one call.
            def _ces(nz, A_safe, w, rho, cd, any_cd, all_cd, has_grp, S_e, S_n):
                if nz.shape[0] == 0:
                    inf = np.full(self.N, np.inf)
                    return inf, inf
                q_e = np.where(nz, S_e / A_safe, 0.0)
                q_n = np.where(nz, S_n / A_safe, 0.0)
                with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                    if all_cd:
                        agg_e = np.exp((w * np.log(np.where(nz, np.maximum(q_e, 1e-12), 1.0))).sum(axis=0))
                        agg_n = np.exp((w * np.log(np.where(nz, np.maximum(q_n, 1e-12), 1.0))).sum(axis=0))
                    else:
                        inv_rho  = 1.0 / rho                        # computed once, used twice
                        rho_bc   = rho[np.newaxis, :]                # broadcast shape (1, N)
                        # Normalised by the per-column maximum so levels cannot under- or overflow.
                        q_e_c    = np.minimum(np.where(nz, np.maximum(q_e, 1e-30), 0.0), 1e30)
                        q_n_c    = np.minimum(np.where(nz, np.maximum(q_n, 1e-30), 0.0), 1e30)
                        ref_e    = np.maximum(q_e_c.max(axis=0), 1e-30)[np.newaxis, :]
                        ref_n    = np.maximum(q_n_c.max(axis=0), 1e-30)[np.newaxis, :]
                        pow_e    = np.where(nz, np.power(np.maximum(q_e_c / ref_e, 1e-12), rho_bc), 0.0)
                        pow_n    = np.where(nz, np.power(np.maximum(q_n_c / ref_n, 1e-12), rho_bc), 0.0)
                        agg_e    = ref_e[0] * np.power(np.maximum((w * pow_e).sum(axis=0), 1e-300), inv_rho)
                        agg_n    = ref_n[0] * np.power(np.maximum((w * pow_n).sum(axis=0), 1e-300), inv_rho)
                        if any_cd:
                            cd_e  = np.exp((w * np.log(np.where(nz, np.maximum(q_e, 1e-12), 1.0))).sum(axis=0))
                            cd_n  = np.exp((w * np.log(np.where(nz, np.maximum(q_n, 1e-12), 1.0))).sum(axis=0))
                            agg_e = np.where(cd, cd_e, agg_e)
                            agg_n = np.where(cd, cd_n, agg_n)
                return np.where(has_grp, agg_e, np.inf), np.where(has_grp, agg_n, np.inf)

            e_rows = self._klems_e_idx
            m_rows = self._klems_m_idx
            E_j, E_j_ns = _ces(
                self._klems_e_nz, self._klems_A_e_safe, self.klems_w_E,
                self._klems_e_rho, self._klems_e_cd, self._klems_e_any_cd, self._klems_e_all_cd,
                self._klems_has_e, S_eff[e_rows, :], S[e_rows, :],
            )
            M_j, M_j_ns = _ces(
                self._klems_m_nz, self._klems_A_m_safe, self.klems_w_M,
                self._klems_m_rho, self._klems_m_cd, self._klems_m_any_cd, self._klems_m_all_cd,
                self._klems_has_m, S_eff[m_rows, :], S[m_rows, :],
            )

            # Two-level top nest, KL against E at sigma_kle and the bundle against M at sigma_klem.
            def _ces2(a, b, wa, wb, wa_pos, wb_pos, rho, cd, any_cd, all_cd):
                with np.errstate(divide="ignore", invalid="ignore",
                                 over="ignore"):
                    def _cd_form():
                        return np.exp(
                            np.where(wa_pos, wa * np.log(np.maximum(a, 1e-12)), 0.0)
                            + np.where(wb_pos, wb * np.log(np.maximum(b, 1e-12)), 0.0))
                    if all_cd:
                        return _cd_form()
                    _a = np.minimum(np.maximum(a, 1e-30), 1e30)
                    _b = np.minimum(np.maximum(b, 1e-30), 1e30)
                    _ref = np.maximum(
                        np.maximum(np.where(wa_pos, _a, 0.0),
                                   np.where(wb_pos, _b, 0.0)), 1e-30)
                    q_pow = (
                        np.where(wa_pos, wa * np.power(np.maximum(_a / _ref, 1e-12), rho), 0.0)
                        + np.where(wb_pos, wb * np.power(np.maximum(_b / _ref, 1e-12), rho), 0.0))
                    out = _ref * np.power(np.maximum(q_pow, 1e-300), 1.0 / rho)
                    if any_cd:
                        out = np.where(cd, _cd_form(), out)
                    return out

            def _kle(e, m):
                # E/M are inf iff has_e/has_m is False, so the masks zero absent-input terms.
                e_f = np.where(self._klems_has_e, e, 0.0)
                m_f = np.where(self._klems_has_m, m, 0.0)
                inner = _ces2(
                    kl_f, e_f, self._klems_w_KL_in, self._klems_w_E_in,
                    self._klems_w_KL_in_pos, self._klems_w_E_in_pos,
                    self._klems_kle_rho, self._klems_kle_cd,
                    self._klems_kle_any_cd, self._klems_kle_all_cd)
                return _ces2(
                    inner, m_f, self._klems_w_IN_top, self.klems_w_M_top,
                    self._klems_w_IN_pos, self._klems_w_M_tp_pos,
                    self._klems_klem_rho, self._klems_klem_cd,
                    self._klems_klem_any_cd, self._klems_klem_all_cd)

            KLE_j    = _kle(E_j,    M_j)
            KLE_j_ns = _kle(E_j_ns, M_j_ns)

            # ND Leontief
            nd_rows = self._klems_nd_idx
            if self._klems_nd_nz.shape[0] > 0:
                q_nd_eff = np.where(self._klems_nd_nz, S_eff[nd_rows, :] / self._klems_A_nd_safe, np.inf)
                q_nd_ns  = np.where(self._klems_nd_nz, S    [nd_rows, :] / self._klems_A_nd_safe, np.inf)
                ND_j    = np.where(self._klems_has_nd, q_nd_eff.min(axis=0), np.inf)
                ND_j_ns = np.where(self._klems_has_nd, q_nd_ns .min(axis=0), np.inf)
            else:
                ND_j = ND_j_ns = np.full(self.N, np.inf)

            xcap    = KLE_j
            xcap_ns = KLE_j_ns
            xinp    = ND_j
            xinp_ns = ND_j_ns

        else:
            raise ValueError(
                f"Unknown production function: {prod_f}. "
                "Supported: 'leontief', 'leontief.adapted', 'linear', 'ces', 'klems'"
            )

        output_constraint = self._period_output_constraints(t)
        # Only KLEMS capacity carries the supplement, so only it needs an own ns arm.
        if prod_f != "klems":
            xcap_ns = xcap
        xcap   = np.nan_to_num(xcap,   nan=0.0,          posinf=NUMERIC_LARGE, neginf=0.0)
        xcap_ns= np.nan_to_num(xcap_ns,nan=0.0,          posinf=NUMERIC_LARGE, neginf=0.0)
        xinp   = np.nan_to_num(xinp,   nan=NUMERIC_LARGE, posinf=NUMERIC_LARGE, neginf=0.0)
        xinp_ns= np.nan_to_num(xinp_ns,nan=NUMERIC_LARGE, posinf=NUMERIC_LARGE, neginf=0.0)
        d      = np.maximum(np.nan_to_num(d, nan=0.0, posinf=NUMERIC_LARGE, neginf=0.0), 0.0)
        output_constraint = np.nan_to_num(output_constraint, nan=np.inf, posinf=np.inf, neginf=0.0)
        capacity_constraint    = np.minimum(xcap,    output_constraint)
        capacity_constraint_ns = np.minimum(xcap_ns, output_constraint)
        x      = np.minimum(np.minimum(capacity_constraint,    xinp),    d)
        x_ns   = np.minimum(np.minimum(capacity_constraint_ns, xinp_ns), d)

        # Zero supplement where it didn't raise output: only charge when binding.
        if has_supplement:
            no_benefit = x <= x_ns + 1e-10
            import_supplement[:, no_benefit] = 0
            # Source revenue follows what actually flowed, prorated per row.
            final = import_supplement.sum(axis=1)
            for _i, (_placed, _alloc) in self._supp_alloc_.items():
                if _placed <= 0.0:
                    continue
                _scale = float(final[_i]) / _placed
                for _src, _amt in _alloc:
                    self._supp_by_source_[_src] = (
                        self._supp_by_source_.get(_src, 0.0) + _amt * _scale)

        x_constraints = np.column_stack([xcap, xinp, d, output_constraint])
        return {
            "output": x,
            "output.constraints": x_constraints,
            "import_supplement_matrix": import_supplement,
        }

    # Shock application methods
    def apply_output_constraint_shock(
        self,
        sector_label: str,
        time_period: int,
        reduction_pct: float,
        baseline_output: float = None,
    ) -> tuple[int, float]:
        """Cap the sector's output for one period at (1 - reduction_pct) times baseline_output."""
        if sector_label not in self.label_to_index:
            raise ValueError(
                f"Sector label '{sector_label}' not found. "
                f"Available labels: {list(self.label_to_index.keys())[:10]}..."
            )
        sector_idx = self.label_to_index[sector_label]
        if time_period < 0 or time_period >= self.TT:
            raise ValueError(f"Time period {time_period} out of range [0, {self.TT-1}]")
        if reduction_pct < 0 or reduction_pct >= 1:
            raise ValueError(f"Reduction percentage must be in [0, 1). Got {reduction_pct}")
        if baseline_output is None:
            baseline_output = self.x0[sector_idx]
        constraint_level = (1 - reduction_pct) * baseline_output
        self.output_constraint_[sector_idx, time_period] = constraint_level
        self.delta_[sector_idx, time_period] = reduction_pct  # activates disruption_active in hire_fire
        return sector_idx, constraint_level

    def apply_consumption_shock(
        self, start: int, duration: int, intensity: float, region: int | None = None
    ) -> None:
        """Set epsilon_r[r, t] = intensity for t in [start, start+duration). epsilon_ stays a view
        into epsilon_r[0]."""
        regions = range(self.n_regions) if region is None else [region]
        for r in regions:
            for t in range(start, min(start + duration, self.TT)):
                self.epsilon_r[r, t] = intensity

    def apply_input_availability_shock(
        self, input_sector_label: str, time_period: int, reduction_pct: float
    ) -> tuple[int, float]:
        """Record a supplier-side input-availability shock capping the supplier's output for the
        period, so downstream shortages arise via reduced deliveries and inventory drawdown."""
        if input_sector_label not in self.label_to_index:
            raise ValueError(
                f"Sector label '{input_sector_label}' not found. "
                f"Available labels: {list(self.label_to_index.keys())[:10]}..."
            )
        input_sector_idx = self.label_to_index[input_sector_label]
        if time_period < 0 or time_period >= self.TT:
            raise ValueError(f"Time period {time_period} out of range [0, {self.TT-1}]")
        if reduction_pct < 0 or reduction_pct >= 1:
            raise ValueError(f"Reduction percentage must be in [0, 1). Got {reduction_pct}")
        if time_period not in self.input_availability_shocks_:
            self.input_availability_shocks_[time_period] = {}
        self.input_availability_shocks_[time_period][input_sector_idx] = reduction_pct
        return input_sector_idx, reduction_pct

    def _build_price_inverses(self, A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Cost-push Leontief inverses (I - A*passthrough)^-1 for upward and downward pass-through,
        where scaling A tunes the retained network amplification."""
        _I = np.eye(self.N)
        try:
            L_pos = np.linalg.inv(_I - A * self.price_passthrough_pos)
            L_neg = np.linalg.inv(_I - A * self.price_passthrough_neg)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "Price pass-through requires (I - A*passthrough) to be invertible. "
                "The input-output matrix appears non-productive."
            ) from exc
        return L_pos, L_neg

    def apply_price_cost_shock(
        self, sector_label: str, time_period: int, delta_cost: float,
        duration: int | None = None,
    ) -> tuple[int, float]:
        """Record a direct unit-cost shock (a fraction of base unit cost, > -1) active for duration
        periods from time_period, duration=None being a permanent level shift."""
        if sector_label not in self.label_to_index:
            raise ValueError(
                f"Sector label '{sector_label}' not found. "
                f"Available labels: {list(self.label_to_index.keys())[:10]}..."
            )
        sector_idx = self.label_to_index[sector_label]
        if time_period < 0 or time_period >= self.TT:
            raise ValueError(f"Time period {time_period} out of range [0, {self.TT-1}]")
        if delta_cost <= -1.0:
            raise ValueError(f"delta_cost must exceed -1; got {delta_cost}")
        if duration is not None and (duration != int(duration) or duration < 1):
            raise ValueError(f"duration must be a positive integer when given; got {duration}")
        end = self.TT if duration is None else min(time_period + int(duration), self.TT)
        # Overwrite any existing shock on this sector and start, matching the other shock setters.
        self.price_cost_shocks_ = [
            _s for _s in self.price_cost_shocks_
            if not (_s["sector"] == sector_idx and _s["start"] == time_period)
        ]
        self.price_cost_shocks_.append({
            "sector": sector_idx, "delta": float(delta_cost),
            "start": time_period, "end": end,
        })
        return sector_idx, float(delta_cost)

    def apply_rationing_shock(
        self,
        supplier_sector_label: str,
        time_period: int,
        capacity_pct: float,
        include_households: bool = True,
    ) -> tuple[int, float]:
        """Cap the supplier's output at capacity_pct of baseline with proportional deliveries."""
        if supplier_sector_label not in self.label_to_index:
            raise ValueError(
                f"Sector label '{supplier_sector_label}' not found. "
                f"Available labels: {list(self.label_to_index.keys())[:10]}..."
            )
        supplier_idx = self.label_to_index[supplier_sector_label]
        if time_period < 0 or time_period >= self.TT:
            raise ValueError(f"Time period {time_period} out of range [0, {self.TT-1}]")
        if capacity_pct <= 0 or capacity_pct > 1:
            raise ValueError(f"Capacity percentage must be in (0, 1]. Got {capacity_pct}")
        if time_period not in self.rationing_shocks_:
            self.rationing_shocks_[time_period] = {}
        self.rationing_shocks_[time_period][supplier_idx] = {
            "capacity_pct": capacity_pct,
            "include_households": include_households,
        }
        self.apply_output_constraint_shock(supplier_sector_label, time_period, 1 - capacity_pct)
        return supplier_idx, capacity_pct

    def apply_bilateral_delivery_shock(
        self,
        supplier_sector_label: str,
        buyer_region: int | str,
        time_period: int,
        capacity_pct: float,
        household_capacity_pct: float | None = None,
    ) -> tuple[int, int, float, float]:
        """Cap deliveries from one supplier sector to one region's buyers at capacity_pct of the
        baseline bilateral flow, binding intermediate orders and household demand."""
        if supplier_sector_label not in self.label_to_index:
            raise ValueError(
                f"Sector label '{supplier_sector_label}' not found. "
                f"Available labels: {list(self.label_to_index.keys())[:10]}..."
            )
        supplier_idx = self.label_to_index[supplier_sector_label]
        if isinstance(buyer_region, str):
            if buyer_region not in self.region_labels:
                raise ValueError(
                    f"Region label '{buyer_region}' not found. "
                    f"Available labels: {list(self.region_labels)}"
                )
            buyer_idx = int(self.region_labels.index(buyer_region))
        else:
            buyer_idx = int(buyer_region)
            if buyer_idx < 0 or buyer_idx >= self.n_regions:
                raise ValueError(
                    f"Buyer region {buyer_idx} out of range [0, {self.n_regions - 1}]"
                )
        if time_period < 0 or time_period >= self.TT:
            raise ValueError(f"Time period {time_period} out of range [0, {self.TT-1}]")
        hh_pct = capacity_pct if household_capacity_pct is None else household_capacity_pct
        for _name, _value in (("Capacity", capacity_pct), ("Household capacity", hh_pct)):
            if _value < 0 or _value > 1:
                raise ValueError(f"{_name} percentage must be in [0, 1]. Got {_value}")
        if time_period not in self.bilateral_delivery_shocks_:
            self.bilateral_delivery_shocks_[time_period] = {}
        self.bilateral_delivery_shocks_[time_period][(supplier_idx, buyer_idx)] = (
            float(capacity_pct), float(hh_pct)
        )
        return supplier_idx, buyer_idx, float(capacity_pct), float(hh_pct)

    def _cap_bilateral_orders(self, O: np.ndarray, t: int) -> None:
        """Scale shocked rows into each buyer region's columns down to the capped level."""
        for (supplier_idx, r), (pct, _) in self.bilateral_delivery_shocks_.get(t, {}).items():
            J_r     = self.region_sector_indices[r]
            level   = pct * float(self.Z0[supplier_idx, J_r].sum())
            desired = float(O[supplier_idx, J_r].sum())
            if desired > level:
                O[supplier_idx, J_r] *= level / desired

    def _cap_bilateral_consumption(self, cd_r: np.ndarray, r: int, t: int) -> np.ndarray:
        """Cap region r's demand on shocked rows, recording the shortfall for the supplement."""
        for (supplier_idx, buyer_idx), (_, pct) in self.bilateral_delivery_shocks_.get(t, {}).items():
            if buyer_idx == r:
                capped = pct * float(self.cons_vec_r[supplier_idx, r])
                short = float(cd_r[supplier_idx]) - capped
                if short > 0.0:
                    self._hh_shortfall_[(supplier_idx, r)] = (
                        self._hh_shortfall_.get((supplier_idx, r), 0.0) + short
                    )
                cd_r[supplier_idx] = min(cd_r[supplier_idx], capped)
        return cd_r

    def apply_import_supplement_cap(
        self,
        supplier_sector_label: str,
        time_period: int,
        cap_level: float,
    ) -> tuple[int, float]:
        """Cap the import supplement from one RoW sector in a period at cap_level (base-table units
        per period), scaling entries down when binding and leaving unregistered pairs unbounded."""
        if supplier_sector_label not in self.label_to_index:
            raise ValueError(
                f"Sector label '{supplier_sector_label}' not found. "
                f"Available labels: {list(self.label_to_index.keys())[:10]}..."
            )
        row_idx = self.label_to_index[supplier_sector_label]
        if self.row_region_idx is None:
            raise ValueError("An import supplement cap requires a RoW region.")

        if time_period < 0 or time_period >= self.TT:
            raise ValueError(f"Time period {time_period} out of range [0, {self.TT-1}]")
        if not np.isfinite(cap_level) or cap_level < 0:
            raise ValueError(f"Cap level must be finite and non-negative. Got {cap_level}")
        self.import_supplement_caps_.setdefault(time_period, {})[row_idx] = float(cap_level)
        return row_idx, float(cap_level)

    def set_initial_inventory_fill(self, input_sector_label: str,
                                   buyer_sector_label: str,
                                   fraction: float) -> None:
        """Start the buyer's stock of one input at fraction x target."""
        for lbl in (input_sector_label, buyer_sector_label):
            if lbl not in self.label_to_index:
                raise ValueError(f"Sector label {lbl!r} not found.")
        if not 0.0 <= float(fraction) <= 1.0:
            raise ValueError(f"fraction must be in [0, 1]; got {fraction}")
        self.initial_inventory_fill_[
            (self.label_to_index[input_sector_label],
             self.label_to_index[buyer_sector_label])
        ] = float(fraction)

    def set_inventory_target_days(self, input_sector_label: str,
                                  buyer_sector_label: str,
                                  days: float) -> None:
        """Override the buyer's inventory-target days for one input."""
        for lbl in (input_sector_label, buyer_sector_label):
            if lbl not in self.label_to_index:
                raise ValueError(f"Sector label {lbl!r} not found.")
        if not float(days) > 0.0:
            raise ValueError(f"days must be positive; got {days}")
        self.inventory_days_pair_[
            (self.label_to_index[input_sector_label],
             self.label_to_index[buyer_sector_label])
        ] = float(days)

    def set_essential_inputs(self, matrix: np.ndarray) -> None:
        """Override the leontief.adapted essential-input classification."""
        m = np.asarray(matrix)
        if m.shape != (self.N, self.N):
            raise ValueError(f"essential matrix must be (N, N); got {m.shape}")
        if not (m > 0).any():
            raise ValueError("no input classified as essential")
        self.A_essential = (m > 0).astype(int)
        self._essential_override_ = True

    def set_supplement_sources(self, input_sector_label: str,
                               source_sector_labels: list[str]) -> None:
        """Register an ordered source list for an input row's supplement, replacing the default RoW
        source, where earlier sources fill first within their registered capacities."""
        for lbl in [input_sector_label] + list(source_sector_labels):
            if lbl not in self.label_to_index:
                raise ValueError(f"Sector label {lbl!r} not found.")
        self.supplement_sources_[self.label_to_index[input_sector_label]] = tuple(
            self.label_to_index[lbl] for lbl in source_sector_labels
        )

    def _row_sources(self, input_idx: int) -> tuple[int, ...]:
        if input_idx in self.supplement_sources_:
            return self.supplement_sources_[input_idx]
        src = int(self.good_to_row_idx[input_idx])
        return (src,) if src >= 0 else ()

    def _cap_import_supplement(self, import_supplement: np.ndarray, t: int) -> None:
        """Allocate each input row's supplement across its ordered sources within their capacities,
        scaling down when binding and recording per-source amounts for the revenue credit."""
        self._supp_by_source_ = {}
        self._supp_alloc_ = {}
        caps = self.import_supplement_caps_.get(t)
        per_input = import_supplement.sum(axis=1)
        used: dict[int, float] = {}
        for i in np.nonzero(per_input > 0)[0]:
            sources = self._row_sources(int(i))
            # A row without a source mapping is neither capped nor credited.
            if not sources:
                continue
            need = float(per_input[i])
            placed = 0.0
            alloc: list = []
            for src in sources:
                room = need - placed
                if caps is not None and src in caps:
                    room = min(room, max(0.0, caps[src] - used.get(src, 0.0)))
                if room <= 0.0:
                    continue
                used[src] = used.get(src, 0.0) + room
                alloc.append((src, room))
                placed += room
                if placed >= need:
                    break
            if placed < need:
                import_supplement[i, :] *= placed / need if need > 0 else 0.0
            self._supp_alloc_[int(i)] = (placed, alloc)

    def apply_technical_change(self, t: int, new_A: np.ndarray) -> None:
        """Replace A by new_A ((N, N), column sums < 1) from period t onwards. Distinct periods are
        cumulative, the same t overwrites, and calls must precede run_model."""
        if new_A.shape != self.A.shape:
            raise ValueError(f"new_A shape {new_A.shape} does not match A shape {self.A.shape}")
        if not (new_A >= 0).all():
            raise ValueError("new_A must be non-negative throughout")
        col_sums = new_A.sum(axis=0)
        if not (col_sums < 1 + 1e-9).all():
            bad = np.where(col_sums >= 1 + 1e-9)[0]
            raise ValueError(
                f"new_A column viability violated: {len(bad)} column(s) sum >= 1 "
                f"(first offender: col {bad[0]}, sum={col_sums[bad[0]]:.6f})"
            )
        self.A_changes[t] = new_A

    def apply_factor_productivity_shock(
        self,
        sector_label: str,
        t: int,
        prod_L: float = 1.0,
        prod_K: float = 1.0,
    ) -> int:
        """Register a factor productivity change from period t onwards, prod_L/prod_K > 1 being more
        efficient. Calls cumulate, the same pair overwrites, and all must precede run_model."""
        if sector_label not in self.label_to_index:
            raise ValueError(
                f"Sector label '{sector_label}' not found. "
                f"Available labels: {list(self.label_to_index.keys())[:10]}..."
            )
        if t < 0 or t >= self.TT:
            raise ValueError(f"Time period {t} out of range [0, {self.TT - 1}]")
        if prod_L <= 0:
            raise ValueError(f"prod_L must be positive; got {prod_L}")
        if prod_K <= 0:
            raise ValueError(f"prod_K must be positive; got {prod_K}")
        sector_idx = self.label_to_index[sector_label]
        if t not in self.fprod_changes:
            self.fprod_changes[t] = {}
        self.fprod_changes[t][sector_idx] = (float(prod_L), float(prod_K))
        return sector_idx

    # Delivery allocation, inventory, profits
    def allocate_deliveries(
        self,
        x: np.ndarray,
        O: np.ndarray,
        cd: np.ndarray,
        fd: dict[str, np.ndarray],
        d: np.ndarray,
        rule: str,
        t: int = None,
    ) -> dict:
        """Split feasible output x into Z, consumption, and realised fd, given the demand d."""
        if rule not in ALLOCATION_RULES:
            raise ValueError(f"rule must be one of {tuple(ALLOCATION_RULES)}; got {rule!r}")
        int_share, fin_share = ALLOCATION_RULES[rule]
        Z = O * int_share(x, O, d)[:, np.newaxis]

        shocks = self.rationing_shocks_.get(t, {})
        for supplier_idx, info in shocks.items():
            Z[supplier_idx, :] = O[supplier_idx, :] * info["capacity_pct"]

        s = fin_share(x, d, Z)
        c = cd * s
        for supplier_idx, info in shocks.items():
            if info["include_households"]:
                c[supplier_idx] = (
                    cd[supplier_idx] * info["capacity_pct"]
                    if s[supplier_idx] > 0
                    else 0
                )

        out = {"Z": Z, "consumption": c}
        for name, comp in fd.items():
            out[name] = comp * s

        # Orders can run above baseline, so the override must not promise more than x.
        for supplier_idx in shocks:
            claimed = (
                float(Z[supplier_idx, :].sum())
                + float(c[supplier_idx])
                + sum(float(out[name][supplier_idx]) for name in fd)
            )
            if claimed > x[supplier_idx]:
                scale = x[supplier_idx] / claimed
                Z[supplier_idx, :] *= scale
                c[supplier_idx]    *= scale
                for name in fd:
                    out[name][supplier_idx] *= scale
        return out

    def inventory_S(
        self,
        x: np.ndarray,
        S: np.ndarray,
        Z: np.ndarray,
        A: np.ndarray,
    ) -> np.ndarray:
        """Update end-of-period inventories: S + Z - A @ diag(x), floored at 0."""
        x = np.nan_to_num(x, nan=0.0, posinf=NUMERIC_LARGE, neginf=0.0)
        S = np.nan_to_num(S, nan=0.0, posinf=NUMERIC_LARGE, neginf=0.0)
        Z = np.nan_to_num(Z, nan=0.0, posinf=NUMERIC_LARGE, neginf=0.0)
        return np.maximum(0, S + Z - A * x[np.newaxis, :])

    def profit_pi(
        self,
        x: np.ndarray,
        Z: np.ndarray,
        l: np.ndarray,
        fprod_k: np.ndarray | None = None,
    ) -> np.ndarray:
        """Profits as output minus intermediate inputs, labour, and cap/tax/import shares."""
        fprod_k_eff = self.fprod_k_ if fprod_k is None else fprod_k
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            eff_cap_share = np.divide(
                self.cap_share, fprod_k_eff,
                out=np.zeros_like(self.cap_share),
                where=fprod_k_eff != 0,
            )
        return (
            x
            - np.sum(Z, axis=0)
            - l
            - eff_cap_share * x
            - self.tax_share * x
            - self.imp_share * x
        )

    def savings_s_regional(
        self,
        household_income_r_t: np.ndarray,
        c_r_t: np.ndarray,
    ) -> np.ndarray:
        """Return (R,) per-region household savings for one period."""
        result = np.zeros(self.n_regions)
        for r in range(self.n_regions):
            extra = self._extra_household_expenditure(
                float(c_r_t[r]), c_other_coef=float(self.c_other_coef_r[r])
            )
            result[r] = float(household_income_r_t[r]) - float(c_r_t[r]) - extra
        return result

    # Post-run validation
    def _validate_run(self, result: dict) -> None:
        """Post-run accounting checks (called when validate=True in run_model)."""
        gdp       = result["gdp"]
        gdp_r     = result["gdp_regional"]
        c_        = result["realised_consumption"]
        c_r       = result["consumption_by_hh_region"]
        savings   = result["savings"]
        savings_r = result["savings_regional"]

        # 1. Regional GDP additivity
        diff_gdp = np.abs(gdp_r.sum(axis=0) - gdp)
        assert np.all(diff_gdp < 1e-6), (
            f"Regional GDP additivity violated: max |sum_r gdp_r - gdp| = {diff_gdp.max():.2e}"
        )

        # 2. Regional consumption additivity
        diff_cons = np.abs(c_r.sum(axis=0) - np.sum(c_, axis=0))
        assert np.all(diff_cons < 1e-6), (
            f"Regional consumption additivity violated: max = {diff_cons.max():.2e}"
        )

        # 3. Savings additivity
        diff_sav = np.abs(savings_r.sum(axis=0) - savings)
        assert np.all(diff_sav < 1e-6), (
            f"Savings additivity violated: max = {diff_sav.max():.2e}"
        )

        # 4. Frozen-region consumption invariance
        for r in range(self.n_regions):
            if self.household_closure_mode_r[r] == "frozen":
                expected = self.base_consumption_total_r[r]
                max_dev  = np.max(np.abs(c_r[r, :] - expected))
                assert max_dev < expected * 0.01 + 1.0, (
                    f"Frozen region {r} consumption drifted: max dev = {max_dev:.4f} "
                    f"from base {expected:.4f}"
                )

        # 5. Trade balance symmetry
        tb_sum = np.abs(result["trade_balance"].sum(axis=0))
        total_flow = np.abs(result["trade_balance"]).sum(axis=0) + 1e-9
        assert np.all(tb_sum / total_flow < 1e-6), (
            f"Trade balance not symmetric: max |sum_r TB_r| / total = {(tb_sum / total_flow).max():.2e}"
        )

        # 6. Expenditure-production GDP identity (diagnostic, rtol=1e-2)
        gov_r_base = np.array([
            self.gov_vec_r[self.region_sector_indices[r], r].sum()
            for r in range(self.n_regions)
        ])
        inv_r_base = np.array([
            (self.inv_vec_r[self.region_sector_indices[r], r]
             + self.invnt_vec_r[self.region_sector_indices[r], r]).sum()
            for r in range(self.n_regions)
        ])
        for t in range(self.TT):
            gdp_exp = (
                result["consumption_by_hh_region"][:, t]
                + gov_r_base
                + inv_r_base
                + result["trade_balance"][:, t]
            )
            max_dev = float(np.abs(gdp_exp - gdp_r[:, t]).max())
            if max_dev > max(np.abs(gdp_r[:, t]).max() * 1e-2, 1e3):
                logger.debug(
                    "Expenditure-production GDP identity: max deviation %.2f at t=%d "
                    "(note: trade_balance covers intermediate trade only)", max_dev, t
                )

        logger.info("run_model validation passed (R=%d, TT=%d)", self.n_regions, self.TT)

    # Simulation loop
    def run_model(
        self,
        store_full_matrices: bool = False,
        validate: bool = False,
        step_callback=None,
        store_delivery_rows=None,
    ) -> dict:
        """Simulate all TT periods and return a results dict. store_delivery_rows keeps per-row
        delivery stacks alone, and step_callback(t, gdp_r_t, gdp_r_0) may raise ConvergenceAbort."""
        for _name, _sched in (("baseline_trend_r", self.baseline_trend_r),
                              ("xi_r", self.xi_r)):
            if _sched is not None and (
                not np.all(np.isfinite(_sched)) or np.any(np.asarray(_sched) <= 0)
            ):
                raise ValueError(f"{_name} must be finite and positive throughout.")
        s    = np.zeros(self.TT)
        pi_  = np.zeros((self.N, self.TT)); pi_[:, 0] = self.profits0
        d_   = np.zeros((self.N, self.TT)); d_[:, 0]  = self.x0
        x_   = np.zeros((self.N, self.TT)); x_[:, 0]  = self.x0
        cd_  = np.zeros((self.N, self.TT)); cd_[:, 0] = self.c0
        c_   = np.zeros((self.N, self.TT)); c_[:, 0]  = self.c0
        l_   = np.zeros((self.N, self.TT)); l_[:, 0]  = self.l0
        # Price index per sector, 1.0 at t=0 numeraire, stays flat when disabled.
        p_   = np.ones((self.N, self.TT))

        R = self.n_regions
        Cdt_r                  = np.zeros((R, self.TT))
        household_income_r     = np.zeros((R, self.TT))
        household_income_sig_r = np.zeros((R, self.TT))
        c_r                    = np.zeros((R, self.TT))
        savings_r              = np.zeros((R, self.TT))
        gdp_r_series           = np.zeros((R, self.TT))
        c_prod_regional        = np.zeros((R, self.TT))
        trade_balance_r        = np.zeros((R, self.TT))

        # t=0 initialisation via reduceat over region permutation.
        _l0_r       = np.add.reduceat(self.l0[self._reg_perm],       self._reg_starts)
        _profits0_r = np.add.reduceat(self.profits0[self._reg_perm], self._reg_starts)
        _va0        = self.x0 - self.Z0.sum(axis=0) - self.imp_share * self.x0
        gdp_r_series[:, 0]    = np.add.reduceat(_va0[self._reg_perm],       self._reg_starts)
        c_prod_regional[:, 0] = np.add.reduceat(self.c0[self._reg_perm],    self._reg_starts)

        for r in range(R):
            Cdt_r[r, 0]                 = self.base_consumption_total_r[r]
            household_income_r[r, 0]    = self._household_income_r(
                r, float(_l0_r[r]), float(_profits0_r[r])
            )
            household_income_sig_r[r, 0] = household_income_r[r, 0]
            c_r[r, 0]                   = self.base_consumption_total_r[r]

        savings_r[:, 0] = self.savings_s_regional(household_income_r[:, 0], c_r[:, 0])
        s[0] = float(np.sum(savings_r[:, 0]))

        # Keynesian: carry EMA and partial-adjustment state; both at base-year steady state at t=0.
        _inv_scale_prev = 1.0
        _savings_ema    = self.base_savings_total  # EMA of raw savings (level)

        # Trade balance at t=0: reduceat Z0 into (R,R) flows; diagonal (intra) flows cancel.
        _Z0_perm = self.Z0[np.ix_(self._reg_perm, self._reg_perm)]
        _exp_t0  = np.add.reduceat(
            np.add.reduceat(_Z0_perm, self._reg_starts, axis=0),
            self._reg_starts, axis=1,
        )
        trade_balance_r[:, 0] = _exp_t0.sum(axis=1) - _exp_t0.sum(axis=0)

        Cdt                  = Cdt_r[0]
        household_income_    = household_income_r[0]
        household_income_signal_ = household_income_sig_r[0]

        A_current           = self.A
        # prod_function may have been switched after construction (sweeps, Monte Carlo).
        if self.prod_function == "leontief.adapted" and self.A_essential is None:
            self.A_essential = self._estimate_essential_inputs(self.A)
        A_essential_current = self.A_essential
        S_tar = A_current * self.x0[np.newaxis, :] * self.n[np.newaxis, :]
        for (_i, _j), _d in self.inventory_days_pair_.items():
            S_tar[_i, _j] = A_current[_i, _j] * self.x0[_j] * _d
        S_prev = S_tar.copy()
        for (_i, _j), _f in self.initial_inventory_fill_.items():
            S_prev[_i, _j] *= _f

        # Local copies so run_model() is safe to call multiple times.
        fprod_l_cur = self.fprod_l_.copy()
        fprod_k_cur = self.fprod_k_.copy()

        # Labour supply reference: (N, TT) from schedule, or tiled l0.
        if self.labour_supply_schedule is not None:
            if self.labour_supply_schedule.shape != (self.N, self.TT):
                raise ValueError(
                    f"labour_supply_schedule shape {self.labour_supply_schedule.shape} "
                    f"must be (N={self.N}, TT={self.TT})."
                )
            l_ref = self.labour_supply_schedule
        else:
            l_ref = np.tile(self.l0[:, np.newaxis], (1, self.TT))

        _pd_r = np.atleast_1d(self.price_deflate_household_income_r)
        if _pd_r.size == 1:
            _pd_r = np.full(self.n_regions, bool(_pd_r[0]))
        elif _pd_r.size != self.n_regions:
            raise ValueError(
                f"price_deflate_household_income has length {_pd_r.size}; "
                f"expected 1 or n_regions={self.n_regions}"
            )

        # Unemployment schedule: validated once here; sliced per period inside the loop.
        u_sched = None
        if self.unemployment_schedule is not None:
            if self.unemployment_schedule.shape != (self.N, self.TT):
                raise ValueError(
                    f"unemployment_schedule shape {self.unemployment_schedule.shape} "
                    f"must be (N={self.N}, TT={self.TT})."
                )
            u_sched = self.unemployment_schedule

        Z_colsum_  = np.zeros((self.N, self.TT))
        Z_colsum_[:, 0] = np.sum(self.Z0, axis=0)
        realised_government_  = np.zeros((self.N, self.TT))
        realised_government_[:, 0]  = self.fd_government_[:, 0]
        realised_investment_  = np.zeros((self.N, self.TT))
        realised_investment_[:, 0]  = self.fd_investment_[:, 0]
        realised_inventories_ = np.zeros((self.N, self.TT))
        realised_inventories_[:, 0] = self.fd_inventories_[:, 0]
        realised_exports_     = np.zeros((self.N, self.TT))
        realised_exports_[:, 0]     = self.fd_exports_[:, 0]
        realised_other_       = np.zeros((self.N, self.TT))
        realised_other_[:, 0]       = self.fd_other_[:, 0]
        desired_l_ = np.zeros((self.N, self.TT))
        desired_l_[:, 0] = self.l0   # at t=0 desired = actual (no gap by construction)

        # Per-period (N, TT) trackers; zero unless the corresponding config field opts in.
        import_supplement_          = np.zeros((self.N, self.TT))  # using-side cost
        import_supplement_by_input_ = np.zeros((self.N, self.TT))  # per-input supplement
        row_export_supplement_      = np.zeros((self.N, self.TT))  # RoW revenue
        hh_supplement_              = np.zeros((self.N, self.TT))  # by source row
        supp_by_source_             = np.zeros((self.N, self.TT))  # intermediates
        export_pull_supplement_     = np.zeros((self.N, self.TT))  # extra exports
        gdp_series    = np.zeros(self.TT)
        gdp_series[0] = float(np.sum(gdp_r_series[:, 0]))

        if store_full_matrices:
            S_list = [S_prev.copy()]
            O_list = [self.Z0.copy()]
            Z_list = [self.Z0.copy()]
            I_list = [np.zeros_like(self.Z0)]
        if store_delivery_rows is not None:
            _rows = np.asarray(store_delivery_rows, dtype=np.intp)
            Zr_list = [self.Z0[_rows, :].copy()]
            Ir_list = [np.zeros((_rows.size, self.N))]
            Sr_list = [S_prev[_rows, :].copy()]

        initial_prod = self.producing_x(
            self.prod_function, A_essential_current, self.x0,
            l_, S_prev, A_current, d_[:, 0], 0,
            fprod_l=fprod_l_cur, fprod_k=fprod_k_cur, S_tar=S_tar,
        )
        # (TT, N, 4) output constraint stack; TT-leading for contiguous per-period slices.
        x_constraints = np.zeros((self.TT, self.N, 4), dtype=np.float64)
        x_constraints[0] = initial_prod["output.constraints"]

        for t in range(1, self.TT):
            # Technical change
            if t in self.A_changes:
                A_current = self.A_changes[t]
                S_tar     = A_current * self.x0[np.newaxis, :] * self.n[np.newaxis, :]
                if self.prod_function != "leontief.adapted":
                    A_essential_current = None
                elif getattr(self, "_essential_override_", False):
                    # An explicit classification is sector knowledge and survives technical change.
                    A_essential_current = self.A_essential
                else:
                    A_essential_current = self._estimate_essential_inputs(A_current)
                # Rebuild price inverses so pass-through tracks the new structure.
                if self.price_passthrough_enabled:
                    self.L_price_pos, self.L_price_neg = self._build_price_inverses(A_current)

            if t in self.fprod_changes:
                for _j, (_pL, _pK) in self.fprod_changes[t].items():
                    fprod_l_cur[_j] = _pL
                    fprod_k_cur[_j] = _pK

            # Labour adjustment
            l_[:, t] = self.hire_fire(
                t, l_, x_, self.delta_, x_constraints[t - 1],
                l_ref_t=l_ref[:, t], fprod_l_t=fprod_l_cur,
                U_r_prev=u_sched[:, t - 1] if u_sched is not None else None,
                U_r_0=u_sched[:, 0]        if u_sched is not None else None,
                desired_l_out=desired_l_[:, t],
            )

            # Consumption demand (per-region loop)
            cd_[:, t] = 0.0
            for r in range(R):
                _xit = float(self.xi_r[r, t]) if self.xi_r is not None else self.xi_[t]
                household_income_sig_r[r, t] = self._household_income_signal_for_period_r(
                    r, household_income_r[r, t - 1], _xit
                )
                if self.price_passthrough_enabled and _pd_r[r]:
                    # Lagged to avoid circularity, weight-sum division giving a base index of 1.0.
                    _w = self.theta_r[r][:, t - 1]
                    P_r_prev = float(_w @ p_[:, t - 1]) / max(float(_w.sum()), 1e-12)
                    household_income_sig_r[r, t] /= max(P_r_prev, 1e-9)
                Cdt_new, cd_new = self._findemand_cd_regional(
                    r, t, Cdt_r[r, t - 1], household_income_sig_r[r, t]
                )
                Cdt_r[r, t]  = Cdt_new
                cd_[:, t]   += self._cap_bilateral_consumption(cd_new, r, t)

            # Government spending (endogenous when gov_income_elasticity != 0)
            if self.gov_income_elasticity != 0.0:
                hh_income_agg = float(np.sum(household_income_r[:, t - 1]))
                gov_scale = 1.0 + self.gov_income_elasticity * (
                    hh_income_agg / self.base_gov_income_total - 1.0
                )
                # Additive: elasticity component + exogenous fd_government_ deviation.
                gov_fd_t = (
                    self.gov_agg_base * gov_scale
                    + (self.fd_government_[:, t] - self.gov_agg_base)
                )
            else:
                gov_fd_t = self.fd_government_[:, t]

            # Investment demand (Keynesian savings-to-investment closure)
            if self.investment_closure == "keynesian":
                # EMA-smooth savings (w=1 = no smoothing), partial-adjust toward target.
                w = self.investment_savings_ema
                raw_savings = float(np.sum(savings_r[:, t - 1]))
                _savings_ema = w * raw_savings + (1.0 - w) * _savings_ema
                target_scale = max(_savings_ema / self.base_savings_total, 0.0)
                alpha = self.investment_adj_speed
                inv_scale = _inv_scale_prev + alpha * (target_scale - _inv_scale_prev)
                inv_scale = max(inv_scale, 0.0)
                _cap = self.investment_scale_growth_cap
                if _cap is not None and _inv_scale_prev > 0:
                    inv_scale = min(inv_scale, _inv_scale_prev * (1.0 + _cap))
                _inv_scale_prev = inv_scale
                # Additive: scale component + exogenous fd_investment_ deviation.
                inv_fd_t = (
                    self.inv_agg_base * inv_scale
                    + (self.fd_investment_[:, t] - self.inv_agg_base)
                )
            else:
                inv_fd_t = self.fd_investment_[:, t]

            O_curr   = self.orders_O(A_current, d_[:, t - 1], self.tau, S_tar, S_prev)
            self._cap_bilateral_orders(O_curr, t)
            d_[:, t] = (
                cd_[:, t]
                + np.sum(O_curr, axis=1)
                + gov_fd_t
                + inv_fd_t
                + self.fd_inventories_[:, t]
                + self.fd_exports_[:, t]
                + self.fd_other_[:, t]
            )
            # Export pull on lagged slack capacity clipped to x0, l0==0 sectors using x0 as proxy.
            if np.any(self.export_pull_vector > 0) and len(self.J_named) > 0:
                _xc_prev = x_constraints[t - 1]   # columns: xcap, xinp, d, output_constraint
                _xcap_eff = np.where(self.l0 == 0, self.x0, _xc_prev[:, 0])
                _cap_prev = np.minimum(_xcap_eff, _xc_prev[:, 3])
                _slack = np.minimum(
                    np.maximum(_cap_prev - x_[:, t - 1], 0.0),
                    self.x0,
                )
                _extra_export = self.export_pull_vector * _slack
                _mask = np.zeros(self.N, dtype=bool)
                _mask[self.J_named] = True
                _extra_export = np.where(_mask, _extra_export, 0.0)
                _extra_export = np.nan_to_num(_extra_export, nan=0.0, posinf=0.0, neginf=0.0)
                export_pull_supplement_[:, t] = _extra_export
                d_[:, t] = d_[:, t] + _extra_export
            prod       = self.producing_x(
                self.prod_function, A_essential_current, self.x0,
                l_, S_prev, A_current, d_[:, t].copy(), t,
                fprod_l=fprod_l_cur, fprod_k=fprod_k_cur, S_tar=S_tar,
            )
            x_[:, t]                 = prod["output"]
            x_constraints[t]         = prod["output.constraints"]
            imp_supp_mat             = prod["import_supplement_matrix"]
            import_supplement_[:, t]          = imp_supp_mat.sum(axis=0)
            import_supplement_by_input_[:, t] = imp_supp_mat.sum(axis=1)

            # Cost-push price level from active unit-cost shocks, inverses split for asymmetry.
            if self.price_passthrough_enabled:
                v = np.zeros(self.N)
                for _sh in self.price_cost_shocks_:
                    if _sh["start"] <= t < _sh["end"]:
                        v[_sh["sector"]] += _sh["delta"]
                v_pos = np.where(v > 0.0, v, 0.0)
                v_neg = np.where(v < 0.0, v, 0.0)
                # Stacked negative shocks can push the index to zero or below, so floor it.
                p_[:, t] = np.maximum(
                    1.0 + self.L_price_pos.T @ v_pos + self.L_price_neg.T @ v_neg, 1e-6
                )

            # Source supplement from RoW's matching sector, capped at row_supply_cap * x.
            if (
                self.row_region_idx is not None
                and np.any(self.row_supply_cap_vector > 0)
                and np.any(imp_supp_mat > 0)
            ):
                row_credit = np.zeros(self.N)
                for _src, _amount in self._supp_by_source_.items():
                    row_credit[_src] += _amount
                _capped_srcs = list(self.import_supplement_caps_.get(t, {}))
                cap_per_row = self.row_supply_cap_vector * np.maximum(x_[:, t], 0.0)
                cap_per_row[_capped_srcs] = np.inf
                row_export_supplement_[:, t] = np.minimum(row_credit, cap_per_row)

            fd_t = {
                "government":  gov_fd_t,
                "investment":  inv_fd_t,
                "inventories": self.fd_inventories_[:, t],
                "exports":     self.fd_exports_[:, t] + export_pull_supplement_[:, t],
                "other":       self.fd_other_[:, t],
            }
            alloc       = self.allocate_deliveries(
                x_[:, t], O_curr, cd_[:, t], fd_t, d_[:, t], self.firm_priority, t
            )
            Z_curr      = alloc["Z"]
            c_[:, t]    = alloc["consumption"]
            realised_government_[:, t]  = alloc["government"]
            realised_investment_[:, t]  = alloc["investment"]
            realised_inventories_[:, t] = alloc["inventories"]
            realised_exports_[:, t]     = alloc["exports"]
            realised_other_[:, t]       = alloc["other"]
            Z_colsum_[:, t] = np.sum(Z_curr, axis=0)

            mask   = cd_[:, t] > 0
            ration = np.where(mask, np.divide(c_[:, t], cd_[:, t], where=mask, out=np.ones_like(c_[:, t])), 1.0)
            self._hh_shortfall_ = {}
            for r in range(R):
                cd_r_j    = self._cap_bilateral_consumption(
                    self.theta_r[r][:, t] * Cdt_r[r, t] * (1.0 - self.epsilon_r[r, t]), r, t
                )
                c_r[r, t] = float(np.dot(cd_r_j, ration))

            # Capped household demand re-sourced from the substitute row within the shared budget.
            if self._hh_shortfall_:
                _budget_used = dict(self._supp_by_source_)
                _caps_t = self.import_supplement_caps_.get(t, {})
                for (_i, _r), _short in self._hh_shortfall_.items():
                    _rf = float(self.import_flexibility_vector[_i])
                    if self.hh_supplement_cover is not None:
                        _rf *= float(self.hh_supplement_cover[_i])
                    _supp = min(_rf, 1.0) * _short \
                        * float(ration[_i] if np.isfinite(ration[_i]) else 1.0)
                    for _src in self._row_sources(int(_i)):
                        if _supp <= 0.0:
                            break
                        _take = _supp
                        if _src in _caps_t:
                            _take = min(_take, max(0.0, _caps_t[_src]
                                                   - _budget_used.get(_src, 0.0)))
                        if _take <= 0.0:
                            continue
                        _budget_used[_src] = _budget_used.get(_src, 0.0) + _take
                        c_[_src, t]  += _take
                        c_r[_r, t]   += _take
                        hh_supplement_[_src, t] += _take
                        _supp -= _take
                if (self.row_region_idx is not None
                        and np.any(self.row_supply_cap_vector > 0)
                        and np.any(hh_supplement_[:, t] > 0)):
                    _cap_row = self.row_supply_cap_vector * np.maximum(x_[:, t], 0.0)
                    _cap_row[list(self.import_supplement_caps_.get(t, {}))] = np.inf
                    row_export_supplement_[:, t] = np.minimum(
                        row_export_supplement_[:, t] + hh_supplement_[:, t], _cap_row)

            _S_before   = S_prev
            S_prev      = self.inventory_S(x_[:, t], _S_before, Z_curr, A_current)
            # Realised use from stock, measured before the supplement restore.
            _use_t      = (_S_before + Z_curr - S_prev).sum(axis=0)
            # Restore imports to stock (prevents supplement from eroding inventory).
            S_prev      = np.maximum(S_prev + imp_supp_mat, 0.0)
            # Deduct intermediate share of export-pull output from value added.
            _ep_correction = np.sum(A_current, axis=0) * export_pull_supplement_[:, t]

            pi_[:, t]   = (
                self.profit_pi(x_[:, t], Z_curr, l_[:, t], fprod_k=fprod_k_cur)
                - import_supplement_[:, t]
                + row_export_supplement_[:, t]
                - _ep_correction
            )

            # Per-region totals of labour income and profits via reduceat.
            _l_r_t  = np.add.reduceat(l_[self._reg_perm, t],  self._reg_starts)
            _pi_r_t = np.add.reduceat(pi_[self._reg_perm, t], self._reg_starts)
            for r in range(R):
                household_income_r[r, t] = self._household_income_r(
                    r, float(_l_r_t[r]), float(_pi_r_t[r])
                ) + float(self.income_spillover_r[r, t])

            savings_r[:, t] = self.savings_s_regional(household_income_r[:, t], c_r[:, t])
            s[t] = float(np.sum(savings_r[:, t]))

            # Value added nets realised use, so stock build is not booked as loss.
            _va_t = (
                x_[:, t]
                - _use_t
                - self.imp_share * x_[:, t]
                + row_export_supplement_[:, t]
            )
            gdp_r_series[:, t]    = np.add.reduceat(_va_t[self._reg_perm], self._reg_starts)
            c_prod_regional[:, t] = np.add.reduceat(c_[self._reg_perm, t], self._reg_starts)

            # (R,R) inter-region trade: diagonal flows cancel in row-sum minus col-sum.
            _Z_perm = Z_curr[np.ix_(self._reg_perm, self._reg_perm)]
            _exp_t  = np.add.reduceat(
                np.add.reduceat(_Z_perm, self._reg_starts, axis=0),
                self._reg_starts, axis=1,
            )
            trade_balance_r[:, t] = _exp_t.sum(axis=1) - _exp_t.sum(axis=0)
            gdp_series[t] = float(np.sum(gdp_r_series[:, t]))

            if step_callback is not None:
                step_callback(t, gdp_r_series[:, t], gdp_r_series[:, 0])

            if store_full_matrices:
                S_list.append(S_prev.copy())
                O_list.append(O_curr)
                Z_list.append(Z_curr)
                I_list.append(imp_supp_mat)
            if store_delivery_rows is not None:
                Zr_list.append(Z_curr[_rows, :].copy())
                Ir_list.append(imp_supp_mat[_rows, :].copy())
                Sr_list.append(S_prev[_rows, :].copy())

        result = {
            "gross_output":              x_,              # (N, TT)
            "labour_compensation":       l_,              # (N, TT) dynamic wage bill from hire_fire
            "desired_l":                 desired_l_,      # (N, TT) target employment from hire_fire (same units as labour_compensation)
            "Z_colsums":                 Z_colsum_,       # (N, TT) total intermediate input per sector (always computed)
            "gdp":                       gdp_series,      # (TT,)
            "realised_consumption":      c_,              # (N, TT)
            "savings":                   s,               # (TT,)
            "household_closure_mode":    self.household_closure_mode,
            # Per-region outputs
            "gdp_regional":              gdp_r_series,           # (R, TT)
            "consumption_by_hh_region":  c_r,                    # (R, TT)
            "consumption_by_prod_region": c_prod_regional,        # (R, TT)
            "household_income_regional": household_income_r,      # (R, TT)
            "savings_regional":          savings_r,               # (R, TT)
            "trade_balance":             trade_balance_r,         # (R, TT)
            "import_supplement":          import_supplement_,           # (N, TT) by using sector
            "import_supplement_by_input": import_supplement_by_input_,  # (N, TT) by input good
            "row_export_supplement":      row_export_supplement_,       # (N, TT)
            "household_supplement":       hh_supplement_,               # (N, TT) by source row
            "supplement_by_source":       supp_by_source_,              # (N, TT)
            "export_pull_supplement":     export_pull_supplement_,      # (N, TT)
            "realised_government":        realised_government_,         # (N, TT)
            "realised_investment":        realised_investment_,         # (N, TT)
            "realised_inventories":       realised_inventories_,        # (N, TT)
            "realised_exports":           realised_exports_,            # (N, TT)
            "realised_other":             realised_other_,              # (N, TT)
            "price_index":                p_,              # (N, TT), 1.0 at t=0, flat when disabled
        }
        if store_full_matrices:
            result["inventories"]             = np.stack(S_list, axis=2)
            result["orders"]                  = np.stack(O_list, axis=2)
            result["intermediate_deliveries"] = np.stack(Z_list, axis=2)
            result["import_supplement_matrices"] = np.stack(I_list, axis=2)
            result["Z_colsums"]                 = Z_colsum_           # (N, TT)
            result["household_income"]          = household_income_   # (TT,) - region 0 view
            result["household_income_signal"]   = household_income_signal_  # (TT,)
        if store_delivery_rows is not None:
            result["delivery_rows"] = _rows
            result["intermediate_deliveries_rows"] = np.stack(Zr_list, axis=2)
            result["import_supplement_rows"] = np.stack(Ir_list, axis=2)
            result["inventory_rows"] = np.stack(Sr_list, axis=2)
        if validate:
            self._validate_run(result)
        return result

    # Plotting (thin wrappers around plotting.py standalone functions)
    def plot_results(
        self,
        current_results: dict,
        baseline_results: dict | None = None,
        title_suffix: str = "",
        save_path: str | None = None,
        uncertainty_data: dict | None = None,
    ) -> None:
        """Plot aggregate gross output. Delegates to plotting.plot_results."""
        from .plotting import plot_results as _plot_results
        _plot_results(self, current_results, baseline_results, title_suffix, save_path, uncertainty_data)

    def plot_regional_results(
        self,
        current_results: dict,
        baseline_results: dict | None = None,
        title_suffix: str = "",
        save_path: str | None = None,
    ) -> None:
        """Plot per-region and aggregate output. Delegates to plotting.plot_regional_results."""
        from .plotting import plot_regional_results as _plot_regional_results
        _plot_regional_results(self, current_results, baseline_results, title_suffix, save_path)

    def _plot_uncertainty_bands(
        self,
        axes,
        time: np.ndarray,
        uncertainty_data: dict,
        baseline_results: dict | None,
        current_results: dict,
        start_period: int = 0,
        colour: str | None = None,
    ) -> None:
        """Draw MC uncertainty bands. Delegates to plotting.plot_uncertainty_bands."""
        from .plotting import plot_uncertainty_bands
        plot_uncertainty_bands(
            axes, time, uncertainty_data, baseline_results, current_results,
            start_period, colour
        )

