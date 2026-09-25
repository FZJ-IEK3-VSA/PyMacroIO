"""
ModelConfig dataclass and script-level run defaults.
ModelConfig holds per-scenario overridable parameters, the script-level constants drive examples/.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from .constants import (
    HOUSEHOLD_CLOSURE_MODES,
    INVESTMENT_CLOSURE_MODES,
    PRODUCTION_FUNCTIONS,
    FIRM_PRIORITY_MODES,
    CES_ELASTICITY_DEFAULT,
    DEFAULT_LES_FRISCH,
    FIRING_SPEED_DAMPING,
    LES_SECTOR_ELASTICITIES,
)


# Scenario-overridable parameters
@dataclass
class ModelConfig:
    """Per-scenario parameters, deep-copied per scenario so mutations never bleed between runs."""

    n_periods: int = 60
    time_frequency: str = "daily"
    tau: np.ndarray | None = None
    gamma_hire: np.ndarray | None = None
    gamma_fire: np.ndarray | None = None
    benefits: float | np.ndarray = 0.1
    c_other_coef: float | np.ndarray = 0.1
    prod_function: str = "leontief.adapted"
    hiringfiring: bool = True
    firm_priority: str = "no"
    inventory_days: np.ndarray | None = None
    inventory_days_daily: float = 2.0
    inventory_days_other: float = 2.0
    # Path to the IO data pickle.
    data_path: str = "data/example_data.pkl"
    savings_rate: None | float | np.ndarray = None
    ces_elasticity: float | list[float] = CES_ELASTICITY_DEFAULT
    income_tax_rate: float | list[float] = 0.0
    # Per-sector fraction of an inventory shortfall coverable by external imports.
    import_flexibility: float | list[float] = 0.0
    # Per-sector cap on RoW output diverted per period for import_flexibility. Ignored without RoW.
    row_supply_cap: float | list[float] = 0.0
    # Per-sector fraction of last period's slack capacity absorbed as extra external demand.
    export_pull: float | list[float] = 0.0
    # Per-sector subsistence floor (fraction of base-year consumption); 0 disables LES.
    subsistence_shares: float | list[float] = 0.0
    household_closure_mode: str | list[str] = "return_to_base"
    # Region count; n_regions=1 gives standard (single-region) behaviour.
    n_regions: int = 1
    region_map: np.ndarray | None = None
    region_labels: list[str] | None = None

    # Income elasticity of government spending. 0.0 = fixed; positive = pro-cyclical.
    gov_income_elasticity: float = 0.0
    # Investment closure: "fixed" holds base-year investment; "keynesian" scales by lagged savings.
    investment_closure: str = "fixed"
    # Partial-adjustment speed for the Keynesian closure (0 < alpha <= 1), 1.0 being instant.
    investment_adj_speed: float = 1.0
    # EMA weight on the savings signal before computing inv_scale (0 < w <= 1), 1.0 no smoothing.
    investment_savings_ema: float = 1.0
    # Per-period growth cap on inv_scale (None = uncapped), preventing runaway compounding.
    investment_scale_growth_cap: float | None = None

    # KLEMS structure, active only when prod_function="klems", defaults being indicative midpoints.
    klems_sigma_e:   float | np.ndarray = 0.5   # CES within energy inputs
    klems_sigma_m:   float | np.ndarray = 0.3   # CES within material inputs
    klems_sigma_kle: float | np.ndarray = 0.3   # KL bundle vs energy (inner nest)
    klems_sigma_klem: float | np.ndarray = 0.3  # KLE bundle vs materials (top nest)
    klems_sigma_kl:  float | np.ndarray = 0.7   # capital vs labour
    klems_sigma_l:   float | np.ndarray = 1.5   # skill tiers within labour
    # "materials" pattern-matches the M nest, "intermediates" puts every non-energy input in M.
    klems_m_scope: str = "materials"

    # Multiplier on gamma_fire when shedding labour (1.0 = symmetric speeds).
    firing_speed_damping: float = FIRING_SPEED_DAMPING

    # Skill-specific savings rates [Low, Med, High]; None = uniform rate.
    savings_rate_by_skill: np.ndarray | None = None

    # Wage curve (Blanchflower-Oswald): scale labour-output ratio by (U/U0)^(-beta).
    wage_curve: bool = False
    wage_curve_beta: float | np.ndarray = 0.10
    # Wage floor as fraction of base-year wage, only meaningful when wage_curve=True.
    wage_floor_ratio: float | np.ndarray | None = None

    # Price pass-through propagates unit-cost shocks through the cost-push Leontief inverse of A.
    price_passthrough_enabled: bool = False
    # Network amplification fraction per direction, the own-cost term always at full strength.
    price_passthrough_pos: float = 1.0
    price_passthrough_neg: float = 0.5
    # Deflates the household income signal by the regional price index, per region or globally.
    price_deflate_household_income: bool | list = False

    def __post_init__(self) -> None:
        """Validate all fields; raise ValueError on invalid input."""
        if self.n_periods <= 0:
            raise ValueError(f"n_periods must be positive; got {self.n_periods}")
        if self.time_frequency not in ("daily", "quarterly"):
            raise ValueError(
                f"time_frequency must be 'daily' or 'quarterly'; got {self.time_frequency!r}"
            )
        if self.savings_rate is not None:
            rates = np.atleast_1d(np.asarray(self.savings_rate, dtype=float))
            if not np.all((rates >= 0) & (rates < 1)):
                raise ValueError(f"savings_rate must be in [0, 1); got {self.savings_rate}")
        _ces = np.atleast_1d(np.asarray(self.ces_elasticity, dtype=float))
        if not np.all(_ces > 0):
            raise ValueError(f"ces_elasticity must be positive for all sectors; got {self.ces_elasticity}")
        _ity = np.atleast_1d(np.asarray(self.income_tax_rate, dtype=float))
        if not np.all((_ity >= 0) & (_ity < 1)):
            raise ValueError(f"income_tax_rate must be in [0, 1) for all regions; got {self.income_tax_rate}")
        _flex = np.atleast_1d(np.asarray(self.import_flexibility, dtype=float))
        if not np.all((_flex >= 0) & (_flex <= 1)):
            raise ValueError(f"import_flexibility must be in [0, 1] for all sectors; got {self.import_flexibility}")
        _cap = np.atleast_1d(np.asarray(self.row_supply_cap, dtype=float))
        if not np.all((_cap >= 0) & (_cap <= 1)):
            raise ValueError(f"row_supply_cap must be in [0, 1] for all sectors; got {self.row_supply_cap}")
        _ep = np.atleast_1d(np.asarray(self.export_pull, dtype=float))
        if not np.all((_ep >= 0) & (_ep <= 1)):
            raise ValueError(f"export_pull must be in [0, 1] for all sectors; got {self.export_pull}")
        for _pt_name in ("price_passthrough_pos", "price_passthrough_neg"):
            _pt_val = float(getattr(self, _pt_name))
            if not (0.0 <= _pt_val <= 1.0):
                raise ValueError(f"{_pt_name} must be in [0, 1]; got {_pt_val}")
        _pd = np.atleast_1d(np.asarray(self.price_deflate_household_income, dtype=bool))
        if _pd.size not in (1, self.n_regions):
            raise ValueError(
                "price_deflate_household_income must be a bool or a length "
                f"n_regions sequence; got size {_pd.size}"
            )
        if _pd.any() and not self.price_passthrough_enabled:
            raise ValueError(
                "price_deflate_household_income requires price_passthrough_enabled=True"
            )
        _ss = np.atleast_1d(np.asarray(self.subsistence_shares, dtype=float))
        if not np.all((_ss >= 0) & (_ss < 1)):
            raise ValueError(f"subsistence_shares must be in [0, 1) for all sectors; got {self.subsistence_shares}")
        for _klems_field in (
            "klems_sigma_e", "klems_sigma_m", "klems_sigma_kle",
            "klems_sigma_kl", "klems_sigma_l",
        ):
            _v = np.atleast_1d(np.asarray(getattr(self, _klems_field), dtype=float))
            if not np.all(_v > 0):
                raise ValueError(
                    f"{_klems_field} must be positive for all sectors; got {getattr(self, _klems_field)}"
                )
        if isinstance(self.household_closure_mode, str):
            if self.household_closure_mode not in HOUSEHOLD_CLOSURE_MODES:
                raise ValueError(
                    f"household_closure_mode must be one of {HOUSEHOLD_CLOSURE_MODES}; "
                    f"got {self.household_closure_mode!r}"
                )
        else:
            if len(self.household_closure_mode) != self.n_regions:
                raise ValueError(
                    f"household_closure_mode list length ({len(self.household_closure_mode)}) "
                    f"must equal n_regions ({self.n_regions})"
                )
            for mode in self.household_closure_mode:
                if mode not in HOUSEHOLD_CLOSURE_MODES:
                    raise ValueError(
                        f"household_closure_mode must be one of {HOUSEHOLD_CLOSURE_MODES}; "
                        f"got {mode!r}"
                    )
        if self.prod_function not in PRODUCTION_FUNCTIONS:
            raise ValueError(
                f"prod_function must be one of {PRODUCTION_FUNCTIONS}; got {self.prod_function!r}"
            )
        if self.firm_priority not in FIRM_PRIORITY_MODES:
            raise ValueError(
                f"firm_priority must be one of {FIRM_PRIORITY_MODES}; got {self.firm_priority!r}"
            )
        if self.klems_m_scope not in ("materials", "intermediates"):
            raise ValueError(
                f"klems_m_scope must be 'materials' or 'intermediates'; got {self.klems_m_scope!r}"
            )
        if not 0.0 < float(self.firing_speed_damping) <= 1.0:
            raise ValueError(
                f"firing_speed_damping must be in (0, 1]; got {self.firing_speed_damping}"
            )
        if self.n_regions < 1:
            raise ValueError(f"n_regions must be >= 1; got {self.n_regions}")
        if isinstance(self.benefits, np.ndarray) and len(self.benefits) != self.n_regions:
            raise ValueError(
                f"benefits array length ({len(self.benefits)}) must equal n_regions ({self.n_regions})"
            )
        if isinstance(self.c_other_coef, np.ndarray) and len(self.c_other_coef) != self.n_regions:
            raise ValueError(
                f"c_other_coef array length ({len(self.c_other_coef)}) must equal n_regions ({self.n_regions})"
            )
        try:
            float(self.gov_income_elasticity)
        except (TypeError, ValueError):
            raise ValueError(
                f"gov_income_elasticity must be a scalar numeric value; got {self.gov_income_elasticity!r}"
            )
        if self.investment_closure not in INVESTMENT_CLOSURE_MODES:
            raise ValueError(
                f"investment_closure must be one of {INVESTMENT_CLOSURE_MODES}; "
                f"got {self.investment_closure!r}"
            )
        _adj = float(self.investment_adj_speed)
        if not (0 < _adj <= 1.0):
            raise ValueError(
                f"investment_adj_speed must be in (0, 1]; got {self.investment_adj_speed}"
            )
        _ema = float(self.investment_savings_ema)
        if not (0 < _ema <= 1.0):
            raise ValueError(
                f"investment_savings_ema must be in (0, 1]; got {self.investment_savings_ema}"
            )
        if self.savings_rate_by_skill is not None:
            _srsk = np.asarray(self.savings_rate_by_skill, dtype=float)
            if _srsk.shape != (3,):
                raise ValueError(
                    f"savings_rate_by_skill must be shape (3,) [Low, Med, High]; got {_srsk.shape}"
                )
            if not np.all((_srsk >= 0) & (_srsk < 1)):
                raise ValueError(
                    f"savings_rate_by_skill values must be in [0, 1); got {self.savings_rate_by_skill}"
                )
        _wcb = np.atleast_1d(np.asarray(self.wage_curve_beta, dtype=float))
        if not np.all(_wcb > 0):
            raise ValueError(
                f"wage_curve_beta must be positive for all sectors; got {self.wage_curve_beta}"
            )
        if self.wage_floor_ratio is not None:
            _wfr = np.atleast_1d(np.asarray(self.wage_floor_ratio, dtype=float))
            if not np.all((_wfr > 0) & (_wfr <= 1)):
                raise ValueError(
                    f"wage_floor_ratio must be in (0, 1] for all sectors; got {self.wage_floor_ratio}"
                )

    def clone(self) -> "ModelConfig":
        """Return a deep copy of this config."""
        return copy.deepcopy(self)


# LES calibration helper
def build_subsistence_shares_vector(
    model,
    frisch: float = DEFAULT_LES_FRISCH,
    sector_elasticities: dict[str, float] | None = None,
) -> np.ndarray:
    """Return an N-vector of LES subsistence shares for ModelConfig.subsistence_shares, share_i =
    1 - eta_i / |frisch| clipped to [0, 0.95], matched by longest substring, unmatched eta=1."""
    if frisch >= 0:
        raise ValueError(f"frisch must be negative; got {frisch}")
    table = LES_SECTOR_ELASTICITIES if sector_elasticities is None else sector_elasticities
    # Default (eta=1): fill covers sectors absent from label_to_index (defensive).
    default_share = float(np.clip(1.0 - 1.0 / abs(frisch), 0.0, 0.95))
    shares = np.full(model.N, default_share, dtype=float)
    for full_label, idx in model.label_to_index.items():
        sector_name = full_label.split(":", 1)[1] if ":" in full_label else full_label
        best_len, best_eta = -1, 1.0
        for key, eta in table.items():
            if key.lower() in sector_name.lower() and len(key) > best_len:
                best_len, best_eta = len(key), eta
        if best_len >= 0:   # matched: update; unmatched: keep the fill (η=1 default)
            shares[idx] = float(np.clip(1.0 - best_eta / abs(frisch), 0.0, 0.95))
    return shares


# KLEMS patterns, matched case-insensitively at word starts, so "gas" cannot catch "biogas".
_ENERGY_PATTERNS: list[str] = [
    "electricity", "gas", "biogas", "coal", "crude oil", "petroleum refin",
    "nuclear", "biomass", "wind", "solar", "tide", "geotherm",
    "steam", "heat", "coke oven", "natural gas distribution",
]
_MATERIALS_PATTERNS: list[str] = [
    "mining", "quarry", "iron", "steel", "alumin", "cement",
    "lime", "plaster", "glass", "basic metal", "chemical",
    "rubber", "plastic", "paper", "wood",
]


def classify_klems(sector_labels: list[str],
                   m_scope: str = "materials") -> dict[str, np.ndarray]:
    """Return boolean (N,) masks keyed 'energy', 'materials', 'services', and 'other' (alias),
    energy preceding materials, m_scope="intermediates" folding all non-energy into the M nest."""
    N = len(sector_labels)
    e_mask = np.zeros(N, dtype=bool)
    m_mask = np.zeros(N, dtype=bool)
    import re
    for i, label in enumerate(sector_labels):
        bare = label.split(":", 1)[1] if ":" in label else label
        bare_lower = bare.lower()
        if any(re.search(r"\b" + re.escape(p), bare_lower)
               for p in _ENERGY_PATTERNS):
            e_mask[i] = True
        elif any(re.search(r"\b" + re.escape(p), bare_lower)
                 for p in _MATERIALS_PATTERNS):
            m_mask[i] = True
    if m_scope == "intermediates":
        m_mask = ~e_mask
    nd_mask = ~e_mask & ~m_mask
    return {"energy": e_mask, "materials": m_mask, "services": nd_mask, "other": nd_mask}


# Script-level defaults (used by examples/ scripts)
SIMULATION_PERIODS = 60
ENABLE_PLOTTING = True
ENABLE_INPUT_AVAILABILITY_SHOCK_PLOT = True
MC_PLOT_SIMULATIONS = 50

# Default calibration for the example input-availability scenario.
INPUT_SHOCK_DEFAULT_REDUCTION_PCT = 0.3
INPUT_SHOCK_DEFAULT_DURATION = 3
INPUT_SHOCK_DEFAULT_START = 2
INPUT_SHOCK_DEFAULT_INVENTORY_DAYS = 5.0

# Tighter stress-test variant.
INPUT_SHOCK_STRESS_REDUCTION_PCT = 0.5
INPUT_SHOCK_STRESS_DURATION = 3
INPUT_SHOCK_STRESS_START = 2
INPUT_SHOCK_STRESS_INVENTORY_DAYS = 1.0
