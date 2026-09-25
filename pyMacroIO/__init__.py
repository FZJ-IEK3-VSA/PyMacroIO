"""
pyMacroIO dynamic disequilibrium model with input-output structure.
The public API is the set of names in __all__ below.
"""

from __future__ import annotations

import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Configuration
from .config import (
    ModelConfig,
    build_subsistence_shares_vector,
    SIMULATION_PERIODS,
    ENABLE_PLOTTING,
    ENABLE_INPUT_AVAILABILITY_SHOCK_PLOT,
    MC_PLOT_SIMULATIONS,
    INPUT_SHOCK_DEFAULT_REDUCTION_PCT,
    INPUT_SHOCK_DEFAULT_DURATION,
    INPUT_SHOCK_DEFAULT_START,
    INPUT_SHOCK_DEFAULT_INVENTORY_DAYS,
    INPUT_SHOCK_STRESS_REDUCTION_PCT,
    INPUT_SHOCK_STRESS_DURATION,
    INPUT_SHOCK_STRESS_START,
    INPUT_SHOCK_STRESS_INVENTORY_DAYS,
)

# Constants
from .constants import (
    HOUSEHOLD_CLOSURE_MODES,
    PRODUCTION_FUNCTIONS,
    FIRM_PRIORITY_MODES,
    DEFAULT_SAVINGS_RATE,
    DEFAULT_TAU,
    DEFAULT_GAMMA_HIRE,
    DEFAULT_GAMMA_FIRE,
    GAMMA_HIRE_MIN, GAMMA_HIRE_MAX,
    GAMMA_FIRE_MIN, GAMMA_FIRE_MAX,
    TAU_MIN, TAU_MAX,
    CES_ELASTICITY_DEFAULT,
    NUMERIC_LARGE,
    LES_SECTOR_ELASTICITIES,
)

# Scenario infrastructure
from .scenarios import (
    Scenario,
    ScenarioRunResult,
    ConsumptionShockSpec,
    InputAvailabilityShockSpec,
    CONSUMPTION_EXAMPLE_SHOCK_SPEC,
    INPUT_AVAILABILITY_EXAMPLE_SHOCK_SPEC,
    INPUT_AVAILABILITY_STRESS_SHOCK_SPEC,
    ScenarioManager,
)

# Core model
from .model import (
    InputOutputModel,
    ConvergenceAbort,
    estimate_essential_inputs_from_io_data,
)

# Uncertainty analysis
from .uncertainty import MonteCarloUncertaintyAnalysis

# Shock runners and helpers
from .shocks import (
    run_consumption_shock_scenario,
    run_input_availability_shock_scenario,
    run_consumption_shock_household_closure_sensitivity,
    run_input_availability_shock_household_closure_sensitivity,
    run_consumption_shock_all_prod_functions,
    run_input_availability_shock_all_prod_functions,
    run_input_availability_sensitivity_panel,
    key_supplier_sector_label,
    key_supplier_sector_label_in_region,
)

# Standalone plotting functions
from .plotting import (
    plot_results,
    plot_regional_results,
    plot_uncertainty_bands,
    plot_household_closure_comparison,
    plot_prod_functions_comparison,
    plot_sensitivity_panel,
)

__all__ = [
    # Exceptions
    "ConvergenceAbort",
    # Config
    "ModelConfig",
    "build_subsistence_shares_vector",
    "LES_SECTOR_ELASTICITIES",
    "SIMULATION_PERIODS",
    "ENABLE_PLOTTING",
    "ENABLE_INPUT_AVAILABILITY_SHOCK_PLOT",
    "MC_PLOT_SIMULATIONS",
    "INPUT_SHOCK_DEFAULT_REDUCTION_PCT",
    "INPUT_SHOCK_DEFAULT_DURATION",
    "INPUT_SHOCK_DEFAULT_START",
    "INPUT_SHOCK_DEFAULT_INVENTORY_DAYS",
    "INPUT_SHOCK_STRESS_REDUCTION_PCT",
    "INPUT_SHOCK_STRESS_DURATION",
    "INPUT_SHOCK_STRESS_START",
    "INPUT_SHOCK_STRESS_INVENTORY_DAYS",
    # Constants
    "HOUSEHOLD_CLOSURE_MODES",
    "PRODUCTION_FUNCTIONS",
    "FIRM_PRIORITY_MODES",
    "DEFAULT_SAVINGS_RATE",
    "DEFAULT_TAU",
    "DEFAULT_GAMMA_HIRE",
    "DEFAULT_GAMMA_FIRE",
    "GAMMA_HIRE_MIN", "GAMMA_HIRE_MAX",
    "GAMMA_FIRE_MIN", "GAMMA_FIRE_MAX",
    "TAU_MIN", "TAU_MAX",
    "CES_ELASTICITY_DEFAULT",
    "NUMERIC_LARGE",
    # Scenarios
    "Scenario",
    "ScenarioRunResult",
    "ConsumptionShockSpec",
    "InputAvailabilityShockSpec",
    "CONSUMPTION_EXAMPLE_SHOCK_SPEC",
    "INPUT_AVAILABILITY_EXAMPLE_SHOCK_SPEC",
    "INPUT_AVAILABILITY_STRESS_SHOCK_SPEC",
    "ScenarioManager",
    # Model
    "InputOutputModel",
    "estimate_essential_inputs_from_io_data",
    # Uncertainty
    "MonteCarloUncertaintyAnalysis",
    # Shocks
    "run_consumption_shock_scenario",
    "run_input_availability_shock_scenario",
    "run_consumption_shock_household_closure_sensitivity",
    "run_input_availability_shock_household_closure_sensitivity",
    "run_consumption_shock_all_prod_functions",
    "run_input_availability_shock_all_prod_functions",
    "run_input_availability_sensitivity_panel",
    "key_supplier_sector_label",
    "key_supplier_sector_label_in_region",
    # Plotting
    "plot_results",
    "plot_regional_results",
    "plot_uncertainty_bands",
    "plot_household_closure_comparison",
    "plot_prod_functions_comparison",
    "plot_sensitivity_panel",
]
