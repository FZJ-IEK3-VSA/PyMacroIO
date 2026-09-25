"""
Entry point for `python -m pyMacroIO`.
Runs the baseline and example scenario figures, equivalent to `python -m examples.run_scenarios`.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Allow running from any cwd by ensuring the repo root is on sys.path.
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from pyMacroIO import (
    ModelConfig,
    ScenarioManager,
    SIMULATION_PERIODS,
    ENABLE_PLOTTING,
    ENABLE_INPUT_AVAILABILITY_SHOCK_PLOT,
    MC_PLOT_SIMULATIONS,
    run_consumption_shock_all_prod_functions,
    run_consumption_shock_household_closure_sensitivity,
    run_input_availability_shock_all_prod_functions,
    run_input_availability_shock_household_closure_sensitivity,
    run_input_availability_sensitivity_panel,
)

logging.getLogger().setLevel(logging.WARNING)

figures_dir = Path("figures")
figures_dir.mkdir(parents=True, exist_ok=True)

# Baseline
manager      = ScenarioManager(
    ModelConfig(n_periods=SIMULATION_PERIODS, time_frequency="daily", prod_function="leontief")
)
baseline_run = manager.run_baseline(force=True)

if ENABLE_PLOTTING:
    baseline_run.model.plot_results(
        baseline_run.results,
        baseline_results=None,
        title_suffix="(Daily Baseline)",
        save_path=str(figures_dir / "baseline.png"),
    )

    run_consumption_shock_all_prod_functions(
        intensity=0.2, duration=3, start=2,
        save_path=str(figures_dir / "consumption_shock_all_prod_functions.png"),
        n_simulations=MC_PLOT_SIMULATIONS,
    )
    run_consumption_shock_household_closure_sensitivity(
        intensity=0.2, duration=3, start=2, prod_function="leontief",
        save_path=str(figures_dir / "consumption_shock_household_closure_sensitivity.png"),
        n_simulations=MC_PLOT_SIMULATIONS,
    )

if ENABLE_PLOTTING and ENABLE_INPUT_AVAILABILITY_SHOCK_PLOT:
    run_input_availability_shock_all_prod_functions(
        input_sector_label=None,
        save_path=str(figures_dir / "input_availability_shock_all_prod_functions.png"),
        n_simulations=MC_PLOT_SIMULATIONS,
    )
    run_input_availability_shock_household_closure_sensitivity(
        input_sector_label=None, prod_function="leontief",
        save_path=str(
            figures_dir / "input_availability_shock_household_closure_sensitivity.png"
        ),
        n_simulations=MC_PLOT_SIMULATIONS,
    )
    run_input_availability_sensitivity_panel(
        input_sector_label=None, prod_function="leontief",
        save_path=str(figures_dir / "input_availability_sensitivity_panel.png"),
    )
