#!/usr/bin/env python3
"""
Baseline and example scenario runner producing figures in figures/.

Run from the PyMacroIO-main directory with python examples/run_scenarios.py.
"""

import logging
import sys
from pathlib import Path

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


def _step(msg: str) -> None:
    print(f"\n[run_scenarios] {msg}", flush=True)


def _done(path: str) -> None:
    print(f"  -> {path}", flush=True)


# Baseline
_step(f"Running baseline  (prod_function=leontief, {SIMULATION_PERIODS} periods) ...")
manager = ScenarioManager(
    ModelConfig(n_periods=SIMULATION_PERIODS, time_frequency="daily", prod_function="leontief")
)
baseline_run = manager.run_baseline(force=True)
print("  Baseline OK", flush=True)

if ENABLE_PLOTTING:
    baseline_run.model.plot_results(
        baseline_run.results,
        baseline_results=None,
        title_suffix="(Daily Baseline)",
        save_path=str(figures_dir / "baseline.png"),
    )
    _done("figures/baseline.png")


# Consumption shock - all production functions
if ENABLE_PLOTTING:
    _step(f"Consumption shock - all prod functions  (MC n={MC_PLOT_SIMULATIONS}) ...")
    run_consumption_shock_all_prod_functions(
        intensity=0.2,
        duration=3,
        start=2,
        save_path=str(figures_dir / "consumption_shock_all_prod_functions.png"),
        n_simulations=MC_PLOT_SIMULATIONS,
    )
    _done("figures/consumption_shock_all_prod_functions.png")

    _step(f"Consumption shock - household closure sensitivity  (MC n={MC_PLOT_SIMULATIONS}) ...")
    run_consumption_shock_household_closure_sensitivity(
        intensity=0.2,
        duration=3,
        start=2,
        prod_function="leontief",
        save_path=str(figures_dir / "consumption_shock_household_closure_sensitivity.png"),
        n_simulations=MC_PLOT_SIMULATIONS,
    )
    _done("figures/consumption_shock_household_closure_sensitivity.png")


# Input-availability shock - all production functions
if ENABLE_PLOTTING and ENABLE_INPUT_AVAILABILITY_SHOCK_PLOT:
    _step(f"Input-availability shock - all prod functions  (MC n={MC_PLOT_SIMULATIONS}) ...")
    run_input_availability_shock_all_prod_functions(
        input_sector_label=None,
        save_path=str(figures_dir / "input_availability_shock_all_prod_functions.png"),
        n_simulations=MC_PLOT_SIMULATIONS,
    )
    _done("figures/input_availability_shock_all_prod_functions.png")

    _step(f"Input-availability shock - household closure sensitivity  (MC n={MC_PLOT_SIMULATIONS}) ...")
    run_input_availability_shock_household_closure_sensitivity(
        input_sector_label=None,
        prod_function="leontief",
        save_path=str(
            figures_dir / "input_availability_shock_household_closure_sensitivity.png"
        ),
        n_simulations=MC_PLOT_SIMULATIONS,
    )
    _done("figures/input_availability_shock_household_closure_sensitivity.png")

    _step("Input-availability sensitivity panel ...")
    run_input_availability_sensitivity_panel(
        input_sector_label=None,
        prod_function="leontief",
        save_path=str(figures_dir / "input_availability_sensitivity_panel.png"),
    )
    _done("figures/input_availability_sensitivity_panel.png")


_step("All done.")
