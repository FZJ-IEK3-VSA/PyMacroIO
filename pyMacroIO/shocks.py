"""
Module-level scenario runner functions and shock helpers.
These orchestrate ScenarioManager, MonteCarloUncertaintyAnalysis, and plotting in single calls.
"""

from __future__ import annotations

import logging
import traceback
from typing import Any

import numpy as np

from .config import (
    SIMULATION_PERIODS,
    MC_PLOT_SIMULATIONS,
    INPUT_SHOCK_DEFAULT_REDUCTION_PCT,
    INPUT_SHOCK_DEFAULT_DURATION,
    INPUT_SHOCK_DEFAULT_START,
    INPUT_SHOCK_DEFAULT_INVENTORY_DAYS,
    INPUT_SHOCK_STRESS_REDUCTION_PCT,
    INPUT_SHOCK_STRESS_DURATION,
    INPUT_SHOCK_STRESS_START,
    INPUT_SHOCK_STRESS_INVENTORY_DAYS,
    ModelConfig,
)
from .constants import HOUSEHOLD_CLOSURE_MODES
from .scenarios import (
    Scenario,
    ScenarioRunResult,
    ScenarioManager,
    ConsumptionShockSpec,
    InputAvailabilityShockSpec,
    INPUT_AVAILABILITY_STRESS_SHOCK_SPEC,
)
from .model import InputOutputModel
from .uncertainty import MonteCarloUncertaintyAnalysis
from .plotting import (
    plot_household_closure_comparison,
    plot_prod_functions_comparison,
    plot_sensitivity_panel,
)

logger = logging.getLogger(__name__)


# Key-supplier helpers
def key_supplier_sector_label(model: InputOutputModel) -> str:
    """Return the sector label with the largest forward supply (row sum of A), the most widely
    relied-upon supplier, so an input-availability shock to it is likely to bind downstream."""
    row_sums = np.sum(model.A, axis=1)
    return model.sector_labels[int(np.argmax(row_sums))]


def key_supplier_sector_label_in_region(
    model: InputOutputModel, r: int
) -> str:
    """Return region r's sector with the largest row sum of A, for targeted supply shocks."""
    J_r      = model.region_sector_indices[r]
    row_sums = model.A[J_r, :].sum(axis=1)
    return model.sector_labels[J_r[int(row_sums.argmax())]]


# Shock-spec resolvers (internal helpers)
def _resolve_consumption_shock_spec(
    intensity: float,
    duration: int,
    start: int,
    shock_spec: ConsumptionShockSpec | None,
) -> ConsumptionShockSpec:
    if shock_spec is not None:
        return shock_spec
    return ConsumptionShockSpec(intensity=intensity, duration=duration, start=start, tier="example")


def _resolve_input_availability_shock_spec(
    reduction_pct: float,
    duration: int,
    start: int,
    inventory_days: float | None,
    input_sector_label: str | None,
    shock_spec: InputAvailabilityShockSpec | None,
) -> InputAvailabilityShockSpec:
    if shock_spec is not None:
        return shock_spec
    return InputAvailabilityShockSpec(
        reduction_pct=reduction_pct,
        duration=duration,
        start=start,
        inventory_days=inventory_days,
        input_sector_label=input_sector_label,
        tier="example",
    )


def _resolve_household_closure_modes(closure_modes: list[str] | None) -> list[str]:
    modes = list(HOUSEHOLD_CLOSURE_MODES) if closure_modes is None else list(closure_modes)
    invalid = [m for m in modes if m not in HOUSEHOLD_CLOSURE_MODES]
    if invalid:
        raise ValueError(
            f"Unknown household closure modes: {invalid}. Supported: {HOUSEHOLD_CLOSURE_MODES}"
        )
    return modes


# Single-scenario runners
def run_consumption_shock_scenario(
    intensity: float = 0.2,
    duration: int = 3,
    start: int = 2,
    prod_function: str = "leontief",
    shock_spec: ConsumptionShockSpec | None = None,
    household_closure_mode: str = "return_to_base",
):
    """Run a consumption-shock scenario and return (scenario_run, baseline_run)."""
    spec = _resolve_consumption_shock_spec(intensity, duration, start, shock_spec)
    manager = ScenarioManager(ModelConfig(
        n_periods=SIMULATION_PERIODS,
        time_frequency="daily",
        prod_function=prod_function,
        household_closure_mode=household_closure_mode,
    ))
    manager.run_baseline(force=True)

    def consumption_shock(model: InputOutputModel) -> None:
        for t in range(spec.start, min(spec.start + spec.duration, model.TT)):
            model.epsilon_[t] = spec.intensity

    scenario = Scenario(
        name="consumption_shock",
        description=(
            f"{spec.intensity * 100:.0f}% consumption shock for "
            f"{spec.duration} periods ({spec.tier})"
        ),
        config=manager.base_config.clone(),
        shocks=[consumption_shock],
    )
    return manager.run_scenario(scenario)


def run_input_availability_shock_scenario(
    input_sector_label: str | None = None,
    reduction_pct: float = INPUT_SHOCK_DEFAULT_REDUCTION_PCT,
    duration: int = INPUT_SHOCK_DEFAULT_DURATION,
    start: int = INPUT_SHOCK_DEFAULT_START,
    prod_function: str = "leontief",
    inventory_days: float | None = INPUT_SHOCK_DEFAULT_INVENTORY_DAYS,
    shock_spec: InputAvailabilityShockSpec | None = None,
    household_closure_mode: str = "return_to_base",
):
    """Run an input-availability shock scenario and return (scenario_run, baseline_run)."""
    spec = _resolve_input_availability_shock_spec(
        reduction_pct, duration, start, inventory_days, input_sector_label, shock_spec
    )
    manager = ScenarioManager(ModelConfig(
        n_periods=SIMULATION_PERIODS,
        time_frequency="daily",
        prod_function=prod_function,
        inventory_days=spec.inventory_days,
        household_closure_mode=household_closure_mode,
    ))
    manager.run_baseline(force=True)
    baseline_run = manager.run_baseline(force=False)
    sector_label = (
        spec.input_sector_label
        if spec.input_sector_label is not None
        else key_supplier_sector_label(baseline_run.model)
    )

    def input_availability_shock(model: InputOutputModel) -> None:
        for t in range(spec.start, min(spec.start + spec.duration, model.TT)):
            model.apply_input_availability_shock(sector_label, t, spec.reduction_pct)

    scenario = Scenario(
        name="input_availability_shock",
        description=(
            f"{spec.reduction_pct * 100:.0f}% input-availability shock "
            f"({sector_label}) for {spec.duration} periods ({spec.tier})"
        ),
        config=manager.base_config.clone(),
        shocks=[input_availability_shock],
    )
    return manager.run_scenario(scenario)


# Household-closure sensitivity sweeps
def run_consumption_shock_household_closure_sensitivity(
    intensity: float = 0.2,
    duration: int = 3,
    start: int = 2,
    prod_function: str = "leontief",
    closure_modes: list[str] | None = None,
    save_path: str | None = None,
    n_simulations: int = MC_PLOT_SIMULATIONS,
) -> dict[str, dict[str, Any]]:
    """Run the consumption shock for each household closure mode with MC bands."""
    modes = _resolve_household_closure_modes(closure_modes)
    results_data: dict[str, dict[str, Any]] = {}

    n_modes = len(modes)
    for idx, closure_mode in enumerate(modes, 1):
        print(f"  [{idx}/{n_modes}] closure_mode={closure_mode} ...", flush=True)
        scenario_run, baseline_run = run_consumption_shock_scenario(
            intensity=intensity,
            duration=duration,
            start=start,
            prod_function=prod_function,
            household_closure_mode=closure_mode,
        )
        mc = MonteCarloUncertaintyAnalysis(baseline_run.model, n_simulations=n_simulations)
        mc.run_uncertainty_analysis(
            shock_scenario="consumption",
            shock_params={"intensity": intensity, "duration": duration, "start": start},
            seed=42,
        )
        results_data[closure_mode] = {
            "scenario_results": scenario_run.results,
            "baseline_results": baseline_run.results,
            "uncertainty_data": mc.get_uncertainty_data_for_plotting(free_raw=True),
            "time_frequency":   scenario_run.model.time_frequency,
        }
        del mc
        print(f"       done", flush=True)

    plot_household_closure_comparison(results_data, save_path=save_path)
    return results_data


def run_input_availability_shock_household_closure_sensitivity(
    input_sector_label: str | None = None,
    reduction_pct: float = INPUT_SHOCK_DEFAULT_REDUCTION_PCT,
    duration: int = INPUT_SHOCK_DEFAULT_DURATION,
    start: int = INPUT_SHOCK_DEFAULT_START,
    prod_function: str = "leontief",
    inventory_days: float | None = INPUT_SHOCK_DEFAULT_INVENTORY_DAYS,
    shock_spec: InputAvailabilityShockSpec | None = None,
    closure_modes: list[str] | None = None,
    save_path: str | None = None,
    n_simulations: int = MC_PLOT_SIMULATIONS,
) -> dict[str, dict[str, Any]]:
    """Run the input-availability shock for each household closure mode with MC bands."""
    spec = _resolve_input_availability_shock_spec(
        reduction_pct, duration, start, inventory_days, input_sector_label, shock_spec
    )
    modes = _resolve_household_closure_modes(closure_modes)
    results_data: dict[str, dict[str, Any]] = {}
    sector_label = spec.input_sector_label

    n_modes = len(modes)
    for idx, closure_mode in enumerate(modes, 1):
        print(f"  [{idx}/{n_modes}] closure_mode={closure_mode} ...", flush=True)
        scenario_run, baseline_run = run_input_availability_shock_scenario(
            prod_function=prod_function,
            shock_spec=InputAvailabilityShockSpec(
                reduction_pct=spec.reduction_pct,
                duration=spec.duration,
                start=spec.start,
                inventory_days=spec.inventory_days,
                input_sector_label=sector_label,
                tier=spec.tier,
            ),
            household_closure_mode=closure_mode,
        )
        if sector_label is None:
            sector_label = key_supplier_sector_label(baseline_run.model)

        mc = MonteCarloUncertaintyAnalysis(baseline_run.model, n_simulations=n_simulations)
        mc.run_uncertainty_analysis(
            shock_scenario="input_availability",
            shock_params={
                "input_sector_label": sector_label,
                "reduction_pct":      spec.reduction_pct,
                "duration":           spec.duration,
                "start":              spec.start,
            },
            seed=42,
        )
        results_data[closure_mode] = {
            "scenario_results": scenario_run.results,
            "baseline_results": baseline_run.results,
            "uncertainty_data": mc.get_uncertainty_data_for_plotting(free_raw=True),
            "time_frequency":   scenario_run.model.time_frequency,
        }
        del mc
        print(f"       done", flush=True)

    plot_household_closure_comparison(results_data, save_path=save_path)
    return results_data


# Production-function comparison runners
def run_consumption_shock_all_prod_functions(
    intensity: float = 0.2,
    duration: int = 3,
    start: int = 2,
    save_path: str | None = None,
    n_simulations: int = MC_PLOT_SIMULATIONS,
    household_closure_mode: str = "return_to_base",
) -> None:
    """Run the consumption shock for all production functions and plot together."""
    production_functions = ["leontief", "leontief.adapted", "linear", "ces", "klems"]
    results_data: dict[str, dict[str, Any]] = {}
    time_frequency = None

    n_funcs = len(production_functions)
    for idx, prod_func in enumerate(production_functions, 1):
        print(f"  [{idx}/{n_funcs}] prod_function={prod_func} ...", flush=True)
        try:
            scenario_run, baseline_run = run_consumption_shock_scenario(
                intensity=intensity,
                duration=duration,
                start=start,
                prod_function=prod_func,
                household_closure_mode=household_closure_mode,
            )
            if time_frequency is None:
                time_frequency = scenario_run.model.time_frequency

            mc = MonteCarloUncertaintyAnalysis(baseline_run.model, n_simulations=n_simulations)
            mc.run_uncertainty_analysis(
                shock_scenario="consumption",
                shock_params={"intensity": intensity, "duration": duration, "start": start},
                seed=42,
            )
            results_data[prod_func] = {
                "scenario_results": scenario_run.results,
                "baseline_results": baseline_run.results,
                "uncertainty_data": mc.get_uncertainty_data_for_plotting(free_raw=True),
                "time_frequency":   scenario_run.model.time_frequency,
            }
            del mc
            print(f"       done", flush=True)
        except Exception as e:
            print(f"  ERROR: prod_function={prod_func} failed: {e}\n{traceback.format_exc()}",
                  flush=True)
            logger.warning("Failed for production function %s: %s", prod_func, e)

    plot_prod_functions_comparison(results_data, time_frequency or "daily", save_path=save_path)


def run_input_availability_shock_all_prod_functions(
    input_sector_label: str | None = None,
    reduction_pct: float = INPUT_SHOCK_STRESS_REDUCTION_PCT,
    duration: int = INPUT_SHOCK_STRESS_DURATION,
    start: int = INPUT_SHOCK_STRESS_START,
    inventory_days: float | None = INPUT_SHOCK_STRESS_INVENTORY_DAYS,
    save_path: str | None = None,
    n_simulations: int = MC_PLOT_SIMULATIONS,
    shock_spec: InputAvailabilityShockSpec | None = None,
    household_closure_mode: str = "return_to_base",
) -> None:
    """Run the input-availability shock for all production functions and plot together, using
    the stress-tier spec by default so that substitution differences are visible."""
    spec = _resolve_input_availability_shock_spec(
        reduction_pct,
        duration,
        start,
        inventory_days,
        input_sector_label,
        shock_spec if shock_spec is not None else INPUT_AVAILABILITY_STRESS_SHOCK_SPEC,
    )
    production_functions = ["leontief", "leontief.adapted", "linear", "ces", "klems"]
    results_data: dict[str, dict[str, Any]] = {}
    time_frequency = None
    sector_label   = spec.input_sector_label

    n_funcs = len(production_functions)
    for idx, prod_func in enumerate(production_functions, 1):
        print(f"  [{idx}/{n_funcs}] prod_function={prod_func} ...", flush=True)
        try:
            scenario_run, baseline_run = run_input_availability_shock_scenario(
                prod_function=prod_func,
                shock_spec=InputAvailabilityShockSpec(
                    reduction_pct=spec.reduction_pct,
                    duration=spec.duration,
                    start=spec.start,
                    inventory_days=spec.inventory_days,
                    input_sector_label=sector_label,
                    tier=spec.tier,
                ),
                household_closure_mode=household_closure_mode,
            )
            if sector_label is None:
                sector_label = key_supplier_sector_label(baseline_run.model)
            if time_frequency is None:
                time_frequency = scenario_run.model.time_frequency

            mc = MonteCarloUncertaintyAnalysis(baseline_run.model, n_simulations=n_simulations)
            mc.run_uncertainty_analysis(
                shock_scenario="input_availability",
                shock_params={
                    "input_sector_label": sector_label,
                    "reduction_pct":      spec.reduction_pct,
                    "duration":           spec.duration,
                    "start":              spec.start,
                },
                seed=42,
            )
            results_data[prod_func] = {
                "scenario_results": scenario_run.results,
                "baseline_results": baseline_run.results,
                "uncertainty_data": mc.get_uncertainty_data_for_plotting(free_raw=True),
                "time_frequency":   scenario_run.model.time_frequency,
            }
            del mc
            print(f"       done", flush=True)
        except Exception as e:
            print(f"  ERROR: prod_function={prod_func} failed: {e}\n{traceback.format_exc()}",
                  flush=True)
            logger.warning("Failed for production function %s: %s", prod_func, e)

    plot_prod_functions_comparison(results_data, time_frequency or "daily", save_path=save_path)


# Sensitivity panel
def run_input_availability_sensitivity_panel(
    input_sector_label: str | None = None,
    prod_function: str = "leontief",
    save_path: str | None = None,
) -> None:
    """Run a sensitivity panel, moderate vs. stress shock crossed with low vs. high inventory."""
    case_specs = [
        {
            "label":          "Stress Test (50%, 1 Day)",
            "colour":         "#2ca02c",            "linestyle":      "-",
            "reduction_pct":  INPUT_SHOCK_STRESS_REDUCTION_PCT,
            "inventory_days": INPUT_SHOCK_STRESS_INVENTORY_DAYS,
        },
        {
            "label":          "Moderate Shock (30%, 1 Day)",
            "colour":         "#1f77b4",            "linestyle":      "--",
            "reduction_pct":  INPUT_SHOCK_DEFAULT_REDUCTION_PCT,
            "inventory_days": INPUT_SHOCK_STRESS_INVENTORY_DAYS,
        },
        {
            "label":          "Stress + Buffer (50%, 5 Days)",
            "colour":         "#ff7f0e",            "linestyle":      "-.",
            "reduction_pct":  INPUT_SHOCK_STRESS_REDUCTION_PCT,
            "inventory_days": INPUT_SHOCK_DEFAULT_INVENTORY_DAYS,
        },
        {
            "label":          "Moderate + Buffer (30%, 5 Days)",
            "colour":         "#d62728",            "linestyle":      ":",
            "reduction_pct":  INPUT_SHOCK_DEFAULT_REDUCTION_PCT,
            "inventory_days": INPUT_SHOCK_DEFAULT_INVENTORY_DAYS,
        },
    ]

    cases_data   = []
    time         = None
    time_frequency = None

    for spec in case_specs:
        scenario_run, baseline_run = run_input_availability_shock_scenario(
            input_sector_label=input_sector_label,
            reduction_pct=spec["reduction_pct"],
            duration=INPUT_SHOCK_DEFAULT_DURATION,
            start=INPUT_SHOCK_DEFAULT_START,
            prod_function=prod_function,
            inventory_days=spec["inventory_days"],
        )
        if time is None:
            time           = np.arange(scenario_run.model.TT)
            time_frequency = scenario_run.model.time_frequency
        current_output  = np.sum(scenario_run.results["gross_output"], axis=0)
        baseline_output = np.sum(baseline_run.results["gross_output"], axis=0)
        output_change   = ((current_output / baseline_output) - 1) * 100
        cases_data.append(
            {
                "label":         spec["label"],
                "colour":        spec["colour"],
                "linestyle":     spec["linestyle"],
                "output_change": output_change,
            }
        )

    if time is not None:
        plot_sensitivity_panel(
            cases_data,
            time,
            time_frequency or "daily",
            save_path=save_path,
        )
