"""
Scenario and run-result data structures, and ScenarioManager.
These objects describe a run, hold its results, and coordinate baseline caching and comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from collections.abc import Callable

import numpy as np

from .config import (
    ModelConfig,
    INPUT_SHOCK_DEFAULT_REDUCTION_PCT,
    INPUT_SHOCK_DEFAULT_DURATION,
    INPUT_SHOCK_DEFAULT_START,
    INPUT_SHOCK_DEFAULT_INVENTORY_DAYS,
    INPUT_SHOCK_STRESS_REDUCTION_PCT,
    INPUT_SHOCK_STRESS_DURATION,
    INPUT_SHOCK_STRESS_START,
    INPUT_SHOCK_STRESS_INVENTORY_DAYS,
)


# Scenario and result dataclasses
@dataclass
class Scenario:
    """A named scenario, a config plus ordered shock callables applied to the model before a run."""

    name: str
    description: str
    config: ModelConfig
    shocks: list[Callable] = field(default_factory=list)


@dataclass
class ScenarioRunResult:
    """Result of a single scenario run."""

    scenario: Scenario
    model: Any          # InputOutputModel - typed as Any to avoid circular import
    results: dict[str, Any]


# Shock specification dataclasses
@dataclass(frozen=True)
class ConsumptionShockSpec:
    """A consumption-shock specification for examples, sensitivity runs, or stress tests."""

    intensity: float = 0.2
    duration: int = 3
    start: int = 2
    tier: str = "example"


@dataclass(frozen=True)
class InputAvailabilityShockSpec:
    """An input-availability shock specification for examples, sensitivity runs, or stress tests."""

    reduction_pct: float = INPUT_SHOCK_DEFAULT_REDUCTION_PCT
    duration: int = INPUT_SHOCK_DEFAULT_DURATION
    start: int = INPUT_SHOCK_DEFAULT_START
    inventory_days: float | None = INPUT_SHOCK_DEFAULT_INVENTORY_DAYS
    input_sector_label: str | None = None
    tier: str = "example"


# Pre-built canonical specs
CONSUMPTION_EXAMPLE_SHOCK_SPEC = ConsumptionShockSpec()
INPUT_AVAILABILITY_EXAMPLE_SHOCK_SPEC = InputAvailabilityShockSpec()
INPUT_AVAILABILITY_STRESS_SHOCK_SPEC = InputAvailabilityShockSpec(
    reduction_pct=INPUT_SHOCK_STRESS_REDUCTION_PCT,
    duration=INPUT_SHOCK_STRESS_DURATION,
    start=INPUT_SHOCK_STRESS_START,
    inventory_days=INPUT_SHOCK_STRESS_INVENTORY_DAYS,
    tier="stress",
)


# Scenario manager
class ScenarioManager:
    """Coordinate baseline and scenario runs, caching the baseline result for comparisons."""

    def __init__(self, base_config: ModelConfig):
        self.base_config = base_config.clone()
        self._baseline_cache: ScenarioRunResult | None = None

    def _instantiate_model(self, config: ModelConfig):
        """Create and return a model instance from the given config."""
        # Import here to avoid a top-level circular dependency
        from .model import InputOutputModel
        return InputOutputModel(
            n_periods=config.n_periods,
            time_frequency=config.time_frequency,
            config=config,
        )

    def run_baseline(self, force: bool = False) -> ScenarioRunResult:
        """Run (or return cached) the baseline scenario; force bypasses the cache."""
        if force or self._baseline_cache is None:
            baseline_config = self.base_config.clone()
            model = self._instantiate_model(baseline_config)
            results = model.run_model()
            baseline_scenario = Scenario(
                name="baseline",
                description="Baseline scenario",
                config=baseline_config,
                shocks=[],
            )
            self._baseline_cache = ScenarioRunResult(baseline_scenario, model, results)
        return self._baseline_cache

    def run_scenario(
        self, scenario: Scenario, use_cached_baseline: bool = True
    ) -> tuple[ScenarioRunResult, ScenarioRunResult | None]:
        """Apply the scenario's shocks, run the model, and return (scenario_run, baseline_run)."""
        baseline = self.run_baseline(force=False) if use_cached_baseline else None
        scenario_model = self._instantiate_model(scenario.config.clone())
        for shock in scenario.shocks:
            shock(scenario_model)
        scenario_results = scenario_model.run_model()
        scenario_run = ScenarioRunResult(scenario, scenario_model, scenario_results)
        return scenario_run, baseline

    @staticmethod
    def compare_regional_to_baseline(
        run: ScenarioRunResult, baseline: ScenarioRunResult
    ) -> dict[str, np.ndarray]:
        """Return per-region GDP and HH consumption % deviations from the baseline."""
        def safe_pct(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            mask = np.abs(b) > 1e-9
            return np.where(mask,
                            np.divide(a - b, np.abs(b), where=mask,
                                      out=np.zeros_like(a, dtype=float)) * 100,
                            0.0)

        return {
            "region_labels": list(getattr(baseline.model, "region_labels", [])),
            "gdp_pct": safe_pct(
                run.results["gdp_regional"],
                baseline.results["gdp_regional"],
            ),
            "consumption_by_hh_region_pct": safe_pct(
                run.results["consumption_by_hh_region"],
                baseline.results["consumption_by_hh_region"],
            ),
        }

    @staticmethod
    def compare_to_baseline(
        run: ScenarioRunResult, baseline: ScenarioRunResult
    ) -> dict[str, np.ndarray]:
        """Return % deviations of GDP and realised consumption from the baseline."""
        def safe_pct_change(current: np.ndarray, reference: np.ndarray) -> np.ndarray:
            pct = np.full_like(current, np.nan, dtype=np.float64)
            nonzero = reference != 0
            pct[nonzero] = (current[nonzero] / reference[nonzero] - 1) * 100
            pct[(~nonzero) & np.isclose(current, reference)] = 0.0
            return pct

        return {
            "gdp_pct": safe_pct_change(run.results["gdp"], baseline.results["gdp"]),
            "consumption_pct": safe_pct_change(
                np.sum(run.results["realised_consumption"], axis=0),
                np.sum(baseline.results["realised_consumption"], axis=0),
            ),
        }
