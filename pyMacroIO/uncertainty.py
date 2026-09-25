"""
Monte Carlo uncertainty analysis.
MonteCarloUncertaintyAnalysis draws parameters, runs the model per draw, and returns quantiles.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .model import InputOutputModel
from .constants import (
    GAMMA_HIRE_MIN,
    GAMMA_HIRE_MAX,
    GAMMA_FIRE_MIN,
    GAMMA_FIRE_MAX,
    TAU_MIN,
    TAU_MAX,
)

logger = logging.getLogger(__name__)


class MonteCarloUncertaintyAnalysis:
    """Monte Carlo uncertainty analysis whose sampling bounds match the model's validated ranges,
    bands reflecting parameter uncertainty only, which may persist after the shock ends."""

    def __init__(self, base_model: InputOutputModel, n_simulations: int = 1000):
        self.base_model    = base_model
        self.n_simulations = n_simulations
        self.results: dict[str, Any] = {}

    def define_parameter_distributions(self) -> dict[str, Any]:
        """Return parameter names, distribution types, bounds, and descriptions, savings_rate and
        benefits being sampled per region and the rest applying to all regions."""
        return {
            "rho1": {
                "distribution": "uniform",
                "bounds": (0.2, 0.9),
                "description": "Consumption persistence (baseline: 0.6)",
            },
            "gamma_hire": {
                "distribution": "uniform",
                "bounds": (GAMMA_HIRE_MIN, GAMMA_HIRE_MAX),
                "description": "Labour hiring speed (baseline: ~0.30)",
            },
            "gamma_fire": {
                "distribution": "uniform",
                "bounds": (GAMMA_FIRE_MIN, GAMMA_FIRE_MAX),
                "description": "Labour firing speed (baseline: ~0.40)",
            },
            "tau": {
                "distribution": "uniform",
                "bounds": (TAU_MIN, TAU_MAX),
                "description": "Inventory adjustment speed (baseline: ~2.17)",
            },
            "savings_rate": {
                "distribution": "uniform",
                "bounds": (0.01, 0.20),
                "description": "Household savings rate - sampled per region (baseline: 0.05)",
            },
            "benefits": {
                "distribution": "uniform",
                "bounds": (0.0, 0.3),
                "description": "Unemployment benefit replacement rate - per region (baseline: 0.1)",
            },
        }

    def sample_parameters(self, seed: int | None = None) -> dict[str, np.ndarray]:
        """Sample from the defined distributions, drawing global parameters once per simulation and
        per-region parameters independently per region with shape (R, n_simulations)."""
        if seed is not None:
            np.random.seed(seed)

        R = self.base_model.n_regions
        distributions  = self.define_parameter_distributions()
        sampled_params: dict[str, np.ndarray] = {}

        for param_name, param_info in distributions.items():
            if param_info["distribution"] != "uniform":
                continue
            lo, hi = param_info["bounds"]

            if param_name == "rho1":
                rho1 = np.random.uniform(lo, hi, size=self.n_simulations)
                sampled_params["rho1"] = rho1
                sampled_params["rho0"] = 1 - rho1
            elif param_name in ("gamma_fire", "gamma_hire", "tau"):
                sampled_params[param_name] = np.random.uniform(
                    lo, hi, size=(self.base_model.N, self.n_simulations)
                )
            elif param_name in ("savings_rate", "benefits"):
                sampled_params[param_name] = np.random.uniform(
                    lo, hi, size=(R, self.n_simulations)
                )

        return sampled_params

    def run_uncertainty_analysis(
        self,
        shock_scenario: str = "baseline",
        shock_params: dict | None = None,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Sample parameters and run the model per draw under shock_scenario ("baseline",
        "consumption", or "input_availability"), failed runs NaN-filled and excluded."""
        sampled_params = self.sample_parameters(seed)
        R  = self.base_model.n_regions
        TT = self.base_model.TT
        self.results = {
            "gdp":          np.zeros((TT, self.n_simulations)),
            "consumption":  np.zeros((TT, self.n_simulations)),
            "gross_output": np.zeros((TT, self.base_model.N, self.n_simulations)),
            "gdp_regional": np.zeros((R, TT, self.n_simulations)),
            "parameters":   sampled_params,
        }
        _print_every = max(1, self.n_simulations // 10)   # progress every ~10 %
        _n_errors    = 0
        print(f"    MC: 0/{self.n_simulations}", end="", flush=True)
        for i in range(self.n_simulations):
            model = self._create_model_with_parameters(sampled_params, i)
            if shock_scenario != "baseline":
                model = self._apply_shock(model, shock_scenario, shock_params)
            try:
                sim = model.run_model()
                self.results["gdp"][:, i]             = sim["gdp"]
                self.results["consumption"][:, i]     = np.sum(sim["realised_consumption"], axis=0)
                self.results["gross_output"][:, :, i] = sim["gross_output"].T
                self.results["gdp_regional"][:, :, i] = sim["gdp_regional"]
            except Exception as e:
                _n_errors += 1
                logger.warning("MC simulation %d failed: %s", i, e)
                self.results["gdp"][:, i]             = np.nan
                self.results["consumption"][:, i]     = np.nan
                self.results["gross_output"][:, :, i] = np.nan
                self.results["gdp_regional"][:, :, i] = np.nan

            if (i + 1) % _print_every == 0 or (i + 1) == self.n_simulations:
                _err_str = f"  ({_n_errors} failed)" if _n_errors else ""
                print(f"\r    MC: {i+1}/{self.n_simulations}{_err_str}   ", end="", flush=True)

        print(flush=True)   # newline after progress line
        if _n_errors:
            print(f"    WARNING: {_n_errors}/{self.n_simulations} MC draws failed "
                  f"(NaN-filled). Check logging output for details.", flush=True)
        return self.results

    def _create_model_with_parameters(
        self, sampled_params: dict, simulation_idx: int
    ) -> InputOutputModel:
        """Create a model instance parameterised from the sampled values, sharing the base model's
        IO data dict by reference so the pkl is not re-read per draw."""
        model = InputOutputModel(
            n_periods=self.base_model.TT,
            time_frequency=self.base_model.time_frequency,
            config=self.base_model.config.clone(),
            _data_dict=self.base_model._raw_data,
        )
        # Copy A_essential from the base model when A is time-invariant.
        if not self.base_model.A_changes:
            model.A_essential = self.base_model.A_essential

        if "rho0" in sampled_params and "rho1" in sampled_params:
            model.rho0 = sampled_params["rho0"][simulation_idx]
            model.rho1 = sampled_params["rho1"][simulation_idx]
        if "gamma_hire" in sampled_params:
            model.gamma_hire = sampled_params["gamma_hire"][:, simulation_idx]
        if "gamma_fire" in sampled_params:
            model.gamma_fire = sampled_params["gamma_fire"][:, simulation_idx]
        if "tau" in sampled_params:
            model.tau = sampled_params["tau"][:, simulation_idx]
        if "savings_rate" in sampled_params:
            draws = sampled_params["savings_rate"]          # (R, n_sims)
            for r in range(model.n_regions):
                sr = model._validate_savings_rate_value(float(draws[r, simulation_idx]))
                model.savings_rate_r[r] = sr
                model._refresh_household_scalars_r(r)
        if "benefits" in sampled_params:
            draws = sampled_params["benefits"]              # (R, n_sims)
            for r in range(model.n_regions):
                model.benefits_r[r] = float(draws[r, simulation_idx])
            model.benefits = float(model.benefits_r[0])

        return model

    def _apply_shock(
        self,
        model: InputOutputModel,
        shock_scenario: str,
        shock_params: dict | None,
    ) -> InputOutputModel:
        """Apply a consumption or input-availability shock to the model in place."""
        if shock_scenario == "consumption" and shock_params:
            model.apply_consumption_shock(
                shock_params.get("start", 2),
                shock_params.get("duration", 3),
                shock_params.get("intensity", 0.2),
                region=None,
            )
        elif shock_scenario == "input_availability" and shock_params:
            label        = shock_params.get("input_sector_label")
            reduction_pct = shock_params.get("reduction_pct", 0.3)
            duration      = shock_params.get("duration", 3)
            start         = shock_params.get("start", 2)
            if label is not None:
                for t in range(start, min(start + duration, model.TT)):
                    model.apply_input_availability_shock(label, t, reduction_pct)
        return model

    def calculate_uncertainty_metrics(self) -> dict[str, Any]:
        """Return mean, std, and quantiles over runs with no NaN in any period."""
        metrics: dict[str, Any] = {}
        for variable in ("gdp", "consumption"):
            if variable not in self.results:
                continue
            data       = self.results[variable]
            valid_mask = ~np.any(np.isnan(data), axis=0)
            valid_data = data[:, valid_mask]
            if valid_data.shape[1] == 0:
                metrics[variable] = {"error": "No valid simulations"}
                continue
            metrics[variable] = {
                "mean":    np.mean(valid_data, axis=1),
                "std":     np.std(valid_data, axis=1),
                "q05":     np.percentile(valid_data,  5, axis=1),
                "q25":     np.percentile(valid_data, 25, axis=1),
                "q75":     np.percentile(valid_data, 75, axis=1),
                "q95":     np.percentile(valid_data, 95, axis=1),
                "n_valid": valid_data.shape[1],
            }

        if "gross_output" in self.results:
            data       = self.results["gross_output"]
            valid_mask = ~np.any(np.isnan(data), axis=(0, 1))
            if np.sum(valid_mask) > 0:
                valid_data = data[:, :, valid_mask]
                if valid_data.shape[2] > 0:
                    metrics["gross_output"] = {
                        "mean":    np.mean(valid_data, axis=2).T,
                        "std":     np.std(valid_data, axis=2).T,
                        "q05":     np.percentile(valid_data,  5, axis=2).T,
                        "q25":     np.percentile(valid_data, 25, axis=2).T,
                        "q75":     np.percentile(valid_data, 75, axis=2).T,
                        "q95":     np.percentile(valid_data, 95, axis=2).T,
                        "n_valid": valid_data.shape[2],
                    }

        return metrics

    def get_uncertainty_data_for_plotting(self, free_raw: bool = True) -> dict[str, Any]:
        """Return uncertainty metrics for plot_results, where free_raw deletes the raw arrays after
        the quantile summaries to bound peak memory in production-function loops."""
        metrics = self.calculate_uncertainty_metrics()
        if free_raw:
            for key in ("gdp", "consumption", "gross_output", "gdp_regional"):
                self.results.pop(key, None)
        return metrics
