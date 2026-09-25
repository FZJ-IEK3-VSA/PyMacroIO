"""
Standalone plotting functions holding all matplotlib logic, with no internal-module imports.
The model class exposes thin wrappers delegating here, so existing call-sites work unchanged.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


# Uncertainty bands (building block used by most higher-level plots)
def plot_uncertainty_bands(
    axes,
    time: np.ndarray,
    uncertainty_data: dict,
    baseline_results: dict | None,
    current_results: dict,
    start_period: int = 0,
    colour: str | None = None,
) -> None:
    """Draw 5-95% and 25-75% MC uncertainty bands (parameter uncertainty only) onto axes, as %
    change from baseline centred on current_results when baseline_results is provided."""
    output_data: dict = {}
    if "gross_output" in uncertainty_data:
        gd = uncertainty_data["gross_output"]
        if "error" not in gd and isinstance(gd, dict) and "mean" in gd:
            output_data = {
                "mean": np.sum(gd["mean"], axis=0),
                "q05":  np.sum(gd["q05"],  axis=0),
                "q25":  np.sum(gd["q25"],  axis=0),
                "q75":  np.sum(gd["q75"],  axis=0),
                "q95":  np.sum(gd["q95"],  axis=0),
            }
    elif "gdp" in uncertainty_data:
        output_data = uncertainty_data["gdp"]

    band_colour = colour if colour is not None else "#2ca02c"
    time_slice = slice(start_period, None) if start_period > 0 else slice(None)

    data = output_data
    if not data or "error" in data:
        return

    if baseline_results is not None:
        baseline_values = np.sum(baseline_results["gross_output"], axis=0)[time_slice]
        current_values  = np.sum(current_results["gross_output"],  axis=0)[time_slice]
        mc_mean_values  = data.get("mean", np.zeros_like(baseline_values))[time_slice]

        current_pct = ((current_values / baseline_values) - 1) * 100
        mc_mean_pct = ((mc_mean_values / baseline_values) - 1) * 100
        offset      = current_pct - mc_mean_pct

        q05 = ((data["q05"][time_slice] / baseline_values) - 1) * 100 + offset
        q25 = ((data["q25"][time_slice] / baseline_values) - 1) * 100 + offset
        q75 = ((data["q75"][time_slice] / baseline_values) - 1) * 100 + offset
        q95 = ((data["q95"][time_slice] / baseline_values) - 1) * 100 + offset
    else:
        current_values = np.sum(current_results["gross_output"], axis=0)[time_slice]
        mc_mean_values = data["mean"][time_slice]
        offset = current_values - mc_mean_values
        q05 = data["q05"][time_slice] + offset
        q25 = data["q25"][time_slice] + offset
        q75 = data["q75"][time_slice] + offset
        q95 = data["q95"][time_slice] + offset

    axes.fill_between(time, q05, q95, alpha=0.12, color=band_colour, label="_nolegend_")
    axes.fill_between(time, q25, q75, alpha=0.20, color=band_colour, label="_nolegend_")


# Single-path total-output plot
def plot_results(
    model,
    current_results: dict,
    baseline_results: dict | None = None,
    title_suffix: str = "",
    save_path: str | None = None,
    uncertainty_data: dict | None = None,
) -> None:
    """Plot aggregate gross output, or % change from baseline when baseline_results is provided."""
    fig, axes = plt.subplots(1, 1, figsize=(7.2, 8))
    if title_suffix:
        fig.suptitle(title_suffix, fontsize=14)

    time = np.arange(model.TT)
    current_output = np.sum(current_results["gross_output"], axis=0)

    if baseline_results is not None:
        baseline_output = np.sum(baseline_results["gross_output"], axis=0)
        output_change = ((current_output / baseline_output) - 1) * 100
    else:
        output_change = current_output

    if uncertainty_data is not None:
        plot_uncertainty_bands(
            axes, time, uncertainty_data, baseline_results, current_results
        )

    axes.plot(time, output_change, color="#2ca02c", linewidth=2, label="Output")
    _apply_time_axis_labels(axes, model.time_frequency)
    axes.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    if time.size > 0:
        axes.set_xlim(time[0], time[-1])
    axes.set_ylabel(
        "Percentage Change from Baseline (%)" if baseline_results is not None
        else "Absolute Value"
    )
    axes.grid(True, alpha=0.3)
    axes.legend(loc="best")
    if baseline_results is not None:
        axes.axhline(y=0, color="black", linestyle="--", alpha=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# Multi-panel regional plot
def plot_regional_results(
    model,
    current_results: dict,
    baseline_results: dict | None = None,
    title_suffix: str = "",
    save_path: str | None = None,
) -> None:
    """Plot per-region and aggregate output in a grid of subplots, as % deviation with
    baseline_results and normalised to period-0=100 without."""
    R    = model.n_regions
    TT   = model.TT
    time = np.arange(TT)

    # Derived series
    x_full = current_results["gross_output"]            # (N, TT)
    x_r    = np.array([
        x_full[model.region_sector_indices[r], :].sum(axis=0) for r in range(R)
    ])                                                   # (R, TT)
    x_agg  = x_full.sum(axis=0)                         # (TT,)
    gdp_r  = current_results["gdp_regional"]            # (R, TT)
    gdp    = current_results["gdp"]                     # (TT,)
    tb_r   = current_results.get("trade_balance", None) # (R, TT) or None

    if baseline_results is not None:
        x_full_b = baseline_results["gross_output"]
        x_r_b    = np.array([
            x_full_b[model.region_sector_indices[r], :].sum(axis=0) for r in range(R)
        ])
        x_agg_b = x_full_b.sum(axis=0)
        gdp_r_b = baseline_results["gdp_regional"]
        gdp_b   = baseline_results["gdp"]

        def pct(cur, ref):
            mask = ref != 0
            return np.where(mask,
                            (np.divide(cur, ref, where=mask,
                                       out=np.ones_like(cur, dtype=float)) - 1.0) * 100.0,
                            0.0)

        x_r_plot   = pct(x_r,   x_r_b)
        x_agg_plot = pct(x_agg, x_agg_b)
        gdp_r_plot = pct(gdp_r, gdp_r_b)
        gdp_plot   = pct(gdp,   gdp_b)
        ylabel_str  = "Change from baseline (%)"
        hline_val   = 0.0
        hline_label = "Baseline (0 %)"
    else:
        def norm100(arr, denom):
            with np.errstate(divide="ignore", invalid="ignore"):
                d = np.where(denom != 0, denom, 1.0)
                return arr / d * 100.0

        x_r_plot   = norm100(x_r,   x_r[:, [0]])
        x_agg_plot = norm100(x_agg, x_agg[0])
        gdp_r_plot = norm100(gdp_r, gdp_r[:, [0]])
        gdp_plot   = norm100(gdp,   gdp[0])
        ylabel_str  = "Index (period 0 = 100)"
        hline_val   = 100.0
        hline_label = "Base year (100)"

    # Layout
    n_panels = R + 1
    ncols    = min(2, n_panels)
    nrows    = int(np.ceil(n_panels / ncols))
    fig, axes_grid = plt.subplots(
        nrows, ncols, figsize=(7.0 * ncols, 4.0 * nrows), squeeze=False
    )
    axes_flat = axes_grid.flatten()

    colours_gdp    = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd",
                      "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]
    colours_output = ["#aec7e8", "#ffbb78", "#98df8a", "#c5b0d5",
                      "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d"]
    xlabel = _time_axis_label(model.time_frequency)

    # Regional panels
    for r in range(R):
        ax  = axes_flat[r]
        lbl = model.region_labels[r]
        ci  = r % len(colours_gdp)

        ax.plot(time, gdp_r_plot[r], color=colours_gdp[ci], lw=2,
                label=f"GDP - {lbl}")
        ax.plot(time, x_r_plot[r],  color=colours_output[ci], lw=1.5,
                linestyle="--", label=f"Gross output - {lbl}")

        if R > 1 and tb_r is not None:
            ax2 = ax.twinx()
            ax2.plot(time, tb_r[r], color="#ff7f0e", lw=1.2,
                     linestyle=":", alpha=0.8, label="Trade balance (RHS)")
            ax2.axhline(0, color="#ff7f0e", lw=0.5, linestyle=":", alpha=0.4)
            ax2.set_ylabel("Trade balance (base-year units)", fontsize=8,
                           color="#ff7f0e")
            ax2.tick_params(axis="y", labelcolor="#ff7f0e", labelsize=7)
            lines2, labels2 = ax2.get_legend_handles_labels()
        else:
            lines2, labels2 = [], []

        ax.axhline(hline_val, color="black", lw=0.8, linestyle="--",
                   alpha=0.5, label=hline_label)
        ax.set_title(f"Region: {lbl}", fontsize=11)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel_str, fontsize=9)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.set_xlim(time[0], time[-1])
        ax.grid(True, alpha=0.3)
        lines1, labels1 = ax.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="best")

    # Aggregate panel
    ax_agg = axes_flat[R]
    ax_agg.plot(time, gdp_plot,   color="#1f77b4", lw=2,   label="GDP (aggregate)")
    ax_agg.plot(time, x_agg_plot, color="#aec7e8", lw=1.5, linestyle="--",
                label="Gross output (aggregate)")
    ax_agg.axhline(hline_val, color="black", lw=0.8, linestyle="--",
                   alpha=0.5, label=hline_label)
    ax_agg.set_title("Aggregate (all regions)", fontsize=11)
    ax_agg.set_xlabel(xlabel, fontsize=9)
    ax_agg.set_ylabel(ylabel_str, fontsize=9)
    ax_agg.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax_agg.set_xlim(time[0], time[-1])
    ax_agg.grid(True, alpha=0.3)
    ax_agg.legend(fontsize=8, loc="best")

    for idx in range(n_panels, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    suptitle = "Regional Output" + (f" - {title_suffix}" if title_suffix else "")
    fig.suptitle(suptitle, fontsize=13)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# Household-closure-mode comparison plot
def plot_household_closure_comparison(
    results_data: dict[str, dict[str, Any]],
    save_path: str | None = None,
) -> None:
    """Plot scenario paths and MC bands for each household closure mode."""
    if not results_data:
        logger.error("No closure-mode results to plot")
        return

    fig, axes = plt.subplots(1, 1, figsize=(7.2, 8))
    colours    = {"return_to_base": "#1f77b4", "scarred": "#d62728", "frozen": "#2ca02c"}
    linestyles = {"return_to_base": "-",       "scarred": "--",      "frozen": ":"}

    first_data = next(iter(results_data.values()))
    # Derive TT and time_frequency from results - no model object needed
    time           = np.arange(len(first_data["scenario_results"]["gdp"]))
    time_frequency = first_data.get("time_frequency", "daily")

    for closure_mode, data in results_data.items():
        colour    = colours.get(closure_mode, "#000000")
        linestyle = linestyles.get(closure_mode, "-")
        if data["uncertainty_data"] is not None:
            plot_uncertainty_bands(
                axes, time,
                data["uncertainty_data"],
                data["baseline_results"],
                data["scenario_results"],
                colour=colour,
            )
        current_output  = np.sum(data["scenario_results"]["gross_output"], axis=0)
        baseline_output = np.sum(data["baseline_results"]["gross_output"],  axis=0)
        output_change   = ((current_output / baseline_output) - 1) * 100
        axes.plot(
            time, output_change,
            color=colour, linewidth=2.5, linestyle=linestyle,
            label=_closure_mode_label(closure_mode),
        )

    _apply_time_axis_labels(axes, time_frequency)
    axes.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    if time.size > 0:
        axes.set_xlim(time[0], time[-1])
    axes.set_ylabel("Percentage Change from Baseline (%)")
    axes.grid(True, alpha=0.3)
    axes.legend(loc="best")
    axes.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info("Figure saved: %s", save_path)
    plt.show()


# Production-function comparison plot
PROD_FUNC_COLOURS = {
    "leontief":         "#2ca02c",
    "leontief.adapted": "#1f77b4",
    "linear":           "#ff7f0e",
    "ces":              "#d62728",
}

PROD_FUNC_LINESTYLES = {
    "leontief":         "-",
    "leontief.adapted": "--",
    "linear":           "-.",
    "ces":              ":",
}


def plot_prod_functions_comparison(
    results_data: dict[str, dict[str, Any]],
    time_frequency: str,
    save_path: str | None = None,
) -> None:
    """Plot scenario paths and MC bands for each production function variant."""
    if not results_data:
        logger.error("No results to plot")
        return

    fig, axes = plt.subplots(1, 1, figsize=(7.2, 8))

    first_data = next(iter(results_data.values()))
    # Derive TT and time_frequency from results - no model object needed
    time = np.arange(len(first_data["scenario_results"]["gdp"]))

    for prod_func, data in results_data.items():
        colour    = PROD_FUNC_COLOURS.get(prod_func, "#000000")
        linestyle = PROD_FUNC_LINESTYLES.get(prod_func, "-")

        if data["uncertainty_data"] is not None:
            plot_uncertainty_bands(
                axes, time,
                data["uncertainty_data"],
                data["baseline_results"],
                data["scenario_results"],
                colour=colour,
            )

        current_output  = np.sum(data["scenario_results"]["gross_output"], axis=0)
        baseline_output = np.sum(data["baseline_results"]["gross_output"],  axis=0)
        output_change   = ((current_output / baseline_output) - 1) * 100
        axes.plot(
            time, output_change,
            color=colour, linewidth=2.5, linestyle=linestyle,
            label=prod_func.replace(".", " ").title(),
        )

    _apply_time_axis_labels(axes, time_frequency)
    axes.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    if time.size > 0:
        axes.set_xlim(time[0], time[-1])
    axes.set_ylabel("Percentage Change from Baseline (%)")
    axes.grid(True, alpha=0.3)
    axes.legend(loc="best")
    axes.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info("Figure saved: %s", save_path)
    plt.show()


# Sensitivity panel
def plot_sensitivity_panel(
    cases_data: list,
    time: "np.ndarray",
    time_frequency: str,
    save_path: "str | None" = None,
) -> None:
    """Plot a multi-case input-availability sensitivity panel."""
    fig, axes = plt.subplots(1, 1, figsize=(7.2, 8))

    for case in cases_data:
        axes.plot(
            time,
            case["output_change"],
            color=case["colour"],
            linewidth=2.5,
            linestyle=case["linestyle"],
            label=case["label"],
        )

    _apply_time_axis_labels(axes, time_frequency)
    axes.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    if time.size > 0:
        axes.set_xlim(time[0], time[-1])
    axes.set_ylabel("Percentage Change from Baseline (%)")
    axes.grid(True, alpha=0.3)
    axes.legend(loc="best")
    axes.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info("Figure saved: %s", save_path)
    plt.show()


# Internal helpers
def _time_axis_label(time_frequency: str) -> str:
    if time_frequency == "daily":
        return "Time Period (Days)"
    if time_frequency == "quarterly":
        return "Time Period (Quarters)"
    return "Time Period"


def _apply_time_axis_labels(axes, time_frequency: str) -> None:
    axes.set_xlabel(_time_axis_label(time_frequency))


def _closure_mode_label(closure_mode: str) -> str:
    return {
        "return_to_base": "Return To Base",
        "scarred":        "Permanent Scarring",
        "frozen":         "Frozen (Base Year)",
    }.get(closure_mode, closure_mode)
