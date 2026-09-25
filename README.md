
<!-- logo:header:start -->
<p align="center">
  <a href="https://www.fz-juelich.de/en/ice/ice-2">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/FZJ-IEK3-VSA/README_assets/v.1.0.0/ICE2_Logos/JSA-Header-dark.svg">
      <img src="https://raw.githubusercontent.com/FZJ-IEK3-VSA/README_assets/v.1.0.0/ICE2_Logos/JSA-Header.svg" alt="Jülich Systems Analysis" height="80">
    </picture>
  </a>
</p>
<!-- logo:header:end -->

# Dynamic Disequilibrium Macroeconomic Model with Input-Output Structure

A Python implementation of a Dynamic Disequilibrium Macroeconomic Model with Input-Output Structure. It features multiple production rules, inventory dynamics, labour adjustment, multi-region support, endogenous macro closure rules, and two exemplary shock types.

## Overview

The model uses an input-output table as its structural backbone but is best understood as a dynamic disequilibrium macroeconomic model. Production is supply- and capacity-constrained, labour and inventories adjust with friction, and quantity rationing takes the place of price-mediated market clearing. It is discrete-time and sector-level. A multi-region structure with R ≥ 1 regions is supported. Production can be Leontief, adapted Leontief, linear, CES, or KLEMS. Household demand follows a Muellbauer or LES rule, whilst government spending and investment can follow endogenous macro closure rules or remain fixed at base-year levels. Full model equations and implementation details are in `docs/Mathematical_summary.pdf`.

## Requirements

Python 3.9+, NumPy, Matplotlib.

```bash
pip install .        # add -e ".[dev]" for an editable install with test dependencies
```

Installing is optional. `pip install -r requirements.txt` gets the dependencies alone. Either way, run from the project root, where the default data file (`data/example_data.pkl`) is found, and invoke scripts in module form.

```bash
python -m pyMacroIO             # baseline and example scenario figures
python -m examples.multiregion  # multi-region baselines and shocks
```

Example data are derived from a subset of EXIOBASE (see References, item 4).

## Quick Start

### Baseline run

```python
from pyMacroIO import ModelConfig, ScenarioManager, MonteCarloUncertaintyAnalysis

config = ModelConfig(
    n_periods=60,
    time_frequency="daily",
    prod_function="leontief.adapted",   # default; also "leontief", "linear", "ces", "klems"
)
manager = ScenarioManager(config)
baseline_run = manager.run_baseline(force=True)
```

### Consumption-shock scenario

```python
from pyMacroIO import run_consumption_shock_scenario, ScenarioManager

scenario_run, baseline_run = run_consumption_shock_scenario(
    intensity=0.2, duration=3, start=2,
)
comparison = ScenarioManager.compare_to_baseline(scenario_run, baseline_run)
# comparison["gdp_pct"], comparison["consumption_pct"]
```

### Input-availability shock

```python
from pyMacroIO import run_input_availability_shock_scenario

scenario_run, baseline_run = run_input_availability_shock_scenario(
    input_sector_label=None,   # uses key supplier (largest forward supply)
    reduction_pct=0.3,
    duration=3,
    start=2,
    inventory_days=5.0,
)
```

### Endogenous macro closures

```python
from pyMacroIO import ModelConfig, ScenarioManager

config = ModelConfig(
    n_periods=60,
    gov_income_elasticity=-0.3,        # counter-cyclical government spending
    investment_closure="keynesian",     # scales investment with lagged savings
)
manager = ScenarioManager(config)
baseline_run = manager.run_baseline(force=True)
```

## Key Features

**Core model**

- `InputOutputModel`: R ≥ 1 regions, production functions `leontief`, `leontief.adapted`, `linear`, `ces`, `klems`
- `ModelConfig`: all scenario-overridable parameters with validation
- Inventories: beginning-of-period stocks constrain production; end-of-period stocks rebuild from realised deliveries, orders carrying the signed gap to target damped by the input row's adjustment speed
- Labour: sector-specific hiring and firing speeds, disruption, and capacity bounds
- Household demand: Muellbauer rule or LES (`subsistence_shares`); closure modes `return_to_base`, `scarred`, `frozen`; LES calibration via `build_subsistence_shares_vector`
- Government spending: income-indexed with elasticity `gov_income_elasticity` (default 0.0 holds spending at the base-year level)
- Investment closure: `investment_closure="keynesian"` scales investment demand with lagged aggregate savings; default `"fixed"` holds it at base year
- Technical change: piecewise-constant productivity events

**Trade closures**

- Import flexibility (`import_flexibility`): per-sector fraction of an inventory shortfall coverable by external imports
- Sourced supplement (`row_supply_cap`): caps the share of RoW output diverted to source the import supplement
- Export pull (`export_pull`): absorbs capacity slack as additional external demand

**Scenarios and uncertainty**

- `ScenarioManager`: cached baseline, scenario runs with shock callables, comparison to baseline
- Shock helpers: `run_consumption_shock_scenario`, `run_input_availability_shock_scenario`, `run_consumption_shock_all_prod_functions`, `run_input_availability_shock_all_prod_functions`, household-closure sensitivity variants, and `run_input_availability_sensitivity_panel`
- `MonteCarloUncertaintyAnalysis`: parameter sampling, ensemble runs, mean and quantile metrics for GDP, consumption, and gross output

## Example data

The default data file is `data/example_data.pkl` (single region) or `data/example_data_2region.pkl` (two-region example). The required entries are `sector_labels`, `Z0` (N×N intermediate flows), `cons_vec`, `gov_vec`, `inv_vec`, `invnt_vec`, `exp_vec` (final-demand vectors), `l0`, `cap0`, `tax0`, `imp0` (value-added components), `consumer_taxes_total`, and `fd_imports_totals`. Multi-region datasets additionally include `region_map` and `region_labels`. Row and value-added identities are checked at load. Full definitions are in `docs/Mathematical_summary.pdf`. Example data are derived from EXIOBASE (see References, item 4).

## Documentation

`docs/Mathematical_summary.pdf` covers all main model equations.

## Licence

The licence for this software is described in the LICENSE file. The example data licence is different; see [4] and the [EXIOBASE licence file](https://zenodo.org/records/15689391/preview/LICENSE.txt) for the authoritative conditions.

### How to cite

Cite this implementation as [0], building on [1–3]. Cite [4] when the included example data (a subset of EXIOBASE) are used, and adhere to its licence terms.

## About Us

We are the <a href="https://www.fz-juelich.de/en/ice/ice-2">Institute of Climate and Energy Systems – Jülich Systems Analysis (ICE-2)</a> at the <a href="https://www.fz-juelich.de/en"> Forschungszentrum Jülich</a>.
Our work focuses on independent, interdisciplinary research in energy, bioeconomy, infrastructure, and sustainability. We support a just, greenhouse gas–neutral transformation through open models and policy-relevant science.

## Acknowledgments

This work was initially supported by the Helmholtz Association under the Joint Initiative ["Energy System 2050   A Contribution of the Research Field Energy"](https://www.helmholtz.de/en/research/energy/energy_system_2050/).

## References

0. Ross, A. G. (2026). Dynamic Disequilibrium Macroeconomic Model with Input-Output Structure. [10.5281/zenodo.21884596](https://doi.org/10.5281/zenodo.21884597)

1. Pichler, A., Pangallo, M., del Rio-Chanona, R. M., Lafond, F., & Farmer, J. D. (2022). Forecasting the propagation of pandemic shocks with a dynamic input–output model. *Journal of Economic Dynamics and Control*, 144, 104527. <https://doi.org/10.1016/j.jedc.2022.104527>

2. Ross, A. G., McGregor, P. G., & Swales, J. K. (2024). Labour market dynamics in the era of technological advancements: The system-wide impacts of labour augmenting technological change. *Technology in Society*, 77, 102539. <https://doi.org/10.1016/j.techsoc.2024.102539>

3. Raseta, M., Ross, A. G., & Voegele, S. (2025). Macro-level implications of the energy system transition to net-zero carbon emissions: Identifying quick wins amid short-term constraints. *Economic Analysis and Policy*, 85, 1065–1078. <https://doi.org/10.1016/j.eap.2025.01.011>

4. Stadler, K., Wood, R., Bulavskaya, T., Södersten, C.-J., Simas, M., Schmidt, S., Usubiaga, A., Acosta-Fernández, J., Kuenen, J., Bruckner, M., Giljum, S., Lutter, S., Merciai, S., Schmidt, J. H., Theurl, M. C., Plutzar, C., Kastner, T., Eisenmenger, N., Erb, K.-H., … Tukker, A. (2025). EXIOBASE 3 (3.9.6) [Data set]. Zenodo. <https://doi.org/10.5281/zenodo.15689391>

5. Miller, R. E., & Blair, P. D. (2009). Input-output analysis: foundations and extensions. Cambridge university press.

<p align="left">
  <!-- logo:helmholtz:start -->
  <a href="https://www.helmholtz.de/en/">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/FZJ-IEK3-VSA/README_assets/v.1.0.0/Helmholtz_Logos/Helmholtz-Logo-White-RGB.svg">
      <img src="https://raw.githubusercontent.com/FZJ-IEK3-VSA/README_assets/v.1.0.0/Helmholtz_Logos/Helmholtz-Logo-Dark-Blue-RGB.svg" alt="Helmholtz Association" width="200">
    </picture>
  </a>
  <!-- logo:helmholtz:end -->
</p>
