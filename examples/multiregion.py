#!/usr/bin/env python3
"""
Multi-region example writing four baseline-replication and shock figures to figures/.

Baselines are indexed to period 0 = 100 and shocks shown as % change from baseline.
"""

from pathlib import Path

import numpy as np

from pyMacroIO import (
    ModelConfig,
    InputOutputModel,
    ENABLE_PLOTTING,
    key_supplier_sector_label,
)

figures_dir = Path("figures")
figures_dir.mkdir(parents=True, exist_ok=True)


# 1. R=1 baseline replication
print("Running R=1 baseline ...")
cfg1 = ModelConfig(n_periods=40, time_frequency="daily", prod_function="leontief.adapted")
m1   = InputOutputModel(config=cfg1)
r1   = m1.run_model(validate=True)

if ENABLE_PLOTTING:
    m1.plot_regional_results(
        r1,
        baseline_results=None,
        title_suffix="R=1 no-shock (index 100 = fixed point)",
        save_path=str(figures_dir / "baseline_1region.png"),
    )
    print("  -> figures/baseline_1region.png")

gdp_drift = np.abs(r1["gdp"] - r1["gdp"][0]).max()
print(f"  R=1 baseline: max GDP drift from t=0 = {gdp_drift:.6f}  (should be ~0)")


# 2. R=2 baseline replication
print("\nRunning R=2 baseline ...")
cfg2 = ModelConfig(
    n_periods=40,
    time_frequency="daily",
    prod_function="leontief.adapted",
    data_path="data/example_data_2region.pkl",
    n_regions=2,
)
m2 = InputOutputModel(config=cfg2)
r2 = m2.run_model(validate=True)

if ENABLE_PLOTTING:
    m2.plot_regional_results(
        r2,
        baseline_results=None,
        title_suffix="R=2 no-shock (index 100 = period-0 value)",
        save_path=str(figures_dir / "baseline_2region.png"),
    )
    print("  -> figures/baseline_2region.png")

# The 2-region test data uses a toy income/consumption split; non-zero GDP drift is expected.
agg_drift = np.abs(r2["gdp"] - r2["gdp"][0]).max()
print(f"  R=2 aggregate GDP drift = {agg_drift:.1f}  (non-zero: toy-data income/consumption mismatch)")
for r in range(m2.n_regions):
    drift  = np.abs(r2["gdp_regional"][r] - r2["gdp_regional"][r, 0]).max()
    steady = r2["gdp_regional"][r, -1]
    print(f"  Region {m2.region_labels[r]}: drift={drift:.0f}, steady-state GDP={steady:.0f}")


# 3. R=2 consumption shock - Region 0 only
print("\nRunning R=2 consumption shock (Region 0, intensity=0.25, duration=5) ...")
cfg2s = ModelConfig(
    n_periods=40,
    time_frequency="daily",
    prod_function="leontief.adapted",
    data_path="data/example_data_2region.pkl",
    n_regions=2,
)
m2s = InputOutputModel(config=cfg2s)

shock_start, shock_dur, shock_intensity = 2, 5, 0.25
m2s.epsilon_r[0, shock_start : shock_start + shock_dur] = shock_intensity

r2s = m2s.run_model(validate=True)

if ENABLE_PLOTTING:
    m2s.plot_regional_results(
        r2s,
        baseline_results=r2,
        title_suffix="R=2 consumption shock - Region 0 (intensity 25 %, 5 periods)",
        save_path=str(figures_dir / "consumption_shock_2region.png"),
    )
    print("  -> figures/consumption_shock_2region.png")

gdp_drop = (
    (r2s["gdp_regional"][:, shock_start + shock_dur - 1]
     / r2["gdp_regional"][:, shock_start + shock_dur - 1]) - 1
) * 100
for r in range(m2s.n_regions):
    print(f"  GDP impact at peak (region {m2s.region_labels[r]}): {gdp_drop[r]:+.2f} %")


# 4. R=2 input-availability shock - key supplier across full economy
print("\nRunning R=2 input-availability shock (key supplier, reduction=0.5, duration=5) ...")
cfg2i = ModelConfig(
    n_periods=40,
    time_frequency="daily",
    prod_function="leontief.adapted",
    data_path="data/example_data_2region.pkl",
    n_regions=2,
)
m2i = InputOutputModel(config=cfg2i)

key_label = key_supplier_sector_label(m2i)
for t in range(shock_start, shock_start + shock_dur):
    m2i.apply_input_availability_shock(key_label, time_period=t, reduction_pct=0.5)

r2i = m2i.run_model(validate=True)

if ENABLE_PLOTTING:
    m2i.plot_regional_results(
        r2i,
        baseline_results=r2,
        title_suffix=f"R=2 input shock - key supplier '{key_label}' (-50 %, 5 periods)",
        save_path=str(figures_dir / "input_shock_2region.png"),
    )
    print("  -> figures/input_shock_2region.png")

gdp_drop_i = (
    (r2i["gdp_regional"][:, shock_start + shock_dur - 1]
     / r2["gdp_regional"][:, shock_start + shock_dur - 1]) - 1
) * 100
for r in range(m2i.n_regions):
    print(f"  GDP impact at peak (region {m2i.region_labels[r]}): {gdp_drop_i[r]:+.2f} %")

# 5. KLEMS baseline replication, R=1 and R=2
print("\nRunning KLEMS R=1 baseline ...")
cfg_k1 = ModelConfig(n_periods=40, time_frequency="daily", prod_function="klems")
mk1    = InputOutputModel(config=cfg_k1)
rk1    = mk1.run_model(validate=True)

gdp_drift_k1 = np.abs(rk1["gdp"] - rk1["gdp"][0]).max()
print(f"  KLEMS R=1 baseline: max GDP drift from t=0 = {gdp_drift_k1:.6f}  (should be ~0)")

if ENABLE_PLOTTING:
    mk1.plot_regional_results(
        rk1,
        baseline_results=None,
        title_suffix="KLEMS R=1 no-shock (index 100 = fixed point)",
        save_path=str(figures_dir / "baseline_klems_1region.png"),
    )
    print("  -> figures/baseline_klems_1region.png")

print("\nRunning KLEMS R=2 baseline ...")
cfg_k2 = ModelConfig(
    n_periods=40,
    time_frequency="daily",
    prod_function="klems",
    data_path="data/example_data_2region.pkl",
    n_regions=2,
)
mk2 = InputOutputModel(config=cfg_k2)
rk2 = mk2.run_model(validate=True)

agg_drift_k2 = np.abs(rk2["gdp"] - rk2["gdp"][0]).max()
print(f"  KLEMS R=2 aggregate GDP drift = {agg_drift_k2:.1f}")
for r in range(mk2.n_regions):
    drift = np.abs(rk2["gdp_regional"][r] - rk2["gdp_regional"][r, 0]).max()
    print(f"  Region {mk2.region_labels[r]}: drift={drift:.0f}")

if ENABLE_PLOTTING:
    mk2.plot_regional_results(
        rk2,
        baseline_results=None,
        title_suffix="KLEMS R=2 no-shock (index 100 = period-0 value)",
        save_path=str(figures_dir / "baseline_klems_2region.png"),
    )
    print("  -> figures/baseline_klems_2region.png")

print("\nDone.")
