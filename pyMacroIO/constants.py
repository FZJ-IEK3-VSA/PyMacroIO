"""
Calibration constants and enumerated mode strings.
All values are hard-coded defaults and validation bounds (ModelConfig carries scenario overrides).
"""

from __future__ import annotations

# Fallback savings rate, used only when a baseline rate cannot be inferred from data.
DEFAULT_SAVINGS_RATE = 0.05

# Default Frisch parameter for LES household demand. Must be negative.
DEFAULT_LES_FRISCH: float = -3.0

# LES calibration, indicative income elasticities pending empirical estimates, unmatched eta=1.0.
LES_SECTOR_ELASTICITIES = {
    # Energy
    "Production of electricity":                    0.45,
    "Transmission of electricity":                  0.45,
    "Distribution and trade of electricity":        0.45,
    "Manufacture of gas; distribution":             0.45,
    "Steam and hot water supply":                   0.45,
    "Petroleum Refinery":                           0.45,
    "Retail sale of automotive fuel":               0.50,
    "Extraction of natural gas":                    0.45,
    "Extraction of crude petroleum":                0.45,
    "Mining of coal and lignite":                   0.45,
    "Manufacture of coke oven products":            0.45,
    "Extraction, liquefaction, and regasification": 0.45,
    "Transport via pipelines":                      0.45,
    # Water
    "Collection, purification and distribution of water": 0.40,
    # Food and agriculture
    "Processing of Food products nec":              0.55,
    "Processing of dairy products":                 0.55,
    "Processing of meat cattle":                    0.55,
    "Processing of meat pigs":                      0.55,
    "Processing of meat poultry":                   0.55,
    "Production of meat products nec":              0.55,
    "Manufacture of fish products":                 0.60,
    "Manufacture of beverages":                     0.80,
    "Manufacture of tobacco products":              0.75,
    "Processed rice":                               0.55,
    "Sugar refining":                               0.55,
    "Processing vegetable oils and fats":           0.55,
    "Raw milk":                                     0.55,
    "Cultivation of paddy rice":                    0.55,
    "Cultivation of wheat":                         0.55,
    "Cultivation of cereal grains":                 0.55,
    "Cultivation of vegetables, fruit":             0.55,
    "Cultivation of oil seeds":                     0.55,
    "Cultivation of sugar cane":                    0.55,
    "Cultivation of plant-based fibers":            0.55,
    "Cultivation of crops nec":                     0.55,
    "Animal products nec":                          0.55,
    "Poultry farming":                              0.55,
    "Cattle farming":                               0.55,
    "Pigs farming":                                 0.55,
    "Meat animals nec":                             0.55,
    "Fishing":                                      0.60,
    "Wool, silk-worm cocoons":                      0.70,
    # Health
    "Health and social work":                       0.50,
    # Education
    "Education":                                    0.70,
    # Housing
    "Real estate activities":                       0.35,
    "Construction":                                 0.90,
    # Telecoms and IT
    "Post and telecommunications":                  0.80,
    "Computer and related activities":              1.10,
    # Transport
    "Transport via railways":                       0.85,
    "Other land transport":                         0.85,
    "Sea and coastal water transport":              1.20,
    "Inland water transport":                       1.10,
    "Air transport":                                1.40,
    "Supporting and auxiliary transport":           1.00,
    "Sale, maintenance, repair of motor vehicles":  1.10,
    # Manufacturing
    "Manufacture of motor vehicles":                1.10,
    "Manufacture of wearing apparel":               1.00,
    "Manufacture of textiles":                      1.00,
    "Manufacture of furniture":                     1.00,
    "Manufacture of other transport equipment":     1.00,
    "Manufacture of electrical machinery":          1.00,
    "Manufacture of office machinery":              1.10,
    "Manufacture of radio, television":             1.10,
    "Manufacture of medical, precision":            1.00,
    "Manufacture of rubber and plastic products":   1.00,
    "Manufacture of wood":                          0.90,
    "Tanning and dressing of leather":              1.10,
    "Manufacture of fabricated metal":              0.95,
    "Manufacture of machinery and equipment":       0.95,
    "Publishing, printing":                         0.90,
    "Paper":                                        0.90,
    "Plastics, basic":                              1.00,
    "Chemicals nec":                                1.00,
    "Manufacture of ceramic goods":                 1.00,
    "Manufacture of glass":                         1.00,
    "Manufacture of other non-metallic":            1.00,
    # Retail and wholesale
    "Retail trade, except of motor vehicles":       1.00,
    "Wholesale trade":                              1.00,
    # Financial services
    "Financial intermediation, except":             1.20,
    "Insurance and pension funding":                1.10,
    "Activities auxiliary to financial":            1.20,
    # Hotels and restaurants
    "Hotels and restaurants":                       1.35,
    # Recreation
    "Recreational, cultural and sporting":          1.45,
    "Activities of membership organisation":        1.30,
    # Other services
    "Other business activities":                    1.20,
    "Research and development":                     1.20,
    "Renting of machinery":                         1.10,
    "Other service activities":                     1.20,
    "Private households with employed persons":     0.80,
    "Public administration and defence":            0.80,
}

# Hire/fire bounds, (1 - delta) clipped to the DELTA range and labour to the CAPACITY range x l0.
DELTA_FLOOR = 0.2
DELTA_CAP = 1.0
CAPACITY_MIN_SCALE = 0.3
CAPACITY_MAX_SCALE = 1.5

# Firing-speed damping factor applied to gamma_fire when reducing labour.
FIRING_SPEED_DAMPING = 0.5

# Consumption demand floors in findemand_cd, as a ratio of baseline consumption.
CONSUMPTION_FLOOR_RATIO = 0.5
CONSUMPTION_FLOOR_LABOUR_RATIO = 0.2

# Enumerated mode strings (used in ModelConfig and validation)
HOUSEHOLD_CLOSURE_MODES = ("return_to_base", "scarred", "frozen")
PRODUCTION_FUNCTIONS = ("leontief", "leontief.adapted", "linear", "ces", "klems")
FIRM_PRIORITY_MODES = ("no", "yes")
INVESTMENT_CLOSURE_MODES = ("fixed", "keynesian")

# Threshold in producing_x above which A_essential marks an input essential in adapted Leontief.
ESSENTIAL_INPUT_THRESHOLD = 0.5

# Default CES substitution elasticity.
CES_ELASTICITY_DEFAULT = 1.5

# IO identity tolerances
ROW_IDENTITY_ATOL = 1e-10
VA_IDENTITY_TOLERANCE = 1.0

# Default sector-level parameters (used when config arrays are not provided)
DEFAULT_TAU = 2.9
DEFAULT_GAMMA_HIRE = 0.375
DEFAULT_GAMMA_FIRE = 0.5

# Bounds wide enough for the daily canon of Pichler et al., doi:10.1016/j.jedc.2022.104527.
GAMMA_HIRE_MIN = 0.01
GAMMA_HIRE_MAX = 0.8
GAMMA_FIRE_MIN = 0.01
GAMMA_FIRE_MAX = 0.8
TAU_MIN = 0.5
TAU_MAX = 90.0

# Large finite value replacing inf/nan in producing_x, dominating any output level in the data.
NUMERIC_LARGE = 1e18

# Coefficient floor below which an input is a trace entry, never essential.
COMBINED_COEF_THRESHOLD = 0.002
