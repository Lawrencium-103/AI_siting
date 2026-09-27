"""
Central configuration. Every numerical assumption used anywhere in the pipeline
lives here, with its source, so that Table 2 of the paper is generated
directly from this file and cannot drift from the code.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
TAB_DIR = OUT_DIR / "tables"

YEARS = [2021, 2022, 2023, 2024, 2025]
SEED = 20260925

# ---------------------------------------------------------------------------
# Study sites (candidate training locations)
# ---------------------------------------------------------------------------
STUDY_SITES = {
    # name: (lat, lon, OWID/Ember country name, note)
    "Oslo":      (59.91, 10.75, "Norway",        "Nordic, hydro-dominated grid"),
    "Paris":     (48.86, 2.35,  "France",        "Temperate, nuclear-dominated grid"),
    "Ashburn":   (39.04, -77.49, "United States", "Temperate-humid, largest US data-centre cluster"),
    "Beijing":   (39.90, 116.40, "China",         "Continental, coal-heavy grid"),
    "Marrakech": (31.63, -8.01, "Morocco",       "Semi-arid, hot summers"),
    "Lagos":     (6.52, 3.38,   "Nigeria",       "Tropical-humid, weak grid reliability"),
}
REFERENCE_SITE = "Ashburn"

# Google campuses with published trailing-twelve-month PUE, used for calibration
# (split into fit and hold-out sets). Coordinates are town-level; ERA5 resolution
# (~0.25 deg) makes finer precision meaningless.
CALIBRATION_SITES = {
    "Loudoun County, Virginia":  (39.04, -77.49, "fit"),
    "St. Ghislain, Belgium":     (50.45, 3.82,   "fit"),
    "Singapore":                 (1.35, 103.70,  "fit"),
    "Council Bluffs, Iowa":      (41.26, -95.86, "fit"),
    "Changhua County, Taiwan":   (24.08, 120.54, "fit"),
    "Quilicura, Chile":          (-33.36, -70.73, "fit"),
    "Midlothian, Texas":         (32.48, -96.99, "fit"),
    "Dublin, Ireland":           (53.40, -6.21,  "holdout"),
    "Eemshaven, Netherlands":    (53.44, 6.83,   "holdout"),
    "Fredericia, Denmark":       (55.57, 9.75,   "holdout"),
    "The Dalles, Oregon":        (45.60, -121.18, "holdout"),
    "Douglas County, Georgia":   (33.75, -84.58, "holdout"),
    "Henderson, Nevada":         (36.04, -114.98, "holdout"),
    "Mayes County, Oklahoma":    (36.31, -95.32, "holdout"),
}
# Hamina (Finland) is deliberately excluded: it is cooled with seawater, which
# the air/tower/dry-cooler model does not represent.

# ---------------------------------------------------------------------------
# External data sources (downloaded at run time, cached under data/raw)
# ---------------------------------------------------------------------------
OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
OWID_ENERGY_CSV = "https://raw.githubusercontent.com/owid/energy-data/master/owid-energy-data.csv"
EPOCH_LARGE_SCALE_CSV = "https://epoch.ai/data/large_scale_ai_models.csv"

# Demand cohort: large-scale training runs published in this window
COHORT_START = "2024-01-01"
COHORT_END = "2025-12-31"
COHORT_MIN_FLOP = 1e24

# ---------------------------------------------------------------------------
# Functional unit
# ---------------------------------------------------------------------------
FU_FLOP = 1e25  # EU AI Act (Reg. 2024/1689, Art. 51) systemic-risk presumption threshold

# ---------------------------------------------------------------------------
# Parameter distributions. Each entry: (dist, a, mode_or_b, c, unit, source)
#   dist = "tri" -> triangular(a, mode, c);  "uni" -> uniform(a, c) (mode ignored)
#   The deterministic ("central") value is the mode for tri, midpoint for uni.
# ---------------------------------------------------------------------------
PARAMS = {
    # --- IT energy -----------------------------------------------------------
    "mfu": ("tri", 0.30, 0.40, 0.50, "-",
            "Llama 3 405B 38-43% BF16 MFU (Dubey et al. 2024); PaLM 46.2% (Chowdhery et al. 2023)"),
    "p_it_per_gpu_kw": ("tri", 0.85, 1.05, 1.30, "kW",
            "H100 SXM TDP 0.70 kW; DGX H100 max system power 10.2 kW / 8 GPUs = 1.275 kW; + fabric/storage share"),
    "overhead_wallclock": ("tri", 1.00, 1.08, 1.20, "-",
            "Checkpointing, restarts and evaluation; Llama 3 reports >90% effective training time"),
    # --- Facility (cooling and electrical) -------------------------------------
    "f_elec": ("tri", 0.02, 0.04, 0.07, "kW/kW_IT",
            "UPS, distribution and lighting losses of hyperscale facilities (Shehabi et al. 2016; 2024)"),
    "f_air": ("tri", 0.01, 0.02, 0.04, "kW/kW_IT",
            "Air-handler fan power, hot/cold aisle containment"),
    "t_chw": ("tri", 16.0, 20.0, 24.0, "degC",
            "Elevated chilled-water supply consistent with ASHRAE TC 9.9 recommended envelope (18-27 degC inlet)"),
    "a_tower": ("tri", 3.0, 4.0, 6.0, "K", "Cooling-tower approach to wet-bulb"),
    "a_dry": ("tri", 6.0, 8.0, 12.0, "K", "Dry-cooler approach to dry-bulb"),
    "eta_carnot": ("tri", 0.45, 0.55, 0.65, "-",
            "Chiller second-law efficiency (fraction of Carnot COP)"),
    "e_pump_tower": ("tri", 0.010, 0.015, 0.030, "kW/kW_heat", "Condenser pumps and tower fans"),
    "e_dry_fan": ("tri", 0.010, 0.020, 0.040, "kW/kW_heat", "Dry-cooler fans"),
    "cycles_conc": ("tri", 3.0, 5.0, 8.0, "-", "Cooling-tower cycles of concentration (blowdown)"),
    "phi_liquid": ("tri", 0.70, 0.80, 0.90, "-",
            "Fraction of IT heat captured by direct-to-chip cold plates"),
    "t_liquid": ("tri", 30.0, 35.0, 40.0, "degC",
            "Facility water supply for warm-water DTC (ASHRAE liquid classes W32-W40)"),
    # --- Grid and backup -------------------------------------------------------
    "grid_rel_err": ("tri", -0.10, 0.0, 0.10, "-",
            "Relative uncertainty applied to Ember annual lifecycle intensity"),
    "eta_diesel": ("tri", 0.30, 0.35, 0.40, "-",
            "Diesel genset electrical efficiency; with IPCC 2006 default 74.1 tCO2/TJ fuel"),
}

# Fixed physical constants and design values (not sampled)
CONST = {
    "h_fg_MJ_per_kg": 2.43,        # latent heat of vaporisation of water at ~30 degC
    "latent_fraction": 0.90,       # share of tower heat rejected by evaporation
    "diesel_tco2_per_TJ": 74.1,    # IPCC 2006 Guidelines, Vol. 2, Ch. 2, gas/diesel oil
    "a_hx": 2.0,                   # economiser heat-exchanger approach, K
    "econ_band": 4.0,              # K over which economiser hands over to chiller
    "evap_lift": 3.0,              # chiller evaporator below supply temperature, K
    "cond_lift_tower": 6.0,        # condensing above tower water, K
    "cond_lift_air": 12.0,         # condensing above dry-bulb for air-cooled chillers, K
    "cop_max": 12.0,               # cap on chiller COP at very low lift
}

# H100 SXM dense BF16 peak (NVIDIA datasheet lists 1,979 TFLOPS with sparsity)
F_PEAK_FLOPS = 989.4e12

# Diesel exposure scenarios for Lagos (share of facility energy from on-site gensets)
DIESEL_SHARES = [0.0, 0.10, 0.25, 0.50]
DIESEL_SHARE_CENTRAL = 0.0   # headline results use grid-only; diesel reported as scenarios

# Cooling architectures evaluated
ARCHITECTURES = ["evaporative", "dry", "dtc_dry"]
ARCH_LABELS = {
    "evaporative": "Air + hybrid cooling tower (evaporative)",
    "dry": "Air + dry cooler / air-cooled chiller",
    "dtc_dry": "Direct-to-chip liquid + dry cooler",
}

# Monte Carlo and Sobol sample sizes
N_MC = 1000
N_SOBOL = 256

# Siting: per-site capacity expressed as a share of portfolio IT energy
CAPACITY_SHARES = [0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.00]
