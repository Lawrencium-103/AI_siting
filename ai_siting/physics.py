"""
Physical models. All functions are vectorised: climate arrays have shape (H,)
and parameters may be scalars or arrays of shape (n, 1), giving outputs of
shape (n, H). Units are SI unless stated; energies are per kW of IT load.
"""
import numpy as np
from . import config as C

K0 = 273.15


def wet_bulb_stull(t_db, rh_pct):
    """Wet-bulb temperature (degC) from dry-bulb (degC) and RH (%), Stull (2011).
    Stated validity: RH 5-99 %, T -20..50 degC; RH is clipped to that range."""
    rh = np.clip(rh_pct, 5.0, 99.0)
    t = np.asarray(t_db, dtype=float)
    return (t * np.arctan(0.151977 * np.sqrt(rh + 8.313659))
            + np.arctan(t + rh) - np.arctan(rh - 1.676331)
            + 0.00391838 * rh ** 1.5 * np.arctan(0.023101 * rh) - 4.686035)


def _cop(t_evap_c, t_cond_c, eta):
    lift = np.maximum(t_cond_c - t_evap_c, 1.0)
    return np.minimum(C.CONST["cop_max"], eta * (t_evap_c + K0) / lift)


def _econ_fraction(t_source_out, t_supply):
    """Share of load that must be met mechanically. 0 = full economiser,
    1 = full chiller; linear hand-over across CONST['econ_band'] kelvin."""
    return np.clip((t_source_out - (t_supply - C.CONST["a_hx"])) / C.CONST["econ_band"], 0.0, 1.0)


def _tower_water_l_per_kwh_heat(p):
    evap = 3.6 / C.CONST["h_fg_MJ_per_kg"] * C.CONST["latent_fraction"]  # kg = L per kWh heat
    return evap * p["cycles_conc"] / (p["cycles_conc"] - 1.0)


def hourly_facility(t_db, t_wb, p, arch):
    """
    Hourly PUE and on-site water use (L per kWh of IT energy).

    evaporative: CRAH air path + chilled-water plant with a hybrid cooling tower.
                 Dry-cooler free cooling when cold enough (no water); otherwise
                 tower-side economiser blending into a water-cooled chiller.
    dry:         CRAH air path + air-cooled chiller with integrated dry-cooler
                 free cooling. No on-site water.
    dtc_dry:     Direct-to-chip cold plates capture `phi_liquid` of IT heat into a
                 warm-water loop rejected by dry coolers (chiller trim if needed);
                 the residual air-side heat is handled as in `dry`.
    """
    t_db = np.asarray(t_db)[None, :]
    t_wb = np.asarray(t_wb)[None, :]
    g = {k: (np.asarray(v).reshape(-1, 1) if np.ndim(v) else v) for k, v in p.items()}
    f_el, f_air = g["f_elec"], g["f_air"]

    if arch in ("evaporative", "dry"):
        q = 1.0 + f_el + f_air  # heat to reject per kW IT
        t_dc_out = t_db + g["a_dry"]
        x_dry = _econ_fraction(t_dc_out, g["t_chw"])

        if arch == "dry":
            cop = _cop(g["t_chw"] - C.CONST["evap_lift"], t_db + C.CONST["cond_lift_air"], g["eta_carnot"])
            p_cool = q * g["e_dry_fan"] + x_dry * q / cop
            water = np.zeros_like(p_cool)
        else:
            dry_ok = x_dry <= 0.0                     # full dry free cooling -> no evaporation
            t_tw = t_wb + g["a_tower"]
            x_ch = _econ_fraction(t_tw, g["t_chw"])
            cop = _cop(g["t_chw"] - C.CONST["evap_lift"], t_tw + C.CONST["cond_lift_tower"], g["eta_carnot"])
            p_tower = q * g["e_pump_tower"] + x_ch * q / cop
            p_dryfc = q * g["e_dry_fan"]
            p_cool = np.where(dry_ok, p_dryfc, p_tower)
            q_rej = q * (1.0 + x_ch / cop)
            water = np.where(dry_ok, 0.0, q_rej * _tower_water_l_per_kwh_heat(g))
        pue = 1.0 + f_el + f_air + p_cool
        return pue, water

    if arch == "dtc_dry":
        phi = g["phi_liquid"]
        q_liq = phi
        f_air_eff = f_air * (1.0 - phi)
        q_air = (1.0 - phi) + f_el + f_air_eff
        # liquid loop
        x_l = _econ_fraction(t_db + g["a_dry"], g["t_liquid"])
        cop_l = _cop(g["t_liquid"] - C.CONST["evap_lift"], t_db + C.CONST["cond_lift_air"], g["eta_carnot"])
        p_liq = q_liq * (g["e_dry_fan"] + g["e_pump_tower"]) + x_l * q_liq / cop_l
        # residual air path
        x_a = _econ_fraction(t_db + g["a_dry"], g["t_chw"])
        cop_a = _cop(g["t_chw"] - C.CONST["evap_lift"], t_db + C.CONST["cond_lift_air"], g["eta_carnot"])
        p_air = q_air * g["e_dry_fan"] + x_a * q_air / cop_a
        pue = 1.0 + f_el + f_air_eff + p_liq + p_air
        return pue, np.zeros_like(pue)

    raise ValueError(arch)


def it_energy_kwh(flop, p):
    """IT energy (kWh) of a training run of `flop` FLOP on H100-class hardware."""
    gpu_seconds = flop / (np.asarray(p["mfu"]) * C.F_PEAK_FLOPS) * np.asarray(p["overhead_wallclock"])
    return gpu_seconds / 3600.0 * np.asarray(p["p_it_per_gpu_kw"])


def gpu_hours(flop, p):
    return flop / (np.asarray(p["mfu"]) * C.F_PEAK_FLOPS) * np.asarray(p["overhead_wallclock"]) / 3600.0


def diesel_ef_kg_per_kwh(eta):
    return C.CONST["diesel_tco2_per_TJ"] * 1e3 / 1e6 * 3.6 / np.asarray(eta)  # t/TJ -> kg/MJ -> kg/kWh_e


def effective_ci(ci_grid, p, diesel_share=0.0):
    """Blended emission factor (kg CO2e per kWh of facility electricity)."""
    grid = np.asarray(ci_grid) * (1.0 + np.asarray(p["grid_rel_err"]))
    return (1.0 - diesel_share) * grid + diesel_share * diesel_ef_kg_per_kwh(p["eta_diesel"])
