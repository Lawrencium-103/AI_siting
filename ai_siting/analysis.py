"""
Core analyses:
  1. climate descriptors for each site
  2. calibration of the facility model against Google campus PUE (fit / hold-out)
  3. validation of the IT-energy model against Meta's Llama 3.1 405B disclosure
  4. Monte Carlo site assessment (PUE, WUE, carbon per functional unit)
  5. variance-based (Sobol) sensitivity analysis
  6. log-ratio decomposition of site differences into grid and facility terms
  7. diesel-exposure scenarios for Lagos
"""
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze

from . import config as C
from . import physics as P
from . import sampling as S

# Parameters fitted during calibration (others keep literature priors)
FIT_KEYS = ["f_elec", "t_chw"]
FIT_BOUNDS = {"f_elec": (0.005, 0.10), "t_chw": (14.0, 25.0)}


# ---------------------------------------------------------------------------
# 1. Climate
# ---------------------------------------------------------------------------
def prepare_climate(df):
    df = df.copy()
    df["t_wb"] = P.wet_bulb_stull(df.t_db.values, df.rh.values)
    df["year"] = df.time.dt.year
    return df


def climate_descriptors(clim, p_central):
    rows = []
    for site, df in clim.items():
        pue_e, _ = P.hourly_facility(df.t_db.values, df.t_wb.values, p_central, "evaporative")
        dry_fc = (df.t_db.values + p_central["a_dry"]) <= (p_central["t_chw"] - C.CONST["a_hx"])
        tower_fc = (df.t_wb.values + p_central["a_tower"]) <= (p_central["t_chw"] - C.CONST["a_hx"])
        n_years = df.year.nunique()
        rows.append({
            "Site": site,
            "Mean T_db (degC)": df.t_db.mean(),
            "P99 T_db (degC)": df.t_db.quantile(0.99),
            "Mean RH (%)": df.rh.mean(),
            "Mean T_wb (degC)": df.t_wb.mean(),
            "P99 T_wb (degC)": df.t_wb.quantile(0.99),
            "Dry free-cooling h/yr": dry_fc.sum() / n_years,
            "Tower free-cooling h/yr": (tower_fc | dry_fc).sum() / n_years,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Calibration against Google campus PUE
# ---------------------------------------------------------------------------
def _annual_pue(df, p, arch="evaporative"):
    out = {}
    for y, g in df.groupby("year"):
        pue, _ = P.hourly_facility(g.t_db.values, g.t_wb.values, p, arch)
        out[int(y)] = float(np.mean(pue))
    return out


def calibrate(calib_clim, google):
    p0 = S.central()
    obs = google.merge(pd.DataFrame(
        [(k, v[2]) for k, v in C.CALIBRATION_SITES.items()], columns=["campus", "split"]), on="campus")

    def predict(theta, subset):
        p = dict(p0)
        p.update(dict(zip(FIT_KEYS, theta)))
        preds = []
        for _, r in subset.iterrows():
            df = calib_clim[r.campus]
            g = df[df.year == r.year]
            pue, _ = P.hourly_facility(g.t_db.values, g.t_wb.values, p, "evaporative")
            preds.append(float(np.mean(pue)))
        return np.array(preds)

    fit = obs[obs.split == "fit"].reset_index(drop=True)
    hold = obs[obs.split == "holdout"].reset_index(drop=True)
    x0 = np.array([p0[k] for k in FIT_KEYS])
    lb = [FIT_BOUNDS[k][0] for k in FIT_KEYS]
    ub = [FIT_BOUNDS[k][1] for k in FIT_KEYS]
    res = least_squares(lambda th: predict(th, fit) - fit.ttm_pue.values, x0, bounds=(lb, ub))
    theta = dict(zip(FIT_KEYS, res.x))

    fit["pred"] = predict(res.x, fit)
    hold["pred"] = predict(res.x, hold)
    prior_pred_hold = predict(x0, hold)

    def metrics(df, col="pred"):
        e = df[col] - df.ttm_pue
        # PUE overhead (PUE-1) is the physically meaningful quantity for relative error
        rel = e / (df.ttm_pue - 1.0)
        return {"MAE": float(np.abs(e).mean()), "RMSE": float(np.sqrt((e ** 2).mean())),
                "Bias": float(e.mean()), "MAPE_overhead": float(np.abs(rel).mean()),
                "n": int(len(df))}

    campus_means = pd.concat([fit, hold]).groupby(["campus", "split"])[["ttm_pue", "pred"]].mean().reset_index()
    from scipy.stats import spearmanr
    rho = spearmanr(campus_means.ttm_pue, campus_means.pred).correlation

    return {
        "theta": theta,
        "fit": fit, "holdout": hold,
        "metrics_fit": metrics(fit), "metrics_holdout": metrics(hold),
        "holdout_prior_MAE": float(np.abs(prior_pred_hold - hold.ttm_pue).mean()),
        "spearman_campus": float(rho),
        "campus_means": campus_means,
    }


def calibrated_params(theta):
    """Re-centre fitted parameters on their calibrated values, keeping prior widths."""
    over = {}
    for k, v in theta.items():
        dist, a, m, b, unit, src = C.PARAMS[k]
        half_lo, half_hi = m - a, b - m
        lo = max(FIT_BOUNDS[k][0], v - half_lo)
        hi = min(FIT_BOUNDS[k][1], v + half_hi)
        over[k] = ("tri", lo, float(v), hi, unit, src + "; mode calibrated to Google campus PUE (this study)")
    return S.with_overrides(C.PARAMS, over)


# ---------------------------------------------------------------------------
# 3. Validation of the IT-energy model (Llama 3.1 405B)
# ---------------------------------------------------------------------------
LLAMA405 = {
    "flop": 3.8e25,                 # Dubey et al. (2024), The Llama 3 Herd of Models
    "gpu_hours_reported": 30.84e6,  # Meta Llama 3.1 model card
    "tco2_reported": 8930.0,        # location-based, Meta Llama 3.1 model card
    "tdp_kw": 0.70,
    "year": 2024,
}


def validate_it_energy(mc, pue_ashburn_2024_draws, ci_us_2024):
    gh = P.gpu_hours(LLAMA405["flop"], mc)
    e_it = P.it_energy_kwh(LLAMA405["flop"], mc)
    co2_full = e_it * pue_ashburn_2024_draws * ci_us_2024 * (1 + mc["grid_rel_err"]) / 1000.0
    # Same calculation restricted to Meta's reporting boundary (GPU TDP only)
    co2_meta_boundary = (LLAMA405["gpu_hours_reported"] * LLAMA405["tdp_kw"]
                         * np.median(pue_ashburn_2024_draws) * ci_us_2024 / 1000.0)
    implied_mfu = LLAMA405["flop"] / (LLAMA405["gpu_hours_reported"] * 3600 * C.F_PEAK_FLOPS)
    q = lambda a: [float(np.percentile(a, 2.5)), float(np.median(a)), float(np.percentile(a, 97.5))]
    return {
        "gpu_hours_pred": q(gh),
        "gpu_hours_reported": LLAMA405["gpu_hours_reported"],
        "gpu_hours_ratio_median": float(np.median(gh) / LLAMA405["gpu_hours_reported"]),
        "gpu_hours_reported_in_95": bool(q(gh)[0] <= LLAMA405["gpu_hours_reported"] <= q(gh)[2]),
        "implied_mfu": float(implied_mfu),
        "tco2_pred_full_boundary": q(co2_full),
        "tco2_meta_boundary_recomputed": float(co2_meta_boundary),
        "tco2_reported": LLAMA405["tco2_reported"],
        "energy_gwh_pred": q(e_it / 1e6),
    }


# ---------------------------------------------------------------------------
# 4. Monte Carlo site assessment
# ---------------------------------------------------------------------------
def site_monte_carlo(clim, grid, mc, archs=C.ARCHITECTURES, chunk=250):
    """
    Returns long DataFrame: site, arch, year, draw, pue, wue_l_per_kwh_it,
    ci_kg_per_kwh (sampled), e_it_kwh_fu, co2_t_fu, water_m3_fu.
    """
    n = len(next(iter(mc.values())))
    e_it_fu = P.it_energy_kwh(C.FU_FLOP, mc)
    rows = []
    for site, df in clim.items():
        country = C.STUDY_SITES[site][2]
        for y, g in df.groupby("year"):
            ci = float(grid[(grid.country == country) & (grid.year == y)].ci_kg_per_kwh.iloc[0])
            ci_s = P.effective_ci(ci, mc, 0.0)
            for arch in archs:
                pue = np.empty(n)
                wue = np.empty(n)
                for s in range(0, n, chunk):
                    sub = {k: v[s:s + chunk] for k, v in mc.items()}
                    ph, wh = P.hourly_facility(g.t_db.values, g.t_wb.values, sub, arch)
                    pue[s:s + chunk] = ph.mean(axis=1)
                    wue[s:s + chunk] = wh.mean(axis=1)
                rows.append(pd.DataFrame({
                    "site": site, "arch": arch, "year": int(y), "draw": np.arange(n),
                    "pue": pue, "wue": wue, "ci": ci_s, "ci_ember": ci,
                    "e_it_kwh_fu": e_it_fu,
                    "co2_t_fu": e_it_fu * pue * ci_s / 1000.0,
                    "water_m3_fu": e_it_fu * wue / 1000.0,
                }))
    return pd.concat(rows, ignore_index=True)


def summarise_sites(mcres):
    """Average over 2021-2025 within each draw, then summarise across draws."""
    per_draw = mcres.groupby(["site", "arch", "draw"])[
        ["pue", "wue", "ci", "co2_t_fu", "water_m3_fu", "e_it_kwh_fu"]].mean().reset_index()

    def q(x):
        return pd.Series({"median": x.median(), "lo": x.quantile(0.025), "hi": x.quantile(0.975)})

    per_draw["co2_per_kwh_it"] = per_draw.co2_t_fu / per_draw.e_it_kwh_fu      # t per kWh_IT
    per_draw["water_per_kwh_it"] = per_draw.water_m3_fu / per_draw.e_it_kwh_fu  # m3 per kWh_IT
    out = []
    for (site, arch), g in per_draw.groupby(["site", "arch"]):
        r = {"site": site, "arch": arch,
             "co2_per_kwh_it": g.co2_per_kwh_it.median(),
             "water_per_kwh_it": g.water_per_kwh_it.median(),
             "e_it_fu_med": g.e_it_kwh_fu.median()}
        for col in ["pue", "wue", "co2_t_fu", "water_m3_fu"]:
            s = q(g[col])
            r[f"{col}_med"], r[f"{col}_lo"], r[f"{col}_hi"] = s["median"], s["lo"], s["hi"]
        out.append(r)
    summ = pd.DataFrame(out)
    # Interannual variability of the central estimate (median across draws per year)
    yearly = mcres.groupby(["site", "arch", "year"])[["pue", "co2_t_fu", "ci_ember"]].median().reset_index()
    return per_draw, summ, yearly


def pairwise_probabilities(per_draw, arch="evaporative", ref=C.REFERENCE_SITE):
    """P(site emits less than reference) using common random numbers (paired draws)."""
    a = per_draw[per_draw.arch == arch].pivot(index="draw", columns="site", values="co2_t_fu")
    return {s: float((a[s] < a[ref]).mean()) for s in a.columns if s != ref}


# ---------------------------------------------------------------------------
# 5. Sobol sensitivity
# ---------------------------------------------------------------------------
SOBOL_KEYS = ["mfu", "p_it_per_gpu_kw", "overhead_wallclock", "f_elec", "f_air", "t_chw",
              "a_tower", "a_dry", "eta_carnot", "e_pump_tower", "e_dry_fan", "cycles_conc",
              "grid_rel_err"]


def sobol_site(df_year, ci, params, arch="evaporative", n=C.N_SOBOL, seed=C.SEED, output="co2"):
    problem = S.sobol_problem(SOBOL_KEYS, params)
    u = sobol_sample.sample(problem, n, calc_second_order=False, seed=seed)
    p = S.from_unit(u, SOBOL_KEYS, params)
    e_it = P.it_energy_kwh(C.FU_FLOP, p)
    pue = np.empty(len(u))
    wue = np.empty(len(u))
    for s in range(0, len(u), 500):
        sub = {k: (v[s:s + 500] if np.ndim(v) else v) for k, v in p.items()}
        ph, wh = P.hourly_facility(df_year.t_db.values, df_year.t_wb.values, sub, arch)
        pue[s:s + 500] = ph.mean(axis=1)
        wue[s:s + 500] = wh.mean(axis=1)
    if output == "co2":
        y = e_it * pue * P.effective_ci(ci, p, 0.0) / 1000.0
    else:
        y = e_it * wue / 1000.0
    si = sobol_analyze.analyze(problem, y, calc_second_order=False, seed=seed)
    return pd.DataFrame({"param": SOBOL_KEYS, "S1": si["S1"], "ST": si["ST"],
                         "ST_conf": si["ST_conf"]})


# ---------------------------------------------------------------------------
# 6. Decomposition of site differences
# ---------------------------------------------------------------------------
def decomposition(summ, grid_mean, arch="evaporative", ref=C.REFERENCE_SITE):
    s = summ[summ.arch == arch].set_index("site")
    rows = []
    for site in s.index:
        ratio_c = s.loc[site, "co2_t_fu_med"] / s.loc[ref, "co2_t_fu_med"]
        ratio_p = s.loc[site, "pue_med"] / s.loc[ref, "pue_med"]
        ratio_g = grid_mean[site] / grid_mean[ref]
        rows.append({"site": site, "carbon_ratio": ratio_c, "ln_carbon": np.log(ratio_c),
                     "ln_grid": np.log(ratio_g), "ln_facility": np.log(ratio_p),
                     "grid_share_abs": abs(np.log(ratio_g)) / (abs(np.log(ratio_g)) + abs(np.log(ratio_p)) + 1e-12)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 7. Diesel exposure (Lagos)
# ---------------------------------------------------------------------------
def diesel_scenarios(mcres, mc, site="Lagos", shares=C.DIESEL_SHARES):
    g = mcres[(mcres.site == site)].groupby(["arch", "draw"])[["pue", "e_it_kwh_fu", "ci_ember"]].mean().reset_index()
    rows = []
    for sh in shares:
        for arch, a in g.groupby("arch"):
            sub = {k: v[a.draw.values] for k, v in mc.items()}
            ci = P.effective_ci(a.ci_ember.values, sub, sh)
            co2 = a.e_it_kwh_fu.values * a.pue.values * ci / 1000.0
            rows.append({"site": site, "arch": arch, "diesel_share": sh,
                         "co2_med": float(np.median(co2)),
                         "co2_lo": float(np.percentile(co2, 2.5)),
                         "co2_hi": float(np.percentile(co2, 97.5))})
    return pd.DataFrame(rows)
