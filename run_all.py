"""
Run the complete analysis for "Siting large AI training runs" and write every table, figure and reported number.

    python run_all.py                  # real data (downloads once, then cached)
    python run_all.py --demo           # synthetic climate, for testing the code path only
    python run_all.py --fallback-cohort path/to/master_gold.parquet

Every number, table and figure reported in the paper is produced here. Nothing is typed by hand.
"""
import argparse
import json
import sys
import time
import numpy as np
import pandas as pd

from ai_siting import config as C, data as D, physics as P, sampling as S
from ai_siting import analysis as A, siting as ST, figures as F


def fmt(x, nd=2):
    return f"{x:,.{nd}f}"


def sci(x, nd=1):
    e = int(np.floor(np.log10(abs(x))))
    return f"{x / 10 ** e:.{nd}f} × 10^{{{e}}}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="synthetic climate; outputs watermarked")
    ap.add_argument("--fallback-cohort", default=None, help="parquet used if Epoch CSV is unavailable")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--n-mc", type=int, default=C.N_MC)
    ap.add_argument("--n-sobol", type=int, default=C.N_SOBOL)
    args = ap.parse_args()
    t0 = time.time()
    for d in (C.OUT_DIR, C.FIG_DIR, C.TAB_DIR, C.RAW_DIR):
        d.mkdir(parents=True, exist_ok=True)
    R = {"DEMO": args.demo}

    # ------------------------------------------------------------------ data
    print("[1/8] climate")
    clim = {s: A.prepare_climate(D.fetch_climate(s, v[0], v[1], demo=args.demo))
            for s, v in C.STUDY_SITES.items()}
    calib_clim = {s: A.prepare_climate(D.fetch_climate(s, v[0], v[1], demo=args.demo))
                  for s, v in C.CALIBRATION_SITES.items()}
    print("[2/8] grid and cohort")
    grid = D.fetch_grid()
    cohort, prov = D.fetch_cohort(fallback_parquet=args.fallback_cohort, offline=args.offline)
    google = D.google_pue_table()

    # ----------------------------------------------------------- calibration
    print("[3/8] calibration")
    cal = A.calibrate(calib_clim, google)
    params = A.calibrated_params(cal["theta"])
    pc = S.central(params)
    mc = S.monte_carlo(args.n_mc, params=params)

    # ------------------------------------------------------ site assessment
    print("[4/8] Monte Carlo site assessment")
    mcres = A.site_monte_carlo(clim, grid, mc)
    per_draw, summ, yearly = A.summarise_sites(mcres)
    probs = A.pairwise_probabilities(per_draw)
    grid_mean = {s: float(grid[(grid.country == v[2])].ci_kg_per_kwh.mean()) for s, v in C.STUDY_SITES.items()}
    decomp = A.decomposition(summ, grid_mean)

    # ------------------------------------------------------------ validation
    print("[5/8] validation")
    ash24 = clim["Ashburn"][clim["Ashburn"].year == 2024]
    # Annual Ashburn 2024 PUE for every Monte Carlo draw, paired with that draw's IT-energy parameters
    pue_a = np.empty(args.n_mc)
    for s0 in range(0, args.n_mc, 250):
        sub = {k: v[s0:s0 + 250] for k, v in mc.items()}
        pue_a[s0:s0 + 250] = P.hourly_facility(ash24.t_db.values, ash24.t_wb.values, sub, "evaporative")[0].mean(axis=1)
    ci_us24 = float(grid[(grid.country == "United States") & (grid.year == 2024)].ci_kg_per_kwh.iloc[0])
    val = A.validate_it_energy(mc, pue_a, ci_us24)

    # ---------------------------------------------------------------- Sobol
    print("[6/8] Sobol sensitivity")
    sob = []
    sob_w = []
    for s, v in C.STUDY_SITES.items():
        g = clim[s][clim[s].year == 2024]
        ci = float(grid[(grid.country == v[2]) & (grid.year == 2024)].ci_kg_per_kwh.iloc[0])
        sob.append(A.sobol_site(g, ci, params, n=args.n_sobol).assign(site=s))
        sob_w.append(A.sobol_site(g, ci, params, n=args.n_sobol, output="water").assign(site=s))
    sob = pd.concat(sob); sob_w = pd.concat(sob_w)

    # ---------------------------------------------------------------- diesel
    diesel = A.diesel_scenarios(mcres, mc)

    # ---------------------------------------------------------------- siting
    print("[7/8] siting optimisation")
    e_fu = float(np.median(P.it_energy_kwh(C.FU_FLOP, mc)))
    cohort["e_it_kwh"] = cohort.flop / C.FU_FLOP * e_fu
    opts_air = ST.build_options(summ, ["evaporative", "dry"])
    opts_all = ST.build_options(summ, C.ARCHITECTURES)
    base = ST.baselines(cohort, ST.build_options(summ, ["evaporative"]))
    sweep_air = ST.capacity_sweep(cohort, opts_air)
    sweep_all = ST.capacity_sweep(cohort, opts_all)
    feas = sweep_air[sweep_air.feasible]
    pareto_cap = float(feas.cap_share.iloc[0]) if len(feas) else None
    par = ST.pareto(cohort, opts_air, pareto_cap) if pareto_cap else pd.DataFrame()

    # ---------------------------------------------------------------- figures
    print("[8/8] figures and tables")
    F.fig_climate(clim, C.FIG_DIR / "fig1_climate.png", args.demo)
    F.fig_calibration(cal, C.FIG_DIR / "fig2_calibration.png", args.demo)
    F.fig_carbon_by_site(summ, C.FIG_DIR / "fig3_carbon_by_site.png", args.demo)
    F.fig_carbon_water(summ, C.FIG_DIR / "fig5_carbon_water.png", args.demo)
    F.fig_sobol(sob, C.FIG_DIR / "fig4_sobol.png", args.demo)
    F.fig_diesel_and_siting(diesel, summ, sweep_air, par, base, C.FIG_DIR / "fig6_diesel_siting.png", args.demo)

    # ---------------------------------------------------------------- tables
    desc = A.climate_descriptors(clim, pc)
    t1 = desc.copy()
    for s in C.STUDY_SITES:
        cty = C.STUDY_SITES[s][2]
        g = grid[grid.country == cty].sort_values("year")
        t1.loc[t1.Site == s, "Grid CI 2021-2025 (gCO2e/kWh)"] = (
            f"{g.ci_kg_per_kwh.min() * 1000:.0f}-{g.ci_kg_per_kwh.max() * 1000:.0f}")
        t1.loc[t1.Site == s, "Country"] = cty
    t1 = t1[["Site", "Country", "Mean T_db (degC)", "P99 T_db (degC)", "Mean RH (%)", "Mean T_wb (degC)",
             "P99 T_wb (degC)", "Dry free-cooling h/yr", "Tower free-cooling h/yr", "Grid CI 2021-2025 (gCO2e/kWh)"]]
    t1.round(1).to_csv(C.TAB_DIR / "table1_sites.csv", index=False)

    t2 = pd.DataFrame([{"Parameter": k, "Distribution": v[0],
                        "Min": v[1], "Mode": v[2], "Max": v[3], "Unit": v[4], "Basis": v[5]}
                       for k, v in params.items()])
    t2.to_csv(C.TAB_DIR / "table2_parameters.csv", index=False)

    t3 = pd.DataFrame([
        {"Check": "Llama 3.1 405B H100-hours (M)", "Model (95% interval)":
            f"{val['gpu_hours_pred'][1] / 1e6:.1f} ({val['gpu_hours_pred'][0] / 1e6:.1f}-{val['gpu_hours_pred'][2] / 1e6:.1f})",
         "Reference": f"{val['gpu_hours_reported'] / 1e6:.2f} (Meta model card)"},
        {"Check": "Llama 3.1 405B location-based tCO2e", "Model (95% interval)":
            f"{val['tco2_pred_full_boundary'][1]:,.0f} ({val['tco2_pred_full_boundary'][0]:,.0f}-{val['tco2_pred_full_boundary'][2]:,.0f}) full IT boundary",
         "Reference": f"{val['tco2_reported']:,.0f} (GPU-TDP boundary); recomputed at that boundary: {val['tco2_meta_boundary_recomputed']:,.0f}"},
        {"Check": "Facility model, fit campuses (n site-years)", "Model (95% interval)":
            f"MAE {cal['metrics_fit']['MAE']:.3f}; bias {cal['metrics_fit']['Bias']:+.3f}",
         "Reference": f"Google TTM PUE (n = {cal['metrics_fit']['n']})"},
        {"Check": "Facility model, hold-out campuses", "Model (95% interval)":
            f"MAE {cal['metrics_holdout']['MAE']:.3f}; bias {cal['metrics_holdout']['Bias']:+.3f}",
         "Reference": f"Google TTM PUE (n = {cal['metrics_holdout']['n']}); prior-only MAE {cal['holdout_prior_MAE']:.3f}"},
    ])
    t3.to_csv(C.TAB_DIR / "table3_validation.csv", index=False)

    t4 = summ.copy()
    t4["Architecture"] = t4.arch.map(C.ARCH_LABELS)
    t4["PUE"] = t4.apply(lambda r: f"{r.pue_med:.3f} ({r.pue_lo:.3f}-{r.pue_hi:.3f})", axis=1)
    t4["WUE (L/kWh_IT)"] = t4.apply(lambda r: f"{r.wue_med:.2f} ({r.wue_lo:.2f}-{r.wue_hi:.2f})", axis=1)
    t4["tCO2e per 1e25 FLOP"] = t4.apply(lambda r: f"{r.co2_t_fu_med:,.0f} ({r.co2_t_fu_lo:,.0f}-{r.co2_t_fu_hi:,.0f})", axis=1)
    t4["Water (thousand m3 per 1e25 FLOP)"] = t4.apply(
        lambda r: f"{r.water_m3_fu_med / 1e3:.1f} ({r.water_m3_fu_lo / 1e3:.1f}-{r.water_m3_fu_hi / 1e3:.1f})", axis=1)
    order = {s: i for i, s in enumerate(C.STUDY_SITES)}
    t4 = t4.sort_values(["site", "arch"], key=lambda c: c.map(order) if c.name == "site" else c)
    t4[["site", "Architecture", "PUE", "WUE (L/kWh_IT)", "tCO2e per 1e25 FLOP", "Water (thousand m3 per 1e25 FLOP)"]] \
        .rename(columns={"site": "Site"}).to_csv(C.TAB_DIR / "table4_site_results.csv", index=False)

    t5 = decomp.copy()
    t5["P(lower than Ashburn)"] = t5.site.map(lambda s: probs.get(s, np.nan))
    t5 = t5.rename(columns={"site": "Site", "carbon_ratio": "Carbon ratio vs Ashburn",
                            "ln_grid": "ln(grid ratio)", "ln_facility": "ln(PUE ratio)",
                            "grid_share_abs": "Grid share of |log difference|"})
    t5[["Site", "Carbon ratio vs Ashburn", "ln(grid ratio)", "ln(PUE ratio)",
        "Grid share of |log difference|", "P(lower than Ashburn)"]].round(3).to_csv(
        C.TAB_DIR / "table5_decomposition.csv", index=False)

    t6 = sweep_air.copy()
    t6["Reduction vs all-at-Ashburn (%)"] = 100 * (1 - t6.co2_t / base["all_reference_co2_t"])
    t6["Reduction vs developer-country siting (%)"] = 100 * (1 - t6.co2_t / base["developer_country_co2_t"])
    t6["Cohort emissions (ktCO2e)"] = t6.co2_t / 1e3
    keep = ["cap_share", "feasible", "Cohort emissions (ktCO2e)", "Reduction vs all-at-Ashburn (%)",
            "Reduction vs developer-country siting (%)"] + [f"share_{s}" for s in C.STUDY_SITES]
    t6 = t6.reindex(columns=keep).rename(columns={"cap_share": "Per-site capacity (share of cohort IT energy)"})
    t6.round(3).to_csv(C.TAB_DIR / "table6_siting.csv", index=False)

    # Supplementary tables
    yearly.to_csv(C.TAB_DIR / "tableS1_interannual.csv", index=False)
    sob.round(4).to_csv(C.TAB_DIR / "tableS2_sobol_carbon.csv", index=False)
    sob_w.round(4).to_csv(C.TAB_DIR / "tableS3_sobol_water.csv", index=False)
    pd.concat([cal["fit"], cal["holdout"]]).to_csv(C.TAB_DIR / "tableS4_calibration_site_years.csv", index=False)
    cohort.to_csv(C.TAB_DIR / "tableS5_cohort.csv", index=False)
    diesel.to_csv(C.TAB_DIR / "tableS6_diesel.csv", index=False)
    sweep_all.to_csv(C.TAB_DIR / "tableS7_siting_with_dtc.csv", index=False)
    par.to_csv(C.TAB_DIR / "tableS8_pareto.csv", index=False)
    grid.to_csv(C.TAB_DIR / "tableS9_grid_intensity.csv", index=False)

    # --------------------------------------------------------- results.json
    ev = summ[summ.arch == "evaporative"].set_index("site")
    dr = summ[summ.arch == "dry"].set_index("site")
    lc = summ[summ.arch == "dtc_dry"].set_index("site")
    order_ev = ev.co2_t_fu_med.sort_values()
    R.update({
        "N_RUNS": str(prov["n_runs"]),
        "COHORT_SOURCE": prov["source"],
        "COHORT_WINDOW": prov["window"],
        "COHORT_TOTAL_FLOP": sci(prov["total_flop"]),
        "COHORT_IT_GWH": fmt(cohort.e_it_kwh.sum() / 1e6, 0),
        "COHORT_LARGEST": f"{cohort.model.iloc[0]} ({sci(cohort.flop.iloc[0])} FLOP)",
        "N_HOURS": f"{len(clim['Oslo']):,}",
        "N_MC": f"{args.n_mc:,}", "N_SOBOL": f"{args.n_sobol:,}",
        "N_SOBOL_EVALS": f"{args.n_sobol * (len(A.SOBOL_KEYS) + 2):,}",
        "CAL_F_ELEC": f"{cal['theta']['f_elec']:.3f}",
        "CAL_T_CHW": f"{cal['theta']['t_chw']:.1f}",
        "CAL_FIT_MAE": f"{cal['metrics_fit']['MAE']:.3f}",
        "CAL_FIT_N": str(cal['metrics_fit']['n']),
        "CAL_HOLD_MAE": f"{cal['metrics_holdout']['MAE']:.3f}",
        "CAL_HOLD_RMSE": f"{cal['metrics_holdout']['RMSE']:.3f}",
        "CAL_HOLD_BIAS": f"{cal['metrics_holdout']['Bias']:+.3f}",
        "CAL_HOLD_N": str(cal['metrics_holdout']['n']),
        "CAL_HOLD_PRIOR_MAE": f"{cal['holdout_prior_MAE']:.3f}",
        "CAL_HOLD_MAPE_OVH": f"{100 * cal['metrics_holdout']['MAPE_overhead']:.0f}",
        "CAL_SPEARMAN": f"{cal['spearman_campus']:.2f}",
        "VAL_GPUH_MED": f"{val['gpu_hours_pred'][1] / 1e6:.1f}",
        "VAL_GPUH_LO": f"{val['gpu_hours_pred'][0] / 1e6:.1f}",
        "VAL_GPUH_HI": f"{val['gpu_hours_pred'][2] / 1e6:.1f}",
        "VAL_GPUH_DIFF": f"{100 * (val['gpu_hours_ratio_median'] - 1):+.0f}",
        "VAL_GPUH_ABSDIFF": f"{abs(100 * (val['gpu_hours_ratio_median'] - 1)):.0f}",
        "VAL_GPUH_IN95": "within" if val["gpu_hours_reported_in_95"] else "outside",
        "VAL_IMPLIED_MFU": f"{val['implied_mfu']:.3f}",
        "VAL_CO2_MED": f"{val['tco2_pred_full_boundary'][1]:,.0f}",
        "VAL_CO2_LO": f"{val['tco2_pred_full_boundary'][0]:,.0f}",
        "VAL_CO2_HI": f"{val['tco2_pred_full_boundary'][2]:,.0f}",
        "VAL_CO2_META_RECOMP": f"{val['tco2_meta_boundary_recomputed']:,.0f}",
        "VAL_E_GWH": f"{val['energy_gwh_pred'][1]:.1f}",
        "FU_E_IT_GWH": f"{e_fu / 1e6:.1f}",
        "FU_E_IT_GWH_LO": f"{np.percentile(P.it_energy_kwh(C.FU_FLOP, mc), 2.5) / 1e6:.1f}",
        "FU_E_IT_GWH_HI": f"{np.percentile(P.it_energy_kwh(C.FU_FLOP, mc), 97.5) / 1e6:.1f}",
        "ORDER_EVAP": " < ".join(order_ev.index),
        "CO2_MIN_SITE": order_ev.index[0], "CO2_MAX_SITE": order_ev.index[-1],
        "CO2_MIN": f"{order_ev.iloc[0]:,.0f}", "CO2_MAX": f"{order_ev.iloc[-1]:,.0f}",
        "CO2_RATIO_MAX_MIN": f"{order_ev.iloc[-1] / order_ev.iloc[0]:.0f}",
        "PUE_EVAP_MIN_SITE": ev.pue_med.idxmin(), "PUE_EVAP_MIN": f"{ev.pue_med.min():.3f}",
        "PUE_EVAP_MAX_SITE": ev.pue_med.idxmax(), "PUE_EVAP_MAX": f"{ev.pue_med.max():.3f}",
        "PUE_EVAP_SPREAD_PCT": f"{100 * (ev.pue_med.max() / ev.pue_med.min() - 1):.1f}",
        "GRID_SPREAD_FACTOR": f"{max(grid_mean.values()) / min(grid_mean.values()):.0f}",
        "DECOMP_MIN_GRID_SHARE": f"{100 * decomp[decomp.site != C.REFERENCE_SITE].grid_share_abs.min():.0f}",
        "DECOMP_MIN_GRID_SHARE_SITE": decomp[decomp.site != C.REFERENCE_SITE].set_index('site').grid_share_abs.idxmin(),
        "DRY_PUE_PENALTY_MAX_SITE": (dr.pue_med / ev.pue_med).idxmax(),
        "DRY_PUE_PENALTY_MAX": f"{100 * ((dr.pue_med / ev.pue_med).max() - 1):.1f}",
        "DRY_CO2_PENALTY_MAX": f"{100 * ((dr.co2_t_fu_med / ev.co2_t_fu_med).max() - 1):.1f}",
        "EVAP_WATER_MAX_SITE": ev.water_m3_fu_med.idxmax(),
        "EVAP_WATER_MAX": f"{ev.water_m3_fu_med.max() / 1e3:.1f}",
        "EVAP_WATER_MIN_SITE": ev.water_m3_fu_med.idxmin(),
        "EVAP_WATER_MIN": f"{ev.water_m3_fu_med.min() / 1e3:.1f}",
        "WUE_EVAP_MAX": f"{ev.wue_med.max():.2f}",
        "DTC_PUE_LAGOS": f"{lc.loc['Lagos', 'pue_med']:.3f}",
        "EVAP_PUE_LAGOS": f"{ev.loc['Lagos', 'pue_med']:.3f}",
        "DTC_CO2_RED_LAGOS": f"{100 * (1 - lc.loc['Lagos', 'co2_t_fu_med'] / ev.loc['Lagos', 'co2_t_fu_med']):.1f}",
        "DTC_CO2_RED_OSLO": f"{100 * (1 - lc.loc['Oslo', 'co2_t_fu_med'] / ev.loc['Oslo', 'co2_t_fu_med']):.1f}",
        "DTC_CO2_RED_MAX_SITE": (1 - lc.co2_t_fu_med / ev.co2_t_fu_med).idxmax(),
        "DTC_CO2_RED_MAX": f"{100 * (1 - lc.co2_t_fu_med / ev.co2_t_fu_med).max():.1f}",
        "LAGOS_VS_ASHBURN_RATIO": f"{ev.loc['Lagos', 'co2_t_fu_med'] / ev.loc['Ashburn', 'co2_t_fu_med']:.2f}",
        "LAGOS_DTC_VS_ASHBURN_EVAP_RATIO": f"{lc.loc['Lagos', 'co2_t_fu_med'] / ev.loc['Ashburn', 'co2_t_fu_med']:.2f}",
    })
    for s in C.STUDY_SITES:
        key = s.upper()
        R[f"CO2_{key}"] = f"{ev.loc[s, 'co2_t_fu_med']:,.0f}"
        R[f"CO2_{key}_CI"] = f"{ev.loc[s, 'co2_t_fu_lo']:,.0f}-{ev.loc[s, 'co2_t_fu_hi']:,.0f}"
        R[f"PUE_{key}"] = f"{ev.loc[s, 'pue_med']:.3f}"
        R[f"PUE_DRY_{key}"] = f"{dr.loc[s, 'pue_med']:.3f}"
        R[f"PUE_DTC_{key}"] = f"{lc.loc[s, 'pue_med']:.3f}"
        R[f"WATER_{key}"] = f"{ev.loc[s, 'water_m3_fu_med'] / 1e3:.1f}"
        R[f"CI_{key}"] = f"{grid_mean[s] * 1000:.0f}"
        if s != C.REFERENCE_SITE:
            R[f"PROB_{key}_LT_REF"] = f"{probs[s]:.2f}"
    # Sobol headline: top driver and its ST at each site
    for s in C.STUDY_SITES:
        d = sob[sob.site == s].sort_values("ST", ascending=False)
        R[f"SOBOL_TOP_{s.upper()}"] = d.param.iloc[0]
        R[f"SOBOL_TOP_ST_{s.upper()}"] = f"{d.ST.iloc[0]:.2f}"
    it_keys = ["mfu", "p_it_per_gpu_kw", "overhead_wallclock"]
    it_share = sob[sob.param.isin(it_keys)].groupby("site").ST.sum()
    R["SOBOL_IT_ST_MIN"] = f"{it_share.min():.2f}"
    R["SOBOL_IT_ST_MAX"] = f"{it_share.max():.2f}"
    fac_keys = [k for k in A.SOBOL_KEYS if k not in it_keys + ["grid_rel_err"]]
    fac_share = sob[sob.param.isin(fac_keys)].groupby("site").ST.sum()
    R["SOBOL_FAC_ST_MAX_SITE"] = fac_share.idxmax()
    R["SOBOL_FAC_ST_MAX"] = f"{fac_share.max():.2f}"
    wtop = sob_w[sob_w.site == R["EVAP_WATER_MAX_SITE"]].sort_values("ST", ascending=False)
    R["SOBOL_WATER_TOP"] = wtop.param.iloc[0]
    R["SOBOL_WATER_TOP_ST"] = f"{wtop.ST.iloc[0]:.2f}"
    # Diesel
    dl = diesel[diesel.arch == "evaporative"].set_index("diesel_share")
    for sh in C.DIESEL_SHARES:
        R[f"LAGOS_DIESEL_{int(sh * 100)}"] = f"{dl.loc[sh, 'co2_med']:,.0f}"
    slope = (dl.co2_med.iloc[-1] - dl.co2_med.iloc[0]) / (100 * (dl.index[-1] - dl.index[0])) * 10
    R["LAGOS_DIESEL_PER_10PP"] = f"{slope:,.0f}"
    R["LAGOS_DIESEL_PER_10PP_PCT"] = f"{100 * slope / dl.co2_med.iloc[0]:.1f}"
    R["DIESEL_EF"] = f"{P.diesel_ef_kg_per_kwh(pc['eta_diesel']):.2f}"
    beij = ev.loc["Beijing", "co2_t_fu_med"]
    if dl.co2_med.iloc[0] >= beij:
        R["LAGOS_BEIJING_DIESEL"] = "already exceeds Beijing without any diesel"
    else:
        xs = dl.index.values * 100
        be = float(np.interp(beij, dl.co2_med.values, xs)) if dl.co2_med.iloc[-1] >= beij else None
        R["LAGOS_BEIJING_DIESEL"] = (f"exceeds Beijing once on-site diesel supplies more than about {be:.0f}% of facility energy"
                                     if be is not None else "remains below Beijing across the diesel range examined")
    # Siting
    R["BASE_ALLREF_KT"] = fmt(base["all_reference_co2_t"] / 1e3, 1)
    R["BASE_DEV_KT"] = fmt(base["developer_country_co2_t"] / 1e3, 1)
    R["SITING_MIN_FEASIBLE"] = f"{100 * pareto_cap:.0f}" if pareto_cap else "n/a"
    for k in C.CAPACITY_SHARES:
        row = sweep_air[(sweep_air.cap_share == k) & (sweep_air.feasible)]
        if len(row):
            R[f"RED_AT_{int(k * 100)}"] = f"{100 * (1 - row.co2_t.iloc[0] / base['all_reference_co2_t']):.1f}"
            R[f"RED_DEV_AT_{int(k * 100)}"] = f"{100 * (1 - row.co2_t.iloc[0] / base['developer_country_co2_t']):.1f}"
        else:
            R[f"RED_AT_{int(k * 100)}"] = "infeasible"
            R[f"RED_DEV_AT_{int(k * 100)}"] = "infeasible"
    if len(par) > 1:
        free = par.iloc[-1] if np.isinf(par.water_cap_m3.iloc[-1]) else par.loc[par.water_m3.idxmax()]
        dry = par.iloc[0]
        R["PARETO_WATER_CUT"] = f"{100 * (1 - dry.water_m3 / max(free.water_m3, 1e-9)):.0f}"
        R["PARETO_CO2_COST"] = f"{100 * (dry.co2_t / free.co2_t - 1):.1f}"
    else:
        R["PARETO_WATER_CUT"] = R["PARETO_CO2_COST"] = "n/a"
    gaps = pd.concat([sweep_air, sweep_all]).get("mip_gap")
    R["MAX_MIP_GAP_PCT"] = f"{100 * np.nanmax(gaps.values):.2f}" if gaps is not None and gaps.notna().any() else "0.00"
    at_bound = [k for k, v in cal["theta"].items()
                if abs(v - A.FIT_BOUNDS[k][0]) < 1e-3 or abs(v - A.FIT_BOUNDS[k][1]) < 1e-3]
    R["CAL_BOUND_NOTE"] = ("" if not at_bound else
        " The calibrated value of " + " and ".join(at_bound) + " lies on the edge of its admissible range, "
        "which indicates that the reported campuses operate more efficiently than the model structure can "
        "represent within physically credible set-points; the resulting estimates are therefore conservative "
        "(slightly high) for best-in-class facilities.")
    mf = sweep_air[sweep_air.feasible]
    R["RED_AT_MINFEAS"] = f"{100 * (1 - mf.co2_t.iloc[0] / base['all_reference_co2_t']):.1f}" if len(mf) else "n/a"
    R["RED_DEV_AT_MINFEAS"] = f"{100 * (1 - mf.co2_t.iloc[0] / base['developer_country_co2_t']):.1f}" if len(mf) else "n/a"
    R["RED_AT_100"] = R.get("RED_AT_100", "n/a")
    lg = cohort.flop.iloc[0] / cohort.flop.sum()
    R["LARGEST_RUN_SHARE"] = f"{100 * lg:.0f}"
    R["RUNTIME_MIN"] = f"{(time.time() - t0) / 60:.1f}"
    R["GRID_YEARS_FILLED"] = ", ".join(
        f"{r.country} {int(r.year)}" for r in grid[grid.get("filled_from").notna()].itertuples()) \
        if "filled_from" in grid else "none"

    # Final figures 2, 4, 5 and 6 are drawn from the output tables, and derived
    # reporting quantities (e.g. interannual swings, DTC comparisons) are added to results.json.
    from ai_siting import derived
    F.make_table_figures(C.TAB_DIR, C.FIG_DIR, args.demo)
    R = derived.derive(R, C.TAB_DIR)
    (C.OUT_DIR / "results.json").write_text(json.dumps(R, indent=2, default=str))
    print(f"results.json written with {len(R)} keys in {R['RUNTIME_MIN']} min")


if __name__ == "__main__":
    sys.exit(main())
