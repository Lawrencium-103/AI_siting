"""
Reporting quantities derived from the output tables (outputs/tables/*.csv).

run_all.py calls derive() after writing the tables, so these keys are
recomputed from whatever tables exist. No number here is typed by hand.
"""
import re
import numpy as np
import pandas as pd

IT_KEYS = ["mfu", "p_it_per_gpu_kw", "overhead_wallclock"]
GRID_KEYS = ["grid_rel_err"]

# Display names used in tables, figures and text
SYMBOL = {
    "mfu": "MFU, u", "p_it_per_gpu_kw": "IT power per GPU, P_IT", "overhead_wallclock": "Wall-clock overhead, o",
    "f_elec": "Electrical overhead, f_el", "f_air": "Air-handler fans, f_air", "t_chw": "Chilled-water supply, T_chw",
    "a_tower": "Tower approach, Δ_tower", "a_dry": "Dry-cooler approach, Δ_dry", "eta_carnot": "Chiller Carnot fraction, η",
    "e_pump_tower": "Tower pumps and fans", "e_dry_fan": "Dry-cooler fans", "cycles_conc": "Cycles of concentration, n_c",
    "phi_liquid": "Liquid-captured heat, φ", "t_liquid": "Liquid-loop supply, T_liq", "grid_rel_err": "Grid intensity error, ε",
    "eta_diesel": "Diesel genset efficiency, η_d",
}
SHORT = {k: v.split(",")[0] for k, v in SYMBOL.items()}


def _num(s):
    return float(str(s).replace(",", ""))


def _parse_ci(s):
    m = re.match(r"\s*([\d,.\-]+)\s*\(([\d,.\-]+)-([\d,.]+)\)", str(s))
    return tuple(_num(x) for x in m.groups())


def site_summary(tab_dir):
    """Numeric medians/intervals reconstructed from table4 (exact to the printed precision)."""
    t4 = pd.read_csv(tab_dir / "table4_site_results.csv")
    arch_of = {"Air + hybrid cooling tower (evaporative)": "evaporative",
               "Air + dry cooler / air-cooled chiller": "dry",
               "Direct-to-chip liquid + dry cooler": "dtc_dry"}
    rows = []
    for _, r in t4.iterrows():
        d = {"site": r["Site"], "arch": arch_of[r["Architecture"]]}
        for col, key in [("PUE", "pue"), ("WUE (L/kWh_IT)", "wue"),
                         ("tCO2e per 1e25 FLOP", "co2"), ("Water (thousand m3 per 1e25 FLOP)", "water_k")]:
            d[f"{key}_med"], d[f"{key}_lo"], d[f"{key}_hi"] = _parse_ci(r[col])
        rows.append(d)
    return pd.DataFrame(rows)


def _pct(x, nd=1):
    return f"{x:.{nd}f}"


def derive(R, tab_dir):
    R = dict(R)
    # ---- calibration residuals -------------------------------------------------
    s4 = pd.read_csv(tab_dir / "tableS4_calibration_site_years.csv")
    s4["res"] = s4.pred - s4.ttm_pue
    camp = s4.groupby("campus").res.mean()
    worst = camp.abs().idxmax()
    R["CAL_WORST_CAMPUS"] = worst
    R["CAL_WORST_RES"] = f"{abs(camp[worst]):.3f}"
    R["CAL_OTHER_MAXABS"] = f"{camp.drop(worst).abs().max():.3f}"
    R["CAL_WORST_DIRECTION"] = "overpredicts" if camp[worst] > 0 else "underpredicts"
    R["CAL_TROPICAL_SENTENCE"] = (
        " Singapore is the only tropical campus in the calibration set, so modelled PUE for Lagos is probably "
        "biased high by a similar margin; the climate penalty reported for Lagos should be read as an upper bound."
        if worst == "Singapore" and camp[worst] > 0 else "")

    # ---- Sobol, grouped ------------------------------------------------------------
    s2 = pd.read_csv(tab_dir / "tableS2_sobol_carbon.csv")
    s3 = pd.read_csv(tab_dir / "tableS3_sobol_water.csv")
    fac = lambda df: df[~df.param.isin(IT_KEYS + GRID_KEYS)]
    top = s2.sort_values("ST", ascending=False).groupby("site").head(1)
    same_top = top.param.nunique() == 1
    R["SOBOL_TOP_NAME"] = SHORT.get(top.param.iloc[0], top.param.iloc[0]) if same_top else "varies by site"
    R["SOBOL_TOP_ST_RANGE"] = (f"{top.ST.min():.2f}" if abs(top.ST.max() - top.ST.min()) < 0.005
                               else f"{top.ST.min():.2f}–{top.ST.max():.2f}")
    R["SOBOL_GRID_ST"] = f"{s2[s2.param.isin(GRID_KEYS)].ST.mean():.2f}"
    R["SOBOL_FAC_ST_MAXSUM"] = f"{fac(s2).groupby('site').ST.sum().max():.3f}"
    wf = fac(s3).groupby("site").ST.sum()
    wi = s3[s3.param.isin(IT_KEYS)].groupby("site").ST.sum()
    R["SOBOL_WATER_FAC_MAX_SITE"] = wf.idxmax()
    R["SOBOL_WATER_FAC_MAX"] = f"{wf.max():.2f}"
    R["SOBOL_WATER_FAC_MIN_SITE"] = wf.idxmin()
    R["SOBOL_WATER_FAC_MIN"] = f"{wf.min():.2f}"
    R["SOBOL_WATER_IT_AT_MIN"] = f"{wi[wf.idxmin()]:.2f}"
    tf = fac(s3[s3.site == wf.idxmax()]).sort_values("ST", ascending=False)
    R["SOBOL_WATER_FAC_TOP2"] = f"{SHORT[tf.param.iloc[0]].lower()} and {SHORT[tf.param.iloc[1]].lower()}"

    # ---- sites, DTC ----------------------------------------------------------------
    ss = site_summary(tab_dir).set_index(["site", "arch"])
    ev = ss.xs("evaporative", level="arch")
    lc = ss.xs("dtc_dry", level="arch")
    dr = ss.xs("dry", level="arch")
    red = 100 * (1 - lc.co2_med / ev.co2_med)
    neg = red[red < -0.05]

    def join(xs):
        xs = list(xs)
        return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]
    s = f"Relative to evaporative cooling, liquid cooling reduces emissions most in {red.idxmax()} ({red.max():.1f}%)"
    if len(neg):
        s += (f"; in {join(neg.index)} it increases them slightly (by {join(f'{-v:.1f}%' for v in neg)}), "
              "because the evaporative plant there already rejects heat very efficiently and the liquid loop "
              "adds pumping and dry-cooler fan power")
    R["DTC_SENTENCE"] = s + "."
    vs_dry = 100 * (1 - lc.pue_med / dr.pue_med)
    R["DTC_VS_DRY_MIN"] = f"{vs_dry.min():.1f}"
    R["DTC_VS_DRY_MAX"] = f"{vs_dry.max():.1f}"
    R["DTC_VS_DRY_MAX_SITE"] = vs_dry.idxmax()
    R["DTC_BEATS_DRY_ALL"] = "every" if (vs_dry > 0).all() else "most"
    gap = ev.pue_med["Lagos"] - ev.pue_med.min()
    R["DTC_GAP_CLOSED_PCT"] = f"{100 * (ev.pue_med['Lagos'] - lc.pue_med['Lagos']) / gap:.0f}"
    R["PUE_EVAP_MIN_VAL"] = f"{ev.pue_med.min():.3f}"

    # ---- probabilities as words --------------------------------------------------------
    t5 = pd.read_csv(tab_dir / "table5_decomposition.csv").set_index("Site")
    p = t5["P(lower than Ashburn)"].dropna()
    always = [s for s, v in p.items() if v >= 0.995]
    never = [s for s, v in p.items() if v <= 0.005]
    other = [f"{s} in {v:.0%} of draws" for s, v in p.items() if 0.005 < v < 0.995]
    parts = []
    if always:
        parts.append(f"{' and '.join(always) if len(always) < 3 else ', '.join(always)} emitted less than Ashburn in every draw")
    if never:
        parts.append(f"{', '.join(never[:-1]) + ' and ' + never[-1] if len(never) > 1 else never[0]} in none")
    if other:
        parts.append("; ".join(other))
    R["PROB_SENTENCE"] = "; ".join(parts) + "."

    # ---- cohort, capacity in MW --------------------------------------------------------
    s5 = pd.read_csv(tab_dir / "tableS5_cohort.csv")
    win = R.get("COHORT_WINDOW", "")
    m = re.match(r"(\d{4}-\d{2}-\d{2}) to (\d{4}-\d{2}-\d{2})", win)
    if m:
        hours = (pd.Timestamp(m.group(2)) - pd.Timestamp(m.group(1))).days * 24 + 24
        mw = s5.e_it_kwh.sum() / 1e3 / hours
        R["COHORT_MEAN_MW"] = f"{mw:.0f}"
        R["CAP_MINFEAS_MW"] = f"{mw * _num(R['SITING_MIN_FEASIBLE']) / 100:.0f}"
        R["CAP_50_MW"] = f"{mw * 0.5:.0f}"
    else:
        R["COHORT_MEAN_MW"] = R["CAP_MINFEAS_MW"] = R["CAP_50_MW"] = "n/a"
    first = s5.country.astype(str).str.split(",").str[0].str.strip()
    vc = first.value_counts()
    R["COHORT_BY_COUNTRY"] = ", ".join(f"{c.replace(' of America', '')} {n}" for c, n in vc.head(3).items()) + \
        (f", other {vc.iloc[3:].sum()}" if len(vc) > 3 else "")

    # ---- interannual variability (Table S1) ------------------------------------------
    s1 = pd.read_csv(tab_dir / "tableS1_interannual.csv")
    s1 = s1[s1.arch == "evaporative"]
    g = s1.groupby("site").agg(pmin=("pue", "min"), pmax=("pue", "max"), cmin=("ci_ember", "min"), cmax=("ci_ember", "max"))
    pue_sw = 100 * (g.pmax / g.pmin - 1)
    ci_sw = 100 * (g.cmax / g.cmin - 1)
    top = ci_sw.idxmax()
    rest = ci_sw.drop(top)
    R["IAV_PUE_MAX"] = f"{pue_sw.max():.1f}"
    R["IAV_GRID_MIN"] = f"{rest.min():.0f}"
    R["IAV_GRID_MAX"] = f"{rest.max():.0f}"
    R["IAV_GRID_TOP_SITE"] = top
    R["IAV_GRID_TOP"] = f"{ci_sw.max():.0f}"

    # ---- siting with liquid cooling ----------------------------------------------------
    t6 = pd.read_csv(tab_dir / "table6_siting.csv")
    s7 = pd.read_csv(tab_dir / "tableS7_siting_with_dtc.csv")
    a = t6[t6.feasible == True].iloc[0]
    b = s7[s7.feasible == True].iloc[0]
    ch = 100 * (b.co2_t / 1e3 / a['Cohort emissions (ktCO2e)'] - 1)
    R["DTC_COHORT_CHANGE_PCT"] = f"{abs(ch):.1f}"
    R["DTC_COHORT_VERB"] = "reduces" if ch < 0 else "increases"
    return R
