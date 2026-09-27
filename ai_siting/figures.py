"""Publication figures (300 dpi PNG, colour-blind-safe palette, no chart junk)."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config as C

PAL = {"Oslo": "#0072B2", "Paris": "#56B4E9", "Ashburn": "#009E73",
       "Beijing": "#E69F00", "Marrakech": "#CC79A7", "Lagos": "#D55E00"}
ARCH_MARK = {"evaporative": "o", "dry": "s", "dtc_dry": "^"}
ARCH_SHORT = {"evaporative": "Air + evaporative", "dry": "Air + dry", "dtc_dry": "Liquid (DTC) + dry"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def _watermark(fig, demo):
    if demo:
        fig.text(0.5, 0.5, "DEMO DATA - NOT FOR PUBLICATION", ha="center", va="center",
                 fontsize=22, color="red", alpha=0.35, rotation=25)


def fig_climate(clim, path, demo=False):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    sites = list(C.STUDY_SITES)
    for ax, col, lab in zip(axes, ["t_db", "t_wb"], ["Dry-bulb temperature (°C)", "Wet-bulb temperature (°C)"]):
        data = [clim[s][col].values for s in sites]
        parts = ax.violinplot(data, showextrema=False, showmedians=True)
        for body, s in zip(parts["bodies"], sites):
            body.set_facecolor(PAL[s]); body.set_alpha(0.75)
        ax.set_xticks(range(1, len(sites) + 1)); ax.set_xticklabels(sites, rotation=30, ha="right")
        ax.set_ylabel(lab)
    axes[0].set_title("a", loc="left", fontweight="bold"); axes[1].set_title("b", loc="left", fontweight="bold")
    _watermark(fig, demo); fig.savefig(path); plt.close(fig)


def fig_calibration(cal, path, demo=False):
    fig, ax = plt.subplots(figsize=(3.5, 3.3))
    cm = cal["campus_means"]
    for split, mk in [("fit", "o"), ("holdout", "D")]:
        d = cm[cm.split == split]
        ax.scatter(d.ttm_pue, d.pred, marker=mk, s=28, label="Fit" if split == "fit" else "Hold-out",
                   facecolor="none" if split == "holdout" else "#555555", edgecolor="#222222")
    lo = min(cm.ttm_pue.min(), cm.pred.min()) - 0.01
    hi = max(cm.ttm_pue.max(), cm.pred.max()) + 0.01
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Reported TTM PUE (Google, 2021-2025 mean)")
    ax.set_ylabel("Modelled annual PUE")
    ax.legend(frameon=False, loc="upper left")
    _watermark(fig, demo); fig.savefig(path); plt.close(fig)


def fig_carbon_by_site(summ, path, demo=False):
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    sites = list(C.STUDY_SITES)
    width = 0.26
    for j, arch in enumerate(C.ARCHITECTURES):
        d = summ[summ.arch == arch].set_index("site").loc[sites]
        x = np.arange(len(sites)) + (j - 1) * width
        ax.bar(x, d.co2_t_fu_med, width, color=[PAL[s] for s in sites],
               alpha=[1.0, 0.65, 0.35][j], edgecolor="black", linewidth=0.4, label=ARCH_SHORT[arch])
        ax.errorbar(x, d.co2_t_fu_med, yerr=[d.co2_t_fu_med - d.co2_t_fu_lo, d.co2_t_fu_hi - d.co2_t_fu_med],
                    fmt="none", ecolor="black", elinewidth=0.7, capsize=2)
    ax.set_xticks(range(len(sites))); ax.set_xticklabels(sites)
    ax.set_ylabel("tCO$_2$e per 10$^{25}$ FLOP")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor="grey", alpha=a, edgecolor="black", linewidth=0.4)
               for a in [1.0, 0.65, 0.35]]
    ax.legend(handles, [ARCH_SHORT[a] for a in C.ARCHITECTURES], frameon=False, ncol=3,
              loc="upper left")
    _watermark(fig, demo); fig.savefig(path); plt.close(fig)


def fig_carbon_water(summ, path, demo=False):
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for _, r in summ.iterrows():
        ax.errorbar(r.water_m3_fu_med / 1000, r.co2_t_fu_med,
                    yerr=[[r.co2_t_fu_med - r.co2_t_fu_lo], [r.co2_t_fu_hi - r.co2_t_fu_med]],
                    xerr=[[(r.water_m3_fu_med - r.water_m3_fu_lo) / 1000], [(r.water_m3_fu_hi - r.water_m3_fu_med) / 1000]],
                    fmt=ARCH_MARK[r.arch], color=PAL[r.site], ms=5, elinewidth=0.6, capsize=0)
    ax.set_xlabel("On-site water consumption (thousand m$^3$ per 10$^{25}$ FLOP)")
    ax.set_ylabel("tCO$_2$e per 10$^{25}$ FLOP")
    ax.set_yscale("log")
    s_handles = [plt.Line2D([], [], marker="o", ls="", color=PAL[s], label=s) for s in C.STUDY_SITES]
    a_handles = [plt.Line2D([], [], marker=ARCH_MARK[a], ls="", color="grey", label=ARCH_SHORT[a]) for a in C.ARCHITECTURES]
    leg1 = ax.legend(handles=s_handles, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.add_artist(leg1)
    ax.legend(handles=a_handles, frameon=False, loc="lower left", bbox_to_anchor=(1.0, 0.0))
    _watermark(fig, demo); fig.savefig(path); plt.close(fig)


def fig_sobol(sob, path, demo=False):
    piv = sob.pivot(index="param", columns="site", values="ST")[list(C.STUDY_SITES)]
    piv = piv.loc[piv.max(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    im = ax.imshow(piv.values, cmap="Greys", vmin=0, vmax=max(0.01, piv.values.max()), aspect="auto")
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns, rotation=30, ha="right")
    ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if v > 0.5 * piv.values.max() else "black")
    fig.colorbar(im, ax=ax, label="Total-order Sobol index S$_T$")
    _watermark(fig, demo); fig.savefig(path); plt.close(fig)


def fig_diesel_and_siting(diesel, summ, sweep, pareto_df, baseline, path, demo=False):
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7))
    ax = axes[0]
    d = diesel[diesel.arch == "evaporative"]
    ax.plot(d.diesel_share * 100, d.co2_med, "-o", color=PAL["Lagos"], ms=3, label="Lagos")
    ax.fill_between(d.diesel_share * 100, d.co2_lo, d.co2_hi, color=PAL["Lagos"], alpha=0.2, lw=0)
    ev = summ[summ.arch == "evaporative"].set_index("site")
    for s in ["Ashburn", "Beijing", "Marrakech"]:
        ax.axhline(ev.loc[s, "co2_t_fu_med"], color=PAL[s], ls="--", lw=0.8, label=s)
    ax.set_xlabel("On-site diesel share (%)"); ax.set_ylabel("tCO$_2$e per 10$^{25}$ FLOP")
    ax.legend(frameon=False, fontsize=7); ax.set_title("a", loc="left", fontweight="bold")

    ax = axes[1]
    f = sweep[sweep.feasible]
    ax.plot(f.cap_share * 100, 100 * (1 - f.co2_t / baseline["all_reference_co2_t"]), "-o", color="black", ms=3)
    ax.set_xlabel("Per-site capacity (% of cohort IT energy)")
    ax.set_ylabel("Reduction vs all-at-reference (%)")
    ax.set_ylim(0, 100); ax.set_title("b", loc="left", fontweight="bold")

    ax = axes[2]
    if len(pareto_df):
        ax.plot(pareto_df.water_m3 / 1e6, pareto_df.co2_t / 1e3, "-o", color="black", ms=3)
    ax.set_xlabel("On-site water (million m$^3$)"); ax.set_ylabel("Cohort emissions (ktCO$_2$e)")
    ax.set_title("c", loc="left", fontweight="bold")
    fig.tight_layout()
    _watermark(fig, demo); fig.savefig(path); plt.close(fig)


# ===========================================================================
# Table-driven publication figures (rebuilt from outputs/tables by
# run_all.py; they can be redrawn from existing tables without re-running the model)
# ===========================================================================
def _tbl_calibration(tab_dir, path, demo=False):
    s4 = pd.read_csv(tab_dir / "tableS4_calibration_site_years.csv")
    cm = s4.groupby(["campus", "split"])[["ttm_pue", "pred"]].mean().reset_index()
    cm["res"] = cm.pred - cm.ttm_pue
    fig, ax = plt.subplots(figsize=(3.8, 3.6))
    for split, mk, face in [("fit", "o", "#555555"), ("holdout", "D", "none")]:
        d = cm[cm.split == split]
        ax.scatter(d.ttm_pue, d.pred, marker=mk, s=28, facecolor=face, edgecolor="#222222",
                   label="Fit (7 campuses)" if split == "fit" else "Hold-out (7 campuses)")
    for _, r in cm[cm.res.abs() >= 0.02].iterrows():
        ax.annotate(r.campus.split(",")[0], (r.ttm_pue, r.pred), xytext=(4, -10),
                    textcoords="offset points", fontsize=7)
    lo = min(cm.ttm_pue.min(), cm.pred.min()) - 0.01
    hi = max(cm.ttm_pue.max(), cm.pred.max()) + 0.01
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
    ax.fill_between([lo, hi], [lo - 0.01, hi - 0.01], [lo + 0.01, hi + 0.01], color="grey", alpha=0.12, lw=0)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Reported PUE (Google, 2021–2025 mean)")
    ax.set_ylabel("Modelled annual PUE")
    ax.legend(frameon=False, loc="upper left", fontsize=7)
    _watermark(fig, demo); fig.savefig(path); plt.close(fig)


def _tbl_sobol(tab_dir, path, demo=False):
    from .derived import IT_KEYS, GRID_KEYS, SHORT
    s2 = pd.read_csv(tab_dir / "tableS2_sobol_carbon.csv")
    s3 = pd.read_csv(tab_dir / "tableS3_sobol_water.csv")
    sites = list(C.STUDY_SITES)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), gridspec_kw={"width_ratios": [1, 1.25]})
    groups = [("IT energy", IT_KEYS, "#444444"), ("Grid intensity", GRID_KEYS, "#E69F00")]
    for k, (df, ttl) in enumerate([(s2, "a  Emissions per FU"), (s3, "b  On-site water per FU (evaporative)")]):
        ax = axes[k]
        bottom = np.zeros(len(sites))
        fac_keys = [p for p in df.param.unique() if p not in IT_KEYS + GRID_KEYS]
        piv = df.pivot(index="param", columns="site", values="ST")[sites].clip(lower=0)
        layers = list(groups)
        if k == 1:
            top_fac = piv.loc[fac_keys].max(axis=1).sort_values(ascending=False).index[:3]
            pal = ["#0072B2", "#56B4E9", "#009E73"]
            layers = [groups[0]] + [(SHORT[p], [p], pal[i]) for i, p in enumerate(top_fac)] + \
                     [("Other facility", [p for p in fac_keys if p not in top_fac], "#CCCCCC")]
        else:
            layers = layers + [("All facility parameters", fac_keys, "#009E73")]
        for name, keys, col in layers:
            v = piv.loc[[p for p in keys if p in piv.index]].sum(axis=0).values
            ax.bar(sites, v, bottom=bottom, color=col, edgecolor="white", linewidth=0.4, label=name)
            bottom += v
        ax.set_ylabel("Total-order index S$_T$")
        ax.set_title(ttl, loc="left", fontweight="bold")
        ax.set_xticks(range(len(sites))); ax.set_xticklabels(sites, rotation=30, ha="right")
        ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(0, -0.28), ncol=2)
    fig.tight_layout()
    _watermark(fig, demo); fig.savefig(path); plt.close(fig)


def _tbl_carbon_water(tab_dir, path, demo=False):
    from .derived import site_summary
    ss = site_summary(tab_dir)
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    offset = {"evaporative": 0.0, "dry": -0.35, "dtc_dry": 0.35}   # thousand m3, only for zero-water options
    for _, r in ss.iterrows():
        x = r.water_k_med + (offset[r.arch] if r.water_k_med == 0 else 0.0)
        xerr = [[r.water_k_med - r.water_k_lo], [r.water_k_hi - r.water_k_med]]
        ax.errorbar(x, r.co2_med, yerr=[[r.co2_med - r.co2_lo], [r.co2_hi - r.co2_med]], xerr=xerr,
                    fmt=ARCH_MARK[r.arch], color=PAL[r.site], ms=5, elinewidth=0.6, capsize=0,
                    mfc="white" if r.arch == "dtc_dry" else PAL[r.site])
    ax.set_xlabel("On-site water (thousand m$^3$ per 10$^{25}$ FLOP)")
    ax.set_ylabel("tCO$_2$e per 10$^{25}$ FLOP")
    ax.set_yscale("log")
    s_handles = [plt.Line2D([], [], marker="o", ls="", color=PAL[s], label=s) for s in C.STUDY_SITES]
    a_handles = [plt.Line2D([], [], marker=ARCH_MARK[a], ls="", color="grey",
                            mfc="white" if a == "dtc_dry" else "grey", label=ARCH_SHORT[a]) for a in C.ARCHITECTURES]
    leg1 = ax.legend(handles=s_handles, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.add_artist(leg1)
    ax.legend(handles=a_handles, frameon=False, loc="lower left", bbox_to_anchor=(1.0, 0.0))
    _watermark(fig, demo); fig.savefig(path); plt.close(fig)


def _tbl_diesel_siting(tab_dir, path, demo=False):
    from .derived import site_summary
    ss = site_summary(tab_dir).set_index(["site", "arch"])
    s6 = pd.read_csv(tab_dir / "tableS6_diesel.csv")
    t6 = pd.read_csv(tab_dir / "table6_siting.csv")
    s8 = pd.read_csv(tab_dir / "tableS8_pareto.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7))
    ax = axes[0]
    d = s6[s6.arch == "evaporative"].sort_values("diesel_share")
    ax.plot(d.diesel_share * 100, d.co2_med, "-o", color=PAL["Lagos"], ms=3, label="Lagos")
    ax.fill_between(d.diesel_share * 100, d.co2_lo, d.co2_hi, color=PAL["Lagos"], alpha=0.18, lw=0)
    for s in ["Ashburn", "Beijing", "Marrakech"]:
        ax.axhline(ss.loc[(s, "evaporative"), "co2_med"], color=PAL[s], ls="--", lw=0.8, label=s)
    ax.set_xlabel("On-site diesel share (%)"); ax.set_ylabel("tCO$_2$e per 10$^{25}$ FLOP")
    ax.legend(frameon=False, fontsize=6.5, loc="upper left"); ax.set_title("a", loc="left", fontweight="bold")

    ax = axes[1]
    f = t6[t6.feasible == True]
    ax.plot(f.iloc[:, 0] * 100, f["Reduction vs all-at-Ashburn (%)"], "-o", color="black", ms=3,
            label="vs all at Ashburn")
    ax.plot(f.iloc[:, 0] * 100, f["Reduction vs developer-country siting (%)"], "--s", color="grey", ms=3,
            label="vs developer country")
    ax.set_xlabel("Per-site capacity (% of cohort IT energy)")
    ax.set_ylabel("Emission reduction (%)"); ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=6.5, loc="lower right"); ax.set_title("b", loc="left", fontweight="bold")

    ax = axes[2]
    if len(s8) > 1:
        free = s8[np.isinf(s8.water_cap_m3)].iloc[0] if np.isinf(s8.water_cap_m3).any() else s8.loc[s8.water_m3.idxmax()]
        x = 100 * (1 - s8.water_m3 / free.water_m3)
        y = 100 * (s8.co2_t / free.co2_t - 1)
        o = np.argsort(x.values)
        ax.plot(x.values[o], y.values[o], "-o", color="black", ms=3)
    ax.set_xlabel("On-site water avoided (%)"); ax.set_ylabel("Emission increase (%)")
    ax.set_xlim(-2, 102); ax.set_ylim(bottom=0)
    ax.set_title("c", loc="left", fontweight="bold")
    fig.tight_layout()
    _watermark(fig, demo); fig.savefig(path); plt.close(fig)


def make_table_figures(tab_dir, fig_dir, demo=False):
    """Regenerate Figs. 2, 4, 5 and 6 from the output tables."""
    _tbl_calibration(tab_dir, fig_dir / "fig2_calibration.png", demo)
    _tbl_sobol(tab_dir, fig_dir / "fig4_sobol.png", demo)
    _tbl_carbon_water(tab_dir, fig_dir / "fig5_carbon_water.png", demo)
    _tbl_diesel_siting(tab_dir, fig_dir / "fig6_diesel_siting.png", demo)
