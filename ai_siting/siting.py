"""
Capacity-constrained siting of a cohort of training runs.

Decision: assign each (indivisible) run r to one option o = (site, cooling architecture).
Capacity is expressed in IT energy (kWh) per site, shared across architectures at that
site, so large runs consume proportionally more capacity than small ones.

    min   sum_{r,o} c_{r,o} x_{r,o}
    s.t.  sum_o x_{r,o} = 1                      for every run r
          sum_{r, o in site s} E_r x_{r,o} <= K_s  for every site s
          [optional] sum_{r,o} w_{r,o} x_{r,o} <= W   (water epsilon-constraint)
          x binary

Solved with HiGHS through scipy.optimize.milp.
"""
import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix

from . import config as C


def build_options(summ, archs):
    s = summ[summ.arch.isin(archs)].copy()
    # per-kWh_IT coefficients are Monte Carlo medians (see analysis.summarise_sites)
    return s[["site", "arch", "co2_per_kwh_it", "water_per_kwh_it"]].reset_index(drop=True)


MIP_REL_GAP = 1e-3   # reported in the paper (Section 2.8)


def _prune(options, water_cap):
    """Remove options that can never be optimal. Without a water constraint only the
    lowest-carbon architecture at each site matters; with one, drop options that are
    dominated in both carbon and water by another option at the same site."""
    if water_cap is None:
        idx = options.groupby("site").co2_per_kwh_it.idxmin()
        return options.loc[idx].reset_index(drop=True)
    keep = []
    for _, g in options.groupby("site"):
        for i, r in g.iterrows():
            dom = ((g.co2_per_kwh_it <= r.co2_per_kwh_it) & (g.water_per_kwh_it <= r.water_per_kwh_it)
                   & ((g.co2_per_kwh_it < r.co2_per_kwh_it) | (g.water_per_kwh_it < r.water_per_kwh_it))).any()
            if not dom:
                keep.append(i)
    return options.loc[keep].reset_index(drop=True)


def solve(runs_e_it, options, cap_share, water_cap=None, time_limit=20):
    options = _prune(options, water_cap)
    R = len(runs_e_it)
    O = len(options)
    sites = list(C.STUDY_SITES)
    c = np.outer(runs_e_it, options.co2_per_kwh_it.values).ravel()     # tCO2
    w = np.outer(runs_e_it, options.water_per_kwh_it.values).ravel()   # m3
    K = cap_share * runs_e_it.sum()
    if runs_e_it.max() > K + 1e-9:
        return None  # a single run exceeds a site's capacity: infeasible by construction

    nvar = R * O
    rows = R + len(sites) + (1 if water_cap is not None else 0)
    A = lil_matrix((rows, nvar))
    lb = np.zeros(rows)
    ub = np.zeros(rows)
    for r in range(R):
        A[r, r * O:(r + 1) * O] = 1
        lb[r] = ub[r] = 1
    for j, s in enumerate(sites):
        idx = np.where(options.site.values == s)[0]
        for r in range(R):
            for o in idx:
                A[R + j, r * O + o] = runs_e_it[r]
        lb[R + j], ub[R + j] = 0, K
    if water_cap is not None:
        A[rows - 1, :] = w
        lb[rows - 1], ub[rows - 1] = 0, water_cap
    res = milp(c=c, constraints=LinearConstraint(A.tocsr(), lb, ub),
               integrality=np.ones(nvar), bounds=Bounds(0, 1),
               options={"time_limit": time_limit, "disp": False, "mip_rel_gap": MIP_REL_GAP})
    if not res.success and res.x is None:
        return None
    x = np.round(res.x).reshape(R, O)
    choice = x.argmax(axis=1)
    return {
        "co2_t": float((x.ravel() * c).sum()),
        "water_m3": float((x.ravel() * w).sum()),
        "assign_site": options.site.values[choice],
        "assign_arch": options.arch.values[choice],
        "mip_gap": float(getattr(res, "mip_gap", np.nan) or 0.0),
        "lower_bound": float(getattr(res, "mip_dual_bound", np.nan) or np.nan),
        "status": res.message,
    }


def baselines(runs, options_evap):
    """Two reference policies, both with the evaporative air-cooled architecture:
    (i) all runs at the reference site; (ii) each run at the study site in the
    developer's country where one exists, otherwise the reference site."""
    per = options_evap.set_index("site")["co2_per_kwh_it"]
    wat = options_evap.set_index("site")["water_per_kwh_it"]
    ref = C.REFERENCE_SITE
    e = runs.e_it_kwh.values
    all_ref = float((e * per[ref]).sum())
    country_to_site = {v[2]: k for k, v in C.STUDY_SITES.items()}

    def home(c):
        if isinstance(c, str):
            for cc, s in country_to_site.items():
                if cc.lower() in c.lower():
                    return s
        return ref
    homes = runs.country.map(home)
    dev = float(sum(e_i * per[h] for e_i, h in zip(e, homes)))
    return {"all_reference_co2_t": all_ref,
            "all_reference_water_m3": float((e * wat[ref]).sum()),
            "developer_country_co2_t": dev,
            "developer_country_sites": homes.value_counts().to_dict()}


def capacity_sweep(runs, options, shares=C.CAPACITY_SHARES):
    out = []
    for k in shares:
        sol = solve(runs.e_it_kwh.values, options, k)
        if sol is None:
            out.append({"cap_share": k, "feasible": False})
            continue
        sites = pd.Series(sol["assign_site"])
        e_by_site = pd.Series(runs.e_it_kwh.values).groupby(sites).sum() / runs.e_it_kwh.sum()
        rec = {"cap_share": k, "feasible": True, "co2_t": sol["co2_t"], "water_m3": sol["water_m3"],
               "mip_gap": sol["mip_gap"]}
        for s in C.STUDY_SITES:
            rec[f"share_{s}"] = float(e_by_site.get(s, 0.0))
        out.append(rec)
    return pd.DataFrame(out)


def pareto(runs, options, cap_share, n_points=9):
    """Carbon-water trade-off by epsilon-constraint on total on-site water."""
    free = solve(runs.e_it_kwh.values, options, cap_share)
    if free is None:
        return pd.DataFrame()
    w_max = free["water_m3"]
    pts = [{"water_cap_m3": np.inf, "co2_t": free["co2_t"], "water_m3": free["water_m3"]}]
    for f in np.linspace(0.0, 1.0, n_points)[:-1]:
        sol = solve(runs.e_it_kwh.values, options, cap_share, water_cap=f * w_max)
        if sol is not None:
            pts.append({"water_cap_m3": f * w_max, "co2_t": sol["co2_t"], "water_m3": sol["water_m3"]})
    return pd.DataFrame(pts).sort_values("water_m3")
