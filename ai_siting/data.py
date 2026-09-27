"""
Data acquisition. All external data are downloaded once and cached under
data/raw so every later run is offline and bit-for-bit repeatable.
"""
import time
import json
import numpy as np
import pandas as pd
import requests

from . import config as C


# ---------------------------------------------------------------------------
# Climate: Open-Meteo historical archive (ERA5-based reanalysis), hourly
# ---------------------------------------------------------------------------
def _slug(name):
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_").lower()


def fetch_climate(name, lat, lon, years=C.YEARS, demo=False):
    """Hourly 2 m temperature and relative humidity (UTC) for `years`."""
    C.RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = C.RAW_DIR / f"climate_{_slug(name)}_{years[0]}_{years[-1]}.csv"
    if demo:
        return synthetic_climate(name, lat, years)
    if path.exists():
        df = pd.read_csv(path, parse_dates=["time"])
        return df
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": f"{years[0]}-01-01", "end_date": f"{years[-1]}-12-31",
        "hourly": "temperature_2m,relative_humidity_2m",
        "timezone": "UTC",
    }
    last_err = None
    for attempt in range(5):
        try:
            r = requests.get(C.OPEN_METEO_ARCHIVE, params=params, timeout=120)
            r.raise_for_status()
            h = r.json()["hourly"]
            df = pd.DataFrame({
                "time": pd.to_datetime(h["time"]),
                "t_db": h["temperature_2m"],
                "rh": h["relative_humidity_2m"],
            })
            n_missing = int(df[["t_db", "rh"]].isna().sum().sum())
            # ERA5 is spatially complete; any residual gaps are linearly interpolated
            # in time and the count is logged for the data-availability statement.
            df[["t_db", "rh"]] = df[["t_db", "rh"]].interpolate(limit_direction="both")
            df.attrs["n_missing"] = n_missing
            df.to_csv(path, index=False)
            meta = C.RAW_DIR / "climate_download_log.jsonl"
            with open(meta, "a") as f:
                f.write(json.dumps({"site": name, "lat": lat, "lon": lon,
                                    "rows": len(df), "n_missing_filled": n_missing,
                                    "downloaded_utc": pd.Timestamp.utcnow().isoformat()}) + "\n")
            time.sleep(1.0)  # be polite to the free API
            return df
        except Exception as e:  # network hiccups, rate limits
            last_err = e
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Open-Meteo download failed for {name}: {last_err}")


def synthetic_climate(name, lat, years):
    """
    DEMO ONLY. A crude sinusoidal climate used to test the code path when the
    Open-Meteo API is unreachable. Never used for reported results: every output
    produced in demo mode is watermarked.
    """
    rng = np.random.default_rng(abs(hash(name)) % 2**32)
    t = pd.date_range(f"{years[0]}-01-01", f"{years[-1]}-12-31 23:00", freq="h")
    doy = t.dayofyear.values
    hr = t.hour.values
    tropical = abs(lat) < 15
    mean = 27.0 if tropical else max(4.0, 28 - 0.45 * abs(lat))
    seas = 1.5 if tropical else 11.0
    sign = 1 if lat >= 0 else -1
    t_db = (mean + sign * seas * np.sin(2 * np.pi * (doy - 110) / 365)
            + (3.0 if tropical else 5.0) * np.sin(2 * np.pi * (hr - 9) / 24)
            + rng.normal(0, 1.5, len(t)))
    rh = np.clip((82 if tropical else 70) - 1.5 * (t_db - mean) + rng.normal(0, 6, len(t)), 10, 100)
    if name in ("Marrakech", "Henderson, Nevada"):
        rh = np.clip(rh - 30, 8, 100)
    return pd.DataFrame({"time": t, "t_db": t_db, "rh": rh})


# ---------------------------------------------------------------------------
# Grid carbon intensity: Ember annual lifecycle intensity via Our World in Data
# ---------------------------------------------------------------------------
def fetch_grid():
    C.RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = C.RAW_DIR / "owid-energy-data.csv"
    if not path.exists():
        r = requests.get(C.OWID_ENERGY_CSV, timeout=300)
        r.raise_for_status()
        path.write_bytes(r.content)
    d = pd.read_csv(path, usecols=["country", "year", "carbon_intensity_elec", "fossil_share_elec"])
    countries = sorted({v[2] for v in C.STUDY_SITES.values()})
    d = d[d.country.isin(countries) & d.year.isin(C.YEARS)].copy()
    d["ci_kg_per_kwh"] = d["carbon_intensity_elec"] / 1000.0
    missing = [(c, y) for c in countries for y in C.YEARS
               if d[(d.country == c) & (d.year == y)].ci_kg_per_kwh.notna().sum() == 0]
    if missing:
        # Carry the latest available year forward and record it (reported in SI)
        fills = []
        for c, y in missing:
            prev = d[(d.country == c) & d.ci_kg_per_kwh.notna()].sort_values("year")
            v = prev.ci_kg_per_kwh.iloc[-1]
            fills.append({"country": c, "year": y, "ci_kg_per_kwh": v, "filled_from": int(prev.year.iloc[-1])})
        d = pd.concat([d, pd.DataFrame(fills)], ignore_index=True)
    return d


# ---------------------------------------------------------------------------
# Training-run cohort: Epoch AI large-scale models
# ---------------------------------------------------------------------------
def _find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    low = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    raise KeyError(f"None of {candidates} in columns {list(df.columns)[:20]}...")


def fetch_cohort(fallback_parquet=None, offline=False):
    """
    Returns (cohort DataFrame with columns model, flop, date, country; provenance dict).
    Tries the live Epoch CSV first; falls back to the user's earlier extract.
    """
    C.RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = C.RAW_DIR / "epoch_large_scale_ai_models.csv"
    prov = {}
    df = None
    if not path.exists() and not offline:
        try:
            r = requests.get(C.EPOCH_LARGE_SCALE_CSV, timeout=120)
            r.raise_for_status()
            path.write_bytes(r.content)
        except Exception as e:
            prov["epoch_download_error"] = str(e)
    if path.exists():
        raw = pd.read_csv(path)
        mcol = _find_col(raw, ["Model", "System"])
        fcol = _find_col(raw, ["Training compute (FLOP)"])
        dcol = _find_col(raw, ["Publication date", "Publication Date"])
        try:
            ccol = _find_col(raw, ["Country (of organization)", "Country (from Organization)", "Country"])
        except KeyError:
            ccol = None
        df = pd.DataFrame({
            "model": raw[mcol],
            "flop": pd.to_numeric(raw[fcol], errors="coerce"),
            "date": pd.to_datetime(raw[dcol], errors="coerce"),
            "country": raw[ccol] if ccol else np.nan,
        })
        df = df[(df.flop >= C.COHORT_MIN_FLOP)
                & (df.date >= C.COHORT_START) & (df.date <= C.COHORT_END)]
        prov.update(source="Epoch AI, Data on Large-Scale AI Models (live CSV)",
                    window=f"{C.COHORT_START} to {C.COHORT_END}",
                    file_mtime=pd.Timestamp(path.stat().st_mtime, unit="s").isoformat())
    elif fallback_parquet is not None:
        raw = pd.read_parquet(fallback_parquet)
        df = pd.DataFrame({
            "model": raw["System"],
            "flop": raw["Training compute (FLOP)"],
            "date": pd.NaT,
            "country": raw.get("Country1", np.nan),
        })
        df = df[df.flop >= C.COHORT_MIN_FLOP]
        prov.update(source="Author's earlier Epoch AI extract (no publication dates; all runs >= 1e24 FLOP)",
                    window="n/a")
    else:
        raise RuntimeError("No cohort data available")
    df = df.dropna(subset=["flop"]).drop_duplicates("model").sort_values("flop", ascending=False)
    prov["n_runs"] = int(len(df))
    prov["total_flop"] = float(df.flop.sum())
    return df.reset_index(drop=True), prov


def google_pue_table():
    return pd.read_csv(C.DATA_DIR / "google_campus_pue.csv")
