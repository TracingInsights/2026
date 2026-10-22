#!/usr/bin/env python3
"""
COTA (Circuit of the Americas) race winner & podium predictor.

Data sources
  1. Jolpica-F1 (open Ergast successor): race results + qualifying, 2014 onwards.
  3. Open-Meteo (free, no key): race-day weather (historical + forecast) and circuit altitude,
     plus track statistics learned from history (pole win rate, DNF rate, safety cars, ...).
  2. TracingInsights (github.com/TracingInsights/<year>): per-session lap times for every
     driver. We turn them into *car pace* features (fuel-corrected race pace, qualifying
     best-lap gap, qualifying top speed) - this is what captures the 2026 cars.

Models   : XGBoost + Ridge (finishing position) and XGBoost classifiers (P(win), P(podium)).
Modes    : post-qualifying (grid / quali features)  or  pre-qualifying (form + pace only).
Backtest : walk-forward on every past US GP at COTA - train only on races BEFORE it.

Usage
-----
    pip install pandas numpy requests scikit-learn xgboost matplotlib
    python cota_predictor.py                       # backtest + predict the upcoming COTA race
    python cota_predictor.py --tracing-from 2022   # faster first run (less history to download)
    python cota_predictor.py --no-tracing          # Jolpica data only
    python cota_predictor.py --eval-all            # big backtest, all circuits + calibration
    python cota_predictor.py --mode pre            # force form/pace-only prediction
    python cota_predictor.py --grid VER,NOR,LEC    # override the grid (e.g. after penalties)
    python cota_predictor.py --plot --no-backtest
    python cota_predictor.py --trace-plot 2025      # race-trace chart of that year's COTA race
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sys
import time
import unicodedata
import warnings
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

# Keep warnings visible: missing features and library changes matter to this model.

BASE = "https://api.jolpi.ca/ergast/f1"
TI_RAW = "https://raw.githubusercontent.com/TracingInsights/{season}/main/{event}/{session}/session_laptimes.json"
TI_DIR = "https://api.github.com/repos/TracingInsights/{season}/contents"
CIRCUIT_ID = "americas"          # Circuit of the Americas in the Ergast/Jolpica database
CACHE = Path(".f1_cache")
BLEND = 0.5                      # weight of XGBoost vs Ridge in the position prediction
FUEL_KG, FUEL_S_PER_KG = 100.0, 0.03   # same fuel-correction assumptions as TracingInsights
SPEED_COLS = ["vst", "vfl", "vi1", "vi2"]
OM_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
OM_FORECAST = "https://api.open-meteo.com/v1/forecast"
OM_HOURLY = "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,cloud_cover"
WET_MM = 0.5                     # >= this much rain around the start -> "wet race"
HTTP = requests.Session()
HTTP.mount("https://", HTTPAdapter(max_retries=Retry(
    total=3, backoff_factor=1, status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=("GET",), respect_retry_after_header=True)))

# --- feature sets ---------------------------------------------------------------------
FEATURES_PRE = [
    "drv_form_pos", "drv_form_pts", "drv_dnf_rate",
    "team_form_pos", "team_form_pts",
    "drv_circuit_avg", "circuit_chaos",
    "drv_inc_rate", "team_mech_rate", "drv_circuit_dnf",      # retirement risk: crashes / reliability
]
FEATURES_QUALI = ["grid", "quali_pos", "quali_gap_pct", "teammate_grid_gap",
                  "q3_gap_pct", "last_seg", "last_seg_gap_pct"]
# car/driver pace from TracingInsights lap data (rolling, same season, previous races only)
FEATURES_PACE = ["drv_rpace_form", "drv_qpace_form", "drv_lap1_form",
                 "team_rpace_form", "team_qpace_form", "team_ts_form",
                 "team_rts_form", "circuit_vmax", "ts_x_circuit",   # straight-line speed x track
                 "team_pit_form", "team_slow_form"]                 # pit-stop crew performance
FEATURES_PACE_NOW = ["q_gap_pct", "team_q_now"]   # this weekend's qualifying pace (post mode)
# the TRACK: what this circuit has historically done to race results
FEATURES_TRACK = ["circuit_pole_win", "circuit_dnf", "circuit_grid_corr", "circuit_sc_rate",
                  "team_circuit_avg", "elevation"]
# the WEATHER at the start of the race (+ who is good in the wet)
FEATURES_WEATHER = ["air_temp", "humidity", "wind", "precip_mm", "wet", "drv_wet_gain", "drv_wet_x"]

XGB_REG = dict(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
               colsample_bytree=0.8, min_child_weight=5, reg_lambda=2.0, random_state=42)
XGB_CLF = dict(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
               colsample_bytree=0.8, min_child_weight=5, reg_lambda=2.0, random_state=42,
               eval_metric="logloss")


def get_features(mode, pace, env=False):
    f = list(FEATURES_PRE)
    if pace:
        f += FEATURES_PACE
    if env:
        f += FEATURES_TRACK + FEATURES_WEATHER
    if mode == "post":
        f += FEATURES_QUALI
        if pace:
            f += FEATURES_PACE_NOW
    return f


# --------------------------------------------------------------------------- #
# Jolpica download (cached on disk)
# --------------------------------------------------------------------------- #
def jget(path, params=None, ttl=None):
    """GET from Jolpica with on-disk caching. ttl=None -> cache forever."""
    params = params or {}
    key = hashlib.md5((path + json.dumps(params, sort_keys=True)).encode()).hexdigest()
    f = CACHE / f"{key}.json"
    if f.exists() and (ttl is None or time.time() - f.stat().st_mtime < ttl):
        return json.loads(f.read_text())
    for attempt in range(6):
        r = HTTP.get(f"{BASE}/{path}", params=params, timeout=30)
        if r.status_code == 429:                      # rate limited -> back off
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        data = r.json()
        CACHE.mkdir(exist_ok=True)
        f.write_text(json.dumps(data))
        time.sleep(0.3)
        return data
    raise RuntimeError(f"Jolpica API kept rate-limiting: {path}")


def fetch_races(season, endpoint, ttl):
    """All races of a season for 'results' or 'qualifying' (handles pagination)."""
    key = "Results" if endpoint == "results" else "QualifyingResults"
    races, offset = {}, 0
    while True:
        mr = jget(f"{season}/{endpoint}.json", {"limit": 100, "offset": offset}, ttl)["MRData"]
        for race in mr["RaceTable"]["Races"]:
            rnd = int(race["round"])
            if rnd not in races:
                races[rnd] = race
                races[rnd][key] = list(race.get(key, []))
            else:                                    # a race can be split across pages
                races[rnd][key].extend(race.get(key, []))
        offset += 100
        if offset >= int(mr["total"]):
            break
    return races


def parse_time(s):
    """'1:32.608' -> 92.608 seconds."""
    if not s:
        return np.nan
    try:
        parts = [float(p) for p in s.split(":")]
    except ValueError:
        return np.nan
    return sum(p * 60 ** i for i, p in enumerate(reversed(parts)))


INCIDENT_WORDS = ("accident", "collision", "spun", "crash")


def dnf_cause(status):
    """-> (incident, mechanical) flags. Finished / lapped (+1 Lap, Lapped) / disqualified -> (0, 0)."""
    st = (status or "").lower()
    if st == "finished" or st.startswith("+") or "lapped" in st or "disqualif" in st or "exclu" in st:
        return 0.0, 0.0
    if any(w in st for w in INCIDENT_WORDS):
        return 1.0, 0.0
    return 0.0, 1.0                                   # engine, gearbox, hydraulics, retired, ...


def load_history(first_season, last_season):
    rows = []
    this_year = dt.date.today().year
    for season in range(first_season, last_season + 1):
        ttl = None if season < this_year else 3600       # refresh the live season hourly
        print(f"  loading {season} ...", end="\r", flush=True)
        res = fetch_races(season, "results", ttl)
        qual = fetch_races(season, "qualifying", ttl)
        for rnd, race in sorted(res.items()):
            q = {x["Driver"]["driverId"]: x
                 for x in qual.get(rnd, {}).get("QualifyingResults", [])}
            for r in race["Results"]:
                d = r["Driver"]["driverId"]
                qq = q.get(d)
                status = r.get("status", "")
                d_inc, d_mech = dnf_cause(status)
                loc = race["Circuit"].get("Location", {})
                rows.append(dict(
                    season=season, rnd=rnd, race_name=race["raceName"],
                    race_date=race.get("date"), race_time=race.get("time"),
                    lat=loc.get("lat"), lon=loc.get("long"),
                    circuitId=race["Circuit"]["circuitId"],
                    driverId=d, code=r["Driver"].get("code", d[:3].upper()),
                    constructorId=r["Constructor"]["constructorId"],
                    grid=int(r["grid"]), position=int(r["position"]),
                    points=float(r["points"]),
                    dnf=float(d_inc + d_mech > 0), status=status,
                    dnf_inc=d_inc, dnf_mech=d_mech,
                    quali_pos=float(qq["position"]) if qq else np.nan,
                    q1=parse_time(qq.get("Q1")) if qq else np.nan,
                    q2=parse_time(qq.get("Q2")) if qq else np.nan,
                    q3=parse_time(qq.get("Q3")) if qq else np.nan,
                ))
    print(" " * 40, end="\r")
    return pd.DataFrame(rows)


def cota_race(season):
    sched = jget(f"{season}.json", ttl=3600)["MRData"]["RaceTable"]["Races"]
    return next((r for r in sched if r["Circuit"]["circuitId"] == CIRCUIT_ID), None)


# --------------------------------------------------------------------------- #
# Open-Meteo: weather at the start of the race (historical archive or forecast)
# --------------------------------------------------------------------------- #
WX_FAILS = {}


def _wx_fail(reason):
    WX_FAILS[reason] = WX_FAILS.get(reason, 0) + 1
    return None


def fetch_weather(lat, lon, date, time_utc):
    """Mean conditions over the first ~2 hours of the race. None if unavailable."""
    if not (lat and lon and date):
        return _wx_fail("missing coordinates/date")
    age = (dt.date.today() - dt.date.fromisoformat(date)).days
    archive = age > 7                                   # archive lags a few days -> use forecast API
    url = OM_ARCHIVE if archive else OM_FORECAST
    params = dict(latitude=round(float(lat), 4), longitude=round(float(lon), 4),
                  start_date=date, end_date=date, timezone="UTC",
                  hourly=OM_HOURLY + ("" if archive else ",precipitation_probability"))
    key = hashlib.md5((url + json.dumps(params, sort_keys=True)).encode()).hexdigest()
    f = CACHE / f"wx_{key}.json"
    if f.exists() and (archive or time.time() - f.stat().st_mtime < 10800):
        js = json.loads(f.read_text())
    else:
        r = None
        for attempt in range(5):                        # retry when rate-limited / server busy
            try:
                r = HTTP.get(url, params=params, timeout=30)
            except requests.RequestException:
                r = None
            if r is not None and r.status_code not in (429, 500, 502, 503, 504):
                break
            time.sleep(2 ** attempt)
        if r is None:
            return _wx_fail("network error")
        if r.status_code != 200:                        # e.g. 400 = beyond the 16-day forecast
            return _wx_fail(f"HTTP {r.status_code}")
        js = r.json()
        CACHE.mkdir(exist_ok=True)
        f.write_text(json.dumps(js))
        time.sleep(0.3)
    h = js.get("hourly") or {}
    times = h.get("time") or []
    hh = int(time_utc[:2]) if time_utc else 13
    target = f"{date}T{hh:02d}:00"
    if target not in times:
        return _wx_fail("race hour not in response")
    i = times.index(target)

    def seg(k):
        return [v for v in (h.get(k) or [])[i:i + 3] if v is not None]

    def mean(k):
        v = seg(k)
        return float(np.mean(v)) if v else np.nan

    prec = seg("precipitation")
    prob = seg("precipitation_probability")
    return dict(air_temp=mean("temperature_2m"), humidity=mean("relative_humidity_2m"),
                wind=mean("wind_speed_10m"), cloud=mean("cloud_cover"),
                precip_mm=float(np.sum(prec)) if prec else np.nan,
                rain_prob=float(np.max(prob)) if prob else np.nan,
                elevation=js.get("elevation", np.nan))


def load_weather(races):
    """races: DataFrame[season, rnd, race_date, race_time, lat, lon]."""
    rows, miss = [], 0
    for _, rc in races.iterrows():
        print(f"  weather {int(rc['season'])} round {int(rc['rnd']):>2}", end="\r", flush=True)
        w = fetch_weather(rc["lat"], rc["lon"], rc["race_date"], rc["race_time"])
        if w is None:
            miss += 1
            continue
        rows.append(dict(season=int(rc["season"]), rnd=int(rc["rnd"]), **w))
    print(" " * 40, end="\r")
    why = f" - reasons: {', '.join(f'{k} x{v}' for k, v in WX_FAILS.items())}" if WX_FAILS else ""
    print(f"Open-Meteo: weather for {len(rows)} races ({miss} not available{why})")
    if not rows:
        return None
    wx = pd.DataFrame(rows)
    wx["wet"] = np.where(wx["precip_mm"].isna(), np.nan, (wx["precip_mm"] >= WET_MM).astype(float))
    return wx


# --------------------------------------------------------------------------- #
# TracingInsights: lap data -> car / driver pace
# --------------------------------------------------------------------------- #
def _norm(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]", "", s.replace("grand prix", ""))


def list_event_folders(season):
    """Event folder names of a TracingInsights season repo (cached for a day)."""
    f = CACHE / f"ti_dirs_{season}.json"
    if f.exists() and time.time() - f.stat().st_mtime < 86400:
        return json.loads(f.read_text())
    try:
        r = HTTP.get(TI_DIR.format(season=season), timeout=30)
        r.raise_for_status()
        names = [x["name"] for x in r.json()
                 if x.get("type") == "dir" and x["name"].endswith("Grand Prix")]
        CACHE.mkdir(exist_ok=True)
        f.write_text(json.dumps(names))
        return names
    except Exception:
        return []                                    # fall back to Jolpica race names


def match_folder(race_name, folders):
    if race_name in folders:
        return race_name
    t = _norm(race_name)
    nf = {_norm(x): x for x in folders}
    if t in nf:
        return nf[t]
    for k, v in nf.items():
        if k and (k in t or t in k):
            return v
    return race_name


def _col(df, name):
    return df[name] if name in df.columns else pd.Series(np.nan, index=df.index)


def _bool(series):
    return series.map(lambda x: x is True)


PIT_MIN, PIT_MAX, SLOW_STOP_S = 12.0, 45.0, 2.5


def _pit_times(full):
    """Pit-lane time per stop (s) = out-time on the next lap minus in-time, per driver."""
    out = {}
    for d, g in full.sort_values(["drv", "lap"]).groupby("drv"):
        pin = g.set_index("lap")["pin"].dropna()
        pout = g.set_index("lap")["pout"].dropna()
        durs = []
        for lap, t_in in pin.items():
            t_out = pout.get(lap + 1)
            if t_out is not None and not pd.isna(t_out):
                durs.append(float(t_out - t_in))
        out[d] = [x for x in durs if PIT_MIN <= x <= PIT_MAX]   # drop red flags / repairs / bad data
    return out


def summarise_session(data, session):
    """session_laptimes.json (dict of parallel arrays) -> {driver_code: metrics}."""
    n = len(data.get("lap", []))
    if n == 0 or "drv" not in data:
        return {}
    df = pd.DataFrame({k: v for k, v in data.items() if isinstance(v, list) and len(v) == n})
    for c in ["time", "lap", "pin", "pout"] + SPEED_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")       # "None" strings -> NaN
    full = df.copy()                                            # keeps in/out laps for pit stops
    df = df[df["time"].notna() & (df["time"] > 0)]
    if df.empty:
        return {}
    speed_cols = [c for c in SPEED_COLS if c in df.columns]
    top = df.groupby("drv")[speed_cols].max().max(axis=1) if speed_cols else pd.Series(dtype=float)

    out = {}
    if session == "Qualifying":
        valid = df
        if "pb" in df.columns and _bool(df["pb"]).any():
            valid = valid[_bool(valid["pb"])]                    # official (not deleted) bests
        if "del" in df.columns:
            valid = valid[~_bool(valid["del"])]
        best = valid.groupby("drv")["time"].min()
        gap = (best / best.min() - 1) * 100
        for d, g in gap.items():
            out[d] = {"q_gap_pct": float(g), "q_top": float(top.get(d, np.nan))}
    else:                                                        # Race
        total = df["lap"].max()
        clean = df[df["lap"] > 1]
        if "status" in clean.columns:
            clean = clean[clean["status"].astype(str) == "1"]    # green-flag laps only
        clean = clean[_col(clean, "pin").isna() & _col(clean, "pout").isna()]
        if clean.empty:
            return {}
        corr = clean["time"] - FUEL_KG * FUEL_S_PER_KG * (1 - clean["lap"] / total)
        clean = clean.assign(corr=corr)
        clean = clean[clean["corr"] <= clean["corr"].quantile(0.10) * 1.07]   # drop slow laps
        grp = clean.groupby("drv")["corr"]
        med = grp.median()[grp.count() >= 5]
        if med.empty:
            return {}
        gap = (med / med.min() - 1) * 100
        sc = float(df["status"].astype(str).str.contains("4").any()) if "status" in df.columns else np.nan
        lap1 = {}
        if "pos" in df.columns:                                  # position at the end of lap 1
            l1 = pd.to_numeric(df.loc[df["lap"] == 1, "pos"], errors="coerce")
            lap1 = dict(zip(df.loc[df["lap"] == 1, "drv"], l1))
        pits = _pit_times(full) if {"pin", "pout"} <= set(full.columns) else {}
        stops = [x for v in pits.values() for x in v]
        med_stop = float(np.median(stops)) if len(stops) >= 10 else np.nan
        if not (15 <= med_stop <= 40):                          # implausible -> units/format differ
            med_stop = np.nan
        for d, g in gap.items():
            st = pits.get(d, [])
            ok = bool(st) and not np.isnan(med_stop)
            out[d] = {"r_gap_pct": float(g), "lap1_pos": float(lap1.get(d, np.nan)), "sc": sc,
                      "r_top": float(top.get(d, np.nan)),
                      "pit_med": med_stop, "pit_n": float(len(st)),
                      "pit_rel": float(np.mean(st) - med_stop) if ok else np.nan,
                      "pit_slow": float(np.mean([x > med_stop + SLOW_STOP_S for x in st])) if ok else np.nan}
    return out


def fetch_session_summary(season, event, session):
    f = CACHE / f"ti5_{season}_{_norm(event)}_{session}.json"
    if f.exists() and (season < dt.date.today().year or time.time() - f.stat().st_mtime < 3600):
        return json.loads(f.read_text())
    url = TI_RAW.format(season=season, event=quote(event), session=session)
    try:
        r = HTTP.get(url, timeout=120)
    except requests.RequestException as e:
        print(f"\n  ! TracingInsights request failed ({e.__class__.__name__}) - skipping {season} {event}")
        return None
    CACHE.mkdir(exist_ok=True)
    if r.status_code == 404:
        if season < dt.date.today().year:                # old seasons never change -> remember
            f.write_text("{}")
        return None
    r.raise_for_status()
    summ = summarise_session(r.json(), session)
    f.write_text(json.dumps(summ))
    return summ


def load_tracing(races, first_season):
    """races: DataFrame[season, rnd, race_name]. Returns per-driver pace per race."""
    rows, n_ok, n_miss = [], 0, 0
    for season, grp in races[races["season"] >= first_season].groupby("season"):
        folders = list_event_folders(season)
        for _, rc in grp.sort_values("rnd").iterrows():
            ev = match_folder(rc["race_name"], folders)
            print(f"  TracingInsights {season} {ev[:28]:<28}", end="\r", flush=True)
            rec = {}
            for session in ("Qualifying", "Race"):
                for code, d in (fetch_session_summary(season, ev, session) or {}).items():
                    rec.setdefault(code, {}).update(d)
            if not rec:
                n_miss += 1
                continue
            n_ok += 1
            for code, d in rec.items():
                rows.append(dict(season=season, rnd=int(rc["rnd"]), code=code,
                                 q_gap_pct=d.get("q_gap_pct"), r_gap_pct=d.get("r_gap_pct"),
                                 q_top=d.get("q_top"), lap1_pos=d.get("lap1_pos"), sc=d.get("sc"),
                                 r_top=d.get("r_top"), pit_rel=d.get("pit_rel"),
                                 pit_slow=d.get("pit_slow"), pit_med=d.get("pit_med"),
                                 pit_n=d.get("pit_n")))
    print(" " * 60, end="\r")
    print(f"TracingInsights: pace data for {n_ok} races ({n_miss} not available)")
    if not rows:
        return None
    tr = pd.DataFrame(rows)
    for c in ("q_gap_pct", "r_gap_pct", "q_top", "lap1_pos", "sc", "r_top", "pit_rel", "pit_slow",
              "pit_med", "pit_n"):
        tr[c] = pd.to_numeric(tr[c], errors="coerce")
    gk = tr.groupby(["season", "rnd"])
    tr["ts_gap"] = gk["q_top"].transform("max") - tr["q_top"]       # km/h below the fastest, qualifying
    tr["rts_gap"] = gk["r_top"].transform("max") - tr["r_top"]      # km/h below the fastest, race
    tr["vmax"] = gk["q_top"].transform("max")                       # fastest speed of the weekend
    n_pit = tr.loc[tr["pit_rel"].notna(), ["season", "rnd"]].drop_duplicates().shape[0]
    print(f"TracingInsights: pit-stop timing found for {n_pit} races"
          + ("" if n_pit else "  (0 = lap-file format differs; pit features stay empty)"))
    if n_pit:                                   # sanity check you can compare with real F1 numbers
        pm = (tr[tr["pit_med"].notna()].groupby(["season", "rnd"])
                .agg(med=("pit_med", "first"), n=("pit_n", "sum")))
        print(f"  pit-lane time (entry to exit) per race: {pm['med'].min():.1f} - {pm['med'].max():.1f} s, "
              f"median {pm['med'].median():.1f} s; about {pm['n'].median():.0f} stops per race")
    return tr.drop(columns=["q_top", "r_top", "pit_med", "pit_n"]).drop_duplicates(["season", "rnd", "code"])


# --------------------------------------------------------------------------- #
# Feature engineering  (every form feature only uses information from PRIOR races)
# --------------------------------------------------------------------------- #
def _roll(s, w):
    return s.shift().rolling(w, min_periods=1).mean()


WX_COLS = ["air_temp", "humidity", "wind", "cloud", "precip_mm", "rain_prob", "elevation", "wet"]


def add_features(df, tr=None, wx=None):
    df = df.copy()
    TR_COLS = ("q_gap_pct", "r_gap_pct", "ts_gap", "lap1_pos", "sc", "rts_gap", "vmax", "pit_rel", "pit_slow")
    for c in TR_COLS + tuple(WX_COLS):
        if c in df.columns:
            df = df.drop(columns=c)
    for c in ("dnf_inc", "dnf_mech"):
        if c not in df.columns:
            df[c] = np.nan
    if tr is not None:
        df = df.merge(tr, on=["season", "rnd", "code"], how="left")
    else:
        for c in TR_COLS:
            df[c] = np.nan
    if wx is not None:
        df = df.merge(wx, on=["season", "rnd"], how="left")
    else:
        for c in WX_COLS:
            df[c] = np.nan
    df = df.sort_values(["season", "rnd", "driverId"]).reset_index(drop=True)
    df["race_idx"] = df.groupby(["season", "rnd"]).ngroup()

    # --- qualifying / grid features (known at race start) ---
    field_size = df.groupby("race_idx")["driverId"].transform("size")
    df["grid"] = df["grid"].replace(0, np.nan).fillna(field_size)
    df["quali_pos"] = df["quali_pos"].fillna(df["grid"])
    df["quali_gap_pct"] = (df["q1"] / df.groupby("race_idx")["q1"].transform("min") - 1) * 100
    df["teammate_grid_gap"] = df["grid"] - df.groupby(["race_idx", "constructorId"])["grid"].transform("mean")
    # full qualifying picture: Q3 gap to pole + gap inside the LAST segment each driver reached
    df["q3_gap_pct"] = (df["q3"] / df.groupby("race_idx")["q3"].transform("min") - 1) * 100
    df["last_seg"] = np.where(df["q3"].notna(), 3, np.where(df["q2"].notna(), 2,
                              np.where(df["q1"].notna(), 1, np.nan)))
    last_t = df["q3"].fillna(df["q2"]).fillna(df["q1"])
    seg_best = last_t.groupby([df["race_idx"], df["last_seg"]]).transform("min")
    df["last_seg_gap_pct"] = (last_t / seg_best - 1) * 100
    df["lap1_gain"] = df["grid"] - df["lap1_pos"]          # places gained on lap 1 (race-day only)

    # --- driver form (results) ---
    g = df.groupby("driverId")
    df["drv_form_pos"] = g["position"].transform(lambda s: s.shift().rolling(5, min_periods=2).mean())
    df["drv_form_pts"] = g["points"].transform(lambda s: s.shift().rolling(5, min_periods=2).mean())
    df["drv_dnf_rate"] = g["dnf"].transform(lambda s: s.shift().rolling(10, min_periods=3).mean())
    df["drv_inc_rate"] = g["dnf_inc"].transform(lambda s: s.shift().rolling(15, min_periods=3).mean())
    df["drv_circuit_dnf"] = (df.groupby(["driverId", "circuitId"])["dnf"]
                               .transform(lambda s: s.shift().expanding().mean()))

    # --- team form (results; car pace proxy) ---
    tm = (df.groupby(["race_idx", "constructorId"], as_index=False)
            .agg(season=("season", "first"), circuitId=("circuitId", "first"),
                 team_pos=("position", "mean"),
                 team_pts=("points", "sum"), team_r=("r_gap_pct", "mean"),
                 team_q=("q_gap_pct", "mean"), team_ts=("ts_gap", "mean"),
                 team_rts=("rts_gap", "mean"), team_pit=("pit_rel", "mean"),
                 team_slow=("pit_slow", "mean"), team_mech=("dnf_mech", "mean")))
    tm = tm.sort_values(["constructorId", "race_idx"])
    gt = tm.groupby("constructorId")
    tm["team_form_pos"] = gt["team_pos"].transform(lambda s: s.shift().rolling(5, min_periods=2).mean())
    tm["team_form_pts"] = gt["team_pts"].transform(lambda s: s.shift().rolling(5, min_periods=2).mean())
    tm["team_pit_form"] = gt["team_pit"].transform(lambda s: s.shift().rolling(6, min_periods=2).mean())
    tm["team_slow_form"] = gt["team_slow"].transform(lambda s: s.shift().rolling(6, min_periods=2).mean())
    tm["team_mech_rate"] = gt["team_mech"].transform(lambda s: s.shift().rolling(10, min_periods=3).mean())

    # --- TracingInsights pace form: rolling over the last 3 races of the SAME season,
    #     so new-regulation cars are judged only on this year's data ---
    gts = tm.groupby(["constructorId", "season"])
    tm["team_rpace_form"] = gts["team_r"].transform(lambda s: _roll(s, 3))
    tm["team_qpace_form"] = gts["team_q"].transform(lambda s: _roll(s, 3))
    tm["team_ts_form"] = gts["team_ts"].transform(lambda s: _roll(s, 3))
    tm["team_rts_form"] = gts["team_rts"].transform(lambda s: _roll(s, 3))
    tm["team_q_now"] = tm["team_q"]                           # same-weekend quali pace (known)
    tm["team_circuit_avg"] = (tm.groupby(["constructorId", "circuitId"])["team_pos"]
                                .transform(lambda s: s.shift().expanding().mean()))
    df = df.merge(tm[["race_idx", "constructorId", "team_form_pos", "team_form_pts",
                      "team_rpace_form", "team_qpace_form", "team_ts_form", "team_q_now",
                      "team_circuit_avg", "team_pit_form", "team_slow_form", "team_mech_rate",
                      "team_rts_form"]],
                  on=["race_idx", "constructorId"], how="left")
    gds = df.groupby(["driverId", "season"])
    df["drv_rpace_form"] = gds["r_gap_pct"].transform(lambda s: _roll(s, 3))
    df["drv_qpace_form"] = gds["q_gap_pct"].transform(lambda s: _roll(s, 3))
    df["drv_lap1_form"] = (df.groupby("driverId")["lap1_gain"]
                             .transform(lambda s: s.shift().rolling(8, min_periods=2).mean()))

    # --- circuit-specific: driver's history at this track + how chaotic the track is ---
    df["drv_circuit_avg"] = (df.groupby(["driverId", "circuitId"])["position"]
                               .transform(lambda s: s.shift().expanding().mean()))
    df["abs_change"] = (df["position"] - df["grid"]).abs()
    rc = df.groupby(["race_idx", "circuitId"], as_index=False)["abs_change"].mean()
    rc["circuit_chaos"] = rc.groupby("circuitId")["abs_change"].transform(
        lambda s: s.shift().expanding().mean())
    df = df.merge(rc[["race_idx", "circuit_chaos"]], on="race_idx", how="left")

    # --- the TRACK: what each circuit has historically done to the results (prior races only) ---
    stats = []
    for idx, g in df.groupby("race_idx"):
        row = dict(race_idx=idx, circuitId=g["circuitId"].iloc[0], pole_win=np.nan, dnf=np.nan,
                   grid_corr=np.nan, sc=g["sc"].max(), vmax=g["vmax"].max())
        if g["position"].notna().sum() >= 10:
            pole = g[g["grid"] == 1]
            row.update(pole_win=float((pole["position"] == 1).any()) if len(pole) else np.nan,
                       dnf=g["dnf"].mean(),
                       grid_corr=g["grid"].rank().corr(g["position"].rank()))
        stats.append(row)
    cs = pd.DataFrame(stats).sort_values("race_idx")
    for col, name in (("pole_win", "circuit_pole_win"), ("dnf", "circuit_dnf"),
                      ("grid_corr", "circuit_grid_corr"), ("sc", "circuit_sc_rate"),
                      ("vmax", "circuit_vmax")):
        cs[name] = cs.groupby("circuitId")[col].transform(lambda s: s.shift().expanding().mean())
    df = df.merge(cs[["race_idx", "circuit_pole_win", "circuit_dnf", "circuit_grid_corr",
                      "circuit_sc_rate", "circuit_vmax"]], on="race_idx", how="left")
    df["ts_x_circuit"] = df["team_ts_form"] * df["circuit_vmax"] / 330.0   # top-speed deficit x track speed

    # --- the WEATHER: who gains places in wet races (prior wet races only) ---
    gain_wet = (df["grid"] - df["position"]).where(df["wet"] == 1)
    df["drv_wet_gain"] = gain_wet.groupby(df["driverId"]).transform(
        lambda s: s.shift().expanding(min_periods=1).mean())
    x = np.where(df["wet"] == 1, df["drv_wet_gain"], 0.0)       # only active on a wet day
    x[df["wet"].isna().to_numpy()] = np.nan
    df["drv_wet_x"] = x
    return df


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def bounded_probabilities(raw, total, lower=None):
    """Rescale marginal probabilities within [lower, 1] to an exact slot total.

    This enforces probability arithmetic, not empirical calibration or a joint
    distribution of finishing orders. Fit calibration on separate past races.
    """
    raw = np.asarray(raw, dtype=float)
    lo = np.zeros_like(raw) if lower is None else np.asarray(lower, dtype=float)
    if (not np.isfinite(raw).all() or not np.isfinite(lo).all()
            or np.any(lo < 0) or np.any(lo > 1)
            or not lo.sum() <= total <= len(raw)):
        raise ValueError("Invalid probability bounds or slot total")
    left, right = 0.0, 1.0
    raw = np.clip(raw, 1e-12, 1)
    while np.clip(raw * right, lo, 1).sum() < total:
        right *= 2
    for _ in range(80):
        mid = (left + right) / 2
        if np.clip(raw * mid, lo, 1).sum() < total:
            left = mid
        else:
            right = mid
    return np.clip(raw * ((left + right) / 2), lo, 1)


def reconcile_probabilities(out):
    """Ensure one winner, three podium places, and nested top-k marginals."""
    out = out.copy()
    for col in ("p_win", "p_podium", "p_top10"):
        if col in out:
            out[col] = out[col].astype("float64")
    for _, race in out.groupby("race_idx"):
        ix, n = race.index, len(race)
        win = bounded_probabilities(race["p_win"], 1)
        podium = bounded_probabilities(race["p_podium"], min(3, n), win)
        out.loc[ix, "p_win"] = win
        out.loc[ix, "p_podium"] = podium
        if "p_top10" in out:
            out.loc[ix, "p_top10"] = bounded_probabilities(race["p_top10"], min(10, n), podium)
    return out


def fit_predict(train, test, feats, extras=False):
    """Fit on `train`, return `test` with pred_pos, p_win, p_podium."""
    if train.empty or test.empty:
        raise ValueError("Training and prediction data must both contain drivers")
    if train["race_idx"].max() >= test["race_idx"].min():
        raise ValueError("Training races must strictly precede prediction races")
    Xtr, Xte = train[feats], test[feats]
    xgb = XGBRegressor(**XGB_REG).fit(Xtr, train["position"])
    rdg = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                        Ridge(alpha=5.0)).fit(Xtr, train["position"])
    out = test.copy()
    out["pred_xgb"], out["pred_ridge"] = xgb.predict(Xte), rdg.predict(Xte)
    out["pred_pos"] = BLEND * out["pred_xgb"] + (1 - BLEND) * out["pred_ridge"]

    for name, y in (("p_win", train["position"] == 1), ("p_podium", train["position"] <= 3)):
        clf = XGBClassifier(**XGB_CLF).fit(Xtr, y.astype(int))
        out[name] = clf.predict_proba(Xte)[:, 1]
    dn = XGBClassifier(**XGB_CLF).fit(Xtr, train["dnf"].astype(int))      # retirement risk
    out["p_dnf"] = dn.predict_proba(Xte)[:, 1]
    if extras:
        t10 = XGBClassifier(**XGB_CLF).fit(Xtr, (train["position"] <= 10).astype(int))
        out["p_top10"] = t10.predict_proba(Xte)[:, 1]
        for name, a in (("pos_lo", 0.1), ("pos_hi", 0.9)):          # 10th / 90th percentile of position
            q = XGBRegressor(**{**XGB_REG, "objective": "reg:quantileerror", "quantile_alpha": a})
            out[name] = q.fit(Xtr, train["position"]).predict(Xte)
        n_cars = out.groupby("race_idx")["code"].transform("size")
        out["pos_lo"] = np.clip(np.minimum(out["pos_lo"], out["pred_pos"]), 1, n_cars).round()
        out["pos_hi"] = np.clip(np.maximum(out["pos_hi"], out["pred_pos"]), 1, n_cars).round()
    return reconcile_probabilities(out)


# --------------------------------------------------------------------------- #
# Walk-forward backtest on past COTA races
# --------------------------------------------------------------------------- #
def backtest(df, mode, variants, start_year=2016):
    """variants: {label: feature list}. All are scored on the SAME past COTA races."""
    done = df[df["position"].notna()]
    cota = done[(done["circuitId"] == CIRCUIT_ID) & (done["season"] >= start_year)]
    acc = {k: [] for k in ["baseline", *variants]}
    detail = []
    for (season, rnd), race in cota.groupby(["season", "rnd"]):
        train = done[done["race_idx"] < race["race_idx"].iloc[0]]
        if len(train) < 300:
            continue
        # baseline: starting grid (post) or simple form ranking (pre)
        b = race["grid"] if mode == "post" else (race["drv_form_pos"].fillna(10) + race["team_form_pos"].fillna(10))
        b_rank = b.rank(method="first")
        actual_pod = set(race.loc[race["position"] <= 3, "code"])
        winner = race.loc[race["position"] == 1, "code"].iloc[0]

        def score(rank, win_code, pod_codes):
            return dict(mae=(rank - race["position"]).abs().mean(), win=float(win_code == winner),
                        pod=len(actual_pod & set(pod_codes)))
        acc["baseline"].append(score(b_rank, race.loc[b_rank.idxmin(), "code"],
                                     race.loc[b_rank.nsmallest(3).index, "code"]))
        for label, feats in variants.items():
            pred = fit_predict(train, race, feats)
            rank = pred["pred_pos"].rank(method="first")
            pod = pred.nlargest(3, "p_podium")["code"].tolist()
            acc[label].append(score(rank, pred.nlargest(1, "p_win")["code"].iloc[0], pod))
            last = (pred.nlargest(1, "p_win")["code"].iloc[0], pod)
        detail.append((season, winner, sorted(actual_pod), last))
    n = len(detail)
    if n == 0:
        print("  (not enough history for a backtest)")
        return
    bl = "grid order" if mode == "post" else "form ranking"
    print(f"\n=== Backtest at COTA | {mode}-qualifying | {n} past races (trained only on earlier races) ===")
    print(f"{'Model':<34}{'Pos. error':>11}{'Winner hit':>12}{'Podium (of 3)':>15}")
    for k, v in acc.items():
        label = f"baseline: {bl}" if k == "baseline" else k
        print(f"{label:<34}{np.mean([x['mae'] for x in v]):>11.2f}"
              f"{int(sum(x['win'] for x in v)):>9}/{n}{np.mean([x['pod'] for x in v]):>15.2f}")
    print(f"\nRace by race (last model: {list(variants)[-1]}):")
    for season, winner, pod, (pw, pp) in detail:
        print(f"  {season}  actual {winner} / {','.join(pod)}   predicted {pw} / {','.join(sorted(pp))}")
    print("(few COTA races exist, so treat these as a sanity check, not proof)")


# --------------------------------------------------------------------------- #
# Big backtest: ALL circuits, last N seasons + probability calibration
# --------------------------------------------------------------------------- #
DNF_BINS = [0, .05, .10, .20, .35, 1.0001]
WIN_BINS = [0, .02, .05, .10, .20, .35, .50, 1.0001]
POD_BINS = [0, .05, .15, .30, .50, .70, .85, 1.0001]


def _hits(p, col, best, k):
    """Per race: how many of the top-k drivers by `col` really finished in the top k."""
    srt = p.sort_values(["race_idx", col], ascending=[True, best == "min"], kind="mergesort")
    top = srt.groupby("race_idx").head(k)
    return (top["position"] <= k).groupby(top["race_idx"]).sum()


def _logloss(pr, y):
    pr = np.clip(pr, 1e-4, 1 - 1e-4)
    return float(-(y * np.log(pr) + (1 - y) * np.log(1 - pr)).mean())


def _calib(p, col, y, bins, title):
    t = pd.DataFrame({"p": p[col].to_numpy(), "y": np.asarray(y, dtype=float)})
    t["bin"] = pd.cut(t["p"], bins, right=False)
    g = t.groupby("bin", observed=True).agg(n=("y", "size"), said=("p", "mean"), actual=("y", "mean"))
    print(f"\n  {title}")
    print(f"  {'Model said':<14}{'Predictions':>12}{'Avg. said':>11}{'Really happened':>17}")
    for iv, r in g.iterrows():
        print(f"  {iv.left:>5.0%} - {min(iv.right, 1):<5.0%}{int(r['n']):>12}{r['said']:>11.1%}{r['actual']:>17.1%}")


def eval_all(df, mode, variants, n_seasons=4, step=3):
    """Walk-forward over every race of the last `n_seasons` seasons (all circuits).
    Models are refit every `step` races on earlier races only, then predict the next block."""
    done = df[df["position"].notna()]
    last = int(done["season"].max())
    if n_seasons <= 0:                              # all history once 2 seasons exist to train on
        races = sorted(done.loc[done["season"] >= int(done["season"].min()) + 2, "race_idx"].unique())
    else:
        races = sorted(done.loc[done["season"] > last - n_seasons, "race_idx"].unique())
    if len(races) < 10:
        print("  (not enough races for the all-circuit evaluation)")
        return
    print(f"\n  evaluating {len(races)} races ({mode}-qualifying) - this takes a few minutes ...")
    parts = {k: [] for k in variants}
    last_label = list(variants)[-1]
    for i in range(0, len(races), step):
        block = races[i:i + step]
        train = done[done["race_idx"] < block[0]]
        if len(train) < 300:
            continue
        test = done[done["race_idx"].isin(block)]
        for label, feats in variants.items():
            parts[label].append(fit_predict(train, test, feats, extras=(label == last_label)))
    res = {k: pd.concat(v, ignore_index=True) for k, v in parts.items() if v}
    if not res:
        return
    ref = next(iter(res.values()))
    n = ref["race_idx"].nunique()
    sea = sorted(ref["season"].unique())
    bl = "grid order" if mode == "post" else "form ranking"
    base = ref["grid"] if mode == "post" else (ref["drv_form_pos"].fillna(10) + ref["team_form_pos"].fillna(10))
    ref = ref.assign(base=base)

    print(f"\n=== ALL-CIRCUIT backtest | {mode}-qualifying | {n} races, {sea[0]}-{sea[-1]} "
          f"| refit every {step} races ===")
    def per_race(p, win_col, pod_col, rank_col):
        w, pod = _hits(p, win_col[0], win_col[1], 1), _hits(p, pod_col[0], pod_col[1], 3)
        rk = p.groupby("race_idx")[rank_col].rank(method="first")
        mae = (rk - p["position"]).abs().groupby(p["race_idx"]).mean()
        return pd.DataFrame({"win": w, "pod": pod, "mae": mae})

    base_key = f"baseline: {bl}"
    pr = {base_key: per_race(ref, ("base", "min"), ("base", "min"), "base")}
    for label, p in res.items():
        pr[label] = per_race(p, ("p_win", "max"), ("p_podium", "max"), "pred_pos")
    pr = {k: v.reindex(pr[base_key].index) for k, v in pr.items()}
    n = len(pr[base_key])

    print(f"\n  {'Model':<32}{'Winner hit':>11}{'Podium (of 3)':>15}{'All 3 right':>13}{'Pos. error':>12}")
    for k, v in pr.items():
        print(f"  {k:<32}{v['win'].mean():>11.0%}{v['pod'].mean():>15.2f}"
              f"{(v['pod'] == 3).mean():>13.0%}{v['mae'].mean():>12.2f}")

    # how much could these numbers move by luck? resample the races 2000 times
    rng = np.random.default_rng(0)
    idx = rng.integers(0, n, size=(2000, n))

    def boot(x):
        return np.percentile(np.asarray(x, float)[idx].mean(axis=1), [2.5, 97.5])

    print(f"\n  How sure are these numbers? 95% range if the {n} races were drawn again (bootstrap)")
    print(f"  {'Model':<32}{'Winner hit':>18}{'Podium (of 3)':>20}{'Pos. error':>18}")
    for k, v in pr.items():
        lw, hw = boot(v["win"])
        lp, hp = boot(v["pod"])
        le, he = boot(v["mae"])
        print(f"  {k:<32}{f'{lw:.0%}-{hw:.0%}':>18}{f'{lp:.2f}-{hp:.2f}':>20}{f'{le:.2f}-{he:.2f}':>18}")

    print(f"\n  Each model vs the baseline on the SAME races (the fair comparison)")
    print(f"  {'Model':<32}{'Winner hit (pts)':>22}{'Pos. error (places)':>24}   Verdict")
    b = pr[base_key]
    for k, v in pr.items():
        if k == base_key:
            continue
        dw, de = (v["win"] - b["win"]).to_numpy(float), (v["mae"] - b["mae"]).to_numpy(float)
        lw, hw = boot(dw)
        le, he = boot(de)
        if he < 0 and de.mean() <= -0.05:
            verdict = "clearly better (error)"
        elif le > 0 and de.mean() >= 0.05:
            verdict = "clearly worse (error)"
        elif he < 0 or le > 0:
            verdict = "real but tiny (<0.05 places)"
        else:
            verdict = "no clear difference"
        print(f"  {k:<32}{f'{dw.mean() * 100:+.1f} [{lw * 100:+.0f},{hw * 100:+.0f}]':>22}"
              f"{f'{de.mean():+.2f} [{le:+.2f},{he:+.2f}]':>24}   {verdict}")
    print("  ([low,high] = 95% range. If it includes 0, the difference could just be luck.)")

    sea_of = ref.drop_duplicates("race_idx").set_index("race_idx")["season"].reindex(b.index)
    cnt = sea_of.value_counts()
    seasons = sorted(cnt.index)
    print(f"\n  Season by season: winner hit / position error (is it stable over time?)")
    print(f"  {'Model':<32}" + "".join(f"{f'{int(sn)} (n={cnt[sn]})':>16}" for sn in seasons))
    for k, v in pr.items():
        g = v.groupby(sea_of)
        wm, em = g["win"].mean(), g["mae"].mean()
        print(f"  {k:<32}" + "".join(f"{f'{wm[sn]:.0%} / {em[sn]:.2f}':>16}" for sn in seasons))

    print(f"\n  Quality of the PROBABILITIES (lower Brier / log loss = better; skill > 0 beats a coin-flip guess)")
    print(f"  {'Model':<32}{'Brier win':>10}{'skill':>7}{'LogLoss':>9}{'Brier podium':>14}{'skill':>7}{'LogLoss':>9}")
    nd = ref.groupby("race_idx")["code"].transform("size").to_numpy()
    yw, yp = (ref["position"] == 1).to_numpy(float), (ref["position"] <= 3).to_numpy(float)
    nb_w, nb_p = np.mean((1 / nd - yw) ** 2), np.mean((3 / nd - yp) ** 2)
    for label, p in res.items():
        bw, bp = np.mean((p["p_win"] - yw) ** 2), np.mean((p["p_podium"] - yp) ** 2)
        print(f"  {label:<32}{bw:>10.4f}{1 - bw / nb_w:>7.2f}{_logloss(p['p_win'], yw):>9.3f}"
              f"{bp:>14.4f}{1 - bp / nb_p:>7.2f}{_logloss(p['p_podium'], yp):>9.3f}")

    label, p = list(res.items())[-1]
    print(f"\n  CALIBRATION of the last model ({label}):")
    _calib(p, "p_win", yw, WIN_BINS, "Win probability - does '30%' really win 30% of the time?")
    _calib(p, "p_podium", yp, POD_BINS, "Podium probability")

    # would a calibration correction help? fit on the earlier races, score on the later ones
    cut = np.median(p["race_idx"].unique())
    a, b = (p["race_idx"] <= cut).to_numpy(), (p["race_idx"] > cut).to_numpy()
    print("\n  Calibration fix check (isotonic, fit on earlier half, scored on later half):")
    for col, y in (("p_win", yw), ("p_podium", yp)):
        iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(p.loc[a, col], y[a])
        before = np.mean((p.loc[b, col] - y[b]) ** 2)
        after = np.mean((iso.predict(p.loc[b, col]) - y[b]) ** 2)
        verdict = "helps" if after < before * 0.98 else "no real gain"
        print(f"    {col:<9} Brier {before:.4f} -> {after:.4f}   ({verdict})")
    print("  (Brier skill: 0 = no better than guessing equal chances for every driver; higher is better)")

    yd = (p["dnf"] == 1).to_numpy(float)
    bd, nbd = np.mean((p["p_dnf"] - yd) ** 2), np.mean((yd.mean() - yd) ** 2)
    print(f"\n  RETIREMENT model: Brier {bd:.4f}, skill {1 - bd / nbd:.2f} vs always saying the average "
          f"rate ({yd.mean():.0%} of cars retire)")
    _calib(p, "p_dnf", yd, DNF_BINS, "Retirement probability - does '20%' really retire 20% of the time?")

    if "p_top10" in p.columns:
        y10 = (p["position"] <= 10).to_numpy(float)
        b10, nb10 = np.mean((p["p_top10"] - y10) ** 2), np.mean((10 / nd - y10) ** 2)
        inside = ((p["position"] >= p["pos_lo"]) & (p["position"] <= p["pos_hi"])).mean()
        print(f"\n  TOP-10 chance: Brier {b10:.4f}, skill {1 - b10 / nb10:.2f}")
        print(f"  80% POSITION RANGE: the real finishing position was inside the range {inside:.0%} of the time "
              f"(target 80%)")


# --------------------------------------------------------------------------- #
# Predict the upcoming COTA race
# --------------------------------------------------------------------------- #
def build_upcoming(hist, season, mode, grid_override):
    race = cota_race(season)
    if race is None:
        sys.exit(f"No COTA race found on the {season} calendar.")
    rnd = int(race["round"])
    if ((hist["season"] == season) & (hist["rnd"] == rnd)).any():
        sys.exit(f"The {season} {race['raceName']} has already been run - see the backtest above.")

    # driver / team line-up = latest completed race of this season (else last season's finale)
    cur = hist[hist["season"] == season]
    base = cur if len(cur) else hist[hist["season"] == season - 1]
    last = base[(base["season"] == base["season"].max()) & (base["rnd"] == base["rnd"].max())]
    entries = last[["driverId", "code", "constructorId"]].drop_duplicates("driverId")

    qd = jget(f"{season}/{rnd}/qualifying.json", ttl=600)["MRData"]["RaceTable"]["Races"]
    qres = {x["Driver"]["driverId"]: x for x in qd[0]["QualifyingResults"]} if qd else {}
    if qres:
        # This weekend's qualifying is a better entry list than the last race.
        entries = pd.DataFrame([dict(driverId=d, code=q["Driver"].get("code", d[:3].upper()),
                                     constructorId=q["Constructor"]["constructorId"])
                                for d, q in qres.items()])

    if mode == "auto":
        mode = "post" if (qres or grid_override) else "pre"
    if mode == "post" and not (qres or grid_override):
        print("! No qualifying results yet - falling back to pre-qualifying mode.")
        mode = "pre"

    rows = []
    for _, e in entries.iterrows():
        q = qres.get(e.driverId)
        rows.append(dict(
            season=season, rnd=rnd, race_name=race["raceName"], circuitId=CIRCUIT_ID,
            driverId=e.driverId, code=e.code, constructorId=e.constructorId,
            grid=0, position=np.nan, points=np.nan, dnf=np.nan,
            quali_pos=float(q["position"]) if q else np.nan,
            q1=parse_time(q.get("Q1")) if q else np.nan,
            q2=parse_time(q.get("Q2")) if q else np.nan,
            q3=parse_time(q.get("Q3")) if q else np.nan,
        ))
    up = pd.DataFrame(rows)
    if mode == "post":
        if grid_override:
            codes = [c.strip().upper() for c in grid_override.split(",")]
            if any(not c for c in codes) or len(codes) != len(set(codes)):
                raise ValueError("Grid codes must be nonempty and unique")
            unknown = set(codes) - set(up["code"])
            if unknown:
                raise ValueError(f"Unknown grid driver codes: {', '.join(sorted(unknown))}")
            if not qres and len(codes) != len(up):
                raise ValueError("Without qualifying results, --grid must specify every driver in order")
            order = {c: i + 1 for i, c in enumerate(codes)}
            up["grid"] = up["code"].map(order)
            rest = up["grid"].isna()           # unlisted drivers line up behind, in quali order
            up.loc[rest, "grid"] = (up.loc[rest, "quali_pos"].fillna(20)
                                      .rank(method="first") + len(order))
        else:
            up["grid"] = up["quali_pos"]
            print("! Grid uses qualifying order. Apply confirmed grid penalties with --grid.")
    return up, rnd, race["raceName"], mode


USE_COLOR = False
TEAM_COLOR = {"red_bull": 33, "ferrari": 196, "mercedes": 51, "mclaren": 208, "aston_martin": 41,
              "alpine": 213, "williams": 39, "rb": 75, "haas": 253, "sauber": 82, "audi": 82,
              "cadillac": 245}


def _c(text, code=None, bold=False):
    """Wrap text in an ANSI colour (256-colour palette) when colours are enabled."""
    if not USE_COLOR:
        return str(text)
    pre = ("\033[1m" if bold else "") + (f"\033[38;5;{code}m" if code is not None else "")
    return f"{pre}{text}\033[0m" if pre else str(text)


def _pc(p):                                   # colour for "chance of something good"
    return 46 if p >= .40 else 82 if p >= .20 else 226 if p >= .10 else 252 if p >= .02 else 240


def _dc(p):                                   # colour for "chance of retiring"
    return 196 if p >= .25 else 214 if p >= .15 else 245


def render_prediction(pred, season, name, mode, pace, env, dnf_base):
    d = pred.copy()
    d["p_dnf_show"] = 0.5 * d["p_dnf"] + 0.5 * dnf_base      # shrunk: the retirement model had no proven skill
    tc = lambda cid: TEAM_COLOR.get(cid, 255)
    win = d.nlargest(1, "p_win").iloc[0]
    pod = d.nlargest(3, "p_podium")
    cont = d[d["p_podium"] >= 0.10].sort_values("p_podium", ascending=False)
    post = mode == "post"

    bar = "=" * 78
    print("\n" + _c(bar, 51))
    print(_c(f"  F1 RACE PREDICTION   |   {season} {name}", 51, True))
    print(_c(f"  model: {mode}-qualifying" + (" + car pace" if pace else "")
             + (" + track & weather" if env else ""), 245))
    print(_c(bar, 51))

    if env:
        r0 = d.iloc[0]
        print(_c("\nWEATHER (start of race)", 51, True))
        if pd.notna(r0["air_temp"]):
            prob = f"   rain chance {r0['rain_prob']:.0f}%" if pd.notna(r0["rain_prob"]) else ""
            state = _c("WET", 226, True) if r0["wet"] == 1 else _c("dry", 46, True)
            print(f"  air {r0['air_temp']:.0f} C   humidity {r0['humidity']:.0f}%   wind {r0['wind']:.0f} km/h"
                  f"   rain {r0['precip_mm']:.1f} mm{prob}   -> {state}")
        else:
            print("  not available yet (forecast covers about 16 days) - treated as unknown")
        print(_c("TRACK PROFILE (what this circuit has done to results before)", 51, True))
        items = [("pole wins the race", "circuit_pole_win", "{:.0%}"),
                 ("cars that retire", "circuit_dnf", "{:.0%}"),
                 ("grid -> finish correlation", "circuit_grid_corr", "{:.2f}"),
                 ("races with a safety car", "circuit_sc_rate", "{:.0%}"),
                 ("avg places moved vs grid", "circuit_chaos", "{:.1f}"),
                 ("altitude", "elevation", "{:.0f} m")]
        parts = [f"{lab}: " + _c(fmt.format(r0[col]), 226) for lab, col, fmt in items if pd.notna(r0[col])]
        for i in range(0, len(parts), 3):
            print("  " + "   |   ".join(parts[i:i + 3]))

    if pace and d["team_rpace_form"].notna().any():
        cp = (d.groupby("constructorId")[["team_rpace_form", "team_qpace_form", "team_ts_form"]]
                .first().sort_values("team_rpace_form"))
        print(_c(f"\nCAR PACE {season} (last 3 races, gap to the fastest; lower = faster)", 51, True))
        print(_c(f"  {'Team':<15}{'Race pace':>11}{'Quali':>9}{'Top speed':>12}", 250, True))
        for cid, r in cp.iterrows():
            ts = f"{r['team_ts_form']:.1f} km/h" if pd.notna(r["team_ts_form"]) else "-"
            print("  " + _c(f"{cid:<15}", tc(cid), True) + f"{r['team_rpace_form']:>10.2f}%{r['team_qpace_form']:>8.2f}%{ts:>12}")

    print(_c("\nRACE", 51, True) + " - most likely winner: "
          + _c(f"{win.code} ({win.p_win:.1%})", tc(win.constructorId), True))
    print("Top podium candidates (unordered):  " + ", ".join(_c(f"{r.code} ({r.p_podium:.0%})", tc(r.constructorId), True)
                                              for _, r in pod.iterrows()))
    print(f"Podium contenders (>=10%): " + ", ".join(_c(r.code, tc(r.constructorId)) for _, r in cont.iterrows()))
    print(f"Expected retirements: about {d['p_dnf_show'].sum():.1f} of {len(d)} cars\n")

    hdr = (f"{'#':>3} {'Driver':<7}{'Team':<15}" + (f"{'Grid':>5}" if post else "")
           + f"{'P(win)':>9}{'P(podium)':>11}{'P(top-10)':>11}{'P(DNF)*':>9}{'E[pos]':>8}  {'80% range':<10}")
    print(_c(hdr, 250, True))
    print(_c("-" * len(hdr), 240))
    for i, r in enumerate(d.itertuples(), 1):
        row = [_c(f"{i:>3} ", 245), _c(f"{r.code:<7}", tc(r.constructorId), True),
               _c(f"{r.constructorId:<15}", tc(r.constructorId))]
        if post:
            row.append(_c(f"{int(r.grid):>5}", 252))
        row += [_c(f"{r.p_win:>9.1%}", _pc(r.p_win)), _c(f"{r.p_podium:>11.1%}", _pc(r.p_podium)),
                _c(f"{r.p_top10:>11.1%}", _pc(r.p_top10 / 2)), _c(f"{r.p_dnf_show:>9.1%}", _dc(r.p_dnf_show)),
                _c(f"{r.pred_pos:>8.1f}", 255), _c(f"  {int(r.pos_lo)}-{int(r.pos_hi)}", 244)]
        print("".join(row))
    print(_c("\n  * P(DNF) is pulled halfway toward the average retirement rate: in testing the retirement model"
             "\n    had no proven skill. E[pos] = expected finishing place; 80% range = places the model gives"
             "\n    a nominal 80% range, whose actual coverage must be checked on held-out races."
             "\n    Probability constraints do not establish calibration. Podium candidates are not P1/P2/P3.", 245))


def predict_upcoming(hist, tr, wx, season, mode, grid_override, plot):
    up, rnd, name, mode = build_upcoming(hist, season, mode, grid_override)
    full = add_features(pd.concat([hist, up], ignore_index=True), tr, wx)
    pace, env = tr is not None, wx is not None
    feats = get_features(mode, pace, env)
    test = full[(full["season"] == season) & (full["rnd"] == rnd)]
    train = full[full["position"].notna() & (full["race_idx"] < test["race_idx"].min())]
    pred = fit_predict(train, test, feats, extras=True).sort_values("pred_pos").reset_index(drop=True)

    render_prediction(pred, season, name, mode, pace, env, float(train["dnf"].mean()))
    pred.to_csv("cota_prediction_improved.csv", index=False)
    latest = train.sort_values(["season", "rnd"]).iloc[-1]
    meta = dict(season=season, round=rnd, mode=mode, pace=pace, environment=env,
                generated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                training_entries=len(train), training_races=int(train["race_idx"].nunique()),
                last_training_race=str(latest["race_name"]),
                last_training_date=str(latest.get("race_date")),
                features=feats, calibrated=False, podium_ordered=False)
    Path("cota_prediction_metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"\nTraining data through {meta['last_training_date']}: {meta['last_training_race']}")
    print(_c("Saved full table -> cota_prediction_improved.csv", 245))

    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        p = pred.sort_values("p_podium")
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.barh(p["code"], p["p_podium"] * 100, color="#d62828", label="Podium %")
        ax.barh(p["code"], p["p_win"] * 100, color="#111", label="Win %")
        ax.set_xlabel("Probability (%)")
        ax.set_title(f"{season} {name} - model prediction ({mode}-quali)")
        ax.legend()
        plt.tight_layout()
        plt.savefig("cota_prediction_improved.png", dpi=150)
        print("Saved plot -> cota_prediction_improved.png")


# --------------------------------------------------------------------------- #
# Race trace plot (gap to a constant-pace reference line, lap by lap)
# --------------------------------------------------------------------------- #
def plot_race_trace(season, race_name=None, out="race_trace.png"):
    """Classic 'race trace': each driver's time vs a reference running at the winner's
    average pace. Up = ahead of the reference; steep drops = pit stops / slow laps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if race_name is None:
        race = cota_race(season)
        if race is None:
            sys.exit(f"No COTA race on the {season} calendar.")
        race_name = race["raceName"]
    ev = match_folder(race_name, list_event_folders(season))
    url = TI_RAW.format(season=season, event=quote(ev), session="Race")
    r = HTTP.get(url, timeout=180)
    if r.status_code == 404:
        sys.exit(f"No TracingInsights race file for {season} {ev} (yet).")
    r.raise_for_status()
    data = r.json()
    n = len(data.get("lap", []))
    df = pd.DataFrame({k: v for k, v in data.items() if isinstance(v, list) and len(v) == n})
    for c in ("lap", "sesT"):
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    df = df.dropna(subset=["lap", "sesT"]).sort_values(["drv", "lap"])
    last = df.groupby("drv").agg(laps=("lap", "max"), t=("sesT", "max"))
    winner = last.sort_values(["laps", "t"], ascending=[False, True]).index[0]
    w = df[df["drv"] == winner].set_index("lap")["sesT"]
    avg = (w.iloc[-1] - w.iloc[0]) / (w.index[-1] - w.index[0])
    ref = lambda lap: w.iloc[0] + (lap - w.index[0]) * avg
    df["trace"] = ref(df["lap"]) - df["sesT"]

    fig, ax = plt.subplots(figsize=(11, 7))
    for d, g in df.groupby("drv"):
        ax.plot(g["lap"], g["trace"], lw=1.4, label=d)
        ax.annotate(d, (g["lap"].iloc[-1], g["trace"].iloc[-1]), fontsize=7, va="center")
    ax.set_xlabel("Lap")
    ax.set_ylabel(f"Seconds ahead of {winner}'s average pace")
    ax.set_title(f"{season} {ev} - race trace")
    ax.set_ylim(df["trace"].quantile(0.01) - 5, df["trace"].max() + 5)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"Saved race trace -> {out}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Predict the COTA race winner & podium")
    ap.add_argument("--season", type=int, default=dt.date.today().year)
    ap.add_argument("--first-season", type=int, default=2014, help="start of training data")
    ap.add_argument("--mode", choices=["auto", "pre", "post"], default="auto")
    ap.add_argument("--grid", default=None, help="driver codes in grid order, e.g. VER,NOR,LEC,...")
    ap.add_argument("--no-tracing", action="store_true", help="skip TracingInsights car-pace data")
    ap.add_argument("--no-env", action="store_true", help="skip track statistics + weather (Open-Meteo)")
    ap.add_argument("--tracing-from", type=int, default=2018,
                    help="first season to download from TracingInsights (later = faster)")
    ap.add_argument("--no-backtest", action="store_true")
    ap.add_argument("--eval-all", action="store_true",
                    help="big backtest over ALL circuits of the last seasons + probability calibration")
    ap.add_argument("--eval-seasons", type=int, default=6,
                    help="seasons used by --eval-all (0 = ALL available history)")
    ap.add_argument("--eval-step", type=int, default=3,
                    help="--eval-all refits every N races (bigger = faster, slightly staler)")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--no-color", action="store_true", help="plain text output (no colours)")
    ap.add_argument("--trace-plot", type=int, metavar="SEASON",
                    help="draw the race trace of that season's COTA race and exit (e.g. 2025)")
    ap.add_argument("--trace-race", default=None, help='race name for --trace-plot (default: COTA)')
    args = ap.parse_args()
    if args.first_season > args.season or args.eval_step < 1:
        ap.error("--first-season must not exceed --season; --eval-step must be positive")
    if not args.no_env:
        print("! Weather backtests use realized historical weather, not forecasts issued before each race."
              "\n  Their scores are hindsight-assisted. Use --no-env for a cleaner comparison.")
    global USE_COLOR
    if os.name == "nt":
        os.system("")                              # switches on ANSI colours in the Windows terminal
    USE_COLOR = (not args.no_color) and sys.stdout.isatty()

    if args.trace_plot:
        plot_race_trace(args.trace_plot, args.trace_race)
        return

    print("Downloading race data (cached after the first run)...")
    hist = load_history(args.first_season, args.season)
    st = hist["status"].value_counts(normalize=True).head(6)
    print("Result statuses in the data: " + ", ".join(f"{k} {v:.0%}" for k, v in st.items())
          + f"  -> counted as retirements: {hist['dnf'].mean():.0%} of entries")

    up_race = cota_race(args.season)               # include the upcoming weekend
    races = hist[["season", "rnd", "race_name", "race_date", "race_time", "lat", "lon"]]
    races = races.drop_duplicates(["season", "rnd"])
    if up_race and not ((races["season"] == args.season) & (races["rnd"] == int(up_race["round"]))).any():
        loc = up_race["Circuit"].get("Location", {})
        races = pd.concat([races, pd.DataFrame([dict(
            season=args.season, rnd=int(up_race["round"]), race_name=up_race["raceName"],
            race_date=up_race.get("date"), race_time=up_race.get("time"),
            lat=loc.get("lat"), lon=loc.get("long"))])], ignore_index=True)

    tr = None if args.no_tracing else load_tracing(races, args.tracing_from)
    wx = None if args.no_env else load_weather(races)
    full = add_features(hist, tr, wx)
    pace, env = tr is not None, wx is not None

    def make_variants(m):
        v = {"Jolpica results only": get_features(m, False, False)}
        if pace:
            v["+ car pace (TracingInsights)"] = get_features(m, True, False)
        if env:
            v["+ car pace + track & weather" if pace else "+ track & weather"] = get_features(m, pace, True)
        return v

    if not args.no_backtest:
        start = max(2016, args.tracing_from + 1) if pace else 2016    # same folds for every variant
        for m in ("post", "pre"):
            backtest(full, m, make_variants(m), start)
    if args.eval_all:
        for m in ("post", "pre"):
            eval_all(full, m, make_variants(m), args.eval_seasons, args.eval_step)
    predict_upcoming(hist, tr, wx, args.season, args.mode, args.grid, args.plot)


if __name__ == "__main__":
    main()
