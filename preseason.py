import gc
import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import fastf1

# ---------------------------------------------------------------------------
# Monkeypatch: Fix FastF1 2026 Pre-Season Path (Day N vs Practice N)
# ---------------------------------------------------------------------------
import fastf1._api
import numpy as np
import orjson
import pandas as pd
import psutil
import requests

_original_make_path = fastf1._api.make_path


def _patched_make_path(wname, wdate, sname, sdate):
    path = _original_make_path(wname, wdate, sname, sdate)
    if "2026" in wdate:
        if "Practice_1" in path:
            path = path.replace("Practice_1", "Day_1")
        elif "Practice_2" in path:
            path = path.replace("Practice_2", "Day_2")
        elif "Practice_3" in path:
            path = path.replace("Practice_3", "Day_3")
    return path


fastf1._api.make_path = _patched_make_path

# ---------------------------------------------------------------------------
# Constants & Configuration
# ---------------------------------------------------------------------------

DEFAULT_YEAR = 2026
# Set these to an integer (e.g. 1) to filter, or None to process all
TARGET_TEST_NUMBER = 2  # e.g. 1 for "Test 1"
TARGET_SESSION_NUMBER = 3  # e.g. 1 for "Practice 1"

# Deprecated but kept for compatibility if needed (though now unused by logic)
SESSION_NUMBER = 3
PROTO = "https"
HOST = "api.multiviewer.app"
HEADERS = {"User-Agent": "FastF1/"}
ORJSON_OPTS = orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_NON_STR_KEYS
EPS = np.finfo(float).eps

# Pre-allocated smoothing kernels
_KERNEL_3 = np.ones(3, dtype=np.float64) / 3.0
_KERNEL_9 = np.ones(9, dtype=np.float64) / 9.0

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("preseason_extraction.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("preseason_extractor")
logging.getLogger("fastf1").setLevel(logging.WARNING)
logging.getLogger("fastf1").propagate = False

_MISSING_TEXT_VALUES = {
    "",
    "null",
    "nan",
    "nat",
    "none",
    "inf",
    "-inf",
    "infinity",
    "-infinity",
}


# ---------------------------------------------------------------------------
# Helper Functions (Copied from main_optimized.py for standalone execution)
# ---------------------------------------------------------------------------
def _write_json(path: str, obj, normalize_missing: bool = False) -> None:
    if normalize_missing:
        obj = _normalize_missing_for_json(obj)
    with open(path, "wb") as f:
        f.write(orjson.dumps(obj, option=ORJSON_OPTS))


def _td_col_to_seconds(series: pd.Series) -> list:
    if series.empty:
        return []
    seconds = series.dt.total_seconds().to_numpy()
    mask = series.isna().to_numpy()
    out = np.round(seconds, 3).astype(object)
    out[mask] = "None"
    return out.tolist()


def _col_to_list_str_or_none(col) -> list:
    if isinstance(col, np.ndarray):
        vals = col
    else:
        if col.empty:
            return []
        vals = col.to_numpy()
    if len(vals) == 0:
        return []
    mask = pd.isna(vals)
    out = np.empty(vals.shape, dtype=object)
    out[mask] = "None"
    valid_vals = vals[~mask]
    s_vals = np.array([str(v).strip().lower() for v in valid_vals])
    missing_mask = np.isin(s_vals, list(_MISSING_TEXT_VALUES))
    str_vals = np.array([str(v) for v in valid_vals])
    out[~mask] = np.where(missing_mask, "None", str_vals)
    return out.tolist()


def _col_to_list_int_or_none(series: pd.Series) -> list:
    if series.empty:
        return []
    vals = series.to_numpy()
    mask = pd.isna(vals)
    out = np.empty(vals.shape, dtype=object)
    out[mask] = "None"
    out[~mask] = vals[~mask].astype(int)
    return out.tolist()


def _col_to_list_bool_or_none(series: pd.Series) -> list:
    if series.empty:
        return []
    vals = series.to_numpy()
    mask = pd.isna(vals)
    out = np.empty(vals.shape, dtype=object)
    out[mask] = "None"
    out[~mask] = vals[~mask].astype(bool)
    return out.tolist()


def _series_to_json_list(series: pd.Series) -> list:
    if series.empty:
        return []

    if pd.api.types.is_timedelta64_dtype(series.dtype):
        return _td_col_to_seconds(series)

    vals = series.to_numpy()
    if pd.api.types.is_float_dtype(series.dtype):
        vals_f = vals.astype(np.float64, copy=False)
        mask = ~np.isfinite(vals_f)
    else:
        mask = pd.isna(vals)
    out = np.empty(vals.shape, dtype=object)
    out[mask] = "None"

    valid = ~mask
    if not valid.any():
        return out.tolist()

    if pd.api.types.is_bool_dtype(series.dtype):
        out[valid] = vals[valid].astype(bool)
    elif pd.api.types.is_integer_dtype(series.dtype):
        out[valid] = vals[valid].astype(int)
    elif pd.api.types.is_float_dtype(series.dtype):
        out[valid] = vals[valid].astype(float)
    else:
        valid_vals = vals[valid]
        s_vals = np.array([str(v).strip().lower() for v in valid_vals])
        missing_mask = np.isin(s_vals, list(_MISSING_TEXT_VALUES))
        str_vals = np.array([str(v) for v in valid_vals])
        out[valid] = np.where(missing_mask, "None", str_vals)

    return out.tolist()


def _scalar_to_json_primitive_or_none(value):
    if isinstance(value, (float, np.floating)):
        return "None" if not np.isfinite(value) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, str):
        return "None" if value.strip().lower() in _MISSING_TEXT_VALUES else value
    if pd.isna(value):
        return "None"
    return value


def _normalize_missing_for_json(value):
    if isinstance(value, dict):
        return {k: _normalize_missing_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize_missing_for_json(v) for v in value]
    if isinstance(value, tuple):
        return [_normalize_missing_for_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_normalize_missing_for_json(v) for v in value.tolist()]
    return _scalar_to_json_primitive_or_none(value)


def _dataframe_to_column_lists(df: pd.DataFrame) -> Dict[str, list]:
    if df is None or df.empty:
        return {}
    return {col: _series_to_json_list(df[col]) for col in df.columns}


_LAP_WEATHER_COL_MAP = (
    ("wT", "Time"),
    ("wAT", "AirTemp"),
    ("wH", "Humidity"),
    ("wP", "Pressure"),
    ("wR", "Rainfall"),
    ("wTT", "TrackTemp"),
    ("wWD", "WindDirection"),
    ("wWS", "WindSpeed"),
)
LAP_WEATHER_KEYS = tuple(k for k, _ in _LAP_WEATHER_COL_MAP)

_RCM_COL_MAP = (
    ("time", "Time"),
    ("cat", "Category"),
    ("msg", "Message"),
    ("status", "Status"),
    ("flag", "Flag"),
    ("scope", "Scope"),
    ("sector", "Sector"),
    ("dNum", "RacingNumber"),
    ("lap", "Lap"),
)


def _session_weather_to_column_lists(weather_df: pd.DataFrame) -> Dict[str, list]:
    if weather_df is None or weather_df.empty:
        return {}

    out: Dict[str, list] = {}
    for short_key, weather_col in _LAP_WEATHER_COL_MAP:
        if weather_col in weather_df.columns:
            out[short_key] = _series_to_json_list(weather_df[weather_col])
    return out


def _session_rcm_to_column_lists(rcm_df: pd.DataFrame) -> Dict[str, list]:
    if rcm_df is None or rcm_df.empty:
        return {}

    out: Dict[str, list] = {}
    for short_key, rcm_col in _RCM_COL_MAP:
        if rcm_col in rcm_df.columns:
            out[short_key] = _series_to_json_list(rcm_df[rcm_col])
    return out


def _lap_weather_to_column_lists(
    laps: pd.DataFrame, weather_df: pd.DataFrame = None
) -> Dict[str, list]:
    n_laps = len(laps)
    if n_laps == 0:
        return {k: [] for k in LAP_WEATHER_KEYS}

    none_row = ["None"] * n_laps
    out = {k: none_row.copy() for k in LAP_WEATHER_KEYS}

    if weather_df is None:
        if not hasattr(laps, "get_weather_data"):
            return out
        try:
            weather_df = laps.get_weather_data()
        except Exception:
            return out

    if weather_df is None:
        return out

    for short_key, weather_col in _LAP_WEATHER_COL_MAP:
        if weather_col not in weather_df.columns:
            continue

        values = _series_to_json_list(weather_df[weather_col])
        if len(values) < n_laps:
            values.extend(["None"] * (n_laps - len(values)))
        elif len(values) > n_laps:
            values = values[:n_laps]
        out[short_key] = values

    return out


def _array_to_list_float_or_none(arr: np.ndarray) -> list:
    if arr.size == 0:
        return []
    mask = ~np.isfinite(arr)
    if not mask.any():
        return arr.tolist()
    out = np.empty(arr.shape, dtype=object)
    out[mask] = "None"
    out[~mask] = arr[~mask]
    return out.tolist()


def _array_to_list_int_or_none(arr: np.ndarray) -> list:
    if arr.size == 0:
        return []
    mask = ~np.isfinite(arr)
    if not mask.any():
        return arr.astype(int).tolist()
    out = np.empty(arr.shape, dtype=object)
    out[mask] = "None"
    out[~mask] = arr[~mask].astype(int)
    return out.tolist()


def _smooth_outliers(arr: np.ndarray, threshold: float, use_abs: bool) -> None:
    if use_abs:
        mask = np.abs(arr) > threshold
    else:
        mask = arr > threshold
    if mask.any():
        indices = np.where(mask)[0]
        indices = indices[(indices >= 1) & (indices < len(arr) - 1)]
        if len(indices) > 0:
            arr[indices] = arr[indices - 1]


def _compute_accelerations(
    speed: np.ndarray,
    time_arr: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    dist: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Convert speed km/h -> m/s as float64
    vx = speed * (1.0 / 3.6)
    if vx.dtype != np.float64:
        vx = vx.astype(np.float64)
    time_f = (time_arr / np.timedelta64(1, "s")).astype(np.float64)

    # Ensure float64 only when needed
    x_f = x if x.dtype == np.float64 else x.astype(np.float64)
    y_f = y if y.dtype == np.float64 else y.astype(np.float64)
    z_f = z if z.dtype == np.float64 else z.astype(np.float64)
    dist_f = dist if dist.dtype == np.float64 else dist.astype(np.float64)

    # --- X acceleration ---
    dtime = np.gradient(time_f)
    ax = np.gradient(vx) / dtime
    _smooth_outliers(ax, 25.0, use_abs=False)
    ax = np.convolve(ax, _KERNEL_3, mode="same")

    # --- Shared gradient for Y and Z ---
    dx = np.gradient(x_f)
    ds = np.gradient(dist_f)

    # --- Y acceleration ---
    dy = np.gradient(y_f)
    theta = np.arctan2(dy, dx + EPS)
    theta[0] = theta[1]
    dtheta = np.gradient(np.unwrap(theta))
    _smooth_outliers(dtheta, 0.5, use_abs=True)
    C = dtheta / (ds + 0.0001)
    ay = np.square(vx) * C
    ay[np.abs(ay) > 150] = 0
    ay = np.convolve(ay, _KERNEL_9, mode="same")

    # --- Z acceleration ---
    dz = np.gradient(z_f)
    z_theta = np.arctan2(dz, dx + EPS)
    z_theta[0] = z_theta[1]
    z_dtheta = np.gradient(np.unwrap(z_theta))
    _smooth_outliers(z_dtheta, 0.5, use_abs=True)
    z_C = z_dtheta / (ds + 0.0001)
    az = np.square(vx) * z_C
    az[np.abs(az) > 150] = 0
    az = np.convolve(az, _KERNEL_9, mode="same")

    return ax, ay, az, time_f


def _process_telemetry_to_dict(telemetry: pd.DataFrame, data_key: str) -> dict:
    time_arr = telemetry["Time"].to_numpy()
    speed = telemetry["Speed"].to_numpy()
    x = telemetry["X"].to_numpy()
    y = telemetry["Y"].to_numpy()
    z = telemetry["Z"].to_numpy()
    dist = telemetry["Distance"].to_numpy()

    ax, ay, az, time_s = _compute_accelerations(speed, time_arr, x, y, z, dist)

    drs_raw = telemetry["DRS"].to_numpy()
    drs = np.isin(drs_raw, [10, 12, 14]).astype(np.int8)
    brake = telemetry["Brake"].to_numpy().astype(bool).astype(np.int8)
    driver_ahead = (
        telemetry["DriverAhead"]
        if "DriverAhead" in telemetry.columns
        else np.full(len(telemetry), np.nan)
    )
    distance_to_driver_ahead = (
        telemetry["DistanceToDriverAhead"].to_numpy()
        if "DistanceToDriverAhead" in telemetry.columns
        else np.full(len(telemetry), np.nan, dtype=np.float64)
    )

    return {
        "tel": {
            "time": _array_to_list_float_or_none(time_s),
            "rpm": _array_to_list_float_or_none(telemetry["RPM"].to_numpy()),
            "speed": _array_to_list_float_or_none(speed),
            "gear": _array_to_list_int_or_none(telemetry["nGear"].to_numpy()),
            "throttle": _array_to_list_float_or_none(telemetry["Throttle"].to_numpy()),
            "brake": _array_to_list_int_or_none(brake),
            "drs": _array_to_list_int_or_none(drs),
            "distance": _array_to_list_float_or_none(dist),
            "rel_distance": _array_to_list_float_or_none(
                telemetry["RelativeDistance"].to_numpy()
                if "RelativeDistance" in telemetry.columns
                else np.full(len(telemetry), np.nan, dtype=np.float64)
            ),
            "DriverAhead": _col_to_list_str_or_none(driver_ahead),
            "DistanceToDriverAhead": _array_to_list_float_or_none(
                distance_to_driver_ahead
            ),
            "acc_x": _array_to_list_float_or_none(ax),
            "acc_y": _array_to_list_float_or_none(ay),
            "acc_z": _array_to_list_float_or_none(az),
            "x": _array_to_list_float_or_none(x),
            "y": _array_to_list_float_or_none(y),
            "z": _array_to_list_float_or_none(z),
            "dataKey": data_key,
        }
    }


def check_memory_usage(threshold_percent=80, session_cache=None, circuit_cache=None):
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_percent = process.memory_percent()

    logger.info(
        f"Current memory usage: {memory_percent:.2f}% "
        f"({memory_info.rss / 1024 / 1024:.2f} MB)"
    )

    if memory_percent > threshold_percent:
        logger.warning(
            f"Memory usage exceeds {threshold_percent}% threshold, clearing caches"
        )
        if session_cache is not None:
            session_cache.clear()
        if circuit_cache is not None:
            circuit_cache.clear()
        gc.collect()

        new_pct = psutil.Process(os.getpid()).memory_percent()
        logger.info(f"New memory usage after clearing caches: {new_pct:.2f}%")
        return True

    return False


# ---------------------------------------------------------------------------
# Pre-Season Extractor
# ---------------------------------------------------------------------------
class PreSeasonExtractor:
    """Extract telemetry from pre-season testing sessions."""

    def __init__(self, year: int = DEFAULT_YEAR):
        self.year = year
        self._session_cache: Dict[str, fastf1.core.Session] = {}
        self._circuit_cache: Dict[str, dict] = {}

    def discover_testing_events(self) -> List[dict]:
        """Return list of testing events with their test_number and session count.
        Forces backend='fastf1' to avoid 'Day 1' session naming issues.
        """
        try:
            schedule = fastf1.get_event_schedule(
                self.year, include_testing=True, backend="fastf1"
            )
        except Exception as e:
            logger.warning(
                f"Failed to load schedule with backend='fastf1': {e}. "
                "Retrying with default backend..."
            )
            schedule = fastf1.get_event_schedule(self.year, include_testing=True)

        testing = schedule[schedule["EventFormat"] == "testing"]

        events = []
        for idx, (_, row) in enumerate(testing.iterrows(), start=1):
            n_sessions = 0
            for s in range(1, 6):
                col = f"Session{s}"
                if col in row.index:
                    val = row[col]
                    if pd.notna(val) and str(val).strip() not in ("", "None"):
                        n_sessions += 1
            events.append(
                {
                    "test_number": idx,
                    "event_name": row.get("EventName", f"Test {idx}"),
                    "sessions": max(n_sessions, 1),
                }
            )

        logger.info(f"Discovered {len(events)} testing event(s) for {self.year}")
        for e in events:
            logger.info(
                f"  Test {e['test_number']}: {e['event_name']} "
                f"({e['sessions']} sessions)"
            )
        return events


    @staticmethod
    def _fix_missing_retired_on_track_laps(f1session: fastf1.core.Session) -> int:
        """Add partial last laps for cars that stopped on track.

        FastF1's ``_fix_missing_laps_retired_on_track`` skips generation when
        *any* prior session-status is ``Finished``. That is wrong for
        multi-segment sessions (Qualifying / Sprint Qualifying): Q1/Q2 already
        emit ``Finished`` before a Q3 crash, so the crash lap is dropped
        (e.g. VER lap 11, 2026 Austrian GP Qualifying).

        Mirror FastF1's generation logic, but treat the driver as finished only
        when the *latest* status at/before the last lap end is ``Finished``.
        """
        if not hasattr(f1session, "_laps") or f1session._laps is None:
            return 0
        if not hasattr(f1session, "session_status") or f1session.session_status is None:
            return 0

        laps = f1session._laps
        if laps.empty or "DriverNumber" not in laps.columns:
            return 0

        session_status = f1session.session_status
        car_data = getattr(f1session, "_car_data", None) or {}
        total_laps = getattr(f1session, "_total_laps", None)
        added = 0
        new_rows: List[pd.DataFrame] = []

        for drv in laps["DriverNumber"].unique():
            drv_laps = laps[laps["DriverNumber"] == drv]
            if drv_laps.empty:
                continue

            if len(drv_laps) == 1 and bool(drv_laps["FastF1Generated"].iloc[0]):
                continue

            ref_time = drv_laps["Time"].iloc[-1]
            if pd.isna(ref_time):
                ref_time = drv_laps["LapStartTime"].iloc[-1]
            if pd.isna(ref_time):
                continue

            next_statuses = session_status[session_status["Time"] > ref_time]
            prev_statuses = session_status[session_status["Time"] <= ref_time]

            if total_laps is not None and drv_laps["LapNumber"].max() == total_laps:
                continue

            # FastF1 bug: used .any() on Finished, which fires after Q1/Q2.
            if (
                not prev_statuses.empty
                and prev_statuses["Status"].iloc[-1] == "Finished"
            ):
                continue

            if next_statuses.empty or not (next_statuses["Status"] == "Finished").any():
                continue

            if not pd.isna(drv_laps["PitInTime"].iloc[-1]):
                continue

            if total_laps is not None and drv_laps.shape[0] >= total_laps:
                continue

            if (
                len(drv_laps) >= 2
                and not pd.isna(drv_laps["PitInTime"].iloc[-2])
                and pd.isna(drv_laps["PitOutTime"].iloc[-1])
            ):
                continue

            next_status = next_statuses.iloc[0]
            if next_status["Status"] == "Aborted":
                assumed_end_time = next_status["Time"]
            else:
                assumed_end_time = None
                if drv in car_data:
                    try:
                        assumed_end_time = car_data[drv].loc[
                            (car_data[drv]["SessionTime"] > ref_time)
                            & (car_data[drv]["Speed"] == 0.0)
                        ].iloc[0]["SessionTime"]
                    except (IndexError, KeyError):
                        assumed_end_time = None
                if assumed_end_time is None:
                    assumed_end_time = ref_time + pd.Timedelta(150, "sec")

            new_last = pd.DataFrame(
                {
                    "LapStartTime": [drv_laps["Time"].iloc[-1]],
                    "Time": [assumed_end_time],
                    "Driver": [drv_laps["Driver"].iloc[-1]],
                    "DriverNumber": [drv_laps["DriverNumber"].iloc[-1]],
                    "Team": [drv_laps["Team"].iloc[-1]],
                    "LapNumber": [drv_laps["LapNumber"].iloc[-1] + 1],
                    "Stint": [drv_laps["Stint"].iloc[-1]],
                    "Compound": [drv_laps["Compound"].iloc[-1]],
                    "TyreLife": [drv_laps["TyreLife"].iloc[-1] + 1],
                    "FreshTyre": [drv_laps["FreshTyre"].iloc[-1]],
                    "Position": [np.nan],
                    "FastF1Generated": [True],
                    "IsAccurate": [False],
                }
            )
            if hasattr(f1session, "_add_track_status_to_laps"):
                f1session._add_track_status_to_laps(new_last)
            new_rows.append(new_last)
            added += 1
            logger.info(
                "Added retired-on-track lap %s for driver %s "
                "(FastF1 skipped it due to multi-segment Finished status)",
                int(new_last["LapNumber"].iloc[0]),
                drv_laps["Driver"].iloc[-1],
            )

        if not new_rows:
            return 0

        f1session._laps = (
            pd.concat([f1session._laps, *new_rows], ignore_index=True)
            .__finalize__(f1session._laps)
            .sort_values(by=["DriverNumber", "LapNumber"])
            .reset_index(drop=True)
        )
        return added

    def get_session(
        self, test_number: int, session_number: int, load_telemetry: bool = False
    ) -> fastf1.core.Session:
        cache_key = f"{self.year}-T{test_number}-S{session_number}"
        cached = self._session_cache.get(cache_key)

        if cached is not None:
            if load_telemetry and not getattr(cached, "_telemetry_loaded", False):
                cached.load(telemetry=True, weather=True, messages=True)
                cached._telemetry_loaded = True
                self._session_cache[cache_key] = cached
            # Idempotent: safe on cache hits; needed after late telemetry load.
            self._fix_missing_retired_on_track_laps(cached)
            return cached

        # Enforce 'fastf1' backend to ensure correct session naming ("Practice N")
        f1session = fastf1.get_testing_session(
            self.year, test_number, session_number, backend="fastf1"
        )
        f1session.load(telemetry=load_telemetry, weather=True, messages=True)
        f1session._telemetry_loaded = load_telemetry
        self._fix_missing_retired_on_track_laps(f1session)
        self._session_cache[cache_key] = f1session
        return f1session

    def session_drivers(
        self, test_number: int, session_number: int
    ) -> Dict[str, List[Dict[str, str]]]:
        try:
            f1session = self.get_session(test_number, session_number)
            laps = f1session.laps
            driver_cols = ["Driver", "Team"]
            has_driver_number = "DriverNumber" in laps.columns
            if has_driver_number:
                driver_cols.append("DriverNumber")
            driver_team = laps.drop_duplicates(subset="Driver")[driver_cols]

            results = f1session.results
            result_by_abbr = {}
            result_by_number = {}
            if results is not None and not results.empty:
                for row in results.itertuples():
                    abbr = getattr(row, "Abbreviation", None)
                    if pd.notna(abbr):
                        result_by_abbr[str(abbr)] = row

                    driver_number = getattr(row, "DriverNumber", None)
                    if pd.notna(driver_number):
                        result_by_number[str(driver_number)] = row

            drivers = [
                self._build_driver_info(
                    row=row,
                    has_driver_number=has_driver_number,
                    result_by_abbr=result_by_abbr,
                    result_by_number=result_by_number,
                )
                for row in driver_team.itertuples(index=False)
            ]
            return {"drivers": drivers}
        except Exception as e:
            logger.error(
                f"Error getting drivers for Test {test_number} Session {session_number}: {e}"
            )
            return {"drivers": []}

    def _build_driver_info(
        self,
        row,
        has_driver_number: bool,
        result_by_abbr: Dict[str, object],
        result_by_number: Dict[str, object],
    ) -> Dict[str, str]:
        driver = _scalar_to_json_primitive_or_none(row.Driver)
        team = _scalar_to_json_primitive_or_none(row.Team)
        driver_number = _scalar_to_json_primitive_or_none(
            row.DriverNumber if has_driver_number else "None"
        )

        result_row = result_by_abbr.get(str(driver))
        if result_row is None and driver_number != "None":
            result_row = result_by_number.get(str(driver_number))

        first_name = "None"
        last_name = "None"
        team_color = "None"
        headshot_url = "None"

        if result_row is not None:
            first_name = _scalar_to_json_primitive_or_none(
                getattr(result_row, "FirstName", "None")
            )
            last_name = _scalar_to_json_primitive_or_none(
                getattr(result_row, "LastName", "None")
            )
            team_color = _scalar_to_json_primitive_or_none(
                getattr(result_row, "TeamColor", "None")
            )
            headshot_url = _scalar_to_json_primitive_or_none(
                getattr(
                    result_row,
                    "HeadshotUrl",
                    getattr(result_row, "HeadShotUrl", "None"),
                )
            )

        return {
            "driver": driver,
            "team": team,
            "dn": driver_number,
            "fn": first_name,
            "ln": last_name,
            "tc": team_color,
            "url": headshot_url,
        }

    def laps_data(
        self,
        driver: str,
        f1session: fastf1.core.Session,
        driver_laps: pd.DataFrame = None,
        session_weather_df: pd.DataFrame = None,
    ) -> Dict[str, list]:
        try:
            if driver_laps is None:
                driver_laps = f1session.laps.pick_drivers(driver)

            lap_weather = _lap_weather_to_column_lists(driver_laps, session_weather_df)

            return {
                "time": _td_col_to_seconds(driver_laps["LapTime"]),
                "lap": _col_to_list_int_or_none(driver_laps["LapNumber"]),
                "compound": _col_to_list_str_or_none(driver_laps["Compound"]),
                "stint": _col_to_list_int_or_none(driver_laps["Stint"]),
                "s1": _td_col_to_seconds(driver_laps["Sector1Time"]),
                "s2": _td_col_to_seconds(driver_laps["Sector2Time"]),
                "s3": _td_col_to_seconds(driver_laps["Sector3Time"]),
                "life": _col_to_list_int_or_none(driver_laps["TyreLife"]),
                "pos": _col_to_list_int_or_none(driver_laps["Position"]),
                "status": _col_to_list_str_or_none(driver_laps["TrackStatus"]),
                "pb": _col_to_list_bool_or_none(driver_laps["IsPersonalBest"]),
                # Additional FastF1 lap columns with shortened names
                "sesT": _td_col_to_seconds(driver_laps["Time"]),
                "drv": _col_to_list_str_or_none(driver_laps["Driver"]),
                "dNum": _col_to_list_str_or_none(driver_laps["DriverNumber"]),
                "pout": _td_col_to_seconds(driver_laps["PitOutTime"]),
                "pin": _td_col_to_seconds(driver_laps["PitInTime"]),
                "s1T": _td_col_to_seconds(driver_laps["Sector1SessionTime"]),
                "s2T": _td_col_to_seconds(driver_laps["Sector2SessionTime"]),
                "s3T": _td_col_to_seconds(driver_laps["Sector3SessionTime"]),
                "vi1": _array_to_list_float_or_none(driver_laps["SpeedI1"].to_numpy()),
                "vi2": _array_to_list_float_or_none(driver_laps["SpeedI2"].to_numpy()),
                "vfl": _array_to_list_float_or_none(driver_laps["SpeedFL"].to_numpy()),
                "vst": _array_to_list_float_or_none(driver_laps["SpeedST"].to_numpy()),
                "fresh": _col_to_list_bool_or_none(driver_laps["FreshTyre"]),
                "team": _col_to_list_str_or_none(driver_laps["Team"]),
                "lST": _td_col_to_seconds(driver_laps["LapStartTime"]),
                "lSD": _col_to_list_str_or_none(driver_laps["LapStartDate"]),
                "del": _col_to_list_bool_or_none(driver_laps["Deleted"]),
                "delR": _col_to_list_str_or_none(driver_laps["DeletedReason"]),
                "ff1G": _col_to_list_bool_or_none(driver_laps["FastF1Generated"]),
                "iacc": _col_to_list_bool_or_none(driver_laps["IsAccurate"]),
                **lap_weather,
            }
        except Exception as e:
            logger.error(f"Error getting lap data for {driver}: {e}")
            return {
                k: []
                for k in (
                    "time",
                    "lap",
                    "compound",
                    "stint",
                    "s1",
                    "s2",
                    "s3",
                    "life",
                    "pos",
                    "status",
                    "pb",
                    "sesT",
                    "drv",
                    "dNum",
                    "pout",
                    "pin",
                    "s1T",
                    "s2T",
                    "s3T",
                    "vi1",
                    "vi2",
                    "vfl",
                    "vst",
                    "fresh",
                    "team",
                    "lST",
                    "lSD",
                    "del",
                    "delR",
                    "ff1G",
                    "iacc",
                    *LAP_WEATHER_KEYS,
                )
            }

    def get_circuit_info(self, test_number: int, session_number: int) -> Optional[Dict]:
        cache_key = f"{self.year}-T{test_number}-S{session_number}"
        if cache_key in self._circuit_cache:
            return self._circuit_cache[cache_key]

        try:
            f1session = self.get_session(test_number, session_number)
            circuit_key = f1session.session_info["Meeting"]["Circuit"]["Key"]

            try:
                circuit_info = f1session.get_circuit_info()
                corners = circuit_info.corners
                result = {
                    "CornerNumber": _series_to_json_list(corners["Number"]),
                    "X": _series_to_json_list(corners["X"]),
                    "Y": _series_to_json_list(corners["Y"]),
                    "Angle": _series_to_json_list(corners["Angle"]),
                    "Distance": _series_to_json_list(corners["Distance"]),
                    "Rotation": _scalar_to_json_primitive_or_none(
                        circuit_info.rotation
                    ),
                }
                self._circuit_cache[cache_key] = result
                return result
            except (AttributeError, KeyError):
                circuit_df, rotation = self._get_circuit_info_from_api(circuit_key)
                if circuit_df is not None:
                    result = {
                        "CornerNumber": _series_to_json_list(circuit_df["Number"]),
                        "X": _series_to_json_list(circuit_df["X"]),
                        "Y": _series_to_json_list(circuit_df["Y"]),
                        "Angle": _series_to_json_list(circuit_df["Angle"]),
                        "Distance": _series_to_json_list(circuit_df["Distance"] / 10),
                        "Rotation": _scalar_to_json_primitive_or_none(rotation),
                    }
                    self._circuit_cache[cache_key] = result
                    return result

            logger.warning(
                f"Could not get corner data for Test {test_number} Session {session_number}"
            )
            return None
        except Exception as e:
            logger.error(
                f"Error getting circuit info for Test {test_number} Session {session_number}: {e}"
            )
            return None

    def _get_circuit_info_from_api(
        self, circuit_key: int
    ) -> Tuple[Optional[pd.DataFrame], float]:
        url = f"{PROTO}://{HOST}/api/v1/circuits/{circuit_key}/{self.year}"
        try:
            response = requests.get(url, headers=HEADERS)
            if response.status_code != 200:
                logger.debug(f"[{response.status_code}] {response.content.decode()}")
                return None, 0.0

            data = response.json()
            rotation = float(data.get("rotation", 0.0))

            rows = [
                (
                    float(e.get("trackPosition", {}).get("x", 0.0)),
                    float(e.get("trackPosition", {}).get("y", 0.0)),
                    int(e.get("number", 0)),
                    str(e.get("letter", "")),
                    float(e.get("angle", 0.0)),
                    float(e.get("length", 0.0)),
                )
                for e in data["corners"]
            ]

            return (
                pd.DataFrame(
                    rows, columns=["X", "Y", "Number", "Letter", "Angle", "Distance"]
                ),
                rotation,
            )
        except Exception as e:
            logger.error(f"Error fetching circuit data from API: {e}")
            return None, 0.0

    def _process_single_lap(
        self,
        driver: str,
        lap_number: int,
        driver_dir: str,
        driver_laps: pd.DataFrame,
        test_number: int,
        session_number: int,
    ) -> bool:
        file_path = f"{driver_dir}/{lap_number}_tel.json"
        try:
            selected = driver_laps[driver_laps.LapNumber == lap_number]
            if selected.empty:
                logger.warning(
                    f"No data for {driver} lap {lap_number} "
                    f"in Test {test_number} Session {session_number}"
                )
                return False

            telemetry = selected.get_telemetry()
            data_key = f"{self.year}-PreSeasonTesting{test_number}-Practice {session_number}-{driver}-{lap_number}"
            tel_data = _process_telemetry_to_dict(telemetry, data_key)
            _write_json(file_path, tel_data)
            return True
        except Exception as e:
            logger.error(f"Error processing lap {lap_number} for {driver}: {e}")
            return False

    def process_driver(
        self,
        test_number: int,
        session_number: int,
        driver: str,
        base_dir: str,
        f1session: fastf1.core.Session = None,
        session_weather_df: pd.DataFrame = None,
    ) -> None:
        driver_dir = f"{base_dir}/{driver}"
        os.makedirs(driver_dir, exist_ok=True)

        try:
            if f1session is None:
                f1session = self.get_session(
                    test_number, session_number, load_telemetry=True
                )

            driver_laps = f1session.laps.pick_drivers(driver)
            driver_laps = driver_laps.assign(
                LapNumber=driver_laps["LapNumber"].astype(int)
            )

            laptimes = self.laps_data(
                driver, f1session, driver_laps, session_weather_df
            )
            _write_json(f"{driver_dir}/laptimes.json", laptimes)

            lap_numbers = driver_laps["LapNumber"].tolist()

            existing = (
                set(os.listdir(driver_dir)) if os.path.isdir(driver_dir) else set()
            )

            for lap_number in lap_numbers:
                fname = f"{lap_number}_tel.json"
                if fname in existing:
                    continue
                self._process_single_lap(
                    driver,
                    lap_number,
                    driver_dir,
                    driver_laps,
                    test_number,
                    session_number,
                )

        except Exception as e:
            logger.error(f"Error processing driver {driver}: {e}")

    def process_testing_session(self, test_number: int, session_number: int) -> None:
        label = f"Test {test_number} - Practice {session_number}"
        logger.info(f"Processing {label}")

        # e.g. /Pre-Season Testing 1/Practice 1, /Pre-Season Testing 2/Practice 1
        base_dir = f"Pre-Season Testing {test_number}/Practice {session_number}"
        os.makedirs(base_dir, exist_ok=True)

        try:
            f1session = self.get_session(
                test_number, session_number, load_telemetry=True
            )

            weather_data = _session_weather_to_column_lists(f1session.weather_data)
            _write_json(
                f"{base_dir}/weather.json", weather_data, normalize_missing=True
            )

            race_control_messages = _session_rcm_to_column_lists(
                f1session.race_control_messages
            )
            _write_json(
                f"{base_dir}/rcm.json", race_control_messages, normalize_missing=True
            )

            drivers_info = self.session_drivers(test_number, session_number)
            _write_json(f"{base_dir}/drivers.json", drivers_info)

            corner_info = self.get_circuit_info(test_number, session_number)
            if corner_info:
                _write_json(f"{base_dir}/corners.json", corner_info)

            drivers = [d["driver"] for d in drivers_info.get("drivers", [])]

            if not drivers:
                logger.warning(f"No drivers found for {label}")
                return

            session_weather_df = None
            if hasattr(f1session.laps, "get_weather_data"):
                try:
                    session_weather_df = f1session.laps.get_weather_data()
                except Exception:
                    pass

            total_drivers = len(drivers)
            for i, driver in enumerate(drivers, 1):
                logger.info(f"Processing driver {driver} ({i}/{total_drivers})")
                self.process_driver(
                    test_number,
                    session_number,
                    driver,
                    base_dir,
                    f1session,
                    session_weather_df,
                )

        except Exception as e:
            logger.error(f"Error processing {label}: {e}")

    def process_all(self) -> None:
        logger.info(f"Starting pre-season testing extraction for {self.year}")
        start_time = time.time()

        events = self.discover_testing_events()
        if not events:
            logger.warning("No testing events found — nothing to extract.")
            return

        for evt in events:
            test_num = evt["test_number"]

            if TARGET_TEST_NUMBER is not None and test_num != TARGET_TEST_NUMBER:
                continue

            n_sessions = evt["sessions"]

            logger.info(
                f"Processing Test {test_num}: {evt['event_name']} "
                f"({n_sessions} sessions)"
            )

            for session_num in range(1, n_sessions + 1):
                if (
                    TARGET_SESSION_NUMBER is not None
                    and session_num != TARGET_SESSION_NUMBER
                ):
                    continue

                try:
                    self.process_testing_session(test_num, session_num)
                except Exception as e:
                    logger.error(f"Failed Test {test_num} Session {session_num}: {e}")
                check_memory_usage(
                    session_cache=self._session_cache,
                    circuit_cache=self._circuit_cache,
                )

        elapsed = time.time() - start_time
        logger.info(f"Pre-season testing extraction completed in {elapsed:.2f} seconds")


# ======================================================================
# Data Availability
# ======================================================================
def is_testing_data_available(year: int) -> bool:
    """Check if any pre-season testing data is available by scanning all sessions."""
    try:
        # Enforce 'fastf1' backend to strictly avoid 'Day 1' session issues
        # and get the full schedule to know what to check
        schedule = fastf1.get_event_schedule(
            year, include_testing=True, backend="fastf1"
        )
        testing = schedule[schedule["EventFormat"] == "testing"]

        if testing.empty:
            logger.info(f"No testing events found in {year} schedule yet.")
            return False

        # Iterate through all testing events and their sessions
        for idx, (_, row) in enumerate(testing.iterrows(), start=1):
            if TARGET_TEST_NUMBER is not None and idx != TARGET_TEST_NUMBER:
                continue

            # Check up to 5 sessions per event
            for s_num in range(1, 6):
                if TARGET_SESSION_NUMBER is not None and s_num != TARGET_SESSION_NUMBER:
                    continue

                col = f"Session{s_num}"
                if col not in row.index:
                    continue

                # Check if session exists in schedule
                val = row[col]
                if pd.isna(val) or str(val).strip() in ("", "None"):
                    continue

                try:
                    # Attempt to load this specific session
                    f1session = fastf1.get_testing_session(
                        year, idx, s_num, backend="fastf1"
                    )
                    # Minimal load to check for data
                    f1session.load(telemetry=False, weather=False, messages=False)

                    if (
                        not f1session.laps.empty
                        and len(f1session.laps["Driver"].unique()) > 0
                    ):
                        logger.info(
                            f"Data detected for Test {idx} Session {s_num}. "
                            "Extraction checks passed."
                        )
                        return True
                except Exception:
                    # If a specific session fails, continue checking others
                    continue

        logger.info(f"No data available yet for any {year} pre-season testing session")
        return False
    except Exception as e:
        logger.warning(f"Testing data check failed: {e}")
        return False


def main():
    try:
        year = DEFAULT_YEAR

        # Use separate cache to avoid pollution from main scripts
        os.makedirs("cache_preseason", exist_ok=True)
        fastf1.Cache.enable_cache("cache_preseason")

        extractor = PreSeasonExtractor(year=year)
        max_attempts = 720
        wait_time = 30
        attempt = 0

        logger.info(f"Starting to wait for {year} pre-season testing data...")

        while attempt < max_attempts:
            if is_testing_data_available(year):
                logger.info(
                    f"Data is available for {year} pre-season testing. "
                    "Starting extraction..."
                )
                extractor.process_all()
                break
            else:
                attempt += 1
                logger.info(
                    f"Data not yet available. Waiting {wait_time}s "
                    f"before retry ({attempt}/{max_attempts})..."
                )
                time.sleep(wait_time)
                gc.collect()

        if attempt >= max_attempts:
            logger.error("Exceeded maximum wait time. Exiting.")

    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise


if __name__ == "__main__":
    main()




