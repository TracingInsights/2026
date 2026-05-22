"""
Season Session Telemetry Extraction Script
==========================================
Extracts telemetry data from non-testing F1 season sessions.

Output directory:
{year}/{event_name}/{session_name}/

Standalone cache:
cache_session
"""

import gc
import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import fastf1
import numpy as np
import orjson
import pandas as pd
import psutil
import requests

# ---------------------------------------------------------------------------
# Constants & Configuration
# ---------------------------------------------------------------------------

DEFAULT_YEAR = 2026
# Keep exactly one uncommented event in this list.
TARGET_EVENT_NAMES_LIST = [
    
    # "Australian Grand Prix",
    # "Chinese Grand Prix",
    # "Japanese Grand Prix",
    # "Bahrain Grand Prix",
    # "Saudi Arabian Grand Prix",
    # "Miami Grand Prix",
    # "Emilia Romagna Grand Prix",
    # "Monaco Grand Prix",
    # "Spanish Grand Prix",
    "Canadian Grand Prix",
    # "Austrian Grand Prix",
    # "British Grand Prix",
    # "Belgian Grand Prix",
    # "Hungarian Grand Prix",
    # "Dutch Grand Prix",
    # "Italian Grand Prix",
    # "Azerbaijan Grand Prix",
    # "Singapore Grand Prix",
    # "United States Grand Prix",
    # "Mexico City Grand Prix",
    # "São Paulo Grand Prix",
    # "Las Vegas Grand Prix",
    # "Qatar Grand Prix",
    # "Abu Dhabi Grand Prix",
]
if len(TARGET_EVENT_NAMES_LIST) != 1:
    raise ValueError(
        "Set exactly one active event in TARGET_EVENT_NAME "
        "(comment all others)."
    )
TARGET_EVENT_NAME = TARGET_EVENT_NAMES_LIST[0]
AVAILABLE_SESSIONS = [
    "Practice 1",
    "Practice 2",
    "Practice 3",
    "Qualifying",
    "Sprint Qualifying",
    "Sprint",
    "Race",
]
# Select one or more sessions from AVAILABLE_SESSIONS.
TARGET_SESSIONS = [
    "Practice 1",
    # "Practice 2",
    # "Practice 3",
    # "Qualifying",
    # "Sprint Qualifying",
    # "Sprint",
    # "Race",
]
invalid_target_sessions = sorted(set(TARGET_SESSIONS) - set(AVAILABLE_SESSIONS))
if invalid_target_sessions:
    raise ValueError(
        "Invalid TARGET_SESSIONS value(s): "
        + ", ".join(invalid_target_sessions)
    )
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
        logging.FileHandler("session_extraction.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("session_extractor")
logging.getLogger("fastf1").setLevel(logging.WARNING)
logging.getLogger("fastf1").propagate = False

_MISSING_TEXT_VALUES = frozenset({
    "",
    "null",
    "nan",
    "nat",
    "none",
    "inf",
    "-inf",
    "infinity",
    "-infinity",
})
_MISSING_TEXT_LIST = list(_MISSING_TEXT_VALUES)


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
    valid = ~mask
    out = np.empty(vals.shape, dtype=object)
    out[mask] = "None"
    valid_vals = vals[valid]
    s_vals = np.array([str(v).strip().lower() for v in valid_vals])
    missing_mask = np.isin(s_vals, _MISSING_TEXT_LIST)
    str_vals = np.array([str(v) for v in valid_vals])
    out[valid] = np.where(missing_mask, "None", str_vals)
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
        missing_mask = np.isin(s_vals, _MISSING_TEXT_LIST)
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


def _lap_weather_to_column_lists(laps: pd.DataFrame, weather_df: pd.DataFrame = None) -> Dict[str, list]:
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
    valid = np.isfinite(arr)
    if valid.all():
        return arr.tolist()
    out = np.empty(arr.shape, dtype=object)
    out[~valid] = "None"
    out[valid] = arr[valid]
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
# Season Session Extractor
# ---------------------------------------------------------------------------
class SeasonSessionExtractor:
    """Extract telemetry from non-testing season sessions."""

    def __init__(self, year: int = DEFAULT_YEAR):
        self.year = year
        self._session_cache: Dict[str, fastf1.core.Session] = {}
        self._circuit_cache: Dict[str, dict] = {}

    def get_session(
        self, event_name: str, session_name: str, load_telemetry: bool = True
    ) -> fastf1.core.Session:
        cache_key = f"{self.year}-{event_name}-{session_name}"
        cached = self._session_cache.get(cache_key)

        if cached is not None:
            if load_telemetry and not getattr(cached, "_telemetry_loaded", False):
                cached.load(telemetry=True, weather=True, messages=True)
                cached._telemetry_loaded = True
                self._session_cache[cache_key] = cached
            return cached

        f1session = fastf1.get_session(self.year, event_name, session_name)
        f1session.load(telemetry=load_telemetry, weather=True, messages=True)
        f1session._telemetry_loaded = load_telemetry
        self._session_cache[cache_key] = f1session
        return f1session

    def session_drivers(
        self, event_name: str, session_name: str, f1session: fastf1.core.Session = None
    ) -> Dict[str, List[Dict[str, str]]]:
        try:
            if f1session is None:
                f1session = self.get_session(event_name, session_name)
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
                f"Error getting drivers for {event_name} {session_name}: {e}"
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
                    "time", "lap", "compound", "stint",
                    "s1", "s2", "s3", "life", "pos", "status", "pb",
                    "sesT", "drv", "dNum", "pout", "pin",
                    "s1T", "s2T", "s3T", "vi1", "vi2",
                    "vfl", "vst", "fresh", "team", "lST",
                    "lSD", "del", "delR", "ff1G", "iacc",
                    *LAP_WEATHER_KEYS,
                )
            }

    def get_circuit_info(
        self, event_name: str, session_name: str
    ) -> Optional[Dict]:
        cache_key = f"{self.year}-{event_name}-{session_name}"
        if cache_key in self._circuit_cache:
            return self._circuit_cache[cache_key]

        try:
            f1session = self.get_session(event_name, session_name)
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
                    "Rotation": _scalar_to_json_primitive_or_none(circuit_info.rotation),
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
                f"Could not get corner data for {event_name} {session_name}"
            )
            return None
        except Exception as e:
            logger.error(
                f"Error getting circuit info for {event_name} {session_name}: {e}"
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
        event_name: str,
        session_name: str,
    ) -> bool:
        file_path = f"{driver_dir}/{lap_number}_tel.json"
        try:
            selected = driver_laps[driver_laps.LapNumber == lap_number]
            if selected.empty:
                logger.warning(
                    f"No data for {driver} lap {lap_number} in {event_name} {session_name}"
                )
                return False

            telemetry = selected.get_telemetry()
            data_key = f"{self.year}-{event_name}-{session_name}-{driver}-{lap_number}"
            tel_data = _process_telemetry_to_dict(telemetry, data_key)
            _write_json(file_path, tel_data)
            return True
        except Exception as e:
            logger.error(f"Error processing lap {lap_number} for {driver}: {e}")
            return False

    def process_driver(
        self,
        event_name: str,
        session_name: str,
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
                    event_name, session_name, load_telemetry=True
                )

            driver_laps = f1session.laps.pick_drivers(driver)
            driver_laps = driver_laps.assign(
                LapNumber=driver_laps["LapNumber"].astype(int)
            )

            laptimes = self.laps_data(driver, f1session, driver_laps, session_weather_df)
            _write_json(f"{driver_dir}/laptimes.json", laptimes)

            lap_numbers = driver_laps["LapNumber"].tolist()

            existing = (
                set(os.listdir(driver_dir))
                if os.path.isdir(driver_dir)
                else set()
            )

            for lap_number in lap_numbers:
                fname = f"{lap_number}_tel.json"
                if fname in existing:
                    continue
                self._process_single_lap(
                    driver, lap_number, driver_dir, driver_laps, event_name, session_name
                )

        except Exception as e:
            logger.error(f"Error processing driver {driver}: {e}")

    def process_event_session(self, event_name: str, session_name: str) -> None:
        label = f"{event_name} - {session_name}"
        logger.info(f"Processing {label}")

        base_dir = f"{event_name}/{session_name}"
        os.makedirs(base_dir, exist_ok=True)

        try:
            f1session = self.get_session(event_name, session_name, load_telemetry=True)

            weather_data = _session_weather_to_column_lists(f1session.weather_data)
            _write_json(f"{base_dir}/weather.json", weather_data)

            session_control_messages = _session_rcm_to_column_lists(
                f1session.race_control_messages
            )
            _write_json(f"{base_dir}/rcm.json", session_control_messages)

            drivers_info = self.session_drivers(event_name, session_name, f1session)
            _write_json(f"{base_dir}/drivers.json", drivers_info)

            corner_info = self.get_circuit_info(event_name, session_name)
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
                    event_name,
                    session_name,
                    driver,
                    base_dir,
                    f1session,
                    session_weather_df,
                )
        except Exception as e:
            logger.error(f"Error processing {label}: {e}")

    def process_all(self) -> None:
        logger.info(f"Starting season session extraction for {self.year}")
        start_time = time.time()

        event_name = TARGET_EVENT_NAME.strip() if TARGET_EVENT_NAME else ""
        if not event_name:
            logger.warning("No TARGET_EVENT_NAME configured — nothing to extract.")
            return

        sessions = [s for s in TARGET_SESSIONS if isinstance(s, str) and s.strip()]
        if not sessions:
            logger.warning("No TARGET_SESSIONS configured — nothing to extract.")
            return

        logger.info(f"Processing {event_name} ({', '.join(sessions)})")
        for session_name in sessions:
            try:
                self.process_event_session(event_name, session_name)
            except Exception as e:
                logger.error(f"Failed {event_name} {session_name}: {e}")
            check_memory_usage(
                session_cache=self._session_cache,
                circuit_cache=self._circuit_cache,
            )

        elapsed = time.time() - start_time
        logger.info(f"Season session extraction completed in {elapsed:.2f} seconds")


# ======================================================================
# Data Availability
# ======================================================================




def is_session_data_available(
    year: int,
    events: Optional[List[str]] = None,
    sessions: Optional[List[str]] = None,
) -> bool:
    """Check if data is available for the first specified event/session pair."""
    try:
        if events is None:
            events = [TARGET_EVENT_NAME] if TARGET_EVENT_NAME else []
        if sessions is None:
            sessions = list(TARGET_SESSIONS)

        if not events or not sessions:
            logger.warning("No events or sessions specified to check")
            return False

        event = events[0]
        session = sessions[0]

        logger.info(f"Checking data availability for {year} {event} {session}...")

        f1session = fastf1.get_session(year, event, session)
        f1session.load(telemetry=False, weather=False, messages=False)

        if f1session.laps.empty:
            logger.info(f"No lap data available yet for {year} {event} {session}")
            return False

        if "Driver" not in f1session.laps.columns:
            logger.info(f"No driver data available yet for {year} {event} {session}")
            return False

        if len(f1session.laps["Driver"].dropna().unique()) == 0:
            logger.info(f"No driver data available yet for {year} {event} {session}")
            return False

        logger.info(f"Data is available for {year} {event} {session}")
        return True

    except Exception as e:
        logger.info(f"Data not yet available: {str(e)}")
        return False


def main():
    try:
        year = DEFAULT_YEAR

        os.makedirs("cache", exist_ok=True)
        fastf1.Cache.enable_cache("cache")

        extractor = SeasonSessionExtractor(year=year)
        max_attempts = 720
        wait_time = 30
        attempt = 0

        logger.info(f"Starting to wait for {year} season session data...")

        while attempt < max_attempts:
            if is_session_data_available(year):
                logger.info(
                    f"Data is available for {year} season sessions. "
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
