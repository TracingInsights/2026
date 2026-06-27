"""
Season Session Laptimes Extraction Script
==========================================
Extracts laptimes.json per driver from non-testing F1 season sessions.

Output directory:
{event_name}/{session_name}/{driver}/laptimes.json

Data sources:
- FastF1: base lap data, weather, sector times, tyre info
- Ergast (Race only): overwrites LapTime with Ergast's official lap times
- OpenF1: adds mini-sector segment columns (ms1, ms2, ms3)
"""

import gc
import logging
import os
import time
from collections import deque
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple, Union

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
    # "Barcelona Grand Prix",
    # "Spanish Grand Prix",
    # "Canadian Grand Prix",
    "Austrian Grand Prix",
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
TARGET_EVENT_NAMES = [e.strip() for e in TARGET_EVENT_NAMES_LIST if e.strip()]
if not TARGET_EVENT_NAMES:
    raise ValueError("Set at least one active event in TARGET_EVENT_NAMES_LIST.")
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
    "Practice 2",
    "Practice 3",
    "Qualifying",
    # "Sprint Qualifying",
    # "Sprint",
    # "Race",
]
invalid_target_sessions = sorted(set(TARGET_SESSIONS) - set(AVAILABLE_SESSIONS))
if invalid_target_sessions:
    raise ValueError(
        "Invalid TARGET_SESSIONS value(s): " + ", ".join(invalid_target_sessions)
    )

ORJSON_OPTS = orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_NON_STR_KEYS

# ---------------------------------------------------------------------------
# External API constants
# ---------------------------------------------------------------------------

ERGAST_BASE_URL = "https://api.jolpi.ca/ergast/"
OPENF1_BASE_URL = "https://api.openf1.org/v1/"

ONE_SECOND = 1
ONE_HOUR = 3600
ONE_MINUTE = 60

ERGAST_MAX_CALLS_PER_SECOND = 4
ERGAST_MAX_CALLS_PER_HOUR = 500

OPENF1_MAX_CALLS_PER_SECOND = 3
OPENF1_MAX_CALLS_PER_MINUTE = 30

MINI_SECTOR_CODE_MAP = {
    2048: 0,  # Yellow
    2049: 1,  # Green
    2050: 2,  # Unknown
    2051: 3,  # Purple
    2052: 4,  # Unknown
    2064: 5,  # Pitlane
    2068: 6,  # Unknown
    0: 7,  # Not Available
}

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

_MISSING_TEXT_VALUES = frozenset(
    {
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
)
_MISSING_TEXT_LIST = list(_MISSING_TEXT_VALUES)


# ---------------------------------------------------------------------------
# JSON helpers
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


# ---------------------------------------------------------------------------
# Weather helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Qualifying helpers
# ---------------------------------------------------------------------------


def _qualifying_session_name(
    session_name: Optional[str],
) -> Optional[Tuple[str, str, str]]:
    if not session_name:
        return None
    normalized = session_name.strip().lower()
    if normalized == "qualifying":
        return ("Q1", "Q2", "Q3")
    if normalized in ("sprint qualifying", "sprint shootout"):
        return ("SQ1", "SQ2", "SQ3")
    return None


def _laps_to_quali_segment(
    driver: str,
    driver_laps: pd.DataFrame,
    f1session: fastf1.core.Session,
    session_name: Optional[str],
) -> list:
    if driver_laps.empty:
        return []

    quali_segments = _qualifying_session_name(session_name)
    if quali_segments is None:
        return ["None"] * len(driver_laps)

    try:
        split_laps = f1session.laps.split_qualifying_sessions()
    except Exception as exc:
        logger.warning(
            "Could not split qualifying sessions for %s in %s: %s",
            driver,
            session_name,
            exc,
        )
        return ["None"] * len(driver_laps)

    lap_to_segment = {}
    for session_laps, segment_name in zip(split_laps, quali_segments):
        if session_laps is None or session_laps.empty:
            continue
        session_driver_laps = session_laps.pick_drivers(driver)
        if session_driver_laps.empty:
            continue
        for lap_num in session_driver_laps["LapNumber"].tolist():
            lap_to_segment[lap_num] = segment_name

    return [
        lap_to_segment.get(lap_num, "None")
        for lap_num in driver_laps["LapNumber"].tolist()
    ]


# ---------------------------------------------------------------------------
# Mini-sector helpers
# ---------------------------------------------------------------------------


def _none_mini_sector_columns(length: int) -> Dict[str, List[str]]:
    none_list = ["None"] * length
    return {key: none_list.copy() for key in ("ms1", "ms2", "ms3")}


def _coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _encode_mini_sector(segments: Any) -> str:
    if not isinstance(segments, list) or not segments:
        return "None"
    fallback_code = MINI_SECTOR_CODE_MAP[0]
    encoded = []
    for segment in segments:
        segment_value = _coerce_int(segment)
        if segment_value is None:
            encoded.append(str(fallback_code))
            continue
        encoded.append(str(MINI_SECTOR_CODE_MAP.get(segment_value, fallback_code)))
    return "".join(encoded)


def _mini_sector_columns_from_laps(
    driver_laps: pd.DataFrame, lap_segments: Dict[int, Dict[str, str]]
) -> Dict[str, List[str]]:
    lap_count = len(driver_laps)
    if lap_count == 0:
        return _none_mini_sector_columns(0)

    if "LapNumber" not in driver_laps.columns:
        return _none_mini_sector_columns(lap_count)

    columns: Dict[str, list] = {key: [] for key in ("ms1", "ms2", "ms3")}
    for lap_number in driver_laps["LapNumber"].tolist():
        lap_key = _coerce_int(lap_number)
        lap_data = lap_segments.get(lap_key, {})
        for column in columns:
            columns[column].append(lap_data.get(column, "None"))

    return columns


# ---------------------------------------------------------------------------
# Memory utilities
# ---------------------------------------------------------------------------


def check_memory_usage(threshold_percent=80, session_cache=None):
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
        gc.collect()

        new_pct = psutil.Process(os.getpid()).memory_percent()
        logger.info(f"New memory usage after clearing caches: {new_pct:.2f}%")
        return True

    return False


# ---------------------------------------------------------------------------
# Ergast client
# ---------------------------------------------------------------------------


class ErgastClient:
    """Fetches official lap times from the Jolpica/Ergast API with retries."""

    def __init__(self, retries: int = 3, backoff_factor: float = 0.3):
        self.session = requests.Session()
        self.cache: Dict[Any, Any] = {}
        self.retries = retries
        self.backoff_factor = backoff_factor
        # Simple token-bucket state for rate limiting (no extra dependencies).
        self._call_times: deque = deque()
        self._lock = Lock()

    def _throttle(self) -> None:
        """Block until both per-second and per-hour limits allow a request."""
        with self._lock:
            while True:
                now = time.monotonic()
                # Drop timestamps older than one hour.
                while self._call_times and now - self._call_times[0] >= ONE_HOUR:
                    self._call_times.popleft()

                recent_second = [t for t in self._call_times if now - t < ONE_SECOND]

                sleep_for = 0.0
                if len(self._call_times) >= ERGAST_MAX_CALLS_PER_HOUR:
                    sleep_for = max(sleep_for, ONE_HOUR - (now - self._call_times[0]))
                if len(recent_second) >= ERGAST_MAX_CALLS_PER_SECOND:
                    sleep_for = max(sleep_for, ONE_SECOND - (now - recent_second[0]))

                if sleep_for <= 0:
                    self._call_times.append(now)
                    return
                time.sleep(sleep_for)

    def _get(self, url: str, params: Optional[Dict[str, int]] = None):
        cache_key = (url, tuple(sorted(params.items())) if params else None)
        if cache_key in self.cache:
            return self.cache[cache_key]

        for attempt in range(self.retries):
            try:
                self._throttle()
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                json_response = response.json()
                self.cache[cache_key] = json_response
                return json_response
            except requests.exceptions.RequestException as exc:
                logger.error("Ergast request failed %s: %s", url, exc)
                if attempt < self.retries - 1:
                    time.sleep(self.backoff_factor * (2**attempt))
                else:
                    return None

    def get_lap_times(self, season: int, round_number: int) -> pd.DataFrame:
        """Fetch all lap times for a race, handling Ergast pagination."""
        all_laps_data = []
        offset = 0
        limit = 100

        while True:
            url = f"{ERGAST_BASE_URL}f1/{season}/{round_number}/laps.json"
            params = {"limit": limit, "offset": offset}
            response = self._get(url, params=params)

            if not response:
                break

            mr_data = response.get("MRData", {})
            race_table = mr_data.get("RaceTable", {})
            races = race_table.get("Races", [])
            if not races:
                break

            for lap in races[0].get("Laps", []):
                lap_number = _coerce_int(lap.get("number"))
                if lap_number is None:
                    continue
                for timing in lap.get("Timings", []):
                    position = _coerce_int(timing.get("position"))
                    all_laps_data.append(
                        {
                            "LapNumber": lap_number,
                            "driverId": timing.get("driverId"),
                            "position": position,
                            "time": timing.get("time"),
                        }
                    )

            total_results = int(mr_data.get("total", 0))
            if offset + limit >= total_results:
                break
            offset += limit

        return pd.DataFrame(all_laps_data)


# ---------------------------------------------------------------------------
# OpenF1 client
# ---------------------------------------------------------------------------


def _normalize_timestamp(value: Any) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts


class OpenF1Client:
    """Fetches session keys and mini-sector data from OpenF1."""

    def __init__(self, retries: int = 3, backoff_factor: float = 0.3):
        self.session = requests.Session()
        self.cache: Dict[Any, Any] = {}
        self.retries = retries
        self.backoff_factor = backoff_factor
        self._request_lock = Lock()
        self._request_times: deque = deque()

    def _throttle_locked(self) -> None:
        """Must be called while holding self._request_lock."""
        while True:
            now = time.monotonic()
            while self._request_times and now - self._request_times[0] >= ONE_MINUTE:
                self._request_times.popleft()

            recent_second = [t for t in self._request_times if now - t < ONE_SECOND]

            sleep_for = 0.0
            if len(self._request_times) >= OPENF1_MAX_CALLS_PER_MINUTE:
                sleep_for = max(sleep_for, ONE_MINUTE - (now - self._request_times[0]))
            if len(recent_second) >= OPENF1_MAX_CALLS_PER_SECOND:
                sleep_for = max(sleep_for, ONE_SECOND - (now - recent_second[0]))

            if sleep_for <= 0:
                self._request_times.append(now)
                return
            time.sleep(sleep_for)

    def _get(
        self, path: str, params: Optional[Dict[str, Union[int, str]]] = None
    ) -> Optional[List[Dict[str, Any]]]:
        url = f"{OPENF1_BASE_URL}{path}"
        cache_key = (url, tuple(sorted(params.items())) if params else None)
        if cache_key in self.cache:
            return self.cache[cache_key]

        for attempt in range(self.retries):
            try:
                with self._request_lock:
                    if cache_key in self.cache:
                        return self.cache[cache_key]
                    self._throttle_locked()
                    response = self.session.get(url, params=params, timeout=30)

                if response.status_code == 404:
                    self.cache[cache_key] = []
                    return []

                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After")
                    wait = (
                        float(retry_after)
                        if retry_after
                        else self.backoff_factor * (2**attempt)
                    )
                    time.sleep(wait)
                    continue

                response.raise_for_status()
                json_response = response.json()
                self.cache[cache_key] = json_response
                return json_response
            except requests.exceptions.RequestException as exc:
                logger.error("OpenF1 request failed %s: %s", url, exc)
                if attempt < self.retries - 1:
                    time.sleep(self.backoff_factor * (2**attempt))
                else:
                    return None

    def get_sessions(
        self,
        year: int,
        session_name: str,
    ) -> List[Dict[str, Any]]:
        params: Dict[str, Union[int, str]] = {
            "year": year,
            "session_name": session_name,
        }
        response = self._get("sessions", params=params)
        return response if isinstance(response, list) else []

    def get_driver_laps(
        self, session_key: int, driver_number: int
    ) -> List[Dict[str, Any]]:
        response = self._get(
            "laps",
            params={"session_key": session_key, "driver_number": driver_number},
        )
        return response if isinstance(response, list) else []


# ---------------------------------------------------------------------------
# Season Session Extractor
# ---------------------------------------------------------------------------


class SeasonSessionExtractor:
    """Extract laptimes from non-testing season sessions."""

    def __init__(self, year: int = DEFAULT_YEAR):
        self.year = year
        self._session_cache: Dict[str, fastf1.core.Session] = {}
        self._ergast_cache: Dict[str, pd.DataFrame] = {}
        self._openf1_session_key_cache: Dict[str, Optional[int]] = {}
        self._openf1_lap_cache: Dict[Tuple[int, int], Dict[int, Dict[str, str]]] = {}
        self.ergast_client = ErgastClient()
        self.openf1_client = OpenF1Client()

    # ------------------------------------------------------------------
    # FastF1 session loading
    # ------------------------------------------------------------------

    def get_session(self, event_name: str, session_name: str) -> fastf1.core.Session:
        cache_key = f"{self.year}-{event_name}-{session_name}"
        cached = self._session_cache.get(cache_key)
        if cached is not None:
            return cached
        f1session = fastf1.get_session(self.year, event_name, session_name)
        f1session.load(telemetry=True, weather=True, messages=True)
        self._session_cache[cache_key] = f1session
        return f1session

    # ------------------------------------------------------------------
    # Ergast
    # ------------------------------------------------------------------

    def _get_ergast_lap_map(
        self,
        event_name: str,
        f1session: fastf1.core.Session,
        driver: str,
    ) -> Dict[int, Any]:
        """
        Returns {lap_number: {"LapTime": timedelta, "position": int}} for a driver from Ergast.
        Result is empty dict if not a Race session or data unavailable.
        """
        cache_key = f"{self.year}-{event_name}-Race"
        if cache_key not in self._ergast_cache:
            try:
                round_number = f1session.event["RoundNumber"]
                all_laps_df = self.ergast_client.get_lap_times(
                    season=self.year, round_number=round_number
                )
                if not all_laps_df.empty:
                    all_laps_df["time"] = all_laps_df["time"].astype(str)
                    all_laps_df["LapTime_Ergast"] = pd.to_timedelta(
                        "00:" + all_laps_df["time"]
                    )
                self._ergast_cache[cache_key] = all_laps_df
            except Exception as exc:
                logger.warning("Ergast fetch failed for %s: %s", event_name, exc)
                self._ergast_cache[cache_key] = pd.DataFrame()

        all_laps_df = self._ergast_cache[cache_key]
        if all_laps_df.empty:
            return {}

        try:
            driver_id = f1session.get_driver(driver)["DriverId"]
        except Exception:
            logger.warning("Could not resolve DriverId for %s", driver)
            return {}

        driver_rows = all_laps_df[all_laps_df["driverId"] == driver_id]
        return {
            int(row["LapNumber"]): {
                "LapTime": row["LapTime_Ergast"],
                "position": row.get("position"),
            }
            for _, row in driver_rows.iterrows()
        }

    # ------------------------------------------------------------------
    # OpenF1
    # ------------------------------------------------------------------

    def _get_openf1_reference_time(
        self,
        f1session: fastf1.core.Session,
        driver_laps: Optional[pd.DataFrame] = None,
    ) -> Optional[pd.Timestamp]:
        for laps in (driver_laps, getattr(f1session, "laps", None)):
            if laps is None or getattr(laps, "empty", True):
                continue
            if "LapStartDate" not in laps.columns:
                continue
            lap_start_dates = laps["LapStartDate"].dropna()
            if lap_start_dates.empty:
                continue
            ref = _normalize_timestamp(lap_start_dates.min())
            if ref is not None:
                return ref

        event_info = getattr(f1session, "event", None)
        if event_info is None:
            return None
        for key in (
            "EventDate",
            "Session5DateUtc",
            "Session4DateUtc",
            "Session3DateUtc",
        ):
            if key not in event_info:
                continue
            ref = _normalize_timestamp(event_info[key])
            if ref is not None:
                return ref
        return None

    def _resolve_openf1_session_key(
        self,
        event_name: str,
        session_name: str,
        f1session: fastf1.core.Session,
        driver_laps: Optional[pd.DataFrame] = None,
    ) -> Optional[int]:
        """
        Resolves and caches the OpenF1 session key for an event/session.
        Call once per event/session before processing any drivers.
        """
        cache_key = f"{self.year}-{event_name}-{session_name}"
        if cache_key in self._openf1_session_key_cache:
            return self._openf1_session_key_cache[cache_key]

        openf1_session_name = getattr(f1session, "name", None) or session_name
        candidates = self.openf1_client.get_sessions(self.year, openf1_session_name)

        if not candidates:
            logger.info("No OpenF1 session found for %s %s", event_name, session_name)
            self._openf1_session_key_cache[cache_key] = None
            return None

        reference_time = self._get_openf1_reference_time(f1session, driver_laps)
        if reference_time is not None:

            def _distance(c: Dict[str, Any]) -> float:
                start = _normalize_timestamp(c.get("date_start"))
                if start is None:
                    return float("inf")
                return abs((start - reference_time).total_seconds())

            candidates = sorted(candidates, key=_distance)

        session_key = _coerce_int(candidates[0].get("session_key"))
        self._openf1_session_key_cache[cache_key] = session_key
        logger.info(
            "Resolved OpenF1 session key %s for %s %s",
            session_key,
            event_name,
            session_name,
        )
        return session_key

    def _get_driver_mini_sector_map(
        self,
        session_key: Optional[int],
        driver: str,
        driver_laps: pd.DataFrame,
        f1session: fastf1.core.Session,
    ) -> Dict[int, Dict[str, str]]:
        """Returns {lap_number: {ms1, ms2, ms3}} for a driver from OpenF1."""
        if session_key is None:
            return {}

        driver_number: Optional[int] = None
        if "DriverNumber" in driver_laps.columns:
            nums = driver_laps["DriverNumber"].dropna()
            if not nums.empty:
                driver_number = _coerce_int(nums.iloc[0])
        if driver_number is None:
            try:
                driver_number = _coerce_int(
                    f1session.get_driver(driver).get("DriverNumber")
                )
            except Exception:
                pass

        if driver_number is None:
            logger.info("Skipping OpenF1 mini-sectors for %s: no driver number", driver)
            return {}

        lap_cache_key = (session_key, driver_number)
        if lap_cache_key in self._openf1_lap_cache:
            return self._openf1_lap_cache[lap_cache_key]

        raw_laps = self.openf1_client.get_driver_laps(session_key, driver_number)
        lap_segments: Dict[int, Dict[str, str]] = {}
        for lap_data in raw_laps:
            lap_number = _coerce_int(lap_data.get("lap_number"))
            if lap_number is None:
                continue
            lap_segments[lap_number] = {
                "ms1": _encode_mini_sector(lap_data.get("segments_sector_1")),
                "ms2": _encode_mini_sector(lap_data.get("segments_sector_2")),
                "ms3": _encode_mini_sector(lap_data.get("segments_sector_3")),
            }

        self._openf1_lap_cache[lap_cache_key] = lap_segments
        return lap_segments

    # ------------------------------------------------------------------
    # Core lap data assembly
    # ------------------------------------------------------------------

    def laps_data(
        self,
        driver: str,
        f1session: fastf1.core.Session,
        driver_laps: pd.DataFrame,
        session_weather_df: pd.DataFrame = None,
        session_name: Optional[str] = None,
        openf1_session_key: Optional[int] = None,
    ) -> Dict[str, list]:
        try:
            session_name = getattr(f1session, "name", None) or session_name

            # Overwrite LapTime with Ergast data for Race sessions.
            if session_name == "Race":
                ergast_map = self._get_ergast_lap_map(
                    f1session.event.get("EventName", ""), f1session, driver
                )
                if ergast_map:
                    driver_laps = driver_laps.copy()
                    ergast_lt_map = {k: v["LapTime"] for k, v in ergast_map.items()}
                    ergast_pos_map = {
                        k: v["position"]
                        for k, v in ergast_map.items()
                        if v.get("position") is not None
                    }
                    driver_laps["LapTime"] = (
                        driver_laps["LapNumber"]
                        .map(ergast_lt_map)
                        .fillna(driver_laps["LapTime"])
                    )
                    if ergast_pos_map:
                        driver_laps["Position"] = (
                            driver_laps["LapNumber"]
                            .map(ergast_pos_map)
                            .fillna(driver_laps["Position"])
                        )

                    # Insert rows for laps Ergast has but FastF1 dropped
                    # (e.g. lap 1, laps above 2:30).
                    existing_laps = set(driver_laps["LapNumber"].dropna().astype(int))
                    missing_laps = sorted(set(ergast_map) - existing_laps)
                    if missing_laps:
                        drv_abbr = driver
                        drv_num = None
                        try:
                            drv_info = f1session.get_driver(driver)
                            drv_abbr = drv_info.get("Abbreviation", driver)
                            drv_num = str(drv_info.get("DriverNumber", ""))
                        except Exception:
                            pass
                        stub_rows = []
                        for lap_num in missing_laps:
                            ergast_entry = ergast_map[lap_num]
                            stub = {col: np.nan for col in driver_laps.columns}
                            stub["LapNumber"] = lap_num
                            stub["LapTime"] = ergast_entry["LapTime"]
                            pos = ergast_entry.get("position")
                            if pos is not None:
                                stub["Position"] = pos
                            stub["Driver"] = drv_abbr
                            if drv_num is not None:
                                stub["DriverNumber"] = drv_num
                            stub_rows.append(stub)
                        stub_df = pd.DataFrame(stub_rows, columns=driver_laps.columns)
                        driver_laps = (
                            pd.concat([driver_laps, stub_df], ignore_index=True)
                            .sort_values("LapNumber")
                            .reset_index(drop=True)
                        )

            lap_weather = _lap_weather_to_column_lists(driver_laps, session_weather_df)
            mini_sector_columns = _mini_sector_columns_from_laps(
                driver_laps,
                self._get_driver_mini_sector_map(
                    openf1_session_key, driver, driver_laps, f1session
                ),
            )

            lap_data = {
                "time": _td_col_to_seconds(driver_laps["LapTime"]),
                "lap": _col_to_list_int_or_none(driver_laps["LapNumber"]),
                "compound": _col_to_list_str_or_none(driver_laps["Compound"]),
                "stint": _col_to_list_int_or_none(driver_laps["Stint"]),
                "s1": _td_col_to_seconds(driver_laps["Sector1Time"]),
                "s2": _td_col_to_seconds(driver_laps["Sector2Time"]),
                "s3": _td_col_to_seconds(driver_laps["Sector3Time"]),
                **mini_sector_columns,
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
            if _qualifying_session_name(session_name) is not None:
                lap_data["qs"] = _laps_to_quali_segment(
                    driver, driver_laps, f1session, session_name
                )

            return lap_data

        except Exception as e:
            logger.error("Error getting lap data for %s: %s", driver, e)
            empty_keys = (
                "time",
                "lap",
                "compound",
                "stint",
                "s1",
                "s2",
                "s3",
                "ms1",
                "ms2",
                "ms3",
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
            empty_lap_data = {k: [] for k in empty_keys}
            if _qualifying_session_name(session_name) is not None:
                empty_lap_data["qs"] = []
            return empty_lap_data

    # ------------------------------------------------------------------
    # Session processing
    # ------------------------------------------------------------------

    def process_event_session(self, event_name: str, session_name: str) -> None:
        label = f"{event_name} - {session_name}"
        logger.info("Processing %s", label)

        base_dir = f"{event_name}/{session_name}"
        os.makedirs(base_dir, exist_ok=True)

        try:
            f1session = self.get_session(event_name, session_name)

            laps = f1session.laps
            if laps.empty or "Driver" not in laps.columns:
                logger.warning("No lap data for %s", label)
                return

            drivers = laps["Driver"].dropna().unique().tolist()
            if not drivers:
                logger.warning("No drivers found for %s", label)
                return

            session_weather_df = None
            if hasattr(laps, "get_weather_data"):
                try:
                    session_weather_df = laps.get_weather_data()
                except Exception:
                    pass

            # Resolve OpenF1 session key once, shared across all drivers.
            openf1_session_key = self._resolve_openf1_session_key(
                event_name, session_name, f1session, laps
            )

            total_drivers = len(drivers)
            for i, driver in enumerate(drivers, 1):
                logger.info("Processing driver %s (%d/%d)", driver, i, total_drivers)
                driver_dir = f"{base_dir}/{driver}"
                os.makedirs(driver_dir, exist_ok=True)

                driver_laps = laps.pick_drivers(driver)
                driver_laps = driver_laps.assign(
                    LapNumber=driver_laps["LapNumber"].astype(int)
                )

                laptimes = self.laps_data(
                    driver,
                    f1session,
                    driver_laps,
                    session_weather_df,
                    session_name,
                    openf1_session_key=openf1_session_key,
                )
                _write_json(f"{driver_dir}/laptimes.json", laptimes)

        except Exception as e:
            logger.error("Error processing %s: %s", label, e)

    def process_all(self) -> None:
        logger.info("Starting laptimes extraction for %d", self.year)
        start_time = time.time()

        if not TARGET_EVENT_NAMES:
            logger.warning("No TARGET_EVENT_NAMES configured — nothing to extract.")
            return

        sessions = [s for s in TARGET_SESSIONS if isinstance(s, str) and s.strip()]
        if not sessions:
            logger.warning("No TARGET_SESSIONS configured — nothing to extract.")
            return

        for event_name in TARGET_EVENT_NAMES:
            logger.info("Processing %s (%s)", event_name, ", ".join(sessions))
            for session_name in sessions:
                try:
                    self.process_event_session(event_name, session_name)
                except Exception as e:
                    logger.error("Failed %s %s: %s", event_name, session_name, e)
                check_memory_usage(session_cache=self._session_cache)

        elapsed = time.time() - start_time
        logger.info("Laptimes extraction completed in %.2f seconds", elapsed)


# ---------------------------------------------------------------------------
# Data Availability check
# ---------------------------------------------------------------------------


def is_session_data_available(
    year: int,
    events: Optional[List[str]] = None,
    sessions: Optional[List[str]] = None,
) -> bool:
    """Check if data is available for the first specified event/session pair."""
    try:
        if events is None:
            events = list(TARGET_EVENT_NAMES)
        if sessions is None:
            sessions = list(TARGET_SESSIONS)

        if not events or not sessions:
            logger.warning("No events or sessions specified to check")
            return False

        event = events[0]
        session = sessions[0]

        logger.info("Checking data availability for %d %s %s...", year, event, session)

        with fastf1.Cache.disabled():
            f1session = fastf1.get_session(year, event, session)
            f1session.load(telemetry=False, weather=False, messages=False)

        if f1session.laps.empty:
            logger.info("No lap data available yet for %d %s %s", year, event, session)
            return False

        if "Driver" not in f1session.laps.columns:
            logger.info(
                "No driver data available yet for %d %s %s", year, event, session
            )
            return False

        if len(f1session.laps["Driver"].dropna().unique()) == 0:
            logger.info(
                "No driver data available yet for %d %s %s", year, event, session
            )
            return False

        logger.info("Data is available for %d %s %s", year, event, session)
        return True

    except Exception as e:
        logger.info("Data not yet available: %s", e)
        return False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    try:
        year = DEFAULT_YEAR

        os.makedirs("cache", exist_ok=True)
        fastf1.Cache.enable_cache("cache")
        logger.info("FastF1 cache enabled at cache")

        extractor = SeasonSessionExtractor(year=year)
        max_attempts = 720
        wait_time = 30
        attempt = 0

        logger.info("Starting to wait for %d season session data...", year)

        while attempt < max_attempts:
            if is_session_data_available(year):
                logger.info(
                    "Data is available for %d season sessions. Starting extraction...",
                    year,
                )
                extractor.process_all()
                break
            else:
                attempt += 1
                logger.info(
                    "Data not yet available. Waiting %ds before retry (%d/%d)...",
                    wait_time,
                    attempt,
                    max_attempts,
                )
                time.sleep(wait_time)
                gc.collect()

        if attempt >= max_attempts:
            logger.error("Exceeded maximum wait time. Exiting.")

    except Exception as e:
        logger.error("Error in main function: %s", e)
        raise


if __name__ == "__main__":
    main()
