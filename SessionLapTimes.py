"""
Session Laptimes Merger
=======================
Combines per-driver laptimes.json files into a single session_laptimes.json
at the session level.

Input:
    {event_name}/{session_name}/{driver}/laptimes.json  (one per driver)

Output:
    {event_name}/{session_name}/session_laptimes.json

All driver arrays are concatenated in alphabetical driver order.
The existing `drv` field in each file identifies which rows belong to which driver.
If session_laptimes.json already exists it is overwritten.
"""

import logging
import os
import sys
from typing import Any, Dict, List, Optional

import orjson

# ---------------------------------------------------------------------------
# Configuration — mirror the style of LapTimes.py
# ---------------------------------------------------------------------------

DEFAULT_YEAR = 2026

# Root directory where event folders live.
# "." means the directory you invoke the script from.
# Override this if your data lives elsewhere, e.g. BASE_DIR = "/data/f1/2024"
BASE_DIR = "."

TARGET_EVENT_NAMES_LIST = [
    # "Australian Grand Prix",
    # "Chinese Grand Prix",
    # "Japanese Grand Prix",
    # "Bahrain Grand Prix",
    # "Saudi Arabian Grand Prix",
    # "Miami Grand Prix",
    # "Emilia Romagna Grand Prix",
    # "Monaco Grand Prix",
    "Barcelona Grand Prix",  
    # "Spanish Grand Prix",
    # "Canadian Grand Prix",
    
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
TARGET_EVENT_NAMES = [e.strip() for e in TARGET_EVENT_NAMES_LIST if e.strip()]
if not TARGET_EVENT_NAMES:
    raise ValueError("Set at least one active event in TARGET_EVENT_NAMES_LIST.")

AVAILABLE_SESSIONS = [
    "Practice 1",
    "Practice 2",
    "Practice 3",
    "Qualifying",
    "Sprint Qualifying",
    # "Sprint Shootout",
    "Sprint",
    "Race",
]

TARGET_SESSIONS = [
    "Practice 1",
    "Practice 2",
    "Practice 3",
    "Qualifying",
    "Sprint Qualifying",
    # "Sprint Shootout",
    "Sprint",
    "Race",
]

invalid_target_sessions = sorted(set(TARGET_SESSIONS) - set(AVAILABLE_SESSIONS))
if invalid_target_sessions:
    raise ValueError(
        "Invalid TARGET_SESSIONS value(s): " + ", ".join(invalid_target_sessions)
    )

ORJSON_OPTS = orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_NON_STR_KEYS

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("merge_laptimes.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("merge_laptimes")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return orjson.loads(f.read())


def _write_json(path: str, obj: Any) -> None:
    with open(path, "wb") as f:
        f.write(orjson.dumps(obj, option=ORJSON_OPTS))


def _driver_lap_count(data: Dict[str, List]) -> int:
    """Return the length of the first non-empty array in a driver's data dict."""
    for v in data.values():
        if isinstance(v, list) and len(v) > 0:
            return len(v)
    return 0


def _discover_drivers(session_dir: str) -> List[str]:
    """
    Return alphabetically sorted list of driver abbreviations that have a
    laptimes.json inside session_dir.
    """
    drivers = []
    try:
        entries = os.listdir(session_dir)
    except FileNotFoundError:
        return drivers

    for entry in sorted(entries):
        driver_dir = os.path.join(session_dir, entry)
        laptimes_path = os.path.join(driver_dir, "laptimes.json")
        if os.path.isdir(driver_dir) and os.path.isfile(laptimes_path):
            drivers.append(entry)

    return drivers  # already sorted because we sorted entries above


# ---------------------------------------------------------------------------
# Core merge logic
# ---------------------------------------------------------------------------


def merge_session(event_name: str, session_name: str) -> bool:
    """
    Merge all driver laptimes.json files for one session into
    session_laptimes.json.

    Returns True on success, False on failure (after logging the error).
    Raises SystemExit on unrecoverable errors (missing drivers, bad files).
    """
    label = f"{event_name} - {session_name}"
    session_dir = os.path.join(BASE_DIR, event_name, session_name)

    if not os.path.isdir(session_dir):
        logger.warning("Session directory not found, skipping: %s", session_dir)
        return True

    drivers = _discover_drivers(session_dir)
    if not drivers:
        logger.warning(
            "No driver directories with laptimes.json found in %s — skipping", session_dir
        )
        return True

    logger.info(
        "%s: found %d driver(s): %s", label, len(drivers), ", ".join(drivers)
    )

    # ------------------------------------------------------------------
    # Load every driver's file — fail the whole session on any problem.
    # ------------------------------------------------------------------
    driver_data: Dict[str, Dict[str, List]] = {}
    for driver in drivers:
        path = os.path.join(session_dir, driver, "laptimes.json")
        try:
            data = _read_json(path)
        except FileNotFoundError:
            logger.error("Missing laptimes.json for driver %s in %s", driver, label)
            return False
        except Exception as exc:
            logger.error(
                "Failed to read laptimes.json for driver %s in %s: %s",
                driver,
                label,
                exc,
            )
            return False

        if not isinstance(data, dict):
            logger.error(
                "laptimes.json for driver %s in %s is not a JSON object", driver, label
            )
            return False

        lap_count = _driver_lap_count(data)
        if lap_count == 0:
            logger.error(
                "laptimes.json for driver %s in %s contains no lap rows", driver, label
            )
            return False

        driver_data[driver] = data
        logger.info("  Loaded %s: %d lap(s)", driver, lap_count)

    # ------------------------------------------------------------------
    # Build the union of all keys across every driver.
    # ------------------------------------------------------------------
    all_keys: List[str] = []
    seen: set = set()
    # Preserve a stable key order: use the first driver's key order as the
    # base, then append any extra keys seen in subsequent drivers.
    for driver in drivers:
        for key in driver_data[driver]:
            if key not in seen:
                all_keys.append(key)
                seen.add(key)

    # ------------------------------------------------------------------
    # Concatenate arrays in alphabetical driver order.
    # Keys missing for a driver are filled with "None" * that driver's lap count.
    # ------------------------------------------------------------------
    merged: Dict[str, List] = {key: [] for key in all_keys}

    for driver in drivers:
        data = driver_data[driver]
        lap_count = _driver_lap_count(data)
        for key in all_keys:
            if key in data:
                values = data[key]
                if not isinstance(values, list):
                    logger.error(
                        "Key '%s' for driver %s in %s is not an array",
                        key,
                        driver,
                        label,
                    )
                    return False
                merged[key].extend(values)
            else:
                # Key present in another driver's file but not this one.
                logger.warning(
                    "Key '%s' missing for driver %s in %s — filling with %d None(s)",
                    key,
                    driver,
                    label,
                    lap_count,
                )
                merged[key].extend(["None"] * lap_count)

    # ------------------------------------------------------------------
    # Sanity-check: every key must have the same total length.
    # ------------------------------------------------------------------
    lengths = {key: len(arr) for key, arr in merged.items()}
    unique_lengths = set(lengths.values())
    if len(unique_lengths) > 1:
        bad = {k: v for k, v in lengths.items() if v != max(unique_lengths)}
        logger.error(
            "Array length mismatch after merge in %s. Mismatched keys: %s",
            label,
            bad,
        )
        return False

    # ------------------------------------------------------------------
    # Write output.
    # ------------------------------------------------------------------
    out_path = os.path.join(session_dir, "session_laptimes.json")
    try:
        _write_json(out_path, merged)
    except Exception as exc:
        logger.error("Failed to write %s: %s", out_path, exc)
        return False

    total_laps = max(lengths.values()) if lengths else 0
    logger.info(
        "%s: wrote %s (%d drivers, %d total laps)",
        label,
        out_path,
        len(drivers),
        total_laps,
    )
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    overall_success = True

    for event_name in TARGET_EVENT_NAMES:
        for session_name in TARGET_SESSIONS:
            label = f"{event_name} - {session_name}"
            logger.info("=== Merging %s ===", label)
            success = merge_session(event_name, session_name)
            if not success:
                logger.error("Merge FAILED for %s", label)
                overall_success = False

    if not overall_success:
        logger.error("One or more sessions failed to merge.")
        sys.exit(1)

    logger.info("All sessions merged successfully.")


if __name__ == "__main__":
    main()
