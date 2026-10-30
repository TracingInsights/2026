"""Helpers shared by the session extraction scripts."""

import datetime
import logging
import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastf1.mvapi.api import get_circuit

logger = logging.getLogger(__name__)


def _float_or_none(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _int_or_none(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _mv_distance(entry: Dict) -> Optional[float]:
    """MultiViewer's own marker distance from the start/finish line, in meters
    (the API's ``length`` is in decimeters)."""
    length = _float_or_none(entry.get("length"))
    return None if length is None else length / 10


def get_mv_circuit_payload(f1session) -> Optional[Dict]:
    """Raw MultiViewer circuit payload for the session's circuit.

    This is the same request ``Session.get_circuit_info()`` makes, so it is
    served from FastF1's HTTP cache. Returns None if the request fails.
    """
    try:
        circuit_key = f1session.session_info["Meeting"]["Circuit"]["Key"]
        return get_circuit(year=f1session.event.year, circuit_key=circuit_key)
    except Exception as e:
        logger.warning(f"Could not fetch MultiViewer circuit data: {e}")
        return None


def _marker_columns(
    prefix: str, entries: List[Dict], telemetry_distances: Optional[List]
) -> Dict[str, list]:
    mv_distances = [_mv_distance(e) for e in entries]
    return {
        f"{prefix}Number": [_int_or_none(e.get("number")) for e in entries],
        f"{prefix}X": [
            _float_or_none((e.get("trackPosition") or {}).get("x")) for e in entries
        ],
        f"{prefix}Y": [
            _float_or_none((e.get("trackPosition") or {}).get("y")) for e in entries
        ],
        f"{prefix}Angle": [_float_or_none(e.get("angle")) for e in entries],
        f"{prefix}Distance": (
            telemetry_distances if telemetry_distances is not None else mv_distances
        ),
        f"{prefix}MVDistance": mv_distances,
    }


def build_circuit_details(payload: Optional[Dict], circuit_info=None) -> Dict:
    """Extra corners.json fields from the raw MultiViewer circuit payload.

    Per-corner columns are aligned with ``CornerNumber`` (FastF1 builds its
    corner rows from the same payload, in the same order). ``circuit_info`` is
    the FastF1 CircuitInfo whose marshal ``Distance`` values were computed from
    the reference lap's telemetry; without it (API fallback path) marshal
    distances fall back to MultiViewer's own distance, as corners do there.
    """
    payload = payload or {}
    corners = payload.get("corners") or []
    letters = [str(c.get("letter") or "") for c in corners]
    pit_loss = payload.get("pitLoss") or {}
    lap = payload.get("candidateLap") or {}

    def telemetry_distances(df) -> Optional[List]:
        if df is None:
            return None
        return [_float_or_none(v) for v in df["Distance"]]

    lights_df = circuit_info.marshal_lights if circuit_info is not None else None
    sectors_df = circuit_info.marshal_sectors if circuit_info is not None else None

    return {
        "CornerLetter": letters,
        "CornerLabel": [
            f"{c.get('number', '')}{letter}" for c, letter in zip(corners, letters)
        ],
        "MVDistance": [_mv_distance(c) for c in corners],
        **_marker_columns(
            "MarshalLight",
            payload.get("marshalLights") or [],
            telemetry_distances(lights_df),
        ),
        **_marker_columns(
            "MarshalSector",
            payload.get("marshalSectors") or [],
            telemetry_distances(sectors_df),
        ),
        "CircuitKey": _int_or_none(payload.get("circuitKey")),
        "CircuitName": payload.get("circuitName"),
        "Location": payload.get("location"),
        "CountryName": payload.get("countryName"),
        "CountryKey": _int_or_none(payload.get("countryKey")),
        "CountryIocCode": payload.get("countryIocCode"),
        "PitLossNormal": _float_or_none(pit_loss.get("normal")),
        "PitLossSC": _float_or_none(pit_loss.get("sc")),
        "PitLossVSC": _float_or_none(pit_loss.get("vsc")),
        "LayoutYear": _int_or_none(payload.get("year")),
        "LayoutRound": _int_or_none(payload.get("round")),
        "LayoutRaceDate": payload.get("raceDate"),
        "LayoutMeetingKey": _int_or_none(payload.get("meetingKey")),
        "LayoutMeetingName": payload.get("meetingName"),
        "LayoutMeetingOfficialName": payload.get("meetingOfficialName"),
        "LayoutLapDriverNumber": lap.get("driverNumber"),
        "LayoutLapNumber": _int_or_none(lap.get("lapNumber")),
        "LayoutLapSession": lap.get("session"),
        "LayoutLapTime": _float_or_none(lap.get("lapTime")),
        "LayoutLapStartDate": lap.get("lapStartDate"),
        "LayoutLapStartSessionTime": _float_or_none(lap.get("lapStartSessionTime")),
        "LayoutLapSessionStartTime": _float_or_none(lap.get("sessionStartTime")),
        "TrackX": list(payload.get("x") or []),
        "TrackY": list(payload.get("y") or []),
        "TrackTime": list(payload.get("trackPositionTime") or []),
        "MiniSectorIndex": list(payload.get("miniSectorsIndexes") or []),
    }


def _seconds_or_none(value: Any) -> Optional[float]:
    """A timedelta as seconds rounded to the millisecond (None for NaT/None)."""
    if value is None or pd.isna(value):
        return None
    return round(pd.Timedelta(value).total_seconds(), 3)


def _gmt_offset(value: datetime.timedelta) -> str:
    """A UTC offset in the live-timing API's own form: ``"-05:00:00"``."""
    total = int(value.total_seconds())
    sign = "-" if total < 0 else ""
    hours, rest = divmod(abs(total), 3600)
    minutes, seconds = divmod(rest, 60)
    return f"{sign}{hours:02d}:{minutes:02d}:{seconds:02d}"


def _info_value(value: Any) -> Any:
    """``Session.session_info`` values as JSON primitives (dates in ISO form)."""
    if isinstance(value, dict):
        return {k: _info_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_info_value(v) for v in value]
    if isinstance(value, datetime.timedelta):
        return _gmt_offset(value)
    if isinstance(value, datetime.datetime):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _status_columns(df: Optional[pd.DataFrame], columns: List[str]) -> Dict[str, list]:
    if df is None or df.empty:
        return {col: [] for col in columns}
    out: Dict[str, list] = {"Time": [_seconds_or_none(t) for t in df["Time"]]}
    for col in columns[1:]:
        out[col] = [None if pd.isna(v) else str(v) for v in df[col]]
    return out


def _session_attr(f1session, name: str) -> Any:
    """A loaded FastF1 session attribute, or None when it was not loaded."""
    try:
        return getattr(f1session, name)
    except Exception as e:
        logger.warning(f"session_info.json: no {name}: {e}")
        return None


def build_session_info(f1session) -> Dict:
    """session_info.json: the live-timing session state FastF1 reads besides laps.

    ``track_status`` and ``session_status`` are the TrackStatus and
    SessionStatus streams (times in seconds of session time),
    ``session_start_time`` the first ``Started`` status, ``t0_date`` the date
    at session time zero, ``total_laps`` the scheduled race distance (race
    and sprint only) and ``session_info`` the SessionInfo payload. None of
    these can be rebuilt exactly from rcm.json, the laps or the schedule.
    Fields FastF1 did not load are None; this never raises.
    """
    t0_date = _session_attr(f1session, "t0_date")
    total_laps = _session_attr(f1session, "total_laps")
    info = _session_attr(f1session, "session_info")
    return {
        "t0_date": None if t0_date is None or pd.isna(t0_date) else t0_date.isoformat(),
        "session_start_time": _seconds_or_none(
            _session_attr(f1session, "session_start_time")
        ),
        "total_laps": None if total_laps is None else int(total_laps),
        "track_status": _status_columns(
            _session_attr(f1session, "track_status"), ["Time", "Status", "Message"]
        ),
        "session_status": _status_columns(
            _session_attr(f1session, "session_status"), ["Time", "Status"]
        ),
        "session_info": _info_value(info) if info is not None else None,
    }
