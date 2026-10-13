"""Helpers shared by the session extraction scripts."""

import logging
import math
from typing import Any, Dict, List, Optional

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
