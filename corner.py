from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field, replace
from typing import Any

import requests

try:
    import fastf1
except ModuleNotFoundError:  # pragma: no cover - allows pure stdlib tests
    fastf1 = None

# Configuration
YEAR = 2026
EVENTS = ["Australian Grand Prix"]
SESSIONS = ["Race", "Qualifying", "Practice 3", "Practice 2", "Practice 1"]
SECTIONS_DIR = "sections"
PROTO = "https"
HOST = "api.multiviewer.app"
HEADERS = {"User-Agent": "FastF1/"}
REQUEST_TIMEOUT = 30

VALID_SECTION_TYPES = {"Straight", "Low", "Medium", "High"}

# Circuit section breakdowns live in sections/<circuit>.json, named after
# the circuit (e.g. sections/marina-bay-street-circuit.json), not the event.
# Event name -> sections/ file name (without the .json extension).
EVENT_SECTIONS_FILE: dict[str, str] = {
    "Australian Grand Prix": "albert-park-circuit",
    "Austrian Grand Prix": "red-bull-ring",
    "Azerbaijan Grand Prix": "baku-city-circuit",
    "Barcelona Grand Prix": "circuit-de-barcelona-catalunya",
    "Belgian Grand Prix": "circuit-de-spa-francorchamps",
    "Dutch Grand Prix": "circuit-zandvoort",
    "Miami Grand Prix": "miami-international-autodrome",
    "Monaco Grand Prix": "circuit-de-monaco",
    "Saudi Arabian Grand Prix": "jeddah-corniche-circuit",
    "Singapore Grand Prix": "marina-bay-street-circuit",
    "Qatar Grand Prix": "lusail-international-circuit",
    "Abu Dhabi Grand Prix": "yas-marina-circuit",
    "United States Grand Prix": "circuit-of-the-americas",
    "Mexico City Grand Prix": "autodrome-hermanos-rodriguez",
    "São Paulo Grand Prix": "interlagos-circuit",
    "Pre-Season Testing": "bahrain-international-circuit",
    "Pre-Season Testing 1": "bahrain-international-circuit",
    "Pre-Season Testing 2": "bahrain-international-circuit",
}

_CIRCUITS_INDEX_CACHE: dict[str, dict[str, Any]] | None = None
_CIRCUIT_PAYLOAD_CACHE: dict[tuple[int, int], dict[str, Any]] = {}


@dataclass(frozen=True)
class LocalCircuitInfo:
    corners: list[dict[str, Any]]
    marshal_lights: list[dict[str, Any]]
    marshal_sectors: list[dict[str, Any]]
    rotation: float
    requested_year: int
    source_year: int
    sections: list[dict[str, Any]] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        return {
            "corners": self.corners,
            "sections": self.sections,
            "marshal_lights": self.marshal_lights,
            "marshal_sectors": self.marshal_sectors,
            "rotation": self.rotation,
        }


def _api_get(path: str) -> dict[str, Any]:
    url = f"{PROTO}://{HOST}{path}"
    response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def _get_circuits_index() -> dict[str, dict[str, Any]]:
    global _CIRCUITS_INDEX_CACHE
    if _CIRCUITS_INDEX_CACHE is None:
        _CIRCUITS_INDEX_CACHE = _api_get("/api/v1/circuits/")
    return _CIRCUITS_INDEX_CACHE


def get_supported_circuit_years(circuit_key: int) -> list[int]:
    circuit_index = _get_circuits_index()
    circuit = circuit_index.get(str(circuit_key))
    if circuit is None:
        raise KeyError(f"No circuit metadata found for circuit key {circuit_key}")

    years = sorted({int(year) for year in circuit.get("years", [])})
    if not years:
        raise ValueError(f"No supported circuit years returned for {circuit_key}")
    return years


def pick_supported_circuit_year(requested_year: int, supported_years: list[int]) -> int:
    years = sorted({int(year) for year in supported_years})
    if not years:
        raise ValueError("supported_years cannot be empty")

    if requested_year in years:
        return requested_year

    past_or_current_years = [year for year in years if year <= requested_year]
    if past_or_current_years:
        return past_or_current_years[-1]

    return years[0]


def _get_circuit_payload(circuit_key: int, year: int) -> dict[str, Any]:
    cache_key = (circuit_key, year)
    if cache_key not in _CIRCUIT_PAYLOAD_CACHE:
        _CIRCUIT_PAYLOAD_CACHE[cache_key] = _api_get(
            f"/api/v1/circuits/{circuit_key}/{year}"
        )
    return _CIRCUIT_PAYLOAD_CACHE[cache_key]


def _as_int_or_none(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _as_float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    result = float(value)
    if not math.isfinite(result):
        return None
    return result


def _marker_rows(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for entry in entries:
        track_position = entry.get("trackPosition") or {}
        rows.append(
            {
                "X": _as_float_or_none(track_position.get("x")),
                "Y": _as_float_or_none(track_position.get("y")),
                "Number": _as_int_or_none(entry.get("number")),
                "Letter": str(entry.get("letter") or ""),
                "Angle": _as_float_or_none(entry.get("angle")),
                "Distance": None,
            }
        )
    return rows


def get_reference_telemetry_samples(reference_lap: Any) -> list[dict[str, float]]:
    telemetry = None
    telemetry_exc = None
    try:
        telemetry = reference_lap.get_telemetry(frequency="original")
    except TypeError:
        try:
            telemetry = reference_lap.get_telemetry()
        except Exception as exc:
            telemetry = None
            telemetry_exc = exc
    except Exception as exc:
        telemetry = None
        telemetry_exc = exc

    if telemetry is not None:
        if telemetry.empty:
            raise ValueError("Reference lap telemetry is empty")

        samples = []
        for row in telemetry.to_dict(orient="records"):
            if row.get("Source") != "pos":
                continue

            x_pos = _as_float_or_none(row.get("X"))
            y_pos = _as_float_or_none(row.get("Y"))
            distance = _as_float_or_none(row.get("Distance"))
            if None in (x_pos, y_pos, distance):
                continue

            samples.append({"X": x_pos, "Y": y_pos, "Distance": distance})

        if samples:
            return samples

    try:
        pos_data = reference_lap.get_pos_data(pad=1, pad_side="both")
    except Exception as pos_exc:
        if telemetry_exc is not None:
            raise RuntimeError(
                "Failed to load telemetry for the reference lap "
                f"({type(telemetry_exc).__name__}: {telemetry_exc}); "
                "position-data fallback also failed "
                f"({type(pos_exc).__name__}: {pos_exc})"
            ) from pos_exc
        raise RuntimeError(
            "Failed to load position data for the reference lap "
            f"({type(pos_exc).__name__}: {pos_exc})"
        ) from pos_exc

    if pos_data.empty:
        raise ValueError("Reference lap position data is empty")

    samples = []
    cumulative_distance = 0.0
    prev_x = None
    prev_y = None
    for row in pos_data.to_dict(orient="records"):
        source = row.get("Source")
        if source not in (None, "pos"):
            continue

        x_pos = _as_float_or_none(row.get("X"))
        y_pos = _as_float_or_none(row.get("Y"))
        if None in (x_pos, y_pos):
            continue

        if prev_x is not None and prev_y is not None:
            cumulative_distance += math.hypot(x_pos - prev_x, y_pos - prev_y) / 10.0

        samples.append({"X": x_pos, "Y": y_pos, "Distance": cumulative_distance})
        prev_x = x_pos
        prev_y = y_pos

    if not samples:
        raise ValueError("No usable positional samples found for the reference lap")

    return samples


def assign_marker_distances(
    markers: list[dict[str, Any]], telemetry_samples: list[dict[str, float]]
) -> list[dict[str, Any]]:
    if not telemetry_samples:
        raise ValueError("telemetry_samples cannot be empty")

    resolved_markers = []
    for marker in markers:
        marker_x = _as_float_or_none(marker.get("X"))
        marker_y = _as_float_or_none(marker.get("Y"))
        if None in (marker_x, marker_y):
            resolved_markers.append({**marker, "Distance": None})
            continue

        best_distance = None
        best_error = None
        for sample in telemetry_samples:
            error = (sample["X"] - marker_x) ** 2 + (sample["Y"] - marker_y) ** 2
            if best_error is None or error < best_error:
                best_error = error
                best_distance = sample["Distance"]

        resolved_markers.append({**marker, "Distance": best_distance})

    return resolved_markers


def _slug_candidates(name: str) -> list[str]:
    """Return hyphenated slug candidates for a circuit/event name.

    "Marina Bay Street Circuit" -> ["marina-bay-street-circuit"]
    """
    lowered = str(name).strip().lower()
    if not lowered:
        return []

    def _slugify(value: str) -> str:
        return "-".join(
            "".join(ch if ch.isalnum() else " " for ch in value).split()
        )

    stripped = _slugify(re.sub(r"\s*grand\s*prix\s*", " ", lowered))
    full_slug = _slugify(lowered)

    candidates = []
    for slug in (stripped, full_slug):
        if slug and slug not in candidates:
            candidates.append(slug)
    return candidates


def _sections_file_stems() -> list[str]:
    """Section file names in SECTIONS_DIR without the .json extension."""
    try:
        return [
            fn[: -len(".json")]
            for fn in os.listdir(SECTIONS_DIR)
            if fn.endswith(".json")
        ]
    except OSError:
        return []


def parse_circuit_sections(raw: Any) -> list[tuple[str, float, float]]:
    """Parse and validate raw section rows: [type, start_percent, end_percent]."""
    if not isinstance(raw, list) or not raw:
        raise ValueError("Circuit sections must be a non-empty list")

    rows = []
    for entry in raw:
        if isinstance(entry, dict):
            section_type = entry.get("Type") or entry.get("type")
            start = entry.get("StartPercent", entry.get("start"))
            end = entry.get("EndPercent", entry.get("end"))
        else:
            section_type, start, end = entry[0], entry[1], entry[2]

        section_type = str(section_type).strip().capitalize()
        if section_type not in VALID_SECTION_TYPES:
            raise ValueError(
                f"Invalid circuit section type {section_type!r}; "
                f"expected one of {sorted(VALID_SECTION_TYPES)}"
            )

        start = float(start)
        end = float(end)
        if not (math.isfinite(start) and math.isfinite(end)):
            raise ValueError("Circuit section bounds must be finite numbers")
        if not (0.0 <= start < end <= 100.0):
            raise ValueError(
                f"Invalid circuit section bounds {start}-{end}; "
                "expected 0 <= start < end <= 100"
            )

        rows.append((section_type, start, end))

    return rows


def load_circuit_sections(
    event_name: str, session_info: dict[str, Any] | None = None
) -> list[tuple[str, float, float]] | None:
    """Load sections/<circuit>.json for this event, if one exists.

    Files are named after circuits, not events, so known events are mapped
    via EVENT_SECTIONS_FILE; other names fall back to slug matching. Returns
    None when no section breakdown has been added for this circuit yet (only
    some circuits have one).
    """
    names = [event_name]
    if session_info:
        meeting = session_info.get("Meeting") or {}
        circuit = meeting.get("Circuit") or {}
        names.extend(
            str(value)
            for value in (
                meeting.get("Name"),
                circuit.get("ShortName"),
                circuit.get("Name"),
                meeting.get("Location"),
            )
            if value
        )

    candidates: list[str] = []
    for name in names:
        mapped = EVENT_SECTIONS_FILE.get(str(name))
        if mapped and mapped not in candidates:
            candidates.append(mapped)
        for slug in _slug_candidates(name):
            if slug not in candidates:
                candidates.append(slug)

    # Exact file-name matches first; then fall back to circuit-named files
    # that contain every word of a candidate (e.g. "monaco" matches
    # "circuit-de-monaco.json").
    available = _sections_file_stems()
    stems = [slug for slug in candidates if slug in available]
    if not stems:
        for slug in candidates:
            tokens = set(slug.split("-"))
            for stem in available:
                if tokens and tokens <= set(stem.split("-")) and stem not in stems:
                    stems.append(stem)

    for slug in stems:
        path = os.path.join(SECTIONS_DIR, f"{slug}.json")
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as f:
            return parse_circuit_sections(json.load(f))

    return None


def build_section_entries(
    section_rows: list[tuple[str, float, float]], total_lap_distance: float
) -> list[dict[str, Any]]:
    """Convert percentage-based sections to absolute distances using the
    reference (fastest) lap's total lap distance."""
    if not math.isfinite(total_lap_distance) or total_lap_distance <= 0:
        raise ValueError("total_lap_distance must be a positive finite number")

    return [
        {
            "Type": section_type,
            "StartPercent": start,
            "EndPercent": end,
            "StartDistance": start / 100.0 * total_lap_distance,
            "EndDistance": end / 100.0 * total_lap_distance,
        }
        for section_type, start, end in section_rows
    ]


def get_section_for_distance(
    distance: float | None,
    section_entries: list[dict[str, Any]],
    total_lap_distance: float,
) -> str | None:
    """Return the section (Straight/Low/Medium/High) containing this distance."""
    if distance is None or not section_entries:
        return None

    percent = distance / total_lap_distance * 100.0
    for index, entry in enumerate(section_entries):
        is_last = index == len(section_entries) - 1
        if entry["StartPercent"] <= percent < entry["EndPercent"]:
            return entry["Type"]
        if is_last and percent >= entry["EndPercent"]:
            return entry["Type"]
    return None


def get_total_lap_distance(telemetry_samples: list[dict[str, float]]) -> float:
    if not telemetry_samples:
        raise ValueError("telemetry_samples cannot be empty")
    return max(sample["Distance"] for sample in telemetry_samples)


def add_sections_local(
    circuit_info: LocalCircuitInfo,
    section_rows: list[tuple[str, float, float]],
    total_lap_distance: float,
) -> LocalCircuitInfo:
    """Attach section breakdown and classify each corner into its section."""
    section_entries = build_section_entries(section_rows, total_lap_distance)

    corners = [
        {
            **corner,
            "Section": get_section_for_distance(
                corner.get("Distance"), section_entries, total_lap_distance
            ),
        }
        for corner in circuit_info.corners
    ]

    return replace(circuit_info, corners=corners, sections=section_entries)


def add_marker_distance_from_samples(
    circuit_info: LocalCircuitInfo, telemetry_samples: list[dict[str, float]]
) -> LocalCircuitInfo:
    return LocalCircuitInfo(
        corners=assign_marker_distances(circuit_info.corners, telemetry_samples),
        marshal_lights=assign_marker_distances(
            circuit_info.marshal_lights, telemetry_samples
        ),
        marshal_sectors=assign_marker_distances(
            circuit_info.marshal_sectors, telemetry_samples
        ),
        rotation=circuit_info.rotation,
        requested_year=circuit_info.requested_year,
        source_year=circuit_info.source_year,
        sections=circuit_info.sections,
    )


def add_marker_distance_local(
    circuit_info: LocalCircuitInfo, reference_lap: Any
) -> LocalCircuitInfo:
    telemetry_samples = get_reference_telemetry_samples(reference_lap)
    return add_marker_distance_from_samples(circuit_info, telemetry_samples)


def get_local_circuit_info(circuit_key: int, requested_year: int) -> LocalCircuitInfo:
    supported_years = get_supported_circuit_years(circuit_key)
    source_year = pick_supported_circuit_year(requested_year, supported_years)
    payload = _get_circuit_payload(circuit_key, source_year)

    return LocalCircuitInfo(
        corners=_marker_rows(payload.get("corners", [])),
        marshal_lights=_marker_rows(payload.get("marshalLights", [])),
        marshal_sectors=_marker_rows(payload.get("marshalSectors", [])),
        rotation=float(payload.get("rotation", 0.0) or 0.0),
        requested_year=requested_year,
        source_year=source_year,
    )


def get_session_circuit_info(
    year: int, event_name: str, session_name: str
) -> LocalCircuitInfo:
    if fastf1 is None:
        raise RuntimeError("fastf1 is required to load session telemetry")

    session = fastf1.get_session(year, event_name, session_name)
    session.load(laps=True, telemetry=True, weather=False, messages=False)

    circuit_key = int(session.session_info["Meeting"]["Circuit"]["Key"])
    fastest_lap = session.laps.pick_fastest()
    if fastest_lap is None:
        raise ValueError(f"No fastest lap available for {event_name}/{session_name}")

    circuit_info = get_local_circuit_info(circuit_key, year)

    # Same process as FastF1's circuit_info.add_marker_distance(): the fastest
    # lap is the reference lap and all marker/section distances are resolved
    # against its telemetry.
    telemetry_samples = get_reference_telemetry_samples(fastest_lap)
    circuit_info = add_marker_distance_from_samples(circuit_info, telemetry_samples)

    section_rows = load_circuit_sections(event_name, session.session_info)
    if section_rows is not None:
        total_lap_distance = get_total_lap_distance(telemetry_samples)
        circuit_info = add_sections_local(
            circuit_info, section_rows, total_lap_distance
        )

    return circuit_info


def main() -> None:
    for event_name in EVENTS:
        for session_name in SESSIONS:
            try:
                circuit_info = get_session_circuit_info(YEAR, event_name, session_name)
            except Exception as exc:
                print(f"✗ Failed {event_name}/{session_name}: {exc}")
                continue

            if circuit_info.source_year != circuit_info.requested_year:
                print(
                    "!"
                    f" Using circuit layout year {circuit_info.source_year} for "
                    f"{event_name} {circuit_info.requested_year}"
                )

            payload = circuit_info.to_payload()
            output_dir = f"{event_name}/{session_name}"
            os.makedirs(output_dir, exist_ok=True)

            output_path = f"{output_dir}/corners.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, allow_nan=False)

            sections_note = (
                f", sections: {len(circuit_info.sections)}"
                if circuit_info.sections
                else ", no section breakdown available"
            )
            print(
                "✓ Corners data saved to "
                f"{output_path} (reference lap: fastest lap, "
                f"layout year: {circuit_info.source_year}{sections_note})"
            )


if __name__ == "__main__":
    main()
