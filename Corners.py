"""
Corners Backfill
================
Rebuilds {event}/{session}/corners.json in a season repository exactly as the
extraction scripts write it (R.py get_circuit_info), without loading a FastF1
session or needing an F1 account:

- corner, marshal light and marshal sector geometry from the MultiViewer API
  (the request FastF1's get_circuit_info makes), as FastF1 parses it;
- their distances matched against the fastest lap's telemetry as FastF1's
  CircuitInfo.add_marker_distance does: the Distance of the position sample
  (``Source == "pos"``) nearest to each marker. The repository's telemetry
  is that lap's merged telemetry; its raw position samples are the ones with
  whole-number x/y (merged car samples carry interpolated positions);
- the section breakdown from sections/ (percentages of the fastest lap's
  total distance), chosen by circuit and layout era (SECTIONS_BY_CIRCUIT);
- the extra MultiViewer fields (utils.build_circuit_details).

The fastest lap is FastF1's pick_fastest(): the quickest personal-best lap.
A session without usable telemetry for it uses the fastest lap of another
session of the same event (Qualifying first; same circuit, only the reference
lap differs); without any, the MultiViewer distances are kept and the section
distances are None.

A file already in the current layout keeps every field (MultiViewer's pit loss
values, for one, change over time) and only gets its section fields added or
corrected; --full rebuilds it too. Older-layout files (CornerNumber, X, Y,
Angle, Distance and Rotation only) and missing files are built in full. A file
is only written when its content changes.

The repository can be this checkout or a partial clone of another season
(git clone --depth 1 --filter=blob:none --sparse ...); the files read are
fetched in batches. Changed files are staged.

Usage:
    python Corners.py                                   # this season, this checkout
    python Corners.py --year 2022 --repo ../2022
    python Corners.py --year 2022 --repo ../2022 "Monaco Grand Prix"
    python Corners.py --full ...                        # rebuild current-layout files too
"""

import argparse
import logging
import os
import re
import subprocess
import time
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import orjson
import requests

from utils import build_circuit_details

DEFAULT_YEAR = 2026
SECTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sections")
MV_URL = "https://api.multiviewer.app/api/v1/circuits/{key}/{year}"
HEADERS = {"User-Agent": "FastF1/"}
VALID_SECTION_TYPES = {"Straight", "Low", "Medium", "High"}

# MultiViewer circuit key -> sections/ file by layout era: (last season, file)
# pairs in season order, the last one open-ended. Circuits whose layout
# changed have a "-2018" file for the seasons before the change (Albert Park
# 2022, Catalunya and Marina Bay 2023, Yas Marina 2021). Circuits without a
# file (Istanbul, Portimão, Nürburgring, Mugello, the Sakhir outer track, Las
# Vegas, Kuala Lumpur, Madring) get no sections.
SECTIONS_BY_CIRCUIT: Dict[int, List[Tuple[Optional[int], str]]] = {
    2: [(None, "silverstone")],
    4: [(None, "hungary")],
    6: [(None, "emilia")],
    7: [(None, "circuit-de-spa-francorchamps")],
    9: [(None, "circuit-of-the-americas")],
    10: [(2021, "albert-park-circuit-2018"), (None, "albert-park-circuit")],
    14: [(None, "interlagos-circuit")],
    15: [(2022, "circuit-de-barcelona-catalunya-2018"), (None, "circuit-de-barcelona-catalunya")],
    19: [(None, "red-bull-ring")],
    22: [(None, "circuit-de-monaco")],
    23: [(None, "canada")],
    28: [(None, "circuit-paul-ricard")],
    34: [(None, "hockenheimring")],
    39: [(None, "monza")],
    46: [(None, "suzuka-circuit")],
    49: [(None, "shanghai-international-circuit")],
    55: [(None, "circuit-zandvoort")],
    61: [(2022, "marina-bay-street-circuit-2018"), (None, "marina-bay-street-circuit")],
    63: [(None, "bahrain-international-circuit")],
    65: [(None, "autodrome-hermanos-rodriguez")],
    70: [(2020, "yas-marina-circuit-2018"), (None, "yas-marina-circuit")],
    79: [(None, "sochi-autodrom")],
    144: [(None, "baku-city-circuit")],
    149: [(None, "jeddah-corniche-circuit")],
    150: [(None, "lusail-international-circuit")],
    151: [(None, "miami-international-autodrome")],
}
# Keys MultiViewer shares between circuits: the short name decides.
SECTIONS_SHORT_NAME = {149: "Jeddah"}

# Sessions whose fastest lap stands in for a session without usable telemetry.
REFERENCE_SESSION_ORDER = [
    "Qualifying", "Race", "Sprint", "Sprint Qualifying", "Sprint Shootout",
    "Practice 3", "Practice 2", "Practice 1",
]

_SESSION_FILE_RE = re.compile(r"^([^/]+)/([^/]+)/([^/]+\.json)$")
_DRIVER_FILE_RE = re.compile(r"^([^/]+)/([^/]+)/([^/]+)/([^/]+)$")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("corners_backfill")


# -- repository access ------------------------------------------------------------


class Repo:
    """Tracked files of a checkout or partial clone, read in batches."""

    def __init__(self, path: str):
        self.path = path
        self.oids: Dict[str, str] = {}
        out = self._git("ls-tree", "-r", "-z", "HEAD")
        for rec in out.decode().split("\0"):
            if rec:
                meta, name = rec.split("\t", 1)
                self.oids[name] = meta.split()[2]

    def _git(self, *args: str, stdin: Optional[bytes] = None) -> bytes:
        return subprocess.run(
            ["git", "-C", self.path, *args], input=stdin, capture_output=True, check=True
        ).stdout

    def read_many(self, paths: List[str]) -> Dict[str, bytes]:
        """Committed contents of ``paths`` (fetching missing blobs in one request)."""
        wanted = {p: self.oids[p] for p in paths if p in self.oids}
        if not wanted:
            return {}
        oids = sorted(set(wanted.values()))
        missing = self._git(
            "cat-file", "--batch-check", stdin="\n".join(oids).encode()
        ).decode().splitlines()
        absent = [line.split()[0] for line in missing if line.endswith(" missing")]
        if absent:
            self._git(
                "-c", "fetch.negotiationAlgorithm=noop", "fetch", "-q", "--no-tags",
                "--no-write-fetch-head", "--recurse-submodules=no", "--filter=blob:none",
                "--stdin", "origin", stdin="\n".join(absent).encode(),
            )
        raw = self._git("cat-file", "--batch", stdin="\n".join(oids).encode())
        blobs, pos = {}, 0
        while pos < len(raw):
            end = raw.index(b"\n", pos)
            oid, _, size = raw[pos:end].decode().split()
            blobs[oid] = raw[end + 1 : end + 1 + int(size)]
            pos = end + 1 + int(size) + 1
        return {p: blobs[o] for p, o in wanted.items()}


# -- sections ---------------------------------------------------------------------


def sections_file(circuit_key: Optional[int], short_name: str, year: int) -> Optional[str]:
    eras = SECTIONS_BY_CIRCUIT.get(circuit_key) if circuit_key is not None else None
    if not eras:
        return None
    expected = SECTIONS_SHORT_NAME.get(circuit_key)
    if expected is not None and short_name != expected:
        return None
    for last_season, stem in eras:
        if last_season is None or year <= last_season:
            return stem
    return None


def load_sections(stem: Optional[str]) -> Optional[List[Tuple[str, float, float]]]:
    """Section rows of sections/<stem>.json, validated as R.py does."""
    if stem is None:
        return None
    with open(os.path.join(SECTIONS_DIR, f"{stem}.json"), "rb") as f:
        raw = orjson.loads(f.read())
    rows = []
    for entry in raw:
        if isinstance(entry, dict):
            section_type = entry.get("Type") or entry.get("type")
            start = entry.get("StartPercent", entry.get("start"))
            end = entry.get("EndPercent", entry.get("end"))
        else:
            section_type, start, end = entry[0], entry[1], entry[2]
        section_type = str(section_type).strip().capitalize()
        start, end = float(start), float(end)
        if section_type not in VALID_SECTION_TYPES or not 0.0 <= start < end <= 100.0:
            raise ValueError(f"invalid section row {entry!r} in {stem}.json")
        rows.append((section_type, start, end))
    return rows


def _section_for_distance(distance, section_rows, reference_lap_distance) -> Optional[str]:
    if not section_rows or reference_lap_distance is None:
        return None
    if distance is None or isinstance(distance, str):
        return None
    percent = float(distance) / reference_lap_distance * 100.0
    if percent != percent:
        return None
    for index, (section_type, start, end) in enumerate(section_rows):
        if start <= percent < end:
            return section_type
        if index == len(section_rows) - 1 and percent >= end:
            return section_type
    return None


def add_sections(corner_info: Dict, section_rows, reference_lap_distance) -> None:
    """R.py's _add_sections_to_corner_info."""
    section_rows = section_rows or []
    corner_info["Section"] = [
        _section_for_distance(d, section_rows, reference_lap_distance)
        for d in corner_info.get("Distance") or []
    ]
    corner_info["SectionType"] = [row[0] for row in section_rows]
    corner_info["SectionStartPercent"] = [row[1] for row in section_rows]
    corner_info["SectionEndPercent"] = [row[2] for row in section_rows]
    if section_rows and reference_lap_distance is not None:
        corner_info["SectionStartDistance"] = [
            row[1] / 100.0 * reference_lap_distance for row in section_rows
        ]
        corner_info["SectionEndDistance"] = [
            row[2] / 100.0 * reference_lap_distance for row in section_rows
        ]
    else:
        corner_info["SectionStartDistance"] = [None] * len(section_rows)
        corner_info["SectionEndDistance"] = [None] * len(section_rows)


# -- reference lap ------------------------------------------------------------------


def fastest_lap(laps: Dict) -> Optional[Tuple[str, int]]:
    """FastF1's pick_fastest(): ``(driver, lap)`` of the quickest personal-best lap."""
    best = None
    for drv, lap, lap_time, pb in zip(
        laps.get("drv") or [], laps.get("lap") or [], laps.get("time") or [], laps.get("pb") or []
    ):
        if pb is not True or isinstance(lap_time, bool) or not isinstance(lap_time, (int, float)):
            continue
        if best is None or lap_time < best[2]:
            best = (drv, int(lap), lap_time)
    return None if best is None else (best[0], best[1])


def reference_samples(tel_payload: Dict):
    """``(position samples [x, y, distance] or None, total lap distance or None)``.

    The total distance (the section reference) needs no positions.
    """
    tel = tel_payload.get("tel") or {}

    def column(name):
        return np.array(
            [v if isinstance(v, (int, float)) and not isinstance(v, bool) else np.nan
             for v in tel.get(name) or []],
            dtype=float,
        )

    x, y, d = column("x"), column("y"), column("distance")
    if not len(d) or np.isnan(d).all():
        return None, None
    total = float(np.nanmax(d))
    total = total if total > 0 else None
    if len(x) != len(d) or len(y) != len(d):
        return None, total
    valid = ~(np.isnan(x) | np.isnan(y) | np.isnan(d))
    pos = valid & (np.mod(x, 1) == 0) & (np.mod(y, 1) == 0)
    if not pos.any():
        return None, total
    return np.column_stack([x[pos], y[pos], d[pos]]), total


def marker_distances(entries: List[Dict], samples) -> List[Optional[float]]:
    """FastF1's add_marker_distance for one marker table."""
    out = []
    for entry in entries:
        pos = entry.get("trackPosition") or {}
        mx, my = float(pos.get("x", 0.0)), float(pos.get("y", 0.0))
        err = (samples[:, 0] - mx) ** 2 + (samples[:, 1] - my) ** 2
        out.append(float(samples[int(np.argmin(err)), 2]))
    return out


# -- MultiViewer --------------------------------------------------------------------

_MV_CACHE: Dict[Tuple[int, int], Optional[Dict]] = {}


def mv_circuit(circuit_key: int, year: int) -> Optional[Dict]:
    key = (circuit_key, year)
    if key not in _MV_CACHE:
        response = requests.get(
            MV_URL.format(key=circuit_key, year=year), headers=HEADERS, timeout=30
        )
        _MV_CACHE[key] = response.json() if response.status_code == 200 else None
    return _MV_CACHE[key]


# -- corners.json -------------------------------------------------------------------


_SECTION_KEYS = (
    "Section", "SectionType", "SectionStartPercent", "SectionEndPercent",
    "SectionStartDistance", "SectionEndDistance",
)
# Fields of the current layout that an older-layout file lacks.
_CURRENT_LAYOUT_KEYS = ("CornerLetter", "MarshalLightNumber", "CircuitKey")


def corners_payload(mv: Dict, samples, total, section_rows) -> Dict:
    """corners.json as R.py builds it from FastF1's CircuitInfo."""
    corners = mv.get("corners") or []
    if samples is not None:
        distances = marker_distances(corners, samples)
        details_info = SimpleNamespace(
            marshal_lights={"Distance": marker_distances(mv.get("marshalLights") or [], samples)},
            marshal_sectors={"Distance": marker_distances(mv.get("marshalSectors") or [], samples)},
        )
    else:
        # No reference lap anywhere: MultiViewer's own distances (meters).
        total = None
        distances = [float(c.get("length", 0.0)) / 10 for c in corners]
        details_info = None
    result = {
        "CornerNumber": [int(c.get("number", 0)) for c in corners],
        "X": [float((c.get("trackPosition") or {}).get("x", 0.0)) for c in corners],
        "Y": [float((c.get("trackPosition") or {}).get("y", 0.0)) for c in corners],
        "Angle": [float(c.get("angle", 0.0)) for c in corners],
        "Distance": distances,
        "Rotation": float(mv.get("rotation", 0.0)),
    }
    add_sections(result, section_rows, total)
    result.update(build_circuit_details(mv, details_info))
    return result


def with_sections(old: Dict, section_rows, total) -> Dict:
    """``old`` with its section fields recomputed, placed after Rotation as R.py writes them."""
    sections = {"Distance": old.get("Distance")}
    add_sections(sections, section_rows, total)
    del sections["Distance"]
    out: Dict = {}
    for key, value in old.items():
        if key in _SECTION_KEYS:
            continue
        out[key] = value
        if key == "Rotation":
            out.update(sections)
    if "Section" not in out:
        out.update(sections)
    return out


def _same_sections(old: Dict, new: Dict) -> bool:
    """The same section content: types, bounds and per-corner sections, distances known alike.

    The section distances follow the reference lap, which an earlier run may
    have taken from another session; that alone does not rewrite a file.
    """
    if any(old.get(k) != new[k] for k in _SECTION_KEYS[:4]):
        return False
    for key in _SECTION_KEYS[4:]:
        a, b = old.get(key), new[key]
        if a is None or len(a) != len(b) or any((x is None) != (y is None) for x, y in zip(a, b)):
            return False
    return True


def backfill(
    year: int, repo_path: str, events: Optional[List[str]] = None, full: bool = False
) -> None:
    start = time.time()
    repo = Repo(repo_path)
    sessions: Dict[Tuple[str, str], set] = {}
    drivers_laps: Dict[Tuple[str, str], List[str]] = {}
    for name in repo.oids:
        m = _SESSION_FILE_RE.match(name)
        if m and not m.group(1).startswith("."):
            sessions.setdefault((m.group(1), m.group(2)), set()).add(m.group(3))
            continue
        m = _DRIVER_FILE_RE.match(name)
        if m and m.group(4) == "laptimes.json":
            drivers_laps.setdefault((m.group(1), m.group(2)), []).append(name)
    targets = sorted(
        k for k, files in sessions.items()
        if (not events or k[0] in events) and ("session_info.json" in files or "corners.json" in files)
    )

    # Batch 1: circuit identity, laps, existing corners.
    first = []
    for e, s in targets:
        first += [f"{e}/{s}/session_info.json", f"{e}/{s}/corners.json"]
        if "session_laptimes.json" in sessions[(e, s)]:
            first.append(f"{e}/{s}/session_laptimes.json")
        else:
            first += drivers_laps.get((e, s), [])
    blobs = repo.read_many(first)

    def laps_of(e, s):
        path = f"{e}/{s}/session_laptimes.json"
        if path in blobs:
            return orjson.loads(blobs[path])
        merged: Dict[str, list] = {"drv": [], "lap": [], "time": [], "pb": []}
        for p in sorted(drivers_laps.get((e, s), [])):
            laps = orjson.loads(blobs[p])
            for k in merged:
                merged[k] += list(laps.get(k) or [])
        return merged

    fastest = {k: fastest_lap(laps_of(*k)) for k in targets}
    tel_paths = {
        k: f"{k[0]}/{k[1]}/{f[0]}/{f[1]}_tel.json" for k, f in fastest.items() if f is not None
    }
    tels = repo.read_many(list(tel_paths.values()))  # batch 2
    reference = {
        k: reference_samples(orjson.loads(tels[p])) if p in tels else (None, None)
        for k, p in tel_paths.items()
    }

    def borrow(e, index):
        for other in REFERENCE_SESSION_ORDER:
            value = reference.get((e, other), (None, None))
            if value[index] is not None:
                return value, other
        return (None, None), None

    written, sections_only, unchanged, failed, notes = [], 0, 0, [], []
    for e, s in targets:
        label = f"{e}/{s}"
        info_raw = blobs.get(f"{label}/session_info.json")
        old_raw = blobs.get(f"{label}/corners.json")
        old = orjson.loads(old_raw) if old_raw is not None else None
        if old is None and fastest.get((e, s)) is None:
            continue  # a session without laps (cancelled) gets no corners.json
        if info_raw is not None:
            circuit = orjson.loads(info_raw)["session_info"]["Meeting"]["Circuit"]
            circuit_key, short_name = circuit.get("Key"), circuit.get("ShortName") or ""
        else:
            circuit_key, short_name = old.get("CircuitKey"), old.get("CircuitName") or ""
        if circuit_key is None:
            failed.append(f"{label}: no circuit key")
            continue
        section_rows = load_sections(sections_file(int(circuit_key), short_name, year))
        samples, total = reference.get((e, s), (None, None))

        if old is not None and not full and all(k in old for k in _CURRENT_LAYOUT_KEYS):
            if total is None:
                (_, total), other = borrow(e, 1)
                if other:
                    notes.append(f"{label}: section distances from the {other} lap")
            new = with_sections(old, section_rows, total)
            if _same_sections(old, new):
                unchanged += 1
                continue
            payload = orjson.dumps(new)
            sections_only += 1
        else:
            mv = mv_circuit(int(circuit_key), year)
            if not mv or not mv.get("corners"):
                failed.append(f"{label}: no MultiViewer circuit data for key {circuit_key}")
                continue
            if samples is None:
                (samples, total), other = borrow(e, 0)
                notes.append(
                    f"{label}: reference lap from {other}" if other
                    else f"{label}: MultiViewer distances only"
                )
            payload = orjson.dumps(corners_payload(mv, samples, total, section_rows))
            if payload == old_raw:
                unchanged += 1
                continue
        out = os.path.join(repo_path, label, "corners.json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "wb") as f:
            f.write(payload)
        written.append(f"{label}/corners.json")

    for i in range(0, len(written), 200):
        repo._git("add", "--sparse", "--", *written[i : i + 200])
    logger.info(
        f"{year} done in {time.time() - start:.1f}s: {len(written)} written and staged "
        f"({sections_only} section fields only, {len(written) - sections_only} built in full), "
        f"{unchanged} unchanged, {len(failed)} failed"
    )
    for line in failed:
        logger.info(f"  failed: {line}")
    for line in notes:
        logger.info(f"  note: {line}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild corners.json in a season repo.")
    parser.add_argument("--year", type=int, default=DEFAULT_YEAR)
    parser.add_argument("--repo", default=".", help="checkout or partial clone of the season")
    parser.add_argument("--full", action="store_true", help="rebuild current-layout files too")
    parser.add_argument("events", nargs="*", help="only these event folders")
    args = parser.parse_args()
    backfill(args.year, args.repo, args.events, args.full)


if __name__ == "__main__":
    main()
