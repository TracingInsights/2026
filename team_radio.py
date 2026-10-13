"""
Team Radio Download
===================
Downloads OpenF1 team radio recordings (MP3) for F1 sessions.

Output per session:
    {event_name}/{session_name}/{driver}/radio/{original_file_name}.mp3
    {event_name}/{session_name}/team_radio.json   (columnar index, sorted by time)

team_radio.json columns:
    drv   - driver abbreviation (e.g. "VER")
    dNum  - driver number as a string
    date  - UTC timestamp of the message (ISO 8601, from OpenF1)
    file  - MP3 path relative to the session folder ("None" if download failed)
    url   - original recording URL on livetiming.formula1.com

Used two ways:
- Imported by LapTimes.py, which downloads radios for the sessions it extracts.
- Run directly (``python team_radio.py``) to backfill every session of the
  season that has already started.
"""

import logging
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Union
from urllib.parse import urlparse

import orjson
import pandas as pd
import requests

logger = logging.getLogger("team_radio")

OPENF1_BASE_URL = "https://api.openf1.org/v1/"
OPENF1_MAX_CALLS_PER_SECOND = 3
OPENF1_MAX_CALLS_PER_MINUTE = 30

DOWNLOAD_WORKERS = 8
DOWNLOAD_RETRIES = 3
DOWNLOAD_TIMEOUT = 30

# Max gap between an OpenF1 session start and a FastF1 schedule slot to treat
# them as the same session (standalone backfill only).
SESSION_MATCH_TOLERANCE = pd.Timedelta(hours=3)

TEAM_RADIO_INDEX_KEYS = ("drv", "dNum", "date", "file", "url")

OpenF1Getter = Callable[[str, Optional[Dict[str, Union[int, str]]]], Any]


# ---------------------------------------------------------------------------
# OpenF1 access
# ---------------------------------------------------------------------------


class _ThrottledOpenF1:
    """Minimal rate-limited OpenF1 getter for standalone runs."""

    def __init__(self, retries: int = 3, backoff_factor: float = 0.5):
        self.session = requests.Session()
        self.retries = retries
        self.backoff_factor = backoff_factor
        self._times: deque = deque()
        self._lock = Lock()

    def _throttle(self) -> None:
        with self._lock:
            while True:
                now = time.monotonic()
                while self._times and now - self._times[0] >= 60:
                    self._times.popleft()
                recent = [t for t in self._times if now - t < 1]
                wait = 0.0
                if len(self._times) >= OPENF1_MAX_CALLS_PER_MINUTE:
                    wait = max(wait, 60 - (now - self._times[0]))
                if len(recent) >= OPENF1_MAX_CALLS_PER_SECOND:
                    wait = max(wait, 1 - (now - recent[0]))
                if wait <= 0:
                    self._times.append(now)
                    return
                time.sleep(wait)

    def get(self, path: str, params: Optional[Dict[str, Union[int, str]]] = None):
        url = f"{OPENF1_BASE_URL}{path}"
        for attempt in range(self.retries):
            try:
                self._throttle()
                response = self.session.get(url, params=params, timeout=30)
                if response.status_code == 404:
                    return []
                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After")
                    time.sleep(
                        float(retry_after)
                        if retry_after
                        else self.backoff_factor * (2**attempt)
                    )
                    continue
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as exc:
                logger.error("OpenF1 request failed %s: %s", url, exc)
                if attempt < self.retries - 1:
                    time.sleep(self.backoff_factor * (2**attempt))
        return None


def fetch_team_radio(get: OpenF1Getter, session_key: int) -> List[Dict[str, Any]]:
    """Return OpenF1 team radio entries for a session (empty list on failure)."""
    response = get("team_radio", {"session_key": session_key})
    return response if isinstance(response, list) else []


# ---------------------------------------------------------------------------
# Download & index
# ---------------------------------------------------------------------------


def _download_file(http: requests.Session, url: str, dest: str) -> bool:
    if os.path.isfile(dest) and os.path.getsize(dest) > 0:
        return True
    tmp = dest + ".part"
    for attempt in range(DOWNLOAD_RETRIES):
        try:
            with http.get(url, timeout=DOWNLOAD_TIMEOUT, stream=True) as response:
                if response.status_code == 404:
                    logger.warning("Team radio not found: %s", url)
                    return False
                response.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in response.iter_content(chunk_size=64 * 1024):
                        f.write(chunk)
            if os.path.getsize(tmp) == 0:
                raise requests.exceptions.RequestException("empty response")
            os.replace(tmp, dest)
            return True
        except (requests.exceptions.RequestException, OSError) as exc:
            logger.warning(
                "Team radio download failed (%d/%d) %s: %s",
                attempt + 1,
                DOWNLOAD_RETRIES,
                url,
                exc,
            )
            if attempt < DOWNLOAD_RETRIES - 1:
                time.sleep(0.5 * (2**attempt))
    if os.path.exists(tmp):
        os.remove(tmp)
    return False


def _driver_code(
    entry: Dict[str, Any], file_name: str, driver_codes: Optional[Dict[int, str]]
) -> str:
    number = entry.get("driver_number")
    if driver_codes and number in driver_codes and driver_codes[number]:
        return str(driver_codes[number])
    # Livetiming file names look like "VER_1_20261011_195602.mp3".
    prefix = file_name.split("_", 1)[0]
    if prefix.isalpha() and len(prefix) == 3:
        return prefix.upper()
    return str(number)


def save_session_radios(
    entries: List[Dict[str, Any]],
    base_dir: str,
    driver_codes: Optional[Dict[int, str]] = None,
    http: Optional[requests.Session] = None,
) -> Dict[str, list]:
    """Download radio MP3s for one session and write team_radio.json.

    ``driver_codes`` maps driver number -> abbreviation so files land in the
    same driver folders as the rest of the session data. Existing files are
    skipped, so re-running only fetches new clips.
    """
    entries = [e for e in entries if e.get("recording_url")]
    entries.sort(key=lambda e: str(e.get("date") or ""))

    index: Dict[str, list] = {k: [] for k in TEAM_RADIO_INDEX_KEYS}
    if not entries:
        return index

    os.makedirs(base_dir, exist_ok=True)
    http = http or requests.Session()

    jobs = []
    for entry in entries:
        url = entry["recording_url"]
        file_name = os.path.basename(urlparse(url).path) or "radio.mp3"
        code = _driver_code(entry, file_name, driver_codes)
        rel_path = f"{code}/radio/{file_name}"
        os.makedirs(os.path.join(base_dir, code, "radio"), exist_ok=True)
        jobs.append((entry, code, rel_path, url))

    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        results = list(
            pool.map(
                lambda job: _download_file(
                    http, job[3], os.path.join(base_dir, job[2])
                ),
                jobs,
            )
        )

    for (entry, code, rel_path, url), ok in zip(jobs, results):
        number = entry.get("driver_number")
        index["drv"].append(code)
        index["dNum"].append(str(number) if number is not None else "None")
        index["date"].append(str(entry.get("date") or "None"))
        index["file"].append(rel_path if ok else "None")
        index["url"].append(url)

    with open(os.path.join(base_dir, "team_radio.json"), "wb") as f:
        f.write(orjson.dumps(index))

    failed = results.count(False)
    logger.info(
        "%s: %d team radio clips saved%s",
        base_dir,
        len(results) - failed,
        f", {failed} failed" if failed else "",
    )
    return index


# ---------------------------------------------------------------------------
# Standalone backfill for a whole season
# ---------------------------------------------------------------------------


def _session_folders_by_start(year: int) -> List[tuple]:
    """Return [(start_utc, folder)] for every scheduled session of the season.

    Folder names follow the existing repo layout:
    "{EventName}/{Session}" and "Pre-Season Testing {n}/Practice {m}".
    """
    import fastf1

    schedule = fastf1.get_event_schedule(year, include_testing=True)
    slots = []
    test_number = 0
    for _, event in schedule.iterrows():
        is_testing = str(event.get("EventFormat", "")).lower() == "testing"
        if is_testing:
            test_number += 1
            event_folder = f"Pre-Season Testing {test_number}"
        else:
            event_folder = str(event["EventName"])
        for n in range(1, 6):
            name = event.get(f"Session{n}")
            start = event.get(f"Session{n}DateUtc")
            if not isinstance(name, str) or not name or pd.isna(start):
                continue
            slots.append(
                (pd.Timestamp(start).tz_localize("UTC"), f"{event_folder}/{name}")
            )
    return slots


def download_season_radios(year: int) -> None:
    client = _ThrottledOpenF1()
    sessions = client.get("sessions", {"year": year}) or []
    slots = _session_folders_by_start(year)
    now = pd.Timestamp.now(tz="UTC")
    http = requests.Session()

    for session in sorted(sessions, key=lambda s: str(s.get("date_start") or "")):
        start = pd.to_datetime(session.get("date_start"), utc=True, errors="coerce")
        if pd.isna(start) or start > now:
            continue
        nearest = min(slots, key=lambda slot: abs(slot[0] - start), default=None)
        if nearest is None or abs(nearest[0] - start) > SESSION_MATCH_TOLERANCE:
            logger.warning(
                "No schedule match for OpenF1 session %s (%s %s)",
                session.get("session_key"),
                session.get("circuit_short_name"),
                session.get("session_name"),
            )
            continue

        folder = nearest[1]
        entries = fetch_team_radio(client.get, session["session_key"])
        if not entries:
            logger.info("%s: no team radio available", folder)
            continue
        save_session_radios(entries, folder, http=http)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logging.getLogger("fastf1").setLevel(logging.WARNING)

    import fastf1

    os.makedirs("cache", exist_ok=True)
    fastf1.Cache.enable_cache("cache")

    year = int(os.environ.get("TEAM_RADIO_YEAR", "2026"))
    download_season_radios(year)


if __name__ == "__main__":
    main()
