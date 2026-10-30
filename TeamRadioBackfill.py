"""
Team Radio Backfill
===================
Downloads the team radio clips of every session of a season repository that
has no team_radio.json yet, from the official F1 live timing archive:

    https://livetiming.formula1.com/static/<session path>TeamRadio.jsonStream
    https://livetiming.formula1.com/static/<session path>TeamRadio/<clip>.mp3

The output is what team_radio.py writes (the 2023+ OpenF1 clips and the
2018-2020 archive clips share it):

    {event}/{session}/{driver}/radio/{clip}.mp3
    {event}/{session}/team_radio.json   columnar: drv, dNum, date, file, url

The session path comes from the session's session_info.json (or, without
one, from the season's Index.json; testing days are the "Practice N"
folders). Clips land in the folder of the driver's abbreviation from
drivers.json; a number missing there is looked up in the other sessions of
the event, then of the season, then read from the clip name ("ROBSHW01_97_..."
-> SHW) when the session has that driver. A session whose feed has no clips
gets no team_radio.json.

The archive requires an F1 TV subscription token since October 2026
(https://github.com/theOehrly/Fast-F1/issues/978): FastF1 >= 3.9.0 reuses
the stored token and signs in again with F1_USERNAME / F1_PASSWORD.

The repository can be this checkout or a partial clone of another season
(git clone --depth 1 --filter=blob:none --sparse ...). Written files are
staged unless --no-stage; --written appends their paths to a file. Clips
already on disk are not downloaded again, so an interrupted run resumes.

Usage:
    python TeamRadioBackfill.py                              # this season, this checkout
    python TeamRadioBackfill.py --year 2022 --repo ../2022
    python TeamRadioBackfill.py --year 2022 --repo ../2022 "Monaco Grand Prix"
"""

import argparse
import logging
import os
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import orjson
import requests

from Corners import Repo

DEFAULT_YEAR = 2026
LIVETIMING_STATIC = "https://livetiming.formula1.com/static/"
TEAM_RADIO_INDEX_KEYS = ("drv", "dNum", "date", "file", "url")
DOWNLOAD_WORKERS = 8
DOWNLOAD_RETRIES = 3
DOWNLOAD_TIMEOUT = 30

_SESSION_FILE_RE = re.compile(r"^([^/]+)/([^/]+)/([^/]+\.json)$")

logger = logging.getLogger("team_radio_backfill")


def livetiming_http_session() -> requests.Session:
    """requests session that sends FastF1's F1 TV token to livetiming only."""
    from fastf1.internals.f1account import (
        LIVETIMING_URL,
        LivetimingAuthAdapter,
        token_manager,
    )

    http = requests.Session()
    http.mount(LIVETIMING_URL, LivetimingAuthAdapter(token_manager))
    http.headers["User-Agent"] = "TracingInsights team radio backfill"
    return http


def _json_stream(text: str) -> List[Dict[str, Any]]:
    """Parse a livetiming .jsonStream: one ``HH:MM:SS.mmm{json}`` per line."""
    messages = []
    for line in text.splitlines():
        brace = line.find("{")
        if brace >= 0:
            try:
                messages.append(orjson.loads(line[brace:]))
            except orjson.JSONDecodeError:
                logger.debug("Skipping malformed stream line: %.80s", line)
    return messages


def fetch_captures(http: requests.Session, api_path: str) -> Optional[List[Dict[str, Any]]]:
    """The session's clips sorted by time ([] without a feed, None when unreadable)."""
    response = http.get(f"{LIVETIMING_STATIC}{api_path}TeamRadio.jsonStream", timeout=DOWNLOAD_TIMEOUT)
    if response.status_code in (403, 404):
        return []  # the archive answers a missing file with 403 (S3) or 404
    if response.status_code == 401:
        logger.error("Livetiming rejected the request (401): is the F1 TV token valid?")
        return None
    response.raise_for_status()
    captures: Dict[str, Dict[str, Any]] = {}
    for message in _json_stream(response.content.decode("utf-8-sig")):
        items = message.get("Captures") or []
        if isinstance(items, dict):  # the first message is a list, later ones {"<i>": {...}}
            items = list(items.values())
        for item in items:
            if isinstance(item, dict) and item.get("Path"):
                captures[item["Path"]] = item
    return sorted(captures.values(), key=lambda c: str(c.get("Utc") or ""))


def season_index_paths(http: requests.Session, year: int) -> Dict[Tuple[str, str], str]:
    """{(meeting name, session name): path} from the season's Index.json."""
    response = http.get(f"{LIVETIMING_STATIC}{year}/Index.json", timeout=DOWNLOAD_TIMEOUT)
    if response.status_code != 200:
        logger.warning("No Index.json for %s (HTTP %s)", year, response.status_code)
        return {}
    index = orjson.loads(response.content.decode("utf-8-sig"))
    paths = {}
    for meeting in index.get("Meetings", []):
        for session in meeting.get("Sessions", []):
            if session.get("Path"):
                paths[(meeting.get("Name", ""), session.get("Name", ""))] = session["Path"]
    return paths


def _download(http: requests.Session, url: str, dest: str) -> bool:
    if os.path.isfile(dest) and os.path.getsize(dest) > 0:
        return True
    tmp = dest + ".part"
    for attempt in range(DOWNLOAD_RETRIES):
        try:
            with http.get(url, timeout=DOWNLOAD_TIMEOUT, stream=True) as response:
                if response.status_code in (403, 404):
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
            logger.warning("Team radio download failed (%d/%d) %s: %s",
                           attempt + 1, DOWNLOAD_RETRIES, url, exc)
            if attempt < DOWNLOAD_RETRIES - 1:
                time.sleep(0.5 * (2**attempt))
    if os.path.exists(tmp):
        os.remove(tmp)
    return False


def _drivers(raw: Optional[bytes]) -> List[Dict[str, Any]]:
    try:
        drivers = orjson.loads(raw).get("drivers", []) if raw else []
    except (orjson.JSONDecodeError, AttributeError):
        return []
    return [d for d in drivers if isinstance(d, dict) and d.get("driver")]


def _driver_codes(raw: Optional[bytes]) -> Dict[str, str]:
    """Driver number -> abbreviation from one drivers.json (older layouts have no numbers)."""
    return {
        str(d["dn"]): str(d["driver"]) for d in _drivers(raw)
        if d.get("dn") not in (None, "", "None")
    }


def _code_from_clip(file_name: str, session_codes: set) -> Optional[str]:
    """The abbreviation in a clip name ("ROBSHW01_97_..." -> "SHW") if the session has it."""
    m = re.match(r"^[A-Z]{3}([A-Z]{3})\d\d_", file_name)
    return m.group(1) if m and m.group(1) in session_codes else None


def backfill(year: int, repo_path: str, events: Optional[List[str]] = None,
             stage: bool = True, written_list: Optional[str] = None) -> None:
    start = time.time()
    repo = Repo(repo_path)
    sessions: Dict[Tuple[str, str], set] = defaultdict(set)
    for name in repo.oids:
        m = _SESSION_FILE_RE.match(name)
        if m and not m.group(1).startswith("."):
            sessions[(m.group(1), m.group(2))].add(m.group(3))
    targets = sorted(
        k for k, files in sessions.items()
        if (not events or k[0] in events) and "drivers.json" in files
        and "team_radio.json" not in files
    )
    blobs = repo.read_many(
        [f"{e}/{s}/{f}" for (e, s), files in sessions.items()
         for f in ("drivers.json", "session_info.json") if f in files]
    )

    by_session = {k: _driver_codes(blobs.get(f"{k[0]}/{k[1]}/drivers.json")) for k in sessions}
    by_event: Dict[str, Dict[str, str]] = defaultdict(dict)
    votes: Dict[str, Counter] = defaultdict(Counter)
    for (e, _), codes in by_session.items():
        for number, code in codes.items():
            by_event[e].setdefault(number, code)
            votes[number][code] += 1
    by_season = {number: c.most_common(1)[0][0] for number, c in votes.items()}

    http = livetiming_http_session()
    index_paths: Optional[Dict[Tuple[str, str], str]] = None
    written: List[str] = []
    saved = no_feed = failed_sessions = clips_failed = 0

    for e, s in targets:
        label = f"{e}/{s}"
        base_dir = os.path.join(repo_path, e, s)
        info_raw = blobs.get(f"{label}/session_info.json")
        api_path = None
        if info_raw:
            api_path = ((orjson.loads(info_raw) or {}).get("session_info") or {}).get("Path")
        if not api_path:
            if index_paths is None:
                index_paths = season_index_paths(http, year)
            day = re.sub(r"^Practice (\d)$", r"Day \1", s)
            api_path = next((p for (meeting, name), p in index_paths.items()
                             if meeting.startswith(e) and name in (s, day)), None)
        if not api_path:
            logger.info("%s: no live timing session path", label)
            continue
        try:
            captures = fetch_captures(http, api_path)
        except requests.exceptions.RequestException as exc:
            logger.error("%s: could not read the team radio feed: %s", label, exc)
            failed_sessions += 1
            continue
        if captures is None:
            failed_sessions += 1
            continue
        if not captures:
            logger.info("%s: no team radio available", label)
            no_feed += 1
            continue

        codes = {**by_season, **by_event[e], **by_session[(e, s)]}
        session_codes = {str(d["driver"]) for d in _drivers(blobs.get(f"{label}/drivers.json"))}
        jobs = []
        for capture in captures:
            number = str(capture.get("RacingNumber") or "")
            file_name = os.path.basename(capture["Path"]) or "radio.mp3"
            code = codes.get(number) or _code_from_clip(file_name, session_codes) or number or "UNK"
            rel_path = f"{code}/radio/{file_name}"
            os.makedirs(os.path.join(base_dir, code, "radio"), exist_ok=True)
            jobs.append((capture, number, code, rel_path, f"{LIVETIMING_STATIC}{api_path}{capture['Path']}"))
        with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
            results = list(pool.map(lambda j: _download(http, j[4], os.path.join(base_dir, j[3])), jobs))

        index: Dict[str, list] = {k: [] for k in TEAM_RADIO_INDEX_KEYS}
        for (capture, number, code, rel_path, url), ok in zip(jobs, results):
            index["drv"].append(code)
            index["dNum"].append(number or "None")
            index["date"].append(str(capture.get("Utc") or "None"))
            index["file"].append(rel_path if ok else "None")
            index["url"].append(url)
            if ok:
                written.append(f"{label}/{rel_path}")
        with open(os.path.join(base_dir, "team_radio.json"), "wb") as f:
            f.write(orjson.dumps(index))
        written.append(f"{label}/team_radio.json")
        saved += 1
        clips_failed += results.count(False)
        unknown = sorted({c for c in index["drv"] if c.isdigit() or c == "UNK"})
        logger.info("%s: %d clips%s%s", label, results.count(True),
                    f", {results.count(False)} failed" if results.count(False) else "",
                    f", numbers without a driver code: {unknown}" if unknown else "")

    if stage:
        for i in range(0, len(written), 500):
            repo._git("add", "--sparse", "--", *written[i : i + 500])
    if written_list:
        with open(written_list, "a", encoding="utf-8") as f:
            f.writelines(f"{p}\n" for p in written)
    logger.info(
        f"{year} done in {time.time() - start:.0f}s: {saved} sessions saved "
        f"({len(written) - saved} clips, {clips_failed} failed), {no_feed} without radio, "
        f"{failed_sessions} unreadable, of {len(targets)} sessions without team_radio.json"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill team radio from the live timing archive.")
    parser.add_argument("--year", type=int, default=DEFAULT_YEAR)
    parser.add_argument("--repo", default=os.path.dirname(os.path.abspath(__file__)),
                        help="season repository (checkout or partial clone)")
    parser.add_argument("--no-stage", action="store_true", help="do not git add the written files")
    parser.add_argument("--written", help="append the written paths to this file")
    parser.add_argument("events", nargs="*", help="only these events")
    args = parser.parse_args()
    logging.getLogger("fastf1").setLevel(logging.WARNING)
    backfill(args.year, os.path.abspath(args.repo), args.events or None,
             stage=not args.no_stage, written_list=args.written)


if __name__ == "__main__":
    main()
