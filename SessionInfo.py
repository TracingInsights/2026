"""
Session Info Backfill
=====================
Writes session_info.json (see utils.build_session_info) into the session
folders of a season repository that were extracted before the extraction
scripts wrote it.

Input:
    {event_name}/{session_name}/rcm.json  (marks an extracted session)

Output:
    {event_name}/{session_name}/session_info.json  (staged with git add)

Only the small live-timing streams are requested (SessionInfo, SessionStatus,
TrackStatus, LapCount: a few KB per session, through FastF1 and its F1
account sign-in), never telemetry: t0_date is LapStartDate - LapStartTime of
the session's laps, which is how FastF1 builds LapStartDate. A file is only
written when the archive path of the loaded streams is the session's own and,
for races and sprints, the session start equals the start of lap 1.

The repository can be this checkout or a partial clone of another season,
which needs no file contents (files are fetched on demand) and is small:

    git clone --depth 1 --filter=blob:none --sparse https://github.com/TracingInsights/2025.git ../2025
    python SessionInfo.py --year 2025 --repo ../2025
    git -C ../2025 status --short            # only "A ... session_info.json"
    git -C ../2025 commit -m "Add session_info.json"
    git -C ../2025 push

Usage:
    python SessionInfo.py                                   # this season, this checkout
    python SessionInfo.py --year 2025 --repo ../2025
    python SessionInfo.py --year 2025 --repo ../2025 "Monaco Grand Prix"   # only these events
    python SessionInfo.py --force ...                       # overwrite existing files
"""

import argparse
import logging
import os
import re
import subprocess
import time
from typing import Dict, List, Optional, Tuple

import fastf1
import numpy as np
import orjson
import pandas as pd
from fastf1.exceptions import RateLimitExceededError

from utils import build_session_info

DEFAULT_YEAR = 2026

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("session_info_backfill")
logging.getLogger("fastf1").setLevel(logging.WARNING)
logging.getLogger("fastf1").propagate = False

_SESSION_FILE_RE = re.compile(r"^([^/]+)/([^/]+)/rcm\.json$")
_LAPTIMES_RE = re.compile(r"^([^/]+)/([^/]+)/[^/]+/laptimes\.json$")
_NUMBERED_RE = re.compile(r"^(.+) (\d+)$")
_PRACTICE_RE = re.compile(r"^Practice (\d+)$")
# The live-timing archive keeps testing days under "Practice_N" in some
# seasons (2023) and "Day_N" in others (2025, 2026), not always the one
# FastF1 builds: both are tried.
_TESTING_PATH_RE = re.compile(r"_Practice_(\d+)/$")


def git(repo: str, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", repo, *args], check=True, capture_output=True
    ).stdout


def read_repo_json(repo: str, path: str):
    """A file of the repository: from disk if checked out, else from HEAD (fetched on demand)."""
    local = os.path.join(repo, path)
    if os.path.exists(local):
        with open(local, "rb") as f:
            return orjson.loads(f.read())
    return orjson.loads(git(repo, "show", f"HEAD:{path}"))


def repo_layout(
    repo: str,
) -> Tuple[set, List[Tuple[str, str]], Dict[Tuple[str, str], List[str]]]:
    """Tracked paths, extracted ``(event, session)`` folders and their driver laptimes."""
    names = git(repo, "ls-tree", "-r", "--name-only", "-z", "HEAD").decode().split("\0")
    sessions = []
    laptimes: Dict[Tuple[str, str], List[str]] = {}
    for name in names:
        match = _SESSION_FILE_RE.match(name)
        if match:
            sessions.append((match.group(1), match.group(2)))
            continue
        match = _LAPTIMES_RE.match(name)
        if match:
            laptimes.setdefault((match.group(1), match.group(2)), []).append(name)
    return set(names), sorted(sessions), laptimes


def laps_timing(repo: str, paths: List[str]) -> Tuple[Optional[pd.Timestamp], Optional[float]]:
    """``(t0_date, lap 1 start in seconds)`` from the first driver files that have them.

    t0_date is the median of LapStartDate - LapStartTime rounded to the
    millisecond (FastF1 sets LapStartDate = LapStartTime + t0_date).
    """
    t0 = lap1 = None
    for path in sorted(paths):
        laps = read_repo_json(repo, path)
        dates, starts = laps.get("lSD") or [], laps.get("lST") or []
        if t0 is None:
            offsets = [
                pd.Timestamp(d).value - round(s * 1e9)
                for d, s in zip(dates, starts)
                if isinstance(d, str) and d != "None" and isinstance(s, (int, float))
            ]
            if offsets:
                t0 = pd.Timestamp(round(float(np.median(offsets)) / 1e6) * 1_000_000)
        if lap1 is None:
            for lap, start in zip(laps.get("lap") or [], starts):
                if lap == 1 and isinstance(start, (int, float)):
                    lap1 = round(float(start), 3)
                    break
        if t0 is not None and lap1 is not None:
            break
    return t0, lap1


def testing_event_names(year: int) -> List[str]:
    """Testing event names of ``year`` in date order (test 1 first)."""
    schedule = fastf1.get_event_schedule(year, include_testing=True)
    testing = schedule[schedule.is_testing()].sort_values("EventDate", kind="stable")
    return [str(name) for name in testing["EventName"]]


def testing_number(testing_names: List[str], folder: str) -> Optional[int]:
    """Test number of a testing event folder ("Pre-Season Testing 2" in 2026 is test 2)."""
    if testing_names.count(folder) == 1:
        return testing_names.index(folder) + 1
    match = _NUMBERED_RE.match(folder)
    if match and match.group(1) in testing_names:
        return int(match.group(2))
    if testing_names.count(folder) > 1:
        return 1  # an unnumbered folder of same-named tests holds the first
    return None


def load_session(year: int, event: str, session: str, testing_names: List[str]):
    """The FastF1 session with only its status streams and session info loaded."""
    test = testing_number(testing_names, event)
    if test is not None:
        practice = _PRACTICE_RE.match(session)
        if practice is None:
            raise ValueError(f"unexpected testing session folder {session!r}")
        f1session = fastf1.get_testing_session(year, test, int(practice.group(1)))
        paths = [f1session.api_path, _TESTING_PATH_RE.sub(r"_Day_\1/", f1session.api_path)]
    else:
        f1session = fastf1.get_session(year, event, session)
        if f1session.event["EventName"] != event:
            raise ValueError(f"FastF1 resolved the event to {f1session.event['EventName']!r}")
        paths = [f1session.api_path]
    for path in dict.fromkeys(paths):
        f1session.api_path = path
        f1session._load_session_info()  # soft-fails: no _session_info on a wrong path
        if hasattr(f1session, "_session_info"):
            break
    f1session._load_session_status_data()
    f1session._load_total_lap_count()
    f1session._load_track_status_data()
    return f1session


def session_info_or_problem(
    year: int,
    repo: str,
    event: str,
    session: str,
    testing_names: List[str],
    laptimes: List[str],
) -> Tuple[Optional[dict], Optional[str]]:
    """``(session_info.json payload, None)``, or ``(None, why it is not written)``."""
    f1session = load_session(year, event, session, testing_names)
    t0, lap1 = laps_timing(repo, laptimes)
    f1session._t0_date = t0
    info = build_session_info(f1session)
    if info["session_info"] is None or not info["session_status"]["Time"]:
        return None, "no session info or session status loaded"
    # The payload's own Path can name the session differently from the path
    # it was served at (2021 sprints, rescheduled or single practice
    # sessions); the meeting must be the same.
    reported = str(info["session_info"].get("Path") or "")
    if reported.split("/")[:2] != f1session.api_path.removeprefix("/static/").split("/")[:2]:
        return None, f"archive path {reported!r} is not the session's meeting"
    if (
        session in f1session._RACE_LIKE_SESSIONS
        and lap1 is not None
        and lap1 != info["session_start_time"]
    ):
        return None, f"session start {info['session_start_time']} is not lap 1 start {lap1}"
    return info, None


def backfill(year: int, repo: str, events: Optional[List[str]] = None, force: bool = False):
    """Write and stage the missing session_info.json files of one season repository.

    FastF1 allows 500 API calls per hour in a process; at the limit this
    waits and retries, so several seasons run back to back in one process
    stay within it.
    """
    start = time.time()
    tracked, sessions, laptimes = repo_layout(repo)
    testing_names = testing_event_names(year)
    to_stage: List[str] = []
    written = skipped = 0
    failed: List[str] = []
    for event, session in sessions:
        if events and event not in events:
            continue
        label = f"{event} - {session}"
        rel_path = f"{event}/{session}/session_info.json"
        out_path = os.path.join(repo, rel_path)
        if (rel_path in tracked or os.path.exists(out_path)) and not force:
            skipped += 1
            if rel_path not in tracked:
                to_stage.append(rel_path)  # written by an interrupted run
            continue
        while True:
            try:
                info, problem = session_info_or_problem(
                    year, repo, event, session, testing_names, laptimes.get((event, session), [])
                )
            except RateLimitExceededError:
                logger.info(f"{label}: FastF1 API limit (500 calls/h) reached, waiting 5 min")
                time.sleep(300)
                continue
            except Exception as e:
                info, problem = None, f"{type(e).__name__}: {e}"
            break
        if problem:
            logger.error(f"{label}: {problem}; not written")
            failed.append(label)
            continue
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(orjson.dumps(info))
        logger.info(f"{label}: wrote {rel_path}")
        to_stage.append(rel_path)
        written += 1

    for i in range(0, len(to_stage), 200):
        git(repo, "add", "--sparse", "--", *to_stage[i : i + 200])
    logger.info(
        f"{year} done in {time.time() - start:.1f}s: {written} written, "
        f"{len(to_stage)} staged, {skipped} kept, {len(failed)} failed"
    )
    for label in failed:
        logger.info(f"  failed: {label}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Write session_info.json into a season repo.")
    parser.add_argument("--year", type=int, default=DEFAULT_YEAR)
    parser.add_argument("--repo", default=".", help="checkout or partial clone of the season")
    parser.add_argument("--force", action="store_true", help="overwrite existing files")
    parser.add_argument("events", nargs="*", help="only these event folders")
    args = parser.parse_args()

    os.makedirs("cache", exist_ok=True)
    fastf1.Cache.enable_cache("cache")
    backfill(args.year, args.repo, args.events, args.force)


if __name__ == "__main__":
    main()
