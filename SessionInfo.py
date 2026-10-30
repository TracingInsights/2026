"""
Session Info Backfill
=====================
Writes session_info.json (see utils.build_session_info) into session folders
extracted before the extraction scripts wrote it.

Input:
    {event_name}/{session_name}/rcm.json  (marks an extracted session)

Output:
    {event_name}/{session_name}/session_info.json

The FastF1 session is loaded with laps and telemetry (t0_date needs the car
and position data), served from the standalone cache when it is there.
Existing session_info.json files are kept unless --force is given.

Usage:
    python SessionInfo.py                          # every extracted session
    python SessionInfo.py "Monaco Grand Prix"      # only these events
    python SessionInfo.py --force "Monaco Grand Prix"
"""

import logging
import os
import re
import sys
import time
from typing import List, Optional, Tuple

import fastf1
import orjson

from utils import build_session_info

DEFAULT_YEAR = 2026
BASE_DIR = "."

# Testing sessions live under "Day_N" in the live-timing archive (as in
# preseason.py); Grand Prix practice sessions keep "Practice_N".
_original_make_path = fastf1._api.make_path


def _patched_make_path(wname, wdate, sname, sdate):
    path = _original_make_path(wname, wdate, sname, sdate)
    if "Testing" in wname and str(DEFAULT_YEAR) in wdate:
        for n in (1, 2, 3):
            if f"Practice_{n}" in path:
                return path.replace(f"Practice_{n}", f"Day_{n}")
    return path


fastf1._api.make_path = _patched_make_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("session_info_backfill")
logging.getLogger("fastf1").setLevel(logging.WARNING)
logging.getLogger("fastf1").propagate = False

# "Pre-Season Testing N" is test N; the unnumbered folder holds a copy of test 1.
_TESTING_RE = re.compile(r"^Pre-Season Testing(?: (\d+))?$")
_PRACTICE_RE = re.compile(r"^Practice (\d+)$")


def session_folders(events: Optional[List[str]]) -> List[Tuple[str, str]]:
    """``(event folder, session folder)`` pairs that hold an extracted session."""
    out = []
    for event in sorted(os.listdir(BASE_DIR)):
        if events and event not in events:
            continue
        event_dir = os.path.join(BASE_DIR, event)
        if not os.path.isdir(event_dir) or event.startswith("."):
            continue
        for session in sorted(os.listdir(event_dir)):
            if os.path.isfile(os.path.join(event_dir, session, "rcm.json")):
                out.append((event, session))
    return out


def load_session(event: str, session: str) -> fastf1.core.Session:
    testing = _TESTING_RE.match(event)
    if testing:
        practice = _PRACTICE_RE.match(session)
        if practice is None:
            raise ValueError(f"unexpected testing session folder {session!r}")
        f1session = fastf1.get_testing_session(
            DEFAULT_YEAR,
            int(testing.group(1) or 1),
            int(practice.group(1)),
            backend="fastf1",
        )
    else:
        f1session = fastf1.get_session(DEFAULT_YEAR, event, session)
    f1session.load(laps=True, telemetry=True, weather=False, messages=False)
    return f1session


def main() -> None:
    args = sys.argv[1:]
    force = "--force" in args
    events = [a for a in args if a != "--force"] or None

    os.makedirs("cache", exist_ok=True)
    fastf1.Cache.enable_cache("cache")

    start = time.time()
    written = skipped = failed = 0
    for event, session in session_folders(events):
        label = f"{event} - {session}"
        out_path = os.path.join(BASE_DIR, event, session, "session_info.json")
        if os.path.exists(out_path) and not force:
            skipped += 1
            continue
        try:
            info = build_session_info(load_session(event, session))
        except Exception as e:
            logger.error(f"{label}: {e}")
            failed += 1
            continue
        if not info["track_status"]["Time"] and not info["session_status"]["Time"]:
            logger.error(f"{label}: no track or session status loaded, not written")
            failed += 1
            continue
        with open(out_path, "wb") as f:
            f.write(orjson.dumps(info))
        logger.info(f"{label}: wrote {out_path}")
        written += 1

    logger.info(
        f"Done in {time.time() - start:.1f}s: {written} written, "
        f"{skipped} kept, {failed} failed"
    )


if __name__ == "__main__":
    main()
