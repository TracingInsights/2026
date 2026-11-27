"""
Overtakes Extraction Script
===========================
Counts the on-track overtakes of every Race and Sprint of the configured
events from F1's live timing running order (overtakes_count.py) and writes:

    {event_name}/{session_name}/overtakes.json   (format: DATA_REFERENCE.md)

Only five small live-timing archive pages are requested per session, never
telemetry (about 1 MB):

    https://livetiming.formula1.com/static/<api_path>TimingData.jsonStream
    ... TrackStatus.jsonStream, RaceControlMessages.jsonStream,
        SessionInfo.json, DriverList.json

The archive requires an F1 TV subscription token since October 2026
(https://github.com/theOehrly/Fast-F1/issues/978). The pages are fetched
through FastF1's own request path (fastf1.req.Cache.requests_get), which
sends FastF1's stored token or signs in with the F1_USERNAME / F1_PASSWORD
environment variables (FastF1 >= 3.9). This script never reads, logs or
stores credentials itself. The pages are requested with the HTTP cache off,
so a re-run always counts the current archive. Raw pages are never written.

The output depends only on the streams: the same streams give a
byte-identical file, so re-running leaves git clean.

Backfill from streams already on disk (<dir>/<year>/<meeting>/<session>/
holding the five pages, plain or .gz; no network, no sign-in):

    python Overtakes.py --year 2025 --from-dir ../livetiming --out-dir ../2025
    python Overtakes.py --year 2025 --from-dir ../livetiming --out-dir ../2025 "Monaco Grand Prix"

Usage:
    python Overtakes.py                          # TARGET_EVENT_NAMES_LIST, this checkout
    python Overtakes.py "Las Vegas Grand Prix"   # only these events
    python Overtakes.py --year 2025 --out-dir ../2025 "Monaco Grand Prix"
"""

import argparse
import logging
import os
import sys
import time
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import overtakes_count  # noqa: E402

# ---------------------------------------------------------------------------
# Constants & Configuration
# ---------------------------------------------------------------------------

DEFAULT_YEAR = 2026
# Uncomment one or more events to process.
TARGET_EVENT_NAMES_LIST = [
    # "Australian Grand Prix",
    # "Chinese Grand Prix",
    # "Japanese Grand Prix",
    # "Miami Grand Prix",
    # "Monaco Grand Prix",
    # "Barcelona Grand Prix",
    # "Canadian Grand Prix",
    # "Austrian Grand Prix",
    # "British Grand Prix",
    # "Belgian Grand Prix",
    # "Hungarian Grand Prix",
    # "Bahrain Grand Prix",
    # "Saudi Arabian Grand Prix",
    # "Dutch Grand Prix",
    # "Italian Grand Prix",
    # "Spanish Grand Prix",
    # "Azerbaijan Grand Prix",
    # "Singapore Grand Prix",
    # "United States Grand Prix",
    # "Mexico City Grand Prix",
    # "São Paulo Grand Prix",
    "Las Vegas Grand Prix",
    # "Qatar Grand Prix",
    # "Abu Dhabi Grand Prix",
]
TARGET_EVENT_NAMES = [e.strip() for e in TARGET_EVENT_NAMES_LIST if e.strip()]
# Overtakes are only counted for these sessions; an event without a sprint
# is skipped for "Sprint".
TARGET_SESSIONS = [
    "Sprint",
    "Race",
]
OUTPUT_NAME = "overtakes.json"
PAGE_RETRIES = 3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("overtakes")
logging.getLogger("fastf1").setLevel(logging.WARNING)
logging.getLogger("fastf1").propagate = False


# ---------------------------------------------------------------------------
# Archive pages
# ---------------------------------------------------------------------------
def fetch_pages(api_path: str) -> Dict[str, bytes]:
    """The five archive pages of a session, through FastF1's authenticated requests."""
    from fastf1 import _api
    from fastf1.req import Cache

    pages: Dict[str, bytes] = {}
    with Cache.disabled():
        for page in overtakes_count.PAGES:
            url = _api.base_url + api_path + page
            for attempt in range(PAGE_RETRIES):
                resp = Cache.requests_get(url, headers=_api.headers, timeout=60)
                if resp.status_code == 200:
                    pages[page] = resp.content
                    break
                if resp.status_code in (401, 403):
                    raise RuntimeError(
                        f"{page}: HTTP {resp.status_code} from the live timing archive "
                        "(F1 TV sign-in missing or rejected; see FastF1 >= 3.9 docs)"
                    )
                time.sleep(2 * (attempt + 1))
            else:
                raise RuntimeError(f"{page}: HTTP {resp.status_code} after {PAGE_RETRIES} attempts")
    return pages


def session_api_path(year: int, event: str, session: str) -> Optional[str]:
    """FastF1's archive path of the session, or None when the event has no such session."""
    import fastf1

    try:
        f1session = fastf1.get_session(year, event, session)
    except ValueError:
        return None
    if f1session.event["EventName"] != event:
        raise ValueError(f"FastF1 resolved the event to {f1session.event['EventName']!r}")
    if f1session.name != session:
        return None
    return f1session.api_path


def archive_status(pages: Dict[str, bytes]) -> str:
    try:
        info = overtakes_count.read_json(pages["SessionInfo.json"])
        return str((info.get("ArchiveStatus") or {}).get("Status") or "")
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Counting and output
# ---------------------------------------------------------------------------
def write_session(pages: Dict[str, bytes], year: int, event: str, session: str, out_root: str) -> bool:
    label = f"{year} {event} - {session}"
    missing = [p for p in ("TimingData.jsonStream", "TrackStatus.jsonStream", "DriverList.json") if p not in pages]
    if missing:
        logger.error(f"{label}: missing pages {missing}; not written")
        return False
    status = archive_status(pages)
    if status and status != "Complete":
        logger.warning(f"{label}: archive status is {status!r}")
    payload = overtakes_count.count_session(pages, year, event, session)
    if payload is None:
        logger.warning(f"{label}: no laps or no running order in the stream; not written")
        return False
    out_path = os.path.join(out_root, event, session, OUTPUT_NAME)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(overtakes_count.dumps(payload))
    made = sum(payload["totals"]["made"])
    logger.info(f"{label}: {made} overtakes ({len(payload['events']['t'])} candidates) -> {out_path}")
    return True


def run_online(year: int, events: List[str], out_root: str, cache_dir: str) -> List[str]:
    import fastf1
    from fastf1.exceptions import RateLimitExceededError

    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)
    failed = []
    for event in events:
        for session in TARGET_SESSIONS:
            label = f"{year} {event} - {session}"
            while True:
                try:
                    api_path = session_api_path(year, event, session)
                    if api_path is None:
                        logger.info(f"{label}: no such session; skipped")
                        break
                    pages = fetch_pages(api_path)
                    if not write_session(pages, year, event, session, out_root):
                        failed.append(label)
                except RateLimitExceededError:
                    logger.info(f"{label}: FastF1 API limit (500 calls/h) reached, waiting 5 min")
                    time.sleep(300)
                    continue
                except Exception as e:
                    logger.error(f"{label}: {type(e).__name__}: {e}")
                    failed.append(label)
                break
    return failed


def run_from_dir(year: int, events: List[str], streams_root: str, out_root: str) -> List[str]:
    """Backfill: count every Race / Sprint folder of ``<streams_root>/<year>``.

    Meeting folders are named as in the archive's Index.json, which are the
    TracingInsights event folder names (checked for 2025 and 2026).
    """
    year_dir = os.path.join(streams_root, str(year))
    failed = []
    for event in sorted(os.listdir(year_dir)):
        if events and event not in events:
            continue
        for session in TARGET_SESSIONS:
            folder = os.path.join(year_dir, event, session)
            if not os.path.isdir(folder):
                continue
            try:
                pages = overtakes_count.read_dir_pages(folder)
                if not write_session(pages, year, event, session, out_root):
                    failed.append(f"{year} {event} - {session}")
            except Exception as e:
                logger.error(f"{year} {event} - {session}: {type(e).__name__}: {e}")
                failed.append(f"{year} {event} - {session}")
    return failed


def main() -> None:
    parser = argparse.ArgumentParser(description="Write overtakes.json for Race and Sprint sessions.")
    parser.add_argument("--year", type=int, default=DEFAULT_YEAR)
    parser.add_argument("--out-dir", default=".", help="season folder to write into (default: this checkout)")
    parser.add_argument("--from-dir", help="count local streams <dir>/<year>/<meeting>/<session>/ instead")
    parser.add_argument("--cache-dir", default="cache", help="FastF1 cache (schedule lookups)")
    parser.add_argument("events", nargs="*", help="only these events (default: TARGET_EVENT_NAMES_LIST; "
                                                  "with --from-dir: every meeting in the folder)")
    args = parser.parse_args()

    start = time.time()
    if args.from_dir:
        failed = run_from_dir(args.year, args.events, args.from_dir, args.out_dir)
    else:
        events = args.events or TARGET_EVENT_NAMES
        if not events:
            raise ValueError("Set at least one active event in TARGET_EVENT_NAMES_LIST.")
        failed = run_online(args.year, events, args.out_dir, args.cache_dir)
    logger.info(f"done in {time.time() - start:.1f}s, {len(failed)} failed")
    for label in failed:
        logger.info(f"  failed: {label}")


if __name__ == "__main__":
    main()
