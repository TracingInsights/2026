"""
2026 Mexico City Grand Prix Race: R_fast.py on the repaired livetiming archive
=============================================================================
The archive for this session holds two recordings on one clock, started
Sat 2026-10-31 17:22:34 UTC:

* day 0 (``HH:MM:SS.mmm``): ~20 minutes of Saturday FP3 written into the Race
  folder (3 laps, TotalLaps 56, a non-existent car #236 "FM1")
* day 1 (``1.HH:MM:SS.mmm``): the actual 71-lap race

FastF1 reads a fixed 12-character timestamp, so it cannot parse the race
lines and only the Saturday data loads ("23 drivers, 44 total laps").

This script runs R_fast.py for that one session and rewrites the Race
``.jsonStream`` bodies in memory before FastF1 / fast_telemetry parse them:

* day-0 lines are dropped, except the ``00:00:00.000`` initial-state line of
  SessionInfo and TrackStatus (the race block has no snapshot of those; a
  normal archive starts with one)
* race lines are shifted back by RACE_OFFSET_MS onto a normal session clock
  (session start ~00:54:52, like other races)
* Saturday race control messages (Utc before race day) are removed from the
  race block's snapshot
* HIR's leftover Saturday tyre stint is removed from TimingAppData until F1
  reuses that stint index for HIR's real second stint
* DriverList details sent after FastF1's 60-minute cutoff (HIR's team, name
  and colour) are folded into the first race-day declaration

The HTTP cache keeps the raw archive responses; only the copies handed to the
parsers are rewritten. Livetiming is requested uncompressed: after the archive
was regenerated, CloudFront kept serving its cached gzip copy of the
Saturday-only files for up to an hour.

Checked against the session keyframes (TimingAppData.json, TimingData.json):
lap counts, every driver's lap-1 tyre and compound sequence match; per-lap
telemetry is identical to FastF1's get_telemetry() on the repaired data.

Usage (from the repository root, like R_fast.py):
    python MexicoRaceArchiveFix.py
"""

import copy
import json
import re

import fastf1._api
from fastf1.req import Cache

RACE_PATH = "/static/2026/2026-11-01_Mexico_City_Grand_Prix/2026-11-01_Race/"
EVENT_NAME = "Mexico City Grand Prix"
SESSION_NAME = "Race"
DAY_MS = 86_400_000
# Earliest race-day line is DriverList at 1.01:46:23.757; floor to the minute.
RACE_OFFSET_MS = DAY_MS + (1 * 3600 + 46 * 60) * 1000
KEEP_INITIAL_STATE = ("SessionInfo.jsonStream", "TrackStatus.jsonStream")
RACE_DAY_UTC = "2026-11-01"
# HIR (#50) entered race day with Saturday's second stint still at index 1;
# F1 kept writing his stint-0 best laps there until it reused index 1 for his
# real second stint (Compound UNKNOWN reset at raw 1.03:30:58.981, lap 37).
STALE_STINT_DRIVER = "50"
STALE_STINT_UNTIL_MS = DAY_MS + ((3 * 60 + 30) * 60 + 58) * 1000 + 981
# fastf1._api.driver_info() ignores DriverList lines after 60 min; HIR's team
# and name only arrive at raw 1.02:53:05.568 (01:07:05 after rebasing).
DRIVER_LIST_CUTOFF_MS = 60 * 60 * 1000
_TS = re.compile(r"^(?:(\d+)\.)?(\d\d):(\d\d):(\d\d)\.(\d{3})")


def _ms(match) -> int:
    d, h, m, s, ms = match.groups()
    return (((int(d or 0) * 24 + int(h)) * 60 + int(m)) * 60 + int(s)) * 1000 + int(ms)


def _fmt(ms: int) -> str:
    s, ms = divmod(ms, 1000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _drop_saturday_messages(payload: str):
    data = json.loads(payload)
    messages = data.get("Messages")
    if isinstance(messages, list):
        kept = [m for m in messages if m.get("Utc", RACE_DAY_UTC) >= RACE_DAY_UTC]
    elif isinstance(messages, dict):
        kept = {k: m for k, m in messages.items()
                if m.get("Utc", RACE_DAY_UTC) >= RACE_DAY_UTC}
    else:
        return payload
    if len(kept) == len(messages):
        return payload
    if not kept:
        return None
    data["Messages"] = kept
    return json.dumps(data, separators=(",", ":"), ensure_ascii=False)


def _drop_stale_stint(payload: str) -> str:
    data = json.loads(payload)
    driver = data.get("Lines", {}).get(STALE_STINT_DRIVER)
    stints = driver.get("Stints") if driver else None
    if isinstance(stints, list) and len(stints) > 1:
        driver["Stints"] = stints[:1]
    elif isinstance(stints, dict) and "1" in stints:
        stints.setdefault("0", {}).update(stints.pop("1"))
    else:
        return payload
    return json.dumps(data, separators=(",", ":"), ensure_ascii=False)


def _fold_late_driver_info(out: list) -> list:
    """Move late DriverList details into the first race-day declaration."""
    if not out:
        return out
    head_match = _TS.match(out[0])
    head = json.loads(out[0][head_match.end():])
    for record in out[1:]:
        match = _TS.match(record)
        if _ms(match) <= DRIVER_LIST_CUTOFF_MS:
            continue
        for drv_num, patch in json.loads(record[match.end():]).items():
            if drv_num in head and isinstance(patch, dict):
                head[drv_num].update(
                    {k: v for k, v in patch.items() if k != "Line"})
    return [head_match.group(0) + json.dumps(
        head, separators=(",", ":"), ensure_ascii=False)] + out[1:]


def repair_stream(page: str, content: bytes) -> bytes:
    """Rewrite one Race .jsonStream body onto a single-day session clock."""
    records = content.decode("utf-8-sig").split("\r\n")[:-1]
    out = []
    for record in records:
        match = _TS.match(record)
        if match is None:
            continue
        t = _ms(match)
        payload = record[match.end():]
        if t < DAY_MS:
            if t == 0 and page in KEEP_INITIAL_STATE:
                out.append(_fmt(0) + payload)
            continue
        if page == "TimingAppData.jsonStream" and t < STALE_STINT_UNTIL_MS:
            payload = _drop_stale_stint(payload)
        t -= RACE_OFFSET_MS
        if not 0 <= t < DAY_MS:
            raise ValueError(
                f"{page}: race line outside one day after rebasing: {record[:20]}"
            )
        if page == "RaceControlMessages.jsonStream":
            payload = _drop_saturday_messages(payload)
            if payload is None:
                continue
        out.append(_fmt(t) + payload)
    if page == "DriverList.jsonStream":
        out = _fold_late_driver_info(out)
    return ("\r\n".join(out) + "\r\n").encode("utf-8")


def install_archive_repair() -> None:
    """Route the Race .jsonStream responses through repair_stream()."""
    orig_requests_get = Cache.requests_get.__func__

    def requests_get(cls, url, **kwargs):
        response = orig_requests_get(cls, url, **kwargs)
        first = url if isinstance(url, str) else url[0]
        if (response.status_code == 200 and RACE_PATH in first
                and first.endswith(".jsonStream")):
            page = first.rsplit("/", 1)[-1]
            response = copy.copy(response)
            response._content = repair_stream(page, response.content)
        return response

    Cache.requests_get = classmethod(requests_get)
    fastf1._api.headers["Accept-Encoding"] = "identity"


def main():
    install_archive_repair()

    import R_fast

    R_fast.TARGET_EVENT_NAMES = [EVENT_NAME]
    R_fast.TARGET_SESSIONS = [SESSION_NAME]
    R_fast.main()


if __name__ == "__main__":
    main()
