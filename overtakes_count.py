"""
Overtake counter on F1's live-timing running order
==================================================
Counts on-track overtakes per driver of a Race or Sprint from the live
timing archive streams (no telemetry), and builds overtakes.json (see
DATA_REFERENCE.md). Used by Overtakes.py.

Input: the raw bytes of five archive pages of the session,

    TimingData.jsonStream, TrackStatus.jsonStream,
    RaceControlMessages.jsonStream, SessionInfo.json, DriverList.json

(RaceControlMessages and SessionInfo are not used by the count.)

Method: TimingData "Line" is F1's running order, re-ranked at every
mini-sector loop. Every pair of cars whose order flips between two
consecutive Line snapshots is a candidate pass by the car that moved ahead;
the start (and a red-flag standing restart) is the net change over its first
5 s. Candidates involving the pit lane, a neutralised track, retired or
stopped cars, timing flicker or jumps, and very slow cars are excluded; every
candidate is kept in the file with its exclusion code. The rules and
thresholds are the ones validated against F1's official per-driver overtake
counts (2026: per-driver MAE 0.65; blind 2025 test: 0.65, 1784 counted
against 1772 official). Changing anything that changes the output requires
bumping ALGO.
"""

import bisect
import collections
import io
import json
from typing import Dict, List, Optional

FORMAT = 1
ALGO = 1
SOURCE = "f1-livetiming"

PAGES = (
    "TimingData.jsonStream",
    "TrackStatus.jsonStream",
    "RaceControlMessages.jsonStream",
    "SessionInfo.json",
    "DriverList.json",
)

P = dict(
    GRID_GAP=150.0,     # s before first lap-1 completion: last snapshot before that is the grid
    L1_CUT=5.0,         # lap-1 net window ends at start + L1_CUT s (None: when the leader completes lap 1)
    L1_GAP=None,        # alternative: lap-1 net window ends at the first pause >= L1_GAP s in Line updates
    L1_BAD_LAPS=1.0,    # lap 1: drop pairs with a car that pits/stops before lap1_t + this * lap1 duration
    FLICKER=1.0,        # s: a swap reversed within this time cancels both events
    MAXJUMP=4,          # a single update moving A up >= MAXJUMP places or B down >= MAXJUMP: not a pass
    RET_T=180.0,        # s: either car Stopped/Retired within this time after the swap
    PIT_T=10.0,         # s: either car enters the pit lane within this time after the swap
    SLOWLAP=99.0,       # victim's current lap (no pit in it) >= SLOWLAP x field median lap -> very slow car
    SLOW_K=5,           # victim of >= SLOW_K otherwise-counted passes within +-SLOW_W s: very slow car
    SLOW_W=10.0,
    SC_STATUS=(4, 5, 6, 7),
    RESTART_NET=True,   # treat red-flag restarts like the start (net window of L1_CUT s)
)

# Exclusion code of each rule in overtakes.json ("x"); 0 = counted.
EXCLUSION_CODES = {
    None: 0,
    "pit": 1,
    "sc": 2,
    "l1sc": 2,          # lap-1 net window while the track is red-flagged
    "retflag": 3,
    "flicker": 4,
    "jump": 5,
    "retsoon": 6,
    "pitsoon": 7,
    "slowlap": 8,
    "slow": 9,
    "l1dns": 10,
    "l1pls": 10,
    "l1stop": 10,
    "l1pit": 10,
    "l1ret": 10,
}

# The loop that triggered a Line snapshot: the passer's loop crossing nearest
# to the snapshot, at most this many seconds before or after it. The Line
# re-rank usually arrives just before the passer's segment patch (2025+2026
# counted passes: 97 % of the passer's nearest crossings are 0-1 s after the
# snapshot, 2 % 0-1 s before it).
LOOP_WINDOW = 1.0


# ---------------------------------------------------------------------------
# Stream parsing
# ---------------------------------------------------------------------------
def tsec(ts):
    h, m, s = ts.split(':')
    return int(h) * 3600 + int(m) * 60 + float(s)


def read_stream(raw: bytes):
    """``[(seconds, patch)]`` of a .jsonStream page (``HH:MM:SS.mmm{json}`` lines)."""
    out = []
    for line in io.StringIO(raw.decode('utf-8-sig'), newline=None):
        line = line.strip().lstrip('﻿')
        if not line:
            continue
        if line[2] == ':':
            ts, data = line[:12], line[12:]
        else:
            p = line.index('.') + 1
            ts, data = line[p:p + 12], line[p + 12:]
        try:
            out.append((tsec(ts), json.loads(data)))
        except Exception:
            pass
    return out


def read_json(raw: bytes):
    return json.loads(raw.decode('utf-8-sig').strip().lstrip('﻿'))


class Step:
    def __init__(self):
        self.t, self.v = [], []

    def add(self, t, v):
        self.t.append(t)
        self.v.append(v)

    def at(self, t, default=None):
        i = bisect.bisect_right(self.t, t) - 1
        return self.v[i] if i >= 0 else default

    def intervals(self):
        out, on = [], None
        for t, v in zip(self.t, self.v):
            if v and on is None:
                on = t
            elif not v and on is not None:
                out.append((on, t))
                on = None
        if on is not None:
            out.append((on, 1e12))
        return out


def _items(x):
    """Index/value pairs of a list (full state) or an index-keyed dict (patch)."""
    if isinstance(x, list):
        return [(i, v) for i, v in enumerate(x)]
    if isinstance(x, dict):
        out = []
        for k, v in x.items():
            try:
                out.append((int(k), v))
            except (TypeError, ValueError):
                pass
        return out
    return []


def load_session(pages: Dict[str, bytes]):
    keys = ('Line', 'NumberOfLaps', 'InPit', 'PitOut', 'Retired', 'Stopped', 'GapToLeader')
    tl = collections.defaultdict(Step)
    line_groups = collections.OrderedDict()
    lap1_t = None
    seg = collections.defaultdict(dict)        # driver -> {(sector, segment): status}
    crossings = collections.defaultdict(list)  # driver -> [(t, sector, segment)]
    nseg = collections.Counter()               # sector -> loop count
    for t, js in read_stream(pages['TimingData.jsonStream']):
        lines = js.get('Lines') or {}
        if not isinstance(lines, dict):
            continue
        for drv, p in lines.items():
            if not isinstance(p, dict):
                continue
            for k in keys:
                if k in p:
                    tl[(drv, k)].add(t, p[k])
            if 'Line' in p:
                line_groups.setdefault(t, []).append((drv, p['Line']))
            if p.get('NumberOfLaps') == 1 and lap1_t is None:
                lap1_t = t
            if 'Sectors' in p:
                _track_segments(t, drv, p['Sectors'], seg[drv], crossings[drv], nseg)
    ts = []
    for t, js in read_stream(pages['TrackStatus.jsonStream']):
        try:
            ts.append((t, int(js.get('Status', 0))))
        except Exception:
            pass
    dl = read_json(pages['DriverList.json'])
    D = {k: v.get('Tla') for k, v in dl.items() if isinstance(v, dict) and v.get('Tla')}
    line, snaps = {}, []
    for t, es in line_groups.items():
        for drv, v in es:
            line[drv] = v
        snaps.append((t, [d for d, _ in sorted(line.items(), key=lambda x: x[1])]))
    loops = [nseg[i] for i in range(max(nseg) + 1)] if nseg else []
    for cr in crossings.values():
        cr.sort(key=lambda c: c[0])
    return dict(tl=tl, snaps=snaps, ts=ts, D=D, lap1_t=lap1_t, crossings=crossings, loops=loops)


def _track_segments(t, drv, sectors, state, out, nseg):
    """Record the mini-sector loop crossings of one TimingData patch of one car.

    A segment whose status goes from 0 to non-zero is a crossing. The first
    loop of the lap (sector 1, segment 0) usually keeps its status from lap
    to lap, so it is not re-sent; its crossing is the patch that resets the
    car's other segments to 0 (the new lap starts there).
    """
    reset = False
    first = None
    for si, sv in _items(sectors):
        if not isinstance(sv, dict) or 'Segments' not in sv:
            continue
        segs = _items(sv['Segments'])
        if isinstance(sv['Segments'], list):
            nseg[si] = max(nseg[si], len(sv['Segments']))
        for gi, gv in segs:
            nseg[si] = max(nseg[si], gi + 1)
            if not isinstance(gv, dict) or 'Status' not in gv:
                continue
            new = gv['Status'] or 0
            old = state.get((si, gi), 0)
            state[(si, gi)] = new
            if old == 0 and new != 0:
                out.append((t, si, gi))
                if (si, gi) == (0, 0):
                    first = True
            elif old != 0 and new == 0:
                reset = True
    if reset and not first and state.get((0, 0), 0) != 0:
        out.append((t, 0, 0))


def status_at(ts, t):
    st = 1
    for tt, s in ts:
        if tt <= t:
            st = s
        else:
            break
    return st


# ---------------------------------------------------------------------------
# Counting (validated rules; do not change without bumping ALGO)
# ---------------------------------------------------------------------------
def analyse(S, p=P):
    """Return (lap1_pairs, events). Each event has 'excl' = first exclusion rule hit or None."""
    tl, D, ts, lap1_t = S['tl'], S['D'], S['ts'], S['lap1_t']
    snaps = S['snaps']
    gi = 0
    for i, (t, o) in enumerate(snaps):
        if t < lap1_t - p['GRID_GAP']:
            gi = i
    snaps = snaps[gi:]
    start = snaps[1][0] if len(snaps) > 1 else lap1_t
    lap1dur = lap1_t - start
    lap1_ref = lap1_t
    if not 30.0 <= lap1dur <= 200.0:
        # e.g. rolling start behind the SC: laps were completed before the first Line change.
        # Use the median green lap time instead, and measure "lap 1" from the first Line change.
        durs = []
        for d in set(snaps[0][1]):
            st_ = tl[(d, 'NumberOfLaps')]
            le_ = [t for t, v in zip(st_.t, st_.v) if isinstance(v, int) and v >= 1 and t > start]
            durs += [b - a for a, b in zip(le_, le_[1:])]
        durs.sort()
        lap1dur = durs[len(durs) // 2] if durs else 90.0
        lap1_ref = start + lap1dur
    drivers = set(snaps[0][1])
    iv = {d: {k: tl[(d, k)].intervals() for k in ('InPit', 'PitOut', 'Retired', 'Stopped')} for d in drivers}

    # ---- lap 1: net inversions grid -> order at lap1_t
    g = snaps[0][1]
    if p.get('L1_GAP'):
        # end of the first burst of Line updates after the start (first pause >= L1_GAP s)
        tt = [t for t, o in snaps[1:]]
        tcut = tt[-1]
        for x, y in zip(tt, tt[1:]):
            if y - x >= p['L1_GAP']:
                tcut = x
                break
        tcut = min(tcut, lap1_ref)
    else:
        tcut = lap1_ref if p.get('L1_CUT') is None else start + p['L1_CUT']
    t1snap, o1 = [(t, o) for t, o in snaps if t <= tcut][-1]
    tbad = lap1_ref + p['L1_BAD_LAPS'] * lap1dur

    def l1bad(d):
        """Lap-1 exclusion category for car d (None = counts)."""
        pits = [(a, b) for a, b in iv[d]['InPit'] if b > start]
        stops = [a for k in ('Retired', 'Stopped') for a, b in iv[d][k] if b > start]
        if pits and pits[0][0] <= start:
            # in the pit lane at the start: never left -> DNS; left later -> pit-lane starter
            return 'dns' if pits[0][1] >= (min(stops) if stops else 1e12) or pits[0][1] > tbad else 'pls'
        if stops and min(stops) < lap1_ref:
            return 'stop'          # stopped / stalled on lap 1
        if pits and pits[0][0] < tbad:
            return 'pit'           # pits on lap 1 or before the leader completes lap 2
        if stops and min(stops) < tbad:
            return 'ret'           # retires right after lap 1
        return None

    lap1 = []
    pg = {d: i for i, d in enumerate(g)}
    po = {d: i for i, d in enumerate(o1)}
    for a in o1:
        for b in o1:
            if a in pg and b in pg and pg[a] > pg[b] and po[a] < po[b]:
                why = l1bad(b) or l1bad(a)
                if why is None and status_at(ts, lap1_t) in (5,):
                    why = 'sc'
                lap1.append(dict(t=tcut, tsnap=t1snap, A=a, B=b, a=D.get(a, a), b=D.get(b, b), lap1=True,
                                 excl=('l1' + why) if why else None))

    # ---- after lap 1: per-snapshot swaps
    # red-flag restarts: like the start, take the net change over the first L1_CUT s after
    # the first Line change following the end of a red flag (standing-restart cascade)
    rwin = []
    if p.get('RESTART_NET'):
        for (ta, sa), (tb, sb) in zip(ts, ts[1:]):
            if sa == 5 and sb != 5 and ta > start:
                nx = [t for t, o in snaps if t >= tb]
                if nx:
                    rwin.append((nx[0], nx[0] + (p['L1_CUT'] or 5.0)))

    def in_rwin(t):
        return any(a <= t < b for a, b in rwin)
    snaps2 = []
    for i, (t, o) in enumerate(snaps):
        if in_rwin(t) and i + 1 < len(snaps) and in_rwin(snaps[i + 1][0]):
            continue        # keep only the last snapshot inside each restart window
        snaps2.append((t, o))
    ev = []
    for (t0, s0), (t1, s1) in zip(snaps2, snaps2[1:]):
        if t1 <= tcut:
            continue
        p0 = {d: i for i, d in enumerate(s0)}
        p1 = {d: i for i, d in enumerate(s1)}
        for a in s1:
            for b in s1:
                if a != b and a in p0 and b in p0 and p0[a] > p0[b] and p1[a] < p1[b]:
                    ev.append(dict(t=t1, A=a, B=b, a=D.get(a, a), b=D.get(b, b), lap1=False,
                                   jump=p0[a] - p1[a], blose=p1[b] - p0[b], restart=in_rwin(t1)))
    last = {}
    for e in ev:
        k = frozenset((e['A'], e['B']))
        e['dtprev'] = e['t'] - last[k]['t'] if k in last else None
        e['dtnext'] = None
        if k in last:
            last[k]['dtnext'] = e['t'] - last[k]['t']
        last[k] = e

    def nxt(d, keys, t):
        best = 1e12
        for k in keys:
            for a, b in iv.get(d, {}).get(k, []):
                if a >= t:
                    best = min(best, a - t)
        return best

    # lap end times per driver (NumberOfLaps increments)
    lapend = {}
    for d in drivers:
        st = tl[(d, 'NumberOfLaps')]
        lapend[d] = [t for t, v in zip(st.t, st.v) if isinstance(v, int) and v >= 1 and t > start]
    allpit = {d: iv[d]['InPit'] for d in drivers}
    laps = []   # (end, dur, driver, haspit)
    for d in drivers:
        le = [start] + lapend[d]
        for a, b in zip(le, le[1:]):
            hp = any(x < b and y > a for x, y in allpit[d])
            laps.append((b, b - a, d, hp))
    laps.sort()
    lap_ends = [x[0] for x in laps]

    def field_median(t, win):
        i0 = bisect.bisect_left(lap_ends, t - win)
        i1 = bisect.bisect_right(lap_ends, t + win)
        ds = sorted(x[1] for x in laps[i0:i1] if not x[3])
        return ds[len(ds) // 2] if ds else None

    def lap_ratio(d, t):
        le = [start] + lapend[d]
        i = bisect.bisect_right(le, t)
        a = le[i - 1]
        if any(x < (le[i] if i < len(le) else 1e12) and y > a for x, y in allpit[d]):
            return 0.0      # in-lap / out-lap: handled by the pit rules
        if i >= len(le):
            # never completes this lap
            return 1e9
        b = le[i]
        med = field_median(b, max(lap1dur, 60.0))
        return (b - a) / med if med else 0.0

    for e in ev:
        t, A, B = e['t'], e['A'], e['B']
        excl = None
        flags = lambda d, k: bool(tl[(d, k)].at(t, False))  # noqa: E731
        if any(flags(d, 'InPit') or flags(d, 'PitOut') for d in (A, B)):
            excl = 'pit'
        elif status_at(ts, t) in p['SC_STATUS']:
            excl = 'sc'
        elif any(flags(d, 'Retired') or flags(d, 'Stopped') for d in (A, B)):
            excl = 'retflag'
        elif not e['restart'] and ((e['dtnext'] is not None and e['dtnext'] < p['FLICKER']) or
                                   (e['dtprev'] is not None and e['dtprev'] < p['FLICKER'])):
            excl = 'flicker'
        elif not e['restart'] and (e['jump'] >= p['MAXJUMP'] or e['blose'] >= p['MAXJUMP']):
            excl = 'jump'
        elif min(nxt(A, ('Retired', 'Stopped'), t), nxt(B, ('Retired', 'Stopped'), t)) < p['RET_T']:
            excl = 'retsoon'
        elif min(nxt(A, ('InPit',), t), nxt(B, ('InPit',), t)) < p['PIT_T']:
            excl = 'pitsoon'
        elif lap_ratio(B, t) >= p['SLOWLAP']:
            excl = 'slowlap'
        e['excl'] = excl
    # very slow car: victim of >= SLOW_K otherwise-counted passes within +-SLOW_W s
    if p['SLOW_K'] < 99:
        vt = collections.defaultdict(list)
        for e in ev:
            if e['excl'] is None:
                vt[e['B']].append(e['t'])
        for e in ev:
            if e['excl'] is None and not e['restart']:
                ts_ = vt[e['B']]
                n = bisect.bisect_right(ts_, e['t'] + p['SLOW_W']) - bisect.bisect_left(ts_, e['t'] - p['SLOW_W'])
                if n >= p['SLOW_K']:
                    e['excl'] = 'slow'
    return lap1, ev


# ---------------------------------------------------------------------------
# overtakes.json
# ---------------------------------------------------------------------------
def _loop_at(S, drv, t):
    """``(sector 1..3, loop index in the lap)`` of the car's crossing that triggered
    the snapshot at t, or ``(0, -1)`` when it has none within LOOP_WINDOW s of t."""
    cr = S['crossings'].get(drv) or []
    i = bisect.bisect_left(cr, (t - LOOP_WINDOW,))
    best = None
    while i < len(cr) and cr[i][0] <= t + LOOP_WINDOW:
        if best is None or abs(cr[i][0] - t) < abs(best[0] - t):
            best = cr[i]
        i += 1
    if best is None:
        return 0, -1
    _, si, gi = best
    return si + 1, sum(S['loops'][:si]) + gi


def _lap_at(S, drv, t):
    """Passer's lap in progress at t (1-based): completed laps + 1."""
    n = S['tl'][(drv, 'NumberOfLaps')].at(t, 0)
    return (n if isinstance(n, int) and not isinstance(n, bool) else 0) + 1


def count_session(pages: Dict[str, bytes], year: int, event: str, session: str) -> Optional[dict]:
    """overtakes.json payload of one session, or None when the streams have no
    lap or no running order (session not run)."""
    S = load_session(pages)
    if S['lap1_t'] is None or len(S['snaps']) < 2:
        return None
    lap1, ev = analyse(S)
    D = S['D']
    cars = {d for _, o in S['snaps'] for d in o}
    drivers = sorted(set(D.values()) | {D.get(d, d) for d in cars})
    idx = {c: i for i, c in enumerate(drivers)}

    rows = []
    for e in lap1:
        rows.append((round(e['tsnap'], 1), 1, idx[e['a']], idx[e['b']], 0, -1,
                     EXCLUSION_CODES[e['excl']]))
    for e in ev:
        s, m = _loop_at(S, e['A'], e['t'])
        rows.append((round(e['t'], 1), _lap_at(S, e['A'], e['t']), idx[e['a']], idx[e['b']], s, m,
                     EXCLUSION_CODES[e['excl']]))
    rows.sort(key=lambda r: (r[0], r[2], r[3], r[1], r[4], r[5], r[6]))

    made = [0] * len(drivers)
    lost = [0] * len(drivers)
    for r in rows:
        if r[6] == 0:
            made[r[2]] += 1
            lost[r[3]] += 1
    cols = list(zip(*rows)) if rows else [()] * 7
    return {
        "format": FORMAT,
        "algo": ALGO,
        "source": SOURCE,
        "year": int(year),
        "event": event,
        "session": session,
        "loops": S['loops'],
        "drivers": drivers,
        "events": {k: list(cols[i]) for i, k in enumerate(("t", "lap", "a", "b", "s", "m", "x"))},
        "totals": {"made": made, "lost": lost},
    }


def dumps(payload: dict) -> bytes:
    """Deterministic serialisation: keys in contract order, compact, UTF-8, trailing newline."""
    return (json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def made_by_driver(payload: dict) -> Dict[str, int]:
    return dict(zip(payload["drivers"], payload["totals"]["made"]))


def read_dir_pages(folder: str) -> Dict[str, bytes]:
    """The five pages of a session from a local streams folder (``<page>`` or ``<page>.gz``)."""
    import gzip
    import os

    pages = {}
    for page in PAGES:
        path = os.path.join(folder, page)
        if os.path.exists(path + ".gz"):
            with gzip.open(path + ".gz", "rb") as f:
                pages[page] = f.read()
        elif os.path.exists(path):
            with open(path, "rb") as f:
                pages[page] = f.read()
    return pages


__all__: List[str] = ["PAGES", "count_session", "dumps", "made_by_driver", "read_dir_pages"]
