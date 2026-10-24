"""Fast paths for FastF1 session loading and per-lap telemetry extraction.

Three independent accelerations, all of which fall back to stock FastF1
behaviour whenever the data does not look exactly like the common case:

1. :class:`FastLapTelemetry` re-implements ``Laps.get_telemetry()`` for a
   single lap with plain numpy. It mirrors FastF1 3.9 step by step (padded
   car/pos slices, driver-ahead distance integration over all drivers,
   three ``merge_channels``/``fill_missing`` passes with the same pandas
   interpolation rules and scipy quadratic splines, edge interpolation and
   the final slice) so that the resulting channels are numerically identical,
   but it avoids the thousands of DataFrame constructions per lap. One lap
   drops from ~650 ms to a few milliseconds.

2. :func:`install_fast_api_parsers` swaps FastF1's pure-Python parsers for
   the ``CarData.z`` and ``Position.z`` streams with versions that decode the
   records in forked worker processes with ``orjson`` and build typed numpy
   arrays directly. Output DataFrames are identical (same columns, dtypes and
   values), and the stage-2 pickle cache files keep their names.

3. :func:`prefetch_session_pages` downloads all livetiming pages a
   ``Session.load()`` call will need through FastF1's own cached HTTP session
   in parallel threads, so the subsequent sequential loader is served from the
   HTTP cache.

:func:`fork_map` is a small helper to run per-driver work in forked worker
processes that share the already loaded session copy-on-write.
"""

from __future__ import annotations

import base64
import concurrent.futures
import json as _stdlib_json
import logging
import multiprocessing
import os
import threading
import time
import zlib
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import orjson

    _json_loads = orjson.loads
except ImportError:  # pragma: no cover - orjson is in requirements.txt
    import json

    _json_loads = json.loads

logger = logging.getLogger("fast_telemetry")

_NS_PER_S = 1_000_000_000
_NS_PER_DAY = 86_400 * _NS_PER_S

_TELEMETRY_CHANNELS = (
    "Time", "Speed", "RPM", "Throttle", "nGear", "Brake", "DRS",
    "X", "Y", "Z", "Distance", "RelativeDistance",
    "DriverAhead", "DistanceToDriverAhead",
)


# ---------------------------------------------------------------------------
# Worker-count helpers
# ---------------------------------------------------------------------------
def default_workers() -> int:
    """Number of worker processes to use (``F1_EXTRACT_WORKERS`` overrides)."""
    env = os.environ.get("F1_EXTRACT_WORKERS")
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            logger.warning(f"Ignoring invalid F1_EXTRACT_WORKERS={env!r}")
    # CPUs this process may actually use (respects affinity / taskset),
    # not the host's total core count.
    if hasattr(os, "process_cpu_count"):
        count = os.process_cpu_count()
    elif hasattr(os, "sched_getaffinity"):
        count = len(os.sched_getaffinity(0))
    else:
        count = os.cpu_count()
    return max(1, count or 1)


def fork_map(func: Callable, items: Sequence, workers: int) -> list:
    """``list(map(func, items))`` using forked worker processes.

    Workers inherit the parent's memory copy-on-write, so large objects
    (e.g. a loaded FastF1 session) must simply exist in the parent before the
    call. ``func`` has to be importable by reference (module-level function).
    Raises if the pool breaks; callers decide how to fall back.
    """
    items = list(items)
    workers = max(1, min(workers, len(items)))
    if workers <= 1 or not hasattr(os, "fork"):
        return [func(item) for item in items]

    ctx = multiprocessing.get_context("fork")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers, mp_context=ctx
    ) as pool:
        return list(pool.map(func, items, chunksize=1))


# ---------------------------------------------------------------------------
# Numeric helpers that mirror pandas / FastF1 behaviour exactly
# ---------------------------------------------------------------------------
def _scalar_total_seconds(ns: int) -> float:
    """``pd.Timedelta(ns).total_seconds()`` (drops the sub-microsecond part)."""
    ns = int(ns)
    days, rem = divmod(ns, _NS_PER_DAY)
    seconds, rem = divmod(rem, _NS_PER_S)
    micros = rem // 1000
    return days * 86400 + seconds + micros / 1_000_000


def _integrate_distance(
    session_time_ns: np.ndarray, start_ns: int, speed: np.ndarray
) -> np.ndarray:
    """``Telemetry.integrate_distance()`` for a slice that starts at
    ``start_ns`` (i.e. ``Time = SessionTime - start``)."""
    time_ns = session_time_ns - start_ns
    ts = time_ns / _NS_PER_S  # Series.dt.total_seconds()
    dt = np.empty(ts.shape, dtype=np.float64)
    dt[1:] = ts[1:] - ts[:-1]  # Series.diff()
    dt[0] = _scalar_total_seconds(time_ns[0])  # Timedelta.total_seconds()
    return np.cumsum(speed / 3.6 * dt)


def _ffill_bfill_float(y: np.ndarray) -> np.ndarray:
    """``Series.ffill().bfill()`` for a float array with NaN gaps."""
    valid = ~np.isnan(y)
    if valid.all() or not valid.any():
        return y
    idx = np.where(valid, np.arange(len(y)), -1)
    np.maximum.accumulate(idx, out=idx)
    idx[idx < 0] = int(np.argmax(valid))
    return y[idx]


def _ffill_bfill_object(y: np.ndarray) -> np.ndarray:
    """``Series.ffill().bfill()`` for an object array with None/NaN gaps."""
    valid = ~pd.isna(y)
    if valid.all() or not valid.any():
        return y
    idx = np.where(valid, np.arange(len(y)), -1)
    np.maximum.accumulate(idx, out=idx)
    idx[idx < 0] = int(np.argmax(valid))
    return y[idx]


def _interp_index_both(x_i8: np.ndarray, y: np.ndarray) -> np.ndarray:
    """``Series.interpolate(method="index", limit_direction="both")`` on a
    DatetimeIndex given as int64 nanoseconds (sorted, unique)."""
    invalid = np.isnan(y)
    if not invalid.any():
        return y
    valid = ~invalid
    if not valid.any():
        return y
    out = y.copy()
    out[invalid] = np.interp(x_i8[invalid], x_i8[valid], y[valid])
    return out


def _interp_quadratic_forward(x_i8: np.ndarray, y: np.ndarray) -> np.ndarray:
    """``Series.interpolate(method="quadratic", fill_value="extrapolate")``
    (default ``limit_direction="forward"``: leading NaNs are preserved)."""
    invalid = np.isnan(y)
    if not invalid.any():
        return y
    valid = ~invalid
    if not valid.any():
        return y
    from scipy.interpolate import interp1d

    out = y.copy()
    terp = interp1d(
        x_i8[valid], y[valid], kind="quadratic",
        fill_value="extrapolate", bounds_error=False,
    )
    out[invalid] = terp(np.asarray(x_i8[invalid]))
    out[: int(np.argmax(valid))] = np.nan
    return out


def _i8(values: np.ndarray, kind: str) -> Optional[np.ndarray]:
    """View a datetime64/timedelta64 array as int64 ns; None if NaT present
    or the dtype is not the expected one."""
    if values.dtype.kind != kind:
        return None
    try:
        values = values.astype(f"{values.dtype.kind}8[ns]", copy=False)
    except (TypeError, ValueError):
        return None
    if np.isnat(values).any():
        return None
    return values.view("int64")


# ---------------------------------------------------------------------------
# Per-lap telemetry
# ---------------------------------------------------------------------------
class FastLapTelemetry:
    """Numpy re-implementation of ``Laps.get_telemetry()`` for single laps.

    Build one instance per loaded session (after any modification of
    ``session.laps``) and call :meth:`lap_arrays` per lap. ``None`` is
    returned whenever the lap does not satisfy the preconditions under which
    the fast path is known to be identical to FastF1; callers must then use
    ``Laps.get_telemetry()``.
    """

    def __init__(self, session):
        self.enabled = False
        self.session = session
        self.car: Dict[str, dict] = {}
        self.pos: Dict[str, dict] = {}
        self.laps: Optional[Dict[str, dict]] = {}
        self.drivers: List[str] = []
        self.t0 = 0
        # Objects the arrays were built from. The extraction scripts replace
        # ``session._laps`` when they add laps (also on later get_session()
        # cache hits), and FastF1's get_telemetry() always uses the current
        # table, so the lap arrays are rebuilt whenever it changes.
        self._laps_src = None
        self._car_src = None
        self._pos_src = None
        try:
            self.enabled = self._build()
        except Exception as exc:
            logger.warning(f"Fast telemetry disabled ({exc!r}); using FastF1")
            self.enabled = False
        if not self.enabled:
            logger.warning("Fast lap telemetry unavailable; using FastF1 path")

    # -- construction ------------------------------------------------------
    def _build(self) -> bool:
        import fastf1

        if fastf1.core.Telemetry.TELEMETRY_FREQUENCY != "original":
            return False
        session = self.session
        t0 = getattr(session, "t0_date", None)
        if t0 is None or pd.isna(t0):
            return False
        self.t0 = int(pd.Timestamp(t0).value)
        self.drivers = [str(d) for d in session.drivers]

        self._car_src = getattr(session, "_car_data", None)
        self._pos_src = getattr(session, "_pos_data", None)
        car_data = self._car_src or {}
        pos_data = self._pos_src or {}
        if not car_data or not pos_data:
            return False
        for drv, tel in car_data.items():
            arrs = self._car_arrays(tel)
            if arrs is None:
                return False
            self.car[str(drv)] = arrs
        for drv, tel in pos_data.items():
            arrs = self._pos_arrays(tel)
            if arrs is None:
                return False
            self.pos[str(drv)] = arrs
        self._laps_src = getattr(session, "_laps", None)
        self.laps = self._lap_arrays(self._laps_src)
        return self.laps is not None

    def _sync_with_session(self) -> bool:
        """Make the arrays match the session's current data; False if the
        fast path cannot be used for it."""
        session = self.session
        if (getattr(session, "_car_data", None) is not self._car_src
                or getattr(session, "_pos_data", None) is not self._pos_src):
            # telemetry was reloaded: rebuild everything
            self.car, self.pos, self.laps = {}, {}, {}
            try:
                self.enabled = self._build()
            except Exception as exc:
                logger.warning(f"Fast telemetry disabled ({exc!r})")
                self.enabled = False
            return self.enabled
        laps_src = getattr(session, "_laps", None)
        if laps_src is not self._laps_src:
            self._laps_src = laps_src
            self.laps = self._lap_arrays(laps_src)
        return self.laps is not None

    def _time_arrays(self, tel) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if "Date" not in tel.columns or "SessionTime" not in tel.columns:
            return None
        date = _i8(tel["Date"].to_numpy(), "M")
        st = _i8(tel["SessionTime"].to_numpy(), "m")
        if date is None or st is None or len(date) == 0:
            return None
        # searchsorted on SessionTime and merging on Date require order
        if not (np.all(np.diff(st) >= 0) and np.all(np.diff(date) > 0)):
            return None
        return date, st

    def _car_arrays(self, tel) -> Optional[dict]:
        times = self._time_arrays(tel)
        if times is None:
            return None
        needed = ("Speed", "RPM", "Throttle", "nGear", "Brake", "DRS")
        if any(c not in tel.columns for c in needed):
            return None
        date, st = times
        try:
            return {
                "date": date,
                "st": st,
                "Speed": tel["Speed"].to_numpy(dtype=np.float64),
                "RPM": tel["RPM"].to_numpy(dtype=np.float64),
                "Throttle": tel["Throttle"].to_numpy(dtype=np.float64),
                "nGear": tel["nGear"].to_numpy(dtype=np.int64),
                "Brake": tel["Brake"].to_numpy(dtype=bool),
                "DRS": tel["DRS"].to_numpy(dtype=np.int64),
            }
        except (TypeError, ValueError):
            return None

    def _pos_arrays(self, tel) -> Optional[dict]:
        times = self._time_arrays(tel)
        if times is None:
            return None
        if any(c not in tel.columns for c in ("X", "Y", "Z")):
            return None
        date, st = times
        try:
            return {
                "date": date,
                "st": st,
                "X": tel["X"].to_numpy(dtype=np.float64),
                "Y": tel["Y"].to_numpy(dtype=np.float64),
                "Z": tel["Z"].to_numpy(dtype=np.float64),
            }
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _lap_arrays(laps) -> Optional[Dict[str, dict]]:
        needed = ("DriverNumber", "LapNumber", "LapStartTime", "Time")
        if laps is None or any(c not in laps.columns for c in needed):
            return None
        drv_col = laps["DriverNumber"].to_numpy()
        num = laps["LapNumber"].to_numpy(dtype=np.float64)
        start = laps["LapStartTime"].to_numpy()
        end = laps["Time"].to_numpy()
        if start.dtype.kind != "m" or end.dtype.kind != "m":
            return None
        start = start.astype("timedelta64[ns]", copy=False)
        end = end.astype("timedelta64[ns]", copy=False)
        start_ok = ~np.isnat(start)
        end_ok = ~np.isnat(end)
        start_i8 = start.view("int64")
        end_i8 = end.view("int64")
        out: Dict[str, dict] = {}
        for drv in pd.unique(drv_col):
            idx = np.flatnonzero(drv_col == drv)  # DataFrame order
            out[str(drv)] = {
                "num": num[idx],
                "start": start_i8[idx],
                "start_ok": start_ok[idx],
                "end": end_i8[idx],
                "end_ok": end_ok[idx],
            }
        return out

    # -- public API --------------------------------------------------------
    def lap_arrays(self, driver_number, lap_start, lap_end) -> Optional[dict]:
        """Telemetry channels for one lap as numpy arrays, or None.

        ``lap_start``/``lap_end`` are the lap's ``LapStartTime`` and ``Time``
        (``pd.Timedelta`` or int nanoseconds).
        """
        if not self.enabled:
            return None
        try:
            if not self._sync_with_session():
                return None
            start_ns = _td_ns(lap_start)
            end_ns = _td_ns(lap_end)
            if start_ns is None or end_ns is None:
                return None
            return self._lap_arrays_impl(str(driver_number), start_ns, end_ns)
        except Exception as exc:
            logger.debug(
                f"Fast telemetry failed for driver {driver_number} "
                f"({lap_start} - {lap_end}): {exc!r}"
            )
            return None

    # -- implementation ----------------------------------------------------
    def _lap_arrays_impl(self, drv: str, S: int, E: int) -> Optional[dict]:
        if E <= S:
            # FastF1 would merge two edge rows with the same date here
            return None
        car = self.car.get(drv)
        pos = self.pos.get(drv)
        if car is None or pos is None:
            return None
        cst = car["st"]
        pst = pos["st"]

        # --- Laps.get_car_data / get_pos_data with pad=1 on both sides ----
        a = int(np.searchsorted(cst, S, "left"))
        b = int(np.searchsorted(cst, E, "right"))
        if b <= a:
            return None
        qa0 = int(np.searchsorted(pst, S, "left"))
        qb0 = int(np.searchsorted(pst, E, "right"))
        if qb0 <= qa0:
            return None
        n_car = len(cst)
        n_pos = len(pst)
        pa, pb = max(0, a - 1), min(n_car, b + 1)
        qa, qb = max(0, qa0 - 1), min(n_pos, qb0 + 1)
        if pb - pa < 3:  # car_data.iloc[1:-1] would be empty
            return None
        # merge_channels() requires a non-zero 'Time' sample in the base data
        if not np.any(cst[pa:pb] != S) or not np.any(pst[qa:qb] != S):
            return None

        # --- car_data.iloc[1:-1].add_driver_ahead() -----------------------
        t_start = int(cst[pa + 1])
        t_end = int(cst[pb - 2])
        ahead_res = self._driver_ahead(drv, t_start, t_end, cst[pa + 1:pb - 1])
        if ahead_res is None:
            return None
        ahead_u, dist_u = ahead_res

        # --- car_data.add_distance().add_relative_distance() --------------
        speed_p = car["Speed"][pa:pb]
        dist_p = _integrate_distance(cst[pa:pb], S, speed_p)
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_p = dist_p / dist_p[-1]  # NaN/inf like pandas, without warning

        # --- car_data.merge_channels(drv_ahead): fill the two padded rows -
        n_p = pb - pa
        cdate = car["date"][pa:pb]
        ahead_p = np.empty(n_p, dtype=object)
        ahead_p[1:-1] = ahead_u
        ahead_p[0] = ahead_u[0]      # bfill
        ahead_p[-1] = ahead_u[-1]    # ffill
        dist_ahead_p = np.full(n_p, np.nan)
        dist_ahead_p[1:-1] = dist_u
        dist_ahead_p = _interp_index_both(cdate, dist_ahead_p)

        # --- pos_data.merge_channels(car_data) ----------------------------
        pdate = pos["date"][qa:qb]
        dates = np.union1d(pdate, cdate)
        m = len(dates)
        ip = np.searchsorted(dates, pdate)
        ic = np.searchsorted(dates, cdate)

        def scatter(idx, vals):
            out = np.full(m, np.nan)
            out[idx] = vals
            return out

        x = _interp_quadratic_forward(dates, scatter(ip, pos["X"][qa:qb]))
        y = _interp_quadratic_forward(dates, scatter(ip, pos["Y"][qa:qb]))
        z = _interp_quadratic_forward(dates, scatter(ip, pos["Z"][qa:qb]))
        speed = _interp_index_both(dates, scatter(ic, speed_p))
        rpm = _interp_index_both(dates, scatter(ic, car["RPM"][pa:pb]))
        throttle = _interp_index_both(dates, scatter(ic, car["Throttle"][pa:pb]))
        ngear = _ffill_bfill_float(scatter(ic, car["nGear"][pa:pb]))
        drs = _ffill_bfill_float(scatter(ic, car["DRS"][pa:pb]))
        brake = _ffill_bfill_float(scatter(ic, car["Brake"][pa:pb]))
        dist = _interp_quadratic_forward(dates, scatter(ic, dist_p))
        rel = _interp_quadratic_forward(dates, scatter(ic, rel_p))
        dist_ahead = _interp_index_both(dates, scatter(ic, dist_ahead_p))
        ahead = np.full(m, None, dtype=object)
        ahead[ic] = ahead_p
        ahead = _ffill_bfill_object(ahead)

        # --- slice_by_lap(interpolate_edges=True): add edge samples -------
        sd = S + self.t0
        ed = E + self.t0
        new_dates = [d for d in (sd, ed) if not _contains_sorted(dates, d)]
        if new_dates:
            dates2 = np.union1d(dates, np.asarray(new_dates, dtype=np.int64))
            old = np.searchsorted(dates2, dates)
            m2 = len(dates2)

            def expand(arr):
                out = np.full(m2, np.nan)
                out[old] = arr
                return out

            x = _interp_quadratic_forward(dates2, expand(x))
            y = _interp_quadratic_forward(dates2, expand(y))
            z = _interp_quadratic_forward(dates2, expand(z))
            speed = _interp_index_both(dates2, expand(speed))
            rpm = _interp_index_both(dates2, expand(rpm))
            throttle = _interp_index_both(dates2, expand(throttle))
            ngear = _ffill_bfill_float(expand(ngear))
            drs = _ffill_bfill_float(expand(drs))
            brake = _ffill_bfill_float(expand(brake))
            dist = _interp_quadratic_forward(dates2, expand(dist))
            rel = _interp_quadratic_forward(dates2, expand(rel))
            dist_ahead = _interp_index_both(dates2, expand(dist_ahead))
            ahead2 = np.full(m2, None, dtype=object)
            ahead2[old] = ahead
            ahead = _ffill_bfill_object(ahead2)
            dates = dates2

        # --- final slice: LapStartTime <= SessionTime <= Time -------------
        lo = int(np.searchsorted(dates, sd, "left"))
        hi = int(np.searchsorted(dates, ed, "right"))
        if hi <= lo:
            return None
        sl = slice(lo, hi)
        ngear_s, drs_s, brake_s = ngear[sl], drs[sl], brake[sl]
        if (np.isnan(ngear_s).any() or np.isnan(drs_s).any()
                or np.isnan(brake_s).any()):
            return None  # FastF1 would fail to restore the integer dtypes
        time_ns = (dates[sl] - self.t0) - S
        return {
            "Time": time_ns.astype("timedelta64[ns]"),
            "Speed": speed[sl],
            "RPM": rpm[sl],
            "Throttle": throttle[sl],
            "nGear": ngear_s.astype(np.int64),
            "Brake": brake_s.astype(bool),
            "DRS": drs_s.astype(np.int64),
            "X": x[sl],
            "Y": y[sl],
            "Z": z[sl],
            "Distance": dist[sl],
            "RelativeDistance": rel[sl],
            "DriverAhead": ahead[sl],
            "DistanceToDriverAhead": dist_ahead[sl],
        }

    def _driver_ahead(
        self, drv: str, t_start: int, t_end: int, own_u_st: np.ndarray
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """``Telemetry.calculate_driver_ahead()`` for the unpadded car slice
        of ``drv`` spanning ``[t_start, t_end]``."""
        own = self.laps.get(drv)
        if own is None:
            return None
        sel = own["start_ok"] & (own["start"] <= t_start)
        if not sel.any():
            return None
        first_lap_number = own["num"][np.flatnonzero(sel)[-1]]

        cols: List[str] = []
        times: List[np.ndarray] = []
        dists: List[np.ndarray] = []
        for d2 in self.drivers:
            car2 = self.car.get(d2)
            laps2 = self.laps.get(d2)
            if car2 is None or laps2 is None or len(laps2["num"]) == 0:
                continue
            num = laps2["num"]
            sel_b = laps2["start_ok"] & (laps2["start"] <= t_start)
            if sel_b.any():
                lap_n_before = num[np.flatnonzero(sel_b)[-1]]
                if lap_n_before < first_lap_number:
                    lap_n_before += 1
            else:
                lap_n_before = _builtin_min(num)
            sel_a = laps2["end_ok"] & (laps2["end"] >= t_end)
            if sel_a.any():
                lap_n_after = num[np.flatnonzero(sel_a)[0]]
            else:
                lap_n_after = _builtin_max(num)

            pad_before = pad_after = 0
            relevant = None
            while True:
                relevant = np.flatnonzero(
                    (num >= lap_n_before - pad_before)
                    & (num <= lap_n_after + pad_after)
                )
                if pad_before >= 1 or pad_after >= 1:
                    break
                if relevant.size == 0:
                    break
                if not laps2["start_ok"][relevant[-1]]:
                    pad_before += 1
                    continue
                if not laps2["end_ok"][relevant[0]]:
                    pad_after += 1
                    continue
                break
            if relevant is None or relevant.size == 0:
                continue

            # slice_by_lap(relevant_laps)
            if relevant.size > 1:
                starts = laps2["start"][relevant][laps2["start_ok"][relevant]]
                ends = laps2["end"][relevant][laps2["end_ok"][relevant]]
                if starts.size == 0 or ends.size == 0:
                    continue
                start = int(starts.min())
                end = int(ends.max())
            else:
                i = relevant[0]
                if not (laps2["start_ok"][i] and laps2["end_ok"][i]):
                    continue
                start = int(laps2["start"][i])
                end = int(laps2["end"][i])
            st2 = car2["st"]
            s0 = int(np.searchsorted(st2, start, "left"))
            s1 = int(np.searchsorted(st2, end, "right"))
            if s1 <= s0:
                continue
            seg_st = st2[s0:s1]
            seg_dist = _integrate_distance(seg_st, start, car2["Speed"][s0:s1])
            # slice_by_time(t_start, t_end)
            r0 = int(np.searchsorted(seg_st, t_start, "left"))
            r1 = int(np.searchsorted(seg_st, t_end, "right"))
            if r1 <= r0:
                continue
            cols.append(d2)
            times.append(seg_st[r0:r1])
            dists.append(seg_dist[r0:r1])

        if drv not in cols or len(cols) < 2:
            return None
        all_t = np.unique(np.concatenate(times))
        j_own = cols.index(drv)
        # FastF1 joins the result back onto the reference lap by position;
        # that only works when every timestamp belongs to the own driver.
        if (all_t.shape != own_u_st.shape or not np.array_equal(all_t, own_u_st)
                or not np.array_equal(times[j_own], own_u_st)):
            return None

        combined = np.full((len(all_t), len(cols)), np.nan)
        for j, (t, dd) in enumerate(zip(times, dists)):
            combined[np.searchsorted(all_t, t), j] = dd

        other_mask = np.ones(len(cols), dtype=bool)
        other_mask[j_own] = False
        drv_map = np.array(cols, dtype=object)[other_mask]
        own_dst = combined[:, j_own]
        other_dst = combined[:, other_mask]
        other_dst[
            np.diff(other_dst, n=1, axis=0, prepend=other_dst[0:1, :]) == 0
        ] = np.nan
        delta = other_dst - own_dst[:, None]
        delta[np.isnan(delta)] = np.inf
        delta[delta < 0] = np.inf
        index_ahead = np.argmin(delta, axis=1)
        all_inf = np.all(delta == np.inf, axis=1)
        drv_ahead = drv_map[index_ahead]
        drv_ahead[all_inf] = ""
        dist_ahead = delta[np.arange(len(index_ahead)), index_ahead]
        dist_ahead[all_inf] = np.nan
        return drv_ahead, dist_ahead


def _builtin_min(values: np.ndarray):
    """Python's ``min()`` over a float array, as FastF1 calls it on a
    LapNumber Series: a leading NaN wins, later NaNs are ignored."""
    return values[0] if np.isnan(values[0]) else np.nanmin(values)


def _builtin_max(values: np.ndarray):
    """Python's ``max()`` over a float array (see ``_builtin_min``)."""
    return values[0] if np.isnan(values[0]) else np.nanmax(values)


def _td_ns(value) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if pd.isna(value):
        return None
    return int(pd.Timedelta(value).value)


def _contains_sorted(arr: np.ndarray, value: int) -> bool:
    i = int(np.searchsorted(arr, value, "left"))
    return i < len(arr) and arr[i] == value


def telemetry_frame_to_arrays(telemetry) -> dict:
    """Extract the channels used by the extraction scripts from a FastF1
    ``Telemetry`` DataFrame (the fallback path)."""
    n = len(telemetry)
    out = {}
    for col in _TELEMETRY_CHANNELS[:-3]:
        out[col] = telemetry[col].to_numpy()
    # optional channels, handled exactly like the original scripts did
    out["RelativeDistance"] = (
        telemetry["RelativeDistance"].to_numpy()
        if "RelativeDistance" in telemetry.columns
        else np.full(n, np.nan, dtype=np.float64)
    )
    out["DriverAhead"] = (
        telemetry["DriverAhead"].to_numpy()
        if "DriverAhead" in telemetry.columns
        else np.full(n, np.nan)
    )
    out["DistanceToDriverAhead"] = (
        telemetry["DistanceToDriverAhead"].to_numpy()
        if "DistanceToDriverAhead" in telemetry.columns
        else np.full(n, np.nan, dtype=np.float64)
    )
    return out


# ---------------------------------------------------------------------------
# Fast CarData.z / Position.z parsers
# ---------------------------------------------------------------------------
_PARSE_RECORDS: Optional[list] = None  # shared with forked workers
_PARSE_MAX_WORKERS = 8
_PARSE_MIN_RECORDS_PER_WORKER = 500
_TS_LENGTH = 12  # len('00:00:00:000')

_CAR_COLUMNS = ["Time", "Date", "RPM", "Speed", "nGear", "Throttle", "Brake",
                "DRS", "Source"]
_POS_COLUMNS = ["Time", "Date", "Status", "X", "Y", "Z", "Source"]


def _loads_like_stdlib(text: str):
    """``json.loads`` semantics with orjson speed: orjson rejects a few
    inputs the standard library accepts (NaN/Infinity literals, integers
    beyond 64 bit, lone surrogates), so retry those with ``json.loads``."""
    try:
        return _json_loads(text)
    except Exception:
        return _stdlib_json.loads(text)


def _parse_zipped_record(text: str) -> dict:
    """``fastf1._api.parse(text, zipped=True)``; raises on anything that is
    not a JSON object (the original raises later when it is indexed)."""
    if text[0] == "{":
        return _loads_like_stdlib(text)
    if text[0] == '"':
        text = text.strip('"')
    raw = zlib.decompress(base64.b64decode(text), -zlib.MAX_WBITS)
    decoded = raw.decode("utf-8-sig")
    if decoded[0] != "{":
        raise ValueError("record is not a JSON object")
    return _loads_like_stdlib(decoded)


def _parse_car_chunk(args) -> tuple:
    i0, i1 = args
    from fastf1.internals.parsing_helpers import to_datetime, to_timedelta

    records = _PARSE_RECORDS[i0:i1]
    rec_times: list = []
    entry_dates: list = []
    data: Dict[str, list] = {}
    errors = 0
    for record in records:
        try:
            time = to_timedelta(record[:_TS_LENGTH])
            jrecord = _parse_zipped_record(record[_TS_LENGTH:])
            ti = len(rec_times)
            rec_times.append(time)
            for entry in jrecord["Entries"]:
                ei = len(entry_dates)
                entry_dates.append(to_datetime(entry["Utc"]))
                cars = entry["Cars"]
                for drv in cars:
                    lists = data.get(drv)
                    if lists is None:
                        lists = data[drv] = [[] for _ in range(8)]
                    try:
                        channels = cars[drv]["Channels"]
                        rpm = channels["0"]
                        speed = channels["2"]
                        ngear = channels["3"]
                        throttle = channels["4"]
                        brake = channels["5"]
                        drs = channels.get("45", 0)
                    except KeyError:
                        continue
                    lists[0].append(ti)
                    lists[1].append(ei)
                    lists[2].append(rpm)
                    lists[3].append(speed)
                    lists[4].append(ngear)
                    lists[5].append(throttle)
                    lists[6].append(brake)
                    lists[7].append(drs)
        except Exception:
            errors += 1
            continue

    times_arr = np.array(rec_times, dtype="timedelta64[ns]")
    dates_arr = np.array(entry_dates, dtype="datetime64[ns]")
    out = {}
    for drv, lists in data.items():
        out[drv] = (
            times_arr[np.asarray(lists[0], dtype=np.intp)],
            dates_arr[np.asarray(lists[1], dtype=np.intp)],
            *[np.asarray(lst).astype(np.int64) for lst in lists[2:]],
        )
    return errors, out


def _parse_pos_chunk(args) -> tuple:
    i0, i1 = args
    from fastf1.internals.parsing_helpers import to_datetime, to_timedelta

    records = _PARSE_RECORDS[i0:i1]
    rec_times: list = []
    sample_dates: list = []
    data: Dict[str, list] = {}
    errors = 0
    for record in records:
        try:
            time = to_timedelta(record[:_TS_LENGTH])
            jrecord = _parse_zipped_record(record[_TS_LENGTH:])
            ti = len(rec_times)
            rec_times.append(time)
            for sample in jrecord["Position"]:
                ei = len(sample_dates)
                sample_dates.append(to_datetime(sample["Timestamp"]))
                entries = sample["Entries"]
                for drv in entries:
                    lists = data.get(drv)
                    if lists is None:
                        lists = data[drv] = [[] for _ in range(6)]
                    try:
                        x = entries[drv]["X"]
                        y = entries[drv]["Y"]
                        z = entries[drv]["Z"]
                    except KeyError:
                        continue
                    try:
                        status = entries[drv]["Status"]
                    except KeyError:
                        status = None
                    if str(status).isdigit():
                        status = "OffTrack" if int(status) else "OnTrack"
                    lists[0].append(ti)
                    lists[1].append(ei)
                    lists[2].append(status)
                    lists[3].append(x)
                    lists[4].append(y)
                    lists[5].append(z)
        except Exception:
            errors += 1
            continue

    times_arr = np.array(rec_times, dtype="timedelta64[ns]")
    dates_arr = np.array(sample_dates, dtype="datetime64[ns]")
    out = {}
    for drv, lists in data.items():
        out[drv] = (
            times_arr[np.asarray(lists[0], dtype=np.intp)],
            dates_arr[np.asarray(lists[1], dtype=np.intp)],
            np.array(lists[2], dtype=object),
            *[np.asarray(lst).astype(np.int64) for lst in lists[3:]],
        )
    return errors, out


def _parse_records(records: list, chunk_func: Callable) -> tuple:
    """Parse all records, in forked workers when there are enough of them."""
    global _PARSE_RECORDS
    n = len(records)
    workers = min(default_workers(), _PARSE_MAX_WORKERS,
                  max(1, n // _PARSE_MIN_RECORDS_PER_WORKER))
    _PARSE_RECORDS = records
    try:
        bounds = np.linspace(0, n, workers + 1).astype(int)
        chunks = [(int(bounds[i]), int(bounds[i + 1])) for i in range(workers)]
        if workers > 1:
            try:
                results = fork_map(chunk_func, chunks, workers)
            except Exception as exc:
                logger.warning(f"Parallel stream parsing failed ({exc!r}); "
                               "parsing in-process")
                results = [chunk_func((0, n))]
        else:
            results = [chunk_func((0, n))]
    finally:
        _PARSE_RECORDS = None

    errors = sum(r[0] for r in results)
    merged: Dict[str, list] = {}
    for _, part in results:
        for drv, arrays in part.items():
            merged.setdefault(drv, []).append(arrays)
    data = {}
    for drv, parts in merged.items():
        if len(parts) == 1:
            data[drv] = parts[0]
        else:
            data[drv] = tuple(
                np.concatenate([p[k] for p in parts]) for k in range(len(parts[0]))
            )
    return errors, data


def _fast_car_data(path, response=None, livedata=None):
    import fastf1._api as _api
    from fastf1.internals.pandas_extensions import create_df_fast

    original = _ORIGINAL_API["car_data"]
    if livedata is not None:
        return original(path, response=response, livedata=livedata)
    if response is None:
        _api._logger.info("Fetching car data...")
        response = _api.fetch_page(path, "car_data")
        if response is None:
            raise _api.SessionNotAvailableError(
                "No data for this session! If this session only finished "
                "recently, please try again in a few minutes."
            )
    _api._logger.info("Parsing car data...")
    try:
        errors, parsed = _parse_records(response, _parse_car_chunk)
    except Exception as exc:
        logger.warning(f"Fast car data parser failed ({exc!r}); using FastF1")
        return original(path, response=response, livedata=livedata)
    if errors > 0:
        _api._logger.warning(f"Car data: failed to decode {errors} "
                             f"messages ({len(response)} messages total)")

    lengths = {len(arrays[0]) for arrays in parsed.values()}
    if len(lengths) > 1:
        # incomplete data for some driver: let FastF1 do its merge logic
        logger.info("Car data is incomplete for some drivers; using FastF1 "
                    "parser")
        return original(path, response=response, livedata=livedata)

    data = {}
    for drv, arrays in parsed.items():
        time, date, rpm, speed, ngear, throttle, brake, drs = arrays
        source = np.full(len(time), "car", dtype=object)
        df = create_df_fast(
            arrays=[time, date, rpm, speed, ngear, throttle, brake, drs, source],
            columns=_CAR_COLUMNS,
        )
        unique_brake = np.unique(brake)
        if ((unique_brake > 0) & (unique_brake < 100)).any():
            _api._logger.warning(f"Driver {drv: >2}: Raw brake data contains "
                                 f"non-boolean values!")
        df["Brake"] = df["Brake"].astype("bool")
        data[drv] = df
    return data


def _fast_position_data(path, response=None, livedata=None):
    import fastf1._api as _api
    from fastf1.internals.pandas_extensions import create_df_fast

    original = _ORIGINAL_API["position_data"]
    if livedata is not None:
        return original(path, response=response, livedata=livedata)
    if response is None:
        _api._logger.info("Fetching position data...")
        response = _api.fetch_page(path, "position")
        if response is None:
            raise _api.SessionNotAvailableError(
                "No data for this session! If this session only finished "
                "recently, please try again in a few minutes."
            )
    _api._logger.info("Parsing position data...")
    if not response:
        return {}
    try:
        errors, parsed = _parse_records(response, _parse_pos_chunk)
    except Exception as exc:
        logger.warning(f"Fast position parser failed ({exc!r}); using FastF1")
        return original(path, response=response, livedata=livedata)
    if errors > 0:
        _api._logger.warning(f"Position data: failed to decode {errors} "
                             f"messages ({len(response)} messages total)")

    lengths = {len(arrays[0]) for arrays in parsed.values()}
    if len(lengths) > 1:
        logger.info("Position data is incomplete for some drivers; using "
                    "FastF1 parser")
        return original(path, response=response, livedata=livedata)

    data = {}
    for drv, arrays in parsed.items():
        time, date, status, x, y, z = arrays
        source = np.full(len(time), "pos", dtype=object)
        data[drv] = create_df_fast(
            arrays=[time, date, status, x, y, z, source],
            columns=_POS_COLUMNS,
        )
    return data


_ORIGINAL_API: Dict[str, Callable] = {}


def install_fast_api_parsers() -> None:
    """Replace ``fastf1._api.car_data``/``position_data`` with the fast
    parsers (keeping FastF1's stage-2 pickle cache wrapper and file names)."""
    import fastf1._api as _api
    from fastf1.req import Cache

    if _ORIGINAL_API:
        return
    for name, impl in (("car_data", _fast_car_data),
                       ("position_data", _fast_position_data)):
        wrapped = getattr(_api, name)
        _ORIGINAL_API[name] = getattr(wrapped, "__wrapped__", wrapped)
        impl.__name__ = name  # cache file name must stay '<name>.ff1pkl'
        impl.__qualname__ = name
        setattr(_api, name, Cache.api_request_wrapper(impl))
    logger.info("Installed fast CarData/Position parsers")


# ---------------------------------------------------------------------------
# Parallel HTTP prefetch
# ---------------------------------------------------------------------------
_PAGE_TO_API_FUNC = {
    "session_info": "session_info",
    "driver_list": "driver_info",
    "session_status": "session_status_data",
    "track_status": "track_status_data",
    "timing_data": "_extended_timing_data",
    "timing_app_data": "timing_app_data",
    "car_data": "car_data",
    "position": "position_data",
    "weather_data": "weather_data",
    "race_control_messages": "race_control_messages",
    "lap_count": "lap_count",
}
# Rough download size order (small -> large). The first request runs alone on
# the main thread, so it should be a small page; the large streams are then
# started first in the thread pool so that their downloads overlap.
_PAGE_SIZE_RANK = {
    "session_info": 0,
    "driver_list": 1,
    "session_status": 2,
    "track_status": 3,
    "lap_count": 4,
    "weather_data": 5,
    "race_control_messages": 6,
    "timing_app_data": 7,
    "timing_data": 8,
    "position": 9,
    "car_data": 10,
}
_PREFETCH_MIN_INTERVAL = 0.25  # match FastF1's own per-request rate limit


def prefetch_session_pages(
    f1session,
    *,
    laps: bool = True,
    telemetry: bool = True,
    weather: bool = True,
    messages: bool = True,
    max_workers: int = 4,
    gate_on_timing_data: bool = False,
) -> bool:
    """Fetch every livetiming page ``Session.load()`` will request, in
    parallel, through FastF1's cached HTTP session, so that the following
    (sequential) ``Session.load()`` is served from the HTTP cache.

    Pages whose parsed result is already in FastF1's pickle cache are
    skipped. Nothing is done when FastF1's HTTP cache is not active, because
    the prefetched pages could not be reused and would be downloaded twice.

    With ``gate_on_timing_data=True`` the TimingData page is requested first.
    If the server answers it with an HTTP error, the session's lap data is
    not published yet: ``Session.load()`` cannot produce any laps without
    that page, so nothing else is downloaded and ``False`` is returned. This
    keeps a polling availability check from requesting every page twice.

    Returns ``True`` in every other case; errors never propagate.
    """
    try:
        import fastf1._api as _api
        from fastf1.req import Cache

        if not getattr(f1session, "f1_api_support", False):
            return True
        if (getattr(Cache, "_requests_session_cached", None) is None
                or getattr(Cache, "_http_tmp_disabled", False)):
            return True
        path = f1session.api_path
        names = ["session_info", "driver_list"]
        if laps:
            names += ["session_status", "track_status", "timing_data",
                      "timing_app_data"]
            race_like = getattr(f1session, "_RACE_LIKE_SESSIONS", ())
            if f1session.name in race_like:
                names.append("lap_count")
        if telemetry:
            names += ["car_data", "position"]
        if weather:
            names.append("weather_data")
        if messages:
            names.append("race_control_messages")

        pending = []
        for name in names:
            try:
                cache_dir = getattr(Cache, "_CACHE_DIR", None)
                if cache_dir and os.path.isfile(
                    Cache._get_cache_file_path(path, _PAGE_TO_API_FUNC[name])
                ):
                    continue
            except Exception:
                pass
            pending.append(name)
        if not pending:
            return True

        lock = threading.Lock()
        last_start = [0.0]

        def fetch(name: str) -> Optional[int]:
            """HTTP status of the final response, None if the request
            raised."""
            page = _api.pages[name]
            urls = (_api.base_url + path + page,
                    _api.base_url_mirror + path + page)
            with lock:  # keep request starts spaced like FastF1's limiter
                wait = _PREFETCH_MIN_INTERVAL - (time.time() - last_start[0])
                if wait > 0:
                    time.sleep(wait)
                last_start[0] = time.time()
            try:
                resp = Cache.requests_get(urls, headers=_api.headers)
                logger.debug(f"Prefetched {name}: HTTP {resp.status_code}")
                return resp.status_code
            except Exception as exc:
                logger.debug(f"Prefetch of {name} failed: {exc!r}")
                return None

        t0 = time.time()
        gate = gate_on_timing_data and "timing_data" in pending
        first = ("timing_data" if gate
                 else min(pending, key=lambda n: _PAGE_SIZE_RANK[n]))
        # The first request runs on the main thread: it initialises the
        # cached HTTP session and, if needed, the F1 account sign-in before
        # any worker thread uses them.
        status = fetch(first)
        if gate and status is not None and status >= 400:
            logger.info(f"TimingData not published yet (HTTP {status})")
            return False
        rest = sorted((n for n in pending if n != first),
                      key=lambda n: _PAGE_SIZE_RANK[n], reverse=True)
        if rest:
            workers = max(1, min(max_workers, len(rest)))
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=workers
            ) as pool:
                list(pool.map(fetch, rest))
        logger.info(f"Prefetched {len(pending)} livetiming pages in "
                    f"{time.time() - t0:.1f}s")
        return True
    except Exception as exc:
        logger.debug(f"Prefetch skipped: {exc!r}")
        return True
