"""Timestamp-gated practice and archived weather features for cota_predictor_v2.

No race-day weather observations are used. Model initialization is NOT its public
availability time: allow eight hours for ECMWF computation/distribution. This is
a conservative assumption, not an independently verified publication log.
"""
import concurrent.futures
import datetime as dt
import gzip
import hashlib
import json
import os
import tempfile
import time
import threading
import warnings
import zlib
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import Ridge

import cota_predictor_improved as base

UTC = dt.timezone.utc
ROOT = Path('.f1_cache/v2')
PRACTICE_FEATURES = ['practice_gap', 'practice_degradation', 'practice_laps',
                     'team_practice_gap', 'sprint_gap', 'sprint_position']
WEATHER_FEATURES = ['forecast_temp', 'forecast_rain', 'forecast_wind', 'forecast_humidity']
SCHEMA = 1
WEATHER_DOWNLOAD = threading.Semaphore(1)


def timestamp(date, clock=None):
    """Parse a calendar date and optional race clock as an aware UTC timestamp."""
    return dt.datetime.fromisoformat(date + 'T' + (clock or '13:00:00Z')).astimezone(UTC)


def cutoff_for(race, mode):
    """Return the fixed historical prediction cutoff for the requested stage."""
    start = timestamp(race['date'], race.get('time'))
    if mode == 'pre':
        return start - dt.timedelta(hours=108)  # Wednesday morning for a Sunday afternoon race
    if mode == 'practice':
        q = race.get('Qualifying')
        return timestamp(q['date'], q.get('time')) - dt.timedelta(hours=1) if q else start-dt.timedelta(hours=26)
    return start - dt.timedelta(hours=2)


def forecast_run(cutoff):
    """Choose the latest six-hour initialization with an eight-hour availability buffer."""
    available = cutoff - dt.timedelta(hours=8)
    return available.replace(hour=(available.hour // 6) * 6, minute=0, second=0, microsecond=0)


def _read_cache(path):
    """Read a complete gzip JSON cache envelope or reject its structure."""
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        saved = json.load(stream)
    if not isinstance(saved, dict) or 'data' not in saved or saved.get('status') != 'ok':
        raise ValueError('Invalid cache envelope')
    return saved['data'], saved['status']


def _atomic_cache_write(path, envelope):
    """Replace a cache only after its unique same-directory gzip file is complete."""
    temporary = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode='w+b', dir=path.parent,
                                         prefix=f'.{path.name}.', suffix='.tmp', delete=False) as binary:
            temporary = Path(binary.name)
            with gzip.open(binary, 'wt', encoding='utf-8') as stream:
                json.dump(envelope, stream)
            binary.flush()
            os.fsync(binary.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError as exc:
                warnings.warn(f'Could not remove temporary cache file: {type(exc).__name__}: {exc}',
                              RuntimeWarning, stacklevel=2)


def cached_get(url, params=None, live=False, offline=False):
    """Return cached or downloaded JSON, recovering from corrupt cache files.

    Offline reads never access the network and report missing, corrupt, or
    unreadable caches. Successful downloads remain usable if caching fails;
    a warning reports the write failure while an older cache is left intact.
    """
    key = hashlib.sha256((url + json.dumps(params or {}, sort_keys=True)).encode()).hexdigest()
    path = ROOT / (key + '.json.gz')
    cache_status = 'not cached'
    try:
        cached_stat = path.stat()
        if offline or not live or time.time()-cached_stat.st_mtime < 3600:
            try:
                return _read_cache(path)
            except (EOFError, ValueError, gzip.BadGzipFile, zlib.error) as exc:
                cache_status = f'corrupt cache: {type(exc).__name__}'
            except OSError as exc:
                cache_status = f'unreadable cache: {type(exc).__name__}'
    except FileNotFoundError:
        pass
    except OSError as exc:
        cache_status = f'unreadable cache: {type(exc).__name__}'
    if offline:
        return None, cache_status
    try:
        # An independent request per worker avoids mutating a shared Session.
        for attempt in range(3):
            try:
                if 'open-meteo.com' in url:
                    with WEATHER_DOWNLOAD:
                        response = requests.get(url, params=params, timeout=35)
                        time.sleep(.2)
                else:
                    response = requests.get(url, params=params, timeout=35)
            except requests.RequestException:
                if attempt == 2:
                    raise
                time.sleep(1 + 2**attempt)
                continue
            if response.status_code not in (429, 500, 502, 503, 504):
                break
            time.sleep(1 + 2**attempt)
        if response.status_code != 200:
            return None, f'HTTP {response.status_code}: {response.text[:120]}'
        data = response.json()
        try:
            _atomic_cache_write(path, dict(data=data, status='ok', url=response.url,
                                          retrieved_utc=dt.datetime.now(UTC).isoformat()))
        except (OSError, TypeError, ValueError, zlib.error) as exc:
            warnings.warn(f'Cache write failed: {type(exc).__name__}: {exc}', RuntimeWarning, stacklevel=2)
        return data, 'ok'
    except (requests.RequestException, ValueError) as exc:
        return None, f'{type(exc).__name__}: {str(exc)[:100]}'


def practice_summary(data, cutoff):
    """Estimate relative long-run pace; fuel loads and traffic remain unobserved.

    Require >=5 clean consecutive laps in a stint. Adjust tyre compound, tyre age,
    and 15-minute session windows with regularized regression. Reject wet laps,
    in/out laps, deleted laps, and laps >3% slower than that driver's stint median.
    Unknown weather or accuracy flags are retained and counted in the audit.
    """
    n = len(data.get('lap', []))
    if not n:
        return {}, {'clean_laps': 0}
    df = pd.DataFrame({k: v for k, v in data.items() if isinstance(v, list) and len(v) == n})
    required = {'drv', 'time', 'lap', 'life', 'stint', 'compound', 'lSD', 'status', 'pin', 'pout'}
    if not required <= set(df):
        return {}, {'missing_columns': sorted(required-set(df))}
    for col in ['time', 'lap', 'life', 'stint', 'sesT', 'pin', 'pout', 'pos']:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    end = pd.to_datetime(df.lSD, errors='coerce', utc=True) + pd.to_timedelta(df.time, unit='s')
    clean = (end <= pd.Timestamp(cutoff)) & df.time.between(40, 200) & df.life.between(1, 30)
    clean &= df.status.astype(str).eq('1') & df.pin.isna() & df.pout.isna()
    clean &= df.compound.isin(['SOFT', 'MEDIUM', 'HARD'])
    for col in ['del', 'wR']:
        if col in df:
            clean &= ~df[col].eq(True)
    if 'iacc' in df:
        clean &= ~df.iacc.eq(False)
    g = df.loc[clean].sort_values(['drv', 'stint', 'lap']).copy()
    if g.empty:
        return {}, {'clean_laps': 0}
    # Removing bad laps can split a stint: never call isolated fast laps a long run.
    med = g.groupby(['drv', 'stint']).time.transform('median')
    g = g[g.time <= med * 1.03].copy()
    new_run = g.groupby(['drv', 'stint']).lap.diff().ne(1)
    g['run'] = new_run.cumsum()
    g = g[g.groupby('run').time.transform('size') >= 5].copy()
    if len(g) < 15 or g.drv.nunique() < 3:
        return {}, {'clean_laps': len(g), 'drivers': int(g.drv.nunique())}
    # Fit only within this already completed session, no race result labels.
    minutes = g.sesT if 'sesT' in g else (pd.to_datetime(g.lSD, utc=True).dt.as_unit('ns').astype('int64') / 1e9)
    design = pd.get_dummies(pd.DataFrame(dict(compound=g.compound,
                                window=(minutes // 900).astype(str))), dtype=float)
    design['tyre_age'] = g.life.astype(float)
    adjustment = Ridge(alpha=10).fit(design, g.time)
    g['residual'] = g.time - adjustment.predict(design)
    pace = g.groupby('drv').residual.median()
    reference = float(pace.min())
    lap_ref = float(g.time.median())
    summary = {}
    for driver, laps in g.groupby('drv'):
        degradation = []
        for _, stint in laps.groupby('run'):
            # Raw within-stint slope; tyre wear and fuel burn are confounded.
            if stint.life.nunique() >= 5:
                degradation.append(float(np.polyfit(stint.life, stint.time, 1)[0]))
        summary[driver] = dict(practice_gap=float((pace[driver]-reference)/lap_ref*100),
                               practice_laps=int(len(laps)),
                               practice_degradation=float(np.median(degradation)) if degradation else None)
    return summary, dict(clean_laps=len(g), drivers=len(summary),
                         last_lap_end=str(end.loc[g.index].max()), fuel_load_known=False,
                         traffic_measured=False)


def session_positions(data, cutoff):
    """Return each driver's last recorded position from laps completed by the cutoff."""
    required = ['drv', 'lap', 'pos', 'lSD', 'time']
    if not all(c in data for c in required):
        return {}
    raw = pd.DataFrame({c: data[c] for c in required})
    for col in ['lap', 'pos', 'time']:
        raw[col] = pd.to_numeric(raw[col], errors='coerce')
    end = pd.to_datetime(raw.lSD, errors='coerce', utc=True) + pd.to_timedelta(raw.time, unit='s')
    raw = raw.loc[end <= pd.Timestamp(cutoff)].dropna(subset=['lap', 'pos']).sort_values('lap')
    return {code: float(g.pos.iloc[-1]) for code, g in raw.groupby('drv')}


def weather_for(race, cutoff, offline=False):
    """Summarize the race window from an available archived initialization and audit it."""
    run = forecast_run(cutoff)
    if run < dt.datetime(2024, 3, 14, tzinfo=UTC):
        return {}, dict(status='outside ECMWF archive', cutoff=cutoff.isoformat())
    location = race['Circuit'].get('Location', {})
    params = dict(latitude=location.get('lat'), longitude=location.get('long'),
                  models='ecmwf_ifs', run=run.strftime('%Y-%m-%dT%H:%M'),
                  forecast_days=10, timezone='UTC',
                  hourly='temperature_2m,precipitation,wind_speed_10m,relative_humidity_2m')
    url = 'https://single-runs-api.open-meteo.com/v1/forecast'
    attempted_runs = []
    for back in range(4):
        selected_run = run - dt.timedelta(hours=6*back)
        params['run'] = selected_run.strftime('%Y-%m-%dT%H:%M')
        data, status = cached_get(url, params, offline=offline)
        attempted_runs.append(dict(run=selected_run.isoformat(), status=status))
        if data:
            run = selected_run
            break
    audit = dict(status=status, run=run.isoformat(), cutoff=cutoff.isoformat(),
                 availability_buffer_hours=8, source=url, model='ecmwf_ifs', attempted_runs=attempted_runs)
    if not data:
        return {}, audit
    h = data.get('hourly', {})
    times = pd.to_datetime(h.get('time', []), utc=True)
    start = timestamp(race['date'], race.get('time'))
    seconds = times.as_unit('ns').asi8 / 1e9
    start_sec, end_sec = start.timestamp(), start.timestamp()+7200
    # Precipitation at t covers (t-1 hour, t], unlike instantaneous temperature.
    rain_weights = np.maximum(0, np.minimum(seconds, end_sec)-np.maximum(seconds-3600, start_sec))/3600
    if not len(seconds) or seconds.min() > start_sec or seconds.max() < end_sec or rain_weights.sum() < 2-1e-8:
        audit['status'] = 'race hours not covered'
        return {}, audit
    out = {}
    for feature, key in [('forecast_temp', 'temperature_2m'), ('forecast_rain', 'precipitation'),
                         ('forecast_wind', 'wind_speed_10m'), ('forecast_humidity', 'relative_humidity_2m')]:
        vals = pd.to_numeric(pd.Series(h.get(key, [])), errors='coerce').to_numpy()
        if len(vals) != len(seconds):
            out[feature] = np.nan
        elif feature == 'forecast_rain':
            used = rain_weights > 0
            out[feature] = float(np.sum(vals[used]*rain_weights[used])) if np.isfinite(vals[used]).all() else np.nan
        else:
            points = np.r_[start_sec, seconds[(seconds > start_sec) & (seconds < end_sec)], end_sec]
            samples = np.interp(points, seconds, vals)
            out[feature] = float(np.trapezoid(samples, points)/7200) if np.isfinite(samples).all() else np.nan
    return out, audit


def weekend_for(race, mode, cutoff=None, weather=True, practice=True, offline=False, folders=None):
    """Collect cutoff-gated weather, practice and sprint features for one weekend."""
    cutoff = cutoff or cutoff_for(race, mode)
    audit = dict(season=int(race['season']), rnd=int(race['round']), mode=mode,
                 cutoff=cutoff.isoformat(), sessions=[])
    weather_values, weather_audit = weather_for(race, cutoff, offline) if weather else ({}, {'status': 'disabled'})
    audit['weather'] = weather_audit
    records = {}
    if practice and mode != 'pre':
        eligible = []
        for calendar_key, session in [('FirstPractice', 'Practice 1'), ('SecondPractice', 'Practice 2'),
                                      ('ThirdPractice', 'Practice 3'), ('Sprint', 'Sprint')]:
            stamp = race.get(calendar_key)
            if stamp and timestamp(stamp['date'], stamp.get('time')) + dt.timedelta(hours=3) <= cutoff:
                eligible.append((calendar_key, session))
        event = base.match_folder(race['raceName'], folders or [])
        for key, session in eligible:
            url = base.TI_RAW.format(season=race['season'], event=quote(event), session=quote(session))
            data, status = cached_get(url, live=int(race['season']) == dt.date.today().year, offline=offline)
            session_audit = dict(session=session, source=url, status=status)
            if data:
                summaries, details = practice_summary(data, cutoff)
                session_audit.update(details)
                for code, values in summaries.items():
                    if session == 'Sprint':
                        records.setdefault(code, {}).update(sprint_gap=values['practice_gap'])
                    else:
                        # Latest usable session replaces an earlier one for that driver.
                        records.setdefault(code, {}).update(values)
                if session == 'Sprint' and 'pos' in data:
                    for code, position in session_positions(data, cutoff).items():
                        records.setdefault(code, {})['sprint_position'] = position
            audit['sessions'].append(session_audit)
    return dict(weather=weather_values, drivers=records), audit


def attach_weekends(frame, calendars, mode, now=None, weather=True, practice=True, offline=False):
    """Downloads independent weekends concurrently; all feature cutoffs recorded."""
    out = frame.copy()
    for c in PRACTICE_FEATURES + WEATHER_FEATURES:
        out[c] = np.nan
    races = frame[['season', 'rnd']].drop_duplicates()
    folders = {}
    if practice and mode != 'pre':
        for season in races.season.unique():
            if season >= 2024:
                saved = base.CACHE / f'ti_dirs_{int(season)}.json'
                if offline:
                    folders[int(season)] = json.loads(saved.read_text()) if saved.exists() else []
                else:
                    folders[int(season)] = base.list_event_folders(int(season))
    tasks = []
    for _, key in races.iterrows():
        race = calendars.get((int(key.season), int(key.rnd)))
        if race is None or int(key.season) < 2024:
            continue
        cutoff = cutoff_for(race, mode)
        is_future = timestamp(race['date'], race.get('time')) > (now or dt.datetime.now(UTC))
        if is_future:
            cutoff = min(cutoff, now or dt.datetime.now(UTC))
        tasks.append((race, cutoff))
    audits = []
    def work(task):
        """Download one independent weekend and preserve its calendar identity."""
        race, cutoff = task
        values, audit = weekend_for(race, mode, cutoff, weather, practice, offline, folders.get(int(race['season'])))
        return race, values, audit
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for i, (race, values, audit) in enumerate(pool.map(work, tasks), 1):
            mask = (out.season == int(race['season'])) & (out.rnd == int(race['round']))
            for col, value in values['weather'].items():
                out.loc[mask, col] = value
            for code, metrics in values['drivers'].items():
                for col, value in metrics.items():
                    out.loc[mask & out.code.eq(code), col] = value
            audits.append(audit)
            print(f'  {mode} weekend data {i}/{len(tasks)}: {race["season"]} {race["raceName"]}', flush=True)
    out['team_practice_gap'] = out.groupby(['race_idx', 'constructorId']).practice_gap.transform('mean')
    return out, audits
