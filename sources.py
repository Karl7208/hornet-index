# -*- coding: utf-8 -*-
"""
관측 자료 모으기 — 모든 출처를 하나의 표준 표로 통일해 연도별 CSV에 쌓는다.

표준 열: date, source, stn, lat, lon, elev, tavg, tmax, tmin, rh, ws, rain, srad
  (srad = 일사량 합 MJ/m²/일, 없으면 빈칸)

저장: data/stations/<연도>-<월>.csv  ← 매일 늘어나는 원자료 보관소 (월별 파일)
       (파이프라인의 '상태'는 이 파일뿐. 격자는 매일 여기서 다시 계산)

출처
  1) 기상청 API허브 AWS 전 지점(ASOS 포함) : KMA_KEY 환경변수 — 하루 27회 호출 (fetch_kma_day)
  2) 수동/외부 자료                        : data/import/*.csv (표준 열) 넣어두면 다음 실행 때 합쳐짐
  산림청 산악기상(AMOS)은 계산에서 제외하므로 받지 않음
"""
import os
import io
import glob
import shutil
import datetime as dt
import numpy as np
import pandas as pd
import requests

ROOT = os.path.dirname(__file__)
ST_DIR = os.path.join(ROOT, 'data', 'stations')
IMPORT_DIR = os.path.join(ROOT, 'data', 'import')
META_CSV = os.path.join(ROOT, 'data', 'static', 'stations_meta.csv')   # stn,lat,lon,elev (선택)
COLS = ['date', 'source', 'stn', 'lat', 'lon', 'elev', 'tavg', 'tmax', 'tmin', 'rh', 'ws', 'rain', 'srad']
KMA_BASE = 'https://apihub.kma.go.kr/api/typ01/url/'
TIMEOUT = 60


# ─────────────────────────────────────────────────────────────
# 보관소
# ─────────────────────────────────────────────────────────────
def load_year(y):
    """data/stations/<연도>-<월>.csv 들을 합쳐 읽음"""
    files = sorted(glob.glob(os.path.join(ST_DIR, f'{y}-*.csv')))
    if not files:
        return pd.DataFrame(columns=COLS)
    return pd.concat([pd.read_csv(f, dtype={'stn': str, 'source': str, 'date': str}) for f in files])


def save_year(y, df):
    """월별 파일로 저장 — 지난달 파일은 내용이 같으면 git 변경도 생기지 않음"""
    os.makedirs(ST_DIR, exist_ok=True)
    df = df[COLS].drop_duplicates(subset=['date', 'source', 'stn'], keep='last').sort_values(['date', 'source', 'stn'])
    for ym, g in df.groupby(df['date'].str[:7]):
        g.to_csv(os.path.join(ST_DIR, f'{ym}.csv'), index=False, float_format='%.2f')


def dates_present(df, source):
    return set(df.loc[df['source'] == source, 'date'])


# ─────────────────────────────────────────────────────────────
# 기상청 API허브 — AWS 전 지점(ASOS 포함 약 720곳) 하루치
#   sfc_aws_day.php : 최고·최저기온, 일강수 (좌표·해발고도 포함)       → 3회
#   awsh.php        : 1~24시 정시 기온·풍속·습도 → 일평균 (18시간 이상) → 24회
#   2026-09 GitHub Actions 에서 접속·형식 확인 (kma_probe.py)
# ─────────────────────────────────────────────────────────────
from concurrent.futures import ThreadPoolExecutor
import urllib.request, urllib.parse

MIN_HOURS = 18
DAY_OBS = {'ta_max': 'tmax', 'ta_min': 'tmin', 'rn_day': 'rain'}


def _kma(endpoint, params, tries=3):
    key = os.environ.get('KMA_KEY', '').strip()
    if not key:
        raise RuntimeError('KMA_KEY 환경변수가 없습니다.')
    url = KMA_BASE + endpoint + '?' + urllib.parse.urlencode({**params, 'authKey': key})
    last = None
    for _ in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=TIMEOUT) as r:
                return r.read().decode('euc-kr', errors='replace')
        except Exception as e:
            last = e
    raise RuntimeError(f'{endpoint} 호출 실패: {last}')


def _data_rows(text):
    return [l.split() for l in text.splitlines() if l.strip() and not l.lstrip().startswith('#')]


def _num(x, lo=None, hi=None):
    try:
        v = float(x)
    except ValueError:
        return np.nan
    if v <= -90 or (lo is not None and v < lo) or (hi is not None and v > hi):
        return np.nan
    return v


def parse_aws_day(text, key, store, meta):
    """YYMMDD STN LON LAT HT VAL 이름"""
    for t in _data_rows(text):
        if len(t) < 6:
            continue
        stn = t[1]
        meta[stn] = (float(t[2]), float(t[3]), float(t[4]))
        lo = 0 if key == 'rain' else -50
        store.setdefault(stn, {})[key] = _num(t[5], lo=lo)


def parse_awsh(text, acc):
    """YYMMDDHHMI STN TA WD WS RN_DAY RN_HR1 HM PA PS"""
    for t in _data_rows(text):
        if len(t) < 8:
            continue
        a = acc.setdefault(t[1], {'ta': [], 'ws': [], 'hm': []})
        for k, i, lo, hi in (('ta', 2, -50, 50), ('ws', 4, 0, 70), ('hm', 7, 0, 100)):
            v = _num(t[i], lo, hi)
            if not np.isnan(v):
                a[k].append(v)


def fetch_kma_day(day):
    D = day.strftime('%Y%m%d')
    store, meta, acc = {}, {}, {}
    for obs, key in DAY_OBS.items():
        parse_aws_day(_kma('sfc_aws_day.php', {'tm2': D, 'obs': obs, 'stn': 0, 'disp': 0, 'help': 0}), key, store, meta)
    base = dt.datetime(day.year, day.month, day.day)
    tms = [(base + dt.timedelta(hours=h)).strftime('%Y%m%d%H00') for h in range(1, 25)]   # 01시 ~ 다음날 00시(=24시)
    with ThreadPoolExecutor(max_workers=6) as ex:
        texts = list(ex.map(lambda tm: _kma('awsh.php', {'tm': tm, 'help': 0}), tms))
    for tx in texts:
        parse_awsh(tx, acc)
    rows = []
    for stn, (lon, lat, ht) in meta.items():
        a = acc.get(stn, {'ta': [], 'ws': [], 'hm': []})
        mean = lambda xs: float(np.mean(xs)) if len(xs) >= MIN_HOURS else np.nan
        v = store.get(stn, {})
        rows.append({'date': day.isoformat(), 'source': 'kma', 'stn': stn, 'lat': lat, 'lon': lon, 'elev': ht,
                     'tavg': mean(a['ta']), 'tmax': v.get('tmax', np.nan), 'tmin': v.get('tmin', np.nan),
                     'rh': mean(a['hm']), 'ws': mean(a['ws']), 'rain': v.get('rain', np.nan), 'srad': np.nan})
    return pd.DataFrame(rows, columns=COLS)


# ─────────────────────────────────────────────────────────────
# 갱신
# ─────────────────────────────────────────────────────────────
def _chunks(d1, d2, days=31):
    s = d1
    while s <= d2:
        e = min(s + dt.timedelta(days=days - 1), d2)
        yield s, e
        s = e + dt.timedelta(days=1)


def import_folder():
    """data/import/*.csv (표준 열) → 연도별 보관소로 합치고 done/ 으로 이동"""
    files = sorted(glob.glob(os.path.join(IMPORT_DIR, '*.csv')))
    for f in files:
        df = pd.read_csv(f, dtype={'stn': str, 'source': str, 'date': str})
        for y, g in df.groupby(df['date'].str[:4]):
            save_year(int(y), pd.concat([load_year(int(y)), g[COLS]]))
        os.makedirs(os.path.join(IMPORT_DIR, 'done'), exist_ok=True)
        shutil.move(f, os.path.join(IMPORT_DIR, 'done', os.path.basename(f)))
        print(f'  가져옴: {os.path.basename(f)} ({len(df)}행)')


def update(upto, log, max_days=400):
    """1월 1일 ~ upto 중 빠진 날을 기상청에서 하루씩 받아 바로 저장 (끊겨도 받은 날까지는 남음)"""
    import_folder()
    y = upto.year
    df = load_year(y)
    have = set(df['date']) if len(df) else set()
    jan1 = dt.date(y, 1, 1)
    need = [jan1 + dt.timedelta(days=i) for i in range((upto - jan1).days + 1)]
    need = [d for d in need if d.isoformat() not in have][:max_days]
    if need:
        log(f'  기상청에서 받을 날짜 {len(need)}일 ({need[0]} ~ {need[-1]})')
    for i, d in enumerate(need, 1):
        try:
            part = fetch_kma_day(d)
        except Exception as e:
            log(f'  {d}: 수집 실패 — {str(e)[:150]}')
            continue
        month = load_year(y)
        save_year(y, pd.concat([month, part]) if len(month) else part)
        if i % 20 == 0 or i == len(need):
            log(f'  {i}/{len(need)}일 받음 (마지막 {d}, 지점 {len(part)}곳)')
    return load_year(y)


if __name__ == '__main__':
    import sys
    if len(sys.argv) >= 3 and sys.argv[1] == 'kma-test':
        d = dt.datetime.strptime(sys.argv[2], '%Y%m%d').date()
        x = fetch_kma_day(d)
        print(x.head(15).to_string()); print(len(x), '지점,', x[['tavg', 'rh', 'ws']].notna().sum().to_dict())
