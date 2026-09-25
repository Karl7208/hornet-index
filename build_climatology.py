# -*- coding: utf-8 -*-
"""
평년 만들기 — 운영판(run_daily.py)과 똑같은 계산을 과거 관측에 돌려 시군구 평년표를 만든다 (PC에서 한 번)

  1) 해마다 계산 (여러 해 동시에)
     python build_climatology.py years 1991 2025
     (지점 정보 기상관측_국내.csv 와 D:\\ASOS\\output, D:\\AWS\\output 은 자동으로 찾음.
      다른 곳에 있으면 --meta "경로" --net ASOS=경로 --net AWS=경로 로 지정, 동시 계산 수는 --workers 6)

  2) 평년표로 묶기 (기준 기간)
     python build_climatology.py aggregate 1991 2020

출력 (data/clim/)
  region_daily_<연도>.csv : 날짜·시군구별 CWRI(하루), CWRI(7일 평균), BAI_E       ← 보관용
  climatology_sgg.csv     : 시군구 × 월일(MM-DD)별 7일 CWRI 평균·10/25/75/90% 범위, 하루 CWRI 평균, BAI_E 평균
                            → 저장소 data/static/ 로 올리면 전망 곡선의 기준선이 됨

계산은 운영판과 동일: ASOS+AWS(산악기상 제외), 고도 보정, 적산온도 활동창(가을 종료 없음),
일사 경험식, 종별 CSI → CWRI 가중합, BAI_E. 2월 29일은 평년표에서 뺌.
"""
import os
import sys
import argparse
import datetime as dt
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, 'data', 'clim')
VARS = ['tavg', 'tmax', 'tmin', 'rh', 'ws', 'rain']
MIN_STATIONS = 30


def one_year(year, meta_path, nets, amos_use='none'):
    """한 해 계산 → region_daily_<year>.csv. 다른 프로세스에서 실행되므로 필요한 것을 안에서 불러옴."""
    import indices as ix
    from grid import Grid
    import local_import as li

    out = os.path.join(OUT, f'region_daily_{year}.csv')
    if os.path.exists(out):
        return year, 'skip (이미 있음)'
    grid = Grid()
    if grid.regions is None:
        return year, '지역 격자 없음 (data/static/regions_1km.tif)'
    meta = li.read_meta(meta_path)

    days = [dt.date(year, 1, 1) + dt.timedelta(days=i) for i in range((dt.date(year, 12, 31) - dt.date(year, 1, 1)).days + 1)]
    obs = {}
    for d in days:
        df, _ = li.import_day(nets, meta, d)
        if df is None or df.empty:
            continue
        if amos_use == 'none':
            df = df[df['source'] != 'amos']
        if len(df) >= MIN_STATIONS:
            obs[d] = df

    if len(obs) < 300:
        return year, f'관측 있는 날 {len(obs)}일뿐 — 건너뜀'

    # 1) 1년치 평균기온 → 종별 활동창
    tstack, doys, last = [], [], None
    for d in days:
        t = grid.interpolate(obs[d], 'tavg') if d in obs else None
        if t is None or np.all(np.isnan(t)):
            t = last if last is not None else np.full(grid.n, np.nan)
        tstack.append(t); doys.append(d.timetuple().tm_yday); last = t
    windows = ix.phenology_windows(tstack, doys)

    # 2) 매일 지수 → 시군구 평균
    codes = grid.regions.astype(int)
    ok = codes > 0
    nreg = int(codes.max()) + 1
    cnt = np.bincount(codes[ok], minlength=nreg).astype(float)
    rows = []
    for d in days:
        if d not in obs:
            continue
        g = obs[d]
        wx = {v: grid.interpolate(g, v) for v in VARS}
        tx = np.maximum(wx['tmax'], wx['tmin'])
        w = {'t': wx['tavg'], 'tx': tx, 'tn': wx['tmin'], 'rh': wx['rh'], 'ws': wx['ws'],
             'pr': np.nan_to_num(wx['rain'], nan=0.0), 'srW': ix.solar_from_temp(wx['tavg'], tx, wx['tmin'])}
        cw, _ = ix.compute_cwri(w, windows, d.timetuple().tm_yday)
        bai = ix.compute_bai_e(w['t'], w['rh'], w['ws'], w['pr'], w['srW'])
        cwm = np.bincount(codes[ok], weights=np.nan_to_num(cw[ok]), minlength=nreg) / np.maximum(cnt, 1)
        bam = np.bincount(codes[ok], weights=np.nan_to_num(bai[ok]), minlength=nreg) / np.maximum(cnt, 1)
        for c in range(1, nreg):
            if cnt[c] > 0:
                rows.append((d.isoformat(), c, round(float(cwm[c]), 2), round(float(bam[c]), 2)))
    df = pd.DataFrame(rows, columns=['date', 'code', 'cwri', 'baie']).sort_values(['code', 'date'])
    df['cwri7'] = df.groupby('code')['cwri'].transform(lambda s: s.rolling(7, min_periods=5).mean()).round(2)
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(out, index=False)
    return year, f'{len(obs)}일, 시군구 {df["code"].nunique()}곳'


def aggregate(y0, y1):
    from grid import STATIC
    files = [os.path.join(OUT, f'region_daily_{y}.csv') for y in range(y0, y1 + 1)]
    have = [f for f in files if os.path.exists(f)]
    if not have:
        raise SystemExit('region_daily_연도.csv 가 없습니다. 먼저 years 단계를 실행하세요.')
    df = pd.concat([pd.read_csv(f) for f in have])
    df['md'] = df['date'].str[5:]
    df = df[df['md'] != '02-29']
    g = df.groupby(['code', 'md'])
    out = pd.DataFrame({
        'cwri7_mean': g['cwri7'].mean(), 'cwri7_p10': g['cwri7'].quantile(.10), 'cwri7_p25': g['cwri7'].quantile(.25),
        'cwri7_p75': g['cwri7'].quantile(.75), 'cwri7_p90': g['cwri7'].quantile(.90),
        'cwri_mean': g['cwri'].mean(), 'baie_mean': g['baie'].mean(), 'n_years': g['cwri'].count()
    }).round(2).reset_index()
    names = pd.read_csv(os.path.join(STATIC, 'regions.csv'), encoding='utf-8-sig').set_index('code')['name']
    out.insert(1, 'region', out['code'].map(names))
    p = os.path.join(OUT, 'climatology_sgg.csv')
    out.to_csv(p, index=False, encoding='utf-8-sig')
    print(f'평년표: {y0}~{y1} 중 {len(have)}개 연도, 시군구 {out["code"].nunique()}곳 × {out["md"].nunique()}일 → {p}')
    print('이 파일을 저장소 data/static/ 에 올리면 전망 곡선의 기준선으로 쓰입니다.')


META_CANDIDATES = [
    '기상관측_국내.csv',
    os.path.join('calculation', '기상관측_국내.csv'),
    os.path.join('..', 'calculation', '기상관측_국내.csv'),
    r'E:\research\research_paper\app\hornet-index\calculation\기상관측_국내.csv',
    r'E:\research\research_paper\app\hornet-report\calculation\기상관측_국내.csv',
]
DEFAULT_NETS = {'ASOS': r'D:\ASOS\output', 'AWS': r'D:\AWS\output'}


def find_meta():
    for c in META_CANDIDATES:
        p = c if os.path.isabs(c) else os.path.join(ROOT, c)
        if os.path.exists(p):
            return os.path.abspath(p)
    return None


def check_ready(meta, nets):
    from grid import STATIC
    need = ['mask_1km.tif', 'regions_1km.tif', 'regions.csv']
    miss = [f for f in need if not os.path.exists(os.path.join(STATIC, f))]
    if miss:
        raise SystemExit(f'격자 파일이 없습니다: {STATIC} 에 {", ".join(miss)} — hornet-index-pipeline\\data\\static 에서 복사하세요.')
    if not os.path.exists(os.path.join(STATIC, 'dem_1km.tif')):
        print('  ⚠ dem_1km.tif 없음 → 고도 보정 없이 계산 (운영판과 달라짐)')
    if not meta or not os.path.exists(meta):
        raise SystemExit('기상관측_국내.csv 를 찾지 못했습니다. --meta "파일 경로" 로 지정하세요.')
    for k, v in nets.items():
        if not os.path.isdir(os.path.join(v, 'Tavg')):
            raise SystemExit(f'{k} 관측 폴더에 Tavg 폴더가 없습니다: {v}')
    print(f'지점 정보: {meta}')
    print('관측 폴더: ' + ', '.join(f'{k}={v}' for k, v in nets.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=['years', 'aggregate'])
    ap.add_argument('y0', type=int); ap.add_argument('y1', type=int)
    ap.add_argument('--meta'); ap.add_argument('--net', action='append')
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    if a.step == 'aggregate':
        return aggregate(a.y0, a.y1)
    nets = dict(x.split('=', 1) for x in a.net) if a.net else dict(DEFAULT_NETS)
    meta = a.meta or find_meta()
    check_ready(meta, nets)
    a.meta = meta
    years = list(range(a.y0, a.y1 + 1))
    print(f'{len(years)}개 연도를 {a.workers}개씩 동시에 계산합니다. 한 해가 끝날 때마다 한 줄씩 나옵니다.')
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(one_year, y, a.meta, nets): y for y in years}
        for f in as_completed(futs):
            try:
                y, msg = f.result()
            except Exception as e:
                y, msg = futs[f], f'오류: {e.__class__.__name__}: {e}'
            print(f'  {y}: {msg}', flush=True)
    print('끝나면:  python build_climatology.py aggregate 1991 2020')


if __name__ == '__main__':
    main()
