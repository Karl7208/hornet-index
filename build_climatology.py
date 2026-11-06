# -*- coding: utf-8 -*-
"""
평년·경보 기준 만들기 — 운영판(run_daily.py)과 똑같은 계산(말벌 논문 2 지수 정의)을 과거 관측에 돌림 (PC에서 한 번)

  1) 해마다 계산 (여러 해 동시에)
     python build_climatology.py years 1991 2025
     (지점 정보 기상관측_국내.csv 와 D:\\ASOS\\output, D:\\AWS\\output 은 자동으로 찾음.
      다른 곳에 있으면 --meta "경로" --net ASOS=경로 --net AWS=경로 로 지정, 동시 계산 수는 --workers 6)

  2) 평년표로 묶기 (기준 기간)
     python build_climatology.py aggregate 1991 2020

  3) 경보 기준 (τ 확인 + 경계·심각)
     python build_climatology.py breakpoints 1991 2020 --check 2017 2024

  (서버 보관 관측으로 한 해 계산: python build_climatology.py archive 2026 2026 [--autumn])

  논문 2 13단계(운영 검증) 입력
     python build_climatology.py autumn 2017 2025          → data/clim/p2/autumn/CWRI_autumn_<연도>.tif (8–10월 하루 CWRI 평균)
     python build_climatology.py stations 1991 2025        → data/clim/p2/stations_<연도>.csv (날마다 지점 수)
     python build_climatology.py export 1991 2026 --last 2026-10-31 --out "E:\\research\\WSPS\\paper2\\ops_hindcast\\region_daily"
        → region_daily_<연도>.csv (date, region 5자리, sido, cwri) + stations_missing_summary.csv

출력 (data/clim/p2/ — 예전 지수로 만든 data/clim/*.csv 와 섞이지 않게 새 폴더)
  region_daily_<연도>.csv : 날짜·시군구별 CWRI(하루), CWRI(7일 평균), BAI_E       ← 보관용
  pixel_sample_<연도>.npz : 8–10월 1 km 칸 표본(하루 3,000칸) — 하루 CWRI, 7일 평균 CWRI
  climatology_sgg.csv     : 시군구 × 월일(MM-DD)별 7일 CWRI 평균·10/25/75/90% 범위, 하루 CWRI 평균, BAI_E 평균
  breakpoints.json        : τ 확인값(8–10월 하루 CWRI > 0 의 P75)과 7일 평균 P50/P75/P90/P97
  → climatology_sgg.csv, breakpoints.json 을 저장소 data/static/ 로 올리면 전망 곡선·경계/심각 단계에 쓰임

계산은 운영판과 동일: ASOS+AWS(산악기상 제외), 고도 보정, 5종 합친 적산온도 활동창(200일 이후 종료 규칙 포함),
Hargreaves 일사(FAO kRs) → 한낮 최대 W m⁻², 종별 CSI → CWRI 가중합(경쟁항 없음), BAI_E. 2월 29일은 평년표에서 뺌.
"""
import os
import sys
import argparse
import datetime as dt
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, 'data', 'clim', 'p2')
SAMPLE_PER_DAY = 3000
SAMPLE_MONTHS = (8, 9, 10)
VARS = ['tavg', 'tmax', 'tmin', 'rh', 'ws', 'rain']
MIN_STATIONS = 30


def year_core(year, obs, out_tag='', autumn=False, write_region=True):
    """obs: {date: DataFrame} (산악기상 제외, 30곳 이상인 날만) → region_daily + pixel_sample + 지점 수 저장.
    autumn=True: 8/1–10/31 하루 CWRI 합·일수를 쌓아 1 km 가을 평균 래스터(autumn/CWRI_autumn_<연도>.tif)도 저장"""
    import indices as ix
    from grid import Grid
    grid = Grid()
    if grid.regions is None:
        return year, '지역 격자 없음 (data/static/regions_1km.tif)'
    last_day = max(obs) if obs else dt.date(year, 12, 31)
    days = [dt.date(year, 1, 1) + dt.timedelta(days=i) for i in range((last_day - dt.date(year, 1, 1)).days + 1)]

    # 1) 평균기온 → 5종 활동창 (합친 창)
    tstack, doys, last = [], [], None
    for d in days:
        t = grid.interpolate(obs[d], 'tavg') if d in obs else None
        if t is None or np.all(np.isnan(t)):
            t = last if last is not None else np.full(grid.n, np.nan)
        tstack.append(t); doys.append(d.timetuple().tm_yday); last = t
    window = ix.activity_window(ix.phenology_windows(tstack, doys))

    # 2) 매일 지수 → 시군구 평균 + 8–10월 1 km 표본
    codes = grid.regions.astype(int)
    ok = codes > 0
    nreg = int(codes.max()) + 1
    cnt = np.bincount(codes[ok], minlength=nreg).astype(float)
    rng = np.random.default_rng(year)
    rows, recent, samp = [], [], {'doy': [], 'cwri': [], 'cwri7': []}
    a_sum, a_n = np.zeros(grid.n), np.zeros(grid.n)
    for d in days:
        if d not in obs:
            recent = recent[-6:]
            continue
        g = obs[d]
        doy = d.timetuple().tm_yday
        wx = {v: grid.interpolate(g, v) for v in VARS}
        tx = np.maximum(wx['tmax'], wx['tmin'])
        w = {'t': wx['tavg'], 'tx': tx, 'tn': wx['tmin'], 'rh': wx['rh'], 'ws': wx['ws'],
             'pr': np.nan_to_num(wx['rain'], nan=0.0), 'srW': grid.solar(tx, wx['tmin'], doy), 'N': ix.daylength(grid.lat, doy)}
        cw, _ = ix.compute_cwri(w, window, doy)
        bai = ix.compute_bai_e(w['t'], w['rh'], w['ws'], w['pr'], w['srW'])
        if autumn and 8 <= d.month <= 10:                       # 활동창 밖은 compute_cwri가 이미 0
            ok_c = np.isfinite(cw)
            a_sum[ok_c] += cw[ok_c]; a_n[ok_c] += 1
        recent = (recent + [cw])[-7:]
        if d.month in SAMPLE_MONTHS and len(recent) >= 5:
            with warnings.catch_warnings():                  # 7일 모두 빈 칸(섬 등)은 NaN으로 두고 경고는 숨김
                warnings.simplefilter('ignore', RuntimeWarning)
                c7 = np.nanmean(np.vstack(recent), axis=0)
            idx = rng.choice(grid.n, size=min(SAMPLE_PER_DAY, grid.n), replace=False)
            samp['doy'].append(np.full(idx.size, doy, np.int16))
            samp['cwri'].append(cw[idx].astype(np.float32)); samp['cwri7'].append(c7[idx].astype(np.float32))
        cwm = np.bincount(codes[ok], weights=np.nan_to_num(cw[ok]), minlength=nreg) / np.maximum(cnt, 1)
        bam = np.bincount(codes[ok], weights=np.nan_to_num(bai[ok]), minlength=nreg) / np.maximum(cnt, 1)
        for c in range(1, nreg):
            if cnt[c] > 0:
                rows.append((d.isoformat(), c, round(float(cwm[c]), 2), round(float(bam[c]), 2)))
    df = pd.DataFrame(rows, columns=['date', 'code', 'cwri', 'baie']).sort_values(['code', 'date'])
    df['cwri7'] = df.groupby('code')['cwri'].transform(lambda s: s.rolling(7, min_periods=5).mean()).round(2)
    os.makedirs(OUT, exist_ok=True)
    st = pd.DataFrame({'date': [d.isoformat() for d in sorted(obs)], 'n_stations': [len(obs[d]) for d in sorted(obs)]})
    st.to_csv(os.path.join(OUT, f'stations_{year}{out_tag}.csv'), index=False)
    if autumn:
        save_autumn(grid, year, a_sum, a_n, out_tag)
    if not write_region:
        return year, f'가을 래스터 {int(a_n.max()) if a_n.size else 0}일 평균'
    df.to_csv(os.path.join(OUT, f'region_daily_{year}{out_tag}.csv'), index=False)
    if samp['doy']:
        np.savez_compressed(os.path.join(OUT, f'pixel_sample_{year}{out_tag}.npz'),
                            **{k: np.concatenate(v) for k, v in samp.items()})
    return year, f'{len(obs)}일, 시군구 {df["code"].nunique()}곳, 8–10월 표본 {sum(map(len, samp["doy"])):,}칸·일'


def save_autumn(grid, year, a_sum, a_n, out_tag=''):
    """8–10월 하루 CWRI 평균을 1 km GeoTIFF로 (EPSG:5186, 바다·자료 없음 = -9999)"""
    import rasterio
    from grid import STATIC
    with rasterio.open(os.path.join(STATIC, 'mask_1km.tif')) as m:
        prof = m.profile.copy(); mask = m.read(1) > 0
    arr = np.full(mask.shape, -9999.0, np.float32)
    vals = np.where(a_n > 0, a_sum / np.maximum(a_n, 1), np.nan)
    flat = np.full(mask.shape, np.nan); flat[mask] = vals                   # grid 칸 순서 = mask 의 True 순서
    arr[np.isfinite(flat)] = flat[np.isfinite(flat)].astype(np.float32)
    prof.update(dtype='float32', nodata=-9999.0, count=1, compress='deflate')
    d = os.path.join(OUT, 'autumn'); os.makedirs(d, exist_ok=True)
    with rasterio.open(os.path.join(d, f'CWRI_autumn_{year}{out_tag}.tif'), 'w', **prof) as o:
        o.write(arr, 1)
        o.update_tags(year=str(year), period='08-01..10-31', days_max=str(int(a_n.max())), days_min=str(int(a_n[a_n > 0].min()) if (a_n > 0).any() else 0),
                      rule='mean of daily CWRI; outside activity window CWRI = 0; paper 2 v4 definition')


def load_pc_year(year, meta_path, nets, amos_use='none'):
    import local_import as li
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
    return obs


def autumn_year(year, meta_path, nets):
    """가을 래스터만 (region_daily는 그대로 둠)"""
    obs = load_pc_year(year, meta_path, nets)
    if len(obs) < 300:
        return year, f'관측 있는 날 {len(obs)}일뿐 — 건너뜀'
    return year_core(year, obs, autumn=True, write_region=False)


def stations_year(year, meta_path, nets):
    """지점 수만 셈 (지수 계산 없음, 빠름)"""
    obs = load_pc_year(year, meta_path, nets)
    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame({'date': [d.isoformat() for d in sorted(obs)], 'n_stations': [len(obs[d]) for d in sorted(obs)]}) \
        .to_csv(os.path.join(OUT, f'stations_{year}.csv'), index=False)
    return year, f'{len(obs)}일'


def export(y0, y1, out_dir, last_date=None):
    """논문 13단계용: region_daily → date, region(5자리), sido, cwri (+ 연도별 지점 수·결측 요약)"""
    from grid import STATIC
    key = pd.read_csv(os.path.join(STATIC, 'regions_sgg_code.csv'), encoding='utf-8-sig', dtype={'sgg_cd': str}).set_index('code')
    os.makedirs(out_dir, exist_ok=True)
    summ = []
    for y in range(y0, y1 + 1):
        f = os.path.join(OUT, f'region_daily_{y}.csv')
        tag = ''
        if not os.path.exists(f):
            f = os.path.join(OUT, f'region_daily_{y}_archive.csv'); tag = '_archive'
        if not os.path.exists(f):
            print(f'  {y}: region_daily 없음 — 건너뜀'); continue
        df = pd.read_csv(f)
        if last_date:
            df = df[df['date'] <= last_date]
        out = pd.DataFrame({'date': df['date'], 'region': df['code'].map(key['sgg_cd']), 'sido': df['code'].map(key['sido']),
                            'cwri': df['cwri']})
        out.to_csv(os.path.join(out_dir, f'region_daily_{y}.csv'), index=False, encoding='utf-8')
        days_all = pd.date_range(f'{y}-01-01', last_date if (last_date and last_date[:4] == str(y)) else f'{y}-12-31')
        sf = os.path.join(OUT, f'stations_{y}{tag}.csv')
        st = pd.read_csv(sf) if os.path.exists(sf) else None
        if st is not None and last_date:
            st = st[st['date'] <= last_date]
        have = set(df['date'].unique())
        summ.append({'year': y, 'days': len(days_all), 'days_with_index': len(have), 'missing_days': len(days_all) - len(have),
                     'stations_median': float(st['n_stations'].median()) if st is not None else '',
                     'stations_min': int(st['n_stations'].min()) if st is not None else '',
                     'stations_max': int(st['n_stations'].max()) if st is not None else '',
                     'source': 'KMA API Hub (운영 관측)' if tag else 'PC 원자료 ASOS+AWS', 'regions': out['region'].nunique()})
        print(f'  {y}: {len(out):,}줄, 시군구 {out["region"].nunique()}곳')
    pd.DataFrame(summ).to_csv(os.path.join(out_dir, 'stations_missing_summary.csv'), index=False, encoding='utf-8-sig')
    print(f'→ {out_dir} (지점 수·결측 요약: stations_missing_summary.csv)')


def one_year(year, meta_path, nets, amos_use='none'):
    """한 해 계산 (PC 원자료). 다른 프로세스에서 실행되므로 필요한 것을 안에서 불러옴."""
    import local_import as li
    if os.path.exists(os.path.join(OUT, f'region_daily_{year}.csv')):
        return year, 'skip (이미 있음)'
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
    return year_core(year, obs)


def archive_year(year, autumn=False):
    """서버 보관소(data/stations) 관측으로 한 해(있는 날까지) 계산 → *_archive 파일"""
    import sources
    o = sources.load_year(year)
    o = o[o['source'] != 'amos']
    obs = {dt.date.fromisoformat(d): g for d, g in o.groupby('date') if len(g) >= MIN_STATIONS}
    return year_core(year, obs, out_tag='_archive', autumn=autumn)


def breakpoints(y0, y1, check=None, tag=''):
    """pixel_sample → τ 확인값과 경계·심각 기준 (7일 평균). check=(a, b) 이면 그 기간도 함께 보고"""
    def load(a, b):
        fs = [os.path.join(OUT, f'pixel_sample_{y}{tag}.npz') for y in range(a, b + 1)]
        fs = [f for f in fs if os.path.exists(f)]
        if not fs:
            return None, 0
        z = [np.load(f) for f in fs]
        return {k: np.concatenate([x[k] for x in z]) for k in ('cwri', 'cwri7')}, len(fs)

    def stats(s, n):
        d = s['cwri'][np.isfinite(s['cwri']) & (s['cwri'] > 0)]
        w = s['cwri7'][np.isfinite(s['cwri7']) & (s['cwri7'] > 0)]
        q = lambda a, p: round(float(np.percentile(a, p)), 2)
        return {'tau_check_p75_daily': q(d, 75), 'n_years': n, 'n_daily': int(d.size),
                'daily': {'p50': q(d, 50), 'p75': q(d, 75), 'p90': q(d, 90), 'p97': q(d, 97), 'n_values': int(d.size)},
                '7day': {'p50': q(w, 50), 'p75': q(w, 75), 'p90': q(w, 90), 'p97': q(w, 97), 'n_values': int(w.size)}}

    import indices as ix
    s, n = load(y0, y1)
    if s is None:
        raise SystemExit('pixel_sample_연도.npz 가 없습니다. 먼저 years 단계를 실행하세요.')
    out = stats(s, n)
    out.update({'use': '7day', 'tau_wasp': ix.TAU_WASP,
                'sample': f'{y0}~{y1}년 8–10월 1 km 칸 하루 {SAMPLE_PER_DAY:,}칸, CWRI > 0 (논문 2 지수 정의)' + (' — 서버 보관 관측, 임시' if tag else ''),
                'built': dt.date.today().isoformat()})
    if check:
        c, m = load(*check)
        if c is not None:
            out['check'] = {'period': f'{check[0]}~{check[1]}', **stats(c, m)}
    p = os.path.join(OUT, f'breakpoints{tag}.json')
    import json
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    r = out['tau_check_p75_daily']
    print(f'τ 확인: {y0}~{y1} 8–10월 하루 CWRI>0 P75 = {r} (논문 τ {ix.TAU_WASP}, 차이 {r - ix.TAU_WASP:+.2f})')
    if 'check' in out:
        print(f'        {out["check"]["period"]} = {out["check"]["tau_check_p75_daily"]}')
    print(f'경계(7일 P90) {out["7day"]["p90"]}, 심각(7일 P97) {out["7day"]["p97"]}  → {p}')
    print('이 파일을 저장소 data/static/breakpoints.json 으로 올리면 경계·심각 단계에 쓰입니다.')


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
    ap.add_argument('step', choices=['years', 'aggregate', 'breakpoints', 'archive', 'autumn', 'stations', 'export'])
    ap.add_argument('y0', type=int); ap.add_argument('y1', type=int)
    ap.add_argument('--meta'); ap.add_argument('--net', action='append')
    ap.add_argument('--check', type=int, nargs=2, help='τ 확인 기간 (예: 2017 2024)')
    ap.add_argument('--tag', default='', help="'_archive' = 서버 보관 관측으로 만든 표본")
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument('--out', default=r'E:\research\WSPS\paper2\ops_hindcast\region_daily', help='export 저장 폴더')
    ap.add_argument('--last', help='export 마지막 날 (예: 2026-10-31)')
    ap.add_argument('--autumn', action='store_true', help='archive 단계에서 가을 래스터도 저장')
    a = ap.parse_args()
    if a.step == 'aggregate':
        return aggregate(a.y0, a.y1)
    if a.step == 'breakpoints':
        return breakpoints(a.y0, a.y1, a.check, a.tag)
    if a.step == 'archive':
        for y in range(a.y0, a.y1 + 1):
            print(' ', *archive_year(y, a.autumn))
        return
    if a.step == 'export':
        return export(a.y0, a.y1, a.out, a.last)
    nets = dict(x.split('=', 1) for x in a.net) if a.net else dict(DEFAULT_NETS)
    meta = a.meta or find_meta()
    check_ready(meta, nets)
    a.meta = meta
    years = list(range(a.y0, a.y1 + 1))
    print(f'{len(years)}개 연도를 {a.workers}개씩 동시에 계산합니다. 한 해가 끝날 때마다 한 줄씩 나옵니다.')
    job = {'years': one_year, 'autumn': autumn_year, 'stations': stations_year}[a.step]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(job, y, a.meta, nets): y for y in years}
        for f in as_completed(futs):
            try:
                y, msg = f.result()
            except Exception as e:
                y, msg = futs[f], f'오류: {e.__class__.__name__}: {e}'
            print(f'  {y}: {msg}', flush=True)
    if a.step == 'years':
        print('끝나면:  python build_climatology.py aggregate 1991 2020')
        print('        python build_climatology.py breakpoints 1991 2020 --check 2017 2024')
    elif a.step == 'autumn':
        print(f'가을 래스터 → {os.path.join(OUT, "autumn")}')


if __name__ == '__main__':
    main()
