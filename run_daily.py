# -*- coding: utf-8 -*-
"""
매일 실행: 관측 수집 → 1km 보간 → 종별 활동창 → 종별 CSI·CWRI·BAI_E → 7일 Gi* → 30일 중첩일수 → 출력

  python run_daily.py                    # 어제(KST) 기준
  python run_daily.py --date 2026-09-23  # 특정 날짜
  python run_daily.py --no-fetch         # 수집 없이 보관된 자료로만 계산

출력
  data/latest/*.png, meta.json, regions.csv   ← 웹 지도가 읽는 것 (저장소에 커밋)
  out/<날짜>/*.tif                            ← GIS용 원본 격자 (커밋 안 함, Actions 산출물로 보관)
"""
import os
import json
import argparse
import datetime as dt
import numpy as np
import pandas as pd

import indices as ix
import sources
from grid import Grid
import export

ROOT = os.path.dirname(os.path.abspath(__file__))
LATEST_FINAL = os.path.join(ROOT, 'data', 'latest')
LATEST = LATEST_FINAL + '_tmp'          # 계산 중에는 임시 폴더에 쓰고 끝나면 한 번에 교체
DAYS = os.path.join(ROOT, 'data', 'days')
VARS = ['tavg', 'tmax', 'tmin', 'rh', 'ws', 'rain']
WINDOW_GI = 7          # Gi*에 쓰는 이동평균 일수
WINDOW_OVERLAP = 30    # 중첩일수 누적 기간
MIN_STATIONS = 30
SEASON_OF_MONTH = {3: 'spring', 4: 'spring', 5: 'spring', 6: 'summer', 7: 'summer',
                   8: 'autumn', 9: 'autumn', 10: 'autumn', 11: 'autumn'}   # 노트북 06 계절 구분 (11월은 가을로)
PERSIST_MIN = 0.87          # 지속성 7/8년 이상 = 상습 지역
# 산악기상(AMOS) 사용 범위: 'none' 전부 제외 / 'temp' 기온·습도만 / 'all' 전부
#   능선 관측소의 센 바람·낮은 기온이 말벌 서식지(저지대·골짜기)를 대표하지 못해 기본은 제외
AMOS_USE = 'none'
AMOS_TEMP_VARS = ('tavg', 'tmax', 'tmin', 'rh')


def pick(g, var):
    if g is None or AMOS_USE == 'all' or (AMOS_USE == 'temp' and var in AMOS_TEMP_VARS):
        return g
    return g[g['source'] != 'amos']
NATIONAL_STAGES = [(70, '심각'), (40, '경계'), (10, '주의'), (0, '관심')]   # 주의 이상 면적(%) 기준


def load_breakpoints():
    p = os.path.join(ROOT, 'data', 'static', 'breakpoints.json')
    if not os.path.exists(p):
        return None
    with open(p, encoding='utf-8') as f:
        j = json.load(f)
    use = j.get('use', 'daily')
    d = j[use]
    return {'use': use, 'p75': d['p75'], 'p90': d['p90'], 'p97': d['p97'], 'sample': j.get('sample', '')}


def load_persistence(grid, season):
    import rasterio
    p = os.path.join(ROOT, 'data', 'static', f'persistence_{season}.tif') if season else None
    if not p or not os.path.exists(p):
        return None
    with rasterio.open(p) as src:
        a = src.read(1).astype('float32')
    return np.where(a < 0, np.nan, a)[grid.mask]


def kst_yesterday():
    return (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=9)).date() - dt.timedelta(days=1)


CLIM = os.path.join(ROOT, 'data', 'static', 'climatology_sgg.csv')
OUTLOOK_DAYS = 14
ANOMALY_DECAY = 0.8          # 오늘의 평년 대비 차이가 하루에 20%씩 줄어 평년으로 수렴 (지속 예보)


def stage_of(v, t):
    t1, t2, t3 = t
    return '심각' if v >= t3 else '경계' if v >= t2 else '주의' if v >= t1 else '관심'


def write_local(grid, day, cw_hist, cw_days, cw7, ac, chronic, thr):
    """시민용 '우리 지역' 자료: data/latest/local/regions.json(목록) + <코드>.json(시군구별)"""
    from pyproj import Transformer
    codes = grid.regions.astype(int)
    nreg = int(codes.max()) + 1
    ok = codes > 0
    cnt = np.bincount(codes[ok], minlength=nreg).astype(float)
    mean = lambda v: np.bincount(codes[ok], weights=np.nan_to_num(v[ok]), minlength=nreg) / np.maximum(cnt, 1)

    daily = np.vstack([mean(c) for c in cw_hist])                       # [날짜, 시군구]
    roll = np.vstack([daily[max(0, i - 6):i + 1].mean(axis=0) for i in range(len(daily))])
    today7 = mean(cw7)
    p1, p2, p3 = mean((ac >= 1).astype(float)), mean((ac >= 2).astype(float)), mean((ac >= 3).astype(float))
    pchr = mean(chronic.astype(float))

    clim = None
    if os.path.exists(CLIM):
        c = pd.read_csv(CLIM, encoding='utf-8-sig').fillna(0)          # 연초 1~4일은 7일 평균 빈칸 → 0 (겨울)
        clim = {(int(a), b): (m, lo, hi) for a, b, m, lo, hi in zip(c['code'], c['md'], c['cwri7_mean'], c['cwri7_p10'], c['cwri7_p90'])}
    md = lambda d: d.strftime('%m-%d') if d.strftime('%m-%d') != '02-29' else '02-28'

    tr = Transformer.from_crs('EPSG:5186', 'EPSG:4326', always_xy=True)
    out_dir = os.path.join(LATEST, 'local'); os.makedirs(out_dir, exist_ok=True)
    listing = []
    for code, name in sorted(grid.region_names.items()):
        m = grid.regions == code
        if not m.any():
            continue
        xy = grid.xy[m]
        lon, lat = tr.transform([xy[:, 0].min() - 500, xy[:, 0].max() + 500], [xy[:, 1].min() - 500, xy[:, 1].max() + 500])
        clon, clat = tr.transform(xy[:, 0].mean(), xy[:, 1].mean())
        v7 = round(float(today7[code]), 1)
        rec = {'code': code, 'name': name, 'bbox': [[round(lat[0], 4), round(lon[0], 4)], [round(lat[1], 4), round(lon[1], 4)]],
               'center': [round(clat, 4), round(clon, 4)], 'cw7': v7, 'stage': stage_of(v7, thr)}
        listing.append(rec)
        series = [{'d': d.isoformat(), 'cw': round(float(daily[i, code]), 1), 'cw7': round(float(roll[i, code]), 1)} for i, d in enumerate(cw_days)]
        detail = {**rec, 'date': day.isoformat(), 'thresholds': [round(float(x), 2) for x in thr],
                  'pct': {'주의': round(float(p1[code] * 100), 1), '경계': round(float(p2[code] * 100), 1), '심각': round(float(p3[code] * 100), 1)},
                  'chronic_pct': round(float(pchr[code] * 100), 1), 'series': series, 'clim': [], 'outlook': []}
        if clim:
            span = [cw_days[0] + dt.timedelta(days=k) for k in range((day - cw_days[0]).days + OUTLOOK_DAYS + 1)]
            detail['clim'] = [{'d': d.isoformat(), 'm': clim[(code, md(d))][0], 'lo': clim[(code, md(d))][1], 'hi': clim[(code, md(d))][2]}
                              for d in span if (code, md(d)) in clim]
            base = clim.get((code, md(day)))
            if base:
                a0 = v7 - base[0]
                for k in range(1, OUTLOOK_DAYS + 1):
                    d = day + dt.timedelta(days=k); b = clim.get((code, md(d)))
                    if b:
                        detail['outlook'].append({'d': d.isoformat(), 'v': round(float(np.clip(b[0] + a0 * ANOMALY_DECAY ** k, 0, 100)), 1)})
                detail['anomaly'] = round(float(a0), 1)
        with open(os.path.join(out_dir, f'{code}.json'), 'w', encoding='utf-8') as f:
            json.dump(detail, f, ensure_ascii=False, separators=(',', ':'))
    with open(os.path.join(out_dir, 'regions.json'), 'w', encoding='utf-8') as f:
        json.dump({'date': day.isoformat(), 'thresholds': [round(float(x), 2) for x in thr], 'regions': listing}, f, ensure_ascii=False, separators=(',', ':'))
    return len(listing)


def publish(day):
    """임시 폴더 → data/latest 교체 + data/days/<날짜>/ 보관 + 날짜 목록 갱신"""
    import shutil
    dd = os.path.join(DAYS, day.isoformat())
    shutil.rmtree(dd, ignore_errors=True)
    shutil.copytree(LATEST, dd)
    old = LATEST_FINAL + '_old'
    shutil.rmtree(old, ignore_errors=True)
    try:
        if os.path.exists(LATEST_FINAL):
            os.replace(LATEST_FINAL, old)
        os.replace(LATEST, LATEST_FINAL)
    except OSError:                                    # 윈도우에서 폴더가 잠겨 있으면 파일 단위로 덮어씀
        os.makedirs(LATEST_FINAL, exist_ok=True)
        for f in os.listdir(LATEST):
            s, t = os.path.join(LATEST, f), os.path.join(LATEST_FINAL, f)
            if os.path.isdir(s):
                shutil.rmtree(t, ignore_errors=True); shutil.copytree(s, t)
            else:
                shutil.copy2(s, t)
        shutil.rmtree(LATEST, ignore_errors=True)
    shutil.rmtree(old, ignore_errors=True)
    dates = sorted(d for d in os.listdir(DAYS) if os.path.isdir(os.path.join(DAYS, d)))
    with open(os.path.join(DAYS, 'index.json'), 'w', encoding='utf-8') as f:
        json.dump(dates, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date'); ap.add_argument('--no-fetch', action='store_true')
    a = ap.parse_args()
    day = dt.datetime.strptime(a.date, '%Y-%m-%d').date() if a.date else kst_yesterday()
    logs, warns = [], []
    log = lambda s: (print(s), logs.append(s))

    grid = Grid()
    log(f'[{day}] 격자 {grid.shape[0]}×{grid.shape[1]}, 유효 칸 {grid.n:,}'
        + (', 고도보정 사용' if grid.dem is not None else ', 고도보정 없음'))
    if grid.dem_missing:
        warns.append(f'DEM이 덮지 않는 칸 {grid.dem_missing:,}개(섬 등) → 그 칸은 고도 보정 없이 계산')

    # 1) 수집 --------------------------------------------------------------
    obs = sources.load_year(day.year) if a.no_fetch else sources.update(day, log)
    obs = obs[obs['date'] <= day.strftime('%Y-%m-%d')]
    by_day = {d: g for d, g in obs.groupby('date')}

    # 2) 1월 1일~오늘 평균기온 격자 → 종별 활동창 -----------------------------
    jan1 = dt.date(day.year, 1, 1)
    days = [jan1 + dt.timedelta(days=i) for i in range((day - jan1).days + 1)]
    tstack, doys, last, gaps = [], [], None, 0
    for d in days:
        g = by_day.get(d.strftime('%Y-%m-%d'))
        t = grid.interpolate(pick(g, 'tavg'), 'tavg') if g is not None else None
        if t is None or np.all(np.isnan(t)):
            gaps += 1
            t = last if last is not None else np.full(grid.n, np.nan)
        tstack.append(t); doys.append(d.timetuple().tm_yday); last = t
    if gaps:
        warns.append(f'1월 1일 이후 관측 없는 날 {gaps}일 (앞날 값으로 대체)')
    windows = ix.phenology_windows(tstack, doys)

    # 3) 최근 30일 지수 ----------------------------------------------------
    cw_hist, cw_days, overlap_days, n_st_today = [], [], np.zeros(grid.n), 0
    today = None
    for d in days[-WINDOW_OVERLAP:]:
        g = by_day.get(d.strftime('%Y-%m-%d'))
        if g is None or len(g) < 3:
            continue
        wx = {v: grid.interpolate(pick(g, v), v) for v in VARS}
        tx = np.maximum(wx['tmax'], wx['tmin'])              # 보간 뒤 최고<최저 역전 방지
        solar = ix.solar_from_temp(wx['tavg'], tx, wx['tmin'])
        for v in ('rh', 'ws', 'rain'):
            if np.all(np.isnan(wx[v])):
                warns.append(f'{d} {v} 자료 없음')
        w = {'t': wx['tavg'], 'tx': tx, 'tn': wx['tmin'], 'rh': wx['rh'], 'ws': wx['ws'],
             'pr': np.nan_to_num(wx['rain'], nan=0.0), 'srW': solar}
        doy = d.timetuple().tm_yday
        cwri, per = ix.compute_cwri(w, windows, doy)
        bai = ix.compute_bai_e(w['t'], w['rh'], w['ws'], w['pr'], w['srW'])
        overlap_days += ((cwri > ix.TAU_WASP) & (bai > ix.TAU_BEE)).astype(float)
        cw_hist.append(cwri); cw_days.append(d)
        if d == day:
            today = dict(cwri=cwri, per=per, bai=bai, wx=w)
            n_st_today = len(pick(g, 'ws'))
    if today is None:
        raise SystemExit(f'{day} 관측 자료가 없어 계산할 수 없습니다.')

    cw7 = np.nanmean(np.vstack(cw_hist[-WINDOW_GI:]), axis=0)
    z, p = ix.getis_ord_gi_star(grid.to_2d(cw7))
    gcls = ix.classify_hotspot(z)
    zc = z[grid.rows, grid.cols]
    # 공개용 경보 3단계 — 미국 산불위험등급(NFDRS)처럼 과거 지수 분포의 백분위수로 단계 경계를 정함
    #   1 주의: 7일 평균 CWRI ≥ τ_wasp 37.76 (논문 2 확정값, 고정)
    #   2 경계: ≥ P90 (breakpoints.json 의 "use" 분포, 기본 권장 7day)
    #   3 심각: ≥ P97
    #   경계값: data/static/breakpoints.json (compute_breakpoints.py, τ_wasp와 같은 표본)
    #   상습 지역(계절 Gi* 지속성 ≥ 7/8년)은 단계와 분리해 빗금 층(chronic)으로 따로 표시
    bp = load_breakpoints()
    t1 = ix.TAU_WASP                    # 주의는 논문 2 확정값으로 고정 (breakpoints의 P75와 무관)
    t2 = bp['p90'] if bp else np.inf
    t3 = bp['p97'] if bp else np.inf
    if bp is None:
        warns.append('breakpoints.json 없음 → 경계·심각 단계 꺼짐 (compute_breakpoints.py 실행 필요)')
    alert_1d = (cw7 >= t1).astype(np.int8) + (cw7 >= t2).astype(np.int8) + (cw7 >= t3).astype(np.int8)
    alert = grid.to_2d(alert_1d.astype('float32'), fill=-9).astype(np.int8)

    season = SEASON_OF_MONTH.get(day.month)
    pers = load_persistence(grid, season)
    chronic_1d = (np.nan_to_num(pers) >= PERSIST_MIN) if pers is not None else np.zeros(grid.n, bool)
    if pers is None and season:
        warns.append(f'지속성 파일(persistence_{season}.tif) 없음 → 상습 지역 표시 없음')
    ac = alert_1d
    pct1 = float(np.mean(ac >= 1) * 100)
    national = next(name for lim, name in NATIONAL_STAGES if pct1 >= lim)

    # 4) 출력 --------------------------------------------------------------
    import shutil
    shutil.rmtree(LATEST, ignore_errors=True)
    os.makedirs(LATEST, exist_ok=True)
    tif_dir = os.path.join(ROOT, 'out', day.isoformat()); os.makedirs(tif_dir, exist_ok=True)
    layers = {'cwri': (today['cwri'], export.PAL_INDEX, False),
              'baie': (today['bai'], export.PAL_BEE, False),
              'cwri7': (cw7, export.PAL_INDEX, False),
              'overlap30': (overlap_days, export.PAL_DAYS, False)}
    for sp in ix.SPECIES:
        layers[f'csi_{sp}'] = (today['per'][sp], export.PAL_INDEX, False)
    bounds = None
    for name, (v, pal, cat) in layers.items():
        a2 = grid.to_2d(v)
        export.write_tif(os.path.join(tif_dir, f'{name}.tif'), a2, grid)
        bounds = export.png4326(os.path.join(LATEST, f'{name}.png'), a2, grid, pal)
    export.write_tif(os.path.join(tif_dir, 'gi_z.tif'), z, grid)
    export.png4326(os.path.join(LATEST, 'gi_class.png'), gcls.astype('float32'), grid, export.PAL_GI, categorical=True)
    export.write_tif(os.path.join(tif_dir, 'alert.tif'), alert.astype('float32'), grid)
    export.png4326(os.path.join(LATEST, 'alert.png'), alert.astype('float32'), grid, export.PAL_ALERT, categorical=True)
    export.png4326(os.path.join(LATEST, 'land.png'), grid.mask.astype('float32'), grid, export.PAL_LAND, categorical=True)
    chronic2d = grid.to_2d(chronic_1d.astype('float32'), fill=0)
    export.write_tif(os.path.join(tif_dir, 'chronic.tif'), chronic2d, grid)
    export.png4326(os.path.join(LATEST, 'chronic.png'), chronic2d, grid, export.PAL_HATCH, categorical=True)

    summ = export.region_summary(grid, {
        'cwri': today['cwri'], 'cwri7': cw7, 'baie': today['bai'], 'overlap30': overlap_days,
        'pct_hot': (zc > 1.96).astype(float), 'pct_alert': (ac >= 2).astype(float), 'pct_chronic': chronic_1d.astype(float), 'pct_wasp_active': (today['cwri'] > ix.TAU_WASP).astype(float),
        **{f'csi_{sp}': today['per'][sp] for sp in ix.SPECIES}})
    if summ:
        pd.DataFrame(summ).to_csv(os.path.join(LATEST, 'regions.csv'), index=False, encoding='utf-8-sig')
        arch = os.path.join(ROOT, 'data', 'archive', str(day.year)); os.makedirs(arch, exist_ok=True)
        pd.DataFrame(summ).assign(date=day.isoformat()).to_csv(
            os.path.join(arch, f'regions_{day.isoformat()}.csv'), index=False, encoding='utf-8-sig')

    # 5) 상식 점검 — 지수가 터무니없는지 스스로 확인 --------------------------
    cw = today['cwri']
    p75 = float(np.nanpercentile(cw, 75))
    pheno = {}
    for sp in ix.SPECIES:
        e = windows[sp][0]
        med = float(np.nanmedian(e)) if np.any(~np.isnan(e)) else None
        ref = ix.PHENOLOGY_PARAMS[sp]['mean_emerge_doy']
        pheno[sp] = {'ko': ix.SPECIES_KO[sp], 'emerge_doy_median': med, 'ref_emerge_doy_jeonju': ref,
                     'pct_active_today': round(float(ix.active_mask(windows[sp], doys[-1]).mean() * 100), 1),
                     'csi_mean_today': round(float(np.nanmean(today['per'][sp])), 1)}
        if med is not None and abs(med - ref) > 30:
            warns.append(f'{ix.SPECIES_KO[sp]} 출현일 중앙값 {med:.0f}일 — 전주 보정값 {ref}일과 30일 이상 차이')
    if n_st_today < MIN_STATIONS:
        warns.append(f'오늘 관측소 {n_st_today}곳 — 보간이 거칠 수 있음')
    t_med = float(np.nanmedian(today['wx']['t']))
    if not (-30 < t_med < 40):
        warns.append(f'평균기온 중앙값 {t_med:.1f}°C — 관측 자료 단위·결측 부호 확인 필요')

    meta = {
        'date': day.isoformat(), 'generated_utc': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
        'bounds': bounds, 'n_stations_today': int(n_st_today),
        'sources_today': {str(k): int(v) for k, v in obs[obs['date'] == day.strftime('%Y-%m-%d')]['source'].value_counts().items()},
        'thresholds': {'tau_wasp': ix.TAU_WASP, 'tau_bee': ix.TAU_BEE, 'gi_radius_cells': ix.GI_RADIUS,
                       'gi_window_days': WINDOW_GI, 'overlap_window_days': WINDOW_OVERLAP},
        'stats': {'cwri_mean': round(float(np.nanmean(cw)), 1), 'cwri_p75': round(p75, 1),
                  'cwri_p75_over_tau': round(p75 / ix.TAU_WASP, 2),
                  'pct_wasp_active': round(float(np.nanmean(cw > ix.TAU_WASP) * 100), 1),
                  'baie_mean': round(float(np.nanmean(today['bai'])), 1),
                  'pct_hotspot_7d': round(float(np.nanmean(zc > 1.96) * 100), 1),
                  'pct_alert': round(float(np.mean(ac >= 2) * 100), 1),
                  'pct_alert_1': round(pct1, 1),
                  'pct_alert_3': round(float(np.mean(ac >= 3) * 100), 1),
                  'pct_chronic': round(float(np.mean(chronic_1d) * 100), 1),
                  'overlap30_mean': round(float(np.nanmean(overlap_days)), 1),
                  'tavg_median': round(t_med, 1),
                  'solar_median': round(float(np.nanmedian(today['wx']['srW'])), 0)},
        'national_stage': national, 'season': season, 'persistence_used': pers is not None,
        'amos_use': AMOS_USE,
        'breakpoints': {'use': bp['use'] if bp else None, 'caution': round(float(t1), 2),
                        'warning': None if not bp else bp['p90'], 'severe': None if not bp else bp['p97'],
                        'sample': bp['sample'] if bp else None},
        'solar_method': 'Solar = -16.357*Tavg + 27.847*Tmax + 25.657*Tgap - 67.734 (낮 8시간 평균 W/m²)',
        'solar_csi_unit': 'MJ/m²' if ix.SOLAR_CSI_MJ else 'W/m² (논문과 동일)',
        'phenology': pheno,
        'layers': {k: f'{k}.png' for k in list(layers) + ['gi_class', 'alert', 'chronic']},
        'alert_rule': '7일 평균 CWRI: 주의 ≥ τ_wasp 37.76(논문 2) / 경계 ≥ P90 / 심각 ≥ P97 (과거 2008~2024 가을 7일 평균 분포). 상습 지역(계절 지속성 ≥ 7/8년)은 별도 빗금. 전국 단계는 주의 이상 면적 10/40/70%',
        'warnings': warns, 'log': logs[-30:],
        'label': '운영 지수 — 논문 검증 설정과 다름, 신고 자료로 검증 중',
    }
    with open(os.path.join(LATEST, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    try:
        n_local = write_local(grid, day, cw_hist, cw_days, cw7, ac, chronic_1d, (t1, t2, t3))
        log(f'  우리 지역 자료: 시군구 {n_local}곳' + (' (평년 전망 포함)' if os.path.exists(CLIM) else ' (평년표 없음 → 전망 생략)'))
    except Exception as e:
        warns.append(f'우리 지역 자료를 만들지 못함: {e.__class__.__name__}: {e}')
    publish(day)
    log(f'완료: CWRI 평균 {meta["stats"]["cwri_mean"]}, P75 {p75:.1f} (τ {ix.TAU_WASP}), '
        f'말벌 활동 칸 {meta["stats"]["pct_wasp_active"]}%, 핫스팟 {meta["stats"]["pct_hotspot_7d"]}%, '
        f'경보 주의 {meta["stats"]["pct_alert_1"]}% · 경계 {meta["stats"]["pct_alert"]}% · 심각 {meta["stats"]["pct_alert_3"]}% '
        f'→ 전국 {national} (상습 지역 {meta["stats"]["pct_chronic"]}%)')
    for w in warns:
        print('  ⚠', w)


if __name__ == '__main__':
    main()
