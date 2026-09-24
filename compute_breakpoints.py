# -*- coding: utf-8 -*-
"""
경보 단계 경계값 계산 — 노트북 02 determine_tau_wasp 와 같은 표본

  python compute_breakpoints.py "E:\\Bee_index\\말벌\\Present\\CWRI"

표본 (τ_wasp를 정할 때와 동일)
  - 2008~2024년 8~10월 날짜 중 random.seed(42)로 30일 추출
  - 각 날짜 CWRI_YYYYMMDD.TIF 에서 NaN·nodata·0 을 뺀 값, 날짜당 최대 10,000개
  - P75 가 논문의 τ_wasp 37.76 과 비슷하게 나오면 같은 표본이 재현된 것

두 가지를 함께 계산
  daily : 하루 CWRI 분포 (τ_wasp와 같은 기준) ← 기본으로 사용
  7day  : 그날 포함 7일 평균 CWRI 분포 (경보가 7일 평균에 걸리므로 참고용)

결과: data/static/breakpoints.json  → run_daily.py 가 읽어 주의/경계/심각 기준으로 사용
"""
import os
import sys
import json
import random
import datetime as dt
from calendar import monthrange
import warnings
import numpy as np
import rasterio
warnings.filterwarnings('ignore', category=RuntimeWarning)

YEARS = range(2008, 2025)
MONTHS = (8, 9, 10)
N_DAYS = 30
PER_DAY = 10000
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'static', 'breakpoints.json')


def find(d, day):
    for ext in ('.TIF', '.tif'):
        p = os.path.join(d, f'CWRI_{day:%Y%m%d}{ext}')
        if os.path.exists(p):
            return p
    return None


def read(p):
    with rasterio.open(p) as r:
        a = r.read(1).astype('float64')
        if r.nodata is not None:
            a[a == r.nodata] = np.nan
    a[a < -1000] = np.nan
    return a


def main():
    cdir = sys.argv[1]
    random.seed(42)
    cand = [(y, m, d) for y in YEARS for m in MONTHS for d in range(1, monthrange(y, m)[1] + 1)]
    cand = random.sample(cand, N_DAYS)
    rng = np.random.default_rng(42)

    daily, week, used = [], [], 0
    for y, m, d in cand:
        day = dt.date(y, m, d)
        p = find(cdir, day)
        if p is None:
            continue
        a = read(p)
        v = a[~np.isnan(a) & (a > 0)]
        if v.size:
            daily.append(rng.choice(v, min(PER_DAY, v.size), replace=False))
            used += 1
        stack = [a]
        for k in range(1, 7):
            q = find(cdir, day - dt.timedelta(days=k))
            if q:
                stack.append(read(q))
        if len(stack) >= 5:
            w = np.nanmean(np.stack(stack), axis=0)
            wv = w[~np.isnan(w) & (w > 0)]
            if wv.size:
                week.append(rng.choice(wv, min(PER_DAY, wv.size), replace=False))

    if not daily:
        raise SystemExit(f'{cdir} 에서 CWRI_YYYYMMDD.TIF 를 찾지 못했습니다.')
    res = {}
    for name, arr in (('daily', daily), ('7day', week)):
        if not arr:
            continue
        x = np.concatenate(arr)
        res[name] = {f'p{q}': round(float(np.percentile(x, q)), 2) for q in (50, 75, 90, 97)}
        res[name]['n_values'] = int(x.size)
        print(f'{name:6s} P50 {res[name]["p50"]:6.2f}  P75 {res[name]["p75"]:6.2f}  '
              f'P90 {res[name]["p90"]:6.2f}  P97 {res[name]["p97"]:6.2f}  ({x.size:,}개 값)')
    print(f'사용한 날짜 {used}/{N_DAYS}일. 논문 τ_wasp = 37.76 과 daily P75 를 비교하세요.')

    res['use'] = 'daily'
    res['sample'] = f'{YEARS.start}~{YEARS.stop - 1}년 {MONTHS} 중 {N_DAYS}일, seed 42, CWRI>0'
    res['source_dir'] = cdir
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f'저장: {OUT}  (7일 평균 기준을 쓰려면 파일의 "use" 를 "7day" 로 바꾸면 됨)')


if __name__ == '__main__':
    main()
