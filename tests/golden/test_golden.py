# -*- coding: utf-8 -*-
"""
앱 지수 코드(indices.py)가 논문 2 기준 코드와 같은 값을 내는지 확인

  python tests/golden/test_golden.py

1) golden_test_vectors.csv (244행): 입력 8변수 → CSI 5종·CWRI·BAI_E   (허용 차이 1e-6)
2) golden_phenology_*.csv: 적산온도 출현·종료·합친 활동창 (정확히 일치)
3) Hargreaves 일사: 앱 함수 vs 기준 코드 함수 (임의 입력 5,000개, 허용 1e-9)
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
import indices as ix                      # noqa: E402
import paper_index_reference as ref       # noqa: E402

TOL = 1e-6
ok_all = True


def report(name, diff, tol=TOL):
    global ok_all
    ok = diff < tol
    ok_all &= ok
    print(f'  {name:28s} max |차이| = {diff:.2e}  {"OK" if ok else "MISMATCH"}')


print('1) 지수 골든 벡터')
df = pd.read_csv(os.path.join(HERE, 'golden_test_vectors.csv'))
wx = {'t': df.Tavg.values, 'tx': df.Tmax.values, 'tn': df.Tmin.values, 'tg': df.Tgap.values,
      'rh': df.HMDT.values, 'ws': df.WDSP.values, 'pr': df.Rn.values, 'srW': df.Solar.values}
cw, per = ix.compute_cwri(wx, None, 200)
for sp in ix.SPECIES:
    report(f'CSI_{sp}', float(np.abs(per[sp] - df[f'CSI_{sp}']).max()))
report('CWRI', float(np.abs(cw - df.CWRI).max()))
bai = ix.compute_bai_e(df.Tavg.values, df.HMDT.values, df.WDSP.values, df.Rn.values, df.Solar.values)
report('BAI_E', float(np.abs(bai - df.BAI_E).max()))
print(f'  ({len(df)}행)')

print('2) 적산온도 활동창')
T = pd.read_csv(os.path.join(HERE, 'golden_phenology_tavg.csv'))
E = pd.read_csv(os.path.join(HERE, 'golden_phenology_expected.csv')).set_index('station')
st = [c for c in T.columns if c != 'doy']
W = ix.phenology_windows([T.loc[i, st].values.astype(float) for i in range(len(T))], T['doy'].values)
start, end = ix.activity_window(W)
bad = 0
for sp in ix.SPECIES:
    bad += int(np.sum(W[sp][0] != E.loc[st, f'emerge_{sp}'].values)) + int(np.sum(W[sp][1] != E.loc[st, f'term_{sp}'].values))
bad += int(np.sum(start != E.loc[st, 'window_start'].values)) + int(np.sum(end != E.loc[st, 'window_end'].values))
n = len(st) * (2 * len(ix.SPECIES) + 2)
ok_all &= bad == 0
print(f'  지점 {len(st)}곳 × (종 5 × 출현·종료 + 창 시작·끝) = {n}값 중 불일치 {bad}  {"OK" if bad == 0 else "MISMATCH"}')
for s, a, b in zip(st, start, end):
    print(f'    {s:14s} 활동창 {a:.0f}–{b:.0f}일')

print('3) Hargreaves 일사 (한낮 최대 W m⁻²)')
rng = np.random.default_rng(42)
k = 5000
lat = rng.uniform(33, 39, k); doy = rng.integers(1, 366, k); elev = rng.uniform(0, 1500, k)
tmin = rng.uniform(-15, 25, k); tmax = tmin + rng.uniform(0, 18, k); krs = rng.choice([0.16, 0.19], k)
app = ix.solar_peak(tmax, tmin, lat, elev, krs, doy)
ra = np.array([ref.extraterrestrial([la], [d])[0, 0] for la, d in zip(lat, doy)])
rs = np.array([ref.hargreaves_rs(np.array([[a]]), np.array([[b]]), np.array([[r]]), np.array([c]), np.array([e]))[0, 0]
               for a, b, r, c, e in zip(tmax, tmin, ra, krs, elev)])
want = np.array([ref.to_raster_unit(np.array([[m]]), [la], [d], 'Wm2_peak')[0, 0] for m, la, d in zip(rs, lat, doy)])
report('Solar (W m-2 peak)', float(np.abs(app - want).max()), 1e-9)
print(f'  (예: 전주 35.8°N, 8월 15일, 최고 31 최저 23, kRs 0.16 → {float(ix.solar_peak(31., 23., 35.8, 50., 0.16, 227)):.0f} W m⁻²)')

print('결과:', 'OK — 앱 코드가 논문 기준 코드와 같음' if ok_all else 'MISMATCH')
sys.exit(0 if ok_all else 1)
