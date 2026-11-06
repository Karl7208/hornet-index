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


# 논문 최종 골든(v3: v4 가중치 + 일사 MJ 기준)을 받으면 이 이름으로 둠. 없으면 v1로 CSI·BAI_E를 보고,
# CWRI는 v1의 종별 CSI에 지금 가중치를 곱해 더한 값과 비교 (가중합 확인)
V2 = next((os.path.join(HERE, f) for f in ('golden_test_vectors_v3.csv', 'golden_test_vectors_v2.csv')
           if os.path.exists(os.path.join(HERE, f))), '')
use_v2 = bool(V2)
print('1) 지수 골든 벡터', f'({os.path.basename(V2)}: 운영 설정)' if use_v2 else '(v1: 일사 변환 끄고 CSI·BAI_E 비교, CWRI는 지금 가중치로 가중합 확인 — v3 대기 중)')
df = pd.read_csv(V2 if use_v2 else os.path.join(HERE, 'golden_test_vectors.csv'))
wx = {'t': df.Tavg.values, 'tx': df.Tmax.values, 'tn': df.Tmin.values, 'tg': df.Tgap.values,
      'rh': df.HMDT.values, 'ws': df.WDSP.values, 'pr': df.Rn.values, 'srW': df.Solar.values}
if use_v2:
    if 'N' in df.columns:
        wx['N'] = df['N'].values
    else:
        wx['N'] = ix.daylength(df['lat'].values, df['doy'].values)
saved = ix.SOLAR_THRESHOLD_MJ
ix.SOLAR_THRESHOLD_MJ = use_v2 and saved
cw, per = ix.compute_cwri(wx, None, 200)
ix.SOLAR_THRESHOLD_MJ = saved
for sp in ix.SPECIES:
    report(f'CSI_{sp}', float(np.abs(per[sp] - df[f'CSI_{sp}']).max()))
cw_ref = df.CWRI.values if use_v2 else np.clip(sum(ix.CWRI_WEIGHTS[sp] * df[f'CSI_{sp}'].values for sp in ix.SPECIES), 0, 100)
report('CWRI' + ('' if use_v2 else ' (v4 가중합)'), float(np.abs(cw - cw_ref).max()))
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

print('4) 일사 기준값 변환 (켬) = 하루 일사량 MJ 를 넣은 것')
n = 3000
lat = rng.uniform(33, 38.6, n); doy = rng.integers(60, 330, n); tx = rng.uniform(5, 36, n); tn = tx - rng.uniform(0.5, 16, n)
sw = ix.solar_peak(tx, tn, lat, 100., 0.16, doy); N = ix.daylength(lat, doy); mj = sw * 2 * N * 3600 / (np.pi * 1e6)
w4 = {'t': (tx + tn) / 2, 'tx': tx, 'tn': tn, 'rh': np.full(n, 70.), 'ws': np.full(n, 1.5), 'pr': np.zeros(n), 'srW': sw, 'N': N}
saved = ix.SOLAR_THRESHOLD_MJ
ix.SOLAR_THRESHOLD_MJ = True;  a4 = {sp: ix.compute_csi(sp, w4) for sp in ix.SPECIES}
ix.SOLAR_THRESHOLD_MJ = False; b4 = {sp: ix.compute_csi(sp, {**w4, 'srW': mj}) for sp in ix.SPECIES}
ix.SOLAR_THRESHOLD_MJ = saved
report('CSI (W 기준값 = MJ 입력)', max(float(np.abs(a4[sp] - b4[sp]).max()) for sp in ix.SPECIES), 1e-9)
print(f'  운영 설정: SOLAR_THRESHOLD_MJ = {ix.SOLAR_THRESHOLD_MJ}, τ_wasp = {ix.TAU_WASP}, 가중치 = ' + ', '.join(f'{sp} {ix.CWRI_WEIGHTS[sp] * 15:.0f}/15' for sp in ix.SPECIES))

print('결과:', 'OK — 앱 코드가 논문 기준 코드와 같음' if ok_all else 'MISMATCH')
sys.exit(0 if ok_all else 1)
