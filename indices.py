# -*- coding: utf-8 -*-
"""
지수 계산부 — 말벌 논문 2(COMPAG 시스템 논문)의 정의를 따름. 기준 코드: tests/golden/paper_index_reference.py
(같은 입력이면 같은 값: python tests/golden/test_golden.py)

  CSI_s  = 100 · 8개 반응함수의 곱 (평균·최고·최저기온, 일교차, 습도, 풍속, 강수, 일사)
  CWRI   = (5·등검은말벌 + 4·장수말벌 + 3·말벌 + 2·꼬마장수말벌 + 1·좀말벌) / 15 (경쟁항 없음)
  활동창 = 5종 적산온도 출현 중 가장 이른 날 ~ 종료 중 가장 늦은 날 (종료 = 200일 이후 기준온도 아래 5일 연속의 첫날),
           창 밖은 CWRI 0
  일사   = Hargreaves–Samani Rs (FAO-56 식 50, kRs 해안 0.19·내륙 0.16, 맑은 날 일사 상한)
           → 한낮 최대 일사 π·Rs/(2N) (W m⁻²) 로 CSI·BAI_E에 넣음. CSI 일사 기준값(MJ d⁻¹)은 같은 식으로 W m⁻²로 바꿔 씀
  BAI_E  = 02_BAI_E_Raster_Pipeline compute_bai_e
  Gi*    = 06_PartA getis_ord_gi_star (7×7 칸, ±3 km)
  임계값 : τ_wasp = 44.354 (1 km 한반도, 8–10월, CWRI > 0, 2017–2024의 P75; v4 가중치·경쟁항 없음·일사 MJ 기준), τ_bee = 30

종 코드 키는 래스터 폴더 이름을 그대로 씀: V_simillima = 좀말벌 V. analis, V_analis = 꼬마장수말벌 V. ducalis.
모든 함수는 numpy 배열을 받아 칸별로 계산한다 (1km 격자 전체를 한 번에).
"""
import numpy as np
from scipy.ndimage import uniform_filter
from scipy import stats as sp_stats

# ─────────────────────────────────────────────────────────────
# 1. 반응함수
# ─────────────────────────────────────────────────────────────
def _clip01(a):
    return np.clip(a, 0.0, 1.0)

def response(x, p):
    t = p['type']
    if t == 'gauss':
        return _clip01(np.exp(-0.5 * ((x - p['opt']) / p['sigma']) ** 2))
    if t == 'gauss_weak':
        f = p.get('floor', 0.5)
        return _clip01(f + (1 - f) * np.exp(-0.5 * ((x - p['opt']) / p['sigma']) ** 2))
    if t == 'sigmoid_neg':
        return _clip01(1.0 / (1.0 + np.exp(p['k'] * (x - p['threshold']))))
    if t == 'sigmoid_pos':
        return _clip01(1.0 / (1.0 + np.exp(-p['k'] * (x - p['threshold']))))
    if t == 'exp_decay':
        return _clip01(np.exp(-p['rate'] * np.maximum(x, 0.0)))
    return np.ones_like(x)

# ─────────────────────────────────────────────────────────────
# 2. 종별 파라미터 (8변수) — 01_Wasp_Activity_Index_v3
# ─────────────────────────────────────────────────────────────
SPECIES_PARAMS = {
    'V_velutina':   {'temp': {'type': 'gauss', 'opt': 24.0, 'sigma': 8.0},
                     'humid': {'type': 'gauss', 'opt': 75.0, 'sigma': 15.0},
                     'wind': {'type': 'sigmoid_neg', 'threshold': 2.0, 'k': 3.0},
                     'precip': {'type': 'exp_decay', 'rate': 0.08},
                     'tmax': {'type': 'gauss', 'opt': 29.0, 'sigma': 8.0},
                     'tmin': {'type': 'sigmoid_pos', 'threshold': 5.0, 'k': 0.5},
                     'tgap': {'type': 'gauss', 'opt': 10.0, 'sigma': 5.0},
                     'solar': {'type': 'sigmoid_pos', 'threshold': 5.0, 'k': 0.3}},
    'V_analis':     {'temp': {'type': 'gauss', 'opt': 25.4, 'sigma': 5.0},
                     'humid': {'type': 'gauss_weak', 'opt': 72.0, 'sigma': 20.0, 'floor': 0.7},
                     'wind': {'type': 'sigmoid_neg', 'threshold': 1.8, 'k': 4.0},
                     'precip': {'type': 'exp_decay', 'rate': 0.15},
                     'tmax': {'type': 'gauss', 'opt': 30.0, 'sigma': 5.5},
                     'tmin': {'type': 'sigmoid_pos', 'threshold': 8.0, 'k': 0.5},
                     'tgap': {'type': 'gauss', 'opt': 9.0, 'sigma': 4.0},
                     'solar': {'type': 'sigmoid_pos', 'threshold': 6.0, 'k': 0.4}},
    'V_mandarinia': {'temp': {'type': 'gauss', 'opt': 26.5, 'sigma': 7.0},
                     'humid': {'type': 'gauss', 'opt': 73.0, 'sigma': 18.0},
                     'wind': {'type': 'sigmoid_neg', 'threshold': 2.8, 'k': 2.5},
                     'precip': {'type': 'exp_decay', 'rate': 0.05},
                     'tmax': {'type': 'gauss', 'opt': 31.0, 'sigma': 7.0},
                     'tmin': {'type': 'sigmoid_pos', 'threshold': 6.0, 'k': 0.4},
                     'tgap': {'type': 'gauss', 'opt': 11.0, 'sigma': 5.0},
                     'solar': {'type': 'sigmoid_pos', 'threshold': 4.0, 'k': 0.3}},
    'V_simillima':  {'temp': {'type': 'sigmoid_pos', 'threshold': 22.0, 'k': 0.3},
                     'humid': {'type': 'gauss', 'opt': 72.0, 'sigma': 14.0},
                     'wind': {'type': 'sigmoid_neg', 'threshold': 2.2, 'k': 3.0},
                     'precip': {'type': 'exp_decay', 'rate': 0.07},
                     'tmax': {'type': 'sigmoid_pos', 'threshold': 25.0, 'k': 0.3},
                     'tmin': {'type': 'sigmoid_pos', 'threshold': 12.0, 'k': 0.5},
                     'tgap': {'type': 'gauss', 'opt': 8.0, 'sigma': 4.0},
                     'solar': {'type': 'sigmoid_pos', 'threshold': 7.0, 'k': 0.4}},
    'V_crabro':     {'temp': {'type': 'gauss', 'opt': 26.5, 'sigma': 10.0},
                     'humid': {'type': 'gauss_weak', 'opt': 70.0, 'sigma': 18.0, 'floor': 0.6},
                     'wind': {'type': 'sigmoid_neg', 'threshold': 3.0, 'k': 2.0},
                     'precip': {'type': 'exp_decay', 'rate': 0.06},
                     'tmax': {'type': 'gauss', 'opt': 30.0, 'sigma': 10.0},
                     'tmin': {'type': 'sigmoid_pos', 'threshold': 4.0, 'k': 0.4},
                     'tgap': {'type': 'gauss', 'opt': 12.0, 'sigma': 6.0},
                     'solar': {'type': 'sigmoid_pos', 'threshold': 4.0, 'k': 0.25}},
}
# 논문 2 본 분석 v4 (2026-11-06): 등검은 5/15, 장수 4/15, 말벌 3/15, 꼬마장수(V_analis 키) 2/15, 좀말벌(V_simillima 키) 1/15
CWRI_WEIGHTS = {'V_velutina': 5 / 15, 'V_mandarinia': 4 / 15, 'V_crabro': 3 / 15,
                'V_analis': 2 / 15, 'V_simillima': 1 / 15}
SPECIES = ['V_velutina', 'V_mandarinia', 'V_crabro', 'V_simillima', 'V_analis']
SPECIES_KO = {'V_velutina': '등검은말벌', 'V_mandarinia': '장수말벌', 'V_crabro': '말벌',
              'V_simillima': '좀말벌', 'V_analis': '꼬마장수말벌'}
SPECIES_SCI = {'V_velutina': 'Vespa velutina', 'V_mandarinia': 'Vespa mandarinia', 'V_crabro': 'Vespa crabro',
               'V_simillima': 'Vespa analis', 'V_analis': 'Vespa ducalis'}      # 키 ≠ 학명 (래스터 폴더 이름 유지)
NATIVE = ['V_mandarinia', 'V_crabro', 'V_simillima', 'V_analis']

# ─────────────────────────────────────────────────────────────
# 일사 — Hargreaves–Samani (FAO-56 식 21·34·37·50) → 한낮 최대 일사 (W m⁻²)
# ─────────────────────────────────────────────────────────────
KRS_COASTAL, KRS_INTERIOR = 0.19, 0.16        # FAO-56 기본값 (지점별 표: data/static/krs_stations.csv)

# 종별 일사 반응함수의 기준값(threshold 4–7, k 0.25–0.4)은 하루 일사량 MJ m⁻² d⁻¹ 규모로 정해진 값.
# 입력은 한낮 최대 일사(W m⁻²)라서 그대로 쓰면 반응값이 늘 1이 됨.
# True: 기준값을 칸·날마다 같은 변환식으로 W m⁻²로 바꿈  thr_W = thr_MJ·f,  k_W = k_MJ / f,  f = π·10⁶ / (2·N·3600)
#       (= Rs(MJ)를 넣는 것과 수학적으로 같음; N = 낮 길이, wx['N'] 필요)
# 논문 2 본 분석으로 확정 (2026-11-06). False = 예전(포화) 정의, 골든 v1 비교용
SOLAR_THRESHOLD_MJ = True


def mj_to_peak_factor(N):
    """하루 일사 1 MJ m⁻² d⁻¹ → 한낮 최대 W m⁻² (반사인 일변화). N=12시간이면 36.4"""
    return np.pi * 1e6 / (2 * np.asarray(N, float) * 3600)


def _solar_decl(doy):
    return 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)


def extraterrestrial(lat_deg, doy):
    """대기 밖 일사 Ra (MJ m⁻² d⁻¹), FAO-56 식 21. lat·doy는 같은 모양으로 맞춰 줄 것."""
    phi = np.radians(np.asarray(lat_deg, float))
    J = np.asarray(doy, float)
    dr = 1 + 0.033 * np.cos(2 * np.pi * J / 365)
    dl = _solar_decl(J)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(dl), -1, 1))
    return 24 * 60 / np.pi * 0.0820 * dr * (ws * np.sin(phi) * np.sin(dl) + np.cos(phi) * np.cos(dl) * np.sin(ws))


def daylength(lat_deg, doy):
    """낮 길이 N (시간), FAO-56 식 34"""
    phi = np.radians(np.asarray(lat_deg, float))
    return 24 / np.pi * np.arccos(np.clip(-np.tan(phi) * np.tan(_solar_decl(np.asarray(doy, float))), -1, 1))


def hargreaves_rs(tmax, tmin, ra, krs, elev):
    """하루 일사 Rs (MJ m⁻² d⁻¹), FAO-56 식 50, 맑은 날 일사 Rso(식 37)로 상한. 고도 결측은 해발 0."""
    rs = np.asarray(krs, float) * np.sqrt(np.maximum(tmax - tmin, 0)) * ra
    z = np.nan_to_num(np.asarray(elev, float), nan=0.0)
    return np.minimum(rs, (0.75 + 2e-5 * z) * ra)


def peak_wm2(rs_mj, lat_deg, doy):
    """하루 일사(MJ) → 한낮 최대 일사 (반사인 일변화): π·Rs·10⁶ / (2·N·3600)"""
    return np.pi * rs_mj * 1e6 / (2 * daylength(lat_deg, doy) * 3600)


def solar_peak(tmax, tmin, lat_deg, elev, krs, doy):
    """기온 → 한낮 최대 일사 (W m⁻²). CSI·BAI_E의 Solar 입력."""
    ra = extraterrestrial(lat_deg, np.full(np.shape(lat_deg), doy) if np.isscalar(doy) else doy)
    return peak_wm2(hargreaves_rs(tmax, tmin, ra, krs, elev), lat_deg, doy)


# 임계값 (논문 2 확정값)
TAU_WASP = 44.354      # 1 km 한반도, 8–10월, CWRI > 0, 2017–2024 P75 (논문 2 v4: 가중치 5:4:3:2:1/15, 일사 MJ 기준)
TAU_BEE = 30.0

# ─────────────────────────────────────────────────────────────
# 3. 적산온도 활동창 — wasp_phenology_model.py / F2_degree_day_params
# ─────────────────────────────────────────────────────────────
PHENOLOGY_PARAMS = {
    'V_analis':     {'T_base': 2.0, 'R_h_emerge': 810.0, 'mean_emerge_doy': 130, 'mean_end_doy': 300},
    'V_velutina':   {'T_base': 8.0, 'R_h_emerge': 165.0, 'mean_emerge_doy': 109, 'mean_end_doy': 331},
    'V_crabro':     {'T_base': 3.0, 'R_h_emerge': 324.0, 'mean_emerge_doy': 98,  'mean_end_doy': 300},
    'V_mandarinia': {'T_base': 3.0, 'R_h_emerge': 453.0, 'mean_emerge_doy': 110, 'mean_end_doy': 312},
    'V_simillima':  {'T_base': 3.0, 'R_h_emerge': 336.0, 'mean_emerge_doy': 99,  'mean_end_doy': 290},
}
N_CONSEC_TERMINATE = 5
USE_TERMINATION = True    # 논문 2: 200일 이후 기준온도 아래 5일 연속이면 종료
TERMINATE_SEARCH_START_DOY = 200


def phenology_windows(tavg_stack, doys):
    """
    1월 1일부터의 일별 Tavg 격자(list of 1D arrays, 격자 유효 칸만) → 종별 (출현 DOY, 종료 DOY)
    predict_emergence_doy / predict_termination_doy 와 같은 규칙.
    미출현 = NaN, 미종료 = 365.
    """
    out = {}
    n = tavg_stack[0].shape[0]
    for sp, p in PHENOLOGY_PARAMS.items():
        cum = np.zeros(n)
        emerge = np.full(n, np.nan, dtype=np.float32)
        streak = np.zeros(n, dtype=np.int32)
        term = np.full(n, np.nan, dtype=np.float32)
        for t, d in zip(tavg_stack, doys):
            v = ~np.isnan(t)
            ne = v & np.isnan(emerge)
            cum[ne] += np.maximum(t[ne] - p['T_base'], 0.0)
            hit = ne & (cum >= p['R_h_emerge'])
            emerge[hit] = d
            if USE_TERMINATION and d >= TERMINATE_SEARCH_START_DOY:
                nt = v & np.isnan(term)
                below = nt & (t < p['T_base'])
                streak[below] += 1
                streak[nt & ~below] = 0
                done = nt & (streak >= N_CONSEC_TERMINATE)
                term[done] = d - N_CONSEC_TERMINATE + 1
        term[np.isnan(term)] = 365.0
        out[sp] = (emerge, term, cum)
    return out


def activity_window(windows):
    """5종 활동창 → 하나의 창 (가장 이른 출현, 가장 늦은 종료). 아무 종도 출현 전이면 NaN."""
    e = np.vstack([np.where(np.isnan(windows[sp][0]), np.inf, windows[sp][0]) for sp in SPECIES]).min(axis=0)
    t = np.vstack([windows[sp][1] for sp in SPECIES]).max(axis=0)          # 미종료 = 365 (논문 코드와 같음)
    start = np.where(np.isfinite(e), e, np.nan)
    return start, t


def active_mask(window, doy):
    """window = (출현, 종료[, ...]) — 종별 창이든 합친 창이든 같은 규칙"""
    emerge, term = window[0], window[1]
    return (~np.isnan(emerge)) & (doy >= emerge) & (doy <= term)

# ─────────────────────────────────────────────────────────────
# 4. CSI / CWRI
# ─────────────────────────────────────────────────────────────
def compute_csi(sp, wx):
    """wx: dict of arrays t, tx, tn, rh, ws, pr, srW (한낮 최대 일사 W m⁻²), 선택 tg(일교차). 반환 0~100"""
    P = SPECIES_PARAMS[sp]
    tgap = wx['tg'] if 'tg' in wx else wx['tx'] - wx['tn']
    m = {'temp': wx['t'], 'humid': wx['rh'], 'wind': wx['ws'], 'precip': wx['pr'],
         'tmax': wx['tx'], 'tmin': wx['tn'], 'tgap': tgap, 'solar': wx['srW']}
    prod = np.ones_like(wx['t'], dtype=np.float64)
    for key, par in P.items():
        if key == 'solar' and SOLAR_THRESHOLD_MJ:              # 기준값 MJ d⁻¹ → W m⁻² (칸·날마다)
            f = mj_to_peak_factor(wx['N'])
            par = {**par, 'threshold': par['threshold'] * f, 'k': par['k'] / f}
        prod *= response(m[key], par)
    return prod * 100.0


def compute_cwri(wx, window, doy):
    """window = activity_window(...) 의 (시작, 끝) — 창 밖은 CWRI·종별 CSI 모두 0. window=None이면 창 없이.
    경쟁항 없음 (논문 1 R1: 등검은말벌의 토착종 억제 근거 없음). 반환 (cwri, {sp: csi})"""
    act = active_mask(window, doy) if window is not None else np.ones(np.shape(wx['t']), bool)
    per = {}
    cwri = np.zeros_like(wx['t'], dtype=np.float64)
    for sp in SPECIES:
        csi = np.where(act, compute_csi(sp, wx), 0.0)
        per[sp] = csi
        cwri += CWRI_WEIGHTS[sp] * csi
    cwri = np.where(np.isnan(wx['t']), np.nan, np.clip(cwri, 0, 100))
    return cwri, per

# ─────────────────────────────────────────────────────────────
# 5. BAI_E — 02_BAI_E_Raster_Pipeline
# ─────────────────────────────────────────────────────────────
def _bell(x, opt, sL, sR):
    return np.where(x <= opt,
                    np.exp(-0.5 * ((x - opt) / max(sL, .01)) ** 2),
                    np.exp(-0.5 * ((x - opt) / max(sR, .01)) ** 2))


def compute_bai_e(T, RH, WS, Rain, SolarWm2):
    """SolarWm2 = 한낮 최대 일사 (W m⁻²)"""
    g = (1 / (1 + np.exp((T - 40) / 1.25))) * (1 / (1 + np.exp(-(T - 5) / 0.75))) \
        * (1 / (1 + np.exp((Rain - 15) / 1.25))) * (1 / (1 + np.exp((WS - 10) / 0.5)))
    ti = _bell(T, 26.6, 7.82, 5.97)
    k = np.log(2) / 10.0
    ri = np.where(Rain > 0, _clip01(np.exp(-k * Rain)), 1.0)
    si = 1 / (1 + np.exp(-0.01 * (SolarWm2 - 290)))
    si = np.where(SolarWm2 > 378, si * np.exp(-0.003 * (SolarWm2 - 378)), si)
    wi = np.where(WS <= 3.0, 1.0, _clip01(1 - (WS - 3.0) / 13.0))
    hi = _bell(RH, 61.5, 20, 20)
    bai = (0.140 * ti + 0.119 * ri + 0.343 * si + 0.221 * wi + 0.176 * hi) * 100 * g
    return np.clip(bai, 0, 100)

# ─────────────────────────────────────────────────────────────
# 6. Gi* — 06_PartA getis_ord_gi_star (반경 3칸 = 7×7 km 이동창)
# ─────────────────────────────────────────────────────────────
GI_RADIUS = 3


def getis_ord_gi_star(raster, radius=GI_RADIUS):
    ks = 2 * radius + 1
    valid = ~np.isnan(raster)
    x = np.where(valid, raster, 0.0).astype(np.float64)
    w = valid.astype(np.float64)
    n = float(np.sum(valid))
    if n < 10:
        return np.full_like(raster, np.nan), np.full_like(raster, np.nan)
    Xb = np.sum(x) / n
    S = np.sqrt(np.sum(x ** 2) / n - Xb ** 2)
    if S < 1e-10:
        return np.full_like(raster, np.nan), np.full_like(raster, np.nan)
    ks2 = float(ks ** 2)
    swx = uniform_filter(x, size=ks, mode='constant', cval=0.0) * ks2
    sw = uniform_filter(w, size=ks, mode='constant', cval=0.0) * ks2
    num = swx - Xb * sw
    dsq = np.maximum((n * sw - sw ** 2) / (n - 1), 1e-10)
    z = np.where(S * np.sqrt(dsq) > 0, num / (S * np.sqrt(dsq)), 0.0)
    z[~valid] = np.nan
    p = np.full_like(z, np.nan)
    vz = ~np.isnan(z)
    p[vz] = 2.0 * (1.0 - sp_stats.norm.cdf(np.abs(z[vz])))
    return z.astype(np.float32), p.astype(np.float32)


def classify_hotspot(z):
    c = np.zeros_like(z, dtype=np.int8)
    c[z > 2.58] = 3
    c[(z > 1.96) & (z <= 2.58)] = 2
    c[(z > 1.65) & (z <= 1.96)] = 1
    c[z < -1.96] = -1
    c[np.isnan(z)] = -9
    return c


# ─────────────────────────────────────────────────────────────
# 2주 전망 (시군구 7일 평균 CWRI) — data/static/outlook.json 의 decay·days
# ─────────────────────────────────────────────────────────────
def outlook(cw7_today, clim_today, clim_future, decay=0.8):
    """오늘 평년 편차가 하루에 decay 배씩 줄어 평년으로 돌아감.
    clim_future[k-1] = k일 뒤 평년(7일 평균 CWRI) → [k일 뒤 전망], 0–100으로 자름"""
    a0 = cw7_today - clim_today
    k = np.arange(1, len(clim_future) + 1)
    return np.clip(np.asarray(clim_future, float) + a0 * decay ** k, 0, 100)
