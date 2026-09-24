# -*- coding: utf-8 -*-
"""
지수 계산부 — Karl의 기존 코드를 그대로 옮긴 것 (새로 만든 식 없음)

  CSI / CWRI      : 01_Wasp_Activity_Index_v3 (말벌_위험진단.html JS 포팅과 동일)
  BAI_E           : 02_BAI_E_Raster_Pipeline compute_bai_e
  적산온도 활동창  : wasp_phenology_model.py (F2_degree_day_params)
  Gi*             : 06_PartA_Phenology_GiStar_EarlyWarning getis_ord_gi_star / classify_hotspot
  임계값          : τ_wasp = 37.76, τ_bee = 30 (논문 2 확정값)

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
CWRI_WEIGHTS = {'V_velutina': 0.35, 'V_mandarinia': 0.25, 'V_crabro': 0.20,
                'V_simillima': 0.12, 'V_analis': 0.08}
SPECIES = ['V_velutina', 'V_mandarinia', 'V_crabro', 'V_simillima', 'V_analis']
SPECIES_KO = {'V_velutina': '등검은말벌', 'V_mandarinia': '장수말벌', 'V_crabro': '말벌',
              'V_simillima': '좀말벌', 'V_analis': '꼬마장수말벌'}

# 일사 경험식 (Karl, 기온 → 일사). 노트북 01·02와 같이 W/m² 규모 값을 CSI·BAI_E에 변환 없이 넣음
def solar_from_temp(tavg, tmax, tmin):
    tgap = tmax - tmin
    return np.maximum(-16.357 * tavg + 27.847 * tmax + 25.657 * tgap - 67.734, 0.0)


# CSI의 일사 기준값(4~7)은 MJ/m²/일 규모. True로 두면 CSI에만 MJ로 환산해 넣음 (BAI_E는 W/m² 그대로)
# 여기서 W/m² = 하루 일사량을 낮 8시간에 나눈 값 → MJ = W/m² ÷ 34.72 (전주 일자료 MJ·W 칸 비율로 확인)
SOLAR_CSI_MJ = False
W_PER_MJ = 1e6 / 28800.0   # 34.72


# 임계값 (논문 2 확정값)
TAU_WASP = 37.76
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
USE_TERMINATION = False   # 운영판: 가을 종료일 고려 안 함 (출현 후 연말까지 열어 두고 추위는 CSI가 반영)
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


def active_mask(window, doy):
    emerge, term, _ = window
    return (~np.isnan(emerge)) & (doy >= emerge) & (doy <= term)

# ─────────────────────────────────────────────────────────────
# 4. CSI / CWRI
# ─────────────────────────────────────────────────────────────
def compute_csi(sp, wx):
    """wx: dict of arrays t, tx, tn, rh, ws, pr, srW (W/m²). 반환 0~100"""
    P = SPECIES_PARAMS[sp]
    solar = wx['srW'] / W_PER_MJ if SOLAR_CSI_MJ else wx['srW']
    m = {'temp': wx['t'], 'humid': wx['rh'], 'wind': wx['ws'], 'precip': wx['pr'],
         'tmax': wx['tx'], 'tmin': wx['tn'], 'tgap': wx['tx'] - wx['tn'], 'solar': solar}
    prod = np.ones_like(wx['t'], dtype=np.float64)
    for key, par in P.items():
        prod *= response(m[key], par)
    return prod * 100.0


def compute_cwri(wx, windows, doy):
    """활동창 밖의 종은 0 → 가중합. 반환 (cwri, {sp: csi})"""
    per = {}
    cwri = np.zeros_like(wx['t'], dtype=np.float64)
    for sp in SPECIES:
        csi = compute_csi(sp, wx)
        csi = np.where(active_mask(windows[sp], doy), csi, 0.0)
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
