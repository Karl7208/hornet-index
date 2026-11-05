# -*- coding: utf-8 -*-
"""
Reference implementation of the hornet and honeybee indices as defined in Wasp Paper 2
(COMPAG system paper, 2026-11). The app must reproduce these functions; golden_test_vectors.csv
holds inputs and expected outputs (check with:  python paper_index_reference.py --test).

Index definition used by the paper (main analysis):
  * CWRI = 0.35·CSI_velutina + 0.25·CSI_mandarinia + 0.20·CSI_crabro + 0.12·CSI_analis + 0.08·CSI_ducalis
    (no competition term; Paper 1 R1 found no suppression of native hornets by V. velutina)
  * CSI_s = 100 · Π of eight response functions (Tavg, RH, wind, precip, Tmax, Tmin, Tgap, Solar)
  * Activity window: earliest degree-day emergence of the five species to latest termination; CWRI = 0 outside
  * tau_wasp = 44.905 = 75th percentile of positive Aug–Oct CWRI, 1-km peninsula grid, 2017–2024
  * Solar enters both indices as midday peak irradiance (W m-2) = pi·Rs/(2N); Rs from Hargreaves–Samani
Internal species keys keep the raster folder names:
  V_analis = V. ducalis (꼬마장수말벌), V_simillima = V. analis (좀말벌).
"""
import argparse
import numpy as np


class C:                                   # constants that the paper code reads from config.py
    SOLAR_UNIT = 'Wm2_peak'
    TERMINATION_CONSEC_DAYS = 5
    TERMINATION_SEARCH_START_DOY = 200
    CWRI_SPECIES = ['V_velutina', 'V_analis', 'V_mandarinia', 'V_simillima', 'V_crabro']
    PHENOLOGY = {'V_analis': {'T_base': 2.0, 'R_h': 810.0}, 'V_velutina': {'T_base': 8.0, 'R_h': 165.0}, 'V_crabro': {'T_base': 3.0, 'R_h': 324.0}, 'V_mandarinia': {'T_base': 3.0, 'R_h': 453.0}, 'V_simillima': {'T_base': 3.0, 'R_h': 336.0}}


SPECIES_LABEL = {'V_velutina': 'V. velutina', 'V_mandarinia': 'V. mandarinia', 'V_crabro': 'V. crabro', 'V_analis': 'V. ducalis', 'V_simillima': 'V. analis'}
TAU_WASP = 44.905
TAU_BEE = 30.0

SPECIES_PARAMS = {'V_velutina': {'temp': ('gauss', 24.0, 8.0), 'humid': ('gauss', 75.0, 15.0), 'wind': ('sigmoid_neg', 2.0, 3.0), 'precip': ('exp_decay', 0.08), 'tmax': ('gauss', 29.0, 8.0), 'tmin': ('sigmoid_pos', 5.0, 0.5), 'tgap': ('gauss', 10.0, 5.0), 'solar': ('sigmoid_pos', 5.0, 0.3)},
    'V_analis': {'temp': ('gauss', 25.4, 5.0), 'humid': ('gauss_weak', 72.0, 20.0, 0.7), 'wind': ('sigmoid_neg', 1.8, 4.0), 'precip': ('exp_decay', 0.15), 'tmax': ('gauss', 30.0, 5.5), 'tmin': ('sigmoid_pos', 8.0, 0.5), 'tgap': ('gauss', 9.0, 4.0), 'solar': ('sigmoid_pos', 6.0, 0.4)},
    'V_mandarinia': {'temp': ('gauss', 26.5, 7.0), 'humid': ('gauss', 73.0, 18.0), 'wind': ('sigmoid_neg', 2.8, 2.5), 'precip': ('exp_decay', 0.05), 'tmax': ('gauss', 31.0, 7.0), 'tmin': ('sigmoid_pos', 6.0, 0.4), 'tgap': ('gauss', 11.0, 5.0), 'solar': ('sigmoid_pos', 4.0, 0.3)},
    'V_simillima': {'temp': ('sigmoid_pos', 22.0, 0.3), 'humid': ('gauss', 72.0, 14.0), 'wind': ('sigmoid_neg', 2.2, 3.0), 'precip': ('exp_decay', 0.07), 'tmax': ('sigmoid_pos', 25.0, 0.3), 'tmin': ('sigmoid_pos', 12.0, 0.5), 'tgap': ('gauss', 8.0, 4.0), 'solar': ('sigmoid_pos', 7.0, 0.4)},
    'V_crabro': {'temp': ('gauss', 26.5, 10.0), 'humid': ('gauss_weak', 70.0, 18.0, 0.6), 'wind': ('sigmoid_neg', 3.0, 2.0), 'precip': ('exp_decay', 0.06), 'tmax': ('gauss', 30.0, 10.0), 'tmin': ('sigmoid_pos', 4.0, 0.4), 'tgap': ('gauss', 12.0, 6.0), 'solar': ('sigmoid_pos', 4.0, 0.25)}}

PARAM_TO_VAR = {'temp': 'Tavg', 'humid': 'HMDT', 'wind': 'WDSP', 'precip': 'Rn', 'tmax': 'Tmax', 'tmin': 'Tmin', 'tgap': 'Tgap', 'solar': 'Solar'}

INTERACTION_BETA = {'V_analis': 0.0, 'V_mandarinia': 0.0, 'V_simillima': 0.0, 'V_crabro': 0.0}   # main analysis

CWRI_WEIGHTS = {'V_velutina': 0.35, 'V_mandarinia': 0.25, 'V_crabro': 0.2, 'V_simillima': 0.12, 'V_analis': 0.08}

NATIVE = ['V_analis', 'V_mandarinia', 'V_simillima', 'V_crabro']


def _response(x, p):
    t = p[0]
    with np.errstate(over='ignore', invalid='ignore'):
        if t == 'gauss':
            r = np.exp(-0.5 * ((x - p[1]) / p[2]) ** 2)
        elif t == 'gauss_weak':
            r = p[3] + (1 - p[3]) * np.exp(-0.5 * ((x - p[1]) / p[2]) ** 2)
        elif t == 'sigmoid_neg':
            r = 1.0 / (1.0 + np.exp(p[2] * (x - p[1])))
        elif t == 'sigmoid_pos':
            r = 1.0 / (1.0 + np.exp(-p[2] * (x - p[1])))
        elif t == 'exp_decay':
            r = np.exp(-p[1] * np.maximum(x, 0))
        else:
            r = np.ones_like(x)
    return np.clip(r, 0, 1)


def cwri(w):
    """w: dict of (station, day) arrays with keys Tavg Tmax Tmin Tgap HMDT WDSP Rn Solar.
    The V. velutina z-score of the IAI is taken across stations on each day
    (the rasters used all peninsula pixels of the day)."""
    csi = {}
    for sp, prm in SPECIES_PARAMS.items():
        prod = np.ones_like(w['Tavg'], dtype=np.float64)
        for k, v in PARAM_TO_VAR.items():
            prod *= _response(w[v], prm[k])
        csi[sp] = prod * 100.0
    vel = csi['V_velutina']
    mu = np.nanmean(vel, axis=0, keepdims=True)
    sd = np.nanstd(vel, axis=0, keepdims=True)
    sd = np.where(sd < 0.01, 1.0, sd)
    z = (vel - mu) / sd
    out = CWRI_WEIGHTS['V_velutina'] * vel
    for sp in NATIVE:
        out += CWRI_WEIGHTS[sp] * np.clip(csi[sp] * np.exp(INTERACTION_BETA[sp] * z), 0, 100)
    out = np.clip(out, 0, 100)
    nan = np.zeros_like(out, bool)
    for v in PARAM_TO_VAR.values():
        nan |= np.isnan(w[v])
    out[nan] = np.nan
    return out


def _bell(x, opt, sl, sr):
    return np.where(x <= opt, np.exp(-0.5 * ((x - opt) / sl) ** 2), np.exp(-0.5 * ((x - opt) / sr) ** 2))


def bai_e(w):
    T, RH, WS, P, S = w['Tavg'], w['HMDT'], w['WDSP'], w['Rn'], w['Solar']
    with np.errstate(over='ignore', invalid='ignore'):
        gate = (1 / (1 + np.exp((T - 40.0) / 1.25)) * 1 / (1 + np.exp(-(T - 5.0) / 0.75))
                * 1 / (1 + np.exp((P - 15.0) / 1.25)) * 1 / (1 + np.exp((WS - 10.0) / 0.5)))
        ti = _bell(T, 26.6, 7.82, 5.97)
        ri = np.where(P > 0, np.clip(np.exp(-np.log(2) / 10.0 * P), 0, 1), 1.0)
        si = 1 / (1 + np.exp(-0.01 * (S - 290))) * np.where(S > 378, np.exp(-0.003 * (S - 378)), 1.0)
        wi = np.where(WS <= 3.0, 1.0, np.clip(1.0 - (WS - 3.0) / 13.0, 0, 1))
        hi = _bell(RH, 61.5, 20, 20)
        b = (0.140 * ti + 0.119 * ri + 0.343 * si + 0.221 * wi + 0.176 * hi) * 100.0 * gate
    b = np.clip(b, 0, 100)
    b[np.isnan(T) | np.isnan(RH)] = np.nan
    return b


def extraterrestrial(lat_deg, doy):
    """FAO-56 eq. 21, MJ m-2 d-1. lat (station,), doy (day,) → (station, day)."""
    phi = np.radians(np.asarray(lat_deg, float))[:, None]
    J = np.asarray(doy, float)[None, :]
    dr = 1 + 0.033 * np.cos(2 * np.pi * J / 365)
    dl = 0.409 * np.sin(2 * np.pi * J / 365 - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(dl), -1, 1))
    return 24 * 60 / np.pi * 0.0820 * dr * (ws * np.sin(phi) * np.sin(dl) + np.cos(phi) * np.cos(dl) * np.sin(ws))


def hargreaves_rs(tmax, tmin, ra, krs, elev):
    """FAO-56 eq. 50, capped at clear-sky radiation (eq. 37). MJ m-2 d-1."""
    rs = krs[:, None] * np.sqrt(np.maximum(tmax - tmin, 0)) * ra
    z = np.nan_to_num(np.asarray(elev, float), nan=0.0)       # missing elevation (North Korea): sea level
    rso = (0.75 + 2e-5 * z)[:, None] * ra
    return np.minimum(rs, rso)


def daylength(lat_deg, doy):
    """Astronomical day length N (h), FAO-56 eq. 34. → (station, day)"""
    phi = np.radians(np.asarray(lat_deg, float))[:, None]
    dl = 0.409 * np.sin(2 * np.pi * np.asarray(doy, float)[None, :] / 365 - 1.39)
    return 24 / np.pi * np.arccos(np.clip(-np.tan(phi) * np.tan(dl), -1, 1))


def to_raster_unit(mj, lat=None, doy=None, unit=None):
    """Convert daily global radiation (MJ m-2 d-1) to the unit of the Solar rasters.
      'MJ'        daily total
      'Wm2'       24-h mean irradiance (× 11.574)
      'Wm2_peak'  midday peak irradiance of a half-sine daily course: π · Rs / (2 N · 3600)"""
    unit = unit or C.SOLAR_UNIT
    if unit == 'MJ':
        return mj
    if unit == 'Wm2':
        return mj * 11.574
    if unit == 'Wm2_peak':
        n = daylength(lat, doy)
        return np.pi * mj * 1e6 / (2 * n * 3600)
    raise ValueError(f'unknown SOLAR_UNIT {unit!r}')


def emergence(T, t_base, r_h):
    """First DOY on which the degree-day sum from 1 January reaches r_h. T (station, day of year)."""
    cum = np.nancumsum(np.maximum(T - t_base, 0.0), axis=1)
    hit = cum >= r_h
    e = hit.argmax(axis=1).astype(float) + 1
    e[~hit.any(axis=1)] = np.nan
    return e


def termination(T, t_base, n=None, start=None):
    """First day of the first run of n consecutive days below t_base starting on or after DOY start;
    365 if none (as in the raster code)."""
    n = n or C.TERMINATION_CONSEC_DAYS
    start = start or C.TERMINATION_SEARCH_START_DOY
    below = (T < t_base).astype(np.int32)
    below[:, :start - 1] = 0
    cs = np.cumsum(below, axis=1)
    run = cs.copy()
    run[:, n:] = cs[:, n:] - cs[:, :-n]
    hit = run >= n
    t = hit.argmax(axis=1).astype(float) + 1 - n + 1
    t[~hit.any(axis=1)] = 365.0
    return t


def activity_window(T):
    e_all = np.full(T.shape[0], np.inf)
    t_all = np.full(T.shape[0], -np.inf)
    for sp in C.CWRI_SPECIES:
        p = C.PHENOLOGY[sp]
        e = emergence(T, p['T_base'], p['R_h'])
        t = termination(T, p['T_base'])
        e_all = np.fmin(e_all, e)
        t_all = np.fmax(t_all, t)
    e_all[~np.isfinite(e_all)] = np.nan
    t_all[~np.isfinite(t_all)] = np.nan
    return e_all, t_all



def _weather(df):
    return {k: df[k].to_numpy(float)[None, :] for k in ('Tavg', 'Tmax', 'Tmin', 'Tgap', 'HMDT', 'WDSP', 'Rn', 'Solar')}


if __name__ == '__main__':
    import pandas as pd
    a = argparse.ArgumentParser()
    a.add_argument('--test', default='golden_test_vectors.csv')
    g = a.parse_args()
    df = pd.read_csv(g.test)
    w = _weather(df)
    got = {'CWRI': cwri(w)[0], 'BAI_E': bai_e(w)[0]}
    for sp in SPECIES_PARAMS:
        prod = np.ones(len(df))
        for k, v in PARAM_TO_VAR.items():
            prod *= _response(w[v][0], SPECIES_PARAMS[sp][k])
        got['CSI_' + sp] = prod * 100
    worst = max(float(np.nanmax(np.abs(got[c] - df[c].to_numpy(float)))) for c in got)
    print(f'{len(df)} rows, max |difference| = {worst:.2e}  ->', 'OK' if worst < 1e-6 else 'MISMATCH')
