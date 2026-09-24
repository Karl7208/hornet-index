# -*- coding: utf-8 -*-
"""
1km 격자 (EPSG:5186, 논문 파이프라인과 동일 좌표계) + 관측소 → 격자 보간

- 마스크: data/static/mask_1km.tif (육지 1, 바다 0) — make_mask.py 로 한 번 생성
- 지역:   data/static/regions_1km.tif + regions.csv (시도 또는 시군 코드) — 요약표용
- 고도:   data/static/dem_1km.tif (선택) — 있으면 기온에 고도 보정(−6.5 °C/km)
- 보간:   가까운 관측소 8곳 역거리가중(IDW, 거리² 가중)
"""
import os
import numpy as np
import rasterio
from pyproj import Transformer
from scipy.spatial import cKDTree

STATIC = os.path.join(os.path.dirname(__file__), 'data', 'static')
LAPSE = 0.0065          # °C / m
IDW_K = 8
IDW_POWER = 2.0
TEMP_VARS = ('tavg', 'tmax', 'tmin')

_to5186 = Transformer.from_crs('EPSG:4326', 'EPSG:5186', always_xy=True)


class Grid:
    def __init__(self, static_dir=STATIC):
        with rasterio.open(os.path.join(static_dir, 'mask_1km.tif')) as src:
            self.mask = src.read(1) > 0
            self.profile = src.profile.copy()
            self.transform = src.transform
            self.crs = src.crs
        self.shape = self.mask.shape
        rows, cols = np.nonzero(self.mask)
        xs, ys = rasterio.transform.xy(self.transform, rows, cols, offset='center')
        self.rows, self.cols = rows, cols
        self.xy = np.column_stack([np.asarray(xs), np.asarray(ys)])
        self.n = len(rows)

        self.dem = None
        self.dem_missing = 0
        p = os.path.join(static_dir, 'dem_1km.tif')
        if os.path.exists(p):
            with rasterio.open(p) as src:
                d = src.read(1).astype(np.float64)
                if d.shape == self.shape:
                    bad = ~np.isfinite(d) | (d < -100) | (d > 3000)     # DEM이 덮지 않는 칸(섬 등)
                    self.dem_missing = int((bad & self.mask).sum())
                    self.dem_ok2d = ~bad
                    self.dem2d = np.where(bad, 0.0, d)
                    self.dem = self.dem2d[self.mask]
                    self.dem_ok = self.dem_ok2d[self.mask]

        self.regions = None
        self.region_names = {}
        p = os.path.join(static_dir, 'regions_1km.tif')
        if os.path.exists(p):
            with rasterio.open(p) as src:
                self.regions = src.read(1)[self.mask]
            import csv
            with open(os.path.join(static_dir, 'regions.csv'), encoding='utf-8-sig') as f:
                for r in csv.DictReader(f):
                    self.region_names[int(r['code'])] = r['name']

    def to_2d(self, v, fill=np.nan):
        out = np.full(self.shape, fill, dtype=np.float32)
        out[self.rows, self.cols] = v
        return out

    def interpolate(self, stations, var):
        """stations: DataFrame(lat, lon, elev, <var>) → 격자 유효 칸 1D 배열"""
        s = stations[['lon', 'lat', 'elev', var]].dropna(subset=[var, 'lon', 'lat'])
        if len(s) < 3:
            return np.full(self.n, np.nan)
        x, y = _to5186.transform(s['lon'].values, s['lat'].values)
        vals = s[var].values.astype(np.float64)
        elev = s['elev'].values.astype(np.float64)
        if self.dem is not None and np.isnan(elev).any():   # 지점 고도가 없으면 DEM에서 읽음
            r, c = rasterio.transform.rowcol(self.transform, x, y)
            r = np.clip(np.asarray(r), 0, self.shape[0] - 1); c = np.clip(np.asarray(c), 0, self.shape[1] - 1)
            elev = np.where(np.isnan(elev) & self.dem_ok2d[r, c], self.dem2d[r, c], elev)
        st_ok = ~np.isnan(elev)                      # 고도를 아는 관측소만 해면 환산
        elev = np.nan_to_num(elev, nan=0.0)
        lapse = var in TEMP_VARS and self.dem is not None
        if lapse:
            vals = np.where(st_ok, vals + LAPSE * elev, vals)   # 해면 기준으로 환산
        tree = cKDTree(np.column_stack([x, y]))
        k = min(IDW_K, len(s))
        dist, idx = tree.query(self.xy, k=k)
        if k == 1:
            dist, idx = dist[:, None], idx[:, None]
        w = 1.0 / np.maximum(dist, 1.0) ** IDW_POWER
        out = np.sum(w * vals[idx], axis=1) / np.sum(w, axis=1)
        if lapse:
            out = np.where(self.dem_ok, out - LAPSE * self.dem, out)   # 칸 고도로 되돌림 (DEM 없는 칸은 그대로)
        return out
