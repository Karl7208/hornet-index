# -*- coding: utf-8 -*-
"""
출력: GeoTIFF(EPSG:5186, Karl의 GIS용) + 웹 지도용 PNG(EPSG:4326, 투명 배경) + 지역 요약표
"""
import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from PIL import Image


def write_tif(path, arr2d, grid, nodata=-9999.0):
    prof = grid.profile.copy()
    prof.update(dtype='float32', count=1, nodata=nodata, compress='deflate')
    with rasterio.open(path, 'w', **prof) as dst:
        dst.write(np.where(np.isnan(arr2d), nodata, arr2d).astype('float32'), 1)


# ── 색 (값 → RGBA) ───────────────────────────────────────────
def _ramp(stops):
    xs = np.array([s[0] for s in stops], dtype=float)
    cs = np.array([s[1] for s in stops], dtype=float)

    def f(v):
        out = np.zeros(v.shape + (4,), dtype=np.uint8)
        ok = ~np.isnan(v)
        for ch in range(4):
            out[..., ch][ok] = np.interp(v[ok], xs, cs[:, ch]).astype(np.uint8)
        return out
    return f


# 말벌 활동(0~100): 낮으면 투명에 가깝게, 높을수록 진한 주황·적갈
PAL_INDEX = _ramp([(0, (255, 250, 220, 75)), (15, (254, 227, 145, 110)), (35, (254, 196, 79, 150)),
                   (55, (254, 153, 41, 190)), (75, (217, 95, 14, 215)), (100, (153, 52, 4, 235))])
# 꿀벌 출입(0~100): 초록 계열
PAL_BEE = _ramp([(0, (247, 252, 245, 75)), (30, (199, 233, 192, 120)), (60, (116, 196, 118, 180)),
                 (100, (0, 109, 44, 230))])
# 중첩 일수(0~30)
PAL_DAYS = _ramp([(0, (245, 244, 250, 75)), (3, (218, 218, 235, 120)), (10, (158, 154, 200, 170)),
                  (20, (106, 81, 163, 210)), (30, (63, 0, 125, 235))])


def PAL_GI(c):
    out = np.zeros(c.shape + (4,), dtype=np.uint8)
    lut = {3: (178, 24, 43, 220), 2: (239, 138, 98, 200), 1: (253, 219, 199, 170), -1: (103, 169, 207, 150)}
    for k, col in lut.items():
        out[c == k] = col
    return out


def PAL_ALERT(c):
    out = np.zeros(c.shape + (4,), dtype=np.uint8)
    lut = {3: (189, 0, 38, 230), 2: (253, 141, 60, 205), 1: (254, 217, 118, 160)}   # 심각·경계·주의
    for k, col in lut.items():
        out[c == k] = col
    return out


def PAL_LAND(c):
    """흰 배경 모드용 남한 육지 (옅은 회색)"""
    out = np.zeros(c.shape + (4,), dtype=np.uint8)
    out[c == 1] = (226, 226, 222, 255)
    return out


def PAL_HATCH(c):
    """상습 지역 빗금 (다른 층 위에 겹쳐 그림)"""
    out = np.zeros(c.shape + (4,), dtype=np.uint8)
    ii, jj = np.indices(c.shape)
    m = (c == 1) & (((ii + jj) % 7) < 2)
    out[m] = (45, 20, 10, 190)
    return out


def png4326(path, arr2d, grid, palette, categorical=False):
    """EPSG:5186 격자 → EPSG:4326 PNG. 반환: [[남, 서], [북, 동]] (Leaflet imageOverlay 범위)"""
    H, W = grid.shape
    left, top = grid.transform.c, grid.transform.f
    right, bottom = left + W * grid.transform.a, top + H * grid.transform.e
    dst_t, dw, dh = calculate_default_transform(grid.crs, 'EPSG:4326', W, H, left, bottom, right, top)
    src = arr2d.astype('float32')
    dst = np.full((dh, dw), np.nan, dtype='float32')
    reproject(src, dst, src_transform=grid.transform, src_crs=grid.crs, dst_transform=dst_t,
              dst_crs='EPSG:4326', resampling=Resampling.nearest, src_nodata=np.nan, dst_nodata=np.nan)
    rgba = palette(np.where(np.isnan(dst), -9, dst).astype(np.int8)) if categorical else palette(dst)
    Image.fromarray(rgba, 'RGBA').save(path, optimize=True)
    west, north = dst_t.c, dst_t.f
    east, south = west + dw * dst_t.a, north + dh * dst_t.e
    return [[round(south, 5), round(west, 5)], [round(north, 5), round(east, 5)]]


def region_summary(grid, layers):
    """layers: {열이름: 1D 배열(유효 칸)} → 지역별 평균 표 (list of dict)"""
    if grid.regions is None:
        return []
    rows = []
    for code, name in sorted(grid.region_names.items()):
        m = grid.regions == code
        if not m.any():
            continue
        r = {'region': name, 'n_cells': int(m.sum())}
        for k, v in layers.items():
            vv = v[m]
            if k.startswith('pct_'):
                r[k] = round(float(np.nanmean(vv) * 100), 1)
            else:
                r[k] = round(float(np.nanmean(vv)), 2)
        rows.append(r)
    return rows
