# -*- coding: utf-8 -*-
"""
한 번만 실행: 지점별 Hargreaves kRs 표 만들기 (FAO-56 기본값)

  python tools/build_krs.py [지점목록.csv]     (기본: tools/kma_stations_list.csv)

FAO-56 식 50의 kRs: 해안(바다 공기의 영향을 받는 곳) 0.19, 내륙 0.16.
해안 판정 = 지점에서 가장 가까운 바다 칸까지 COASTAL_KM 이하 (1 km DEM에서 해발 0 이하이고 육지 마스크 밖인 칸 = 바다).
결과: data/static/krs_stations.csv (ID, 지점명, 기관, lat, lon, dist_sea_km, coastal, krs)
→ grid.py 가 1 km 칸마다 가장 가까운 지점의 kRs를 씀.
"""
import os
import sys
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from scipy.ndimage import distance_transform_edt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC = os.path.join(ROOT, 'data', 'static')
KRS_COASTAL, KRS_INTERIOR = 0.19, 0.16
COASTAL_KM = 10.0


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'tools', 'kma_stations_list.csv')
    st = pd.read_csv(src, encoding='utf-8-sig')
    st = st.rename(columns={'지점명': 'name', '기관명': 'net', '위도': 'lat', '경도': 'lon'})[['ID', 'name', 'net', 'lat', 'lon']]
    st = st[st['net'] != '북한기상청'].dropna(subset=['lat', 'lon'])
    with rasterio.open(os.path.join(STATIC, 'mask_1km.tif')) as m:
        land = m.read(1) > 0
        tf = m.transform
    with rasterio.open(os.path.join(STATIC, 'dem_1km.tif')) as d:
        dem = d.read(1).astype(float)
    sea = (~land) & ~(dem > 0)
    dist = distance_transform_edt(~sea) * abs(tf.a) / 1000.0          # km to nearest sea cell
    x, y = Transformer.from_crs('EPSG:4326', 'EPSG:5186', always_xy=True).transform(st['lon'].values, st['lat'].values)
    r, c = rasterio.transform.rowcol(tf, x, y)
    r, c = np.asarray(r), np.asarray(c)
    inside = (r >= 0) & (r < land.shape[0]) & (c >= 0) & (c < land.shape[1])
    st = st[inside].copy()
    st['dist_sea_km'] = dist[r[inside], c[inside]].round(1)
    st['coastal'] = st['dist_sea_km'] <= COASTAL_KM
    st['krs'] = np.where(st['coastal'], KRS_COASTAL, KRS_INTERIOR)
    out = os.path.join(STATIC, 'krs_stations.csv')
    st.to_csv(out, index=False, encoding='utf-8-sig')
    print(f'{len(st)}곳 → {out}')
    print(st.groupby('net')['coastal'].agg(['size', 'sum', 'mean']).round(2).to_string())


if __name__ == '__main__':
    main()
