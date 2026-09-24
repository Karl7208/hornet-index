# -*- coding: utf-8 -*-
"""
한 지점의 모든 값 확인 — 지도에 색이 안 나오는 곳 진단, 신고 지점의 지수 확인용

  python check_point.py <위도> <경도> <날짜>
  예) python check_point.py 33.38 126.53 2026-09-23      (제주)
      python check_point.py 35.84 127.12 2026-09-23      (전주)
"""
import os
import sys
import glob
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
import sources

ROOT = os.path.dirname(os.path.abspath(__file__))


def at(path, x, y):
    with rasterio.open(path) as r:
        row, col = r.index(x, y)
        if not (0 <= row < r.height and 0 <= col < r.width):
            return '격자 밖'
        v = r.read(1)[row, col]
        if r.nodata is not None and v == r.nodata:
            return '값 없음(nodata)'
        return round(float(v), 2)


def main():
    lat, lon, day = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
    x, y = Transformer.from_crs('EPSG:4326', 'EPSG:5186', always_xy=True).transform(lon, lat)
    st = os.path.join(ROOT, 'data', 'static')
    print(f'지점 {lat}, {lon} → TM {x:.0f}, {y:.0f}')

    print('\n[고정 격자]')
    for f in ('mask_1km', 'regions_1km', 'dem_1km', 'persistence_spring', 'persistence_summer', 'persistence_autumn'):
        p = os.path.join(st, f + '.tif')
        print(f'  {f:20s}', at(p, x, y) if os.path.exists(p) else '파일 없음')
    rid = at(os.path.join(st, 'regions_1km.tif'), x, y)
    if isinstance(rid, float) and os.path.exists(os.path.join(st, 'regions.csv')):
        names = pd.read_csv(os.path.join(st, 'regions.csv'), encoding='utf-8-sig').set_index('code')['name']
        print('  지역 이름             ', names.get(int(rid), '없음'))

    print(f'\n[{day} 계산 결과]')
    d = os.path.join(ROOT, 'out', day)
    if not os.path.isdir(d):
        print('  out 폴더에 이 날짜 결과 없음 — run_daily.py 를 먼저 실행')
    for p in sorted(glob.glob(os.path.join(d, '*.tif'))):
        print(f'  {os.path.basename(p)[:-4]:20s}', at(p, x, y))

    print(f'\n[{day} 주변 관측소 (30km 이내)]')
    obs = sources.load_year(int(day[:4]))
    o = obs[obs['date'] == day].dropna(subset=['lat', 'lon']).copy()
    ox, oy = Transformer.from_crs('EPSG:4326', 'EPSG:5186', always_xy=True).transform(o['lon'].values, o['lat'].values)
    o['km'] = np.hypot(ox - x, oy - y) / 1000
    near = o[o['km'] <= 30].sort_values('km')
    if near.empty:
        print('  없음 — 가장 가까운 관측소:', o.sort_values('km').iloc[0][['source', 'stn', 'km']].to_dict() if len(o) else '자료 없음')
    else:
        print(near[['source', 'stn', 'km', 'tavg', 'tmax', 'tmin', 'rh', 'ws', 'rain']].round(1).head(12).to_string(index=False))


if __name__ == '__main__':
    main()
