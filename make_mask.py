# -*- coding: utf-8 -*-
"""
한 번만 실행: 경계 파일(시도·시군구 GeoJSON 또는 SHP, zip 풀어서) → 1km 마스크·지역 격자

  python make_mask.py <경계파일> <이름필드> [--dem 원본DEM.tif]

예)
  python make_mask.py sig.shp SIG_KOR_NM --code-field SIG_CD --dem E:/GIS/DEM/DEM_ITRF2000_TM.tif
  python make_mask.py sig_project_penisula_do_fix.shp CTP_KOR_NM --like <기존 CWRI tif>

출력: data/static/mask_1km.tif, regions_1km.tif, regions.csv, (dem_1km.tif)
좌표계 EPSG:5186, 1km — 논문 파이프라인과 같은 격자 체계.
논문과 칸 위치까지 완전히 맞추려면 기존 CWRI GeoTIFF 한 장을 --like 로 주면 그 격자를 그대로 씀.
"""
import os
import sys
import json
import argparse
import numpy as np
import rasterio
from rasterio import features
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling
from pyproj import Transformer
from shapely.geometry import shape, mapping
from shapely.ops import transform as shp_transform

OUT = os.path.join(os.path.dirname(__file__), 'data', 'static')
RES = 1000.0


SIDO = {'11': '서울', '26': '부산', '27': '대구', '28': '인천', '29': '광주', '30': '대전', '31': '울산',
        '36': '세종', '41': '경기', '42': '강원', '51': '강원', '43': '충북', '44': '충남', '45': '전북',
        '52': '전북', '46': '전남', '47': '경북', '48': '경남', '50': '제주'}


def guess_crs(bounds):
    """.prj가 없을 때 좌표 범위로 추정: UTM-K(5179)는 x≈70만~140만, y≈140만~210만"""
    minx, miny, maxx, maxy = bounds
    if 100 <= minx <= 180 and 20 <= miny <= 50:
        return 'EPSG:4326'
    if minx > 500000 and miny > 1000000:
        return 'EPSG:5179'
    return 'EPSG:5186'


def read_features(path, name_field, code_field=None, src_crs=None):
    if path.lower().endswith(('.json', '.geojson')):
        gj = json.load(open(path, encoding='utf-8'))
        crs = src_crs or ('EPSG:5186' if '5186' in str(gj.get('crs', '')) else 'EPSG:4326')
        feats = [(shape(f['geometry']), str(f['properties'][name_field]),
                  str(f['properties'].get(code_field, '')) if code_field else '') for f in gj['features']]
        return feats, crs
    import shapefile                                   # pyshp (geopandas 없이 SHP 읽기)
    r = None
    for enc in ('cp949', 'utf-8'):
        try:
            r = shapefile.Reader(path, encoding=enc)
            r.records()
            break
        except UnicodeDecodeError:
            r = None
    names = [f[0] for f in r.fields[1:]]
    feats = []
    for sr in r.iterShapeRecords():
        rec = dict(zip(names, sr.record))
        feats.append((shape(sr.shape.__geo_interface__), str(rec[name_field]),
                      str(rec.get(code_field, '')) if code_field else ''))
    prj = os.path.splitext(path)[0] + '.prj'
    if src_crs:
        crs = src_crs
    elif os.path.exists(prj):
        from pyproj import CRS
        crs = CRS.from_wkt(open(prj).read()).to_string()
    else:
        crs = guess_crs(r.bbox)
        print(f'  .prj 없음 → 좌표 범위로 {crs} 추정')
    return feats, crs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('boundary'); ap.add_argument('name_field')
    ap.add_argument('--code-field', help='지역 코드 열 (예: SIG_CD). 앞 두 자리로 시도명을 붙여 이름 중복을 막음')
    ap.add_argument('--src-crs', help='경계 파일 좌표계 (예: EPSG:5179). 없으면 .prj 또는 좌표 범위로 추정')
    ap.add_argument('--dem', help='해발고도 DEM (GeoTIFF) → 1km 평균으로 재표본')
    ap.add_argument('--like', help='기존 CWRI GeoTIFF — 논문과 같은 격자를 그대로 사용')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    feats, crs = read_features(a.boundary, a.name_field, a.code_field, a.src_crs)
    if a.code_field:
        feats = [(g, f'{SIDO.get(c[:2], c[:2])} {n}', c) for g, n, c in feats]
    feats = [(g, n) for g, n, _ in feats]
    if crs != 'EPSG:5186':
        tr = Transformer.from_crs(crs, 'EPSG:5186', always_xy=True).transform
        feats = [(shp_transform(tr, g), n) for g, n in feats]

    if a.like:
        with rasterio.open(a.like) as src:
            transform, H, W = src.transform, src.height, src.width
    else:
        minx = min(g.bounds[0] for g, _ in feats); miny = min(g.bounds[1] for g, _ in feats)
        maxx = max(g.bounds[2] for g, _ in feats); maxy = max(g.bounds[3] for g, _ in feats)
        minx, miny = np.floor(minx / RES) * RES, np.floor(miny / RES) * RES
        maxx, maxy = np.ceil(maxx / RES) * RES, np.ceil(maxy / RES) * RES
        W, H = int((maxx - minx) / RES), int((maxy - miny) / RES)
        transform = from_origin(minx, maxy, RES, RES)

    names = sorted(set(n for _, n in feats))
    code = {n: i + 1 for i, n in enumerate(names)}
    regions = features.rasterize([(mapping(g), code[n]) for g, n in feats],
                                 out_shape=(H, W), transform=transform, fill=0, dtype='int16')
    mask = (regions > 0).astype('uint8')

    prof = dict(driver='GTiff', height=H, width=W, count=1, crs='EPSG:5186',
                transform=transform, compress='deflate')
    with rasterio.open(os.path.join(OUT, 'mask_1km.tif'), 'w', dtype='uint8', **prof) as dst:
        dst.write(mask, 1)
    with rasterio.open(os.path.join(OUT, 'regions_1km.tif'), 'w', dtype='int16', **prof) as dst:
        dst.write(regions, 1)
    with open(os.path.join(OUT, 'regions.csv'), 'w', encoding='utf-8-sig') as f:
        f.write('code,name\n')
        for n in names:
            f.write(f'{code[n]},{n}\n')

    if a.dem:
        dem = np.full((H, W), -9999, dtype='float32')
        with rasterio.open(a.dem) as src:
            reproject(rasterio.band(src, 1), dem, dst_transform=transform, dst_crs='EPSG:5186',
                      resampling=Resampling.average, dst_nodata=-9999)
        with rasterio.open(os.path.join(OUT, 'dem_1km.tif'), 'w', dtype='float32', nodata=-9999, **prof) as dst:
            dst.write(dem, 1)

    print(f'격자 {H}×{W}, 유효 칸 {int(mask.sum()):,}, 지역 {len(names)}개 → {OUT}')


if __name__ == '__main__':
    main()
