# -*- coding: utf-8 -*-
"""
한 번만 실행: 논문(노트북 06)의 계절별 Gi* 지속성 래스터 → 운영 격자로 옮김

  python prep_persistence.py <지속성 파일들이 있는 폴더>
  예) python prep_persistence.py tools/persistence_paper2_v4      (논문 2 v4 인계 static/persistence_*.tif, 2017–2024)

폴더 안(하위 폴더 포함)의 persistence_<계절>.tif 또는 Gi_persistence_<계절>.TIF 를 찾아
data/static/persistence_<계절>.tif 로 저장 (최근접 재표본, 값 0~1 = 가을 Gi* z > 1.96 이었던 해의 비율)
논문 래스터는 한반도 1103×673 (ITRF2000 TM, EPSG:5186과 같은 투영 모수)이고 칸 중심이 운영 격자와 수백 m 어긋나 있어
투영 정의를 운영 격자 것으로 맞춘 뒤(같은 모수, 좌표 이동 없음) 최근접으로 옮김.
"""
import os
import sys
import glob
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from grid import Grid, STATIC


def main():
    root = sys.argv[1]
    g = Grid()
    for s in ('spring', 'summer', 'autumn'):
        hits = [h for pat in (f'persistence_{s}.tif', f'Gi_persistence_{s}.TIF', f'Gi_persistence_{s}.tif')
                for h in glob.glob(os.path.join(root, '**', pat), recursive=True)
                if os.path.abspath(os.path.dirname(h)) != os.path.abspath(STATIC)]
        if not hits:
            print(f'{s}: 파일 없음'); continue
        with rasterio.open(hits[0]) as r:
            dst = np.full(g.shape, -9999, dtype='float32')
            src = r.read(1)
            reproject(src, dst, src_transform=r.transform, src_crs=g.crs, src_nodata=r.nodata, dst_nodata=-9999,
                      dst_transform=g.transform, dst_crs=g.crs, resampling=Resampling.nearest)
            sv = src[(src >= 0) & np.isfinite(src)]
        prof = g.profile.copy(); prof.update(dtype='float32', nodata=-9999, compress='deflate')
        with rasterio.open(os.path.join(STATIC, f'persistence_{s}.tif'), 'w', **prof) as w:
            w.write(dst, 1)
        v = dst[g.mask]; v = v[v >= 0]
        print(f'{s}: {hits[0]}\n    원본 한반도 0.75 이상 {100*(sv >= 0.75).mean():.1f}% → 운영 격자(남한) 칸 {v.size:,}: '
              f'0.75 이상 {100*(v >= 0.75).mean():.1f}%, 7/8년(0.87) 이상 {100*(v >= 0.87).mean():.1f}%, '
              f'빈칸 {int((g.mask & (dst < 0)).sum()):,}')


if __name__ == '__main__':
    main()
