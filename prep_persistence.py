# -*- coding: utf-8 -*-
"""
한 번만 실행: 논문(노트북 06)의 계절별 Gi* 지속성 래스터 → 운영 격자로 옮김

  python prep_persistence.py <Gi_persistence 파일들이 있는 폴더>
  예) python prep_persistence.py "E:\\Bee_index\\말벌\\Gi_Star\\B_GiStar\\present"

폴더 안(하위 폴더 포함)의 Gi_persistence_spring/summer/autumn.TIF 를 찾아
data/static/persistence_<계절>.tif 로 저장 (최근접 재표본, 값 0~1 = 핫스팟이었던 해의 비율)
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
        hits = glob.glob(os.path.join(root, '**', f'Gi_persistence_{s}.TIF'), recursive=True) \
            + glob.glob(os.path.join(root, '**', f'Gi_persistence_{s}.tif'), recursive=True)
        if not hits:
            print(f'{s}: 파일 없음'); continue
        with rasterio.open(hits[0]) as r:
            dst = np.full(g.shape, -9999, dtype='float32')
            reproject(rasterio.band(r, 1), dst, src_nodata=r.nodata, dst_nodata=-9999,
                      dst_transform=g.transform, dst_crs=g.crs, resampling=Resampling.nearest)
        prof = g.profile.copy(); prof.update(dtype='float32', nodata=-9999, compress='deflate')
        with rasterio.open(os.path.join(STATIC, f'persistence_{s}.tif'), 'w', **prof) as w:
            w.write(dst, 1)
        v = dst[g.mask]; v = v[v >= 0]
        print(f'{s}: {hits[0]} → 남한 칸 {v.size:,}, 7/8년 이상 {100*(v >= 0.87).mean():.1f}%')


if __name__ == '__main__':
    main()
