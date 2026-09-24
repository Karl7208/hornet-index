# -*- coding: utf-8 -*-
"""
로컬 일별 관측 파일 → 표준 관측표 (data/stations/<연도>-<월>.csv)

폴더 구조 (Karl PC)
  D:\\ASOS\\output\\{Tavg,Tmax,Tmin,HMDT,WDSP,Rn,...}\\<변수>_YYYYMMDD.txt
  D:\\AWS\\output\\ ...
  D:\\AMOS\\output\\ ...      (산림청 산악기상)
파일 형식: "ID,Value" / 결측은 Null
지점 정보: 기상관측_국내.csv (ID, 기관명, 위도, 경도)

사용
  python local_import.py --meta "E:\\...\\기상관측_국내.csv" ^
      --net ASOS=D:\\ASOS\\output --net AWS=D:\\AWS\\output --net AMOS=D:\\AMOS\\output ^
      --start 2026-01-01 --end 2026-09-23

같은 ID가 두 관측망에 다 있으면 ASOS > AWS > AMOS 순으로 하나만 남김 (보간에서 이중 가중 방지).
"""
import os
import argparse
import datetime as dt
import numpy as np
import pandas as pd
import sources

VAR_DIRS = {'tavg': 'Tavg', 'tmax': 'Tmax', 'tmin': 'Tmin', 'rh': 'HMDT', 'ws': 'WDSP', 'rain': 'Rn'}
PRIORITY = ['ASOS', 'AWS', 'AMOS']
# 물리적으로 말이 안 되는 값은 결측 처리
RANGE = {'tavg': (-40, 45), 'tmax': (-40, 50), 'tmin': (-45, 40), 'rh': (0, 100), 'ws': (0, 60), 'rain': (0, 800)}


def read_meta(path):
    for enc in ('utf-8-sig', 'cp949'):
        try:
            m = pd.read_csv(path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    m = m.rename(columns={'위도': 'lat', '경도': 'lon', '기관명': 'net'})
    m['stn'] = m['ID'].astype(int).astype(str)
    m = m[m['net'].isin(PRIORITY)]                       # 북한 지점 제외
    return m[['stn', 'lat', 'lon', 'net']].drop_duplicates('stn')


def read_var(root, var_dir, day):
    p = os.path.join(root, var_dir, f'{var_dir}_{day:%Y%m%d}.txt')
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p, na_values=['Null', 'null', 'NULL', ''], dtype={'ID': str})
    df.columns = ['stn', 'value']
    df['stn'] = df['stn'].str.strip()
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    return df.dropna(subset=['stn']).drop_duplicates('stn').set_index('stn')['value']


def import_day(nets, meta, day):
    frames = []
    for net, root in nets.items():
        cols = {}
        for k, vd in VAR_DIRS.items():
            s = read_var(root, vd, day)
            if s is not None:
                lo, hi = RANGE[k]
                cols[k] = s.where((s >= lo) & (s <= hi))
        if 'tavg' not in cols:
            continue
        df = pd.DataFrame(cols)
        df.index.name = 'stn'
        df = df.reset_index()
        df['source'] = net.lower()
        frames.append(df)
    if not frames:
        return None, 0
    df = pd.concat(frames, ignore_index=True)
    df['pri'] = df['source'].str.upper().map({n: i for i, n in enumerate(PRIORITY)}).fillna(9)
    df = df.sort_values('pri').drop_duplicates('stn').drop(columns='pri')
    n_before = len(df)
    df = df.merge(meta[['stn', 'lat', 'lon']], on='stn', how='inner')
    df['date'] = day.isoformat()
    df['elev'] = np.nan                                  # 지점 고도 없음 → DEM에서 읽음 (grid.py)
    df['srad'] = np.nan                                  # 일사는 격자에서 경험식으로 계산
    for c in sources.COLS:
        if c not in df.columns:
            df[c] = np.nan
    return df[sources.COLS], n_before - len(df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--meta', required=True)
    ap.add_argument('--net', action='append', required=True, help='이름=폴더 (예: ASOS=D:\\ASOS\\output)')
    ap.add_argument('--start', required=True); ap.add_argument('--end', required=True)
    a = ap.parse_args()
    nets = dict(x.split('=', 1) for x in a.net)
    meta = read_meta(a.meta)
    d0 = dt.date.fromisoformat(a.start); d1 = dt.date.fromisoformat(a.end)

    rows, no_meta, empty = [], 0, []
    d = d0
    while d <= d1:
        df, miss = import_day(nets, meta, d)
        if df is None or df.empty:
            empty.append(d.isoformat())
        else:
            rows.append(df); no_meta += miss
        d += dt.timedelta(days=1)
    if not rows:
        raise SystemExit('읽은 자료가 없습니다. 폴더 경로와 파일 이름(Tavg_YYYYMMDD.txt)을 확인하세요.')
    allr = pd.concat(rows, ignore_index=True)
    for y, g in allr.groupby(allr['date'].str[:4]):
        old = sources.load_year(int(y))
        sources.save_year(int(y), pd.concat([old, g]) if len(old) else g)

    per_day = allr.groupby('date').size()
    print(f'{d0}~{d1}: {len(per_day)}일, 하루 평균 관측소 {per_day.mean():.0f}곳 '
          f'(망별 {allr.groupby("source").stn.nunique().to_dict()})')
    print(f'지점 정보에 없어 뺀 관측 {no_meta}건 (신설 지점 등) — 지점 파일을 갱신하면 들어감')
    if empty:
        print(f'자료 없는 날 {len(empty)}일: {", ".join(empty[:10])}{" …" if len(empty) > 10 else ""}')


if __name__ == '__main__':
    main()
