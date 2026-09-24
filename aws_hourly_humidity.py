# -*- coding: utf-8 -*-
"""
AWS 시간자료 → 일평균 습도 파일 (기존 폴더 형식 그대로)

  python aws_hourly_humidity.py --in D:\\AWS\\raw\\hourly\\2026 --out D:\\AWS\\output

입력: OBS_AWS_TIM_*.csv (기상자료개방포털, cp949)  열: 지점, 지점명, 일시, …, 습도(%)
출력: <out>\\HMDT\\HMDT_YYYYMMDD.txt  ("ID,Value", 결측은 Null)

규칙
- 날짜 경계는 기상청 일자료와 같게 01시~24시 (자정 00:00 값은 전날 24시로 계산)
- 하루 유효 시간이 18시간 미만이면 결측(Null)
- 이미 있는 날짜 파일은 건드리지 않음 (--overwrite 를 주면 새로 씀)
그다음 local_import.py 를 다시 실행하면 AWS 습도가 들어감.
"""
import os
import glob
import argparse
import pandas as pd

MIN_HOURS = 18


def read_hourly(path):
    for enc in ('cp949', 'utf-8-sig'):
        try:
            head = pd.read_csv(path, encoding=enc, nrows=0)
            break
        except UnicodeDecodeError:
            continue
    cols = list(head.columns)
    stn_c = cols[0]
    tm_c = next(c for c in cols if '일시' in c)
    rh_c = next(c for c in cols if '습도' in c)
    parts = []
    for ch in pd.read_csv(path, encoding=enc, usecols=[stn_c, tm_c, rh_c], chunksize=500_000):
        ch.columns = ['stn', 'tm', 'rh']
        ch['rh'] = pd.to_numeric(ch['rh'], errors='coerce')
        ch.loc[(ch['rh'] < 0) | (ch['rh'] > 100), 'rh'] = float('nan')
        t = pd.to_datetime(ch['tm'], errors='coerce')
        ch['date'] = (t - pd.Timedelta(minutes=1)).dt.date     # 00:00 → 전날 24시
        parts.append(ch.dropna(subset=['date']))
    return pd.concat(parts, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='src', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--overwrite', action='store_true')
    a = ap.parse_args()
    out_dir = os.path.join(a.out, 'HMDT')
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(a.src, 'OBS_AWS_TIM_*.csv')))
    if not files:
        raise SystemExit(f'{a.src} 에 OBS_AWS_TIM_*.csv 가 없습니다.')
    written = skipped = 0
    for f in files:
        h = read_hourly(f)
        g = h.groupby(['date', 'stn'])['rh'].agg(['mean', 'count']).reset_index()
        g.loc[g['count'] < MIN_HOURS, 'mean'] = float('nan')
        for d, gd in g.groupby('date'):
            if gd['mean'].isna().all():          # 파일 경계에 걸친 한두 시간뿐인 날은 쓰지 않음
                continue
            p = os.path.join(out_dir, f'HMDT_{d:%Y%m%d}.txt')
            if os.path.exists(p) and not a.overwrite:
                skipped += 1
                continue
            with open(p, 'w', encoding='utf-8') as w:
                w.write('ID,Value\n')
                for s, v in zip(gd['stn'], gd['mean']):
                    w.write(f'{int(s)},{"Null" if pd.isna(v) else f"{v:.1f}"}\n')
            written += 1
        print(f'  {os.path.basename(f)}: {g["date"].nunique()}일, 지점 {g["stn"].nunique()}곳')
    print(f'완료: 새로 쓴 날짜 {written}일, 이미 있어 건너뜀 {skipped}일 → {out_dir}')


if __name__ == '__main__':
    main()
