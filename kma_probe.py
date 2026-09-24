# -*- coding: utf-8 -*-
"""
기상청 API허브 접속 시험 — GitHub 서버(해외)에서도 되는지 확인
환경변수 KMA_KEY 필요. 키는 GitHub Secrets에만 두고 코드에는 넣지 않음.
PC에서도 같은 시험 가능:  set KMA_KEY=키값  →  python kma_probe.py
"""
import os, sys, datetime as dt, urllib.request, urllib.parse

KEY = os.environ.get('KMA_KEY', '').strip()
if not KEY:
    sys.exit('KMA_KEY 가 없습니다 (GitHub: Settings > Secrets and variables > Actions).')
BASE = 'https://apihub.kma.go.kr/api/typ01/url/'
day = (dt.datetime.utcnow() + dt.timedelta(hours=9)).date() - dt.timedelta(days=1)
D = day.strftime('%Y%m%d')


def call(name, endpoint, params, show=12):
    url = BASE + endpoint + '?' + urllib.parse.urlencode({**params, 'authKey': KEY})
    print(f'\n=== {name} ===')
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            raw = r.read()
            print(f'HTTP {r.status}, {len(raw):,} bytes')
    except Exception as e:
        print(f'실패: {e.__class__.__name__}: {e}')
        return None
    text = raw.decode('euc-kr', errors='replace')
    lines = text.splitlines()
    data = [l for l in lines if l.strip() and not l.startswith('#')]
    print(f'자료 줄 수: {len(data)}')
    for l in lines[:show]:
        print('  ' + l[:160])
    if data:
        print('  … 첫 자료 줄: ' + data[0][:160])
    return data


ok = {}
ok['ASOS 일자료'] = call('ASOS 일자료 (전 지점, 어제)', 'kma_sfcdd3.php', {'tm1': D, 'tm2': D, 'stn': 0, 'help': 1}, show=30)
for obs in ['ta_avg', 'ta_max', 'ta_min', 'hm_avg', 'ws_avg', 'rn_day']:
    ok[f'AWS 일자료 {obs}'] = call(f'AWS 일자료 {obs} (전 지점, 어제)', 'sfc_aws_day.php',
                                  {'tm2': D, 'obs': obs, 'stn': 0, 'disp': 0, 'help': 1 if obs == 'ta_avg' else 0}, show=20 if obs == 'ta_avg' else 3)
ok['지점정보 AWS'] = call('지점정보 AWS', 'stn_inf.php', {'inf': 'AWS', 'stn': '', 'tm': D + '0900', 'help': 0}, show=3)

print('\n=== 요약 ===')
for k, v in ok.items():
    print(f'{k:20s} {"성공 " + str(len(v)) + "줄" if v else "실패 또는 자료 없음"}')
