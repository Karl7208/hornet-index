# -*- coding: utf-8 -*-
"""
기상청 API허브 접속 시험 2차 — AWS 시간자료 열 이름, AWS 일자료 요소 이름 확인
환경변수 KMA_KEY 필요 (GitHub Secrets)
"""
import os, sys, datetime as dt, urllib.request, urllib.parse

KEY = os.environ.get('KMA_KEY', '').strip()
if not KEY:
    sys.exit('KMA_KEY 가 없습니다.')
BASE = 'https://apihub.kma.go.kr/api/typ01/url/'
day = (dt.datetime.utcnow() + dt.timedelta(hours=9)).date() - dt.timedelta(days=1)
D = day.strftime('%Y%m%d')


def get(endpoint, params):
    url = BASE + endpoint + '?' + urllib.parse.urlencode({**params, 'authKey': KEY})
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            return r.read().decode('euc-kr', errors='replace')
    except Exception as e:
        return f'#실패 {e.__class__.__name__}: {e}'


def rows(text):
    return [l for l in text.splitlines() if l.strip() and not l.lstrip().startswith('#')]


print('=== 1. AWS 시간자료 (전 지점, 어제 15시) — 열 설명 ===')
t = get('awsh.php', {'tm': D + '1500', 'help': 1})
for l in t.splitlines()[:45]:
    print('  ' + l[:170])
r = rows(t)
print(f'  자료 줄 수: {len(r)}')
for l in r[:3]:
    print('  예: ' + l[:170])

print('\n=== 2. AWS 일자료 요소 이름 찾기 (어제) ===')
for obs in ['ta', 'ta_avg', 'ta_day', 'hm', 'hm_avg', 'hm_min', 'rhm', 'ws', 'ws_avg', 'ws_max', 'wd_max', 'ws_ins', 'rn_day', 'td', 'pa']:
    n = len(rows(get('sfc_aws_day.php', {'tm2': D, 'obs': obs, 'stn': 0, 'disp': 0, 'help': 0})))
    print(f'  obs={obs:8s} → {n}줄')

print('\n=== 3. ASOS 일자료 자료 줄 모양 ===')
r = rows(get('kma_sfcdd3.php', {'tm1': D, 'tm2': D, 'stn': 0, 'help': 0}))
print(f'  자료 줄 수: {len(r)}')
for l in r[:3]:
    print('  예: ' + l[:200])
