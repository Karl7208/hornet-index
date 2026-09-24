# 말벌 활동 지수 (hornet-index)

매일 06:30(KST) GitHub Actions가 기상청 API허브에서 전날 관측(AWS 전 지점, ASOS 포함 약 720곳)을 받아
Karl의 모델(종별 CSI·CWRI·BAI_E·적산온도 활동창·Gi*)로 1km 격자 지수를 계산하고, 결과를 GitHub Pages로 공개합니다.
PC는 필요 없습니다.

## 저장소에 있어야 하는 것
- 코드: `run_daily.py`, `indices.py`, `grid.py`, `export.py`, `sources.py`, `index.html`, `requirements.txt`
- `.github/workflows/daily.yml` (매일 자동 실행)
- `data/static/` ← **Karl PC의 `hornet-index-pipeline\data\static` 폴더 내용 그대로**
  (`mask_1km.tif`, `regions_1km.tif`, `regions.csv`, `dem_1km.tif`, `persistence_*.tif` 3개, `breakpoints.json`)
- `data/stations/` ← 비워 두면 첫 실행 때 1월 1일부터 기상청에서 받아 채움 (이후 매일 하루치씩 추가)

## 설정
- Settings > Secrets and variables > Actions: `KMA_KEY` (API허브 키)
- Settings > Pages > Source: **GitHub Actions**

## 자료
- 관측: `sfc_aws_day.php`(최고·최저기온, 강수) + `awsh.php` 1~24시(평균기온·습도·풍속, 18시간 이상일 때) — 하루 27회 호출
- 일사: 경험식 `Solar = −16.357·Tavg + 27.847·Tmax + 25.657·Tgap − 67.734`
- 산림청 산악기상(AMOS)은 계산에서 제외

그 밖의 도구(`make_mask.py`, `prep_persistence.py`, `compute_breakpoints.py`, `local_import.py`, `aws_hourly_humidity.py`, `check_point.py`)는 PC에서 한 번씩 쓰는 준비용입니다.
