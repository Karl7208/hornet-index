# 말벌 활동 지수 (hornet-index)

매일 06:30(KST) GitHub Actions가 기상청 API허브에서 전날 관측(AWS 전 지점, ASOS 포함 약 730곳)을 받아
말벌 논문 2(COMPAG 시스템 논문)와 같은 정의로 1km 격자 지수(종별 CSI·CWRI·BAI_E·적산온도 활동창·Gi*)를 계산하고,
결과를 GitHub Pages로 공개합니다. PC는 필요 없습니다.

- 전국(담당자): https://karl7208.github.io/hornet-index/ (`?lang=en` 영문)
- 우리 지역(시민): https://karl7208.github.io/hornet-index/local.html

## 지수 정의 (논문 2와 같음)
| 항목 | 정의 |
|---|---|
| CSI | 종마다 8개 반응함수의 곱 × 100 |
| CWRI | 0.35·등검은말벌 *V. velutina* + 0.25·장수말벌 *V. mandarinia* + 0.20·말벌 *V. crabro* + 0.12·좀말벌 *V. analis* + 0.08·꼬마장수말벌 *V. ducalis*, **경쟁항 없음** |
| 활동창 | 5종 적산온도 출현 중 가장 이른 날 ~ 종료 중 가장 늦은 날(200일 이후 기준온도 아래 5일 연속의 첫날), 창 밖 CWRI = 0 |
| 일사 | Hargreaves–Samani Rs(FAO-56 식 50, 맑은 날 상한) → 한낮 최대 일사 π·Rs/(2N), W m⁻². kRs = FAO 기본값(해안 0.19, 내륙 0.16; 바다에서 10 km 이내 = 해안), 칸마다 가장 가까운 지점 값 (`data/static/krs_stations.csv`) |
| τ_wasp (주의) | 44.905 = 1 km 한반도, 8–10월, CWRI > 0, 2017–2024의 75번째 백분위 |
| 경계·심각 | 새 지수 평년의 7일 평균 CWRI P90·P97 (`data/static/breakpoints.json`) |
| 상습 지역 | 계절 Gi*(7×7 칸, ±3 km) z > 1.96 이었던 해의 비율 ≥ 0.87 (2017–2024, 논문 2 래스터) |

코드 키는 래스터 폴더 이름을 그대로 씁니다: `V_simillima` = 좀말벌 *V. analis*, `V_analis` = 꼬마장수말벌 *V. ducalis*.

**골든 테스트:** `python tests/golden/test_golden.py` — 앱 코드가 논문 기준 코드(`tests/golden/paper_index_reference.py`)와 같은 값을 내는지 확인 (CSI 5종·CWRI·BAI_E 244행, 활동창, 일사).

## 저장소에 있어야 하는 것
- 코드: `run_daily.py`, `indices.py`, `grid.py`, `export.py`, `sources.py`, `index.html`, `local.html`, `requirements.txt`
- `.github/workflows/daily.yml` (매일 자동 실행)
- `data/static/`: `mask_1km.tif`, `regions_1km.tif`, `regions.csv`, `dem_1km.tif`, `persistence_*.tif` 3개, `breakpoints.json`, `climatology_sgg.csv`, `krs_stations.csv`, `beekeepers_sgg.csv`(양봉 농가 수, 자료가 오면 채움)
- `data/stations/` (관측 보관소, 매일 자동 추가), `data/archive/` (시군구 요약, `runlog.csv` 매일 실행 기록)

## 설정
- Settings > Secrets and variables > Actions: `KMA_KEY` (API허브 키)
- Settings > Pages > Source: **GitHub Actions**

## 자료
- 관측: `sfc_aws_day.php`(최고·최저기온, 강수) + `awsh.php` 1~24시(평균기온·습도·풍속, 18시간 이상일 때) — 하루 27회 호출
- 산림청 산악기상(AMOS)은 계산에서 제외

## PC에서 한 번씩 (평년·경보 기준 다시 만들기)
```
python build_climatology.py years 1991 2025
python build_climatology.py aggregate 1991 2020
python build_climatology.py breakpoints 1991 2020 --check 2017 2024
```
결과 `data/clim/p2/climatology_sgg.csv`, `data/clim/p2/breakpoints.json` → 저장소 `data/static/` 로 올림.
그 밖의 도구: `prep_persistence.py`(상습 지역 층), `tools/build_krs.py`(kRs 표), `make_mask.py`, `local_import.py`, `aws_hourly_humidity.py`, `check_point.py`.

## 양봉 농가 노출층
`data/static/beekeepers_sgg.csv`의 `farms`(필요하면 `colonies`)를 채우면 전국 화면에 시군구별 "주의 이상 면적"(위해)과 농가 수(노출)를
나란히 보여 주고 2차원(둘 다 높음 / 위해만 / 노출만 / 둘 다 낮음)으로 나눕니다. **지수와 농가 수를 곱해 하나의 점수로 만들지 않습니다.**
