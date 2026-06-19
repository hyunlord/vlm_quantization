# PAPER MULTILING — 추가 언어 멀티링구얼 평가 (확장 ②)

C1/확장① 파이프라인을 **재학습 없이** 멀티링구얼 벤치마크 **XM3600**(Crossmodal-3600)로 확장. 목적 = 헤드라인 "multilingual" 주장(지금까지 EN+KO 2개)을 **36개 언어**로 강화하고, **언어별 최적 오프라인 인코더**(배포 정보)를 얻는다. **eval-only** — 백본·인코더·헤드 모두 frozen, 벤치마크 이미지를 ft113 `img_h`로 인코딩한 **별도 갤러리**만 새로 만든다. COCO `index.bin`/`img_h`/`common.py`/기존 헤드 **무변경**.

## 데이터 — XM3600 (Crossmodal-3600)
- **3600 이미지**(Open Images, 지역 균형) × **36개 언어 사람 캡션**(한국어 `ko` 포함). 멀티링구얼 image-text 검색 표준(EMNLP'22).
- 직접 다운로드(HF `datasets` 의존 없음): 캡션 `captions.jsonl`(google.github.io/crossmodal-3600) + 이미지 `images.tgz`(Open Images S3). `web/paper_multiling_fetch.sh` 참고.
- 캡션 수/언어 ≈ 7,200(이미지당 ~2 캡션; en 7,200 · ko 7,650 · de 8,643 · bn 3,600 등 언어별 상이). 각 캡션을 개별 text→image 쿼리로 사용.
- **XM3600 ≠ COCO 분포**: 절대 R@K는 COCO 5K 표(서버 EN 79.92/KO 71.08)와 **직접 비교 불가** — 이미지 분포·캡션 작성 방식이 다름. 별도 벤치마크로 보고. 포인트 = **언어 커버리지 + 언어별 offline−server gap + 인코더별 언어 프로파일**.
- fallback(미사용): XTD10(COCO 이미지, 8개 언어, MT 캡션) — 스크립트에 포함(`--bench xtd`).

## 방법 — 재학습 없는 갤러리 + 5개 인코더
- **갤러리(공유)**: 3600 이미지 → so400m vision tower(MAP-head `pooler_output`) → **frozen ft113 `img_h`** → 1024-bit pack. 오프라인 헤드가 모두 이 ft113 img_h 코드 공간에 정렬돼 있어 **한 갤러리를 모든 인코더가 공유**.
- **언어별 평가**: 각 언어 캡션을 인코더별 **trained config**로 임베딩 → 그 인코더의 학습된 헤드 → 1024-bit pack → 갤러리에 faiss `IndexBinaryFlat` Hamming Top-K → **text→image R@{1,5,10}** (gold = 캡션→해당 이미지). 캡션이 이미지당 복수면 각 캡션을 개별 쿼리로.
- **인코더(행) 5개** — 오프라인 헤드는 확장①에서 학습된 것을 그대로 재사용(헤드 .pt에 student·prefix 기록, 동일 pooling/prefix로 캡션 인코딩, 재학습 0):

  | 인코더 | 임베딩 경로 | pooling/prefix | dim |
  |---|---|---|---|
  | `so400m-float (ceiling)` | so400m text vs image **cosine**(해시 없음) | MAP-pool, L2 | 1152 |
  | `server (so400m+ft113)` | so400m text → ft113 `txt_h` | MAP-pool, L2(norm_in) | 1152 |
  | `e5-small (C1, no-prefix)` | multilingual-e5-small → `txt_h_e5` | mean-pool, no-prefix, L2 | 384 |
  | `MiniLM-L12-v2` | paraphrase-multilingual-MiniLM-L12-v2 → head | mean-pool, no-prefix, L2 | 384 |
  | `e5-base (query:)` | multilingual-e5-base → head | mean-pool, **"query: "**, L2 | 768 |

  `so400m-float`은 서버 해시의 **상한(ceiling)** 으로만 포함(배포 비대상) — 양자화 갭·언어별 작동을 calibration.

## 비교표 — 대표 14개 언어 (text→image **R@10**; R@1/5/10·36개 전부 = `paper/multiling.csv`)
**굵게** = server / 최고 offline. `*` = offline이 server를 **추월**.

| lang | float(ceiling) | **server** | e5-small | **MiniLM** | e5-base |
|---|---|---|---|---|---|
| ko | 88.9 | **83.8** | 54.2 | **60.2** | 59.2 |
| en | 83.4 | **77.9** | 59.2 | **68.7** | 65.0 |
| vi | 91.0 | **87.0** | 40.1 | **70.7** | 56.6 |
| id | 90.4 | **85.6** | 43.4 | **73.2** | 52.7 |
| ru | 91.0 | **84.0** | 43.6 | **69.9** | 53.4 |
| fr | 92.6 | **86.5** | 49.9 | **66.8** | 57.1 |
| es | 83.6 | **74.3** | 44.3 | **60.3** | 56.0 |
| ar | 83.6 | **77.3** | 32.0 | **56.8** | 39.2 |
| zh | 81.9 | **76.3** | 38.6 | **71.4** | 50.6 |
| ja | 80.6 | **74.9** | 42.3 | **68.7** | 50.4 |
| tr | 80.0 | **73.5** | 29.7 | **61.3** | 43.6 |
| th | 60.0 | 57.5 | 39.4 | **69.5*** | 48.4 |
| hi | 40.9 | 43.4 | 30.5 | **52.4*** | 30.0 |
| de | 37.6 | 30.1 | 46.4 | **66.6*** | 55.7 |

전체 36개 언어 평균 R@10: float **66.9** · server **62.4** · MiniLM **53.9** · e5-base **43.3** · e5-small **34.1**.

## 해석
- **(i) 멀티링구얼 작동 = YES (헤드라인 강화)**. server가 **36개 중 28개 언어에서 R@10 ≥ 50**(평균 62.4), 해시는 float ceiling 대비 **R@10의 92.9%를 유지**(언어 전반 일관 — 양자화 갭이 작음). 시스템이 EN-편향이 아님: **KO(83.8)가 EN(77.9)보다 높고**, 상위권(fr 86.5 · vi 87.0 · id 85.6 · ru 84.0). 즉 EN+KO 2개 → **30개+ 언어로 주장 확장**. 저자원 3개(mi 1.5 · quz 7.7 · sw 13.9)는 모든 인코더가 실패(언어 미커버) — so400m·XLM-R 공통 한계.
- **(ii) 언어별 offline−server gap = 언어 의존적**. 커버 언어에서 **최고 offline이 server R@10의 ~86.7%를 회복**. 단, 갭은 언어마다 크게 다름:
  - **고자원(server 강세)**: fr/ru/vi/id/ar/ko/zh/ja/es/it … server ≫ offline(offline이 60–85% 회복). so400m 텍스트 타워가 강한 언어.
  - **offline ≥ server (8개 언어, so400m 텍스트가 병목)**: **de(server 30.1 → MiniLM 66.6, 2.2×)**, **te(6.2 → e5-base 28.4)**, **th(57.5 → MiniLM 69.5)**, **hi(43.4 → MiniLM 52.4)** + 한계적 fi·hr·mi·sw. 이 언어들은 so400m **텍스트** 임베딩이 약해(이미지·해시 문제 아님 — 동일 갤러리, float ceiling도 동반 저하), 전용 멀티링구얼 텍스트 인코더 + head-adapt가 server를 **추월**.
  - **독일어(de)는 두드러진 outlier**: float ceiling 37.6 — 다른 주요 유럽어(fr/es/it/pt ~83–93)보다 현저히 낮음 = **so400m 텍스트 타워의 독일어 약점**(배포 contract max_len=64 하). offline MiniLM이 2.2× 회복.
- **(iii) 인코더별 언어 프로파일(배포 권고)**: **MiniLM-L12-v2가 36개 중 30개 언어에서 최고 offline**(주요 Latin/Cyrillic/CJK + Thai; 평균 R@10 53.9 ≫ e5-base 43.3 ≫ e5-small 34.1). 확장①의 EN 결론(MiniLM이 EN 최고)이 **멀티링구얼 전반으로 일반화**. e5-base는 **특정 저자원/스크립트(fil·mi·quz·sw·te)에서만** 최고, e5-small은 bn 1개. → **단일 오프라인 인코더로는 MiniLM이 광범위 멀티링구얼 최적**(현 배포 e5-small 대비 평균 R@10 +19.8pt); 위 특정 저자원어 타깃 시에만 e5-base 고려.

## 재현
```bash
# DGX. 데이터 취득(재실행 안전):
bash web/paper_multiling_fetch.sh data
# 전체 평가(36개 언어, 재학습 없음):
HEAD_PATH=/tmp/ft_ko_113.pt .venv/bin/python web/paper_multiling_run.py --bench xm3600 --data data/xm3600 --out paper/multiling.csv
```
산출물: `web/paper_multiling_run.py`(평가) + `web/paper_multiling_fetch.sh`(데이터) + `paper/multiling.csv`(language×encoder, R@1/5/10) + `paper/multiling.json`(그림용). 다운로드 데이터·헤드(`/tmp/txt_h_*.pt`, 확장①에서 생성) 미커밋. 앵커: COCO 서버(79.92/71.08)는 별도 분포라 직접 비교 대상 아님(§데이터).

## 채우는 표/섹션 + 커밋
- **기여 ① / §6 Experiments — 멀티링구얼 평가**: EN+KO → **36개 언어**(28개 R@10≥50)로 확장; **언어별 offline 배포 프로파일**(MiniLM 30/36 최고). so400m-float ceiling 행은 **양자화 갭(해시가 float의 92.9% 유지)** 도 언어 전반에서 입증.
- 브랜치 **`paper-multiling`**(`web-v3-hybrid`에서 분기). 평가 커밋 **`971185b`** → `origin/paper-multiling` push 완료.
