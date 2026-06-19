# Hash Quality — Experiment Ledger (append-only)

> **목적**: cross-modal 1-bit 해싱의 "품질 격차" 조사 전 과정을 **빠짐없이, 하나의 파일에 누적**한다. 모든 옵션의 벤치마크 지표·실험 설정·분석을 시간순으로 append하고, 언제든 복기할 수 있게 한다.
> **규칙**: 기존 항목은 **지우거나 덮어쓰지 않는다**(정정은 새 줄로 "⚠ 정정"). 새 실험은 §7 Experiment Log에 날짜 항목으로 추가. §2 Running Conclusions만 살아있는 요약으로 갱신.
> 최종 갱신: 2026-06-18

---

## 1. 한 줄 요약 (TL;DR)
**전 옵션을 1024-bit(최적 비트)로 재학습·재측정(§4d)** — 256 기본값은 과소평가였음. 1024에서:
- **pair-R@K**: hash R@10이 float의 **~99%**(256에선 95%)로 좁혀짐. best CC12M 403K R@10 80.5 / R@1 41.4 (float 81.1 / 47.7).
- **category-mAP**: CC12M/데이터가 baseline 70 → **75**로 실제 개선(I2T·T2I 일관), hash가 float(66.6)·NDCG도 초과.
- **1024에선 데이터가 양 축(pair-R@1 + category-mAP) 모두 유리** — 256에서 보이던 trade-off가 완화(974K-uncapped만 R@10↓=mix-ratio 흔적).
- **best 종합 = 혼합비 1:1~1:2 (cap 113K/226K)** — R@1 42.2~42.5 (CC12M 403K 41.4·uncapped 40.6 능가) + mAP 최대 유지. **데이터의 진짜 레버는 "양"이 아니라 "COCO:추가 혼합비"**(§4f). demo(h768)는 R@1 최저 → 깨끗한 레시피 > 추가 용량.
- 비트 trade-off: 1024=최고품질·최대비용(128B·faiss 18ms/1M), 256=최저비용(32B·4ms). `agreement@100`은 폐기(데이터 효과 거꾸로 본 artifact).

---

## 2. Running Conclusions (살아있는 요약 — 갱신 대상)
- **이진화는 무죄**: float-hash logits vs 1-bit 코드 top-k 100% 동일(B∩C=100%). 양자화 손실 ≈ 0.
- **검색단 트릭 전부 막다른 길**: rerank(bit→float/bit→bit), asymmetric — 이득 0 (이유 §7-R1).
- **데이터의 진짜 레버 = 혼합비(COCO:추가 step 비율), "양"이 아님 (§4f 양성 발견)**: 비율 스윕 결과 pair-R@1이 뒤집힌 U자 — **COCO-heavy 1:1~1:2에서 정점(42.2~42.5, baseline 39.8 대비 +2.4~2.7pt)**, 추가 과다(uncapped 1:8.6)면 40.6으로 깎임. category-mAP는 추가데이터 유무로만 결정(70→75)·비율 무관. → **best = 1:1~1:2.** (이전 "데이터=레버 아님"은 403K 비율 1점만 본 불완전 결론이었음.)
- **데이터 효과는 지표 의존**: category-mAP(semantic)에선 CC12M가 baseline +5pt 개선, pair-R은 혼합비에 따라 ±. 볼륨 자체는 plateau(403K≈974K 동일비율)지만 **비율이 레버**.
- **★ 2축 정리 (§4g 도메인-외)**: **혼합비(1:1~1:2)=in-domain(COCO) 정밀도 레버 / CC12M 데이터=out-of-domain 일반화 레버.** Flickr30K(미지)에서 CC12M이 baseline 대비 R@1 +7~9pt, bit가 float R@10의 95~99% 유지. 데이터의 진짜 가치는 *미지 데이터 일반화*에서 드러남(in-domain은 과소평가).
- **정규화는 gross 의미오류를 안 고침**: 통제 ablation에서 NORM·RAW 둘 다 "running lab→개". 정규화는 agreement@100만 흔들고 gold R@K엔 무영향.
- **`agreement@100`은 품질 목표로 부적합** (§5): float이 정답 아님 + 유효다양성을 페널티 + top-100 노이즈 + distillation 순환. **gold = 정답 쌍 R@K/MRR/MedR.**
- **코드용량↑ 레버 = 검증됨(§4d)**: 전 옵션 bits→1024 재학습 → hash R@10 float의 95%→**99%**(256→1024 +2.3~2.7pt), R@1 +3~4pt. **1024가 정확한-쌍 격차를 실질적으로 메움.** (256 기본값은 오판이었음.)
- **1024+데이터는 양 축 모두 개선**: CC12M가 pair-R@1·category-mAP 둘 다 올림(256의 trade-off 완화). best 종합 = CC12M 403K.
- **bit-old(데모)는 신규 1024 레시피들보다 오히려 낮음**(R@1 38.1 최저, hidden 768인데도). 깨끗한 캐시-emb 레시피 > 추가 용량. 데모 "도서관"은 단일 손쿼리 artifact.
- **쿼리 robustness(오타/노이즈): "비트가 더 견고" 가설 대체로 반증(§4e)** — bit1024가 결과셋 안정성(Stability@10)은 근소 우위지만 정답 유지력(R@10 retention)은 float 우위. 무승부~float 약우위. 데모의 "반지의 제왕" 견고성은 양자화가 아니라 다국어 아티팩트였음.

---

## 3. Setup & Infra (안정 레퍼런스)
- **태스크**: 이미지+텍스트 → SigLIP2 So400m(frozen, 1152-d) → NestedHashLayer → 1-bit 코드 → Hamming/XOR 검색.
- **NestedHashLayer**: Linear(1152→hidden)→LayerNorm→GELU→Dropout→Linear(hidden→max_bit), per-bit prefix(Matryoshka)→BatchNorm→L2norm→tanh("continuous")/SignSTE("binary").
- **데이터**: COCO Karpathy(train+restval 113,287 / val 5K / test 5K). 확대용: CC12M(pixparse 이미지 + CaptionEmporium llavanext clean 캡션, key join 100%), OI Localized Narratives(167,711 image-text).
- **손실**(CombinedHashLoss): InfoNCE(1.0)+EAQL(quant,ramp)+OrthoHash(0.1)+BitBalance(0.01)+Consistency(0.5)+LCS(0.5). broadening 실험은 +RKD(distance-wise, μ정규화 Huber)+CroVCA coding-rate diversity.
- **인프라**: DGX Spark GB10(`ssh dgx-spark`). sm_121 PyTorch fallback(느림, but GPU 사용). 통합메모리 ~120GiB. 캐시 임베딩 학습(frozen backbone→cache→tiny head 분 단위).
- **핵심 캐시/파일**: `/tmp/emb_cache.pt`(COCO test img/txt/ids), `/tmp/emb_aug.pt`(train clean/weak/strong/txt), `/tmp/hp_results.json`(Optuna best_params, hidden 384, 256-bit 튜닝), `/tmp/oi_index_167k.npz`(OI 코퍼스 emb+paths+packed_256), `/tmp/cc12m_pairs.pt`(403K), `/tmp/cc12m_pairs_big.pt`(806K), `/tmp/oi_pairs.pt`(167K), `/tmp/combined_pairs.pt`(974K).
- **스크립트**: `train_broaden.py`(modes·RKD·CroVCA), `train_ablate.py`(정규화 on/off 통제), `train_mixctrl.py`(OI_CAP 혼합비통제 + 고정평가쿼리), `eval_retrieval.py`(gold R@K/MRR/MedR).

---

## 4. ★ Gold 재측정 — 진짜 검색 품질 (2026-06-18, `eval_retrieval.py`)
**COCO test 5K, 정답 쌍 기준 cross-modal 검색, 256-bit prefix.** 각 헤드는 학습 시 입력규약(demo=raw, 나머지=L2norm)대로 평가.

| 옵션 | bit | I2T R@1 / R@5 / R@10 / MRR / MedR | T2I R@1 / R@5 / R@10 / MRR / MedR |
|---|---|---|---|
| **float (천장)** | emb | **47.68 / 71.82 / 81.10 / 58.85 / 2** | **49.74 / 72.88 / 81.58 / 60.53 / 2** |
| bit-old demo (raw) | 256 | 34.96 / 65.98 / 76.94 / 48.99 / 3 | 35.36 / 64.94 / 76.76 / 48.94 / 3 |
| coco baseline (A_coco) | 256 | 36.18 / 66.58 / 77.52 / 49.97 / 3 | 37.00 / 65.70 / 77.12 / 50.32 / 3 |
| CC12M 403K (A) | 256 | 36.96 / 65.92 / 76.64 / 50.15 / 3 | 37.12 / 65.48 / 76.44 / 50.35 / 2 |
| rich 974K capped (B) | 256 | 36.70 / 65.06 / 76.58 / 49.78 / 3 | 36.76 / 64.58 / 75.80 / 49.68 / 3 |
| ablate NORM | 256 | 36.76 / 66.24 / 77.42 / 50.26 / 3 | 37.58 / 66.50 / 77.20 / 50.58 / 3 |
| ablate RAW (raw) | 256 | 36.58 / 66.34 / 77.88 / 50.15 / 3 | 37.36 / 65.50 / 76.86 / 50.46 / 3 |

**해석**: hash 변종 전부 ±1~2pt(노이즈). 데이터·정규화·레시피 무영향. 유일 실질격차 = hash↔float(R@10 ~77 vs ~81 = float의 95%; R@1 ~37 vs ~48). bit-old는 R@1만 ~1.5pt 낮음.

### 4b. ★ Category-mAP — semantic gold (2026-06-18, `category_map.py`)
**relevance = COCO 80-cat 1개 이상 공유** (deep-hashing 문헌 표준). 쿼리당 평균 relevant=1644/5000(=32%, **느슨함 — 절대값 과대해석 금물, 상대비교용**). AP@K = Σ(P@k·rel_k)/#rel_in_topK.

| 옵션 | I2T mAP@100 / @1000 | T2I mAP@100 / @1000 |
|---|---|---|
| float (천장) | 84.03 / 66.56 | **90.52 / 77.26** |
| bit-old demo (raw) | 88.08 / 66.09 | 87.46 / 66.07 |
| coco baseline | 88.69 / 68.39 | 88.15 / 68.32 |
| **CC12M 403K (A)** | **90.51 / 72.98** | **89.61 / 71.89** |
| rich 974K (B) | 90.08 / 72.85 | 89.20 / 71.77 |
| ablate NORM | 88.49 / 68.11 | 88.01 / 68.14 |
| ablate RAW (raw) | 88.51 / 68.18 | 88.00 / 68.20 |

**해석**: **CC12M가 baseline 대비 +1.8(@100)~+4.6pt(@1000) 일관 개선** — pair-R@K(§4)가 못 본 효과. 즉 CC12M은 "정확한 쌍"이 아니라 "**같은 개념**" 검색을 키운다. I2T는 hash가 float 초과(느슨한 relevance + balance/BN이 카테고리 균등 분산 효과로 추정 — 과대해석 주의). T2I는 float 우위. 볼륨 plateau(403K≈974K)·정규화 무효(NORM≈RAW)는 여기서도 동일. **agreement@100은 이 효과를 거꾸로 봤다(데이터 많을수록 나쁘다 했으나 mAP는 좋다).**

### 4c. ★ 전 세팅 종합 — 비트길이·hash-native·진단 (2026-06-18, `eval_all.py`+`eval_hashing.py`+`build_eval_html.py`)
**인터랙티브 HTML: `claudedocs/hash_eval_comparison.html`** (헤드 10개 × 비트 8→256[demo는 1024] × 12지표 × I2T/T2I, 정렬 가능 마스터표 + SVG 곡선).
- **256b mAP@1000(I2T) 단조 상승**: baseline 68.39 → CroVCA-only 70.5 → CC12M 403K 72.98 → 974K capped 72.85 → **974K uncapped(R6) 74.18**. ↔ 같은 세팅 pair-R@10은 반대로 baseline 77.5 → uncapped 74.3로 **하락**. **명확한 trade-off: 데이터/넓힘 → 개념검색↑ 정확한쌍↓.** (CroVCA-only도 mAP +2pt — diversity 손실 단독 효과 확인.)
- **비트길이**: 256까지 가파름, 이후 평탄. demo 256→1024 R@10 76.9→79.1(+2.2pt뿐), mAP@1000 66.1→66.9. **256이 sweet spot.**
- **★ hash-native(Hamming 반경2)**: 8b cover 100%(R@H2~30%, P@H2~45%) → 16b cover100%(R@H2~3%) → 32b cover ~20% → **64b cover<0.5%, ≥128b cover 0%**. **즉 ≥128b는 O(1) 버킷 룩업 불가 → 선형 Hamming 스캔(faiss IndexBinaryFlat) 필수.** float은 이 지표 자체 없음(해시 고유 강점 지표).
- **★ 코드 진단(256b, 이미지코드)**: COCO 계열 bit_balance ~0.03·entropy ~0.999(우수) vs **CC12M 계열 bit_balance ~0.12·entropy ~0.98(악화)**. bit_indep ~0.07~0.10(비트↑→상관↑), quant_err 0.73(8b)→0.95(256b) 전 세팅 공통. **데이터 확대는 semantic mAP↑ 대가로 bit 균형↓.**
- hash가 float 초과(I2T mAP@1000 CC12M 72.98 vs float 66.56)는 §4b와 동일(느슨 relevance·BN 분산 효과 추정, 과대해석 주의). T2I는 float 우위.

---

### 4d. ★★ 전 옵션 1024-bit 재학습 + 재측정 (2026-06-18, `train_1024.py` → `eval_all`+`eval_hashing`)
모든 옵션을 bits→1024로 재학습(k1024_*, hidden 384; demo만 h768). **사용자 지적이 옳았음 — 256 기본은 전 세팅을 과소평가했었다.** COCO test 5K, **1024-bit, I2T**:

| 옵션 | R@1 | R@10 | MRR | mAP@1000 | NDCG@10 |
|---|---|---|---|---|---|
| **float (천장)** | **47.68** | **81.10** | **58.85** | 66.56 | 73.48 |
| bit-old demo (h768) | 38.12 | 79.06 | 51.77 | 66.93 | 75.78 |
| COCO baseline | 39.78 | 80.22 | 53.48 | 70.07 | 76.02 |
| COCO + CroVCA | 40.64 | 80.90 | 53.86 | 72.90 | 76.76 |
| **CC12M 403K** | 41.38 | **80.54** | 54.48 | **74.89** | 76.63 |
| 974K uncapped | 40.58 | 78.18 | 53.21 | **75.04** | 75.81 |
| 974K capped | **41.50** | 79.80 | 54.48 | 74.96 | 76.36 |
| COCO raw-input | 40.62 | 80.72 | 54.23 | 70.28 | 76.23 |

- **1024가 float 격차를 크게 좁힘**: 같은 k1024 헤드 256→1024 — baseline R@10 77.94→**80.22**, CC12M 77.86→**80.54**(+2.7pt). hash R@10이 float의 **~95%→~99%**로 상승. R@1도 +3~4pt. **256 기본값은 오판이었음(1024가 최적 비교 지점).**
- **CC12M/데이터는 양 축 모두 유리해짐**: 1024에선 CC12M가 pair-R@1(41.4, baseline 39.8 대비↑)·category-mAP(74.89 vs 70.07) 둘 다 올림. 256에서 보이던 "pair 무효" trade-off가 1024+데이터에서 **완화**(여전히 974K-uncapped만 R@10 78.18로 낮음=mix-ratio 흔적).
- **best 종합 = CC12M 403K**(R@10 80.54 + mAP 74.89 둘 다 상위), best R@1 = 974K capped 41.5.
- **demo(h768)가 R@1 최저(38.12)** — hidden이 더 큰데도 신규 384 레시피들에 밀림 → **깨끗한 캐시-emb 레시피 > 추가 용량**(demo의 한국어/full-image 레시피 열위).
- **정규화 무효 재확인**: raw-COCO R@10 80.72 ≈ baseline 80.22. hash가 category mAP·NDCG에선 float 초과(NDCG 76 vs float 73.5).
- 저장/속도(§HTML ⑥⑦): 1024는 128B(float 36×↓)·1M faiss 18.4ms(float 13×↑). 256은 32B(144×↓)·4ms(62×↑). **1024 최고품질 ↔ 256 최저비용**.
- 산출물: `claudedocs/hash_eval_comparison.html`(1024 기본, 비트선택, 설명 ①, 저장 ⑥, 속도 ⑦), 헤드 `/tmp/k1024_*.pt`, `scripts/train_1024.py`·`bench_speed_storage.py`.

### 4e. 쿼리 robustness — "비트가 오타에 더 강한가?" (2026-06-18, `scripts/robustness_bench.py`)
계기: 데모에서 "반지의 제왕" 오타 쿼리에 bit는 반지(ring), float은 한국문화 이미지로 흩어짐 → "비트가 더 견고?" 가설. COCO test 300쿼리, 섭동(오타15%/단어드롭30%/truncation60%) 후 top-10 측정(CC12M-403K head).

**① Stability@10** (top10 clean∩perturbed, 높을수록 견고):
| method | 오타 | 단어드롭 | trunc |
|---|---|---|---|
| float | 40.3 | 51.8 | 38.4 |
| bit256 | 39.6 | 49.1 | 37.7 |
| **bit1024** | **44.1** | **52.9** | **39.9** |

**② R@10 유지율** (섭동 후 *정답* top-10 잔류):
| method | clean | 오타 | 단어드롭 | trunc |
|---|---|---|---|---|
| **float** | 82.7 | 57.0(**69%**) | 61.7(**75%**) | 50.3(**61%**) |
| bit256 | 78.7 | 49.0(62%) | 55.3(70%) | 45.0(57%) |
| bit1024 | 81.0 | 52.3(65%) | 57.7(71%) | 46.0(57%) |

- **가설 대체로 반증**: 두 robustness가 갈림 — **결과셋 안정성(①)은 bit1024가 float보다 근소 우위**(전 항목 일관, 오타 +3.8pt), 그러나 **정답 유지력(②)은 float 우위**(오타 69% vs 65%). 즉 bit1024 결과가 더 일관되나, 그 결과가 정답일 확률은 float이 높음. 차이 수 pt(약함) → **robustness↔precision 무승부, 정답 기준 float 약우위.**
- **데모의 "반지의 제왕" 효과는 양자화 견고성이 아니라 다국어 아티팩트**(float이 한국어 "제왕"을 충실 반영 vs 영어학습 hash head가 "반지"에 집중) + 일화. 영어 오타 통제 실험에선 비트 견고성 우위 안 나타남.
- 교훈: 단일 일화로 "비트가 더 견고"라 결론짓지 말 것 — 측정하니 정답성은 float이 약우위.

### 4f. ★★ 혼합비 스윕 — 데이터의 진짜 레버는 "비율" (2026-06-18, `scripts/sweep_eval.py`) — 양성 발견
결합 974K 풀 고정, OI_CAP(epoch당 추가데이터 step)만 변경 → COCO:추가 비율 스윕. 1024-bit, I2T, gold:
| 세팅 | COCO:추가 | R@1 | R@10 | MRR | mAP@1k | NDCG10 |
|---|---|---|---|---|---|---|
| float (천장) | — | 47.68 | 81.10 | 58.85 | 66.56 | 73.48 |
| baseline (추가0) | 1:0 | 39.78 | 80.22 | 53.48 | 70.07 | 76.02 |
| **cap 113K** | **1:1** | **42.18** | **80.56** | **55.28** | 74.93 | **76.86** |
| **cap 226K** | **1:2** | **42.46** | 80.48 | 55.15 | 74.89 | 76.71 |
| cap 403K | 1:3.6 | 41.50 | 79.80 | 54.48 | 74.96 | 76.36 |
| cap 605K | 1:5.3 | 40.66 | 79.20 | 53.78 | 75.24 | 76.23 |
| uncapped 974K | 1:8.6 | 40.58 | 78.18 | 53.21 | 75.04 | 75.81 |
(6점 곡선 완성. 1:5.3은 하강부에 정확히 안착 — 뒤집힌 U자 확정.)

- **pair-R@1이 뒤집힌 U자**: baseline 39.78 → **1:1~1:2 정점 42.2~42.5** → 1:3.6 41.5 → uncapped 40.58. **추가데이터가 COCO-heavy 비율(1:1~1:2)에선 정확한-쌍 +2.4~2.7pt 향상**(과다 추가 시 깎임). R@10·MRR·NDCG도 1:1에서 정점.
- **category-mAP는 추가데이터 유무로만 결정(70→75, +5pt), 비율엔 거의 무관**(74.9~75.0 전 구간).
- **sweet spot = 1:1~1:2**: pair-R 최고 + mAP 최대 유지. **이전 best "CC12M 403K"(R@1 41.4)를 1:2(42.5)가 능가** → 새 최강 세팅.
- **이전 결론 정밀화**: §4d/mixctrl의 "혼합비 통제해도 데이터 무익"은 **403K 비율 1점만 본 것**이었음. 비율 스윕하니 **데이터의 진짜 레버 = 혼합비(양 아님)** — COCO-heavy 1:1~1:2가 pair-R을 실제로 올림. 헤드 `/tmp/sweep_c{113287,226574}_*.pt`, `k1024_974{c,u}_*.pt`.

### 4g. ★★ 도메인-외 일반화 — Flickr30K (2026-06-18, `scripts/flickr_bit_bench.py`)
**한 번도 학습 안 한** Flickr30K-1K test(5-caption 프로토콜)로 float vs bit 헤드 평가. 학습=COCO+CC12M+OI뿐.
| 세팅 | I2T R@1/5/10 | T2I R@1/5/10 |
|---|---|---|
| float (천장) | **92.9** / 99.3 / 99.9 | **79.02** / 94.18 / 96.98 |
| **CC12M 403K** @1024 | **85.4** / 97.5 / 99.3 | **69.94** / 89.8 / 93.92 |
| best 1:2 @1024 | 83.8 / 97.2 / 98.7 | 67.52 / 89.24 / 93.66 |
| COCO baseline @1024 | 76.7 / 93.8 / 97.1 | 62.6 / 85.66 / 90.96 |

- **bit 해시가 미지 데이터에도 잘 일반화**: R@10이 float의 **95~99%**(I2T 99.3 vs 99.9, T2I 93.9 vs 97.0).
- **★ 반전: out-of-domain 최강 bit = CC12M 403K**(I2T R@1 85.4 / T2I 69.9), best-1:2(83.8/67.5)·baseline(76.7/62.6) 능가. **in-domain COCO에선 best-1:2가 pair-R 최고였는데 Flickr에선 CC12M이 더 잘 일반화.**
- **★★ 데이터(CC12M)의 진짜 가치 = 일반화**: baseline → CC12M로 **I2T R@1 +8.7pt(76.7→85.4), T2I +7.3pt(62.6→69.9)**. in-domain COCO 평가는 이 효과를 과소평가(거기선 +1~2pt)했음.
- **정밀화된 2축 결론**: **혼합비(1:1~1:2) = in-domain 정밀도 레버 / CC12M 데이터 = out-of-domain 일반화 레버.** 배포 도메인이 다양/미지면 CC12M 403K, COCO-유사면 best 1:2.

**DOCCI test (5000, 1:1 dense 캡션, 미지, `scripts/docci_bit_bench.py`)** — Flickr 확증·강화:
| 세팅 | I2T R@1/5/10 | T2I R@1/5/10 |
|---|---|---|
| float (천장) | 65.38 / 88.28 / 93.14 | 67.14 / 87.98 / 92.58 |
| **CC12M 403K** @1024 | **47.5** / 74.74 / 83.26 | **50.16** / 76.06 / 83.58 |
| best 1:2 @1024 | 46.5 / 74.28 / 82.96 | 49.92 / 75.76 / 83.98 |
| COCO baseline @1024 | 29.4 / 54.8 / 65.66 | 32.44 / 58.34 / 68.74 |

- **CC12M의 일반화 이득이 두 벤치 모두 + DOCCI가 더 큼**: baseline→CC12M I2T R@1 — Flickr +8.7pt(76.7→85.4), **DOCCI +18.1pt(29.4→47.5)**. **도메인 차이 클수록 데이터 일반화 이득↑** (DOCCI dense 캡션이 COCO 짧은 캡션과 멀어 baseline 절반 추락).
- **CC12M 403K ≥ best 1:2 (두 미지 벤치 모두)** — out-of-domain 최강 일관(in-domain best-1:2 우위와 대조).
- bit가 float R@10의 ~89%(DOCCI)~99%(Flickr) 유지 — 어려운 도메인서 양자화 격차↑지만 견고.
- **2축 결론 교차 벤치로 확증.**

## 5. 지표와 그 타당성
### gold (정답 기준 — 신뢰)
- **R@K (pair)**: 쿼리의 정답 쌍이 top-K에 드는 비율. retrieval 논문 표준(SigLIP2 R@1/5/10).
- **MRR**: 정답 첫 등장 rank의 역수 평균. 단일정답이면 mAP_pair과 동일.
- **MedR**: 정답 rank 중앙값(낮을수록↑).
- (후보) **category-mAP@K**: relevance=COCO 80-cat 공유. deep-hashing 고전 표준(mAP@5000). 80-cat 멀티핫 라벨 필요 → 현재 캐시에 없음(추가 작업).
- (후보) **NDCG@K**: graded relevance.

### diagnostic only (품질 목표로 부적합)
- **`agreement@100`** = hash top-100 ∩ float-emb top-100 / 100. ⚠ float은 정답 아닌 proxy / 유효다양성을 페널티 / top-100 경계 노이즈(±수pt) / distillation과 순환. **상대 진단엔 OK, 절대 품질·최적화 목표엔 부적합.** §4 gold가 이를 직접 입증(NORM vs RAW agreement 2배차 → gold 동일).

---

## 6. 데이터 확대 궤적 (agreement@100, 참고용 — gold는 §4)
| round | OI-loop 데이터 | agreement (coco→+data) | 평가쿼리 | 비고 |
|---|---|---|---|---|
| round-2a | OI 167K | 20.3→25.0 | OI 손쿼리 | corrected CroVCA 첫 양성 |
| round-3 | CC12M 403K | 17.6→**34.7** | cc12m(inflated) | sweet spot |
| round-5 | CC12M 806K | 17.4→33.0 | cc12m | 볼륨2배 무익 |
| round-6 | CC12M+OI 974K | 18.5→29.5 | combined | 결합 악화 |
| mixctrl A | CC12M 403K | 20.3→**30.5** | **고정 OI(비교가능)** | 진짜 기준 |
| mixctrl B | 974K→403K캡 | 20.3→25.3 | 고정 OI | 혼합비통제에도 A>B |
**⚠ cross-round agreement는 mixctrl(고정쿼리) 외엔 쿼리셋이 달라 직접 비교 불가. §4 gold가 최종 판정.**

---

## 7. Experiment Log (append-only, 시간순)

### R1 — 검색단 트릭 (2026-06-15)
- 가설: bit→float rerank / bit→bit rerank / asymmetric로 품질 회복.
- 결과: 전부 이득 0. rerank bit→float=저장 폭증, bit→bit=Matryoshka prefix 상관(256→1024 재랭킹=full-1024 동일), asymmetric=이진화가 lossy 아니라 무의미.
- 결론: 검색단은 막다른 길. 품질은 학습/표현 문제.

### R2 — naive distillation 실패 → corrected RKD/CroVCA (2026-06-15~16)
- naive(절대 sim L2, w=3.0): COCO R@10 0.782→0.705, 잘되던 쿼리 파괴(58→17%). InfoNCE와 충돌.
- 정정: **상대(scale-invariant)·저가중·decoupled**로. RKD distance-wise(μ정규화 Huber δ=1, w=1.0), CroVCA coding-rate diversity(w=0.1).
- coding_rate 버그수정: per-sample L2norm + XᵀX + (B+d)/(B·d) 재스케일(누락시 ~30x 오류).
- round-2a(robust 200쿼리×k=100): coco 20.3 → crovca 22.7(COCO 무손실) → coco_oi_rkd_crovca 25.0. corrected diversity 첫 신뢰성 양성.

### R3 — CC12M 403K (2026-06-17) ["성공"으로 보였음]
- CC12M(llavanext clean 캡션) 403,280쌍 + RKD@1.0 + CroVCA@0.1. agreement 17.6→34.7(~2배), COCO R@10 0.775→0.766.
- ⚠ 후속 정정: agreement는 cc12m 쿼리로 잰 inflated값. gold(§4)에선 CC12M가 baseline 대비 무영향.

### R4_1024 — 1024-bit (2026-06-17)
- 별도 전용 런(hidden은 broaden 기본). coco R@10 0.8024 / coco_oi_rkd_crovca 0.8056. running lab 0→8%, library 17→75%.
- 256-max 기본 런과 비교: 긴 코드 한계효용 작음(+1.3pt 수준). 1024 제대로는 hidden 768 필요.

### Ablation — 입력 정규화 on/off (2026-06-17, `train_ablate.py`)
- COCO-only·seed42·동일HP·256-max, 해시입력 정규화만 토글. float 기준선 항상 normalized 고정. CPU.
- running lab: **NORM=개 4/4, RAW=개 3/4** → 정규화는 gross 오류 안 뒤집음.
- agreement@100: NORM 20.0% vs RAW 10.9%(~2배). COCO R@10 0.7742 vs 0.7788(동일). float R@10 0.811.
- 결론: 정규화 효과는 agreement에만, gross 의미·정답성엔 무영향. **앞선 "정규화가 running-lab 고침" 주장 2회 과잉귀속 → 통제실험으로 반증.** 옛데모 "도서관"=옛 레시피 전체(아래) 탓, 단일원인 미규명.

### bit-old(데모) 실측 차이 (2026-06-17)
- demo_hashheads.pt: hidden **768**, bits→**1024**(확장 Matryoshka), experiment_highbit.yaml 계열(한국어 coco_ko_train.jsonl 혼합, lr1e-3·bs256·30ep, full-image Lightning).
- bit-base/ablation: hidden 384, bits→256, Optuna HP, 캐시학습, COCO만.
- **손실 6종·가중치 양쪽 완전 동일**(앞서 "손실구성 차이/Matryoshka 부재"라 쓴 건 ⚠ 오류 — 정정).
- 확정 차이: 한국어 데이터 / Matryoshka 1024확장 / hidden 768 / HP / full-image. running-lab "도서관" 단일원인 미규명(한국어 또는 1024-prefix 후보).

### R5 — CC12M 806K (볼륨2배, 2026-06-17)
- big.pt 806,560쌍(403K+403K APPEND). agreement coco 17.4→33.0(403K 34.7과 동일/노이즈). COCO R@10 0.752.
- 결론: **순수 볼륨 한계효용 도달.**

### R6 — 전 데이터 결합 974K (다양성, 2026-06-17)
- CC12M806K+OI167K=974,271. agreement 18.5→29.5, COCO R@10 0.7426(단조 하락).
- ⚠ 교란: COCO loop 113K 고정인데 OI loop만 커짐 → non-COCO step비↑(mix-ratio artifact 의심).

### Mixctrl — 혼합비 통제 (2026-06-18, `train_mixctrl.py`)
- OI loop를 epoch당 OI_CAP=403K로 캡 → COCO:OI step비 고정. 평가쿼리 고정(oi_pairs seed0 200캡, OI_PAIRS와 독립 → 비교가능).
- A(CC12M 403K, cap없음): agreement 20.3→**30.5%**, R@10 0.7664.
- B(결합974K→403K캡): agreement 20.3→**25.3%**, R@10 0.7658.
- 결론: **같은 예산·같은 비율에도 B(풍부풀)가 A보다 −5.2pt** → 악화는 artifact 아닌 실제. **데이터=레버 아님 확정.** A는 OI 미학습인데도 OI평가에서 B(OI학습)보다 높음.
- ⚠ 노이즈: agreement@100 200쿼리 ±수pt(멀티시드 미실시). gold(§4)가 최종.

### Gold 재측정 — pair-R@K (2026-06-18) → §4 표 참조
- 모든 헤드 COCO test 5K R@K/MRR/MedR. **데이터·정규화·레시피 전부 노이즈급 차이.** 유일 격차 hash↔float(R@10 95%). agreement@100 artifact 직접 입증.

### Gold 재측정 — category-mAP (2026-06-18) → §4b 표 참조
- 80-cat relevance, mAP@100·@1000. **CC12M가 baseline +1.8~4.6pt 실제 개선**(pair-R@K가 못 본 semantic 효과). I2T는 hash>float(느슨 relevance 주의). 볼륨 plateau·정규화 무효 재확인. **agreement@100이 데이터 효과를 반대로 봤음을 확정.**
- 라벨: `data/coco/annotations/instances_{train,val}2014.json`(80-cat), test id=cocoid 5000/5000 조인.

### 전 세팅 종합 비교 + hash-native/진단 (2026-06-18) → §4c, HTML `hash_eval_comparison.html`
- 헤드 10개 × 비트 8→256 × 12지표. 연구에이전트(scientist)로 hashing 문헌 지표 보강 → P@H≤2/R@H≤2/cover, NDCG@1000, bit_balance/indep/entropy/quant_err 추가.
- **trade-off 발견**: 데이터/넓힘 → category-mAP↑ but pair-R@10↓ + bit_balance↓. **Hamming 반경2 룩업 ≥128b 붕괴(cover0%)** → 선형스캔 필수. 256=sweet spot.

### Korean — 멀티링구얼 평가 (eval-first, 2026-06-18, `scripts/eval_korean.py`)
- 동기: 사용자 "한글 쿼리 강화". 학습 전 **기존 헤드의 한국어 작동 여부부터 측정**(over-engineering 방지).
- 셋업: coco_ko 한국어 캡션 text-only 임베딩(이미지emb 재사용) → `coco_ko_pairs.pt` 113,287 학습쌍(train+restval, 누수無) + `coco_ko_test.pt` 5K(test split). 같은 5K COCO test 이미지로 **EN vs KO T2I**(paired gold) R@1/5/10·MRR, 비트 64/256/1024.
- 학습데이터에 **한국어 0개** 상태 결과(1024bit T2I):

  | 헤드 | EN R@10 | KO R@10 | gap@10 | KO MRR |
  |---|---|---|---|---|
  | baseline (COCO만) | 80.02 | 62.18 | 17.84 | 0.3696 |
  | best (1:2 mix, COCO+CC12M+OI) | 79.70 | **65.32** | **14.38** | 0.4015 |
  | coco+cc12m+oi (974c) | 79.26 | 65.02 | 14.24 | 0.3971 |

- 발견: (1) **한국어 이미 작동** — 학습 한국어 0개인데 KO R@10 62~65%(SigLIP2 멀티링구얼 + 텍스트헤드 영/한 공유). (2) **다양한 데이터가 격차 축소** — baseline gap 17.84 → sweet-spot 14.38(한국어 학습 없이도 −3.5pt 전이; CC12M/OI 일반화가 한국어에 전이). (3) 그래도 **14pt 격차 잔존**(EN 80 vs KO 65) → 한국어 쌍 추가 재학습 가치 확인.
- ✅ **한국어 학습효과 분리측정**: `ml226_` = `pool_ml.pt`(974K + 한국어113K=10.4%), `OI_CAP=226574`(=**1:2 혼합비**). **정확한 동일조건 baseline = `sweep_c226574`(best 1:2, pair-R@1 42.46 = 전체 best)** — 둘 다 OI_CAP=226574, 유일 차이=한국어 포함. ⚠**정정**: 앞서 baseline으로 쓴 `974c`는 cap 403K(=1:3.6 혼합비)라 동일조건 아니었음(§4f). (학습 1422s)

  | 헤드 (둘 다 1:2) | EN R@10 | KO R@10 | gap@10 (1024) | gap@10 (256) |
  |---|---|---|---|---|
  | best 1:2 `sweep_c226574` (한국어無) | 79.70 | 65.32 | 14.38 | 16.44 |
  | ml226 (+한국어10.4%) | 78.76 | **66.60** | **12.16** | **12.06** |

  → 한국어만 추가(동일 1:2): KO R@10 **+1.28pt(1024)/+2.32pt(256)**, **gap −2.22pt(1024)/−4.38pt(256)**, EN **−0.94pt(1024)/−2.06pt(256)** (정확한 동일조건에선 영어 비용이 974c 비교보다 큼). **한국어 학습 효과 실재 확인.** 한국어 10.4%로 제한적 → 최종 통합헤드에서 비중↑ 여지.
- **float ceiling 대비** (1152-dim cosine, T2I, `scripts/eval_float_ceiling.py`): float EN R@10 **81.58** / KO **66.06**.

  | | float ceiling | best 1:2(한국어無) | ml226 hash(한국어有) |
  |---|---|---|---|
  | EN R@10 | 81.58 | 79.70 (97.7%) | 78.76 (96.5%) |
  | KO R@10 | 66.06 | 65.32 (98.9%) | **66.60 (100.8%)** |

  - EN: hash=float **96.5~97.7%** (정상 1-bit 양자화 손실).
  - KO: **float ceiling 자체가 EN보다 15.5pt 낮음**(SigLIP2 raw의 한국어 약점). best 1:2(한국어無)도 float 아래(98.9%)지만 **ml226 hash는 float을 넘어섬(100.8%)** → 한국어 학습 헤드가 단순 양자화가 아니라 **SigLIP2의 한국어 정렬 약점을 보정**. 영어는 SigLIP2가 이미 잘 정렬돼 float이 진짜 상한이나, 한국어는 학습헤드가 raw cosine을 능가 가능.
- ✅ **(A) fine-tune vs (B) from-scratch** (사용자 호기심: "한국어만 이어학습=forgetting?"). `scripts/finetune_korean.py`: best 1:2(`sweep_c226574`) 로드 → 한국어만 5ep, lr×0.1(3.06e-5), MIX_EN=0(영어 안 섞음). 223s.

  | 1024bit | EN R@10 | KO R@10 | gap | KO MRR |
  |---|---|---|---|---|
  | base best 1:1 (한글無) | 80.16 | 65.36 | 14.80 | 0.3936 |
  | base best 1:2 (한글無) | 79.70 | 65.32 | 14.38 | 0.4015 |
  | B: ml226 (1:2 풀10% from-scratch) | 78.76 | 66.60 | 12.16 | 0.3993 |
  | B: ml113 (1:1 풀10% from-scratch) | 79.30 | 65.92 | 13.38 | 0.4015 |
  | A: ft226 (1:2 한국어 fine-tune) | 79.62 | 70.22 | 9.40 | 0.4417 |
  | **A': ft113 (1:1 한국어 fine-tune)** | **79.98** | **71.10** | **8.88** | 0.4376 |

  → **A(fine-tune) >> B(from-scratch) 확정** (동일 best1:1 베이스 직접비교): **A' ft113 KO 71.10 vs B ml113 KO 65.92 = +5.18pt**, EN도 A'(79.98) > B(79.30). **from-scratch는 한국어 풀10% 섞어도 baseline 65.3대에서 거의 안 오름**(65.9~66.6, 희석); **fine-tune은 한국어 100% 집중 → KO +5~6pt + forgetting 거의無**(A' EN 80.16→79.98 −0.18pt). **A' ft113 = 종합 최강**: KO 71.10(전옵션 최고, float 66.06의 **107.6% 추월**)·EN 79.98·gap 8.88. 이유 ①낮은lr×0.1+5ep로 헤드 거의 불변 ②한국어img=COCOimg(앵커동일) ③SigLIP2 멀티링구얼. → **배포 후보 = ft113**. **교훈: 소수 언어/도메인 강화는 from-scratch 혼합보다 fine-tune이 압도적**(혼합은 희석, fine-tune은 집중).
- ✅ **ft 영어 OOD 검증** (한번도 학습 안 한 Flickr30K-1K + DOCCI, `scripts/{flickr,docci}_bit_bench.py`, 1024bit — 한국어ft가 영어 OOD 해치는지):

  | R@10 | Flickr T2I | Flickr I2T | DOCCI T2I | DOCCI I2T |
  |---|---|---|---|---|
  | best 1:1 | 93.20 | 98.6 | 83.02 | 82.32 |
  | ft113 (1:1+koft) | 93.72 | 99.0 | 82.82 | 80.72 |
  | best 1:2 | 93.66 | 98.7 | 83.98 | 82.96 |
  | ft226 (1:2+koft) | 93.76 | 98.7 | 83.98 | 82.72 |

  → **한국어 fine-tune이 영어 OOD 거의 불해침**. Flickr: ft 동급~약간 우위(회귀無). DOCCI: ft226 거의 유지(I2T −0.24), **ft113만 DOCCI I2T −1.6pt 소폭 회귀**(T2I −0.2 미미). 종합: ft113=한국어 최고(KO71.1)+DOCCI I2T 소폭 trade-off, **ft226=OOD 가장 안전(회귀≈0)+KO70.2**. 둘 다 영어 거의 유지하며 한국어 대폭↑ → fine-tune의 forgetting은 in-domain·OOD 모두 미미 확정.
- ✅ **마스터표 gold 12지표 반영** (eval_all/eval_hashing 재실행, 14헤드 — sweep 2 + ml/ft 4 추가). 영어 COCO test I2T 1024bit: ft113 R@10 80.32·mAP@1000 74.58 ≈ best1:1 80.56·74.93(영어 gold 거의 무손실 재확인). **fine-tune(ft)이 from-scratch(ml)보다 영어 R@1 보존 우수**: ft 41.3~42.0 vs ml 40.0~40.3(혼합이 영어 R@1 더 깎음 = 한국어 결론과 일관). HTML(`hash_eval_comparison.html`) 마스터표②·곡선③·hash-native④·진단⑤에 6헤드 추가 + ⑨ OOD ft + ⑫ 한국어 멀티링구얼 섹션 신규.
- ✅ **CC12M 1.21M 확대 평가** (`cc12m_pairs_12.pt` 806K+403K=1.21M → `combined_12.pt`+OI=1.38M. `k12b113`=확대 best1:1, `ft12_113`=확대+한국어ft):
  - **in-domain**(1024 T2I): k12b113 EN 80.30/KO 65.54 ≈ best1:1 80.16/65.36 (**볼륨 plateau 재확인**). ft12_113 KO 71.06 ≈ ft113 71.10, EN 79.20 < 79.98(약간↓).
  - **OOD** R@10: 확대(best1:1→k12b113) Flickr T2I 93.20→93.58·DOCCI T2I 83.02→83.72(+0.7)·I2T 82.32→82.5 — **소폭 개선**. ft12_113 ≈ ft113. CC12M 403K(순수, OI無)가 OOD 여전히 최고(Flickr T2I 93.92).
  - **결론: 806K→1.21M 확대는 효과 미미**(in-domain plateau, OOD +0.2~0.7pt). **볼륨≠레버 재확인**(806K도 이미 충분 다양성). **ft113(806K+한국어ft)이 best 실용 헤드** — 확대는 디스크11GB·시간만↑, 추가 가치 거의 無.
- ✅ **데모 ft113 추가 완료** (`demo/live_server.py` HEAD_FILES 맨 앞 '한국어+영어 (ft113)' = `/tmp/ft_ko_113.pt`, 코퍼스 1,086,903 인코딩, 포트 8200, faiss IndexBinaryFlat). 한국어 쿼리 '고양이'→고양이 이미지 정확 검색 확인. **전체 마무리**: HTML 완전판(14헤드×12지표 + ⑫한국어 + ⑨OOD), 레저 전부 기록, 데모 가동.

---

## 8. Open Questions / 다음 레버
0. **★ 배포 목표 = 어느 gold?** pair-R@K(정확한 쌍) vs category-mAP(개념 검색). 목표가 후자면 **CC12M은 이미 효과적**이고 더 키울 가치 있음. 전자면 코드용량이 관건. → 제품 의도 확정이 다음 실험 방향을 가른다.
1. **코드 용량↑** (pair-R@K 격차용): hidden 384→768 + 1024-bit + HP 재튜닝(옛 데모 스펙). hash↔float R@10 4pt 격차가 양자화 용량 한계인지 검증.
2. **상대구조 손실 강화**: RKD 가중/형태 조정 — pair-R@K가 오르나(category-mAP는 이미 CC12M로 개선됨).
3. ✅ **category-mAP@K 완료**(§4b) — agreement 대체 gold로 확립. 후속: mAP@100도 cross-modal i2i/t2t 확장 가능.
4. **본질 ceiling**: pair-R@10 95%가 1-bit×256×소형head 천장일 수 있음(데이터/정규화로 안 움직임). category-mAP는 아직 데이터로 개선 여지.
5. **멀티시드**: 주요 비교 ±분산 측정으로 노이즈 확정(특히 category-mAP의 hash>float, agreement@100 ±수pt).
6. **I2T에서 hash>float 규명**: balance/BN의 카테고리 균등분산 효과 가설 검증(느슨 relevance artifact인지 진짜인지).
