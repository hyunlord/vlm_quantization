# V2 HEADADAPT SPIKE — 텍스트 헤드를 e5에 맞춰 재학습 (사이클 5, C1)

**판정 = 조건부 GO** (근사 오프라인 모드용). 이미지 코드 고정 + **native e5 → 새 텍스트 헤드 `txt_h'`** 학습 → 사이클 4 대비 **R@10 +3.1(EN)/+3.7(KO) 개선**(KO는 서버 −4.9로 5pt 바 안). 단 **Top-10 overlap은 오히려 하락**(0.42/0.36 vs c4 0.51/0.45). 학습 스파이크(브라우저 통합/ONNX/UI 없음). `img_h`/`index.bin`/`common.py`/v1 전부 무변경.

## 한 줄 결론
C1은 cycle 4의 근본 원인(e5≠so400m × so400m-head 비트민감도)을 **head를 e5에 직접 맞춰** 제거 → **recall(R@10) 개선**. 그러나 서버와 *다른 head*라 **서버 top-K 일치(overlap)는 구조적으로 하락**. 즉 "서버와 동일"은 못 되지만 "오프라인 자체 품질"은 cycle 4보다 좋고 서버에 근접. 게다가 **배포가 더 깨끗**(stock e5 ONNX + 2.2MB `txt_h'`; cycle 4는 472MB 커스텀 e5 필요).

## 사전 조사
- **손실 스택**: `src/losses/combined.py:CombinedHashLoss` = InfoNCE(cross-modal contrastive) + EAQL(quant) + OrthoHash + BitBalance + LCS — **각 head의 continuous 출력**에 작동(per bit length). `train_1024.py`가 frozen 백본 + 캐시 임베딩으로 head만 학습해 71/80을 낸 그 레시피.
- **고정 이미지 코드 소스**: **ft113 `img_h`**(= `index.bin`을 만든 바로 그 head)를 `emb_aug.train.clean`(so400m 이미지 emb, 113K, L2)에 적용 → frozen anchor. 같은 head라 코드 공간 동일(Hamming 비교 가능).
- **학습 쌍**: 113K EN(이미지×caption[0]) + 113K KO(이미지×coco_ko_train caption[0], 매칭) = **226,574 pairs**.
- **e5 native emb**: `intfloat/multilingual-e5-small` mean-pool, **frozen·precompute**(384-d, 캐시 `/tmp/e5_train_emb.pt`). (cycle 4의 so400m-distilled e5가 아니라 **native** e5.)

## 학습 설정
- frozen ft113 `img_h`(anchor) + 학습가능 **`txt_h' = NestedHashLayer(384→hidden 384→1024-bit)`** on native e5(frozen). 이미지 브랜치 detach.
- 손실 = CombinedHashLoss(hp_results 가중치: ortho 0.18 / quant 0.13 / balance 2e-4 / cons 0.65 / lcs 0.65 / temp 0.083), AdamW lr 3.06e-4 + OneCycleLR, BS 512, **25 epochs = 11,050 steps, 15min**(캐시 임베딩이라 head 학습은 빠름). loss 2.88→1.52.
- **C1b(e5 backbone fine-tune)는 미실행** — 근거는 go/no-go 참조.

## 파리티 표 (hold-out 5K test, eval_korean 프로토콜 = 71/80 사과-대-사과)
| 경로 | EN R@10 (Δ서버, Δc4) | KO R@10 (Δ서버, Δc4) | Top-10 overlap EN/KO (Δc4) | 매칭쌍 Hamming off/srv | 인코드 |
|---|---|---|---|---|---|
| 서버 so400m | 79.92 | 71.08 | 1.0 / 1.0 | 234 / 245 (/1024) | (서버) |
| 사이클 4 (distill e5→so400m txt_h) | 70.86 (−9.06) | 62.50 (−8.58) | 0.512 / 0.451 | — | — |
| **C1 (native e5 → txt_h')** | **74.0 (−5.92, +3.14)** | **66.2 (−4.88, +3.70)** | 0.420 / 0.361 (−0.092 / −0.090) | 242 / 257 | ~1.2 ms/q |
- 서버 R@10(79.92/71.08) = 공개 ft113 → 하니스 정확(cycle 4와 동일).
- **R@10: cycle 4 대비 +3.1(EN)/+3.7(KO)** 개선; KO는 서버 −4.88로 **5pt 바 안**, EN −5.92로 근접.
- **overlap: 0.42/0.36 — cycle 4보다 하락(−0.09), 0.8 바 크게 미달.**
- 코드공간 진단(매칭쌍 텍스트–이미지 Hamming): C1 242/257 vs 서버 234/245 → C1 텍스트 코드가 제 이미지 코드에서 약간 더 멀지만 recall은 더 좋음(상대 랭킹이 gold를 잘 띄움).

## go/no-go + 근거
- **메커니즘(중요)**: cycle 4는 **서버와 같은 head**(so400m txt_h)에 근사 입력 → overlap 중간(0.5)·recall 낮음(입력 근사 오차). C1은 **서버와 다른 head**(txt_h')에 native 입력 → 입력 mismatch 제거로 **recall↑**, 그러나 서버와 다른 head라 **서버 top-K 일치↓**. **overlap↓는 버그가 아니라 설계상 당연.**
- **overlap≥0.8 = "서버와 동일"은 어떤 독립 오프라인 인코더로도 구조적으로 도달 불가**(텍스트 기하가 so400m과 다름). 따라서 오프라인은 "서버와 동일"이 아니라 **"서버에 근접한 별도 품질"**로 포지셔닝해야 함.
- **판정**: 엄격 바(Δ≤5 **&** overlap≥0.8) 미충족. 하지만 **"쓸 만한 근사 오프라인"** 기준(브리핑의 calibration)으로는 **C1이 cycle 4보다 명확히 우수**(recall +3~4pt, KO 서버 5pt 내) **+ 배포 더 깨끗**(stock e5 + 2.2MB txt_h') → **조건부 GO**.
- **C1b 미실행 이유**: 브리핑상 "C1a 개선했으나 바 미달"이면 C1b 후보지만 — (1) overlap 하락은 *head가 다르다*는 구조적 원인이라 e5 fine-tune으로도 overlap이 0.8로 갈 가능성 낮음(서버 head를 흉내내지 않음), (2) C1b는 stock-e5 깔끔함을 잃고 472MB 커스텀 e5 필요, (3) 예상 이득은 EN R@10 몇 pt(증분, 결정 불변). 스파이크 정신(결정 지향, 불필요한 grind 회피)상 미실행. 필요하면 후속으로 실행 가능(헤드+e5 동시 학습 ~37min).

## 사이클 6 오프라인 경로 권고
- **C1(native e5 → `txt_h'`) 채택**: recall이 cycle 4보다 좋고 서버에 근접, **배포 최단**(브라우저 = stock e5-small ONNX[int8 ~118MB, 공유/CDN 가능] + `txt_h'` 2.2MB + tokenizer.json[GO]). 서버 `/encode_query`는 **정확/레퍼런스 경로**로 유지(hybrid).
- **UX 포지셔닝**: 오프라인 모드 = "완전 오프라인·근사"(서버와 결과가 다를 수 있으나 품질 근접), 정확 모드 = 서버. 사용자에 라벨.
- 파이프라인(사이클 6에서 구현): `JS e5 토크나이저 → ONNX stock e5 → mean-pool → txt_h'(폴딩) → ±1 → pack → 브라우저 Hamming(index.bin)`.

## 결정 / 블로커 / 다음
- **결정**: C1을 사이클 6 오프라인 경로로 채택(hybrid: 서버 정확 + e5-offline 근사). `img_h`/`index.bin`/`common.py`/`query_server`/v1 무변경 유지.
- **블로커**: 없음(근사 오프라인 수용 시). "서버 동일 오프라인"은 구조적 불가 — 그 목표면 재정의 필요.
- **다음**: 사이클 6 = C1의 브라우저 통합(ONNX 익스포트+폴딩, app.js 토글, PWA). 선택 레버: C1b(e5 fine-tune)로 R@10 추가 +pt 시도(overlap 개선은 불확실), 또는 더 큰 학생.

## 산출물 / 재현
- `web/headadapt_train.py` — frozen ft113 img_h anchor + native e5 → `txt_h'` 학습 → `/tmp/txt_h_e5.pt`(2.2MB, 커밋 안 함). e5 emb 캐시 `/tmp/e5_train_emb.pt`.
- `web/headadapt_eval.py` — 오프라인(native e5→txt_h') vs 서버 파리티(R@10/overlap/매칭쌍 Hamming) + cycle 4 델타.
- 모델/임베딩 미커밋(DGX `/tmp`).

## 커밋·푸시
- 브랜치 **`web-v2-headadapt`**(`web-v2-distill`에서 분기). 스크립트 + 보고서만 커밋(모델 미커밋).
- **커밋 `13d6f3b`**(C1 scripts + 보고서) → `origin/web-v2-headadapt` **푸시 완료**. (이 SHA 기록 커밋이 뒤따름.)
