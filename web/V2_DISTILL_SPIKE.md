# V2 DISTILL SPIKE — 소형 텍스트 인코더 (사이클 4)

**go/no-go = NO-GO**(엄격 바 기준) — 풀-오프라인 쿼리 인코더를 e5-small 크기로 출시하는 건 **충실도 미달**. **단, 막힌 축은 크기가 아니라 검색 충실도**다(int8 118MB는 통과; R@10/overlap이 미달). **권장 = hybrid**(서버 정확 경로 유지 + 선택적 "근사 오프라인 모드"). 학습 타당성 스파이크(브라우저 통합/ONNX/UI 없음).

## 한 줄 결론
e5-small(117.7M)을 so400m 텍스트 임베딩에 distill(3ep, val cos 0.89) → hold-out에서 **EN R@10 70.9(Δ−9.1)/KO 62.5(Δ−8.6), Top-10 overlap 0.45–0.51**. 바(Δ≤5 & overlap≥0.8) 미달. **크기는 OK(int8 118MB)** — 병목은 학생 용량/head 비트민감도. KO cos(0.92)>EN cos(0.87)라 **KO 데이터는 병목 아님**.

## 사전 조사
- **교사 캐시(재사용, 무료)**: 사이클 2~3에서 쓴 캐시가 곧 교사 타깃. **검증: 캐시 임베딩 == teacher(caption[0]) cos ≥0.9999**(EN/KO 둘 다) → 708M 타워 재임베딩 불필요.
  - EN: `/tmp/emb_aug.pt['train']['txt']` 113,287 (so400m EN 텍스트 emb), 문자열 = `dataset_coco.json` caption[0] by id.
  - KO: `/tmp/coco_ko_pairs.pt['txt_emb']` 113,287 (so400m KO 텍스트 emb), 문자열 = `coco_ko_train.jsonl` caption[0](build_korean 매칭 순서).
- **KO 텍스트 재고**: coco_ko_train 113K(이미지당 5캡션; [0] 사용). EN 가용 566K(train+restval) 중 113K 사용. → **KO 데이터는 충분**(사전 우려와 달랐던 점).
- **Hold-out(학습 제외)**: `emb_cache['test']`(5K EN + 문자열), `coco_ko_test`(5K KO). emb_aug train = Karpathy train+restval라 test 쿼리는 구조적으로 미포함.
- **학생/토크나이저**: `intfloat/multilingual-e5-small`(117.7M, XLM-R, hidden 384, fast tokenizer vocab 250002) → transformers.js 호환.

## 학습 설정
- 학생 = e5-small backbone + mean-pool + 학습가능 `Linear(384→1152)` → L2. 손실 = `1 − cos(student, teacher)`(norm_in=1 → 방향만 매칭).
- 데이터 226,574 pairs(EN 113,287 + KO 113,287), batch 256, lr 3e-5(backbone)/3e-4(proj), **3 epochs = 2,634 steps, 37min(GB10)**. val cos **0.846 → 0.877 → 0.892**(상승 둔화, 천장 ~0.90±).
- (후보 2/더 큰 학생은 후보 1이 바 미달이라 미진행.)

## 파리티 표 (hold-out 5K test, eval_korean 프로토콜 = 71/80과 사과-대-사과)
| 경로 | EN R@10 (Δ서버) | KO R@10 (Δ서버) | Top-10 overlap (EN/KO) | cos(학생,교사) (EN/KO) | 크기 |
|---|---|---|---|---|---|
| **서버 so400m**(측정) | 79.92 | 71.08 | 1.0 / 1.0 | — | 708MB int8(타워, 사이클3) |
| **학생 fp32** | **70.86 (−9.06)** | **62.50 (−8.58)** | **0.512 / 0.451** | 0.866 / 0.916 | 472MB |
| 학생 fp16 | 미측정* | 미측정* | — | — | 236MB |
| 학생 int8 | (≤fp32) | (≤fp32) | — | — | **118MB** |
- 서버 R@10(79.92/71.08) = 공개 ft113 80/71 재현 → 하니스 정확.
- 인코드 레이턴시 ~**1.2 ms/q**(PyTorch GB10; 브라우저 ORT는 다름).
- *fp16: GB10에서 사소한 dtype-cast 버그(proj Float vs backbone Half)로 미측정. **fp32가 최선치인데 이미 바 미달**이라 fp16/int8(≤fp32)은 판정 불변 → 추격 안 함. 크기만 보고(236/118MB).
- **판정**: Δ≈−9pt(바 ≤5) **그리고** overlap 0.45–0.51(바 ≥0.8) → **미달**. 크기 int8 118MB는 **통과**.

## 토크나이저-인-JS — GO
- transformers.js `AutoTokenizer('intfloat/multilingual-e5-small')` → Python 토큰 id **KO+EN 완전 일치**(BOS=0, EOS=2). 학생 토크나이저는 브라우저에서 그대로 됨. (`web/distill_tok_check.mjs`)

## go/no-go + 근거
- **NO-GO**(엄격 바): 풀-오프라인 충실 인코더를 e5-small로 출시 불가.
- **근거**: cos 0.87–0.92로도 R@10 −9pt·overlap ~0.5 — head(`txt_h`)가 0 근처 비트에서 민감(사이클 2 관측의 연장)해 임베딩 방향 오차가 Hamming Top-K를 크게 흔듦. **크기는 문제 아님**(118MB).
- **무엇이 needle을 움직이나**:
  1. **학생 용량 ↑** = 가장 직접적. e5-base(278M, int8 ~278MB)/e5-large(560M, int8 ~560MB)면 cos↑로 바 근접 가능하나 **150MB 예산 초과**(크기↔충실도 정면 충돌). → 출시하려면 **크기 예산 재결정** 필요.
  2. **KO 데이터 ↑ = 레버 아님**(KO cos 0.92 > EN 0.87; KO R@10 절대값이 낮은 건 서버 baseline(71)이 낮아서). 사전 가설 반증.
  3. **학습 더/레시피 개선**(더 많은 epoch, e5 "query:" prefix, hard-neg-aware/MSE 혼합) = cos 아직 미세 상승 중이라 +2~4pt 여지이나 overlap 0.5→0.8까지는 **불확실**(단독으론 부족할 가능성).

## 결정 / 블로커 / 다음
- **결정**: 엄격-오프라인을 e5-small로 강행하지 않음. `txt_h`/`img_h`/`index.bin`/`common.py`/`/encode_query` 전부 **무변경**(v1 그대로 동작).
- **블로커**: 충실도(학생 용량 vs 150MB 예산). 토크나이저·KO데이터·크기 자체는 클리어.
- **권장(다음)**:
  - **(A) Hybrid 출시(권장, 저비용)**: v1 서버 `/encode_query`를 정확 경로로 유지 + e5-small(118MB)을 **"근사 오프라인/PWA 모드"로 라벨링**해 옵션 제공(붕괴는 아님 — gold를 top-10에 ~2/3 회수). 논문 마일스톤("오프라인 동작")과 정확도를 동시에. 사이클 5 = 이 hybrid의 브라우저 통합(ONNX 학생+txt_h 폴딩, app.js 토글).
  - **(B) 엄격-오프라인이 필수면**: 크기 예산을 ~280MB로 올려 **e5-base distill 재스파이크**(같은 게이트로 Δ≤5/overlap≥0.8 도달 확인) 또는 개선 distill 레시피. 도달 보장 없음 → 먼저 측정.

## 산출물 / 재현
- `web/distill_train.py` — 교사 캐시 재사용 + e5-small distill → `/tmp/distill_e5.pt`(미커밋, 472MB).
- `web/distill_eval.py` — 파리티 평가(R@10/overlap/cos, fp32/fp16). `… web/distill_eval.py --student /tmp/distill_e5.pt`.
- `web/distill_tok_check.mjs` — 학생 토크나이저 JS 파리티.
- 모델/임베딩 산출물: 미커밋(DGX `/tmp/distill_e5.pt`).

## 커밋·푸시
- 브랜치 **`web-v2-distill`**(`web-v1`에서 분기). 스크립트 + 보고서만 커밋(모델/임베딩 미커밋).
- **커밋 `8b64465`**(distill scripts + 보고서) → `origin/web-v2-distill` **푸시 완료**. (이 SHA 기록 커밋이 뒤따름.)
