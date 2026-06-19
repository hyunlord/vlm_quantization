# PAPER EVALS — 논문 보강 측정 (eval-only)

재학습/인덱스 재생성 없이 기존 캐시·헤드(ft113)로 측정한 4종. 프로토콜 = `eval_korean` 5K test(갤러리 `img_h(te_img)`, gold caption_i→image_i). **앵커 재현: 1024-bit 서버 EN R@10 79.92 / KO 71.08 ✓.** 스크립트: `web/eval_paper.py`(A/B/C), `web/eval_scaling.py`+`eval_scaling.mjs`(D). 데이터: `paper/*.csv`.

## A. 상한 (→ 표 T2 맨 윗줄) — `paper/upper_bound.csv`  *(정정됨 — 헤드-연속이 진짜 이진화 상한)*
| 경로 | EN R@1/5/10 | KO R@1/5/10 |
|---|---|---|
| **head-continuous ceiling** (sign 직전, `F.normalize(BN(hash_head))`) | 42.74 / 70.52 / **80.52** | 32.34 / 60.68 / **72.04** |
| 1024-bit 서버(ft113, 이진화) | 41.64 / 68.98 / 79.92 | 30.52 / 59.00 / 71.08 |
| backbone-float, no head (raw so400m cosine) | 49.74 / 72.88 / 81.58 | 31.80 / 55.22 / 66.06 |
- **진짜 이진화 상한 = head-continuous**(SignSTE가 이진화하는 바로 그 연속 텐서). **EN·KO 둘 다 ≥ 1bit 서버** ✓ (80.52≥79.92, 72.04≥71.08) — 이진화 손실 EN ~0.6pt / KO ~1.0pt. **이게 T2 상한 행.**
- **backbone-float(no head)는 상한이 아니라 베이스라인**: 헤드 없는 raw so400m. EN 81.58(>1bit, 백본이 더 높음 — 헤드가 EN 약간 희생) / **KO 66.06(<1bit 71.08, ft113 한국어 fine-tune이 raw 백본을 추월** = 헤드 가치). 직전 사이클의 "KO float<1bit 역전"은 이 베이스라인을 상한으로 오인했던 것 — 정정 완료(head-continuous는 둘 다 1bit 이상).

## B. 비트길이 스윕 (→ 그림 F-bits) — `paper/bits_sweep.csv`
| bits | B/img | 50K idx | 서버 EN R@1/5/10 | 서버 KO R@1/5/10 | 오프라인(C1) EN R@10 | 오프라인 KO R@10 |
|---|---|---|---|---|---|---|
| 64 | 8 | 0.4 MB | 24.9/53.2/66.5 | 18.8/42.7/55.4 | 58.5 | 49.2 |
| 128 | 16 | 0.8 MB | 33.1/61.5/74.1 | 24.0/50.3/63.8 | 67.0 | 57.5 |
| 256 | 32 | 1.6 MB | 37.9/66.0/77.6 | 27.2/54.9/67.8 | 70.7 | 62.6 |
| 512 | 64 | 3.2 MB | 40.0/68.2/79.2 | 30.0/58.0/69.8 | 73.1 | 65.2 |
| **1024** | 128 | 6.4 MB | 41.6/69.0/**79.9** | 30.5/59.0/**71.1** | **74.0** | **66.2** |
- 단조 증가, 명확한 size↔품질 곡선. **256-bit(32B, 1.6MB/50K)**가 좋은 knee(서버 EN 77.6/KO 67.8, 1024 대비 −2~3pt에 인덱스 1/4). 오프라인(C1)은 전 구간 서버보다 ~5–7pt 아래로 평행 추적; 1024 오프라인 74.0/66.2 = 사이클5 C1과 **정확히 일치**(교차검증).

## C. 정밀도 민감도 (→ §4.1 표 T-prec) — `paper/precision.csv`
1024-bit 서버 기준, fp32 대비. (overlap = fp32 대비 Top-10 일치율.)
| 대상 | dtype | flip /1024 | Top-10 overlap | EN R@10 | KO R@10 | 조건 |
|---|---|---|---|---|---|---|
| head | fp32 | 0.00 | 1.000 | 79.92 | 71.08 | baseline |
| head | fp16 | 0.12 | 0.998 | 79.98 | 71.08 | head 캐스팅 |
| head | bf16 | 0.95 | 0.990 | 79.94 | 71.00 | head 캐스팅 |
| head | int8 | 18.97 | 0.929 | 79.70 | 70.52 | dynamic-quant(Linear) |
| emb | fp16 | 0.02 | 1.000 | 79.92 | 71.06 | 임베딩 storage-cast |
| emb | bf16 | 0.18 | 0.998 | 79.92 | 71.04 | 임베딩 storage-cast |
| emb | int8 | 5.45 | 0.966 | 79.84 | 70.84 | 임베딩 storage-cast(per-tensor) |
| **text_tower** | **bf16** | **2.57** | — | 79.98 | 70.98 | **so400m 타워 compute bf16 vs fp32 cache, head fp32, FULL 5K** |
- **bf16 명시 재측정(full 5K)**: 아웃라인의 "~2.7/1024 flip"은 **텍스트 타워를 bf16으로 *연산***할 때(27레이어 누적) = **2.57/1024**(full 5K; n=1000에선 2.72)로 재현 ✓, R@10 79.98/70.98 = fp32 앵커(79.92/71.08)와 **사실상 동일**(Top-K 무영향, 이제 사과-대-사과). 단순 임베딩 storage-cast bf16은 0.18, head-cast bf16은 0.95 — **조건(어디를 bf16으로 두는가)에 따라 다름**.
- 정밀도 영향: fp16 무시 가능(flip≤0.12, R@10 동일). bf16도 R@10 ~무영향. **int8: head(19 flip, KO −0.6pt) > emb(5.5 flip, KO −0.2pt)** — head가 0-근처 비트에 더 민감(사이클 2 관찰 재확인). 검색 품질은 전 dtype에서 product-viable.

## D. 인덱스 스케일링 레이턴시 (→ 그림 F-scale) — `paper/scaling.csv`
머신 **DGX GB10(aarch64) CPU**, warm, median of 30. 합성 랜덤 128B 코드, 단일 쿼리 Hamming Top-10.
| N | index | faiss IndexBinaryFlat(18 threads) | JS `search.js`(Node, 단일스레드) |
|---|---|---|---|
| 50K | 6.4 MB | 5.24 ms | **25.8 ms** |
| 500K | 64 MB | 10.45 ms | 256.8 ms |
| 5M | 640 MB | 116.3 ms | 1336.7 ms |
- **브라우저(JS 단일스레드)**: 50K ~26ms(데모 즉답), 500K ~257ms(수용), 5M ~1.3s(느림 → Web Worker/WASM-SIMD/faiss-wasm 필요). 데모 코퍼스 50K는 브라우저로 충분. **faiss(서버)**: 5M에서도 116ms — 서버 스케일 모드 여유. F-scale는 브라우저 단일스레드의 선형 스케일 한계와 "언제 전략 전환"을 보여줌.

## 앵커/교차검증 + 채우는 표·그림
- 앵커 1024 서버 **79.92 / 71.08 재현** ✓. 오프라인 1024 **74.0/66.2 = 사이클5 C1** ✓. text_tower bf16 **2.57/1024(full 5K) ≈ 사이클2** ✓. head-continuous 상한 **80.52/72.04 ≥ 1bit 둘 다** ✓.
- **A → T2 top row**(head-continuous = 이진화 상한; backbone-float = no-head 베이스라인). **B → F-bits**(+ T2/T3 size 행). **C → §4.1 T-prec**. **D → F-scale**.
- 정정(이번 follow-up): (E) 상한을 raw-backbone → **head-continuous**로 교체(둘 다 1bit 이상; "KO 역전" 해소). (F) text_tower bf16 R@10을 **full 5K**로 재측정(79.98/70.98, flip 2.57; n=1000 캐비엇 제거).

## 산출물 / 커밋
- 스크립트: `web/eval_paper.py`(A/B/C), `web/eval_paper_fix.py`(E/F 정정), `web/eval_scaling.py`+`eval_scaling.mjs`(D). 데이터: `paper/{upper_bound,bits_sweep,precision,scaling}.csv`. (합성 코드는 in-memory, 미저장.)
- 재현: `web/eval_paper.py --txt-head /tmp/txt_h_e5.pt` → `web/eval_paper_fix.py`(상한·bf16 정정) → `web/eval_scaling.py`. (모두 `HEAD_PATH=/tmp/ft_ko_113.pt .venv/bin/python …`.)
- 브랜치 **`paper-evals`**(`web-v3-hybrid`에서 분기). A/B/C/D = `1bf470e`+`ece4e1e`; **정정 E/F = `21122b3`** → `origin/paper-evals` 푸시 완료.
