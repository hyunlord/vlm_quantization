# 검색 속도 업그레이드 — Float vs 바이너리 해시, 그리고 faiss 전환

_측정 환경: DGX(GB10, aarch64), numpy 2.x(`bitwise_count`), faiss 1.14.3(OpenMP 20스레드), 실제 OI 임베딩 코드(41,620장)를 타일링해 N 확대. 별도 표기 없으면 256-bit, top-k=12._

---

## TL;DR
- **문제**: ~10만 corpus에서 float 검색과 비트 검색의 체감 속도가 비슷했다. "비트는 XOR하고 정렬만 하면 되는데 왜 안 빠르지?"
- **원인 규명(분해 측정)**: 비트의 거리계산 자체는 빨랐지만 numpy 경로가 ① 싱글스레드 ② 임시배열 3번 할당 ③ `sum(axis=1)` 병목 ④ 양쪽 공유 고정비용(argpartition·dedup·JSON·HTTP)에 묻힘.
- **업그레이드**: 비트 검색을 **faiss `IndexBinaryFlat`**(하드웨어 POPCNT + 멀티스레드 + 단일패스, packed 32B 유지)로 교체. numpy(uint64) fallback 유지.
- **결과(GB10, 1M, 256-bit)**: 비트 검색 numpy 34.6ms → **faiss 9.6ms (3.6×)**. float 대비 **25.9×**. 정확도(top-k)는 numpy와 **완전 동일**(거리 multiset 일치).

---

## 1. 원인 규명 — 왜 10만에서 차이가 작았나

검색 코드와 동일 연산을 컴포넌트별로 분해 측정.

### 1-1. 비트 거리계산의 진짜 병목은 `sum`
```
xor:                  0.36 ms
xor + popcount:       0.47 ms     ← 여기까진 거의 공짜
xor + popcount + sum: 1.20 ms     ← sum(axis=1)이 0.7ms로 최대 병목
```
직관("XOR하고 더하면 끝")은 옳았다. 느린 범인은 XOR/popcount가 아니라 **(N,32) uint8을 axis=1로 리덕션**하는 메모리 패턴. packed를 uint64로 보면(32→4 원소) ~1.5–2.5× 개선.

### 1-2. 이론(144×)이 실측(~1.8×)으로 줄어든 이유
- float `emb @ q`는 **BLAS 행렬곱 = 멀티스레드 + SIMD**. 메모리를 144× 많이 읽어도 전 코어로 빠르게 훑음.
- numpy 비트 경로는 **싱글스레드 + 임시배열 3회 할당**. 적게 읽어도 1코어만 사용.
- → "비트가 빠르다"는 하드웨어 잠재력을 numpy가 못 살림.

### 1-3. 공유 고정비용이 비율을 1:1로 뭉갬
`argpartition`(top-k 선택), dedup 루프, 결과 dict 빌드, JSON 직렬화, HTTP 왕복 — **양쪽이 동일하게 지불**. 실제 거리계산이 1–2ms일 때 고정비용이 수~수십ms면 비율이 1:1로 보인다. (당시 5K×200 타일 데모는 dedup 후보가 2,412개라 고정비용이 더 컸고, `argpartition(range(k))` 오용으로 부분정렬까지 겹쳐 비트가 0.6s 났던 버그도 있었음 → 단일 kth로 기수정.)

---

## 2. 백엔드 비교 (GB10, 실측)

같은 corpus에 4가지 거리계산 백엔드. top-k는 numpy 기준과 **완전 일치 검증**.

### 256-bit (데모 기본값)
| corpus | numpy uint8(기존) | numpy uint64 | matmul ±1(BLAS) | **faiss** |
|--------:|---:|---:|---:|---:|
| 100K | 3.78 ms | 2.48 ms | 9.27 ms | **0.31 ms (12×)** |
| 500K | 19.77 ms | 12.94 ms | 44.53 ms | **4.88 ms (4.1×)** |
| 1M | 43.46 ms | 29.51 ms | 87.27 ms | **8.76 ms (5.0×)** |

### 64-bit
| corpus | numpy uint8 | numpy uint64 | matmul ±1 | faiss |
|--------:|---:|---:|---:|---:|
| 100K | 1.42 ms | **0.14 ms** | 2.51 ms | 0.74 ms |
| 1M | 21.79 ms | 2.11 ms | 22.42 ms | **1.99 ms (10.9×)** |

### 1024-bit
| corpus | numpy uint8 | numpy uint64 | matmul ±1 | faiss |
|--------:|---:|---:|---:|---:|
| 100K | 14.00 ms | 7.27 ms | 40.27 ms | **2.16 ms (6.5×)** |
| 1M | 142.05 ms | 56.47 ms | 370.81 ms | **23.66 ms (6.0×)** |

**관찰**
- **faiss가 거의 전 구간 승리**, 코드가 길수록 우위 확대(1024-bit/1M 6×).
- **matmul ±1 트릭은 GB10(ARM)에서 오히려 느림** — ARM BLAS가 x86만큼 최적화 안 됨 + unpack 비용 + packed 32B를 1024B로 풀어 **저장 이점 상실** → 폐기. (로컬 Mac에선 빨랐으나 타깃 하드웨어에서 반전 → 실측의 중요성)
- numpy uint64는 의존성 0의 공짜 1.5–2.5× → faiss 미설치 시 fallback.

---

## 3. 업그레이드 후 재비교 — Float vs 비트(구/신)

전체 top-k 검색 경로(거리계산 + argpartition/정렬), GB10, 256-bit.

| corpus | float cosine(기존) | 비트 numpy(구) | **비트 faiss(신)** | **faiss vs float** | faiss vs numpy |
|--------:|---:|---:|---:|---:|---:|
| 100K | 29.6 ms | 3.5 ms | **2.1 ms** | **14.4×** | 1.7× |
| 500K | 142.2 ms | 23.5 ms | **5.7 ms** | **25.1×** | 4.1× |
| 1M | 248.4 ms | 34.6 ms | **9.6 ms** | **25.9×** | 3.6× |

- GB10(ARM)에선 **float 행렬곱 BLAS가 느려**(1M 248ms), 비트(faiss) 대비 격차가 **최대 25.9×**로 매우 크다.
- **정직한 주석**: float-vs-bit 격차의 일부는 "GB10 float BLAS가 느린" 탓 → x86 호스트(좋은 BLAS)면 float이 더 빨라 격차는 줄어든다. 반면 **비트 numpy→faiss 3.6–4.1× 와 저장 144×는 하드웨어와 무관하게 성립**.

---

## 4. faiss `IndexBinaryFlat` 가 빠른 이유

| 항목 | numpy 경로 | faiss IndexBinaryFlat |
|---|---|---|
| popcount | `bitwise_count`(범용) | **CPU POPCNT 하드웨어 명령** |
| 스레드 | 싱글스레드 | **OpenMP 멀티스레드(20)** |
| 메모리 | xor→popcount→sum, **임시 3회 할당·3패스** | **단일 패스·무할당**(블록 fused) |
| 레이아웃 | 범용 ndarray | Hamming 전용 캐시 친화 |
| 저장 | packed 32B | **packed 32B 유지**(풀지 않음) → 저장 144× 유지 |
| 정확도 | 정확(전수) | 정확(전수, "Flat") — 동일 결과 |

```python
ix = faiss.IndexBinaryFlat(256)   # 비트 수
ix.add(packed_uint8)              # (N, 32) — 내부 복사본 보관
dists, ids = ix.search(query, k)  # Hamming 오름차순 상위 k (정확)
```
- **정확성 검증**: 전 비트 길이에서 numpy와 **K개 최소 Hamming 거리 multiset 완전 일치**. 짧은 코드의 인덱스 차이는 순수 동점(8-bit는 거리 0에 176개 동점)일 뿐.
- 더 큰 스케일(수천만~억)은 `IndexBinaryIVF`(근사, sublinear)/GPU로 확장 가능. 데모 규모(~110만)는 Flat이 정확+충분히 빠른 최적해.

---

## 5. 반영 (코드)

`demo/server.py`
- `search/bit`: **faiss `IndexBinaryFlat` 우선** + numpy(uint64) fallback, 응답에 `backend` 필드 노출
- 시작 시 비트 길이별 인덱스 빌드(packed 32B 유지)
- numpy fallback도 packed를 uint64로 보아 ~1.5–2.5× 개선
- 버그 수정: `_HAVE_FAISS` 정의 순서

**End-to-end 검증**(실서버, 41K corpus): `/search/bit` → `backend: faiss IndexBinaryFlat (HW POPCNT, multithread)`, 2.4ms · `/search/float` 7.4ms → 41K에서도 이미 3×, 대용량에서 25×로 확대.

---

## 6. 재현

```bash
# 백엔드 4종 비교(+정확성 검증)
INDEX=/tmp/oi_index.npz BITS=64,256,1024 NS=100000,500000,1000000 \
  python scripts/bench_hamming.py

# float vs 비트(구/신) 재비교
INDEX=/tmp/oi_index.npz BIT=256 NS=100000,500000,1000000 \
  python scripts/bench_float_vs_bit.py
```

## 7. 더 깊은 조사 — 재랭킹·ANN: "속도"가 아니라 "저장"이 진짜 축

faiss 전환 후 "더 빠르게 할 수 있나"를 파고들어 3가지 2단계 파이프라인을 추가 측정
(167K 실제 코드, i2i self-retrieval, recall@10 vs 정확 float, GB10).

| 파이프라인 | latency | recall@10 | 저장 |
|---|---:|---:|---|
| full float (정확, gold) | 40 ms | 100% | 4608 B ✗ |
| **float→float (faiss IVFFlat 근사)** | **2.4 ms** | **99%** | 큼 ✗ |
| **bit→bit (256 shortlist→1024 재랭킹)** | **2.3 ms** | 30% | 160 B ✓ |
| bit→float (256 shortlist→float 재랭킹) | 48 ms | 80% | 큼 ✗ |

**결정적 재포지셔닝**:
- **float→float(IVF)가 2.4ms·99%** — float도 근사 ANN으로 비트만큼 빠르다(17×↑). → **이 규모에서 "비트가 빠르다"는 더 이상 차별점이 아니다.**
- **진짜 차별점은 저장**: float 계열은 float 벡터를 들고 있어야 함(저장 이점 없음). **비트만 144× 작다.**
- **bit→float 재랭킹은 GB10에서 48ms** — float `emb`(770MB) 흩어진 gather가 이 박스에서 느림(swap 영향). float 저장도 필요 → 이득 없음. (빠른 x86이면 ~3ms로 빠르나 저장 이점은 여전히 없음)
- **bit→bit은 저장 작게(160B) 유지하나 recall 30%** — coarse shortlist + Matryoshka prefix 공유로 품질 회복 실패.

→ 정직한 포지셔닝: **"비트 = 144× 작은 저장(품질 약간 손해)" vs "float IVF = 99% 품질·동급 속도(저장 큼)".** 저장 제약(온디바이스)이 핵심이면 비트, 아니면 float IVF.

## 8. 품질 한계 — 양자화는 "다의어"에서 의미가 무너진다

텍스트 쿼리 실측(데모 API)에서 드러난 중요한 뉘앙스 — **낮은 recall이 곧 "틀림"은 아니다**:

| 쿼리 | bit-256 top-12가 float과 겹침 | bit-256 shortlist에 float 정답 포함 |
|---|---:|---|
| "a dog running in grass" (명확) | 0% | **top-500에 12/12 (100%)** |
| "a labrador retriever dog running" (명시적) | 17% | top-500에 10/12 (83%) |
| "running lab" (다의어) | 0% | top-2000에도 4/12 (33%)뿐 |

- **명확한 쿼리**: 정답이 bit shortlist에 다 들어있음 → 단지 top-k **순서 churn**(같은 개들, 순위만 다름). bit이 "이해 못 한" 게 아님. 재랭킹하면 복원됨.
- **다의어("lab"=Labrador vs 도서관)**: bit이 "도서관" 영역으로 튐 → 정답이 shortlist에 거의 없음. **1024-bit로도(prefix 공유), 재랭킹으로도(후보에 없음) 못 고침.** 1비트/차원 양자화가 다의어를 가르는 미세 방향을 잃음.

**결론**: 이해력이 전반적으로 붕괴한 게 아니라 **세밀/다의 의미에서 선택적으로 저하**된다. 이는 검색 트릭이 아니라 **해시 학습**으로 풀 문제(다의어 보존 손실 설계).

## 9. 품질 조사 — 검색단 트릭 전멸, 진짜 레버는 데이터+상대 distillation

"bit 검색이 float보다 의미를 덜 이해한다"(예: "running lab"→float은 래브라도, bit는 도서관)를 끝까지 파고든 결과:

**(a) 이진화는 무죄 (cross-modal·i2i 둘 다 입증).** float hash logits vs 1-bit 코드 top-k가 **100% 동일**(B∩C=100%). 즉 1비트 양자화 손실 ≈ 0. 품질 격차는 **emb→hash head 투영 + 좁은 COCO 학습**에서 발생.

**(b) 검색단 트릭 전부 실패.**
- bit→float rerank: float 저장 필요 → 144× 이점 상실 (의미 없음)
- bit→bit rerank: 256→1024 재랭킹 = full-1024와 **완전 동일(30.3%=30.3%)** — Matryoshka prefix 상관으로 이득 0
- asymmetric(쿼리 float 유지): 이진화가 lossy가 아니라 **이득 0** (float-hash=symmetric=asymmetric=29.0%)

**(c) naive distillation 실패.** hash 유사도행렬 ≈ emb 유사도행렬 절대 L2(w=3.0) → COCO 256-bit R@10 **0.782→0.705 하락** + 잘 되던 쿼리 파괴(library 58→17%). InfoNCE와 충돌.

**(d) "running lab"의 진짜 정체 — churn vs 진짜 오류.** bit shortlist 포함률 측정: "a dog running in grass"는 top-500에 정답 12/12(100%, 단지 순서 churn), "running lab"(다의어)은 top-2000에도 33%뿐(진짜 오류). 명확 쿼리는 bit이 멀쩡하고, **다의어에서만 의미 붕괴** — 코드 길이↑·재랭킹으로 못 고침(학습 문제).

**(e) 리서치 결론(개선 방향).** 검색단은 다 막혔고 유일한 레버 = **학습**:
1. 데이터 확대 (CC12M/OI Narratives 등) — 좁은 COCO가 병목
2. **상대(relative)·저가중치·decoupled distillation** (RKD 거리 μ정규화+Huber / CroVCA coding-rate diversity) — 우리가 한 절대 L2의 올바른 버전
3. backbone unfreeze/교체는 비추 (broad 지식 파괴 / 병목 아님)
→ 진행: OI Localized Narratives(167K)로 image-text 쌍 → COCO+OI/+RKD/+CroVCA 재학습 비교 중.

## 핵심 교훈
1. "느리다"는 분해 측정으로 원인을 짚어야 한다 — 비트의 병목은 XOR이 아니라 `sum`과 싱글스레드·고정비용이었다.
2. 같은 알고리즘도 **타깃 하드웨어에서 측정**해야 한다 — matmul 트릭은 Mac에선 이득, GB10(ARM)에선 손해였다.
3. 올바른 도구(faiss IndexBinaryFlat)는 packed 포맷을 유지한 채 하드웨어 명령·병렬·무할당으로 잠재력을 끝까지 짜낸다 → 저장 144×와 속도 둘 다 확보.
4. **속도는 양쪽 다 풀린다(float IVF 2.4ms ≈ bit 2.3ms). 진짜 축은 저장(144×).** 낮은 recall의 상당수는 churn이며, 진짜 한계는 다의어 — 학습으로 풀 문제다.
