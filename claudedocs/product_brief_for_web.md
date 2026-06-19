# VLM 1-bit Hashing — 제품화 핸드오프 문서

> **목적**: 지금까지의 R&D(크로스모달 바이너리 해시)를 웹/모바일 **제품**으로 전환하기 위한 전체 핸드오프. 이 문서 하나만 읽으면 기술·실험결과·강점·제품후보·경쟁·다음단계를 모두 파악할 수 있도록 작성. **Claude Web 등에서 제품 플랜 수립용**으로 통째로 붙여넣어 사용.
> 생성일 기준: 2026-06-19. 상세 실험 로그는 `claudedocs/hash_quality_ledger.md`(append-only), 평가 비교는 `claudedocs/hash_eval_comparison.html`.

---

## 1. 한 줄 요약

이미지와 텍스트를 **1024비트(128바이트) 바이너리 해시 코드**로 인코딩해, **멀티링구얼(한국어 특히 강함) 크로스모달 검색**을 한다. 코드가 극도로 작아(이미지당 128B) **온디바이스/오프라인** 인덱스가 가능하고, faiss Hamming(하드웨어 POPCNT)으로 초고속 검색한다.

핵심 한 문장: **"한국어 자연어로, 폰 안에서도 돌아갈 만큼 가벼운, 빠른 이미지 검색 엔진."**

---

## 2. 기술 스택 (현재 구현)

| 구성 | 내용 |
|---|---|
| 백본(인코더) | **SigLIP2 So400m** (`google/siglip2-so400m-patch14-384`, 1152-dim, **frozen**, 멀티링구얼 텍스트 인코더) |
| 해시 헤드 | **NestedHashLayer** — Matryoshka prefix-nested(8~1024bit 한 모델에서 다 추출), Linear→LayerNorm→GELU→Dropout→Linear→BatchNorm→L2norm→SignSTE(±1) |
| 손실 | InfoNCE(contrastive) + Quantization + Balance + Consistency + Ortho + (옵션) RKD/CroVCA |
| 검색 | **faiss IndexBinaryFlat** (Hamming distance, POPCNT) — 선형 스캔이지만 1M에서 ms 단위 |
| 학습 데이터 | COCO Captions(113K) + OpenImages(167K) + CC12M(806K~1.21M) — 캐시된 임베딩 위에서 헤드만 학습(백본 동결) |
| 인프라 | DGX Spark **GB10** 박스 (`ssh dgx-spark`), 통합 120GB 메모리 |

**중요 설계 포인트**
- 백본은 동결, **해시 헤드(작은 MLP)만 학습** → 학습이 가볍고 빠름(헤드 1개 ~16-25분).
- Matryoshka 구조라 한 헤드에서 8/16/.../1024bit를 동시에 제공 → 정확도-용량 trade-off를 런타임에 선택 가능.
- 텍스트/이미지 둘 다 같은 코드 공간 → 크로스모달(텍스트로 이미지, 이미지로 이미지).

---

## 3. 핵심 실험 결과 (제품 의사결정에 중요한 것만)

> 전체 누적 로그: `hash_quality_ledger.md`. 아래는 제품 관점 요약.

### 3-1. 평가 방법론
- 초기에 쓰던 `agreement@100`(float과 hash의 top-100 일치율) 지표가 **artifact(오해 소지)**임을 밝히고 폐기.
- **gold 지표**로 재정립: **pair-R@K**(정답 쌍을 정확히 찾나, R@1/5/10·MRR·MedR) + **category-mAP**(같은 개념/카테고리를 찾나, COCO 80-cat 기반 mAP@100/1000·NDCG). + hash-native(Hamming 반경 룩업) + 진단(bit balance/independence/entropy).

### 3-2. 데이터에 대한 핵심 발견
- **"양"은 레버가 아니다.** CC12M을 403K→806K→1.21M로 늘려도 in-domain 성능은 plateau, OOD도 +0.2~0.7pt 소폭. 볼륨 확대는 디스크·시간만 더 씀.
- **레버는 "다양성 + 혼합비"**다. CC12M을 넣는 것 자체(다양성)는 baseline 대비 category-mAP +4pt, OOD +8~18pt로 컸지만, 이미 403~806K에서 포화. 그리고 학습 시 **COCO:추가데이터 = 1:1~1:2 혼합비**가 sweet spot(pair-R@1 정점 42.5).

### 3-3. 양자화 손실 (hash vs float ceiling)
- 영어: hash(1024bit) R@10이 float(원본 1152-dim cosine)의 **95~98%**. 즉 1-bit로 압축해도 정확도 거의 유지.
- 비트 trade-off: **≥128bit는 Hamming 반경-2 해시테이블 룩업(O(1))이 붕괴**(코드 공간이 희박해 거리 2 안에 후보 없음) → **선형 스캔(faiss POPCNT) 필수**. 다행히 선형 스캔도 1M에서 ms. 정확도는 1024bit가 최고.

### 3-4. 한국어 (가장 중요한 제품 차별점)
- SigLIP2가 멀티링구얼이라 **학습에 한국어가 0개여도 KO R@10 65%** 작동(텍스트 헤드를 영/한이 공유).
- **소수 언어 강화는 fine-tune ≫ from-scratch 혼합** (이번 핵심 교훈):
  - best 헤드를 **한국어 쌍으로만 낮은 lr·5epoch fine-tune** → `ft113`: KO R@10 **65→71.1** (원본 float ceiling 66.06조차 **추월** = 107.6%), 영어는 거의 무손실(−0.18pt).
  - 같은 데이터를 from-scratch로 풀에 10% 섞으면 → KO 65.9로 **거의 안 오름**(희석).
- 한국어 fine-tune이 영어 OOD(Flickr/DOCCI)도 거의 안 해침.

### 3-5. 일반화 (학습 안 한 도메인)
- Flickr30K R@10 ~93, DOCCI R@10 ~83 (한 번도 학습 안 한 벤치) — 도메인 밖에서도 잘 작동.

---

## 4. 현재 보유 자산

| 자산 | 설명 | 위치 |
|---|---|---|
| **배포 후보 헤드 `ft113`** | best 1:1 + 한국어 fine-tune. KO R@10 71.1, EN 80. | `/tmp/ft_ko_113.pt` (DGX) |
| **라이브 데모** | 1,086,903장 코퍼스(COCO+OI+CC12M), 한국어/영어 자연어 검색, 10개 헤드 옵션, faiss. | DGX 포트 8200, `demo/live_server.py`+`live.html` |
| **평가 HTML** | 14개 헤드 × 12지표 비교(마스터표·곡선·OOD·한국어 등). | `claudedocs/hash_eval_comparison.html` |
| **학습/평가 스크립트** | 헤드 학습, 한국어 fine-tune, 평가 일습. | `scripts/train_1024.py`, `finetune_korean.py`, `eval_korean.py`, `eval_float_ceiling.py`, `flickr/docci_bit_bench.py` |
| **임베딩 캐시** | COCO/OI/CC12M SigLIP2 임베딩(헤드 학습용). | DGX `/tmp/*.pt` |

---

## 5. 제품으로서의 고유 강점 (차별점 3가지)

1. **초소형 코드 — 이미지당 128 bytes(1024bit).** 100만 장 인덱스 = **128MB**. 보통 시각검색은 이미지당 수 KB(float 벡터)라 서버가 필수인데, 우리는 **폰·브라우저에 통째로 올려 오프라인 검색** 가능. → 가장 드문 강점.
2. **한국어 자연어 검색이 float 수준.** ft113로 한국어 R@10 71(영어와 격차 ~8pt까지 좁힘, 원본 임베딩 ceiling 추월). 영어 위주인 시각검색 시장에서 흔치 않음.
3. **크로스모달 + 초고속.** 텍스트↔이미지 양방향, faiss Hamming(POPCNT)으로 1M에서 ms.

---

## 6. 제품 의사결정용 성능 수치

| 지표 | 값 |
|---|---|
| 한국어 텍스트→이미지 R@10 | 71.1 (ft113, 1024bit) |
| 영어 텍스트→이미지 R@10 | ~80 |
| float ceiling 대비 | 영어 95~98%, 한국어는 추월 |
| 저장 용량 | 1024bit = 128B/이미지 → 100만장 128MB, 1000만장 1.28GB |
| 검색 속도 | faiss 이진 1M에서 ms 단위 |
| OOD(미학습 도메인) R@10 | Flickr ~93, DOCCI ~83 |
| 헤드 재학습 비용 | 백본 동결, 헤드만 ~16-25분(GB10), 한국어 fine-tune ~1-4분 |

---

## 7. 제품 후보 (강점 매핑 + 현실성)

### A. 한국어 자연어 이미지 검색 — 웹 (현재 데모 제품화)
- **무엇**: 공개/스톡/사내 이미지 DB를 한국어 자연어로 검색하는 웹 서비스 또는 검색 API.
- **살리는 강점**: ②한국어 + ③속도. 데모가 이미 1.09M에서 동작 → **가장 빠른 MVP**.
- **도전**: 코퍼스 소스 확보, 구글/스톡 대비 차별점(한국어·속도·가격), 수익모델.

### B. 온디바이스 사진 검색 — 모바일 앱
- **무엇**: 폰 갤러리 사진을 한국어/영어 자연어로 오프라인 검색("작년 바다 여행", "강아지").
- **살리는 강점**: ①128byte(수만 장도 폰에) + ②한국어 + 프라이버시(온디바이스).
- **도전**: 모바일에서 SigLIP2 추론(백본이 큼) → ONNX/CoreML 양자화 또는 경량 distill 필요. 검색(Hamming)은 폰에서 쉬움, **인코딩이 관건**.
- **비고**: 이 기술의 **가장 독특한 포지션**(초소형 코드가 진짜 빛남). Apple/Google Photos가 이미 온디바이스 검색 제공 → 차별점(한국어·오픈/커스텀 코퍼스) 필요.

### C. 이커머스/콘텐츠 시각 검색 — B2B 위젯/API
- **무엇**: 상품·콘텐츠 이미지를 텍스트·이미지로 검색하는 B2B 임베드/API.
- **살리는 강점**: ③크로스모달 + ②한국어 + ①저비용 인덱스(대량 카탈로그도 작은 인덱스).
- **도전**: 도메인 적응(상품 이미지), 영업/통합, 기존 벡터DB 솔루션과 경쟁.

### D. 유사·중복 이미지 탐지 — 도구/라이브러리
- **무엇**: 해시 기반 near-duplicate 탐지/클러스터링.
- **살리는 강점**: ①1bit 코드의 dedup 속도.
- **도전**: 시각검색보다 틈새, 기존 perceptual hash(pHash) 대비 차별(시맨틱 유사).

---

## 8. 경쟁 / 유사 서비스 분석

> 웹 검색(firecrawl)으로 확인한 사실 위주. 불확실한 부분은 명시. **핵심 결론: 1bit·한국어·온디바이스 각 축은 이미 누군가 하고 있고, 우리 자리는 "세 축의 교집합"뿐이며, 카카오가 가장 큰 위협.**

### 8.1 자연어 이미지 검색 (스톡 / 소비자 사진앱)
| 서비스 | 설명 | 우리 대비 |
|---|---|---|
| Shutterstock AI search (NVIDIA 협업) | 구도까지 지정하는 시맨틱 검색 | 서버 SaaS, 자사 카탈로그 한정. 우리는 임의 코퍼스+온디바이스 |
| Getty/iStock AI search | 서술형 문장 검색 | 폐쇄형 카탈로그, 클라우드 전용 |
| Adobe Stock/Firefly | 생성형 결합 | 검색보다 생성 무게 |
| Google Photos "Ask Photos" (Gemini) | 개인 앨범 자연어 질의 | 클라우드/Gemini 의존, 대형모델. 우리는 경량·오프라인 |
- **판단**: 자연어 이미지 검색 자체는 **이미 보편화된 레드오션**(빅테크+스톡). 단 대부분 **클라우드 SaaS·자사 카탈로그 종속** → "임의 코퍼스에 꽂는 초경량 엔진"은 빈틈 여지.

### 8.2 온디바이스 / 오프라인 시각 검색
| 서비스 | 설명 | 우리 대비 |
|---|---|---|
| **Apple Photos (온디바이스 시맨틱)** | 사진 온디바이스 색인, 객체·장면 검색 | 가장 유사. iOS 한정·OS통합. 우리는 크로스플랫폼·임의앱 임베드 |
| Apple Enhanced Visual Search (iOS18+) | 온디바이스 임베딩 + **동형암호로 서버 글로벌 인덱스 조회** | **완전 오프라인 아님** → 우리의 "완전 오프라인"이 차별 |
| **Queryable (오픈소스 iOS)** | CLIP/MobileCLIP로 앨범 **오프라인** 검색 | 컨셉 거의 동일. float·영어. 우리는 1bit+한국어 |
| **Apple MobileCLIP** | 모바일 최적화 멀티모달 모델(distill), CoreML | **핵심 위협**: 온디바이스 백본 문제를 Apple이 풀어 무료 공개 → 우리 SigLIP2 모바일 부담의 대안이 시장에 존재 |
- **판단**: 온디바이스는 **빈틈이라기보다 빠르게 채워지는 영역**. 우리 차별은 "온디바이스" 자체가 아니라 **(한국어)+(1bit 저장/속도)+(플랫폼 비종속)의 조합**.

### 8.3 벡터 / 시맨틱 검색 인프라
| 서비스 | 설명 | 우리 대비 |
|---|---|---|
| Pinecone/Qdrant/Weaviate/Milvus | 벡터 DB 표준 인프라 | 대부분 float 1차, **바이너리 양자화(BQ)는 이미 옵션 제공** |
| **Qdrant Binary Quantization** | float→1bit, **최대 40x 속도·32x 메모리↓** (단 고차원+rescoring 필요) | **우리 핵심 주장과 정면 중첩**. 차별: 그들은 사후 양자화+float rescoring, 우리는 **처음부터 1bit end-to-end 학습** |
| Weaviate BQ / OpenSearch binary / Vespa Matryoshka-binary | 동일 비용절감 옵션 | 동일 메시지. 우리만의 독점 아님 |
| faiss / ScaNN | Hamming 네이티브 지원 | **우리가 이미 쓰는 도구** = 차별 아님 |
- **판단 (중요)**: **"1bit가 저장·속도 유리"는 이미 인프라 업계 상식**이며 모든 메이저 벡터DB가 옵션 제공. 우리 차별은 "바이너리 검색"이 아니라 **"멀티모달(I/T)+한국어+end-to-end 학습 해시"**여야 함. 저장/속도만 내세우면 Qdrant/Cohere에 묻힘.

### 8.4 CLIP 기반 오픈소스 / 서비스
| 서비스 | 설명 | 우리 대비 |
|---|---|---|
| clip-retrieval (LAION) | LAION-5B KNN 인덱스 | float·영어·연구용 |
| **Marqo** | AI-네이티브 멀티모달 벡터 검색 엔진 | **직접 경쟁군**. float·서버형. 우리는 1bit·온디바이스 |
| Elasticsearch + CLIP | KNN+CLIP 레시피 | 범용 패턴 |
| **Jina Embeddings v4** | **3.8B 멀티링구얼 멀티모달**(한국어 포함) | 직접 경쟁. 거대모델→모바일 부적합. **한국어는 카카오 평가에서 약했음** |
- **판단**: 멀티모달 검색은 오픈소스·스타트업 두터움(Marqo/Jina/clip-retrieval). 대부분 **float+영어+서버형**. 각 축 단독은 이미 누군가 함.

### 8.5 바이너리 / 해시 임베딩의 산업 적용 (★핵심)
| 사례 | 내용 | 시사점 |
|---|---|---|
| **eBay Visual Search** | 프로덕션 시각검색에 **바이너리 해시**, "**98.4% 메모리 절감**" 명시 | **우리 핵심 주장(초소형 코드)이 이미 대규모 프로덕션에서 검증·운영 중** |
| **Pinterest Unified Embedding** | 수십억 이미지를 바이너리화해 검색 | 빅테크가 이미 채택 |
| **Cohere int8 & binary Embeddings** | **1bit, 32x 메모리↓·40x 속도↑·1M 20ms**, 정확도 90~98%(float rescoring으로 90→95%) | **우리 수치의 업계 레퍼런스**. 단 **Cohere는 텍스트 전용(크로스모달 아님)** |
| 딥해싱 학술(CSQ, DHD 등) | Hamming deep hashing 활발 | 우리 기법(SignSTE)은 학계 표준 계열, 신규성보다 적용이 관건 |
- **판단 (가장 중요)**: **우리의 "1bit 초소형" 강점은 빈틈이 아니라 검증된 산업 표준**(eBay/Pinterest 프로덕션, Cohere/Qdrant 제품). 수치(32x/40x/90~98%)는 Cohere와 거의 동일 = 신뢰 가능하나 독창적 우위 아님. **단 결정적 빈틈 하나: Cohere binary는 텍스트 전용, eBay/Pinterest는 I2I 중심 → "크로스모달(T↔I) 1bit 해시" 제품화 사례는 공개 검색상 미확인**(내부 미공개 가능성은 배제 못 함). 또한 모든 사례가 **float rescoring으로 정확도 보완** = 순수 1bit Hamming만이면 정확도 갭 존재 가능성.

### 8.6 한국어 멀티모달 검색 (국내) (★핵심 경쟁자)
| 서비스 | 내용 | 우리 대비 위협 |
|---|---|---|
| **카카오 Kanana-v-embedding** | **한국어 특화 멀티모달 임베딩**. T-T/T-I/I-I 전부, **Matryoshka 64~2048dim**, 사내 광고심사 적용, **온디바이스·초개인화 앨범검색을 로드맵 명시**("뽀삐 작년 해변사진"), KoEmbed 한국어 데이터셋 자체구축 | **가장 위협적인 직접 경쟁**. 우리 포지셔닝(한국어+멀티모달+온디바이스)과 **거의 완전 중첩**. 단 백본 **2B VLM→모바일엔 무거움**, **1bit는 아직 안 함**(Matryoshka 차원축소까지) |
| 네이버 스마트렌즈 + 멀티모달 AI | 이미지+자연어 동시 이해, "렌즈x AI 브리핑"(2025.7) | 서버형·네이버 생태계 한정 |
| 카카오 Kanana-o | 이미지·음성 멀티모달 LLM | 생성형, 직접 경쟁 아님 |
- **판단 (충격)**: **우리가 빈틈이라 본 "한국어 멀티모달 검색"을 카카오가 이미 정조준해 발표**(Kanana-v-embedding). 한국어 특화·멀티모달·**온디바이스 로드맵 명시**. 우리의 마지막 방어선은 그들이 **2B VLM + 1bit 미적용**이라는 점(우리는 SigLIP2 400M + 1bit로 더 경량).

### 8.7 ★ 냉정한 평가
**3대 강점이 진짜 빈틈인가:**
| 강점 | 현실 | 평가 |
|---|---|---|
| ① 1bit 128byte | eBay/Pinterest/Cohere/Qdrant 다 함 | **빈틈 아님**(검증된 표준). 단 "크로스모달 1bit"만 미확인 틈새 |
| ② 한국어 자연어 | 카카오 Kanana 정조준, 네이버, Jina(약함) | **빈틈 빠르게 사라지는 중**. 카카오가 최대 위협. 단 경량 온디바이스는 우리 우위 가능 |
| ③ 온디바이스 오프라인 | Apple Photos/MobileCLIP/Queryable, 카카오 로드맵 | **빈틈 아님이나 미성숙**. 특히 "한국어+완전오프라인+1bit" 조합은 아직 빈자리 |

**약점 (정직하게):**
1. **정확도 갭** — 모든 바이너리 사례가 float rescoring으로 보완(Cohere 90→95%). 순수 1bit는 full-float SOTA 대비 손실 가능 → **"float 대비 X%"를 정직한 벤치마크로 제시** 필요.
2. **모바일 백본 부담** — SigLIP2 So400m ≈ **400M 파라미터**. 검색(Hamming)은 가볍지만 **인코딩(백본 추론)이 무거움**. Apple은 이를 위해 MobileCLIP을 따로 만듦 → **"온디바이스" 주장하려면 백본 경량화/distill 계획 필수**.
3. **단일 축은 전부 선점** — 우리 가치는 **오직 "세 축의 교집합"에만** 존재. 하나만 떼면 거대 경쟁자에 묻힘.

**가장 현실적인 포지셔닝:**
> **"한국어에 강한, 완전 오프라인으로 도는, 크로스모달(T↔I) 1bit 해시 검색 엔진"** — 세 조건 **동시 충족** 솔루션은 현재 시장 미확인.

- **(A) 엣지/임베디드 SDK** — 벡터DB와 정면승부(패배) 대신, **클라우드 못 쓰는 환경**(온프레미스, 규제산업, 프라이버시 앨범, 오프라인 디바이스) 타깃. Apple/카카오가 안 가는 "플랫폼 비종속+완전 오프라인" 자리.
- **(B) 카카오 대비 초경량** — Kanana(2B)/Jina(3.8B) 대비 SigLIP2(400M)+1bit. **"메모리 1/32, 모델 1/5~1/10로 비슷한 품질"**을 벤치로 증명(백본 경량화 전제).
- **(C) 정직한 정확도 스토리** — "full-float SOTA는 아니나 저장·속도 제약 환경 실용 정확도(float의 95%)" 트레이드오프를 명시(경쟁사도 다 rescoring 보완 → 약점 아닌 정직한 포지셔닝).
- **(D) 한국어 1순위 + 카카오와 분리** — "카카오가 서버에서 하는 걸, 우리는 기기 안에서 1/32 메모리로".

**한 줄 결론**: 세 강점은 개별로는 모두 선점됐고 우리 수치는 신뢰할 만하나 독창적이지 않다. **유일한 빈자리는 "한국어+완전오프라인+크로스모달 1bit"의 교집합**이며, 그조차 카카오 Kanana 온디바이스 로드맵이 1~2년 내 잠식할 위험. **방어선 = 초경량 백본 + 완전 오프라인 + 플랫폼 비종속 + 정직한 정확도 벤치마크.**

**불확실성 표기:**
- "크로스모달 1bit 해시 상용 사례 없음"은 *공개 검색 기준*(빅테크 내부 미공개 가능성 배제 못 함).
- 우리 1024bit 정확도 vs full-float/Cohere binary는 **직접 비교 벤치마크 필요**(현재는 Cohere 90~98% 유비 적용).
- 확인됨: 저장/속도 수치, eBay 98.4%, SigLIP2=400M, 카카오/네이버/Apple/Cohere/Marqo/Jina 기능은 출처 URL 검증.

**출처(주요)**: Shutterstock(developer.nvidia.com), Google Ask Photos(blog.google), Apple Photos/MobileCLIP/동형암호(machinelearning.apple.com·support.apple.com/122033), Queryable(github.com/mazzzystar/Queryable), Qdrant BQ(qdrant.tech/articles/binary-quantization), Cohere binary(cohere.com/blog/int8-binary-embeddings), eBay Visual Search(ResearchGate), Marqo(marqo.ai), Jina v4(jina.ai·arxiv 2506.18902), 카카오 Kanana(tech.kakao.com/posts/801), 네이버(navercorp.com), SigLIP2(arxiv 2502.14786).

---

## 9. 열린 질문 / 플랜 수립 입력값

제품 플랜을 짜려면 아래가 정해져야 함:

1. **타겟 사용자/시장** — 개인(B2C, 내 사진) vs 기업(B2B, 카탈로그/스톡) vs 개발자(API)?
2. **플랫폼 우선순위** — 웹(빠른 MVP) vs 모바일(독특하나 추론 엔지니어링)?
3. **코퍼스 소스** — 자체 보유 데이터 / 공개 데이터(Unsplash 등) / 사용자 업로드 / 스톡 제휴?
4. **핵심 포지셔닝** — "한국어 검색"으로 갈지, "온디바이스·프라이버시"로 갈지, "초저비용·고속 인덱스"로 갈지 (셋 중 하나를 주력으로).
5. **차별점의 방어 가능성** — Apple/Google Photos, 스톡 AI 검색이 이미 하는 것과 무엇이 다른가? (8장 경쟁분석 참고)
6. **MVP 범위** — 검색만 vs 검색+업로드+계정, 코퍼스 규모, 한국어 only vs 다국어.
7. **비즈니스 모델** — 구독 / API 사용량 / B2B 라이선스 / 앱 판매.

---

## 10. 제품화 시 기술 과제

| 과제 | 메모 |
|---|---|
| **모바일 백본 추론** | SigLIP2 So400m은 큼. 온디바이스(B안)면 ONNX/CoreML 양자화 또는 경량 모델 distill 필요. 텍스트 인코더만 온디바이스 + 이미지 인덱싱은 서버 1회도 가능. |
| **코퍼스 인코딩 파이프라인** | 대량 이미지 → SigLIP2 임베딩 → 헤드 코드. GB10에서 ~14 img/s(현재 PyTorch fallback, 느림). 프로덕션은 GPU 최적화/배치 필요. |
| **증분 인덱싱** | 새 이미지 추가 시 faiss 인덱스 갱신. IndexBinaryFlat는 add 쉬움. |
| **스케일** | 1M ms 단위 OK. 1억+면 IndexBinaryHash 또는 multi-index 검토(단 ≥128bit 반경룩업 붕괴 주의 → 샤딩/IVF). |
| **콜드스타트/서빙** | 백본 GPU 상주 필요(쿼리 인코딩). 데모는 faiss thread warmup으로 첫 쿼리 outlier 제거. |
| **정확도 한계** | 1-bit는 full-float SOTA(CLIP/SigLIP 원본 cosine)보다 약간 낮음(영어 95~98%). 정밀도 최우선 용도면 float rerank 단계 고려 가능. |

---

## 11. 부록 — 파일/경로 레퍼런스

- 실험 누적 로그(상세): `claudedocs/hash_quality_ledger.md`
- 평가 비교 HTML: `claudedocs/hash_eval_comparison.html` (브라우저로 열기)
- 데모 서버: `demo/live_server.py` (FastAPI, 포트 8200), 프론트 `demo/live.html`
- 배포 헤드: `/tmp/ft_ko_113.pt` (DGX)
- 학습: `scripts/train_1024.py` (헤드 학습), `scripts/finetune_korean.py` (한국어 fine-tune)
- 평가: `scripts/eval_korean.py`, `scripts/eval_float_ceiling.py`, `scripts/flickr_bit_bench.py`, `scripts/docci_bit_bench.py`
- 인프라 접속: `ssh dgx-spark` (passwordless), GB10
