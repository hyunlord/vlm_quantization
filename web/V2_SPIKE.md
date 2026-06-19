# V2 SPIKE — 클라이언트 텍스트 인코더 타당성 (사이클 3)

**go/no-go = NO-GO** (현 형태). "풀 SigLIP2-so400m 텍스트 타워를 브라우저로" 옮기는 v2는 **크기 때문에 불가**. 토크나이저·정확도는 문제 아님 → **distill 또는 hybrid로 진행 권장.** 이건 타당성 스파이크다(브라우저 UI/통합 없음, 모델 변경 없음).

## 한 줄 결론
텍스트 타워가 **707.8M params**(so400m의 *큰* 쪽; vision은 428M) → **int8조차 ~708 MB**. 브라우저 첫 로딩으로 배포 불가. 토크나이저는 transformers.js로 **Python과 토큰 id 완전 일치**. 정확도(파리티)는 size no-go로 게이트아웃(미측정). → 풀 타워 중단, **작은 인코더 distill**이 진짜 v2 경로.

## 사전 조사
- **데모 텍스트 그래프**(`web/common.py:Encoder`, 확인): `tokenize(max_length=64, padding="max_length", truncation)` → `text_model` → `pooler_output` → `_prep`(norm_in=1 → L2) → `txt_h`(NestedHashLayer 1152→384→1024) → `sign`(±1) → `pack_bits`(128 B). 데모 토크나이저 = **GemmaTokenizer**(is_fast, vocab 256000, pad=0, EOS=1, BOS 없음).
- **transformers.js 지원**: HF repo `google/siglip2-so400m-patch14-384`가 **`tokenizer.json` 동봉**(+config/special_tokens) → transformers.js `AutoTokenizer`가 직접 로드. 텍스트 *모델*(가중치) 자체의 transformers.js/ONNX 경로는 size로 no-go라 미진행(아래). 토크나이저 JS 경로는 확정.
- **가정과 달랐던 점**: (1) `AutoProcessor.from_pretrained(...)` **콜드 로드가 이 transformers 버전에서 실패**(`config_model_type` None) — 데모는 모델 먼저 로드 또는 `GemmaTokenizer` 폴백으로 우회(정상 동작). (2) 텍스트 타워가 vision보다 **더 크다**(708M > 428M) — "so400m=400M"은 vision 기준, 전체는 1136M.

## 익스포트 (크기)
| 정밀도 | 텍스트 타워 크기(추정) | head txt_h |
|---|---|---|
| fp32 | **~2,831 MB** | 0.85M params (~3.4 MB) — 폴딩 자명 |
| fp16 | **~1,416 MB** | |
| int8(dynamic) | **~708 MB** | |
- 크기는 **정확한 param count에서 도출**(가중치가 지배, ONNX 그래프 오버헤드 수% → 실제 on-disk는 이 값 + α). 재현: `python web/spike_probe.py`.
- **실제 ONNX 익스포트는 의도적으로 미수행**(transparency): §0/§5 "size 과대면 멈춰 보고, 억지로 진행 말 것"에 따름. 정밀도와 무관하게 ≥708MB는 브라우저 배포 불가이므로, optimum/onnx/onnxruntime(미설치) 깔고 2.8GB fp32 export + 700M 모델 int8 양자화(느리고 메모리 큼)를 도는 것은 이미 결정난 make-or-break 질문에 비싼 비용. **정확한 on-disk 바이트가 필요하면 후속 export 가능하나 권장하지 않음**(결정 불변).
- head 폴딩 여부: txt_h는 0.85M로 자명하게 폴딩 가능(토큰→…→txt_h→±1까지 그래프화, JS엔 sign+pack만). 단, 폴딩은 풀 타워를 전제하므로 이번엔 미실행.

## 파리티 (정확도)
- **미측정 — size no-go로 게이트아웃**(측정 안 한 값은 주장하지 않음).
- 참고(사이클 2 근거): bf16 텍스트 타워만으로 cosine **0.9999**, packed **~2.7/1024비트** 차였음. → ONNX **fp16**도 cosine ≥0.999·Top-K 거의 불변이 **예상**. **int8**은 0-민감 head 때문에 Top-K 드리프트 **위험**(미검증). 어차피 크기로 배포 불가라 이 축은 distill 경로에서 사이클 2 방법론(`verify_fidelity.py` cosine≥0.999 + `smoke_retrieval.py` Top-K)으로 게이트할 것.

## 토크나이저-인-JS — **GO**
- `transformers.js@4.2.0` `AutoTokenizer.from_pretrained('google/siglip2-so400m-patch14-384')` → KO+EN 샘플 토큰 id가 Python `web/common.py` 토크나이저와 **완전 일치**:
  - `"바닷가 강아지"` → `[238131,246067,236361,84608,236655,236183,1]` (py==js)
  - `"a dog on the beach"` → `[235250,5929,611,573,8318,1]` (py==js)
  - `"two giraffes"` → `[9721,48523,61525,1]` (py==js)
- EOS=1 포함, BOS 없음, canonicalization 차이 없음. → **토크나이저는 브라우저에서 그대로 됨**(distill 경로도 동일 토크나이저 재사용 가능). 재현: `web/spike_tok_check.mjs`(+`spike_probe.py`로 ref 생성).

## 권고
- **NO-GO**: 풀 SigLIP2-so400m 텍스트 타워 브라우저화. **유일 블로커 = 크기(708M params)**.
- **첫 로딩 추정**(최선 int8): 텍스트모델 ~708MB + tokenizer.json ~16MB + `index.bin` 6MB + `meta.json` 6.7MB ≈ **~737MB**(fp16면 **~1.45GB**). 일반 회선·모바일·브라우저 캐시/메모리 한계 초과 → 불가.
- **권장 경로 (대안)**:
  1. **Distill (진짜 v2)**: 작은 멀티링구얼 텍스트 인코더(≈20–120M)를 SigLIP2 텍스트 임베딩(1152-d, `txt_h` 입력)에 distill → cosine ≥0.999 타깃. 브라우저엔 **distilled 인코더(int8 ~20–120MB) + txt_h(0.85M) + tokenizer.json**만. `index.bin`(이미지측) 불변 = 완전 오프라인 PWA 가능. 런타임: 토크나이저=transformers.js(검증됨), 인코더=onnxruntime-web(int8/fp16). *별도 학습 사이클 — 이번 스파이크 범위 밖.*
  2. **Hybrid (저비용)**: v1의 `/encode_query` 서버 인코딩 유지(현 상태) + distill 성공 시 오프라인 옵션 추가. 데모는 "브라우저 검색 + 초경량 텍스트-only 서버 인코드"로 차별점 유지.
- 이미지 인코딩은 빌드타임 서버 유지(vision 428M도 브라우저 불가지만 빌드타임이라 무관).

## 결정 / 블로커 / 다음
- **결정**: 풀 타워 브라우저화 중단. `web/common.py`·`/encode_query` 변경 없음(v1 그대로 동작).
- **블로커**: 텍스트 타워 크기 708M. 그 외 축(토크나이저=GO, 예상 정확도=양호)은 클리어.
- **다음(사이클 4 후보)**: (A) **Distill 스파이크** — 소형 멀티링구얼 인코더를 `txt_h` 입력 임베딩에 distill, 사이클 2 게이트(cosine≥0.999 + Top-K 불변)로 합격선 탐색 + 크기 측정. 합격 시에만 브라우저 통합(사이클 5). (B) 또는 v1 hybrid를 제품으로 확정하고 v2는 distill 성공까지 연기.

## 산출물 / 재현
- `web/spike_probe.py` — 크기(param 기반) + 토크나이저 ref 덤프. `python web/spike_probe.py`.
- `web/spike_tok_check.mjs` — transformers.js 토큰 id 파리티. `npm i @huggingface/transformers@4.2.0` 후 `node web/spike_tok_check.mjs`.
- ONNX/모델 산출물: **미생성/미커밋**(애초에 export 안 함). transformers.js는 DGX `/tmp/v2spike_js`에 설치(미커밋).

## 커밋·푸시
- 브랜치 **`web-v2-spike`**(`web-v1`에서 분기). 스크립트 + 이 보고서만 커밋.
- **커밋 SHA**: (아래 후속 기록 커밋 참조) · **origin/web-v2-spike 푸시 완료**.
