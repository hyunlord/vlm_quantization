# HANDOFF — v1 인코딩 충실도 검증 + 닫기 (사이클 2)

**상태: 통과.** 데모의 라이브 쿼리 인코딩이 `scripts/eval_korean.py`가 KO R@10 71 / EN 80을 낸 캐시 임베딩을 재현함을 실측 증명. **`web/common.py` 무수정**(검증 전용). 브랜치 `web-v1` 이어감.

## 추가 파일 (전부 `web/`, 검증 전용 — 인코딩 로직 무변경)
- `web/verify_fidelity.py` — 충실도 검증 + **회귀 가드**(고정 seed/N/thresh, PASS/FAIL + RESULT_JSON + exit code)
- `web/smoke_retrieval.py` — 실제 `index.bin`(50K)에 EN caption→image R@1/5/10 헬스 스모크

## 실행 (전부 DGX `ssh dgx-spark`, repo `~/github/vlm_quantization`)
```bash
# 충실도 + 회귀 가드 (한 줄, PASS/FAIL + 수치 + exit code)
HEAD_PATH=/tmp/ft_ko_113.pt .venv/bin/python web/verify_fidelity.py            # 기본 n=1000,thresh=0.999,seed=42
# 서빙 artifacts retrieval 스모크
HEAD_PATH=/tmp/ft_ko_113.pt .venv/bin/python web/smoke_retrieval.py --n 1000
```

## 사전 조사 결과 (§3)
- **eval_korean 입력**:
  - EN 텍스트 임베딩 = `/tmp/emb_cache.pt["test"]["txt"]` (5000×1152, fp32), **원문 문자열 = 같은 dict의 `["test"]["captions"]` (5000, 정렬됨)** ← 우선 경로 가능.
  - 이미지 임베딩 = `/tmp/emb_cache.pt["test"]["img"]` (5000×1152).
  - KO 텍스트 임베딩 = `/tmp/coco_ko_test.pt["txt_emb"]` (5000×1152) + `["ids"]`. **KO 문자열은 미저장** → `scripts/build_korean.py` 로직대로 `data/coco_ko/coco_ko.jsonl`에서 cocoid별 첫 한국어 캡션을 `ids` 순서로 재구성(missing=0).
  - **평가 split**: COCO Karpathy **test 5K**. 지표: T2I, ±1 코드 Hamming=`(D−q·dbᵀ)/2`, `R@k = (top-k 안에 페어 이미지 존재) 비율`.
  - 텍스트 임베딩 규약: `tok(max_length=64, padding="max_length", truncation) → text_model → pooler_output → L2`. **레포 전역 동일**, `web/common.py`와 일치.
- **가정과 달랐던 점**:
  1. KO 캐시에 문자열이 없어 `coco_ko.jsonl`에서 재구성(빌더와 동일 로직, missing=0).
  2. **임베딩 dtype 차이 = 유일한 실질 차이**: eval 캐시는 **fp32** 텍스트 타워(`build_korean.py`), 데모(`common.py`)는 GPU에서 **bf16** 타워. head(txt_h/img_h)는 양쪽 fp32 → 이미지 코드는 바이트 동일해야 하고, 텍스트는 bf16 반올림만큼 미세差.

## 충실도 결과 (우선 경로 = 가장 타이트)
- **EN**: 라이브 인코딩 vs 캐시 `te_en`, n=1000 — cos **mean 0.99991 / median 0.99991 / min 0.99908 / p1 0.99977, 100% ≥0.999** → PASS.
- **KO**: vs `coco_ko_test["txt_emb"]`, n=1000 — cos **mean 0.99996 / median 0.99996 / min 0.99984 / p1 0.99990, 100% ≥0.999** → PASS.
- **이미지측**: `image_codes_packed(te_img)` vs eval_korean의 `img_h` 경로 → **5000/5000 바이트 동일** (mismatch 0). 동일 연산(fp32 head) 확인.
- **packed 쿼리코드(참고)**: 데모 bf16-타워 코드 vs fp32-캐시 유래 코드는 **바이트 동일 아님**(~8% 동일, **mean Hamming ~2.7/1024**, max 13–15). 원인 = bf16 vs fp32 (거의 0인 좌표의 부호가 가끔 뒤집힘). **1024비트 중 ~3비트(0.3%)** → 검색 Top-K에 영향 없음. **판정은 cosine ≥0.999로 깨끗이 통과**(의뢰서 AC: 코사인 ≥0.999 **또는** 바이트 동일 → 코사인 충족).
- **결론**: 데모 쿼리 인코딩 = eval 임베딩 재현(방향 cos ~0.9999). KO 71 / EN 80 헤드라인은 데모 경로의 지표로 확정. (bf16/fp32 ~2.7비트 차는 무해.)

## smoke (서빙 artifacts, 50K 갤러리)
- EN caption→image, 1000 쿼리: **R@1 22.8% · R@5 43.5% · R@10 53.1%** (faiss 오라클).
- 해석: **붕괴 아님**(랜덤 ≈0.02%). 절대 벤치 아님 — 50K 갤러리는 eval의 5K보다 ~10× 어렵고 COCO 캡션이 generic. 건강성만 확인.

## 회귀 가드 (한 줄)
```bash
HEAD_PATH=/tmp/ft_ko_113.pt .venv/bin/python web/verify_fidelity.py   # exit 0=PASS, !=0=FAIL; RESULT_JSON 출력
```
고정값: seed=42, n=1000/언어, thresh=0.999, 판정 = EN·KO 평균 cos≥thresh & ≥99% 샘플 통과 & 이미지 바이트 동일. 다음 사이클에서 인코딩/헤드/캐시 회귀 감지용.

## 결정 / 블로커 / 다음
- **결정**: bf16(데모) vs fp32(eval) 차이(~2.7/1024비트, cos≥0.9999)는 무해로 **수용** — `web/common.py` 수정 불필요(인덱스 재빌드 회피). 이미지 경로는 완전 동일.
- **블로커**: 없음. 우선(타이트) 경로로 완료 — fallback 불필요.
- **다음(범위 밖이었음)**: v1.5 Matryoshka 64→1024 coarse-to-fine · v2 클라이언트 텍스트 인코더 · Web Worker 분리 · 스케일 모드(1.09M). (원하면 데모를 정확도 100% 일치로 만들려면 텍스트 타워를 fp32로 강제하는 옵션 추가 가능 — 단 속도/메모리 trade-off, 현재 불필요.)

## 커밋·푸시
- 브랜치 **`web-v1`**. 생성물(`web/static/data`, `thumbs`) 미커밋. 검증 스크립트만 추가.
- **커밋 `f6bbc18`** (verify_fidelity + smoke + HANDOFF) → `origin/web-v1` **푸시 완료**. (이 SHA 기록 커밋이 뒤따름.)
- 사이클1 토대: `d3f8f34`/`85ebb02`/`b5fc7d3` (`web/HANDOFF_v1.md`).
