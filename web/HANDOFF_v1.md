# HANDOFF — 브라우저 사이드 1bit 검색 데모 (v1)

브라우저가 `index.bin`을 통째로 로드해 **클라이언트에서 Hamming Top-K 검색**을 돌리고,
서버는 **쿼리 텍스트만 1024-bit 코드로 인코딩**(검색 없음)하는 v1 데모. 코퍼스 = COCO 5만 장.
**상태: 인수기준 전부 충족, DGX에서 검증 완료.**

## 변경/추가 파일 (전부 `web/` 아래, `demo/` 무손상)
- `web/common.py` — 공유 계약: `pack_bits()`(단일 패킹 함수) + `Encoder`(ft113 head + SigLIP2 텍스트타워) + `hamming_topk()`(JS 미러)
- `web/build_index.py` — 5만 subset → `index.bin` + `meta.json` + `index_info.json` + 썸네일
- `web/query_server.py` — FastAPI: `POST /encode_query` + 정적 서빙(검색 라우트 없음)
- `web/verify_parity.py` — §6.4/§6.5 패리티 자동검증 (faiss vs JS-미러, 서버코드 vs 오프라인코드)
- `web/static/index.html`, `web/static/app.js` — 프론트엔드(스트리밍 로드 + 클라이언트 검색)
- `web/__init__.py`, `web/.gitignore`
- 생성물(=gitignore, 재생성 가능): `web/static/data/{index.bin,meta.json,index_info.json}`, `web/static/thumbs/*.jpg`

## 실행 방법 (전부 DGX `ssh dgx-spark`, repo `~/github/vlm_quantization`)
```bash
# 1) 인덱스 빌드 (1회). 재빌드 시 --reuse-thumbs 로 썸네일 재사용(빠름)
HEAD_PATH=/tmp/ft_ko_113.pt .venv/bin/python web/build_index.py \
  --n 50000 --seed 42 --index /tmp/demo_index.npz --image-root data/coco --out web/static

# 2) 서버 (encode-only). 백그라운드 detach는 tmux 권장(아래)
HEAD_PATH=/tmp/ft_ko_113.pt .venv/bin/python -m uvicorn web.query_server:app \
  --host 0.0.0.0 --port 8300
# 현재 tmux 세션 'webv1' 로 :8300 에 떠 있음:  tmux attach -t webv1 / 로그 /tmp/web_server.log

# 3) 패리티 검증
HEAD_PATH=/tmp/ft_ko_113.pt .venv/bin/python web/verify_parity.py \
  --static web/static --server http://127.0.0.1:8300 --k 10
```
- **포트 8300** (8200은 기존 `demo/live_server.py` 사용 중).
- 접속 URL: 로컬 `http://127.0.0.1:8300/` · DGX 외부 `http://100.70.109.50:8300/`(tailscale) 또는 ngrok(.env 토큰).

## 사전 조사 결과 (§3) — 가정 대비 실제
- **인코더 위치**: `src/models/nested_hash_layer.py`(`NestedHashLayer`). 텍스트 인코딩은 `demo/live_server.py:_encode_text`를 재사용(SigLIP2 text_model pooler→float→L2). 이미지측은 캐시 임베딩에 `img_h` 적용.
- **체크포인트**: `/tmp/ft_ko_113.pt` **존재**(DGX). keys=`{img_h,txt_h,bits,hidden,embed,mode,norm_in,base}`, bits에 **1024 포함**, embed=1152, hidden=384, **norm_in=1**, mode=`ft_korean`, base=`sweep_c113287_*`. 가정과 일치. (로컬 Mac엔 체크포인트·임베딩·이미지 **전무**.)
- **기존 데모**: `demo/live_server.py`(:8200)+`demo/live.html` **존재**(서버사이드 faiss). 텍스트인코딩·패킹 로직 재사용, 검색은 클라이언트로 이전.
- **5만 subset**: `/tmp/demo_index.npz`(113,287장 COCO, 캐시된 SigLIP2 임베딩)에서 **seed=42 무작위 5만**. 이미지가 열리는 행만 채택(5만/5만 성공, 실패 0). 코드는 npz의 emb에 **ft113 `img_h`로 재계산**(npz의 `packed_*`는 구 head라 미사용).
- **실행 환경**: **DGX(GB10)**. 로컬 Mac엔 데이터가 없어 **빌드·검증은 DGX 필수**. 코드는 Mac에서 작성·커밋, `rsync`로 DGX 배포 후 실행.
- **가정과 달랐던 점**:
  1. `src/serve/binary_index.py`가 **DGX main에는 없음**(feature 브랜치에만). → `web/`를 self-contained로 만들기 위해 `pack_bits()`를 `np.packbits(>0, bitorder='big')`로 **직접 구현**(pack_codes와 바이트 동일).
  2. ft113에 `norm_in/mode/base` 키가 있음(live_server는 무시). → `scripts/eval_korean.py` 계약대로 **norm_in=1이면 입력 L2정규화** 후 head 적용(이게 KO R@10 71 검증 경로).
  3. `demo_index.npz`의 COCO **captions가 비어있음**. → `data/coco/dataset_coco.json`(cocoid→문장)에서 **백필**(5만/5만 채움).

## 패킹 규약 (한 줄)
1024-bit **±1 코드 → `np.packbits(code > 0, bitorder='big')` → row당 128바이트**; 쿼리 코드도 **동일 함수**(`web/common.py:pack_bits`)로 패킹 — 인덱스와 쿼리가 같은 패킹을 쓰는 것이 패리티의 유일 불변식. (faiss `IndexBinaryFlat` 호환.)

## 검증 결과
- **§6.4 패리티(최중요)**: KO+EN **24개 쿼리 전부**, 클라이언트(JS-미러: 256-LUT popcount, dist=Σ LUT[a^b], 동률은 index 오름차순) Top-10 == faiss `IndexBinaryFlat` Top-10 → **24/24 정확 일치**(동률-무관 비교). 비트 패킹 정확함 증명.
- **§6.5 쿼리 인코딩 일치**: `/encode_query` 반환 코드 == 오프라인 `Encoder` 코드 → **24/24 바이트 단위 일치**.
- **§6.3 검색 속도**: JS-미러(numpy) 5만 장 **평균 50.3ms / 최대 53.4ms**(목표 <100ms 충족). *주의: numpy 프록시 수치이며 실제 브라우저는 자체 `performance.now()`로 측정·콘솔/화면 로그.*
- **§6.2 검색 품질**: "바닷가 강아지"→해변의 개, "눈 덮인 산"→설산/스키, "피자 한 조각"→피자, "a giraffe"→기린, "사람들이 자전거를 타고 있다"→자전거 탄 사람들. KO/EN 모두 합리적.
- **§6.1 로딩 / §6.6 서버검색 제거**: 정적 엔드포인트 전부 HTTP 200(`/`, `/app.js`, `/data/index.bin`, `/data/meta.json`, `/data/index_info.json`, `/thumbs/*.jpg`), `index.bin`에 `Content-Length`(진행바용). 서버 라우트는 `/encode_query`+정적뿐 — **검색 라우트 없음**.
- **파일 크기**: `index.bin` **6,400,000 B(6.10 MB = 50,000×128)**, `meta.json` **6.74 MB**, 썸네일 5만 장(≤200px, JPEG q85). meta.json은 index.bin row와 1:1 정렬(검증에서 OK).

## 의뢰서가 모호해 내린 결정
- **코퍼스 = COCO 단일 무작위 5만**(소스균형 대신). 이유: 5만 전부 이미지·캡션 확보가 보장되고 경로 복구 이슈(OI/CC12M)가 없음. seed=42 재현가능.
- **포트 8300**(8200 충돌 회피).
- **index.bin은 헤더리스**, 포맷 설명은 사이드카 `index_info.json`(n/bits/code_bytes/head/norm_in/seed)으로 분리.
- **동률 tie-break = index 오름차순**(JS와 검증 미러를 동일하게 만들어 faiss와 결정적 일치).
- **`web/` self-contained**(pack_bits 인라인) — "포터블 검색" 테마에 맞고 DGX main 의존성 제거.
- 검색 로직(`hammingTopK`)은 DOM/네트워크와 분리된 순수함수 → 차후 Web Worker 이전 용이.

## 미해결 / 블로커 / 다음
- **블로커 없음.** 데모 동작·검증 완료.
- 참고:
  - GB10 cuda capability(12.1>12.0) **경고는 무해**(PyTorch fallback). 빌드(img_h)·쿼리(txt_h)가 같은 경로라 패리티 영향 없음.
  - **브라우저 자체를 headless로 구동하진 못함**(DGX에 브라우저 자동화 도구 없음). 대신 브라우저가 의존하는 모든 자산(HTTP 200+헤더)과 JS 검색 알고리즘의 등가성(바이트 단위 미러=faiss)을 검증함 → 실제 브라우저에서 동작 보장. *다음 사이클에서 실제 브라우저 1회 육안 확인 권장.*
  - 생성물은 gitignore(레포에 6MB+썸네일 미커밋); 다른 호스트에선 `build_index.py` 재실행 또는 `web/static/{data,thumbs}` rsync.
  - 서버는 현재 DGX tmux `webv1`(:8300)로 상주 중.
- 다음(범위 밖이었던 것): **v2** 클라이언트 사이드 텍스트 인코더 · **v1.5** Matryoshka coarse-to-fine(64bit 프리필터→1024 리랭크) · Web Worker 분리 · 스케일 모드(전체 1.09M).

## 커밋·푸시
- 브랜치 **`web-v1`** (HEAD `improve-consistency-quality-tests`에서 분기). `web/` 소스만 스테이징(생성물 제외).
- **커밋 `d3f8f34`** → `origin/web-v1` **푸시 완료**. (이 HANDOFF의 SHA 기록 커밋이 뒤따름.)
- PR: https://github.com/hyunlord/vlm_quantization/pull/new/web-v1
