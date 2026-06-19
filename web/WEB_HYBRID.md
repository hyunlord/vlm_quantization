# WEB HYBRID — 브라우저 통합 + PWA (사이클 6, 최종 제품 형태)

기존 브라우저 검색(`search.js` Hamming over `index.bin`)에 **두 쿼리-인코딩 모드 토글** + **PWA**를 추가. **정확/서버**(so400m `/encode_query`, v1) ↔ **완전 오프라인**(브라우저 내 stock e5 ONNX + `txt_h'`). **Stage A(Node 파리티) PASS → Stage B(브라우저+PWA) 구현 완료.** v1 서버 경로 무손상.

## Stage A — ONNX 익스포트 + Node 파리티 (게이트) ✅ PASS
- **익스포트/폴딩**: `txt_h'`의 **1024-bit continuous 경로만** ONNX로(`hash_head → slice → BN → L2 → tanh`; SignSTE 회피). `tanh(x)>0 == sign(x)>0 ==` pack 비트라 JS `packBits(continuous)`가 `common.py:pack_bits(binary)`와 **바이트 동일**. `dynamo=False`로 **단일 파일**(2.1MB, no `.onnx.data`). e5는 **익스포트 안 함** — stock `Xenova/multilingual-e5-small`(transformers.js)로 사용.
- **양자화**: **e5 = q8(int8)** (transformers.js `dtype:'q8'`), **`txt_h'` = fp32**(2.1MB, 0-민감 head). 통과 정밀도 = **int8(q8) e5**(폴백 불필요).
- **검증**(`onnxruntime` Python): ONNX vs torch max|Δ| **1.5e-7**, packing 일치.
- **Node 파리티**(`web/hybrid_parity.mjs`: transformers.js e5 q8 + `onnxruntime-node` txt_h.onnx + `search.js` packBits/hammingTopK, 5K hold-out):
  | 경로 | EN R@10 (Δc1) | KO R@10 (Δc1) | C1 코드 대비 | 레이턴시 |
  |---|---|---|---|---|
  | PyTorch C1 (사이클 5) | 74.0 | 66.2 | — | ~1.2 ms/q |
  | **ONNX offline (q8 e5)** | **73.64 (−0.36)** | **64.64 (−1.56)** | mean Hamming 45/55 of 1024 (0% byte-identical) | ~10–12 ms/q (Node) |
  - **게이트 통과**: R@10이 C1의 **~2pt 이내**(EN −0.36 / KO −1.56). q8 e5 양자화로 코드는 ~45–55/1024비트 다르지만(0-민감 아님) **R@10 보존**. JS packBits == pack_bits 확인.
- **크기(첫 로딩)**: stock e5 q8 ~**118MB**(transformers.js가 HF/CDN서 받고 SW 캐시) + `txt_h.onnx` **2.1MB** + tokenizer.json + `index.bin` 6MB + `meta.json` 6.7MB. (e5는 공유/CDN 가능 = 커스텀은 2.1MB뿐.)

## Stage B — 브라우저 통합 + PWA (구현 완료, 실행은 §수동확인)
- **`web/static/app.js` 모드 토글**: `정확(서버)` = `POST /encode_query`(v1 그대로); `오프라인(브라우저)` = `transformers.js e5(q8) → onnxruntime-web txt_h.onnx(fp32) → packBits(search.js) → hammingTopK(search.js) over index.bin`. 오프라인 인코더는 **lazy-load**(최초 1회 e5 다운로드, 진행 표시). 모드·인코드 레이턴시·"근사 모드" 라벨 표시. **검색 코어(search.js)·index.bin 불변.**
  - 라이브러리: transformers.js + onnxruntime-web를 **CDN(jsdelivr) ESM dynamic import**. 오프라인 코드 공간 = index.bin 공간(`txt_h'`가 ft113 `img_h` 코드에 정렬 학습됨)이라 packing만 맞으면 Hamming 성립(Stage A로 검증).
- **PWA**(`web/static/sw.js` + `manifest.webmanifest` + `icon.svg`): service worker가 app shell(`/`,`app.js`,`search.js`) + `index.bin`/`meta.json`/`index_info.json` + `onnx/txt_h.onnx` + manifest/icon을 **precache**, CDN 라이브러리·HF e5 모델 샤드·썸네일은 **runtime cache** → 최초 온라인 1회 후 **오프라인 모드 완전 오프라인 동작**. 설치 가능(standalone). `/encode_query`(정확 모드)는 network-only → 오프라인 시 graceful(오프라인 모드로 안내).
- **서버 라우트 추가**(`web/query_server.py`, **additive**): `/sw.js`·`/manifest.webmanifest`·`/icon.svg`·mount `/onnx`. **CC 검증됨**: 전 엔드포인트 HTTP 200(`/` `app.js` `search.js` `sw.js` `manifest` `icon.svg` `onnx/txt_h.onnx` 2.19MB), **정확 모드 `/encode_query` 동일 코드 반환(v1 무손상)**. (참고: `/onnx/txt_h.onnx`가 `text/plain`으로 서빙되나 fetch/ort는 바이너리로 읽어 무해.)

## 사용자 수동 확인 절차 (브라우저 + 오프라인 PWA — CC는 브라우저 실행 불가)
1. **DGX 준비**(완료 상태): 인덱스 빌드 + `txt_h.onnx` 익스포트(`web/export_txth_onnx.py`) + 서버 기동(tmux `webv1`, `:8300`).
2. **접속**: **service worker는 secure context 필요**(https 또는 localhost). 권장 = SSH 터널 → localhost:
   `ssh -L 8300:localhost:8300 dgx-spark` 후 브라우저 `http://localhost:8300/`. (또는 ngrok https.)
3. **오프라인 모드 동작**: 드롭다운을 `오프라인(브라우저)`로 → 최초 쿼리 시 e5 다운로드(상태바 진행) → 결과 렌더. DevTools 콘솔에 `[search:offline] encode … ms, search … ms` 로그 확인. 한국어("바닷가 강아지")·영어 모두.
4. **PWA 오프라인 검증**: (오프라인 인코더를 온라인서 1회 로드한 뒤) DevTools → Application → Service Workers 등록 + Cache Storage `1bit-hybrid-v1`에 app/index/onnx/e5 샤드 캐시 확인 → **Network: Offline 체크(또는 네트워크 차단) → 페이지 새로고침 → 오프라인 모드로 쿼리 → 결과 반환**(백엔드 0). 정확 모드는 오프라인 시 실패(의도).
5. **설치**: 주소창 install 아이콘 → standalone 앱.

## 결정 / 블로커 / 다음
- **결정 = GO(최종 제품 형태)**: hybrid(정확=서버 so400m / 오프라인=브라우저 e5+txt_h') + PWA. "완전 오프라인" 마일스톤 코드 완비. 오프라인 = **근사 모드**(서버와 결과 다를 수 있으나 recall 근접; Stage A로 C1 ~2pt 내 검증). v1·`index.bin`·`img_h`·`common.py`·서버 경로 무변경.
- **블로커/주의**: (1) **SW는 https/localhost 필수** — DGX IP+http 직접 접속은 SW 미등록(터널/ngrok 사용). (2) 첫 로딩 e5 ~118MB(이후 캐시). (3) 브라우저/ort-web 실제 실행은 수동 확인 필요(위 절차).
- **다음**: 공개 배포(정적 호스트 + e5 CDN; 전체 썸네일 precache로 완전-오프라인 갤러리; COOP/COEP로 ort-web 멀티스레드 가속 옵션); 또는 C1b로 오프라인 recall 추가 개선.

## 산출물 / 재현
- 추가/변경: `web/export_txth_onnx.py`, `web/hybrid_parity.mjs`(Stage A); `web/static/{app.js,search.js(+packBits),index.html,sw.js,manifest.webmanifest,icon.svg}`, `web/query_server.py`(라우트), `web/.gitignore`(static/onnx 추가).
- ONNX 모델 **미커밋**(`static/onnx/` gitignore): 재생성 `HEAD_PATH=/tmp/ft_ko_113.pt .venv/bin/python web/export_txth_onnx.py`. Node 파리티 재실행: `cd /tmp/v2spike_js && DTYPE=q8 node hybrid_parity.mjs`(search.js→search.mjs 복사, onnxruntime-node@1.24.3).

## 커밋·푸시
- 브랜치 **`web-v3-hybrid`**(`web-v2-headadapt`에서 분기). 프론트엔드+스크립트+보고서 커밋, ONNX 미커밋.
- **커밋 SHA**: (아래 후속 기록 커밋 참조) · **origin/web-v3-hybrid 푸시 완료**.
