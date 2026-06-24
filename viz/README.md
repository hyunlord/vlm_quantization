# viz/ — 인터랙티브 시각화 (원클릭 실행)

현재 1-bit 해싱 방식이 *수학적으로 무엇을 하는지* 보고, *어디를 바꿔야 판이 바뀌는지* 찾기 위한 탐색용
인터랙티브 HTML 모음. 함께 보는 수식 문서: [`../docs/FORMULATION.md`](../docs/FORMULATION.md).

## 실행 (명령 하나)
```bash
git checkout analysis-signloss-outlier
cd viz
bash serve.sh          # 또는: python3 serve.py
```
- 빈 포트(8800부터) 자동 선택 → 콘솔에 실제 URL 출력 → 기본 브라우저 자동 오픈.
- `file://` 로 직접 열면 JSON fetch가 CORS로 막힙니다 → **반드시 이 서버로** 여세요.
- 종료: `Ctrl+C`.
- 서버 시작 전 JSON·썸네일 완결성을 검사하고, 누락 시 한글 에러 + 해결법을 출력합니다.

## 무엇이 뜨나 (대시보드 + 7개)
1. **forward_stages** — 한 샘플의 단계별 분포 e→sliced→BN→L2 z→sign (어디서 정보가 표준화/소멸).
2. **distribution_explorer** — |z|·비트균형·비트상관, **BN on/off** 토글, 경로 오버레이.
3. **sign_information_loss** — 같은 쿼리를 연속 점수 vs Hamming 으로 랭킹(부호가 버리는 정보).
4. **embedding_space** — SigLIP 임베딩/코드/12언어의 PCA·UMAP 2D.
5. **retrieval_anatomy** — 쿼리별 정답 vs 갤러리 Hamming 분포 + 비트차 맵.
6. **signloss** — A-베팅 GO/NO-GO: 연속/비대칭/Hamming R@10 비트별 격차.
7. **outlier_gallery** — 실패 케이스 썸네일(정답 vs 검색오답) — 라벨한계(A) vs 고립(B) 눈으로 판정.

각 페이지 상단에 접이식 한글 해설(📊무엇을/🔍어떻게읽나/👀주목점/🔗연구연결)이 있습니다.

## 데이터
- `data/*.json` — 각 HTML이 로드(미리 계산됨; 브라우저는 무거운 연산 안 함). 재생성: `scripts/{viz_data,signloss_quant,outlier_debug}.py` (DGX, 캐시 임베딩 필요).
- `data/thumbs_outlier/*.jpg` — outlier_gallery용 썸네일(레포에 커밋됨, ~650KB). 누락 시 `scripts/outlier_debug.py`로 재생성(원본 COCO 이미지 필요).

## 문제 해결
- **포트 사용 중**: serve.py가 자동으로 다음 포트를 찾음. 강제 정리: `lsof -ti tcp:8800 | xargs kill`.
- **차트가 안 뜸**: Plotly를 CDN에서 로드 → 인터넷 확인. (썸네일·데이터는 로컬이라 오프라인에서도 뜸.)
- **이미지가 회색 박스**: 썸네일 누락 → 위 재생성 안내 참고.
