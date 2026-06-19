# PAPER ENCODERS — head-adapt 일반화 (확장 ①)

C1(head-adapt) 레시피를 **인코더만 바꿔** 다른 소형 멀티링구얼 텍스트 인코더에 적용. 목적 = 기여 ④("head-adapt > encoder-matching")가 **e5 전용이 아니라 일반 레시피**임을 입증. eval-only(헤드만 학습), 동일 recipe/hp/split/갤러리 — **유일 변수 = 인코더**. 앵커·C1 재현 ✓.

## 비교표 — `paper/encoders.csv`
| 인코더 | 계열 | dim | params | int8 MB | 배포가능 | EN R@1/5/10 (Δ서버) | KO R@1/5/10 (Δ서버) | overlap EN/KO |
|---|---|---|---|---|---|---|---|---|
| **server** (so400m+ft113) | SigLIP2-so400m | 1152 | 707.8M | 708 | ✗ | 41.6/69.0/**79.92** | 30.5/59.0/**71.08** | 1.0/1.0 |
| e5-small (C1, no-prefix) | e5/XLM-R | 384 | 117.7M | 118 | ✓ | 32.2/61.7/**74.0** (−5.9) | 24.1/52.3/**66.2** (−4.9) | 0.42/0.36 |
| e5-small **+"query:"** | e5/XLM-R | 384 | 117.7M | 118 | ✓ | 32.1/62.1/**74.12** (−5.8) | 25.3/52.5/**65.86** (−5.2) | 0.42/0.37 |
| **MiniLM-L12-v2** | paraphrase/XLM-R | 384 | 117.7M | 118 | ✓ | 37.1/66.6/**78.16** (**−1.8**) | 25.8/52.8/**65.44** (−5.6) | 0.48/0.36 |
| **e5-base** | e5/XLM-R | 768 | 278.0M | 278 | ✓ | 34.8/64.6/**77.02** (−2.9) | 27.4/55.5/**68.86** (**−2.2**) | 0.46/0.38 |

## 학습/공정성 설정
- **레시피(C1과 동일, 유일 변수=인코더)**: frozen 인코더(mean-pool) → 학습가능 `txt_h'=NestedHashLayer(dim→384→1024)` → **frozen ft113 img_h 코드에 정렬**(CombinedHashLoss: InfoNCE+EAQL+ortho+balance+lcs). hp=`hp_results`(lr 3.06e-4, ortho .18/quant .13/cons·lcs .65/temp .083), 25ep, BS512, OneCycle. 학습 226,574쌍(EN 113K + KO 113K), 갤러리=5K test `img_h` 코드, gold caption_i→image_i(eval_korean). 인코더 frozen.
- **pooling+prefix(공정 비교 핵심)**: 전부 mean-pool + L2. **e5-base = "query: " prefix**(e5 권장), **MiniLM = prefix 없음**(sentence-transformers 권장). 
- **공정성 caveat 해소(재측정 완료)**: e5-small을 e5 권장 **"query: " prefix**로도 측정 → **EN 74.12 / KO 65.86**, no-prefix(74.0/66.2)와 **사실상 동일(EN +0.12, KO −0.34)**. **head-adapt에선 prefix가 거의 무영향** — head가 prefixed/raw 임베딩 분포를 학습으로 흡수하므로(prefix는 *zero-shot* e5 품질엔 중요하나 head 학습 시 상쇄). → 아래 비교들이 prefix 오염 없이 성립.

## 해석
- **head-adapt 일반화 = YES**: 세 소형 인코더(≤278M, 두 계열, 384/768-d) **모두** 708M 서버 대비 **EN −1.8~−5.9 / KO −2.2~−5.6** 한 자릿수 pt 내로 회복. 레시피는 e5 전용이 아니라 **일반적**(다른 계열 MiniLM·다른 차원 e5-base에서도 동일하게 대부분의 갭을 회복).
- **prefix 이득 ≈ 0**: e5-small +query: vs no-prefix = EN +0.12 / KO −0.34. head-adapt가 prefix를 흡수 → prefix는 배포에서 채택할 가치 없음(아래 §배포).
- **계열/크기 패턴(prefix-clean)**: **MiniLM(다른 계열, 118M)이 EN 최고(78.16, 서버 −1.8)** — e5-small을 *best config(+prefix, 74.12)*로 둬도 MiniLM이 **EN +4.0** 우세 → MiniLM의 EN 우위는 *실재*(prefix 아티팩트 아님). **e5-base(같은 계열, 768-d, 278M)가 KO 최고(68.86, 서버 −2.2)** — e5-small+prefix(KO 65.86) 대비 e5-base +3.0 → **e5-base의 우위는 *크기*(768-d/278M) 때문, prefix 아님**(둘 다 prefix). 즉 **EN엔 MiniLM, KO엔 e5-base**; head-adapt가 둘 다 끌어올림.
- **overlap ~0.36–0.48**(전 인코더): 독립 인코더라 서버 top-K와 부분 일치(설계상 — 서버와 다른 head). recall(R@K)이 운영 지표. C1 관찰(다른 head=낮은 overlap·좋은 recall)이 인코더 전반에 일관.

## 배포 함의
- **MiniLM-L12-v2 (118MB, Xenova ONNX 존재)**: 현 배포 e5-small보다 **EN 우수·동일 크기** → **더 나은 오프라인 인코더 후보**(특히 EN). KO는 e5-small과 비슷.
- **e5-base (278MB)**: KO 최고지만 2.4× 크기 — KO 중시 + 크기 예산 허용 시.
- **prefix 채택 = 불필요(비채택 권고)**: 이득 ≈0(EN +0.12/KO −0.34)인데 app.js에 "query: " 프리픽스 추가 + `txt_h.onnx` 재익스포트 + Node 파리티 재검증(후속 ①~④)의 복잡도만 늘림. **현 배포(e5-small no-prefix) 유지에 문제 없음**; 업그레이드 레버는 *prefix가 아니라 인코더 교체*(→ MiniLM, EN↑). MiniLM 채택 시에만 ②~④(onnx 재익스포트·파리티·app.js 인코더 교체) 필요, prefix 단계는 불요.
- 셋 다 transformers.js/ONNX 배포 가능. v3 hybrid 오프라인 모드는 인코더 교체만으로 업그레이드 가능(`txt_h.onnx`만 재익스포트, e5는 stock 교체).

## 채우는 표/섹션 + 커밋
- **기여 ④ / §4.3 일반화**(head-adapt는 인코더-불문 일반 레시피; 배포 후보 비교).
- 산출물: `web/paper_encoder_run.py`(인코더 스윕) + `web/paper_encoder_e5prefix.py`(e5-small +query: 정정) + `paper/encoders.csv`. 학습 헤드(`/tmp/txt_h_*.pt`) 미커밋(재생성 스크립트만). img_h/index.bin/common.py/C1 산출물 무변경.
- 재현: `…python web/paper_encoder_run.py` → `…python web/paper_encoder_e5prefix.py`(둘 다 `HEAD_PATH=/tmp/ft_ko_113.pt`).
- 브랜치 **`paper-encoders`**. 스윕 `323795c`/`7042da4`; **e5-small+prefix 정정 = (아래 후속 기록 커밋)** → `origin/paper-encoders` push.
