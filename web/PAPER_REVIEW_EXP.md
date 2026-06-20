# PAPER REVIEW EXP — 리뷰 대비 측정 배치 (항목 #1·#2·#3·#5)

리뷰어 대비 4개 측정. **1024-bit, eval_korean 5K 프로토콜**, 기존 so400m 캐시 위 평가/측정(항목 #1만 새 베이스라인 파이프라인). COCO ft113·`index.bin`·`common.py` 무변경. 산출 `paper/{baseline,review_seeds,review_map,review_buildcost}.csv` + 스크립트 `web/paper_{baseline,review_exp}.py`.

## #1 오프-더-셸프 베이스라인 ★ — "스톡 작은 CLIP 통째 바이너리화로 충분한가?"
> **모델 대체(보고)**: 1순위 **M-CLIP**·보조 **MobileCLIP** 둘 다 환경의 **transformers 5.1과 비호환**(M-CLIP은 `from_pretrained` meta-device init 실패 — 예전 jina와 동일류; MobileCLIP은 비-transformers 패키지). 작업의뢰서 지침대로 **transformers-네이티브 스톡 CLIP으로 대체**: **AltCLIP-m18**(멀티링구얼, XLM-R+CLIP-ViT-L, 1024-d) + **openai/clip-vit-base-patch32**(작은 EN-전용, 512-d, MobileCLIP 역할).

| 모델 | variant | dim | EN R@10 | KO R@10 |
|---|---|---|---|---|
| **CLIP-ViT-B/32** (작은 스톡, EN전용) | **(a) naive** sign | 512 | 45.1 | 0.3 |
| **AltCLIP-m18** (스톡 mCLIP) | **(a) naive** sign | 1024 | 68.0 | 62.2 |
| SigLIP2-So400m (참조) | (a) naive sign (head 없음) | 1152 | 70.2 | 52.2 |
| AltCLIP-m18 | **(b) head** (1024, Ext③) | 1024 | 77.96 | 70.52 |
| — SigLIP2+e5-small (배포 offline) | head-adapt | 1024 | 74.0 | 66.2 |
| — SigLIP2+MiniLM (head-adapt) | head-adapt | 1024 | 78.16 | 65.44 |
| **— SigLIP2-So400m server** | head (1-bit) | 1024 | **79.92** | **71.08** |

- **판정: 스톡 CLIP "통째 naive 바이너리화"는 부족 → SigLIP2 + 학습 헤드 경로가 필요**. 최고 naive(AltCLIP 68.0/62.2)도 배포 offline(74.0/66.2)·server(79.9/71.1)에 EN −6~12·KO 미달; 작은 스톡 CLIP(CLIP-B/32 naive 45.1 EN / KO ≈0)은 훨씬 낮음(EN전용이라 KO 붕괴).
- **헤드의 기여가 핵심**: 같은 백본 so400m에서 naive(70.2/52.2) → 학습 헤드 server(79.92/71.08) = **EN +9.7 / KO +18.8**. AltCLIP도 naive(68/62)→head(78/70.5) = EN +10/KO +8.4. 즉 *백본만 바이너리화*가 아니라 *학습 헤드*가 1-bit 품질을 만든다.
- 정직 노트: 베이스라인 naive는 **자체 갤러리**(모델별 원본 임베딩 바이너리)라 절대 비교는 *베이스라인 내부* 한정. AltCLIP을 ViT-L(작지 않음)로 대체했으므로 "작은 스톡 CLIP"의 *상한*에 가까움 — 그래도 server/head-adapt에 못 미침 → 결론 강화. clip-B/32 (b)head는 미실행(AltCLIP가 (a)+(b) 완비 + so400m naive↔head로 헤드 효과 입증되어 판정에 불요; 후속 가능).

## #2 분산/시드 — head-adapt 안정성
- **MiniLM head-adapt(no-prefix) × 3시드(0/1/2)**: EN R@10 **78.45 ± 0.13** | KO R@10 **65.81 ± 0.17** (n=3; seed별 EN 78.28/78.46/78.6, KO 66.02/65.82/65.6). `paper/review_seeds.csv`.
- **관측 분산 ±0.1~0.2 pt** — head-adapt는 시드에 매우 안정적. server ft113은 고정 단일 체크포인트(79.92/71.08, 재학습 비쌈 → 단일값).

## #3 mAP — 핵심 행 mAP@10 (R@10 옆) `paper/review_map.csv`
| 행 | EN R@10 / mAP@10 | KO R@10 / mAP@10 |
|---|---|---|
| server (so400m+ft113) | 79.92 / **53.34** | 71.08 / **42.49** |
| MiniLM head-adapt | 78.16 / **49.68** | 65.44 / **37.24** |
| e5-small 배포 offline | 74.0 / **44.84** | 66.2 / **36.28** |

(mAP@10 = single-gold 평균 정밀도 = mean(1/rank, top-10 내). R@10 전부 앵커 재현 — server 79.92/71.08, MiniLM 78.16/65.44, e5-small 74.0/66.2.)

## #5 인덱스 구축 비용 · 메모리 (50K) `paper/review_buildcost.csv`
- **구축 시간**: so400m vision 인코딩 **20.8 img/s**(bf16, PIL 디코드-바운드) → 50K ≈ **40 min**(인코딩이 지배적). img_h+pack(50K) **2.69 s**(18.5K img/s) · 인덱스 직렬화 0.006 s = 헤드/패킹은 무시할 수준.
- **메모리**: 인덱스 **6.4 MB**(50K × 128 B/img) · JS 검색 footprint ≈ **6.6 MB**(인덱스 + 쿼리코드 128 B + popcount LUT + 거리버퍼 Int32 50K). 브라우저 상주 가능.

## 채우는 논문 표/문장 + 커밋
- **베이스라인 비교행**(#1): 스톡 CLIP naive < SigLIP2+head → 방법(강한 백본+학습 헤드) 정당화. **분산 표기**(#2): head-adapt R@10 = mean±0.1~0.2pt(시드 안정). **mAP 컬럼**(#3): 핵심 행 mAP@10. **구축비용**(#5): 50K 40min 인코딩·6.4MB 인덱스·~6.6MB JS.
- 산출 스크립트: `web/paper_baseline.py`(#1 naive) + `web/paper_review_exp.py`(#2/#3/#5). 캐시/헤드(`/tmp/*`) 미커밋.
- 브랜치 **`paper-review-exp`**(`web-v3-hybrid`에서 분기). 커밋 **`7a104f0`** → `origin/paper-review-exp` push 완료.
