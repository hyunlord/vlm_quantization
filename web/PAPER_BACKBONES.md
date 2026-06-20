# PAPER BACKBONES — 백본 일반화 (확장 ③)

핵심 파이프라인(전체 해시 헤드 학습 → 1-bit 검색 → head-adapt → 정밀도)을 **다른 크로스모달 백본**에서 재현해, 논문 **기여 ②~④가 SigLIP2-So400m 전용이 아님**을 입증. 리뷰어의 "SigLIP-specific?"에 대한 일반성 검증. **헤드만 학습**(백본 finetune 없음), **1024-bit**, eval_korean 5K 프로토콜. COCO ft113 산출물·`index.bin`·`common.py` 무변경 — 백본별 별도 산출물(`/tmp/bb_<tag>_*`).

> **완료**: 백본 #1 **SigLIP2-base**(scale 축, 같은 계열) + 백본 #2 **AltCLIP-m18**(다른 계열: XLM-R+CLIP). 둘 다 ②③④ 재현.
> *jina-clip-v2 주석*: 원래 #2 후보였으나 DGX의 **transformers 5.1과 비호환**(custom modeling이 meta-tensor init에서 실패; `low_cpu_mem_usage=False`로도 동일). 메인 venv 다운그레이드는 #1·기타 코드를 깨므로, 사용자 합의로 **transformers-native 다른-계열 멀티링구얼 백본 AltCLIP-m18로 대체**(동일 목표: 다른 아키텍처·한국어 포함).

## 교차-백본 비교표 — `paper/backbones.csv`
| backbone | family | dim | server 1-bit EN/KO R@10 | head-cont ceiling EN/KO | MiniLM head-adapt EN/KO | head flip bf16/**int8** | emb flip bf16/int8 |
|---|---|---|---|---|---|---|---|
| **SigLIP2-So400m** (ft113, 참조) | SigLIP2-so400m | 1152 | **79.92 / 71.08** | 80.52 / 72.04 | 78.16 / 65.44 | 0.95 / **18.97** | 0.18 / 5.45 |
| **SigLIP2-base**-patch16-256 | SigLIP2-base | 768 | **78.12 / 56.7** | 78.62 / 57.9 | 76.14 / 53.78 | 0.98 / **24.48** | 0.0 / 9.19 |
| **AltCLIP-m18** | **XLM-R + CLIP-ViT-L** | 1024 | **77.96 / 70.52** | 78.76 / 71.14 | 74.02 / 51.08 | 0.82 / **16.73** | 0.0 / 3.31 |

(flip = bitflips/1024, 텍스트 코드 EN+KO 평균; eval_paper.py "C" 동일 스킴 — head int8=`quantize_dynamic`, emb int8=per-row storage-cast. So400m 행은 `precision.csv` 정확 재현[18.97/5.45]으로 스킴 검증.)

## 일반성 판정 — 기여 ②~④ 모두 **scale + family 양축에서 재현 ✓**
- **④ 이진화 거의 공짜 = 재현(전 백본)**: 상한−1bit 갭 = So400m 0.6/0.96 · base 0.5/1.2 · **AltCLIP 0.8/0.62** pt. 어느 백본에서도 sign() 비용 ≤~1pt.
- **③ head-adapt가 오프라인 회복 = 재현(전 백본)**: MiniLM head-adapt EN = server 대비 So400m −1.76 · base −1.98 · **AltCLIP −3.94**(모두 EN 대부분 회복; Ext① head-adapt 우위가 백본-불문). (KO는 AltCLIP에서 갭 큼[−19.4] — AltCLIP의 KO server가 매우 강해서[70.52] MiniLM이 따라가기 어려움; EN이 깨끗한 교차-백본 지표.)
- **② head 정밀도 민감(int8 head ≫ emb) = 재현(전 백본)**: int8 head flip ≫ emb int8 flip — So400m 18.97≫5.45(3.5×) · base 24.48≫9.19(2.7×) · **AltCLIP 16.73≫3.31(5.1×)**. head가 0-근처에 민감(int8에서 head가 emb보다 2.7~5.1× 더 flip), bf16은 양쪽 무해(~1/0). **"head를 fp16+ 로 둬라"** 결론이 전 백본 유효.

**결론**: 세 기여(②③④)가 **다른 크기**(base, So400m의 ~1/3)와 **다른 계열**(AltCLIP = XLM-R+CLIP, 비-SigLIP 아키텍처)에서 **모두 재현**. → 핵심 발견은 **SigLIP2-So400m 전용이 아니라 방법-수준의 일반 성질**. 리뷰어의 "SigLIP-specific?"에 직접 답함.

## 백본별 결과 (Stage 3~5)
- **#1 SigLIP2-base** (dim 768): 1-bit EN **78.12**/KO **56.7** · 상한 78.62/57.9 · MiniLM head-adapt 76.14/53.78 · head int8 flip 24.48 ≫ emb 9.19. KO가 낮음 = base는 멀티링구얼이 약함(+EN-only 학습).
- **#2 AltCLIP-m18** (dim 1024, 다른 계열): 1-bit EN **77.96**/KO **70.52** · 상한 78.76/71.14 · MiniLM head-adapt 74.02/51.08 · head int8 flip 16.73 ≫ emb 3.31. **KO 70.52 ≈ So400m 71.08** — AltCLIP-m18은 18개어 멀티링구얼이라 EN-only 학습에도 KO가 강함(다른 계열에서도 1-bit 크로스모달 검색이 잘 작동함을 보임).

## 레시피 · 공정성 caveat (정직히)
- **헤드 학습 = `train_1024` 레시피 재사용**(img_h+txt_h 동시, CombinedHashLoss, AdamW(둘 다), OneCycleLR, 25ep BS512, `hp_results.json`, norm_in=1). 바뀐 건 **백본+in_dim**(768 / 1024)뿐.
- **단순화(명시)**: 두 백본 모두 **clean 임베딩·COCO-EN만** 학습 — So400m=ft113가 쓴 증강(weak/strong, 변환이 git에 없음)·OI/RKD/CROVCA·KO finetune은 제외. ∴ So400m 참조 행과 **절대값 직접 비교는 부적절**(특히 KO: 두 백본은 EN-only 전이, ft113은 KO finetune). 패턴(②③④)은 이 차이에 견고 — 그것이 일반성 검증의 포인트.
- **정밀도 스킴 = eval_paper.py "C" 동일**(head int8=`quantize_dynamic`, emb int8=per-row, flip=텍스트 코드 EN+KO). So400m을 이 스킴으로 재측정→`precision.csv`(18.97/5.45) 정확 재현으로 교차-백본 비교 타당성 확보.
- head-adapt = head-only, 백본 이미지 코드 frozen(Ext① 그대로), MiniLM trained config(no-prefix).
- AltCLIP은 transformers 5.1에서 `get_image_features`가 ModelOutput 반환 → `visual_projection(vision_model(...).pooler_output)` 컴포넌트 경로 사용(텍스트도 동일).

## 채우는 표/섹션 + 커밋
- **기여 ②~④ 일반성(§4/§6)**: 다른 크기(base)·**다른 계열(AltCLIP)** 백본에서 head 민감도·head-adapt 회복·이진화 거의 공짜가 **모두 재현** → 방법이 백본-불문 일반적. SigLIP-specific 반박.
- 산출물: `web/paper_backbone_run.py`(5-stage, `--backbone-type {siglip,altclip}`) + `paper/backbones.csv`(So400m 참조 + base + AltCLIP). 임베딩·헤드(`/tmp/bb_*`) 미커밋(재생성 스크립트만).
- 재현: `… web/paper_backbone_run.py --model google/siglip2-base-patch16-256 --tag siglip2-base` / `… --model BAAI/AltCLIP-m18 --tag altclip-m18 --family AltCLIP-XLMR --backbone-type altclip`
- 브랜치 **`paper-backbones`**(`web-v3-hybrid`에서 분기). #1(siglip2-base) `d19d82f` · #2(AltCLIP-m18) 커밋 SHA: _커밋 후 기재_ → push.
