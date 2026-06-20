# PAPER BACKBONES — 백본 일반화 (확장 ③)

핵심 파이프라인(전체 해시 헤드 학습 → 1-bit 검색 → head-adapt → 정밀도)을 **다른 크로스모달 백본**에서 재현해, 논문 **기여 ②~④가 SigLIP2-So400m 전용이 아님**을 입증. 리뷰어의 "SigLIP-specific?"에 대한 일반성 검증. **헤드만 학습**(백본 finetune 없음), **1024-bit**, eval_korean 5K 프로토콜. COCO ft113 산출물·`index.bin`·`common.py` 무변경 — 백본별 별도 산출물(`/tmp/bb_<tag>_*`).

> **진행 상태**: 백본 #1 **siglip2-base 완료**. 백본 #2 **jina-clip-v2 대기**(다른 계열; einops/timm 등 의존성 설치 + 자체 전처리 통합 필요 — 사용자 확인 후 진행).

## 교차-백본 비교표 — `paper/backbones.csv`
| backbone | dim | server 1-bit EN/KO R@10 | head-cont ceiling EN/KO | MiniLM head-adapt EN/KO R@10 | head flip bf16/int8 | emb flip bf16/int8 |
|---|---|---|---|---|---|---|
| **SigLIP2-So400m** (ft113, 참조) | 1152 | **79.92 / 71.08** | 80.52 / 72.04 | 78.16 / 65.44 | 0.95 / **18.97** | 0.18 / 5.45 |
| **SigLIP2-base**-patch16-256 | 768 | **78.12 / 56.7** | 78.62 / 57.9 | 76.14 / 53.78 | 0.98 / **24.48** | 0.0 / 9.19 |

(flip = bitflips/1024, 텍스트 코드 EN+KO 평균; eval_paper.py "C" 동일 스킴 — head int8=`quantize_dynamic`, emb int8=per-row storage-cast. So400m 행은 `precision.csv` 재현: head int8 18.97 / emb 5.45 정확 일치로 스킴 검증.)

## 백본 #1 — SigLIP2-base 결과 (Stage 3~5)
- **Stage 3 (1-bit + 이진화 상한)**: server 1-bit EN R@10 **78.12** / KO **56.7**(R@1 38.2/21.1). head-continuous 상한 EN 78.62 / KO 57.9. → **1-bit 크로스모달 검색이 base에서 작동**.
- **Stage 4 (head-adapt)**: MiniLM(현 최적 오프라인)을 base의 frozen img_h 코드에 head-adapt → EN **76.14** / KO **53.78**.
- **Stage 5 (정밀도)**: head bf16 flip 0.98 / int8 **24.48**; emb bf16 0.0 / int8 9.19.

## 일반성 판정 — 기여 ②~④ 모두 **재현 ✓**
- **④ 이진화 거의 공짜 = 재현**: base 상한−1bit 갭 = **EN 0.5pt / KO 1.2pt**(So400m 0.6 / 0.96와 동급). sign() 비용이 base에서도 ~1pt.
- **③ head-adapt가 오프라인 회복 = 재현**: base MiniLM head-adapt EN 76.14 = server 78.12 대비 **−1.98pt**(So400m EN −1.76와 동급; ~97% 회복). Ext①의 head-adapt 우위가 base에서도 성립.
- **② head 정밀도 민감(int8 head ≫ emb) = 재현**: base head int8 flip **24.48 ≫ emb int8 9.19**(2.7×). So400m 18.97 ≫ 5.45(3.5×)와 동일 패턴 — head가 0-근처에 민감(int8에서 head가 emb보다 훨씬 더 flip), bf16은 양쪽 무해(~1 / 0). **head를 fp16+ 로 둬야 한다는 결론이 base에서도 유효**.

**결론**: 세 기여 모두 **다른 크기의 같은-계열 백본(base, 768-d, 256px, So400m의 ~3분의 1)** 에서 재현. 핵심 발견은 **So400m 전용이 아니라 일반적**. (절대값은 백본마다 다름 — base는 멀티링구얼이 약하고 *EN-only 학습*(아래 caveat)이라 KO가 특히 낮음. 작업의뢰서대로 *패턴 재현*이 포인트이며 절대값 일치는 아님.)

## 레시피 · 공정성 caveat (정직히)
- **헤드 학습 = `train_1024` 레시피 재사용**(img_h+txt_h 동시, CombinedHashLoss[contrastive+ortho+quant+balance+cons+lcs], AdamW(둘 다), OneCycleLR, 25ep BS512, `hp_results.json` 파라미터, norm_in=1). 바뀐 건 **백본+in_dim(768)** 뿐.
- **단순화(명시)**: base 헤드는 **clean 임베딩·COCO-EN만**으로 학습 — So400m=ft113가 쓴 *증강(weak/strong) 뷰*(변환이 git에 없음)·*OI/RKD/CROVCA*·*KO finetune*은 제외. 따라서 So400m 참조 행과 **절대값 직접 비교는 부적절**(특히 KO: base는 EN-only 전이, ft113은 KO finetune). EN이 더 깨끗한 교차-백본 지표. 패턴(②③④)은 이 차이에 견고.
- **정밀도 스킴 = eval_paper.py "C"와 동일**으로 맞춤(head int8=`torch.ao.quantize_dynamic`, emb int8=per-row, flip=텍스트 코드 EN+KO). So400m을 이 스킴으로 재측정 → `precision.csv`(18.97/5.45) **정확 재현**으로 교차-백본 비교 타당성 확보.
- head-adapt = head-only, base 이미지 코드 frozen(Ext① 그대로), MiniLM trained config(no-prefix).

## 채우는 표/섹션 + 커밋
- **기여 ②~④ 일반성(§4/§6)**: 다른 크기·같은 계열 백본(base)에서 head 민감도·head-adapt 회복·이진화 거의 공짜가 **모두 재현** → 방법이 백본-불문 일반적. (#2 jina-clip-v2로 *다른 계열* 추가 검증 예정.)
- 산출물: `web/paper_backbone_run.py`(5-stage 파이프라인, 백본 인자화) + `paper/backbones.csv`(누적). 임베딩·헤드(`/tmp/bb_*`) 미커밋(재생성 스크립트만).
- 재현: `HEAD_PATH=/tmp/ft_ko_113.pt .venv/bin/python web/paper_backbone_run.py --model google/siglip2-base-patch16-256 --tag siglip2-base --epochs 25`
- 브랜치 **`paper-backbones`**(`web-v3-hybrid`에서 분기). #1 커밋 SHA: _커밋 후 기재_ → push.
