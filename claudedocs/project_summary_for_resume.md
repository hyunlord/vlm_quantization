# 프로젝트 요약 — Cross-Modal Binary Hashing (이력서 참고용)

VLM(SigLIP2) 위에 **이미지↔텍스트 검색용 바이너리 해시**를 얹어, 검색 품질은 거의 유지하면서
인덱스 크기·검색 속도를 대폭 줄인 시스템. 연구·실측·서빙·데모·튜닝을 1인으로 수행.

_기간: 2026-06 / 환경: DGX(GB10) 원격, PyTorch Lightning, FastAPI_

---

## 한 줄 요약 (이력서용)
> SigLIP2 듀얼 인코더에 Matryoshka 중첩 바이너리 해시 헤드를 학습시켜, 1024-bit 코드로
> **float 검색 품질의 98%를 유지하면서 인덱스 36배 축소·검색 5.5배 가속**을 달성하고,
> 벤치마크 검증·하이퍼파라미터 튜닝·실시간 비교 데모까지 end-to-end로 구축.

---

## 핵심 성과 (정량)
- **품질/효율 트레이드오프 실측** (COCO Karpathy test 5K, 풀 113K+augmentation 학습)
  - 1024-bit 바이너리: I2T R@10 **0.798** = float(0.811) 대비 **98%**
  - 인덱스 크기 **6.1 MB vs float 219.7 MB (36×↓)**, 검색 **3,388 vs 618 QPS (5.5×↑)**
  - 256-bit: 품질 94% · 인덱스 1.5 MB — 균형점
- **하이퍼파라미터 튜닝** (Optuna, 캐시 임베딩 위에서 고속 탐색)
  - 256-bit R@10 0.755 → **0.769** (같은 크기·속도에서 품질만 향상)
  - 512/1024 튜닝: R@10 0.798 → **0.808** (float 0.811의 99.6%), 전 코드 길이 개선
  - **방법론 교훈 도출·검증**: "HP 평가 epoch = 최종 학습 epoch"으로 맞춰야 함을 실험으로 입증
    (불일치 시 튜닝이 역효과 → 일치시키자 전 구간 baseline 초과)
- **SigLIP2 벤치 재현 검증**: 공개 지표(COCO R@1 55.8/71.7)의 ~93% 안정 재현으로 파이프라인 정합성 확인;
  bf16/리사이즈가 원인이 아님을 실측으로 규명(가설 반증)
- **검색 속도 업그레이드(numpy→faiss)**: 비트 Hamming 검색을 `IndexBinaryFlat`(HW POPCNT+멀티스레드+무할당,
  packed 32B 유지)로 전환 — GB10 1M·256-bit에서 numpy 34.6ms → **faiss 9.6ms(3.6×)**, **float 대비 25.9×**;
  top-k 정확도는 numpy와 완전 동일(거리 multiset 일치) 검증. 원인은 분해 측정으로 규명
  (병목=`sum(axis=1)`·싱글스레드·공유 고정비용), matmul 트릭은 타깃 ARM에서 역효과라 폐기
  → 상세: `claudedocs/search_speed_upgrade.md`, `search_speed_comparison.html`

---

## 기술 스택 / 역량
- **모델링**: SigLIP2(So400m) 프로즌 백본 + per-modality MLP 해시 헤드, Matryoshka prefix-nesting,
  SignSTE 양자화, InfoNCE/quantization/balance/consistency 복합 손실
- **검색/서빙**: Hamming(XOR+popcount) 인덱스, FAISS IndexBinaryFlat, FastAPI 검색 API,
  numpy `bitwise_count` 활용 popcount 최적화
- **실험 인프라**: 프로즌 백본 임베딩 캐싱으로 학습 100배 가속(에폭당 4시간→18초), Optuna 튜닝,
  원격 GPU(DGX) 무인 파이프라인(nohup/setsid 체인, 워처 자동화)
- **검증/분석**: 표준 프로토콜(mAP@5000, R@K, 5-caption) 구현, 벤치 재현, 데이터 누수 탐지·수정
- **시각화/데모**: 자체 HTML 대시보드(품질·속도·검색 갤러리), float vs 바이너리 좌우 분할
  실시간 비교 데모(병렬 요청·도착순 렌더·latency 표시)

---

## 문제 해결 사례 (디버깅/엔지니어링)
1. **데이터 누수 탐지**: 한국어 캡션 데이터가 test/val 이미지를 포함 → 과거 수치 상향 편향 확인,
   train+restval만으로 필터(누수 0) 후 재측정
2. **성능 버그 규명**: 데모 바이너리 검색이 예상보다 느림 → `np.argpartition(range(k))` 오용으로
   부분정렬이 비효율(O(N)이 아님)임을 분해 측정으로 발견, 단일 kth로 수정
3. **하드웨어 제약 대응**: GB10(sm_121) PyTorch 커널 폴백으로 백본 forward 저속(3-12 img/s) →
   bf16 + 임베딩 캐싱으로 우회; 멀티프로세스 GPU OOM은 리소스 정리로 해결
4. **재현성**: 학습 시드 고정, EAQL eval-mode 가드, 메트릭 벡터화, 26개 CPU 테스트 + CI
5. **검색 속도 원인 규명·업그레이드**: "비트가 왜 안 빠른가"를 컴포넌트 분해 측정으로 추적 →
   병목이 XOR가 아니라 `sum(axis=1)`·싱글스레드·공유 고정비용임을 규명, faiss IndexBinaryFlat로 3.6–25.9× 가속.
   matmul 트릭은 Mac에선 이득이나 GB10(ARM)에선 손해임을 실측으로 확인해 폐기(타깃 하드웨어 측정의 중요성)
6. **무인 파이프라인 견고화**: 대용량 임베딩이 다운로드(128스레드)와 스레드 경합으로 중단된 것을 진단,
   워커 수 throttle(16+16)·성공 시에만 완료 플래그·단일 스크립트 기동으로 복구(거짓 완료 신호 차단)

---

## 산출물
- 코드: 해시 레이어/손실/모델, 바이너리 인덱스, 검색 API, 벤치마크 스크립트, 데이터 fetch
- 문서: 데이터·벤치(`DATA_AND_BENCHMARKS`), 로드맵(`ROADMAP`), 선행연구·novelty(`PRIOR_ART_AND_NOVELTY`),
  데이터셋 인벤토리(`DATASETS`), 프로덕션 가이드(`PRODUCTION`)
- 시각화: 옵션별 비교 HTML, 튜닝 전후 비교 HTML, 좌우 분할 속도 데모
- 테스트 + GitHub Actions CI, PR 1건(13 커밋)

---

## 선행연구 대비 위치 (정직한 평가)
- 개별 요소(바이너리 임베딩, cross-modal 해싱, Matryoshka)는 기존 기술
- **"learned + prefix-nested + 1-bit + cross-modal + SigLIP2" 조합**은 출판 논문/제품에서 미발견
  (가장 가까운 CroVCA·QAMA가 양옆에서 bracket) — 좁지만 실재하는 niche로 판단
- 응용: 온디바이스 자연어 사진 검색(오프라인·다국어·경량) — 빅테크/오픈소스 미충족 영역

---

## 비즈니스/제품 관점
- 바이너리 해시의 36× 저장·5.5× 속도 이점을 "수백만 장 온디바이스 사진 검색"으로 연결
- 시장 분석(Apple/Google/immich 등)으로 차별점(오프라인+바이너리+다국어+오픈소스) 도출
- 한계도 명시: 바이너리는 float 대비 정확도 손해, 특정 인물·반려동물은 개인화(few-shot) 필요
