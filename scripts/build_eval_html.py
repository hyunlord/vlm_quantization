"""Generate the comprehensive evaluation-comparison HTML from the two metric JSONs
(eval_all.json + eval_hashing.json). Embeds the data and renders sortable tables,
Matryoshka bit-length curves (SVG), hash-native tables, and code diagnostics.
"""
import json, sys, os

EVAL = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "/tmp/eval_all.json"))
HASH = json.load(open(sys.argv[2] if len(sys.argv) > 2 else "/tmp/eval_hashing_local.json"))
OUT = sys.argv[3] if len(sys.argv) > 3 else "/Users/rexxa/github/vlm_quantization/claudedocs/hash_eval_comparison.html"
SPEED_PATH = sys.argv[4] if len(sys.argv) > 4 else "/tmp/speed_storage.json"
SPEED = json.load(open(SPEED_PATH)) if os.path.exists(SPEED_PATH) else None

# ---- storage (analytical, exact) ----
FLOAT_BYTES = 1152 * 4   # SigLIP2 1152-d float32
STORAGE = []
for b in [8, 16, 32, 64, 128, 256, 512, 1024]:
    by = b // 8
    STORAGE.append({"bits": b, "bytes": by, "ratio": round(FLOAT_BYTES / by),
                    "gb_per_1M": round(by * 1_000_000 / 1e9, 4),
                    "gb_per_100M": round(by * 100_000_000 / 1e9, 2)})

# ---- mix-ratio sweep (1024-bit I2T, COCO test gold) — §4f ----
SWEEP = [  # {label, x(extra:coco multiple), R@1, R@10, MRR, mAP, NDCG}
    {"label": "baseline", "x": 0,   "R@1": 39.78, "R@10": 80.22, "MRR": 53.48, "mAP": 70.07, "NDCG": 76.02},
    {"label": "1:1",      "x": 1,   "R@1": 42.18, "R@10": 80.56, "MRR": 55.28, "mAP": 74.93, "NDCG": 76.86},
    {"label": "1:2",      "x": 2,   "R@1": 42.46, "R@10": 80.48, "MRR": 55.15, "mAP": 74.89, "NDCG": 76.71},
    {"label": "1:3.6",    "x": 3.6, "R@1": 41.50, "R@10": 79.80, "MRR": 54.48, "mAP": 74.96, "NDCG": 76.36},
    {"label": "1:5.3",    "x": 5.3, "R@1": 40.66, "R@10": 79.20, "MRR": 53.78, "mAP": 75.24, "NDCG": 76.23},
    {"label": "1:8.6",    "x": 8.6, "R@1": 40.58, "R@10": 78.18, "MRR": 53.21, "mAP": 75.04, "NDCG": 75.81},
]
SWEEP_FLOAT = {"R@1": 47.68, "R@10": 81.10, "mAP": 66.56}

# ---- out-of-domain generalization (never trained on) — §4g ----
OOD = {
    "Flickr30K (1K, 5-cap)": {
        "float":            {"I2T": [92.9, 99.3, 99.9],  "T2I": [79.02, 94.18, 96.98]},
        "best 1:1":         {"I2T": [82.0, 97.4, 98.6],  "T2I": [67.38, 89.18, 93.2]},
        "best 1:2":         {"I2T": [83.8, 97.2, 98.7],  "T2I": [67.52, 89.24, 93.66]},
        "ft113 (한글ft)":   {"I2T": [80.2, 96.9, 99.0],  "T2I": [66.76, 89.04, 93.72]},
        "ft226 (한글ft)":   {"I2T": [82.5, 96.7, 98.7],  "T2I": [67.4, 89.26, 93.76]},
        "CC12M 403K":       {"I2T": [85.4, 97.5, 99.3],  "T2I": [69.94, 89.8, 93.92]},
        "COCO baseline":    {"I2T": [76.7, 93.8, 97.1],  "T2I": [62.6, 85.66, 90.96]},
    },
    "DOCCI (5K, 1:1 dense)": {
        "float":            {"I2T": [65.38, 88.28, 93.14], "T2I": [67.14, 87.98, 92.58]},
        "best 1:1":         {"I2T": [45.28, 73.14, 82.32], "T2I": [49.88, 75.54, 83.02]},
        "best 1:2":         {"I2T": [46.5, 74.28, 82.96],  "T2I": [49.92, 75.76, 83.98]},
        "ft113 (한글ft)":   {"I2T": [43.8, 71.8, 80.72],   "T2I": [49.14, 74.98, 82.82]},
        "ft226 (한글ft)":   {"I2T": [46.1, 73.18, 82.72],  "T2I": [49.86, 75.72, 83.98]},
        "CC12M 403K":       {"I2T": [47.5, 74.74, 83.26],  "T2I": [50.16, 76.06, 83.58]},
        "COCO baseline":    {"I2T": [29.4, 54.8, 65.66],   "T2I": [32.44, 58.34, 68.74]},
    },
}

# ---- Korean multilingual (EN vs KO T2I, 1024-bit, same 5K COCO test images) ----
KO = [
    {"label": "float ceiling", "EN": 81.58, "KO": 66.06, "gap": 15.52, "KO_MRR": 0.4326, "kind": "float"},
    {"label": "best 1:1 (한글無)", "EN": 80.16, "KO": 65.36, "gap": 14.80, "KO_MRR": 0.3936, "kind": "base"},
    {"label": "best 1:2 (한글無)", "EN": 79.70, "KO": 65.32, "gap": 14.38, "KO_MRR": 0.4015, "kind": "base"},
    {"label": "B: ml226 (1:2 from-scratch)", "EN": 78.76, "KO": 66.60, "gap": 12.16, "KO_MRR": 0.3993, "kind": "B"},
    {"label": "B: ml113 (1:1 from-scratch)", "EN": 79.30, "KO": 65.92, "gap": 13.38, "KO_MRR": 0.4015, "kind": "B"},
    {"label": "A: ft226 (1:2 fine-tune)", "EN": 79.62, "KO": 70.22, "gap": 9.40, "KO_MRR": 0.4417, "kind": "A"},
    {"label": "A': ft113 (1:1 fine-tune)", "EN": 79.98, "KO": 71.10, "gap": 8.88, "KO_MRR": 0.4376, "kind": "best"},
]

# ---- query robustness (perturbations) — §4e ----
ROBUST = {
    "perts": ["typo", "worddrop", "trunc"],
    "stability": {"float": [40.3, 51.8, 38.4], "bit256": [39.6, 49.1, 37.7], "bit1024": [44.1, 52.9, 39.9]},
    "r10_clean": {"float": 82.7, "bit256": 78.7, "bit1024": 81.0},
    "r10_pert": {"float": [57.0, 61.7, 50.3], "bit256": [49.0, 55.3, 45.0], "bit1024": [52.3, 57.7, 46.0]},
}

html = """<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hash 품질 — 전 세팅 종합 벤치마크</title>
<style>
:root{--bg:#0d1117;--card:#161b22;--bd:#30363d;--fg:#e6edf3;--mut:#8b949e;--accent:#58a6ff;--good:#3fb950;--warn:#d29922;--bad:#f85149}
*{box-sizing:border-box}body{background:var(--bg);color:var(--fg);font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:0;padding:24px;max-width:1280px;margin:0 auto}
h1{font-size:24px;margin:0 0 4px}h2{font-size:19px;margin:30px 0 10px;border-bottom:1px solid var(--bd);padding-bottom:6px}h3{font-size:15px;margin:18px 0 8px;color:var(--accent)}
.sub{color:var(--mut);margin:0 0 18px}
table{border-collapse:collapse;width:100%;font-size:12.5px;margin:8px 0}
th,td{border:1px solid var(--bd);padding:5px 8px;text-align:right;white-space:nowrap}
th{background:#1c2230;cursor:pointer;position:sticky;top:0;user-select:none}th:first-child,td:first-child{text-align:left}
tr:nth-child(even) td{background:#11161f}
.float td{background:#1a2740!important;color:#9ecbff;font-weight:600}
.best{color:var(--good);font-weight:700}.worst{color:var(--bad)}
.cap{background:#161b22;border:1px solid var(--bd);border-left:3px solid var(--accent);padding:10px 14px;border-radius:5px;margin:12px 0;color:#c9d1d9}
.cap.good{border-left-color:var(--good)}.cap.warn{border-left-color:var(--warn)}.cap.bad{border-left-color:var(--bad)}
.tog{display:inline-flex;gap:0;margin:6px 0}.tog button{background:#1c2230;color:var(--fg);border:1px solid var(--bd);padding:5px 14px;cursor:pointer}.tog button.on{background:var(--accent);color:#0d1117;font-weight:700}
.mut{color:var(--mut);font-size:12px}
svg{background:#0b0f15;border:1px solid var(--bd);border-radius:6px}
.lg{display:flex;flex-wrap:wrap;gap:10px;margin:6px 0;font-size:12px}.lg span{display:inline-flex;align-items:center;gap:5px}.sw{width:11px;height:11px;border-radius:2px;display:inline-block}
code{background:#1c2230;padding:1px 5px;border-radius:3px;font-size:12px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px}@media(max-width:900px){.grid2{grid-template-columns:1fr}}
</style></head><body>
<h1>Hash 품질 — 전 세팅 종합 벤치마크</h1>
<p class="sub">COCO test 5K · cross-modal 검색 · 학습한 모든 헤드 × Matryoshka 비트 전구간 × 12개 지표 (gold = 정답 기반, agreement@100 폐기). 생성: eval_all.py + eval_hashing.py</p>

<div class="cap warn"><b>왜 이 문서?</b> 이전엔 "개선됐다"만 적었는데, 여기서는 <b>그동안 학습한 세팅 10개를 전부</b>, <b>여러 gold 지표로</b> 평가해 한 표에서 비교합니다. 핵심: <b>지표에 따라 결론이 갈립니다</b> — 정확한 쌍(pair-R@K)이냐 같은 개념(category-mAP)이냐.</div>

<style>
details.gx{margin:8px 0;border:1px solid var(--bd);border-radius:6px;padding:4px 14px;background:#11161f}
details.gx summary{cursor:pointer;font-size:15px;color:var(--accent);padding:6px 0;font-weight:600}
table.gl td,table.gl th{text-align:left;white-space:normal;vertical-align:top;font-size:12.5px;line-height:1.5}
table.gl td:first-child,table.gl th:first-child{white-space:nowrap;color:#e6edf3;font-weight:700}
table.gl th{background:#1c2230}
.k{color:#79c0ff;font-weight:700}.up{color:var(--good)}.dn{color:var(--warn)}
</style>

<h2>① 용어 · 지표 · 항목 — 상세 설명</h2>
<p class="mut">아래 표/그래프를 읽는 데 필요한 모든 용어·지표·세팅 정의입니다. 길어서 접어뒀습니다 — 제목을 클릭해 펼치세요.</p>

<details class="gx" open><summary>1·A. 기본 용어</summary>
<table class="gl"><thead><tr><th>용어</th><th>뜻</th></tr></thead><tbody>
<tr><td>cross-modal</td><td>서로 다른 모달(이미지↔텍스트) 간 검색. 같은 의미의 이미지와 텍스트를 가까운 코드로 만든다.</td></tr>
<tr><td>I2T / T2I</td><td><b>I2T</b>=이미지를 쿼리로 텍스트를 검색 / <b>T2I</b>=텍스트를 쿼리로 이미지를 검색. 두 방향을 따로 측정.</td></tr>
<tr><td>query / gallery(corpus)</td><td>찾는 쪽(query) / 검색 대상 풀(gallery=corpus). 여기서는 COCO test 5,000개.</td></tr>
<tr><td>pair (정답 쌍)</td><td>같은 이미지에서 나온 이미지-캡션 1:1 짝(같은 cocoid). "정확히 그 짝"만 정답으로 보는 엄격 기준.</td></tr>
<tr><td>category relevance</td><td>COCO 80개 물체 카테고리(사람·개·자동차…)를 <b>1개라도 공유</b>하면 "관련"으로 인정하는 느슨·등급(graded) 기준.</td></tr>
<tr><td>float embedding</td><td>SigLIP2가 뽑은 1152차원 <b>실수</b> 벡터. 양자화(비트화) 전 원본 → 성능의 <b>천장(ceiling)</b>.</td></tr>
<tr><td>hash code (binary)</td><td>float을 1비트(+1/−1)들로 압축한 코드. 저장·검색이 수십 배 싸고 빠름. 이 프로젝트의 산출물.</td></tr>
<tr><td>Matryoshka nested code</td><td>한 번 학습으로 8·16·…·1024비트를 <b>"앞에서 잘라"</b> 모두 쓸 수 있게 만든 중첩 코드. 256비트 = 1024비트의 앞 256개.</td></tr>
<tr><td>prefix</td><td>nested code에서 앞 K비트만 취한 것. 그래서 한 헤드가 여러 비트길이를 동시에 제공.</td></tr>
<tr><td>Hamming distance</td><td>두 비트코드에서 <b>다른 비트의 개수</b>. XOR 후 1을 세면(popcount) 됨. 해시 검색의 거리 척도.</td></tr>
<tr><td>continuous vs binary (SignSTE)</td><td>학습 땐 tanh(연속값), 추론 땐 sign(±1 이진). STE(straight-through)로 sign의 기울기를 흘려 학습 가능하게 함.</td></tr>
<tr><td>frozen backbone</td><td>SigLIP2(backbone)는 <b>고정</b>(학습 안 함). 그 위의 작은 hash head만 학습 → 빠르고 broad 지식 보존.</td></tr>
<tr><td>RKD (관계형 distillation)</td><td>float의 <b>샘플 간 거리 구조</b>를 hash가 따라하게 하는 손실. 절대 유사도가 아닌 상대 구조라 InfoNCE와 충돌 안 함.</td></tr>
<tr><td>CroVCA coding-rate</td><td>코드가 한쪽으로 쏠리지 않게 <b>다양성(coding rate)</b>을 키우는 손실.</td></tr>
<tr><td>capped / uncapped</td><td>(혼합비 통제) 추가 데이터(OI/CC12M)를 epoch당 <b>고정량만</b> 쓰면 capped, 전부 쓰면 uncapped. COCO 대비 학습 비중을 맞추려는 장치.</td></tr>
<tr><td>normalization (정규화)</td><td>hash head 입력 float을 L2정규화하느냐(norm) 원본 그대로냐(raw). ablation으로 영향을 분리 측정.</td></tr>
</tbody></table></details>

<details class="gx"><summary>1·B. 지표 상세 — 정의·범위·방향·한계 (3계열 + 진단)</summary>
<p class="mut">방향: <span class="up">↑</span>=높을수록 좋음, <span class="dn">↓</span>=낮을수록 좋음.</p>
<table class="gl"><thead><tr><th>지표</th><th>계열</th><th>정의 / 수식</th><th>범위·방향</th><th>무엇을 보나 / 한계</th></tr></thead><tbody>
<tr><td>R@1 / R@5 / R@10</td><td>pair</td><td>쿼리의 <b>정답 쌍</b>이 상위 K개 안에 있으면 1, 전 쿼리 평균.</td><td>0–100% <span class="up">↑</span></td><td>정확한 짝을 얼마나 상위에 올리나. <i>한계: 정답이 1개라 가정 → "다르지만 유효"는 0점.</i></td></tr>
<tr><td>MRR</td><td>pair</td><td>mean(1 / 정답의 순위).</td><td>0–100% <span class="up">↑</span></td><td>정답이 평균적으로 얼마나 위인가(상위 가중).</td></tr>
<tr><td>MedR</td><td>pair</td><td>정답 순위의 <b>중앙값</b>.</td><td>1–N <span class="dn">↓</span></td><td>쿼리 절반이 이 순위 안에 정답. 낮을수록 좋음.</td></tr>
<tr><td>mAP@100 / @1000</td><td>category</td><td>상위 K에서 Σ(정밀도@k · 관련@k) / (상위 K 내 관련 수).</td><td>0–100% <span class="up">↑</span></td><td>"같은 카테고리"를 상위에 얼마나 모으나. <i>한계: relevance가 느슨(쿼리당 평균 32% 관련) → 절대값이 높게 나옴, 상대비교용.</i></td></tr>
<tr><td>NDCG@10</td><td>category</td><td>Σ(공유카테고리수 / log₂(순위+1)) ÷ 이상적 정렬값(IDCG).</td><td>0–100% <span class="up">↑</span></td><td>더 많이 겹치는 항목을 더 위에 두는가(<b>graded</b> 보상).</td></tr>
<tr><td>P@10</td><td>category</td><td>상위 10개 중 관련(카테고리 공유) 비율.</td><td>0–100% <span class="up">↑</span></td><td>첫 화면(10개)의 정밀도.</td></tr>
<tr><td>P@H≤2</td><td>hash-native</td><td>Hamming 거리 ≤2로 검색된 것 중 관련 비율.</td><td>0–100% <span class="up">↑</span></td><td>해시 버킷(반경2)의 <b>순도</b>. <i>float은 측정 불가(반경 개념 없음).</i></td></tr>
<tr><td>R@H≤2</td><td>hash-native</td><td>반경2 안의 관련 수 ÷ 전체 관련 수.</td><td>0–100% <span class="up">↑</span></td><td>버킷 룩업의 <b>커버리지</b>. <i>긴 코드일수록 0에 수렴(코드가 흩어져 반경2가 텅 빔).</i></td></tr>
<tr><td>cover@H2</td><td>hash-native</td><td>반경2 안에 1개라도 있는 쿼리 비율.</td><td>0–100% <span class="up">↑</span></td><td><b>O(1) 버킷 룩업이 작동하는 쿼리 비율.</b> 0%면 선형 스캔 강제.</td></tr>
<tr><td>bit_balance</td><td>진단</td><td>mean_k | (비트 k의 배치 평균) |.</td><td>0–1 <span class="dn">↓</span>(0=균형)</td><td>각 비트가 +/− 50:50인가. 쏠리면 그 비트는 정보 0.</td></tr>
<tr><td>bit_indep</td><td>진단</td><td>‖ HᵀH/N − I ‖_F / K.</td><td><span class="dn">↓</span>(0=독립)</td><td>비트끼리 독립인가. 상관 높으면 실효 코드길이↓.</td></tr>
<tr><td>entropy</td><td>진단</td><td>비트당 섀넌 엔트로피 H(p_k)의 평균.</td><td>0–1 <span class="up">↑</span>(1=최대)</td><td>비트당 실제 정보량.</td></tr>
<tr><td>quant_err</td><td>진단</td><td>mean(1 − |연속코드|).</td><td>0–1 <span class="dn">↓</span>(0=확실)</td><td>sign으로 얼마나 확실히 ±1 갈렸나. 높으면 0 근처 애매비트 많음.</td></tr>
<tr><td><s>agreement@100</s> (폐기)</td><td>—</td><td>hash top100 ∩ float top100 / 100.</td><td>—</td><td><b>폐기.</b> float=정답 아닌 proxy + 유효다양성 페널티 + top-100 경계노이즈 + distillation과 순환. gold로 재측정하니 데이터 효과를 <b>거꾸로</b> 봤음이 드러남.</td></tr>
</tbody></table></details>

<details class="gx"><summary>1·C. 옵션(세팅) 설명 — 마스터표의 각 행</summary>
<table class="gl"><thead><tr><th>세팅</th><th>학습 데이터</th><th>특징</th></tr></thead><tbody>
<tr><td>float (ceiling)</td><td>—</td><td>양자화 안 한 SigLIP2 원본 임베딩 코사인 검색. 모든 hash가 도달하려는 <b>성능 천장</b>.</td></tr>
<tr><td>bit-old demo</td><td>COCO + 한국어, full-image</td><td>옛 데모 모델. hidden <b>768</b>, bits→<b>1024</b>. 레시피가 달라 직접 비교 시 용량차 주의.</td></tr>
<tr><td>COCO baseline</td><td>COCO 113K</td><td>기본(정규화 입력). 추가 데이터·distillation 없음. = 정규화 ablation의 norm 쪽.</td></tr>
<tr><td>COCO + CroVCA</td><td>COCO 113K</td><td>+ coding-rate 다양성 손실만 추가.</td></tr>
<tr><td>CC12M 403K</td><td>COCO + CC12M 40만쌍</td><td>clean 캡션 대규모 + RKD + CroVCA. 데이터 확대의 핵심.</td></tr>
<tr><td>974K uncapped</td><td>COCO + CC12M+OI 97만</td><td>전 데이터 결합, epoch당 전부 사용.</td></tr>
<tr><td>974K capped</td><td>〃 (epoch당 40만으로 제한)</td><td>COCO:추가데이터 step 비율을 고정(혼합비 통제 실험).</td></tr>
<tr><td>raw-COCO</td><td>COCO 113K, 비정규화 입력</td><td>정규화 ablation의 raw 쪽. norm(=baseline)과 비교용.</td></tr>
</tbody></table>
<p class="mut">전부 bits→1024로 학습(데이터 확대분은 hidden 384, demo만 768). 옛 256-max 결과는 ledger(`hash_quality_ledger.md`)에 보존.</p></details>

<details class="gx"><summary>1·D. 각 표·그래프 읽는 법</summary>
<table class="gl"><thead><tr><th>항목</th><th>읽는 법</th></tr></thead><tbody>
<tr><td>② 마스터표</td><td><b>비트 버튼</b>으로 코드 길이 선택, <b>I2T/T2I</b> 토글로 방향, <b>열 머리글 클릭</b>으로 정렬. <span class="best">초록</span>=hash 중 그 열 최고, <span style="color:#9ecbff">파랑 행</span>=float 천장, <b>–</b>=그 비트로 학습 안 된 세팅.</td></tr>
<tr><td>③ Matryoshka 곡선</td><td>비트길이↑에 따른 R@10·mAP@1000 변화. <b>점선</b>=float 천장. 색=세팅(아래 범례). 256 이후 평탄 = 한계효용.</td></tr>
<tr><td>④ hash-native 표</td><td>비트별 P@H2 / R@H2 / cover. <span class="best">초록 cover</span>=룩업 잘 됨(≈100%), <span class="worst">빨강 cover</span>=0%(반경2 안 텅 빔 → 버킷 룩업 불가).</td></tr>
<tr><td>⑤ 진단 표</td><td>최장 비트 코드의 품질 4종(균형·독립·엔트로피·양자화오차). <span class="best">초록</span>=그 열 최고.</td></tr>
</tbody></table></details>

<h2>② ★ 마스터 비교표 — 비트길이 선택</h2>
<div class="tog" id="dir1"><button data-d="I2T" class="on">I2T (이미지→텍스트)</button><button data-d="T2I">T2I (텍스트→이미지)</button></div>
<div class="tog" id="bitsel" style="margin-left:8px"></div>
<p class="mut">비트 버튼/방향 토글 + 열 머리글 클릭=정렬. 초록=해당 열 최고(hash 중), 파랑행=float 천장. <b>–</b> = 그 세팅은 해당 비트로 학습 안 됨(256-max). float은 비트 무관(임베딩).</p>
<div id="master"></div>

<h2>③ Matryoshka 비트길이 곡선</h2>
<p class="mut">짧은 코드→긴 코드. 모든 세팅이 비슷한 곡선; 256 이후 한계효용 급감. demo만 512/1024 보유.</p>
<div class="grid2">
<div><h3>pair R@10 (I2T) vs 비트</h3><div id="curveR"></div></div>
<div><h3>category mAP@1000 (I2T) vs 비트</h3><div id="curveM"></div></div>
</div>
<div id="legend" class="lg"></div>

<h2>④ Hash-native — Hamming 반경 2 룩업 (I2T)</h2>
<p class="mut">진짜 해시 버킷 룩업이 되는지. float은 이 지표 자체가 없음(반경 개념 없음).</p>
<div id="hball"></div>
<div class="cap bad"><b>핵심: 반경2 룩업은 ≥128bit에서 완전 붕괴(cover 0%).</b> 32bit도 cover ~20%뿐. 즉 256bit 배포에선 <b>O(1) 버킷 룩업 불가 → 선형 Hamming 스캔(faiss IndexBinaryFlat)이 필수.</b> 짧은 코드(8bit)는 cover 100%지만 R@H2가 ~30%로 정밀도-재현율 trade.</div>

<h2>⑤ 코드 품질 진단 — 256-bit (이미지 코드)</h2>
<div id="diag"></div>
<div class="cap warn"><b>CC12M/데이터 확대는 bit_balance를 악화</b>(COCO ~0.03 → CC12M ~0.12). balance 손실이 COCO엔 잘 듣지만 넓은 데이터에선 일부 비트가 치우침(entropy도 0.999→0.98). 즉 semantic mAP↑의 이면에 코드 균형↓ 비용이 있음.</div>

<h2>⑥ 저장용량 비교 (비트길이별 · 분석적·정확)</h2>
<p class="mut">벡터당 바이트 = 비트/8. float 기준 = SigLIP2 1152-d × 4B = <b>4,608 B</b>. 비율 = float ÷ 코드. 코퍼스 크기별 총량도 표기.</p>
<div id="storage"></div>
<div class="cap good"><b>해시의 핵심 이점 ①.</b> 256-bit = 32B로 float 대비 <b>144× 작음</b>(1024-bit도 128B, 36×). 1억 벡터 코퍼스: float ≈ 461 GB vs 256-bit ≈ <b>3.2 GB</b>. 메모리 상주·인덱스 비용을 좌우.</div>

<h2>⑦ 검색속도 비교 (faiss 이진 vs float · 실측)</h2>
<p class="mut" id="speedmeta"></p>
<div class="tog" id="ssel"></div>
<div id="speed"></div>
<div class="cap good"><b>해시의 핵심 이점 ②.</b> faiss <code>IndexBinaryFlat</code>(HW popcount)는 float 코사인 대비 <b>수십 배 빠름</b>. 비트가 길수록 Hamming 비용↑(이득↓)이지만 여전히 압도적. numpy popcount는 느려서 faiss가 실배포 경로. ⚠ 학습 중 측정이라 절대 ms는 ±, 상대 배율은 견고.</div>

<h2>⑧ 혼합비 스윕 — in-domain sweet spot (COCO test, 1024-bit, I2T)</h2>
<p class="mut">결합 974K 풀 고정, COCO:추가 데이터 step 비율만 변경. pair-R은 뒤집힌 U자(1:1~1:2 정점), category-mAP는 추가데이터만 있으면 포화.</p>
<div class="grid2"><div><h3>pair R@1 vs 혼합비</h3><div id="sweepR1"></div></div><div><h3>category mAP@1000 vs 혼합비</h3><div id="sweepM"></div></div></div>
<div id="sweepTbl"></div>
<div class="cap good"><b>sweet spot = 1:1~1:2</b>: pair-R@1 42.2~42.5 (baseline 39.8·uncapped 40.6 능가) + mAP 최대. <b>데이터의 in-domain 레버는 "양"이 아니라 "혼합비".</b></div>

<h2>⑨ 도메인-외 일반화 — Flickr30K · DOCCI (학습 안 한 벤치)</h2>
<p class="mut">학습=COCO+CC12M+OI뿐. 미지 데이터셋 retrieval로 일반화 측정. 막대=R@1, 회색 점선=float 천장.</p>
<div id="ood"></div>
<div class="cap good"><b>데이터(CC12M)가 도메인-외 일반화의 지배적 레버</b>: baseline→CC12M I2T R@1 Flickr +8.7pt / DOCCI +18.1pt (도메인 차 클수록 이득↑). <b>CC12M ≥ best-1:2 (두 벤치 모두)</b> — in-domain best-1:2 우위와 대조. bit가 float R@10의 89%(DOCCI)~99%(Flickr) 유지.</div>

<h2>⑩ 쿼리 robustness — 섭동(오타/단어드롭/truncation)</h2>
<p class="mut">clean 쿼리 vs 섭동 쿼리. Stability@10=결과셋 안정성(↑robust), R@10 retention=정답 잔류율.</p>
<div id="robust"></div>
<div class="cap warn"><b>"비트가 더 견고" 가설 대체로 반증</b>: bit1024가 결과셋 안정성(Stability)은 근소 우위지만, 정답 유지력(R@10 retention)은 float 우위. 무승부~float 약우위. 데모의 "반지의 제왕"은 양자화가 아닌 다국어 아티팩트.</div>

<h2>⑫ 한국어 멀티링구얼 — 영어 vs 한국어 · fine-tune vs from-scratch</h2>
<p class="sub">같은 5K COCO test 이미지, T2I, 1024-bit. 한국어 캡션 = coco_ko (test split, train 누수 없음). 학습 데이터에 한국어를 넣는 두 방식 비교 — B: from-scratch 혼합(COCO+KO 동시 학습), A: 기존 best 헤드를 한국어로 fine-tune.</p>
<div id="koTable"></div>
<div id="koChart"></div>
<div class="cap good"><b>① 한국어 이미 작동</b> — 학습에 한국어가 0개인데 KO R@10이 65%(best 헤드)에 달함. SigLIP2가 멀티링구얼이고 텍스트 헤드를 영·한이 공유하기 때문.</div>
<div class="cap good"><b>② fine-tune ≫ from-scratch</b> — 같은 best 1:1 베이스에서 A'(ft113) KO 71.10 vs B(ml113) 65.92로 +5.18pt. from-scratch는 한국어가 풀의 10%로 희석되고, fine-tune은 한국어에 100% 집중하기 때문.</div>
<div class="cap good"><b>③ ft113 = 배포 후보</b> — KO 71.10 (float 66.06의 107.6% 추월) + EN 79.98 (영어 거의 손실 없음, in-domain −0.18pt) + Flickr/DOCCI OOD 회귀 미미 (⑨ 표 참조).</div>
<div class="cap"><b>④ 교훈</b> — 소수 언어/도메인 강화는 from-scratch 혼합보다 fine-tune이 압도적으로 효율적이다.</div>

<h2>⑪ 종합 결론</h2>
<div id="concl"></div>

<script>
const EVAL=__EVAL__;const HASH=__HASH__;const STORAGE=__STORAGE__;const SPEED=__SPEED__;
const SWEEP=__SWEEP__;const SWEEP_FLOAT=__SWEEPFLOAT__;const OOD=__OOD__;const ROBUST=__ROBUST__;const KO=__KO__;
const HEADS=Object.keys(EVAL).filter(k=>k!=="float (ceiling)");
const ORDER=Object.keys(EVAL).filter(k=>k!=="float (ceiling)");  // auto from data (robust to label changes)
const PALETTE=["#f85149","#8b949e","#d29922","#3fb950","#58a6ff","#bc8cff","#56d4dd","#ff9bce","#79c0ff","#e3b341"];
const COLORS=ORDER.map((_,i)=>PALETTE[i%PALETTE.length]);
const PMETR=["R@1","R@5","R@10","MRR","MedR","mAP@100","mAP@1000","NDCG@10","P@10"];
function bitsOf(h){return Object.keys(EVAL[h]).filter(b=>b!=="float").sort((a,b)=>+a-+b);}
function cell(h,bit,dir,m){const o=(h==="float (ceiling)")?EVAL[h]["float"]:EVAL[h][bit];return o&&o[dir]?o[dir][m]:null;}

// ---- master table ----
const ALLBITS=["8","16","32","64","128","256","512","1024"];
let curDir="I2T",curBit="1024",sortM="mAP@1000",sortAsc=false;
document.getElementById('bitsel').innerHTML=ALLBITS.map(b=>`<button data-b="${b}" class="${b==="1024"?"on":""}">${b}b</button>`).join('');
function master(){
  const data=[{h:"float (ceiling)",fl:1,v:Object.fromEntries(PMETR.map(m=>[m,cell("float (ceiling)","float",curDir,m)]))}]
    .concat(ORDER.map(h=>({h,fl:0,v:Object.fromEntries(PMETR.map(m=>[m,cell(h,curBit,curDir,m)]))})));
  // best per metric among hash rows that HAVE this bit
  const best={};PMETR.forEach(m=>{const vals=data.filter(d=>!d.fl&&d.v[m]!=null).map(d=>d.v[m]);best[m]=vals.length?((m==="MedR")?Math.min(...vals):Math.max(...vals)):null;});
  const hashRows=data.filter(d=>!d.fl).sort((a,b)=>{const x=a.v[sortM],y=b.v[sortM];if(x==null)return 1;if(y==null)return -1;return sortAsc?x-y:y-x;});
  const ordered=[data[0]].concat(hashRows);
  let h=`<table><thead><tr><th>세팅 @ ${curBit}-bit</th>`+PMETR.map(m=>`<th data-m="${m}">${m}</th>`).join('')+'</tr></thead><tbody>';
  ordered.forEach(d=>{
    h+=`<tr class="${d.fl?'float':''}"><td>${d.h}</td>`+PMETR.map(m=>{
      const v=d.v[m];let c='';if(!d.fl&&v!=null&&v===best[m])c='best';
      return `<td class="${c}">${v==null?'–':v}</td>`;}).join('')+'</tr>';
  });
  h+='</tbody></table>';
  document.getElementById('master').innerHTML=h;
  document.querySelectorAll('#master th[data-m]').forEach(th=>th.onclick=()=>{const m=th.dataset.m;if(sortM===m)sortAsc=!sortAsc;else{sortM=m;sortAsc=(m==="MedR");}master();});
}
document.querySelectorAll('#dir1 button').forEach(b=>b.onclick=()=>{document.querySelectorAll('#dir1 button').forEach(x=>x.classList.remove('on'));b.classList.add('on');curDir=b.dataset.d;master();});
document.querySelectorAll('#bitsel button').forEach(b=>b.onclick=()=>{document.querySelectorAll('#bitsel button').forEach(x=>x.classList.remove('on'));b.classList.add('on');curBit=b.dataset.b;master();});

// ---- SVG curve ----
const BITSX=[8,16,32,64,128,256,512,1024];
function curve(elt,metric,dir){
  const W=560,H=300,pl=44,pr=12,pt=14,pb=28;
  const ys=[];HEADS.concat(["float (ceiling)"]).forEach(h=>{});
  let vmin=1e9,vmax=-1e9;
  const series=ORDER.map((h,i)=>{const pts=bitsOf(h).map(b=>[+b,cell(h,b,dir,metric)]).filter(p=>p[1]!=null&&BITSX.indexOf(p[0])>=0);pts.forEach(p=>{vmin=Math.min(vmin,p[1]);vmax=Math.max(vmax,p[1]);});return{h,i,pts};});
  const fl=cell("float (ceiling)","float",dir,metric);vmax=Math.max(vmax,fl);vmin=Math.min(vmin,fl);
  vmin=Math.floor(vmin/10)*10;vmax=Math.ceil(vmax/10)*10;
  const xi=b=>pl+(BITSX.indexOf(b))/(BITSX.length-1)*(W-pl-pr);
  const yi=v=>pt+(1-(v-vmin)/(vmax-vmin))*(H-pt-pb);
  let s=`<svg viewBox="0 0 ${W} ${H}" width="100%">`;
  for(let g=vmin;g<=vmax;g+=10){s+=`<line x1="${pl}" y1="${yi(g)}" x2="${W-pr}" y2="${yi(g)}" stroke="#21262d"/><text x="${pl-6}" y="${yi(g)+3}" fill="#8b949e" font-size="9" text-anchor="end">${g}</text>`;}
  BITSX.forEach(b=>{s+=`<text x="${xi(b)}" y="${H-10}" fill="#8b949e" font-size="9" text-anchor="middle">${b}</text>`;});
  s+=`<line x1="${pl}" y1="${yi(fl)}" x2="${W-pr}" y2="${yi(fl)}" stroke="#9ecbff" stroke-dasharray="4 3" stroke-width="1.3"/><text x="${W-pr}" y="${yi(fl)-4}" fill="#9ecbff" font-size="9" text-anchor="end">float ${fl}</text>`;
  series.forEach(se=>{if(!se.pts.length)return;const d=se.pts.map((p,k)=>(k?'L':'M')+xi(p[0])+' '+yi(p[1])).join(' ');s+=`<path d="${d}" fill="none" stroke="${COLORS[se.i]}" stroke-width="1.8"/>`;se.pts.forEach(p=>{s+=`<circle cx="${xi(p[0])}" cy="${yi(p[1])}" r="2.3" fill="${COLORS[se.i]}"/>`;});});
  s+='</svg>';elt.innerHTML=s;
}
curve(document.getElementById('curveR'),"R@10","I2T");
curve(document.getElementById('curveM'),"mAP@1000","I2T");
document.getElementById('legend').innerHTML=ORDER.map((h,i)=>`<span><span class="sw" style="background:${COLORS[i]}"></span>${h}</span>`).join('')+'<span><span class="sw" style="background:#9ecbff"></span>float (점선)</span>';

// ---- hash-native table ----
function hballTbl(){
  const bits=["8","16","32","64","128","256"];
  let h='<table><thead><tr><th>세팅</th>'+bits.map(b=>`<th>${b}b<br><span class="mut">P@H2 / R@H2 / cov</span></th>`).join('')+'</tr></thead><tbody>';
  ORDER.forEach(hd=>{h+=`<tr><td>${hd}</td>`+bits.map(b=>{const o=HASH[hd]&&HASH[hd][b]?HASH[hd][b]["I2T"]:null;if(!o)return '<td>–</td>';const cov=o["cover@H2"];const cc=cov==0?'worst':(cov>=99?'best':'');return `<td class="${cc}">${o["P@H2"]} / ${o["R@H2"]} / ${cov}%</td>`;}).join('')+'</tr>';});
  h+='</tbody></table>';document.getElementById('hball').innerHTML=h;
}
hballTbl();

// ---- diagnostics 256 ----
function diagTbl(){
  const cols=["bit_balance","bit_indep","entropy","quant_err"];
  const good={bit_balance:"min",bit_indep:"min",entropy:"max",quant_err:"min"};
  const data=ORDER.map(h=>({h,d:(HASH[h]&&HASH[h]["256"])?HASH[h]["256"]["diag_img"]:null})).filter(x=>x.d);
  const best={};cols.forEach(c=>{const v=data.map(x=>x.d[c]);best[c]=good[c]==="min"?Math.min(...v):Math.max(...v);});
  let h='<table><thead><tr><th>세팅 (256b)</th>'+cols.map(c=>`<th>${c}</th>`).join('')+'</tr></thead><tbody>';
  data.forEach(x=>{h+=`<tr><td>${x.h}</td>`+cols.map(c=>`<td class="${x.d[c]===best[c]?'best':''}">${x.d[c]}</td>`).join('')+'</tr>';});
  h+='</tbody></table>';document.getElementById('diag').innerHTML=h;
}
diagTbl();

// ---- conclusions (computed) ----
function g(h,b,d,m){return cell(h,b,d,m);}
const B1="1024";
const concl=`
<div class="cap good"><b>1. 정확한 쌍(pair-R@K) @${B1}b — 데이터·정규화는 노이즈, 실격차는 hash↔float.</b> I2T R@10 — COCO baseline ${g("COCO baseline",B1,"I2T","R@10")} / CC12M403K ${g("CC12M 403K",B1,"I2T","R@10")} / raw-input ${g("COCO raw-input",B1,"I2T","R@10")} / demo ${g("bit-old demo (h768)",B1,"I2T","R@10")} — vs float 천장 ${g("float (ceiling)","float","I2T","R@10")}.</div>
<div class="cap good"><b>2. 같은 개념(category-mAP@1000) @${B1}b — 데이터가 레버.</b> baseline ${g("COCO baseline",B1,"I2T","mAP@1000")} · CroVCA ${g("COCO + CroVCA",B1,"I2T","mAP@1000")} · CC12M403K ${g("CC12M 403K",B1,"I2T","mAP@1000")} · 974K uncapped ${g("974K uncapped",B1,"I2T","mAP@1000")} · 974K capped ${g("974K capped",B1,"I2T","mAP@1000")} (float ${g("float (ceiling)","float","I2T","mAP@1000")}).</div>
<div class="cap warn"><b>3. Trade-off (정확한 쌍 ↔ 개념).</b> 데이터 확대는 category-mAP를 올리는 경향이나 pair-R@10·bit_balance는 함께 오르지 않을 수 있음 — ①(R@10)과 ②(mAP) 수치를 직접 대조. "정확한 쌍" vs "개념 검색" 목적이 선택을 가른다.</div>
<div class="cap"><b>4. 비트길이 (길수록 pair-R↑, 한계효용 감소).</b> demo I2T R@10: 256 ${g("bit-old demo (h768)","256","I2T","R@10")} → 512 ${g("bit-old demo (h768)","512","I2T","R@10")} → 1024 ${g("bit-old demo (h768)","1024","I2T","R@10")}. category-mAP@1000은 256 부근서 포화(256 ${g("bit-old demo (h768)","256","I2T","mAP@1000")} → 1024 ${g("bit-old demo (h768)","1024","I2T","mAP@1000")}).</div>
<div class="cap bad"><b>5. Hamming 룩업 한계.</b> ≥128b는 반경2 cover 0% → O(1) 버킷 룩업 불가, 선형 Hamming 스캔(faiss) 필수. 8b만 cover 100%(대신 R@H2 ~30%).</div>
<div class="cap"><b>6. agreement@100 폐기.</b> 정규화 차가 agreement 2배였으나 모든 gold에서 무영향 → artifact 확정.</div>
<div class="cap"><b>7. 용량 주석.</b> demo만 hidden 768, 나머지는 384(튜닝 HP). 512/1024 절대비교 시 demo가 용량으로 유리할 수 있음 — 데이터 효과는 같은 384 내(baseline↔CC12M)에서 비교가 공정.</div>
<div class="cap good"><b>8. 혼합비 sweet spot (§⑧).</b> in-domain pair-R은 COCO:추가 <b>1:1~1:2에서 정점</b>(R@1 42.5, baseline 39.8·uncapped 40.6 능가). <b>데이터의 in-domain 레버는 "양"이 아니라 "혼합비".</b></div>
<div class="cap good"><b>9. 도메인-외 일반화 (§⑨).</b> CC12M이 미지 벤치(Flickr/DOCCI)에서 baseline 대비 R@1 <b>+8.7~18.1pt</b> — bit가 float R@10의 89~99% 유지. <b>2축: 혼합비=in-domain 정밀도 / CC12M 데이터=out-of-domain 일반화.</b> 배포 도메인이 다양/미지면 CC12M 403K, COCO-유사면 best 1:2.</div>`;
document.getElementById('concl').innerHTML=concl;

// ---- storage table ----
(function(){
  let h='<table><thead><tr><th>코드</th><th>벡터당</th><th>float 대비</th><th>1M 코퍼스</th><th>100M 코퍼스</th></tr></thead><tbody>';
  h+='<tr class="float"><td>float (1152-d)</td><td>4,608 B</td><td>1×</td><td>4.61 GB</td><td>461 GB</td></tr>';
  STORAGE.forEach(s=>{h+=`<tr><td>${s.bits}-bit</td><td>${s.bytes} B</td><td class="best">${s.ratio}×</td><td>${s.gb_per_1M} GB</td><td>${s.gb_per_100M} GB</td></tr>`;});
  document.getElementById('storage').innerHTML=h+'</tbody></table>';
})();

// ---- speed table ----
(function(){
  const el=document.getElementById('speed');
  if(!SPEED||!SPEED.bench){el.innerHTML='<p class="mut">속도 벤치 데이터 없음.</p>';return;}
  const sizes=Object.keys(SPEED.bench);let curN=sizes[sizes.length-1];
  document.getElementById('speedmeta').innerHTML=`측정: ${SPEED.meta.device} · top-${SPEED.meta.K} · 쿼리 ${SPEED.meta.Q}회 평균 · ms/쿼리. 코퍼스 크기 선택 ↓`;
  document.getElementById('ssel').innerHTML=sizes.map(n=>`<button data-n="${n}" class="${n===curN?'on':''}">${(+n).toLocaleString()} 벡터</button>`).join('');
  function draw(){
    const bb=SPEED.bench[curN];
    let h='<table><thead><tr><th>코드</th><th>float 코사인</th><th>bit numpy popcount</th><th>bit faiss</th><th>faiss vs float</th></tr></thead><tbody>';
    Object.keys(bb).forEach(bits=>{const r=bb[bits];
      h+=`<tr><td>${bits}-bit</td><td>${r.float_ms} ms</td><td>${r.bit_np_ms} ms</td><td class="best">${r.bit_faiss_ms} ms</td><td class="best">×${r.speedup_faiss_vs_float}</td></tr>`;});
    el.innerHTML=h+'</tbody></table>';
  }
  document.querySelectorAll('#ssel button').forEach(b=>b.onclick=()=>{document.querySelectorAll('#ssel button').forEach(x=>x.classList.remove('on'));b.classList.add('on');curN=b.dataset.n;draw();});
  draw();
})();
// ---- mix-ratio sweep ----
(function(){
  const W=540,H=250,pl=42,pr=14,pt=14,pb=32, xmax=Math.max(...SWEEP.map(s=>s.x));
  function line(elt,key,fv,lo,hi){
    const xi=x=>pl+(x/xmax)*(W-pl-pr), yi=v=>pt+(1-(v-lo)/(hi-lo))*(H-pt-pb);
    let s=`<svg viewBox="0 0 ${W} ${H}" width="100%">`;
    for(let g=lo;g<=hi+0.01;g+=(hi-lo)/5){s+=`<line x1="${pl}" y1="${yi(g)}" x2="${W-pr}" y2="${yi(g)}" stroke="#21262d"/><text x="${pl-5}" y="${yi(g)+3}" fill="#8b949e" font-size="9" text-anchor="end">${g.toFixed(0)}</text>`;}
    SWEEP.forEach(p=>s+=`<text x="${xi(p.x)}" y="${H-11}" fill="#8b949e" font-size="9" text-anchor="middle">${p.label}</text>`);
    if(fv!=null)s+=`<line x1="${pl}" y1="${yi(fv)}" x2="${W-pr}" y2="${yi(fv)}" stroke="#9ecbff" stroke-dasharray="4 3"/><text x="${W-pr}" y="${yi(fv)-3}" fill="#9ecbff" font-size="9" text-anchor="end">float ${fv}</text>`;
    s+=`<path d="${SWEEP.map((p,k)=>(k?'L':'M')+xi(p.x)+' '+yi(p[key])).join(' ')}" fill="none" stroke="#3fb950" stroke-width="2"/>`;
    SWEEP.forEach(p=>{const b=(key==='R@1'&&(p.label==='1:1'||p.label==='1:2'));s+=`<circle cx="${xi(p.x)}" cy="${yi(p[key])}" r="${b?4:2.5}" fill="${b?'#f0d000':'#3fb950'}"/>`;});
    elt.innerHTML=s+'</svg>';
  }
  line(document.getElementById('sweepR1'),'R@1',SWEEP_FLOAT['R@1'],38,50);
  line(document.getElementById('sweepM'),'mAP',SWEEP_FLOAT['mAP'],64,78);
  let h='<table><thead><tr><th>COCO:추가</th><th>R@1</th><th>R@10</th><th>MRR</th><th>mAP@1k</th><th>NDCG10</th></tr></thead><tbody>';
  h+=`<tr class="float"><td>float (천장)</td><td>${SWEEP_FLOAT['R@1']}</td><td>${SWEEP_FLOAT['R@10']}</td><td>-</td><td>${SWEEP_FLOAT['mAP']}</td><td>-</td></tr>`;
  SWEEP.forEach(p=>{const on=(p.label==='1:1'||p.label==='1:2');h+=`<tr><td>${on?'★ ':''}${p.label}</td><td class="${on?'best':''}">${p['R@1']}</td><td>${p['R@10']}</td><td>${p['MRR']}</td><td>${p['mAP']}</td><td>${p['NDCG']}</td></tr>`;});
  document.getElementById('sweepTbl').innerHTML=h+'</tbody></table>';
})();
// ---- OOD ----
(function(){
  let h='';
  Object.keys(OOD).forEach(ds=>{
    const D=OOD[ds], heads=Object.keys(D), fv=D['float']['I2T'][0];
    h+=`<h3>${ds}</h3><table><thead><tr><th>세팅</th><th>I2T R@1 / R@5 / R@10</th><th>T2I R@1 / R@5 / R@10</th></tr></thead><tbody>`;
    heads.forEach(hd=>{h+=`<tr class="${hd==='float'?'float':''}"><td>${hd}</td><td>${D[hd]['I2T'].join(' / ')}</td><td>${D[hd]['T2I'].join(' / ')}</td></tr>`;});
    h+='</tbody></table><div style="margin:4px 0 16px">';
    heads.filter(x=>x!=='float').forEach(hd=>{const v=D[hd]['I2T'][0],pct=v/fv*100,col=hd.includes('CC12M')?'#3fb950':hd.includes('best')?'#58a6ff':'#8b949e';
      h+=`<div style="display:flex;align-items:center;gap:8px;margin:2px 0"><span style="width:120px;font-size:11px">${hd}</span><div style="flex:1;background:#0b0f15;border-radius:3px;height:15px;position:relative"><div style="width:${pct}%;background:${col};height:100%;border-radius:3px"></div></div><span style="width:120px;font-size:11px;color:${col}">${v} (${Math.round(pct)}% of float)</span></div>`;});
    h+=`<div class="mut">막대 = I2T R@1 (float 천장 ${fv})</div></div>`;
  });
  document.getElementById('ood').innerHTML=h;
})();
// ---- robustness ----
(function(){
  const P=ROBUST.perts, MS=['float','bit256','bit1024'];
  let h='<table><thead><tr><th>Stability@10 (↑robust)</th>'+P.map(p=>`<th>${p}</th>`).join('')+'</tr></thead><tbody>';
  MS.forEach(m=>{const b=m==='bit1024';h+=`<tr><td>${m}</td>`+ROBUST.stability[m].map(v=>`<td class="${b?'best':''}">${v}%</td>`).join('')+'</tr>';});
  h+='</tbody></table><table style="margin-top:8px"><thead><tr><th>R@10 clean→섭동 (retention)</th><th>clean</th>'+P.map(p=>`<th>${p}</th>`).join('')+'</tr></thead><tbody>';
  MS.forEach(m=>{const c=ROBUST.r10_clean[m],b=m==='float';h+=`<tr><td>${m}</td><td>${c}</td>`+ROBUST.r10_pert[m].map(v=>`<td class="${b?'best':''}">${v} (${Math.round(v/c*100)}%)</td>`).join('')+'</tr>';});
  document.getElementById('robust').innerHTML=h+'</tbody></table>';
})();
// ---- Korean multilingual table + chart ----
(function(){
  // table
  var cols=["헤드","EN R@10","KO R@10","gap@10","KO MRR"];
  var th='<table><thead><tr>'+cols.map(function(c){return '<th>'+c+'</th>';}).join('')+'</tr></thead><tbody>';
  KO.forEach(function(r){
    var fl=r.kind==="float", best=r.kind==="best";
    var rowcls=fl?' class="float"':'';
    var koCell=best?'<td class="best">'+r.KO+'</td>':'<td>'+r.KO+'</td>';
    th+='<tr'+rowcls+'><td>'+r.label+'</td><td>'+r.EN+'</td>'+koCell+'<td>'+r.gap+'</td><td>'+(r.KO_MRR*100).toFixed(2)+'</td></tr>';
  });
  document.getElementById('koTable').innerHTML=th+'</tbody></table>';

  // chart — grouped bar: EN(blue) vs KO(green) per head
  var W=620,H=280,pl=44,pr=14,pt=18,pb=52;
  var vals=[];KO.forEach(function(r){vals.push(r.EN,r.KO);});
  var vmin=Math.floor(Math.min.apply(null,vals)/10)*10;
  var vmax=Math.ceil(Math.max.apply(null,vals)/10)*10;
  var N=KO.length, bw=Math.floor((W-pl-pr)/(N*2+N+1)), gap=Math.floor(bw*0.5);
  var xi=function(i,j){return pl+(gap*(i+1))+(i*2+j)*bw;};
  var yi=function(v){return pt+(1-(v-vmin)/(vmax-vmin))*(H-pt-pb);};
  var s='<svg viewBox="0 0 '+W+' '+H+'" width="100%">';
  for(var g=vmin;g<=vmax;g+=10){s+='<line x1="'+pl+'" y1="'+yi(g)+'" x2="'+(W-pr)+'" y2="'+yi(g)+'" stroke="#21262d"/><text x="'+(pl-4)+'" y="'+(yi(g)+3)+'" fill="#8b949e" font-size="9" text-anchor="end">'+g+'</text>';}
  KO.forEach(function(r,i){
    var xEN=xi(i,0), xKO=xi(i,1);
    var yEN=yi(r.EN), yKO=yi(r.KO), yBase=yi(vmin);
    var hEN=yBase-yEN, hKO=yBase-yKO;
    s+='<rect x="'+xEN+'" y="'+yEN+'" width="'+bw+'" height="'+hEN+'" fill="#58a6ff" opacity="0.85"/>';
    s+='<rect x="'+xKO+'" y="'+yKO+'" width="'+bw+'" height="'+hKO+'" fill="#3fb950" opacity="0.85"/>';
    var labelX=xEN+bw;
    var labelY=H-pb+14;
    s+='<text x="'+labelX+'" y="'+labelY+'" fill="#8b949e" font-size="8" text-anchor="middle" transform="rotate(-35,'+labelX+','+labelY+')">'+r.label+'</text>';
  });
  s+='</svg>';
  s+='<div class="lg"><span><span class="sw" style="background:#58a6ff"></span>EN R@10</span><span><span class="sw" style="background:#3fb950"></span>KO R@10</span></div>';
  document.getElementById('koChart').innerHTML=s;
})();
master();
</script>
</body></html>"""

html = (html.replace("__EVAL__", json.dumps(EVAL, ensure_ascii=False))
            .replace("__HASH__", json.dumps(HASH, ensure_ascii=False))
            .replace("__STORAGE__", json.dumps(STORAGE, ensure_ascii=False))
            .replace("__SPEED__", json.dumps(SPEED, ensure_ascii=False))
            .replace("__SWEEP__", json.dumps(SWEEP, ensure_ascii=False))
            .replace("__SWEEPFLOAT__", json.dumps(SWEEP_FLOAT, ensure_ascii=False))
            .replace("__OOD__", json.dumps(OOD, ensure_ascii=False))
            .replace("__ROBUST__", json.dumps(ROBUST, ensure_ascii=False))
            .replace("__KO__", json.dumps(KO, ensure_ascii=False)))
open(OUT, "w").write(html)
print("wrote", OUT, len(html), "bytes")
