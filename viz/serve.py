#!/usr/bin/env python3
"""원클릭 viz 서버 — 브라우저에서 모든 인터랙티브 시각화를 본다.
stdlib만 사용. 실행: `python3 serve.py` (또는 `bash serve.sh`).
file:// 의 CORS 문제를 피하려 로컬 HTTP 서버로 띄운다. JSON/썸네일 완결성을 먼저 검사한다.
"""
from __future__ import annotations
import http.server, socketserver, os, sys, socket, threading, webbrowser, glob

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

PAGES = [  # (파일, 한글설명)
    ("index.html", "대시보드 (전체 7개 링크)"),
    ("forward_stages.html", "① head 내부 변환 단계 (sliced→BN→L2→z→sign)"),
    ("distribution_explorer.html", "② 코드 분포 · BN on/off"),
    ("sign_information_loss.html", "③ sign이 버리는 정보 (연속 vs Hamming 랭킹)"),
    ("embedding_space.html", "④ 임베딩/코드 2D 공간 (PCA/UMAP, 12언어)"),
    ("retrieval_anatomy.html", "⑤ 성공/실패 쿼리 해부"),
    ("signloss.html", "⑥ A-베팅 GO/NO-GO (연속/비대칭/Hamming R@10)"),
    ("outlier_gallery.html", "⑦ 실패 케이스 A/B 썸네일 갤러리"),
]
REQ_JSON = ["distributions.json", "forward_stages.json", "sign_info.json",
            "embedding_2d.json", "retrieval_anatomy.json", "signloss.json", "outlier_cases.json"]


def check():
    miss = []
    for p, _ in PAGES:
        if not os.path.exists(p):
            miss.append(f"viz/{p}")
    for j in REQ_JSON:
        if not os.path.exists(os.path.join("data", j)):
            miss.append(f"viz/data/{j}")
    thumbs = glob.glob(os.path.join("data", "thumbs_outlier", "*.jpg"))
    ok = True
    if miss:
        ok = False
        print("\n❌ 필수 파일 누락:")
        for m in miss:
            print(f"   - {m}")
        print("   → 이 브랜치(analysis-signloss-outlier)를 제대로 clone/checkout 했는지 확인하세요.")
        print("   → 재생성(DGX): scripts/viz_data.py · signloss_quant.py · outlier_debug.py")
    if not thumbs:
        ok = False
        print("\n❌ 썸네일 없음: viz/data/thumbs_outlier/*.jpg 가 비어 있습니다.")
        print("   → outlier_gallery.html 의 이미지가 안 뜹니다. clone에 포함됐는지 확인.")
        print("   → 재생성(DGX, 이미지 필요): REPO=$(pwd) .venv/bin/python scripts/outlier_debug.py")
    else:
        print(f"✅ 썸네일 {len(thumbs)}개 확인")
    if ok:
        print(f"✅ JSON {len(REQ_JSON)}개 · HTML {len(PAGES)}개 모두 존재")
    return ok


def free_port(start=8800, tries=20):
    for p in range(start, start + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:  # connect 실패 = 비어 있음
                return p
    return None


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)  # 즉시 출력(파이프/리다이렉트에서도 URL 바로 보이게)
    except Exception:
        pass
    print("=" * 60)
    print(" vlm_quantization — viz 로컬 서버")
    print("=" * 60)
    complete = check()
    if not complete and "--force" not in sys.argv:
        print("\n위 누락을 해결한 뒤 다시 실행하세요. (무시하고 띄우려면 --force)")
        sys.exit(1)
    port = free_port()
    if port is None:
        print("\n❌ 8800~8819 포트가 모두 사용 중입니다. 기존 서버를 끄세요: lsof -ti tcp:8800 | xargs kill")
        sys.exit(1)
    url = f"http://localhost:{port}/"
    handler = http.server.SimpleHTTPRequestHandler

    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True   # SO_REUSEADDR: TIME_WAIT 포트 재바인드 허용
        daemon_threads = True

    httpd = Server(("127.0.0.1", port), handler)
    print("\n" + "-" * 60)
    print(f"🌐 서버 시작: {url}")
    print("-" * 60)
    for p, desc in PAGES:
        print(f"   {url}{p:30}  {desc}")
    print("-" * 60)
    print("브라우저가 자동으로 열립니다. 안 열리면 위 대시보드 URL을 직접 입력하세요.")
    print("종료: Ctrl+C")
    print("(차트는 Plotly CDN을 쓰므로 인터넷이 필요합니다. 오프라인이면 차트만 안 뜹니다.)\n")
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n서버를 종료합니다. 안녕히.")
        httpd.shutdown()


if __name__ == "__main__":
    main()
