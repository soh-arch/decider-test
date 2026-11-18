"""優先度ソートのライブデモ用サーバー。

使い方（テスト環境で）:
    python demo/demo_server.py
    → ブラウザで http://localhost:8765 を開く

やること:
    - 起動時に decider-0.8b を読み込み、基準の過去案件（reference.csv）を採点して「ものさし」を作る
    - 画面（priority_demo.html）を配信する
    - 画面から届いた案件を採点し、期待値と優先度パーセンタイルを返す

外部のライブラリは decider と torch だけ（サーバーは Python 標準の http.server）。
"""
import argparse
import bisect
import csv
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from decider.infer import Decider

HERE = Path(__file__).resolve().parent

# 並べ替えに強い尋ね方：3段階の説明付きラベル
QUESTION = {"urgency": {
    "type": "score",
    "instructions": "この問い合わせの緊急度は？",
    "criteria": ["低い：急ぎではない", "中くらい：数日以内に対応", "高い：今すぐ対応が必要"],
}}

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Mapika/decider-0.8b")
ap.add_argument("--cases", default=str(HERE.parent / "data" / "data.csv"), help="デモに流す案件（text 列）")
ap.add_argument("--reference", default=str(HERE.parent / "data" / "reference.csv"), help="基準の過去案件（text 列）")
ap.add_argument("--port", type=int, default=8765)
args = ap.parse_args()


class Scorer:
    def __init__(self, model):
        print(f"モデル読み込み: {model}")
        self.d = Decider(model, device="mps", dtype=torch.bfloat16, use_graphs=False)
        self.lock = threading.Lock()   # GPU は1つなので、採点は1件ずつ

    def score(self, text):
        with self.lock:
            t = time.time()
            a = self.d.system_one(text, QUESTION)["answers"]["urgency"]
            ms = (time.time() - t) * 1000
        p = a["probabilities"]
        expected = sum(int(k) * v for k, v in p.items())   # 期待値（0〜2）
        return {"expected": expected, "p_high": p["2"], "ms": ms}


def read_texts(path):
    return [r for r in csv.DictReader(open(path, encoding="utf-8")) if r.get("text")]


scorer = Scorer(args.model)
scorer.score("ウォームアップ")

print("基準の過去案件を採点中...")
reference = sorted(scorer.score(r["text"])["expected"] for r in read_texts(args.reference))
print(f"  {len(reference)}件 / 期待値 {reference[0]:.2f}〜{reference[-1]:.2f}")

cases = [{"id": r.get("id", str(i)), "text": r["text"]} for i, r in enumerate(read_texts(args.cases), 1)]
print(f"デモ用の案件: {len(cases)}件")


def priority_percentile(x):
    """基準データの中で下から何%の位置か（同じ値が複数あるときは中間）。0〜100"""
    lo = bisect.bisect_left(reference, x)
    hi = bisect.bisect_right(reference, x)
    return (lo + hi) / 2 / len(reference) * 100


class Handler(BaseHTTPRequestHandler):
    def _send(self, body, ctype="application/json; charset=utf-8", code=200):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send((HERE / "priority_demo.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/cases":
            self._send({"cases": cases, "model": args.model, "reference_n": len(reference)})
        else:
            self._send({"error": "not found"}, code=404)

    def do_POST(self):
        if self.path != "/api/score":
            return self._send({"error": "not found"}, code=404)
        length = int(self.headers.get("Content-Length", 0))
        text = json.loads(self.rfile.read(length) or b"{}").get("text", "")
        if not text:
            return self._send({"error": "text is empty"}, code=422)
        r = scorer.score(text)
        r["percentile"] = priority_percentile(r["expected"])
        self._send(r)

    def log_message(self, *a):   # アクセスログは出さない
        pass


print(f"\n準備完了: ブラウザで http://localhost:{args.port} を開いてください（終了は Ctrl+C）")
ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
