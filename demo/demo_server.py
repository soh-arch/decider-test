"""percentile-ranked triage のデモ用サーバー。

使い方:
    python demo/demo_server.py
    ブラウザで http://localhost:8765 を開く

起動時に decider を読み込み、基準集合（data/reference.csv）の案件を採点しておく。
画面（priority_demo.html）から届いた案件を採点し、優先度スコアと優先度パーセンタイルを返す。
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
DATA = HERE.parent / "data"

QUESTION = {"urgency": {
    "type": "score",
    "instructions": "この問い合わせの緊急度は？",
    "criteria": ["低い：急ぎではない", "中くらい：数日以内に対応", "高い：今すぐ対応が必要"],
}}


def pick_device():
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.bfloat16
    return "cpu", torch.float32


class Scorer:
    def __init__(self, model):
        device, dtype = pick_device()
        print(f"モデルを読み込み中: {model}（{device}）")
        self.decider = Decider(model, device=device, dtype=dtype, use_graphs=False)
        self.lock = threading.Lock()  # GPU は1つなので、採点は1件ずつ行う

    def score(self, text):
        with self.lock:
            start = time.time()
            answer = self.decider.system_one(text, QUESTION)["answers"]["urgency"]
            ms = (time.time() - start) * 1000
        p = answer["probabilities"]
        expected = sum(int(level) * prob for level, prob in p.items())
        return {"expected": expected, "p_high": p["2"], "ms": ms}


class ReferenceSet:
    """基準集合の優先度スコアを並べ替えて持ち、新しいスコアの位置を求める。"""

    def __init__(self, scores):
        self.scores = sorted(scores)

    def percentile(self, x):
        lo = bisect.bisect_left(self.scores, x)
        hi = bisect.bisect_right(self.scores, x)
        return (lo + hi) / 2 / len(self.scores) * 100


def read_texts(path):
    with open(path, encoding="utf-8") as f:
        return [row for row in csv.DictReader(f) if row.get("text")]


def make_handler(scorer, reference, cases, model):
    page = (HERE / "priority_demo.html").read_bytes()

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
                self._send(page, "text/html; charset=utf-8")
            elif self.path == "/api/cases":
                self._send({"cases": cases, "model": model, "reference_n": len(reference.scores)})
            else:
                self._send({"error": "not found"}, code=404)

        def do_POST(self):
            if self.path != "/api/score":
                return self._send({"error": "not found"}, code=404)
            length = int(self.headers.get("Content-Length", 0))
            text = json.loads(self.rfile.read(length) or b"{}").get("text", "")
            if not text:
                return self._send({"error": "text is empty"}, code=422)
            result = scorer.score(text)
            result["percentile"] = reference.percentile(result["expected"])
            self._send(result)

        def log_message(self, *args):
            pass

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Mapika/decider-0.8b")
    ap.add_argument("--cases", default=str(DATA / "data.csv"), help="画面に届く案件（text 列）")
    ap.add_argument("--reference", default=str(DATA / "reference.csv"), help="基準集合の案件（text 列）")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    scorer = Scorer(args.model)
    scorer.score("ウォームアップ")

    print("基準集合を採点中...")
    reference = ReferenceSet(scorer.score(row["text"])["expected"] for row in read_texts(args.reference))
    print(f"  {len(reference.scores)}件、優先度スコア {reference.scores[0]:.2f}〜{reference.scores[-1]:.2f}")

    cases = [{"id": row.get("id", str(i)), "text": row["text"]} for i, row in enumerate(read_texts(args.cases), 1)]
    print(f"デモの案件: {len(cases)}件")

    handler = make_handler(scorer, reference, cases, args.model)
    print(f"\n準備ができました。ブラウザで http://localhost:{args.port} を開いてください（終了は Ctrl+C）")
    ThreadingHTTPServer(("127.0.0.1", args.port), handler).serve_forever()


if __name__ == "__main__":
    main()
