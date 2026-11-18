"""質問ごとの処理時間を測る。どの質問が重いかを調べる。

使い方:
    python experiments/time_questions.py                        # 4b
    python experiments/time_questions.py --model Mapika/decider-2b
"""
import argparse
import csv
import json
import statistics
import time

import torch
from decider.infer import Decider

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Mapika/decider-4b")
ap.add_argument("--task", default="data/task.json")
ap.add_argument("--data", default="data/data.csv")
ap.add_argument("--n", type=int, default=10, help="使う件数")
args = ap.parse_args()

questions = json.load(open(args.task, encoding="utf-8"))
texts = [r["text"] for r in csv.DictReader(open(args.data, encoding="utf-8"))][: args.n]

d = Decider(args.model, device="mps", dtype=torch.bfloat16, use_graphs=False)
d.system_one(texts[0], questions)  # ウォームアップ

cases = {f"{q} だけ": {q: questions[q]} for q in questions}
cases["全部まとめて"] = questions

print(f"\nモデル {args.model} / {len(texts)}件の中央値\n")
for name, qs in cases.items():
    times = []
    for t in texts:
        s = time.time()
        d.system_one(t, qs)
        times.append((time.time() - s) * 1000)
    print(f"  {name:<24} {statistics.median(times):7.0f} ms")
