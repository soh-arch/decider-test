"""メッセージを緊急度の高い順に並べる。

使い方:
    python experiments/sort_by_urgency.py                           # data.csv の先頭20件、2b
    python experiments/sort_by_urgency.py --model Mapika/decider-0.8b --n 20
    python experiments/sort_by_urgency.py --data mymessages.csv     # text 列があればよい
"""
import argparse
import csv
import time

import torch
from decider.infer import Decider

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Mapika/decider-2b")
ap.add_argument("--data", default="data/data.csv")
ap.add_argument("--n", type=int, default=20)
args = ap.parse_args()

# 緊急度だけを聞く（ほかの質問は並べ替えに不要なので省く）
URGENCY = {"urgency": {
    "type": "score",
    "instructions": "この問い合わせの緊急度は？",
    "criteria": ["低い：急ぎではない", "中くらい：数日以内に対応", "高い：今すぐ対応が必要"],
}}
LABELS = ["低", "中", "高"]

texts = [r["text"] for r in csv.DictReader(open(args.data, encoding="utf-8"))][: args.n]
d = Decider(args.model, device="mps", dtype=torch.bfloat16, use_graphs=False)
d.system_one(texts[0], URGENCY)  # ウォームアップ

rows = []
start = time.time()
for t in texts:
    p = d.system_one(t, URGENCY)["answers"]["urgency"]["probabilities"]
    p = [p[str(k)] for k in range(3)]
    expected = 0 * p[0] + 1 * p[1] + 2 * p[2]   # 期待値（0〜2の連続値）
    rows.append((expected, p[2], t))
elapsed = time.time() - start

# 期待値の高い順。同点なら「高」の確率が大きいほうを上に
rows.sort(key=lambda r: (r[0], r[1]), reverse=True)

print(f"\n{args.model} / {len(texts)}件 / 合計 {elapsed:.1f}秒\n")
print(f"{'順位':>3}  {'期待値':>5}  {'高の確率':>6}  目安  メッセージ")
for i, (e, ph, t) in enumerate(rows, 1):
    print(f"{i:>3}  {e:5.2f}  {ph:6.2f}   {LABELS[round(e)]}   {t[:40]}")
