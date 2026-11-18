"""Kubernetes の issue で優先度ソートを評価する。

使い方:
    python experiments/sort_k8s.py                              # 0.8b
    python experiments/sort_k8s.py --model Mapika/decider-2b

やること:
    1. 各 issue の緊急度を3段階の説明付きラベルで聞き、期待値を求める
    2. 期待値の大きい順に並べる（優先度ソート）
    3. メンテナーが付けた5段階の優先度と比べる
"""
import argparse
import csv
import statistics
import time

import torch
from decider.infer import Decider

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Mapika/decider-0.8b")
ap.add_argument("--data", default="data/k8s_issues.csv")
ap.add_argument("--out", default="results/k8s_sorted.csv")
args = ap.parse_args()

QUESTION = {"urgency": {
    "type": "score",
    "instructions": "この案件の緊急度は？",
    "criteria": ["低い：急ぎではない", "中くらい：数日以内に対応", "高い：今すぐ対応が必要"],
}}
NAMES = ["awaiting-more-evidence", "backlog", "important-longterm", "important-soon", "critical-urgent"]

rows = list(csv.DictReader(open(args.data, encoding="utf-8")))
d = Decider(args.model, device="mps", dtype=torch.bfloat16, use_graphs=False)
d.system_one(rows[0]["text"], QUESTION)  # ウォームアップ

times = []
for i, r in enumerate(rows, 1):
    t = time.time()
    a = d.system_one(r["text"], QUESTION)["answers"]["urgency"]
    times.append((time.time() - t) * 1000)
    r["expected"] = a["score"] if "score" in a else sum(int(k) * p for k, p in a["probabilities"].items())
    r["p_high"] = a["probabilities"]["2"]
    r["level"] = int(r["level"])
    print(f"  {i}/{len(rows)}  {times[-1]:.0f} ms")

rows.sort(key=lambda r: (r["expected"], r["p_high"]), reverse=True)
for rank, r in enumerate(rows, 1):
    r["rank"] = rank
with open(args.out, "w", encoding="utf-8", newline="") as f:
    cols = ["rank", "expected", "p_high", "level", "priority", "title", "created_at", "url"]
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    rk = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            rk[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return rk


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    ma, mb = statistics.mean(ra), statistics.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den


def order_agreement(s, g):
    agree = total = 0.0
    for i in range(len(g)):
        for j in range(i + 1, len(g)):
            if g[i] == g[j]:
                continue
            total += 1
            if s[i] == s[j]:
                agree += 0.5
            elif (s[i] > s[j]) == (g[i] > g[j]):
                agree += 1
    return agree / total


exp = [r["expected"] for r in rows]
gold = [r["level"] for r in rows]
n = len(rows)

print(f"\n===== {args.model} / {n}件 =====\n")
print(f"速度: 中央値 {statistics.median(times):.0f} ms / 最大 {max(times):.0f} ms")
print(f"順位相関（メンテナーの優先度と）: {spearman(exp, gold):.2f}")
print(f"順序一致率: {order_agreement(exp, gold):.1%}（50%で偶然と同じ）")

print("\n===== メンテナーの優先度ごとの期待値 =====\n")
for lv in range(4, -1, -1):
    vals = [r["expected"] for r in rows if r["level"] == lv]
    if vals:
        print(f"  {NAMES[lv]:<24} 平均 {statistics.mean(vals):.2f}  範囲 {min(vals):.2f}〜{max(vals):.2f}（{len(vals)}件）")

print("\n===== 上位・下位に何が来たか =====\n")
for k in (5, 10):
    top = [r["level"] for r in rows[:k]]
    print(f"  上位{k}件の内訳: " + " / ".join(f"{NAMES[lv]} {top.count(lv)}" for lv in range(4, -1, -1)))
bottom = [r["level"] for r in rows[-10:]]
print(f"  下位10件の内訳: " + " / ".join(f"{NAMES[lv]} {bottom.count(lv)}" for lv in range(4, -1, -1)))

print("\n===== 並べ替えた結果（上位15件）=====\n")
for r in rows[:15]:
    print(f"  {r['rank']:>2}  {r['expected']:.2f}  [{r['priority']:<22}] {r['title'][:60]}")
print(f"\n全件は {args.out} に保存しました。")
