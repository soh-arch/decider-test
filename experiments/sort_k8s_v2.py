"""Kubernetes の issue で、尋ね方を変えたときの優先度ソートを比べる。

使い方:
    python experiments/sort_k8s_v2.py                              # 2b
    python experiments/sort_k8s_v2.py --model Mapika/decider-4b

比べる尋ね方:
    A: これまでの尋ね方（日本語の緊急度・3段階説明付き）
    B: Kubernetes 公式の優先度ラベルの定義をそのまま使う（英語・5段階）
       → 正解ラベルを付けたメンテナーと同じ物差しで聞く
"""
import argparse
import csv
import statistics
import time

import torch
from decider.infer import Decider

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Mapika/decider-2b")
ap.add_argument("--data", default="data/k8s_issues.csv")
ap.add_argument("--out", default="results/k8s_sorted_v2.csv")
args = ap.parse_args()

CONDITIONS = {
    "A 緊急度（日本語3段階）": {
        "instructions": "この案件の緊急度は？",
        "criteria": ["低い：急ぎではない", "中くらい：数日以内に対応", "高い：今すぐ対応が必要"],
    },
    # 説明文は Kubernetes の公式ラベル説明（低い順）。backlog は公式説明が
    # 「awaiting-more-evidence より高い」だけなので、その意味で書いている
    "B 公式の優先度定義（英語5段階）": {
        "instructions": "Which priority label would the Kubernetes maintainers assign to this issue?",
        "criteria": [
            "awaiting-more-evidence: Lowest priority. Possibly useful, but not yet enough support to actually get it done.",
            "backlog: Higher priority than awaiting-more-evidence, but no one is expected to work on it soon.",
            "important-longterm: Important over the long term, but may not be staffed and/or may need multiple releases to complete.",
            "important-soon: Must be staffed and worked on either currently, or very soon, ideally in time for the next release.",
            "critical-urgent: Highest priority. Must be actively worked on as someone's top priority right now.",
        ],
    },
}
NAMES = ["awaiting-more-evidence", "backlog", "important-longterm", "important-soon", "critical-urgent"]
TEST_WORDS = ("failing", "flak", "failure cluster")

rows = list(csv.DictReader(open(args.data, encoding="utf-8")))
for r in rows:
    r["level"] = int(r["level"])
    r["is_test"] = any(w in r["title"].lower() for w in TEST_WORDS)

d = Decider(args.model, device="mps", dtype=torch.bfloat16, use_graphs=False)


def score(text, cond):
    q = {"p": {"type": "score", **cond}}
    p = d.system_one(text, q)["answers"]["p"]["probabilities"]
    n = len(cond["criteria"])
    return sum(k * p[str(k)] for k in range(n)) / (n - 1)   # 0〜1 に正規化


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


score(rows[0]["text"], CONDITIONS["A 緊急度（日本語3段階）"])  # ウォームアップ

res, times = {}, {}
for name, cond in CONDITIONS.items():
    res[name], times[name] = [], []
    for i, r in enumerate(rows, 1):
        t = time.time()
        res[name].append(score(r["text"], cond))
        times[name].append((time.time() - t) * 1000)
    print(f"  {name} 完了（中央値 {statistics.median(times[name]):.0f} ms）")

gold = [r["level"] for r in rows]
nt = [i for i, r in enumerate(rows) if not r["is_test"]]

print(f"\n===== {args.model} / {len(rows)}件（うちテスト失敗の報告 {len(rows) - len(nt)}件）=====\n")
print(f"{'尋ね方':<22}{'順位相関':>8}{'順序一致率':>10}{'テスト報告を除く相関':>14}")
for name, s in res.items():
    s_nt, g_nt = [s[i] for i in nt], [gold[i] for i in nt]
    print(f"{name:<22}{spearman(s, gold):>8.2f}{order_agreement(s, gold):>11.1%}{spearman(s_nt, g_nt):>16.2f}")

print("\n===== メンテナーの優先度ごとの平均スコア（0〜1）=====\n")
print(f"{'優先度':<24}" + "".join(f"{n[:12]:>14}" for n in res))
for lv in range(4, -1, -1):
    print(f"{NAMES[lv]:<24}" + "".join(
        f"{statistics.mean(s[i] for i in range(len(rows)) if gold[i] == lv):>14.2f}" for s in res.values()))

print(f"\n尋ね方どうしの順位相関: {spearman(*res.values()):.2f}")

for name, s in res.items():
    order = sorted(range(len(rows)), key=lambda i: s[i], reverse=True)
    print(f"\n===== {name}：上位10件 =====")
    for rank, i in enumerate(order[:10], 1):
        r = rows[i]
        print(f"  {rank:>2}  {s[i]:.2f}  [{r['priority']:<22}] {r['title'][:55]}")

with open(args.out, "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "level", "priority", "is_test", *res.keys(), "title"])
    for i, r in enumerate(rows):
        w.writerow([r["id"], r["level"], r["priority"], r["is_test"],
                    *[round(s[i], 3) for s in res.values()], r["title"]])
print(f"\n全件は {args.out} に保存しました。")
