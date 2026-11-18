"""deciderが「文脈が足りない」と感じるかどうかと、優先度の乖離の関係を調べる。

使い方:
    python experiments/fetch_k8s.py --per-level 20 --out data/k8s_issues_100.csv   # 先にデータを増やす
    python experiments/eval_context.py --data data/k8s_issues_100.csv              # 2b

やること:
    1. 優先度を聞く（前回もっとも成績の良かった尋ね方A：日本語の緊急度3段階）
    2. 文脈依存を3つの側面に分けて yes/no で聞く（真実ではなく、deciderがどう感じるか）
    3. 各件の「優先度乖離」を求め、文脈依存の答えとの相関を見る

乖離の計算:
    deciderのスコアとメンテナーの優先度を、それぞれ全件の中での順位（0〜1のパーセンタイル）に変換し、
    その差の絶対値を乖離とする。尺度（3段階と5段階）の違いに左右されない。

歪みへの対処:
    - 正解の段階ごとに分けた相関も出す（両端の段階ほど乖離が大きく出やすいため）
    - 本文の長さの影響を取り除いた相関（偏相関）も出す
"""
import argparse
import csv
import statistics
import time

import torch
from decider.infer import Decider

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Mapika/decider-2b")
ap.add_argument("--data", default="data/k8s_issues_100.csv")
ap.add_argument("--out", default="results/k8s_context.csv")
args = ap.parse_args()

PRIORITY = {"type": "score", "instructions": "この案件の緊急度は？",
            "criteria": ["低い：急ぎではない", "中くらい：数日以内に対応", "高い：今すぐ対応が必要"]}

# 文脈依存の3つの側面。sign=+1 は「yesなら文脈依存」、-1 は「yesなら文脈は足りている」
CONTEXT = {
    "外部情報が必要": (+1, "この案件の優先度を決めるには、本文に書かれていない情報（リリースの予定、担当者の状況、ほかの議論など）が必要か？"),
    "影響が明記":     (-1, "この問題が誰に・どの程度の影響を与えるかが、本文にはっきり書かれているか？"),
    "根拠が十分":     (-1, "問題の深刻さを判断できるだけの根拠（再現手順、ログ、発生範囲など）が本文にそろっているか？"),
}
NAMES = ["awaiting-more-evidence", "backlog", "important-longterm", "important-soon", "critical-urgent"]

rows = list(csv.DictReader(open(args.data, encoding="utf-8")))
d = Decider(args.model, device="mps", dtype=torch.bfloat16, use_graphs=False)
d.system_one(rows[0]["text"], {"p": PRIORITY})  # ウォームアップ

start = time.time()
for i, r in enumerate(rows, 1):
    r["level"] = int(r["level"])
    r["length"] = len(r["text"])
    a = d.system_one(r["text"], {"p": PRIORITY})["answers"]["p"]["probabilities"]
    r["score"] = sum(int(k) * p for k, p in a.items())
    for name, (sign, text) in CONTEXT.items():
        p_yes = d.system_one(r["text"], {"q": {"type": "noul", "instructions": text}})["answers"]["q"]["noul"]
        r[name] = p_yes if sign > 0 else 1 - p_yes          # すべて「文脈依存の度合い」にそろえる
    r["文脈依存（平均）"] = statistics.mean(r[n] for n in CONTEXT)
    print(f"  {i}/{len(rows)}  経過 {time.time() - start:.0f}秒")


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
    if len(a) < 3:
        return float("nan")
    ra, rb = ranks(a), ranks(b)
    ma, mb = statistics.mean(ra), statistics.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else float("nan")


def partial(x, y, z):
    """z の影響を取り除いた x と y の順位相関"""
    rxy, rxz, ryz = spearman(x, y), spearman(x, z), spearman(y, z)
    return (rxy - rxz * ryz) / (((1 - rxz ** 2) * (1 - ryz ** 2)) ** 0.5)


def pct(xs):
    n = len(xs)
    return [(r - 1) / (n - 1) for r in ranks(xs)]


n = len(rows)
p_score, p_gold = pct([r["score"] for r in rows]), pct([r["level"] for r in rows])
for r, a, b in zip(rows, p_score, p_gold):
    r["乖離"] = abs(a - b)
div = [r["乖離"] for r in rows]
length = [r["length"] for r in rows]
gold = [r["level"] for r in rows]
aspects = list(CONTEXT) + ["文脈依存（平均）"]

print(f"\n===== {args.model} / {n}件 =====\n")
print(f"優先度ソートの順位相関（メンテナーと）: {spearman([r['score'] for r in rows], gold):.2f}")
print(f"乖離の平均: {statistics.mean(div):.3f}（0で完全一致、偶然なら約0.33）")

print("\n===== 文脈依存の度合いと乖離の相関（正なら「文脈不足を感じる件ほど外す」）=====\n")
print(f"{'側面':<16}{'全体':>7}{'段階別の平均':>12}{'長さの影響を除く':>14}")
for a in aspects:
    x = [r[a] for r in rows]
    per_level = [spearman([r[a] for r in rows if r["level"] == lv], [r["乖離"] for r in rows if r["level"] == lv])
                 for lv in range(5)]
    per_level = [v for v in per_level if v == v]
    print(f"{a:<16}{spearman(x, div):>7.2f}{statistics.mean(per_level):>12.2f}{partial(x, div, length):>16.2f}")

print("\n===== 3つの側面どうしの一致（低ければ、問いの立て方で答えが大きく変わる）=====\n")
names3 = list(CONTEXT)
print(" " * 16 + "".join(f"{s:>12}" for s in names3))
for a in names3:
    print(f"{a:<14}" + "".join(f"{spearman([r[a] for r in rows], [r[b] for r in rows]):>12.2f}" for b in names3))

print("\n===== 参考：文脈依存の度合いと、他の要素との相関 =====\n")
x = [r["文脈依存（平均）"] for r in rows]
print(f"  本文の長さ         {spearman(x, length):>6.2f}")
print(f"  メンテナーの優先度 {spearman(x, gold):>6.2f}")
print(f"  deciderの優先度    {spearman(x, [r['score'] for r in rows]):>6.2f}")

print("\n===== 文脈依存の度合いで3つに分けたときの乖離 =====\n")
order = sorted(range(n), key=lambda i: rows[i]["文脈依存（平均）"])
third = n // 3
for label, idx in (("文脈依存が低い1/3", order[:third]),
                   ("中くらいの1/3", order[third:2 * third]),
                   ("文脈依存が高い1/3", order[2 * third:])):
    sub = [rows[i] for i in idx]
    print(f"  {label:<14} 乖離の平均 {statistics.mean(r['乖離'] for r in sub):.3f}"
          f"  / この中での順位相関 {spearman([r['score'] for r in sub], [r['level'] for r in sub]):.2f}（{len(sub)}件）")

print("\n===== 乖離が大きかった上位10件 =====\n")
for r in sorted(rows, key=lambda r: -r["乖離"])[:10]:
    print(f"  乖離 {r['乖離']:.2f}  文脈依存 {r['文脈依存（平均）']:.2f}  [{r['priority']:<22}] {r['title'][:50]}")

cols = ["id", "level", "priority", "score", "乖離", "length", *aspects, "title", "url"]
with open(args.out, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
print(f"\n全件は {args.out} に保存しました。")
