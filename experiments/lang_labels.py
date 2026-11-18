"""緊急度の3段階ラベル（低・中・高）を5言語で比べる。

使い方:
    python experiments/lang_labels.py                               # 0.8b
    python experiments/lang_labels.py --model Mapika/decider-2b

条件をそろえるための工夫:
    - 問い合わせ文と質問文（「この問い合わせの緊急度は？」）は全条件で同じ日本語
    - 変えるのは3つのラベルの言葉だけ（説明文は付けない）
    - 比較用に、これまでの説明付きラベル（基準）も測る
"""
import argparse
import csv
import statistics

import torch
from decider.infer import Decider

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Mapika/decider-0.8b")
ap.add_argument("--data", default="data/data.csv")
args = ap.parse_args()

INSTRUCTION = "この問い合わせの緊急度は？"
CONDITIONS = {
    "基準（説明付き）": ["低い：急ぎではない", "中くらい：数日以内に対応", "高い：今すぐ対応が必要"],
    "日本語": ["低い", "中くらい", "高い"],
    "英語": ["low", "medium", "high"],
    "中国語": ["低", "中等", "高"],
    "アラビア語": ["منخفض", "متوسط", "مرتفع"],
    "トルコ語": ["düşük", "orta", "yüksek"],
}

rows = list(csv.DictReader(open(args.data, encoding="utf-8")))
texts = [r["text"] for r in rows]
gold = [int(r["urgency"]) for r in rows]

d = Decider(args.model, device="mps", dtype=torch.bfloat16, use_graphs=False)


def ask(text, labels):
    q = {"urgency": {"type": "score", "instructions": INSTRUCTION, "criteria": labels}}
    p = d.system_one(text, q)["answers"]["urgency"]["probabilities"]
    return [p[str(k)] for k in range(3)]


def ranks(xs):
    """順位（同点は平均順位）"""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    ma, mb = statistics.mean(ra), statistics.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den


def order_agreement(exp, g):
    agree = total = 0.0
    for i in range(len(g)):
        for j in range(i + 1, len(g)):
            if g[i] == g[j]:
                continue
            total += 1
            if exp[i] == exp[j]:
                agree += 0.5
            elif (exp[i] > exp[j]) == (g[i] > g[j]):
                agree += 1
    return agree / total


ask(texts[0], CONDITIONS["日本語"])  # ウォームアップ

results = {}
for name, labels in CONDITIONS.items():
    probs = [ask(t, labels) for t in texts]
    exp = [p[1] + 2 * p[2] for p in probs]
    top = [max(range(3), key=lambda k: p[k]) for p in probs]
    results[name] = {"probs": probs, "exp": exp, "top": top}
    print(f"  {name} 完了")

print(f"\n===== {args.model} / {len(texts)}件 =====\n")
print(f"{'条件':<12} {'最有力の段階 低/中/高':>18} {'中の確率 平均':>10} {'期待値 平均':>9} {'期待値の幅':>12} {'順序一致率':>8}")
for name, r in results.items():
    c = [r["top"].count(k) for k in range(3)]
    mid = statistics.mean(p[1] for p in r["probs"])
    print(f"{name:<12} {c[0]:>8}/{c[1]:>2}/{c[2]:>2} {mid:>14.2f} {statistics.mean(r['exp']):>11.2f}"
          f" {min(r['exp']):>6.2f}〜{max(r['exp']):.2f} {order_agreement(r['exp'], gold):>10.1%}")

print("\n===== 並び順の似かより（順位相関。1.00で完全に同じ並び）=====\n")
names = list(results)
print(" " * 14 + "".join(f"{n[:5]:>8}" for n in names))
for a in names:
    print(f"{a:<12}" + "".join(f"{spearman(results[a]['exp'], results[b]['exp']):>8.2f}" for b in names))

print("\n===== 「中」は本当に真ん中か =====")
print("正解の段階ごとに、各条件の期待値の平均（低→中→高の順に増えていれば、中は低と高の間として機能している）\n")
print(f"{'条件':<12} {'正解 低':>8} {'正解 中':>8} {'正解 高':>8}")
for name, r in results.items():
    means = [statistics.mean(e for e, g in zip(r["exp"], gold) if g == k) for k in range(3)]
    print(f"{name:<12} {means[0]:>8.2f} {means[1]:>8.2f} {means[2]:>8.2f}")
