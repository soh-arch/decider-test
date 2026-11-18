"""言葉のラベルの段階数（3・4・5）と説明の有無で、並べ替えの質を比べる。

使い方:
    python experiments/word_levels.py                           # 0.8b
    python experiments/word_levels.py --model Mapika/decider-2b
    python experiments/word_levels.py --out results/word_levels_08b.csv   # 1件ごとのスコアも保存

条件をそろえるための工夫:
    - 質問文と問い合わせ文は全条件で同じ
    - 段階数が違うので、期待値 ÷（段階数−1）で 0〜1 に揃える（正規化スコア）
    - 段階の言葉と説明は下の LEVELS で自由に書き換えられる
"""
import argparse
import csv
import statistics

import torch
from decider.infer import Decider

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Mapika/decider-0.8b")
ap.add_argument("--data", default="data/data.csv")
ap.add_argument("--out", help="1件ごとの正規化スコアを書き出す CSV のパス")
args = ap.parse_args()

INSTRUCTION = "この問い合わせの緊急度は？"

# 段階の言葉と説明（低い順）。ここを書き換えて模索する
LEVELS = {
    3: [("低い", "急ぎではない"),
        ("中くらい", "数日以内に対応"),
        ("高い", "今すぐ対応が必要")],
    4: [("低い", "急ぎではない"),
        ("やや低い", "数日以内に対応"),
        ("やや高い", "今日中に対応"),
        ("高い", "今すぐ対応が必要")],
    5: [("低い", "急ぎではない"),
        ("やや低い", "1週間以内に対応"),
        ("中くらい", "数日以内に対応"),
        ("やや高い", "今日中に対応"),
        ("高い", "今すぐ対応が必要")],
}

CONDITIONS = {}
for n, lv in LEVELS.items():
    CONDITIONS[f"{n}段階・説明なし"] = [w for w, _ in lv]
    CONDITIONS[f"{n}段階・説明付き"] = [f"{w}：{desc}" for w, desc in lv]

rows = list(csv.DictReader(open(args.data, encoding="utf-8")))
texts = [r["text"] for r in rows]
gold = [int(r["urgency"]) for r in rows]

d = Decider(args.model, device="mps", dtype=torch.bfloat16, use_graphs=False)


def score(text, labels):
    q = {"urgency": {"type": "score", "instructions": INSTRUCTION, "criteria": labels}}
    p = d.system_one(text, q)["answers"]["urgency"]["probabilities"]
    n = len(labels)
    return sum(k * p[str(k)] for k in range(n)) / (n - 1)


def ranks(xs):
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


def separation(s):
    lo = [x for x, g in zip(s, gold) if g == 0]
    hi = [x for x, g in zip(s, gold) if g == 2]
    pooled = ((statistics.pvariance(lo) * len(lo) + statistics.pvariance(hi) * len(hi)) / (len(lo) + len(hi))) ** 0.5
    return (statistics.mean(hi) - statistics.mean(lo)) / pooled


score(texts[0], CONDITIONS["3段階・説明付き"])  # ウォームアップ

res = {}
for name, labels in CONDITIONS.items():
    res[name] = [score(t, labels) for t in texts]
    print(f"  {name} 完了")

if args.out:
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "urgency_gold", *res])
        for i, r in enumerate(rows):
            w.writerow([r["id"], gold[i], *(f"{res[n][i]:.4f}" for n in res)])

print(f"\n===== {args.model} / {len(texts)}件（スコアは0〜1に正規化）=====\n")
print(f"{'条件':<14}{'順序一致率':>8}{'全体の幅':>13}{'標準偏差':>8}{'高と低の分離度':>12}")
for name, s in res.items():
    print(f"{name:<14}{order_agreement(s, gold):>9.1%}  {min(s):>5.2f}〜{max(s):.2f}"
          f"{statistics.pstdev(s):>9.3f}{separation(s):>12.2f}")

print("\n===== 正解の段階ごとの平均スコア =====\n")
print(f"{'条件':<14}{'正解 低':>8}{'正解 中':>8}{'正解 高':>8}{'低→中':>8}{'中→高':>8}")
for name, s in res.items():
    m = [statistics.mean(x for x, g in zip(s, gold) if g == lv) for lv in range(3)]
    print(f"{name:<14}{m[0]:>8.2f}{m[1]:>8.2f}{m[2]:>8.2f}{m[1] - m[0]:>8.2f}{m[2] - m[1]:>8.2f}")

print("\n===== 並び順の似かより（順位相関）=====\n")
names = list(res)
short = [n.replace("段階・", "") for n in names]
print(" " * 12 + "".join(f"{s:>9}" for s in short))
for a, sa in zip(names, short):
    print(f"{sa:<10}" + "".join(f"{spearman(res[a], res[b]):>9.2f}" for b in names))


def avg_corr(A, B):
    vals = [spearman(res[a], res[b]) for a in A for b in B if a != b]
    return statistics.mean(vals)


plain = [n for n in names if "説明なし" in n]
desc = [n for n in names if "説明付き" in n]
print("\n===== まとめ：グループ内とグループ間の順位相関の平均 =====\n")
print(f"  説明なしどうし   {avg_corr(plain, plain):.2f}")
print(f"  説明付きどうし   {avg_corr(desc, desc):.2f}")
print(f"  説明なしと付き   {avg_corr(plain, desc):.2f}")
