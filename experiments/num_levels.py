"""緊急度を「数値の段階」で表したときの並べ替えの質を、段階数（3〜7）と言葉のラベルで比べる。

使い方:
    python experiments/num_levels.py                            # 0.8b
    python experiments/num_levels.py --model Mapika/decider-2b
    python experiments/num_levels.py --out results/num_levels_08b.csv   # 1件ごとのスコアも保存

比べ方:
    - 段階数が違うと期待値の尺度が変わるので、期待値 ÷（段階数−1）で 0〜1 に揃える（正規化スコア）
    - 見たいこと：①言葉と数値で違いが出るか ②段階の数で違いが出るか
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

CONDITIONS = {
    "言葉：低中高": ("この問い合わせの緊急度は？", ["低い", "中くらい", "高い"]),
    "言葉：説明付き": ("この問い合わせの緊急度は？",
                ["低い：急ぎではない", "中くらい：数日以内に対応", "高い：今すぐ対応が必要"]),
}
for n in range(3, 8):
    CONDITIONS[f"数値：{n}段階"] = (
        f"この問い合わせの緊急度を1から{n}の数字で評価してください。1が最も低く、{n}が最も高い緊急度です。",
        [str(i) for i in range(1, n + 1)],
    )

rows = list(csv.DictReader(open(args.data, encoding="utf-8")))
texts = [r["text"] for r in rows]
gold = [int(r["urgency"]) for r in rows]

d = Decider(args.model, device="mps", dtype=torch.bfloat16, use_graphs=False)


def score(text, instruction, labels):
    q = {"urgency": {"type": "score", "instructions": instruction, "criteria": labels}}
    p = d.system_one(text, q)["answers"]["urgency"]["probabilities"]
    n = len(labels)
    exp = sum(k * p[str(k)] for k in range(n))
    return exp / (n - 1)  # 0〜1 に正規化


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


score(texts[0], *CONDITIONS["数値：5段階"])  # ウォームアップ

res = {}
for name, (inst, labels) in CONDITIONS.items():
    res[name] = [score(t, inst, labels) for t in texts]
    print(f"  {name} 完了")

if args.out:
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "urgency_gold", *res])
        for i, r in enumerate(rows):
            w.writerow([r["id"], gold[i], *(f"{res[n][i]:.4f}" for n in res)])

def separation(s):
    """正解「高」と「低」の平均の差を、段階内のばらつきで割った値（大きいほど高と低をはっきり分けている）"""
    lo = [x for x, g in zip(s, gold) if g == 0]
    hi = [x for x, g in zip(s, gold) if g == 2]
    pooled = ((statistics.pvariance(lo) * len(lo) + statistics.pvariance(hi) * len(hi)) / (len(lo) + len(hi))) ** 0.5
    return (statistics.mean(hi) - statistics.mean(lo)) / pooled


print(f"\n===== {args.model} / {len(texts)}件（スコアは0〜1に正規化）=====\n")
print(f"{'条件':<14}{'順序一致率':>8}{'全体の幅':>13}{'高と低の分離度':>12}")
for name, s in res.items():
    print(f"{name:<14}{order_agreement(s, gold):>9.1%}  {min(s):>5.2f}〜{max(s):.2f}{separation(s):>12.2f}")

print("\n===== 正解の段階ごとの平均スコア（低・中・高の間隔が均等なほど尺度として扱いやすい）=====\n")
print(f"{'条件':<14}{'正解 低':>8}{'正解 中':>8}{'正解 高':>8}{'低→中':>8}{'中→高':>8}")
for name, s in res.items():
    m = [statistics.mean(x for x, g in zip(s, gold) if g == lv) for lv in range(3)]
    print(f"{name:<14}{m[0]:>8.2f}{m[1]:>8.2f}{m[2]:>8.2f}{m[1] - m[0]:>8.2f}{m[2] - m[1]:>8.2f}")

print("\n===== 並び順の似かより（順位相関）=====\n")
names = list(res)
short = [n.replace("言葉：", "").replace("数値：", "") for n in names]
print(" " * 14 + "".join(f"{s[:5]:>7}" for s in short))
for a, sa in zip(names, short):
    print(f"{sa:<12}" + "".join(f"{spearman(res[a], res[b]):>7.2f}" for b in names))

print("\n===== まとめ：グループ内とグループ間の順位相関の平均 =====\n")
words = [n for n in names if n.startswith("言葉")]
nums = [n for n in names if n.startswith("数値")]


def avg_corr(A, B):
    vals = [spearman(res[a], res[b]) for a in A for b in B if a != b]
    return statistics.mean(vals)


print(f"  言葉どうし   {avg_corr(words, words):.2f}")
print(f"  数値どうし   {avg_corr(nums, nums):.2f}")
print(f"  言葉と数値   {avg_corr(words, nums):.2f}")
