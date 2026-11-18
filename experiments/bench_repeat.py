"""「まとめて」と「1つずつ」を交互に5回ずつ繰り返し、速さの差が偶然でないかを確かめる。

使い方:
    python experiments/bench_repeat.py                  # 4b、5回
    python experiments/bench_repeat.py --rounds 5 --model Mapika/decider-2b

偏りを防ぐ工夫:
    - 回ごとに順番を入れ替える（奇数回は「まとめて」が先、偶数回は「1つずつ」が先）
      → 時間とともにMacが熱で遅くなる影響などを打ち消す
    - 同じ30件を両方式で測り、1件ごとに比べる
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
ap.add_argument("--rounds", type=int, default=5)
args = ap.parse_args()

questions = json.load(open(args.task, encoding="utf-8"))
texts = [r["text"] for r in csv.DictReader(open(args.data, encoding="utf-8"))]

d = Decider(args.model, device="mps", dtype=torch.bfloat16, use_graphs=False)


def run(text, mode):
    if mode == "まとめて":
        return d.system_one(text, questions)["answers"]
    return {q: d.system_one(text, {q: s})["answers"][q] for q, s in questions.items()}


def top(ans):
    """各質問の答え（比較用）"""
    out = {}
    for q, a in ans.items():
        if "choice" in a:
            out[q] = a["choice"]
        elif "noul" in a:
            out[q] = a["noul"] >= 0.5
        else:
            p = a["probabilities"]
            out[q] = max(p, key=p.get)
    return out


def probs(ans):
    """確率を1列に並べる（ずれの比較用）"""
    v = []
    for q in sorted(ans):
        a = ans[q]
        if "noul" in a and "probabilities" not in a:
            v.append(a["noul"])
        else:
            v.extend(a["probabilities"][k] for k in sorted(a["probabilities"]))
    return v


print("ウォームアップ中...")
for m in ("まとめて", "1つずつ"):
    run(texts[0], m)

modes = ("まとめて", "1つずつ")
times = {m: [[None] * len(texts) for _ in range(args.rounds)] for m in modes}
answers = {m: [[None] * len(texts) for _ in range(args.rounds)] for m in modes}

for r in range(args.rounds):
    order = modes if r % 2 == 0 else modes[::-1]
    for m in order:
        for i, t in enumerate(texts):
            s = time.time()
            ans = run(t, m)
            times[m][r][i] = (time.time() - s) * 1000
            answers[m][r][i] = ans
        print(f"  {r + 1}回目 {m:<5} 中央値 {statistics.median(times[m][r]):5.0f} ms")

print("\n===== 回ごとの中央値 =====")
print(f"{'回':>3}  {'まとめて':>8}  {'1つずつ':>8}  {'短縮率':>6}")
for r in range(args.rounds):
    a, b = statistics.median(times["まとめて"][r]), statistics.median(times["1つずつ"][r])
    print(f"{r + 1:>3}  {a:8.0f}  {b:8.0f}  {(1 - b / a):6.1%}")

allA = [x for rr in times["まとめて"] for x in rr]
allB = [x for rr in times["1つずつ"] for x in rr]
print(f"\n全{len(allA)}回の中央値  まとめて {statistics.median(allA):.0f} ms / 1つずつ {statistics.median(allB):.0f} ms"
      f" / 短縮率 {1 - statistics.median(allB) / statistics.median(allA):.1%}")

# 1件ごとの比較（各件で5回の中央値どうしを比べる）
per_item = []
for i in range(len(texts)):
    a = statistics.median(times["まとめて"][r][i] for r in range(args.rounds))
    b = statistics.median(times["1つずつ"][r][i] for r in range(args.rounds))
    per_item.append(1 - b / a)
faster = sum(x > 0 for x in per_item)
print(f"1件ごと: {len(texts)}件中 {faster}件で「1つずつ」が速い / 短縮率 最小 {min(per_item):.1%}・"
      f"中央値 {statistics.median(per_item):.1%}・最大 {max(per_item):.1%}")

# 答えが毎回・両方式で同じか
base = [top(a) for a in answers["まとめて"][0]]
diff_answers = 0
max_dp = 0.0
for m in modes:
    for r in range(args.rounds):
        for i, ans in enumerate(answers[m][r]):
            if top(ans) != base[i]:
                diff_answers += 1
            p0, p1 = probs(answers["まとめて"][0][i]), probs(ans)
            max_dp = max(max_dp, max(abs(x - y) for x, y in zip(p0, p1)))
print(f"\n===== 答えの一致 =====")
print(f"答えが1回目の「まとめて」と違った件数: {diff_answers} / {2 * args.rounds * len(texts)}")
print(f"確率の最大のずれ: {max_dp:.4f}")
