"""decider の速度と精度を、正解付きCSVで評価する。

使い方:
    python experiments/eval_decider.py                      # task.json と data.csv を使う
    python experiments/eval_decider.py --model Mapika/decider-2b
    python experiments/eval_decider.py --data mydata.csv --task mytask.json

CSVの形式:
    - text 列: 判定対象の文章（state になる）
    - task.json の質問IDと同じ名前の列: 正解
        choice -> 選択肢のキー（例: billing）
        noul   -> true / false
        score  -> 段階の番号（0始まり）
    - 正解が空欄のセルは採点しない
"""
import argparse
import csv
import json
import statistics
import time

import torch
from decider.infer import Decider


def predict(answer, qtype):
    """モデルの答えから (予測ラベル, 予測の確率) を取り出す"""
    if qtype == "choice":
        return answer["choice"], answer["x_p_max"]
    if qtype == "noul":
        p = answer["noul"]
        return ("true" if p >= 0.5 else "false"), max(p, 1 - p)
    if qtype == "score":
        probs = answer["probabilities"]
        level = max(probs, key=probs.get)
        return level, probs[level]
    raise ValueError(qtype)


def expected_level(answer):
    """score の期待値（確率で重み付けした段階）"""
    return sum(int(k) * p for k, p in answer["probabilities"].items())


def order_agreement(items):
    """正解の段階が違う2件の組のうち、期待値の大小が正解と同じ順だった割合。同点は0.5"""
    agree = total = 0.0
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            (ei, gi), (ej, gj) = items[i], items[j]
            if gi == gj:
                continue
            total += 1
            if ei == ej:
                agree += 0.5
            elif (ei > ej) == (gi > gj):
                agree += 1
    return agree / total if total else float("nan"), int(total)


def ece(pairs, bins=5):
    """(確率, 正解か) の組から、確率のずれ（ECE）を出す。小さいほど良い"""
    if not pairs:
        return float("nan")
    total, err = len(pairs), 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        group = [(p, c) for p, c in pairs if lo < p <= hi or (b == 0 and p == 0)]
        if group:
            conf = sum(p for p, _ in group) / len(group)
            acc = sum(c for _, c in group) / len(group)
            err += len(group) / total * abs(conf - acc)
    return err


def pct(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(round(q * (len(values) - 1))))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Mapika/decider-4b")
    ap.add_argument("--task", default="data/task.json")
    ap.add_argument("--data", default="data/data.csv")
    ap.add_argument("--out", default="results/results.csv")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--separate", action="store_true", help="質問を1つずつ別々に投げる")
    args = ap.parse_args()

    questions = json.load(open(args.task, encoding="utf-8"))
    rows = list(csv.DictReader(open(args.data, encoding="utf-8")))

    print(f"モデル読み込み: {args.model} ({args.dtype})")
    t = time.time()
    d = Decider(args.model, device="mps", dtype=getattr(torch, args.dtype), use_graphs=False)
    print(f"  {time.time() - t:.1f}秒")

    print("ウォームアップ中...")
    d.system_one(rows[0]["text"], questions)

    latencies, tokens = [], []
    stats = {q: {"n": 0, "ok": 0, "pairs": [], "order": []} for q in questions}
    out_rows, mistakes = [], []

    for i, row in enumerate(rows, 1):
        t = time.time()
        if args.separate:
            r = {"answers": {}, "usage": {}}
            for q, spec in questions.items():
                r["answers"][q] = d.system_one(row["text"], {q: spec})["answers"][q]
        else:
            r = d.system_one(row["text"], questions)
        ms = (time.time() - t) * 1000
        latencies.append(ms)
        tokens.append(r.get("usage", {}).get("input_tokens", 0))

        out = {"id": row.get("id", i), "text": row["text"], "ms": round(ms)}
        for q, spec in questions.items():
            pred, prob = predict(r["answers"][q], spec["type"])
            out[f"{q}_pred"], out[f"{q}_prob"] = pred, round(prob, 3)
            gold = (row.get(q) or "").strip().lower()
            if spec["type"] == "score":
                exp = expected_level(r["answers"][q])
                out[f"{q}_exp"] = round(exp, 3)
                if gold:
                    stats[q]["order"].append((exp, int(gold)))
            if gold:
                ok = pred.lower() == gold
                s = stats[q]
                s["n"] += 1
                s["ok"] += ok
                s["pairs"].append((prob, ok))
                out[f"{q}_gold"], out[f"{q}_ok"] = gold, ok
                if not ok:
                    mistakes.append((out["id"], q, gold, pred, prob, row["text"]))
        out_rows.append(out)
        print(f"  {i}/{len(rows)}  {ms:.0f} ms")

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    print("\n===== 速度（1リクエスト = 全質問まとめて）=====")
    print(f"件数 {len(latencies)} / 質問数 {len(questions)}")
    print(f"中央値 {statistics.median(latencies):.0f} ms / p90 {pct(latencies, 0.9):.0f} ms / 最大 {max(latencies):.0f} ms")
    slow = sorted(zip(latencies, tokens, rows), key=lambda x: -x[0])[:3]
    print("遅かった上位3件:")
    for ms, tok, row in slow:
        print(f"  {ms:.0f} ms / {tok} tokens / {row['text'][:40]}")

    print("\n===== 精度 =====")
    for q, s in stats.items():
        if s["n"]:
            right = [p for p, c in s["pairs"] if c]
            wrong = [p for p, c in s["pairs"] if not c]
            print(f"{q}: 正答率 {s['ok']}/{s['n']} = {s['ok'] / s['n']:.1%} / ECE {ece(s['pairs']):.3f}")
            print(f"    確率の平均  正解時 {statistics.mean(right) if right else float('nan'):.2f}"
                  f" / 不正解時 {statistics.mean(wrong) if wrong else float('nan'):.2f}")
            if s["order"]:
                rate, n_pairs = order_agreement(s["order"])
                print(f"    順序の一致率 {rate:.1%}（正解の段階が違う {n_pairs} 組のうち、期待値の大小が正解と同じ順）")
                for lv in sorted({g for _, g in s["order"]}):
                    vals = [e for e, g in s["order"] if g == lv]
                    print(f"      正解 {lv} の期待値  平均 {statistics.mean(vals):.2f} / 範囲 {min(vals):.2f}〜{max(vals):.2f}（{len(vals)}件）")

    print("\n===== 間違えた問題 =====")
    for id_, q, gold, pred, prob, text in mistakes:
        print(f"[{id_}] {q}: 正解 {gold} / 予測 {pred} ({prob:.2f})  {text[:50]}")
    print(f"\n詳細は {args.out} に保存しました。")


if __name__ == "__main__":
    main()
