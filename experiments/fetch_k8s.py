"""Kubernetes の issue を優先度ラベル付きで集め、CSV にする。

使い方（Mac のテスト環境で）:
    python experiments/fetch_k8s.py                 # 5段階×10件 = 50件
    python experiments/fetch_k8s.py --per-level 12  # 1段階あたりの件数を変える

GitHub にログインしていなくても動く（検索APIは1分に10回までの制限あり。このスクリプトは5回だけ使う）。
トークンを持っていれば、環境変数 GITHUB_TOKEN に入れると制限が緩くなる。

「妥当な」issue の条件:
    - プルリクエストではなく issue である
    - 優先度ラベルがちょうど1つだけ付いている
    - 作成者がボットではない（自動生成の flaky テスト報告などを除く）
    - 本文が英語で、テンプレートのコメントを除いて200文字以上ある
"""
import argparse
import csv
import json
import os
import random
import re
import time
import urllib.parse
import urllib.request

# 優先度（数字が大きいほど優先度が高い）。定義は Kubernetes の公式ラベル説明による
LEVELS = [
    ("priority/awaiting-more-evidence", 0),
    ("priority/backlog", 1),
    ("priority/important-longterm", 2),
    ("priority/important-soon", 3),
    ("priority/critical-urgent", 4),
]
PRIORITY_NAMES = {name for name, _ in LEVELS}

ap = argparse.ArgumentParser()
ap.add_argument("--per-level", type=int, default=10)
ap.add_argument("--out", default="data/k8s_issues.csv")
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()
random.seed(args.seed)

headers = {"Accept": "application/vnd.github+json", "User-Agent": "decider-priority-test"}
if os.environ.get("GITHUB_TOKEN"):
    headers["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"


def search(label):
    q = f'repo:kubernetes/kubernetes is:issue label:"{label}"'
    url = "https://api.github.com/search/issues?" + urllib.parse.urlencode(
        {"q": q, "per_page": 100, "sort": "created", "order": "desc"})
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))["items"]


def clean(body):
    body = re.sub(r"<!--.*?-->", "", body or "", flags=re.S)   # テンプレートのコメントを除く
    body = re.sub(r"```.*?```", "[code]", body, flags=re.S)      # 長いコードやログを短縮
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    return body


def is_english(text):
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and sum(c.isascii() for c in letters) / len(letters) > 0.95


def is_bot(user):
    login = (user or {}).get("login", "").lower()
    return (user or {}).get("type") == "Bot" or "bot" in login


rows = []
for label, level in LEVELS:
    print(f"取得中: {label}")
    items = search(label)
    ok = []
    for it in items:
        if "pull_request" in it or is_bot(it.get("user")):
            continue
        prios = [l["name"] for l in it["labels"] if l["name"] in PRIORITY_NAMES]
        if prios != [label]:
            continue
        body = clean(it.get("body"))
        if len(body) < 200 or not is_english(it["title"] + body):
            continue
        ok.append(it)
    picked = random.sample(ok, min(args.per_level, len(ok)))
    print(f"  条件を満たした {len(ok)} 件から {len(picked)} 件を採用")
    for it in picked:
        body = clean(it["body"])
        rows.append({
            "id": it["number"],
            "level": level,
            "priority": label.split("/")[1],
            "title": it["title"],
            "text": f"{it['title']}\n\n{body[:2000]}",
            "created_at": it["created_at"],
            "closed_at": it.get("closed_at") or "",
            "state": it["state"],
            "url": it["html_url"],
        })
    time.sleep(7)  # 検索APIの回数制限を避ける

rows.sort(key=lambda r: r["created_at"])  # 時系列順に並べる
with open(args.out, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print(f"\n{len(rows)} 件を {args.out} に保存しました。")
for label, level in LEVELS:
    print(f"  {label}: {sum(r['level'] == level for r in rows)} 件")
