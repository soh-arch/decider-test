"""Generate score_spread.svg: a grouped beeswarm of normalized priority scores by question style."""
import csv
import math
from pathlib import Path

OUT = Path(__file__).with_name("score_spread.svg")
DATA = Path(__file__).resolve().parents[3] / "results" / "num_levels_08b.csv"

BG = "#F7F3ED"
TEXT, TEXT2, TEXT3 = "#231E16", "#534C41", "#736D62"
LINE_L, LINE_M = "#E4DFD6", "#B5B0A7"
CARD = "#FFFEFB"
POINT = {
    0: (CARD, TEXT2),
    1: ("#C9C3B9", "#908A80"),
    2: (TEXT, TEXT),
}
FONT = "'Hiragino Sans','Hiragino Kaku Gothic ProN',sans-serif"

TITLE = "質問の仕方ごとの正規化した優先度スコアの散らばり"
AXIS_TITLE = "正規化した優先度スコア"
LEGEND_TITLE = "想定する正解"
CLASSES = ["低い", "中くらい", "高い"]
GROUPS = [
    ("数値：3段階", "数字・3段階"),
    ("数値：5段階", "数字・5段階"),
    ("数値：7段階", "数字・7段階"),
    ("言葉：低中高", "言葉・3段階"),
    ("言葉：説明付き", "言葉・3段階（説明付き）"),
]
RECOMMENDED = "言葉・3段階（説明付き）"

W = 500
PAD = 16
R = 3.1
GAP = 0.9


def load():
    with DATA.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    cls = [int(r["urgency_gold"]) for r in rows]
    groups = []
    for col, name in GROUPS:
        groups.append((name, [float(r[col]) for r in rows]))
    return cls, groups


def text_width(s, size):
    w = 0.0
    for ch in s:
        w += size * (0.6 if ord(ch) < 0x2E80 else 1.0)
    return w


def swarm(xs, d, side="both"):
    """Deterministic dodge: place points in ascending x at the offset closest to 0."""
    order = sorted(range(len(xs)), key=lambda i: (xs[i], i))
    placed = []
    ys = [0.0] * len(xs)
    for i in order:
        x = xs[i]
        near = [(px, py) for px, py in placed if abs(px - x) < d]
        cands = [0.0]
        for px, py in near:
            dy = math.sqrt(d * d - (px - x) ** 2)
            cands += [py + dy, py - dy]
        if side == "up":
            cands = [c for c in cands if c >= -1e-9]
        cands.sort(key=lambda c: (abs(c), -c))
        for c in cands:
            if all((px - x) ** 2 + (py - c) ** 2 >= d * d - 1e-6 for px, py in near):
                ys[i] = c
                break
        placed.append((x, ys[i]))
    return ys


def quantile(v, q):
    s = sorted(v)
    pos = (len(s) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


class Svg:
    def __init__(self):
        self.parts = []

    def add(self, s):
        self.parts.append(s)

    def text(self, x, y, s, size=12, fill=TEXT, weight=400, anchor="start", baseline="auto"):
        self.add(
            f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
            f'font-weight="{weight}" text-anchor="{anchor}" dominant-baseline="{baseline}">{s}</text>'
        )

    def line(self, x1, y1, x2, y2, stroke, width=1.0, cap="butt"):
        self.add(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{stroke}" stroke-width="{width}" stroke-linecap="{cap}"/>'
        )

    def rect(self, x, y, w, h, fill, rx=0):
        self.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}"/>')

    def dot(self, x, y, c, r=R):
        fill, stroke = POINT[c]
        self.add(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="0.8"/>'
        )

    def render(self, h):
        body = "\n".join(self.parts)
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{h:.0f}" '
            f'viewBox="0 0 {W} {h:.0f}" font-family="{FONT}">\n'
            f'<rect width="100%" height="100%" fill="{BG}"/>\n{body}\n</svg>\n'
        )


def header(svg, y=PAD):
    """Title and a one-line legend; returns the y below them."""
    svg.text(PAD, y + 15, TITLE, size=15, weight=700)
    ly = y + 15 + 24
    svg.text(PAD, ly, LEGEND_TITLE, size=12, fill=TEXT2, baseline="central")
    x = PAD + text_width(LEGEND_TITLE, 12) + 14
    for c, name in enumerate(CLASSES):
        svg.dot(x + R, ly, c, r=3.6)
        svg.text(x + 2 * R + 6, ly, name, size=12, baseline="central")
        x += 2 * R + 6 + text_width(name, 12) + 16
    return ly + 18


def x_axis(svg, x0, x1, y_top, y_bot, grid=True, frame=False):
    """Gridlines at 0, 0.5, 1 from y_top to y_bot, tick labels and axis title below."""
    for v in (0, 0.5, 1):
        x = x0 + (x1 - x0) * v
        if grid:
            svg.line(x, y_top, x, y_bot, LINE_L, 1)
        else:
            svg.line(x, y_bot - 4, x, y_bot, LINE_M, 1)
    if frame:
        svg.line(x0, y_bot, x1, y_bot, LINE_M, 1)
    ty = y_bot + 16
    for v, lab in ((0, "0"), (0.5, "0.5"), (1, "1")):
        svg.text(x0 + (x1 - x0) * v, ty, lab, size=12, fill=TEXT3, anchor="middle")
    svg.text((x0 + x1) / 2, ty + 20, AXIS_TITLE, size=12, fill=TEXT2, anchor="middle")
    return ty + 20 + PAD


def offsets(groups, span, side="both"):
    d = 2 * R + GAP
    res = [swarm([v * span for v in vals], d, side) for _, vals in groups]
    ext = max(max(abs(y) for y in ys) for ys in res) + R
    return res, ext


def label_style(name):
    rec = name == RECOMMENDED
    return (TEXT if rec else TEXT2), (700 if rec else 400)


# ---------------------------------------------------------------- versions

def v1(cls, groups, band=True, summary=False):
    """Left labels, centered swarm, shared gridlines, recommended row on a light band."""
    svg = Svg()
    top = header(svg) + 6
    lw = max(text_width(n, 12) for n, _ in groups)
    x0 = PAD + lw + 14 + R
    x1 = W - PAD - R
    ys, ext = offsets(groups, x1 - x0)
    extra = 0
    row = 2 * ext + 12 + (6 if summary else 0)
    grid_top = top
    rows_y = []
    for i in range(len(groups)):
        rows_y.append(top + row * i + row / 2 - extra / 2)
    bottom = top + row * len(groups)
    if band:
        for (name, _), cy in zip(groups, rows_y):
            if name == RECOMMENDED:
                svg.rect(PAD - 6, cy - row / 2 + extra / 2 + 1, W - 2 * PAD + 12, row - 2, CARD, rx=4)
    for v in (0, 0.5, 1):
        x = x0 + (x1 - x0) * v
        svg.line(x, grid_top, x, bottom, LINE_L, 1)
    for (name, vals), cy, yo in zip(groups, rows_y, ys):
        fill, weight = label_style(name)
        svg.text(x0 - R - 14, cy, name, fill=fill, weight=weight, anchor="end", baseline="central")
        svg.line(x0 + min(vals) * (x1 - x0), cy, x0 + max(vals) * (x1 - x0), cy, LINE_M, 1)
        if summary:
            q1, med, q3 = (quantile(vals, q) for q in (0.25, 0.5, 0.75))
            own = max(abs(y) for y in yo) + R + 4
            qx0, qx1 = x0 + q1 * (x1 - x0), x0 + q3 * (x1 - x0)
            svg.rect(qx0, cy - own, qx1 - qx0, 2 * own, LINE_L, rx=3)
            mx = x0 + med * (x1 - x0)
            svg.line(mx, cy - own, mx, cy + own, LINE_M, 1.6)
        for v, y, c in zip(vals, yo, cls):
            svg.dot(x0 + v * (x1 - x0), cy - y, c)
    h = x_axis(svg, x0, x1, grid_top, bottom, grid=False)
    return svg.render(h)


def main():
    cls, groups = load()
    s = v1(cls, groups)
    OUT.write_text(s, encoding="utf-8")
    print(OUT, s.split('height="', 1)[1].split('"', 1)[0])


if __name__ == "__main__":
    main()
