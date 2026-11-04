# -*- coding: utf-8 -*-
"""讓球主隊截距要不要依讓球深度分開？（11/05：讓 1.5 / 2.5 買客隊的長期 ROI 比讓 0.5 好）

正式版：logit(主隊贏盤) = a + logit(power 去水)（整數 / 半球盤，排除走盤）。
比較：再加「|讓球| ≥ 1.5」的額外截距 c。逐季留一，跟正式版比 log loss；並看兩種模型下「挑期望值較高一邊」的半球盤 ROI。
"""
from __future__ import annotations

import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import calibration as cal  # noqa: E402
import data_football  # noqa: E402
from strategies import settle_handicap  # noqa: E402

OUT = pathlib.Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
LINES: list[str] = []


def say(s=""):
    print(s, flush=True)
    LINES.append(s)


def fit_offset(X, off, y, iters=40):
    """logit(p) = off + X @ w（off 的係數固定 1）。"""
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(off + X @ w)))
        g = X.T @ (p - y)
        H = (X * (p * (1 - p))[:, None]).T @ X + 1e-6 * np.eye(X.shape[1])
        w -= np.linalg.solve(H, g)
    return w


fb = data_football.load_history().dropna(subset=["c_ah_line", "c_ah_h", "c_ah_a"])
fb = fb[(fb.c_ah_line * 2 % 1 == 0) & (fb.c_ah_h > 1) & (fb.c_ah_a > 1)].reset_index(drop=True)
m = (fb.hg - fb.ag + fb.c_ah_line).to_numpy(float)
fb, m = fb[m != 0].reset_index(drop=True), m[m != 0]
y = (m > 0).astype(float)
lp = np.log(cal._power_matrix(fb[["c_ah_h", "c_ah_a"]].to_numpy(float))[:, 0])
lp = lp - np.log(1 - np.exp(lp))
deep = (fb.c_ah_line.abs() >= 1.5).to_numpy(float)
X0 = np.ones((len(fb), 1))
X1 = np.column_stack([np.ones(len(fb)), deep])
season = fb.season.to_numpy()
say(f"# 讓球主隊截距依深度分開？（{len(fb)} 場整數 / 半球盤，深盤 {int(deep.sum())} 場）")
w_all = fit_offset(X1, lp, y)
say(f"全部資料估計：一般截距 {w_all[0]:+.3f}、深盤再加 {w_all[1]:+.3f}（正式版只用一個截距 {fit_offset(X0, lp, y)[0]:+.3f}）")
out, better = [], 0
for S in sorted(set(season)):
    te = season == S
    w0, w1 = fit_offset(X0[~te], lp[~te], y[~te]), fit_offset(X1[~te], lp[~te], y[~te])
    p0 = 1 / (1 + np.exp(-(lp[te] + X0[te] @ w0)))
    p1 = 1 / (1 + np.exp(-(lp[te] + X1[te] @ w1)))
    ll = lambda p: -(y[te] * np.log(p) + (1 - y[te]) * np.log(1 - p))
    d = ll(p1) - ll(p0)
    dd = d[deep[te] == 1]
    better += d.mean() < 0
    out.append(f"{S} 全部 {d.mean()*1000:+.2f}‰／深盤 {dd.mean()*1000:+.2f}‰（深盤再加 {w1[1]:+.3f}）")
say(f"逐季留一：{better}/{len(out)} 季改善")
for o in out:
    say(f"- {o}")

# 半球盤：兩種模型「挑期望值較高的一邊」的 ROI（逐季留一估計的係數）
half = ((fb.c_ah_line * 2 % 1 == 0) & (fb.c_ah_line % 1 != 0)).to_numpy()
rows = []
for S in sorted(set(season)):
    te = (season == S) & half
    tr = season != S
    for name, X in (("正式版", X0), ("深盤分開", X1)):
        w = fit_offset(X[tr], lp[tr], y[tr])
        ph = 1 / (1 + np.exp(-(lp[te] + X[te] @ w)))
        f = fb[te]
        ev_h, ev_a = ph * f.c_ah_h.to_numpy() - 1, (1 - ph) * f.c_ah_a.to_numpy() - 1
        mg = (f.hg - f.ag).to_numpy(float); ln = f.c_ah_line.to_numpy(float)
        for eh, ea, mm, l, oh, oa in zip(ev_h, ev_a, mg, ln, f.c_ah_h, f.c_ah_a):
            home = eh >= ea
            ev = max(eh, ea)
            prof = settle_handicap(mm, l, oh) if home else settle_handicap(-mm, -l, oa)
            rows.append((name, S, ev, prof, abs(l) >= 1.5))
r = pd.DataFrame(rows, columns=["model", "season", "ev", "profit", "deep"])
say("\n## 半球盤：每場挑期望值較高的一邊（市場平均收盤賠率）")
for name, g in r.groupby("model"):
    for lab, s in (("全部", g), ("期望值 ≥ 0", g[g.ev >= 0]), ("期望值 ≥ +2%", g[g.ev >= 0.02]), ("深盤", g[g.deep])):
        if not len(s):
            continue
        by = s.groupby("season").profit.mean()
        say(f"- {name}｜{lab}：{len(s)} 注 ROI {s.profit.mean()*100:+.1f}%｜{(by > 0).sum()}/{len(by)} 季為正")

(OUT / "football_ah_depth.md").write_text("\n".join(LINES), encoding="utf-8")
