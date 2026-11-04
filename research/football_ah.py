# -*- coding: utf-8 -*-
"""足球讓球（半球盤，跟 DK / 運彩一樣）：每場挑模型期望值較高的一邊，什麼條件下長期接近打平或賺錢？

模型 = 正式版：power 去水 + 主隊截距。依期望值區間、讓球深度、熱門 / 冷門、主 / 客分組，逐季看 ROI（市場平均收盤）。
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


def summary(name, d):
    if not len(d):
        say(f"- {name}：0 注")
        return
    prof = d.profit.to_numpy(float)
    by = d.groupby("season").profit.agg(["size", "mean"])
    t = prof.mean() / (prof.std(ddof=1) / math.sqrt(len(prof))) if len(prof) > 2 and prof.std() > 0 else 0
    p = 0.5 * math.erfc(t / math.sqrt(2))
    say(f"- {name}：{len(prof)} 注 ROI {prof.mean()*100:+.1f}%（p={p:.3f}）｜命中 {(prof > 0).mean()*100:.0f}%｜{(by['mean'] > 0).sum()}/{len(by)} 季為正｜"
        + "、".join(f"{s} {r['mean']*100:+.0f}%" for s, r in by.iterrows()))


fb = data_football.load_history().dropna(subset=["c_ah_line", "c_ah_h", "c_ah_a", "c_avg_h", "c_avg_a"])
fb = fb[(fb.c_ah_line * 2 % 1 == 0) & (fb.c_ah_line % 1 != 0) & (fb.c_ah_h > 1) & (fb.c_ah_a > 1)].reset_index(drop=True)
a = cal.coefs().get("football_spread") or -0.066
ph = cal._power_matrix(fb[["c_ah_h", "c_ah_a"]].to_numpy(float))[:, 0]
ph = 1 / (1 + np.exp(-(a + np.log(ph / (1 - ph)))))
ev_h, ev_a = ph * fb.c_ah_h - 1, (1 - ph) * fb.c_ah_a - 1
home = ev_h >= ev_a
mg = (fb.hg - fb.ag).to_numpy(float)
line = fb.c_ah_line.to_numpy(float)
profit = np.where(home, [settle_handicap(m, l, o) for m, l, o in zip(mg, line, fb.c_ah_h)],
                  [settle_handicap(-m, -l, o) for m, l, o in zip(mg, line, fb.c_ah_a)])
d = pd.DataFrame({"season": fb.season, "ev": np.where(home, ev_h, ev_a), "profit": profit, "home": home,
                  "fav": np.where(home, line < 0, line > 0), "depth": np.abs(line), "odds": np.where(home, fb.c_ah_h, fb.c_ah_a),
                  "league": fb.league})
say(f"# 足球讓球（半球盤）每場挑期望值較高的一邊：{len(d)} 場、{d.season.nunique()} 季（主隊截距 {a:+.3f}）")
summary("全部", d)
say("\n## 依模型期望值")
for lo, hi in ((0.0, 1), (-0.01, 0.0), (-0.02, -0.01), (-0.03, -0.02), (-1, -0.03)):
    summary(f"期望值 {lo*100:+.0f}% ~ {hi*100:+.0f}%", d[(d.ev >= lo) & (d.ev < hi)])
say("\n## 依選到的一邊")
summary("選到客隊", d[~d.home]); summary("選到主隊", d[d.home])
summary("選到受讓方（冷門）", d[~d.fav]); summary("選到讓球方（熱門）", d[d.fav])
say("\n## 依讓球深度")
for dep in (0.5, 1.5, 2.5):
    summary(f"讓 {dep:g}", d[d.depth == dep])
summary("讓 3.5 以上", d[d.depth >= 3.5])
say("\n## 依聯賽")
for lg, g in d.groupby("league"):
    summary(lg, g)
say("\n## 組合：客隊受讓 0.5 / 1.5")
summary("客隊受讓（客隊是冷門）", d[~d.home & ~d.fav])
summary("客隊受讓 0.5", d[~d.home & ~d.fav & (d.depth == 0.5)])
summary("客隊受讓 1.5", d[~d.home & ~d.fav & (d.depth == 1.5)])

(OUT / "football_ah.md").write_text("\n".join(LINES), encoding="utf-8")
