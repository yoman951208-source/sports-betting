# -*- coding: utf-8 -*-
"""足球：哪裡找得到可以買的盤？（11/05 使用者：推薦的命中率不低，但全部 🔴 不能買）

A. 「每場最不虧」選法（跟最推薦一樣：大小、讓球、賠率 ≥1.5 的不讓分）長期 ROI，依賠率區間、盤口、季別。
B. 盤口之間的矛盾（勝平負、大小、讓球各自定價）：
   B1 大小 vs 勝平負：勝平負的和局機率暗示的進球數，跟大小盤不一致時，買勝平負那邊暗示的方向
   B2 讓球 vs 比分模型：用勝平負 + 大小算出的讓球贏盤機率（Dixon-Coles + 淨勝校準），跟讓球盤的價格比，
      期望值 ≥ 門檻就買
全部用市場平均收盤賠率結算（運彩通常更低），另附各家最高價當作上限參考。逐季看是否穩定。
"""
from __future__ import annotations

import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import analyze  # noqa: E402
import calibration as cal  # noqa: E402
import data_football  # noqa: E402
from strategies import lam_from_over, settle_handicap  # noqa: E402

OUT = pathlib.Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
LINES: list[str] = []


def say(s=""):
    print(s, flush=True)
    LINES.append(s)


def summary(name, prof, seasons):
    prof, seasons = np.asarray(prof, float), np.asarray(seasons)
    if not len(prof):
        say(f"- {name}：0 注")
        return
    by = pd.Series(prof).groupby(seasons).agg(["size", "mean"])
    t = prof.mean() / (prof.std(ddof=1) / math.sqrt(len(prof))) if len(prof) > 2 and prof.std() > 0 else 0
    p = 0.5 * math.erfc(t / math.sqrt(2))
    say(f"- {name}：{len(prof)} 注 ROI {prof.mean()*100:+.1f}%（p={p:.3f}）｜{(by['mean'] > 0).sum()}/{len(by)} 季為正｜"
        + "、".join(f"{s} {r['mean']*100:+.0f}%({int(r['size'])})" for s, r in by.iterrows()))


fb = data_football.load_history()
need = ["c_avg_h", "c_avg_d", "c_avg_a", "c_avg_over", "c_avg_under"]
fb = fb.dropna(subset=need)
fb = fb[(fb[need] > 1).all(axis=1)].sort_values("date").reset_index(drop=True)
a_int = cal.coefs().get("football_spread") or -0.066
say(f"# 足球：哪裡找得到可以買的盤（{len(fb)} 場、{fb.season.nunique()} 季；讓球主隊截距 {a_int:+.3f}）")

rows = []
for r in fb.itertuples():
    pc = cal.power_devig([r.c_avg_h, r.c_avg_d, r.c_avg_a])
    po = cal.power_devig([r.c_avg_over, r.c_avg_under])
    if not pc or not po:
        continue
    lam = lam_from_over(po[0], 2.5)
    gm = analyze.goal_model(tuple(pc), lam)
    rows.append((r, pc, po, lam, gm))
say(f"可用 {len(rows)} 場")

# ----------------------------------------------------------------------------- A
picks = []
for r, pc, po, lam, gm in rows:
    goals, mg = r.hg + r.ag, r.hg - r.ag
    opts = []   # (名稱, 賠率, 估計勝率, 損益, 最高價損益)
    for k, nm, o, mx in ((0, "主勝", r.c_avg_h, r.c_max_h), (1, "和局", r.c_avg_d, r.c_max_d), (2, "客勝", r.c_avg_a, r.c_max_a)):
        if o >= 1.5:
            win = (r.ftr == "HDA"[k])
            opts.append(("勝平負", nm, o, pc[k], (o - 1) if win else -1.0, ((mx - 1) if win else -1.0) if mx == mx else np.nan))
    ov = goals >= 3
    opts.append(("大小", "大2.5", r.c_avg_over, po[0], (r.c_avg_over - 1) if ov else -1.0, np.nan))
    opts.append(("大小", "小2.5", r.c_avg_under, po[1], (r.c_avg_under - 1) if not ov else -1.0, np.nan))
    line = r.c_ah_line
    if line == line and r.c_ah_h == r.c_ah_h and r.c_ah_a == r.c_ah_a and (line * 2) % 1 == 0 and (line % 1) != 0:  # 半球盤（跟 DK 一樣）
        ph = cal._power_matrix(np.array([[r.c_ah_h, r.c_ah_a]]))[0, 0]
        ph = 1 / (1 + math.exp(-(a_int + math.log(ph / (1 - ph)))))
        opts.append(("讓球", "讓球主", r.c_ah_h, ph, settle_handicap(mg, line, r.c_ah_h), np.nan))
        opts.append(("讓球", "讓球客", r.c_ah_a, 1 - ph, settle_handicap(-mg, -line, r.c_ah_a), np.nan))
    best = max(opts, key=lambda x: x[3] * x[2] - 1)
    picks.append({"season": r.season, "market": best[0], "name": best[1], "odds": best[2], "p": best[3], "ev": best[3] * best[2] - 1,
                  "profit": best[4], "win": best[4] > 0})
pk = pd.DataFrame(picks)
say("\n## A. 每場「最不虧」選項（跟最推薦同樣的選法），市場平均收盤賠率")
summary("全部", pk.profit, pk.season)
say(f"  命中 {pk.win.mean()*100:.1f}%｜模型平均勝率 {pk.p.mean()*100:.1f}%｜平均賠率 {pk.odds.mean():.2f}｜模型平均期望值 {pk.ev.mean()*100:+.1f}%")
for lo, hi in ((1.0, 1.6), (1.6, 2.0), (2.0, 2.6), (2.6, 10)):
    s = pk[(pk.odds >= lo) & (pk.odds < hi)]
    summary(f"賠率 {lo}–{hi}（命中 {s.win.mean()*100:.0f}% vs 模型 {s.p.mean()*100:.0f}%）", s.profit, s.season)
for mk in ("大小", "讓球", "勝平負"):
    s = pk[pk.market == mk]
    summary(f"盤口：{mk}", s.profit, s.season)

# ----------------------------------------------------------------------------- B1
say("\n## B1. 大小 vs 勝平負：勝平負的和局機率比大小盤推出的高 → 勝平負覺得進球少")
d_draw = np.array([pc[1] - sum(p for d, p in gm["margin"].items() if d == 0) for r, pc, po, lam, gm in rows])
season = np.array([r.season for r, *_ in rows])
over = np.array([(r.hg + r.ag) >= 3 for r, *_ in rows])
o_over = np.array([r.c_avg_over for r, *_ in rows]); o_under = np.array([r.c_avg_under for r, *_ in rows])
mo_over = np.array([r.c_max_over if hasattr(r, "c_max_over") else np.nan for r, *_ in rows])
say(f"差距分布：中位數 {np.median(d_draw)*100:+.1f}pp、10% {np.percentile(d_draw, 10)*100:+.1f}pp、90% {np.percentile(d_draw, 90)*100:+.1f}pp")
for q in (0.05, 0.10, 0.20):                     # 用分位數：矛盾最大的前 q（市場彼此很一致，固定門檻選不到比賽）
    u = d_draw >= np.quantile(d_draw, 1 - q)
    o = d_draw <= np.quantile(d_draw, q)
    summary(f"勝平負覺得進球最少的前 {q*100:.0f}% → 買小2.5", np.where(~over[u], o_under[u] - 1, -1.0), season[u])
    summary(f"勝平負覺得進球最多的前 {q*100:.0f}% → 買大2.5", np.where(over[o], o_over[o] - 1, -1.0), season[o])

# ----------------------------------------------------------------------------- B2
say("\n## B2. 讓球盤 vs 比分模型（勝平負 + 大小推出的讓球機率；全部讓球盤，含四分之一盤）")
ev_h, ev_a, pr_h, pr_a, s_ah, half = [], [], [], [], [], []
for r, pc, po, lam, gm in rows:
    line = r.c_ah_line
    if not (line == line and r.c_ah_h == r.c_ah_h and r.c_ah_a == r.c_ah_a):
        continue
    mg = r.hg - r.ag
    eh = sum(p * settle_handicap(d, line, r.c_ah_h) for d, p in gm["margin"].items())
    ea = sum(p * settle_handicap(-d, -line, r.c_ah_a) for d, p in gm["margin"].items())
    ev_h.append(eh); ev_a.append(ea)
    pr_h.append(settle_handicap(mg, line, r.c_ah_h)); pr_a.append(settle_handicap(-mg, -line, r.c_ah_a))
    s_ah.append(r.season); half.append((line * 2) % 1 == 0 and (line % 1) != 0)
ev_h, ev_a, pr_h, pr_a, s_ah, half = map(np.array, (ev_h, ev_a, pr_h, pr_a, s_ah, half))
say(f"{len(ev_h)} 場；比分模型算的期望值（市場平均價）：主 中位數 {np.median(ev_h)*100:+.1f}%、客 {np.median(ev_a)*100:+.1f}%")
for q in (0.05, 0.10, 0.20):                     # 模型期望值最高的前 q（不論正負）：比分模型的排序有沒有資訊
    cut = np.quantile(np.concatenate([ev_h, ev_a]), 1 - q)
    bh, ba = ev_h >= cut, ev_a >= cut
    prof = np.concatenate([pr_h[bh], pr_a[ba]]); ss = np.concatenate([s_ah[bh], s_ah[ba]])
    summary(f"模型期望值最高的前 {q*100:.0f}%（≥{cut*100:+.1f}%；主 {bh.sum()}、客 {ba.sum()}）", prof, ss)
    hb, hba = bh & half, ba & half
    summary("　只看半球盤", np.concatenate([pr_h[hb], pr_a[hba]]), np.concatenate([s_ah[hb], s_ah[hba]]))
summary("對照：讓球盤全部都買（主客各一）", np.concatenate([pr_h, pr_a]), np.concatenate([s_ah, s_ah]))

(OUT / "football_value.md").write_text("\n".join(LINES), encoding="utf-8")
