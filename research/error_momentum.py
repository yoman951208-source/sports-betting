# -*- coding: utf-8 -*-
"""市場的誤差會不會延續？（10/28 使用者：實際勝率比模型高的類型，是不是可以多推薦）

做法：同一聯賽、同一季，前 k 場「實際 − 市場機率」的平均（例如這季進球一直比盤口多），
當作下一場的特徵，看能不能讓預測更準。逐季留一，跟「只重新校準」比（公平比較）。
另外算直接下注：前 k 場誤差 > 門檻時，跟著買（例如大分）的各季 ROI。
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
import data_nba  # noqa: E402

OUT = pathlib.Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
LINES: list[str] = []


def say(s=""):
    print(s, flush=True)
    LINES.append(s)


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def bll(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def prior_mean(df, group, resid, k, min_n):
    """同一組（聯賽-季）之前 k 場的平均誤差（不含這場）。df 已依日期排序。"""
    return df.groupby(group)[resid].transform(lambda s: s.shift().rolling(k, min_periods=min_n).mean())


def test(name, df, y, p, feat, odds=None, thr=0.05):
    d = pd.DataFrame({"s": df["season"].to_numpy(), "y": np.asarray(y, float), "lp": logit(p), "f": np.asarray(feat, float),
                      "o": np.nan if odds is None else np.asarray(odds, float)}).dropna(subset=["y", "lp", "f"])
    out, better = [], 0
    for S in sorted(d.s.unique()):
        te = (d.s == S).to_numpy()
        if te.sum() < 300:
            continue
        w0 = cal._fit(d.lp.to_numpy()[~te, None], d.y.to_numpy()[~te])
        w1 = cal._fit(d[["lp", "f"]].to_numpy()[~te], d.y.to_numpy()[~te])
        p0 = 1 / (1 + np.exp(-(w0[0] + w0[1] * d.lp.to_numpy()[te])))
        p1 = 1 / (1 + np.exp(-(w1[0] + d[["lp", "f"]].to_numpy()[te] @ np.array(w1[1:]))))
        diff = bll(p1, d.y.to_numpy()[te]) - bll(p0, d.y.to_numpy()[te])
        better += diff.mean() < 0
        out.append(f"{S} {diff.mean()*1000:+.2f}‰")
    w = cal._fit(d[["lp", "f"]].to_numpy(), d.y.to_numpy())
    line = f"- **{name}**：{better}/{len(out)} 季改善｜特徵係數 {w[2]:+.2f}｜" + "　".join(out)
    if odds is not None:
        sel = d[(d.f >= thr) & d.o.notna()]
        prof = np.where(sel.y > 0, sel.o - 1, -1.0)
        by = pd.Series(prof).groupby(sel.s.to_numpy()).mean()
        line += (f"\n  　跟著買（前段誤差 ≥ {thr*100:.0f}pp）：{len(sel)} 注 ROI {prof.mean()*100:+.1f}%｜{(by > 0).sum()}/{len(by)} 季為正"
                 if len(sel) else "\n  　（沒有符合的場次）")
    say(line)


# ----------------------------------------------------------------------------- 足球
fb = data_football.load_history().dropna(subset=["c_avg_h", "c_avg_d", "c_avg_a", "c_avg_over", "c_avg_under"])
fb = fb.sort_values(["date"]).reset_index(drop=True)
ok = (fb[["c_avg_h", "c_avg_d", "c_avg_a", "c_avg_over", "c_avg_under"]] > 1).all(axis=1)
fb = fb[ok].reset_index(drop=True)
p3 = cal._power_matrix(fb[["c_avg_h", "c_avg_d", "c_avg_a"]].to_numpy(float))
fb["pH"], fb["pD"], fb["pA"] = p3[:, 0], p3[:, 1], p3[:, 2]
fb["pO"] = cal._power_matrix(fb[["c_avg_over", "c_avg_under"]].to_numpy(float))[:, 0]
fb["yO"] = ((fb.hg + fb.ag) >= 3).astype(float)
fb["yD"] = (fb.ftr == "D").astype(float)
fb["yH"] = (fb.ftr == "H").astype(float)
fb["yA"] = (fb.ftr == "A").astype(float)
fb["grp"] = fb["div"] + fb["season"]
for col in ("O", "D", "H", "A"):
    fb[f"r{col}"] = fb[f"y{col}"] - fb[f"p{col}"]
say(f"# 市場誤差會不會延續（足球 {len(fb)} 場、8 季）")
for k in (30, 80):
    say(f"\n## 足球：同一聯賽本季前 {k} 場的誤差")
    test(f"大2.5（前 {k} 場 實際大分率 − 市場）", fb, fb.yO, fb.pO, prior_mean(fb, "grp", "rO", k, k // 2), fb.c_avg_over)
    test(f"小2.5（同上，反向）", fb, 1 - fb.yO, 1 - fb.pO, -prior_mean(fb, "grp", "rO", k, k // 2), fb.c_avg_under)
    test(f"和局（前 {k} 場 實際和局率 − 市場）", fb, fb.yD, fb.pD, prior_mean(fb, "grp", "rD", k, k // 2), fb.c_avg_d)
    test(f"主勝（前 {k} 場 實際主勝率 − 市場）", fb, fb.yH, fb.pH, prior_mean(fb, "grp", "rH", k, k // 2), fb.c_avg_h)
    test(f"客勝（前 {k} 場 實際客勝率 − 市場）", fb, fb.yA, fb.pA, prior_mean(fb, "grp", "rA", k, k // 2), fb.c_avg_a)
# 亞盤主隊贏盤（整數 / 半球盤）
ah = fb.dropna(subset=["c_ah_line", "c_ah_h", "c_ah_a"]).copy()
ah = ah[(ah.c_ah_line * 2) % 1 == 0]
m = ah.hg - ah.ag + ah.c_ah_line
ah = ah[m != 0].copy()
ah["y"] = ((ah.hg - ah.ag + ah.c_ah_line) > 0).astype(float)
ah["p"] = cal._power_matrix(ah[["c_ah_h", "c_ah_a"]].to_numpy(float))[:, 0]
ah["r"] = ah.y - ah.p
for k in (30, 80):
    test(f"亞盤主隊贏盤（前 {k} 場 實際 − 市場）", ah, ah.y, ah.p, prior_mean(ah, "grp", "r", k, k // 2), ah.c_ah_h)

# ----------------------------------------------------------------------------- NBA
nb = data_nba.load_history().dropna(subset=["c_spread_home", "c_total"]).sort_values(["game_date", "event_id"]).reset_index(drop=True)
mg = nb.home_score - nb.away_score
cov = mg + nb.c_spread_home
ov = nb.home_score + nb.away_score - nb.c_total
nb["yC"] = np.where(cov > 0, 1.0, np.where(cov < 0, 0.0, np.nan))
nb["yO"] = np.where(ov > 0, 1.0, np.where(ov < 0, 0.0, np.nan))
nb["pC"] = cal._power_matrix(nb[["c_spread_home_odds", "c_spread_away_odds"]].fillna(1.91).to_numpy(float))[:, 0]
nb["pO"] = cal._power_matrix(nb[["c_over_odds", "c_under_odds"]].fillna(1.91).to_numpy(float))[:, 0]
nb["rC"], nb["rO"] = nb.yC - nb.pC, nb.yO - nb.pO
nb["grp"] = nb.season
say(f"\n# NBA（{len(nb)} 場、{nb.season.nunique()} 季）：全聯盟本季前 k 場的誤差")
for k in (100, 250):
    test(f"主隊讓分贏盤（前 {k} 場 實際 − 市場）", nb, nb.yC, nb.pC, prior_mean(nb, "grp", "rC", k, k // 2), nb.c_spread_home_odds)
    test(f"客隊讓分贏盤（同上，反向）", nb, 1 - nb.yC, 1 - nb.pC, -prior_mean(nb, "grp", "rC", k, k // 2), nb.c_spread_away_odds)
    test(f"大分（前 {k} 場 實際 − 市場）", nb, nb.yO, nb.pO, prior_mean(nb, "grp", "rO", k, k // 2), nb.c_over_odds)
    test(f"小分（同上，反向）", nb, 1 - nb.yO, 1 - nb.pO, -prior_mean(nb, "grp", "rO", k, k // 2), nb.c_under_odds)
# 各隊自己的誤差延續（這隊本季打出盤口的比例）
long = pd.concat([pd.DataFrame({"i": nb.index, "season": nb.season, "team": nb.home, "r": nb.rC}),
                  pd.DataFrame({"i": nb.index, "season": nb.season, "team": nb.away, "r": -nb.rC})]).sort_values("i")
long["f"] = long.groupby(["season", "team"])["r"].transform(lambda s: s.shift().rolling(15, min_periods=8).mean())
fh = long.drop_duplicates("i", keep="first").set_index("i")["f"]
fa = long.drop_duplicates("i", keep="last").set_index("i")["f"]
test("各隊本季前 15 場打出盤口（主 − 客）", nb, nb.yC, nb.pC, (fh - fa).reindex(nb.index), nb.c_spread_home_odds, thr=0.25)

(OUT / "error_momentum.md").write_text("\n".join(LINES), encoding="utf-8")
