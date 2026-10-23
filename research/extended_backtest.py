# -*- coding: utf-8 -*-
"""更多資料的回測（2026-10-23）：足球 2019-20 起 8 季、NBA 2016-17 起 11 季、MLB 2019 起 7 季（不含 2020 縮水季）。

1. 資料覆蓋率
2. 策略回測：原本的季別 vs 新增的季別（真正的樣本外）vs 全部
3. 校準層：逐季留一交叉驗證（季別變多，結論更可靠）
4. 歷史相似盤口該給多少權重（shrink）：逐季留一的 log loss，以及「每場最佳選項 EV ≥ 2%」推薦的實際 ROI
5. 莊家角度逐季：主客、冷熱門、讓球盤主隊
6. NBA 動能（上一場大勝 / 打出盤口後，跟買或反買）逐季
7. NBA 球員狀態（疲勞、缺陣）加入 2023-24 後重新驗證
"""
from __future__ import annotations

import math
import pathlib
import sys
import time
import traceback

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import calibration as cal  # noqa: E402
import config  # noqa: E402
import data_football  # noqa: E402
import data_mlb  # noqa: E402
import data_nba  # noqa: E402

FB_SEASONS = ["1920", "2021", "2122", "2223", "2324", "2425", "2526", "2627"]
NBA_EXTRA = [("2016-17", "2016-10-25", "2017-06-12"), ("2017-18", "2017-10-17", "2018-06-08"),
             ("2018-19", "2018-10-16", "2019-06-13"), ("2019-20", "2019-10-22", "2020-10-11"),
             ("2020-21", "2020-12-22", "2021-07-20"), ("2021-22", "2021-10-19", "2022-06-16"),
             ("2022-23", "2022-10-18", "2023-06-12"), ("2023-24", "2023-10-24", "2024-06-17")]
MLB_EXTRA = [("2019", "2019-03-20", "2019-10-31"), ("2021", "2021-03-31", "2021-11-03"),
             ("2022", "2022-04-06", "2022-11-06"), ("2023", "2023-03-29", "2023-11-02")]
ORIG = {"football": {"2024-25", "2025-26", "2026-27"}, "nba": {"2024-25", "2025-26", "2026-27"}, "mlb": {"2024", "2025", "2026"}}

data_football.FOOTBALL_SEASONS = config.FOOTBALL_SEASONS = FB_SEASONS
if not any(s[0] == "2016-17" for s in data_nba.NBA_SEASONS):
    data_nba.NBA_SEASONS = config.NBA_SEASONS = NBA_EXTRA + list(data_nba.NBA_SEASONS)
if not any(s[0] == "2019" for s in data_mlb.MLB_SEASONS):
    data_mlb.MLB_SEASONS = MLB_EXTRA + list(data_mlb.MLB_SEASONS)

OUT = pathlib.Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
REPORT: list[str] = []
T0 = time.time()


def say(s: str = "") -> None:
    print(s, flush=True)
    REPORT.append(s)


def save() -> None:
    (OUT / "extended_backtest.md").write_text("\n".join(REPORT), encoding="utf-8")


def section(title: str):
    def deco(fn):
        def run(*a, **k):
            say(f"\n## {title}（{time.time() - T0:.0f}s）")
            try:
                return fn(*a, **k)
            except Exception:
                say("```\n" + traceback.format_exc() + "\n```")
            finally:
                save()
        return run
    return deco


def sig(x):
    return 1 / (1 + np.exp(-x))


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def bll(p, y):
    p = np.clip(np.asarray(p, float), 0.02, 0.98)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def tstat(d):
    d = np.asarray(d, float)
    return d.mean() / (d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 2 and d.std() > 0 else 0.0


def pct(x, digits=1):
    return f"{x * 100:+.{digits}f}%"


def two_way_power(o1, o2):
    o = np.column_stack([np.asarray(o1, float), np.asarray(o2, float)])
    ok = np.isfinite(o).all(axis=1) & (o > 1).all(axis=1)
    p = np.full(len(o), np.nan)
    if ok.any():
        p[ok] = cal._power_matrix(o[ok])[:, 0]
    return p


def prop2(o1, o2):
    a, b = 1 / np.asarray(o1, float), 1 / np.asarray(o2, float)
    return a / (a + b)


def ah_sign(margin, line):
    """亞洲讓球（含四分之一盤）的輸贏方向：+1 贏（含贏半）、0 走盤、−1 輸（含輸半）。"""
    margin, line = np.asarray(margin, float), np.asarray(line, float)
    quarter = np.abs((line * 4) % 2 - 1) < 1e-9
    whole = np.sign(margin + line)
    q = np.sign(np.sign(margin + line - 0.25) + np.sign(margin + line + 0.25))
    return np.where(quarter, q, whole)


# =============================================================================
# 資料
# =============================================================================
fb = data_football.load_history().reset_index(drop=True)
nba = data_nba.load_history().reset_index(drop=True)
mlb = data_mlb.load_history().reset_index(drop=True)
say(f"# 更多資料的回測（{time.strftime('%Y-%m-%d %H:%M')}）")


@section("1. 資料覆蓋率")
def coverage():
    say("| 足球季別 | 場數 | 收盤勝平負 | 亞盤收盤 | 大小收盤 | 開盤勝平負 |")
    say("|---|---|---|---|---|---|")
    for s, g in fb.groupby("season"):
        say(f"| {s} | {len(g)} | {g.c_avg_h.notna().mean():.0%} | {g.c_ah_h.notna().mean():.0%} | "
            f"{g.c_avg_over.notna().mean():.0%} | {g.o_avg_h.notna().mean():.0%} |")
    for name, df, cols in (("NBA", nba, ["c_spread_home", "c_ml_home", "c_total", "o_spread_home", "o_ml_home"]),
                           ("MLB", mlb, ["c_ml_home", "c_spread_home", "c_total", "o_ml_home", "o_total"])):
        say(f"\n| {name} 季別 | 場數 | " + " | ".join(cols) + " | 主要公司 |")
        say("|---|---|" + "---|" * len(cols) + "---|")
        for s, g in df.groupby("season"):
            prov = g["provider"].value_counts().index[0] if g["provider"].notna().any() else "-"
            say(f"| {s} | {len(g)} | " + " | ".join(f"{g[c].notna().mean():.0%}" if c in g else "0%" for c in cols) + f" | {prov} |")


# =============================================================================
# 2. 策略
# =============================================================================
@section("2. 策略回測：原本季別 vs 新增季別（樣本外）vs 全部")
def strategies_section():
    import json
    from backtest import evaluate, run_bets, select
    from strategies import football_strategies, mlb_strategies, nba_strategies
    fs, ns, ms = football_strategies(), nba_strategies(), mlb_strategies()
    bets = {**run_bets(fs, fb), **run_bets(ns, nba), **run_bets(ms, mlb)}
    strats = fs + ns + ms
    old = {}
    try:
        for s in json.loads((ROOT / "data" / "strategies.json").read_text(encoding="utf-8"))["strategies"]:
            old[s["sid"]] = ("⭐" if s["enabled"] else "☆" if s["weak"] else "")
    except Exception:
        pass
    rows = []
    for s in strats:
        b = bets.get(s.sid, pd.DataFrame())
        s.stats = evaluate(b) if len(b) else {"n": 0}
        if not len(b):
            continue
        o, n = b[b.season.isin(ORIG[s.sport])], b[~b.season.isin(ORIG[s.sport])]
        rows.append((s, evaluate(o) if len(o) else {"n": 0}, evaluate(n) if len(n) else {"n": 0}))
    select(strats)

    def cell(st):
        return f"{st['n']} 注 {pct(st['roi'])} (p={st['p']:.2f})" if st.get("n") else "-"

    say("原本啟用（⭐）或參考（☆）的策略，在新增季別的表現：")
    say("| 原標記 | 策略 | 原本季別 | 新增季別（樣本外） | 全部 | 全部資料下 |")
    say("|---|---|---|---|---|---|")
    for s, o, n in rows:
        if old.get(s.sid):
            now = "⭐ 啟用" if s.enabled else "☆ 參考" if s.weak else "✖"
            say(f"| {old[s.sid]} | {s.title} | {cell(o)} | {cell(n)} | {cell(s.stats)} | {now} |")
    say("\n全部資料下會被啟用的策略（樣本 ≥ 門檻、p<0.10、每季 ROI 都 > 0）：")
    en = [s for s in strats if s.enabled]
    for s in en:
        bs = "、".join(f"{k} {pct(v['roi'], 0)}" for k, v in s.stats["by_season"].items())
        say(f"- ⭐ {s.title}：{cell(s.stats)}｜各季 {bs}")
    if not en:
        say("- （沒有）")
    say("\n全部資料 t 值最高的 12 個策略：")
    top = sorted([r for r in rows if r[0].stats.get("n", 0) >= 80], key=lambda r: -r[0].stats["t"])[:12]
    for s, o, n in top:
        pos = sum(v["roi"] > 0 for v in s.stats["by_season"].values())
        say(f"- {s.title}：全部 {cell(s.stats)}｜新增季別 {cell(n)}｜{pos}/{len(s.stats['by_season'])} 季為正")


# =============================================================================
# 3. 校準
# =============================================================================
def cv_binary(name, season, y, X, base_lp, min_n=300):
    season, y, X, base_lp = np.asarray(season), np.asarray(y, float), np.asarray(X, float), np.asarray(base_lp, float)
    out, better = [], 0
    for S in sorted(set(season)):
        te = season == S
        if te.sum() < min_n:
            continue
        w = np.array(cal._fit(X[~te], y[~te]))
        p1 = sig(w[0] + X[te] @ w[1:])
        d = bll(p1, y[te]) - bll(sig(base_lp[te]), y[te])
        better += d.mean() < 0
        out.append(f"{S} {d.mean() * 1000:+.2f}‰(t={tstat(d):+.1f})")
    say(f"- **{name}**：{better}/{len(out)} 季改善｜" + "　".join(out))


@section("3. 校準層：逐季留一交叉驗證（‰ = log loss 變化，負的是改善）")
def calibration_section():
    # 足球勝平負（三向）
    x = fb.dropna(subset=["c_avg_h", "c_avg_d", "c_avg_a"]).reset_index(drop=True)
    odds = x[["c_avg_h", "c_avg_d", "c_avg_a"]].to_numpy(float)
    pw = cal._power_matrix(odds)
    inv = 1 / odds
    pr = inv / inv.sum(axis=1, keepdims=True)
    res = x.ftr.map({"H": 0, "D": 1, "A": 2}).to_numpy()
    lp = cal._logit(pw)
    out, better = [], 0
    for S in sorted(x.season.unique()):
        te = (x.season == S).to_numpy()
        if te.sum() < 300:
            continue
        Xs, ys = [], []
        for k in range(3):
            dr = np.full((~te).sum(), float(k == 1))
            Xs.append(np.column_stack([lp[~te, k], dr, lp[~te, k] * dr]))
            ys.append((res[~te] == k).astype(float))
        w = cal._fit(np.vstack(Xs), np.concatenate(ys))
        drm = np.array([0.0, 1.0, 0.0])
        raw = cal._sigmoid(w[0] + w[1] * lp[te] + w[2] * drm + w[3] * lp[te] * drm)
        pc = raw / raw.sum(axis=1, keepdims=True)
        i = np.arange(te.sum())
        d = -np.log(pc[i, res[te]]) + np.log(pw[te][i, res[te]])
        dp = -np.log(pw[te][i, res[te]]) + np.log(pr[te][i, res[te]])
        better += d.mean() < 0
        out.append(f"{S} {d.mean() * 1000:+.2f}‰(t={tstat(d):+.1f}; power 去水 vs 等比例 {dp.mean() * 1000:+.2f}‰)")
    say(f"- **足球勝平負**（目前採用：logit、和局、logit×和局）：{better}/{len(out)} 季改善｜" + "　".join(out))

    # 足球讓球：主隊截距（只用整數 / 半球盤）
    ah = fb.dropna(subset=["c_ah_line", "c_ah_h", "c_ah_a"]).copy()
    ah = ah[(ah.c_ah_line * 2) % 1 == 0]
    m = (ah.hg - ah.ag + ah.c_ah_line).to_numpy(float)
    ah, m = ah[m != 0], m[m != 0]
    lpa = logit(two_way_power(ah.c_ah_h, ah.c_ah_a))
    y = (m > 0).astype(float)
    out, better = [], 0
    for S in sorted(ah.season.unique()):
        te = (ah.season == S).to_numpy()
        if te.sum() < 200:
            continue
        a = cal._fit_intercept(lpa[~te], y[~te])
        d = bll(sig(a + lpa[te]), y[te]) - bll(sig(lpa[te]), y[te])
        better += d.mean() < 0
        out.append(f"{S} 截距{a:+.3f} 主隊實際{y[te].mean() * 100:.1f}%/市場{sig(lpa[te]).mean() * 100:.1f}% {d.mean() * 1000:+.2f}‰")
    say(f"- **足球讓球主隊截距**（10/23 採用）：{better}/{len(out)} 季改善｜" + "　".join(out))

    # 足球大小 2.5
    ou = fb.dropna(subset=["c_avg_over", "c_avg_under"])
    lpo = logit(two_way_power(ou.c_avg_over, ou.c_avg_under))
    cv_binary("足球大小 2.5 校準（目前不採用）", ou.season, ((ou.hg + ou.ag) >= 3).astype(float), lpo[:, None], lpo)

    # NBA
    n = nba.dropna(subset=["c_ml_home", "c_ml_away"])
    lpm = logit(two_way_power(n.c_ml_home, n.c_ml_away))
    cv_binary("NBA 不讓分：只用收盤（全部 11 季）", n.season, (n.home_score > n.away_score).astype(float), lpm[:, None], lpm)
    no = n.dropna(subset=["o_ml_home", "o_ml_away"])
    lpc = logit(two_way_power(no.c_ml_home, no.c_ml_away))
    lpo_ = logit(two_way_power(no.o_ml_home, no.o_ml_away))
    cv_binary("NBA 不讓分：收盤 + 開盤→收盤變化（目前採用；有開盤的季別）", no.season,
              (no.home_score > no.away_score).astype(float), np.column_stack([lpc, lpc - lpo_]), lpc)
    sp = nba.dropna(subset=["c_spread_home"]).copy()
    mm = (sp.home_score - sp.away_score + sp.c_spread_home).to_numpy(float)
    sp, mm = sp[mm != 0], mm[mm != 0]
    lps = logit(two_way_power(sp.c_spread_home_odds.fillna(1.91), sp.c_spread_away_odds.fillna(1.91)))
    cv_binary("NBA 讓分：只用收盤（全部季別）", sp.season, (mm > 0).astype(float), lps[:, None], lps)
    so = sp.o_spread_home.notna().to_numpy()
    dsp = -(sp.c_spread_home - sp.o_spread_home).to_numpy(float)
    cv_binary("NBA 讓分：+ 讓分移動（目前採用；有開盤的季別）", sp.season[so], (mm[so] > 0).astype(float),
              np.column_stack([lps[so], dsp[so]]), lps[so])
    tt = nba.dropna(subset=["c_total"]).copy()
    ov = (tt.home_score + tt.away_score - tt.c_total).to_numpy(float)
    tt, ov = tt[ov != 0], ov[ov != 0]
    lpt = logit(two_way_power(tt.c_over_odds.fillna(1.91), tt.c_under_odds.fillna(1.91)))
    cv_binary("NBA 大小分校準（目前不採用）", tt.season, (ov > 0).astype(float), lpt[:, None], lpt)

    # MLB
    b = mlb.dropna(subset=["c_ml_home", "c_ml_away"])
    lpb = logit(two_way_power(b.c_ml_home, b.c_ml_away))
    cv_binary("MLB 不讓分校準（目前不採用）", b.season, (b.home_score > b.away_score).astype(float), lpb[:, None], lpb)
    bt = mlb.dropna(subset=["c_total", "c_over_odds", "c_under_odds"]).copy()
    ov = (bt.home_score + bt.away_score - bt.c_total).to_numpy(float)
    bt, ov = bt[ov != 0], ov[ov != 0]
    lpt = logit(two_way_power(bt.c_over_odds, bt.c_under_odds))
    cv_binary("MLB 大小分校準（目前不採用）", bt.season, (ov > 0).astype(float), lpt[:, None], lpt)


# =============================================================================
# 4. 歷史相似盤口的權重
# =============================================================================
SHRINKS = [150, 300, 600, 1200, 2500, 5000, float("inf")]


def blend(pm, ph, n, s):
    if s == float("inf"):
        return pm
    ph = np.where(np.isnan(ph), pm, ph)
    return (pm * s + ph * n) / (s + n)


def report_shrink(sport, seasons, opts):
    """opts: list of dict(market, pm, ph, n, y(0/1/nan), odds, profit_if_win, profit) arrays over games."""
    say(f"\n**{sport}**（逐季留一：每季的相似盤口只從其他季別找）")
    header = "| shrink | " + " | ".join(o["name"] for o in opts) + " | 每場最佳選項 EV≥2%：注數、ROI、各季為正 |"
    say(header)
    say("|---|" + "---|" * len(opts) + "---|")
    for s in SHRINKS:
        cells = []
        evs = []
        for o in opts:
            p = blend(o["pm"], o["ph"], o["n"], s)
            ok = ~np.isnan(o["y"]) & ~np.isnan(p)
            cells.append(f"{bll(p[ok], o['y'][ok]).mean():.4f}")
            evs.append(np.where(np.isnan(p) | np.isnan(o["odds"]), -9, p * o["odds"] - 1))
        E = np.column_stack(evs)
        k = E.argmax(axis=1)
        best = E[np.arange(len(E)), k]
        pick = best >= 0.02
        prof = np.array([opts[j]["profit"][i] for i, j in zip(np.where(pick)[0], k[pick])])
        sea = np.asarray(seasons)[pick]
        by = pd.Series(prof).groupby(sea).mean() if len(prof) else pd.Series(dtype=float)
        kinds = pd.Series([opts[j]["name"] for j in k[pick]]).value_counts().head(3).to_dict()
        label = "只看市場" if s == float("inf") else f"{s:g}"
        say(f"| {label} | " + " | ".join(cells) +
            f" | {pick.sum()} 注 {pct(prof.mean()) if len(prof) else '-'}｜{(by > 0).sum()}/{len(by)} 季｜最常選 {kinds} |")


@section("4. 歷史相似盤口該給多少權重（shrink 越大越相信市場；log loss 越低越好）")
def shrink_section():
    # ---------------- 足球
    h = fb.dropna(subset=["c_avg_h", "c_avg_d", "c_avg_a", "c_avg_over", "c_avg_under"]).reset_index(drop=True)
    odds = h[["c_avg_h", "c_avg_d", "c_avg_a"]].to_numpy(float)
    inv = 1 / odds
    prop = inv / inv.sum(axis=1, keepdims=True)
    pO = prop2(h.c_avg_over, h.c_avg_under)
    pc = cal.football_1x2_matrix(odds)
    res = h.ftr.map({"H": 0, "D": 1, "A": 2}).to_numpy()
    R = np.column_stack([(res == k) - pc[:, k] for k in range(3)])
    goals = (h.hg + h.ag).to_numpy(float)
    margin = (h.hg - h.ag).to_numpy(float)
    season = h.season.to_numpy()
    line = h.c_ah_line.to_numpy(float)
    has_ah = ~np.isnan(line) & h.c_ah_h.notna().to_numpy() & h.c_ah_a.notna().to_numpy()
    lines = sorted(set(np.round(line[has_ah] * 4) / 4))
    cover_by_line = {L: ah_sign(margin, np.full(len(h), L)) for L in lines}
    n_arr = np.zeros(len(h))
    hist = np.full((len(h), 5), np.nan)       # H, D, A, over2.5, AH 主隊贏盤
    for S in sorted(set(season)):
        pool = np.where(season != S)[0]
        PH, PA, PO = prop[pool, 0], prop[pool, 2], pO[pool]
        for i in np.where(season == S)[0]:
            for tol, tol_o in ((0.03, 0.04), (0.045, 0.06), (0.06, 0.08), (0.08, 0.12)):
                msk = (np.abs(PH - prop[i, 0]) <= tol) & (np.abs(PA - prop[i, 2]) <= tol) & (np.abs(PO - pO[i]) <= tol_o)
                if msk.sum() >= 120:
                    break
            sel = pool[msk]
            n_arr[i] = len(sel)
            if not len(sel):
                continue
            hist[i, :3] = np.clip(pc[i] + R[sel].mean(axis=0), 0.02, 0.97)
            hist[i, 3] = (goals[sel] >= 3).mean()
            if has_ah[i]:
                c = cover_by_line[round(line[i] * 4) / 4][sel]
                c = c[c != 0]
                hist[i, 4] = (c > 0).mean() if len(c) else np.nan
    pOU = two_way_power(h.c_avg_over, h.c_avg_under)
    a_int = cal.coefs().get("football_spread") or 0.0
    pAH = sig(a_int + logit(two_way_power(h.c_ah_h, h.c_ah_a)))
    ysign = np.where(has_ah, ah_sign(margin, np.nan_to_num(line)), np.nan)
    yah = np.where(ysign > 0, 1.0, np.where(ysign < 0, 0.0, np.nan))

    def ah_profit(side):
        o = (h.c_ah_h if side == "h" else h.c_ah_a).to_numpy(float)
        mg = margin if side == "h" else -margin
        ln = line if side == "h" else -line
        return np.array([settle_handicap_safe(a, b, c) for a, b, c in zip(mg, ln, o)])

    over = goals >= 3
    opts = []
    for k, nm, col in ((0, "主勝", "c_avg_h"), (1, "和局", "c_avg_d"), (2, "客勝", "c_avg_a")):
        o = h[col].to_numpy(float)
        y = (res == k).astype(float)
        opts.append({"name": nm, "pm": pc[:, k], "ph": hist[:, k], "n": n_arr, "y": y, "odds": o, "profit": np.where(y > 0, o - 1, -1.0)})
    oo, ou_ = h.c_avg_over.to_numpy(float), h.c_avg_under.to_numpy(float)
    opts.append({"name": "大2.5", "pm": pOU, "ph": hist[:, 3], "n": n_arr, "y": over.astype(float), "odds": oo, "profit": np.where(over, oo - 1, -1.0)})
    opts.append({"name": "小2.5", "pm": 1 - pOU, "ph": 1 - hist[:, 3], "n": n_arr, "y": (~over).astype(float), "odds": ou_, "profit": np.where(~over, ou_ - 1, -1.0)})
    opts.append({"name": "亞盤主", "pm": pAH, "ph": hist[:, 4], "n": n_arr, "y": yah, "odds": np.where(has_ah, h.c_ah_h.to_numpy(float), np.nan), "profit": ah_profit("h")})
    opts.append({"name": "亞盤客", "pm": 1 - pAH, "ph": 1 - hist[:, 4], "n": n_arr, "y": 1 - yah, "odds": np.where(has_ah, h.c_ah_a.to_numpy(float), np.nan), "profit": ah_profit("a")})
    say(f"足球：{len(h)} 場，相似盤口平均 {n_arr.mean():.0f} 場（原本只有 3 季時約 190 場）")
    report_shrink("足球（賠率用市場平均收盤）", season, opts)

    # ---------------- NBA
    g = nba.dropna(subset=["c_spread_home", "c_total"]).reset_index(drop=True)
    sp, tot = g.c_spread_home.to_numpy(float), g.c_total.to_numpy(float)
    mg = (g.home_score - g.away_score).to_numpy(float)
    pts = (g.home_score + g.away_score).to_numpy(float)
    pH = prop2(g.c_ml_home, g.c_ml_away)
    cov_raw, ov_raw = mg + sp, pts - tot
    win_res = (mg > 0) - pH
    season = g.season.to_numpy()
    n_arr = np.zeros(len(g))
    hist = np.full((len(g), 3), np.nan)     # cover, over, win_resid
    for S in sorted(set(season)):
        pool = np.where(season != S)[0]
        SP, TT = sp[pool], tot[pool]
        for i in np.where(season == S)[0]:
            sel = pool[(np.abs(SP - sp[i]) <= 1) & (np.abs(TT - tot[i]) <= 6)]
            n_arr[i] = len(sel)
            if not len(sel):
                continue
            c = cov_raw[sel]
            c = c[c != 0]
            hist[i, 0] = (c > 0).mean() if len(c) else np.nan
            v = ov_raw[sel]
            v = v[v != 0]
            hist[i, 1] = (v > 0).mean() if len(v) else np.nan
            wr = win_res[sel]
            hist[i, 2] = np.nanmean(wr) if np.isfinite(wr).any() else np.nan
    sho, sao = g.c_spread_home_odds.fillna(1.91).to_numpy(float), g.c_spread_away_odds.fillna(1.91).to_numpy(float)
    oov, oun = g.c_over_odds.fillna(1.91).to_numpy(float), g.c_under_odds.fillna(1.91).to_numpy(float)
    mlh, mla = g.c_ml_home.to_numpy(float), g.c_ml_away.to_numpy(float)
    p_sp = two_way_power(sho, sao)
    p_t = prop2(oov, oun)
    p_ml = two_way_power(mlh, mla)
    win_hist = np.clip(p_ml + hist[:, 2], 0.03, 0.97)
    ycov = np.where(cov_raw > 0, 1.0, np.where(cov_raw < 0, 0.0, np.nan))
    yov = np.where(ov_raw > 0, 1.0, np.where(ov_raw < 0, 0.0, np.nan))
    ywin = (mg > 0).astype(float)
    prof = lambda y, o: np.where(np.isnan(y), 0.0, np.where(y > 0, o - 1, -1.0))
    opts = [
        {"name": "主讓分", "pm": p_sp, "ph": hist[:, 0], "n": n_arr, "y": ycov, "odds": sho, "profit": prof(ycov, sho)},
        {"name": "客讓分", "pm": 1 - p_sp, "ph": 1 - hist[:, 0], "n": n_arr, "y": 1 - ycov, "odds": sao, "profit": prof(1 - ycov, sao)},
        {"name": "大分", "pm": p_t, "ph": hist[:, 1], "n": n_arr, "y": yov, "odds": oov, "profit": prof(yov, oov)},
        {"name": "小分", "pm": 1 - p_t, "ph": 1 - hist[:, 1], "n": n_arr, "y": 1 - yov, "odds": oun, "profit": prof(1 - yov, oun)},
        {"name": "主不讓分", "pm": p_ml, "ph": win_hist, "n": n_arr, "y": ywin, "odds": mlh, "profit": prof(ywin, mlh)},
        {"name": "客不讓分", "pm": 1 - p_ml, "ph": 1 - win_hist, "n": n_arr, "y": 1 - ywin, "odds": mla, "profit": prof(1 - ywin, mla)},
    ]
    say(f"\nNBA：{len(g)} 場，相似盤口平均 {n_arr.mean():.0f} 場")
    report_shrink("NBA（賠率用 ESPN 收盤）", season, opts)

    # ---------------- MLB
    g = mlb.dropna(subset=["c_ml_home", "c_ml_away", "c_total"]).reset_index(drop=True)
    pH = prop2(g.c_ml_home, g.c_ml_away)
    L = g.c_total.to_numpy(float)
    mg = (g.home_score - g.away_score).to_numpy(float)
    runs = (g.home_score + g.away_score).to_numpy(float)
    season = g.season.to_numpy()
    n_arr = np.zeros(len(g))
    hist = np.full((len(g), 2), np.nan)     # win, over
    winr = (mg > 0) - pH
    for S in sorted(set(season)):
        pool = np.where(season != S)[0]
        PHp, LL = pH[pool], L[pool]
        for i in np.where(season == S)[0]:
            for tol, tol_t in ((0.025, 0.0), (0.04, 0.5), (0.06, 0.5), (0.08, 1.0)):
                msk = (np.abs(PHp - pH[i]) <= tol) & (np.abs(LL - L[i]) <= tol_t + 1e-9)
                if msk.sum() >= 150:
                    break
            sel = pool[msk]
            n_arr[i] = len(sel)
            if not len(sel):
                continue
            hist[i, 0] = np.clip(pH[i] + winr[sel].mean(), 0.03, 0.97)
            adj = runs[sel] - L[sel] + L[i]
            adj = adj[adj != L[i]]
            hist[i, 1] = (adj > L[i]).mean() if len(adj) else np.nan
    mlh, mla = g.c_ml_home.to_numpy(float), g.c_ml_away.to_numpy(float)
    ovo, uno = g.c_over_odds.to_numpy(float), g.c_under_odds.to_numpy(float)
    p_t = prop2(ovo, uno)
    ywin = (mg > 0).astype(float)
    yov = np.where(runs > L, 1.0, np.where(runs < L, 0.0, np.nan))
    opts = [
        {"name": "主不讓分", "pm": pH, "ph": hist[:, 0], "n": n_arr, "y": ywin, "odds": mlh, "profit": prof(ywin, mlh)},
        {"name": "客不讓分", "pm": 1 - pH, "ph": 1 - hist[:, 0], "n": n_arr, "y": 1 - ywin, "odds": mla, "profit": prof(1 - ywin, mla)},
        {"name": "大分", "pm": p_t, "ph": hist[:, 1], "n": n_arr, "y": yov, "odds": ovo, "profit": prof(yov, ovo)},
        {"name": "小分", "pm": 1 - p_t, "ph": 1 - hist[:, 1], "n": n_arr, "y": 1 - yov, "odds": uno, "profit": prof(1 - yov, uno)},
    ]
    say(f"\nMLB：{len(g)} 場，相似盤口平均 {n_arr.mean():.0f} 場")
    report_shrink("MLB（賠率用 ESPN 收盤）", season, opts)


def settle_handicap_safe(margin, line, odds):
    from strategies import settle_handicap
    if not (np.isfinite(margin) and np.isfinite(line) and np.isfinite(odds)):
        return 0.0
    return settle_handicap(margin, line, odds)


# =============================================================================
# 5. 莊家角度逐季
# =============================================================================
@section("5. 莊家角度逐季（全部用收盤價、每注 1 單位的 ROI）")
def bookie_section():
    rows = []
    for s, g in fb.groupby("season"):
        r = {"季別": s}
        for k, nm in (("h", "主勝"), ("d", "和局"), ("a", "客勝")):
            o = g[f"c_avg_{k}"]
            ok = o.notna()
            r[nm] = np.where(g.ftr[ok] == k.upper(), o[ok] - 1, -1.0).mean()
        a = g.dropna(subset=["c_ah_line", "c_ah_h", "c_ah_a"])
        r["亞盤主"] = np.mean([settle_handicap_safe(x, l, o) for x, l, o in zip(a.hg - a.ag, a.c_ah_line, a.c_ah_h)])
        r["亞盤客"] = np.mean([settle_handicap_safe(x, -l, o) for x, l, o in zip(a.ag - a.hg, a.c_ah_line, a.c_ah_a)])
        b3 = g.dropna(subset=["c_b365_h", "c_b365_d", "c_b365_a"])
        oo = b3[["c_b365_h", "c_b365_d", "c_b365_a"]].to_numpy(float).ravel()
        ww = np.column_stack([(b3.ftr == x).to_numpy() for x in "HDA"]).ravel()
        prof = np.where(ww, oo - 1, -1.0)
        r["B365 賠率≥5"] = prof[oo >= 5].mean()
        r["B365 賠率<1.5"] = prof[oo < 1.5].mean()
        rows.append(r)
    t = pd.DataFrame(rows).set_index("季別")
    say("足球（市場平均收盤；B365 欄是 Bet365 收盤依賠率分組）：")
    say("| 季別 | " + " | ".join(t.columns) + " |")
    say("|---|" + "---|" * len(t.columns))
    for s, r in t.iterrows():
        say(f"| {s} | " + " | ".join(pct(v) for v in r) + " |")
    say(f"| 全部 | " + " | ".join(pct(v) for v in t.mean()) + " |（各季平均）")

    rows = []
    for s, g in nba.groupby("season"):
        r = {"季別": s}
        sp = g.dropna(subset=["c_spread_home"])
        mg = (sp.home_score - sp.away_score).to_numpy(float)
        ho = sp.c_spread_home_odds.fillna(1.91).to_numpy(float)
        ao = sp.c_spread_away_odds.fillna(1.91).to_numpy(float)
        ph = np.array([settle_handicap_safe(x, l, o) for x, l, o in zip(mg, sp.c_spread_home, ho)])
        pa = np.array([settle_handicap_safe(-x, -l, o) for x, l, o in zip(mg, sp.c_spread_home, ao)])
        fav_home = sp.c_spread_home.to_numpy(float) < 0
        r["主隊讓分"], r["客隊讓分"] = ph.mean(), pa.mean()
        r["熱門讓分"] = np.where(fav_home, ph, pa).mean()
        r["冷門讓分"] = np.where(fav_home, pa, ph).mean()
        ml = g.dropna(subset=["c_ml_home", "c_ml_away"])
        o = np.concatenate([ml.c_ml_home, ml.c_ml_away])
        w = np.concatenate([ml.home_score > ml.away_score, ml.away_score > ml.home_score])
        prof = np.where(w, o - 1, -1.0)
        r["不讓分 賠率≥4"] = prof[o >= 4].mean() if (o >= 4).any() else np.nan
        r["不讓分 賠率<1.5"] = prof[o < 1.5].mean() if (o < 1.5).any() else np.nan
        rows.append(r)
    t = pd.DataFrame(rows).set_index("季別")
    say("\nNBA：")
    say("| 季別 | " + " | ".join(t.columns) + " |")
    say("|---|" + "---|" * len(t.columns))
    for s, r in t.iterrows():
        say(f"| {s} | " + " | ".join(pct(v) if pd.notna(v) else "-" for v in r) + " |")

    rows = []
    for s, g in mlb.groupby("season"):
        ml = g.dropna(subset=["c_ml_home", "c_ml_away"])
        if not len(ml):
            continue
        hw = (ml.home_score > ml.away_score).to_numpy()
        ho, ao = ml.c_ml_home.to_numpy(float), ml.c_ml_away.to_numpy(float)
        ph, pa = np.where(hw, ho - 1, -1.0), np.where(~hw, ao - 1, -1.0)
        fav_home = ho < ao
        rows.append({"季別": s, "主隊": ph.mean(), "客隊": pa.mean(), "熱門": np.where(fav_home, ph, pa).mean(),
                     "冷門": np.where(fav_home, pa, ph).mean()})
    t = pd.DataFrame(rows).set_index("季別")
    say("\nMLB 不讓分：")
    say("| 季別 | " + " | ".join(t.columns) + " |")
    say("|---|" + "---|" * len(t.columns))
    for s, r in t.iterrows():
        say(f"| {s} | " + " | ".join(pct(v) for v in r) + " |")


# =============================================================================
# 6. NBA 動能
# =============================================================================
@section("6. NBA 動能：上一場大勝 / 大幅打出盤口之後，跟買或反買（讓分收盤）")
def momentum_section():
    g = nba.dropna(subset=["c_spread_home"]).sort_values(["game_date", "event_id"]).reset_index(drop=True)
    mg = (g.home_score - g.away_score).astype(float)
    long = pd.concat([
        pd.DataFrame({"i": g.index, "season": g.season, "team": g.home, "m": mg, "ats": mg + g.c_spread_home, "side": "h"}),
        pd.DataFrame({"i": g.index, "season": g.season, "team": g.away, "m": -mg, "ats": -mg - g.c_spread_home, "side": "a"}),
    ]).sort_values("i")
    grp = long.groupby(["season", "team"])
    long["last_m"] = grp["m"].shift()
    long["last_ats"] = grp["ats"].shift()
    piv = long.pivot_table(index="i", columns="side", values=["last_m", "last_ats"], aggfunc="first")
    ho = g.c_spread_home_odds.fillna(1.91).to_numpy(float)
    ao = g.c_spread_away_odds.fillna(1.91).to_numpy(float)
    ph = np.array([settle_handicap_safe(x, l, o) for x, l, o in zip(mg, g.c_spread_home, ho)])
    pa = np.array([settle_handicap_safe(-x, -l, o) for x, l, o in zip(mg, g.c_spread_home, ao)])
    for lab, col, thr in (("上一場大勝 ≥15 分", "last_m", 15), ("上一場大勝 ≥20 分", "last_m", 20), ("上一場打出盤口 ≥12 分", "last_ats", 12)):
        hh = (piv[(col, "h")] >= thr).reindex(g.index, fill_value=False).to_numpy()
        aa = (piv[(col, "a")] >= thr).reindex(g.index, fill_value=False).to_numpy()
        only_h, only_a = hh & ~aa, aa & ~hh
        for mode in ("跟買", "反買"):
            if mode == "跟買":
                prof = np.concatenate([ph[only_h], pa[only_a]])
            else:
                prof = np.concatenate([pa[only_h], ph[only_a]])
            sea = np.concatenate([g.season.to_numpy()[only_h], g.season.to_numpy()[only_a]])
            by = pd.Series(prof).groupby(sea).mean()
            p = 0.5 * math.erfc(tstat(prof) / math.sqrt(2))
            say(f"- {mode}「{lab}」：{len(prof)} 注 {pct(prof.mean())}（p={p:.3f}）｜{(by > 0).sum()}/{len(by)} 季為正｜"
                + "、".join(f"{k} {pct(v, 0)}" for k, v in by.items()))


# =============================================================================
# 7. NBA 球員狀態
# =============================================================================
@section("7. NBA 球員狀態（主力疲勞、缺陣）：加入 2023-24 後逐季驗證")
def form_section():
    import data_nba_box
    import nba_form
    open_seasons = nba.loc[nba.o_spread_home.notna(), "season"].unique()
    d0 = nba[nba.season.isin(open_seasons)]
    box = data_nba_box.load_boxes(d0.event_id.astype(str).tolist())
    F = nba_form.game_features(d0, box)
    d = d0.astype({"event_id": str}).merge(F, on="event_id")
    d = d[(d.h_n_prior >= 10) & (d.a_n_prior >= 10)]
    say(f"有開盤的季別 {list(open_seasons)}，可用 {len(d)} 場")
    s = d.dropna(subset=["c_spread_home", "o_spread_home"]).copy()
    m = (s.home_score - s.away_score + s.c_spread_home).to_numpy(float)
    s, m = s[m != 0], m[m != 0]
    y = (m > 0).astype(float)
    lp = logit(two_way_power(s.c_spread_home_odds.fillna(1.91), s.c_spread_away_odds.fillna(1.91)))
    dsp = -(s.c_spread_home - s.o_spread_home).to_numpy(float)
    out, better = [], 0
    for S in sorted(s.season.unique()):
        te = (s.season == S).to_numpy()
        if te.sum() < 300:
            continue
        w0 = np.array(cal._fit(np.column_stack([lp, dsp])[~te], y[~te]))
        base = w0[0] + w0[1] * lp + w0[2] * dsp
        X = np.column_stack([base, s.d_star_out.fillna(0), s.d_star_load.fillna(0)])
        w1 = np.array(cal._fit(X[~te], y[~te]))
        d_ = bll(sig(w1[0] + X[te] @ w1[1:]), y[te]) - bll(sig(base[te]), y[te])
        better += d_.mean() < 0
        out.append(f"{S} {d_.mean() * 1000:+.2f}‰(t={tstat(d_):+.1f}；缺陣係數{w1[2]:+.3f}、疲勞係數{w1[3]:+.3f})")
    say(f"- **讓分 + 主力缺陣差、疲勞差**：{better}/{len(out)} 季改善｜" + "　".join(out))
    ml = d.dropna(subset=["c_ml_home", "c_ml_away", "o_ml_home", "o_ml_away"])
    y = (ml.home_score > ml.away_score).astype(float).to_numpy()
    lc = logit(two_way_power(ml.c_ml_home, ml.c_ml_away))
    lo = logit(two_way_power(ml.o_ml_home, ml.o_ml_away))
    out, better = [], 0
    for S in sorted(ml.season.unique()):
        te = (ml.season == S).to_numpy()
        if te.sum() < 300:
            continue
        w0 = np.array(cal._fit(np.column_stack([lc, lc - lo])[~te], y[~te]))
        base = w0[0] + w0[1] * lc + w0[2] * (lc - lo)
        X = np.column_stack([base, ml.d_star_load.fillna(0)])
        w1 = np.array(cal._fit(X[~te], y[~te]))
        d_ = bll(sig(w1[0] + X[te] @ w1[1:]), y[te]) - bll(sig(base[te]), y[te])
        better += d_.mean() < 0
        out.append(f"{S} {d_.mean() * 1000:+.2f}‰(t={tstat(d_):+.1f}；疲勞係數{w1[2]:+.3f})")
    say(f"- **不讓分 + 主力疲勞差**：{better}/{len(out)} 季改善｜" + "　".join(out))


coverage()
strategies_section()
calibration_section()
shrink_section()
bookie_section()
momentum_section()
form_section()
say(f"\n（總耗時 {time.time() - T0:.0f} 秒）")
save()
