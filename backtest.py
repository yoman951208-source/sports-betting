# -*- coding: utf-8 -*-
"""兩年歷史回測：找出「盤口變化 / 各家賠率差 / 球員休息」中真的有優勢的規律。

篩選原則（config.py 可調）：
  1. 樣本數夠大（足球 ≥150 注、NBA ≥80 注）
  2. 單尾 p 值 < 0.10（不是運氣）
  3. 每一季 ROI 都 > 0（不是只有某一季好）
同一類策略只啟用 t 值最高的一個門檻，避免同一場比賽重複推播。
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime

import pandas as pd

import config
from core import DATA_DIR, STRATEGY_FILE, log
from strategies import (Strategy, bet_odds, football_strategies, market_prob, mlb_strategies, nba_strategies, settle,
                        settle_handicap,
                        trigger)


def _clean(v):
    return None if isinstance(v, float) and math.isnan(v) else v


def run_bets(strats: list[Strategy], df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    rows = [{k: _clean(v) for k, v in r.items()} for r in df.to_dict("records")]
    bets: dict[str, list] = defaultdict(list)
    for g in rows:
        for s in strats:
            t = trigger(s, g)
            if not t:
                continue
            pick, line, why = t
            odds = bet_odds(s, g, pick)
            if odds is None or odds <= 1 or (s.market == "ah" and line is None):
                continue
            profit = settle(s, g, pick, line, odds)
            bets[s.sid].append({"season": g["season"], "date": g.get("date") or g.get("game_date"),
                                "pick": pick, "odds": odds, "profit": profit, "why": why,
                                "p_mkt": market_prob(s, g, pick)})
    return {k: pd.DataFrame(v) for k, v in bets.items()}


def evaluate(b: pd.DataFrame) -> dict:
    n = len(b)
    if n == 0:
        return {"n": 0}
    p = b["profit"]
    mean, sd = p.mean(), p.std(ddof=1) if n > 1 else 0.0
    t = mean / (sd / math.sqrt(n)) if sd > 0 else 0.0
    pval = 0.5 * math.erfc(t / math.sqrt(2))
    cum = p.cumsum()
    drawdown = float((cum.cummax().clip(lower=0) - cum).max())
    losing, streak = 0, 0
    for x in p:
        streak = streak + 1 if x < 0 else 0
        losing = max(losing, streak)
    decided = b[p != 0]
    # uplift：實際勝率比市場（去水）機率高多少 —— 即時訊息用來換算「運彩賠率至少要多少才划算」
    dm = decided.dropna(subset=["p_mkt"])
    uplift = float(((dm["profit"] > 0).astype(float) - dm["p_mkt"]).mean()) if len(dm) else 0.0
    by_season = {}
    for season, grp in b.groupby("season"):
        by_season[season] = {"n": int(len(grp)), "roi": round(float(grp["profit"].mean()), 4),
                             "hit": round(float((grp["profit"] > 0).mean()), 4)}
    return {
        "n": int(n), "hit": round(float((decided["profit"] > 0).mean()) if len(decided) else 0.0, 4),
        "roi": round(float(mean), 4), "units": round(float(p.sum()), 2), "avg_odds": round(float(b["odds"].mean()), 3),
        "t": round(float(t), 2), "p": round(float(pval), 4), "max_drawdown": round(drawdown, 2),
        "max_losing_streak": int(losing), "by_season": by_season, "uplift": round(uplift, 4),
        "first": str(pd.Timestamp(b["date"].min()).date()), "last": str(pd.Timestamp(b["date"].max()).date()),
    }


def select(strats: list[Strategy]) -> None:
    for s in strats:
        st = s.stats
        if st.get("n", 0) == 0:
            continue
        seasons_ok = all(v["roi"] > 0 for v in st["by_season"].values() if v["n"] >= 40) \
            if config.REQUIRE_POSITIVE_EACH_SEASON else True
        s.enabled = st["n"] >= config.MIN_BETS[s.sport] and st["p"] < config.MAX_P_VALUE and seasons_ok and st["roi"] > 0
        s.weak = not s.enabled and st["roi"] > 0 and st["n"] >= config.MIN_BETS[s.sport] // 2
    # 同一類（family + mode）只留 t 值最高的一個
    best: dict[tuple, Strategy] = {}
    for s in strats:
        if s.enabled:
            key = (s.family, s.mode)
            if key not in best or s.stats["t"] > best[key].stats["t"]:
                best[key] = s
    for s in strats:
        if s.enabled and best[(s.family, s.mode)] is not s:
            s.enabled, s.weak = False, True


def baselines(fb: pd.DataFrame, nba: pd.DataFrame) -> dict:
    """盲目下注的基準報酬（看出莊家抽水有多重）。"""
    out = {}
    for k, name in (("h", "足球盲買主勝"), ("d", "足球盲買和局"), ("a", "足球盲買客勝")):
        o = fb[f"c_avg_{k}"]
        won = fb["ftr"] == k.upper()
        prof = (won * (o - 1) - (~won)).where(o.notna())
        out[name] = round(float(prof.mean()), 4)
    m = fb[["c_avg_h", "c_avg_d", "c_avg_a"]]
    out["足球市場平均抽水"] = round(float((1 / m).sum(axis=1).mean() - 1), 4)
    if len(nba):
        margin = nba["home_score"] - nba["away_score"] + nba["c_spread_home"]
        o = nba["c_spread_home_odds"].fillna(1.91)
        prof = ((margin > 0) * (o - 1) - (margin < 0)).where(nba["c_spread_home"].notna())
        out["NBA盲買主隊讓分"] = round(float(prof.mean()), 4)
        mh, ma = nba["c_ml_home"], nba["c_ml_away"]
        out["NBA不讓分平均抽水"] = round(float((1 / mh + 1 / ma).mean() - 1), 4)
    return out


def mlb_baselines(mlb: pd.DataFrame) -> dict:
    """MLB 盲目下注基準（例行賽 / 季後賽分開看）。"""
    out = {}
    d = mlb.dropna(subset=["c_ml_home", "c_ml_away"])
    home_win = d["home_score"] > d["away_score"]
    fav_home = d["c_ml_home"] <= d["c_ml_away"]
    fav_odds = d["c_ml_home"].where(fav_home, d["c_ml_away"])
    fav_won = home_win == fav_home
    for tag, mask in (("例行賽", ~d["playoffs"]), ("季後賽", d["playoffs"])):
        x = d[mask]
        if not len(x):
            continue
        out[f"MLB{tag}盲買主隊"] = round(float((home_win[mask] * (x["c_ml_home"] - 1) - ~home_win[mask]).mean()), 4)
        out[f"MLB{tag}盲買熱門"] = round(float((fav_won[mask] * (fav_odds[mask] - 1) - ~fav_won[mask]).mean()), 4)
        t = x.dropna(subset=["c_total"])
        runs = t["home_score"] + t["away_score"]
        dec = runs != t["c_total"]
        out[f"MLB{tag}大分比例"] = round(float((runs[dec] > t["c_total"][dec]).mean()), 4)
    out["MLB季後賽場數"] = int(d["playoffs"].sum())
    out["MLB不讓分平均抽水"] = round(float((1 / d["c_ml_home"] + 1 / d["c_ml_away"]).mean() - 1), 4)
    return out


# -----------------------------------------------------------------------------
# 讓分 × 大小 交叉表（描述性統計，看各種盤口組合下的實際報酬）
# -----------------------------------------------------------------------------
def _side_profit(margin_home, line_home, odds_home, odds_away, side_home) -> list[float]:
    """side_home=True 買主隊讓分，否則買客隊。"""
    out = []
    for m, l, oh, oa, sh in zip(margin_home, line_home, odds_home, odds_away, side_home):
        out.append(settle_handicap(m if sh else -m, l if sh else -l, oh if sh else oa))
    return out


def _cells(d: pd.DataFrame, rows: str, cols: str, metrics: dict[str, str]) -> list[dict]:
    out = []
    for (r, c), grp in d.groupby([rows, cols], observed=True):
        out.append({"row": r, "col": c, "n": len(grp), **{k: float(grp[v].mean()) for k, v in metrics.items()}})
    return out


def cross_tables(fb: pd.DataFrame, nba: pd.DataFrame) -> list[dict]:
    tables = []
    # ⚽ 1. 收盤讓球深度 × 預期進球
    d = fb.dropna(subset=["c_ah_line", "c_ah_h", "c_ah_a", "c_avg_over", "c_avg_under"]).copy()
    io = 1 / d[["c_avg_over", "c_avg_under"]]
    d["pO"] = io["c_avg_over"] / io.sum(axis=1)
    d["r"] = pd.cut(d["c_ah_line"].abs(), [-0.01, 0.25, 0.75, 1.25, 9], labels=["讓0~0.25", "讓0.5~0.75", "讓1~1.25", "讓≥1.5"])
    d["c"] = pd.cut(d["pO"], [0, 0.5, 0.6, 1.01], right=False, labels=["大2.5<50%", "大2.5 50~60%", "大2.5≥60%"])
    fav_home = d["c_ah_line"] <= 0
    margin = d["hg"] - d["ag"]
    d["fav"] = _side_profit(margin, d["c_ah_line"], d["c_ah_h"], d["c_ah_a"], fav_home)
    d["dog"] = _side_profit(margin, d["c_ah_line"], d["c_ah_h"], d["c_ah_a"], ~fav_home)
    d["over"] = (d["hg"] + d["ag"] >= 3) * (d["c_avg_over"] - 1) - (d["hg"] + d["ag"] < 3)
    tables.append({"title": "⚽ 收盤讓球深度 × 預期進球", "metrics": {"讓球方": "fav", "受讓方": "dog", "大2.5": "over"},
                   "cells": _cells(d, "r", "c", {"fav": "fav", "dog": "dog", "over": "over"})})

    # ⚽ 2. 讓球盤變化 × 大小盤變化
    d = fb.dropna(subset=["o_ah_line", "c_ah_line", "c_ah_h", "c_ah_a", "o_avg_over", "o_avg_under",
                          "c_avg_over", "c_avg_under"]).copy()
    dah = d["c_ah_line"] - d["o_ah_line"]
    po = (1 / d["o_avg_over"]) / (1 / d["o_avg_over"] + 1 / d["o_avg_under"])
    pc = (1 / d["c_avg_over"]) / (1 / d["c_avg_over"] + 1 / d["c_avg_under"])
    dou = pc - po
    d["r"] = pd.Categorical(["主隊被買" if x <= -0.25 else "客隊被買" if x >= 0.25 else "讓球沒動" for x in dah],
                            ["主隊被買", "讓球沒動", "客隊被買"])
    d["c"] = pd.Categorical(["大球被買" if x >= 0.03 else "小球被買" if x <= -0.03 else "大小沒動" for x in dou],
                            ["大球被買", "大小沒動", "小球被買"])
    side_home = dah <= 0  # 讓球沒動時看主隊
    d["steam"] = _side_profit(d["hg"] - d["ag"], d["c_ah_line"], d["c_ah_h"], d["c_ah_a"], side_home)
    d["over"] = (d["hg"] + d["ag"] >= 3) * (d["c_avg_over"] - 1) - (d["hg"] + d["ag"] < 3)
    tables.append({"title": "⚽ 讓球盤變化 × 大小盤變化（開盤→收盤）", "metrics": {"跟讓球": "steam", "大2.5": "over"},
                   "cells": _cells(d, "r", "c", {"steam": "steam", "over": "over"})})

    # 🏀 3. 收盤讓分大小 × 大小分相對聯盟平均
    d = nba.dropna(subset=["c_spread_home", "total_rel", "c_total"]).copy()
    d["r"] = pd.cut(d["c_spread_home"].abs(), [-0.1, 3.4, 6.9, 9.9, 99], labels=["讓≤3", "讓3.5~6.5", "讓7~9.5", "讓≥10"])
    d["c"] = pd.cut(d["total_rel"], [-99, -4, 4, 99], right=False, labels=["大小偏低≥4", "大小正常", "大小偏高≥4"])
    oh, oa = d["c_spread_home_odds"].fillna(1.91), d["c_spread_away_odds"].fillna(1.91)
    fav_home = d["c_spread_home"] <= 0
    margin = d["home_score"] - d["away_score"]
    d["fav"] = _side_profit(margin, d["c_spread_home"], oh, oa, fav_home)
    d["dog"] = _side_profit(margin, d["c_spread_home"], oh, oa, ~fav_home)
    pts = d["home_score"] + d["away_score"]
    d["over"] = [settle_handicap(p - t, 0, o) for p, t, o in zip(pts, d["c_total"], d["c_over_odds"].fillna(1.91))]
    tables.append({"title": "🏀 收盤讓分 × 大小分（相對近期平均）", "metrics": {"讓分方": "fav", "受讓方": "dog", "大分": "over"},
                   "cells": _cells(d, "r", "c", {"fav": "fav", "dog": "dog", "over": "over"})})

    # 🏀 4. 讓分變化 × 大小分變化
    d = nba.dropna(subset=["o_spread_home", "c_spread_home", "o_total", "c_total"]).copy()
    dsp, dt = d["c_spread_home"] - d["o_spread_home"], d["c_total"] - d["o_total"]
    d["r"] = pd.Categorical(["主隊被買" if x <= -2 else "客隊被買" if x >= 2 else "讓分沒動" for x in dsp],
                            ["主隊被買", "讓分沒動", "客隊被買"])
    d["c"] = pd.Categorical(["大小↑≥2" if x >= 2 else "大小↓≥2" if x <= -2 else "大小沒動" for x in dt],
                            ["大小↑≥2", "大小沒動", "大小↓≥2"])
    oh, oa = d["c_spread_home_odds"].fillna(1.91), d["c_spread_away_odds"].fillna(1.91)
    d["steam"] = _side_profit(d["home_score"] - d["away_score"], d["c_spread_home"], oh, oa, dsp <= 0)
    pts = d["home_score"] + d["away_score"]
    d["over"] = [settle_handicap(p - t, 0, o) for p, t, o in zip(pts, d["c_total"], d["c_over_odds"].fillna(1.91))]
    tables.append({"title": "🏀 讓分變化 × 大小分變化（開盤→收盤）", "metrics": {"跟讓分": "steam", "大分": "over"},
                   "cells": _cells(d, "r", "c", {"steam": "steam", "over": "over"})})
    return tables


def cross_markdown(tables: list[dict]) -> str:
    out = ["", "## 讓分 × 大小 交叉表", "", "格子內：樣本數｜各選項以收盤價下注的 ROI（讓球沒動的列＝買主隊）", ""]
    for t in tables:
        cols = list(dict.fromkeys(c["col"] for c in t["cells"]))
        rows = list(dict.fromkeys(c["row"] for c in t["cells"]))
        out += [f"### {t['title']}", "", "| | " + " | ".join(map(str, cols)) + " |", "|---" * (len(cols) + 1) + "|"]
        for r in rows:
            cells = []
            for c in cols:
                x = next((x for x in t["cells"] if x["row"] == r and x["col"] == c), None)
                cells.append("" if x is None else f"n={x['n']}<br>" + "<br>".join(
                    f"{k} {x[v]*100:+.1f}%" for k, v in t["metrics"].items()))
            out.append(f"| {r} | " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out)


def cross_telegram(tables: list[dict]) -> str:
    from html import escape
    lines = ["<b>🔀 讓分 × 大小 交叉分析</b>（以收盤價下注的 ROI；盲買基準約 -3%~-6%）", ""]
    for t in tables:
        lines.append(f"<b>{escape(t['title'])}</b>")
        for x in t["cells"]:
            if x["n"] < 40:
                continue
            vals = "  ".join(f"{k}{x[v]*100:+.0f}%" for k, v in t["metrics"].items())
            flag = " ⬅" if any(x[v] > 0.03 for v in t["metrics"].values()) and x["n"] >= 150 else ""
            lines.append(f"{escape(str(x['row']))}｜{escape(str(x['col']))}（{x['n']}）{vals}{flag}")
        lines.append("")
    lines.append("⬅ = 樣本 ≥150 且有選項 ROI > +3%（是否穩定見下方策略篩選結果）")
    return "\n".join(lines)


def run(send: bool = False, dry_run: bool = False) -> list[Strategy]:
    import data_football
    import data_mlb
    import data_nba
    from core import send_telegram

    fb = data_football.load_history()
    nba = data_nba.load_history()
    mlb = data_mlb.load_history()
    fs, ns, ms = football_strategies(), nba_strategies(), mlb_strategies()
    all_bets = {**run_bets(fs, fb), **run_bets(ns, nba), **run_bets(ms, mlb)}
    strats = fs + ns + ms
    for s in strats:
        s.stats = evaluate(all_bets.get(s.sid, pd.DataFrame()))
    select(strats)
    base = {**baselines(fb, nba), **mlb_baselines(mlb)}
    meta = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "football_games": int(len(fb)), "football_range": [str(fb["date"].min().date()), str(fb["date"].max().date())],
        "nba_games": int(len(nba)), "nba_range": [str(nba["game_date"].min().date()), str(nba["game_date"].max().date())],
        "mlb_games": int(len(mlb)), "mlb_range": [str(mlb["game_date"].min().date()), str(mlb["game_date"].max().date())],
        "baselines": base,
    }
    STRATEGY_FILE.write_text(json.dumps({"meta": meta, "strategies": [s.to_dict() for s in strats]},
                                        ensure_ascii=False, indent=1), encoding="utf-8")
    tables = cross_tables(fb, nba)
    (DATA_DIR / "backtest_report.md").write_text(markdown_report(strats, meta) + cross_markdown(tables), encoding="utf-8")
    log.info("回測完成：%d 個策略，啟用 %d 個", len(strats), sum(s.enabled for s in strats))
    if send:
        send_telegram(telegram_report(strats, meta), dry_run=dry_run)
        send_telegram(cross_telegram(tables), dry_run=dry_run)
    return strats


def _line(s: Strategy) -> str:
    st = s.stats
    seasons = " / ".join(f"{k[2:]}:{v['roi']*100:+.1f}%({v['n']})" for k, v in st["by_season"].items())
    return (f"{s.title}\n   樣本 {st['n']} 注｜命中 {st['hit']*100:.1f}%｜均賠 {st['avg_odds']:.2f}｜"
            f"ROI {st['roi']*100:+.1f}%｜p={st['p']:.3f}\n   各季 {seasons}")


def telegram_report(strats: list[Strategy], meta: dict) -> str:
    from html import escape
    lines = ["<b>📊 兩年回測報告</b>",
             f"足球 {meta['football_games']} 場（{meta['football_range'][0]}~{meta['football_range'][1]}）",
             f"NBA {meta['nba_games']} 場（{meta['nba_range'][0]}~{meta['nba_range'][1]}）"]
    if meta.get("mlb_games"):
        lines.append(f"MLB {meta['mlb_games']} 場（{meta['mlb_range'][0]}~{meta['mlb_range'][1]}）")
    lines.append("")
    b = meta["baselines"]
    lines.append("<b>基準（盲目下注）</b>")
    lines += [f"• {k}：{v}" if isinstance(v, int) else
              f"• {k}：{v*100:.1f}%" if "比例" in k else f"• {k}：{v*100:+.1f}%" for k, v in b.items()]
    for sport, icon in (("football", "⚽"), ("nba", "🏀"), ("mlb", "⚾")):
        en = sorted([s for s in strats if s.sport == sport and s.enabled], key=lambda s: -s.stats["t"])
        weak = sorted([s for s in strats if s.sport == sport and s.weak], key=lambda s: -s.stats["t"])[:4]
        lines.append(f"\n{icon} <b>⭐ 通過檢驗、會即時推播</b>（{len(en)}）")
        lines += [escape(_line(s)) for s in en] or ["（無 —— 兩年資料中沒有找到穩定的優勢）"]
        if weak:
            lines.append(f"{icon} <b>參考（正報酬但不夠顯著）</b>")
            lines += [escape(_line(s)) for s in weak]
    lines.append("\n⚠️ 過去績效不代表未來；市場會修正已知規律。請量力而為。")
    return "\n".join(lines)


def markdown_report(strats: list[Strategy], meta: dict) -> str:
    out = [f"# 回測報告（{meta['generated']}）", "",
           f"- 足球：{meta['football_games']} 場，{meta['football_range'][0]} ~ {meta['football_range'][1]}",
           f"- NBA：{meta['nba_games']} 場，{meta['nba_range'][0]} ~ {meta['nba_range'][1]}"]
    if meta.get("mlb_games"):
        out.append(f"- MLB：{meta['mlb_games']} 場，{meta['mlb_range'][0]} ~ {meta['mlb_range'][1]}")
    out += ["", "## 基準", ""]
    out += [f"- {k}：{v}" if isinstance(v, int) else f"- {k}：{v*100:+.2f}%" for k, v in meta["baselines"].items()]
    out += ["", "## 全部策略", "",
            "| 狀態 | 策略 | 即時資料 | 注數 | 命中 | 均賠 | ROI | p | 最大回撤(u) | 最長連敗 | 各季 ROI(注數) |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in sorted(strats, key=lambda s: (s.sport, -s.stats.get("t", -99))):
        st = s.stats
        if not st.get("n"):
            out.append(f"| - | {s.title} | {s.live_source} | 0 | | | | | | | |")
            continue
        tag = "⭐啟用" if s.enabled else ("參考" if s.weak else "")
        seasons = "<br>".join(f"{k}: {v['roi']*100:+.1f}% ({v['n']})" for k, v in st["by_season"].items())
        out.append(f"| {tag} | {s.title} | {s.live_source} | {st['n']} | {st['hit']*100:.1f}% | {st['avg_odds']:.2f} | "
                   f"{st['roi']*100:+.1f}% | {st['p']:.3f} | {st['max_drawdown']} | {st['max_losing_streak']} | {seasons} |")
    return "\n".join(out) + "\n"


def load_strategies() -> tuple[list[Strategy], dict]:
    if not STRATEGY_FILE.exists():
        return [], {}
    data = json.loads(STRATEGY_FILE.read_text(encoding="utf-8"))
    return [Strategy(**d) for d in data["strategies"]], data.get("meta", {})


if __name__ == "__main__":
    run()
