# -*- coding: utf-8 -*-
"""賽後檢討：用開賽前最後一次存下的 DK 盤口，重算當時的賽前分析，再跟實際賽果比對。

看三件事：
  1. 市場準不準：熱門打出幾場、爆冷幾場、進球數 vs 賽前預期
  2. 盤口變化（資金方向）有沒有說中
  3. 模型（市場 + 歷史相似盤口）比單看市場好還是差；假設照模型下注的損益
"""
from __future__ import annotations

import json
import math
from datetime import timedelta
from html import escape

import live
from analyze import MIN_EDGE, football_model, zh
from core import db, now_utc, parse_iso, to_tpe
from strategies import settle_handicap

UPSET_P = 0.25   # 實際賽果的賽前機率低於這個就算爆冷


def _prematch_odds(conn, game_id: str, kickoff) -> tuple[dict | None, object]:
    row = conn.execute("SELECT * FROM snapshots WHERE game_id=? AND ts<=? ORDER BY ts DESC LIMIT 1",
                       (game_id, kickoff.isoformat(timespec="seconds"))).fetchone()
    if not row:
        return None, None
    odds = json.loads(row["open_json"] or "{}")
    odds.update({"c_ml_home": row["ml_home"], "c_ml_draw": row["ml_draw"], "c_ml_away": row["ml_away"],
                 "c_spread_home": row["spread_home"], "c_spread_home_odds": row["spread_home_odds"],
                 "c_spread_away_odds": row["spread_away_odds"], "c_total": row["total"],
                 "c_over_odds": row["over_odds"], "c_under_odds": row["under_odds"]})
    return odds, parse_iso(row["ts"])


def _profit(opt: dict, m: dict, hs: int, as_: int) -> float:
    market, pick = opt["key"]
    if market == "1x2":
        res = "H" if hs > as_ else "A" if hs < as_ else "D"
        return opt["odds"] - 1 if pick == res else -1.0
    if market == "total":
        goals = hs + as_
        return settle_handicap(goals - m["L"] if pick == "over" else m["L"] - goals, 0, opt["odds"])
    margin = hs - as_ if pick == "home" else as_ - hs
    return settle_handicap(margin, m["sp"] if pick == "home" else -m["sp"], opt["odds"])


def _mark(x: float) -> str:
    return "✅" if x > 0 else "❌" if x < 0 else "➖"


def review_games(hours: float = 24) -> list[dict]:
    """過去 hours 小時內完賽、且有賽前快照的足球比賽。"""
    games = live.fetch_games()
    now = now_utc()
    conn = db()
    out = []
    try:
        for g in games:
            kick = parse_iso(g["kickoff"])
            if g["sport"] != "football" or not g["completed"] or not 0 < (now - kick).total_seconds() / 3600 <= hours:
                continue
            odds, ts = _prematch_odds(conn, g["game_id"], kick)
            if not odds:
                continue
            gg = dict(g, odds=odds)
            m = football_model(gg)
            if not m:
                continue
            hs, as_ = g["home_score"], g["away_score"]
            res_i = 0 if hs > as_ else 2 if hs < as_ else 1
            for o in m["options"]:
                o["profit"] = _profit(o, m, hs, as_)
            p_model = next(o["p"] for o in m["options"] if o["key"] == ("1x2", "HDA"[res_i]))
            steam = None
            if m["po"]:
                moves = [b - a for a, b in zip(m["po"], m["pc"])]
                i = max(range(3), key=lambda k: abs(moves[k]))
                if abs(moves[i]) >= 0.03:
                    # 資金湧入的結果有沒有打出；若是某結果被拉高賠率（機率下降），看它是否沒打出
                    steam = {"i": i, "move": moves[i], "right": (res_i == i) == (moves[i] > 0)}
            out.append({"g": g, "m": m, "ts": ts, "mins_before": (kick - ts).total_seconds() / 60,
                        "res_i": res_i, "p_mkt": m["pc"][res_i], "p_model": p_model, "steam": steam})
    finally:
        conn.close()
    return sorted(out, key=lambda r: (r["g"]["league"], r["g"]["kickoff"]))


def match_block(r: dict) -> str:
    g, m = r["g"], r["m"]
    hs, as_ = g["home_score"], g["away_score"]
    res_name = ["主勝", "和局", "客勝"][r["res_i"]]
    upset = " 🔥爆冷" if r["p_mkt"] < UPSET_P else ""
    lines = [f"⚽ {escape(g['league'])}｜{live.kickoff_str(g)}",
             f"<b>{escape(zh(g['home']))} {hs} - {as_} {escape(zh(g['away']))}</b>",
             f"賽前（開賽前 {r['mins_before']:.0f} 分鐘）：勝/和/負 " + "/".join(f"{p*100:.0f}" for p in m["pc"]) + "%"
             + (f"｜大 {m['L']:g} 球 {m['pOL']*100:.0f}%（預期 {m['lam']:.1f} 球）" if m["lam"] else ""),
             f"實際：{res_name}（市場給 {r['p_mkt']*100:.0f}%）{upset}｜{hs + as_} 球"]
    picks = []
    for o in m["options"]:
        if o["key"][0] != "1x2":
            picks.append(f"{o['name']}{_mark(o['profit'])}")
    if picks:
        lines.append("　" + "　".join(escape(p) for p in picks))
    if r["steam"]:
        s = r["steam"]
        lines.append(f"資金方向：{['主勝', '和局', '客勝'][s['i']]} {s['move']*100:+.0f}pp → {'✅ 說中' if s['right'] else '❌ 沒說中'}")
    b = m["best"]
    if b:
        tag = "建議" if b["ev"] >= MIN_EDGE else "觀望（若硬買）"
        lines.append(f"模型最佳：{escape(b['name'])}（EV {b['ev']*100:+.1f}%，{tag}）→ {_mark(b['profit'])} {b['profit']:+.2f} 注")
    return "\n".join(lines)


def summary(rs: list[dict], hours: float) -> str:
    if not rs:
        return f"過去 {hours:g} 小時沒有可檢討的比賽（需要賽前有存到盤口快照）。"
    n = len(rs)
    leagues = {}
    for r in rs:
        leagues[r["g"]["league"]] = leagues.get(r["g"]["league"], 0) + 1
    fav_hits = sum(r["res_i"] == max(range(3), key=lambda i: r["m"]["pc"][i]) for r in rs)
    upsets = [r for r in rs if r["p_mkt"] < UPSET_P]
    goals = sum(r["g"]["home_score"] + r["g"]["away_score"] for r in rs)
    exp_goals = sum(r["m"]["lam"] or 0 for r in rs)
    overs = [next((o for o in r["m"]["options"] if o["key"] == ("total", "over")), None) for r in rs]
    over_w = sum(1 for o in overs if o and o["profit"] > 0)
    over_l = sum(1 for o in overs if o and o["profit"] < 0)
    steams = [r["steam"] for r in rs if r["steam"]]
    ll_mkt = sum(-math.log(r["p_mkt"]) for r in rs) / n
    ll_mod = sum(-math.log(r["p_model"]) for r in rs) / n
    best = [r["m"]["best"] for r in rs if r["m"]["best"]]
    best_units = sum(b["profit"] for b in best)
    rec = [b for b in best if b["ev"] >= MIN_EDGE]
    fav_units = sum(next(o["profit"] for o in r["m"]["options"]
                         if o["key"] == ("1x2", "HDA"[max(range(3), key=lambda i: r["m"]["pc"][i])])) for r in rs)
    t0 = to_tpe(min(parse_iso(r["g"]["kickoff"]) for r in rs))
    lines = [f"🧾 <b>賽後檢討</b>｜{t0:%m/%d} 起 {n} 場（" + "、".join(f"{k} {v}" for k, v in leagues.items()) + "）", "",
             "<b>市場準不準</b>",
             f"• 熱門結果打出 {fav_hits}/{n} 場（賽前平均預期約 {sum(max(r['m']['pc']) for r in rs)/n*100:.0f}%）",
             f"• 爆冷（賽前機率 &lt;{UPSET_P*100:.0f}%）{len(upsets)} 場"
             + (f"：" + "、".join(f"{escape(zh(r['g']['home']))} {r['g']['home_score']}-{r['g']['away_score']} "
                                   f"{escape(zh(r['g']['away']))}（{r['p_mkt']*100:.0f}%）" for r in upsets) if upsets else ""),
             f"• 總進球 {goals}（平均 {goals/n:.2f}）vs 賽前預期 {exp_goals:.1f}（平均 {exp_goals/n:.2f}）",
             f"• DK 大小盤：大 {over_w} 場、小 {over_l} 場"]
    if steams:
        lines.append(f"• 盤口變化 ≥3pp 的 {len(steams)} 場：資金方向說中 {sum(s['right'] for s in steams)} 場")
    lines += ["", "<b>模型表現</b>",
              f"• 實際結果的賽前平均機率：市場 {sum(r['p_mkt'] for r in rs)/n*100:.1f}% / 模型 {sum(r['p_model'] for r in rs)/n*100:.1f}%"
              f"（log loss {ll_mkt:.3f} / {ll_mod:.3f}，越低越好）",
              f"• 系統建議下注（EV ≥{MIN_EDGE*100:.0f}%）：{len(rec)} 注" + (f"，{sum(b['profit'] for b in rec):+.2f} 注" if rec else "（全部觀望 → 0 損失）"),
              f"• 假設每場硬買「EV 最高」的選項：{sum(b['profit'] > 0 for b in best)} 勝 {sum(b['profit'] < 0 for b in best)} 負，{best_units:+.2f} 注",
              f"• 對照：每場買勝平負熱門：{fav_units:+.2f} 注"]
    return "\n".join(lines)


def build(hours: float = 24) -> list[str]:
    rs = review_games(hours)
    return [summary(rs, hours)] + [match_block(r) for r in rs]


# =============================================================================
# 每日盤後分析（足球 + MLB/NBA + 串關 + 學習）
# =============================================================================
def us_sports_blocks(conn, hours: float) -> list[str]:
    """MLB / NBA：用 predictions 表（開賽前最後一次記錄的模型預測）對照賽果。"""
    since = (now_utc() - timedelta(hours=hours + 4)).isoformat()
    games = conn.execute("""SELECT DISTINCT p.game_id, g.* FROM predictions p JOIN games g ON g.game_id = p.game_id
                            WHERE p.sport IN ('mlb','nba') AND g.status='final' AND g.kickoff >= ?
                            ORDER BY g.kickoff""", (since,)).fetchall()
    blocks = []
    for g in games:
        opts = conn.execute("SELECT * FROM predictions WHERE game_id=?", (g["game_id"],)).fetchall()
        icon = "⚾" if g["sport"] == "mlb" else "🏀"
        gg = dict(g)
        title = f"{live.team_name(gg, 'away')} {g['away_score']} @ {g['home_score']} {live.team_name(gg, 'home')}"
        ml = {o["pick"]: o for o in opts if o["market"] == "ml"}
        winner = "home" if g["home_score"] > g["away_score"] else "away"
        lines = [f"{icon} {g['league']}｜{to_tpe(parse_iso(g['kickoff'])):%m/%d %H:%M}", f"<b>{escape(title)}</b>"]
        if winner in ml:
            p = ml[winner]["p_mkt"]
            lines.append(f"實際：{escape(live.team_name(gg, winner))} 勝（賽前市場給 {p*100:.0f}%）" + (" 🔥爆冷" if p < 0.35 else ""))
        marks = [f"{_opt_name(gg, o)}{_mark(o['profit'] or 0)}" for o in opts if o["result"] in ("W", "L", "P")]
        lines.append("　" + "　".join(escape(x) for x in marks))
        best = max(opts, key=lambda o: o["p_model"] * o["odds"] - 1)
        ev = best["p_model"] * best["odds"] - 1
        lines.append(f"模型最佳：{escape(_opt_name(gg, best))}（EV {ev*100:+.1f}%{'' if ev >= MIN_EDGE else '，觀望'}）→ "
                     f"{_mark(best['profit'] or 0)} {best['profit'] or 0:+.2f} 注")
        blocks.append("\n".join(lines))
    return blocks


def _opt_name(g: dict, o) -> str:
    if o["market"] == "ml":
        return f"{live.team_name(g, o['pick']).split()[0]} 不讓分"
    if o["market"] == "total":
        return f"{'大' if o['pick'] == 'over' else '小'} {o['line']:g}"
    if o["market"] == "spread":
        line = o["line"] if o["pick"] == "home" else -o["line"]
        return f"{live.team_name(g, o['pick']).split()[0]} {line:+g}"
    return o["pick"]


def parlay_blocks(conn, hours: float) -> list[str]:
    import parlay
    conn.executescript(parlay.PARLAY_SCHEMA)
    since = (now_utc() - timedelta(hours=hours + 6)).isoformat()
    rows = conn.execute("SELECT * FROM parlays WHERE result IS NOT NULL AND settled >= ? ORDER BY id", (since,)).fetchall()
    lines = ["🎟️ <b>串關結果</b>"]
    if rows:
        for r in rows:
            legs = json.loads(r["legs"])
            names = " + ".join(escape(l.get("name", "")) for l in legs)
            lines.append(f"{'✅中' if r['result'] == 'W' else '❌'} {escape(r['label'])}：{names}"
                         f"（命中率 {r['p']*100:.0f}%，{r['profit']:+.2f} 注）")
    else:
        lines.append("（這段期間沒有結算的串關）")
    lines.append("近 7 天：" + parlay.record_lines(conn, 7)[0])
    lines.append("累計：" + parlay.record_lines(conn)[0])
    return ["\n".join(lines)]


def daily(hours: float = 24) -> list[str]:
    """每天的盤後分析：先結算、更新學習，再組訊息。回傳 [摘要, 各段落...]。"""
    import learning
    import parlay
    conn = db()
    try:
        learning.settle_predictions(conn)
        parlay.settle_parlays(conn)
        learn = learning.update(conn)
        calib = learning.calibration(conn)
        fb = review_games(hours)
        us = us_sports_blocks(conn, hours)
        par = parlay_blocks(conn, hours)
    finally:
        conn.close()
    head = [summary(fb, hours) if fb else f"🧾 <b>盤後分析</b>｜過去 {hours:g} 小時沒有可檢討的足球比賽"]
    tail = ["🧠 <b>模型學習狀態</b>"] + (learn or ["（還沒有已結算的預測）"])
    if calib:
        tail += ["校準（模型給的機率 vs 實際發生比例，越接近越好）："] + calib
    tail.append("模型會自動調整「市場 vs 歷史相似盤口」的權重：哪個比較準就多信哪個。")
    return head + par + ["\n".join(tail)] + [match_block(r) for r in fb] + us
