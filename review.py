# -*- coding: utf-8 -*-
"""盤後分析：每場只列「我們推薦的那一注」有沒有中（10/23 介面整理），加上一行市場摘要。

推薦 = 開賽前最後一次盤口算出的模型中，期望值最高的選項。EV ≥ +2% 才算真的推薦（🟢）；
其他場次標「沒推薦」，括號附上最接近的選項有沒有中，給模型檢討用。
足球用賽前快照重算；NBA / MLB 用開賽前記錄的 predictions。最推薦的單關／串關另外列在最前面。
"""
from __future__ import annotations

import json
from datetime import timedelta
from html import escape

import live
import ui
from analyze import MIN_EDGE, football_model, zh_short
from core import db, now_utc, parse_iso, to_tpe
from strategies import settle_handicap

UPSET_P = 0.25   # 足球：實際賽果的賽前機率低於這個就算爆冷


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
            out.append({"g": g, "m": m, "res_i": res_i, "p_mkt": m["pc"][res_i]})
    finally:
        conn.close()
    return sorted(out, key=lambda r: (r["g"]["league"], r["g"]["kickoff"]))


# =============================================================================
# 共用：每場一行推薦結果、推薦 / 沒推薦的小計
# =============================================================================
def _pick_line(name: str, odds: float, ev: float, profit: float | None) -> str:
    if ev >= MIN_EDGE:
        return f"　🟢 推薦 <b>{escape(name)}</b> @{odds:.2f} → {ui.result_text(profit)}"
    return f"　⚪ 沒推薦（最接近的 {escape(name)} {ui.mark(profit)}）"


def _tally(picks: list[tuple[float, float]]) -> list[str]:
    """picks = [(期望值, 損益)] → 推薦 / 沒推薦的戰績。"""
    rec = [p or 0 for ev, p in picks if ev >= MIN_EDGE]
    rest = [p or 0 for ev, p in picks if ev < MIN_EDGE]
    out = [f"🟢 推薦 {len(rec)} 注：中 {sum(p > 0 for p in rec)}、沒中 {sum(p < 0 for p in rec)}｜<b>{sum(rec):+.2f} 注</b>"
           if rec else "🟢 推薦 0 注（全部觀望）"]
    if rest:
        out.append(f"⚪ 沒推薦 {len(rest)} 場（若硬買最接近的：中 {sum(p > 0 for p in rest)}、沒中 {sum(p < 0 for p in rest)}，{sum(rest):+.2f} 注）")
    return out


def _opt_name(g: dict, o) -> str:
    if o["market"] == "ml":
        return f"{live.team_name(g, o['pick']).split()[0]} 不讓分"
    if o["market"] == "total":
        return f"{'大' if o['pick'] == 'over' else '小'} {o['line']:g}"
    if o["market"] == "spread":
        line = o["line"] if o["pick"] == "home" else -o["line"]
        return f"{live.team_name(g, o['pick']).split()[0]} {line:+g}"
    return o["pick"]


def _ev(o) -> float:
    return o["p_model"] * o["odds"] - 1


# =============================================================================
# 足球
# =============================================================================
def football_section(rs: list[dict]) -> str | None:
    if not rs:
        return None
    n = len(rs)
    fav_hits = sum(r["res_i"] == max(range(3), key=lambda i: r["m"]["pc"][i]) for r in rs)
    goals = sum(r["g"]["home_score"] + r["g"]["away_score"] for r in rs)
    exp_goals = sum(r["m"]["lam"] or 0 for r in rs)
    picks = [(r["m"]["best"]["ev"], r["m"]["best"]["profit"]) for r in rs if r["m"]["best"]]
    lines = [f"⚽ <b>足球</b>（{n} 場）", *_tally(picks),
             f"市場：熱門打出 {fav_hits}/{n}｜爆冷 {sum(r['p_mkt'] < UPSET_P for r in rs)} 場｜"
             f"進球 {goals}（賽前預期 {exp_goals:.0f}）"]
    league = None
    for r in rs:
        g, b = r["g"], r["m"]["best"]
        if g["league"] != league:
            league = g["league"]
            lines += ["", f"<b>【{escape(league)}】</b>"]
        upset = " 🔥爆冷" if r["p_mkt"] < UPSET_P else ""
        gm = r["m"].get("goals")
        pred = ""
        if gm:
            top = gm["top"][0][0]
            pred = f"（預測 {top}{' 🎯猜中' if top == f'{g['home_score']}-{g['away_score']}' else ''}）"
        lines.append(f"{escape(zh_short(g['home']))} <b>{g['home_score']}-{g['away_score']}</b> {escape(zh_short(g['away']))}{upset}{pred}")
        if b:
            lines.append(_pick_line(b.get("short", b["name"]), b["odds"], b["ev"], b["profit"]))
    return "\n".join(lines)


# =============================================================================
# NBA / MLB（用開賽前記錄的 predictions）
# =============================================================================
def _nba_player_line(rows: list[dict], team: str) -> tuple[str, list[str]]:
    """某隊的得分王與缺陣者。回傳（得分王文字, 缺陣名單）。"""
    mine = [r for r in rows if r["team"] == team]
    played = sorted([r for r in mine if not r["dnp"] and (r.get("min") or 0) > 0], key=lambda r: -(r.get("pts") or 0))
    top = (f"{played[0]['name'].split()[-1]} {int(played[0]['pts'] or 0)}分{int(played[0]['reb'] or 0)}板"
           f"{int(played[0]['ast'] or 0)}助") if played else ""
    out = [r["name"] for r in mine
           if r["dnp"] and r.get("reason") and any(k in r["reason"].upper() for k in ("INJURY", "ILLNESS", "REST", "OUT", "PERSONAL"))]
    return top, out[:3]


def nba_section(conn, hours: float) -> str | None:
    import data_nba_box
    since = (now_utc() - timedelta(hours=hours + 4)).isoformat()
    games = conn.execute("""SELECT * FROM games WHERE sport='nba' AND status='final' AND kickoff >= ?
                            ORDER BY kickoff""", (since,)).fetchall()
    rows = []
    for g in games:
        odds, _ts = _prematch_odds(conn, g["game_id"], parse_iso(g["kickoff"]))
        if not odds or odds.get("c_spread_home") is None:
            continue
        margin = g["home_score"] - g["away_score"]
        pm = None
        if odds.get("c_ml_home") and odds.get("c_ml_away"):
            pm = (1 / odds["c_ml_home"]) / (1 / odds["c_ml_home"] + 1 / odds["c_ml_away"])
        opts = conn.execute("SELECT * FROM predictions WHERE game_id=? AND result IN ('W','L','P')", (g["game_id"],)).fetchall()
        rows.append({"g": dict(g), "odds": odds, "box": data_nba_box.fetch_box(g["game_id"].split(":")[-1]) or [],
                     "margin": margin, "cover": margin + odds["c_spread_home"], "pts": g["home_score"] + g["away_score"],
                     "tot": odds.get("c_total"), "upset": pm is not None and ((margin > 0 and pm < 0.35) or (margin < 0 and pm > 0.65)),
                     "best": max(opts, key=_ev) if opts else None})
    if not rows:
        return None
    decided = [r for r in rows if r["cover"] != 0]
    fav_cover = sum((r["cover"] > 0) == (r["odds"]["c_spread_home"] <= 0) for r in decided)
    overs = sum(1 for r in rows if r["tot"] and r["pts"] > r["tot"])
    unders = sum(1 for r in rows if r["tot"] and r["pts"] < r["tot"])
    picks = [(_ev(r["best"]), r["best"]["profit"]) for r in rows if r["best"]]
    lines = [f"🏀 <b>NBA</b>（{len(rows)} 場）", *_tally(picks),
             f"市場：讓分熱門贏盤 {fav_cover}/{len(decided)}｜大 {overs} 小 {unders}｜爆冷 {sum(r['upset'] for r in rows)} 場"]
    for r in rows:
        g = r["g"]
        lines += ["", f"{escape(live.team_name(g, 'away'))} <b>{g['away_score']} @ {g['home_score']}</b> "
                      f"{escape(live.team_name(g, 'home'))}" + (" 🔥爆冷" if r["upset"] else "")]
        if r["best"]:
            b = r["best"]
            lines.append(_pick_line(_opt_name(g, b), b["odds"], _ev(b), b["profit"]))
        if r["box"]:
            tops, outs = [], []
            for side, team in (("away", g["away_abbr"]), ("home", g["home_abbr"])):
                top, out = _nba_player_line(r["box"], team)
                if top:
                    tops.append(f"{live.team_name(g, side)} {top}")
                outs += out
            if tops:
                lines.append("　🏅 " + escape("｜".join(tops)))
            if outs:
                lines.append("　🚑 缺陣：" + escape("、".join(outs)))
    return "\n".join(lines)


def mlb_section(conn, hours: float) -> str | None:
    since = (now_utc() - timedelta(hours=hours + 4)).isoformat()
    games = conn.execute("""SELECT DISTINCT g.* FROM predictions p JOIN games g ON g.game_id = p.game_id
                            WHERE p.sport = 'mlb' AND g.status='final' AND g.kickoff >= ?
                            ORDER BY g.kickoff""", (since,)).fetchall()
    rows = []
    for g in games:
        opts = conn.execute("SELECT * FROM predictions WHERE game_id=? AND result IN ('W','L','P')", (g["game_id"],)).fetchall()
        if opts:
            rows.append((dict(g), max(opts, key=_ev)))
    if not rows:
        return None
    lines = [f"⚾ <b>MLB</b>（{len(rows)} 場）", *_tally([(_ev(b), b["profit"]) for _, b in rows])]
    for g, b in rows:
        lines += ["", f"{escape(live.team_name(g, 'away'))} <b>{g['away_score']} @ {g['home_score']}</b> {escape(live.team_name(g, 'home'))}",
                  _pick_line(_opt_name(g, b), b["odds"], _ev(b), b["profit"])]
    return "\n".join(lines)


# =============================================================================
# 最推薦與串關
# =============================================================================
def _leg_profit(conn, leg: dict) -> float | None:
    from learning import leg_profit
    g = conn.execute("SELECT status, home_score, away_score FROM games WHERE game_id=?", (leg["game_id"],)).fetchone()
    if not g or g["status"] != "final" or g["home_score"] is None:
        return None
    return leg_profit(leg["sport"], leg["market"], leg["pick"], leg["line"], leg["odds"], g["home_score"], g["away_score"])


def _ticket_line(conn, r, label: str) -> str:
    legs = json.loads(r["legs"])
    if len(legs) == 1:
        l = legs[0]
        body = f"{escape(l.get('name', ''))} @{l['odds']:.2f}（{escape(l.get('title', ''))}）"
    else:
        body = " ＋ ".join(f"{escape(l.get('name', ''))}{ui.mark(_leg_profit(conn, l))}" for l in legs)
    return f"{ui.mark(r['profit'])} <b>{escape(label)}</b>：{body} → {r['profit']:+.2f} 注"


def best_section(conn, hours: float) -> list[str]:
    """最推薦（單關 / 二串一 / 三串一）的結果，最重要所以放最前面。"""
    import parlay
    conn.executescript(parlay.PARLAY_SCHEMA)
    since = (now_utc() - timedelta(hours=hours + 6)).isoformat()
    rows = conn.execute("SELECT * FROM parlays WHERE result IS NOT NULL AND settled >= ? AND label LIKE '最推薦｜%' ORDER BY id",
                        (since,)).fetchall()
    lines = ["🏆 <b>最推薦結果</b>"]
    lines += [_ticket_line(conn, r, r["label"].split("｜", 1)[-1]) for r in rows] or ["（這段期間沒有結算的最推薦）"]
    since30 = (now_utc() - timedelta(days=30)).isoformat()
    rec = []
    for k in ("單關", "二串一", "三串一"):
        rs = conn.execute("SELECT result, profit FROM parlays WHERE result IN ('W','L','P') AND created >= ? AND label = ?",
                          (since30, "最推薦｜" + k)).fetchall()
        rec.append(f"{k} {sum(r['result'] == 'W' for r in rs)}/{len(rs)} 中 {sum(r['profit'] for r in rs):+.1f} 注" if rs else f"{k} 尚無")
    lines.append("近 30 天：" + "｜".join(rec))
    return lines


def parlay_section(conn, hours: float) -> str:
    import parlay
    since = (now_utc() - timedelta(hours=hours + 6)).isoformat()
    rows = conn.execute("SELECT * FROM parlays WHERE result IS NOT NULL AND settled >= ? AND label NOT LIKE '最推薦｜%' ORDER BY id",
                        (since,)).fetchall()
    lines = ["🎟️ <b>每日串關結果</b>"]
    lines += [_ticket_line(conn, r, r["label"]) for r in rows] or ["（這段期間沒有結算的串關）"]
    lines += ["近 7 天：" + parlay.record_lines(conn, 7)[0], "累計：" + parlay.record_lines(conn)[0]]
    return "\n".join(lines)


# =============================================================================
# 組合
# =============================================================================
def _sections(conn, hours: float) -> tuple[str, list[str]]:
    fb = review_games(hours)
    t0 = to_tpe(now_utc() - timedelta(hours=hours))
    head = [f"🧾 <b>盤後分析</b>｜{t0:%m/%d %H:%M} 之後完賽的比賽", "✅ 中　❌ 沒中　➖ 走盤", ""] + best_section(conn, hours)
    blocks = [b for b in (football_section(fb), nba_section(conn, hours), mlb_section(conn, hours)) if b]
    if not blocks:
        blocks = [f"過去 {hours:g} 小時沒有可檢討的比賽（需要賽前有存到盤口快照）。"]
    return "\n".join(head), blocks + [parlay_section(conn, hours)]


def build(hours: float = 24) -> list[str]:
    """「賽後」指令：結算後組訊息（不做模型學習）。回傳 [摘要, 各段落...]。"""
    import learning
    import parlay
    conn = db()
    try:
        learning.settle_predictions(conn)
        parlay.settle_parlays(conn)
        head, blocks = _sections(conn, hours)
    finally:
        conn.close()
    return [head] + blocks


def daily(hours: float = 24) -> list[str]:
    """每天的盤後分析：先結算、更新學習，再組訊息。回傳 [摘要, 各段落...]。"""
    import learning
    import parlay
    conn = db()
    try:
        learning.settle_predictions(conn)
        parlay.settle_parlays(conn)
        learn = learning.update(conn) + learning.check_calibration(conn)
        calib = learning.calibration(conn)
        head, blocks = _sections(conn, hours)
    finally:
        conn.close()
    tail = ["🧠 <b>模型學習狀態</b>"] + (learn or ["（還沒有已結算的預測）"])
    if calib:
        tail += ["校準（模型給的機率 vs 實際發生比例，越接近越好）："] + calib
    return [head] + blocks + ["\n".join(tail)]
