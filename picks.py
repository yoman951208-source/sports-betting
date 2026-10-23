# -*- coding: utf-8 -*-
"""今日最推薦：掃描所有比賽（足球 / NBA / MLB）的所有選項，給出最推薦的單關、二串一、三串一。

排序依據（都來自前面驗證過的模型與資料）：
  1. 期望值 EV = 估計勝率 × DK 賠率 − 1（估計勝率 = 校準後市場機率 + 球員狀態；⭐ 訊號成立時加上回測的 uplift）
  2. 只考慮賠率 1.20–4.50（10/19 分析：高賠率冷門長期虧最多）
  3. 通過歷史回測的訊號（⭐）額外加分
評等：🟢 推薦（EV ≥ +2%）／🟡 接近公平（−2% ~ +2%）／🔴 偏貴（< −2%，只是「最不虧」）
"""
from __future__ import annotations

import math
from html import escape
from itertools import combinations

import live
import ui
from analyze import game_model, game_title, upcoming
from backtest import load_strategies
from core import db
from strategies import no_vig2

SINGLE_ODDS = (1.20, 4.50)
LEG_ODDS = (1.25, 2.60)       # 串關每關：太高的冷門會讓整張命中率掉太多
LEG_MIN_EV = -0.03
MIN_HIT = {2: 0.25, 3: 0.12}
SIGNAL_BONUS = 0.02
ICON = {"football": "⚽", "nba": "🏀", "mlb": "⚾"}


grade = ui.grade


def _signals(g: dict, m: dict, strats) -> None:
    """⭐ 回測訊號成立的選項：估計勝率加上回測的 uplift（analyze.apply_signals，和單場分析一致）。"""
    from analyze import apply_signals
    conn = db()
    try:
        f = live.features(conn, g, {})
    finally:
        conn.close()
    apply_signals(g, m, strats, f)


def _open_prob(o: dict, g: dict, m: dict) -> float | None:
    od = g["odds"]
    market, pick = o["key"]
    if market == "1x2" and m.get("po"):
        return m["po"]["HDA".index(pick)]
    if market == "ml":
        p = no_vig2(od.get("o_ml_home"), od.get("o_ml_away"))
        return None if p is None else (p if pick == "home" else 1 - p)
    return None


def reasons(o: dict, g: dict, m: dict, signal: str | None) -> list[str]:
    r = []
    ev = o["ev"]
    r.append(f"DK 賠率高於合理價（期望值 {ev*100:+.1f}%）" if ev > 0 else f"在所有選項中最接近公平價（期望值 {ev*100:+.1f}%）")
    if signal:
        r.append(f"⭐ 通過歷史回測的訊號：{signal}（回測勝率比市場高 {(o['p'] - o['p_mkt'])*100:.1f}pp）")
    po = _open_prob(o, g, m)
    if po is not None and o["p_mkt"] - po >= 0.03:
        r.append(f"資金往這邊：開盤 {po*100:.0f}% → 現在 {o['p_mkt']*100:.0f}%")
    form = m.get("form") or {}
    if g["sport"] == "nba" and form.get("d_star_load") is not None and o["key"][0] in ("spread", "ml"):
        d = form["d_star_load"] if o["key"][1] == "home" else -form["d_star_load"]
        if d <= -1.5:
            r.append(f"對手主力最近比較累（近 3 場多打 {abs(d):.1f} 分鐘）")
    raw = o.get("p_raw")
    if raw is not None and o["p_mkt"] - raw >= 0.01 and g["sport"] == "football" and o["key"][0] == "1x2":
        r.append(f"冷熱門校準：原始 {raw*100:.0f}% → 校準後 {o['p_mkt']*100:.0f}%（熱門通常被低估）")
    if o["odds"] < 1.50:
        r.append("賠率低於 1.5：抽水最輕的區間（8 季回測約 −2.6%，5 倍以上冷門約 −17%）")
    if g["sport"] == "football" and o["key"][0] == "spread" and m.get("sp") is not None and abs(m["sp"]) >= 1.5:
        dog = "home" if m["sp"] > 0 else "away"
        if o["key"][1] == dog:
            r.append("深盤買受讓方：歷史上比讓球方少虧")
    if g["sport"] == "football" and o["key"] == ("spread", "away") and raw is not None and o["p_mkt"] - raw >= 0.01:
        r.append(f"莊家角度：讓球盤主隊常被多抽，客隊這邊校準後 {raw*100:.0f}% → {o['p_mkt']*100:.0f}%（8 季回測中 7 季都一樣）")
    return r


def candidates(games: list[dict], hours: float) -> list[dict]:
    strats = load_strategies()[0]
    out = []
    for g in upcoming(games, hours):
        if g["sport"] not in ICON or not g["odds"]:
            continue
        try:
            m = game_model(g)
        except Exception:
            continue
        if not m:
            continue
        _signals(g, m, strats)
        for o in m["options"]:
            if not SINGLE_ODDS[0] <= o["odds"] <= SINGLE_ODDS[1]:
                continue
            s = o.get("signal")
            out.append({**o, "g": g, "m": m, "signal": s, "score": o["ev"] + (SIGNAL_BONUS if s else 0)})
    return sorted(out, key=lambda x: (-x["score"], -x["p"]))


def best_parlay(cands: list[dict], size: int) -> dict | None:
    legs = [c for c in cands if LEG_ODDS[0] <= c["odds"] <= LEG_ODDS[1] and c["ev"] >= LEG_MIN_EV][:30]
    best = None
    for combo in combinations(legs, size):
        if len({c["g"]["game_id"] for c in combo}) < size:
            continue
        p = math.prod(c["p"] for c in combo)
        if p < MIN_HIT[size]:
            continue
        ev = math.prod(c["p"] * c["odds"] for c in combo) - 1
        score = ev + SIGNAL_BONUS * sum(bool(c["signal"]) for c in combo)
        if best is None or score > best["score"]:
            best = {"legs": combo, "p": p, "ev": ev, "odds": math.prod(c["odds"] for c in combo), "fair": 1 / p, "score": score}
    return best


def _pick_block(c: dict, title: str) -> str:
    g = c["g"]
    lines = [f"<b>{title}</b>",
             f"{ICON[g['sport']]} {escape(g['league'])}｜{live.kickoff_str(g)}",
             f"<b>{escape(game_title(g))}</b>",
             f"👉 <b>{escape(c['name'])}</b>",
             f"   估計勝率 <b>{c['p']*100:.0f}%</b>｜合理賠率 {c['fair']:.2f}｜DK {c['odds']:.2f}",
             f"   📌 <b>運彩賠率 ≥ {c['fair']:.2f} 才下</b>",
             f"   評等：{grade(c['ev'])}",
             "   理由："]
    lines += [f"   • {escape(x)}" for x in reasons(c, g, c["m"], c["signal"])]
    return "\n".join(lines)


def _parlay_block(t: dict, title: str) -> str:
    lines = [f"<b>{title}</b>"]
    for i, c in enumerate(t["legs"], 1):
        g = c["g"]
        star = " ⭐" if c["signal"] else ""
        lines.append(f"{i}. {ICON[g['sport']]} {live.kickoff_str(g)} {escape(game_title(g))}\n"
                     f"   👉 <b>{escape(c['name'])}</b>{star}｜{c['p']*100:.0f}%｜合理 {c['fair']:.2f}（DK {c['odds']:.2f}）")
    lines.append(f"命中率 <b>{t['p']*100:.0f}%</b>｜📌 <b>運彩組合賠率 ≥ {t['fair']:.2f} 才下</b>（DK 組合 {t['odds']:.2f}）")
    lines.append(f"評等：{grade(t['ev'])}（期望值 {t['ev']*100:+.1f}%）")
    # 賠率 × 等比例去水機率 = 1 /（1 + 抽水），串關時逐關相乘
    keep = [c["odds"] * (c.get("p_raw") or c["p_mkt"]) for c in t["legs"]]
    lines.append("🏦 莊家抽水：每關 " + "/".join(f"{(1 - k)*100:.0f}%" for k in keep)
                 + f" → 整張累積約 <b>{(1 - math.prod(keep))*100:.0f}%</b>（DK；運彩通常更高）")
    return "\n".join(lines)


def report(games: list[dict], hours: float = 24, save_conn=None) -> list[str]:
    cands = candidates(games, hours)
    n_games = len({c["g"]["game_id"] for c in cands})
    if not cands:
        return [f"🏆 <b>今日最推薦</b>\n未來 {hours:g} 小時沒有可分析的比賽（足球 / NBA / MLB）。"]
    single = cands[0]
    p2, p3 = best_parlay(cands, 2), best_parlay(cands, 3)
    green = [c for c in cands if c["ev"] >= 0.02]
    head = [f"🏆 <b>今日最推薦</b>（未來 {hours:g} 小時，掃描 {n_games} 場、{len(cands)} 個選項）", ""]
    if green:
        head.append(f"✅ 今天有 <b>{len(green)} 個</b>選項的 DK 賠率高於合理價（🟢）。")
    else:
        head.append("⚠️ <b>今天沒有明顯有價值的選項</b>：下面是「最不虧」的選擇，不買也是好選擇。")
    head.append("判斷方法只有一條：<b>運彩賠率 ≥ 📌 的數字才下</b>，低於就跳過；只用小注。")
    blocks = ["\n".join(head), _pick_block(single, "🥇 最推薦單關")]
    if len(cands) > 1:
        others = [c for c in cands[1:4]]
        lines = ["<b>🥈 其他單關候選</b>"]
        for c in others:
            lines.append(f"{ui.grade_icon(c['ev'])} {ICON[c['g']['sport']]} {live.kickoff_str(c['g'])} {escape(game_title(c['g']))}\n"
                         f"　 👉 {escape(c['name'])} @{c['odds']:.2f}｜{c['p']*100:.0f}%｜📌 運彩 ≥ {c['fair']:.2f}")
        blocks.append("\n".join(lines))
    for t, size, cn in ((p2, 2, "二串一"), (p3, 3, "三串一")):
        blocks.append(_parlay_block(t, f"🎫 最推薦{cn}") if t else
                      f"🎫 <b>最推薦{cn}</b>\n今天找不到命中率 ≥{MIN_HIT[size]*100:.0f}% 且每關都接近公平價的組合 → 建議不串")
    if save_conn is not None:
        _save(save_conn, single, p2, p3)
        import parlay
        rec = [f"單關 {parlay.record_lines(save_conn, 30, '最推薦｜單關')[0]}",
               f"二串一 {parlay.record_lines(save_conn, 30, '最推薦｜二串一')[0]}",
               f"三串一 {parlay.record_lines(save_conn, 30, '最推薦｜三串一')[0]}"]
        blocks.append("📒 <b>最推薦近 30 天戰績</b>\n" + "\n".join(rec))
    return blocks


def _save(conn, single: dict, p2: dict | None, p3: dict | None) -> None:
    import parlay
    tickets = [("最推薦｜單關", {"legs": (single,), "p": single["p"], "fair": single["fair"], "odds": single["odds"]})]
    if p2:
        tickets.append(("最推薦｜二串一", p2))
    if p3:
        tickets.append(("最推薦｜三串一", p3))
    parlay.save_tickets(conn, tickets)
