# -*- coding: utf-8 -*-
"""運彩賠率比價（10/24）：使用者在 Telegram 輸入運彩開的賠率，馬上判斷值不值得買。

一行一關，好幾行就當成一張串關：
  湖人 1.75            → 湖人不讓分（足球：那隊勝）
  湖人 -3.5 1.80       → 湖人讓 3.5（也可以寫「讓3.5」；受讓寫 +3.5 或「受讓3.5」）
  湖人 大228.5 1.75    → 那場大分（小分寫「小」）
  利物浦 和 3.40       → 足球和局
估計勝率用和單場分析同一套模型（校準後的市場機率，⭐ 訊號成立時加回測 uplift）。
運彩開的盤口跟 DK 不同時換算：足球用 Poisson 比分模型，NBA / MLB 用常態近似（以 DK 盤口為中心）。
"""
from __future__ import annotations

import math
import re
from html import escape

import config
import live
import ui
from strategies import settle_handicap

MAX_ODDS = 30.0
SIGMA = {"nba": (12.5, 18.0), "mlb": (4.4, 4.3)}   # （分差、總分）的標準差，用來換算不同的盤口

_TOTAL = re.compile(r"(大|小)\s*(\d+(?:\.\d+)?)")
_HCP = re.compile(r"(受讓|讓)\s*(\d+(?:\.\d+)?)|(?<![\d.])([+-]\d+(?:\.\d+)?)")
_DRAW = re.compile(r"(?:^|\s)(和局|平局|和|平)(?=\s|$|\d)")
_NUM = re.compile(r"\d+(?:\.\d+)?")
_NOISE = re.compile(r"不讓分|獨贏|勝|贏|主隊|客隊|@|x|X|×")


def parse_leg(line: str) -> dict | None:
    """一行 → {team, market(ml/draw/total/spread), pick, line, odds}；看不出賠率就回傳 None。"""
    s = line.strip()
    leg: dict = {"raw": s, "market": "ml", "line": None}
    m = _TOTAL.search(s)
    if m:
        leg.update(market="total", pick="over" if m.group(1) == "大" else "under", line=float(m.group(2)))
        s = s[:m.start()] + " " + s[m.end():]
    m = _HCP.search(s)
    if m and leg["market"] == "ml":
        if m.group(1):
            v = float(m.group(2))
            leg.update(market="spread", line=-v if m.group(1) == "讓" else v)
        else:
            leg.update(market="spread", line=float(m.group(3)))
        s = s[:m.start()] + " " + s[m.end():]
    if leg["market"] == "ml" and _DRAW.search(s):
        leg["market"] = "draw"
        s = _DRAW.sub(" ", s)
    nums = [(n.group(), n.start(), n.end()) for n in _NUM.finditer(s)]
    odds = [x for x in nums if "." in x[0] and 1.0 < float(x[0]) <= MAX_ODDS]
    if not odds:
        return None
    val, a, b = odds[-1]
    leg["odds"] = float(val)
    s = s[:a] + " " + s[b:]
    leg["team"] = _NOISE.sub(" ", s).strip(" ,，、+")
    leg["team"] = re.sub(r"\s+", " ", leg["team"]).strip()
    return leg if leg["team"] else None


def parse(text: str) -> list[dict]:
    parts = [p for p in re.split(r"[\n；;]+", text) if p.strip()]
    return [leg for leg in (parse_leg(p) for p in parts) if leg]


def looks_like(text: str) -> bool:
    """訊息裡有「隊名 + 賠率」→ 當作比價（否則照原本的隊名搜尋）。"""
    return bool(parse(text))


# ----------------------------------------------------------------------------- 機率
def _side(g: dict, q: str) -> str | None:
    import analyze
    ql = q.lower()
    target = analyze.ALIASES.get(q, analyze.ALIASES.get(ql, ql))
    zh = config.NBA_TEAM_ZH if g["sport"] == "nba" else config.MLB_TEAM_ZH if g["sport"] == "mlb" else {}
    for side in ("home", "away"):
        name, abbr = g[side].lower(), g[f"{side}_abbr"].lower()
        if q in (zh.get(g[f"{side}_abbr"]), analyze.FOOTBALL_ZH.get(name)) or ql == abbr or target in name or (name and name in target):
            return side
    return None


def _phi(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _phi_inv(p: float) -> float:
    lo, hi = -8.0, 8.0
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if _phi(mid) < p else (lo, mid)
    return (lo + hi) / 2


def _fair_from_dist(dist: dict[float, float], line: float) -> float | None:
    """dist = {所選方的淨勝（或 總分−0）: 機率}；回傳讓 EV = 0 的賠率（處理走盤、四分之一盤）。"""
    ev = lambda o: sum(p * settle_handicap(m, line, o) for m, p in dist.items())
    a = ev(3.0) - ev(2.0)
    if a <= 1e-9:
        return None
    b = a - ev(2.0)
    return 1 + b / a


def _option(m: dict, key: tuple) -> dict | None:
    return next((o for o in m["options"] if o["key"] == key), None)


def leg_fair(g: dict, m: dict, leg: dict) -> tuple[float | None, str]:
    """這一關的合理賠率與說明文字。"""
    import analyze
    sport = g["sport"]
    side = _side(g, leg["team"])
    if leg["market"] in ("ml", "spread") and side is None:
        return None, "看不出是哪一隊"
    if sport == "football":
        pc, lam = m["pc"], m.get("lam") or 2.6
        if leg["market"] == "ml":
            p = (_option(m, ("1x2", "H" if side == "home" else "A")) or {}).get("p")
            return (1 / p if p else None), f"{escape(live.team_name(g, side))} 勝（90 分鐘）"
        if leg["market"] == "draw":
            p = (_option(m, ("1x2", "D")) or {}).get("p")
            return (1 / p if p else None), "和局"
        gm = m.get("goals") or analyze.goal_model(pc, lam)
        if leg["market"] == "total":
            dist = {(n - leg["line"]) if leg["pick"] == "over" else (leg["line"] - n): p for n, p in gm["total"].items()}
        else:
            dist = {(k if side == "home" else -k): p for k, p in gm["margin"].items()}
        if leg["market"] == "total":
            return _fair_from_dist(dist, 0.0), f"{'大' if leg['pick'] == 'over' else '小'} {leg['line']:g} 球"
        return _fair_from_dist(dist, leg["line"]), f"{escape(live.team_name(g, side))} {leg['line']:+g}"
    # NBA / MLB
    o = g["odds"]
    s_m, s_t = SIGMA[sport]
    if leg["market"] == "draw":
        return None, "籃球 / 棒球沒有和局"
    if leg["market"] == "ml":
        p = (_option(m, ("ml", side)) or {}).get("p")
        return (1 / p if p else None), f"{escape(live.team_name(g, side))} 不讓分"
    if leg["market"] == "total":
        L, base = o.get("c_total"), _option(m, ("total", "over"))
        if L is None or not base:
            return None, "目前沒有大小分盤口"
        if abs(leg["line"] - L) < 1e-9:
            p = base["p"] if leg["pick"] == "over" else _option(m, ("total", "under"))["p"]
        else:
            mu = L + s_t * _phi_inv(base["p"])
            p_over = 1 - _phi((leg["line"] - mu) / s_t)
            p = p_over if leg["pick"] == "over" else 1 - p_over
        return 1 / p, f"{'大' if leg['pick'] == 'over' else '小'} {leg['line']:g}" + ("" if abs(leg["line"] - L) < 1e-9 else f"（DK 是 {L:g}，換算）")
    sp, base = o.get("c_spread_home"), _option(m, ("spread", "home"))
    if sp is None or not base:
        return None, "目前沒有讓分盤口"
    home_line = leg["line"] if side == "home" else -leg["line"]
    if abs(home_line - sp) < 1e-9:
        p = base["p"] if side == "home" else _option(m, ("spread", "away"))["p"]
    else:
        mu = s_m * _phi_inv(base["p"]) - sp        # 主隊預期淨勝分（讓 DK 盤口的贏盤機率剛好等於模型）
        p_home = _phi((mu + home_line) / s_m)
        p = p_home if side == "home" else 1 - p_home
    note = "" if abs(home_line - sp) < 1e-9 else f"（DK 是 主{sp:+g}，換算）"
    return 1 / p, f"{escape(live.team_name(g, side))} {leg['line']:+g}{note}"


# ----------------------------------------------------------------------------- 報告
def _verdict(ev: float) -> str:
    return ("🟢 <b>值得買</b>" if ev >= ui.GOOD else "🟡 <b>接近公平</b>（可小注）" if ev >= ui.FAIR
            else "🔴 <b>不值得</b>")


def report(games: list[dict], text: str) -> list[str]:
    import analyze
    from backtest import load_strategies
    from core import db
    strats = load_strategies()[0]
    legs, blocks, seen = parse(text), [], []
    for leg in legs:
        found = analyze.search(games, leg["team"])
        if not found:
            blocks.append(f"❓ 「{escape(leg['raw'])}」：找不到「{escape(leg['team'])}」未來 4 天內的比賽")
            leg["fair"] = None
            continue
        g = found[0]
        m = analyze.game_model(g) if g["odds"] else None
        if not m:
            blocks.append(f"❓ 「{escape(leg['raw'])}」：{escape(analyze.game_title(g))} 目前沒有盤口")
            leg["fair"] = None
            continue
        conn = db()
        try:
            analyze.apply_signals(g, m, strats, live.features(conn, g, {}))
        finally:
            conn.close()
        fair, what = leg_fair(g, m, leg)
        leg.update(fair=fair, g=g)
        seen.append(g["game_id"])
        icon = {"football": "⚽", "nba": "🏀", "mlb": "⚾"}.get(g["sport"], "")
        head = f"{icon} {escape(g['league'])}｜{live.kickoff_str(g)}\n<b>{escape(analyze.game_title(g))}</b>"
        if not fair:
            blocks.append(f"{head}\n❓ {what}：算不出合理賠率")
            continue
        ev = leg["odds"] / fair - 1
        verdict = _verdict(ev)
        blocks.append(f"{head}\n👉 {what}｜運彩 <b>{leg['odds']:.2f}</b>\n"
                      f"估計勝率 {100 / fair:.0f}%｜合理賠率 <b>{fair:.2f}</b>\n"
                      f"期望值 {ev * 100:+.1f}% → {verdict}"
                      + ("" if ev >= ui.GOOD else f"\n（運彩要 ≥ {fair * (1 + ui.GOOD):.2f} 才值得）"))
    ok = [leg for leg in legs if leg.get("fair")]
    if len(ok) >= 2:
        fair = math.prod(l["fair"] for l in ok)
        odds = math.prod(l["odds"] for l in ok)
        ev = odds / fair - 1
        same = len(set(seen)) < len(seen)
        blocks.append(f"🎫 <b>{len(ok)} 串 1 合計</b>\n組合賠率 {odds:.2f}｜合理 {fair:.2f}｜命中率約 {100 / fair:.0f}%\n"
                      f"期望值 {ev * 100:+.1f}% → {_verdict(ev)}"
                      + ("\n⚠️ 有兩關在同一場比賽，結果會互相影響，上面的命中率不準" if same else "")
                      + "\n串越多關抽水越疊加：只有每一關都接近公平時才值得串")
    if not legs:
        return ["看不懂這個格式。例：<code>湖人 1.75</code>、<code>湖人 -3.5 1.80</code>、<code>曼城 大2.5 1.85</code>"]
    return ["💱 <b>運彩賠率比價</b>（合理賠率 = 用 DK 盤口與模型算的公平價）"] + blocks + [live.DISCLAIMER]


HOWTO = """💱 <b>運彩賠率比價</b>
把運彩開的賠率直接打給我（一行一關，好幾行就算成串關）：
<code>湖人 1.75</code>　→ 湖人不讓分（足球是「那隊勝」）
<code>湖人 -3.5 1.80</code>　→ 湖人讓 3.5（也可寫 讓3.5；受讓寫 +3.5 或 受讓3.5）
<code>湖人 大228.5 1.75</code>　→ 那場大分（小分寫「小」）
<code>利物浦 和 3.40</code>　→ 足球和局
我會回：估計勝率、合理賠率、期望值，值不值得買。"""
