# -*- coding: utf-8 -*-
"""單場比賽深入分析（足球 / NBA）。

每場會看：
  1. 盤口（DraftKings 開盤 → 現在）的勝平負、讓分、大小，換算成去水機率與「合理賠率」
  2. 盤口變化 → 對照兩年回測中「同樣幅度的變化」跟進/反向的實際報酬
  3. 歷史相似盤口：找勝平負機率 + 大小球機率都接近的歷史比賽（讓分 × 大小交叉），看實際賽果
  4. 近況、積分（足球）、休息天數與主力傷兵（NBA）
  5. 綜合估計勝率 = 市場機率與歷史相似盤口的加權（大幅偏向市場，避免小樣本雜訊）
     → 期望值與「運彩賠率至少要多少才划算」
"""
from __future__ import annotations

import math
from html import escape

import pandas as pd

import config
import live
from backtest import load_strategies
from core import get_json, log, now_utc, parse_iso, to_tpe
from strategies import Strategy, lam_from_over, no_vig2, no_vig3, p_over, settle_handicap, trigger

SHRINK = 600          # 歷史樣本要多大才跟市場一樣重要（越大越相信市場）
MIN_EDGE = 0.02       # 綜合期望值 ≥ 2% 才列為「可考慮」

FOOTBALL_ZH = {
    "manchester city": "曼城", "manchester united": "曼聯", "liverpool": "利物浦", "arsenal": "兵工廠",
    "chelsea": "切爾西", "tottenham hotspur": "熱刺", "newcastle united": "紐卡索", "aston villa": "阿斯頓維拉",
    "real madrid": "皇馬", "barcelona": "巴薩", "atlético madrid": "馬競", "atletico madrid": "馬競",
    "bayern munich": "拜仁", "borussia dortmund": "多特蒙德", "bayer leverkusen": "勒沃庫森", "rb leipzig": "萊比錫",
    "vfb stuttgart": "斯圖加特", "eintracht frankfurt": "法蘭克福", "paris saint-germain": "巴黎聖日耳曼",
    "internazionale": "國際米蘭", "ac milan": "AC米蘭", "juventus": "尤文圖斯", "napoli": "拿坡里", "as roma": "羅馬",
    "atalanta": "亞特蘭大", "como": "科莫", "fc porto": "波圖", "benfica": "本菲卡", "sporting cp": "里斯本競技",
    "psv eindhoven": "PSV", "feyenoord rotterdam": "費耶諾德", "ajax amsterdam": "阿賈克斯", "galatasaray": "加拉塔薩雷",
    "fenerbahce": "費內巴切", "club brugge": "布魯日", "celtic": "塞爾提克", "villarreal": "比利亞雷亞爾",
    "real betis": "貝提斯", "lille": "里爾", "lens": "朗斯", "marseille": "馬賽", "monaco": "摩納哥",
    "shakhtar donetsk": "頓內次克礦工", "slavia prague": "布拉格斯拉維亞", "aek athens": "AEK雅典",
    "bodo/glimt": "博德閃耀", "lask linz": "林茨", "viking fk": "維京", "slovan bratislava": "布拉提斯拉瓦",
    "sabah fk": "薩巴",
}
ALIASES = {**{v: k for k, v in FOOTBALL_ZH.items()},
           "巴黎": "paris saint-germain", "國米": "internazionale", "米蘭": "ac milan", "尤文": "juventus",
           "巴塞": "barcelona", "巴薩隆納": "barcelona", "皇家馬德里": "real madrid", "阿森納": "arsenal",
           "槍手": "arsenal", "多特": "borussia dortmund", "藥廠": "bayer leverkusen", "熱刺": "tottenham hotspur",
           "psg": "paris saint-germain", "man city": "manchester city", "man utd": "manchester united",
           "inter": "internazionale", "atleti": "atlético madrid", "barca": "barcelona", "spurs": "tottenham hotspur"}


def zh(name: str) -> str:
    z = FOOTBALL_ZH.get(name.lower())
    return f"{z}({name})" if z else name


# =============================================================================
# 歷史資料（載入一次後留在記憶體）
# =============================================================================
_FB: pd.DataFrame | None = None
_NBA: pd.DataFrame | None = None


def football_history() -> pd.DataFrame:
    global _FB
    if _FB is None:
        import data_football
        df = data_football.load_history()
        inv = 1 / df[["c_avg_h", "c_avg_d", "c_avg_a"]]
        s = inv.sum(axis=1)
        df["pH"], df["pD"], df["pA"] = inv["c_avg_h"] / s, inv["c_avg_d"] / s, inv["c_avg_a"] / s
        io = 1 / df[["c_avg_over", "c_avg_under"]]
        df["pO25"] = io["c_avg_over"] / io.sum(axis=1)
        df["goals"] = df["hg"] + df["ag"]
        df["margin"] = df["hg"] - df["ag"]
        _FB = df.dropna(subset=["pH", "pA", "pO25"])
    return _FB


def nba_history() -> pd.DataFrame:
    global _NBA
    if _NBA is None:
        import data_nba
        df = data_nba.load_history()
        df["margin"] = df["home_score"] - df["away_score"]
        df["points"] = df["home_score"] + df["away_score"]
        _NBA = df.dropna(subset=["c_spread_home", "c_total"])
    return _NBA


def football_analogs(pH: float, pA: float, pO: float | None) -> pd.DataFrame:
    """勝平負機率 + 大2.5機率都接近的歷史比賽；樣本太少時逐步放寬。"""
    h = football_history()
    for tol, tol_o in ((0.03, 0.04), (0.045, 0.06), (0.06, 0.08), (0.08, 0.12)):
        m = ((h["pH"] - pH).abs() <= tol) & ((h["pA"] - pA).abs() <= tol)
        if pO is not None:
            m &= (h["pO25"] - pO).abs() <= tol_o
        a = h[m]
        if len(a) >= 120:
            return a
    return a


# =============================================================================
# 共用：選項評估
# =============================================================================
def blend(p_mkt: float, p_hist: float | None, n: int, shrink: float | None = None) -> float:
    """市場機率與歷史相似盤口加權；shrink 越大越相信市場（由 learning.py 依實戰結果每天調整）。"""
    shrink = SHRINK if shrink is None else shrink
    if p_hist is None or n == 0 or shrink == float("inf"):
        return p_mkt
    return (p_mkt * shrink + p_hist * n) / (shrink + n)


def option_rows(options: list[dict], sport: str | None = None) -> tuple[list[str], dict | None]:
    """options: {name, odds, p_mkt, p_hist, n} → 文字列與最佳選項。sport 決定用哪個學到的 shrink。"""
    import learning
    shrink = learning.get_shrink(sport) if sport else SHRINK
    lines, best = [], None
    for o in options:
        if not o.get("odds") or o.get("p_mkt") is None:
            continue
        p = blend(o["p_mkt"], o.get("p_hist"), o.get("n", 0), shrink)
        o["p"], o["ev"], o["fair"] = p, p * o["odds"] - 1, 1 / p
        if best is None or o["ev"] > best["ev"]:
            best = o
    for o in options:
        if "fair" in o:
            hist = f"｜歷史 {o['p_hist']*100:.0f}%" if o.get("p_hist") is not None else ""
            lines.append(f"  {escape(o['name'])}：合理 <b>{o['fair']:.2f}</b>（市場 {o['p_mkt']*100:.0f}%{hist}）DK {o['odds']:.2f} → EV {o['ev']*100:+.1f}%")
    return lines, best


def _move_verdict(strats: list[Strategy], family: str, move: float, unit: str = "pp") -> str:
    """找回測中「跟進」門檻 ≤ 本次變動的最大門檻，回報跟進與反向的成績。"""
    cands = [s for s in strats if s.family == family and s.mode == "follow" and s.stats.get("n")
             and abs(move) >= s.param - 1e-9]
    if not cands:
        return "變動不大"
    s = max(cands, key=lambda s: s.param)
    st = s.stats
    verdict = "⭐ 可跟" if s.enabled else ("偏正面但不顯著" if st["roi"] > 0 else "不建議追")
    th = f"{s.param*100:.0f}pp" if unit == "pp" else f"{s.param:g} 分"
    fade = next((x for x in strats if x.family == family and x.mode == "fade" and x.param == s.param), None)
    fade_s = f"；反向 ROI {fade.stats['roi']*100:+.1f}%" if fade and fade.stats.get("n") else ""
    return f"回測跟進 ≥{th}：{st['n']} 注 ROI {st['roi']*100:+.1f}%（{verdict}）{fade_s}"


# =============================================================================
# 足球
# =============================================================================
def _summary(g: dict) -> dict:
    _, path, eid = g["game_id"].split(":", 2)
    return get_json(f"https://site.api.espn.com/apis/site/v2/sports/{path}/summary", params={"event": eid},
                    cache_name=f"live/summary_{eid}.json", max_age_hours=1) or {}


def _form(summary: dict, team_name: str) -> str:
    for t in summary.get("lastFiveGames") or []:
        if t.get("team", {}).get("displayName") != team_name:
            continue
        tid, res, gf, ga = t["team"]["id"], [], 0, 0
        for ev in t.get("events", []):
            res.append(ev.get("gameResult", "?").replace("W", "勝").replace("D", "和").replace("L", "負"))
            try:
                hs, as_ = int(ev["homeTeamScore"]), int(ev["awayTeamScore"])
                home = ev.get("homeTeamId") == tid
                gf += hs if home else as_
                ga += as_ if home else hs
            except (KeyError, TypeError, ValueError):
                pass
        return f"{''.join(res)}（進{gf}失{ga}）" if res else ""
    return ""


def _standing(summary: dict, team_name: str) -> str:
    for grp in (summary.get("standings") or {}).get("groups", []):
        for e in grp.get("standings", {}).get("entries", []):
            if e.get("team") == team_name:
                st = {s["name"]: s.get("displayValue") for s in e.get("stats", [])}
                return f"第{st.get('rank', '?')}（{st.get('points', '?')}分，{st.get('overall', '')}）"
    return ""


# =============================================================================
# Poisson 比分模型：把市場的勝平負 + 預期總進球拆成兩隊各自的預期進球，
# 就能算出任何讓球 / 大小盤口的合理賠率（運彩開的盤口常跟 DK 不同）
# =============================================================================
def _score_matrix(lh: float, la: float, kmax: int = 12) -> list[list[float]]:
    ph = [math.exp(-lh) * lh ** i / math.factorial(i) for i in range(kmax + 1)]
    pa = [math.exp(-la) * la ** j / math.factorial(j) for j in range(kmax + 1)]
    return [[x * y for y in pa] for x in ph]


def _hda(mat) -> tuple[float, float, float]:
    h = sum(mat[i][j] for i in range(len(mat)) for j in range(len(mat)) if i > j)
    d = sum(mat[i][i] for i in range(len(mat)))
    return h, d, 1 - h - d


def team_lambdas(pc: tuple, lam: float) -> tuple[float, float]:
    """找主客預期進球的分配，使「主勝 /（主勝+客勝）」符合市場。"""
    target = pc[0] / (pc[0] + pc[2])
    lo, hi = 0.02, 0.98
    for _ in range(40):
        s = (lo + hi) / 2
        h, _d, a = _hda(_score_matrix(s * lam, (1 - s) * lam))
        if h / (h + a) < target:
            lo = s
        else:
            hi = s
    return s * lam, (1 - s) * lam


def handicap_cover(lh: float, la: float, line: float) -> float:
    """主隊讓 line（如 -1.5）贏盤機率；整數盤扣掉走盤。"""
    mat = _score_matrix(lh, la)
    win = lose = 0.0
    for i, row in enumerate(mat):
        for j, p in enumerate(row):
            x = i - j + line
            if x > 1e-9:
                win += p
            elif x < -1e-9:
                lose += p
    return win / (win + lose)


def alt_lines(pc: tuple, lam: float) -> list[str]:
    """運彩常見盤口的合理賠率（Poisson 近似）。"""
    out = []
    totals = sorted({max(0.5, round(lam) - 0.5), round(lam) + 0.5})
    out.append("大小 " + "｜".join(f"{t:g}球 大{1/p_over(lam, t):.2f}/小{1/(1-p_over(lam, t)):.2f}" for t in totals))
    lh, la = team_lambdas(pc, lam)
    fav_home = pc[0] >= pc[2]
    hcp = []
    for k in ((1.5, 2.5) if max(pc[0], pc[2]) >= 0.65 else (1.5,)):
        line = -k if fav_home else k            # 主隊的讓分
        p = handicap_cover(lh, la, line)
        fav_p = p if fav_home else 1 - p
        hcp.append(f"{'主' if fav_home else '客'}讓{k:g} {1/fav_p:.2f}/受讓 {1/(1-fav_p):.2f}")
    out.append("讓球 " + "｜".join(hcp))
    return out


def football_model(g: dict) -> dict | None:
    """足球的機率模型：市場去水機率 + 歷史相似盤口 → 各選項（含 key / p / ev / fair）。賽前分析與賽後檢討共用。"""
    o = g["odds"]
    pc = no_vig3(o.get("c_ml_home"), o.get("c_ml_draw"), o.get("c_ml_away"))
    if not pc:
        return None
    po = no_vig3(o.get("o_ml_home"), o.get("o_ml_draw"), o.get("o_ml_away"))
    # 大小球 → λ（預期總進球）
    L, pOL = o.get("c_total"), no_vig2(o.get("c_over_odds"), o.get("c_under_odds"))
    lam = lam_from_over(pOL, L) if L and pOL else None
    pO25 = p_over(lam, 2.5) if lam else None
    Lo, pOLo = o.get("o_total"), no_vig2(o.get("o_over_odds"), o.get("o_under_odds"))
    lam_o = lam_from_over(pOLo, Lo) if Lo and pOLo else None
    sp = o.get("c_spread_home")
    p_sp = no_vig2(o.get("c_spread_home_odds"), o.get("c_spread_away_odds"))

    # 歷史相似盤口（勝平負 × 大小 交叉）
    a = football_analogs(pc[0], pc[2], pO25)
    n = len(a)
    hist = {}
    if n:
        hist = {"H": (a["ftr"] == "H").mean(), "D": (a["ftr"] == "D").mean(), "A": (a["ftr"] == "A").mean(),
                "over25": (a["goals"] >= 3).mean(), "goals": a["goals"].mean()}
        if L:
            dec = a[a["goals"] != L]
            hist["overL"] = (dec["goals"] > L).mean() if len(dec) else None
        if sp is not None:
            cover = a["margin"].map(lambda m: settle_handicap(m, sp, 2.0))
            dec = cover[cover != 0]
            hist["cover"] = (dec > 0).mean() if len(dec) else None

    home, away = g["home"], g["away"]
    options = [
        {"key": ("1x2", "H"), "name": f"主勝 {zh(home)}", "odds": o.get("c_ml_home"), "p_mkt": pc[0], "p_hist": hist.get("H"), "n": n},
        {"key": ("1x2", "D"), "name": "和局", "odds": o.get("c_ml_draw"), "p_mkt": pc[1], "p_hist": hist.get("D"), "n": n},
        {"key": ("1x2", "A"), "name": f"客勝 {zh(away)}", "odds": o.get("c_ml_away"), "p_mkt": pc[2], "p_hist": hist.get("A"), "n": n},
    ]
    if lam:
        options += [{"key": ("total", "over"), "name": f"大 {L:g} 球", "odds": o.get("c_over_odds"), "p_mkt": pOL,
                     "p_hist": hist.get("overL"), "n": n},
                    {"key": ("total", "under"), "name": f"小 {L:g} 球", "odds": o.get("c_under_odds"), "p_mkt": 1 - pOL,
                     "p_hist": None if hist.get("overL") is None else 1 - hist["overL"], "n": n}]
    if sp is not None and p_sp:
        options += [{"key": ("spread", "home"), "name": f"主 {sp:+g}", "odds": o.get("c_spread_home_odds"), "p_mkt": p_sp,
                     "p_hist": hist.get("cover"), "n": n},
                    {"key": ("spread", "away"), "name": f"客 {-sp:+g}", "odds": o.get("c_spread_away_odds"), "p_mkt": 1 - p_sp,
                     "p_hist": None if hist.get("cover") is None else 1 - hist["cover"], "n": n}]
    rows, best = option_rows(options, sport="football")
    return {"pc": pc, "po": po, "L": L, "pOL": pOL, "lam": lam, "lam_o": lam_o, "pO25": pO25, "sp": sp, "p_sp": p_sp,
            "n": n, "hist": hist, "options": [x for x in options if "p" in x], "rows": rows, "best": best}


def analyze_football(g: dict, strats: list[Strategy]) -> tuple[str, dict | None]:
    o = g["odds"]
    home, away = g["home"], g["away"]
    lines = [f"⚽ {escape(g['league'])}｜{live.kickoff_str(g)}",
             f"<b>{escape(zh(home))} vs {escape(zh(away))}</b>"]
    summ = _summary(g)
    fh, fa = _form(summ, home), _form(summ, away)
    if fh or fa:
        lines.append(f"近5場：主 {fh or '-'}｜客 {fa or '-'}")
    sh, sa = _standing(summ, home), _standing(summ, away)
    if sh or sa:
        lines.append(f"積分：主 {sh or '-'}｜客 {sa or '-'}")

    m = football_model(g)
    if not m:
        lines.append("（目前沒有盤口資料）")
        return "\n".join(lines), None
    pc, po, L, pOL, lam, lam_o, pO25 = m["pc"], m["po"], m["L"], m["pOL"], m["lam"], m["lam_o"], m["pO25"]
    sp, p_sp, n, hist = m["sp"], m["p_sp"], m["n"], m["hist"]

    lines.append("<b>盤口（DK 開盤→現在，去水機率）</b>")
    lines.append("  勝/和/負 " + " / ".join(f"{a*100:.0f}→{b*100:.0f}%" for a, b in zip(po or pc, pc))
                 + f"｜賠率 {o['c_ml_home']:.2f}/{o['c_ml_draw']:.2f}/{o['c_ml_away']:.2f}")
    if lam:
        lines.append(f"  大小 {L:g} 球：大 {pOL*100:.0f}%（預期總進球 {lam_o or lam:.2f}→{lam:.2f}，換算大2.5 {pO25*100:.0f}%）")
    sp = o.get("c_spread_home")
    p_sp = no_vig2(o.get("c_spread_home_odds"), o.get("c_spread_away_odds"))
    if sp is not None and p_sp:
        lines.append(f"  讓球 主 {sp:+g}（主 {o['c_spread_home_odds']:.2f} / 客 {o['c_spread_away_odds']:.2f}）")
    if lam:
        lines.append("<b>運彩其他盤口的合理賠率</b>（Poisson 近似，運彩 ≥ 這個才划算）")
        lines += [f"  {escape(x)}" for x in alt_lines(pc, lam)]

    # 盤口變化 vs 回測
    if po:
        moves = [b - a for a, b in zip(po, pc)]
        i = max(range(3), key=lambda k: abs(moves[k]))
        if abs(moves[i]) >= 0.02:
            who = ["主勝", "和局", "客勝"][i]
            lines.append(f"📈 {who} {moves[i]*100:+.1f}pp → {_move_verdict(strats, 'fb_steam', moves[i])}")
    if lam and lam_o and abs(p_over(lam, 2.5) - p_over(lam_o, 2.5)) >= 0.02:
        d = p_over(lam, 2.5) - p_over(lam_o, 2.5)
        lines.append(f"📈 大2.5 {d*100:+.1f}pp → {_move_verdict(strats, 'fb_ou', d)}")

    if n:
        lines.append(f"<b>歷史相似盤口</b>（{n} 場，11 個聯賽 2024–26）")
        lines.append(f"  主勝 {hist['H']*100:.0f}%｜和 {hist['D']*100:.0f}%｜客勝 {hist['A']*100:.0f}%｜"
                     f"大2.5 {hist['over25']*100:.0f}%｜平均 {hist['goals']:.2f} 球"
                     + (f"｜主 {sp:+g} 贏盤 {hist['cover']*100:.0f}%" if hist.get("cover") is not None else ""))

    # 策略訊號（⭐ 啟用 / ☆ 參考）
    conn = live.db()
    try:
        f = live.features(conn, g, {})
    finally:
        conn.close()
    sig = [("⭐" if s.enabled else "☆") + s.title for s in strats
           if s.sport == "football" and s.live_source == "espn" and (s.enabled or s.weak) and trigger(s, f)]
    if sig:
        lines += [f"🔔 {escape(x)}" for x in sig]

    lines.append("<b>評估</b>（運彩賠率 ≥ 合理賠率才有正期望值）")
    lines += m["rows"]
    lines.append(_verdict(m["best"]))
    return "\n".join(lines), m["best"]


def _verdict(best: dict | None) -> str:
    if best and best["ev"] >= MIN_EDGE:
        return f"👉 <b>相對最有價值：{escape(best['name'])}</b>（DK {best['odds']:.2f}，EV {best['ev']*100:+.1f}%；運彩 ≥ {best['fair']:.2f} 才划算）"
    if best:
        return f"👉 沒有明顯價值（最好的是 {escape(best['name'])}，EV {best['ev']*100:+.1f}%）；除非運彩賠率高於合理賠率，否則觀望"
    return "👉 資料不足"


# =============================================================================
# NBA
# =============================================================================
def analyze_nba(g: dict, strats: list[Strategy]) -> tuple[str, dict | None]:
    o = g["odds"]
    conn = live.db()
    try:
        f = live.features(conn, g, {})
        keys = live.key_players()
        inj = [r for r in conn.execute("SELECT * FROM injuries") if r["team"] in (g["home_abbr"], g["away_abbr"])
               and r["player_id"] in keys]
    finally:
        conn.close()
    lines = [f"🏀 NBA｜{live.kickoff_str(g)}",
             f"<b>{escape(live.team_name(g, 'away'))} @ {escape(live.team_name(g, 'home'))}</b>"]
    rest = []
    for side in ("home", "away"):
        if f.get(f"{side}_rest") == 1:
            rest.append(f"{live.team_name(g, side)} 背靠背")
    if rest:
        lines.append("😴 " + "、".join(rest))
    for r in inj:
        lines.append(f"🚑 {escape(r['name'])}（{r['team']}，{keys[r['player_id']][1]:.1f} 分）{escape(r['status'])}")
    if o.get("c_spread_home") is None:
        lines.append("（目前沒有盤口資料）")
        return "\n".join(lines), None
    lines.append(f"<b>盤口（DK 開盤→現在）</b>\n  {escape(live.line_summary(g))}")
    sp, tot = o["c_spread_home"], o.get("c_total")
    if o.get("o_spread_home") is not None and abs(sp - o["o_spread_home"]) >= 1:
        lines.append(f"📈 讓分移動 {sp - o['o_spread_home']:+g} 分 → {_move_verdict(strats, 'nba_spread', sp - o['o_spread_home'], '分')}")
    if tot and o.get("o_total") and abs(tot - o["o_total"]) >= 2:
        lines.append(f"📈 大小分移動 {tot - o['o_total']:+g} 分 → {_move_verdict(strats, 'nba_total', tot - o['o_total'], '分')}")
    pm_o, pm_c = no_vig2(f.get("o_ml_home"), f.get("o_ml_away")), no_vig2(f.get("c_ml_home"), f.get("c_ml_away"))
    if pm_o and pm_c and abs(pm_c - pm_o) >= 0.03:
        lines.append(f"📈 主隊勝率 {(pm_c - pm_o)*100:+.1f}pp → {_move_verdict(strats, 'nba_ml', pm_c - pm_o)}")

    m = nba_model(g, f)
    if m["n"]:
        hist = m["hist"]
        lines.append(f"<b>歷史相似盤口</b>（讓分±1、大小±6，{m['n']} 場）")
        lines.append(f"  主隊贏盤 {hist['cover']*100:.0f}%｜大分 {hist['over']*100:.0f}%｜主勝 {hist['win']*100:.0f}%")
    sig = [("⭐" if s.enabled else "☆") + s.title for s in strats
           if s.sport == "nba" and s.live_source == "espn" and (s.enabled or s.weak) and trigger(s, f)]
    lines += [f"🔔 {escape(x)}" for x in sig]
    lines.append("<b>評估</b>（運彩賠率 ≥ 合理賠率才有正期望值）")
    lines += m["rows"]
    lines.append(_verdict(m["best"]))
    return "\n".join(lines), m["best"]


def nba_model(g: dict, f: dict | None = None) -> dict | None:
    """NBA 機率模型：市場去水機率 + 歷史相似盤口（讓分 ±1、大小 ±6）。f = live.features（不讓分清理過）。"""
    o = g["odds"]
    sp, tot = o.get("c_spread_home"), o.get("c_total")
    if sp is None:
        return None
    if f is None:
        conn = live.db()
        try:
            f = live.features(conn, g, {})
        finally:
            conn.close()
    h = nba_history()
    a = h[((h["c_spread_home"] - sp).abs() <= 1) & ((h["c_total"] - (tot or h["c_total"].median())).abs() <= 6)]
    n = len(a)
    hist = {}
    if n:
        cover = a["margin"].map(lambda m: settle_handicap(m, sp, 2.0))
        hist["cover"] = float((cover[cover != 0] > 0).mean())
        over = a["points"] - (tot or 0)
        hist["over"] = float((over[over != 0] > 0).mean()) if tot else None
        hist["win"] = float((a["margin"] > 0).mean())
    p_sp = no_vig2(o.get("c_spread_home_odds") or 1.91, o.get("c_spread_away_odds") or 1.91)
    p_t = no_vig2(o.get("c_over_odds") or 1.91, o.get("c_under_odds") or 1.91)
    pm_c = no_vig2(f.get("c_ml_home"), f.get("c_ml_away"))
    inv = lambda x: None if x is None else 1 - x
    home, away = live.team_name(g, "home"), live.team_name(g, "away")
    options = [
        {"key": ("spread", "home"), "name": f"{home} {sp:+g}", "odds": o.get("c_spread_home_odds"), "p_mkt": p_sp,
         "p_hist": hist.get("cover"), "n": n},
        {"key": ("spread", "away"), "name": f"{away} {-sp:+g}", "odds": o.get("c_spread_away_odds"), "p_mkt": 1 - p_sp,
         "p_hist": inv(hist.get("cover")), "n": n},
    ]
    if tot:
        options += [{"key": ("total", "over"), "name": f"大 {tot:g}", "odds": o.get("c_over_odds"), "p_mkt": p_t,
                     "p_hist": hist.get("over"), "n": n},
                    {"key": ("total", "under"), "name": f"小 {tot:g}", "odds": o.get("c_under_odds"), "p_mkt": 1 - p_t,
                     "p_hist": inv(hist.get("over")), "n": n}]
    if pm_c:
        options += [{"key": ("ml", "home"), "name": f"{home} 不讓分", "odds": f.get("c_ml_home"), "p_mkt": pm_c,
                     "p_hist": hist.get("win"), "n": n},
                    {"key": ("ml", "away"), "name": f"{away} 不讓分", "odds": f.get("c_ml_away"), "p_mkt": 1 - pm_c,
                     "p_hist": inv(hist.get("win")), "n": n}]
    rows, best = option_rows(options, sport="nba")
    return {"sp": sp, "L": tot, "n": n, "hist": hist, "options": [x for x in options if "p" in x],
            "rows": rows, "best": best}


# =============================================================================
# MLB
# =============================================================================
_MLB: pd.DataFrame | None = None


def mlb_history() -> pd.DataFrame:
    global _MLB
    if _MLB is None:
        import data_mlb
        df = data_mlb.load_history().dropna(subset=["c_ml_home", "c_ml_away", "c_total"])
        inv_h, inv_a = 1 / df["c_ml_home"], 1 / df["c_ml_away"]
        df["pH"] = inv_h / (inv_h + inv_a)
        df["margin"] = df["home_score"] - df["away_score"]
        df["runs"] = df["home_score"] + df["away_score"]
        _MLB = df
    return _MLB


def mlb_analogs(ph: float, total: float) -> pd.DataFrame:
    h = mlb_history()
    for tol, tol_t in ((0.025, 0.0), (0.04, 0.5), (0.06, 0.5), (0.08, 1.0)):
        a = h[((h["pH"] - ph).abs() <= tol) & ((h["c_total"] - total).abs() <= tol_t + 1e-9)]
        if len(a) >= 150:
            return a
    return a


def _p_over_emp(runs: pd.Series, line: float) -> float | None:
    dec = runs[runs != line]
    return float((dec > line).mean()) if len(dec) else None


def mlb_model(g: dict) -> dict | None:
    """MLB 機率模型：市場去水機率 + 歷史相似盤口（勝率 ±2.5pp、同一個大小分）。"""
    o = g["odds"]
    ph = no_vig2(o.get("c_ml_home"), o.get("c_ml_away"))
    L = o.get("c_total")
    if ph is None or not L:
        return None
    pOL = no_vig2(o.get("c_over_odds"), o.get("c_under_odds"))
    sp = o.get("c_spread_home")
    p_sp = no_vig2(o.get("c_spread_home_odds"), o.get("c_spread_away_odds"))
    a = mlb_analogs(ph, L)
    n = len(a)
    hist = {}
    if n:
        hist["win"] = float((a["margin"] > 0).mean())
        hist["over"] = _p_over_emp(a["runs"], L)
        if sp is not None:
            cover = a["margin"] + sp
            dec = cover[cover != 0]
            hist["cover"] = float((dec > 0).mean()) if len(dec) else None
        # 運彩常開不同的大小分：用「市場機率 + 歷史上相鄰盤口的差距」推估
        hist["alt"] = {}
        base = _p_over_emp(a["runs"], L)
        for alt in (L - 1, L + 1):
            p_alt = _p_over_emp(a["runs"], alt)
            if p_alt is not None and base is not None and pOL:
                hist["alt"][alt] = min(0.97, max(0.03, pOL + (p_alt - base)))
    home, away = live.team_name(g, "home"), live.team_name(g, "away")
    options = [
        {"key": ("ml", "home"), "name": f"{home} 不讓分", "odds": o.get("c_ml_home"), "p_mkt": ph, "p_hist": hist.get("win"), "n": n},
        {"key": ("ml", "away"), "name": f"{away} 不讓分", "odds": o.get("c_ml_away"), "p_mkt": 1 - ph,
         "p_hist": None if hist.get("win") is None else 1 - hist["win"], "n": n},
    ]
    if pOL:
        options += [{"key": ("total", "over"), "name": f"大 {L:g} 分", "odds": o.get("c_over_odds"), "p_mkt": pOL,
                     "p_hist": hist.get("over"), "n": n},
                    {"key": ("total", "under"), "name": f"小 {L:g} 分", "odds": o.get("c_under_odds"), "p_mkt": 1 - pOL,
                     "p_hist": None if hist.get("over") is None else 1 - hist["over"], "n": n}]
    if sp is not None and p_sp:
        options += [{"key": ("spread", "home"), "name": f"{home} {sp:+g}", "odds": o.get("c_spread_home_odds"), "p_mkt": p_sp,
                     "p_hist": hist.get("cover"), "n": n},
                    {"key": ("spread", "away"), "name": f"{away} {-sp:+g}", "odds": o.get("c_spread_away_odds"), "p_mkt": 1 - p_sp,
                     "p_hist": None if hist.get("cover") is None else 1 - hist["cover"], "n": n}]
    rows, best = option_rows(options, sport="mlb")
    return {"ph": ph, "L": L, "pOL": pOL, "sp": sp, "n": n, "hist": hist,
            "options": [x for x in options if "p" in x], "rows": rows, "best": best}


def _pitcher_line(pid: str, name: str) -> str:
    data = get_json(f"https://site.web.api.espn.com/apis/common/v3/sports/baseball/mlb/athletes/{pid}/overview",
                    cache_name=f"live/mlb_pitcher_{pid}.json", max_age_hours=12) or {}
    st = data.get("statistics") or {}
    names = st.get("names") or []

    def parse(split):
        try:
            v = dict(zip(names, split.get("stats", [])))
            ip_raw = str(v["innings"])
            whole, _, frac = ip_raw.partition(".")
            ip = int(whole) + (int(frac) / 3 if frac else 0)
            if ip <= 0:
                return None
            er, h, bb, k = (float(v[x]) for x in ("earnedRuns", "hits", "walks", "strikeouts"))
            return f"{ip_raw} 局 ERA {er * 9 / ip:.2f}｜WHIP {(h + bb) / ip:.2f}｜K/9 {k * 9 / ip:.1f}"
        except (KeyError, ValueError, TypeError, ZeroDivisionError):
            return None

    parts = {s.get("displayName"): parse(s) for s in st.get("splits", [])}
    reg, post = parts.get("Regular Season"), parts.get("Postseason")
    return f"{escape(name)}：例行賽 {reg or '-'}" + (f"\n      季後賽 {post}" if post else "")


def _mlb_injuries(summary: dict, max_per_team: int = 3) -> list[str]:
    out = []
    for team in summary.get("injuries") or []:
        abbr = (team.get("team") or {}).get("abbreviation", "")
        items = []
        for inj in team.get("injuries", []):
            if "60-Day" in inj.get("status", ""):   # 長期傷兵早已反映在盤口，略過
                continue
            ath = inj.get("athlete") or {}
            items.append(f"{ath.get('displayName', '?')}({inj.get('status', '')})")
            if len(items) >= max_per_team:
                break
        if items:
            out.append(f"{abbr}：" + "、".join(items))
    return out


def analyze_mlb(g: dict, strats: list[Strategy]) -> tuple[str, dict | None]:
    o = g["odds"]
    home, away = live.team_name(g, "home"), live.team_name(g, "away")
    rec = g.get("records", {})
    lines = [f"⚾ MLB {escape(g.get('note') or '')}｜{live.kickoff_str(g)}",
             f"<b>{escape(away)} @ {escape(home)}</b>"
             + (f"（{escape(rec.get('away', ''))} / {escape(rec.get('home', ''))}）" if rec else "")]
    if g.get("series"):
        lines.append(f"系列賽：{escape(g['series'])}")
    probs = g.get("probables") or {}
    if probs:
        lines.append("<b>先發投手</b>")
        for side in ("away", "home"):
            if side in probs:
                lines.append(f"  {'客' if side == 'away' else '主'} " + _pitcher_line(*probs[side]))
    summ = _summary(g)
    pred = summ.get("predictor") or {}
    try:
        espn_home = float(pred["homeTeam"]["gameProjection"]) / 100
    except (KeyError, TypeError, ValueError):
        espn_home = None
    inj = _mlb_injuries(summ)
    if inj:
        lines.append("🚑 " + "｜".join(escape(x) for x in inj))

    m = mlb_model(g)
    if not m:
        lines.append("（目前沒有盤口資料）")
        return "\n".join(lines), None
    lines.append(f"<b>盤口（DK 開盤→現在）</b>\n  {escape(live.line_summary(g))}")
    if espn_home is not None:
        lines.append(f"  ESPN 模型：主隊勝率 {espn_home*100:.0f}%（市場去水 {m['ph']*100:.0f}%）")
    po = no_vig2(o.get("o_ml_home"), o.get("o_ml_away"))
    if po and abs(m["ph"] - po) >= 0.02:
        lines.append(f"📈 主隊勝率 {(m['ph'] - po)*100:+.1f}pp → {_move_verdict(strats, 'mlb_ml', m['ph'] - po)}")
    if o.get("o_total") and abs(m["L"] - o["o_total"]) >= 0.5:
        lines.append(f"📈 大小分 {o['o_total']:g}→{m['L']:g} → {_move_verdict(strats, 'mlb_total', m['L'] - o['o_total'], '分')}")
    hist = m["hist"]
    if m["n"]:
        lines.append(f"<b>歷史相似盤口</b>（勝率±2.5pp、大小相同，{m['n']} 場，2024–26）")
        lines.append(f"  主隊勝 {hist['win']*100:.0f}%"
                     + (f"｜大 {m['L']:g} {hist['over']*100:.0f}%" if hist.get("over") is not None else "")
                     + (f"｜主 {m['sp']:+g} 贏盤 {hist['cover']*100:.0f}%" if hist.get("cover") is not None else ""))
    if g.get("playoffs"):
        b = load_strategies()[1].get("baselines", {})
        if b.get("MLB季後賽場數"):
            lines.append(f"  季後賽基準（{b['MLB季後賽場數']} 場）：盲買熱門 {b['MLB季後賽盲買熱門']*100:+.1f}%、"
                         f"盲買主隊 {b['MLB季後賽盲買主隊']*100:+.1f}%、大分 {b['MLB季後賽大分比例']*100:.0f}%（樣本小，僅參考）")
    if hist.get("alt"):
        lines.append("<b>運彩其他大小分盤口的合理賠率</b>")
        lines.append("  " + "｜".join(f"{k:g} 分 大{1/p:.2f}/小{1/(1-p):.2f}" for k, p in sorted(hist["alt"].items())))
    conn = live.db()
    try:
        f = live.features(conn, g, {})
    finally:
        conn.close()
    sig = [("⭐" if s.enabled else "☆") + s.title for s in strats
           if s.sport == "mlb" and (s.enabled or s.weak) and trigger(s, f)]
    lines += [f"🔔 {escape(x)}" for x in sig]
    lines.append("<b>評估</b>（運彩賠率 ≥ 合理賠率才有正期望值）")
    lines += m["rows"]
    lines.append(_verdict(m["best"]))
    return "\n".join(lines), m["best"]


def game_model(g: dict) -> dict | None:
    return {"mlb": mlb_model, "football": football_model, "nba": nba_model}[g["sport"]](g) if g["sport"] in ("mlb", "football", "nba") else None


def game_title(g: dict) -> str:
    if g["sport"] in ("nba", "mlb"):
        return f"{live.team_name(g, 'away')} @ {live.team_name(g, 'home')}"
    return f"{zh(g['home'])} vs {zh(g['away'])}"


# =============================================================================
# 入口
# =============================================================================
def analyze_game(g: dict, strats: list[Strategy] | None = None) -> tuple[str, dict | None]:
    strats = strats if strats is not None else load_strategies()[0]
    try:
        if g["sport"] == "nba":
            return analyze_nba(g, strats)
        if g["sport"] == "mlb":
            return analyze_mlb(g, strats)
        return analyze_football(g, strats)
    except Exception as exc:  # 單場失敗不影響其他場
        log.exception("分析失敗 %s", g.get("game_id"))
        return f"{escape(g['home'])} vs {escape(g['away'])}：分析失敗（{escape(str(exc))}）", None


def upcoming(games: list[dict], hours: float = 36, league: str | None = None, sport: str | None = None) -> list[dict]:
    now = now_utc()
    out = [g for g in games if g["state"] == "pre" and 0 < (parse_iso(g["kickoff"]) - now).total_seconds() / 3600 <= hours
           and (league is None or g["league"] == league) and (sport is None or g["sport"] == sport)]
    return sorted(out, key=lambda g: g["kickoff"])


def search(games: list[dict], query: str, hours: float = 96) -> list[dict]:
    """用隊名（中英文、縮寫）或聯賽名稱找比賽。"""
    q = query.strip().lower()
    if not q:
        return []
    leagues = {g["league"] for g in games}
    if query.strip() in leagues:
        return upcoming(games, hours, league=query.strip())
    target = ALIASES.get(query.strip(), ALIASES.get(q, q))
    # 中文隊名 → 縮寫（「勇士」同時是 NBA 勇士與 MLB 勇士，兩邊都找）
    zh_abbr = {("nba", k) for k, v in config.NBA_TEAM_ZH.items() if v == query.strip()} | \
              {("mlb", k) for k, v in config.MLB_TEAM_ZH.items() if v == query.strip()}
    out = []
    for g in upcoming(games, hours):
        names = [g["home"].lower(), g["away"].lower(), g["home_abbr"].lower(), g["away_abbr"].lower()]
        if any(sp == g["sport"] and ab.lower() in names[2:] for sp, ab in zh_abbr):
            out.append(g)
        elif zh_abbr:
            continue
        elif any(target in n for n in names) or any(n and n in target for n in names[:2]):
            out.append(g)
    return out


def league_report(games: list[dict], league: str, hours: float = 36) -> list[str]:
    """整個聯賽／盃賽的分析：一則總表 + 每場一則詳細。"""
    strats = load_strategies()[0]
    sel = upcoming(games, hours, league=league)
    if not sel:
        return [f"未來 {hours:g} 小時沒有{league}比賽。"]
    details, summary = [], []
    for g in sel:
        text, best = analyze_game(g, strats)
        details.append(text)
        tag = (f"✅ {escape(best['name'])}（EV {best['ev']*100:+.1f}%，運彩≥{best['fair']:.2f}）"
               if best and best["ev"] >= MIN_EDGE else
               f"➖ 觀望（最佳 {escape(best['name'])} EV {best['ev']*100:+.1f}%）" if best else "➖ 無盤口")
        note = f" [{escape(g['note'])}]" if g.get("note") and g["sport"] == "mlb" else ""
        summary.append(f"{to_tpe(parse_iso(g['kickoff'])):%m/%d %H:%M} {escape(game_title(g))}{note}\n   {tag}")
    head = [f"🏆 <b>{escape(league)} 分析總表</b>（未來 {hours:g} 小時，{len(sel)} 場）", ""] + summary + [
        "", "EV = 用 DK 現價計算的期望值；運彩賠率通常較低，請以「運彩 ≥ 合理賠率」為準。",
        "估計勝率 = 市場去水機率 × 歷史相似盤口（偏向市場加權）。", live.DISCLAIMER]
    return ["\n".join(head)] + details
