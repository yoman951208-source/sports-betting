# -*- coding: utf-8 -*-
"""單場比賽深入分析（足球 / NBA）。

每場會看：
  1. 盤口（DraftKings 開盤 → 現在）的勝平負、讓分、大小，換算成去水機率與「合理賠率」
  2. 盤口變化 → 對照歷史回測中「同樣幅度的變化」跟進/反向的實際報酬
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
import ui
from backtest import load_strategies
from core import get_json, log, now_utc, parse_iso, to_tpe
from strategies import Strategy, lam_from_over, no_vig2, no_vig3, p_over, settle_handicap, trigger

SHRINK = 1e9          # 只用市場機率；歷史相似盤口只當參考（10/23：8–11 季回測加權沒有幫助，見 learning.py）
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


def zh_short(name: str) -> str:
    """有中文名就只用中文（表格、清單用）。"""
    return FOOTBALL_ZH.get(name.lower(), name)


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
        df = df.dropna(subset=["pH", "pA", "pO25"]).copy()
        # 校準後的市場機率（相似盤口殘差要跟這場用同一套校準比，才不會重複修正冷熱門偏差）
        import calibration
        cal = calibration.football_1x2_matrix(df[["c_avg_h", "c_avg_d", "c_avg_a"]].to_numpy(float))
        df["pHc"], df["pDc"], df["pAc"] = cal[:, 0], cal[:, 1], cal[:, 2]
        _FB = df
    return _FB


def nba_history() -> pd.DataFrame:
    global _NBA
    if _NBA is None:
        import data_nba
        df = data_nba.load_history()
        df["margin"] = df["home_score"] - df["away_score"]
        df["points"] = df["home_score"] + df["away_score"]
        inv_h, inv_a = 1 / df["c_ml_home"], 1 / df["c_ml_away"]
        df["pH"] = inv_h / (inv_h + inv_a)
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
    """市場機率與歷史相似盤口加權；shrink 越大越相信市場（10/23 起 learning.get_shrink 一律只用市場）。"""
    shrink = SHRINK if shrink is None else shrink
    if p_hist is None or n == 0 or shrink >= 1e9:
        return p_mkt
    return (p_mkt * shrink + p_hist * n) / (shrink + n)


def option_rows(options: list[dict], sport: str | None = None) -> tuple[list[str], dict | None]:
    """options: {name, short, odds, p_mkt, p_hist, n} → （選項表格, 最佳選項）。sport 決定用哪個學到的 shrink。"""
    import learning
    shrink = learning.get_shrink(sport) if sport else SHRINK
    best = None
    for o in options:
        if not o.get("odds") or o.get("p_mkt") is None:
            continue
        p = blend(o["p_mkt"], o.get("p_hist"), o.get("n", 0), shrink)
        o["p"], o["ev"], o["fair"] = p, p * o["odds"] - 1, 1 / p
        if best is None or o["ev"] > best["ev"]:
            best = o
    return [ui.options_table(options, best)], best


def apply_signals(g: dict, m: dict | None, strats: list[Strategy], f: dict) -> list[str]:
    """回測訊號：⭐（通過回測）成立時，把回測量到的「實際勝率比市場高多少」（uplift）加到那個選項，
    跟即時推播的算法一致，再重新挑最佳選項。☆（參考）只顯示、不調整。回傳要顯示的訊號文字。"""
    out = []
    for s in strats:
        if s.sport != g["sport"] or s.live_source != "espn" or not (s.enabled or s.weak):
            continue
        t = trigger(s, f)
        if not t:
            continue
        out.append(f"🔔 {'⭐' if s.enabled else '☆'}{escape(s.title)}")
        if not (s.enabled and m):
            continue
        key = ({"ou25": "total", "ah": "spread"}.get(s.market, s.market), t[0])
        o = next((x for x in m["options"] if x["key"] == key), None)
        up = s.stats.get("uplift", 0.0)
        if o is not None and up > 0 and o["p_mkt"] + up > o["p"]:
            p = min(0.97, o["p_mkt"] + up)
            o.update(p=p, ev=p * o["odds"] - 1, fair=1 / p, signal=s.title)
    if m and m["options"]:
        m["best"] = pick_best(m)
        m["rows"] = [ui.options_table(m["options"], m["best"])]
    return out


def compose(head: list[str], m: dict | None, signals: list[str], info: list[str]) -> str:
    """單場分析的版面：結論（推薦 / 觀望）→ 訊號 → 全部選項表格 → 參考資料。"""
    out = head + [""] + ui.recommend_block(m["best"] if m else None) + signals
    if m and m.get("rows"):
        out += ["", "<b>📊 全部選項</b>（合理 = 運彩賠率至少要這麼高才划算）", *m["rows"], *m.get("extra", [])]
    if info:
        out += ["", "<b>📚 參考資料</b>", *info]
    return "\n".join(out)


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


RHO = -0.08                      # Dixon-Coles 低比分修正：純 Poisson 低估和局（22.9% vs 實際 26.1%）；3 季比分 log loss 在 -0.05~-0.10 最低
MARGIN_CAL = (-0.045, 1.029)      # 淨勝 ≥2 球機率的 logit 校準：純 Poisson 高估大比分（熱門讓 1.5 贏 31% vs 實際 28%）
                                  # research/goals_calibration.py：8 季逐季留一 7 季改善（總進球同樣校準只有 4 季 → 不用）


def goal_model(pc: tuple, lam: float) -> dict:
    """比分模型：兩隊預期進球 → 各比分機率（Dixon-Coles 修正 + 淨勝 ≥2 球校準）。
    回傳 lh、la、top（最可能的 3 個比分）、margin {主隊淨勝: 機率}、total {總進球: 機率}。"""
    lh, la = team_lambdas(pc, lam)
    mat = _score_matrix(lh, la)
    mat[0][0] *= 1 - lh * la * RHO
    mat[0][1] *= 1 + lh * RHO
    mat[1][0] *= 1 + la * RHO
    mat[1][1] *= 1 - RHO
    s = sum(map(sum, mat))
    margin, total, cells = {}, {}, []
    for i, row in enumerate(mat):
        for j, p in enumerate(row):
            p /= s
            margin[i - j] = margin.get(i - j, 0.0) + p
            total[i + j] = total.get(i + j, 0.0) + p
            cells.append((p, i, j))
    a, b = MARGIN_CAL
    for sign in (1, -1):      # 各方向「贏 ≥k 球」的累積機率：k ≥ 2 才校準（贏 ≥1 球 = 勝率，維持跟市場一致）
        kmax = max(k * sign for k in margin)
        surv = {k: sum(p for m, p in margin.items() if m * sign >= k) for k in range(1, kmax + 2)}
        for k in range(2, kmax + 1):
            x = min(max(surv[k], 1e-9), 1 - 1e-9)
            surv[k] = 1 / (1 + math.exp(-(a + b * math.log(x / (1 - x)))))
        for k in range(1, kmax + 1):
            margin[k * sign] = max(0.0, surv[k] - surv[k + 1])
    # 三個比分預測：單一機率最高的幾乎都是 1-1、1-0（機率太分散），所以另外給「照預期進球」與「大比分」情境
    ranked = [(f"{i}-{j}", p, i, j) for p, i, j in sorted(cells, reverse=True)]
    eh, ea = round(lh), round(la)
    want = [("最可能", lambda i, j: True),
            ("照預期進球", lambda i, j: (i, j) == (eh, ea)),
            # 跟「最可能」同一個比分時：較強的一方多進一球（1-1 → 2-1）
            ("照預期進球", lambda i, j: (i, j) == ((eh + 1, ea) if lh >= la else (eh, ea + 1))),
            ("大比分", lambda i, j: i + j >= max(4, eh + ea + 1))]
    top, used = [], set()
    for label, ok in want:
        hit = None if label in {t[0] for t in top} else next((c for c in ranked if ok(c[2], c[3]) and c[0] not in used), None)
        if hit:
            top.append((label, hit[0], hit[1]))
            used.add(hit[0])
    return {"lh": lh, "la": la, "top": top, "margin": margin, "total": total}


def score_line(gm: dict) -> str:
    return (f"🎯 <b>比分預測</b>（預期進球 主 {gm['lh']:.1f} : 客 {gm['la']:.1f}）\n　"
            + "｜".join(f"{lab} <b>{s}</b>（{p*100:.0f}%）" for lab, s, p in gm["top"]))


def pred_scores(gm: dict) -> str:
    return " / ".join(s for _, s, _ in gm["top"])


def line_table(gm: dict, pc: tuple) -> str:
    """運彩常見的大小、讓球盤口的合理賠率（比分模型換算）。"""
    rows = []
    lam = gm["lh"] + gm["la"]
    for L in (1.5, 2.5, 3.5) + ((4.5,) if lam >= 3 else ()):
        p = sum(v for n, v in gm["total"].items() if n > L)
        rows += [[f"{p*100:.0f}%", f"{1/p:.2f}", f"大 {L:g}"], [f"{(1-p)*100:.0f}%", f"{1/(1-p):.2f}", f"小 {L:g}"]]
    fav_home = pc[0] >= pc[2]
    fav, dog = ("主", "客") if fav_home else ("客", "主")
    for k in (1.5, 2.5) if max(pc[0], pc[2]) >= 0.55 else (1.5,):
        p = sum(v for m, v in gm["margin"].items() if (m if fav_home else -m) > k)
        rows += [[f"{p*100:.0f}%", f"{1/p:.2f}", f"{fav}讓 {k:g}"], [f"{(1-p)*100:.0f}%", f"{1/(1-p):.2f}", f"{dog}受讓 {k:g}"]]
    return ui.table(["勝率", "合理", "運彩盤口"], rows, right=(0, 1))


def football_model(g: dict) -> dict | None:
    """足球的機率模型：市場去水機率 + 歷史相似盤口 → 各選項（含 key / p / ev / fair）。賽前分析與賽後檢討共用。"""
    import calibration
    o = g["odds"]
    pc_raw = no_vig3(o.get("c_ml_home"), o.get("c_ml_draw"), o.get("c_ml_away"))
    if not pc_raw:
        return None
    # 市場機率：power 去水 + 歷史校準（修正冷熱門偏差，見 calibration.py）
    pc = calibration.football_1x2(o.get("c_ml_home"), o.get("c_ml_draw"), o.get("c_ml_away")) or pc_raw
    po = (calibration.football_1x2(o.get("o_ml_home"), o.get("o_ml_draw"), o.get("o_ml_away"))
          or no_vig3(o.get("o_ml_home"), o.get("o_ml_draw"), o.get("o_ml_away")))
    # 大小球 → λ（預期總進球）；大小用 power 去水（樣本外略好，校準反而變差所以不做）
    L, pOL = o.get("c_total"), calibration.two_way(o.get("c_over_odds"), o.get("c_under_odds"))
    lam = lam_from_over(pOL, L) if L and pOL else None
    pO25 = p_over(lam, 2.5) if lam else None
    Lo, pOLo = o.get("o_total"), calibration.two_way(o.get("o_over_odds"), o.get("o_under_odds"))
    lam_o = lam_from_over(pOLo, Lo) if Lo and pOLo else None
    sp = o.get("c_spread_home")
    # 讓球：power 去水 + 扣回莊家對主隊多抽的部分（10/23 莊家角度評估）
    p_sp = calibration.football_spread(o.get("c_spread_home_odds"), o.get("c_spread_away_odds"))

    # 歷史相似盤口（勝平負 × 大小 交叉）—— 用未校準的機率找相似比賽（與歷史欄位同一套定義）
    a = football_analogs(pc_raw[0], pc_raw[2], pO25)
    n = len(a)
    hist = {}
    if n:
        # 勝平負：相似比賽「打敗自己校準後市場機率多少」加回這場的校準機率（10/19 修正）
        clip = lambda x: float(min(0.97, max(0.02, x)))
        hist = {"H": clip(pc[0] + ((a["ftr"] == "H") - a["pHc"]).mean()),
                "D": clip(pc[1] + ((a["ftr"] == "D") - a["pDc"]).mean()),
                "A": clip(pc[2] + ((a["ftr"] == "A") - a["pAc"]).mean()),
                "over25": (a["goals"] >= 3).mean(), "goals": a["goals"].mean()}
        if L:
            dec = a[a["goals"] != L]
            hist["overL"] = (dec["goals"] > L).mean() if len(dec) else None
        if sp is not None:
            cover = a["margin"].map(lambda m: settle_handicap(m, sp, 2.0))
            dec = cover[cover != 0]
            hist["cover"] = (dec > 0).mean() if len(dec) else None

    home, away = g["home"], g["away"]
    options = []
    if lam:
        options += [{"key": ("total", "over"), "name": f"大 {L:g} 球", "odds": o.get("c_over_odds"), "p_mkt": pOL,
                     "p_hist": hist.get("overL"), "n": n},
                    {"key": ("total", "under"), "name": f"小 {L:g} 球", "odds": o.get("c_under_odds"), "p_mkt": 1 - pOL,
                     "p_hist": None if hist.get("overL") is None else 1 - hist["overL"], "n": n}]
    if sp is not None and p_sp:
        options += [{"key": ("spread", "home"), "name": f"主 {sp:+g}（{zh(home)}）", "short": f"主 {sp:+g}",
                     "odds": o.get("c_spread_home_odds"), "p_mkt": p_sp, "p_hist": hist.get("cover"), "n": n},
                    {"key": ("spread", "away"), "name": f"客 {-sp:+g}（{zh(away)}）", "short": f"客 {-sp:+g}",
                     "odds": o.get("c_spread_away_odds"), "p_mkt": 1 - p_sp,
                     "p_hist": None if hist.get("cover") is None else 1 - hist["cover"], "n": n}]
    options += [
        {"key": ("1x2", "H"), "name": f"主勝 {zh(home)}", "short": "主勝", "odds": o.get("c_ml_home"), "p_mkt": pc[0],
         "p_hist": hist.get("H"), "n": n},
        {"key": ("1x2", "D"), "name": "和局", "short": "和局", "odds": o.get("c_ml_draw"), "p_mkt": pc[1], "p_hist": hist.get("D"), "n": n},
        {"key": ("1x2", "A"), "name": f"客勝 {zh(away)}", "short": "客勝", "odds": o.get("c_ml_away"), "p_mkt": pc[2],
         "p_hist": hist.get("A"), "n": n},
    ]
    # 原始（等比例去水、未校準）機率一起記錄，讓 learning.py 用實戰結果檢驗校準有沒有幫助
    raw = {("1x2", "H"): pc_raw[0], ("1x2", "D"): pc_raw[1], ("1x2", "A"): pc_raw[2]}
    q = no_vig2(o.get("c_over_odds"), o.get("c_under_odds"))
    if q:
        raw[("total", "over")], raw[("total", "under")] = q, 1 - q
    q = no_vig2(o.get("c_spread_home_odds"), o.get("c_spread_away_odds"))
    if q:
        raw[("spread", "home")], raw[("spread", "away")] = q, 1 - q
    for x in options:
        x["p_raw"] = raw.get(x["key"], x["p_mkt"])
    option_rows(options, sport="football")
    m = {"pc": pc, "po": po, "L": L, "pOL": pOL, "lam": lam, "lam_o": lam_o, "pO25": pO25, "sp": sp, "p_sp": p_sp,
         "n": n, "hist": hist, "options": [x for x in options if "p" in x], "focus": ("total", "spread"),
         "goals": goal_model(pc, lam) if lam else None}
    m["best"] = pick_best(m)
    m["rows"] = [ui.options_table(m["options"], m["best"])]
    if m["goals"]:
        m["extra"] = ["", "<b>📐 運彩常見盤口的合理賠率</b>（比分模型換算，運彩 ≥ 合理才划算）", line_table(m["goals"], pc)]
    return m


FOOTBALL_1X2_MIN_ODDS = 1.5      # 足球以大小、讓球為主；不讓分賠率至少這麼高才推薦（使用者 10/25）


def in_focus(m: dict, o: dict) -> bool:
    """這個選項能不能進推薦、最推薦、串關。"""
    focus = m.get("focus")
    return not focus or o["key"][0] in focus or (o["key"][0] == "1x2" and o["odds"] >= FOOTBALL_1X2_MIN_ODDS)


def pick_best(m: dict) -> dict | None:
    """期望值最高、而且 in_focus 的選項。"""
    pool = [o for o in m["options"] if in_focus(m, o)] or m["options"]
    return max(pool, key=lambda o: o["ev"]) if pool else None


def analyze_football(g: dict, strats: list[Strategy]) -> tuple[str, dict | None]:
    o = g["odds"]
    home, away = g["home"], g["away"]
    head = [f"⚽ {escape(g['league'])}｜{live.kickoff_str(g)}",
            f"<b>{escape(zh(home))} vs {escape(zh(away))}</b>"]
    lines = []          # 參考資料
    summ = _summary(g)
    fh, fa = _form(summ, home), _form(summ, away)
    if fh or fa:
        lines.append(f"近5場：主 {fh or '-'}｜客 {fa or '-'}")
    sh, sa = _standing(summ, home), _standing(summ, away)
    if sh or sa:
        lines.append(f"積分：主 {sh or '-'}｜客 {sa or '-'}")

    m = football_model(g)
    if not m:
        return "\n".join(head + lines + ["（目前沒有盤口資料）"]), None
    pc, po, L, pOL, lam, lam_o, pO25 = m["pc"], m["po"], m["L"], m["pOL"], m["lam"], m["lam_o"], m["pO25"]
    sp, p_sp, n, hist = m["sp"], m["p_sp"], m["n"], m["hist"]
    if m["goals"]:
        head.append(score_line(m["goals"]))
        g["_pred"] = f"預測 {pred_scores(m['goals'])}｜大2.5 {sum(v for k, v in m['goals']['total'].items() if k > 2.5)*100:.0f}%"

    lines.append("<b>盤口（DK 開盤→現在，去水機率）</b>")
    lines.append("  勝/和/負 " + " / ".join(f"{a*100:.0f}→{b*100:.0f}%" for a, b in zip(po or pc, pc))
                 + f"｜賠率 {o['c_ml_home']:.2f}/{o['c_ml_draw']:.2f}/{o['c_ml_away']:.2f}")
    if lam:
        lines.append(f"  大小 {L:g} 球：大 {pOL*100:.0f}%（預期總進球 {lam_o or lam:.2f}→{lam:.2f}，換算大2.5 {pO25*100:.0f}%）")
    sp = o.get("c_spread_home")
    p_sp = no_vig2(o.get("c_spread_home_odds"), o.get("c_spread_away_odds"))
    if sp is not None and p_sp:
        lines.append(f"  讓球 主 {sp:+g}（主 {o['c_spread_home_odds']:.2f} / 客 {o['c_spread_away_odds']:.2f}）")

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
        lines.append(f"<b>歷史相似盤口</b>（{n} 場，11 個聯賽 2019–26）")
        lines.append(f"  主勝 {hist['H']*100:.0f}%｜和 {hist['D']*100:.0f}%｜客勝 {hist['A']*100:.0f}%｜"
                     f"大2.5 {hist['over25']*100:.0f}%｜平均 {hist['goals']:.2f} 球"
                     + (f"｜主 {sp:+g} 贏盤 {hist['cover']*100:.0f}%" if hist.get("cover") is not None else ""))

    # 策略訊號（⭐ 啟用 / ☆ 參考）
    conn = live.db()
    try:
        f = live.features(conn, g, {})
    finally:
        conn.close()
    return compose(head, m, apply_signals(g, m, strats, f), lines), m["best"]


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
    head = [f"🏀 NBA｜{live.kickoff_str(g)}",
            f"<b>{escape(live.team_name(g, 'away'))} @ {escape(live.team_name(g, 'home'))}</b>"]
    lines = []          # 參考資料
    rest = []
    for side in ("home", "away"):
        if f.get(f"{side}_rest") == 1:
            rest.append(f"{live.team_name(g, side)} 背靠背")
    if rest:
        lines.append("😴 " + "、".join(rest))
    for r in inj:
        lines.append(f"🚑 {escape(r['name'])}（{r['team']}，{keys[r['player_id']][1]:.1f} 分）{escape(r['status'])}")
    if o.get("c_spread_home") is None:
        return "\n".join(head + lines + ["（目前沒有盤口資料）"]), None
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
    lines += nba_form_lines(g, m)
    if m["n"]:
        hist = m["hist"]
        lines.append(f"<b>歷史相似盤口</b>（讓分±1、大小±6，{m['n']} 場）")
        pct = lambda k: f"{hist[k]*100:.0f}%" if hist.get(k) is not None else "-"
        lines.append(f"  主隊贏盤 {pct('cover')}｜大分 {pct('over')}｜主勝（校正後）{pct('win')}")
    return compose(head, m, apply_signals(g, m, strats, f), lines), m["best"]


def nba_form_lines(g: dict, m: dict) -> list[str]:
    """球隊近 10 場、主力 3 人近況與疲勞、球員狀態對勝率的調整。"""
    form = (m or {}).get("form") or {}
    out = [f"<b>📋 本季近況與主力</b>（{form.get('season', '')}）"]
    any_stars = False
    for side in ("away", "home"):
        st = form.get(side) or {}
        name = escape(live.team_name(g, side))
        if st.get("games10"):
            out.append(f"  {name}：近{st['games10']}場 {st['wins10']}勝{st['games10'] - st['wins10']}負｜"
                       f"平均淨勝 {st['margin10']:+.1f}｜打出盤口 {st.get('ats_w', 0)}-{st.get('ats_l', 0)}")
        for s in st.get("stars", []):
            any_stars = True
            flags = []
            if s["load"] >= 3:
                flags.append(f"近3場多打 {s['load']:+.1f} 分鐘⚠️")
            elif s["load"] <= -3:
                flags.append(f"近3場少打 {s['load']:+.1f} 分鐘")
            if s["pid"] in form.get("out_pids", set()):
                flags.append("🚑 Out")
            trend = "↑" if s["pts_last5"] > s["pts_avg"] * 1.15 else "↓" if s["pts_last5"] < s["pts_avg"] * 0.85 else ""
            out.append(f"    • {escape(s['name'])}：季均 {s['pts_avg']:.1f}分/{s['min_avg']:.0f}分鐘｜近5場 {s['pts_last5']:.1f}分{trend}"
                       + (f"｜{'｜'.join(flags)}" if flags else ""))
    if not any_stars:
        out.append("  （本季出賽未滿 5 場，主力近況與疲勞調整要等第 5 場之後才會啟用）")
        return out
    adj = []
    if form.get("d_star_load") is not None and m.get("p_sp") is not None and form.get("p_sp_before") is not None:
        adj.append(f"主隊贏盤 {form['p_sp_before']*100:.1f}% → {m['p_sp']*100:.1f}%")
    if (form.get("d_star_load") is not None and m.get("p_ml") is not None and form.get("p_ml_before") is not None
            and abs(m["p_ml"] - form["p_ml_before"]) >= 0.0005):
        adj.append(f"主勝 {form['p_ml_before']*100:.1f}% → {m['p_ml']*100:.1f}%")
    if adj:
        out.append(f"  🧮 球員狀態調整（疲勞差 {form['d_star_load']:+.1f} 分鐘、缺陣差 {form.get('d_star_out', 0):+.0f}）：" + "；".join(adj))
    return out


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
        # 每場相似比賽都對「自己的」讓分與大小分盤口結算，避免盤口差 1~6 分造成偏差（10/19 修正）
        cover = a["margin"] + a["c_spread_home"]
        hist["cover"] = float((cover[cover != 0] > 0).mean())
        over = a["points"] - a["c_total"]
        hist["over"] = float((over[over != 0] > 0).mean()) if tot else None
        hist["win_resid"] = float(((a["margin"] > 0) - a["pH"]).dropna().mean())   # 相似比賽主隊勝率比市場高多少
    import calibration
    # 讓分：power 去水 + 主力缺陣 / 疲勞調整（見 calibration.py）；不讓分只用 power 去水；大小維持原始市場
    p_sp = calibration.nba_spread(o.get("c_spread_home_odds"), o.get("c_spread_away_odds"), sp, o.get("o_spread_home"))
    p_t = no_vig2(o.get("c_over_odds") or 1.91, o.get("c_under_odds") or 1.91)
    pm_c = calibration.nba_ml(f.get("c_ml_home"), f.get("c_ml_away"), f.get("o_ml_home"), f.get("o_ml_away"))
    # 球員狀態（10/19）：主力疲勞差、主力缺陣差（傷兵名單 Out）——本季出賽 ≥5 場才有
    form = {}
    try:
        import nba_form
        conn = live.db()
        try:
            out_pids = {r["player_id"] for r in conn.execute("SELECT player_id FROM injuries WHERE status='Out'")}
        finally:
            conn.close()
        form = nba_form.live_features(g["home_abbr"], g["away_abbr"], out_pids)
        form["out_pids"] = out_pids
        form["p_sp_before"], form["p_ml_before"] = p_sp, pm_c
        p_sp = calibration.nba_form_adjust(p_sp, "spread", form.get("d_star_out"), form.get("d_star_load"))
        pm_c = calibration.nba_form_adjust(pm_c, "ml", None, form.get("d_star_load"))
    except Exception:
        log.exception("NBA 球員狀態計算失敗 %s", g.get("game_id"))
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
    if pm_c and hist.get("win_resid") is not None:
        hist["win"] = min(0.97, max(0.03, pm_c + hist["win_resid"]))
    if pm_c:
        options += [{"key": ("ml", "home"), "name": f"{home} 不讓分", "odds": f.get("c_ml_home"), "p_mkt": pm_c,
                     "p_hist": hist.get("win"), "n": n},
                    {"key": ("ml", "away"), "name": f"{away} 不讓分", "odds": f.get("c_ml_away"), "p_mkt": 1 - pm_c,
                     "p_hist": inv(hist.get("win")), "n": n}]
    raw = {}
    q = no_vig2(o.get("c_spread_home_odds") or 1.91, o.get("c_spread_away_odds") or 1.91)
    raw[("spread", "home")], raw[("spread", "away")] = q, 1 - q
    q = no_vig2(f.get("c_ml_home"), f.get("c_ml_away"))
    if q:
        raw[("ml", "home")], raw[("ml", "away")] = q, 1 - q
    for x in options:
        x["p_raw"] = raw.get(x["key"], x["p_mkt"])
    rows, best = option_rows(options, sport="nba")
    return {"sp": sp, "L": tot, "n": n, "hist": hist, "form": form, "p_sp": p_sp, "p_ml": pm_c, "options": [x for x in options if "p" in x],
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
        hist["win"] = float(min(0.97, max(0.03, ph + ((a["margin"] > 0) - a["pH"]).mean())))  # 市場機率 + 相似比賽的偏差
        # 相似比賽的大小分不一定等於這場（樣本不夠時會放寬到 ±1 分），
        # 所以把每場的總分平移成「相對自己盤口」再比，否則 6.5 分的盤會被 7.5 分的比賽灌水（10/19 修正）
        adj = a["runs"] - a["c_total"] + L
        hist["over"] = _p_over_emp(adj, L)
        if sp is not None:
            cover = a["margin"] + sp
            dec = cover[cover != 0]
            hist["cover"] = float((dec > 0).mean()) if len(dec) else None
        # 運彩常開不同的大小分：用「市場機率 + 歷史上相鄰盤口的差距」推估
        hist["alt"] = {}
        base = _p_over_emp(adj, L)
        for alt in (L - 1, L + 1):
            p_alt = _p_over_emp(adj, alt)
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
    head = [f"⚾ MLB {escape(g.get('note') or '')}｜{live.kickoff_str(g)}",
            f"<b>{escape(away)} @ {escape(home)}</b>"
            + (f"（{escape(rec.get('away', ''))} / {escape(rec.get('home', ''))}）" if rec else "")]
    lines = []          # 參考資料
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
        return "\n".join(head + lines + ["（目前沒有盤口資料）"]), None
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
        lines.append(f"<b>歷史相似盤口</b>（勝率±2.5pp、大小相同，{m['n']} 場，2019–26）")
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
    return compose(head, m, apply_signals(g, m, strats, f), lines), m["best"]


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
    details, recs, others = [], [], []
    for g in sel:
        text, best = analyze_game(g, strats)
        details.append(text)
        g["_best_ev"] = best["ev"] if best else None      # bot 的比賽按鈕會顯示評等
        when = f"{to_tpe(parse_iso(g['kickoff'])):%m/%d %H:%M}"
        note = f" [{escape(g['note'])}]" if g.get("note") and g["sport"] == "mlb" else ""
        pred = f"\n　🎯 {g['_pred']}" if g.get("_pred") else ""
        if best and best["ev"] >= MIN_EDGE:
            recs.append(f"{when} {escape(game_title(g))}{note}\n"
                        f"　👉 <b>{escape(best['name'])}</b> @{best['odds']:.2f}｜📌 運彩 ≥ {best['fair']:.2f}{pred}")
        else:
            icon = ui.grade_icon(best["ev"]) if best else "⚪"
            others.append(f"{icon} {when} {escape(game_title(g))}{note}{pred}")
    head = [f"🏆 <b>{escape(league)} 分析總表</b>（未來 {hours:g} 小時，{len(sel)} 場）", ""]
    head += ([f"🟢 <b>推薦（{len(recs)} 場）</b>"] + recs) if recs else ["🟢 <b>推薦</b>：這段時間沒有明顯有價值的選項"]
    if others:
        head += ["", f"<b>其他 {len(others)} 場</b>（🟡 接近公平：運彩賠率夠高才考慮｜🔴 偏貴：觀望）"] + others
    head += ["", "📌 = 運彩賠率至少要這麼高才划算（DK 的價格通常比運彩好）", live.DISCLAIMER]
    return ["\n".join(head)] + details
