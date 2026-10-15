# -*- coding: utf-8 -*-
"""策略定義 —— 回測（歷史資料）與即時監控（ESPN / The Odds API）共用同一套觸發條件。

每個策略都只用「下注當下就看得到」的資訊（開盤→收盤的盤口變化、休息天數、各家賠率差），
避免偷看未來。下注價格一律用收盤賠率（市場平均或該博彩公司收盤價）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

# -----------------------------------------------------------------------------
# 結算
# -----------------------------------------------------------------------------


def _single_line(margin: float, handicap: float, odds: float) -> float:
    x = margin + handicap
    if x > 1e-9:
        return odds - 1
    if x < -1e-9:
        return -1.0
    return 0.0


def settle_handicap(margin: float, handicap: float, odds: float) -> float:
    """讓分 / 亞洲讓球結算（單位注 1 的損益）。margin = 所選隊得分 - 對手得分，
    handicap = 所選隊的讓分（-0.5 代表讓半球）。四分之一盤（-0.25、-0.75…）拆成兩半注。"""
    if abs((handicap * 4) % 2 - 1) < 1e-9:  # .25 / .75
        return (_single_line(margin, handicap - 0.25, odds) + _single_line(margin, handicap + 0.25, odds)) / 2
    return _single_line(margin, handicap, odds)


def settle_total(points: float, line: float, pick: str, odds: float) -> float:
    margin = points - line if pick == "over" else line - points
    return settle_handicap(margin, 0.0, odds)


def settle_1x2(home_goals: int, away_goals: int, pick: str, odds: float) -> float:
    res = "H" if home_goals > away_goals else "A" if home_goals < away_goals else "D"
    return odds - 1 if res == pick else -1.0


def poisson_cdf(k: int, lam: float) -> float:
    return sum(math.exp(-lam) * lam ** i / math.factorial(i) for i in range(k + 1))


def p_over(lam: float, line: float) -> float:
    """總進球 ~ Poisson(lam) 時「大 line 球」的機率；整數盤扣掉走盤。"""
    if abs(line - round(line)) < 1e-9:
        k = int(round(line))
        p_push = math.exp(-lam) * lam ** k / math.factorial(k)
        return (1 - poisson_cdf(k, lam)) / max(1e-9, 1 - p_push)
    return 1 - poisson_cdf(int(math.floor(line)), lam)


def lam_from_over(p: float, line: float) -> float:
    """由「大 line 球」機率反推預期總進球（二分法）。用來把 3.5 / 1.5 球的盤口換成大 2.5 機率。"""
    lo, hi = 0.1, 9.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if p_over(mid, line) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def no_vig2(a: float, b: float) -> float | None:
    if not a or not b or a <= 1 or b <= 1 or math.isnan(a) or math.isnan(b):
        return None
    return (1 / a) / (1 / a + 1 / b)


def no_vig3(h: float, d: float, a: float) -> tuple[float, float, float] | None:
    vals = (h, d, a)
    if any(v is None or (isinstance(v, float) and math.isnan(v)) or v <= 1 for v in vals):
        return None
    inv = [1 / v for v in vals]
    s = sum(inv)
    return tuple(x / s for x in inv)


# -----------------------------------------------------------------------------
# 策略描述
# -----------------------------------------------------------------------------
@dataclass
class Strategy:
    sid: str
    sport: str            # football / nba
    family: str
    mode: str             # follow / fade / ...
    param: float
    market: str           # 1x2 / ou25 / ah / spread / total / ml
    title: str            # 中文說明
    live_source: str      # espn / oddsapi / none
    stats: dict = field(default_factory=dict)
    enabled: bool = False
    weak: bool = False

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in
                ("sid", "sport", "family", "mode", "param", "market", "title", "live_source", "stats", "enabled", "weak")}


PICK_ZH = {"H": "主勝", "D": "和局", "A": "客勝", "home": "主隊", "away": "客隊", "over": "大分", "under": "小分"}


def football_strategies() -> list[Strategy]:
    out = []
    for t in (0.03, 0.05, 0.07, 0.10):
        out.append(Strategy(f"fb_steam_follow_{t}", "football", "fb_steam", "follow", t, "1x2",
                            f"1X2 某結果的去水機率在開盤→收盤上升 ≥{t*100:.0f}pp（資金湧入），跟進下注", "espn"))
        out.append(Strategy(f"fb_steam_fade_{t}", "football", "fb_steam", "fade", t, "1x2",
                            f"1X2 某結果的去水機率下降 ≥{t*100:.0f}pp（賠率被拉高），反向買這個被冷落的結果", "espn"))
    for t in (0.03, 0.05, 0.08):
        out.append(Strategy(f"fb_ou_follow_{t}", "football", "fb_ou", "follow", t, "ou25",
                            f"大小 2.5 球的去水機率變動 ≥{t*100:.0f}pp，跟進變動方向", "espn"))
        out.append(Strategy(f"fb_ou_fade_{t}", "football", "fb_ou", "fade", t, "ou25",
                            f"大小 2.5 球的去水機率變動 ≥{t*100:.0f}pp，反向下注", "espn"))
    for k in (0.25, 0.5):
        out.append(Strategy(f"fb_ah_follow_{k}", "football", "fb_ah", "follow", k, "ah",
                            f"亞洲讓球盤口開盤→收盤移動 ≥{k} 球，跟進盤口移動方向", "none"))
        out.append(Strategy(f"fb_ah_fade_{k}", "football", "fb_ah", "fade", k, "ah",
                            f"亞洲讓球盤口移動 ≥{k} 球，反向下注", "none"))
    for d in (0.01, 0.02, 0.03):
        out.append(Strategy(f"fb_sharp_div_{d}", "football", "fb_sharp", "follow", d, "1x2",
                            f"Betfair 交易所（尖銳盤）比市場平均更看好某結果 ≥{d*100:.0f}pp，以平均賠率跟尖銳盤", "oddsapi"))
    for e in (0.03, 0.05, 0.08):
        out.append(Strategy(f"fb_value_best_{e}", "football", "fb_value", "close", e, "1x2",
                            f"收盤時最高賠率比 Betfair 公平賠率高 ≥{e*100:.0f}%（各家比價找價值）", "oddsapi"))
        out.append(Strategy(f"fb_value_open_{e}", "football", "fb_value", "open", e, "1x2",
                            f"早盤最高賠率比 Betfair 早盤公平賠率高 ≥{e*100:.0f}%（提早下注找價值）", "oddsapi"))
    # ---- 讓分 × 大小 交叉（收盤盤口的組合）----
    for t in (1.0, 1.5):
        for mode, desc in (("dog_low", "預期低進球（大2.5 <50%）→ 買受讓方"), ("fav_low", "預期低進球（大2.5 <50%）→ 買讓球方"),
                           ("dog_high", "預期高進球（大2.5 ≥60%）→ 買受讓方"), ("fav_high", "預期高進球（大2.5 ≥60%）→ 買讓球方")):
            out.append(Strategy(f"fb_cross_{mode}_{t}", "football", "fb_cross", mode, t, "ah",
                                f"[交叉] 讓球 ≥{t:g} 球的比賽，{desc}", "espn"))
    for mode, desc in (("deep_over", "讓球 ≥1.5 球（實力懸殊）→ 買大 2.5"), ("deep_under", "讓球 ≥1.5 球（實力懸殊）→ 買小 2.5"),
                       ("level_over", "讓球 ≤0.25 球（勢均力敵）→ 買大 2.5"), ("level_under", "讓球 ≤0.25 球（勢均力敵）→ 買小 2.5")):
        out.append(Strategy(f"fb_crossou_{mode}", "football", "fb_crossou", mode, 0, "ou25", f"[交叉] {desc}", "espn"))
    # ---- 讓分變化 × 大小變化 交叉 ----
    for mode, desc in (("agree", "讓球盤往某隊移動、大小盤也往同一劇本移動（強隊+大球 / 弱隊+小球）→ 跟進讓球"),
                       ("disagree", "讓球盤與大小盤的移動方向互相矛盾 → 跟進讓球")):
        out.append(Strategy(f"fb_ahou_{mode}", "football", "fb_ahou", mode, 0.25, "ah", f"[交叉] {desc}", "none"))
    return out


def mlb_strategies() -> list[Strategy]:
    out = []
    for t in (0.03, 0.05, 0.08):
        out.append(Strategy(f"mlb_ml_follow_{t}", "mlb", "mlb_ml", "follow", t, "ml",
                            f"MLB 不讓分勝率（去水）開盤→收盤變動 ≥{t*100:.0f}pp，跟進", "espn"))
        out.append(Strategy(f"mlb_ml_fade_{t}", "mlb", "mlb_ml", "fade", t, "ml",
                            f"MLB 不讓分勝率變動 ≥{t*100:.0f}pp，反向", "espn"))
    for k in (0.5, 1.0):
        out.append(Strategy(f"mlb_total_follow_{k}", "mlb", "mlb_total", "follow", k, "total",
                            f"MLB 大小分盤口移動 ≥{k:g} 分，跟進", "espn"))
        out.append(Strategy(f"mlb_total_fade_{k}", "mlb", "mlb_total", "fade", k, "total",
                            f"MLB 大小分盤口移動 ≥{k:g} 分，反向", "espn"))
    # ---- 交叉：不讓分變動 × 大小分變動 ----
    for mode, desc in (("tot_flat", "不讓分勝率變動 ≥4pp、但大小分沒動 → 跟進不讓分"),
                       ("tot_moved", "不讓分勝率變動 ≥4pp、大小分也動了 → 跟進不讓分")):
        out.append(Strategy(f"mlb_mx_{mode}", "mlb", "mlb_mx", mode, 0.04, "ml", f"[交叉] MLB {desc}", "espn"))
    # ---- 交叉：讓分（±1.5）× 大小分 ----
    for mode, desc in (("fav_high", "大小 ≥9 分（打擊戰）→ 買熱門 -1.5"), ("dog_high", "大小 ≥9 分 → 買冷門 +1.5"),
                       ("fav_low", "大小 ≤7.5 分（投手戰）→ 買熱門 -1.5"), ("dog_low", "大小 ≤7.5 分（投手戰）→ 買冷門 +1.5")):
        out.append(Strategy(f"mlb_rlx_{mode}", "mlb", "mlb_rlx", mode, 0.58, "spread",
                            f"[交叉] MLB 熱門勝率 ≥58% 的比賽，{desc}", "espn"))
    return out


def nba_strategies() -> list[Strategy]:
    out = []
    for k in (1.0, 2.0, 3.0, 4.0):  # NBA 盤口都是 .5 結尾，移動一定是整數分
        out.append(Strategy(f"nba_spread_follow_{k}", "nba", "nba_spread", "follow", k, "spread",
                            f"讓分盤開盤→收盤移動 ≥{k:g} 分，跟進移動方向", "espn"))
        out.append(Strategy(f"nba_spread_fade_{k}", "nba", "nba_spread", "fade", k, "spread",
                            f"讓分盤移動 ≥{k:g} 分，反向下注（賭市場反應過度）", "espn"))
    for k in (2.0, 3.0, 4.0, 6.0):
        out.append(Strategy(f"nba_total_follow_{k}", "nba", "nba_total", "follow", k, "total",
                            f"大小分盤口移動 ≥{k:g} 分，跟進移動方向", "espn"))
        out.append(Strategy(f"nba_total_fade_{k}", "nba", "nba_total", "fade", k, "total",
                            f"大小分盤口移動 ≥{k:g} 分，反向下注", "espn"))
    for t in (0.03, 0.05, 0.08):
        out.append(Strategy(f"nba_ml_follow_{t}", "nba", "nba_ml", "follow", t, "ml",
                            f"不讓分勝率（去水）變動 ≥{t*100:.0f}pp，跟進", "espn"))
        out.append(Strategy(f"nba_ml_fade_{t}", "nba", "nba_ml", "fade", t, "ml",
                            f"不讓分勝率（去水）變動 ≥{t*100:.0f}pp，反向", "espn"))
    out.append(Strategy("nba_b2b_fade", "nba", "nba_rest", "fade", 1, "spread",
                        "背靠背（昨天剛比賽）的球隊 vs 休息 ≥1 天的對手：買對手讓分", "espn"))
    out.append(Strategy("nba_b2b_follow", "nba", "nba_rest", "follow", 1, "spread",
                        "背靠背的球隊 vs 休息 ≥1 天的對手：買背靠背球隊讓分（市場可能過度扣分）", "espn"))
    # ---- 讓分 × 大小 交叉（收盤盤口的組合；大小分跟近期聯盟平均比）----
    for k in (6.0, 9.0):
        for mode, desc in (("dog_low", "大小分比近期平均低 ≥4 分（慢節奏）→ 買受讓方"), ("fav_low", "大小分比近期平均低 ≥4 分 → 買讓分方"),
                           ("dog_high", "大小分比近期平均高 ≥4 分（快節奏）→ 買受讓方"), ("fav_high", "大小分比近期平均高 ≥4 分 → 買讓分方")):
            out.append(Strategy(f"nba_cross_{mode}_{k}", "nba", "nba_cross", mode, k, "spread",
                                f"[交叉] 讓分 ≥{k:g} 分的比賽，{desc}", "espn"))
    for mode, desc in (("big_over", "讓分 ≥10 分（實力懸殊）→ 買大分"), ("big_under", "讓分 ≥10 分（垃圾時間）→ 買小分"),
                       ("close_over", "讓分 ≤3 分（勢均力敵）→ 買大分"), ("close_under", "讓分 ≤3 分（勢均力敵）→ 買小分")):
        out.append(Strategy(f"nba_crossou_{mode}", "nba", "nba_crossou", mode, 0, "total", f"[交叉] {desc}", "espn"))
    # ---- 讓分變化 × 大小變化 交叉 ----
    for mode, desc in (("tot_up", "讓分盤移動 ≥2 分，且大小分同時上升 ≥2 分 → 跟進讓分"),
                       ("tot_down", "讓分盤移動 ≥2 分，且大小分同時下降 ≥2 分 → 跟進讓分"),
                       ("tot_flat", "讓分盤移動 ≥2 分，但大小分幾乎沒動 → 跟進讓分")):
        out.append(Strategy(f"nba_mx_sp_{mode}", "nba", "nba_mx_sp", mode, 2.0, "spread", f"[交叉] {desc}", "espn"))
    for mode, desc in (("sp_moved", "大小分移動 ≥3 分，且讓分也移動 ≥2 分 → 跟進大小"),
                       ("sp_flat", "大小分移動 ≥3 分，但讓分沒動 → 跟進大小")):
        out.append(Strategy(f"nba_mx_tot_{mode}", "nba", "nba_mx_tot", mode, 3.0, "total", f"[交叉] {desc}", "espn"))
    return out


# -----------------------------------------------------------------------------
# 觸發條件（輸入一場比賽的特徵 dict，回傳 (pick, line, 說明) 或 None）
# 回測與即時共用，特徵欄位名稱與 data_football / data_nba 一致
# -----------------------------------------------------------------------------
ODDS_RANGE = (1.25, 8.0)   # 太低沒賺頭、太高變異太大，只考慮這區間的賠率


def _ok_odds(o) -> bool:
    return o is not None and not (isinstance(o, float) and math.isnan(o)) and ODDS_RANGE[0] <= o <= ODDS_RANGE[1]


def _num(v) -> float | None:
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else v


def p_over25(g: dict) -> float | None:
    """大 2.5 球去水機率：即時資料已用 Poisson 換算好放在 p_over25，歷史資料用市場平均收盤賠率。"""
    return _num(g.get("p_over25")) or no_vig2(g.get("c_avg_over"), g.get("c_avg_under"))


def _other(side: str) -> str:
    return "away" if side == "home" else "home"


def _trigger_cross(s: Strategy, g: dict) -> tuple[str, float | None, str] | None:
    """讓分 × 大小 交叉策略。"""
    f = s.family
    if f == "fb_cross":
        line, po = _num(g.get("c_ah_line")), p_over25(g)
        if line is None or po is None or line == 0 or abs(line) < s.param - 1e-9:
            return None
        if (s.mode.endswith("low") and po >= 0.50) or (s.mode.endswith("high") and po < 0.60):
            return None
        fav = "home" if line < 0 else "away"
        pick = _other(fav) if s.mode.startswith("dog") else fav
        if not _ok_odds(_num(g.get("c_ah_h" if pick == "home" else "c_ah_a"))):
            return None
        return pick, line, f"讓球 主 {line:+g}｜大2.5 機率 {po*100:.0f}%"
    if f == "fb_crossou":
        line, po = _num(g.get("c_ah_line")), p_over25(g)
        if line is None or po is None or not _num(g.get("c_avg_over")):
            return None
        if (s.mode.startswith("deep") and abs(line) < 1.5 - 1e-9) or (s.mode.startswith("level") and abs(line) > 0.25 + 1e-9):
            return None
        return ("over" if s.mode.endswith("over") else "under"), 2.5, f"讓球 主 {line:+g}｜大2.5 機率 {po*100:.0f}%"
    if f == "fb_ahou":
        lo, lc = _num(g.get("o_ah_line")), _num(g.get("c_ah_line"))
        po, pc = no_vig2(g.get("o_avg_over"), g.get("o_avg_under")), no_vig2(g.get("c_avg_over"), g.get("c_avg_under"))
        if lo is None or lc is None or po is None or pc is None or lc == 0:
            return None
        d, dou = lc - lo, pc - po
        if abs(d) < s.param - 1e-9 or abs(dou) < 0.03:
            return None
        steamed, fav = ("home" if d < 0 else "away"), ("home" if lc < 0 else "away")
        agree = (steamed == fav) == (dou > 0)
        if agree != (s.mode == "agree"):
            return None
        return steamed, lc, f"讓球 主 {lo:+g}→{lc:+g}｜大2.5 {po*100:.0f}→{pc*100:.0f}%"
    if f == "nba_cross":
        sp, rel = _num(g.get("c_spread_home")), _num(g.get("total_rel"))
        if sp is None or rel is None or abs(sp) < s.param - 1e-9:
            return None
        if (s.mode.endswith("low") and rel > -4) or (s.mode.endswith("high") and rel < 4):
            return None
        fav = "home" if sp < 0 else "away"
        return (_other(fav) if s.mode.startswith("dog") else fav), sp, \
            f"讓分 主 {sp:+g}｜大小 {g.get('c_total'):g}（比近期平均 {rel:+.1f}）"
    if f == "nba_crossou":
        sp, tot = _num(g.get("c_spread_home")), _num(g.get("c_total"))
        if sp is None or tot is None:
            return None
        if (s.mode.startswith("big") and abs(sp) < 10) or (s.mode.startswith("close") and abs(sp) > 3):
            return None
        return ("over" if s.mode.endswith("over") else "under"), tot, f"讓分 主 {sp:+g}｜大小 {tot:g}"
    if f == "mlb_mx":
        po, pc = no_vig2(g.get("o_ml_home"), g.get("o_ml_away")), no_vig2(g.get("c_ml_home"), g.get("c_ml_away"))
        to, tc = _num(g.get("o_total")), _num(g.get("c_total"))
        if po is None or pc is None or to is None or tc is None or abs(pc - po) < s.param:
            return None
        flat = abs(tc - to) < 0.5 - 1e-9
        if flat != (s.mode == "tot_flat"):
            return None
        pick = "home" if pc > po else "away"
        if not _ok_odds(_num(g.get(f"c_ml_{pick}"))):
            return None
        return pick, None, f"主隊勝率 {po*100:.0f}%→{pc*100:.0f}%｜大小 {to:g}→{tc:g}"
    if f == "mlb_rlx":
        sp, tot = _num(g.get("c_spread_home")), _num(g.get("c_total"))
        ph = no_vig2(g.get("c_ml_home"), g.get("c_ml_away"))
        if sp is None or tot is None or ph is None:
            return None
        fav = "home" if ph >= 0.5 else "away"
        if max(ph, 1 - ph) < s.param or (fav == "home") != (sp < 0):   # 熱門要是讓 1.5 分的那邊
            return None
        if (s.mode.endswith("low") and tot > 7.5) or (s.mode.endswith("high") and tot < 9):
            return None
        pick = fav if s.mode.startswith("fav") else _other(fav)
        return pick, sp, f"熱門勝率 {max(ph, 1 - ph)*100:.0f}%｜讓分 主{sp:+g}｜大小 {tot:g}"
    if f in ("nba_mx_sp", "nba_mx_tot"):
        so, sc = _num(g.get("o_spread_home")), _num(g.get("c_spread_home"))
        to, tc = _num(g.get("o_total")), _num(g.get("c_total"))
        if None in (so, sc, to, tc):
            return None
        dsp, dt = sc - so, tc - to
        why = f"讓分 {so:+g}→{sc:+g}｜大小 {to:g}→{tc:g}"
        if f == "nba_mx_sp":
            if abs(dsp) < 2:
                return None
            ok = {"tot_up": dt >= 2, "tot_down": dt <= -2, "tot_flat": abs(dt) < 2}[s.mode]
            return (("home" if dsp < 0 else "away"), sc, why) if ok else None
        if abs(dt) < 3:
            return None
        ok = abs(dsp) >= 2 if s.mode == "sp_moved" else abs(dsp) < 2
        return (("over" if dt > 0 else "under"), tc, why) if ok else None
    return None


def trigger(s: Strategy, g: dict) -> tuple[str, float | None, str] | None:
    f = s.family
    if f in ("fb_cross", "fb_crossou", "fb_ahou", "nba_cross", "nba_crossou", "nba_mx_sp", "nba_mx_tot", "mlb_mx", "mlb_rlx"):
        return _trigger_cross(s, g)
    if f == "fb_steam":
        po = no_vig3(g.get("o_avg_h"), g.get("o_avg_d"), g.get("o_avg_a"))
        pc = no_vig3(g.get("c_avg_h"), g.get("c_avg_d"), g.get("c_avg_a"))
        if not po or not pc:
            return None
        moves = {k: pc[i] - po[i] for i, k in enumerate("HDA")}
        if s.mode == "follow":
            k = max(moves, key=moves.get)
            hit = moves[k] >= s.param
        else:
            k = min(moves, key=moves.get)
            hit = moves[k] <= -s.param
        if hit and _ok_odds(g.get(f"c_avg_{k.lower()}")):
            i = "HDA".index(k)
            return k, None, f"{PICK_ZH[k]} 機率 {po[i]*100:.0f}% → {pc[i]*100:.0f}%（{moves[k]*100:+.1f}pp）"
        return None
    if f == "fb_ou":
        po, pc = no_vig2(g.get("o_avg_over"), g.get("o_avg_under")), no_vig2(g.get("c_avg_over"), g.get("c_avg_under"))
        if po is None or pc is None:
            return None
        d = pc - po
        if abs(d) < s.param:
            return None
        steamed = "over" if d > 0 else "under"
        pick = steamed if s.mode == "follow" else ("under" if steamed == "over" else "over")
        return pick, 2.5, f"大 2.5 機率 {po*100:.0f}% → {pc*100:.0f}%（{d*100:+.1f}pp）"
    if f == "fb_ah":
        lo, lc = g.get("o_ah_line"), g.get("c_ah_line")
        if lo is None or lc is None or math.isnan(lo) or math.isnan(lc):
            return None
        d = lc - lo
        if abs(d) < s.param - 1e-9:
            return None
        steamed = "home" if d < 0 else "away"
        pick = steamed if s.mode == "follow" else ("away" if steamed == "home" else "home")
        return pick, lc, f"主隊讓球 {lo:+g} → {lc:+g}"
    if f == "fb_sharp":
        pb = no_vig3(g.get("c_bfe_h"), g.get("c_bfe_d"), g.get("c_bfe_a"))
        pa = no_vig3(g.get("c_avg_h"), g.get("c_avg_d"), g.get("c_avg_a"))
        if not pb or not pa:
            return None
        diffs = {k: pb[i] - pa[i] for i, k in enumerate("HDA")}
        k = max(diffs, key=diffs.get)
        if diffs[k] >= s.param and _ok_odds(g.get(f"c_avg_{k.lower()}")):
            i = "HDA".index(k)
            return k, None, f"{PICK_ZH[k]}：交易所 {pb[i]*100:.1f}% vs 市場平均 {pa[i]*100:.1f}%"
        return None
    if f == "fb_value":
        t = "o" if s.mode == "open" else "c"
        pb = no_vig3(g.get(f"{t}_bfe_h"), g.get(f"{t}_bfe_d"), g.get(f"{t}_bfe_a"))
        if not pb:
            return None
        evs = {}
        for i, k in enumerate("HDA"):
            o = g.get(f"{t}_max_{k.lower()}")
            if _ok_odds(o):
                evs[k] = o * pb[i] - 1
        if not evs:
            return None
        k = max(evs, key=evs.get)
        if evs[k] >= s.param:
            i = "HDA".index(k)
            return k, None, f"{PICK_ZH[k]} 最高賠率 {g[f'{t}_max_{k.lower()}']:.2f} vs 公平賠率 {1/pb[i]:.2f}"
        return None
    if f == "nba_spread":
        lo, lc = g.get("o_spread_home"), g.get("c_spread_home")
        if lo is None or lc is None or math.isnan(lo) or math.isnan(lc):
            return None
        d = lc - lo
        if abs(d) < s.param - 1e-9:
            return None
        steamed = "home" if d < 0 else "away"
        pick = steamed if s.mode == "follow" else ("away" if steamed == "home" else "home")
        return pick, lc, f"主隊讓分 {lo:+g} → {lc:+g}"
    if f in ("nba_total", "mlb_total"):
        lo, lc = g.get("o_total"), g.get("c_total")
        if lo is None or lc is None or math.isnan(lo) or math.isnan(lc):
            return None
        d = lc - lo
        if abs(d) < s.param - 1e-9:
            return None
        steamed = "over" if d > 0 else "under"
        pick = steamed if s.mode == "follow" else ("under" if steamed == "over" else "over")
        return pick, lc, f"大小分 {lo:g} → {lc:g}"
    if f in ("nba_ml", "mlb_ml"):
        po, pc = no_vig2(g.get("o_ml_home"), g.get("o_ml_away")), no_vig2(g.get("c_ml_home"), g.get("c_ml_away"))
        if po is None or pc is None:
            return None
        d = pc - po
        if abs(d) < s.param:
            return None
        steamed = "home" if d > 0 else "away"
        pick = steamed if s.mode == "follow" else ("away" if steamed == "home" else "home")
        if not _ok_odds(g.get(f"c_ml_{pick}")):
            return None
        return pick, None, f"主隊勝率 {po*100:.0f}% → {pc*100:.0f}%（{d*100:+.1f}pp）"
    if f == "nba_rest":
        hr, ar = g.get("home_rest"), g.get("away_rest")
        if hr is None or ar is None or g.get("c_spread_home") is None or math.isnan(g["c_spread_home"]):
            return None
        if hr == 1 and ar >= 2:
            tired = "home"
        elif ar == 1 and hr >= 2:
            tired = "away"
        else:
            return None
        pick = tired if s.mode == "follow" else ("away" if tired == "home" else "home")
        return pick, g["c_spread_home"], f"{PICK_ZH[tired]}背靠背（休息 {int(min(hr, ar)) - 1} 天）"
    return None


def bet_odds(s: Strategy, g: dict, pick: str) -> float | None:
    """策略下注用的收盤價格。"""
    m = s.market
    if m == "1x2":
        if s.family == "fb_value":
            return g.get(f"{'o' if s.mode == 'open' else 'c'}_max_{pick.lower()}")
        key = f"c_avg_{pick.lower()}"
        return g.get(key)
    if m == "ou25":
        return g.get(f"c_avg_{pick}")
    if m == "ah":
        return g.get("c_ah_h" if pick == "home" else "c_ah_a")
    if m == "spread":
        o = g.get("c_spread_home_odds" if pick == "home" else "c_spread_away_odds")
        return o if o and not math.isnan(o) else 1.91
    if m == "total":
        o = g.get("c_over_odds" if pick == "over" else "c_under_odds")
        return o if o and not math.isnan(o) else 1.91
    if m == "ml":
        return g.get(f"c_ml_{pick}")
    return None


def market_prob(s: Strategy, g: dict, pick: str) -> float | None:
    """下注當下市場（去水）認為這個選項會贏的機率；fb_value 用 Betfair 公平機率。"""
    m = s.market
    if m == "1x2":
        if s.family == "fb_value":
            t = "o" if s.mode == "open" else "c"
            p = no_vig3(g.get(f"{t}_bfe_h"), g.get(f"{t}_bfe_d"), g.get(f"{t}_bfe_a"))
        else:
            p = no_vig3(g.get("c_avg_h"), g.get("c_avg_d"), g.get("c_avg_a"))
        return p["HDA".index(pick)] if p else None
    pairs = {"ou25": ("c_avg_over", "c_avg_under"), "ah": ("c_ah_h", "c_ah_a"),
             "spread": ("c_spread_home_odds", "c_spread_away_odds"), "total": ("c_over_odds", "c_under_odds"),
             "ml": ("c_ml_home", "c_ml_away")}
    a, b = pairs[m]
    first = pick in ("home", "over")
    x, y = g.get(a), g.get(b)
    if m in ("spread", "total"):
        x, y = x or 1.91, y or 1.91
    p = no_vig2(x, y) if x and y else None
    return None if p is None else (p if first else 1 - p)


def settle(s: Strategy, g: dict, pick: str, line: float | None, odds: float) -> float:
    """依比賽結果計算損益。g 需含 hg/ag（足球）或 home_score/away_score（NBA）。"""
    if s.sport == "football":
        hg, ag = g["hg"], g["ag"]
        if s.market == "1x2":
            return settle_1x2(hg, ag, pick, odds)
        if s.market == "ou25":
            return settle_total(hg + ag, 2.5, pick, odds)
        if s.market == "ah":
            margin = hg - ag if pick == "home" else ag - hg
            hcp = line if pick == "home" else -line
            return settle_handicap(margin, hcp, odds)
    else:
        hs, as_ = g["home_score"], g["away_score"]
        if s.market == "spread":
            margin = hs - as_ if pick == "home" else as_ - hs
            hcp = line if pick == "home" else -line
            return settle_handicap(margin, hcp, odds)
        if s.market == "total":
            return settle_total(hs + as_, line, pick, odds)
        if s.market == "ml":
            won = (hs > as_) if pick == "home" else (as_ > hs)
            return odds - 1 if won else -1.0
    raise ValueError(s.market)
