# -*- coding: utf-8 -*-
"""NBA 球隊與球員近況特徵（2026-10-19 加入）。

每場比賽、每一隊，只用「這場之前」同一季的比賽計算（不偷看未來）：
  球隊  margin10  近 10 場平均淨勝分
        ats10     近 10 場「打出盤口」平均：淨勝分 + 自己的收盤讓分（>0 代表比市場預期好）
        ou10      近 10 場總分 − 收盤大小分（>0 代表比賽常開出大分）
  主力  主力 = 本季場均上場時間最多的 3 人（至少已出賽 5 場）
        star_form 主力近 5 場得分相對季平均的變化（比例）
        star_out  這場沒上場的主力人數（歷史從 box score 得知；即時用傷兵名單）
        star_load 主力近 3 場上場時間 − 季平均（分鐘，疲勞指標）
"""
from __future__ import annotations

import numpy as np
import pandas as pd

KEY_N = 3
MIN_PRIOR = 5

_LIVE: dict = {}


def current(season: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """本季已完賽比賽與 box score（快取 30 分鐘）。季初資料不足時由呼叫端決定是否沿用上季。"""
    import time
    import config
    import data_nba
    import data_nba_box
    season = season or config.NBA_SEASONS[-1][0]
    hit = _LIVE.get(season)
    if hit and time.time() - hit[2] < 1800:
        return hit[0], hit[1], season
    games = data_nba.load_history()
    games = games[games["season"] == season]
    box = data_nba_box.load_boxes(games["event_id"].astype(str).tolist()) if len(games) else pd.DataFrame()
    _LIVE[season] = (games, box, time.time())
    return games, box, season


def live_features(home: str, away: str, out_pids: set[str]) -> dict:
    """即時：兩隊主力疲勞差（主 − 客）與主力缺陣差（依傷兵名單 Out）。主力需本季已出賽 ≥5 場。"""
    games, box, season = current()
    if box.empty:
        return {"season": season, "home": {"team": home, "stars": []}, "away": {"team": away, "stars": []}}
    h, a = team_state(games, box, home, season), team_state(games, box, away, season)
    res = {"season": season, "home": h, "away": a}
    if h.get("star_load") is not None and a.get("star_load") is not None:
        res["d_star_load"] = h["star_load"] - a["star_load"]
    if len(h.get("stars", [])) == KEY_N and len(a.get("stars", [])) == KEY_N:
        res["d_star_out"] = float(sum(s["pid"] in out_pids for s in h["stars"]) - sum(s["pid"] in out_pids for s in a["stars"]))
    return res


def team_games(games: pd.DataFrame) -> pd.DataFrame:
    """每場拆成兩列（主、客各一），附上以該隊角度的淨勝分、讓分、打出盤口。"""
    rows = []
    for side, opp, sign in (("home", "away", 1), ("away", "home", -1)):
        d = games[["event_id", "season", "game_date", side, opp, "home_score", "away_score", "c_spread_home", "c_total"]].copy()
        d = d.rename(columns={side: "team", opp: "opp"})
        d["side"] = side
        d["margin"] = sign * (d["home_score"] - d["away_score"])
        d["spread"] = sign * d["c_spread_home"]
        d["ats"] = d["margin"] + d["spread"]
        d["ou"] = d["home_score"] + d["away_score"] - d["c_total"]
        rows.append(d)
    t = pd.concat(rows, ignore_index=True).sort_values(["game_date", "event_id"])
    g = t.groupby(["season", "team"])
    for col in ("margin", "ats", "ou"):
        t[f"{col}10"] = g[col].transform(lambda s: s.shift().rolling(10, min_periods=MIN_PRIOR).mean())
    t["n_prior"] = g.cumcount()
    return t


def player_features(games: pd.DataFrame, box: pd.DataFrame) -> pd.DataFrame:
    """每個（比賽, 球隊）的主力特徵。"""
    meta = games[["event_id", "season", "game_date"]].astype({"event_id": str})
    p = box.astype({"event_id": str}).merge(meta, on="event_id", how="inner")
    p = p.sort_values(["pid", "game_date"])
    played = (~p["dnp"]) & (p["min"].fillna(0) > 0)
    for c in ("min", "pts", "pm"):
        p[c] = p[c].where(played)
    g = p.groupby(["season", "pid"])
    p["gp_prior"] = g["min"].transform(lambda s: s.notna().shift(fill_value=False).cumsum())
    p["min_avg"] = g["min"].transform(lambda s: s.shift().expanding().mean())
    p["pts_avg"] = g["pts"].transform(lambda s: s.shift().expanding().mean())
    p["pts_last5"] = g["pts"].transform(lambda s: s.shift().rolling(5, min_periods=2).mean())
    p["min_last3"] = g["min"].transform(lambda s: s.shift().rolling(3, min_periods=2).mean())
    p["played"] = played

    def agg(df: pd.DataFrame) -> pd.Series:
        k = df[df["gp_prior"] >= MIN_PRIOR].nlargest(KEY_N, "min_avg")
        if len(k) < KEY_N:
            return pd.Series({"star_form": np.nan, "star_out": np.nan, "star_load": np.nan, "stars": ""})
        form = ((k["pts_last5"] - k["pts_avg"]) / k["pts_avg"].clip(lower=5)).mean()
        return pd.Series({"star_form": form, "star_out": float((~k["played"]).sum()),
                          "star_load": (k["min_last3"] - k["min_avg"]).mean(), "stars": ",".join(k["name"])})

    return p.groupby(["event_id", "team"]).apply(agg).reset_index()


def team_state(games: pd.DataFrame, box: pd.DataFrame, team: str, season: str) -> dict:
    """某隊「到目前為止」的狀態（給即時分析用）：近 10 場、主力 3 人的季平均 / 近況 / 疲勞。"""
    g = games[(games["season"] == season) & ((games["home"] == team) | (games["away"] == team))]
    g = g.sort_values("game_date")
    out: dict = {"team": team, "n": len(g)}
    if len(g):
        margin = [(r.home_score - r.away_score) * (1 if r.home == team else -1) for r in g.itertuples()]
        spread = [(r.c_spread_home if r.home == team else -r.c_spread_home) for r in g.itertuples()]
        last = list(zip(margin, spread))[-10:]
        out.update({"wins10": sum(m > 0 for m, _ in last), "games10": len(last),
                    "margin10": sum(m for m, _ in last) / len(last),
                    "ats_w": sum(m + s > 0 for m, s in last if s == s), "ats_l": sum(m + s < 0 for m, s in last if s == s)})
    p = box[box["event_id"].isin(set(g["event_id"].astype(str))) & (box["team"] == team)].copy()
    if p.empty:
        out["stars"] = []
        return out
    p = p.merge(g[["event_id", "game_date"]].astype({"event_id": str}), on="event_id").sort_values("game_date")
    p = p[(~p["dnp"]) & (p["min"].fillna(0) > 0)]
    stars = []
    for pid, d in p.groupby("pid"):
        if len(d) < MIN_PRIOR:
            continue
        stars.append({"pid": pid, "name": d["name"].iloc[-1], "gp": len(d), "min_avg": d["min"].mean(),
                      "pts_avg": d["pts"].mean(), "pts_last5": d["pts"].tail(5).mean(),
                      "min_last3": d["min"].tail(3).mean(), "pm_last5": d["pm"].tail(5).mean()})
    stars = sorted(stars, key=lambda x: -x["min_avg"])[:KEY_N]
    for s in stars:
        s["load"] = s["min_last3"] - s["min_avg"]
    out["stars"] = stars
    out["star_load"] = float(np.mean([s["load"] for s in stars])) if len(stars) == KEY_N else None
    return out


def game_features(games: pd.DataFrame, box: pd.DataFrame) -> pd.DataFrame:
    """每場一列：主客各項特徵與差值（主 − 客）。"""
    t = team_games(games.astype({"event_id": str}))
    pf = player_features(games, box)
    t = t.merge(pf, on=["event_id", "team"], how="left")
    cols = ["margin10", "ats10", "ou10", "star_form", "star_out", "star_load", "n_prior"]
    h = t[t["side"] == "home"].set_index("event_id")[cols].add_prefix("h_")
    a = t[t["side"] == "away"].set_index("event_id")[cols].add_prefix("a_")
    f = h.join(a)
    for c in ("margin10", "ats10", "star_form", "star_out", "star_load"):
        f[f"d_{c}"] = f[f"h_{c}"] - f[f"a_{c}"]
    f["s_ou10"] = f["h_ou10"] + f["a_ou10"]
    return f.reset_index()
