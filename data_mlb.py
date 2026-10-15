# -*- coding: utf-8 -*-
"""MLB 歷史資料：ESPN 賽程/比分 + DraftKings / ESPN BET 開盤與收盤（不讓分、讓分 ±1.5、大小分）。

快取在 data/cache/mlb，第一次下載約數分鐘（約 7,000 場）。
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import pandas as pd

from core import get_json, log
from data_nba import parse_core_odds

SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard"
CORE_ODDS = ("https://sports.core.api.espn.com/v2/sports/baseball/leagues/mlb/events/{eid}"
             "/competitions/{eid}/odds")

# (季別, 開始日, 結束日)：只取例行賽 + 季後賽（season.type 2 / 3），日期抓寬一點
MLB_SEASONS = [
    ("2024", "2024-03-20", "2024-11-02"),
    ("2025", "2025-03-18", "2025-11-03"),
    ("2026", "2026-03-25", "2026-11-05"),
]


def _dates(start: str, end: str):
    d, last = date.fromisoformat(start), min(date.fromisoformat(end), date.today() - timedelta(days=1))
    while d <= last:
        yield d
        d += timedelta(days=1)


def _scoreboard(d: date) -> dict | None:
    key = d.strftime("%Y%m%d")
    recent = d >= date.today() - timedelta(days=3)
    return get_json(SCOREBOARD, params={"dates": key, "limit": 50},
                    cache_name=f"mlb/scoreboard_{key}.json", max_age_hours=6 if recent else None)


def _odds(eid: str) -> dict:
    return parse_core_odds(get_json(CORE_ODDS.format(eid=eid), cache_name=f"mlb/odds_{eid}.json"))


def load_history() -> pd.DataFrame:
    games = []
    for season, start, end in MLB_SEASONS:
        days = list(_dates(start, end))
        with ThreadPoolExecutor(max_workers=8) as pool:
            boards = list(pool.map(_scoreboard, days))
        for d, sb in zip(days, boards):
            for ev in (sb or {}).get("events", []):
                stype = ev.get("season", {}).get("type")
                if stype not in (2, 3):
                    continue
                comp = ev["competitions"][0]
                if not comp.get("status", {}).get("type", {}).get("completed"):
                    continue
                teams = {c["homeAway"]: c for c in comp["competitors"]}
                try:
                    hs, as_ = int(teams["home"]["score"]), int(teams["away"]["score"])
                except (KeyError, TypeError, ValueError):
                    continue
                games.append({
                    "event_id": ev["id"], "season": season, "game_date": pd.Timestamp(d), "playoffs": stype == 3,
                    "home": teams["home"]["team"]["abbreviation"], "away": teams["away"]["team"]["abbreviation"],
                    "home_score": hs, "away_score": as_,
                })
    log.info("MLB 賽程：%d 場，開始抓取盤口…", len(games))
    with ThreadPoolExecutor(max_workers=8) as pool:
        odds = list(pool.map(lambda g: _odds(g["event_id"]), games))
    df = pd.DataFrame([{**g, **o} for g, o in zip(games, odds)])
    df = df.drop_duplicates("event_id").sort_values(["game_date", "event_id"]).reset_index(drop=True)
    # 清掉抽水不合理的不讓分賠率（ESPN 少數開盤資料有誤）
    for p in ("o", "c"):
        over = 1 / df[f"{p}_ml_home"] + 1 / df[f"{p}_ml_away"] - 1
        bad = (over < 0) | (over > 0.12)
        df.loc[bad, [f"{p}_ml_home", f"{p}_ml_away"]] = float("nan")
    log.info("MLB 歷史資料：%d 場（有收盤不讓分 %d 場，季後賽 %d 場）",
             len(df), int(df["c_ml_home"].notna().sum()), int(df["playoffs"].sum()))
    return df


if __name__ == "__main__":
    d = load_history()
    print(d.groupby(["season", "playoffs"]).agg(n=("event_id", "size"), ml=("c_ml_home", "count"),
                                                 o_ml=("o_ml_home", "count"), total=("c_total", "count"),
                                                 rl=("c_spread_home", "count")))
    print(d.tail(3).T)
