# -*- coding: utf-8 -*-
"""NBA 歷史資料：ESPN 賽程/比分 + 博彩公司（ESPN BET / DraftKings）開盤與收盤盤口。

每天一個 scoreboard 請求、每場比賽一個 odds 請求，全部快取在 data/cache/nba，
第一次下載約需數分鐘，之後只補新日期。
"""
from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta

import pandas as pd

from config import NBA_SEASONS
from core import CACHE_DIR, american_to_decimal, get_json, log, parse_line

SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
CORE_ODDS = ("https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events/{eid}"
             "/competitions/{eid}/odds")


def _dates(start: str, end: str):
    d, last = date.fromisoformat(start), min(date.fromisoformat(end), date.today() - timedelta(days=1))
    while d <= last:
        yield d
        d += timedelta(days=1)


def _price(block: dict | None, key: str) -> float | None:
    """從 {'spread': {'decimal': 1.9, 'american': '-110'}} 這類結構取小數賠率。"""
    v = (block or {}).get(key) or {}
    if isinstance(v, dict):
        if v.get("decimal"):
            return float(v["decimal"])
        return american_to_decimal(v.get("american") or v.get("alternateDisplayValue"))
    return None


def _line(block: dict | None, key: str) -> float | None:
    v = (block or {}).get(key) or {}
    return parse_line(v.get("american") or v.get("alternateDisplayValue")) if isinstance(v, dict) else None


def _market(it: dict, market: str) -> dict:
    """某家博彩公司某個盤（spread / ml / total）的開盤與收盤。舊季別只有 current（當作收盤）。"""
    h, a = it.get("homeTeamOdds") or {}, it.get("awayTeamOdds") or {}
    out = {}
    for tag in ("open", "close"):
        hb = h.get(tag) or (h.get("current") if tag == "close" else None) or {}
        ab = a.get(tag) or (a.get("current") if tag == "close" else None) or {}
        tb = it.get(tag) or (it.get("current") if tag == "close" else None) or {}
        p = tag[0]  # o_ / c_
        if market == "spread":
            out.update({f"{p}_spread_home": _line(hb, "pointSpread"), f"{p}_spread_home_odds": _price(hb, "spread"),
                        f"{p}_spread_away_odds": _price(ab, "spread")})
        elif market == "ml":
            out.update({f"{p}_ml_home": _price(hb, "moneyLine"), f"{p}_ml_away": _price(ab, "moneyLine")})
        else:
            out.update({f"{p}_total": _line(tb, "total"), f"{p}_over_odds": _price(tb, "over"),
                        f"{p}_under_odds": _price(tb, "under")})
    return out


_MARKET_KEY = {"spread": "spread_home", "ml": "ml_home", "total": "total"}


def parse_core_odds(data: dict | None) -> dict:
    """解析 core API 的 odds（不含 Live）。每個盤各自挑博彩公司：優先順序最高、有收盤的那家；
    那家沒有開盤時改用同時有開盤與收盤的下一家 —— 開盤與收盤一定來自同一家，盤口變化才不會失真
    （10/23：舊季別常常只有部分公司有某個盤，例如 MLB 2023 的 Bet365 沒有不讓分）。"""
    items = [it for it in (data or {}).get("items", []) if "live" not in it.get("provider", {}).get("name", "").lower()]
    if not items:
        return {}
    items = sorted(items, key=lambda x: x.get("provider", {}).get("priority", 99))
    row = {"provider": items[0].get("provider", {}).get("name")}
    for market, key in _MARKET_KEY.items():
        cands = [_market(it, market) for it in items]
        closed = [c for c in cands if c.get(f"c_{key}") is not None]
        both = [c for c in closed if c.get(f"o_{key}") is not None]
        row.update(both[0] if both else closed[0] if closed else cands[0])
    return row


def _scoreboard(d: date) -> dict | None:
    key = d.strftime("%Y%m%d")
    recent = d >= date.today() - timedelta(days=3)
    return get_json(SCOREBOARD, params={"dates": key, "limit": 50},
                    cache_name=f"nba/scoreboard_{key}.json", max_age_hours=6 if recent else None)


def _odds(eid: str) -> dict:
    return parse_core_odds(get_json(CORE_ODDS.format(eid=eid), cache_name=f"nba/odds_{eid}.json"))


def _season_games(season: str, start: str, end: str) -> list[dict]:
    days = list(_dates(start, end))
    with ThreadPoolExecutor(max_workers=8) as pool:
        boards = list(pool.map(_scoreboard, days))
    games = []
    for d, sb in zip(days, boards):
        for ev in (sb or {}).get("events", []):
            if ev.get("season", {}).get("type") not in (2, 3):  # 2=例行賽 3=季後賽
                continue
            comp = ev["competitions"][0]
            if not comp.get("status", {}).get("type", {}).get("completed"):
                continue
            teams = {c["homeAway"]: c for c in comp["competitors"]}
            if "home" not in teams or "away" not in teams:
                continue
            try:
                hs, as_ = int(teams["home"]["score"]), int(teams["away"]["score"])
            except (KeyError, TypeError, ValueError):
                continue
            games.append({
                "event_id": ev["id"], "season": season, "game_date": pd.Timestamp(d),
                "playoffs": ev["season"]["type"] == 3,
                "home": teams["home"]["team"]["abbreviation"], "away": teams["away"]["team"]["abbreviation"],
                "home_score": hs, "away_score": as_,
            })
    return games


def season_frame(sport: str, season: str, start: str, end: str, games_fn, odds_fn) -> pd.DataFrame:
    """一季的賽程 + 盤口。已結束的季別整季存成一個 CSV（之後不用再解析上千個 JSON），MLB 也共用。
    檔名帶 PARSED_VERSION：parse_core_odds 改了就把版本加 1，舊的整季檔自動不用。"""
    path = CACHE_DIR / sport / f"season_{season}_v{PARSED_VERSION}.csv"
    done = date.fromisoformat(end) < date.today() - timedelta(days=3)
    if done and path.exists():
        return pd.read_csv(path, dtype={"event_id": str, "season": str, "home": str, "away": str, "provider": str},
                           keep_default_na=False, na_values=[""], parse_dates=["game_date"])
    games = games_fn(season, start, end)
    with ThreadPoolExecutor(max_workers=8) as pool:
        odds = list(pool.map(lambda g: odds_fn(g["event_id"]), games))
    df = pd.DataFrame([{**g, **o} for g, o in zip(games, odds)])
    if done and len(df):
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    return df


PARSED_VERSION = 2
_MEMO: tuple[pd.DataFrame, float] | None = None


def load_history() -> pd.DataFrame:
    """所有 NBA_SEASONS 的已完賽比賽與盤口。同一個程式 10 分鐘內重複呼叫直接回傳記憶體裡的副本。"""
    global _MEMO
    if _MEMO and time.time() - _MEMO[1] < 600:
        return _MEMO[0].copy()
    frames = [season_frame("nba", s, a, b, _season_games, _odds) for s, a, b in NBA_SEASONS]
    df = pd.concat([f for f in frames if len(f)], ignore_index=True)
    log.info("NBA 賽程：%d 場（%d 季）", len(df), len(NBA_SEASONS))
    df = df.sort_values(["game_date", "event_id"]).reset_index(drop=True)
    df = clean(df)
    df = add_rest_days(df)
    # 近期聯盟平均大小分（前 150 場、不含當場，避免偷看未來）→ 這場盤口相對偏高或偏低
    df["total_base"] = df["c_total"].rolling(150, min_periods=50).mean().shift(1)
    df["total_rel"] = df["c_total"] - df["total_base"]
    have = df["c_spread_home"].notna().sum()
    log.info("NBA 歷史資料：%d 場（有收盤讓分 %d 場、有開盤讓分 %d 場）", len(df), have, int(df["o_spread_home"].notna().sum()))
    _MEMO = (df, time.time())
    return df.copy()


def _spread_prob(spread: float) -> float:
    """讓分 → 主隊勝率的粗估（NBA 分差標準差約 12.5 分）。"""
    return 0.5 * (1 + math.erf((-spread / 12.5) / math.sqrt(2)))


def ml_consistent(ml_home: float, ml_away: float, spread_home: float, tol: float = 0.10) -> bool:
    """不讓分賠率是否合理：抽水在 0~12%，且與讓分推得的勝率差距 ≤ tol。即時監控也用這個檢查。"""
    over = 1 / ml_home + 1 / ml_away - 1
    if over < 0 or over > 0.12:
        return False
    p_ml = (1 / ml_home) / (1 / ml_home + 1 / ml_away)
    return abs(p_ml - _spread_prob(spread_home)) <= tol


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """ESPN 少數比賽的開盤不讓分賠率是錯的（例如兩隊都 1.32、或讓 2.5 分卻開 1.10），
    會被誤判成「盤口大幅變動」。不讓分與讓分推得的勝率差太多、或抽水不合理時，清掉該時點的不讓分賠率。"""
    bad_total = 0
    for p in ("o", "c"):
        h, a, s = df[f"{p}_ml_home"], df[f"{p}_ml_away"], df[f"{p}_spread_home"]
        over = 1 / h + 1 / a - 1
        p_ml = (1 / h) / (1 / h + 1 / a)
        p_sp = s.map(lambda x: _spread_prob(x) if pd.notna(x) else float("nan"))
        bad = (over < 0) | (over > 0.12) | ((p_ml - p_sp).abs() > 0.10)
        df.loc[bad, [f"{p}_ml_home", f"{p}_ml_away"]] = float("nan")
        bad_total += int(bad.sum())
    if bad_total:
        log.info("NBA 清掉 %d 筆不合理的不讓分賠率", bad_total)
    return df


def add_rest_days(df: pd.DataFrame) -> pd.DataFrame:
    """每隊距上一場的天數（1 = 背靠背）。季初第一場設為 7。"""
    long = pd.concat([
        df[["event_id", "season", "game_date", "home"]].rename(columns={"home": "team"}).assign(side="home"),
        df[["event_id", "season", "game_date", "away"]].rename(columns={"away": "team"}).assign(side="away"),
    ]).sort_values("game_date")
    long["rest"] = long.groupby(["season", "team"])["game_date"].diff().dt.days.fillna(7).clip(upper=7)
    rest = long.pivot_table(index="event_id", columns="side", values="rest", aggfunc="first")
    df["home_rest"] = df["event_id"].map(rest["home"])
    df["away_rest"] = df["event_id"].map(rest["away"])
    return df


if __name__ == "__main__":
    t0 = datetime.now()
    d = load_history()
    print(d.groupby("season").agg(n=("event_id", "size"), spread=("c_spread_home", "count"),
                                  open_spread=("o_spread_home", "count"), total=("c_total", "count")))
    print(d.head(3).T)
    print("耗時", datetime.now() - t0)
