# -*- coding: utf-8 -*-
"""即時監控：抓盤口快照 → 套用回測過的策略 → Telegram 推播 → 賽後自動結算追蹤成績。

資料來源：
  ESPN scoreboard  NBA / 各足球聯賽的 DraftKings 開盤 & 即時盤口、比分（免費、免金鑰）
  ESPN injuries    NBA 傷兵名單
  The Odds API     （選用）多家博彩公司即時賠率，用來比價找價值
"""
from __future__ import annotations

import difflib
import json
import re
import sqlite3
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from html import escape
from zoneinfo import ZoneInfo

import config
import data_nba
from backtest import load_strategies
from core import (american_to_decimal, db, get_json, http_get, kv_get, kv_set, log, no_vig, now_utc, parse_iso,
                  parse_line, send_telegram, to_tpe)
from strategies import (PICK_ZH, Strategy, bet_odds, lam_from_over, market_prob, no_vig2, no_vig3, p_over, settle,
                        trigger)

ET = ZoneInfo("America/New_York")
SITE = "https://site.api.espn.com/apis/site/v2/sports/{path}/scoreboard"
INJURIES = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
BYATHLETE = "https://site.web.api.espn.com/apis/common/v3/sports/basketball/nba/statistics/byathlete"
WEEKDAY_ZH = "一二三四五六日"
DISCLAIMER = "⚠️ 研究用途，非投注建議；盤口可能再變動，請量力而為。"


# =============================================================================
# 1. 抓 ESPN 賽程與盤口
# =============================================================================
def _leagues() -> list[tuple[str, str, str]]:
    out = [("nba", "basketball/nba", "NBA"), ("mlb", "baseball/mlb", "MLB")]
    out += [("football", f"soccer/{slug}", name) for slug, name in config.FOOTBALL_LEAGUES.values()]
    out += [("football", f"soccer/{slug}", name) for slug, name in config.FOOTBALL_CUPS.items()]
    return out


def _dig(d: dict, *keys):
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def parse_site_odds(lst: list | None) -> dict:
    """ESPN scoreboard 的 odds：open = 開盤，close = 目前（賽前）/ 收盤（賽後）。"""
    for o in lst or []:
        if not isinstance(o, dict):
            continue
        ml, ps, tot = o.get("moneyline") or {}, o.get("pointSpread") or {}, o.get("total") or {}
        if not (ml or ps or tot):
            continue
        r = {"provider": _dig(o, "provider", "name") or "DraftKings"}
        for tag, p in (("open", "o"), ("close", "c")):
            r[f"{p}_ml_home"] = american_to_decimal(_dig(ml, "home", tag, "odds"))
            r[f"{p}_ml_away"] = american_to_decimal(_dig(ml, "away", tag, "odds"))
            r[f"{p}_ml_draw"] = american_to_decimal(_dig(ml, "draw", tag, "odds"))
            r[f"{p}_spread_home"] = parse_line(_dig(ps, "home", tag, "line"))
            r[f"{p}_spread_home_odds"] = american_to_decimal(_dig(ps, "home", tag, "odds"))
            r[f"{p}_spread_away_odds"] = american_to_decimal(_dig(ps, "away", tag, "odds"))
            r[f"{p}_total"] = parse_line(_dig(tot, "over", tag, "line"))
            r[f"{p}_over_odds"] = american_to_decimal(_dig(tot, "over", tag, "odds"))
            r[f"{p}_under_odds"] = american_to_decimal(_dig(tot, "under", tag, "odds"))
        return r
    return {}


def parse_event(ev: dict, sport: str, path: str, league: str) -> dict | None:
    comp = (ev.get("competitions") or [{}])[0]
    teams = {c.get("homeAway"): c for c in comp.get("competitors", [])}
    if "home" not in teams or "away" not in teams:
        return None
    if sport in ("nba", "mlb") and _dig(ev, "season", "type") not in (2, 3):  # 排除季前賽、明星賽
        return None
    st = _dig(comp, "status", "type") or {}

    def score(c):
        try:
            return int(float(c.get("score")))
        except (TypeError, ValueError):
            return None

    g = {
        "game_id": f"espn:{path}:{ev['id']}", "sport": sport, "league": league, "kickoff": ev["date"],
        "home": teams["home"]["team"].get("displayName", ""), "away": teams["away"]["team"].get("displayName", ""),
        "home_abbr": teams["home"]["team"].get("abbreviation", ""), "away_abbr": teams["away"]["team"].get("abbreviation", ""),
        "state": st.get("state"), "completed": bool(st.get("completed")), "status_name": st.get("name", ""),
        "home_score": score(teams["home"]), "away_score": score(teams["away"]),
        "odds": parse_site_odds(comp.get("odds")),
        "note": ((comp.get("notes") or [{}])[0] or {}).get("headline", ""),
        "series": (comp.get("series") or {}).get("summary", ""),
        "playoffs": _dig(ev, "season", "type") == 3,
    }
    if sport == "mlb":
        g["probables"] = {}
        g["records"] = {}
        for side, c in teams.items():
            p = next((x for x in c.get("probables") or [] if x.get("name") == "probableStartingPitcher"), None)
            if p and p.get("athlete"):
                g["probables"][side] = (str(p["athlete"].get("id")), p["athlete"].get("displayName", ""))
            recs = c.get("records") or []
            if recs:
                g["records"][side] = recs[0].get("summary", "")
    return g


def fetch_games() -> list[dict]:
    today = datetime.now(ET).date()
    jobs = []
    for sport, path, league in _leagues():
        ahead = {"nba": config.NBA_LOOKAHEAD_DAYS, "mlb": config.MLB_LOOKAHEAD_DAYS}.get(sport, config.FOOTBALL_LOOKAHEAD_DAYS)
        for off in range(-2, ahead + 1):
            jobs.append((sport, path, league, today + timedelta(days=off), -1 <= off <= 1))

    def load(job):
        sport, path, league, d, near = job
        data = get_json(SITE.format(path=path), params={"dates": d.strftime("%Y%m%d")},
                        cache_name=f"live/{path.replace('/', '_')}_{d:%Y%m%d}.json",
                        max_age_hours=0.4 if near else 3)
        return [g for ev in (data or {}).get("events", []) if (g := parse_event(ev, sport, path, league))]

    games: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for lst in pool.map(load, jobs):
            for g in lst:
                games[g["game_id"]] = g
    return list(games.values())


def save_games(conn: sqlite3.Connection, games: list[dict]) -> None:
    ts = now_utc().isoformat(timespec="seconds")
    for g in games:
        status = "final" if g["completed"] else g["state"]
        conn.execute(
            """INSERT INTO games(game_id, sport, league, kickoff, home, away, home_abbr, away_abbr, status,
                                 home_score, away_score, updated)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(game_id) DO UPDATE SET kickoff=excluded.kickoff, status=excluded.status,
                 home_score=excluded.home_score, away_score=excluded.away_score, updated=excluded.updated""",
            (g["game_id"], g["sport"], g["league"], parse_iso(g["kickoff"]).isoformat(), g["home"], g["away"],
             g["home_abbr"], g["away_abbr"], status, g["home_score"], g["away_score"], ts))
        o = g["odds"]
        if g["state"] == "pre" and o and (o.get("c_ml_home") or o.get("c_spread_home") is not None):
            conn.execute(
                "INSERT OR IGNORE INTO snapshots VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (g["game_id"], ts, "espn", o.get("c_ml_home"), o.get("c_ml_draw"), o.get("c_ml_away"),
                 o.get("c_spread_home"), o.get("c_spread_home_odds"), o.get("c_spread_away_odds"),
                 o.get("c_total"), o.get("c_over_odds"), o.get("c_under_odds"),
                 json.dumps({k: v for k, v in o.items() if k.startswith("o_")})))
    conn.commit()


# =============================================================================
# 2. 特徵（欄位名稱與回測一致）
# =============================================================================
def _nba_played_yesterday(game_day) -> set[str]:
    d = game_day - timedelta(days=1)
    data = get_json(SITE.format(path="basketball/nba"), params={"dates": d.strftime("%Y%m%d")},
                    cache_name=f"live/basketball_nba_{d:%Y%m%d}.json", max_age_hours=6)
    out = set()
    for ev in (data or {}).get("events", []):
        if _dig(ev, "season", "type") in (2, 3):
            for c in ev["competitions"][0]["competitors"]:
                out.add(c["team"]["abbreviation"])
    return out


def _football_reference(conn: sqlite3.Connection, g: dict) -> dict:
    """回測的足球「開盤」是 football-data 在賽前 1–2 天擷取的賠率，
    所以優先用我們在開賽前 18–48 小時存下的快照當基準；沒有的話才用 DraftKings 開盤。"""
    o = g["odds"]
    kick = parse_iso(g["kickoff"])
    row = conn.execute(
        """SELECT ml_home, ml_draw, ml_away, total, over_odds, under_odds FROM snapshots
           WHERE game_id=? AND source='espn' AND ts BETWEEN ? AND ? ORDER BY ts LIMIT 1""",
        (g["game_id"], (kick - timedelta(hours=48)).isoformat(timespec="seconds"),
         (kick - timedelta(hours=18)).isoformat(timespec="seconds"))).fetchone()
    if row and row["ml_home"] and row["ml_draw"] and row["ml_away"]:
        return {"h": row["ml_home"], "d": row["ml_draw"], "a": row["ml_away"], "total": row["total"],
                "over": row["over_odds"], "under": row["under_odds"], "src": "賽前1-2天快照"}
    return {"h": o.get("o_ml_home"), "d": o.get("o_ml_draw"), "a": o.get("o_ml_away"), "total": o.get("o_total"),
            "over": o.get("o_over_odds"), "under": o.get("o_under_odds"), "src": "開盤"}


def features(conn: sqlite3.Connection, g: dict, played_yesterday: dict) -> dict:
    o = g["odds"]
    f = {"home_score": g["home_score"], "away_score": g["away_score"]}
    if g["sport"] == "mlb":
        f.update({k: v for k, v in o.items() if k[:2] in ("o_", "c_")})
        for p in ("o", "c"):  # 抽水不合理的不讓分賠率當作沒有（與回測清理規則一致）
            h, a = f.get(f"{p}_ml_home"), f.get(f"{p}_ml_away")
            if h and a and not 0 <= 1 / h + 1 / a - 1 <= 0.12:
                f[f"{p}_ml_home"] = f[f"{p}_ml_away"] = None
    elif g["sport"] == "nba":
        f.update({k: v for k, v in o.items() if k[:2] in ("o_", "c_")})
        for p in ("o", "c"):
            h, a, s = f.get(f"{p}_ml_home"), f.get(f"{p}_ml_away"), f.get(f"{p}_spread_home")
            if h and a and s is not None and not data_nba.ml_consistent(h, a, s):
                f[f"{p}_ml_home"] = f[f"{p}_ml_away"] = None
        day = parse_iso(g["kickoff"]).astimezone(ET).date()
        if day not in played_yesterday:
            played_yesterday[day] = _nba_played_yesterday(day)
        f["home_rest"] = 1 if g["home_abbr"] in played_yesterday[day] else 2
        f["away_rest"] = 1 if g["away_abbr"] in played_yesterday[day] else 2
        if f.get("c_total"):
            f["total_rel"] = f["c_total"] - nba_total_baseline(conn)
    else:
        ref = _football_reference(conn, g)
        f.update({"o_avg_h": ref["h"], "o_avg_d": ref["d"], "o_avg_a": ref["a"],
                  "c_avg_h": o.get("c_ml_home"), "c_avg_d": o.get("c_ml_draw"), "c_avg_a": o.get("c_ml_away"),
                  "ref_src": ref["src"],
                  # DK 足球讓球是 ±0.5 / ±1.5 這種半球盤，等同亞洲讓球的同一盤口
                  "o_ah_line": o.get("o_spread_home"), "c_ah_line": o.get("c_spread_home"),
                  "c_ah_h": o.get("c_spread_home_odds"), "c_ah_a": o.get("c_spread_away_odds")})
        if o.get("c_total") == 2.5:
            f.update({"c_avg_over": o.get("c_over_odds"), "c_avg_under": o.get("c_under_odds")})
            if ref["total"] == 2.5:
                f.update({"o_avg_over": ref["over"], "o_avg_under": ref["under"]})
        p = no_vig2(o.get("c_over_odds"), o.get("c_under_odds"))
        if o.get("c_total") and p:  # 3.5 / 1.5 球盤口用 Poisson 換算成大 2.5 機率
            f["p_over25"] = p_over(lam_from_over(p, o["c_total"]), 2.5)
    return f


_NBA_BASE: float | None = None


def nba_total_baseline(conn: sqlite3.Connection) -> float:
    """近期聯盟平均大小分：優先用資料庫中最近 150 場已完賽比賽的最後盤口；季初不夠就用上季例行賽最後 150 場。"""
    rows = conn.execute(
        """SELECT s.total FROM snapshots s JOIN games g ON g.game_id = s.game_id
           WHERE g.sport='nba' AND g.status='final' AND s.total IS NOT NULL
             AND s.ts = (SELECT MAX(ts) FROM snapshots WHERE game_id = s.game_id)
           ORDER BY g.kickoff DESC LIMIT 150""").fetchall()
    if len(rows) >= 40:
        return sum(r["total"] for r in rows) / len(rows)
    global _NBA_BASE
    if _NBA_BASE is None:
        h = data_nba.load_history()
        reg = h[~h["playoffs"]].dropna(subset=["c_total"])
        _NBA_BASE = float(reg["c_total"].tail(150).mean())
    return _NBA_BASE


# =============================================================================
# 3. 訊息格式
# =============================================================================
def team_name(g: dict, side: str) -> str:
    if g["sport"] in ("nba", "mlb"):
        abbr = g[f"{side}_abbr"]
        names = config.NBA_TEAM_ZH if g["sport"] == "nba" else config.MLB_TEAM_ZH
        return f"{names.get(abbr, g[side])} {abbr}"
    return g[side]


def kickoff_str(g: dict) -> str:
    t = to_tpe(parse_iso(g["kickoff"]))
    return f"{t:%m/%d}({WEEKDAY_ZH[t.weekday()]}) {t:%H:%M}"


def pick_text(g: dict, market: str, pick: str, line: float | None) -> str:
    if market in ("1x2",):
        return {"H": f"{team_name(g, 'home')} 勝", "D": "和局", "A": f"{team_name(g, 'away')} 勝"}[pick]
    if market == "ou25":
        return f"{PICK_ZH[pick]}（2.5 球）"
    if market in ("spread", "ah"):
        hcp = line if pick == "home" else -line
        return f"{team_name(g, pick)} 讓分 {hcp:+g}"
    if market == "total":
        return f"總分 {PICK_ZH[pick]} {line:g}"
    if market == "ml":
        return f"{team_name(g, pick)} 不讓分"
    return pick


def line_summary(g: dict) -> str:
    o = g["odds"]
    parts = []
    if g["sport"] == "mlb":
        po, pc = no_vig2(o.get("o_ml_home"), o.get("o_ml_away")), no_vig2(o.get("c_ml_home"), o.get("c_ml_away"))
        if po and pc:
            parts.append(f"主勝率 {po*100:.0f}→{pc*100:.0f}%（{o['c_ml_home']:.2f}/{o['c_ml_away']:.2f}）")
        if o.get("c_spread_home") is not None and o.get("c_spread_home_odds"):
            parts.append(f"讓分 主{o['c_spread_home']:+g} {o['c_spread_home_odds']:.2f}/{o['c_spread_away_odds']:.2f}")
        if o.get("c_total"):
            parts.append(f"大小 {o.get('o_total') or o['c_total']:g}→{o['c_total']:g}")
    elif g["sport"] == "nba":
        if o.get("o_spread_home") is not None and o.get("c_spread_home") is not None:
            parts.append(f"讓分(主) {o['o_spread_home']:+g}→{o['c_spread_home']:+g}")
        if o.get("o_total") and o.get("c_total"):
            parts.append(f"大小 {o['o_total']:g}→{o['c_total']:g}")
        po, pc = no_vig2(o.get("o_ml_home"), o.get("o_ml_away")), no_vig2(o.get("c_ml_home"), o.get("c_ml_away"))
        if po and pc:
            parts.append(f"主勝率 {po*100:.0f}→{pc*100:.0f}%")
    else:
        po = no_vig3(o.get("o_ml_home"), o.get("o_ml_draw"), o.get("o_ml_away"))
        pc = no_vig3(o.get("c_ml_home"), o.get("c_ml_draw"), o.get("c_ml_away"))
        if po and pc:
            parts.append("勝平負 " + "/".join(f"{a*100:.0f}→{b*100:.0f}" for a, b in zip(po, pc)) + "%")
        if o.get("c_ml_home"):
            parts.append(f"賠率 {o['c_ml_home']:.2f}/{o['c_ml_draw'] or 0:.2f}/{o['c_ml_away']:.2f}")
    return "｜".join(parts)


def stats_line(s: Strategy) -> str:
    st = s.stats
    seasons = all(v["roi"] > 0 for v in st["by_season"].values() if v["n"] >= 40)
    return (f"回測 {st['n']} 注｜命中 {st['hit']*100:.1f}%｜ROI {st['roi']*100:+.1f}%｜p={st['p']:.2f}"
            + ("｜每季皆正" if seasons else ""))


def breakeven_line(s: Strategy, p_mkt: float | None) -> str:
    """把回測的「實際勝率 - 市場機率」加回現在的市場機率，換算成運彩至少要開多少才有正期望值。"""
    if p_mkt is None:
        return ""
    p_est = min(0.97, max(0.03, p_mkt + s.stats.get("uplift", 0.0)))
    return f"估計勝率 {p_est*100:.1f}% → 運彩賠率 ≥ <b>{1/p_est:.2f}</b> 才有正期望值（盤口需相同）"


# =============================================================================
# 4. 策略訊號
# =============================================================================
def scan_signals(conn: sqlite3.Connection, games: list[dict], strats: list[Strategy], dry_run: bool) -> int:
    active = [s for s in strats if s.live_source == "espn" and s.enabled]
    if not active:
        return 0
    now, sent, rest_cache = now_utc(), 0, {}
    for g in games:
        if g["state"] != "pre" or not g["odds"]:
            continue
        hours = (parse_iso(g["kickoff"]) - now).total_seconds() / 3600
        if not 0 < hours <= 72:
            continue
        f = features(conn, g, rest_cache)
        window = config.SIGNAL_WINDOW_HOURS[g["sport"]]
        items = []
        for s in active:
            if s.sport != g["sport"] or not (t := trigger(s, f)):
                continue
            pick, line, why = t
            if hours <= window:
                tier, key = "signal", f"{g['game_id']}|{s.sid}|{pick}"
            elif config.EARLY_STEAM_ALERT:
                tier, key = "early", f"{g['game_id']}|{s.family}|{pick}|early"
            else:
                continue
            if conn.execute("SELECT 1 FROM alerts WHERE alert_key=?", (key,)).fetchone():
                continue
            items.append((tier, key, s, pick, line, why, bet_odds(s, f, pick), market_prob(s, f, pick)))
        if not items:
            continue
        if send_telegram(signal_message(g, items, hours, f), dry_run=dry_run):
            for tier, key, s, pick, line, why, odds, p in items:
                conn.execute(
                    """INSERT OR IGNORE INTO alerts(alert_key, game_id, sport, strategy, market, pick, line, odds,
                       p_mkt, sent_at, tier, close_line, close_odds) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (key, g["game_id"], g["sport"], s.sid, s.market, pick, line, odds, p,
                     now.isoformat(timespec="seconds"), tier, line, odds))
            conn.commit()
            sent += len(items)
    return sent


def signal_message(g: dict, items: list, hours: float, f: dict) -> str:
    icon = "🏀" if g["sport"] == "nba" else "⚽"
    early = all(it[0] == "early" for it in items)
    head = "👀 盤口異動預警（尚未到下注時機）" if early else "⭐ 下注訊號"
    lines = [f"{head} {icon} {escape(g['league'])}",
             f"<b>{escape(team_name(g, 'away'))} @ {escape(team_name(g, 'home'))}</b>" if g["sport"] == "nba"
             else f"<b>{escape(g['home'])} vs {escape(g['away'])}</b>",
             f"🕐 {kickoff_str(g)} 台灣時間（約 {hours:.1f} 小時後）", ""]
    for tier, _key, s, pick, line, why, odds, p in items:
        lines.append(f"📈 {escape(s.title)}")
        lines.append(f"   {escape(why)}" + (f"（基準：{f.get('ref_src')}）" if f.get("ref_src") else ""))
        price = f" @ {odds:.2f}" if odds else ""
        lines.append(f"👉 <b>{escape(pick_text(g, s.market, pick, line))}{price}</b>（{escape(g['odds'].get('provider', ''))} 現價）")
        lines.append(f"   {stats_line(s)}")
        if be := breakeven_line(s, p):
            lines.append(f"   {be}")
        if tier == "early":
            lines.append(f"   ⏳ 回測是以收盤價下注，開賽前 {config.SIGNAL_WINDOW_HOURS[g['sport']]} 小時內若仍成立會再通知")
        lines.append("")
    lines += [f"目前盤口：{escape(line_summary(g))}", DISCLAIMER]
    return "\n".join(lines)


# =============================================================================
# 5. NBA 傷兵
# =============================================================================
def nba_season_year(d=None) -> int:
    d = d or datetime.now(ET).date()
    return d.year + 1 if d.month >= 9 else d.year


def key_players() -> dict[str, tuple[str, float]]:
    """場均得分 ≥ 門檻的球員（本季出賽 ≥5 場；季初資料不足時沿用上季）。"""
    out: dict[str, tuple[str, float]] = {}
    year = nba_season_year()
    for season in (year, year - 1):
        data = get_json(BYATHLETE, params={"season": season, "seasontype": 2, "limit": 250,
                                           "sort": "offensive.avgPoints:desc"},
                        cache_name=f"live/nba_scorers_{season}.json", max_age_hours=24)
        if not data:
            continue
        names = {c["name"]: c.get("names", []) for c in data.get("categories", [])}
        try:
            i_pts, i_gp = names["offensive"].index("avgPoints"), names["general"].index("gamesPlayed")
        except (KeyError, ValueError):
            continue
        for a in data.get("athletes", []):
            cats = {c["name"]: c.get("values", []) for c in a.get("categories", [])}
            try:
                ppg, gp = float(cats["offensive"][i_pts]), float(cats["general"][i_gp])
            except (KeyError, IndexError, TypeError, ValueError):
                continue
            aid = str(a.get("athlete", {}).get("id"))
            if gp >= 5 and ppg >= config.INJURY_KEY_PLAYER_MIN_PPG and aid not in out:
                out[aid] = (a["athlete"].get("displayName", ""), ppg)
    return out


def _athlete_id(ath: dict) -> str | None:
    if ath.get("id"):
        return str(ath["id"])
    for link in ath.get("links", []):
        if m := re.search(r"/id/(\d+)", link.get("href", "")):
            return m.group(1)
    return None


def check_injuries(conn: sqlite3.Connection, games: list[dict], dry_run: bool) -> int:
    data = get_json(INJURIES)
    if not data:
        return 0
    current = {}
    for team in data.get("injuries", []):
        for inj in team.get("injuries", []):
            ath = inj.get("athlete", {})
            pid = _athlete_id(ath)
            if pid:
                current[pid] = {"team": _dig(ath, "team", "abbreviation") or "", "name": ath.get("displayName", ""),
                                "status": inj.get("status", ""), "comment": inj.get("shortComment", "")}
    prev = {r["player_id"]: dict(r) for r in conn.execute("SELECT * FROM injuries")}
    first_run = not prev and kv_get(conn, "injuries_init") is None
    ts = now_utc().isoformat(timespec="seconds")
    conn.execute("DELETE FROM injuries")
    conn.executemany("INSERT INTO injuries VALUES (?,?,?,?,?,?)",
                     [(pid, c["team"], c["name"], c["status"], c["comment"], ts) for pid, c in current.items()])
    kv_set(conn, "injuries_init", ts)
    if first_run:
        log.info("傷兵名單初始化：%d 筆（第一次不推播）", len(current))
        return 0

    now = now_utc()
    upcoming = {}
    for g in sorted(games, key=lambda x: x["kickoff"]):
        if g["sport"] == "nba" and g["state"] == "pre":
            hrs = (parse_iso(g["kickoff"]) - now).total_seconds() / 3600
            if 0 < hrs <= config.INJURY_LOOKAHEAD_HOURS:
                upcoming.setdefault(g["home_abbr"], g)
                upcoming.setdefault(g["away_abbr"], g)
    keys = key_players()
    bad = config.INJURY_ALERT_STATUSES
    msgs = []
    for pid in set(current) | set(prev):
        c, p = current.get(pid), prev.get(pid)
        if pid not in keys or (c and p and c["status"] == p["status"]):
            continue
        info = c or p
        g = upcoming.get(info["team"])
        if not g:
            continue
        name, ppg = keys[pid]
        if c and c["status"] in bad:
            head = f"🚑 <b>{escape(name)}</b>（{escape(info['team'])}，場均 {ppg:.1f} 分）→ <b>{escape(c['status'])}</b>"
            detail = c["comment"]
        elif p and p["status"] in bad:
            head = f"✅ <b>{escape(name)}</b>（{escape(info['team'])}，場均 {ppg:.1f} 分）{escape(p['status'])} → {escape(c['status'] if c else '可出賽')}"
            detail = c["comment"] if c else ""
        else:
            continue
        side = "home" if g["home_abbr"] == info["team"] else "away"
        msgs.append("\n".join(filter(None, [
            head, f"   {escape(detail)}" if detail else "",
            f"   下一場：{escape(team_name(g, 'away'))} @ {escape(team_name(g, 'home'))}｜{kickoff_str(g)}",
            f"   盤口（開盤→現在）：{escape(line_summary(g))}",
            f"   {'⚠️ 盤口還沒明顯反應，可留意' if _line_unmoved(g, side) else '盤口已有反應'}",
        ])))
    if msgs:
        send_telegram("🏀 <b>NBA 主力傷兵異動</b>\n\n" + "\n\n".join(msgs), dry_run=dry_run)
    return len(msgs)


def _line_unmoved(g: dict, side: str) -> bool:
    o = g["odds"]
    if o.get("o_spread_home") is None or o.get("c_spread_home") is None:
        return False
    return abs(o["c_spread_home"] - o["o_spread_home"]) < 1


# =============================================================================
# 6. 選用：The Odds API 各家比價
# =============================================================================
SHARP_BOOKS = ("pinnacle", "betfair_ex_eu", "betfair_ex_uk", "matchbook")


def _norm(name: str) -> str:
    name = name.lower()
    for w in ("fc", "cf", "afc", "sc", "ac", "club", "de", "the", "1."):
        name = re.sub(rf"\b{re.escape(w)}\b", " ", name)
    return re.sub(r"[^a-z0-9]+", " ", name).strip()


def _match_game(games: list[dict], sport: str, home: str, away: str, kickoff) -> dict | None:
    best, best_score = None, 0.0
    for g in games:
        if g["sport"] != sport or abs((parse_iso(g["kickoff"]) - kickoff).total_seconds()) > 3 * 3600:
            continue
        score = (difflib.SequenceMatcher(None, _norm(g["home"]), _norm(home)).ratio()
                 + difflib.SequenceMatcher(None, _norm(g["away"]), _norm(away)).ratio()) / 2
        if score > best_score:
            best, best_score = g, score
    return best if best_score >= 0.6 else None


def oddsapi_sweep(conn: sqlite3.Connection, games: list[dict], strats: list[Strategy], dry_run: bool,
                  force: bool = False) -> int:
    if not config.ODDS_API_KEY:
        return 0
    now = now_utc()
    last = kv_get(conn, "oddsapi_last")
    if not force and last and now - parse_iso(last) < timedelta(hours=24 / config.ODDS_API_SWEEPS_PER_DAY):
        return 0
    kv_set(conn, "oddsapi_last", now.isoformat(timespec="seconds"))
    ref = {s.sid: s for s in strats}.get("fb_value_open_0.05")
    found, remaining = [], None
    for sport_key, league in config.ODDS_API_SPORTS.items():
        resp = http_get(f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds",
                        params={"apiKey": config.ODDS_API_KEY, "regions": "eu", "markets": "h2h", "oddsFormat": "decimal"})
        if resp is None:
            log.warning("The Odds API 讀取失敗：%s", sport_key)
            continue
        remaining = resp.headers.get("x-requests-remaining", remaining)
        sport = "nba" if sport_key.startswith("basketball") else "football"
        for ev in resp.json():
            kick = parse_iso(ev["commence_time"])
            if not 0 < (kick - now).total_seconds() / 3600 <= 72:
                continue
            books = {b["key"]: (b["title"], {o["name"]: o["price"] for m in b.get("markets", []) if m["key"] == "h2h"
                                              for o in m.get("outcomes", [])}) for b in ev.get("bookmakers", [])}
            sharp_key = next((k for k in ("pinnacle", "betfair_ex_eu") if k in books), None)
            if not sharp_key:
                continue
            names = [ev["home_team"], ev["away_team"]] + (["Draw"] if sport == "football" else [])
            fair = no_vig([books[sharp_key][1].get(n) for n in names])
            if not fair:
                continue
            for i, n in enumerate(names):
                cands = [(v[1][n], v[0]) for k, v in books.items() if k not in SHARP_BOOKS and v[1].get(n)]
                if not cands:
                    continue
                best, book = max(cands)
                edge = best * fair[i] - 1
                if edge >= config.ODDS_API_VALUE_EDGE and 1.25 <= best <= 8.0:
                    found.append((sport, league, ev, kick, i, n, best, book, fair[i], edge, sharp_key))
    log.info("The Odds API 掃描完成，找到 %d 個價值選項，剩餘額度 %s", len(found), remaining)

    lines, new = [], 0
    for sport, league, ev, kick, i, n, best, book, p_fair, edge, sharp_key in sorted(found, key=lambda x: -x[9]):
        g = _match_game(games, sport, ev["home_team"], ev["away_team"], kick)
        pick = ("H", "A", "D")[i] if sport == "football" else ("home", "away")[i]
        key = f"oddsapi|{ev['id']}|{pick}|{to_tpe(now):%Y%m%d}"
        if conn.execute("SELECT 1 FROM alerts WHERE alert_key=?", (key,)).fetchone():
            continue
        conn.execute(
            """INSERT INTO alerts(alert_key, game_id, sport, strategy, market, pick, line, odds, p_mkt, sent_at, tier,
               close_line, close_odds) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (key, g["game_id"] if g else f"oddsapi:{ev['id']}", sport, "oddsapi_value",
             "1x2" if sport == "football" else "ml", pick, None, best, p_fair,
             now.isoformat(timespec="seconds"), "value", None, None))
        new += 1
        t = to_tpe(kick)
        pick_name = "和局" if n == "Draw" else f"{n} 勝"
        lines.append(f"{'⚽' if sport == 'football' else '🏀'} {escape(league)}｜{t:%m/%d %H:%M}\n"
                     f"   {escape(ev['home_team'])} vs {escape(ev['away_team'])}\n"
                     f"   👉 <b>{escape(pick_name)} @ {best:.2f}</b>（{escape(book)}）\n"
                     f"   公平賠率 {1/p_fair:.2f}（{'Pinnacle' if sharp_key == 'pinnacle' else 'Betfair'} 去水）｜期望值 <b>{edge*100:+.1f}%</b>")
    conn.commit()
    if lines:
        head = "💰 <b>各家賠率比價：高於尖銳盤公平價</b>"
        foot = []
        if ref and ref.stats.get("n"):
            foot.append(f"回測參考（足球早盤 ≥5%）：{ref.stats['n']} 注、ROI {ref.stats['roi']*100:+.1f}%、p={ref.stats['p']:.2f}（未達顯著，僅供比價）")
        foot.append("運彩賠率若 ≥ 公平賠率，同樣代表有正期望值。" + DISCLAIMER)
        send_telegram("\n\n".join([head] + lines[:15] + ["\n".join(foot)]), dry_run=dry_run)
    return new


# =============================================================================
# 7. 結算與 CLV
# =============================================================================
def _current_price(g: dict, market: str, pick: str) -> tuple[float | None, float | None]:
    o = g["odds"]
    if market == "ml":
        return None, o.get(f"c_ml_{pick}")
    if market == "1x2":
        return None, o.get({"H": "c_ml_home", "D": "c_ml_draw", "A": "c_ml_away"}[pick])
    if market == "spread":
        return o.get("c_spread_home"), o.get("c_spread_home_odds" if pick == "home" else "c_spread_away_odds")
    if market in ("total", "ou25"):
        return o.get("c_total"), o.get("c_over_odds" if pick == "over" else "c_under_odds")
    return None, None


def settle_alerts(conn: sqlite3.Connection, games: list[dict]) -> int:
    by_id = {g["game_id"]: g for g in games}
    rows = conn.execute("""SELECT a.*, g.status, g.home_score, g.away_score, g.kickoff FROM alerts a
                           LEFT JOIN games g ON a.game_id = g.game_id WHERE a.result IS NULL""").fetchall()
    done, now = 0, now_utc()
    for r in rows:
        g = by_id.get(r["game_id"])
        if g and g["state"] == "pre":                        # 賽前：持續更新「最後盤口」算 CLV
            line, odds = _current_price(g, r["market"], r["pick"]) if r["tier"] != "value" else (None, None)
            if odds:
                conn.execute("UPDATE alerts SET close_line=?, close_odds=? WHERE alert_key=?", (line, odds, r["alert_key"]))
            continue
        if r["status"] == "final" and r["home_score"] is not None:
            dummy = Strategy("", r["sport"], "", "", 0, r["market"], "", "")
            game = {"hg": r["home_score"], "ag": r["away_score"], "home_score": r["home_score"], "away_score": r["away_score"]}
            line = r["line"] if r["line"] is not None else 0.0
            profit = settle(dummy, game, r["pick"], line, r["odds"]) if r["odds"] else 0.0
            result = "W" if profit > 0 else "L" if profit < 0 else "P"
            conn.execute("UPDATE alerts SET result=?, profit=?, settled_at=? WHERE alert_key=?",
                         (result, profit, now.isoformat(timespec="seconds"), r["alert_key"]))
            done += 1
        elif r["kickoff"] and now - parse_iso(r["kickoff"]) > timedelta(days=3):
            conn.execute("UPDATE alerts SET result='void', profit=0, settled_at=? WHERE alert_key=?",
                         (now.isoformat(timespec="seconds"), r["alert_key"]))
        elif r["game_id"].startswith("oddsapi:") and now - parse_iso(r["sent_at"]) > timedelta(days=4):
            conn.execute("UPDATE alerts SET result='void', profit=0, settled_at=? WHERE alert_key=?",
                         (now.isoformat(timespec="seconds"), r["alert_key"]))
    conn.commit()
    return done


def record_summary(conn: sqlite3.Connection, days: int | None = None) -> list[str]:
    since = (now_utc() - timedelta(days=days)).isoformat() if days else "0"
    rows = conn.execute("""SELECT tier, strategy, result, profit, odds, close_odds, line, close_line FROM alerts
                           WHERE result IN ('W','L','P') AND sent_at >= ?""", (since,)).fetchall()
    if not rows:
        return ["（尚無已結算的訊號）"]
    groups = defaultdict(list)
    for r in rows:
        groups[(r["tier"], r["strategy"])].append(r)
    tier_zh = {"signal": "⭐訊號", "early": "👀預警", "value": "💰比價"}
    out = []
    for (tier, sid), rs in sorted(groups.items()):
        w = sum(r["result"] == "W" for r in rs)
        l_ = sum(r["result"] == "L" for r in rs)
        p = sum(r["result"] == "P" for r in rs)
        units = sum(r["profit"] for r in rs)
        clv = [r["odds"] / r["close_odds"] - 1 for r in rs
               if r["close_odds"] and r["odds"] and (r["line"] == r["close_line"])]
        clv_s = f"｜CLV {sum(clv)/len(clv)*100:+.1f}%" if clv else ""
        out.append(f"{tier_zh.get(tier, tier)} {sid}：{w}勝{l_}敗{p}和｜{units:+.2f}u｜ROI {units/len(rs)*100:+.1f}%{clv_s}")
    return out


# =============================================================================
# 8. 每日總覽 / 每週成績
# =============================================================================
def daily_digest(conn: sqlite3.Connection, games: list[dict], strats: list[Strategy], dry_run: bool) -> None:
    now = now_utc()
    soon = [g for g in games if g["state"] == "pre"
            and 0 < (parse_iso(g["kickoff"]) - now).total_seconds() / 3600 <= 24]
    soon.sort(key=lambda g: g["kickoff"])
    usable = [s for s in strats if s.live_source == "espn" and (s.enabled or (s.weak and config.SHOW_WEAK_SIGNALS_IN_DIGEST))]
    rest_cache = {}
    t = to_tpe(now)
    lines = [f"📅 <b>每日賽事總覽</b> {t:%m/%d}({WEEKDAY_ZH[t.weekday()]})｜未來 24 小時", ""]

    def marks(g):
        f = features(conn, g, rest_cache)
        out = []
        for s in usable:
            if s.sport == g["sport"] and (tr := trigger(s, f)):
                out.append(("⭐" if s.enabled else "☆") + pick_text(g, s.market, tr[0], tr[1]))
        return out

    nba = [g for g in soon if g["sport"] == "nba"]
    for sport, head in (("mlb", "⚾ <b>MLB</b>"), ("nba", "🏀 <b>NBA</b>")):
        lst = [g for g in soon if g["sport"] == sport]
        if not lst:
            continue
        lines.append(f"{head}（{len(lst)} 場）")
        for g in lst:
            m = marks(g)
            tag = f" [{escape(g['note'])}｜{escape(g['series'])}]" if g.get("note") else ""
            lines.append(f"{to_tpe(parse_iso(g['kickoff'])):%H:%M} {escape(team_name(g, 'away'))} @ {escape(team_name(g, 'home'))}{tag}"
                         + (f"\n   {escape(line_summary(g))}" if g["odds"] else "")
                         + (f"\n   {escape(' / '.join(m))}" if m else ""))
        lines.append("")
    fb = [g for g in soon if g["sport"] == "football"]
    if fb:
        notable = []
        for g in fb:
            o = g["odds"]
            po = no_vig3(o.get("o_ml_home"), o.get("o_ml_draw"), o.get("o_ml_away"))
            pc = no_vig3(o.get("c_ml_home"), o.get("c_ml_draw"), o.get("c_ml_away"))
            move = max(abs(a - b) for a, b in zip(po, pc)) if po and pc else 0
            m = marks(g)
            if m or move >= 0.03:
                notable.append((g, m))
        lines.append(f"⚽ <b>足球</b>（監控 {len(fb)} 場，以下為盤口明顯變動 ≥3pp 或有訊號的比賽）")
        for g, m in notable[:25]:
            lines.append(f"{to_tpe(parse_iso(g['kickoff'])):%H:%M} [{escape(g['league'])}] {escape(g['home'])} vs {escape(g['away'])}\n"
                         f"   {escape(line_summary(g))}" + (f"\n   {escape(' / '.join(m))}" if m else ""))
        if not notable:
            lines.append("（目前沒有明顯變動）")
        lines.append("")
    if not soon:
        lines.append("未來 24 小時沒有監控中的比賽。\n")
    keys = key_players()
    teams = {g["home_abbr"] for g in nba} | {g["away_abbr"] for g in nba}
    inj = [r for r in conn.execute("SELECT * FROM injuries") if r["team"] in teams and r["player_id"] in keys
           and r["status"] in config.INJURY_ALERT_STATUSES]
    if inj:
        lines.append("🚑 <b>今日主力傷兵</b>")
        lines += [f"• {escape(r['name'])}（{r['team']}，{keys[r['player_id']][1]:.1f} 分）{escape(r['status'])}" for r in inj]
        lines.append("")
    lines.append("⭐ = 通過回測、開賽前會即時推播　☆ = 參考（正報酬但不顯著）")
    lines.append("<b>近 30 天實戰紀錄</b>")
    lines += record_summary(conn, 30)
    send_telegram("\n".join(lines), dry_run=dry_run)


def weekly_report(conn: sqlite3.Connection, dry_run: bool) -> None:
    lines = ["📊 <b>每週實戰成績</b>", "", "<b>近 7 天</b>"] + record_summary(conn, 7)
    lines += ["", "<b>全部累計</b>"] + record_summary(conn)
    lines += ["", "CLV = 推播時賠率 vs 開賽前最後賠率；長期 CLV 為正，代表訊號確實比收盤盤口早一步。"]
    send_telegram("\n".join(lines), dry_run=dry_run)


# =============================================================================
# 9. 一輪監控
# =============================================================================
def run_cycle(dry_run: bool = False) -> None:
    strats, _meta = load_strategies()
    if not strats:
        log.warning("找不到 data/strategies.json，請先執行 python main.py backtest")
    conn = db(dry_run)
    try:
        games = fetch_games()
        save_games(conn, games)
        n_sig = scan_signals(conn, games, strats, dry_run)
        n_inj = check_injuries(conn, games, dry_run)
        n_val = oddsapi_sweep(conn, games, strats, dry_run)
        n_set = settle_alerts(conn, games)
        import learning
        import parlay
        n_rec = learning.record_predictions(conn, games)   # 開賽前記錄模型預測（學習用）
        n_set += learning.settle_predictions(conn) + parlay.settle_parlays(conn)
        if n_rec:
            log.info("記錄模型預測 %d 筆", n_rec)
        pre = sum(g["state"] == "pre" and bool(g["odds"]) for g in games)
        log.info("監控完成：%d 場比賽（%d 場有盤口），訊號 %d、傷兵 %d、比價 %d、結算 %d",
                 len(games), pre, n_sig, n_inj, n_val, n_set)
    finally:
        conn.close()
