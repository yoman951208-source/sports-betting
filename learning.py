# -*- coding: utf-8 -*-
"""持續學習：每場比賽開賽前記錄模型預測 → 賽後自動結算 → 每天檢查模型準不準。

估計勝率原本是「市場機率 × shrink + 歷史相似盤口 × n」的加權，每天依實戰結果調 shrink。
10/23 用足球 8 季、NBA 11 季、MLB 7 季逐季留一回測：加上歷史相似盤口沒有讓預測更準，
用它挑出的「EV ≥ 2%」選項長期約 −5%（足球 8 季 0 季為正）；實戰也一樣（足球推薦 14 注 10 注和局、只中 1 注）。
所以現在估計勝率 = 校準後的市場機率，歷史相似盤口只當參考資料；這裡每天只「顯示」兩者的實戰 log loss 對照。
另外輸出校準表（模型說 60% 的事，實際發生幾成），放在賽後檢討裡。
"""
from __future__ import annotations

import math
import sqlite3
from datetime import timedelta

from core import db, kv_get, kv_set, log, now_utc, parse_iso

MARKET_ONLY = 1e9                       # 等同「完全相信市場」
COMPARE_SHRINK = 600.0                  # 每天對照用：舊的「市場 + 歷史相似盤口」權重
RECORD_WINDOW_HOURS = 3                 # 開賽前幾小時內開始記錄（之後每輪更新，開賽前最後一次為準）

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    game_id TEXT, market TEXT, pick TEXT,
    sport TEXT, league TEXT, kickoff TEXT, line REAL, odds REAL,
    p_mkt REAL, p_hist REAL, n INTEGER, p_model REAL, shrink REAL,
    recorded TEXT, result TEXT, profit REAL,
    PRIMARY KEY (game_id, market, pick)
);
"""


def ensure(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(predictions)")}
    if "p_raw" not in cols:  # 10/19 新增：未校準的市場機率（檢驗校準層用）
        conn.execute("ALTER TABLE predictions ADD COLUMN p_raw REAL")
        conn.commit()


def get_shrink(sport: str) -> float:
    """估計勝率要多相信市場：10/23 起一律只用市場機率（見模組說明）。"""
    return MARKET_ONLY


# -----------------------------------------------------------------------------
# 記錄與結算
# -----------------------------------------------------------------------------
def _line_for(m: dict, market: str) -> float | None:
    return m.get("L") if market == "total" else m.get("sp") if market == "spread" else None


def record_model(conn: sqlite3.Connection, g: dict, m: dict, ts: str | None = None) -> int:
    ts = ts or now_utc().isoformat(timespec="seconds")
    shrink = get_shrink(g["sport"])
    rows = 0
    for o in m["options"]:
        market, pick = o["key"]
        conn.execute(
            """INSERT INTO predictions(game_id, market, pick, sport, league, kickoff, line, odds, p_mkt, p_hist, n,
                                       p_model, shrink, recorded, p_raw)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(game_id, market, pick) DO UPDATE SET line=excluded.line, odds=excluded.odds,
                 p_mkt=excluded.p_mkt, p_hist=excluded.p_hist, n=excluded.n, p_model=excluded.p_model,
                 shrink=excluded.shrink, recorded=excluded.recorded, p_raw=excluded.p_raw
               WHERE predictions.result IS NULL""",
            (g["game_id"], market, pick, g["sport"], g["league"], parse_iso(g["kickoff"]).isoformat(),
             _line_for(m, market), o["odds"], o["p_mkt"], o.get("p_hist"), o.get("n", 0), o["p"], shrink, ts,
             o.get("p_raw", o["p_mkt"])))
        rows += 1
    return rows


def record_predictions(conn: sqlite3.Connection, games: list[dict]) -> int:
    """開賽前 RECORD_WINDOW_HOURS 小時內的比賽：記錄（或更新）每個選項的模型預測。"""
    from analyze import game_model
    ensure(conn)
    now, total = now_utc(), 0
    for g in games:
        if g["state"] != "pre" or not g["odds"]:
            continue
        if not 0 < (parse_iso(g["kickoff"]) - now).total_seconds() / 3600 <= RECORD_WINDOW_HOURS:
            continue
        try:
            m = game_model(g)
        except Exception:  # 單場模型失敗不影響其他場
            log.exception("記錄預測失敗 %s", g["game_id"])
            continue
        if m:
            total += record_model(conn, g, m)
    conn.commit()
    return total


def leg_profit(sport: str, market: str, pick: str, line: float | None, odds: float, hs: int, as_: int) -> float:
    """單一選項以 1 注計的損益（1x2 / ml / spread / total）。"""
    from strategies import settle_handicap, settle_total
    if market == "1x2":
        res = "H" if hs > as_ else "A" if hs < as_ else "D"
        return odds - 1 if pick == res else -1.0
    if market == "ml":
        if hs == as_:
            return 0.0
        return odds - 1 if (hs > as_) == (pick == "home") else -1.0
    if market == "total":
        return settle_total(hs + as_, line, pick, odds)
    margin = hs - as_ if pick == "home" else as_ - hs
    return settle_handicap(margin, line if pick == "home" else -line, odds)


def settle_predictions(conn: sqlite3.Connection) -> int:
    ensure(conn)
    rows = conn.execute("""SELECT p.*, g.status, g.home_score, g.away_score FROM predictions p
                           JOIN games g ON g.game_id = p.game_id WHERE p.result IS NULL""").fetchall()
    done = 0
    now = now_utc()
    for r in rows:
        if r["status"] == "final" and r["home_score"] is not None:
            pr = leg_profit(r["sport"], r["market"], r["pick"], r["line"], r["odds"], r["home_score"], r["away_score"])
            res = "W" if pr > 0 else "L" if pr < 0 else "P"
            conn.execute("UPDATE predictions SET result=?, profit=? WHERE game_id=? AND market=? AND pick=?",
                         (res, pr, r["game_id"], r["market"], r["pick"]))
            done += 1
        elif now - parse_iso(r["kickoff"]) > timedelta(days=3):
            conn.execute("UPDATE predictions SET result='void', profit=0 WHERE game_id=? AND market=? AND pick=?",
                         (r["game_id"], r["market"], r["pick"]))
    conn.commit()
    return done


def backfill(conn: sqlite3.Connection) -> int:
    """用資料庫裡的賽前快照，替已完賽、還沒有預測紀錄的比賽補記錄（第一次啟用時用）。"""
    from analyze import game_model
    from review import _prematch_odds
    ensure(conn)
    games = conn.execute("""SELECT * FROM games g WHERE status='final' AND sport IN ('football','mlb','nba')
                            AND NOT EXISTS (SELECT 1 FROM predictions p WHERE p.game_id = g.game_id)""").fetchall()
    n = 0
    for row in games:
        kick = parse_iso(row["kickoff"])
        odds, ts = _prematch_odds(conn, row["game_id"], kick)
        if not odds:
            continue
        g = {**dict(row), "odds": odds, "state": "post", "kickoff": row["kickoff"]}
        try:
            m = game_model(g)
        except Exception:
            log.exception("補記錄失敗 %s", row["game_id"])
            continue
        if m:
            record_model(conn, g, m, ts.isoformat(timespec="seconds"))
            n += 1
    conn.commit()
    return n


# -----------------------------------------------------------------------------
# 學習
# -----------------------------------------------------------------------------
def _blend(p_mkt: float, p_hist: float | None, n: int, s: float) -> float:
    if p_hist is None or not n or s >= MARKET_ONLY:
        return p_mkt
    return (p_mkt * s + p_hist * n) / (s + n)


def _logloss(rows, s: float) -> float:
    tot = 0.0
    for r in rows:
        p = min(0.98, max(0.02, _blend(r["p_mkt"], r["p_hist"], r["n"], s)))
        tot += -math.log(p if r["result"] == "W" else 1 - p)
    return tot / len(rows)


def settled_rows(conn: sqlite3.Connection, sport: str, limit_games: int = 600):
    ensure(conn)
    return conn.execute(
        """SELECT * FROM predictions WHERE sport=? AND result IN ('W','L') AND game_id IN (
               SELECT game_id FROM predictions WHERE sport=? AND result IN ('W','L')
               GROUP BY game_id ORDER BY MAX(kickoff) DESC LIMIT ?)""", (sport, sport, limit_games)).fetchall()


def update(conn: sqlite3.Connection) -> list[str]:
    """每天跑一次：實戰中「只看市場」與「加上歷史相似盤口」誰比較準（只顯示，不再自動調整）。"""
    lines = []
    for sport, zh in (("football", "足球"), ("mlb", "MLB"), ("nba", "NBA")):
        rows = settled_rows(conn, sport)
        games = len({r["game_id"] for r in rows})
        if games == 0:
            continue
        ll_mkt, ll_hist = _logloss(rows, MARKET_ONLY), _logloss(rows, COMPARE_SHRINK)
        lines.append(f"• {zh}：近 {games} 場｜log loss 只看市場 {ll_mkt:.4f} / 加上歷史相似盤口 {ll_hist:.4f}"
                     f"（{'市場較準' if ll_mkt <= ll_hist else '歷史加權較準'}；模型只用市場機率）")
    return lines


CAL_MIN_GAMES = 200      # 實戰至少這麼多場才判斷校準層是否該關掉
CAL_T = 2.0              # 校準比原始差、且 t 值超過這個才關


def strategy_uplift(s) -> float:
    """⭐ 策略的「實際勝率比市場機率高多少」：回測與實戰推播的結果按注數合併（10/28 使用者：實際勝率比模型高就多推）。
    實戰打得比回測好 → 加分變大、更常被推薦；打得差 → 自動變小。只合併通過回測的策略：
    研究（research/error_momentum.py）顯示「最近一直贏的類型」本身不會延續，只有回測過的規律才值得加碼。"""
    bt_n, bt_up = s.stats.get("n", 0), s.stats.get("uplift", 0.0)
    try:
        conn = db()
        try:   # 同一場同一邊可能先預警後訊號，只算一次
            rows = conn.execute("""SELECT MAX(result = 'W') w, AVG(p_mkt) p FROM alerts
                                   WHERE strategy = ? AND result IN ('W','L') AND p_mkt IS NOT NULL
                                   GROUP BY game_id, pick""", (s.sid,)).fetchall()
        finally:
            conn.close()
    except sqlite3.Error:
        return bt_up
    live = sum(r["w"] - r["p"] for r in rows)
    return (bt_n * bt_up + live) / (bt_n + len(rows)) if bt_n + len(rows) else 0.0


def check_calibration(conn: sqlite3.Connection) -> list[str]:
    """用實戰結果比較「校準後市場機率」與「原始等比例去水」：校準明顯變差就自動關閉該運動的校準。"""
    ensure(conn)
    out = []
    for sport, zh in (("football", "足球"), ("nba", "NBA")):
        rows = conn.execute("""SELECT game_id, p_mkt, p_raw, result FROM predictions
                               WHERE sport=? AND result IN ('W','L') AND p_raw IS NOT NULL
                                 AND abs(p_raw - p_mkt) > 1e-9""", (sport,)).fetchall()
        games = {}
        for r in rows:
            y = r["result"] == "W"
            pc = min(0.98, max(0.02, r["p_mkt"]))
            pr = min(0.98, max(0.02, r["p_raw"]))
            d = -math.log(pc if y else 1 - pc) + math.log(pr if y else 1 - pr)
            games[r["game_id"]] = games.get(r["game_id"], 0.0) + d
        n = len(games)
        if n < 20:
            continue
        diffs = list(games.values())
        mean = sum(diffs) / n
        sd = math.sqrt(sum((x - mean) ** 2 for x in diffs) / (n - 1)) or 1e-9
        t = mean / (sd / math.sqrt(n))
        off = kv_get(conn, f"calibration_off:{sport}") == "1"
        verdict = "校準較準" if mean < 0 else "原始較準"
        if n >= CAL_MIN_GAMES and t > CAL_T and not off:
            kv_set(conn, f"calibration_off:{sport}", "1")
            off = True
            log.info("學習：%s 校準在實戰明顯變差（t=%.2f），自動關閉", sport, t)
        out.append(f"• {zh}校準層：{n} 場，{verdict}（每場 log loss 差 {mean:+.4f}，t={t:+.2f}）"
                   + ("｜已自動關閉" if off else f"｜滿 {CAL_MIN_GAMES} 場且 t>{CAL_T:g} 才會關閉"))
    return out


def calibration(conn: sqlite3.Connection, sport: str | None = None) -> list[str]:
    """校準表：模型給的機率 vs 實際發生比例。"""
    ensure(conn)
    q = "SELECT p_model, result FROM predictions WHERE result IN ('W','L')" + (" AND sport=?" if sport else "")
    rows = conn.execute(q, (sport,) if sport else ()).fetchall()
    if len(rows) < 30:
        return []
    bins = [(0, 0.2), (0.2, 0.35), (0.35, 0.5), (0.5, 0.65), (0.65, 0.8), (0.8, 1.01)]
    out = []
    for lo, hi in bins:
        sel = [r for r in rows if lo <= r["p_model"] < hi]
        if len(sel) >= 10:
            avg = sum(r["p_model"] for r in sel) / len(sel)
            act = sum(r["result"] == "W" for r in sel) / len(sel)
            out.append(f"  模型說 {avg*100:.0f}% → 實際 {act*100:.0f}%（{len(sel)} 個選項）")
    return out


if __name__ == "__main__":
    c = db()
    print("補記錄", backfill(c), "場；結算", settle_predictions(c), "筆")
    print("\n".join(update(c)))
    print("\n".join(calibration(c)))
