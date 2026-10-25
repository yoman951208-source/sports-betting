# -*- coding: utf-8 -*-
"""串關建議（二串一、三串一）。

每一關都來自不同比賽（彼此獨立）：
  整張命中率 = 各關估計勝率相乘
  合理組合賠率 = 1 / 整張命中率      → 運彩組合賠率 ≥ 這個才有正期望值
  期望值（以 DK 價格）= Π(勝率 × 賠率) - 1

注意：串關會把每一關的抽水相乘，各關都是小負期望值時，關數越多整張虧越多。
"""
from __future__ import annotations

import json
import math
from datetime import timedelta
from html import escape
from itertools import combinations

import live
from analyze import game_model, game_title, in_focus, upcoming
from core import parse_iso, to_tpe

LEG_ODDS = (1.30, 4.5)     # 單關賠率範圍（太低沒意義、太高命中率太低）
LEG_MIN_EV = -0.05          # 單關期望值至少 -5%（以 DK 價格）
MAX_LEG_REUSE = 2           # 同一關在「所有」單子中最多出現幾次（10/15 檢討：重複的關一輸就連帶好幾張）
MAX_GAME_REUSE = 2          # 同一場比賽（不論選項）在所有單子中最多出現幾次
MIN_HIT = {2: 0.15, 3: 0.08}  # 整張命中率下限，避免變成彩券


PARLAY_SCHEMA = """
CREATE TABLE IF NOT EXISTS parlays (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created TEXT, label TEXT, legs TEXT,          -- legs: JSON [{game_id, sport, market, pick, line, odds, p, name, title, kickoff}]
    p REAL, fair REAL, odds REAL,
    result TEXT, profit REAL, settled TEXT
);
"""


def candidate_legs(games: list[dict], league: str | None, hours: float, sport: str | None = None) -> list[dict]:
    legs = []
    for g in upcoming(games, hours, league=league, sport=sport):
        if g["sport"] not in ("football", "mlb", "nba"):
            continue
        m = game_model(g)
        if not m:
            continue
        for o in m["options"]:
            if LEG_ODDS[0] <= o["odds"] <= LEG_ODDS[1] and o["ev"] >= LEG_MIN_EV and in_focus(m, o):
                legs.append({**o, "g": g, "m": m})
    return legs


def _ticket(legs: tuple) -> dict:
    p = math.prod(l["p"] for l in legs)
    odds = math.prod(l["odds"] for l in legs)
    return {"legs": legs, "p": p, "odds": odds, "fair": 1 / p, "ev": math.prod(l["p"] * l["odds"] for l in legs) - 1}


def build_tickets(legs: list[dict], size: int, used: dict[str, int] | None = None) -> list[tuple[str, dict]]:
    """used：跨呼叫共用的出現次數（關 id 與比賽 id），讓二串一、三串一加起來也不會集中在同一關。"""
    used = {} if used is None else used
    min_p = MIN_HIT[size]
    combos = [t for c in combinations(legs, size)
              if len({l["g"]["game_id"] for l in c}) == size and (t := _ticket(c))["p"] >= min_p]
    if not combos:
        return []
    band = (2.5, 5.0) if size == 2 else (4.5, 10.0)
    styles = [
        ("期望值最高", lambda t: t["ev"]),
        ("命中率最高", lambda t: t["p"] + (t["ev"] + 1) * 0.001),
        (f"平衡（組合賠率 {band[0]:g}~{band[1]:g}）", lambda t: t["ev"] if band[0] <= t["odds"] <= band[1] else -9),
    ]
    chosen = []
    for name, key in styles:
        for t in sorted(combos, key=key, reverse=True):
            ids = [f"{l['g']['game_id']}|{l['key']}" for l in t["legs"]]
            gids = [f"game|{l['g']['game_id']}" for l in t["legs"]]
            if any(used.get(i, 0) >= MAX_LEG_REUSE for i in ids) or any(t is c for _, c in chosen):
                continue
            if any(used.get(g, 0) >= MAX_GAME_REUSE for g in gids):
                continue
            if any(set(ids) == {f"{l['g']['game_id']}|{l['key']}" for l in c["legs"]} for _, c in chosen):
                continue
            if key(t) == -9:
                break
            for i in ids + gids:
                used[i] = used.get(i, 0) + 1
            chosen.append((name, t))
            break
    return chosen


def _leg_line(i: int, l: dict) -> str:
    g = l["g"]
    t = to_tpe(parse_iso(g["kickoff"]))
    return (f"{i}. {t:%m/%d %H:%M} {escape(game_title(g))}\n"
            f"   👉 <b>{escape(l['name'])}</b>｜估計 {l['p']*100:.0f}%｜合理 {l['fair']:.2f}（DK {l['odds']:.2f}）")


def ticket_text(label: str, t: dict) -> str:
    lines = [f"🎫 <b>{label}</b>"]
    lines += [_leg_line(i, l) for i, l in enumerate(t["legs"], 1)]
    lines.append(f"命中率 <b>{t['p']*100:.0f}%</b>｜合理組合賠率 <b>{t['fair']:.2f}</b>｜DK 組合 {t['odds']:.2f}（EV {t['ev']*100:+.1f}%）")
    lines.append(f"✅ 運彩組合賠率 ≥ <b>{t['fair']:.2f}</b> 才划算；每關的運彩賠率也可以逐關對照「合理」")
    return "\n".join(lines)


def _line_for(l: dict) -> float | None:
    m = l["m"]
    return m.get("L") if l["key"][0] == "total" else m.get("sp") if l["key"][0] == "spread" else None


def save_tickets(conn, tickets: list[tuple[str, dict]]) -> None:
    from core import now_utc
    conn.executescript(PARLAY_SCHEMA)
    ts = now_utc().isoformat(timespec="seconds")
    for label, t in tickets:
        legs = [{"game_id": l["g"]["game_id"], "sport": l["g"]["sport"], "market": l["key"][0], "pick": l["key"][1],
                 "line": _line_for(l), "odds": l["odds"], "p": l["p"], "name": l["name"], "title": game_title(l["g"]),
                 "kickoff": l["g"]["kickoff"]} for l in t["legs"]]
        conn.execute("INSERT INTO parlays(created, label, legs, p, fair, odds) VALUES (?,?,?,?,?,?)",
                     (ts, label, json.dumps(legs, ensure_ascii=False), t["p"], t["fair"], t["odds"]))
    conn.commit()


def settle_parlays(conn) -> int:
    """所有關都完賽才結算：每關係數 = 1 + 單關損益（贏=賠率、走盤=1、輸=0），整張損益 = Π 係數 - 1。"""
    from core import now_utc, parse_iso
    from learning import leg_profit
    conn.executescript(PARLAY_SCHEMA)
    done = 0
    for r in conn.execute("SELECT * FROM parlays WHERE result IS NULL").fetchall():
        legs = json.loads(r["legs"])
        factor, finished = 1.0, True
        for l in legs:
            g = conn.execute("SELECT status, home_score, away_score FROM games WHERE game_id=?", (l["game_id"],)).fetchone()
            if not g or g["status"] != "final" or g["home_score"] is None:
                if now_utc() - parse_iso(l["kickoff"]) > timedelta(days=3):
                    continue              # 延賽／取消：該關視為走盤
                finished = False
                break
            factor *= 1 + leg_profit(l["sport"], l["market"], l["pick"], l["line"], l["odds"], g["home_score"], g["away_score"])
        if not finished:
            continue
        profit = factor - 1
        conn.execute("UPDATE parlays SET result=?, profit=?, settled=? WHERE id=?",
                     ("W" if profit > 0 else "L" if profit < 0 else "P", profit, now_utc().isoformat(timespec="seconds"), r["id"]))
        done += 1
    conn.commit()
    return done


def record_lines(conn, days: int | None = None, label_prefix: str | None = None) -> list[str]:
    """串關實戰紀錄（以 DK 賠率、每張 1 注）。label_prefix 只看某一類（例如「最推薦｜單關」）。"""
    from core import now_utc
    conn.executescript(PARLAY_SCHEMA)
    since = (now_utc() - timedelta(days=days)).isoformat() if days else "0"
    q = "SELECT * FROM parlays WHERE result IN ('W','L','P') AND created >= ?"
    args: tuple = (since,)
    if label_prefix:
        q += " AND label LIKE ?"
        args += (label_prefix + "%",)
    rows = conn.execute(q, args).fetchall()
    if not rows:
        return ["（尚無已結算的紀錄）"]
    w = sum(r["result"] == "W" for r in rows)
    units = sum(r["profit"] for r in rows)
    exp = sum(r["p"] for r in rows)
    return [f"{len(rows)} 張｜中 {w} 張（模型預期約 {exp:.1f} 張）｜{units:+.2f} 注（DK 賠率，每張 1 注）"]


def report(games: list[dict], league: str | None = "歐冠", hours: float = 24, save_conn=None,
           sport: str | None = None) -> list[str]:
    legs = candidate_legs(games, league, hours, sport)
    title = league or ("足球" if sport == "football" else "全部（足球 + MLB + NBA）")
    n_games = len({l["g"]["game_id"] for l in legs})
    if n_games < 2:
        return [f"未來 {hours:g} 小時{title}可用的比賽不足 2 場，無法組串關。"]
    blocks, tickets = [], []
    used: dict[str, int] = {}
    for size, cn in ((2, "二串一"), (3, "三串一")):
        for name, t in build_tickets(legs, size, used):
            blocks.append(ticket_text(f"{cn}｜{name}", t))
            tickets.append((f"{cn}｜{name}", t))
    avg_leg_ev = sum(l["ev"] for l in legs) / len(legs)
    head = "\n".join([
        f"🎟️ <b>{escape(title)} 串關建議</b>（{n_games} 場可選，候選 {len(legs)} 個選項）",
        "",
        "⚠️ <b>先講清楚</b>：串關會把每一關的抽水<b>相乘</b>。",
        f"這批候選單關在 DK 的平均期望值約 {avg_leg_ev*100:+.1f}%，二串一約 {((1+avg_leg_ev)**2-1)*100:+.1f}%、"
        f"三串一約 {((1+avg_leg_ev)**3-1)*100:+.1f}%；運彩賠率通常更低。",
        "所以<b>只有運彩的組合賠率 ≥「合理組合賠率」</b>時才值得買，否則建議小注娛樂或不買。",
        "",
        "選法：每關來自不同比賽；「估計」= 市場去水機率 + 歷史相似盤口加權；",
        "期望值最高 = 每關最接近公平價；命中率最高 = 最穩；平衡 = 常見的運彩賠率區間。",
        f"分散風險：同一關最多出現在 {MAX_LEG_REUSE} 張、同一場比賽最多 {MAX_GAME_REUSE} 張，避免一關輸就全部輸。",
    ])
    if save_conn is not None and tickets:
        save_tickets(save_conn, tickets)
    return [head] + blocks
