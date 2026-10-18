# -*- coding: utf-8 -*-
"""NBA 球員逐場數據（ESPN box score）。

原始 summary 每場約 370KB，所以只存精簡版（每場約 3KB）：data/cache/nba_box/<event>.json
欄位：球員 id、名字、球隊、先發、是否沒上場（與原因）、上場分鐘、得分、籃板、助攻、正負值。
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from core import CACHE_DIR, http_get, log

SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"
BOX_DIR = CACHE_DIR / "nba_box"


def _num(v) -> float | None:
    try:
        return float(str(v).replace("+", ""))
    except (TypeError, ValueError):
        return None


def parse_box(summary: dict) -> list[dict]:
    out = []
    for team in (summary.get("boxscore") or {}).get("players", []):
        abbr = team.get("team", {}).get("abbreviation", "")
        for st in team.get("statistics", [])[:1]:
            names = st.get("names") or st.get("labels") or []
            idx = {n: i for i, n in enumerate(names)}
            for a in st.get("athletes", []):
                stats = a.get("stats") or []
                get = lambda k: _num(stats[idx[k]]) if k in idx and idx[k] < len(stats) else None
                out.append({
                    "pid": str(a.get("athlete", {}).get("id", "")), "name": a.get("athlete", {}).get("displayName", ""),
                    "team": abbr, "starter": bool(a.get("starter")), "dnp": bool(a.get("didNotPlay")) or not stats,
                    "reason": a.get("reason", ""), "min": get("MIN"), "pts": get("PTS"), "reb": get("REB"),
                    "ast": get("AST"), "pm": get("+/-"),
                })
    return out


def fetch_box(eid: str, force: bool = False) -> list[dict] | None:
    """抓一場的精簡 box score（已完賽的才存檔）。"""
    path = BOX_DIR / f"{eid}.json"
    if path.exists() and not force:
        return json.loads(path.read_text(encoding="utf-8"))
    resp = http_get(SUMMARY, params={"event": eid})
    if resp is None:
        return None
    try:
        data = resp.json()
    except ValueError:
        return None
    done = (((data.get("header") or {}).get("competitions") or [{}])[0].get("status") or {}).get("type", {}).get("completed")
    rows = parse_box(data)
    if rows and done:
        BOX_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return rows


def load_boxes(event_ids: list[str]) -> pd.DataFrame:
    """回傳所有指定比賽的球員數據（缺的會補抓）。"""
    missing = [e for e in event_ids if not (BOX_DIR / f"{e}.json").exists()]
    if missing:
        # 每場原始資料約 370KB；10/19 一次 8 執行緒抓 2,600 場被 ESPN 暫時封鎖（403），所以放慢
        log.info("NBA box score：補抓 %d 場…", len(missing))

        def slow(eid):
            time.sleep(0.4)
            return fetch_box(eid)

        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(slow, missing))
    rows = []
    for e in event_ids:
        path = BOX_DIR / f"{e}.json"
        if path.exists():
            for r in json.loads(path.read_text(encoding="utf-8")):
                rows.append({"event_id": e, **r})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import data_nba
    games = data_nba.load_history()
    box = load_boxes(games["event_id"].astype(str).tolist())
    print(len(box), "player-games;", box["event_id"].nunique(), "games;", int(box["dnp"].sum()), "DNP rows")
    print(box.head())
