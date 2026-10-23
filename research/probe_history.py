# -*- coding: utf-8 -*-
"""更多回測資料的可行性檢查：更早的季別，各家盤口資料齊不齊、可不可信。

足球：football-data.co.uk 各季 11 個聯賽的開盤 / 收盤欄位覆蓋率。
NBA / MLB：ESPN 每季抽 12 天、每天 4 場，看選到的博彩公司、收盤 / 開盤覆蓋率、跟其他家的盤口差多少。
"""
from __future__ import annotations

import pathlib
import sys
from datetime import date, timedelta

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import data_football  # noqa: E402
import data_mlb  # noqa: E402
import data_nba  # noqa: E402
from config import FOOTBALL_LEAGUES  # noqa: E402
from core import get_json, parse_line  # noqa: E402

OUT = pathlib.Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
lines: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True)
    lines.append(s)


# ----------------------------------------------------------------------------- 足球
COLS = {"AvgH": "平均開", "AvgCH": "平均收", "B365CH": "B365收", "PSCH": "Pin收", "MaxCH": "最高收", "BFEH": "交易所開",
        "BFECH": "交易所收", "AvgC>2.5": "大小收", "AHh": "亞盤開", "AHCh": "亞盤收", "AvgCAHH": "亞盤收賠率"}
say("# 足球（football-data.co.uk，11 個聯賽）")
for season in ["1617", "1718", "1819", "1920", "2021", "2122", "2223", "2324"]:
    total, cov = 0, dict.fromkeys(COLS, 0)
    for div in FOOTBALL_LEAGUES:
        raw = data_football._download(div, season, False)
        if raw is None or "HomeTeam" not in raw:
            continue
        raw = raw.dropna(subset=["HomeTeam"])
        total += len(raw)
        for c in COLS:
            cov[c] += int(raw[c].notna().sum()) if c in raw else 0
    say(f"- {season}: {total} 場｜" + "｜".join(f"{v} {cov[c] / max(total, 1) * 100:.0f}%" for c, v in COLS.items()))


# ----------------------------------------------------------------------------- NBA / MLB
def probe(sport: str, mod, seasons: list[tuple[str, str, str]]) -> None:
    say(f"\n# {sport}（ESPN）")
    for season, s, e in seasons:
        d0, d1 = date.fromisoformat(s), date.fromisoformat(e)
        days = [d0 + timedelta(days=int(i)) for i in np.linspace(10, (d1 - d0).days - 40, 12)]
        rows = []
        for d in days:
            sb = mod._scoreboard(d)
            for ev in (sb or {}).get("events", [])[:4]:
                raw = get_json(mod.CORE_ODDS.format(eid=ev["id"]), cache_name=f"{sport.lower()}/odds_{ev['id']}.json")
                o = mod.parse_core_odds(raw)
                items = [it for it in (raw or {}).get("items", []) if "live" not in it.get("provider", {}).get("name", "").lower()]
                spreads = []
                for it in items:
                    h = it.get("homeTeamOdds") or {}
                    blk = h.get("close") or h.get("current") or {}
                    v = parse_line((blk.get("pointSpread") or {}).get("american") or (blk.get("pointSpread") or {}).get("alternateDisplayValue"))
                    if v is not None:
                        spreads.append(v)
                med = float(np.median(spreads)) if spreads else None
                rows.append({**o, "n_prov": len(items), "spread_med": med})
        df = pd.DataFrame(rows)
        if df.empty:
            say(f"- {season}: 抓不到比賽")
            continue
        pct = lambda c: f"{df[c].notna().mean() * 100:.0f}%" if c in df else "0%"
        dev = (df["c_spread_home"] - df["spread_med"]).abs() if "c_spread_home" in df else pd.Series(dtype=float)
        provs = df["provider"].value_counts().head(3).to_dict() if "provider" in df else {}
        say(f"- {season}: 抽樣 {len(df)} 場｜收盤讓分 {pct('c_spread_home')}、不讓分 {pct('c_ml_home')}、大小 {pct('c_total')}｜"
            f"開盤讓分 {pct('o_spread_home')}｜選到的公司 {provs}｜跟各家中位數差 >2 分 {int((dev > 2).sum())} 場")


probe("NBA", data_nba, [("2016-17", "2016-10-25", "2017-06-12"), ("2017-18", "2017-10-17", "2018-06-08"),
                        ("2018-19", "2018-10-16", "2019-06-13"), ("2019-20", "2019-10-22", "2020-10-11"),
                        ("2020-21", "2020-12-22", "2021-07-20"), ("2021-22", "2021-10-19", "2022-06-16"),
                        ("2022-23", "2022-10-18", "2023-06-12"), ("2023-24", "2023-10-24", "2024-06-17")])
probe("MLB", data_mlb, [("2019", "2019-03-28", "2019-10-30"), ("2021", "2021-04-01", "2021-11-02"),
                        ("2022", "2022-04-07", "2022-11-05"), ("2023", "2023-03-30", "2023-11-01")])

(OUT / "probe_history.md").write_text("\n".join(lines), encoding="utf-8")
