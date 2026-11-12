# -*- coding: utf-8 -*-
"""主力缺陣（11/12）：沒進比賽名單的「常規先發」人數，能不能預測讓分盤收盤線沒算到的部分？

ESPN 賽後名單 → 每隊近 10 場先發 ≥50% 的球員 = 常規先發；缺席 = 不在這場 20 人名單內。
檢定：實際淨勝球 + 收盤亞盤線（殘差）對 (主隊缺席 − 客隊缺席) 迴歸。係數 < 0 = 主隊缺人時，盤口仍高估主隊。
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402

import data_football  # noqa: E402

B = "https://site.api.espn.com/apis/site/v2/sports/soccer"
CACHE = pathlib.Path(__file__).parent / "out" / "absence_cache.json"
CACHE.parent.mkdir(exist_ok=True)
LEAGUES = {"E0": "eng.1", "SP1": "esp.1", "I1": "ita.1", "D1": "ger.1", "F1": "fra.1"}
cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}


def get(url):
    return requests.get(url, timeout=30).json()


def lineups(lg, eid):
    k = f"{lg}:{eid}"
    if k not in cache:
        d = get(f"{B}/{lg}/summary?event={eid}")
        cache[k] = [{"team": r["team"]["displayName"], "home": r.get("homeAway") == "home",
                     "ids": [p["athlete"]["id"] for p in r.get("roster", [])],
                     "start": [p["athlete"]["id"] for p in r.get("roster", []) if p.get("starter")]}
                    for r in d.get("rosters", [])]
    return cache[k]


def main():
    df = data_football.load_history()
    df = df[df["div"].isin(LEAGUES) & df["c_ah_line"].notna()]
    rows = []
    for div, lg in LEAGUES.items():
        d = df[df["div"] == div]
        evs = {}
        for day in sorted(d["date"].dt.normalize().unique()):   # ESPN 的日期區間查詢不可用 → 逐個比賽日查
            for ev in get(f"{B}/{lg}/scoreboard?dates={pd.Timestamp(day):%Y%m%d}").get("events", []):
                c = ev["competitions"][0]["competitors"]
                try:
                    sc = {x["homeAway"]: int(x["score"]) for x in c}
                except (KeyError, ValueError):
                    continue
                evs[ev["id"]] = (pd.Timestamp(ev["date"]).tz_localize(None).normalize(), sc["home"], sc["away"])
        hist: dict[str, list] = {}
        for eid, (day, hg, ag) in sorted(evs.items(), key=lambda x: x[1][0]):
            m = d[((d["date"].dt.normalize() == day)) & (d["hg"] == hg) & (d["ag"] == ag)]
            if len(m) != 1:
                continue
            ln = lineups(lg, eid)
            if len(ln) != 2:
                continue
            ab = {}
            for t in ln:
                past = hist.get(t["team"], [])[-10:]
                if len(past) >= 5:
                    cnt: dict[str, int] = {}
                    for p in past:
                        for i in p["start"]:
                            cnt[i] = cnt.get(i, 0) + 1
                    regs = {i for i, n in cnt.items() if n >= len(past) / 2}
                    ab[t["home"]] = len(regs - set(t["ids"]))
                hist.setdefault(t["team"], []).append(t)
            if True in ab and False in ab:
                r = m.iloc[0]
                rows.append({"div": div, "season": r["season"], "dabs": ab[True] - ab[False], "habs": ab[True], "aabs": ab[False],
                             "resid": (hg - ag) + r["c_ah_line"], "goals": hg + ag})
        CACHE.write_text(json.dumps(cache))
        print(div, len(rows), flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(CACHE.parent / "absence_rows.csv", index=False)
    x, y = r["dabs"].values.astype(float), r["resid"].values
    b = np.polyfit(x, y, 1)
    se = np.sqrt(np.var(y - np.polyval(b, x)) / (len(x) * np.var(x)))
    print(f"樣本 {len(r)}；殘差 = {b[1]:+.3f} {b[0]:+.3f} × (主缺−客缺)；t = {b[0]/se:+.2f}")
    print("缺席差分組：", r.groupby(r["dabs"].clip(-2, 2))["resid"].agg(["count", "mean"]).round(3).to_string())
    print("總進球 vs 雙方缺席數：", np.corrcoef(r["habs"] + r["aabs"], r["goals"])[0, 1].round(3))


if __name__ == "__main__":
    main()
