# -*- coding: utf-8 -*-
"""足球歷史資料：football-data.co.uk 的賽果 + 多家博彩公司開盤/收盤賠率。

欄位說明（https://www.football-data.co.uk/notes.txt）：
  Avg*  = 市場平均（多家博彩公司）  Max* = 最高賠率   BFE* = Betfair 交易所（尖銳盤參考）
  *C*   = 收盤賠率；沒有 C 的是「開盤」（週末賽事於週五下午、週中賽事於週二下午擷取）
  AHh / AHCh = 亞洲讓球（主隊）開盤 / 收盤盤口
"""
from __future__ import annotations

import io
import time

import pandas as pd

from config import FOOTBALL_LEAGUES, FOOTBALL_SEASONS
from core import CACHE_DIR, http_get, log

URL = "https://www.football-data.co.uk/mmz4281/{season}/{div}.csv"

# 原始欄位 → 統一欄位名稱
COLMAP = {
    "AvgH": "o_avg_h", "AvgD": "o_avg_d", "AvgA": "o_avg_a",
    "AvgCH": "c_avg_h", "AvgCD": "c_avg_d", "AvgCA": "c_avg_a",
    "MaxH": "o_max_h", "MaxD": "o_max_d", "MaxA": "o_max_a",
    "MaxCH": "c_max_h", "MaxCD": "c_max_d", "MaxCA": "c_max_a",
    "BFEH": "o_bfe_h", "BFED": "o_bfe_d", "BFEA": "o_bfe_a",
    "BFECH": "c_bfe_h", "BFECD": "c_bfe_d", "BFECA": "c_bfe_a",
    "B365H": "o_b365_h", "B365D": "o_b365_d", "B365A": "o_b365_a",
    "B365CH": "c_b365_h", "B365CD": "c_b365_d", "B365CA": "c_b365_a",
    "Avg>2.5": "o_avg_over", "Avg<2.5": "o_avg_under",
    "AvgC>2.5": "c_avg_over", "AvgC<2.5": "c_avg_under",
    "AHh": "o_ah_line", "AvgAHH": "o_ah_h", "AvgAHA": "o_ah_a",
    "AHCh": "c_ah_line", "AvgCAHH": "c_ah_h", "AvgCAHA": "c_ah_a",
}


def _season_label(code: str) -> str:
    return f"20{code[:2]}-{code[2:]}"


def _download(div: str, season: str, current: bool) -> pd.DataFrame | None:
    path = CACHE_DIR / "football" / f"{season}_{div}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    stale = current and (not path.exists() or time.time() - path.stat().st_mtime > 12 * 3600)
    if not path.exists() or stale:
        resp = http_get(URL.format(season=season, div=div))
        if resp is None:
            if not path.exists():
                log.warning("下載失敗：%s %s", div, season)
                return None
        else:
            path.write_bytes(resp.content)
    try:
        return pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip")
    except (UnicodeDecodeError, pd.errors.ParserError):
        return pd.read_csv(path, encoding="latin-1", on_bad_lines="skip")


def load_history() -> pd.DataFrame:
    """回傳所有聯賽、所有季別的已完賽比賽（統一欄位）。"""
    frames = []
    for season in FOOTBALL_SEASONS:
        current = season == FOOTBALL_SEASONS[-1]
        for div, (_espn, name) in FOOTBALL_LEAGUES.items():
            raw = _download(div, season, current)
            if raw is None or raw.empty or "FTR" not in raw.columns:
                continue
            raw = raw.dropna(subset=["HomeTeam", "AwayTeam", "FTR"])
            df = pd.DataFrame({
                "div": div, "league": name, "season": _season_label(season),
                "date": pd.to_datetime(raw["Date"], dayfirst=True, errors="coerce"),
                "home": raw["HomeTeam"], "away": raw["AwayTeam"],
                "hg": pd.to_numeric(raw["FTHG"], errors="coerce"),
                "ag": pd.to_numeric(raw["FTAG"], errors="coerce"),
                "ftr": raw["FTR"],
            })
            for src, dst in COLMAP.items():
                df[dst] = pd.to_numeric(raw[src], errors="coerce") if src in raw.columns else float("nan")
            frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["date", "hg", "ag"]).sort_values("date").reset_index(drop=True)
    log.info("足球歷史資料：%d 場（%s ~ %s）", len(out), out["date"].min().date(), out["date"].max().date())
    return out


if __name__ == "__main__":
    d = load_history()
    print(d.groupby(["season", "league"]).size().unstack(0))
