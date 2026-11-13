# -*- coding: utf-8 -*-
"""情境因子批次回測（11/13）：會影響盤口、或莊家可能故意偏移的情境，收盤盤口有沒有算夠？

做法：每個因子決定「買哪一邊」，以收盤賠率結算。前面幾季（train）找規律、最後幾季（test）驗證——
一次測幾十個因子一定會有幾個純靠運氣看起來賺錢，所以只有 train、test 都成立的才算數。
足球：亞洲讓球 + 大小 2.5（市場平均收盤）。NBA：讓分 + 大小分（DK 收盤）。
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import data_football  # noqa: E402
import data_nba  # noqa: E402
from strategies import no_vig3, settle_handicap, settle_total  # noqa: E402

OUT = pathlib.Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
LINES: list[str] = []


def say(s=""):
    print(s, flush=True)
    LINES.append(s)


def team_features(df: pd.DataFrame, big_margin: float, over_col: str) -> pd.DataFrame:
    """每隊賽前的近況（只用這場之前的比賽）。df 需要：gid, season, date, home, away, margin(主-客), cover_h(主隊過盤 1/0/-1), over_col。"""
    parts = []
    for side, sign in (("home", 1), ("away", -1)):
        parts.append(pd.DataFrame({"gid": df["gid"], "season": df["season"], "date": df["date"], "team": df[side], "side": side,
                                   "m": sign * df["margin"], "cover": sign * df["cover_h"], "over": df[over_col],
                                   "away_game": side == "away"}))
    lg = pd.concat(parts).sort_values(["date", "gid"]).reset_index(drop=True)
    g = lg.groupby(["season", "team"], sort=False)
    lg["n"] = g.cumcount()                                    # 本季已踢場數
    lg["rest"] = g["date"].diff().dt.days
    lg["prev_m"] = g["m"].shift()
    lg["d3"] = g["date"].diff(2).dt.days                      # 這場與前兩場的間隔（≤4 = 四天三戰）

    # streak 要在同隊同季內連續 → 先依隊排序再算
    lg = lg.sort_values(["season", "team", "date", "gid"]).reset_index(drop=True)
    for name, cond in (("wstreak", lg["m"] > 0), ("lstreak", lg["m"] < 0), ("cstreak", lg["cover"] > 0),
                       ("ncstreak", lg["cover"] < 0), ("roadstreak", lg["away_game"])):
        key = lg["season"].astype(str) + "|" + lg["team"].astype(str)
        c = cond.astype(int)
        blk = ((c == 0) | (key != key.shift())).cumsum()
        run = c.groupby(blk).cumsum()
        lg[name] = run.groupby(key).shift().fillna(0)
    lg["over3"] = lg.groupby(["season", "team"])["over"].transform(lambda s: s.shift().rolling(3).sum())
    cols = ["n", "rest", "prev_m", "d3", "wstreak", "lstreak", "cstreak", "ncstreak", "roadstreak", "over3"]
    out = df.copy()
    for side, p in (("home", "h_"), ("away", "a_")):
        f = lg[lg["side"] == side].set_index("gid")[cols].add_prefix(p)
        out = out.join(f, on="gid")
    return out


def one_side(h: pd.Series, a: pd.Series, back: bool) -> np.ndarray:
    """只有一隊符合條件時下注：back=True 買符合的那隊，False 買對手。回傳 +1 主 / -1 客 / 0。"""
    s = np.where(h & ~a, 1, np.where(a & ~h, -1, 0))
    return s if back else -s


def evaluate(df: pd.DataFrame, sigs: dict[str, np.ndarray], pos: str, neg: str, test_seasons: list[str], title: str) -> list[dict]:
    """sigs: +1 → 買 pos 那邊（損益欄 pos）、-1 → 買 neg 那邊。"""
    say(f"\n## {title}")
    say(f"{'因子':<34}{'場數':>6}{'ROI':>8}{'t':>6} ｜{'train n':>8}{'ROI':>8}{'t':>6} ｜{'test n':>7}{'ROI':>8}{'t':>6}")
    is_test = df["season"].isin(test_seasons).values
    rows = []
    for name, s in sigs.items():
        s = np.asarray(s)
        pnl = np.where(s > 0, df[pos].values, np.where(s < 0, df[neg].values, np.nan))
        ok = ~np.isnan(pnl)

        def stat(mask):
            x = pnl[mask & ok]
            if len(x) < 2 or x.std() == 0:
                return len(x), float("nan"), float("nan")
            return len(x), x.mean(), x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
        (n, r, t), (n1, r1, t1), (n2, r2, t2) = stat(np.ones(len(df), bool)), stat(~is_test), stat(is_test)
        flag = " ★" if n1 >= 100 and n2 >= 50 and t1 >= 1.64 and r2 > 0 else ""
        say(f"{name:<34}{n:>6}{r*100:>+7.1f}%{t:>+6.2f} ｜{n1:>8}{r1*100:>+7.1f}%{t1:>+6.2f} ｜{n2:>7}{r2*100:>+7.1f}%{t2:>+6.2f}{flag}")
        rows.append({"market": title, "signal": name, "n": n, "roi": r, "t": t, "n_train": n1, "roi_train": r1, "t_train": t1,
                     "n_test": n2, "roi_test": r2, "t_test": t2, "star": bool(flag)})
    return rows


# ============================================================ 足球
def football() -> list[dict]:
    df = data_football.load_history()
    df = df.dropna(subset=["c_ah_line", "c_ah_h", "c_ah_a"]).reset_index(drop=True)
    df["gid"] = df.index
    df["margin"] = df["hg"] - df["ag"]
    df["goals"] = df["hg"] + df["ag"]
    df["pnl_h"] = [settle_handicap(m, l, o) for m, l, o in zip(df["margin"], df["c_ah_line"], df["c_ah_h"])]
    df["pnl_a"] = [settle_handicap(-m, -l, o) for m, l, o in zip(df["margin"], df["c_ah_line"], df["c_ah_a"])]
    df["cover_h"] = np.sign(df["pnl_h"])
    df["over"] = (df["goals"] > 2.5).astype(float)
    has_ou = df["c_avg_over"].notna() & df["c_avg_under"].notna()
    df["pnl_o"] = [settle_total(g, 2.5, "over", o) if k else np.nan for g, o, k in zip(df["goals"], df["c_avg_over"], has_ou)]
    df["pnl_u"] = [settle_total(g, 2.5, "under", o) if k else np.nan for g, o, k in zip(df["goals"], df["c_avg_under"], has_ou)]
    df = team_features(df, 3, "over")

    # 上一季積分排名（同聯賽）：前 3 = 人氣球隊；上一季不在這個聯賽 = 升班馬
    pts = pd.concat([
        pd.DataFrame({"div": df["div"], "season": df["season"], "team": df["home"], "p": np.where(df["margin"] > 0, 3, np.where(df["margin"] == 0, 1, 0))}),
        pd.DataFrame({"div": df["div"], "season": df["season"], "team": df["away"], "p": np.where(df["margin"] < 0, 3, np.where(df["margin"] == 0, 1, 0))}),
    ]).groupby(["div", "season", "team"])["p"].sum().reset_index()
    pts["rank"] = pts.groupby(["div", "season"])["p"].rank(ascending=False, method="min")
    seasons = sorted(df["season"].unique())
    nxt = {s: seasons[i + 1] for i, s in enumerate(seasons[:-1])}
    pts["season"] = pts["season"].map(nxt)
    prev_rank = pts.dropna(subset=["season"]).set_index(["div", "season", "team"])["rank"]
    for side, p in (("home", "h_"), ("away", "a_")):
        df[p + "prank"] = [prev_rank.get((d, s, t), np.nan) for d, s, t in zip(df["div"], df["season"], df[side])]
    has_prev = df["season"] != seasons[0]
    h_big, a_big = (df["h_prank"] <= 3), (df["a_prank"] <= 3)
    h_new, a_new = has_prev & df["h_prank"].isna(), has_prev & df["a_prank"].isna()

    line = df["c_ah_line"]
    dog = np.where(line > 0, 1, np.where(line < 0, -1, 0))        # 受讓那一邊
    rd = (df["h_rest"] - df["a_rest"]).where((df["h_rest"] <= 14) & (df["a_rest"] <= 14))
    midweek = df["date"].dt.dayofweek.isin([1, 2, 3]).values
    early = ((df["h_n"] < 4) & (df["a_n"] < 4)).values
    late = ((df["h_n"] >= 30) & (df["a_n"] >= 30)).values

    shade = np.zeros(len(df))                                    # 軟莊家（B365）比交易所更壓低哪一邊 = 散戶買哪邊
    for i, r in enumerate(df[["c_b365_h", "c_b365_d", "c_b365_a", "c_bfe_h", "c_bfe_d", "c_bfe_a"]].itertuples(index=False)):
        if any(pd.isna(x) for x in r):
            continue
        pb, pe = no_vig3(*r[:3]), no_vig3(*r[3:])
        if pb and pe:
            shade[i] = (pb[0] - pe[0]) - (pb[2] - pe[2])         # >0：B365 相對壓低主隊賠率

    sp = {
        "基準：全買主隊": np.ones(len(df)), "基準：全買客隊": -np.ones(len(df)), "基準：全買受讓方": dog, "基準：全買讓球方": -dog,
        "主場受讓": np.where(line > 0, 1, 0), "客場受讓": np.where(line < 0, -1, 0),
        "休息多 3 天以上（買休息多）": np.where(rd >= 3, 1, np.where(rd <= -3, -1, 0)),
        "休息少（≤3天）對手正常（買休息少）": one_side(df["h_rest"] <= 3, df["a_rest"] <= 3, True),
        "上一場大敗 3 球+（買反彈）": one_side(df["h_prev_m"] <= -3, df["a_prev_m"] <= -3, True),
        "上一場大勝 3 球+（反買）": one_side(df["h_prev_m"] >= 3, df["a_prev_m"] >= 3, False),
        "連勝 4+（反買）": one_side(df["h_wstreak"] >= 4, df["a_wstreak"] >= 4, False),
        "連敗 4+（買）": one_side(df["h_lstreak"] >= 4, df["a_lstreak"] >= 4, True),
        "連過盤 3+（反買）": one_side(df["h_cstreak"] >= 3, df["a_cstreak"] >= 3, False),
        "連不過盤 3+（買）": one_side(df["h_ncstreak"] >= 3, df["a_ncstreak"] >= 3, True),
        "人氣球隊（上季前3）反買": one_side(h_big, a_big, False),
        "人氣球隊作客（反買）": np.where(a_big & ~h_big, 1, 0),
        "升班馬（買）": one_side(h_new, a_new, True),
        "升班馬前 10 場（買）": one_side(h_new & (df["h_n"] < 10), a_new & (df["a_n"] < 10), True),
        "季初前 4 場：買受讓方": np.where(early, dog, 0), "季末（30 場後）：買受讓方": np.where(late, dog, 0),
        "週中比賽：買受讓方": np.where(midweek, dog, 0),
        "軟莊壓低的那邊（跟散戶）": np.where(shade >= 0.02, 1, np.where(shade <= -0.02, -1, 0)),
        "軟莊壓低的另一邊（反散戶）": np.where(shade >= 0.02, -1, np.where(shade <= -0.02, 1, 0)),
        "賠率較高的那邊（≥2.00）": np.where(df["c_ah_h"] >= 2.0, 1, np.where(df["c_ah_a"] >= 2.0, -1, 0)),
    }
    o3 = df["h_over3"] + df["a_over3"]
    deep = line.abs() >= 1.5
    tot = {
        "基準：全買大": np.ones(len(df)), "基準：全買小": -np.ones(len(df)),
        "兩隊近 3 場多為大分（≥5/6）→ 買小": np.where(o3 >= 5, -1, 0), "兩隊近 3 場多為大分 → 買大": np.where(o3 >= 5, 1, 0),
        "兩隊近 3 場多為小分（≤1/6）→ 買大": np.where(o3 <= 1, 1, 0), "兩隊近 3 場多為小分 → 買小": np.where(o3 <= 1, -1, 0),
        "深盤（|讓球|≥1.5）→ 買大": np.where(deep, 1, 0), "深盤 → 買小": np.where(deep, -1, 0),
        "平手盤（|讓球|≤0.25）→ 買小": np.where(line.abs() <= 0.25, -1, 0),
        "週中 → 買小": np.where(midweek, -1, 0), "季初前 4 場 → 買大": np.where(early, 1, 0), "季末 → 買大": np.where(late, 1, 0),
        "雙方都休息少（≤3天）→ 買小": np.where((df["h_rest"] <= 3) & (df["a_rest"] <= 3), -1, 0),
        "賠率較高的那邊（≥2.00）": np.where(df["c_avg_over"] >= 2.0, 1, np.where(df["c_avg_under"] >= 2.0, -1, 0)),
    }
    test = seasons[-3:]
    say(f"\n# 足球：{len(df)} 場，train {seasons[:-3]}｜test {test}")
    return (evaluate(df, sp, "pnl_h", "pnl_a", test, "足球 亞洲讓球（市場平均收盤）")
            + evaluate(df, tot, "pnl_o", "pnl_u", test, "足球 大小 2.5（市場平均收盤）"))


# ============================================================ NBA
def nba() -> list[dict]:
    df = data_nba.load_history()
    df = df[~df["playoffs"]].dropna(subset=["c_spread_home", "c_spread_home_odds", "c_spread_away_odds", "home_score"]).reset_index(drop=True)
    df["gid"] = df.index
    df["date"] = pd.to_datetime(df["game_date"])
    df["margin"] = df["home_score"] - df["away_score"]
    df["pts"] = df["home_score"] + df["away_score"]
    df["pnl_h"] = [settle_handicap(m, l, o) for m, l, o in zip(df["margin"], df["c_spread_home"], df["c_spread_home_odds"])]
    df["pnl_a"] = [settle_handicap(-m, -l, o) for m, l, o in zip(df["margin"], df["c_spread_home"], df["c_spread_away_odds"])]
    df["cover_h"] = np.sign(df["pnl_h"])
    has_t = df["c_total"].notna() & df["c_over_odds"].notna() & df["c_under_odds"].notna()
    df["over"] = np.where(has_t, (df["pts"] > df["c_total"]).astype(float), np.nan)
    df["pnl_o"] = [settle_total(p, l, "over", o) if k else np.nan for p, l, o, k in zip(df["pts"], df["c_total"], df["c_over_odds"], has_t)]
    df["pnl_u"] = [settle_total(p, l, "under", o) if k else np.nan for p, l, o, k in zip(df["pts"], df["c_total"], df["c_under_odds"], has_t)]
    df = team_features(df, 20, "over")

    line = df["c_spread_home"]
    dog = np.where(line > 0, 1, np.where(line < 0, -1, 0))
    rd = df["h_rest"] - df["a_rest"]
    h_b2b, a_b2b = df["h_rest"] == 1, df["a_rest"] == 1
    h_34, a_34 = df["h_d3"] <= 4, df["a_d3"] <= 4
    early = ((df["h_n"] < 8) & (df["a_n"] < 8)).values
    alt = df["home"].isin(["DEN", "UTAH"])
    sp = {
        "基準：全買主隊": np.ones(len(df)), "基準：全買客隊": -np.ones(len(df)), "基準：全買受讓方": dog, "基準：全買讓分方": -dog,
        "主場受讓": np.where(line > 0, 1, 0), "客場受讓": np.where(line < 0, -1, 0),
        "大熱門（讓 10+）反買": np.where(line.abs() >= 10, dog, 0), "大熱門（讓 10+）買": np.where(line.abs() >= 10, -dog, 0),
        "小盤（|讓|≤3）買主": np.where(line.abs() <= 3, 1, 0),
        "休息多 2 天以上（買）": np.where(rd >= 2, 1, np.where(rd <= -2, -1, 0)),
        "背靠背 vs 對手沒有（買背靠背）": one_side(h_b2b, a_b2b, True),
        "客隊背靠背、主隊沒有（買主）": np.where(a_b2b & ~h_b2b, 1, 0),
        "四天三戰 vs 對手沒有（反買）": one_side(h_34, a_34, False),
        "高原主場（DEN/UTAH）＋客隊背靠背（買主）": np.where(alt & a_b2b, 1, 0),
        "上一場大敗 20+（買反彈）": one_side(df["h_prev_m"] <= -20, df["a_prev_m"] <= -20, True),
        "上一場大勝 20+（反買）": one_side(df["h_prev_m"] >= 20, df["a_prev_m"] >= 20, False),
        "連勝 5+（反買）": one_side(df["h_wstreak"] >= 5, df["a_wstreak"] >= 5, False),
        "連敗 5+（買）": one_side(df["h_lstreak"] >= 5, df["a_lstreak"] >= 5, True),
        "連過盤 4+（反買）": one_side(df["h_cstreak"] >= 4, df["a_cstreak"] >= 4, False),
        "連不過盤 4+（買）": one_side(df["h_ncstreak"] >= 4, df["a_ncstreak"] >= 4, True),
        "連續客場第 4 場+（反買客隊）": np.where(df["a_roadstreak"] >= 3, 1, 0),
        "連續客場第 4 場+（買客隊）": np.where(df["a_roadstreak"] >= 3, -1, 0),
        "季初前 8 場：買受讓方": np.where(early, dog, 0), "季初前 8 場：買主隊": np.where(early, 1, 0),
        "賠率較高的那邊（≥1.95）": np.where(df["c_spread_home_odds"] >= 1.95, 1, np.where(df["c_spread_away_odds"] >= 1.95, -1, 0)),
    }
    o3 = df["h_over3"] + df["a_over3"]
    rel = df["total_rel"]
    tot = {
        "基準：全買大": np.ones(len(df)), "基準：全買小": -np.ones(len(df)),
        "兩隊近 3 場多為大分（≥5/6）→ 買小": np.where(o3 >= 5, -1, 0), "兩隊近 3 場多為小分（≤1/6）→ 買大": np.where(o3 <= 1, 1, 0),
        "大小分比聯盟平均高 8+ → 買小": np.where(rel >= 8, -1, 0), "大小分比聯盟平均低 8+ → 買大": np.where(rel <= -8, 1, 0),
        "雙方都背靠背 → 買小": np.where(h_b2b & a_b2b, -1, 0), "任一隊背靠背 → 買小": np.where(h_b2b | a_b2b, -1, 0),
        "任一隊背靠背 → 買大": np.where(h_b2b | a_b2b, 1, 0),
        "雙方都休息 3 天+ → 買小": np.where((df["h_rest"] >= 3) & (df["a_rest"] >= 3), -1, 0),
        "大熱門（讓 10+）→ 買小": np.where(line.abs() >= 10, -1, 0), "大熱門（讓 10+）→ 買大": np.where(line.abs() >= 10, 1, 0),
        "小盤（|讓|≤3）→ 買小": np.where(line.abs() <= 3, -1, 0),
        "季初前 8 場 → 買大": np.where(early, 1, 0), "季初前 8 場 → 買小": np.where(early, -1, 0),
        "賠率較高的那邊（≥1.95）": np.where(df["c_over_odds"] >= 1.95, 1, np.where(df["c_under_odds"] >= 1.95, -1, 0)),
    }
    seasons = sorted(df["season"].unique())
    test = seasons[-4:]
    say(f"\n# NBA：{len(df)} 場例行賽，train {seasons[:-4]}｜test {test}")
    return (evaluate(df, sp, "pnl_h", "pnl_a", test, "NBA 讓分（DK 收盤）")
            + evaluate(df, tot, "pnl_o", "pnl_u", test, "NBA 大小分（DK 收盤）"))


def _selfcheck() -> None:
    d = pd.DataFrame({"gid": range(4), "season": "s", "date": pd.to_datetime(["2024-01-01", "2024-01-04", "2024-01-08", "2024-01-10"]),
                      "home": ["A", "A", "B", "A"], "away": ["B", "C", "A", "C"], "margin": [2, 1, -3, 0], "cover_h": [1, 1, -1, -1],
                      "over": [1.0, 0.0, 1.0, 1.0]})
    f = team_features(d, 3, "over").set_index("gid")
    assert f.loc[3, "h_wstreak"] == 3 and f.loc[3, "h_n"] == 3 and f.loc[3, "h_rest"] == 2, f.loc[3]   # A 連贏三場（含客場那場）
    assert f.loc[2, "a_roadstreak"] == 0 and f.loc[3, "a_roadstreak"] == 1 and f.loc[3, "h_over3"] == 2
    assert f.loc[3, "h_prev_m"] == 3 and pd.isna(f.loc[0, "h_rest"])


if __name__ == "__main__":
    _selfcheck()
    rows = football() + nba()
    pd.DataFrame(rows).to_csv(OUT / "situational.csv", index=False, encoding="utf-8-sig")
    stars = [r for r in rows if r["star"]]
    say(f"\n# 總結：共測 {len(rows)} 個因子，train t≥1.64 且 test ROI>0 的有 {len(stars)} 個（★）")
    say(f"純靠運氣，{len(rows)} 個裡預期也會有約 {len(rows) * 0.05 * 0.5:.1f} 個同時通過這兩關。")
    for r in stars:
        say(f"★ {r['market']}｜{r['signal']}：train {r['roi_train']*100:+.1f}%（t={r['t_train']:+.2f}, n={r['n_train']}）"
            f"｜test {r['roi_test']*100:+.1f}%（t={r['t_test']:+.2f}, n={r['n_test']}）")
    (OUT / "situational.md").write_text("\n".join(LINES), encoding="utf-8")
