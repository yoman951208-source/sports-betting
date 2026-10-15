# -*- coding: utf-8 -*-
"""日本聯賽盃（ルヴァンカップ）分析 —— 沒有即時盤口時的「模型合理賠率」。

ESPN 與免費賠率網站都沒有聯賽盃盤口，所以改用：
  1. football-data.co.uk 的 J1 收盤賠率（市場平均，去水）→ 換算每場的預期淨勝球
  2. 用最小平方法解出每隊的「市場隱含實力」（單位：球／場，相對 J1 平均），近期比賽權重較高
  3. J2 / J3 球隊沒有 J1 資料時，用「降級前的 J1 實力」或分級先驗值（假設，可信度低）
  4. 盃賽主力輪換：實力差打 8 折
  5. Poisson 比分模型 → 90 分鐘勝平負、讓球、大小的合理賠率
"""
from __future__ import annotations

import io
import math
import time
from html import escape

import numpy as np
import pandas as pd

from analyze import _hda, _score_matrix, handicap_cover
from core import CACHE_DIR, http_get, log
from strategies import p_over

CSV = "https://www.football-data.co.uk/new/JPN.csv"
HOME_ADV_PRIOR = 0.25
CUP_SHRINK = 0.8          # 盃賽輪換：實力差打折
TIER_PRIOR = {"J2": -0.55, "J3": -0.95}   # 沒有 J1 資料的球隊：相對 J1 平均的實力（假設值）

# 今日（2026-10-14）ルヴァンカップ 1st ラウンド 3 回戦；(主, 客, 主隊級別, 開球台灣時間)
# 隊名用 football-data 的寫法；級別依 2026/27 ESPN J1 名單判斷
FIXTURES = [
    ("Yokohama FC", "Urawa Reds", "J2", "17:30"),
    ("Matsumoto Yamaga", "Sanfrecce Hiroshima", "J3", "18:00"),
    ("Yokohama F. Marinos", "Shimizu S-Pulse", "J1", "18:00"),
    ("Nagoya Grampus", "Okayama", "J1", "18:00"),
    ("Iwata", "Kawasaki Frontale", "J2", "18:00"),
    ("Sagan Tosu", "Avispa Fukuoka", "J2", "18:00"),
    ("Gainare Tottori", "Cerezo Osaka", "J3", "18:00"),
    ("Mito", "FC Tokyo", "J1", "18:00"),
]
ZH = {
    "Yokohama FC": "橫濱FC", "Urawa Reds": "浦和紅鑽", "Matsumoto Yamaga": "松本山雅", "Sanfrecce Hiroshima": "廣島三箭",
    "Yokohama F. Marinos": "橫濱水手", "Shimizu S-Pulse": "清水心跳", "Nagoya Grampus": "名古屋鯨八",
    "Okayama": "岡山綠雉", "Iwata": "磐田喜悅", "Kawasaki Frontale": "川崎前鋒", "Sagan Tosu": "鳥栖砂岩",
    "Avispa Fukuoka": "福岡黃蜂", "Gainare Tottori": "鳥取飛翔", "Cerezo Osaka": "大阪櫻花", "Mito": "水戶蜀葵",
    "FC Tokyo": "FC東京",
}


def load() -> pd.DataFrame:
    path = CACHE_DIR / "football" / "JPN.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or time.time() - path.stat().st_mtime > 12 * 3600:
        resp = http_get(CSV)
        if resp is not None:
            path.write_bytes(resp.content)
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    return df.dropna(subset=["date", "AvgCH", "AvgCD", "AvgCA"])


def supremacy(pH: float, pA: float, lam: float) -> float:
    """市場勝負機率 → Poisson 模型下的預期淨勝球（主 - 客）。"""
    lo, hi = -3.5, 3.5
    target = pH / (pH + pA)
    for _ in range(40):
        s = (lo + hi) / 2
        lh, la = max(0.05, (lam + s) / 2), max(0.05, (lam - s) / 2)
        h, _d, a = _hda(_score_matrix(lh, la))
        if h / (h + a) < target:
            lo = s
        else:
            hi = s
    return s


def fit_ratings(df: pd.DataFrame) -> tuple[dict[str, float], float, float, dict[str, str]]:
    """近期加權最小平方：s = 主場優勢 + r_主 - r_客。回傳（實力, 主場優勢, 平均總進球, 每隊最後出現的季別）。"""
    d = df[df["date"] >= "2024-01-01"].copy()
    inv = 1 / d[["AvgCH", "AvgCD", "AvgCA"]]
    s = inv.sum(axis=1)
    d["pH"], d["pA"] = inv["AvgCH"] / s, inv["AvgCA"] / s
    lam_by_season = (d["HG"] + d["AG"]).groupby(d["Season"]).mean()
    d["lam"] = d["Season"].map(lam_by_season)
    d["sup"] = [supremacy(h, a, l) for h, a, l in zip(d["pH"], d["pA"], d["lam"])]
    # 權重：本季 3、2025 年 1、2024 年 0.4（讓降級隊仍有估計值，但以近期為主）
    w = np.where(d["Season"] == "2026/2027", 3.0, np.where(d["date"].dt.year == 2025, 1.0, 0.4))
    teams = sorted(set(d["Home"]) | set(d["Away"]))
    idx = {t: i for i, t in enumerate(teams)}
    X = np.zeros((len(d) + 1, len(teams) + 1))
    y = np.zeros(len(d) + 1)
    for k, (h, a, sp) in enumerate(zip(d["Home"], d["Away"], d["sup"])):
        X[k, idx[h]], X[k, idx[a]], X[k, -1], y[k] = 1, -1, 1, sp
    X[-1, :-1] = 1  # 約束：實力總和 = 0
    sw = np.sqrt(np.append(w, 50.0))
    coef, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    last_season = d.groupby("Home")["Season"].max().to_dict()
    cur = d[d["Season"] == "2026/2027"]
    cur_teams = set(cur["Home"]) | set(cur["Away"])
    # 以本季 J1 球隊平均為 0
    base = np.mean([coef[idx[t]] for t in cur_teams]) if cur_teams else 0.0
    ratings = {t: float(coef[idx[t]] - base) for t in teams}
    # 本季只有約 90 場，平均進球雜訊大 → 與前兩季平均各半
    cur_lam = lam_by_season.get("2026/2027", lam_by_season.iloc[-1])
    prev_lam = lam_by_season.drop("2026/2027", errors="ignore").mean()
    lam_now = float((cur_lam + prev_lam) / 2)
    return ratings, float(coef[-1]), lam_now, {t: last_season.get(t, "") for t in teams}


def match_model(home: str, away: str, home_tier: str, ratings: dict, last: dict, hfa: float, lam: float):
    notes = []

    def rating(team: str, tier: str) -> tuple[float, str]:
        if team in ratings and (tier == "J1" or last.get(team) in ("2025", "2026/2027")):
            r = ratings[team]
            if tier != "J1":
                r -= 0.15  # 降級後一般會再弱一些
                notes.append(f"{ZH.get(team, team)} 用降級前（{last.get(team)}）的 J1 實力再扣 0.15")
            return r, "高" if tier == "J1" else "中"
        if team in ratings:
            r = min(ratings[team] - 0.25, TIER_PRIOR.get(tier, 0))
            notes.append(f"{ZH.get(team, team)} 最後在 J1 是 {last.get(team)}，資料舊，取較保守值")
            return r, "低"
        notes.append(f"{ZH.get(team, team)}（{tier}）沒有 J1 資料，用 {tier} 先驗值 {TIER_PRIOR.get(tier, 0):+.2f}")
        return TIER_PRIOR.get(tier, 0.0), "低"

    rh, ch = rating(home, home_tier)
    ra, ca = rating(away, "J1")
    sup = hfa + CUP_SHRINK * (rh - ra)
    lh, la = max(0.1, (lam + sup) / 2), max(0.1, (lam - sup) / 2)
    p = _hda(_score_matrix(lh, la))
    conf = "低" if "低" in (ch, ca) else ("中" if "中" in (ch, ca) else "中高")
    return {"p": p, "lh": lh, "la": la, "sup": sup, "conf": conf, "notes": notes, "rh": rh, "ra": ra}


def fair(p: float) -> str:
    return f"{1/p:.2f}" if p > 0.01 else "-"


def report() -> list[str]:
    df = load()
    ratings, hfa, lam, last = fit_ratings(df)
    log.info("J1 實力模型：%d 隊，主場優勢 %.2f 球，平均總進球 %.2f", len(ratings), hfa, lam)
    blocks = []
    summary = []
    for home, away, tier, ko in FIXTURES:
        m = match_model(home, away, tier, ratings, last, hfa, lam)
        pH, pD, pA = m["p"]
        fav_home = pH >= pA
        line = -1.5 if fav_home else 1.5
        cov = handicap_cover(m["lh"], m["la"], line)
        fav_cov = cov if fav_home else 1 - cov
        p_dc_home = pH + pD                    # 主隊不敗（受讓 0.5）
        po25 = p_over(m["lh"] + m["la"], 2.5)
        hz, az = ZH.get(home, home), ZH.get(away, away)
        lines = [f"⚽ 日本聯賽盃｜10/14 {ko}（台灣）",
                 f"<b>{escape(hz)}（{tier}）vs {escape(az)}（J1）</b>",
                 f"模型實力（球/場，J1 平均=0）：主 {m['rh']:+.2f}｜客 {m['ra']:+.2f}｜預期比分 {m['lh']:.2f} : {m['la']:.2f}",
                 f"90 分鐘：主勝 {pH*100:.0f}%｜和 {pD*100:.0f}%｜客勝 {pA*100:.0f}%｜大2.5 {po25*100:.0f}%｜可信度 <b>{m['conf']}</b>",
                 "<b>合理賠率</b>（運彩 ≥ 這個才划算）",
                 f"  主勝 {fair(pH)}｜和 {fair(pD)}｜客勝 {fair(pA)}",
                 f"  {'主' if fav_home else '客'}讓1.5 {fair(fav_cov)}｜受讓1.5 {fair(1 - fav_cov)}｜主隊不敗 {fair(p_dc_home)}",
                 f"  大2.5 {fair(po25)}｜小2.5 {fair(1 - po25)}"]
        lines += [f"  ※ {escape(n)}" for n in m["notes"]]
        blocks.append("\n".join(lines))
        best = max((("主勝", pH), ("和", pD), ("客勝", pA)), key=lambda x: x[1])
        summary.append(f"{ko} {escape(hz)} vs {escape(az)}：主/和/客 {pH*100:.0f}/{pD*100:.0f}/{pA*100:.0f}%｜"
                       f"大2.5 {po25*100:.0f}%｜可信度 {m['conf']}")
    head = "\n".join([
        "🇯🇵 <b>日本聯賽盃（ルヴァンカップ）1st Round 3 回戰｜今天傍晚</b>", "",
        "⚠️ <b>這次沒有即時盤口</b>（ESPN 與免費賠率網站都沒有這個盃賽），以下是<b>模型估計</b>，不是市場價格：",
        f"• J1 球隊實力來自 2024–2026/27 J1 收盤賠率（本季權重最高）；主場優勢 {hfa:+.2f} 球、平均總進球 {lam:.2f}",
        "• J2/J3 球隊沒有本季 J1 資料，用降級前實力或分級假設值 → <b>可信度低</b>",
        "• 大小球只用聯盟平均進球（資料沒有大小盤），各場都一樣 → 大小只當參考，不建議據此下注",
        f"• 盃賽常大幅輪換，實力差先打 {CUP_SHRINK*100:.0f} 折；以 90 分鐘（含傷停）賽果計算", "",
    ] + summary + ["", "用法：把運彩開的賠率拿來比「合理賠率」，<b>運彩 ≥ 合理才考慮</b>；可信度低的場次建議不買或極小注。"])
    return [head] + blocks


def parlay_report(min_conf: tuple[str, ...] = ("中高", "中")) -> list[str]:
    """沒有市場價格，所以只能依模型命中率選關；每張附「合理組合賠率」，運彩組合賠率 ≥ 它才有正期望值。"""
    from itertools import combinations
    df = load()
    ratings, hfa, lam, last = fit_ratings(df)
    legs = []
    for home, away, tier, ko in FIXTURES:
        m = match_model(home, away, tier, ratings, last, hfa, lam)
        if m["conf"] not in min_conf:
            continue
        pH, pD, pA = m["p"]
        fav_home = pH >= pA
        cov = handicap_cover(m["lh"], m["la"], -1.5 if fav_home else 1.5)
        dog_cov = 1 - cov if fav_home else cov
        hz, az = ZH.get(home, home), ZH.get(away, away)
        dog = az if fav_home else hz   # 受讓 1.5 的是非熱門那隊
        for name, p in ((f"{hz} 勝", pH), (f"{az} 勝", pA), (f"{dog} 受讓1.5", dog_cov)):
            if 0.38 <= p <= 0.85:
                legs.append({"match": f"{ko} {hz} vs {az}", "name": name, "p": p, "conf": m["conf"]})
    blocks = []
    for size, cn in ((2, "二串一"), (3, "三串一")):
        combos = [c for c in combinations(legs, size) if len({l["match"] for l in c}) == size]
        if not combos:
            continue
        prob = lambda c: math.prod(l["p"] for l in c)
        picks = [("命中率最高", max(combos, key=prob))]
        # 不含「受讓1.5」的純勝負組合，組合賠率較高
        pure = [c for c in combos if all("受讓" not in l["name"] for l in c)]
        if pure:
            alt = max(pure, key=prob)
            if alt is not picks[0][1]:
                picks.append(("純勝負（賠率較高）", alt))
        for label, c in picks:
            p = prob(c)
            lines = [f"🎫 <b>日本聯賽盃 {cn}｜{label}</b>"]
            lines += [f"{i}. {escape(l['match'])}\n   👉 <b>{escape(l['name'])}</b>｜模型 {l['p']*100:.0f}%｜合理 {1/l['p']:.2f}（可信度 {l['conf']}）"
                      for i, l in enumerate(c, 1)]
            lines.append(f"命中率約 <b>{p*100:.0f}%</b>｜合理組合賠率 <b>{1/p:.2f}</b>")
            lines.append(f"✅ 運彩組合賠率 ≥ <b>{1/p:.2f}</b> 才划算（沒有市場價可比，只能靠這個門檻）")
            blocks.append("\n".join(lines))
    head = "\n".join([
        "🇯🇵 <b>日本聯賽盃 串關（模型版）</b>",
        "⚠️ 這個盃賽沒有市場盤口，以下只用「可信度中以上」的場次（J1 對 J1、橫濱FC vs 浦和），依模型命中率挑選；",
        "模型沒被市場驗證過，<b>請一定要比對運彩組合賠率</b>，低於合理組合賠率就不要買。",
    ])
    return [head] + blocks


if __name__ == "__main__":
    for b in report():
        print(b, "\n")
