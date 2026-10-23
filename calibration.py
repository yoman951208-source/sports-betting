# -*- coding: utf-8 -*-
"""市場機率的校準層（2026-10-19 加入，10/23 用更多季別重新驗證後精簡）。

做法：
  1. 去水用 power 方法（找 k 使 Σ(1/賠率)^k = 1）—— 等比例去水會高估冷門（足球 8 季中 7 季較準）
  2. 只保留「逐季留一」（用其他季別估計、在沒看過的那季測試）每一季都比原始市場準的修正：
     足球讓分：只加截距（斜率固定 1）—— 莊家在讓球盤對主隊多抽，主隊實際贏盤率比去水機率低約 2pp
               （8 季中 7 季改善）
     NBA 讓分：主力缺陣差、主力疲勞差（有開盤的 3 季都改善）
10/23 拿掉的（原本只用 2–3 季驗證，季別變多後就不穩）：
  足球勝平負的 logit／和局校準（8 季只有 4 季改善）、NBA 不讓分加盤口變化（3 季 1 季）、
  NBA 讓分加讓分移動（3 季 2 季）、NBA 不讓分加主力疲勞（3 季 2 季）。MLB、所有大小分本來就不校準。
係數每週隨回測重新估計，存在資料庫 kv「calibration」。CAL_VERSION 改了雲端會自動重估。
"""
from __future__ import annotations

import json
import math
import time

import numpy as np

from core import db, kv_get, kv_set, log

L2 = 1.0
_cache: tuple[dict, float] | None = None


# -----------------------------------------------------------------------------
# 去水
# -----------------------------------------------------------------------------
def power_devig(odds) -> list[float] | None:
    vals = [float(o) for o in odds if o is not None]
    if len(vals) != len(odds) or any(not math.isfinite(o) or o <= 1 for o in vals):
        return None
    inv = [1 / o for o in vals]
    lo, hi = 0.5, 3.0
    for _ in range(60):
        k = (lo + hi) / 2
        if sum(x ** k for x in inv) < 1:
            hi = k
        else:
            lo = k
    p = [x ** k for x in inv]
    s = sum(p)
    return [x / s for x in p]


def _power_matrix(odds: np.ndarray) -> np.ndarray:
    inv = 1 / odds
    lo, hi = np.full(len(odds), 0.5), np.full(len(odds), 3.0)
    for _ in range(60):
        k = (lo + hi) / 2
        s = (inv ** k[:, None]).sum(axis=1)
        hi = np.where(s < 1, k, hi)
        lo = np.where(s >= 1, k, lo)
    p = inv ** k[:, None]
    return p / p.sum(axis=1, keepdims=True)


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


# -----------------------------------------------------------------------------
# 估計
# -----------------------------------------------------------------------------
def _fit(X: np.ndarray, y: np.ndarray) -> list[float]:
    """L2 正則化往「不校準」(a=0, b=1, 其他=0) 收縮的邏輯迴歸（牛頓法）。"""
    X1 = np.column_stack([np.ones(len(X)), X])
    prior = np.zeros(X1.shape[1])
    prior[1] = 1.0
    w = prior.copy()
    reg = np.r_[0.0, np.ones(X1.shape[1] - 1)] * L2
    for _ in range(50):
        p = _sigmoid(X1 @ w)
        g = X1.T @ (p - y) + reg * (w - prior)
        H = (X1 * (p * (1 - p))[:, None]).T @ X1 + np.diag(reg)
        w = w - np.linalg.solve(H, g)
    return [round(float(v), 5) for v in w]


def _fit_intercept(lp: np.ndarray, y: np.ndarray) -> float:
    """斜率固定 1、只估截距：logit(p真) = a + logit(p市場)。"""
    a = 0.0
    for _ in range(30):
        p = _sigmoid(a + lp)
        a -= float((p - y).sum() / (p * (1 - p)).sum())
    return round(a, 5)


CAL_VERSION = 3        # 10/23：季別變多後重新驗證、拿掉不穩的校準


def fit_all() -> dict:
    import data_football
    import data_nba
    out: dict = {"fitted": time.strftime("%Y-%m-%d %H:%M"), "version": CAL_VERSION}
    fb = data_football.load_history()
    # 讓球：只用整數 / 半球盤（DK 的足球讓分都是半球盤；四分之一盤會半贏半輸，不能當成輸贏二元）
    ah = fb.dropna(subset=["c_ah_line", "c_ah_h", "c_ah_a"])
    ah = ah[(ah["c_ah_line"] * 2) % 1 == 0]
    m = ah["hg"] - ah["ag"] + ah["c_ah_line"]
    ah, m = ah[m != 0], m[m != 0]
    lp = _logit(_power_matrix(ah[["c_ah_h", "c_ah_a"]].to_numpy(float))[:, 0])
    out["football_spread"] = _fit_intercept(lp, (m > 0).to_numpy(float))
    out["n"] = {"football_spread": int(len(ah))}
    try:
        out.update(_fit_nba_form(data_nba.load_history()))
    except Exception:  # box score 抓不到時不影響其他校準
        log.exception("NBA 球員狀態校準失敗")
    return out


def _fit_nba_form(nba) -> dict:
    """NBA 讓分：power 去水之上加主力缺陣差、主力疲勞差（10/23：有開盤的 3 季逐季留一都改善）。"""
    import data_nba_box
    import nba_form
    # 只用近幾季（有開盤、box score 已在快取）；舊季別只有收盤，不必下載上萬場 box score
    nba = nba[nba["season"].isin(nba.loc[nba["o_spread_home"].notna(), "season"].unique())]
    box = data_nba_box.load_boxes(nba["event_id"].astype(str).tolist())
    F = nba_form.game_features(nba, box)
    d = nba.astype({"event_id": str}).merge(F, on="event_id")
    d = d[(d["h_n_prior"] >= 10) & (d["a_n_prior"] >= 10)]
    s = d.dropna(subset=["c_spread_home"])
    m = s["home_score"] - s["away_score"] + s["c_spread_home"]
    s, m = s[m != 0], m[m != 0]
    ps = _power_matrix(s[["c_spread_home_odds", "c_spread_away_odds"]].fillna(1.91).to_numpy(float))[:, 0]
    X = np.column_stack([_logit(ps), s["d_star_out"].fillna(0), s["d_star_load"].fillna(0)])
    return {"nba_form_spread": _fit(X, (m > 0).to_numpy(float)), "n_nba_form": int(len(s))}


def nba_form_adjust(p: float | None, kind: str, d_star_out: float | None = None,
                    d_star_load: float | None = None) -> float | None:
    """把球員狀態加到校準後的 NBA 機率上。kind = 'spread' / 'ml'。特徵算不出來（季初）就不調整。"""
    if p is None or d_star_load is None:
        return p
    c = coefs()
    w = c.get(f"nba_form_{kind}")
    if not w or c.get("raw_nba"):
        return p
    lp = float(_logit(p))
    if kind == "spread":
        return float(_sigmoid(w[0] + w[1] * lp + w[2] * (d_star_out or 0.0) + w[3] * d_star_load))
    return float(_sigmoid(w[0] + w[1] * lp + w[2] * d_star_load))


def refit() -> dict:
    """重新估計並存進資料庫（每週回測時呼叫）。"""
    global _cache
    coef = fit_all()
    conn = db()
    try:
        kv_set(conn, "calibration", json.dumps(coef))
    finally:
        conn.close()
    _cache = (coef, time.time())
    log.info("校準係數更新：%s", coef)
    return coef


def coefs() -> dict:
    global _cache
    if _cache and time.time() - _cache[1] < 600:
        return _cache[0]
    try:
        conn = db()
        try:
            v = kv_get(conn, "calibration")
            off = {s for s in ("football", "nba") if kv_get(conn, f"calibration_off:{s}") == "1"}
        finally:
            conn.close()
        c = json.loads(v) if v else {}
    except Exception:
        c, off = {}, set()
    # learning.check_calibration 發現實戰變差而關閉的運動：拿掉係數 → 退回原始等比例去水
    if "football" in off:
        c.pop("football_1x2", None)
        c.pop("football_spread", None)
        c["raw_football"] = True
    if "nba" in off:
        c.pop("nba_ml", None)
        c.pop("nba_spread", None)
        c["raw_nba"] = True
    _cache = (c, time.time())
    return c


def _prop(odds) -> list[float] | None:
    vals = [o for o in odds if o]
    if len(vals) != len(odds) or any(o <= 1 for o in vals):
        return None
    inv = [1 / o for o in vals]
    return [x / sum(inv) for x in inv]


# -----------------------------------------------------------------------------
# 套用（沒有係數時退回 power 去水 / 原始機率）
# -----------------------------------------------------------------------------
def football_1x2(h, d, a) -> tuple[float, float, float] | None:
    if coefs().get("raw_football"):
        q = _prop([h, d, a])
        return tuple(q) if q else None
    p = power_devig([h, d, a])
    if not p:
        return None
    w = coefs().get("football_1x2")
    if not w:
        return tuple(p)
    raw = []
    for k, pk in enumerate(p):
        lp = float(_logit(pk))
        draw = 1.0 if k == 1 else 0.0
        raw.append(float(_sigmoid(w[0] + w[1] * lp + w[2] * draw + w[3] * lp * draw)))
    s = sum(raw)
    return tuple(x / s for x in raw)


def football_1x2_matrix(odds: np.ndarray) -> np.ndarray:
    """歷史資料一次算（給相似盤口的殘差用）。"""
    p = _power_matrix(odds)
    w = coefs().get("football_1x2")
    if not w:
        return p
    lp = _logit(p)
    draw = np.array([0.0, 1.0, 0.0])
    raw = _sigmoid(w[0] + w[1] * lp + w[2] * draw + w[3] * lp * draw)
    return raw / raw.sum(axis=1, keepdims=True)


def two_way(o1, o2) -> float | None:
    p = power_devig([o1, o2])
    return p[0] if p else None


def football_spread(odds_home, odds_away) -> float | None:
    """足球讓分的主隊贏盤機率：power 去水 + 主隊截距（莊家對主隊多抽的部分扣回來）。"""
    p = two_way(odds_home, odds_away)
    a = coefs().get("football_spread")
    if p is None or a is None:
        return p
    return float(_sigmoid(a + float(_logit(p))))


def nba_ml(c_home, c_away, o_home=None, o_away=None) -> float | None:
    if coefs().get("raw_nba"):
        q = _prop([c_home, c_away])
        return q[0] if q else None
    pc = power_devig([c_home, c_away])
    if not pc:
        return None
    w = coefs().get("nba_ml")
    if not w:
        return pc[0]
    po = power_devig([o_home, o_away]) if o_home and o_away else None
    lp = float(_logit(pc[0]))
    dl = lp - float(_logit(po[0])) if po else 0.0
    return float(_sigmoid(w[0] + w[1] * lp + w[2] * dl))


def nba_spread(odds_home, odds_away, sp_close, sp_open=None) -> float | None:
    if coefs().get("raw_nba"):
        return _prop([odds_home or 1.91, odds_away or 1.91])[0]
    p = power_devig([odds_home or 1.91, odds_away or 1.91])
    if not p:
        return None
    w = coefs().get("nba_spread")
    if not w or sp_close is None:
        return p[0]
    dsp = -(sp_close - sp_open) if sp_open is not None else 0.0
    return float(_sigmoid(w[0] + w[1] * float(_logit(p[0])) + w[2] * dsp))


if __name__ == "__main__":
    print(refit())
