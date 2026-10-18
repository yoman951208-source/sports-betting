# -*- coding: utf-8 -*-
"""市場機率的校準層（2026-10-19 加入）。

做法：
  1. 去水改用 power 方法（找 k 使 Σ(1/賠率)^k = 1）—— 等比例去水會高估冷門
  2. 在 logit 尺度上做邏輯迴歸校準：logit(p真) = a + b·logit(p市場) + 其他特徵
只採用「一季訓練、另一季測試」每次都比原始市場準的組合（樣本外驗證，見 10/19 評估）：
  足球 1X2  ：logit(p)、是否和局、logit(p)×和局          （3 個測試季都改善）
  NBA 不讓分：logit(p)、開盤→收盤的 logit 變化          （2 季都改善；資金方向在收盤後仍有資訊）
  NBA 讓分  ：logit(p)、讓分移動（主隊被買為正，單位：分）（2 季都改善）
  其餘（MLB、所有大小分）：校準在樣本外變差 → 維持原始市場機率（足球大小只換成 power 去水）
係數每週隨回測重新估計，存在資料庫 kv「calibration」（雲端與本機會一起同步）。
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


def fit_all() -> dict:
    import pandas as pd
    import data_football
    import data_nba
    out: dict = {"fitted": time.strftime("%Y-%m-%d %H:%M")}
    fb = data_football.load_history().dropna(subset=["c_avg_h", "c_avg_d", "c_avg_a"])
    pc = _power_matrix(fb[["c_avg_h", "c_avg_d", "c_avg_a"]].to_numpy(float))
    X, y = [], []
    for k, lab in enumerate("HDA"):
        lp = _logit(pc[:, k])
        draw = np.full(len(fb), float(lab == "D"))
        X.append(np.column_stack([lp, draw, lp * draw]))
        y.append((fb["ftr"] == lab).to_numpy(float))
    out["football_1x2"] = _fit(np.vstack(X), np.concatenate(y))

    nba = data_nba.load_history()
    ml = nba.dropna(subset=["c_ml_home", "c_ml_away", "o_ml_home", "o_ml_away"])
    pc = _power_matrix(ml[["c_ml_home", "c_ml_away"]].to_numpy(float))[:, 0]
    po = _power_matrix(ml[["o_ml_home", "o_ml_away"]].to_numpy(float))[:, 0]
    out["nba_ml"] = _fit(np.column_stack([_logit(pc), _logit(pc) - _logit(po)]),
                         (ml["home_score"] > ml["away_score"]).to_numpy(float))
    sp = nba.dropna(subset=["c_spread_home", "o_spread_home"])
    m = sp["home_score"] - sp["away_score"] + sp["c_spread_home"]
    sp, m = sp[m != 0], m[m != 0]
    ps = _power_matrix(sp[["c_spread_home_odds", "c_spread_away_odds"]].fillna(1.91).to_numpy(float))[:, 0]
    dsp = -(sp["c_spread_home"] - sp["o_spread_home"]).to_numpy(float)
    out["nba_spread"] = _fit(np.column_stack([_logit(ps), dsp]), (m > 0).to_numpy(float))
    out["n"] = {"football": int(len(fb)), "nba_ml": int(len(ml)), "nba_spread": int(len(sp))}
    return out


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
