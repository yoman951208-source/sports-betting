# -*- coding: utf-8 -*-
"""模型與解析的基本檢查（不需要網路）。執行：python tests/test_models.py（或 pytest）。"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import analyze  # noqa: E402
import oddscheck  # noqa: E402


def test_goal_model_is_a_distribution():
    gm = analyze.goal_model((0.55, 0.25, 0.20), 2.7)
    assert abs(sum(gm["margin"].values()) - 1) < 1e-9
    assert abs(sum(gm["total"].values()) - 1) < 1e-9
    assert len({s for _, s, _ in gm["top"]}) == 3          # 三個比分預測不重複


def test_margin_calibration_keeps_win_prob_and_trims_big_wins():
    pc, lam = (0.70, 0.18, 0.12), 3.2
    cal = analyze.goal_model(pc, lam)
    old, analyze.MARGIN_CAL = analyze.MARGIN_CAL, (0.0, 1.0)  # 不校準
    try:
        raw = analyze.goal_model(pc, lam)
    finally:
        analyze.MARGIN_CAL = old
    win = lambda gm, k: sum(p for d, p in gm["margin"].items() if d >= k)
    assert abs(win(cal, 1) - win(raw, 1)) < 1e-9             # 勝率不變（跟市場一致）
    assert win(cal, 2) < win(raw, 2)                         # 大勝的機率變小（純 Poisson 高估）


def test_line_table_uses_dk_for_the_dk_line():
    gm = analyze.goal_model((0.5, 0.27, 0.23), 2.6)
    m = {"L": 2.5, "sp": None, "options": [{"key": ("total", "over"), "p": 0.4321}]}
    assert "43%" in analyze.line_table(gm, (0.5, 0.27, 0.23), m).split("大 2.5")[0].splitlines()[-1]


def test_in_focus():
    m = {"focus": ("total", "spread")}
    assert analyze.in_focus(m, {"key": ("total", "over"), "odds": 1.2})
    assert analyze.in_focus(m, {"key": ("1x2", "H"), "odds": 1.6})
    assert not analyze.in_focus(m, {"key": ("ml", "home"), "odds": 1.3})
    assert analyze.in_focus(m, {"key": ("ml", "home"), "odds": 1.3, "signal": "⭐"})
    assert analyze.in_focus({}, {"key": ("ml", "home"), "odds": 1.1})   # MLB 沒有限制


def test_odds_parser():
    leg = oddscheck.parse_leg("湖人 -3.5 1.80")
    assert (leg["team"], leg["market"], leg["line"], leg["odds"]) == ("湖人", "spread", -3.5, 1.8)
    leg = oddscheck.parse_leg("曼城 大2.5 1.85")
    assert (leg["market"], leg["pick"], leg["line"], leg["odds"]) == ("total", "over", 2.5, 1.85)
    assert oddscheck.parse_leg("利物浦 和 3.40")["market"] == "draw"
    assert oddscheck.parse_leg("湖人") is None
    assert len(oddscheck.parse("曼城 2.10\n湖人 受讓3.5 1.80")) == 2


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
