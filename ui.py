# -*- coding: utf-8 -*-
"""Telegram 訊息的共用排版（10/23 介面整理）：評等、對錯符號、等寬表格。

表格放在 <pre> 裡用等寬字：中文字算 2 格寬，而且中文只放在最後一欄（手機字型的中文寬度不一定剛好 2 格，
放最後一欄就不會把後面的欄位推歪）。
"""
from __future__ import annotations

import unicodedata
from html import escape

GOOD, FAIR = 0.02, -0.02     # 期望值 ≥ +2% 推薦；−2% ~ +2% 接近公平；更低就是偏貴


def grade_icon(ev: float) -> str:
    return "🟢" if ev >= GOOD else "🟡" if ev >= FAIR else "🔴"


def grade(ev: float) -> str:
    if ev >= GOOD:
        return "🟢 推薦"
    if ev >= FAIR:
        return "🟡 接近公平"
    return "🔴 偏貴（只是最不虧）"


def mark(profit: float | None) -> str:
    p = profit or 0
    return "✅" if p > 0 else "❌" if p < 0 else "➖"


def result_text(profit: float | None) -> str:
    p = profit or 0
    return f"✅ 中 {p:+.2f} 注" if p > 0 else f"❌ 沒中 {p:+.2f} 注" if p < 0 else "➖ 走盤 0 注"


def _width(s: str) -> int:
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 0 if unicodedata.combining(c) else 1 for c in s)


def _pad(s: str, w: int, right: bool) -> str:
    gap = " " * max(0, w - _width(s))
    return gap + s if right else s + gap


def table(header: list[str], rows: list[list[str]], right: tuple[int, ...] = ()) -> str:
    """等寬表格（已 HTML escape，直接放進 Telegram HTML 訊息）。right = 靠右對齊的欄位（數字）。"""
    widths = [max(_width(str(x)) for x in col) for col in zip(header, *rows)]
    out = ["  ".join(_pad(str(c), widths[i], i in right) for i, c in enumerate(r)).rstrip() for r in [header, *rows]]
    return "<pre>" + escape("\n".join(out)) + "</pre>"


def options_table(options: list[dict], best: dict | None) -> str:
    """賽前分析的所有選項：勝率、合理賠率、DK 賠率、期望值；推薦的那個標 👈。"""
    rows = []
    for o in options:
        if "fair" not in o:
            continue
        tag = " 👈" if best is not None and o is best and o["ev"] >= GOOD else ""
        rows.append([f"{o['p']*100:.0f}%", f"{o['fair']:.2f}", f"{o['odds']:.2f}", f"{o['ev']*100:+.1f}%",
                     o.get("short", o["name"]) + tag])
    return table(["勝率", "合理", "DK", "EV", "選項"], rows, right=(0, 1, 2, 3))


def recommend_block(best: dict | None) -> list[str]:
    """賽前分析最上面的結論：推薦哪個、運彩要多少賠率才下；沒有價值就直接說觀望。"""
    if not best:
        return ["⚪ <b>資料不足，無法評估</b>"]
    ev = best["ev"]
    if ev >= GOOD:
        return [f"🟢 <b>推薦：{escape(best['name'])}</b>",
                f"　 DK {best['odds']:.2f}｜估計勝率 {best['p']*100:.0f}%｜期望值 {ev*100:+.1f}%",
                f"　 📌 <b>運彩賠率 ≥ {best['fair']:.2f} 才下</b>"]
    if ev >= FAIR:
        return [f"🟡 <b>沒有明顯價值</b>：最接近公平的是 {escape(best['name'])}（EV {ev*100:+.1f}%）",
                f"　 運彩賠率 ≥ {best['fair']:.2f} 才考慮，否則觀望"]
    return [f"🔴 <b>觀望</b>：每個選項都偏貴（最好的是 {escape(best['name'])}，EV {ev*100:+.1f}%）"]
