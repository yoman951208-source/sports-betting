# -*- coding: utf-8 -*-
"""Telegram 指令：在常駐模式中用 long polling 接收訊息，只回應 .env 裡設定的 chat_id。

  /help                 說明
  /ucl 或 歐冠           歐冠未來 36 小時全部比賽分析（歐霸、英超、西甲… 也可以直接打聯賽名稱）
  /nba                  NBA 未來 24 小時比賽分析
  /today 或 總覽         每日總覽
  /report 或 成績        實戰成績
  直接打隊名             例如：曼城、Arsenal、湖人、LAL → 分析該隊下一場比賽
"""
from __future__ import annotations

import time

import requests

import config
from core import _telegram_credentials, db, kv_get, kv_set, log, send_blocks, send_telegram

HELP = """🤖 <b>運彩分析機器人 指令</b>

直接輸入<b>隊名</b>（中英文都可）→ 分析該隊下一場比賽
  例：曼城、皇馬、Arsenal、湖人、LAL

<b>聯賽名稱</b> → 整個聯賽未來 36 小時的分析
  例：歐冠、歐霸、英超、西甲、義甲、德甲、法甲

/nba　NBA 未來 24 小時
MLB（或 棒球、季後賽）　MLB 未來 48 小時，含先發投手、系列賽、傷兵
/today　每日總覽
串關　未來 24 小時的二串一、三串一建議（預設歐冠；可加聯賽，例：串關 英超、串關 全部）
/review 或 賽後　過去 24 小時完賽比賽的賽後檢討（賽前預測 vs 實際）
/report　實戰成績
/help　顯示這個說明

每場分析會列出各選項的「合理賠率」：<b>運彩開的賠率 ≥ 合理賠率</b>才有正期望值。"""

LEAGUE_WORDS = {"ucl": "歐冠", "uel": "歐霸", "epl": "英超", "mlb": "MLB", "美職": "MLB", "棒球": "MLB", "季後賽": "MLB", **{v[1]: v[1] for v in config.FOOTBALL_LEAGUES.values()},
                **{v: v for v in config.FOOTBALL_CUPS.values()}}


def _has_league(league: str, hours: float = 24) -> bool:
    import analyze
    import live
    return bool(analyze.upcoming(live.fetch_games(), hours, league=league))


def handle(text: str) -> None:
    import analyze
    import live

    t = text.strip()
    cmd, _, arg = t.lstrip("/").partition(" ")
    cmd = cmd.split("@")[0].lower()
    if cmd in ("start", "help", "說明", "指令"):
        send_telegram(HELP)
        return
    if cmd in ("today", "總覽", "今日"):
        conn = db()
        try:
            live.daily_digest(conn, live.fetch_games(), live.load_strategies()[0], False)
        finally:
            conn.close()
        return
    if cmd in ("parlay", "串關", "串"):
        import parlay
        lg = LEAGUE_WORDS.get(arg.strip(), arg.strip() or None) if arg.strip() != "全部" else None
        msgs = parlay.report(live.fetch_games(), lg if arg.strip() else "歐冠" if _has_league("歐冠") else None, 24)
        send_telegram(msgs[0])
        send_blocks(msgs[1:])
        return
    if cmd in ("review", "賽後", "檢討", "賽果"):
        import review
        hours = float(arg) if arg.replace(".", "", 1).isdigit() else 24
        head, *blocks = review.build(hours)
        send_telegram(head)
        send_blocks(blocks)
        return
    if cmd in ("report", "成績", "戰績"):
        conn = db()
        try:
            live.weekly_report(conn, False)
        finally:
            conn.close()
        return

    games = live.fetch_games()
    league = "NBA" if cmd in ("nba", "籃球") else LEAGUE_WORDS.get(cmd) or LEAGUE_WORDS.get(t)
    if league:
        hours = {"NBA": 24, "MLB": 48}.get(league, 36)
        send_telegram(f"🔍 分析{league}未來 {hours} 小時的比賽中…")
        head, *details = analyze.league_report(games, league, hours)
        send_telegram(head)
        send_blocks(details)
        return

    query = arg if cmd in ("a", "analyze", "分析") else t
    found = analyze.search(games, query)
    if not found:
        send_telegram(f"找不到「{query}」未來 4 天內的比賽。\n可以輸入隊名（例：曼城、Arsenal、湖人、LAL）或聯賽（例：歐冠）。/help 看全部指令")
        return
    if len(found) > 5:
        send_telegram(f"找到 {len(found)} 場，先分析最近的 5 場；可以打更完整的隊名縮小範圍。")
    send_blocks([analyze.analyze_game(g)[0] for g in found[:5]])


def poll(timeout: int = 20) -> None:
    """等待新訊息最多 timeout 秒並處理；沒設定 Telegram 時就單純睡覺。"""
    token, chat_id = _telegram_credentials()
    if not token or not chat_id:
        time.sleep(timeout)
        return
    conn = db()
    try:
        offset = int(kv_get(conn, "tg_offset", "0"))
        resp = requests.get(f"https://api.telegram.org/bot{token}/getUpdates",
                            params={"offset": offset, "timeout": timeout, "allowed_updates": '["message"]'},
                            timeout=timeout + 15)
        updates = resp.json().get("result", []) if resp.ok else []
        if not resp.ok:
            time.sleep(5)
        for u in updates:
            kv_set(conn, "tg_offset", str(u["update_id"] + 1))  # 先記錄，避免出錯時無限重試同一則
            msg = u.get("message") or {}
            text = (msg.get("text") or "").strip()
            if str(msg.get("chat", {}).get("id")) != chat_id or not text:
                continue
            log.info("收到指令：%s", text[:50])
            try:
                handle(text)
            except Exception as exc:
                log.exception("指令處理失敗")
                send_telegram(f"處理「{text[:30]}」時發生錯誤：{exc}", parse_mode=None)
    except (requests.RequestException, ValueError) as exc:
        log.warning("Telegram 輪詢失敗：%s", str(exc).replace(token, "***"))
        time.sleep(5)
    finally:
        conn.close()
