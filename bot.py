# -*- coding: utf-8 -*-
"""Telegram 介面：固定選單按鈕 + 訊息內按鈕（inline keyboard），只回應 .env 裡設定的 chat_id。

固定選單（輸入框下方）：🏆今日推薦 🎟️串關 🏀NBA ⚾MLB ⚽足球 🧾賽後檢討 📅今日總覽 📒戰績 ❓說明
訊息內按鈕：
  聯賽總表 → 每場比賽一顆按鈕，點了才展開那場的詳細分析（不會一次洗十幾則訊息）
  今日推薦 → 🔄 重新整理 / 🎟️ 更多串關 / 📒 戰績
也仍然可以直接打字：隊名（曼城、湖人、LAL）、聯賽（歐冠、英超）、推薦、串關、賽後…
"""
from __future__ import annotations

import time

import requests

import config
from core import DATA_DIR, TELEGRAM_API, _telegram_credentials, db, log, send_blocks, send_telegram

OFFSET_FILE = DATA_DIR / "tg_offset.txt"   # Telegram 讀取位置（本機專用，不隨雲端同步）

# ---------------------------------------------------------------- 按鈕定義
MENU = {
    "keyboard": [
        [{"text": "🏆 今日推薦"}, {"text": "🎟️ 串關"}],
        [{"text": "🏀 NBA"}, {"text": "⚾ MLB"}, {"text": "⚽ 足球"}],
        [{"text": "🧾 賽後檢討"}, {"text": "📅 今日總覽"}],
        [{"text": "💱 運彩比價"}, {"text": "📒 戰績"}, {"text": "❓ 說明"}],
    ],
    "resize_keyboard": True,
    "is_persistent": True,
    "input_field_placeholder": "按下面的按鈕，或輸入隊名（例：湖人、曼城）",
}
BUTTONS = {"🏆 今日推薦": "推薦", "🎟️ 串關": "串關", "🏀 NBA": "nba", "⚾ MLB": "mlb", "⚽ 足球": "足球",
           "🧾 賽後檢討": "賽後", "📅 今日總覽": "today", "📒 戰績": "report", "❓ 說明": "help", "💱 運彩比價": "比價"}
BEST_BUTTONS = {"inline_keyboard": [
    [{"text": "🔄 重新整理", "callback_data": "c:推薦"}],
    [{"text": "🎟️ 更多串關", "callback_data": "c:串關 全部"}, {"text": "📒 戰績", "callback_data": "c:report"}],
]}
BOT_COMMANDS = [("best", "🏆 今日最推薦（單關、二串一、三串一）"), ("parlay", "🎟️ 串關建議"),
                ("nba", "🏀 NBA 比賽"), ("mlb", "⚾ MLB 比賽"), ("football", "⚽ 足球（選聯賽）"),
                ("review", "🧾 賽後檢討"), ("today", "📅 今日總覽"), ("report", "📒 實戰戰績"),
                ("menu", "顯示按鈕選單"), ("help", "說明")]

HELP = """🤖 <b>運彩分析機器人</b>

👇 <b>直接按下面的按鈕</b>就可以用：
🏆 <b>今日推薦</b>：掃描所有比賽，給出最推薦的單關、二串一、三串一
🎟️ <b>串關</b>：更多二串一、三串一組合
🏀 <b>NBA</b>／⚾ <b>MLB</b>／⚽ <b>足球</b>：先看總表，<b>點比賽按鈕</b>才展開詳細分析
🧾 <b>賽後檢討</b>：賽前預測 vs 實際結果
📅 <b>今日總覽</b>　📒 <b>戰績</b>
💱 <b>運彩比價</b>：直接打運彩的賠率，例如 <code>湖人 1.75</code>、<code>湖人 -3.5 1.80</code>、<code>曼城 大2.5 1.85</code>（好幾行 = 串關）

也可以直接打字：隊名（曼城、皇馬、湖人、LAL）、聯賽（歐冠、英超、西甲）

判斷原則只有一條：<b>運彩賠率 ≥ 合理賠率才下</b>，低於就跳過，只用小注。"""

LEAGUE_WORDS = {"ucl": "歐冠", "uel": "歐霸", "epl": "英超", "mlb": "MLB", "美職": "MLB", "棒球": "MLB", "季後賽": "MLB",
                **{v[1]: v[1] for v in config.FOOTBALL_LEAGUES.values()}, **{v: v for v in config.FOOTBALL_CUPS.values()}}

DASHBOARD_URL = f"https://{config.GITHUB_REPO.split('/')[0]}.github.io/{config.GITHUB_REPO.split('/')[1]}/"

_DETAIL: dict[str, tuple[str, float]] = {}   # game_id → (分析文字, 時間)：聯賽總表算過的直接給，點按鈕秒回


# ---------------------------------------------------------------- 工具
def _api(method: str, **payload) -> dict:
    token, _ = _telegram_credentials()
    try:
        return requests.post(TELEGRAM_API.format(token=token, method=method), json=payload, timeout=15).json()
    except (requests.RequestException, ValueError):
        return {}


def setup_commands() -> None:
    """設定左下角「/」指令選單（只需設一次，重複設定無害）。"""
    _api("setMyCommands", commands=[{"command": c, "description": d} for c, d in BOT_COMMANDS])


def _game_label(g: dict) -> str:
    import analyze
    from core import parse_iso, to_tpe
    import ui
    t = to_tpe(parse_iso(g["kickoff"]))
    ev = g.get("_best_ev")          # league_report 算過的最佳期望值 → 按鈕上直接看得到評等
    icon = "" if ev is None else ui.grade_icon(ev) + " "
    label = f"{icon}{t:%m/%d %H:%M} {analyze.game_title(g)}"
    return label if len(label) <= 40 else label[:39] + "…"


def _league_view(games: list[dict], league: str, hours: float) -> None:
    """聯賽總表 + 每場一顆按鈕。"""
    import analyze
    sel = analyze.upcoming(games, hours, league=league)
    if not sel:
        send_telegram(f"未來 {hours:g} 小時沒有{league}比賽。", reply_markup=MENU)
        return
    head, *details = analyze.league_report(games, league, hours)
    now = time.time()
    for g, d in zip(sel, details):
        _DETAIL[g["game_id"]] = (d, now)
    rows = [[{"text": _game_label(g), "callback_data": f"g:{g['game_id']}"}] for g in sel[:30]]
    rows.append([{"text": f"🎟️ {league} 串關", "callback_data": f"c:串關 {league}"},
                 {"text": "🏆 今日推薦", "callback_data": "c:推薦"}])
    send_telegram(head + "\n\n👇 <b>點比賽看詳細分析</b>", reply_markup={"inline_keyboard": rows})


def _football_picker(games: list[dict]) -> None:
    """⚽ 足球：列出未來 36 小時有比賽的聯賽，點了才看那個聯賽。"""
    import analyze
    counts: dict[str, int] = {}
    for g in analyze.upcoming(games, 36, sport="football"):
        counts[g["league"]] = counts.get(g["league"], 0) + 1
    if not counts:
        send_telegram("未來 36 小時沒有監控中的足球比賽。", reply_markup=MENU)
        return
    btns = [{"text": f"{lg}（{n}）", "callback_data": f"lg:{lg}"} for lg, n in sorted(counts.items(), key=lambda x: -x[1])]
    rows = [btns[i:i + 3] for i in range(0, len(btns), 3)]
    rows.append([{"text": "🎟️ 全部足球串關", "callback_data": "c:串關 全部"}])
    send_telegram("⚽ <b>選擇聯賽</b>（括號是未來 36 小時的場數）", reply_markup={"inline_keyboard": rows})


def _game_detail(game_id: str) -> None:
    import analyze
    import live
    hit = _DETAIL.get(game_id)
    if hit and time.time() - hit[1] < 900:
        text = hit[0]
    else:
        g = next((x for x in live.fetch_games() if x["game_id"] == game_id), None)
        if not g:
            send_telegram("找不到這場比賽（可能已經開賽或結束）。")
            return
        text = analyze.analyze_game(g)[0]
        _DETAIL[game_id] = (text, time.time())
    send_telegram(text, reply_markup={"inline_keyboard": [[{"text": "🔄 用最新盤口重算", "callback_data": f"r:{game_id}"}]]})


# ---------------------------------------------------------------- 指令
def handle(text: str) -> None:
    import analyze
    import live

    t = BUTTONS.get(text.strip(), text.strip())
    cmd, _, arg = t.lstrip("/").partition(" ")
    cmd = cmd.split("@")[0].lower()
    if cmd in ("start", "help", "說明", "指令", "menu", "選單"):
        send_telegram(HELP, reply_markup=MENU)
        return
    if cmd in ("today", "總覽", "今日"):
        conn = db()
        try:
            live.daily_digest(conn, live.fetch_games(), live.load_strategies()[0], False)
        finally:
            conn.close()
        return
    if cmd in ("best", "推薦", "最推薦", "今日推薦", "建議"):
        import picks
        send_telegram("🔍 掃描所有比賽中…")
        send_blocks(picks.report(live.fetch_games(), float(arg) if arg.replace(".", "", 1).isdigit() else 24),
                    reply_markup=BEST_BUTTONS)
        return
    if cmd in ("parlay", "串關", "串"):
        import parlay
        a = arg.strip()
        lg = None if a in ("", "全部") else LEAGUE_WORDS.get(a, a)
        msgs = parlay.report(live.fetch_games(), lg, 24)
        send_telegram(msgs[0])
        send_blocks(msgs[1:], reply_markup={"inline_keyboard": [[{"text": "🏆 今日推薦", "callback_data": "c:推薦"}]]})
        return
    if cmd in ("review", "賽後", "檢討", "賽果"):
        import review
        hours = float(arg) if arg.replace(".", "", 1).isdigit() else 24
        head, *blocks = review.build(hours)
        send_telegram(head)
        send_blocks(blocks)
        return
    if cmd in ("比價", "運彩比價", "odds"):
        import oddscheck
        send_telegram(oddscheck.HOWTO)
        return
    if cmd in ("report", "成績", "戰績"):
        conn = db()
        try:
            live.weekly_report(conn, False)
        finally:
            conn.close()
        send_telegram(f"📊 <b>戰績網頁</b>（圖表、每張最推薦的結果，每天更新）\n{DASHBOARD_URL}")
        return

    games = live.fetch_games()
    import oddscheck
    if oddscheck.looks_like(t):          # 「隊名 + 賠率」→ 運彩比價
        send_blocks(oddscheck.report(games, t))
        return
    if cmd in ("football", "足球"):
        _football_picker(games)
        return
    league = "NBA" if cmd in ("nba", "籃球") else LEAGUE_WORDS.get(cmd) or LEAGUE_WORDS.get(t)
    if league:
        _league_view(games, league, {"NBA": 24, "MLB": 48}.get(league, 36))
        return

    query = arg if cmd in ("a", "analyze", "分析") else t
    found = analyze.search(games, query)
    if not found:
        send_telegram(f"找不到「{query}」未來 4 天內的比賽。\n可以按下面的按鈕，或輸入隊名（例：曼城、湖人、LAL）。",
                      reply_markup=MENU)
        return
    if len(found) > 1:   # 多場：先給按鈕讓使用者選
        rows = [[{"text": _game_label(g), "callback_data": f"g:{g['game_id']}"}] for g in found[:10]]
        send_telegram(f"找到 {len(found)} 場，點要看的比賽：", reply_markup={"inline_keyboard": rows})
        return
    _game_detail(found[0]["game_id"])


def handle_callback(data: str) -> None:
    kind, _, val = data.partition(":")
    if kind == "c":
        handle(val)
    elif kind == "lg":
        import live
        _league_view(live.fetch_games(), val, {"NBA": 24, "MLB": 48}.get(val, 36))
    elif kind == "g":
        _game_detail(val)
    elif kind == "r":
        _DETAIL.pop(val, None)
        _game_detail(val)


# ---------------------------------------------------------------- 接收
def poll(timeout: int = 20) -> None:
    """等待新訊息 / 按鈕最多 timeout 秒並處理；沒設定 Telegram 時就單純睡覺。"""
    token, chat_id = _telegram_credentials()
    if not token or not chat_id:
        time.sleep(timeout)
        return
    # 不在等待期間開著資料庫：Windows 上開著的檔案無法被雲端同步覆蓋。
    try:
        offset = int(OFFSET_FILE.read_text().strip() or 0) if OFFSET_FILE.exists() else 0
    except (OSError, ValueError):
        offset = 0
    try:
        resp = requests.get(f"https://api.telegram.org/bot{token}/getUpdates",
                            params={"offset": offset, "timeout": timeout,
                                    "allowed_updates": '["message","callback_query"]'},
                            timeout=timeout + 15)
        updates = resp.json().get("result", []) if resp.ok else []
        if not resp.ok:
            time.sleep(5)
        for u in updates:
            OFFSET_FILE.write_text(str(u["update_id"] + 1))  # 先記錄，避免出錯時無限重試同一則
            cq = u.get("callback_query")
            if cq:
                if str(((cq.get("message") or {}).get("chat") or {}).get("id")) != chat_id:
                    continue
                _api("answerCallbackQuery", callback_query_id=cq["id"], text="處理中…")
                data = cq.get("data") or ""
                log.info("按鈕：%s", data[:60])
                try:
                    handle_callback(data)
                except Exception as exc:
                    log.exception("按鈕處理失敗")
                    send_telegram(f"處理按鈕時發生錯誤：{exc}", parse_mode=None)
                continue
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
