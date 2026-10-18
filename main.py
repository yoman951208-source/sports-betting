# -*- coding: utf-8 -*-
"""
運彩分析機器人（足球 + NBA）→ Telegram
=====================================

  python main.py backtest [--send]     下載兩年歷史資料並回測，--send 把報告發到 Telegram
  python main.py run [--dry-run]       跑一輪即時監控（抓盤口、找訊號、傷兵、結算）
  python main.py digest [--dry-run]    立即發送每日賽事總覽
  python main.py report [--dry-run]    立即發送實戰成績
  python main.py oddsapi [--dry-run]   立即用 The Odds API 比價一次（需 ODDS_API_KEY）
  python main.py strategies            列出目前啟用的策略
  python main.py league 歐冠 [--hours 36]  分析整個聯賽／盃賽的比賽並發到 Telegram
  python main.py analyze 曼城           分析指定球隊的下一場比賽
  python main.py parlay 歐冠            串關建議（二串一、三串一）；全部 = 所有足球
  python main.py test-notify           發一則 Telegram 測試訊息
  python main.py loop                  完整本機常駐模式（不用雲端時）
  python main.py cloud                 雲端執行一次（GitHub Actions 每 30 分鐘呼叫）
  python main.py bot                   本機只回 Telegram 指令，資料每 5 分鐘從 GitHub 同步

免責聲明：本程式僅供研究，訊號不構成投注建議；任何博彩都有輸錢風險。
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback

import schedule

import config
from core import STRATEGY_FILE, db, log, send_blocks, send_telegram


def cmd_backtest(send: bool, dry_run: bool) -> None:
    import backtest
    old = {s.sid for s in backtest.load_strategies()[0] if s.enabled}
    strats = backtest.run(send=send, dry_run=dry_run)
    new = {s.sid for s in strats if s.enabled}
    if not dry_run:
        import calibration
        safe(calibration.refit)  # 市場機率校準層也每週用最新資料重新估計
    if not send and old and old != new:
        added, removed = new - old, old - new
        send_telegram("🔄 <b>每週重新回測：啟用策略有變動</b>\n"
                      + "".join(f"➕ {sid}\n" for sid in sorted(added))
                      + "".join(f"➖ {sid}\n" for sid in sorted(removed)), dry_run=dry_run)


def cmd_strategies() -> None:
    import backtest
    strats, meta = backtest.load_strategies()
    if not strats:
        print("尚未回測，請先執行：python main.py backtest")
        return
    print(f"回測時間：{meta.get('generated')}")
    for s in sorted(strats, key=lambda s: (not s.enabled, not s.weak, -s.stats.get("t", -9))):
        if s.enabled or s.weak:
            st = s.stats
            print(f"{'⭐' if s.enabled else '☆'} {s.sid:<26} n={st['n']:<5} ROI={st['roi']*100:+.1f}% "
                  f"p={st['p']:.3f} 即時資料={s.live_source}")


def send_review(hours: float, dry_run: bool) -> None:
    """每日盤後分析：足球 + MLB/NBA + 串關結果 + 模型學習（先結算、更新學習再發）。"""
    import review
    head, *blocks = review.daily(hours)
    send_telegram(head, dry_run=dry_run)
    send_blocks(blocks, dry_run=dry_run)


def send_daily_parlays(dry_run: bool) -> None:
    """每日串關：未來 24 小時全部（足球 + MLB + NBA），存進資料庫供賽後自動結算。"""
    import live
    import parlay
    conn = db(dry_run)
    try:
        head, *blocks = parlay.report(live.fetch_games(), None, 24, save_conn=conn)
        record = parlay.record_lines(conn, 30)[0]
    finally:
        conn.close()
    send_telegram(head + f"\n\n📒 近 30 天串關實戰：{record}", dry_run=dry_run)
    send_blocks(blocks, dry_run=dry_run)


def run_due_jobs(jobs: list[tuple[str, int | None, int, object]]) -> None:
    """每日工作的「補跑」機制：電腦睡著錯過時間，醒來後當天仍會補跑一次（用資料庫記錄當天是否做過）。"""
    from datetime import datetime
    from core import kv_get, kv_set
    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    conn = db()
    try:
        for name, hour, minute, fn in jobs:
            if hour is None or (now.hour, now.minute) < (hour, minute):
                continue
            if kv_get(conn, f"job:{name}") == today:
                continue
            kv_set(conn, f"job:{name}", today)   # 先記錄，避免失敗時每 20 秒重試
            log.info("執行每日工作：%s", name)
            safe(fn)
    finally:
        conn.close()


def safe(fn, *args):
    try:
        fn(*args)
    except Exception:  # 常駐模式不能因為單次錯誤整個停掉
        log.error("執行失敗：\n%s", traceback.format_exc())


def _digest() -> None:
    import live
    conn = db()
    try:
        live.daily_digest(conn, live.fetch_games(), live.load_strategies()[0], False)
    finally:
        conn.close()


def _weekly() -> None:
    from datetime import datetime
    import live
    if datetime.now().weekday() != config.WEEKLY_REPORT_WEEKDAY:
        return
    conn = db()
    try:
        live.weekly_report(conn, False)
    finally:
        conn.close()
    safe(cmd_backtest, False, False)  # 每週更新歷史資料並重新回測


def send_nba_preview() -> None:
    """NBA 賽前報告：未來 14 小時的每場比賽（盤口、主力近況與疲勞、傷兵、球員狀態調整後的合理賠率）。"""
    import analyze
    import live
    games = live.fetch_games()
    if not analyze.upcoming(games, 14, league="NBA"):
        return
    head, *details = analyze.league_report(games, "NBA", 14)
    send_telegram(head.replace("分析總表", "賽前報告（含主力近況、疲勞、傷兵）"))
    send_blocks(details)


def daily_jobs() -> list[tuple[str, int | None, int, object]]:
    """每日工作（名稱, 時, 分, 函式）：時間到了才跑；錯過就補跑，一天一次（台灣時間）。"""
    return [
        ("nba_preview", config.NBA_PREVIEW_HOUR, 0, send_nba_preview),
        ("review", config.DAILY_REVIEW_HOUR, 0, lambda: send_review(24, False)),
        ("digest", config.DAILY_DIGEST_HOUR, 0, _digest),
        ("parlay", config.DAILY_PARLAY_HOUR, config.DAILY_PARLAY_MINUTE, lambda: send_daily_parlays(False)),
        ("weekly", config.DAILY_DIGEST_HOUR, 5, _weekly),
    ]


def cloud_once() -> None:
    """雲端（GitHub Actions）每次執行：一輪監控 + 到期的每日工作。狀態由 workflow 存回 data 分支。"""
    import calibration
    import live
    from core import kv_get
    if not STRATEGY_FILE.exists():
        safe(cmd_backtest, False, False)
    conn = db()
    try:
        cal = kv_get(conn, "calibration") or ""
    finally:
        conn.close()
    if "nba_form_spread" not in cal:   # 沒有校準、或還沒有 10/19 新增的球員狀態係數 → 重新估計
        safe(calibration.refit)
    safe(live.run_cycle)
    run_due_jobs(daily_jobs())


def sync_state() -> bool:
    """從 GitHub data 分支下載雲端最新的資料庫與策略檔（本機只負責回答 Telegram 指令時用）。"""
    import os
    import requests
    from core import DATA_DIR
    ok = True
    for name in ("betting.db", "strategies.json"):
        # 優先用 GitHub API（不經 CDN 快取，raw.githubusercontent 可能慢 5 分鐘）；未登入每小時 60 次，夠用
        api = f"https://api.github.com/repos/{config.GITHUB_REPO}/contents/{name}?ref=data"
        raw = f"https://raw.githubusercontent.com/{config.GITHUB_REPO}/data/{name}"
        try:
            resp = requests.get(api, headers={"Accept": "application/vnd.github.raw"}, timeout=60)
            if not resp.ok:
                resp = requests.get(raw, timeout=60)
            if not resp.ok or not resp.content:
                ok = False
                continue
            tmp = DATA_DIR / f"{name}.tmp"
            tmp.write_bytes(resp.content)
            for attempt in range(5):  # Windows：檔案正被讀取時無法覆蓋，稍等重試
                try:
                    os.replace(tmp, DATA_DIR / name)
                    break
                except PermissionError:
                    if attempt == 4:
                        raise
                    time.sleep(1.5)
        except (requests.RequestException, OSError) as exc:
            log.warning("同步 %s 失敗：%s", name, exc)
            ok = False
    return ok


def bot_mode() -> None:
    """本機「只回指令」模式：監控與每日工作由雲端負責；每 5 分鐘同步一次雲端資料庫。"""
    import bot
    log.info("本機指令模式啟動：監控與每日報告在 GitHub Actions 執行，這裡只回答 Telegram 指令")
    last = 0.0
    while True:
        if time.time() - last > 300:
            sync_state()
            last = time.time()
        bot.poll(timeout=20)


def loop() -> None:
    """完整本機模式（不用雲端時）：監控 + 每日工作 + 回答指令。"""
    import live
    schedule.every(config.CHECK_INTERVAL_MINUTES).minutes.do(safe, live.run_cycle)
    if not STRATEGY_FILE.exists():
        safe(cmd_backtest, False, False)
    log.info("常駐模式啟動：每 %d 分鐘監控一次；盤後分析 %s:00、總覽 %02d:00、串關 %02d:%02d（錯過會補跑）",
             config.CHECK_INTERVAL_MINUTES, config.DAILY_REVIEW_HOUR, config.DAILY_DIGEST_HOUR,
             config.DAILY_PARLAY_HOUR, config.DAILY_PARLAY_MINUTE)
    safe(live.run_cycle)
    import bot
    jobs = daily_jobs()
    while True:
        schedule.run_pending()
        run_due_jobs(jobs)
        bot.poll(timeout=20)  # 等 Telegram 指令最多 20 秒，順便當作迴圈間隔


def main() -> int:
    ap = argparse.ArgumentParser(description="運彩分析機器人（足球 + NBA）")
    ap.add_argument("command", choices=["backtest", "run", "digest", "report", "oddsapi", "strategies", "test-notify", "loop",
                                        "analyze", "league", "review", "parlay", "cloud", "bot", "sync"])
    ap.add_argument("query", nargs="?", default="", help="analyze：隊名；league：聯賽名稱（如 歐冠）")
    ap.add_argument("--hours", type=float, default=36, help="league：分析未來幾小時的比賽")
    ap.add_argument("--send", action="store_true", help="backtest：把報告發到 Telegram")
    ap.add_argument("--dry-run", action="store_true", help="只印出訊息，不實際發送 Telegram")
    a = ap.parse_args()

    if a.command == "backtest":
        cmd_backtest(a.send, a.dry_run)
    elif a.command == "strategies":
        cmd_strategies()
    elif a.command == "test-notify":
        ok = send_telegram("✅ 運彩分析機器人連線測試成功（足球 + NBA）", dry_run=a.dry_run)
        print("發送成功" if ok else "發送失敗，請檢查 .env")
        return 0 if ok else 1
    elif a.command == "cloud":
        cloud_once()
    elif a.command == "bot":
        bot_mode()
    elif a.command == "sync":
        print("同步成功" if sync_state() else "同步失敗（data 分支可能還不存在）")
    elif a.command == "loop":
        loop()
    else:
        import live
        if a.command == "run":
            live.run_cycle(dry_run=a.dry_run)
            return 0
        if a.command == "parlay":
            import parlay
            league = None if a.query == "全部" else (a.query or "歐冠")
            head, *blocks = parlay.report(live.fetch_games(), league, a.hours if a.hours != 36 else 24)
            send_telegram(head, dry_run=a.dry_run)
            send_blocks(blocks, dry_run=a.dry_run)
            return 0
        if a.command == "review":
            send_review(a.hours if a.hours != 36 else 18, a.dry_run)
            return 0
        if a.command in ("analyze", "league"):
            import analyze
            games = live.fetch_games()
            if a.command == "league":
                head, *details = analyze.league_report(games, a.query or "歐冠", a.hours)
                send_telegram(head, dry_run=a.dry_run)
                send_blocks(details, dry_run=a.dry_run)
            else:
                found = analyze.search(games, a.query)
                send_blocks([analyze.analyze_game(g)[0] for g in found[:5]] or [f"找不到「{a.query}」的比賽"],
                            dry_run=a.dry_run)
            return 0
        conn = db(a.dry_run)
        try:
            if a.command == "digest":
                live.daily_digest(conn, live.fetch_games(), live.load_strategies()[0], a.dry_run)
            elif a.command == "report":
                live.weekly_report(conn, a.dry_run)
            elif a.command == "oddsapi":
                if not config.ODDS_API_KEY:
                    print("未設定 ODDS_API_KEY（.env），請先到 https://the-odds-api.com/ 免費註冊")
                    return 1
                games = live.fetch_games()
                live.save_games(conn, games)
                n = live.oddsapi_sweep(conn, games, live.load_strategies()[0], a.dry_run, force=True)
                print(f"找到 {n} 個新的價值選項")
        finally:
            conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
