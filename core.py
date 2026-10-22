# -*- coding: utf-8 -*-
"""共用工具：設定載入、HTTP（含快取）、賠率換算、Telegram 推播、SQLite。"""
from __future__ import annotations

import json
import logging
import math
import os
import shutil
import sqlite3
import sys
import time
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

try:  # 選用：從同目錄 .env 讀取金鑰
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:  # 選用：改用作業系統憑證庫（Norton 等防毒的 HTTPS 掃描會讓 certifi 驗證失敗）
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
DB_PATH = DATA_DIR / "betting.db"
STRATEGY_FILE = DATA_DIR / "strategies.json"
TPE = ZoneInfo("Asia/Taipei")

DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
if load_dotenv is not None:
    load_dotenv(BASE_DIR / ".env")


# -----------------------------------------------------------------------------
# Log
# -----------------------------------------------------------------------------
def _setup_log() -> logging.Logger:
    logger = logging.getLogger("betting")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = RotatingFileHandler(BASE_DIR / "betting.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    if sys.stdout is not None:  # pythonw 執行時沒有 stdout
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    return logger


log = _setup_log()


# -----------------------------------------------------------------------------
# HTTP
# -----------------------------------------------------------------------------
_session = requests.Session()
_session.headers["User-Agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) sports-betting-research/1.0"


def http_get(url: str, params: dict | None = None, retries: int = 3, timeout: int = 20) -> requests.Response | None:
    """GET 並重試；4xx（429 除外）直接放棄，回傳 None。"""
    for attempt in range(1, retries + 1):
        try:
            resp = _session.get(url, params=params, timeout=timeout)
            if resp.ok:
                return resp
            if resp.status_code == 429:
                time.sleep(5 * attempt)
                continue
            if 400 <= resp.status_code < 500:
                log.debug("HTTP %s %s", resp.status_code, url)
                return None
        except requests.RequestException as exc:
            log.debug("連線失敗（第 %d 次）%s：%s", attempt, url, exc)
        time.sleep(1.5 * attempt)
    return None


def get_json(url: str, params: dict | None = None, cache_name: str | None = None,
             max_age_hours: float | None = None) -> dict | None:
    """取得 JSON。給 cache_name 時會寫入 data/cache；max_age_hours=None 代表永久有效。"""
    path = CACHE_DIR / cache_name if cache_name else None
    if path and path.exists():
        age_h = (time.time() - path.stat().st_mtime) / 3600
        if max_age_hours is None or age_h <= max_age_hours:
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except ValueError:
                pass
    resp = http_get(url, params=params)
    if resp is None:
        return None
    try:
        data = resp.json()
    except ValueError:
        return None
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data


# -----------------------------------------------------------------------------
# 賠率換算
# -----------------------------------------------------------------------------
def american_to_decimal(value) -> float | None:
    """美式賠率（-110 / +150 / "EVEN"）→ 歐式小數賠率。"""
    if value is None:
        return None
    s = str(value).strip().upper()
    if s in ("", "OFF", "NONE", "NAN"):
        return None
    if s in ("EVEN", "EV", "PK"):
        return 2.0
    try:
        a = float(s)
    except ValueError:
        return None
    if a == 0:
        return None
    return round(1 + a / 100, 4) if a > 0 else round(1 + 100 / abs(a), 4)


def parse_line(value) -> float | None:
    """'o232.5' / 'u2.5' / '+1.5' / '-2.5' / 'PK' → 浮點盤口。"""
    if value is None:
        return None
    s = str(value).strip().lower().lstrip("ou")
    if s in ("pk", "even"):
        return 0.0
    try:
        return float(s)
    except ValueError:
        return None


def no_vig(odds: list[float | None]) -> list[float] | None:
    """去除莊家抽水後的機率（等比例法）。任一賠率缺漏回傳 None。"""
    if not odds or any(o is None or not isinstance(o, (int, float)) or math.isnan(o) or o <= 1 for o in odds):
        return None
    inv = [1 / o for o in odds]
    s = sum(inv)
    return [x / s for x in inv]


def overround(odds: list[float]) -> float:
    return sum(1 / o for o in odds) - 1


def fmt_odds(o: float | None) -> str:
    return "-" if o is None else f"{o:.2f}"


def fmt_pct(x: float | None, digits: int = 1, sign: bool = True) -> str:
    if x is None:
        return "-"
    return f"{x * 100:+.{digits}f}%" if sign else f"{x * 100:.{digits}f}%"


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def to_tpe(dt: datetime) -> datetime:
    return dt.astimezone(TPE)


def parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


# -----------------------------------------------------------------------------
# Telegram
# -----------------------------------------------------------------------------
TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"


def _telegram_credentials() -> tuple[str, str]:
    return os.getenv("TELEGRAM_BOT_TOKEN", "").strip(), os.getenv("TELEGRAM_CHAT_ID", "").strip()


def _split_message(text: str, limit: int = 4000) -> list[str]:
    """超過 Telegram 上限時依換行切段。"""
    if len(text) <= limit:
        return [text]
    parts, buf = [], ""
    for line in text.split("\n"):
        if len(buf) + len(line) + 1 > limit and buf:
            parts.append(buf)
            buf = ""
        buf += line + "\n"
    if buf.strip():
        parts.append(buf)
    return parts


def send_telegram(text: str, parse_mode: str | None = "HTML", dry_run: bool = False, retries: int = 3,
                  reply_markup: dict | None = None) -> bool:
    """發送訊息（自動分段）。reply_markup = Telegram 按鈕（附在最後一段）。dry_run 時只印出來。"""
    if dry_run:
        print("=" * 60 + "\n" + text + "\n" + "=" * 60)
        if reply_markup:
            print("[按鈕]", reply_markup)
        return True
    token, chat_id = _telegram_credentials()
    if not token or not chat_id:
        log.error("未設定 TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID（.env），無法發送")
        return False
    ok = True
    chunks = _split_message(text)
    for i, chunk in enumerate(chunks):
        ok &= _send_one(chunk, token, chat_id, parse_mode, retries, reply_markup if i == len(chunks) - 1 else None)
    return ok


BLOCK_SEP = "\n\n" + "━" * 12 + "\n\n"


def send_blocks(blocks: list[str], sep: str = BLOCK_SEP, dry_run: bool = False, reply_markup: dict | None = None) -> bool:
    """把多段內容盡量塞進同一則訊息（每則 ≤4000 字），避免洗版又不把一段拆開。"""
    msgs, buf = [], ""
    for b in blocks:
        if buf and len(buf) + len(sep) + len(b) > 4000:
            msgs.append(buf)
            buf = ""
        buf = f"{buf}{sep}{b}" if buf else b
    if buf:
        msgs.append(buf)
    return all([send_telegram(m, dry_run=dry_run, reply_markup=reply_markup if i == len(msgs) - 1 else None)
                for i, m in enumerate(msgs)])


def _send_one(text: str, token: str, chat_id: str, parse_mode: str | None, retries: int,
              reply_markup: dict | None = None) -> bool:
    url = TELEGRAM_API.format(token=token, method="sendMessage")
    payload = {"chat_id": chat_id, "text": text, "disable_web_page_preview": True}
    if parse_mode:
        payload["parse_mode"] = parse_mode
    if reply_markup:
        payload["reply_markup"] = reply_markup
    for attempt in range(1, retries + 1):
        wait = 2 * attempt
        try:
            resp = requests.post(url, json=payload, timeout=15)
            data = resp.json()
            if resp.ok and data.get("ok"):
                return True
            desc = data.get("description", "")
            if resp.status_code == 429:
                wait = data.get("parameters", {}).get("retry_after", wait)
            elif resp.status_code == 400 and "parse" in desc and "parse_mode" in payload:
                log.warning("訊息格式解析失敗，改以純文字重送：%s", desc)
                payload.pop("parse_mode")
                continue
            elif 400 <= resp.status_code < 500:
                log.error("Telegram 拒絕請求 (%s)：%s", resp.status_code, desc)
                return False
        except (requests.RequestException, ValueError) as exc:
            log.warning("Telegram 連線失敗（第 %d 次）：%s", attempt, str(exc).replace(token, "***"))
        if attempt < retries:
            time.sleep(wait)
    return False


# -----------------------------------------------------------------------------
# SQLite
# -----------------------------------------------------------------------------
SCHEMA = """
CREATE TABLE IF NOT EXISTS games (
    game_id     TEXT PRIMARY KEY,      -- espn:<league>:<event id>
    sport       TEXT, league TEXT, kickoff TEXT,
    home TEXT, away TEXT, home_abbr TEXT, away_abbr TEXT,
    status TEXT, home_score INTEGER, away_score INTEGER, updated TEXT
);
CREATE TABLE IF NOT EXISTS snapshots (
    game_id TEXT, ts TEXT, source TEXT,
    ml_home REAL, ml_draw REAL, ml_away REAL,
    spread_home REAL, spread_home_odds REAL, spread_away_odds REAL,
    total REAL, over_odds REAL, under_odds REAL,
    open_json TEXT,
    PRIMARY KEY (game_id, ts, source)
);
CREATE TABLE IF NOT EXISTS alerts (
    alert_key TEXT PRIMARY KEY,         -- 去重用
    game_id TEXT, sport TEXT, strategy TEXT, market TEXT, pick TEXT,
    line REAL, odds REAL, p_mkt REAL, sent_at TEXT, tier TEXT,   -- tier: signal / early / value
    close_line REAL, close_odds REAL,   -- 開賽前最後看到的盤口（算 CLV 用）
    result TEXT, profit REAL, settled_at TEXT
);
CREATE TABLE IF NOT EXISTS injuries (
    player_id TEXT PRIMARY KEY, team TEXT, name TEXT, status TEXT, comment TEXT, updated TEXT
);
CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT);
"""


def db(dry_run: bool = False) -> sqlite3.Connection:
    """dry_run 時改用正式資料庫的暫存複本，測試不會影響之後的去重與成績紀錄。"""
    path = DB_PATH
    if dry_run:
        path = DATA_DIR / "dryrun.db"
        if DB_PATH.exists():
            shutil.copyfile(DB_PATH, path)
        elif path.exists():
            path.unlink()
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def kv_get(conn: sqlite3.Connection, key: str, default: str | None = None) -> str | None:
    row = conn.execute("SELECT v FROM kv WHERE k=?", (key,)).fetchone()
    return row["v"] if row else default


def kv_set(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute("INSERT OR REPLACE INTO kv(k, v) VALUES (?, ?)", (key, value))
    conn.commit()
