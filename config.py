# -*- coding: utf-8 -*-
"""設定區 —— 一般只需要改這裡。"""
import os

# 足球聯賽：football-data.co.uk 代碼 → (ESPN 代碼, 中文名稱)
# 歷史回測用 football-data（多家博彩公司開盤/收盤賠率 + Betfair 交易所），即時監控用 ESPN。
FOOTBALL_LEAGUES = {
    "E0":  ("eng.1", "英超"),
    "E1":  ("eng.2", "英冠"),
    "SP1": ("esp.1", "西甲"),
    "I1":  ("ita.1", "義甲"),
    "D1":  ("ger.1", "德甲"),
    "F1":  ("fra.1", "法甲"),
    "N1":  ("ned.1", "荷甲"),
    "P1":  ("por.1", "葡超"),
    "SC0": ("sco.1", "蘇超"),
    "B1":  ("bel.1", "比甲"),
    "T1":  ("tur.1", "土超"),
}
# 只有即時監控、沒有歷史資料的盃賽（套用從聯賽學到的盤口規律，訊息會註明）
FOOTBALL_CUPS = {
    "uefa.champions": "歐冠",
    "uefa.europa": "歐霸",
}
# 回測季別（football-data 的季別代碼）。10/23 起往前加到 2019-20（football-data 從這季開始有收盤、亞盤、大小盤）
FOOTBALL_SEASONS = ["1920", "2021", "2122", "2223", "2324", "2425", "2526", "2627"]

# NBA 回測季別：(季別名稱, 開始日, 結束日) —— 只取例行賽 + 季後賽
# 2016-17 ~ 2022-23 只有收盤（ESPN 沒存開盤）：校準、歷史相似盤口、收盤類回測會用到；盤口變化類只用 2023-24 起
NBA_SEASONS = [
    ("2016-17", "2016-10-25", "2017-06-12"),
    ("2017-18", "2017-10-17", "2018-06-08"),
    ("2018-19", "2018-10-16", "2019-06-13"),
    ("2019-20", "2019-10-22", "2020-10-11"),
    ("2020-21", "2020-12-22", "2021-07-20"),
    ("2021-22", "2021-10-19", "2022-06-16"),
    ("2022-23", "2022-10-18", "2023-06-12"),
    ("2023-24", "2023-10-24", "2024-06-17"),
    ("2024-25", "2024-10-22", "2025-06-23"),
    ("2025-26", "2025-10-21", "2026-06-21"),
    ("2026-27", "2026-10-20", "2027-06-30"),   # 本季：已完賽的比賽會逐日加入（球員近況、回測都會用到）
]
NBA_PREVIEW_HOUR = 6                 # 每天幾點發 NBA 賽前報告（台灣時間；NBA 多在 07:00–11:00 開打）；None = 不發

# ---------------- 即時監控 ----------------
CHECK_INTERVAL_MINUTES = 30          # 常駐模式多久抓一次盤口
FOOTBALL_LOOKAHEAD_DAYS = 3          # 監控未來幾天的足球賽
NBA_LOOKAHEAD_DAYS = 2
# 回測策略以「收盤賠率」下注，所以訊號要在接近開賽時發（此時盤口最接近收盤）
SIGNAL_WINDOW_HOURS = {"football": 4, "nba": 3, "mlb": 3}
# 提早預警：距開賽超過上面時間、但盤口已大幅變動時，先發「觀察」訊息
EARLY_STEAM_ALERT = True
DAILY_DIGEST_HOUR = 12               # 每天幾點（台灣時間）發當日賽事總覽
DAILY_REVIEW_HOUR = 10               # 每天幾點發盤後分析（足球+MLB/NBA+串關結果+模型學習）；None = 不自動發
DAILY_PARLAY_HOUR = 12               # 每天幾點發串關建議（未來 24 小時），會存檔供賽後自動結算
DAILY_PARLAY_MINUTE = 30
WEEKLY_REPORT_WEEKDAY = 0            # 每週一發實戰成績（0=週一）

# ---------------- 策略篩選（回測 → 實戰） ----------------
# 一個策略要同時滿足才會被標為「⭐ 啟用」並即時推播：
MIN_BETS = {"football": 150, "nba": 80, "mlb": 150}   # 樣本數下限
MAX_P_VALUE = 0.10                         # 統計顯著性（單尾）
REQUIRE_POSITIVE_EACH_SEASON = True        # 每一季 ROI 都要 > 0（避免只是某一季運氣好）
# 沒過門檻、但整體 ROI > 0 的策略只放在每日總覽當「參考」，不即時推播
SHOW_WEAK_SIGNALS_IN_DIGEST = True

# ---------------- 傷兵（NBA） ----------------
INJURY_ALERT_STATUSES = {"Out", "Doubtful"}   # 主力球員變成這些狀態才通知
INJURY_KEY_PLAYER_MIN_PPG = 15.0              # 場均得分 ≥ 此值視為主力（約每隊 2–4 人）
INJURY_LOOKAHEAD_HOURS = 36                   # 該隊在這段時間內有比賽才通知

# ---------------- 選用：The Odds API（多家博彩公司即時賠率比價） ----------------
# 免費方案每月 500 credits：https://the-odds-api.com/ 註冊後把金鑰填進 .env 的 ODDS_API_KEY
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "").strip()
ODDS_API_SPORTS = {
    "soccer_epl": "英超",
    "soccer_spain_la_liga": "西甲",
    "soccer_italy_serie_a": "義甲",
    "soccer_germany_bundesliga": "德甲",
    "soccer_france_ligue_one": "法甲",
    "soccer_uefa_champs_league": "歐冠",
    "basketball_nba": "NBA",
}
ODDS_API_SWEEPS_PER_DAY = 2          # 7 個聯賽 × 2 次/天 × 30 天 ≈ 420 credits
ODDS_API_VALUE_EDGE = 0.04           # 博彩公司賠率比 Pinnacle/Betfair 公平賠率高 4% 以上才通知

MLB_LOOKAHEAD_DAYS = 2

MLB_TEAM_ZH = {
    "ARI": "響尾蛇", "ATL": "勇士", "BAL": "金鶯", "BOS": "紅襪", "CHC": "小熊", "CHW": "白襪", "CIN": "紅人",
    "CLE": "守護者", "COL": "洛磯", "DET": "老虎", "HOU": "太空人", "KC": "皇家", "LAA": "天使", "LAD": "道奇",
    "MIA": "馬林魚", "MIL": "釀酒人", "MIN": "雙城", "NYM": "大都會", "NYY": "洋基", "ATH": "運動家", "OAK": "運動家",
    "PHI": "費城人", "PIT": "海盜", "SD": "教士", "SF": "巨人", "SEA": "水手", "STL": "紅雀", "TB": "光芒",
    "TEX": "遊騎兵", "TOR": "藍鳥", "WSH": "國民",
}

NBA_TEAM_ZH = {
    "ATL": "老鷹", "BOS": "塞爾提克", "BKN": "籃網", "CHA": "黃蜂", "CHI": "公牛", "CLE": "騎士",
    "DAL": "獨行俠", "DEN": "金塊", "DET": "活塞", "GS": "勇士", "HOU": "火箭", "IND": "溜馬",
    "LAC": "快艇", "LAL": "湖人", "MEM": "灰熊", "MIA": "熱火", "MIL": "公鹿", "MIN": "灰狼",
    "NO": "鵜鶘", "NY": "尼克", "OKC": "雷霆", "ORL": "魔術", "PHI": "76人", "PHX": "太陽",
    "POR": "拓荒者", "SAC": "國王", "SA": "馬刺", "TOR": "暴龍", "UTAH": "爵士", "WSH": "巫師",
}

# 雲端：GitHub Actions 每 30 分鐘執行，資料庫存在這個 repo 的 data 分支；本機 bot 模式從這裡同步
GITHUB_REPO = "yoman951208-source/sports-betting"
