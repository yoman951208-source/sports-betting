# -*- coding: utf-8 -*-
"""戰績儀表板（10/24）：把資料庫裡的最推薦、串關、推播訊號戰績做成一個靜態網頁（GitHub Pages）。

python main.py dashboard  → site/index.html（資料直接嵌在網頁裡，手機打開就能看）
由 .github/workflows/dashboard.yml 每天自動更新並發布。
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import timedelta
from html import escape

from core import DATA_DIR, db, now_utc, parse_iso, to_tpe

SITE = DATA_DIR.parent / "site"
SERIES = [("best1", "最推薦單關"), ("best2", "最推薦二串一"), ("best3", "最推薦三串一"),
          ("daily", "每日串關"), ("signal", "⭐ 推播訊號")]
TIER = {"signal": "⭐ 訊號", "early": "👀 預警", "value": "💰 比價"}


def _series_of_parlay(label: str) -> str:
    return {"最推薦｜單關": "best1", "最推薦｜二串一": "best2", "最推薦｜三串一": "best3"}.get(label, "daily")


def _mark(profit) -> str:
    p = profit or 0
    return "✅" if p > 0 else "❌" if p < 0 else "➖"


def collect(conn) -> dict:
    bets = defaultdict(list)          # series → [(結算時間, 損益)]
    tickets = []
    for r in conn.execute("SELECT * FROM parlays WHERE result IN ('W','L','P') ORDER BY settled"):
        s = _series_of_parlay(r["label"])
        t = r["settled"] or r["created"]
        bets[s].append((t, r["profit"]))
        if s != "daily":
            legs = json.loads(r["legs"])
            tickets.append({"date": f"{to_tpe(parse_iso(r['created'])):%m/%d}", "kind": r["label"].split("｜")[-1],
                            "legs": " ＋ ".join(f"{l.get('name', '')}（{l.get('title', '')}）" for l in legs),
                            "odds": round(r["odds"] or 0, 2), "result": _mark(r["profit"]), "profit": round(r["profit"], 2)})
    alerts = []
    for r in conn.execute("SELECT * FROM alerts WHERE result IN ('W','L','P') ORDER BY settled_at"):
        if r["tier"] == "signal":
            bets["signal"].append((r["settled_at"] or r["sent_at"], r["profit"]))
        alerts.append({"date": f"{to_tpe(parse_iso(r['sent_at'])):%m/%d}", "tier": TIER.get(r["tier"], r["tier"]),
                       "strategy": r["strategy"], "pick": f"{r['market']} {r['pick']}" + (f" {r['line']:+g}" if r["line"] is not None else ""),
                       "odds": round(r["odds"] or 0, 2), "result": _mark(r["profit"]), "profit": round(r["profit"] or 0, 2)})
    # 累計損益（每天一點）
    days = sorted({f"{to_tpe(parse_iso(t)):%Y-%m-%d}" for v in bets.values() for t, _ in v})
    curves = {}
    for key, _ in SERIES:
        per_day = defaultdict(float)
        for t, p in bets.get(key, []):
            per_day[f"{to_tpe(parse_iso(t)):%Y-%m-%d}"] += p or 0
        run, pts = 0.0, []
        for d in days:
            run += per_day.get(d, 0.0)
            pts.append(round(run, 2))
        curves[key] = pts
    summary = []
    for key, name in SERIES:
        v = [p or 0 for _, p in bets.get(key, [])]
        summary.append({"key": key, "name": name, "n": len(v), "hits": sum(p > 0 for p in v),
                        "units": round(sum(v), 2), "roi": round(sum(v) / len(v) * 100, 1) if v else None})
    tier_rows = []
    for tier, name in TIER.items():
        v = [a["profit"] for a in alerts if a["tier"] == name]
        if v:
            tier_rows.append({"name": name, "n": len(v), "hits": sum(p > 0 for p in v), "units": round(sum(v), 2),
                              "roi": round(sum(v) / len(v) * 100, 1)})
    calib = []
    rows = conn.execute("SELECT p_model, result FROM predictions WHERE result IN ('W','L')").fetchall()
    for lo, hi in ((0, 0.2), (0.2, 0.35), (0.35, 0.5), (0.5, 0.65), (0.65, 0.8), (0.8, 1.01)):
        sel = [r for r in rows if lo <= r["p_model"] < hi]
        if len(sel) >= 10:
            calib.append({"band": f"{lo*100:.0f}–{min(hi, 1)*100:.0f}%", "n": len(sel),
                          "model": round(sum(r["p_model"] for r in sel) / len(sel) * 100, 1),
                          "actual": round(sum(r["result"] == "W" for r in sel) / len(sel) * 100, 1)})
    return {"updated": f"{to_tpe(now_utc()):%Y-%m-%d %H:%M}", "days": days, "curves": curves,
            "series": [{"key": k, "name": n} for k, n in SERIES], "summary": summary, "tiers": tier_rows,
            "tickets": tickets[-30:][::-1], "alerts": alerts[-30:][::-1], "calib": calib}


def build() -> str:
    conn = db()
    try:
        data = collect(conn)
    finally:
        conn.close()
    SITE.mkdir(exist_ok=True)
    html = TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
    path = SITE / "index.html"
    path.write_text(html, encoding="utf-8")
    return str(path)


TEMPLATE = r"""<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>運彩分析戰績</title>
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface-1: #fcfcfb; --text-primary: #0b0b0b; --text-secondary: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10); --good: #006300; --bad: #d03b3b;
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #eda100; --s5: #e87ba4;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --page: #0d0d0d; --surface-1: #1a1a19; --text-primary: #f0efec; --text-secondary: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10); --good: #0ca30c; --bad: #e66767;
    --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --s5: #d55181;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface-1: #1a1a19; --text-primary: #f0efec; --text-secondary: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10); --good: #0ca30c; --bad: #e66767;
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --s5: #d55181;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--text-primary);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", "Noto Sans TC", sans-serif; }
main { max-width: 980px; margin: 0 auto; padding: 20px 16px 48px; }
h1 { font-size: 22px; margin: 0 0 2px; }
h2 { font-size: 17px; margin: 28px 0 10px; }
.sub { color: var(--text-secondary); font-size: 13px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin-top: 16px; }
.tile { background: var(--surface-1); border: 1px solid var(--border); border-radius: 12px; padding: 12px 14px; }
.tile .name { color: var(--text-secondary); font-size: 13px; display: flex; align-items: center; gap: 6px; }
.tile .val { font-size: 24px; font-weight: 650; margin-top: 2px; }
.tile .meta { color: var(--muted); font-size: 12px; }
.dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; flex: none; }
.card { background: var(--surface-1); border: 1px solid var(--border); border-radius: 12px; padding: 14px; }
.chart-wrap { position: relative; height: 300px; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 14px; margin: 0 0 10px; font-size: 13px; color: var(--text-secondary); }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.scroll { overflow-x: auto; -webkit-overflow-scrolling: touch; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { text-align: left; padding: 7px 8px; border-bottom: 1px solid var(--grid); white-space: nowrap; }
td.wrap { white-space: normal; min-width: 220px; }
th { color: var(--text-secondary); font-weight: 600; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
.pos { color: var(--good); } .neg { color: var(--bad); }
.note { color: var(--muted); font-size: 12px; margin-top: 24px; }
#tip { position: absolute; pointer-events: none; background: var(--surface-1); border: 1px solid var(--border);
  border-radius: 8px; padding: 8px 10px; font-size: 12px; box-shadow: 0 4px 14px rgba(0,0,0,.12); display: none; z-index: 2; }
#tip .row { display: flex; align-items: center; gap: 6px; }
</style>
</head>
<body>
<main>
  <h1>運彩分析戰績</h1>
  <div class="sub" id="updated"></div>
  <div class="tiles" id="tiles"></div>

  <h2>累計損益（每注 1 單位，DK 賠率）</h2>
  <div class="card">
    <div class="legend" id="legend"></div>
    <div class="chart-wrap"><svg id="chart" width="100%" height="100%" role="img" aria-label="累計損益折線圖"></svg><div id="tip"></div></div>
    <div class="scroll" style="margin-top:10px"><table id="sumtable"></table></div>
  </div>

  <h2>推播訊號</h2>
  <div class="card scroll"><table id="tiers"></table></div>

  <h2>最近的最推薦</h2>
  <div class="card scroll"><table id="tickets"></table></div>

  <h2>最近結算的推播</h2>
  <div class="card scroll"><table id="alerts"></table></div>

  <h2>模型準不準（說 X% 的事，實際發生幾成）</h2>
  <div class="card scroll"><table id="calib"></table></div>

  <p class="note">研究用途，非投注建議。損益以 DK 賠率、每注 1 單位計算；運彩賠率通常較低，實際報酬會更差。</p>
</main>
<script>
const D = __DATA__;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const color = i => css(`--s${i + 1}`);
const fmt = v => (v > 0 ? "+" : "") + v.toFixed(2);
const cls = v => v > 0 ? "pos" : v < 0 ? "neg" : "";
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
document.getElementById("updated").textContent = `更新時間：${D.updated}（台灣時間）`;

function tiles() {
  document.getElementById("tiles").innerHTML = D.summary.map((s, i) => `
    <div class="tile"><div class="name"><span class="dot" style="background:${color(i)}"></span>${esc(s.name)}</div>
    <div class="val ${cls(s.units)}">${s.n ? fmt(s.units) + " 注" : "—"}</div>
    <div class="meta">${s.n ? `${s.n} 張｜中 ${s.hits}｜ROI ${s.roi > 0 ? "+" : ""}${s.roi}%` : "尚無結算"}</div></div>`).join("");
}

function table(id, head, rows) {
  document.getElementById(id).innerHTML = `<thead><tr>${head.map(h => `<th class="${h[1] || ""}">${h[0]}</th>`).join("")}</tr></thead>`
    + `<tbody>${rows.length ? rows.join("") : `<tr><td colspan="${head.length}" style="color:var(--muted)">尚無資料</td></tr>`}</tbody>`;
}

function chart() {
  const svg = document.getElementById("chart"), tip = document.getElementById("tip");
  const box = svg.parentElement.getBoundingClientRect(), W = box.width, H = box.height, m = {l: 44, r: 12, t: 10, b: 26};
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const days = D.days, n = days.length;
  const live = D.series.map((s, i) => ({...s, i, pts: D.curves[s.key]})).filter(s => D.summary[s.i].n > 0);
  if (!n || !live.length) { svg.innerHTML = `<text x="${W/2}" y="${H/2}" text-anchor="middle" fill="${css("--muted")}" font-size="13">尚無已結算的紀錄</text>`; return; }
  const all = live.flatMap(s => s.pts).concat([0]);
  let lo = Math.min(...all), hi = Math.max(...all); if (hi - lo < 2) { hi += 1; lo -= 1; }
  const pad = (hi - lo) * 0.08; lo -= pad; hi += pad;
  const x = k => m.l + (n === 1 ? (W - m.l - m.r) / 2 : k * (W - m.l - m.r) / (n - 1));
  const y = v => m.t + (hi - v) * (H - m.t - m.b) / (hi - lo);
  const step = Math.pow(10, Math.floor(Math.log10((hi - lo) / 4))); const ticks = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) ticks.push(v);
  const thin = ticks.length > 7 ? ticks.filter((_, k) => k % 2 === 0) : ticks;
  let g = thin.map(v => `<line x1="${m.l}" x2="${W - m.r}" y1="${y(v)}" y2="${y(v)}" stroke="${Math.abs(v) < 1e-9 ? css("--axis") : css("--grid")}" stroke-width="1"/>
    <text x="${m.l - 6}" y="${y(v) + 4}" text-anchor="end" font-size="11" fill="${css("--muted")}">${+v.toFixed(2)}</text>`).join("");
  const every = Math.max(1, Math.ceil(n / 6));
  const anchor = k => n > 1 && k === n - 1 ? "end" : n > 1 && k === 0 ? "start" : "middle";
  g += days.map((d, k) => (k % every === 0 && n - 1 - k >= every / 2) || k === n - 1 ? `<text x="${x(k)}" y="${H - 6}" text-anchor="${anchor(k)}" font-size="11" fill="${css("--muted")}">${d.slice(5)}</text>` : "").join("");
  g += live.map(s => `<polyline fill="none" stroke="${color(s.i)}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"
    points="${s.pts.map((v, k) => `${x(k)},${y(v)}`).join(" ")}"/>`).join("");
  g += `<line id="cross" y1="${m.t}" y2="${H - m.b}" stroke="${css("--axis")}" stroke-width="1" visibility="hidden"/>`;
  g += live.map(s => `<circle class="hov" data-i="${s.i}" r="4" fill="${color(s.i)}" stroke="${css("--surface-1")}" stroke-width="2" visibility="hidden"/>`).join("");
  g += `<rect x="${m.l}" y="0" width="${W - m.l - m.r}" height="${H}" fill="transparent" id="hit"/>`;
  svg.innerHTML = g;
  const hit = svg.querySelector("#hit"), cross = svg.querySelector("#cross"), dots = svg.querySelectorAll(".hov");
  const show = ev => {
    const r = svg.getBoundingClientRect(), px = (ev.touches ? ev.touches[0].clientX : ev.clientX) - r.left;
    const k = Math.max(0, Math.min(n - 1, Math.round(n === 1 ? 0 : (px - m.l) / ((W - m.l - m.r) / (n - 1)))));
    cross.setAttribute("x1", x(k)); cross.setAttribute("x2", x(k)); cross.setAttribute("visibility", "visible");
    dots.forEach(c => { const s = live.find(z => z.i == c.dataset.i); c.setAttribute("cx", x(k)); c.setAttribute("cy", y(s.pts[k])); c.setAttribute("visibility", "visible"); });
    tip.innerHTML = `<div style="color:${css("--text-secondary")};margin-bottom:4px">${days[k]}</div>` +
      live.slice().sort((a, b) => b.pts[k] - a.pts[k]).map(s => `<div class="row"><span class="dot" style="background:${color(s.i)}"></span>${esc(s.name)}　<b>${fmt(s.pts[k])}</b></div>`).join("");
    tip.style.display = "block";
    const tx = x(k) + 12 + tip.offsetWidth > W ? x(k) - 12 - tip.offsetWidth : x(k) + 12;
    tip.style.left = tx + "px"; tip.style.top = "8px";
  };
  const hide = () => { tip.style.display = "none"; cross.setAttribute("visibility", "hidden"); dots.forEach(c => c.setAttribute("visibility", "hidden")); };
  hit.addEventListener("mousemove", show); hit.addEventListener("touchstart", show, {passive: true});
  hit.addEventListener("touchmove", show, {passive: true}); hit.addEventListener("mouseleave", hide);
}

function render() {
  tiles();
  document.getElementById("legend").innerHTML = D.series.map((s, i) => D.summary[i].n ? `<span><span class="dot" style="background:${color(i)}"></span>${esc(s.name)}</span>` : "").join("");
  chart();
  table("sumtable", [["類別"], ["張數", "num"], ["中", "num"], ["累計損益", "num"], ["ROI", "num"]],
    D.summary.map(s => `<tr><td>${esc(s.name)}</td><td class="num">${s.n}</td><td class="num">${s.hits}</td>
      <td class="num ${cls(s.units)}">${s.n ? fmt(s.units) : "—"}</td><td class="num">${s.roi === null ? "—" : s.roi + "%"}</td></tr>`));
  table("tiers", [["種類"], ["注數", "num"], ["中", "num"], ["累計損益", "num"], ["ROI", "num"]],
    D.tiers.map(t => `<tr><td>${esc(t.name)}</td><td class="num">${t.n}</td><td class="num">${t.hits}</td>
      <td class="num ${cls(t.units)}">${fmt(t.units)}</td><td class="num">${t.roi}%</td></tr>`));
  table("tickets", [["日期"], ["類型"], ["選項"], ["賠率", "num"], ["結果"], ["損益", "num"]],
    D.tickets.map(t => `<tr><td>${t.date}</td><td>${esc(t.kind)}</td><td class="wrap">${esc(t.legs)}</td>
      <td class="num">${t.odds.toFixed(2)}</td><td>${t.result}</td><td class="num ${cls(t.profit)}">${fmt(t.profit)}</td></tr>`));
  table("alerts", [["日期"], ["種類"], ["策略"], ["選項"], ["賠率", "num"], ["結果"], ["損益", "num"]],
    D.alerts.map(a => `<tr><td>${a.date}</td><td>${esc(a.tier)}</td><td>${esc(a.strategy)}</td><td>${esc(a.pick)}</td>
      <td class="num">${a.odds.toFixed(2)}</td><td>${a.result}</td><td class="num ${cls(a.profit)}">${fmt(a.profit)}</td></tr>`));
  table("calib", [["模型給的機率"], ["選項數", "num"], ["模型平均", "num"], ["實際發生", "num"]],
    D.calib.map(c => `<tr><td>${c.band}</td><td class="num">${c.n}</td><td class="num">${c.model}%</td><td class="num">${c.actual}%</td></tr>`));
}
render();
let rt; addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(chart, 150); });
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", render);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    print(build())
