// NBA 場中分析（Cloudflare Worker 直接處理，約 2–4 秒回覆；不經過 GitHub Actions，因為場中盤口幾秒就變）
//
// 資料：ESPN scoreboard / summary（比分、時間、球員即時數據、ESPN 勝率模型、賽前盤口）
//       The Odds API（DraftKings 場中讓分、大小、不讓分；ESPN 的場中盤口是過期的，不能用）
// 每次抓盤口用 3 點額度（h2h + spreads + totals），30 秒內重複按用快取、不再扣。
// 場中盤口沒有歷史資料可以回測，所以這裡只整理資訊，不給推薦。

const ZH = {
  ATL: "老鷹", BOS: "塞爾提克", BKN: "籃網", CHA: "黃蜂", CHI: "公牛", CLE: "騎士", DAL: "獨行俠", DEN: "金塊",
  DET: "活塞", GS: "勇士", HOU: "火箭", IND: "溜馬", LAC: "快艇", LAL: "湖人", MEM: "灰熊", MIA: "熱火",
  MIL: "公鹿", MIN: "灰狼", NO: "鵜鶘", NY: "尼克", OKC: "雷霆", ORL: "魔術", PHI: "76人", PHX: "太陽",
  POR: "拓荒者", SAC: "國王", SA: "馬刺", TOR: "暴龍", UTAH: "爵士", WSH: "巫師",
};
const ESPN = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba";
const CREDIT_WARN = 40;

const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const name = (c) => `${ZH[c.team.abbreviation] || c.team.shortDisplayName} ${c.team.abbreviation}`;
const sgn = (x, d = 1) => (x > 0 ? "+" : "") + Number(x).toFixed(d).replace(/\.0$/, "");
const american = (ml) => (ml > 0 ? 1 + ml / 100 : 1 + 100 / Math.abs(ml));
const noVig = (a, b) => (1 / a) / (1 / a + 1 / b);

async function getJson(url, init) {
  const r = await fetch(url, init);
  if (!r.ok) throw new Error(`${new URL(url).hostname} ${r.status}`);
  return { json: await r.json(), headers: r.headers };
}

/** 秒數表示的時間：回傳 { elapsed, remaining }（分鐘）。延長賽每節 5 分鐘。 */
function clock(period, display) {
  const m = String(display).includes(":") ? display.split(":").reduce((a, v, i) => a + Number(v) / (i ? 60 : 1), 0) : Number(display) / 60;
  if (period <= 4) return { elapsed: (period - 1) * 12 + 12 - m, remaining: (4 - period) * 12 + m };
  return { elapsed: 48 + (period - 5) * 5 + 5 - m, remaining: m };
}

/** 犯規麻煩：第 1 節 2 犯、上半場 3 犯、第 3 節 4 犯、任何時候 5 犯。 */
const foulTrouble = (pf, period) => pf >= 5 || (pf >= 4 && period <= 3) || (pf >= 3 && period <= 2) || (pf >= 2 && period <= 1);

async function liveOdds(env) {
  if (!env.ODDS_API_KEY) return { events: [], note: "（沒有設定 ODDS_API_KEY，無法取得場中盤口）" };
  const url = `https://api.the-odds-api.com/v4/sports/basketball_nba/odds?apiKey=${env.ODDS_API_KEY}` +
    "&regions=us&markets=h2h,spreads,totals&oddsFormat=decimal&bookmakers=draftkings";
  const { json, headers } = await getJson(url, { cf: { cacheTtl: 30, cacheEverything: true } });
  const left = Number(headers.get("x-requests-remaining"));
  const note = Number.isFinite(left) && left < CREDIT_WARN ? `⚠️ The Odds API 本月只剩 ${left} 點額度` : "";
  return { events: json, note };
}

/** 把 The Odds API 的事件對到 ESPN 的比賽（用球隊暱稱比對：Clippers、Trail Blazers、76ers…）。 */
function findOdds(events, home, away) {
  const e = events.find((x) => x.home_team.endsWith(" " + home.team.name) && x.away_team.endsWith(" " + away.team.name));
  const bk = e && e.bookmakers && e.bookmakers[0];
  if (!bk) return null;
  const mk = Object.fromEntries(bk.markets.map((m) => [m.key, m]));
  const pick = (m, pred) => (m ? m.outcomes.find(pred) : undefined);
  const h2h = (n) => pick(mk.h2h, (o) => o.name === n);
  const sp = (n) => pick(mk.spreads, (o) => o.name === n);
  const tt = (n) => pick(mk.totals, (o) => o.name === n);
  const hs = sp(e.home_team), as = sp(e.away_team);
  const stamp = [mk.spreads, mk.totals, mk.h2h].filter(Boolean).map((m) => Date.parse(m.last_update)).sort((a, b) => b - a)[0];
  return {
    mlHome: h2h(e.home_team)?.price, mlAway: h2h(e.away_team)?.price,
    spread: hs?.point, spreadHome: hs?.price, spreadAway: as?.price,
    total: tt("Over")?.point, over: tt("Over")?.price, under: tt("Under")?.price,
    age: stamp ? Math.max(0, Math.round((Date.now() - stamp) / 1000)) : null,
  };
}

function players(box, period) {
  const L = box.statistics[0].labels, ix = (k) => L.indexOf(k);
  const rows = box.statistics[0].athletes.map((a) => {
    const s = a.stats || [];
    const n = (k) => Number(s[ix(k)] ?? 0) || 0;
    return { name: a.athlete.shortName || a.athlete.displayName, min: n("MIN"), pts: n("PTS"), reb: n("REB"), ast: n("AST"),
      pf: n("PF"), pm: s[ix("+/-")] ?? "0", dnp: a.didNotPlay, reason: a.reason || "", ejected: a.ejected };
  });
  const lines = rows.filter((r) => r.min > 0 || r.ejected).sort((a, b) => b.min - a.min).slice(0, 5).map((r) => {
    const flags = [];
    if (r.ejected) flags.push("🚫已被驅逐");
    if (foulTrouble(r.pf, period)) flags.push("⚠️犯規麻煩");
    return `　• ${esc(r.name)} ${r.pts}分 ${r.reb}板 ${r.ast}助｜${r.min}分鐘｜${r.pf}犯｜+/- ${esc(r.pm)}${flags.length ? "｜" + flags.join(" ") : ""}`;
  });
  const out = rows.filter((r) => r.dnp && /INJURY|ILLNESS|PERSONAL|SUSPEN|HEALTH/i.test(r.reason)).slice(0, 3);
  if (out.length) lines.push("　🚑 缺陣：" + out.map((r) => `${esc(r.name)}（${esc(r.reason.toLowerCase())}）`).join("、"));
  return lines;
}

/** 單場詳細：盤口、勝率、節奏、球員狀態。 */
export async function liveDetail(env, eventId) {
  const [{ json: d }, odds] = await Promise.all([getJson(`${ESPN}/summary?event=${eventId}`), liveOdds(env)]);
  const comp = d.header.competitions[0], st = comp.status;
  if (st.type.state !== "in") return `這場比賽目前不是進行中（${esc(st.type.description)}）。`;
  const home = comp.competitors.find((c) => c.homeAway === "home"), away = comp.competitors.find((c) => c.homeAway === "away");
  const hs = Number(home.score), as = Number(away.score), tot = hs + as, margin = hs - as;
  const { elapsed, remaining } = clock(st.period, st.displayClock);
  const o = findOdds(odds.events, home, away);
  const wp = (d.winprobability || []).slice(-1)[0]?.homeWinPercentage;
  const pre = (d.pickcenter || [])[0];
  const L = [`🔴 <b>場中｜${esc(name(away))} ${as} @ ${hs} ${esc(name(home))}</b>`,
    `⏱ 第 ${st.period} 節 ${esc(st.displayClock)}｜已打 ${elapsed.toFixed(1)} 分鐘、剩 ${remaining.toFixed(1)} 分鐘`, ""];
  L.push(`<b>📈 盤口（DK 場中${o && o.age != null ? `，${o.age} 秒前更新` : ""}）</b>`);
  if (o && o.spread != null) {
    const h = esc(ZH[home.team.abbreviation] || home.team.abbreviation), a = esc(ZH[away.team.abbreviation] || away.team.abbreviation);
    L.push(`讓分：${h} ${sgn(o.spread)}（${o.spreadHome.toFixed(2)}）／${a} ${sgn(-o.spread)}（${o.spreadAway.toFixed(2)}）` +
      (pre?.spread != null ? `｜賽前 ${h} ${sgn(pre.spread)}` : ""));
    L.push(`大小：${o.total}（大 ${o.over.toFixed(2)}／小 ${o.under.toFixed(2)}）` + (pre?.overUnder != null ? `｜賽前 ${pre.overUnder}` : ""));
    if (o.mlHome && o.mlAway) L.push(`不讓分：${h} ${o.mlHome.toFixed(2)}／${a} ${o.mlAway.toFixed(2)}`);
  } else {
    L.push("（目前抓不到這場的場中盤口）");
  }
  if (odds.note) L.push(odds.note);
  L.push("", "<b>🧮 分析</b>");
  const pm = o?.mlHome && o?.mlAway ? noVig(o.mlHome, o.mlAway) : null;
  const pg = pre?.homeTeamOdds?.moneyLine && pre?.awayTeamOdds?.moneyLine ? noVig(american(pre.homeTeamOdds.moneyLine), american(pre.awayTeamOdds.moneyLine)) : null;
  const hz = ZH[home.team.abbreviation] || home.team.abbreviation;
  L.push(`${esc(hz)}勝率：` + [pm != null ? `盤口隱含 ${(pm * 100).toFixed(0)}%` : null, wp != null ? `ESPN 模型 ${(wp * 100).toFixed(0)}%` : null,
    pg != null ? `賽前 ${(pg * 100).toFixed(0)}%` : null].filter(Boolean).join("｜"));
  if (pm != null && wp != null && Math.abs(pm - wp) >= 0.1) L.push(`　⚠️ 盤口與 ESPN 模型差 ${(Math.abs(pm - wp) * 100).toFixed(0)}pp：可能盤口還沒更新，或其中一邊沒反映球員狀況`);
  if (o?.spread != null) L.push(`淨勝：目前 ${esc(hz)} ${sgn(margin, 0)}；盤口預期最終 ${sgn(-o.spread)} → 剩餘時間預期 ${sgn(-o.spread - margin)}`);
  if (elapsed >= 12) {
    const pace = tot / elapsed, proj = pace * (elapsed + remaining);
    L.push(`總分：目前 ${tot}，照目前節奏（${pace.toFixed(2)} 分/分鐘）全場約 ${proj.toFixed(0)}` +
      (o?.total ? `；場中大小 ${o.total} → ${proj > o.total ? "節奏比盤口快" : "節奏比盤口慢"} ${Math.abs(proj - o.total).toFixed(1)} 分` : ""));
  } else {
    L.push("總分：比賽才剛開始，節奏還不準，先不估");
  }
  L.push("", "<b>👤 球員狀態</b>（上場時間前 5 名）");
  for (const c of [away, home]) {
    const box = d.boxscore.players.find((p) => p.team.abbreviation === c.team.abbreviation);
    L.push(`<b>${esc(name(c))}</b>`, ...(box ? players(box, st.period) : ["　（沒有球員數據）"]));
  }
  L.push("", "ℹ️ 場中盤口沒有歷史資料可以回測，這裡只整理資訊，不是推薦；盤口幾秒就會變，可能已經不一樣。節奏估計沒考慮垃圾時間與末段犯規戰術。");
  return L.join("\n");
}

/** 進行中的比賽清單，每場一顆按鈕。 */
export async function liveList(env) {
  const [{ json: sb }, odds] = await Promise.all([getJson(`${ESPN}/scoreboard`), liveOdds(env)]);
  const games = sb.events.filter((e) => e.status.type.state === "in");
  if (!games.length) {
    const next = sb.events.filter((e) => e.status.type.state === "pre").sort((a, b) => a.date.localeCompare(b.date))[0];
    const when = next && new Date(next.date).toLocaleString("zh-TW", { timeZone: "Asia/Taipei", month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit", hour12: false });
    return { text: "🔴 目前沒有進行中的 NBA 比賽。" + (next ? `\n下一場：${esc(next.shortName)}（台灣時間 ${when}）` : "") };
  }
  const rows = [], lines = [`🔴 <b>NBA 場中（${games.length} 場）</b>`, ""];
  for (const e of games) {
    const c = e.competitions[0], home = c.competitors.find((x) => x.homeAway === "home"), away = c.competitors.find((x) => x.homeAway === "away");
    const o = findOdds(odds.events, home, away), hz = ZH[home.team.abbreviation] || home.team.abbreviation;
    const pm = o?.mlHome && o?.mlAway ? noVig(o.mlHome, o.mlAway) : null;
    lines.push(`<b>${esc(name(away))} ${away.score} @ ${home.score} ${esc(name(home))}</b>｜${esc(e.status.type.shortDetail)}`,
      o && o.spread != null ? `　讓分 ${esc(hz)} ${sgn(o.spread)}｜大小 ${o.total}` + (pm != null ? `｜${esc(hz)}勝率 ${(pm * 100).toFixed(0)}%` : "") : "　（沒有場中盤口）");
    rows.push([{ text: `${ZH[away.team.abbreviation] || away.team.abbreviation} @ ${hz}　${away.score}-${home.score}`, callback_data: `lv:${e.id}` }]);
  }
  if (odds.note) lines.push("", odds.note);
  lines.push("", "👇 點比賽看盤口分析與球員狀態");
  return { text: lines.join("\n"), keyboard: { inline_keyboard: rows } };
}
