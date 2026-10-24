// Telegram 按鈕接收器（Cloudflare Worker，免費、24 小時在線）
//
// Telegram 收到你的訊息或按鈕 → 送到這裡（webhook）
//   1. 檢查密語（只接受 Telegram 送來的）與 chat_id（只回應你本人）
//   2. 說明／選單：直接回覆（秒回）
//   3. 其他指令：立刻回「處理中」，再觸發 GitHub Actions 的 command.yml 去跑分析（約 1 分鐘回覆）
//
// 需要的設定（wrangler secret）：BOT_TOKEN、CHAT_ID、GH_TOKEN、WEBHOOK_SECRET；變數 GH_REPO 在 wrangler.toml

const MENU = {
  keyboard: [
    [{ text: "🏆 今日推薦" }, { text: "🎟️ 串關" }],
    [{ text: "🏀 NBA" }, { text: "⚾ MLB" }, { text: "⚽ 足球" }],
    [{ text: "🧾 賽後檢討" }, { text: "📅 今日總覽" }],
    [{ text: "💱 運彩比價" }, { text: "📒 戰績" }, { text: "❓ 說明" }],
  ],
  resize_keyboard: true,
  is_persistent: true,
  input_field_placeholder: "按下面的按鈕，或輸入隊名（例：湖人、曼城）",
};

const HELP = `🤖 <b>運彩分析機器人</b>

👇 <b>直接按下面的按鈕</b>就可以用：
🏆 <b>今日推薦</b>：掃描所有比賽，給出最推薦的單關、二串一、三串一
🎟️ <b>串關</b>：更多二串一、三串一組合
🏀 <b>NBA</b>／⚾ <b>MLB</b>／⚽ <b>足球</b>：先看總表，<b>點比賽按鈕</b>才展開詳細分析
🧾 <b>賽後檢討</b>　📅 <b>今日總覽</b>　📒 <b>戰績</b>
💱 <b>運彩比價</b>：直接打運彩的賠率，例如 <code>湖人 1.75</code>、<code>湖人 -3.5 1.80</code>（好幾行 = 串關）

也可以直接打字：隊名（曼城、皇馬、湖人、LAL）、聯賽（歐冠、英超）

⏱ 由雲端處理，按下後約 1 分鐘回覆（電腦關機也能用）
判斷原則只有一條：<b>運彩賠率 ≥ 合理賠率才下</b>，低於就跳過，只用小注。`;

const INSTANT = new Set(["/start", "/help", "/menu", "❓ 說明", "說明", "選單", "指令"]);

const HOWTO = `💱 <b>運彩賠率比價</b>
把運彩開的賠率直接打給我（一行一關，好幾行就算成串關）：
<code>湖人 1.75</code>　→ 湖人不讓分（足球是「那隊勝」）
<code>湖人 -3.5 1.80</code>　→ 湖人讓 3.5（也可寫 讓3.5；受讓寫 +3.5 或 受讓3.5）
<code>湖人 大228.5 1.75</code>　→ 那場大分（小分寫「小」）
<code>利物浦 和 3.40</code>　→ 足球和局
我會回：估計勝率、合理賠率、期望值，值不值得買。`;
const HOWTO_KEYS = new Set(["💱 運彩比價", "比價", "運彩比價", "/odds"]);

export default {
  async fetch(request, env) {
    if (request.method !== "POST") return new Response("ok");
    if (request.headers.get("X-Telegram-Bot-Api-Secret-Token") !== env.WEBHOOK_SECRET) {
      return new Response("forbidden", { status: 403 });
    }
    let update;
    try {
      update = await request.json();
    } catch {
      return new Response("ok");
    }

    const tg = (method, body) =>
      fetch(`https://api.telegram.org/bot${env.BOT_TOKEN}/${method}`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(body),
      });

    let kind, payload, chatId, callbackId;
    if (update.callback_query) {
      kind = "callback";
      payload = update.callback_query.data || "";
      chatId = String(update.callback_query.message?.chat?.id ?? "");
      callbackId = update.callback_query.id;
    } else if (update.message?.text) {
      kind = "message";
      payload = update.message.text.trim();
      chatId = String(update.message.chat.id);
    } else {
      return new Response("ok");
    }
    if (chatId !== String(env.CHAT_ID)) return new Response("ok"); // 只回應本人

    // 說明／選單：秒回
    if (kind === "message" && INSTANT.has(payload.split("@")[0])) {
      await tg("sendMessage", { chat_id: chatId, text: HELP, parse_mode: "HTML", reply_markup: MENU });
      return new Response("ok");
    }

    if (kind === "message" && HOWTO_KEYS.has(payload)) {
      await tg("sendMessage", { chat_id: chatId, text: HOWTO, parse_mode: "HTML" });
      return new Response("ok");
    }

    // 其他：觸發 GitHub Actions（command.yml），輸入只當資料傳遞
    const resp = await fetch(
      `https://api.github.com/repos/${env.GH_REPO}/actions/workflows/command.yml/dispatches`,
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${env.GH_TOKEN}`,
          Accept: "application/vnd.github+json",
          "X-GitHub-Api-Version": "2022-11-28",
          "User-Agent": "sports-betting-telegram-worker",
        },
        body: JSON.stringify({ ref: "main", inputs: { kind, payload: payload.slice(0, 200) } }),
      },
    );
    const ok = resp.status === 204;
    if (callbackId) {
      await tg("answerCallbackQuery", {
        callback_query_id: callbackId,
        text: ok ? "⏳ 雲端處理中，約 1 分鐘…" : "⚠️ 雲端暫時無法處理，請稍後再試",
      });
    } else {
      await tg("sendMessage", {
        chat_id: chatId,
        text: ok ? "⏳ 收到，雲端處理中（約 1 分鐘）…" : `⚠️ 雲端暫時無法處理（${resp.status}），請稍後再試`,
      });
    }
    return new Response("ok");
  },
};
