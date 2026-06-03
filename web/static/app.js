const messagesEl = document.querySelector("#messages");
const appEl = document.querySelector("#app");
const formEl = document.querySelector("#composer");
const inputEl = document.querySelector("#queryInput");
const sendButton = document.querySelector("#sendButton");
const clearButton = document.querySelector("#clearButton");

let busy = false;

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function resizeInput() {
  inputEl.style.height = "0px";
  inputEl.style.height = `${Math.min(inputEl.scrollHeight, 180)}px`;
}

function setBusy(value) {
  busy = value;
  sendButton.disabled = value;
  sendButton.classList.toggle("loading", value);
  inputEl.disabled = value;
}

function showConversation() {
  appEl.classList.remove("empty");
}

function appendMessage(role, text) {
  showConversation();
  const item = document.createElement("article");
  item.className = `message ${role}`;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  item.appendChild(bubble);

  messagesEl.appendChild(item);
  item.scrollIntoView({ behavior: "smooth", block: "end" });
  return item;
}

function renderSources(parent, hits) {
  if (!hits.length) return;

  const sources = document.createElement("div");
  sources.className = "sources";

  hits.slice(0, 6).forEach((hit, index) => {
    const detail = document.createElement("details");
    detail.className = "source";

    const summary = document.createElement("summary");
    summary.innerHTML = `
      <span class="source-rank">[${index + 1}]</span>
      <span class="source-path">${escapeHtml(hit.source_id)}</span>
    `;

    const snippet = document.createElement("p");
    snippet.className = "source-snippet";
    snippet.textContent = hit.snippet || hit.path;

    detail.appendChild(summary);
    detail.appendChild(snippet);
    sources.appendChild(detail);
  });

  parent.appendChild(sources);
}

function renderThink(parent, think) {
  if (!think?.trim()) return;

  const detail = document.createElement("details");
  detail.className = "think";

  const summary = document.createElement("summary");
  summary.textContent = "思考";

  const body = document.createElement("p");
  body.textContent = think.trim();

  detail.appendChild(summary);
  detail.appendChild(body);
  parent.insertBefore(detail, parent.querySelector(".bubble"));
}

function ensureThink(parent) {
  let detail = parent.querySelector(".think");
  if (!detail) {
    detail = document.createElement("details");
    detail.className = "think";
    detail.open = true;

    const summary = document.createElement("summary");
    summary.textContent = "思考";

    const body = document.createElement("p");
    body.dataset.role = "think-body";

    detail.appendChild(summary);
    detail.appendChild(body);
    parent.insertBefore(detail, parent.querySelector(".bubble"));
  }
  return detail.querySelector("[data-role='think-body']");
}

function parseSseBlock(block) {
  const lines = block.split(/\r?\n/);
  let event = "";
  const data = [];
  for (const line of lines) {
    if (line.startsWith("event:")) {
      event = line.slice("event:".length).trim();
    } else if (line.startsWith("data:")) {
      data.push(line.slice("data:".length).trimStart());
    }
  }
  if (!event || !data.length) return null;
  try {
    const payload = JSON.parse(data.join("\n"));
    payload.event = payload.event || event;
    return payload;
  } catch {
    return { event: "error", error: "stream parse error" };
  }
}

async function streamAsk(query, onEvent) {
  const response = await fetch("/api/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, top_k: 6 }),
  });

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }

  const reader = response.body?.getReader();
  if (!reader) throw new Error("stream unavailable");

  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const blocks = buffer.split(/\n\n/);
    buffer = blocks.pop() || "";
    for (const block of blocks) {
      const payload = parseSseBlock(block.trim());
      if (payload) onEvent(payload);
    }
  }
  buffer += decoder.decode();
  if (buffer.trim()) {
    const payload = parseSseBlock(buffer.trim());
    if (payload) onEvent(payload);
  }
}

async function submitQuestion() {
  const query = inputEl.value.trim();
  if (!query || busy) return;

  inputEl.value = "";
  resizeInput();
  appendMessage("user", query);
  const assistantItem = appendMessage("assistant", "正在检索资料...");
  const bubble = assistantItem.querySelector(".bubble");
  let hits = [];
  let answerStarted = false;

  setBusy(true);
  try {
    await streamAsk(query, (payload) => {
      if (payload.event === "sources") {
        hits = payload.hits || [];
        bubble.textContent = "正在思考...";
        renderSources(assistantItem, hits);
      } else if (payload.event === "think_delta") {
        const thinkBody = ensureThink(assistantItem);
        thinkBody.textContent += payload.delta || "";
      } else if (payload.event === "answer_delta") {
        if (!answerStarted) {
          answerStarted = true;
          bubble.textContent = "";
        }
        bubble.textContent += payload.delta || "";
      } else if (payload.event === "done") {
        const meta = document.createElement("div");
        meta.className = "meta";
        const elapsed = payload.latency_sec;
        meta.textContent = `检索 ${hits.length} 条资料${elapsed ? ` · ${elapsed.toFixed(1)}s` : ""}`;
        assistantItem.appendChild(meta);
      } else if (payload.event === "error") {
        throw new Error(payload.error || "stream error");
      }
    });
    if (!answerStarted && bubble.textContent === "正在思考...") {
      bubble.textContent = "根据当前资料无法确认。";
    }
  } catch (error) {
    bubble.textContent = `请求失败：${error instanceof Error ? error.message : String(error)}`;
  } finally {
    setBusy(false);
    inputEl.focus();
    assistantItem.scrollIntoView({ behavior: "smooth", block: "end" });
  }
}

formEl.addEventListener("submit", (event) => {
  event.preventDefault();
  void submitQuestion();
});

inputEl.addEventListener("input", resizeInput);
inputEl.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    void submitQuestion();
  }
});

clearButton.addEventListener("click", () => {
  messagesEl.innerHTML = "";
  appEl.classList.add("empty");
  inputEl.value = "";
  resizeInput();
  inputEl.focus();
});

resizeInput();
inputEl.focus();
