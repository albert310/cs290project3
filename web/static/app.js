const messagesEl = document.querySelector("#messages");
const appEl = document.querySelector("#app");
const formEl = document.querySelector("#composer");
const inputEl = document.querySelector("#queryInput");
const sendButton = document.querySelector("#sendButton");
const clearButton = document.querySelector("#clearButton");
const thinkingToggle = document.querySelector("#thinkingToggle");
const verificationToggle = document.querySelector("#verificationToggle");
const rerankToggle = document.querySelector("#rerankToggle");
const tokenBudgetInput = document.querySelector("#tokenBudgetInput");
const modeInputs = [thinkingToggle, verificationToggle, rerankToggle, tokenBudgetInput];
const DEFAULT_TOKEN_BUDGET = 2048;
const THINKING_TOKEN_BUDGET = 8192;
const MIN_TOKEN_BUDGET = 128;
const MAX_TOKEN_BUDGET = 8192;

let busy = false;
let scrollFrame = 0;
let activeController = null;

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

function scrollToEnd() {
  window.cancelAnimationFrame(scrollFrame);
  scrollFrame = window.requestAnimationFrame(() => {
    messagesEl.lastElementChild?.scrollIntoView({ behavior: "smooth", block: "end" });
  });
}

function setBusy(value) {
  busy = value;
  sendButton.disabled = value;
  sendButton.classList.toggle("loading", value);
  appEl.classList.toggle("busy", value);
  inputEl.disabled = value;
  modeInputs.forEach((input) => {
    input.disabled = value;
  });
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
  scrollToEnd();
  return item;
}

function sourceLabel(hit) {
  const title = hit.source_id.match(/title=([^;]+)/)?.[1]?.trim();
  if (title) return title;
  const path = hit.path || hit.source_id;
  const clean = path.split(/[\\/]/).filter(Boolean).pop() || hit.source_id;
  return clean.replace(/#chunk=\d+.*/, "");
}

function renderSources(parent, hits) {
  if (!hits.length) return;
  parent.querySelector(".sources")?.remove();

  const sources = document.createElement("div");
  sources.className = "sources";

  hits.slice(0, 6).forEach((hit, index) => {
    const detail = document.createElement("details");
    detail.className = "source";

    const summary = document.createElement("summary");
    summary.innerHTML = `
      <span class="source-rank">${index + 1}</span>
      <span class="source-path">${escapeHtml(sourceLabel(hit))}</span>
    `;
    summary.title = hit.source_id;

    const snippet = document.createElement("p");
    snippet.className = "source-snippet";
    snippet.textContent = hit.snippet || hit.path;

    detail.appendChild(summary);
    detail.appendChild(snippet);
    sources.appendChild(detail);
  });

  parent.appendChild(sources);
  scrollToEnd();
}

function renderQueryKeywords(parent, keywords, searchQuery) {
  if (!keywords.length) return;

  const detail = document.createElement("details");
  detail.className = "query-keywords";
  detail.open = true;

  const summary = document.createElement("summary");
  summary.textContent = "检索词";

  const list = document.createElement("div");
  list.className = "keyword-list";
  keywords.forEach((keyword) => {
    const chip = document.createElement("span");
    chip.textContent = keyword;
    list.appendChild(chip);
  });

  detail.appendChild(summary);
  detail.appendChild(list);

  if (searchQuery?.trim()) {
    const queryEl = document.createElement("p");
    queryEl.className = "search-query";
    queryEl.textContent = searchQuery.trim();
    detail.appendChild(queryEl);
  }

  parent.appendChild(detail);
  scrollToEnd();
}

function renderRolloutStep(parent, payload) {
  const item = document.createElement("div");
  item.className = "rollout-step";

  const action = payload.action === "search" ? "继续搜索" : "证据足够";
  const step = payload.step ?? "";
  const keywords = payload.keywords?.length ? ` · ${payload.keywords.join(" / ")}` : "";
  const hits = payload.action === "search" ? ` · 新增 ${payload.new_hit_count ?? 0}/${payload.hit_count ?? 0}` : "";
  item.textContent = `Step ${step}: ${action}${keywords}${hits}`;
  if (payload.note) {
    item.title = payload.note;
  }

  parent.appendChild(item);
  scrollToEnd();
}

function renderAnswerVerification(parent, payload) {
  const item = document.createElement("div");
  item.className = "verification-step";

  const keywords = payload.keywords?.length ? ` · ${payload.keywords.join(" / ")}` : "";
  const hits = ` · 新增 ${payload.new_hit_count ?? 0}/${payload.hit_count ?? 0}`;
  item.textContent = `二次核验${keywords}${hits}`;
  const queries = payload.search_queries?.length ? payload.search_queries.join("\n") : payload.search_query;
  if (queries?.trim()) item.title = queries.trim();

  parent.appendChild(item);
  scrollToEnd();
}

function renderLLMRerank(parent, payload) {
  const item = document.createElement("div");
  item.className = "rerank-step";

  const stage = payload.stage ? ` · ${payload.stage}` : "";
  const error = payload.error ? " · 回退" : "";
  const selected = payload.selected_count ?? 0;
  const candidates = payload.candidate_count ?? 0;
  item.textContent = `LLM重排${stage} · ${selected}/${candidates}${error}`;
  if (payload.error) {
    item.title = payload.error;
  }

  parent.appendChild(item);
  scrollToEnd();
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
  scrollToEnd();
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

function ensureStatus(parent) {
  let status = parent.querySelector("[data-role='status']");
  if (!status) {
    status = document.createElement("div");
    status.className = "status-line active";
    status.dataset.role = "status";
    parent.insertBefore(status, parent.querySelector(".bubble"));
  }
  return status;
}

function setStatus(parent, text, active = true) {
  const status = ensureStatus(parent);
  status.textContent = text;
  status.classList.toggle("active", active);
  scrollToEnd();
}

function clearStatus(parent) {
  parent.querySelector("[data-role='status']")?.remove();
}

function readTokenBudget() {
  const parsed = Number.parseInt(tokenBudgetInput.value, 10);
  const safe = Number.isFinite(parsed) ? parsed : DEFAULT_TOKEN_BUDGET;
  const clamped = Math.min(MAX_TOKEN_BUDGET, Math.max(MIN_TOKEN_BUDGET, safe));
  tokenBudgetInput.value = String(clamped);
  return clamped;
}

function currentTokenBudgetValue() {
  const parsed = Number.parseInt(tokenBudgetInput.value, 10);
  return Number.isFinite(parsed) ? parsed : null;
}

function syncThinkingBudget() {
  const current = currentTokenBudgetValue();
  if (thinkingToggle.checked) {
    if (current === null || current < THINKING_TOKEN_BUDGET) {
      tokenBudgetInput.value = String(THINKING_TOKEN_BUDGET);
    }
  } else if (current === THINKING_TOKEN_BUDGET) {
    tokenBudgetInput.value = String(DEFAULT_TOKEN_BUDGET);
  }
  readTokenBudget();
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

async function streamAsk(query, options, signal, onEvent) {
  const response = await fetch("/api/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    signal,
    body: JSON.stringify({
      query,
      top_k: 20,
      enable_thinking: options.enableThinking,
      verify_answer: options.verifyAnswer,
      llm_rerank: options.llmRerank,
      max_tokens: options.maxTokens,
    }),
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
  const assistantItem = appendMessage("assistant", "");
  const bubble = assistantItem.querySelector(".bubble");
  const controller = new AbortController();
  activeController = controller;
  const options = {
    enableThinking: thinkingToggle.checked,
    verifyAnswer: verificationToggle.checked,
    llmRerank: rerankToggle.checked,
    maxTokens: readTokenBudget(),
  };
  let hits = [];
  let answerStarted = false;

  setBusy(true);
  assistantItem.classList.add("streaming");
  setStatus(assistantItem, "正在检索资料");
  try {
    await streamAsk(query, options, controller.signal, (payload) => {
      if (controller.signal.aborted) return;
      if (payload.event === "query_keywords") {
        setStatus(assistantItem, "已生成检索词");
        renderQueryKeywords(assistantItem, payload.keywords || [], payload.search_query);
      } else if (payload.event === "search_rollout_step") {
        setStatus(assistantItem, "正在扩展搜索");
        renderRolloutStep(assistantItem, payload);
      } else if (payload.event === "answer_verification") {
        setStatus(assistantItem, "正在二次核验");
        renderAnswerVerification(assistantItem, payload);
      } else if (payload.event === "llm_rerank") {
        setStatus(assistantItem, "正在重排资料");
        renderLLMRerank(assistantItem, payload);
      } else if (payload.event === "sources") {
        hits = payload.hits || [];
        setStatus(assistantItem, "正在阅读资料");
        renderSources(assistantItem, hits);
      } else if (payload.event === "think_delta") {
        setStatus(assistantItem, "正在思考");
        const thinkBody = ensureThink(assistantItem);
        thinkBody.textContent += payload.delta || "";
        scrollToEnd();
      } else if (payload.event === "answer_delta") {
        if (!answerStarted) {
          answerStarted = true;
          clearStatus(assistantItem);
          bubble.textContent = "";
        }
        bubble.textContent += payload.delta || "";
        scrollToEnd();
      } else if (payload.event === "done") {
        assistantItem.classList.remove("streaming");
        clearStatus(assistantItem);
        const meta = document.createElement("div");
        meta.className = "meta";
        const elapsed = payload.latency_sec;
        const finish = payload.finish_reason ? ` · ${payload.finish_reason}` : "";
        const budget = payload.max_tokens || options.maxTokens;
        meta.textContent = `检索 ${hits.length} 条资料 · 预算 ${budget} tokens${elapsed ? ` · ${elapsed.toFixed(1)}s` : ""}${finish}`;
        assistantItem.appendChild(meta);
      } else if (payload.event === "error") {
        throw new Error(payload.error || "stream error");
      }
    });
    if (!answerStarted && !bubble.textContent.trim()) {
      clearStatus(assistantItem);
      bubble.textContent = "根据当前资料无法确认。";
    }
  } catch (error) {
    if (controller.signal.aborted) return;
    assistantItem.classList.remove("streaming");
    clearStatus(assistantItem);
    bubble.textContent = `请求失败：${error instanceof Error ? error.message : String(error)}`;
  } finally {
    if (activeController === controller) {
      activeController = null;
      assistantItem.classList.remove("streaming");
      setBusy(false);
      inputEl.focus();
      scrollToEnd();
    }
  }
}

formEl.addEventListener("submit", (event) => {
  event.preventDefault();
  void submitQuestion();
});

inputEl.addEventListener("input", resizeInput);
thinkingToggle.addEventListener("change", syncThinkingBudget);
tokenBudgetInput.addEventListener("change", readTokenBudget);
tokenBudgetInput.addEventListener("blur", readTokenBudget);
inputEl.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    void submitQuestion();
  }
});

clearButton.addEventListener("click", () => {
  activeController?.abort();
  activeController = null;
  messagesEl.innerHTML = "";
  appEl.classList.add("empty");
  setBusy(false);
  inputEl.value = "";
  resizeInput();
  inputEl.focus();
});

syncThinkingBudget();
resizeInput();
inputEl.focus();
