type SourceHit = {
  source_id: string;
  path: string;
  chunk_index: number;
  rank: number;
  snippet: string;
};

type ChatResponse = {
  answer: string;
  think?: string;
  hits: SourceHit[];
  latency_sec?: number;
  error?: string;
};

type StreamPayload = {
  event: "sources" | "think_delta" | "answer_delta" | "done" | "error";
  delta?: string;
  hits?: SourceHit[];
  latency_sec?: number;
  error?: string;
};

const messagesEl = document.querySelector<HTMLDivElement>("#messages")!;
const appEl = document.querySelector<HTMLElement>("#app")!;
const formEl = document.querySelector<HTMLFormElement>("#composer")!;
const inputEl = document.querySelector<HTMLTextAreaElement>("#queryInput")!;
const sendButton = document.querySelector<HTMLButtonElement>("#sendButton")!;
const clearButton = document.querySelector<HTMLButtonElement>("#clearButton")!;

let busy = false;

function escapeHtml(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function resizeInput(): void {
  inputEl.style.height = "0px";
  inputEl.style.height = `${Math.min(inputEl.scrollHeight, 180)}px`;
}

function setBusy(value: boolean): void {
  busy = value;
  sendButton.disabled = value;
  sendButton.classList.toggle("loading", value);
  inputEl.disabled = value;
}

function showConversation(): void {
  appEl.classList.remove("empty");
}

function appendMessage(role: "user" | "assistant", text: string): HTMLDivElement {
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

function renderSources(parent: HTMLElement, hits: SourceHit[]): void {
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

function renderThink(parent: HTMLElement, think?: string): void {
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

function ensureThink(parent: HTMLElement): HTMLParagraphElement {
  let detail = parent.querySelector<HTMLDetailsElement>(".think");
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
  return detail.querySelector<HTMLParagraphElement>("[data-role='think-body']")!;
}

function parseSseBlock(block: string): StreamPayload | null {
  const lines = block.split(/\r?\n/);
  let event = "";
  const data: string[] = [];
  for (const line of lines) {
    if (line.startsWith("event:")) {
      event = line.slice("event:".length).trim();
    } else if (line.startsWith("data:")) {
      data.push(line.slice("data:".length).trimStart());
    }
  }
  if (!event || !data.length) return null;
  try {
    const payload = JSON.parse(data.join("\n")) as StreamPayload;
    payload.event = payload.event || (event as StreamPayload["event"]);
    return payload;
  } catch {
    return { event: "error", error: "stream parse error" };
  }
}

async function streamAsk(
  query: string,
  onEvent: (payload: StreamPayload) => void,
): Promise<void> {
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

async function submitQuestion(): Promise<void> {
  const query = inputEl.value.trim();
  if (!query || busy) return;

  inputEl.value = "";
  resizeInput();
  appendMessage("user", query);
  const assistantItem = appendMessage("assistant", "正在检索资料...");
  const bubble = assistantItem.querySelector<HTMLDivElement>(".bubble")!;
  let hits: SourceHit[] = [];
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
