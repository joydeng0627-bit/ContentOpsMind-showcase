const STORAGE_KEY = "contentopsmind_conversations_v2";
const UI_STORAGE_KEY = "contentopsmind_ui_v1";
const CAMPAIGN_CREATE_STORAGE_KEY = "contentopsmind_pending_campaign_create_v1";
const API_BASE_URL = window.location.protocol === "file:" ? "http://localhost:8010" : "";
const REQUEST_TIMEOUT_MS = 120000;
const state = {
  conversations: [],
  activeId: null,
  lastRequestId: null,
  currentTrace: null,
  currentView: "workspace",
  activeRequest: null,
  typingToken: 0,
  historyLoadError: false,
  sidebarCollapsed: false,
  taskOptionsTouched: false,
  campaignRequest: null,
  campaignWorkflows: [],
  activeCampaignWorkflowId: null,
  campaignDetail: null,
  campaignRevisionTarget: null,
  campaignBriefEditWorkflowId: null,
  pendingCampaignCreate: null,
};
const $ = (id) => document.getElementById(id);
const uid = () =>
  crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random().toString(16).slice(2)}`;

function showToast(message) {
  const toast = $("toast");
  toast.textContent = message;
  toast.classList.add("show");
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => toast.classList.remove("show"), 2400);
}

async function api(path, options = {}) {
  const res = await fetch(`${API_BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const type = res.headers.get("content-type") || "";
  const data = type.includes("application/json") ? await res.json() : await res.text();
  if (!res.ok) {
    const error = new Error(typeof data === "string" ? data : data.detail || JSON.stringify(data));
    error.status = res.status;
    error.response = data;
    throw error;
  }
  return data;
}

function loadPendingCampaignCreate() {
  if (state.pendingCampaignCreate) return state.pendingCampaignCreate;
  try {
    const parsed = JSON.parse(sessionStorage.getItem(CAMPAIGN_CREATE_STORAGE_KEY) || "null");
    if (parsed?.idempotencyKey && parsed?.requestSnapshot) state.pendingCampaignCreate = parsed;
  } catch (_) {
    sessionStorage.removeItem(CAMPAIGN_CREATE_STORAGE_KEY);
  }
  return state.pendingCampaignCreate;
}

function clearPendingCampaignCreate() {
  state.pendingCampaignCreate = null;
  sessionStorage.removeItem(CAMPAIGN_CREATE_STORAGE_KEY);
}

function campaignCreateOperation(payload) {
  const requestSnapshot = JSON.stringify(payload);
  const pending = loadPendingCampaignCreate();
  if (pending?.requestSnapshot === requestSnapshot) return pending;
  const operation = { idempotencyKey: uid(), requestSnapshot };
  state.pendingCampaignCreate = operation;
  sessionStorage.setItem(CAMPAIGN_CREATE_STORAGE_KEY, JSON.stringify(operation));
  return operation;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function renderInlineMarkdown(text) {
  let value = text;
  const inlineCode = [];
  value = value.replace(/`([^`\n]+)`/g, (_, code) => {
    const token = `\u0000INLINE${inlineCode.length}\u0000`;
    inlineCode.push(`<code>${code}</code>`);
    return token;
  });
  value = value.replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>");
  value = value.replace(/\[([^\]\n]+)\]\(([^)\s]+)\)/g, (match, label, href) => {
    const decoded = href.replaceAll("&amp;", "&");
    if (!/^(https?:|mailto:)/i.test(decoded)) return match;
    return `<a href="${href}" target="_blank" rel="noopener noreferrer">${label}</a>`;
  });
  inlineCode.forEach((code, index) => {
    value = value.replace(`\u0000INLINE${index}\u0000`, code);
  });
  return value;
}

function renderMarkdownLite(text) {
  const codeBlocks = [];
  let source = String(text || "").replace(/\r\n?/g, "\n");
  source = source.replace(/```([A-Za-z0-9_-]*)[ \t]*\n([\s\S]*?)```/g, (_, language, body) => {
    const token = `\u0000BLOCK${codeBlocks.length}\u0000`;
    const className = language ? ` class="language-${escapeHtml(language)}"` : "";
    codeBlocks.push(`<pre><code${className}>${escapeHtml(body.replace(/\n$/, ""))}</code></pre>`);
    return token;
  });
  source = escapeHtml(source);
  const lines = source.split("\n");
  const output = [];
  let listType = null;
  let orderedItemCount = 0;
  const closeList = () => {
    if (listType) output.push(`</${listType}>`);
    listType = null;
  };

  for (const rawLine of lines) {
    const line = rawLine.trimEnd();
    const blockMatch = line.match(/^\u0000BLOCK(\d+)\u0000$/);
    if (blockMatch) {
      closeList();
      orderedItemCount = 0;
      output.push(codeBlocks[Number(blockMatch[1])]);
      continue;
    }
    const heading = line.match(/^(#{1,3})\s+(.+)$/);
    if (heading) {
      closeList();
      orderedItemCount = 0;
      const level = heading[1].length;
      output.push(`<h${level}>${renderInlineMarkdown(heading[2])}</h${level}>`);
      continue;
    }
    if (/^\s*(---+|\*\*\*+)\s*$/.test(line)) {
      closeList();
      orderedItemCount = 0;
      output.push("<hr>");
      continue;
    }
    const unordered = line.match(/^\s*[-*+]\s+(.+)$/);
    const ordered = line.match(/^\s*\d+[.)]\s+(.+)$/);
    if (unordered || ordered) {
      const nextType = unordered ? "ul" : "ol";
      if (listType !== nextType) {
        closeList();
        listType = nextType;
        const start =
          listType === "ol" && orderedItemCount > 0 ? ` start="${orderedItemCount + 1}"` : "";
        output.push(`<${listType}${start}>`);
      }
      output.push(`<li>${renderInlineMarkdown((unordered || ordered)[1])}</li>`);
      if (ordered) orderedItemCount += 1;
      continue;
    }
    const quote = line.match(/^\s*&gt;\s?(.*)$/);
    if (quote) {
      closeList();
      output.push(`<blockquote>${renderInlineMarkdown(quote[1])}</blockquote>`);
      continue;
    }
    closeList();
    if (!line.trim()) {
      if (output.at(-1) !== "") output.push("");
    } else {
      output.push(`<p>${renderInlineMarkdown(line)}</p>`);
    }
  }
  closeList();
  return output.join("").replaceAll("\u0000", "");
}

function nowLabel() {
  return new Date().toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" });
}
function activeConversation() {
  return state.conversations.find((item) => item.id === state.activeId) || null;
}
function conversationById(id) {
  return state.conversations.find((item) => item.id === id) || null;
}

function loadConversations() {
  try {
    const parsed = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
    state.conversations = Array.isArray(parsed) ? parsed : [];
  } catch {
    state.conversations = [];
    state.historyLoadError = true;
  }
  state.activeId = state.conversations[0]?.id || null;
  try {
    state.sidebarCollapsed = Boolean(
      JSON.parse(localStorage.getItem(UI_STORAGE_KEY) || "{}").sidebarCollapsed,
    );
  } catch {
    state.sidebarCollapsed = false;
  }
}

function saveConversations() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(state.conversations.slice(0, 50)));
}

function createConversation() {
  state.taskOptionsTouched = false;
  const item = {
    id: uid(),
    convId: null,
    title: "新对话",
    createdAt: Date.now(),
    updatedAt: Date.now(),
    messages: [],
  };
  state.conversations.unshift(item);
  state.activeId = item.id;
  saveConversations();
  renderConversationList();
  renderChat();
  switchView("workspace");
  $("messageInput").focus();
  return item;
}

function selectConversation(id) {
  state.activeId = id;
  renderConversationList();
  renderChat();
  switchView("workspace");
  closeSidebar();
}

function deleteConversation(id) {
  state.conversations = state.conversations.filter((item) => item.id !== id);
  if (state.activeId === id) state.activeId = state.conversations[0]?.id || null;
  saveConversations();
  renderConversationList();
  renderChat();
}

function renderConversationList() {
  const list = $("conversationList");
  if (state.historyLoadError) {
    list.innerHTML = '<div class="history-empty">历史会话加载失败，本地记录未被覆盖。</div>';
    return;
  }
  if (!state.conversations.length) {
    list.innerHTML = '<div class="history-empty">还没有历史对话</div>';
    return;
  }
  list.innerHTML = state.conversations
    .map(
      (item) =>
        `<button class="conversation-item ${item.id === state.activeId ? "active" : ""}" data-conversation-id="${escapeHtml(item.id)}" title="${escapeHtml(item.title)}"><span>${escapeHtml(item.title)}</span><span class="conversation-delete" data-delete-id="${escapeHtml(item.id)}" role="button" tabindex="0" aria-label="删除对话">×</span></button>`,
    )
    .join("");
  list
    .querySelectorAll(".conversation-item")
    .forEach((button) =>
      button.addEventListener("click", () => selectConversation(button.dataset.conversationId)),
    );
  list.querySelectorAll(".conversation-delete").forEach((button) =>
    button.addEventListener("click", (event) => {
      event.stopPropagation();
      deleteConversation(button.dataset.deleteId);
    }),
  );
}

function messageTemplate(message, index) {
  const isUser = message.role === "user";
  const status = message.status || (message.error ? "failed" : "success");
  const knowledgeStatus = message.meta
    ? message.meta.knowledgeUsed
      ? message.meta.executionSuccess === false
        ? "已检索知识库，但回答生成失败"
        : {
            product_supported: "已找到产品依据",
            guidance_only: "仅找到写作规范",
            no_match: "产品资料不足",
          }[message.meta.evidenceStatus] || "知识库已参与本次回答"
      : "知识库未参与本次回答"
    : "";
  const meta = message.meta
    ? `<div class="message-meta">${[
        message.meta.intent && `Intent: ${message.meta.intent}`,
        message.meta.agents && `Agents: ${message.meta.agents}`,
        knowledgeStatus,
        message.meta.latency != null && `${message.meta.latency} ms`,
      ]
        .filter(Boolean)
        .map((value) => `<span>${escapeHtml(value)}</span>`)
        .join("")}</div>`
    : "";
  const actions = !isUser
    ? `<div class="message-actions">
    ${["failed", "timeout", "cancelled", "empty"].includes(status) ? `<button type="button" data-retry-id="${escapeHtml(message.id || "")}">重试</button>` : ""}
    ${status === "success" ? `<button type="button" data-regenerate-id="${escapeHtml(message.id || "")}">重新生成</button>` : ""}
    ${message.meta?.requestId ? `<button type="button" data-trace-id="${escapeHtml(message.meta.requestId)}">查看运行详情</button>` : ""}
  </div>`
    : "";
  const pending = status === "sending" || status === "executing";
  const content = pending
    ? `<div class="thinking"><i></i><i></i><i></i><span>${status === "sending" ? "正在发送任务" : "Agent 正在执行"}</span></div>`
    : isUser
      ? `<p>${escapeHtml(message.content).replaceAll("\n", "<br>")}</p>`
      : renderMarkdownLite(message.content);
  return `<article class="message ${isUser ? "message-user" : "message-agent"}" data-message-index="${index}" data-message-id="${escapeHtml(message.id || "")}"><div class="avatar ${isUser ? "user" : "assistant"}">${isUser ? "你" : "C"}</div><div><div class="message-head"><strong>${isUser ? "你" : "ContentOpsMind"}</strong><span class="message-time">${escapeHtml(message.time || "")}</span></div><div class="message-content ${message.error ? "error-message" : ""}">${content}</div>${meta}${actions}</div></article>`;
}

function renderChat() {
  const conversation = activeConversation();
  const messages = conversation?.messages || [];
  $("welcomeState").hidden = messages.length > 0;
  $("messageList").innerHTML = messages.map(messageTemplate).join("");
  $("currentConversationTitle").textContent = conversation?.title || "新对话";
  updateRequestUI();
  requestAnimationFrame(scrollToBottom);
}

function typeAssistantMessage(conversationId, messageIndex) {
  const conversation = activeConversation();
  const message = conversation?.messages?.[messageIndex];
  if (!message || message.role !== "assistant" || conversation.id !== conversationId) return;
  const article = document.querySelector(`.message[data-message-index="${messageIndex}"]`);
  const content = article?.querySelector(".message-content");
  if (!content || window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;

  const token = ++state.typingToken;
  const text = String(message.content || "");
  const chunkSize = text.length > 700 ? 4 : text.length > 260 ? 2 : 1;
  let position = 0;
  content.textContent = "";
  content.classList.add("typing-cursor");

  const tick = () => {
    if (token !== state.typingToken || state.activeId !== conversationId || !content.isConnected)
      return;
    position = Math.min(position + chunkSize, text.length);
    content.textContent = text.slice(0, position);
    scrollToBottom();
    if (position < text.length) {
      window.setTimeout(tick, 18);
    } else {
      content.classList.remove("typing-cursor");
      content.innerHTML = renderMarkdownLite(text);
      scrollToBottom();
    }
  };
  tick();
}

function scrollToBottom() {
  const el = $("chatScroll");
  el.scrollTop = el.scrollHeight;
}

function requestSnapshot(raw, conversation) {
  const snapshot = {
    message: raw,
    user_id: $("userId").value.trim() || "ops_demo",
    conv_id: conversation.convId || undefined,
  };
  if (state.taskOptionsTouched) {
    snapshot.task_type = $("taskType").value;
    snapshot.channel = $("channel").value;
    snapshot.tone = $("tone").value;
  }
  return snapshot;
}

function updateRequestUI() {
  const active = state.activeRequest;
  const sendButton = $("sendBtn");
  const input = $("messageInput");
  if (!sendButton || !input) return;
  sendButton.classList.toggle("is-cancelling", Boolean(active));
  sendButton.setAttribute("aria-label", active ? "取消当前请求" : "发送消息");
  sendButton.disabled = !active && !input.value.trim();
  $("requestStatus").textContent = active ? "正在执行内容任务；切换会话不会改变结果归属。" : "";
  $("composerStatus").textContent = active ? "仅允许一个活动请求" : "";
}

function cancelActiveRequest() {
  if (!state.activeRequest) return;
  state.activeRequest.terminationReason = "user_cancelled";
  state.activeRequest.controller.abort();
}

function classifyRequestError(error, request) {
  if (error?.name === "AbortError" && request.terminationReason === "timeout") {
    return {
      status: "timeout",
      content: "模型响应超时。你可以重试此请求。",
      toast: "模型响应超时",
    };
  }
  if (error?.name === "AbortError" && request.terminationReason === "user_cancelled") {
    return {
      status: "cancelled",
      content: "请求已取消。后端任务可能仍在执行。",
      toast: "请求已取消",
    };
  }
  if (/Failed to fetch|NetworkError|503|服务未就绪/i.test(error?.message || "")) {
    return {
      status: "failed",
      content: "后端服务当前不可用。请检查服务状态后重试。",
      toast: "后端服务不可用",
    };
  }
  return {
    status: "failed",
    content: `请求失败：${error?.message || "未知错误"}`,
    toast: "请求失败",
  };
}

async function executeChat({ conversation, snapshot, placeholder, replacePlaceholder = false }) {
  if (state.activeRequest || state.campaignRequest) {
    showToast("当前已有任务正在执行");
    return;
  }
  const request = {
    requestToken: uid(),
    localConversationId: conversation.id,
    backendConversationId: conversation.convId,
    controller: new AbortController(),
    timeoutHandle: null,
    terminationReason: null,
    snapshot: { ...snapshot },
    placeholderId: placeholder.id,
    replacePlaceholder,
  };
  state.activeRequest = request;
  placeholder.status = "executing";
  placeholder.error = false;
  placeholder.content = "";
  saveConversations();
  renderConversationList();
  renderChat();
  request.timeoutHandle = window.setTimeout(() => {
    if (state.activeRequest?.requestToken !== request.requestToken) return;
    request.terminationReason = "timeout";
    request.controller.abort();
  }, REQUEST_TIMEOUT_MS);

  let animateTarget = null;
  try {
    const data = await api("/chat", {
      method: "POST",
      signal: request.controller.signal,
      body: JSON.stringify(snapshot),
    });
    const target = conversationById(request.localConversationId);
    if (!target) return;
    const targetIndex = target.messages.findIndex((item) => item.id === request.placeholderId);
    if (targetIndex < 0) return;
    const responseText = String(data.response ?? "").trim();
    const executionSuccess = data.success !== false;
    target.convId = data.conv_id;
    target.messages[targetIndex] = {
      ...target.messages[targetIndex],
      role: "assistant",
      content: responseText || "本次请求没有返回可展示内容。",
      time: nowLabel(),
      status: executionSuccess ? (responseText ? "success" : "empty") : "failed",
      error: !executionSuccess || !responseText,
      request: request.snapshot,
      meta: {
        intent: data.intent,
        agents: (data.routed_agents || []).join(", ") || data.agent_type,
        knowledgeUsed: Boolean(data.knowledge_used),
        evidenceStatus: data.evidence_status,
        executionSuccess,
        errorCode: data.error_code || null,
        latency: data.latency_ms,
        requestId: data.request_id,
      },
    };
    target.updatedAt = Date.now();
    state.lastRequestId = data.request_id;
    animateTarget =
      executionSuccess && responseText
        ? { conversationId: target.id, messageIndex: targetIndex }
        : null;
    loadTraceList(false);
  } catch (error) {
    const target = conversationById(request.localConversationId);
    const targetMessage = target?.messages.find((item) => item.id === request.placeholderId);
    if (targetMessage) {
      const failure = classifyRequestError(error, request);
      targetMessage.status = failure.status;
      targetMessage.content = failure.content;
      targetMessage.error = true;
      targetMessage.time = nowLabel();
      targetMessage.request = request.snapshot;
      target.updatedAt = Date.now();
      showToast(failure.toast);
    }
  } finally {
    window.clearTimeout(request.timeoutHandle);
    if (state.activeRequest?.requestToken === request.requestToken) state.activeRequest = null;
    saveConversations();
    renderConversationList();
    renderChat();
    if (animateTarget && state.activeId === animateTarget.conversationId) {
      typeAssistantMessage(animateTarget.conversationId, animateTarget.messageIndex);
    }
  }
}

async function sendChat() {
  if (state.activeRequest) {
    cancelActiveRequest();
    return;
  }
  const raw = $("messageInput").value.trim();
  if (!raw) return;
  const conversation = activeConversation() || createConversation();
  if (!conversation.messages.length)
    conversation.title = raw.replace(/\s+/g, " ").slice(0, 24) || "新对话";
  const snapshot = requestSnapshot(raw, conversation);
  conversation.messages.push({
    id: uid(),
    role: "user",
    content: raw,
    time: nowLabel(),
    request: snapshot,
  });
  const placeholder = {
    id: uid(),
    role: "assistant",
    content: "",
    time: nowLabel(),
    status: "sending",
    request: snapshot,
  };
  conversation.messages.push(placeholder);
  conversation.updatedAt = Date.now();
  $("messageInput").value = "";
  resizeComposer();
  await executeChat({ conversation, snapshot, placeholder });
}

async function retryMessage(messageId) {
  if (state.activeRequest || state.campaignRequest) return showToast("当前已有任务正在执行");
  const conversation = activeConversation();
  const placeholder = conversation?.messages.find((item) => item.id === messageId);
  if (
    !placeholder?.request ||
    !["failed", "timeout", "cancelled", "empty"].includes(placeholder.status)
  )
    return;
  await executeChat({
    conversation,
    snapshot: { ...placeholder.request },
    placeholder,
    replacePlaceholder: true,
  });
}

async function regenerateMessage(messageId) {
  if (state.activeRequest || state.campaignRequest) return showToast("当前已有任务正在执行");
  const conversation = activeConversation();
  const source = conversation?.messages.find((item) => item.id === messageId);
  if (!source?.request || source.status !== "success") return;
  const placeholder = {
    id: uid(),
    role: "assistant",
    content: "",
    time: nowLabel(),
    status: "executing",
    request: { ...source.request },
  };
  conversation.messages.push(placeholder);
  conversation.updatedAt = Date.now();
  await executeChat({ conversation, snapshot: { ...source.request }, placeholder });
}

function resizeComposer() {
  const input = $("messageInput");
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 180)}px`;
}
function prettyJson(data) {
  return JSON.stringify(data, null, 2);
}

async function checkHealth() {
  try {
    const data = await api("/health");
    const ok = data.status === "ok";
    $("apiStatus").textContent = ok ? "正常" : "异常";
    $("statusDot").classList.toggle("online", ok);
    $("connectionState").innerHTML =
      `<span class="status-dot ${ok ? "online" : ""}" aria-hidden="true"></span>${ok ? "服务已连接" : "服务异常"}`;
  } catch {
    $("apiStatus").textContent = "不可用";
    $("statusDot").classList.remove("online");
    $("connectionState").innerHTML =
      '<span class="status-dot" aria-hidden="true"></span>服务不可用';
  }
}

function renderTraceList(items) {
  const list = $("traceList");
  if (!items?.length) {
    list.innerHTML = '<div class="empty-state">暂无 Trace，请先发送一条消息。</div>';
    return;
  }
  list.innerHTML = items
    .map(
      (item) =>
        `<button class="trace-item" data-request-id="${escapeHtml(item.request_id)}"><strong>${escapeHtml(item.message_preview || item.request_id)}</strong><span>${escapeHtml(item.intent?.primary || "unknown")} · ${escapeHtml((item.routing || []).join(", ") || "-")} · ${item.latency_ms || 0} ms</span><span>${escapeHtml(item.request_id)}</span></button>`,
    )
    .join("");
  list
    .querySelectorAll(".trace-item")
    .forEach((button) =>
      button.addEventListener("click", () => loadTraceDetail(button.dataset.requestId)),
    );
}

async function loadTraceList(notify = true) {
  try {
    const data = await api("/trace/recent?limit=20");
    renderTraceList(data.items || []);
    if (notify) showToast("Trace 已刷新");
  } catch (error) {
    $("traceList").innerHTML =
      `<div class="empty-state">加载失败：${escapeHtml(error.message)}</div>`;
  }
}

function renderRetrievalAudit(metadata) {
  const table = (items) =>
    `<div style="overflow:auto"><table><thead><tr><th>序号</th><th>标题 / 来源</th><th>检索分数</th><th>正文</th></tr></thead><tbody>${items.map((i, n) => `<tr><td>${n}</td><td>${escapeHtml(i.title || "")}<br>${escapeHtml(i.source_url || "")}</td><td>${escapeHtml(i.score)}<br>词法 ${escapeHtml(i.lexical_score)} / 向量 ${escapeHtml(i.vector_score)}</td><td>${escapeHtml(String(i.content || "").slice(0, 800))}</td></tr>`).join("")}</tbody></table></div>`;
  const queries = (metadata.query_results || [])
    .map(
      (q) =>
        `<details><summary>${escapeHtml(q.lane)} · ${escapeHtml(q.query)} · ${(q.candidates || []).length} 个候选</summary>${table(q.candidates || [])}</details>`,
    )
    .join("");
  const rankings = (metadata.rankings || [])
    .map(
      (r) =>
        `<details open><summary>${escapeHtml(r.lane)} 重排：${escapeHtml(r.status)} ${escapeHtml(r.reason || "")}</summary><p>模型顺序：${escapeHtml(JSON.stringify(r.model_order || []))}；最终顺序：${escapeHtml(JSON.stringify(r.effective_order || []))}</p>${table(r.candidates || [])}</details>`,
    )
    .join("");
  return `<section><h3>召回与重排</h3><p>证据状态：${escapeHtml(metadata.evidence_status || "no_match")}；模型仅排序，没有生成重排分数。</p>${queries}${rankings}</section>`;
}

function renderTraceDetail(trace) {
  const retrieval = trace.retrieval || {},
    synthesis = trace.synthesis || {};
  const steps = [
    [
      "Intent Recognition",
      `primary=${trace.intent?.primary || "-"}; secondary=${(trace.intent?.secondary || []).join(", ") || "-"}`,
    ],
    ["Routing", (trace.routing || []).join(", ") || "-"],
    [
      "RAG Retrieval",
      retrieval.used
        ? `queries=${(retrieval.queries || []).join(" | ")}; docs=${(retrieval.docs || []).map((d) => d.title).join(", ")}; latency=${retrieval.latency_ms || 0} ms`
        : "not used",
    ],
    ...(trace.agent_execution || []).map((agent) => [
      `Agent: ${agent.agent} · ${agent.phase || "execute"}`,
      `success=${agent.success}; latency=${agent.latency_ms || 0} ms; tools=${(agent.tool_calls || []).map((call) => call.tool).join(", ") || "none"}`,
    ]),
    ["Synthesis", `used=${synthesis.used || false}; latency=${synthesis.latency_ms || 0} ms`],
    ["Total", `status=${trace.status || "-"}; latency=${trace.latency_ms || 0} ms`],
  ];
  $("traceDetail").classList.remove("empty-state");
  $("traceDetail").innerHTML =
    `<div class="timeline">${steps.map(([title, body]) => `<div class="timeline-step"><h4>${escapeHtml(title)}</h4><p>${escapeHtml(body)}</p></div>`).join("")}</div>${renderRetrievalAudit(retrieval.metadata || {})}<pre class="json-box">${escapeHtml(prettyJson(trace))}</pre>`;
}

async function loadTraceDetail(id) {
  try {
    const data = await api(`/trace/${encodeURIComponent(id)}`);
    state.currentTrace = data;
    renderTraceDetail(data);
  } catch (error) {
    $("traceDetail").innerHTML =
      `<div class="empty-state">加载失败：${escapeHtml(error.message)}</div>`;
  }
}

function renderMetrics(data) {
  const cards = [
    ["请求量", data.request_count ?? 0],
    ["平均延迟", `${data.avg_latency_ms ?? 0} ms`],
    ["知识库使用率", `${Math.round((data.rag_hit_rate ?? 0) * 1000) / 10}%`],
    ["工具调用", data.tool_call_count ?? 0],
    ["工具成功率", `${Math.round((data.tool_success_rate ?? 0) * 1000) / 10}%`],
    ["参数失败率", `${Math.round((data.tool_validation_failure_rate ?? 0) * 1000) / 10}%`],
    ["失败率", `${Math.round((data.failure_rate ?? 0) * 1000) / 10}%`],
    ["检索延迟", `${data.avg_retrieval_latency_ms ?? 0} ms`],
  ];
  $("metricGrid").innerHTML = cards
    .map(
      ([label, value]) =>
        `<div class="metric-card"><span>${label}</span><strong>${value}</strong></div>`,
    )
    .join("");
}

async function loadDashboard() {
  try {
    const data = await api("/dashboard");
    renderMetrics(data);
    $("dashboardOutput").textContent = prettyJson(data);
  } catch (error) {
    $("dashboardOutput").textContent = `加载失败：${error.message}`;
  }
}

async function runEval(mode) {
  const buttons = {
      smoke: $("runSmokeEvalBtn"),
      intent: $("runIntentEvalBtn"),
      full: $("runFullEvalBtn"),
    },
    button = buttons[mode];
  button.disabled = true;
  $("evalOutput").textContent = "评测运行中，请稍候...";
  try {
    let body = {};
    if (mode === "intent") body = { dialog_cases: [] };
    if (mode === "smoke")
      body = {
        intent_cases: [],
        dialog_cases: [
          { question: "帮我写一条双十一满399减80的短信文案" },
          { question: "检查文案中的广告法风险" },
          { question: "把新品文案改成小红书风格" },
        ],
      };
    const data = await api("/eval/run", { method: "POST", body: JSON.stringify(body) });
    $("evalOutput").textContent = prettyJson({
      pass_rate: data.pass_rate,
      total: data.total,
      passed: data.passed,
      avg_scores: data.avg_scores,
      regressions: data.regressions,
      recommendations: data.recommendations,
    });
  } catch (error) {
    $("evalOutput").textContent = `评测失败：${error.message}`;
  } finally {
    button.disabled = false;
  }
}

function renderKnowledgeEvalResults(results = []) {
  const container = $("knowledgeEvalResults");
  if (!results.length) {
    container.innerHTML = "";
    return;
  }
  container.innerHTML = results
    .map((item) => {
      const failedChecks = Object.entries(item.assertions || {})
        .filter(([, passed]) => !passed)
        .map(([name]) => name);
      const detail = item.infrastructure_error || failedChecks.join(", ") || "全部断言通过";
      const traceButton = item.request_id
        ? `<button class="eval-trace-button" data-eval-trace-id="${escapeHtml(item.request_id)}">查看 Trace</button>`
        : "";
      return `<div class="eval-case ${item.passed ? "is-passed" : "is-failed"}"><div><strong>${item.passed ? "通过" : "失败"} · ${escapeHtml(item.id)}</strong><span>${escapeHtml(detail)}</span></div>${traceButton}</div>`;
    })
    .join("");
  container.querySelectorAll("[data-eval-trace-id]").forEach((button) => {
    button.addEventListener("click", () => openTraceFromMessage(button.dataset.evalTraceId));
  });
}

async function runKnowledgeEval() {
  const button = $("runKnowledgeEvalBtn");
  button.disabled = true;
  $("evalOutput").textContent = "知识问答回归运行中，将调用真实 /chat 与 /trace...";
  $("knowledgeEvalResults").innerHTML = "";
  try {
    const data = await api("/eval/knowledge", {
      method: "POST",
      body: JSON.stringify({ fail_fast: false }),
    });
    const summaryKeys = [
      "total",
      "passed",
      "failed",
      "pass_rate",
      "expected_coverage_rate",
      "forbidden_violation_rate",
      "intent_accuracy",
      "source_match_rate",
      "memory_policy_pass_rate",
      "infrastructure_failures",
      "avg_latency_ms",
      "p50_latency_ms",
      "p95_latency_ms",
    ];
    const summary = Object.fromEntries(summaryKeys.map((key) => [key, data[key]]));
    $("evalOutput").textContent = prettyJson(summary);
    renderKnowledgeEvalResults(data.results || []);
  } catch (error) {
    $("evalOutput").textContent = `知识问答回归失败：${error.message}`;
  } finally {
    button.disabled = false;
  }
}

async function searchKnowledge() {
  const button = $("searchBtn");
  button.disabled = true;
  try {
    const data = await api(
      `/search?query=${encodeURIComponent($("searchInput").value.trim())}&top_k=5`,
      { method: "POST" },
    );
    $("knowledgeOutput").textContent = prettyJson(data);
  } catch (error) {
    $("knowledgeOutput").textContent = `检索失败：${error.message}`;
  } finally {
    button.disabled = false;
  }
}
async function loadKnowledgeStats() {
  try {
    $("knowledgeOutput").textContent = prettyJson(await api("/knowledge/stats"));
  } catch (error) {
    $("knowledgeOutput").textContent = `统计失败：${error.message}`;
  }
}

const CAMPAIGN_TIMEOUT_MS = 300000;
const CAMPAIGN_STATUS_LABELS = {
  draft: "草稿",
  validated: "Brief 已校验",
  planned: "已规划",
  retrieving: "知识检索中",
  generating: "内容生成中",
  needs_clarification: "等待补充或修正 Brief",
  fact_checking: "事实核验中",
  compliance_reviewing: "合规审核中",
  reviewing: "并行审核中",
  review_incomplete: "审核未完成",
  awaiting_approval: "等待人工审批",
  changes_requested: "需要修改",
  approved: "已批准",
  delivered: "已交付",
  failed: "执行失败",
  cancelled: "已取消",
};
const CAMPAIGN_STAGES = [
  ["brief_validation", "Brief"],
  ["planning", "规划"],
  ["knowledge_retrieval", "检索"],
  ["content_generation", "生成"],
  ["parallel_review", "并行审核"],
  ["human_approval", "人工审批"],
  ["delivery", "交付"],
];
const LEGACY_CAMPAIGN_STAGES = [
  ["brief_validation", "Brief"],
  ["planning", "规划"],
  ["knowledge_retrieval", "检索"],
  ["content_generation", "生成"],
  ["fact_check", "事实核验"],
  ["compliance_review", "合规审核"],
  ["human_approval", "人工审批"],
  ["delivery", "交付"],
];

function campaignDate(value) {
  if (!value) return "-";
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? "-"
    : date.toLocaleString("zh-CN", {
        month: "2-digit",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
      });
}

function campaignLines(value) {
  return String(value || "")
    .split(/\r?\n/)
    .map((item) => item.trim())
    .filter(Boolean);
}

function campaignFacts(value) {
  return Object.fromEntries(
    campaignLines(value)
      .map((line) => {
        const index = line.search(/[:：]/);
        return index > 0
          ? [line.slice(0, index).trim(), line.slice(index + 1).trim()]
          : [line, "已确认"];
      })
      .filter(([, fact]) => fact),
  );
}

function campaignLatestVersions(detail = state.campaignDetail) {
  if (!detail?.workflow) return [];
  const ids = new Set(Object.values(detail.workflow.latest_versions || {}));
  return (detail.versions || []).filter((version) => ids.has(version.version_id));
}

function campaignApprovalReady(detail = state.campaignDetail) {
  return Boolean(
    detail?.workflow?.status === "awaiting_approval" &&
    detail.review_complete === true &&
    detail.review_status === "pass" &&
    campaignLatestVersions(detail).length,
  );
}

function campaignPendingReviews(detail = state.campaignDetail) {
  if (!detail) return 0;
  const rows = campaignLatestVersions(detail).flatMap((v) =>
    ["fact_check", "compliance", "content_quality"].map(
      (type) => detail.review_progress?.[v.version_id]?.[type],
    ),
  );
  const count = rows.filter((r) => !r || r.decision !== "pass").length;
  return campaignApprovalReady(detail) ? 0 : Math.max(1, count);
}

function campaignRevisionVersion(detail = state.campaignDetail) {
  const versions = campaignLatestVersions(detail);
  return (
    versions.find((v) =>
      (detail.reviews || []).some(
        (r) =>
          r.version_id === v.version_id && ["changes_required", "blocked"].includes(r.decision),
      ),
    ) || versions[0]
  );
}

function showCampaignReview(reviewId) {
  const row = document.getElementById(`campaign-review-${reviewId}`);
  if (!row) return;
  row.querySelectorAll("details").forEach((node) => {
    node.open = true;
  });
  row.scrollIntoView({ behavior: "smooth", block: "center" });
  row.tabIndex = -1;
  row.focus({ preventScroll: true });
}

function campaignSourceUrl(value) {
  try {
    const url = new URL(value);
    return ["https:", "http:"].includes(url.protocol) ? url.href : "";
  } catch {
    return "";
  }
}

function renderCampaignSources(detail) {
  const documents = detail.workflow?.knowledge_snapshot?.documents || [];
  const items = documents.length ? documents : detail.citations || [];
  const groups = new Map();
  items.forEach((item) => {
    const title = String(item.structured_data?.source_title || item.title || "来源资料").replace(
      /\s*\|\s*\d+:\d+\s*$/,
      "",
    );
    const url = campaignSourceUrl(item.source_url),
      key = url || item.structured_data?.source_id || title;
    if (!groups.has(key)) groups.set(key, { title, url, items: [] });
    groups.get(key).items.push(item);
  });
  $("campaignCitationList").innerHTML =
    [...groups.values()]
      .map((group) => {
        const publisher =
          group.url &&
          (new URL(group.url).hostname === "starbucks.com.cn" ||
            new URL(group.url).hostname.endsWith(".starbucks.com.cn"))
            ? "星巴克官方新闻"
            : "来源资料";
        const ids = new Set(group.items.map((i) => i.document_id));
        const facts = [
          ...new Set(
            (detail.reviews || [])
              .filter((r) => r.review_type === "fact_check")
              .flatMap((r) => r.evidence || [])
              .filter((e) => (e.evidence_refs || []).some((id) => ids.has(id)))
              .map((e) => e.matched_text)
              .filter(Boolean),
          ),
        ];
        const factReviews = facts.length
          ? facts
          : [
              ...new Set(
                (detail.reviews || [])
                  .filter(
                    (r) =>
                      r.review_type === "fact_check" &&
                      r.decision === "pass" &&
                      (detail.versions || []).some(
                        (v) =>
                          v.version_id === r.version_id &&
                          (v.citation_ids || []).some((id) => ids.has(id)),
                      ),
                  )
                  .map((r) => normalizedReviewView(r).summary)
                  .filter(Boolean),
              ),
            ];
        return `<article class="campaign-source-card"><small>${escapeHtml(publisher)}</small><strong>${escapeHtml(group.title)}</strong><span>引用了 ${group.items.length} 个片段</span><details><summary>查看引用内容</summary>${group.items.map((item, index) => `<section class="campaign-source-excerpt"><strong>片段 ${index + 1}</strong><p>${escapeHtml(item.content || "未保存片段正文")}</p></section>`).join("")}${factReviews.length ? `<section><strong>关联事实核验</strong><ul>${factReviews.map((f) => `<li>${escapeHtml(f)}</li>`).join("")}</ul></section>` : ""}</details>${group.url ? `<a href="${escapeHtml(group.url)}" target="_blank" rel="noopener noreferrer">打开原文</a>` : ""}</article>`;
      })
      .join("") || '<div class="campaign-ledger-empty small">尚未形成知识快照</div>';
}

function updateCampaignRequestUI(message = "") {
  const busy = Boolean(state.campaignRequest);
  $("createCampaignBtn").disabled = busy;
  const workflow = state.campaignDetail?.workflow;
  const hasVersions = campaignLatestVersions().length > 0;
  const availability = {
    runCampaignBtn: workflow
      ? ![
          "validated",
          "planned",
          "failed",
          "reviewing",
          "review_incomplete",
          "changes_requested",
        ].includes(workflow.status)
      : true,
    editCampaignBriefBtn: !workflow || workflow.status !== "needs_clarification",
    submitCampaignRevisionBtn: !state.campaignRevisionTarget,
    approveCampaignBtn: !campaignApprovalReady(),
    requestCampaignChangesBtn: !workflow || workflow.status !== "awaiting_approval" || !hasVersions,
    rejectCampaignBtn: !workflow || workflow.status !== "awaiting_approval" || !hasVersions,
    downloadCampaignJsonBtn: state.campaignDetail?.publish_allowed !== true,
    downloadCampaignMarkdownBtn: state.campaignDetail?.publish_allowed !== true,
  };
  Object.entries(availability).forEach(([id, disabled]) => {
    if ($(id)) $(id).disabled = busy || disabled;
  });
  const ready = campaignApprovalReady();
  $("approveCampaignBtn").classList.toggle("primary", ready);
  $("approveCampaignBtn").classList.toggle("secondary", !ready);
  $("campaignApprovalGate").textContent = ready
    ? "审核已通过，可对当前版本作出审批决定。"
    : `仍有 ${campaignPendingReviews()} 项审核待解决，解决后才能批准。`;
  if ($("cancelCampaignBtn")) $("cancelCampaignBtn").hidden = !busy;
  if (message) $("campaignActionStatus").textContent = message;
}

async function withCampaignRequest(label, operation) {
  if (state.activeRequest || state.campaignRequest) {
    showToast("当前已有请求正在执行");
    return null;
  }
  const request = {
    controller: new AbortController(),
    terminationReason: null,
    timeoutHandle: null,
  };
  state.campaignRequest = request;
  $("campaignActionStatus").classList.remove("is-error");
  updateCampaignRequestUI(label);
  request.timeoutHandle = window.setTimeout(() => {
    if (state.campaignRequest !== request) return;
    request.terminationReason = "timeout";
    request.controller.abort();
  }, CAMPAIGN_TIMEOUT_MS);
  try {
    return await operation(request.controller.signal);
  } catch (error) {
    $("campaignActionStatus").classList.add("is-error");
    if (error?.name === "AbortError" && request.terminationReason === "timeout")
      $("campaignActionStatus").textContent = "工作流请求超时，可稍后刷新查看后端执行结果。";
    else if (error?.name === "AbortError" && request.terminationReason === "user_cancelled")
      $("campaignActionStatus").textContent = "已取消浏览器等待；后端工作流可能仍在执行。";
    else $("campaignActionStatus").textContent = `操作失败：${error?.message || "未知错误"}`;
    return null;
  } finally {
    window.clearTimeout(request.timeoutHandle);
    if (state.campaignRequest === request) state.campaignRequest = null;
    updateCampaignRequestUI();
  }
}

function cancelCampaignRequest() {
  if (!state.campaignRequest) return;
  state.campaignRequest.terminationReason = "user_cancelled";
  state.campaignRequest.controller.abort();
}

function renderCampaignWorkflowList() {
  const items = state.campaignWorkflows || [];
  $("campaignWorkflowCount").textContent = String(items.length);
  $("campaignWorkflowList").innerHTML = items.length
    ? items
        .map(
          (item) => `
    <button class="campaign-workflow-item ${item.workflow_id === state.activeCampaignWorkflowId ? "active" : ""}" type="button" data-campaign-id="${escapeHtml(item.workflow_id)}">
      <span class="campaign-workflow-state" data-status="${escapeHtml(item.status)}"></span>
      <span><strong>${escapeHtml(item.brief?.campaign_name || "未命名活动")}</strong><small>${escapeHtml((item.brief?.channels || []).join(" · "))}</small></span>
      <time>${escapeHtml(campaignDate(item.updated_at))}</time>
    </button>`,
        )
        .join("")
    : '<div class="campaign-list-empty">暂无活动工作流</div>';
}

async function loadCampaignWorkflows(selectId = null) {
  try {
    const userId = $("userId").value.trim();
    const data = await api(`/campaign-workflows?user_id=${encodeURIComponent(userId)}&limit=30`);
    state.campaignWorkflows = data.items || [];
    renderCampaignWorkflowList();
    const target = selectId || state.activeCampaignWorkflowId;
    if (target) await loadCampaignDetail(target);
  } catch (error) {
    $("campaignWorkflowList").innerHTML =
      `<div class="campaign-list-empty is-error">加载失败：${escapeHtml(error.message)}</div>`;
  }
}

async function createCampaignWorkflow(event) {
  event.preventDefault();
  const channels = [...document.querySelectorAll('input[name="campaignChannel"]:checked')].map(
    (input) => input.value,
  );
  if (!channels.length) {
    $("campaignFormStatus").textContent = "请至少选择一个交付渠道。";
    return;
  }
  const payload = {
    user_id: $("userId").value.trim() || "anonymous",
    brief: {
      campaign_name: $("campaignName").value.trim(),
      objective: $("campaignObjective").value.trim(),
      product: $("campaignProduct").value.trim(),
      audience: $("campaignAudience").value.trim(),
      channels,
      facts: campaignFacts($("campaignFacts").value),
      constraints: campaignLines($("campaignConstraints").value),
      brief_text: $("campaignBriefText").value.trim(),
    },
  };
  if (
    !payload.brief.campaign_name ||
    !payload.brief.objective ||
    !payload.brief.product ||
    !payload.brief.audience
  ) {
    $("campaignFormStatus").textContent = "请完整填写活动名称、目标、产品和受众。";
    return;
  }
  if (state.campaignBriefEditWorkflowId) {
    const detail = state.campaignDetail;
    const updated = await withCampaignRequest("正在保存修正后的 Brief…", (signal) =>
      api(`/campaign-workflows/${encodeURIComponent(state.campaignBriefEditWorkflowId)}/brief`, {
        method: "PATCH",
        signal,
        body: JSON.stringify({
          brief: payload.brief,
          expected_workflow_revision: detail.workflow.revision,
        }),
      }),
    );
    if (updated) {
      const workflowId = state.campaignBriefEditWorkflowId;
      state.campaignBriefEditWorkflowId = null;
      $("createCampaignBtn").textContent = "创建并保存 Brief";
      $("campaignFormStatus").textContent = "Brief 已修正，请重新启动工作流。";
      await loadCampaignWorkflows(workflowId);
    }
    return;
  }
  const operation = campaignCreateOperation(payload);
  payload.idempotency_key = operation.idempotencyKey;
  let requestError = null;
  const created = await withCampaignRequest("正在保存活动 Brief…", async (signal) => {
    try {
      return await api("/campaign-workflows", {
        method: "POST",
        signal,
        body: JSON.stringify(payload),
      });
    } catch (error) {
      requestError = error;
      throw error;
    }
  });
  if (!created) {
    // 网络中断、超时和 5xx 无法确认服务端是否已创建成功，保留同一个键供重试。
    // 明确的 4xx 表示请求未被接受或发生业务冲突，下一次提交应视为新操作。
    if (
      requestError?.status >= 400 &&
      requestError.status < 500 &&
      requestError.status !== 408 &&
      requestError.status !== 429
    ) {
      clearPendingCampaignCreate();
    } else {
      $("campaignFormStatus").textContent =
        "保存结果尚未确认；未修改 Brief 时再次提交会安全复用同一幂等键。";
    }
    return;
  }
  clearPendingCampaignCreate();
  state.activeCampaignWorkflowId = created.workflow_id;
  $("campaignFormStatus").textContent = "Brief 已保存，可在右侧启动工作流。";
  await loadCampaignWorkflows(created.workflow_id);
}

function beginCampaignBriefEdit() {
  const detail = state.campaignDetail;
  if (!detail?.workflow || detail.workflow.status !== "needs_clarification") return;
  const brief = detail.workflow.brief;
  state.campaignBriefEditWorkflowId = detail.workflow.workflow_id;
  $("campaignName").value = brief.campaign_name || "";
  $("campaignObjective").value = brief.objective || "";
  $("campaignProduct").value = brief.product || "";
  $("campaignAudience").value = brief.audience || "";
  document.querySelectorAll('input[name="campaignChannel"]').forEach((input) => {
    input.checked = (brief.channels || []).includes(input.value);
  });
  $("campaignFacts").value = Object.entries(brief.facts || {})
    .map(([key, value]) => `${key}: ${value}`)
    .join("\n");
  $("campaignConstraints").value = (brief.constraints || []).join("\n");
  $("campaignBriefText").value = brief.brief_text || "";
  $("createCampaignBtn").textContent = "保存 Brief 修改";
  $("campaignFormStatus").textContent = "请修正冲突项；保存后系统将重新检索和生成。";
  $("campaignName").focus();
}

function resetCampaignBriefMode() {
  state.campaignBriefEditWorkflowId = null;
  $("createCampaignBtn").textContent = "创建并保存 Brief";
  $("campaignFormStatus").textContent = "Brief 保存后可启动 Agent 工作流。";
  clearPendingCampaignCreate();
}

async function loadCampaignDetail(workflowId) {
  try {
    if (state.campaignBriefEditWorkflowId && state.campaignBriefEditWorkflowId !== workflowId) {
      state.campaignBriefEditWorkflowId = null;
      $("campaignBriefForm").reset();
    }
    state.activeCampaignWorkflowId = workflowId;
    state.campaignDetail = await api(`/campaign-workflows/${encodeURIComponent(workflowId)}`);
    renderCampaignWorkflowList();
    renderCampaignDetail();
  } catch (error) {
    $("campaignActionStatus").classList.add("is-error");
    $("campaignActionStatus").textContent = `详情加载失败：${error.message}`;
  }
}

function renderCampaignStages(workflow, events) {
  const completed = new Set(
    (events || []).filter((event) => event.status === "succeeded").map((event) => event.stage),
  );
  const failed = new Set(
    (events || []).filter((event) => event.status === "failed").map((event) => event.stage),
  );
  const legacyStages = new Set(["fact_check", "compliance_review", "content_review"]);
  const stages =
    legacyStages.has(workflow.current_stage) &&
    !(events || []).some((event) => event.stage === "parallel_review")
      ? LEGACY_CAMPAIGN_STAGES
      : CAMPAIGN_STAGES;
  $("campaignStageRail").innerHTML = stages
    .map(([stage, label], index) => {
      const needsHumanRevision =
        stage === "human_approval" && workflow.status === "changes_requested";
      const awaitingApproval =
        stage === "human_approval" && workflow.status === "awaiting_approval";
      const approvalDone =
        stage === "human_approval" && ["approved", "delivered"].includes(workflow.status);
      const deliveryActive = stage === "delivery" && workflow.status === "approved";
      const deliveryDone = stage === "delivery" && workflow.status === "delivered";
      const mode = needsHumanRevision
        ? "action"
        : awaitingApproval || deliveryActive
          ? "active"
          : approvalDone || deliveryDone
            ? "done"
            : failed.has(stage)
              ? "failed"
              : completed.has(stage)
                ? "done"
                : workflow.current_stage === stage
                  ? "active"
                  : "pending";
      const visibleLabel = needsHumanRevision ? "人工修订" : label;
      return `<div class="campaign-stage is-${mode}"><span>${String(index + 1).padStart(2, "0")}</span><strong>${visibleLabel}</strong></div>`;
    })
    .join("");
}

function renderCampaignVersions(detail) {
  const latestIds = new Set(Object.values(detail.workflow.latest_versions || {}));
  const versions = [...(detail.versions || [])].sort(
    (a, b) =>
      Number(!latestIds.has(a.version_id)) - Number(!latestIds.has(b.version_id)) ||
      a.channel.localeCompare(b.channel, "zh-CN") ||
      b.version - a.version,
  );
  if (!versions.length) {
    $("campaignVersionList").innerHTML =
      '<div class="campaign-ledger-empty">尚未生成渠道内容。启动工作流后将形成首个版本。</div>';
    return;
  }
  const byId = Object.fromEntries(versions.map((item) => [item.version_id, item]));
  const revisionEnabled = !["reviewing", "review_incomplete"].includes(detail.workflow.status);
  $("campaignVersionList").innerHTML = versions
    .map((version) => {
      const latest = latestIds.has(version.version_id);
      const related = (detail.reviews || []).filter(
        (review) => review.version_id === version.version_id,
      );
      return `<article class="campaign-version ${latest ? "is-latest" : "is-history"}">
      <header><div><span class="campaign-channel">${escapeHtml(version.channel)}</span><strong>v${version.version}</strong>${latest ? "<em>当前版本</em>" : "<em>历史版本</em>"}</div><time>${escapeHtml(campaignDate(version.created_at))}</time></header>
      <div class="campaign-version-content markdown-body">${renderMarkdownLite(version.content || "")}</div>
      ${version.parent_version_id && byId[version.parent_version_id] ? `<details class="campaign-diff"><summary>查看与 v${byId[version.parent_version_id].version} 的差异</summary>${renderTextDiff(byId[version.parent_version_id].content, version.content)}</details>` : ""}
      <footer><div>${related.map((review) => `<button type="button" data-review-record="${escapeHtml(review.review_id)}" class="campaign-review-chip is-${escapeHtml(review.decision)}">${escapeHtml(REVIEW_TYPE_LABELS[review.review_type] || "渠道校验")}：${escapeHtml(REVIEW_DECISION_LABELS[review.decision] || review.decision)}</button>`).join("") || '<span class="campaign-review-chip">暂无审核</span>'}</div>
      ${latest && revisionEnabled ? `<button class="secondary campaign-revise-button" type="button" data-revise-version="${escapeHtml(version.version_id)}" data-revise-channel="${escapeHtml(version.channel)}">创建修订</button>` : ""}</footer>
    </article>`;
    })
    .join("");
}

function renderGenerationResolution(detail) {
  const panel = $("campaignClarificationPanel");
  const resolution = detail.generation_resolution;
  if (!resolution || detail.workflow.status !== "needs_clarification") {
    panel.hidden = true;
    panel.innerHTML = "";
    return;
  }
  const rows = (resolution.results || []).flatMap((result) =>
    (result.conflicts || []).map(
      (conflict) => `
    <div class="campaign-evidence is-high">
      <div><strong>${escapeHtml(result.channel)} · ${escapeHtml(conflict.category || "业务事实冲突")}</strong><em>需要澄清</em></div>
      <span>Brief 要求：${escapeHtml(conflict.requested_value || "未提供")}</span>
      <small>已确认依据：${escapeHtml(conflict.verified_value || "未提供")}</small>
      <small>证据：${escapeHtml([...(conflict.brief_refs || []), ...(conflict.evidence_refs || [])].join(", "))}</small>
      <small>建议：${escapeHtml(conflict.suggestion || "请修改 Brief 后重新运行")}</small>
    </div>`,
    ),
  );
  const rejected = (resolution.results || []).flatMap(
    (result) => result.rejected_constraints || [],
  );
  panel.innerHTML = `<div><strong>需要补充或修正 Brief</strong><span>当前尚未创建任何待审核内容版本</span></div>
    ${rows.join("") || '<div class="campaign-ledger-empty small">生成阶段要求澄清，但未返回可展示的冲突详情。</div>'}
    ${rejected.length ? `<p><strong>已拒绝的营销要求：</strong>${escapeHtml(rejected.join("；"))}</p>` : ""}`;
  panel.hidden = false;
}

function renderTextDiff(before = "", after = "") {
  const left = String(before).split(/\r?\n/),
    right = String(after).split(/\r?\n/);
  const removed = left.filter((line) => !right.includes(line));
  const added = right.filter((line) => !left.includes(line));
  return `<div class="text-diff">${removed.map((line) => `<div class="diff-remove">− ${escapeHtml(line)}</div>`).join("")}${added.map((line) => `<div class="diff-add">+ ${escapeHtml(line)}</div>`).join("") || '<div class="diff-same">文本无逐行变化</div>'}</div>`;
}

function normalizedReviewView(review) {
  let parsed = null;
  const rawSummary = String(review.summary || "").trim();
  if (rawSummary.startsWith("{")) {
    try {
      parsed = JSON.parse(rawSummary);
    } catch (_) {
      parsed = null;
    }
  }
  const parsedItems = Array.isArray(parsed?.risk_items)
    ? parsed.risk_items.map((item) => ({
        rule_id: item.rule_id || "unknown_rule",
        category: item.rule_id || "unknown_rule",
        matched_text: item.matched_text || "",
        source: (item.evidence_refs || []).join(", ") || "Campaign Brief",
        severity: item.severity || "medium",
        suggestion: item.suggestion || item.reason || "",
        impact: item.impact || "required_change",
      }))
    : [];
  return {
    summary: parsed?.summary || rawSummary || "未提供审核摘要",
    evidence:
      Array.isArray(review.evidence) && review.evidence.length ? review.evidence : parsedItems,
  };
}

const REVIEW_TYPE_LABELS = {
  fact_check: "事实审核",
  compliance: "合规审核",
  content_quality: "内容质量",
  channel_validation: "渠道校验",
};
const REVIEW_DECISION_LABELS = {
  pass: "通过",
  changes_required: "需修改",
  blocked: "已阻断",
  execution_failed: "执行失败",
};
const REVIEW_IMPACT_LABELS = {
  blocking_issue: "阻断问题",
  required_change: "必须修改",
  non_blocking_suggestion: "优化建议",
  delivery_configuration: "发布配置",
};
const REVIEW_RULE_LABELS = {
  missing_delivery_link: "活动链接待配置",
  incorrect_end_date: "活动截止时间不一致",
  missing_scope_qualifier: "适用范围披露不完整",
  missing_threshold: "优惠门槛披露不完整",
  missing_unsubscribe: "缺少退订提示",
  channel_length_exceeded: "渠道字数超限",
  promotion_tier_mismatch: "优惠档位不一致",
  ineligible_promotion_recommendation: "商品组合不满足优惠门槛",
  unsupported_efficacy: "无依据功效主张",
  absolute_price: "绝对价格主张",
  unknown_rule: "未识别风险规则",
};

function renderCampaignReviews(detail) {
  const reviews = [...(detail.reviews || [])].reverse();
  const progress = detail.review_progress || {};
  const latestIds = Object.values(detail.workflow?.latest_versions || {});
  const versionsById = Object.fromEntries(
    (detail.versions || []).map((version) => [version.version_id, version]),
  );
  const progressRows = latestIds
    .map((versionId) => {
      const version = versionsById[versionId];
      const heading = version ? `${version.channel} · v${version.version}` : "未关联版本";
      const chips = ["fact_check", "compliance", "content_quality"]
        .map((reviewType) => {
          const item = progress[versionId]?.[reviewType],
            decision = item?.decision || "pending";
          const label = item ? REVIEW_DECISION_LABELS[decision] || decision : "待执行";
          return `<span class="campaign-review-chip is-${escapeHtml(decision)}">${escapeHtml(REVIEW_TYPE_LABELS[reviewType])} · ${escapeHtml(label)}${item?.attempt ? ` · #${Number(item.attempt)}` : ""}</span>`;
        })
        .join("");
      return `<section class="campaign-review-channel"><strong>${escapeHtml(heading)}</strong><div>${chips}</div></section>`;
    })
    .join("");
  const aggregation = detail.review_aggregation || {};
  const progressHeader = latestIds.length
    ? `<div class="campaign-review-progress"><strong>并行审核进度</strong><div>${progressRows}</div><small>聚合状态：${escapeHtml(CAMPAIGN_STATUS_LABELS[aggregation.status] || aggregation.status || "待执行")}</small></div>`
    : "";
  $("campaignReviewList").innerHTML =
    progressHeader +
    (reviews.length
      ? reviews
          .map((review) => {
            const version = versionsById[review.version_id];
            const versionLabel = version
              ? `${version.channel} · v${version.version}`
              : "未关联版本";
            const view = normalizedReviewView(review),
              expanded = review.decision !== "pass" ? " open" : "";
            const evidence = view.evidence
              .map((item) => {
                const ruleId = item.rule_id || item.category || "unknown_rule";
                return `<div class="campaign-evidence is-${escapeHtml(item.severity || "medium")}">
      <div><strong title="${escapeHtml(ruleId)}">${escapeHtml(REVIEW_RULE_LABELS[ruleId] || "其他审核问题")}</strong><em>${escapeHtml(REVIEW_IMPACT_LABELS[item.impact] || item.impact || "必须修改")}</em></div>
      ${item.matched_text ? `<span>命中文本：${escapeHtml(item.matched_text)}</span>` : ""}
      <small>依据：${escapeHtml(item.source || "未标明")}</small>
      ${item.suggestion ? `<small>建议：${escapeHtml(item.suggestion)}</small>` : ""}</div>`;
              })
              .join("");
            return `<article id="campaign-review-${escapeHtml(review.review_id)}" class="campaign-review-row"><span class="campaign-decision-mark is-${escapeHtml(review.decision)}"></span>
      <div class="campaign-review-body"><header><strong>${escapeHtml(versionLabel)} · ${escapeHtml(REVIEW_TYPE_LABELS[review.review_type] || review.review_type)} · ${escapeHtml(REVIEW_DECISION_LABELS[review.decision] || review.decision)}</strong>${review.request_id ? `<button type="button" data-campaign-trace="${escapeHtml(review.request_id)}">Trace</button>` : ""}</header>
      <p>${escapeHtml(view.summary)}</p><small>${escapeHtml(review.agent)} · ${escapeHtml(campaignDate(review.created_at))}</small>
      ${evidence ? `<details class="campaign-evidence-list"${expanded}><summary>${view.evidence.length} 项结构化证据</summary>${evidence}</details>` : ""}</div></article>`;
          })
          .join("")
      : '<div class="campaign-ledger-empty">暂无审核记录</div>');
  const approvals = [...(detail.approvals || [])].reverse();
  $("campaignApprovalList").innerHTML = approvals.length
    ? approvals
        .map(
          (item) =>
            `<div class="campaign-approval-row"><strong>${escapeHtml(item.decision)}</strong><span>${escapeHtml(item.reviewer_id)} · ${escapeHtml(campaignDate(item.created_at))}</span><p>${escapeHtml(item.comment || "未填写意见")}</p></div>`,
        )
        .join("")
    : '<div class="campaign-ledger-empty small">尚未提交人工审批</div>';
}

function renderCampaignEvents(events) {
  $("campaignEventCount").textContent = String(events.length);
  $("campaignEventList").innerHTML = events.length
    ? [...events]
        .reverse()
        .map(
          (event) =>
            `<div class="campaign-event-row"><span class="is-${escapeHtml(event.status)}"></span><strong>${escapeHtml(event.stage)}</strong><small>${escapeHtml(event.channel || "全局")} · ${Math.round(Number(event.latency_ms || 0))} ms</small>${event.request_id ? `<button type="button" data-campaign-trace="${escapeHtml(event.request_id)}">Trace</button>` : ""}</div>`,
        )
        .join("")
    : '<div class="campaign-ledger-empty small">暂无执行事件</div>';
}

function renderCampaignDetail() {
  const detail = state.campaignDetail;
  if (!detail?.workflow) {
    $("campaignEmpty").hidden = false;
    $("campaignDetail").hidden = true;
    return;
  }
  const workflow = detail.workflow;
  $("campaignEmpty").hidden = true;
  $("campaignDetail").hidden = false;
  $("campaignDetailName").textContent = workflow.brief.campaign_name;
  const latestVersionLabels = campaignLatestVersions(detail)
    .map((version) => `${version.channel} v${version.version}`)
    .join(" / ");
  $("campaignWorkflowMeta").textContent =
    `${workflow.brief.product} · ${workflow.brief.audience} · ${workflow.brief.channels.join(" / ")} · 内容版本 ${latestVersionLabels || "未生成"} · 流程记录 #${workflow.revision}`;
  $("campaignStatusBadge").textContent = CAMPAIGN_STATUS_LABELS[workflow.status] || workflow.status;
  $("campaignStatusBadge").dataset.status = workflow.status;
  $("editCampaignBriefBtn").hidden = workflow.status !== "needs_clarification";
  $("runCampaignBtn").textContent =
    workflow.status === "changes_requested"
      ? `修改${campaignRevisionVersion(detail)?.channel || "渠道"}文案`
      : ["reviewing", "review_incomplete"].includes(workflow.status)
        ? "重试未完成审核"
        : workflow.status === "failed"
          ? "从失败阶段恢复"
          : workflow.status === "needs_clarification"
            ? "等待修改 Brief"
            : "启动工作流";
  const clarifying = workflow.status === "needs_clarification";
  [
    "approveCampaignBtn",
    "requestCampaignChangesBtn",
    "rejectCampaignBtn",
    "downloadCampaignJsonBtn",
    "downloadCampaignMarkdownBtn",
  ].forEach((id) => {
    $(id).hidden = clarifying;
  });
  $("campaignActionStatus").textContent = workflow.last_error
    ? `最近错误：${workflow.last_error}`
    : `当前阶段：${CAMPAIGN_STATUS_LABELS[workflow.status] || workflow.status}`;
  renderCampaignStages(workflow, detail.events || []);
  renderGenerationResolution(detail);
  if (workflow.status === "changes_requested") {
    const issue = (detail.reviews || []).find(
      (r) =>
        r.version_id === campaignRevisionVersion(detail)?.version_id &&
        r.decision === "changes_required",
    );
    if (issue?.summary) $("campaignActionStatus").textContent = issue.summary;
  }
  renderCampaignSources(detail);
  const summary = detail.execution_summary || {},
    recovery = detail.recoverability || {};
  const usageText = summary.usage_available
    ? Number(summary.total_tokens || 0).toLocaleString()
    : "历史数据未记录";
  const costText = summary.cost_available
    ? `$${Number(summary.estimated_cost_usd || 0).toFixed(6)}`
    : "成本未配置";
  $("campaignExecutionSummary").innerHTML =
    `<div class="campaign-summary-grid"><span>累计模型调用<strong>${summary.usage_available ? Number(summary.model_calls || 0) : "—"}</strong></span><span>输入 / 输出 Token<strong>${summary.usage_available ? `${Number(summary.input_tokens || 0).toLocaleString()} / ${Number(summary.output_tokens || 0).toLocaleString()}` : "—"}</strong></span><span>累计 Token<strong>${usageText}</strong></span><span>估算成本<strong>${costText}</strong></span><span title="仅统计阶段请求重试，不含内容修订或内部格式修复">阶段重试<strong>${Number(summary.retry_count || 0)}</strong></span><span>内容修订<strong>${Number(summary.revision_count || 0)}</strong></span><span>统计口径<strong>${summary.usage_accounting_version === 2 ? (summary.usage_complete ? "请求隔离 · 含检索" : "部分用量未返回") : "旧口径 · 可能不准"}</strong></span><span>恢复状态<strong>${recovery.recoverable ? `可从 ${escapeHtml(recovery.resume_from || "失败阶段")} 恢复` : "无需恢复"}</strong></span></div>`;
  renderCampaignVersions(detail);
  renderCampaignReviews(detail);
  renderCampaignEvents(detail.events || []);
  updateCampaignRequestUI();
}

async function runCampaignWorkflow() {
  const id = state.activeCampaignWorkflowId;
  if (!id) return;
  if (state.campaignDetail?.workflow?.status === "changes_requested") {
    const version = campaignRevisionVersion();
    if (version) openCampaignRevision(version.version_id, version.channel);
    const issue = (state.campaignDetail.reviews || []).find(
      (r) => r.version_id === version?.version_id && r.decision === "changes_required",
    );
    if (issue)
      $("campaignActionStatus").textContent = issue.summary || "请按审核建议修改当前文案。";
    return;
  }
  const result = await withCampaignRequest(
    "Agent 工作流执行中：检索、生成与审核可能需要数分钟…",
    (signal) =>
      api(`/campaign-workflows/${encodeURIComponent(id)}/run`, { method: "POST", signal }),
  );
  if (result) {
    $("campaignActionStatus").textContent = "工作流执行完成，正在刷新结果…";
    await loadCampaignWorkflows(id);
  }
}

function openCampaignRevision(versionId, channel) {
  state.campaignRevisionTarget = { versionId, channel };
  $("campaignRevisionChannel").textContent = `${channel} · 基于当前版本`;
  $("campaignRevisionInstruction").value = "";
  $("campaignRevisionPanel").hidden = false;
  updateCampaignRequestUI();
  $("campaignRevisionInstruction").focus();
}

function closeCampaignRevision() {
  $("campaignRevisionPanel").hidden = true;
  state.campaignRevisionTarget = null;
  updateCampaignRequestUI();
}

async function submitCampaignRevision() {
  const target = state.campaignRevisionTarget,
    detail = state.campaignDetail;
  const instruction = $("campaignRevisionInstruction").value.trim();
  if (!target || !detail || !instruction) {
    showToast("请填写明确的修订要求");
    return;
  }
  const payload = {
    channel: target.channel,
    instruction,
    parent_version_id: target.versionId,
    expected_workflow_revision: detail.workflow.revision,
  };
  const result = await withCampaignRequest(`正在重新生成 ${target.channel} 版本并复审…`, (signal) =>
    api(`/campaign-workflows/${encodeURIComponent(detail.workflow.workflow_id)}/revisions`, {
      method: "POST",
      signal,
      body: JSON.stringify(payload),
    }),
  );
  if (result) {
    closeCampaignRevision();
    await loadCampaignWorkflows(detail.workflow.workflow_id);
  }
}

async function submitCampaignApproval(decision) {
  const detail = state.campaignDetail,
    versions = campaignLatestVersions(detail);
  const reviewer = $("campaignReviewerId").value.trim(),
    comment = $("campaignApprovalComment").value.trim();
  if (!detail || !versions.length) {
    showToast("当前没有可审批的渠道版本");
    return;
  }
  if (decision === "approved" && !campaignApprovalReady(detail)) {
    showToast("仍有审核待解决，暂不能批准。");
    return;
  }
  if (!reviewer) {
    showToast("请填写审批人");
    return;
  }
  if (decision !== "approved" && !comment) {
    showToast("驳回或要求修改时必须填写审批意见");
    return;
  }
  const payload = {
    version_ids: versions.map((item) => item.version_id),
    decision,
    reviewer_id: reviewer,
    comment,
    expected_workflow_revision: detail.workflow.revision,
  };
  const result = await withCampaignRequest("正在提交人工审批决策…", (signal) =>
    api(`/campaign-workflows/${encodeURIComponent(detail.workflow.workflow_id)}/approvals`, {
      method: "POST",
      signal,
      body: JSON.stringify(payload),
    }),
  );
  if (result) {
    $("campaignApprovalComment").value = "";
    await loadCampaignWorkflows(detail.workflow.workflow_id);
  }
}

async function downloadCampaignPackage(format) {
  const id = state.activeCampaignWorkflowId;
  if (!id) return;
  const result = await withCampaignRequest(
    `正在准备 ${format.toUpperCase()} 交付包…`,
    async (signal) => {
      const response = await fetch(
        `${API_BASE_URL}/campaign-workflows/${encodeURIComponent(id)}/package?format=${encodeURIComponent(format)}`,
        { signal },
      );
      if (!response.ok) {
        const body = await response.text();
        throw new Error(body);
      }
      return {
        blob: await response.blob(),
        filename: `${id}.${format === "markdown" ? "md" : "json"}`,
      };
    },
  );
  if (!result) return;
  const url = URL.createObjectURL(result.blob),
    link = document.createElement("a");
  link.href = url;
  link.download = result.filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
  $("campaignActionStatus").textContent = `交付包已下载：${result.filename}`;
}

function switchView(name) {
  state.currentView = name;
  document
    .querySelectorAll(".nav-item")
    .forEach((button) => button.classList.toggle("active", button.dataset.view === name));
  document.querySelectorAll(".view").forEach((view) => view.classList.remove("active"));
  $(`${name}View`).classList.add("active");
  const viewMeta = {
    workspace: ["智能对话", "内容任务分析、Agent 编排与知识增强", "Agent + RAG", ""],
    campaign: ["活动工作流", "Brief、生成、审核、审批与交付闭环", "Campaign DAG", "内容生产闭环"],
    trace: ["Agent Trace", "查看意图、路由、检索与执行链路", "Observability", "运行记录"],
    evaluation: ["评测看板", "回归测试与系统质量指标", "Evaluation", "质量保障"],
    knowledge: ["知识库", "内容运营知识检索与数据统计", "Content RAG", "业务知识"],
  };
  const meta = viewMeta[name] || viewMeta.workspace;
  $("viewTitle").textContent = meta[0];
  $("viewSubtitle").textContent = meta[1];
  $("modelBadge").innerHTML = `<span aria-hidden="true"></span>${escapeHtml(meta[2])}`;
  if (name === "workspace") renderChat();
  else $("currentConversationTitle").textContent = meta[3];
  if (name === "trace") loadTraceList(false);
  if (name === "evaluation") loadDashboard();
  if (name === "campaign" && !state.campaignWorkflows.length) loadCampaignWorkflows();
  closeSidebar();
}

function openSidebar() {
  $("sidebar").classList.add("open");
  $("sidebarBackdrop").classList.add("show");
}
function closeSidebar() {
  $("sidebar").classList.remove("open");
  $("sidebarBackdrop").classList.remove("show");
}
function applySidebarState() {
  document.body.classList.toggle("sidebar-collapsed", state.sidebarCollapsed);
  $("collapseSidebarBtn").setAttribute("aria-expanded", String(!state.sidebarCollapsed));
  $("collapseSidebarBtn").setAttribute(
    "aria-label",
    state.sidebarCollapsed ? "展开侧栏" : "折叠侧栏",
  );
  $("collapseSidebarBtn").title = state.sidebarCollapsed ? "展开侧栏" : "折叠侧栏";
}
function toggleSidebarCollapse() {
  state.sidebarCollapsed = !state.sidebarCollapsed;
  localStorage.setItem(
    UI_STORAGE_KEY,
    JSON.stringify({ sidebarCollapsed: state.sidebarCollapsed }),
  );
  applySidebarState();
}

function refreshCurrentView() {
  if (state.currentView === "trace") loadTraceList(true);
  else if (state.currentView === "campaign") loadCampaignWorkflows(state.activeCampaignWorkflowId);
  else if (state.currentView === "evaluation") loadDashboard();
  else if (state.currentView === "knowledge") loadKnowledgeStats();
  else {
    checkHealth();
    showToast("服务状态已刷新");
  }
}

function openTraceFromMessage(requestId) {
  switchView("trace");
  loadTraceDetail(requestId);
}

function bindEvents() {
  document
    .querySelectorAll(".nav-item")
    .forEach((button) => button.addEventListener("click", () => switchView(button.dataset.view)));
  document.querySelectorAll("[data-prompt]").forEach((button) =>
    button.addEventListener("click", () => {
      $("messageInput").value = button.dataset.prompt;
      resizeComposer();
      updateRequestUI();
      $("messageInput").focus();
    }),
  );
  $("newChatBtn").addEventListener("click", createConversation);
  $("sendBtn").addEventListener("click", sendChat);
  $("messageInput").addEventListener("input", () => {
    resizeComposer();
    updateRequestUI();
  });
  $("messageInput").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendChat();
    }
  });
  $("messageList").addEventListener("click", (event) => {
    const retry = event.target.closest("[data-retry-id]");
    const regenerate = event.target.closest("[data-regenerate-id]");
    const trace = event.target.closest("[data-trace-id]");
    if (retry) retryMessage(retry.dataset.retryId);
    else if (regenerate) regenerateMessage(regenerate.dataset.regenerateId);
    else if (trace) openTraceFromMessage(trace.dataset.traceId);
  });
  $("toggleOptionsBtn").addEventListener("click", () => {
    const open = $("taskOptions").classList.toggle("open");
    if (open) state.taskOptionsTouched = true;
    $("toggleOptionsBtn").setAttribute("aria-expanded", String(open));
  });
  ["taskType", "channel", "tone"].forEach((id) => {
    $(id).addEventListener("change", () => {
      state.taskOptionsTouched = true;
    });
  });
  $("clearHistoryBtn").addEventListener("click", () => {
    if (!state.conversations.length || !confirm("确定清空本机保存的全部对话记录吗？")) return;
    state.conversations = [];
    state.activeId = null;
    saveConversations();
    renderConversationList();
    renderChat();
  });
  $("refreshAllBtn").addEventListener("click", refreshCurrentView);
  $("loadTraceBtn").addEventListener("click", () => loadTraceList(true));
  $("loadDashboardBtn").addEventListener("click", loadDashboard);
  $("runKnowledgeEvalBtn").addEventListener("click", runKnowledgeEval);
  $("runSmokeEvalBtn").addEventListener("click", () => runEval("smoke"));
  $("runIntentEvalBtn").addEventListener("click", () => runEval("intent"));
  $("runFullEvalBtn").addEventListener("click", () => runEval("full"));
  $("searchBtn").addEventListener("click", searchKnowledge);
  $("statsBtn").addEventListener("click", loadKnowledgeStats);
  $("campaignBriefForm").addEventListener("submit", createCampaignWorkflow);
  $("campaignBriefForm").addEventListener("input", clearPendingCampaignCreate);
  $("campaignBriefForm").addEventListener("reset", clearPendingCampaignCreate);
  $("campaignBriefForm").addEventListener("reset", resetCampaignBriefMode);
  $("loadCampaignWorkflowsBtn").addEventListener("click", () =>
    loadCampaignWorkflows(state.activeCampaignWorkflowId),
  );
  $("refreshCampaignBtn").addEventListener(
    "click",
    () => state.activeCampaignWorkflowId && loadCampaignDetail(state.activeCampaignWorkflowId),
  );
  $("runCampaignBtn").addEventListener("click", runCampaignWorkflow);
  $("editCampaignBriefBtn").addEventListener("click", beginCampaignBriefEdit);
  $("cancelCampaignBtn").addEventListener("click", cancelCampaignRequest);
  $("submitCampaignRevisionBtn").addEventListener("click", submitCampaignRevision);
  $("cancelCampaignRevisionBtn").addEventListener("click", closeCampaignRevision);
  $("approveCampaignBtn").addEventListener("click", () => submitCampaignApproval("approved"));
  $("requestCampaignChangesBtn").addEventListener("click", () =>
    submitCampaignApproval("changes_requested"),
  );
  $("rejectCampaignBtn").addEventListener("click", () => submitCampaignApproval("rejected"));
  $("downloadCampaignJsonBtn").addEventListener("click", () => downloadCampaignPackage("json"));
  $("downloadCampaignMarkdownBtn").addEventListener("click", () =>
    downloadCampaignPackage("markdown"),
  );
  $("campaignView").addEventListener("click", (event) => {
    const workflow = event.target.closest("[data-campaign-id]");
    const revision = event.target.closest("[data-revise-version]");
    const trace = event.target.closest("[data-campaign-trace]");
    const review = event.target.closest("[data-review-record]");
    if (review) {
      showCampaignReview(review.dataset.reviewRecord);
      return;
    }
    if (workflow) loadCampaignDetail(workflow.dataset.campaignId);
    else if (revision)
      openCampaignRevision(revision.dataset.reviseVersion, revision.dataset.reviseChannel);
    else if (trace) openTraceFromMessage(trace.dataset.campaignTrace);
  });
  $("openSidebarBtn").addEventListener("click", openSidebar);
  $("closeSidebarBtn").addEventListener("click", closeSidebar);
  $("sidebarBackdrop").addEventListener("click", closeSidebar);
  $("collapseSidebarBtn").addEventListener("click", toggleSidebarCollapse);
}

window.addEventListener("DOMContentLoaded", () => {
  if (API_BASE_URL) $("docsLink").href = `${API_BASE_URL}/docs`;
  loadConversations();
  bindEvents();
  applySidebarState();
  renderConversationList();
  renderChat();
  resizeComposer();
  updateRequestUI();
  checkHealth();
  loadTraceList(false);
  if (state.historyLoadError) showToast("历史会话加载失败，本地记录未被覆盖");
});
