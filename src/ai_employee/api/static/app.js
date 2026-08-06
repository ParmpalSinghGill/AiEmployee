/* AI Employee Console — dependency-free SPA.
 * Tabs = sessions; each session holds an initial run + follow-ups. We poll each
 * running task and render a live n8n-style flow, logs, and the final result. */

const TERMINAL = new Set(["completed", "failed", "awaiting_approval", "waiting_for_user", "cancelled"]);
const state = {
  sessions: new Map(),        // sid -> session object
  order: [],                  // tab order of sids
  active: null,               // active sid
  cache: new Map(),           // taskId -> snapshot
  pollers: new Map(),         // taskId -> intervalId
  explainOpen: new Map(),     // taskId -> Map(data-xk -> bool) for Explain expand state
};

// Latest brain config (/config) + coding backend options, cached so the Explain
// header can show which models are doing the reasoning and the coding.
let _brainCfg = null;
const HOSTED_BRAIN_PROVIDERS = ["deepseek", "openai", "gemini", "glm", "opencode"];
const BRAIN_BALANCE_REFRESH_MS = 5 * 60 * 1000;
let _brainBalanceTimer = null;

const $ = (sel, root = document) => root.querySelector(sel);
const el = (html) => { const t = document.createElement("template"); t.innerHTML = html.trim(); return t.content.firstChild; };
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
// Escape then turn http(s) URLs into clickable links (Explain + Result prose).
// Also link local workspace media paths to the inline file viewer.
function linkify(s) {
  const safe = esc(s);
  const withHttp = safe.replace(
    /(https?:\/\/[^\s<>&"]+)/g,
    (url) => {
      let href = url;
      let tail = "";
      while (/[.,;:!?)}\]]$/.test(href)) {
        tail = href.slice(-1) + tail;
        href = href.slice(0, -1);
      }
      if (!href) return url;
      return `<a href="${href}" target="_blank" rel="noopener noreferrer" class="x-link">${href}</a>${tail}`;
    }
  );
  return withHttp.replace(
    /(?:\.\/)?(?:data\/)?(?:workspace\/)?(?:sessions|models|data)\/[\w./+-]+\.(?:mp4|webm|mov|mkv|m4v|avi|ogv|jpg|jpeg|png|gif|webp|bmp|svg)/gi,
    (path) => {
      // Skip if already inside an href attribute we just wrote
      return `<a href="${workspaceFileUrl(path)}" target="_blank" rel="noopener noreferrer" class="x-link media-local">${path}</a>`;
    }
  );
}

/* ── Markdown ──────────────────────────────────────────────────────────────
 * Agents answer in Markdown, so rendering it as literal '#'/'**' text reads as
 * broken output. This is a deliberately small subset — headings, emphasis, code,
 * lists, tables, rules, links/images — with no external dependency, keeping the
 * console dependency-free and offline-capable.
 *
 * Everything is escaped FIRST (via linkify, which also handles URLs and local
 * media paths), then inline markers are replaced on the already-safe string, so
 * model output can never inject HTML. */
function mdInline(s) {
  let t = linkify(s);
  // Code spans first — their contents must not be re-processed for emphasis.
  const code = [];
  t = t.replace(/`([^`]+)`/g, (_m, c) => `\u0000${code.push(c) - 1}\u0000`);
  // [label](url) — the url is already escaped; only allow http(s) and our own paths.
  t = t.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (m, label, href) => {
    if (/^(https?:|\/)/.test(href)) {
      return `<a href="${href}" target="_blank" rel="noopener noreferrer" class="x-link">${label}</a>`;
    }
    return m;
  });
  t = t.replace(/\*\*\*([^*]+)\*\*\*/g, "<strong><em>$1</em></strong>");
  t = t.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  t = t.replace(/(^|[\s(])\*([^*\n]+)\*(?=[\s.,;:!?)]|$)/g, "$1<em>$2</em>");
  t = t.replace(/(^|[\s(])_([^_\n]+)_(?=[\s.,;:!?)]|$)/g, "$1<em>$2</em>");
  t = t.replace(/~~([^~]+)~~/g, "<del>$1</del>");
  return t.replace(/\u0000(\d+)\u0000/g, (_m, i) => `<code>${code[+i]}</code>`);
}

/* Some providers (notably DeepSeek) sometimes emit a tool call as literal text
 * instead of a structured call, dumping provider markup into the answer. It is
 * noise to a reader, so strip it at display time — the real call is still visible
 * in Explain and Logs, which show raw output deliberately. */
function stripToolCallMarkup(s) {
  let t = String(s == null ? "" : s);
  // U+FF5C (fullwidth vertical bar) is what DeepSeek emits, not an ASCII pipe.
  if (!/DSML|｜|invoke name=/.test(t)) return t;
  // Whole sentinel tags: <｜｜DSML｜｜tool_calls>, <｜｜DSML｜｜invoke name="x">, closers.
  t = t.replace(/<\/?[｜|\s]*DSML[｜|][^>]*>/g, "");
  t = t.replace(/<\/?[｜|][^>]*>/g, "");
  t = t.replace(/<\/?(?:tool_calls|invoke|parameter)\b[^>]*>/gi, "");
  t = t.replace(/\bstring="true"\s*/gi, "");
  return t.replace(/\n{3,}/g, "\n\n").trim();
}

/* Block-level Markdown → HTML. Falls back to linkify() for plain text. */
function renderMarkdown(src) {
  const lines = stripToolCallMarkup(src).split("\n");
  const out = [];
  let i = 0, para = [];

  const flushPara = () => {
    if (para.length) { out.push(`<p>${mdInline(para.join("\n"))}</p>`); para = []; }
  };

  while (i < lines.length) {
    const ln = lines[i];

    // Fenced code — emitted verbatim (escaped), never inline-processed.
    const fence = ln.match(/^\s*```+\s*([\w+-]*)\s*$/);
    if (fence) {
      flushPara();
      const lang = fence[1] || "";
      const buf = [];
      i++;
      while (i < lines.length && !/^\s*```+\s*$/.test(lines[i])) buf.push(lines[i++]);
      i++;  // closing fence
      out.push(`<pre class="md-pre"><code${lang ? ` class="lang-${esc(lang)}"` : ""}>${esc(buf.join("\n"))}</code></pre>`);
      continue;
    }

    if (/^\s*(?:---+|\*\*\*+|___+)\s*$/.test(ln)) { flushPara(); out.push(`<hr class="md-hr">`); i++; continue; }

    const h = ln.match(/^(#{1,6})\s+(.*)$/);
    if (h) {
      flushPara();
      const lvl = Math.min(h[1].length + 2, 6);   // '#' → h3, so it sits under the page's own h1/h2
      out.push(`<h${lvl} class="md-h md-h${h[1].length}">${mdInline(h[2].trim())}</h${lvl}>`);
      i++; continue;
    }

    if (/^\s*>\s?/.test(ln)) {
      flushPara();
      const buf = [];
      while (i < lines.length && /^\s*>\s?/.test(lines[i])) buf.push(lines[i++].replace(/^\s*>\s?/, ""));
      out.push(`<blockquote class="md-quote">${mdInline(buf.join("\n"))}</blockquote>`);
      continue;
    }

    // Table: a header row followed by a |---|---| separator.
    if (/^\s*\|.*\|\s*$/.test(ln) && i + 1 < lines.length && /^\s*\|[\s:|-]+\|\s*$/.test(lines[i + 1])) {
      flushPara();
      const cells = (row) => row.trim().replace(/^\||\|$/g, "").split("|").map(c => c.trim());
      const head = cells(lines[i]);
      i += 2;
      const body = [];
      while (i < lines.length && /^\s*\|.*\|\s*$/.test(lines[i])) body.push(cells(lines[i++]));
      out.push(`<div class="md-table-wrap"><table class="md-table"><thead><tr>` +
        head.map(c => `<th>${mdInline(c)}</th>`).join("") + `</tr></thead><tbody>` +
        body.map(r => `<tr>${r.map(c => `<td>${mdInline(c)}</td>`).join("")}</tr>`).join("") +
        `</tbody></table></div>`);
      continue;
    }

    const li = ln.match(/^(\s*)([-*+]|\d+[.)])\s+(.*)$/);
    if (li) {
      flushPara();
      const ordered = /\d/.test(li[2]);
      const items = [];
      while (i < lines.length) {
        const m = lines[i].match(/^(\s*)([-*+]|\d+[.)])\s+(.*)$/);
        if (!m || /\d/.test(m[2]) !== ordered) break;
        const buf = [m[3]];
        i++;
        // Continuation lines belonging to this bullet.
        while (i < lines.length && /^\s+\S/.test(lines[i]) &&
               !/^(\s*)([-*+]|\d+[.)])\s+/.test(lines[i])) buf.push(lines[i++].trim());
        items.push(buf.join(" "));
      }
      const tag = ordered ? "ol" : "ul";
      out.push(`<${tag} class="md-list">` + items.map(t => `<li>${mdInline(t)}</li>`).join("") + `</${tag}>`);
      continue;
    }

    if (!ln.trim()) { flushPara(); i++; continue; }
    para.push(ln);
    i++;
  }
  flushPara();
  return out.join("");
}

const _MEDIA_EXT_RE = /\.(mp4|webm|mov|mkv|m4v|avi|ogv|jpg|jpeg|png|gif|webp|bmp|svg)$/i;
const _LOCAL_PATH_RE = /(?:(?:\.\/)?(?:data\/)?(?:workspace\/)?(?:sessions|models|data)\/[\w./+-]+|\/[\w./+-]*workspace\/[\w./+-]+)(?:\.\w+)?/gi;
/* Charts and figures are written straight to the workspace root (e.g.
   ``revenue_by_region.png``), so a bare image filename must be picked up too —
   the path-prefixed regex above would miss them entirely. */
const _BARE_IMAGE_RE = /(?:^|[\s"'`(<[])((?:\.\/)?(?:[\w+-]+\/)*[\w+-]+\.(?:png|jpg|jpeg|gif|webp|bmp|svg))(?=$|[\s"'`)>\].,;:!?])/gim;

function workspaceFileUrl(path) {
  return `/ui/api/workspace/file?path=${encodeURIComponent(String(path || "").replace(/\\/g, "/"))}`;
}

/** Collect local workspace paths from result prose, audits, and tool events. */
function collectLocalMediaPaths(task) {
  const out = [];
  const seen = new Set();
  const add = (p) => {
    const s = String(p || "").trim().replace(/^["'`]+|["'`]+$/g, "");
    if (!s || /^(NOT DOWNLOADED|N\/A|NONE)$/i.test(s)) return;
    if (seen.has(s)) return;
    seen.add(s);
    out.push(s);
  };

  const r = task?.result || {};
  for (const st of r.steps || []) {
    const resources = st.resource_audit?.resources || [];
    for (const res of resources) add(res.local_path);
  }

  const textBlobs = [r.answer || ""];
  for (const st of r.steps || []) textBlobs.push(st.output || "");
  for (const e of task.events || []) {
    if (e.event === "tool.result" && e.tool === "download_resource") {
      textBlobs.push(typeof e.output === "string" ? e.output : JSON.stringify(e.output || ""));
      try {
        const obj = typeof e.output === "string" ? JSON.parse(e.output) : e.output;
        if (obj && typeof obj === "object") add(obj.path || obj.local_path);
      } catch { /* not JSON */ }
    }
  }
  for (const blob of textBlobs) {
    const localLine = /(?:\*\*)?Local path(?:\*\*)?\s*:\s*(.+)$/gim;
    let m;
    while ((m = localLine.exec(blob))) add(m[1].trim());
    const pathHits = blob.match(_LOCAL_PATH_RE) || [];
    for (const p of pathHits) {
      if (_MEDIA_EXT_RE.test(p) || /\/(sessions|models|data)\//.test(p)) add(p);
    }
    // Bare image filenames written to the workspace root (charts, figures).
    let bm;
    _BARE_IMAGE_RE.lastIndex = 0;
    while ((bm = _BARE_IMAGE_RE.exec(blob))) {
      const p = bm[1];
      if (!/^https?:/i.test(p)) add(p.replace(/^\.\//, ""));
    }
  }
  return out;
}

function mediaCard(item, kind) {
  const src = esc(item.url || workspaceFileUrl(item.path));
  const name = esc(item.name || item.path || "file");
  const open = `<a class="media-open" href="${src}" target="_blank" rel="noopener noreferrer">Open</a>`;
  if (kind === "video") {
    return `<div class="media-card">
      <video controls preload="metadata" src="${src}"></video>
      <div class="media-cap"><span class="media-name" title="${name}">${name}</span>${open}</div>
    </div>`;
  }
  return `<div class="media-card">
    <a href="${src}" target="_blank" rel="noopener noreferrer"><img src="${src}" alt="${name}" loading="lazy" /></a>
    <div class="media-cap"><span class="media-name" title="${name}">${name}</span>${open}</div>
  </div>`;
}

function mediaSectionHtml(title, items, kind) {
  if (!items || !items.length) return "";
  return `<div class="steps-list media-section"><h3>${title} <span class="x-muted">(${items.length})</span></h3>
    <div class="media-grid">${items.map(it => mediaCard(it, kind)).join("")}</div></div>`;
}

/* Unix seconds for when a run started, with a grace window for clock skew between
 * the task record and the filesystem. Used to scope workspace listings to one run. */
function runStartedSince(task) {
  const started = Date.parse(task?.created_at || "");
  return Number.isNaN(started) ? null : (started - 60_000) / 1000;
}

const _KIND_ICON = {
  report: "📄", spreadsheet: "📊", page: "🌐",
  data: "🗂", notebook: "📓", code: "⌨", archive: "🗜",
};

function fmtBytes(n) {
  if (!n && n !== 0) return "";
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`;
  if (n < 1024 * 1024 * 1024) return `${(n / 1048576).toFixed(1)} MB`;
  return `${(n / 1073741824).toFixed(1)} GB`;
}

/* Files the run produced that aren't images — spreadsheets, reports, dashboards.
 * Previously these were written to the workspace and never surfaced anywhere. */
function deliverablesHtml(data) {
  const files = (data && data.files) || [];
  if (!files.length) return "";
  const note = data.truncated
    ? `<span class="x-muted">(showing ${data.shown} of ${data.total})</span>`
    : `<span class="x-muted">(${files.length})</span>`;
  const rows = files.map(f => {
    const href = esc(f.url || workspaceFileUrl(f.path));
    const dir = String(f.path || "").includes("/")
      ? `<span class="dl-dir">${esc(f.path.slice(0, f.path.lastIndexOf("/")))}/</span>`
      : "";
    return `<a class="dl-row" href="${href}" target="_blank" rel="noopener noreferrer"
        title="${esc(f.path)}">
      <span class="dl-icon">${_KIND_ICON[f.kind] || "📎"}</span>
      <span class="dl-name">${dir}${esc(f.name)}</span>
      <span class="dl-kind">${esc(f.kind || "")}</span>
      <span class="dl-size">${fmtBytes(f.size)}</span>
    </a>`;
  }).join("");
  return `<div class="steps-list deliverables"><h3>Files produced ${note}</h3>
    <div class="dl-list">${rows}</div></div>`;
}

async function loadDeliverables(taskId) {
  const mount = document.getElementById(`deliverables-${taskId}`);
  if (!mount) return;
  const task = state.cache.get(taskId);
  if (!task || task.status !== "completed") { mount.innerHTML = ""; return; }
  try {
    const since = runStartedSince(task);
    const data = await api("GET",
      `/ui/api/workspace/deliverables${since != null ? `?since=${since}` : ""}`);
    mount.innerHTML = deliverablesHtml(data);
  } catch (err) {
    mount.innerHTML = `<div class="x-muted">Could not list produced files: ${esc(err.message || err)}</div>`;
  }
}

async function loadMediaGallery(taskId) {
  const mount = document.getElementById(`media-gallery-${taskId}`);
  if (!mount) return;
  const task = state.cache.get(taskId);
  if (!task) { mount.innerHTML = ""; return; }
  const paths = collectLocalMediaPaths(task);
  // Charts are generated freely but only sometimes named in the answer, so once a
  // run is finished ask for every workspace image rather than just the named ones.
  const wantAll = task.status === "completed";
  if (!paths.length && !wantAll) { mount.innerHTML = ""; return; }
  try {
    const parts = paths.map(p => `paths=${encodeURIComponent(p)}`);
    if (wantAll) {
      parts.push("all_images=true");
      // Scope to files this run wrote — the workspace persists across runs, so an
      // unscoped listing shows every chart every earlier task produced.
      const since = runStartedSince(task);
      if (since != null) parts.push(`since=${since}`);
    }
    const data = await api("GET", `/ui/api/workspace/media?${parts.join("&")}`);
    if (!data.total) { mount.innerHTML = ""; return; }
    const note = data.sampled
      ? `<div class="media-sample-note x-muted">Showing ${data.shown} of ${data.total} media files (sample)</div>`
      : "";
    mount.innerHTML = note
      + mediaSectionHtml("Video", data.videos, "video")
      + mediaSectionHtml("Image", data.images, "image");
  } catch (err) {
    mount.innerHTML = `<div class="x-muted">Could not load local media previews: ${esc(err.message || err)}</div>`;
  }
}

async function api(method, path, body) {
  const res = await fetch(path, {
    method,
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  return res.json();
}

/* ── Saved-task sidebar ────────────────────────────────────────────────── */
async function loadSessionList() {
  let data;
  try { data = await api("GET", "/ui/api/sessions"); }
  catch { return; }
  const list = $("#session-list");
  list.innerHTML = "";
  const sessions = (data.sessions || []).slice().reverse();  // newest first
  if (!sessions.length) {
    list.innerHTML = `<div class="sidebar-empty">No saved tasks yet.<br>Run one above.</div>`;
    return;
  }
  for (const s of sessions) {
    const last = s.runs[s.runs.length - 1] || {};
    const status = s.runs.some(r => r.status === "running" || r.status === "pending")
      ? "running" : (last.status || "completed");
    const item = el(`<div class="sess-item ${state.sessions.has(s.id) ? "open" : ""}">
      <span class="dot ${status}"></span>
      <div class="meta"><div class="t">${esc(s.title)}</div>
        <div class="sub">${s.runs.length} run${s.runs.length === 1 ? "" : "s"} · ${fmtDate(s.created_at)}</div></div>
      <span class="clr-mem" title="Clear this task's memory">🧠</span>
      <span class="del" title="Delete task">🗑</span></div>`);
    item.addEventListener("click", (e) => {
      if (e.target.classList.contains("del")) { deleteSession(s.id); return; }
      if (e.target.classList.contains("clr-mem")) { clearSessionMemory(s.id, s.title); return; }
      openSession(s.id);
    });
    list.appendChild(item);
  }
}

function fmtDate(iso) {
  try { const d = new Date(iso); return d.toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }); }
  catch { return ""; }
}

async function deleteFollowup(sid, taskId, index) {
  if (index === 0) return;
  const s = state.sessions.get(sid);
  const run = s?.runs.find(r => r.id === taskId);
  if (!run) return;
  const preview = run.goal.length > 80 ? run.goal.slice(0, 80) + "…" : run.goal;
  if (!confirm(`Delete this follow-up?\n\n"${preview}"\n\nThe original task and other runs are kept.`)) return;
  try { await api("DELETE", `/ui/api/sessions/${sid}/runs/${taskId}`); }
  catch (e) { alert("Delete failed: " + e.message); return; }
  clearInterval(state.pollers.get(taskId));
  state.pollers.delete(taskId);
  state.cache.delete(taskId);
  s.runs = s.runs.filter(r => r.id !== taskId);
  if (s.activeRun === taskId) s.activeRun = s.runs[s.runs.length - 1]?.id || null;
  renderSession(s);
  renderTabs();
  loadSessionList();
}

async function deleteSession(sid) {
  if (!confirm("Delete this saved task and its history?\n\nThis also removes recall memory for this task.")) return;
  try { await api("DELETE", `/ui/api/sessions/${sid}`); } catch {}
  if (state.sessions.has(sid)) closeSession(sid);
  loadSessionList();
}

async function clearSessionMemory(sid, title) {
  const label = title ? `"${title.slice(0, 48)}"` : "this task";
  if (!confirm(`Clear recall memory for ${label}?\n\nThe saved run history stays — only what the AI remembers for follow-ups is removed.`)) return;
  try {
    const res = await api("DELETE", `/ui/api/sessions/${sid}/memory`);
    const n = res.records_removed;
    alert(n != null ? `Cleared ${n} memory record${n === 1 ? "" : "s"} for this task.` : "Task memory cleared.");
  } catch (e) {
    alert("Failed to clear task memory: " + e.message);
  }
}

async function deleteAllSessions() {
  if (!confirm(
    "Delete ALL saved tasks?\n\n" +
    "This removes every task from the sidebar and deletes their run history. " +
    "Each task's recall memory is cleared too.\n\n" +
    "Global memory (🧹) and other tasks' unrelated recall are not wiped unless you use Clear all memories."
  )) return;
  const btn = $("#delete-all-sessions");
  btn.disabled = true;
  try {
    const res = await api("DELETE", "/ui/api/sessions");
    for (const sid of [...state.order]) closeSession(sid);
    alert(`Deleted ${res.deleted} saved task${res.deleted === 1 ? "" : "s"}.`);
    loadSessionList();
  } catch (e) {
    alert("Failed to delete tasks: " + e.message);
  }
  btn.disabled = false;
}

/* Open (or resume) a saved session as a tab. */
async function openSession(sid) {
  let detail;
  try { detail = await api("GET", `/ui/api/sessions/${sid}`); }
  catch { return; }
  detail.runs.forEach(r => state.cache.set(r.id, r));
  if (state.sessions.has(sid)) {
    setActive(sid);
    return;
  }
  const runs = (detail.runs || []).map(r => ({ id: r.id, goal: r.goal, status: r.status }));
  const session = {
    id: sid, title: detail.title, runs,
    activeRun: runs.length ? runs[runs.length - 1].id : null, view: "flow", explainOrder: "grouped",
  };
  state.sessions.set(sid, session);
  state.order.push(sid);
  buildSection(session);
  setActive(sid);
  // Resume polling for any run still in flight (e.g. opened in another tab).
  runs.filter(r => !TERMINAL.has(r.status)).forEach(r => poll(r.id, sid));
  renderTabs();
  loadSessionList();
}

/* ── Task creation ─────────────────────────────────────────────────────── */
const AUTO_APPROVE_KEY = "aiemployee.autoApprove";

function isAutoApprove() {
  const el = $("#auto-approve");
  if (el) return !!el.checked;
  try { return localStorage.getItem(AUTO_APPROVE_KEY) === "1"; } catch { return false; }
}

function syncAutoApproveUi() {
  const el = $("#auto-approve");
  if (!el) return;
  try { el.checked = localStorage.getItem(AUTO_APPROVE_KEY) === "1"; } catch { /* ignore */ }
  el.title = el.checked
    ? "Auto Approve ON — gated ops run without asking"
    : "Auto Approve OFF — pause for shell/download/etc. approvals";
}

function setAutoApprove(on) {
  const el = $("#auto-approve");
  if (el) el.checked = !!on;
  try { localStorage.setItem(AUTO_APPROVE_KEY, on ? "1" : "0"); } catch { /* ignore */ }
  syncAutoApproveUi();
  if (on) autoApprovePausedTasks();
}

/** If Auto Approve is toggled on mid-pause, resume every waiting run. */
function autoApprovePausedTasks() {
  if (!isAutoApprove()) return;
  for (const s of state.sessions.values()) {
    for (const run of s.runs || []) {
      const snap = state.cache.get(run.id);
      const status = snap?.status || run.status;
      if (status === "awaiting_approval") {
        approveTask(s.id, run.id, { approve_all_similar: true }).catch(() => {});
      } else if (status === "waiting_for_user") {
        approveTask(s.id, run.id, {
          decision: "proceed_without",
          approve_all_similar: true,
        }).catch(() => {});
      }
    }
  }
}

async function newTask(goal, mode) {
  const body = {
    goal,
    mode: mode || "task",
    auto_approve: isAutoApprove(),
  };
  const { task_id, session_id } = await api("POST", "/ui/api/tasks", body);
  const session = {
    id: session_id, title: goal.slice(0, 42), runs: [{ id: task_id, goal, status: "pending", mode: mode || "task" }],
    activeRun: task_id, view: "flow", explainOrder: "grouped",
  };
  state.sessions.set(session_id, session);
  state.order.push(session_id);
  buildSection(session);
  setActive(session_id);
  poll(task_id, session_id);
  renderTabs();
  loadSessionList();
}

async function followUp(sid, goal, approvedOps, mode) {
  const body = {
    goal,
    approved_ops: approvedOps || [],
    mode: mode || "task",
    auto_approve: isAutoApprove(),
  };
  const { task_id } = await api("POST", `/ui/api/sessions/${sid}/followup`, body);
  const s = state.sessions.get(sid);
  s.runs.push({ id: task_id, goal, status: "pending", mode: mode || "task" });
  s.activeRun = task_id;
  poll(task_id, sid);
  renderSession(s);
  renderTabs();
}

/* Approve a paused run and resume it IN PLACE — same run, same tab (no new run
 * is created). The paused task flips back to running and the poller re-attaches. */
function resourceApprovalBanner(task, session, na, waiting = false) {
  const urls = (na.candidate_urls || na.args?.candidate_urls || []).slice(0, 8);
  const hubs = (na.hubs_searched || na.args?.hubs_searched || []).join(", ");
  const missing = (na.missing_credentials || na.args?.missing_credentials || []).join(", ");
  const desc = na.resource_description || na.args?.resource_description || na.message || "Required resource";
  const reason = na.download_failure_reason || na.args?.download_failure_reason || "";
  const preset = na.message || na.args?.message || "";
  const urlList = urls.length
    ? `<ul class="x-list">${urls.map(u => `<li><a href="${esc(u)}" target="_blank" class="x-link">${esc(u)}</a></li>`).join("")}</ul>`
    : `<p class="x-muted">No candidate URLs — searched: ${esc(hubs || "all hubs")}</p>`;
  const id = `res-appr-${task.id}`;
  const head = waiting
    ? `<div><b>Status:</b> <span class="wait-tag">WAITING_FOR_USER</span></div>
       <div class="wait-msg">${linkify(preset || `I found a suitable resource: ${desc}`)}</div>
       ${reason ? `<div class="x-muted" style="margin-top:8px">Reason: ${linkify(reason)}</div>` : ""}`
    : `<div><b>Resource needed:</b> ${linkify(desc)}</div>
       ${missing ? `<div class="x-muted">Missing API keys: ${esc(missing)} — add in 🔑 Keys</div>` : ""}`;
  const explain = `<div class="appr-body result-appr-explain">${approvalExplainBody(na, task, { linkify: true })}</div>`;
  return `<div class="approval-banner resource-banner ${waiting ? "waiting-banner" : ""}" id="${id}">
    <div class="result-appr-head">⏸ Approval required — <b>${esc(na.operation || "resource")}</b></div>
    ${explain}
    ${head}
    <div class="x-label" style="margin-top:8px">${waiting ? "Download URL" : "Candidate links"}</div>${urlList}
    <div class="resource-appr-form">
      <label>Decision
        <select class="res-decision">
          <option value="provide_local_path">I downloaded it — use local path</option>
          <option value="provide_url">Try this URL instead</option>
          <option value="proceed_without">Proceed without this resource</option>
          <option value="abort">Stop — resource is required</option>
        </select>
      </label>
      <label>Local path (workspace) <input class="res-path" placeholder="./data/my_file.mp4" /></label>
      <label>URL <input class="res-url" placeholder="https://…" /></label>
      <label>Note <input class="res-note" placeholder="optional" /></label>
      ${approvalAllowAllHtml(task.id, na)}
      <button class="btn primary res-approve-btn">${waiting ? "Continue with path" : "Continue with choice"}</button>
    </div>
  </div>`;
}

function wireResourceApproval(task, session) {
  const wrap = document.getElementById(`res-appr-${task.id}`);
  if (!wrap) return;
  const btn = $(".res-approve-btn", wrap);
  if (!btn || btn._wired) return;
  btn._wired = true;
  btn.addEventListener("click", () => {
    const decision = $(".res-decision", wrap).value;
    const body = {
      decision,
      local_path: $(".res-path", wrap).value.trim() || undefined,
      resource_url: $(".res-url", wrap).value.trim() || undefined,
      note: $(".res-note", wrap).value.trim() || undefined,
    };
    approveTask(session.id, task.id, body);
  });
}

async function approveTask(sid, taskId, body) {
  const cb = document.getElementById(`appr-allow-all-${taskId}`);
  const payload = { ...(body || {}) };
  payload.approve_all_similar = cb ? cb.checked : true;
  try { await api("POST", `/ui/api/tasks/${taskId}/approve`, payload); }
  catch (e) { alert("Approve failed: " + e.message); return; }
  const s = state.sessions.get(sid);
  if (s) {
    const run = s.runs.find(r => r.id === taskId);
    if (run) {
      run.status = "running";
      if (run.result) run.result = { ...run.result, needs_approval: null };
    }
    s.activeRun = taskId;
    renderSession(s);
  }
  const cached = state.cache.get(taskId);
  if (cached) {
    state.cache.set(taskId, {
      ...cached,
      status: "running",
      result: cached.result ? { ...cached.result, needs_approval: null } : cached.result,
    });
  }
  poll(taskId, sid);                   // resume polling the same task id
  renderTabs();
}

async function resumeTask(sid, taskId) {
  try { await api("POST", `/ui/api/tasks/${taskId}/resume`); }
  catch (e) { alert("Resume failed: " + e.message); return; }
  const s = state.sessions.get(sid);
  if (s) {
    const run = s.runs.find(r => r.id === taskId);
    if (run) run.status = "running";
    s.activeRun = taskId;
    renderSession(s);
  }
  const cached = state.cache.get(taskId);
  if (cached) {
    state.cache.set(taskId, {
      ...cached,
      status: "running",
      resumable: false,
      error: null,
    });
  }
  poll(taskId, sid);
  renderTabs();
}

/* Stop a run in flight. The poller picks up the resulting "cancelled" status. */
async function cancelTask(taskId) {
  try { await api("POST", `/ui/api/tasks/${taskId}/cancel`); } catch {}
}

/* ── Polling ───────────────────────────────────────────────────────────── */
function poll(taskId, sid) {
  const tick = async () => {
    let snap;
    try { snap = await api("GET", `/ui/api/tasks/${taskId}`); }
    catch { return; }
    state.cache.set(taskId, snap);
    const s = state.sessions.get(sid);
    if (s) {
      const run = s.runs.find(r => r.id === taskId);
      if (run) run.status = snap.status;
      if (run && snap.result !== undefined) {
        run.result = snap.result;
        run.status = snap.status;
      }
      // Adopt the auto-generated session title once it's ready.
      if (snap.session_title && s.title !== snap.session_title) {
        s.title = snap.session_title;
        loadSessionList();
      }
    }
    if (state.active === sid && s && s.activeRun === taskId) renderActiveView(s);
    renderTabs();
    // Safety net: if Auto Approve is on and a pause still surfaces, approve immediately.
    if (isAutoApprove() && !snap._autoApproving) {
      if (snap.status === "awaiting_approval") {
        snap._autoApproving = true;
        state.cache.set(taskId, snap);
        approveTask(sid, taskId, { approve_all_similar: true }).catch(() => {});
      } else if (snap.status === "waiting_for_user") {
        snap._autoApproving = true;
        state.cache.set(taskId, snap);
        approveTask(sid, taskId, {
          decision: "proceed_without",
          approve_all_similar: true,
        }).catch(() => {});
      }
    }
    if (TERMINAL.has(snap.status)) {
      clearInterval(state.pollers.get(taskId));
      state.pollers.delete(taskId);
      loadSessionList();  // reflect final status in the saved-tasks panel
    }
  };
  tick();
  state.pollers.set(taskId, setInterval(tick, 1000));
}

/* ── Tabs ──────────────────────────────────────────────────────────────── */
function sessionStatus(s) {
  if (s.runs.some(r => r.status === "running" || r.status === "pending")) return "running";
  return s.runs[s.runs.length - 1]?.status || "pending";
}

function renderTabs() {
  const tabs = $("#tabs");
  tabs.innerHTML = "";
  for (const sid of state.order) {
    const s = state.sessions.get(sid);
    const tab = el(`<div class="tab ${sid === state.active ? "active" : ""}">
      <span class="dot ${sessionStatus(s)}"></span>
      <span class="title">${esc(s.title)}</span>
      <span class="close" title="Close">×</span></div>`);
    tab.addEventListener("click", (e) => {
      if (e.target.classList.contains("close")) { closeSession(sid); return; }
      setActive(sid);
    });
    tabs.appendChild(tab);
  }
}

function closeSession(sid) {
  const s = state.sessions.get(sid);
  s.runs.forEach(r => { clearInterval(state.pollers.get(r.id)); state.pollers.delete(r.id); });
  if (s._el) s._el.remove();
  state.sessions.delete(sid);
  state.order = state.order.filter(x => x !== sid);
  if (state.active === sid) state.active = state.order[state.order.length - 1] || null;
  if (state.active) setActive(state.active); else showEmpty();
  renderTabs();
}

function showEmpty() {
  $("#workspace").querySelectorAll(".session").forEach(n => n.style.display = "none");
  let empty = $("#empty-state");
  if (empty) empty.style.display = "flex";
}

function setActive(sid) {
  state.active = sid;
  const empty = $("#empty-state"); if (empty) empty.style.display = "none";
  $("#workspace").querySelectorAll(".session").forEach(n => n.style.display = "none");
  const s = state.sessions.get(sid);
  s._el.style.display = "flex";
  renderSession(s);
  renderTabs();
}

/* ── Section (per session) ─────────────────────────────────────────────── */
function buildSection(session) {
  const node = $("#session-template").content.firstElementChild.cloneNode(true);
  session._el = node;
  $("#workspace").appendChild(node);
  wireExplainView($(".explain-view", node), session);

  node.querySelectorAll(".subtab").forEach(btn => {
    btn.addEventListener("click", () => {
      session.view = btn.dataset.view;
      renderSession(session);
    });
  });
  wireFlowJumps($(".flow-view", node), session);
  $(".followup", node).addEventListener("submit", (e) => {
    e.preventDefault();
    const input = $(".followup-input", node);
    const goal = input.value.trim();
    if (!goal) return;
    input.value = "";
    const active = state.cache.get(session.activeRun);
    const keepMode = active?.mode === "download" || active?.mode === "plan" || active?.mode === "research"
      ? active.mode
      : "task";
    followUp(session.id, goal, undefined, keepMode);
  });
  $(".sess-clear-memory", node).addEventListener("click", () => clearSessionMemory(session.id, session.title));
  $(".sess-delete", node).addEventListener("click", () => deleteSession(session.id));
}

/* Clicking a flow node opens the Explain tab at that node's card. Delegated from
 * the container so it keeps working across the poller's re-renders of .flow-view. */
function wireFlowJumps(flow, session) {
  if (!flow || flow._jumpWired) return;
  flow._jumpWired = true;
  const go = (e) => {
    const node = e.target.closest(".node[data-jump]");
    if (!node || !flow.contains(node)) return;
    jumpToExplain(session, node.getAttribute("data-jump"));
  };
  flow.addEventListener("click", go);
  flow.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); go(e); }
  });
}

/* Switch to Explain, expand the matching component, scroll it into view and
 * flash it so it's obvious which card the click landed on. */
function jumpToExplain(session, key) {
  if (!key) return;
  session.view = "explain";
  // Force this card open even if the user had collapsed it previously.
  let open = state.explainOpen.get(session.activeRun);
  if (!open) { open = new Map(); state.explainOpen.set(session.activeRun, open); }
  open.set(key, true);
  renderSession(session);
  // Wait for the re-render, then locate and highlight the target.
  requestAnimationFrame(() => {
    const explain = $(".explain-view", session._el);
    if (!explain) return;
    const card = explain.querySelector(`details[data-xk="${cssEscape(key)}"]`);
    if (!card) return;
    card.open = true;
    card.scrollIntoView({ behavior: "smooth", block: "start" });
    card.classList.remove("x-flash");
    void card.offsetWidth;  // restart the animation if the same card is re-picked
    card.classList.add("x-flash");
    setTimeout(() => card.classList.remove("x-flash"), 1600);
  });
}

/* Attribute selectors need escaping — our keys contain ':' (e.g. comp:step:1). */
function cssEscape(s) {
  return window.CSS && CSS.escape ? CSS.escape(s) : String(s).replace(/["\\]/g, "\\$&");
}

function captureExplainOpen(explain, taskId) {
  if (!taskId || !explain.querySelector("details[data-xk]")) return;
  let m = state.explainOpen.get(taskId);
  if (!m) { m = new Map(); state.explainOpen.set(taskId, m); }
  explain.querySelectorAll("details[data-xk]").forEach(d => m.set(d.getAttribute("data-xk"), d.open));
}

function restoreExplainOpen(explain, taskId) {
  const prevOpen = state.explainOpen.get(taskId);
  if (!prevOpen) return;
  explain.querySelectorAll("details[data-xk]").forEach(d => {
    const k = d.getAttribute("data-xk");
    if (prevOpen.has(k)) d.open = prevOpen.get(k);
  });
}

function wireExplainView(explain, session) {
  if (explain._wired) return;
  explain._wired = true;
  explain.addEventListener("toggle", (e) => {
    const det = e.target;
    if (!det.matches("details[data-xk]")) return;
    const taskId = session.activeRun;
    if (!taskId) return;
    let m = state.explainOpen.get(taskId);
    if (!m) { m = new Map(); state.explainOpen.set(taskId, m); }
    m.set(det.getAttribute("data-xk"), det.open);
    // Expanding an LLM call opens both INP and OUT; user can collapse either after.
    if (det.classList.contains("x-llm-call") && det.open) {
      det.querySelectorAll(".x-llm-section[data-xk]").forEach(s => {
        s.open = true;
        m.set(s.getAttribute("data-xk"), true);
      });
    }
  }, true);
}

function renderSession(s) {
  // runs strip
  const strip = $(".runs-strip", s._el);
  strip.innerHTML = "";
  s.runs.forEach((r, i) => {
    const label = i === 0 ? "Task" : "Follow-up " + i;
    const chip = el(`<div class="run-chip ${r.id === s.activeRun ? "active" : ""}">
      <span class="dot ${r.status}"></span>
      <span class="run-chip-label">${label}: ${esc(r.goal.slice(0, 30))}</span>
      ${i > 0 ? `<button type="button" class="run-chip-del" title="Delete this follow-up">×</button>` : ""}
    </div>`);
    chip.addEventListener("click", (e) => {
      if (e.target.closest(".run-chip-del")) return;
      s.activeRun = r.id;
      renderSession(s);
    });
    const del = $(".run-chip-del", chip);
    if (del) del.addEventListener("click", (e) => {
      e.stopPropagation();
      deleteFollowup(s.id, r.id, i);
    });
    strip.appendChild(chip);
  });
  // Stop button for the active run while it's still in flight.
  const active = s.runs.find(r => r.id === s.activeRun);
  const activeTask = active ? state.cache.get(active.id) : null;
  if (active && !TERMINAL.has(active.status)) {
    const stop = el(`<button class="btn stop-btn" title="Stop this run">■ Stop</button>`);
    stop.addEventListener("click", () => {
      stop.disabled = true;
      stop.textContent = "Stopping…";
      cancelTask(active.id);
    });
    strip.appendChild(stop);
  } else if (active && activeTask && activeTask.resumable) {
    const resume = el(`<button class="btn primary resume-btn" title="Resume this interrupted run">▶ Resume</button>`);
    resume.addEventListener("click", () => {
      resume.disabled = true;
      resume.textContent = "Resuming…";
      resumeTask(s.id, active.id);
    });
    strip.appendChild(resume);
  }
  // subtab active state
  s._el.querySelectorAll(".subtab").forEach(b => b.classList.toggle("active", b.dataset.view === s.view));
  renderActiveView(s);
}

function wireExplainControls(explain, session) {
  explain.querySelectorAll("[data-explain-order]").forEach(btn => {
    btn.addEventListener("click", () => {
      session.explainOrder = btn.dataset.explainOrder || "grouped";
      renderSession(session);
    });
  });
}

function renderActiveView(s) {
  const task = state.cache.get(s.activeRun);
  const flow = $(".flow-view", s._el), explain = $(".explain-view", s._el),
        logs = $(".logs-view", s._el), result = $(".result-view", s._el);
  flow.classList.toggle("active", s.view === "flow");
  explain.classList.toggle("active", s.view === "explain");
  logs.classList.toggle("active", s.view === "logs");
  result.classList.toggle("active", s.view === "result");
  const lc = $(".log-count", s._el);
  if (lc) lc.textContent = task ? `(${(task.events || []).length})` : "";

  if (!task) { flow.innerHTML = explain.innerHTML = logs.innerHTML = result.innerHTML = `<div class="placeholder">Waiting for the employee to start…</div>`; return; }
  if (s.view === "flow") flow.innerHTML = renderFlow(task, s);
  else if (s.view === "explain") {
    const taskId = s.activeRun;
    captureExplainOpen(explain, taskId);
    explain.innerHTML = problemHeader(task) + renderExplain(task, s);
    wireExplainControls(explain, s);
    restoreExplainOpen(explain, taskId);
  }
  else if (s.view === "logs") { logs.innerHTML = renderLogs(task); logs.scrollTop = logs.scrollHeight; }
  else result.innerHTML = problemHeader(task) + renderResult(task, s);
}

/* The base problem (the goal the user gave) — pinned to the top of Result & Explain. */
function problemHeader(task) {
  const goal = (task && task.goal) || "";
  if (!goal) return "";
  const tag = task.mode === "research"
    ? `<span class="pb-tag">🔬 Research</span>`
    : task.mode === "plan"
    ? `<span class="pb-tag">📋 Plan</span>`
    : (task.mode === "download" ? `<span class="pb-tag">⬇ Download</span>` : "");
  return `<div class="problem-banner"><div class="pb-head"><span class="pb-label">Problem</span>${tag}${modelBadges()}</div>
    <div class="pb-goal">${linkify(goal)}</div></div>`;
}

/* Which models are doing the work — brain (reasoning) + coding backend — shown
 * beside the task so it's clear what produced the output. Sourced from /config
 * and /ui/api/coding/options (cached by the top-bar badges). */
function modelBadges() {
  const cfg = _brainCfg;
  if (!cfg) return "";
  const brain = `${cfg.provider}${cfg.model ? " · " + cfg.model : ""}`;
  // Coding step: "brain" backend reuses the brain model; a CLI backend runs its own.
  const cb = (_codingOpts && _codingOpts.current) || "brain";
  const code = cb === "brain"
    ? `brain · ${cfg.model || cfg.provider}`
    : `${cb} (CLI)`;
  return `<span class="pb-models">` +
    `<span class="pb-model" title="Reasoning brain (LLM provider · model)">🧠 ${esc(brain)}</span>` +
    `<span class="pb-model" title="Coding backend that writes &amp; runs code">⌨ ${esc(code)}</span>` +
    `</span>`;
}

/* ── Flow (derive an n8n-style graph from the event stream) ────────────── */
function deriveFlow(task) {
  let plan = null, synth = false, reflect = false, approval = null;
  const steps = new Map();
  const ensure = (n, agent) => {
    if (!steps.has(n)) steps.set(n, { step: n, agent: agent || "?", tools: new Map(), verified: null, confidence: null, retries: 0 });
    const s = steps.get(n); if (agent) s.agent = agent; return s;
  };
  for (const e of (task.events || [])) {
    const f = e.fields || {};
    switch (e.event) {
      case "planner.plan_ready":
      case "workflow.planned": plan = { steps: f.steps ?? plan?.steps ?? 0 }; break;
      case "agent.start": if (f.node === "execute" && f.step != null) ensure(f.step, f.agent); break;
      case "tool.execute": if (f.step != null && f.tool) { const s = ensure(f.step); if (!s.tools.has(f.tool)) s.tools.set(f.tool, { name: f.tool, status: "running" }); } break;
      case "tool.result": if (f.step != null && f.tool) { const s = ensure(f.step); s.tools.set(f.tool, { name: f.tool, status: f.ok ? "ok" : "fail", latency: f.latency_ms }); } break;
      case "tool.approval_required": if (f.step != null && f.tool) { ensure(f.step).tools.set(f.tool, { name: f.tool, status: "approval" }); } approval = { ...(approval || {}), tool: f.tool, args: f.args, key: f.key, step: f.step }; break;
      case "verifier.verdict": if (f.step != null) { const s = ensure(f.step); s.verified = f.passed; s.confidence = f.confidence; } break;
      case "workflow.retry": if (f.step != null) ensure(f.step).retries = f.attempt || 0; break;
      case "workflow.node": if (f.node === "synthesize") synth = true; if (f.node === "reflect") reflect = true; break;
      case "agent.needs_approval": approval = { ...(approval || {}), operation: f.op, tool: f.tool ?? approval?.tool, args: f.args ?? approval?.args }; break;
    }
  }
  // Backfill agent names the event stream didn't carry (see buildTrace).
  for (const r of (task.result?.steps || [])) {
    const s = steps.get(r.id);
    if (s && (!s.agent || s.agent === "?") && r.agent) s.agent = r.agent;
  }
  return { plan, steps: [...steps.values()].sort((a, b) => a.step - b.step), synth, reflect, approval };
}

/* ``jumpKey`` links a flow node to its Explain card (matching ``data-xk``), so
 * clicking the node opens the trace for exactly that part of the run. Nodes with
 * no explainable trace (e.g. Result) simply omit it and stay non-interactive. */
function nodeHtml(cls, type, icon, title, bodyHtml = "", jumpKey = "") {
  const jump = jumpKey
    ? ` data-jump="${esc(jumpKey)}" role="button" tabindex="0" title="See what happened here"`
    : "";
  return `<div class="node ${cls}${jumpKey ? " jumpable" : ""}"${jump}>
    <div class="node-head"><span class="node-icon">${icon}</span>${esc(title)}</div>
    <div class="node-type">${esc(type)}</div>${bodyHtml}
    ${jumpKey ? `<div class="node-jump">Explain →</div>` : ""}</div>`;
}
const conn = () => `<div class="connector"></div>`;

/* Turn a pending approval into a human explanation of *what* will run and its
 * key detail (for shell: the exact command; for file writes: the path/content). */
function approvalWhat(a) {
  const tool = a.tool || "";
  const args = a.args || {};
  const batched = a.batched_calls || [];
  if (tool === "shell")
    return { verb: "run a shell command", detail: args.command || "", mono: true };
  if (tool === "write_file")
    return { verb: `write the file ${args.path || ""}`.trim(),
             detail: args.content != null ? String(args.content).slice(0, 800) : "", mono: true };
  if (tool === "make_dir")
    return { verb: `create the directory ${args.path || ""}`.trim(), detail: "", mono: false };
  if (tool === "download_resource")
    return { verb: "download a resource into the workspace", detail: `${args.url || ""} → ${args.destination || ""}`.trim(), mono: true };
  if (batched.length > 1 && batched.every(b => b.tool === "download_resource")) {
    const lines = batched.map(b => {
      const u = (b.args || {}).url || "";
      const d = (b.args || {}).destination || "";
      return `${u} → ${d}`.trim();
    });
    return {
      verb: `download ${batched.length} resources into the workspace`,
      detail: lines.join("\n"),
      mono: true,
    };
  }
  if (tool === "request_resource_input" || a.operation === "resource_acquisition")
    return { verb: "ask you how to proceed (resource missing or not downloaded)",
             detail: (a.message || args.message || "") + (args.candidate_urls ? "\nURLs: " + args.candidate_urls.join(", ") : ""),
             mono: false };
  if (a.operation === "waiting_for_user")
    return { verb: "wait for you to download the resource and provide a local path",
             detail: a.message || args.message || "",
             mono: false };
  const keys = Object.keys(args);
  return { verb: tool ? `use the ${tool} tool` : "perform a sensitive operation",
           detail: keys.length ? JSON.stringify(args, null, 2) : "", mono: true };
}

/** True when the task is paused and the approval UI should be shown. */
function taskAwaitingApproval(task) {
  return task.status === "awaiting_approval" || task.status === "waiting_for_user";
}

function approvalAllowAllHtml(taskId, approval) {
  const batched = (approval && approval.batched_calls) || [];
  const n = batched.length;
  const hint = n > 1
    ? ` (covers all ${n} pending downloads)`
  : " (won't ask again for each video/folder write in this run)";
  return `<label class="appr-allow-all">
    <input type="checkbox" id="appr-allow-all-${taskId}" class="appr-allow-all-cb" checked />
    <span>Allow all similar operations for this run${hint}</span>
  </label>`;
}

/** Shared “what is it asking for?” block — used in Flow and Results. */
function approvalExplainBody(a, task, opts = {}) {
  const w = approvalWhat(a);
  const detailText = w.detail || "";
  const detail = detailText
    ? `<div class="appr-detail ${w.mono ? "mono" : ""}">${
        opts.linkify ? linkify(detailText) : esc(detailText)
      }</div>`
    : "";
  const via = a.tool ? ` <span class="muted">via <code>${esc(a.tool)}</code></span>` : "";
  let actions = "";
  if (
    opts.showActions
    && (task.status === "awaiting_approval" || task.status === "waiting_for_user")
  ) {
    const btnId = opts.buttonId || `approve-${task.id}`;
    const btnLabel = opts.buttonLabel
      || (task.status === "waiting_for_user" ? "Continue" : "Approve & continue");
    const btnClass = opts.buttonClass ? ` ${opts.buttonClass}` : "";
    actions = `${approvalAllowAllHtml(task.id, a)}<button class="btn primary${btnClass}" id="${esc(btnId)}">${btnLabel}</button>`;
  }
  return `<div class="appr-what">Wants to <b>${esc(w.verb)}</b>${via}</div>${detail}${actions}`;
}

function approvalNode(a, task) {
  const op = a.operation || a.key || "sensitive operation";
  const paused = task.status === "awaiting_approval" || task.status === "waiting_for_user";
  const body = approvalExplainBody(a, task, paused ? {
    showActions: true,
    buttonId: `fapprove-${task.id}`,
    buttonClass: "appr-btn",
    buttonLabel: task.status === "waiting_for_user" ? "Continue" : "Approve & continue",
  } : {});
  return `<div class="node approval">
    <div class="node-head"><span class="node-icon">⏸</span>Approval required</div>
    <div class="node-type">${esc(op)}${a.step != null ? " · step " + esc(a.step) : ""}</div>
    <div class="appr-body">${body}</div></div>`;
}

function renderFlow(task, session) {
  const fl = deriveFlow(task);
  const agents = new Set(fl.steps.map(s => s.agent));
  if (fl.synth) agents.add("ceo");
  const tools = new Set();
  fl.steps.forEach(s => s.tools.forEach(t => tools.add(t.name)));

  const summary = `<div class="flow-summary">
    <div class="stat"><div class="v">${fl.steps.length || (fl.plan?.steps || 0)}</div><div class="k">Steps</div></div>
    <div class="stat"><div class="v">${agents.size}</div><div class="k">Agents</div></div>
    <div class="stat"><div class="v">${tools.size}</div><div class="k">Tools used</div></div>
    <div class="stat"><div class="v">${esc(statusLabel(task.status))}</div><div class="k">Status</div></div>
  </div>`;

  const nodes = [];
  nodes.push(nodeHtml("plan", "Planner", "◇",
    "Plan", `<div class="node-sub">${fl.plan ? fl.plan.steps + " step(s) decomposed" : "planning…"}</div>`,
    "comp:planner"));

  for (const s of fl.steps) {
    let chips = "";
    if (s.tools.size) {
      chips = `<div class="tool-chips">` + [...s.tools.values()].map(t =>
        `<span class="chip"><span class="s ${t.status}"></span>${esc(t.name)}${t.latency != null ? `<span class="muted">${Math.round(t.latency)}ms</span>` : ""}</span>`).join("") + `</div>`;
    }
    let badge = "";
    if (s.verified === true) badge = `<div class="badge ok">✓ verified${s.confidence != null ? " · " + s.confidence : ""}</div>`;
    else if (s.verified === false) badge = `<div class="badge fail">✗ needs work</div>`;
    const retry = s.retries ? `<div class="node-sub">retried ×${s.retries}</div>` : "";
    nodes.push(nodeHtml("step", `Step ${s.step} · agent`, "▣", s.agent, chips + badge + retry,
      "comp:step:" + s.step));
  }
  if (fl.approval && (task.status === "awaiting_approval" || task.status === "waiting_for_user"))
    nodes.push(approvalNode(fl.approval, task));
  if (fl.synth) nodes.push(nodeHtml("synthesize", "CEO", "★", "Synthesize", `<div class="node-sub">integrate step outputs</div>`, "comp:synthesize"));
  if (fl.reflect) nodes.push(nodeHtml("reflect", "Memory", "↻", "Reflect", `<div class="node-sub">store outcome + learn skill</div>`, "comp:reflect"));
  // The deliverable itself is produced by synthesis, so Result jumps there.
  if (task.status === "completed") nodes.push(nodeHtml("result", "Output", "✓", "Result", `<div class="node-sub">final deliverable ready</div>`, fl.synth ? "comp:synthesize" : ""));

  // Wire the in-flow approve button once it's in the DOM (mirrors the Result tab).
  if (fl.approval && (task.status === "awaiting_approval" || task.status === "waiting_for_user") && session) {
    setTimeout(() => {
      const btn = document.getElementById(`fapprove-${task.id}`);
      if (btn) btn.addEventListener("click", () => {
        const body = task.status === "waiting_for_user"
          ? { decision: "proceed_without" }
          : undefined;
        approveTask(session.id, task.id, body);
      });
    }, 0);
  }

  return summary + `<div class="flow">` + nodes.join(conn()) + `</div>`;
}

function statusLabel(s) {
  return { running: "Running", pending: "Starting", completed: "Done", failed: "Failed", awaiting_approval: "Needs approval", waiting_for_user: "Waiting for you", cancelled: "Cancelled" }[s] || s;
}

/* ── Explain (input → reasoning → output, per component) ───────────────── */
/* Walk the event stream and group it into the components the user sees: the
 * Planner, each executed Step (with its LLM calls, tool calls and verification),
 * the Synthesis, and Reflection. Every LLM call captures the exact messages in
 * and the content/usage out; every tool call its args in and output out. */
function buildTrace(task) {
  const comps = [];
  const steps = new Map();
  let planner = null, synth = null, reflect = null;

  const mkPlanner = () => planner || (comps.push(planner = { kind: "planner", title: "Planner", icon: "◇", input: null, output: null, llm: [], timeline: [] }), planner);
  const mkStep = (n) => {
    if (!steps.has(n)) { const c = { kind: "step", step: n, agent: "?", title: "Step " + n, icon: "▣", input: null, answer: null, llm: [], tools: [], timeline: [], verify: { input: null, verdict: null, llm: [], timeline: [] } }; steps.set(n, c); comps.push(c); }
    return steps.get(n);
  };
  const mkSynth = () => synth || (comps.push(synth = { kind: "synthesize", title: "Synthesize", icon: "★", agent: "ceo", input: null, output: null, llm: [], timeline: [] }), synth);
  const mkReflect = () => reflect || (comps.push(reflect = { kind: "reflect", title: "Reflect", icon: "↻", llm: [], timeline: [], notes: "Persist the outcome to long-term memory and distill a reusable skill." }), reflect);

  const traceTarget = (f) => {
    if (f.step != null) {
      const s = mkStep(f.step);
      return f.phase === "verify" ? { box: s.verify, comp: s } : { box: s, comp: s };
    }
    if (f.node === "plan" || f.purpose === "planner") return { box: mkPlanner(), comp: mkPlanner() };
    if (f.node === "synthesize" || f.purpose === "synthesize") return { box: mkSynth(), comp: mkSynth() };
    if (f.node === "reflect") return { box: mkReflect(), comp: mkReflect() };
    return { box: mkSynth(), comp: mkSynth() };
  };

  for (const e of (task.events || [])) {
    const f = e.fields || {};
    switch (e.event) {
      case "planner.input": mkPlanner().input = { goal: f.goal, agents: f.agents || [] }; break;
      case "planner.plan_ready": mkPlanner().output = { steps: f.steps, reasoning: f.reasoning, plan: f.plan || [] }; break;
      case "agent.start": if (f.step != null) mkStep(f.step).agent = f.agent || "?"; break;
      case "agent.context":
        if (f.step != null) { const s = mkStep(f.step); s.agent = f.agent || s.agent; s.input = { system: f.system, memories: f.memories || [], task: f.task }; }
        else if (f.node === "synthesize") mkSynth().input = { system: f.system, memories: f.memories || [], task: f.task };
        break;
      case "agent.answer":
        if (f.step != null) { mkStep(f.step).answer = { output: f.output, note: f.note }; }
        else if (f.node === "synthesize") mkSynth().output = { output: f.output };
        break;
      case "brain.request": {
        const t = traceTarget(f);
        const call = { req: f, resp: null };
        t.box.llm.push(call);
        t.box.timeline.push({ type: "llm", ref: call });
        break;
      }
      case "brain.response": {
        const b = traceTarget(f).box.llm;
        for (let i = b.length - 1; i >= 0; i--) if (!b[i].resp) { b[i].resp = f; break; }
        break;
      }
      case "tool.execute":
        if (f.step != null) {
          const s = mkStep(f.step);
          const tool = { name: f.tool, args: f.args, status: "running" };
          s.tools.push(tool);
          s.timeline.push({ type: "tool", ref: tool });
        }
        break;
      case "tool.result": if (f.step != null) { const ts = mkStep(f.step).tools; for (let i = ts.length - 1; i >= 0; i--) if (ts[i].name === f.tool && ts[i].status === "running") { Object.assign(ts[i], { status: f.ok ? "ok" : "fail", output: f.output, error: f.error, latency: f.latency_ms }); break; } } break;
      case "verifier.input": if (f.step != null) mkStep(f.step).verify.input = { task: f.task, criteria: f.criteria, output: f.output }; break;
      case "verifier.verdict": if (f.step != null) mkStep(f.step).verify.verdict = { passed: f.passed, confidence: f.confidence, issues: f.issues || [], suggestion: f.suggestion }; break;
      case "workflow.node": if (f.node === "synthesize") mkSynth(); if (f.node === "reflect") mkReflect(); break;
      case "approval.granted": comps.push({ kind: "approval", idx: comps.length, title: "Approved", icon: "✅", op: f.op, tool: f.tool, args: f.args }); break;
    }
  }
  // Some steps never emit an event carrying the agent name (it comes from the
  // bound log context). The finished result records it per step, so backfill
  // rather than leaving the card labelled "?".
  for (const r of (task.result?.steps || [])) {
    const c = steps.get(r.id);
    if (c && (!c.agent || c.agent === "?") && r.agent) c.agent = r.agent;
  }
  return comps;
}

const _jsonBlk = (obj) => `<pre class="x-json">${esc(typeof obj === "string" ? obj : JSON.stringify(obj, null, 2))}</pre>`;
const _field = (label, html) => `<div class="x-field"><div class="x-label">${esc(label)}</div>${html}</div>`;
const _text = (t) => `<div class="x-text">${linkify(t ?? "")}</div>`;

function msgsBlock(msgs) {
  if (!Array.isArray(msgs) || !msgs.length) return _text("(no messages)");
  return `<div class="x-msgs">` + msgs.map((m, idx) => {
    let extra = "";
    if (m.tool_calls && m.tool_calls.length) extra = `<div class="x-toolcalls">→ calls: ${m.tool_calls.map(c => `<code>${esc(c.name)}(${esc(JSON.stringify(c.args))})</code>`).join(" ")}</div>`;
    const roleLabel = m.role === "tool" && m.name ? `${m.role}:${esc(m.name)}` : esc(m.role);
    return `<div class="x-msg"><span class="x-role ${esc(m.role)}">${roleLabel}</span><span class="x-msg-idx">#${idx + 1}</span><div class="x-body"><span class="x-content">${linkify(m.content || (extra ? "" : "(empty)"))}</span>${extra}</div></div>`;
  }).join("") + `</div>`;
}

function llmInpSummary(messages) {
  if (!messages || !messages.length) return "0 msg";
  const roles = messages.map(m => m.role === "tool" && m.name ? `tool:${m.name}` : m.role);
  return `${messages.length} msg · ${roles.join(" → ")}`;
}

function llmOutSummary(resp, req) {
  if (!resp) return "waiting…";
  if (req.kind === "structured") return `JSON · ${esc(req.schema || "structured")}`;
  const tc = resp.tool_calls || [];
  if (tc.length) return `tool_calls · ${tc.map(t => t.name).join(", ")}`;
  const txt = (resp.content || "").trim();
  if (txt) return `text · ${txt.length > 48 ? txt.slice(0, 48) + "…" : txt}`;
  if (resp.reasoning) return "reasoning only";
  return esc(resp.finish_reason || "—");
}

function llmInpDelta(prevMessages, messages) {
  if (!prevMessages || !messages || messages.length <= prevMessages.length) return "";
  const added = messages.length - prevMessages.length;
  const newOnes = messages.slice(prevMessages.length);
  const roles = newOnes.map(m => m.role === "tool" && m.name ? `tool:${m.name}` : m.role);
  return `+${added} since previous call (${roles.join(", ")}) — this is why INP looks similar at the top; scroll down for new context.`;
}

function llmBlock(call, i, keyBase = "", prevCall = null) {
  const req = call.req || {}, resp = call.resp || {};
  const prevMsgs = prevCall && prevCall.req ? prevCall.req.messages : null;
  const inpNote = llmInpDelta(prevMsgs, req.messages);
  const tag = req.kind === "structured" ? `JSON ${esc(req.schema || "")}` : (resp.finish_reason ? esc(resp.finish_reason) : "");
  const tokens = resp.usage ? `${resp.usage.total_tokens} tok` : "";
  const outShort = llmOutSummary(resp, req);
  const llmKey = `${keyBase}:llm:${i}`;
  const summary =
    `🧠 LLM ${i + 1} · <span class="x-io"><span class="x-inp-tag">INP</span>→<span class="x-out-tag">OUT</span></span>` +
    ` · <span class="x-muted">${esc(llmInpSummary(req.messages))}</span>` +
    ` · <span class="x-out-short">OUT: ${outShort}</span>` +
    (tag ? ` · ${tag}` : "") +
    (tokens ? ` · <span class="x-muted">${tokens}</span>` : "") +
    (!call.resp ? ` · <span class="x-muted">running…</span>` : "");

  let inpBody = "";
  if (inpNote) inpBody += `<div class="x-inp-delta">${esc(inpNote)}</div>`;
  inpBody += msgsBlock(req.messages);
  if (req.tools && req.tools.length) inpBody += _field("Tools offered to model", `<div class="x-text">${req.tools.map(t => `<code>${esc(t)}</code>`).join(" ")}</div>`);

  let outBody = "";
  if (!call.resp) outBody += _text("(waiting for response…)");
  else if (req.kind === "structured") outBody += _jsonBlk(resp.output);
  else {
    if (resp.reasoning) outBody += _field("Reasoning (chain-of-thought)", _text(resp.reasoning));
    outBody += _field("Text content", _text(resp.content || "(no text — see tool calls below)"));
    if (resp.tool_calls && resp.tool_calls.length) outBody += _field("Tool calls requested", _jsonBlk(resp.tool_calls));
    if (resp.finish_reason) outBody += _field("Finish reason", _text(resp.finish_reason));
    if (resp.usage) outBody += _field("Token usage", _text(`${resp.usage.prompt_tokens} prompt + ${resp.usage.completion_tokens} completion = ${resp.usage.total_tokens} total`));
  }

  const inpSummary =
    `<span class="x-inp-tag">INP</span> — messages sent to the model` +
    ` · <span class="x-muted">${esc(llmInpSummary(req.messages))}</span>`;
  const outSummary =
    `<span class="x-out-tag">OUT</span> — model response` +
    ` · <span class="x-muted">${outShort}</span>`;

  const inpSection =
    `<details class="x-llm-section x-llm-inp" data-xk="${esc(llmKey)}:inp" open>` +
    `<summary>${inpSummary}</summary><div class="x-llm-section-body">${inpBody}</div></details>`;
  const outSection =
    `<details class="x-llm-section x-llm-out" data-xk="${esc(llmKey)}:out" open>` +
    `<summary>${outSummary}</summary><div class="x-llm-section-body">${outBody}</div></details>`;

  return `<details class="x-call x-llm-call" data-xk="${esc(llmKey)}"><summary>${summary}</summary><div class="x-call-body">${inpSection}${outSection}</div></details>`;
}

function toolBlock(t, i = 0, keyBase = "") {
  const stClass = t.status === "ok" ? "ok" : (t.status === "fail" ? "fail" : "run");
  const lat = t.latency != null ? `<span class="x-muted">· ${Math.round(t.latency)}ms</span>` : "";
  const mark = t.status === "ok" ? "✓" : (t.status === "fail" ? "✗" : "…");
  let body = _field("Input (arguments)", _jsonBlk(t.args || {}));
  if (t.error) body += _field("Error", _text(t.error));
  body += _field("Output", _text(t.output ?? "(no output captured)"));
  return `<details class="x-call tool ${stClass}" data-xk="${esc(keyBase)}:tool:${i}"><summary>🔧 ${esc(t.name)} <span class="x-mark ${stClass}">${mark}</span> ${lat}</summary><div class="x-call-body">${body}</div></details>`;
}

function verifyBlock(v, keyBase = "") {
  const vd = v.verdict;
  const badge = vd ? (vd.passed ? `<span class="x-mark ok">✓ passed</span>` : `<span class="x-mark fail">✗ needs work</span>`) : `<span class="x-muted">running…</span>`;
  const conf = vd && vd.confidence != null ? `<span class="x-muted">· ${(vd.confidence * 100).toFixed(0)}% conf</span>` : "";
  let body = "";
  if (v.input) body += _field("Input (what was reviewed)", _text(v.input.output));
  v.llm.forEach((c, i) => body += llmBlock(c, i, keyBase + ":verify", i > 0 ? v.llm[i - 1] : null));
  if (vd) {
    if (vd.issues && vd.issues.length) body += _field("Issues found", `<ul class="x-list">${vd.issues.map(x => `<li>${linkify(x)}</li>`).join("")}</ul>`);
    if (vd.suggestion) body += _field("Suggested fix", _text(vd.suggestion));
  }
  return `<details class="x-call verify" data-xk="${esc(keyBase)}:verify"><summary>🔎 Verification ${badge} ${conf}</summary><div class="x-call-body">${body || _text("(no verification data)")}</div></details>`;
}

function procBlocksGrouped(llm, tools, keyBase) {
  let body = "";
  llm.forEach((l, i) => body += llmBlock(l, i, keyBase, i > 0 ? llm[i - 1] : null));
  tools.forEach((t, i) => body += toolBlock(t, i, keyBase));
  return body;
}

function procBlocksTimeline(timeline, keyBase) {
  let body = "";
  let llmIdx = 0;
  let toolIdx = 0;
  let prevLlm = null;
  for (const item of (timeline || [])) {
    if (item.type === "llm") {
      body += llmBlock(item.ref, llmIdx, keyBase, prevLlm);
      prevLlm = item.ref;
      llmIdx += 1;
    } else if (item.type === "tool") {
      body += toolBlock(item.ref, toolIdx, keyBase);
      toolIdx += 1;
    }
  }
  return body;
}

function compCard(c, explainOrder = "grouped") {
  // Stable key per component so the poller can restore expand/collapse state
  // across re-renders (otherwise a running task collapses whatever you open).
  const ck = "comp:" + c.kind + (c.kind === "step" ? ":" + c.step : (c.kind === "approval" ? ":" + c.idx : ""));
  let body = "";
  if (c.kind === "approval") {
    const w = approvalWhat({ tool: c.tool, args: c.args });
    body += `<div class="appr-what">You approved: <b>${esc(w.verb)}</b>${c.tool ? ` <span class="muted">via <code>${esc(c.tool)}</code></span>` : ""}</div>`;
    if (w.detail) body += `<div class="appr-detail ${w.mono ? "mono" : ""}">${linkify(w.detail)}</div>`;
  } else if (c.kind === "planner") {
    if (c.input) body += _field("Input (goal)", _text(c.input.goal)) + _field("Available agents", `<div class="x-text">${(c.input.agents || []).map(a => `<code>${esc(a)}</code>`).join(" ")}</div>`);
    body += explainOrder === "timeline" ? procBlocksTimeline(c.timeline, ck) : procBlocksGrouped(c.llm, [], ck);
    if (c.output) {
      if (c.output.reasoning) body += _field("Reasoning", _text(c.output.reasoning));
      body += _field("Output (plan)", (c.output.plan || []).length
        ? `<ol class="x-plan">${c.output.plan.map(p => `<li><b>${esc(p.agent)}</b>: ${linkify(p.description)}${p.success_criteria ? `<div class="x-muted">✓ ${linkify(p.success_criteria)}</div>` : ""}</li>`).join("")}</ol>`
        : _text("(planning…)"));
    }
  } else if (c.kind === "step") {
    if (c.input) {
      body += _field("Input (task given to agent)", _text(c.input.task));
      if (c.input.system) body += _field("Agent role (system prompt)", _text(c.input.system));
      if (c.input.memories && c.input.memories.length) body += _field("Recalled knowledge", `<ul class="x-list">${c.input.memories.map(m => `<li>${linkify(m)}</li>`).join("")}</ul>`);
    }
    if (c.llm.length || c.tools.length) {
      body += `<div class="x-label">Reasoning & actions</div><div class="x-proc">`;
      body += explainOrder === "timeline"
        ? procBlocksTimeline(c.timeline, ck)
        : procBlocksGrouped(c.llm, c.tools, ck);
      body += `</div>`;
    }
    if (c.answer) body += _field("Output (agent's answer)", _text(c.answer.output) + (c.answer.note ? `<div class="x-muted">⚠ ${linkify(c.answer.note)}</div>` : ""));
    if (c.verify.input || c.verify.verdict || c.verify.llm.length) body += verifyBlock(c.verify, ck);
  } else if (c.kind === "synthesize") {
    if (c.input) body += _field("Input (team's step outputs)", _text(c.input.task));
    body += explainOrder === "timeline" ? procBlocksTimeline(c.timeline, ck) : procBlocksGrouped(c.llm, [], ck);
    if (c.output) body += _field("Output (final deliverable)", _text(c.output.output));
  } else if (c.kind === "reflect") {
    body += _text(c.notes);
    body += explainOrder === "timeline" ? procBlocksTimeline(c.timeline, ck) : procBlocksGrouped(c.llm, [], ck);
  }

  let sub = "";
  if (c.kind === "step") sub = `agent: ${esc(c.agent)}`;
  else if (c.kind === "synthesize") sub = "CEO integrates the work";
  return `<details class="x-comp ${c.kind}" data-xk="${esc(ck)}" open><summary><span class="x-icon">${c.icon || "•"}</span><b>${esc(c.title)}</b>${sub ? `<span class="x-sub">${sub}</span>` : ""}</summary><div class="x-comp-body">${body || _text("…")}</div></details>`;
}

function renderExplain(task, session) {
  const comps = buildTrace(task);
  if (!comps.length) return `<div class="placeholder">No explainable steps yet… the trace appears as the employee works.</div>`;
  const order = session?.explainOrder || "grouped";
  const intro = `<div class="x-intro">Each 🧠 LLM row is one model round-trip: <span class="x-inp-tag">INP</span> (messages sent) and <span class="x-out-tag">OUT</span> (what came back) — both expand when you open the call; collapse <span class="x-inp-tag">INP</span> to focus on output. 🔧 tools run <i>between</i> LLM calls and are not LLM calls.</div>`;
  const toolbar = `<div class="x-explain-toolbar">
    <div class="x-explain-toolbar-label">Order</div>
    <div class="x-seg">
      <button type="button" class="x-seg-btn ${order === "grouped" ? "active" : ""}" data-explain-order="grouped">Grouped</button>
      <button type="button" class="x-seg-btn ${order === "timeline" ? "active" : ""}" data-explain-order="timeline">Execution Order</button>
    </div>
  </div>`;
  return intro + toolbar + `<div class="x-trace">` + comps.map(c => compCard(c, order)).join(`<div class="x-conn"></div>`) + `</div>`;
}

/* ── Logs ──────────────────────────────────────────────────────────────── */
function renderLogs(task) {
  const lines = (task.events || []).map(e => {
    const f = e.fields || {};
    const fields = Object.entries(f).map(([k, v]) =>
      `<b>${esc(k)}</b>=${esc(typeof v === "object" ? JSON.stringify(v) : v)}`).join(" ");
    const ts = (e.ts || "").slice(11, 19);
    const approved = e.event === "approval.granted";
    return `<div class="log-line${approved ? " approved" : ""}">
      <span class="log-ts">${esc(ts)}</span>
      <span class="log-lvl ${esc(e.level)}">${approved ? "✅" : esc(e.level)}</span>
      <span class="log-event">${approved ? "APPROVED" : esc(e.event)}</span>
      <span class="log-fields">${fields}</span></div>`;
  });
  if (!lines.length) return `<div class="placeholder">No log events yet…</div>`;
  return `<div class="logs">${lines.join("")}</div>`;
}

/* ── Result ────────────────────────────────────────────────────────────── */
/* Live progress for a run that hasn't produced its final answer yet. Reads the
 * plan and each finished step's output straight from the event stream so the
 * Result tab fills in as work completes rather than sitting empty. */
function renderPartialResult(task) {
  const fl = deriveFlow(task);
  const answers = new Map();     // step id -> agent's answer
  let planSteps = [], synthesizing = false;
  for (const e of (task.events || [])) {
    const f = e.fields || {};
    if (e.event === "planner.plan_ready" && (f.plan || []).length) planSteps = f.plan;
    else if (e.event === "agent.answer" && f.step != null) answers.set(f.step, f.output || "");
    else if (e.event === "agent.context" && f.node === "synthesize") synthesizing = true;
  }

  if (!planSteps.length && !answers.size) {
    const what = fl.plan ? "Planning the work…" : "Starting up…";
    return `<div class="placeholder"><span class="spin">◌</span> ${what}</div>`;
  }

  const done = answers.size;
  const total = planSteps.length || fl.steps.length || done;
  const pct = total ? Math.round((done / total) * 100) : 0;

  let html = `<div class="partial-head">
    <div class="partial-title"><span class="spin">◌</span> Working — ${done} of ${total} step(s) done</div>
    <div class="conf-bar partial-bar"><div class="conf-fill" style="width:${pct}%"></div></div>
  </div>`;

  // The plan as a live checklist: done / running / queued.
  if (planSteps.length) {
    html += `<div class="partial-plan"><h3>Plan</h3>` + planSteps.map(p => {
      const isDone = answers.has(p.id);
      const running = !isDone && fl.steps.some(s => s.step === p.id);
      const mark = isDone ? "✓" : (running ? "◌" : "·");
      const cls = isDone ? "done" : (running ? "running" : "queued");
      return `<div class="partial-step ${cls}"><span class="pm">${mark}</span>
        <span class="pa">${esc(p.agent || "")}</span>
        <span class="pd">${linkify(p.description || "")}</span></div>`;
    }).join("") + `</div>`;
  }

  // Outputs already produced, newest last — the actual "results so far".
  const ids = [...answers.keys()].sort((a, b) => a - b);
  if (ids.length) {
    html += `<div class="steps-list"><h3>Results so far</h3>` + ids.map(id => {
      const st = fl.steps.find(s => s.step === id);
      const agent = st?.agent && st.agent !== "?" ? st.agent : (planSteps.find(p => p.id === id)?.agent || "");
      const ok = st?.verified === true ? " ✓" : "";
      return `<div class="step-card"><h4>Step ${esc(id)}${agent ? " · " + esc(agent) : ""}${ok}</h4>
        <div class="out md">${renderMarkdown((answers.get(id) || "").slice(0, 1200))}</div></div>`;
    }).join("") + `</div>`;
  }

  if (synthesizing) html += `<div class="partial-note"><span class="spin">◌</span> Integrating the step outputs into the final answer…</div>`;
  return html;
}

function renderResult(task, session) {
  if (task.resumable) {
    const reason = task.error || "This run was interrupted before it finished.";
    const partial = task.result && task.result.answer
      ? `<div class="result-answer md">${renderMarkdown(task.result.answer)}</div>`
      : `<div class="x-muted" style="margin-top:10px">No final answer was produced yet. Resume to continue from the last saved step.</div>`;
    setTimeout(() => {
      const btn = document.getElementById(`resume-${task.id}`);
      if (btn) btn.addEventListener("click", () => resumeTask(session.id, task.id));
    }, 0);
    return `<div class="approval-banner result-approval-banner">
      <div class="result-appr-head">⏸ Run interrupted</div>
      <div class="appr-body">${linkify(reason)}</div>
      <button class="btn primary" id="resume-${task.id}" style="margin-top:10px">Resume from last checkpoint</button>
    </div>${partial}`;
  }
  if (task.status === "failed") return `<div class="approval-banner" style="border-color:var(--red);color:var(--red);background:rgba(239,94,107,.12)">Task failed: ${linkify(task.error || "unknown error")}</div>`;
  if (task.status === "cancelled") return `<div class="approval-banner" style="border-color:var(--muted);color:var(--muted);background:rgba(140,140,150,.10)">Task stopped: ${linkify(task.error || "cancelled by user")}</div>`;
  const r = task.result;
  // ``task.result`` only lands when the whole pipeline finishes, but finished
  // steps are already in the event stream — show them as they arrive instead of
  // a spinner that looks like nothing is happening.
  if (!r) return renderPartialResult(task);

  if (task.mode === "research" && r.result) return renderResearch(task, session, r);
  if (task.mode === "plan" && r.result) return renderPlan(task, session, r);

  let html = "";
  if (r.needs_approval && taskAwaitingApproval(task)) {
    const na = r.needs_approval;
    const op = na.operation || "operation";
    if (op === "waiting_for_user") {
      html += resourceApprovalBanner(task, session, na, true);
    } else if (op === "resource_acquisition" || na.tool === "request_resource_input") {
      html += resourceApprovalBanner(task, session, na, false);
    } else {
      html += `<div class="approval-banner result-approval-banner">
        <div class="result-appr-head">⏸ Approval required — <b>${esc(op)}</b></div>
        <div class="appr-body">${approvalExplainBody(na, task, {
          showActions: true,
          linkify: true,
          buttonId: `approve-${task.id}`,
        })}</div>
      </div>`;
    }
  }
  const conf = r.confidence != null ? r.confidence : 0;
  html += `<div class="result-answer md">${renderMarkdown(r.answer || "(no answer produced)")}</div>
    <div class="result-meta">
      <span class="muted">Confidence</span>
      <div class="conf-bar"><div class="conf-fill" style="width:${Math.round(conf * 100)}%"></div></div>
      <span>${(conf * 100).toFixed(0)}%</span>
    </div>`;

  html += `<div id="media-gallery-${esc(task.id)}" class="media-gallery"></div>`;
  html += `<div id="deliverables-${esc(task.id)}"></div>`;

  const steps = r.steps || [];
  if (steps.length) {
    html += `<div class="steps-list"><h3>Step outputs</h3>` + steps.map(st =>
      `<div class="step-card"><h4>Step ${esc(st.id)} · ${esc(st.agent)} ${st.verified ? "✓" : ""}</h4>
        <div class="out md">${renderMarkdown((st.output || "").slice(0, 1200))}</div></div>`).join("") + `</div>`;
  }

  html += tokensBlock(r.tokens);

  // Defer wiring the approve button until the node is in the DOM.
  setTimeout(() => {
    const btn = document.getElementById(`approve-${task.id}`);
    if (btn) btn.addEventListener("click", () => {
      const body = task.status === "waiting_for_user"
        ? { decision: "proceed_without" }
        : undefined;
      approveTask(session.id, task.id, body);
    });
    wireResourceApproval(task, session);
    loadMediaGallery(task.id);
    loadDeliverables(task.id);
  }, 0);
  return html;
}

/* ── Token usage card (overall + per-agent) ─────────────────────────────── */
function tokensBlock(tokens) {
  if (!tokens || !tokens.total_tokens) return "";
  const by = tokens.by_purpose || {};
  const vals = Object.values(by).map(v => v.total_tokens);
  const max = Math.max(1, ...vals);
  const rows = Object.entries(by)
    .sort((a, b) => b[1].total_tokens - a[1].total_tokens)
    .map(([k, v]) => `<div class="tok-row">
      <span class="tok-name">${esc(k.replace(/^agent:/, ""))}</span>
      <span class="tok-bar"><span style="width:${Math.round(v.total_tokens / max * 100)}%"></span></span>
      <span class="tok-val">${v.total_tokens.toLocaleString()} <span class="x-muted">· ${v.calls} call${v.calls === 1 ? "" : "s"}</span></span>
    </div>`).join("");
  return `<div class="tokens-card"><h3>🧮 Token usage
    <span class="tok-total">${tokens.total_tokens.toLocaleString()} total · ${tokens.calls} calls</span></h3>
    ${rows || `<div class="x-muted">no per-agent breakdown</div>`}</div>`;
}

/* ── Research result (found models / datasets / papers / hardware) ──────── */
function _link(url) { return url ? `<a href="${esc(url)}" target="_blank" class="x-link">↗</a>` : ""; }

function findingsBlock(title, items, fmt) {
  items = items || [];
  if (!items.length)
    return `<div class="steps-list"><h3>${title}</h3><div class="step-card x-muted">none found</div></div>`;
  return `<div class="steps-list"><h3>${title} <span class="x-muted">(${items.length})</span></h3>` +
    items.map(it => `<div class="step-card"><div class="out">${fmt(it)}</div></div>`).join("") + `</div>`;
}

function renderResearch(task, session, r) {
  const res = r.result || {};
  let html = "";
  if (r.needs_approval && taskAwaitingApproval(task)) {
    const na = r.needs_approval;
    const op = na.operation || "operation";
    html += `<div class="approval-banner result-approval-banner">
      <div class="result-appr-head">⏸ Approval required — <b>${esc(op)}</b> <span class="muted">(run generated code locally)</span></div>
      <div class="appr-body">${approvalExplainBody(na, task, {
        showActions: true,
        linkify: true,
        buttonId: `approve-${task.id}`,
        buttonLabel: "Approve &amp; run",
      })}</div>
    </div>`;
  }
  const branchLabel = {
    both: "Model + dataset found", model_only: "Model only",
    data_only: "Dataset only", neither: "No ready model/dataset — reproduced from paper",
  }[res.branch] || res.branch || "";
  html += `<div class="result-answer"><b>${esc(branchLabel)}.</b> ${linkify(res.summary || "")}</div>`;

  html += findingsBlock("📦 Models", res.found_model, m =>
    `<b>${esc(m.id || "")}</b> ${_link(m.url)} ${m.task ? `<span class="x-muted">· ${esc(m.task)}</span>` : ""} ${m.popularity ? `<span class="x-muted">· ${esc(String(m.popularity))}</span>` : ""}`);
  html += findingsBlock("🗂 Datasets", res.found_dataset, d =>
    `<b>${esc(d.id || "")}</b> ${_link(d.url)}`);
  html += findingsBlock("📄 Papers", res.papers, p =>
    `<b>${esc(p.title || "")}</b> ${_link(p.url)} ${p.code_url ? `· code ${_link(p.code_url)}` : ""}`);

  if (res.hardware && res.hardware.summary)
    html += `<div class="steps-list"><h3>🖥 Hardware</h3><div class="step-card"><div class="out">${linkify(res.hardware.summary)}</div></div></div>`;
  if (res.generated && res.generated.instructions)
    html += `<div class="steps-list"><h3>🛠 Generated code &amp; instructions</h3><div class="step-card"><div class="out">${linkify(res.generated.instructions.slice(0, 4000))}</div></div></div>`;
  if (res.accuracy && res.accuracy.report)
    html += `<div class="steps-list"><h3>🎯 Accuracy</h3><div class="step-card"><div class="out">${linkify(res.accuracy.report.slice(0, 2000))}</div></div></div>`;

  html += tokensBlock(r.tokens || res.tokens);

  setTimeout(() => {
    const btn = document.getElementById(`approve-${task.id}`);
    if (btn) btn.addEventListener("click", () => approveTask(session.id, task.id));
  }, 0);
  return html;
}

/* ── Plan result (step-by-step guide + OSS / proprietary tools) ─────────── */
function _toolKindBadge(kind) {
  const k = String(kind || "unknown").toLowerCase();
  const label = {
    open_source: "Open source",
    proprietary: "Proprietary",
    freemium: "Freemium",
    saas: "SaaS",
    unknown: "Unknown",
  }[k] || k;
  return `<span class="tool-kind tool-kind-${esc(k)}">${esc(label)}</span>`;
}

function _fmtTool(t) {
  const lic = t.license ? `<span class="x-muted">· ${esc(t.license)}</span>` : "";
  const purpose = t.purpose ? `<div class="x-muted">${esc(t.purpose)}</div>` : "";
  const notes = t.notes ? `<div class="x-muted">${linkify(t.notes)}</div>` : "";
  return `<div class="tool-row"><b>${esc(t.name || "")}</b> ${_link(t.url)} ${_toolKindBadge(t.kind)} ${lic}${purpose}${notes}</div>`;
}

function renderPlan(task, session, r) {
  const res = r.result || {};
  let html = "";
  html += `<div class="result-answer"><b>How-to plan.</b> ${linkify(res.summary || "")}</div>`;
  if (res.approach)
    html += `<div class="steps-list"><h3>Approach</h3><div class="step-card"><div class="out">${linkify(res.approach)}</div></div></div>`;

  const steps = res.steps || [];
  if (steps.length) {
    html += `<div class="steps-list plan-steps"><h3>Step-by-step guide <span class="x-muted">(${steps.length})</span></h3>` +
      steps.map((st, i) => {
        const tools = (st.tools || []).map(n => `<code>${esc(n)}</code>`).join(" ");
        const deps = (st.depends_on || []).length
          ? `<div class="x-muted">After step ${(st.depends_on || []).join(", ")}</div>` : "";
        const ok = st.success_criteria
          ? `<div class="plan-ok">✓ ${esc(st.success_criteria)}</div>` : "";
        const tips = st.tips ? `<div class="x-muted">Tip: ${linkify(st.tips)}</div>` : "";
        return `<div class="step-card plan-step">
          <h4><span class="plan-num">${esc(String(st.id || i + 1))}</span> ${esc(st.title || "Step")}</h4>
          <div class="out">${linkify(st.instruction || "")}</div>
          ${tools ? `<div class="plan-tools">Tools: ${tools}</div>` : ""}
          ${deps}${ok}${tips}
        </div>`;
      }).join("") + `</div>`;
  } else {
    html += `<div class="steps-list"><h3>Step-by-step guide</h3><div class="step-card x-muted">no steps produced</div></div>`;
  }

  html += findingsBlock("🛠 Open-source tools", res.tools_open_source, _fmtTool);
  html += findingsBlock("🏢 Proprietary / SaaS tools", res.tools_proprietary, _fmtTool);
  if ((res.tools_other || []).length)
    html += findingsBlock("🧰 Other tools", res.tools_other, _fmtTool);
  if (!(res.tools_open_source || []).length && !(res.tools_proprietary || []).length && (res.tools || []).length)
    html += findingsBlock("🧰 Tools", res.tools, _fmtTool);

  if (res.alternatives)
    html += `<div class="steps-list"><h3>Alternatives</h3><div class="step-card"><div class="out">${linkify(res.alternatives)}</div></div></div>`;
  if (res.risks)
    html += `<div class="steps-list"><h3>Risks &amp; blockers</h3><div class="step-card"><div class="out">${linkify(res.risks)}</div></div></div>`;

  html += tokensBlock(r.tokens || res.tokens);
  return html;
}

/* ── Bootstrap ─────────────────────────────────────────────────────────── */
function syncComposerModeUi() {
  const modeSel = $("#task-mode");
  const input = $("#new-task-input");
  const submit = $("#new-task-form button[type='submit']");
  if (!modeSel || !input || !submit) return;
  const mode = modeSel.value || "task";
  if (mode === "download") {
    input.placeholder = "Paste a YouTube, GitHub, Hugging Face, Kaggle, or file URL…";
    submit.textContent = "⬇ Download";
  } else if (mode === "research") {
    input.placeholder = "Describe the model/dataset/paper you want researched…";
    submit.textContent = "▶ Run";
  } else if (mode === "plan") {
    input.placeholder = "What do you want a step-by-step plan for? (tools + how-to)…";
    submit.textContent = "📋 Plan";
  } else {
    input.placeholder = "Give the employee a goal…  (e.g. Research X and write a brief)";
    submit.textContent = "▶ Run";
  }
}

$("#new-task-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const input = $("#new-task-input");
  const goal = input.value.trim();
  if (!goal) return;
  input.value = "";
  const mode = $("#task-mode") ? $("#task-mode").value : "task";
  newTask(goal, mode).catch(err => alert("Failed to start task: " + err.message));
});
if ($("#task-mode")) {
  $("#task-mode").addEventListener("change", syncComposerModeUi);
  syncComposerModeUi();
}
if ($("#auto-approve")) {
  syncAutoApproveUi();
  $("#auto-approve").addEventListener("change", (e) => setAutoApprove(e.target.checked));
}

$("#refresh-sessions").addEventListener("click", loadSessionList);
$("#delete-all-sessions").addEventListener("click", deleteAllSessions);

async function clearAllMemories() {
  if (!confirm(
    "Clear ALL stored memories?\n\n" +
    "This removes long-term recall, skills, conversation history, and knowledge " +
    "chunks. Saved tasks stay — but follow-ups will not remember past runs.\n\n" +
    "A backup is kept so you can restore with ↩ if needed."
  )) return;
  const btn = $("#clear-memories");
  btn.disabled = true;
  try {
    const res = await api("DELETE", "/ui/api/memory");
    const n = res.records_removed;
    alert(n != null ? `Cleared ${n} memory record${n === 1 ? "" : "s"}.` : "All memories cleared.");
    await loadMemoryBadge();
  } catch (e) {
    alert("Failed to clear memories: " + e.message);
  }
  btn.disabled = false;
}

async function restoreArchivedMemories() {
  if (!confirm("Restore memory from before the last clear?\n\nThis replaces current memory with the archived backup.")) return;
  const btn = $("#restore-memories");
  btn.disabled = true;
  try {
    const res = await api("POST", "/ui/api/memory/restore");
    alert(`Restored ${res.records_restored} memory record${res.records_restored === 1 ? "" : "s"}.`);
    await loadMemoryBadge();
  } catch (e) {
    alert("Failed to restore memory: " + e.message);
  }
  btn.disabled = false;
}

let _memoryPop = null;
let _memorySettings = null;

async function loadMemoryBadge() {
  const badge = $("#memory-badge");
  let st;
  try { st = await api("GET", "/ui/api/memory/settings"); }
  catch { badge.textContent = "memory: ?"; return; }
  _memorySettings = st;
  const on = st.use_stored_memory;
  badge.classList.toggle("warn", !on);
  badge.innerHTML =
    `<span class="b-led"></span><span>Memory: ${on ? "On" : "Off"}</span><span class="caret">▾</span>`;
  badge.title = on
    ? "Using stored memories for recall. Click to change."
    : "Ignoring stored memories (fresh context). Click to change.";
  const restoreBtn = $("#restore-memories");
  if (restoreBtn) {
    restoreBtn.disabled = !st.backup_available;
    restoreBtn.title = st.backup_available
      ? `Restore ${st.backup_records} archived record(s) from before last clear`
      : "No archived memory to restore";
  }
}

function openMemoryPicker() {
  if (_memoryPop) { _memoryPop.remove(); _memoryPop = null; return; }
  const st = _memorySettings || { use_stored_memory: true, backup_available: false };
  const pop = el(`<div class="brain-pop memory-pop">
    <h4>Stored memory</h4>
    <label class="bp-save-wrap"><input type="checkbox" class="mp-use" ${st.use_stored_memory ? "checked" : ""} />
      <span>Use stored memory (recall prior knowledge &amp; save new outcomes)</span></label>
    <label class="bp-save-wrap"><input type="checkbox" class="mp-save-default" />
      <span>Set as default (remember after refresh)</span></label>
    <p class="hint mp-hint">${st.use_stored_memory
      ? "Agents will recall long-term memory and skills when planning."
      : "Tasks run with a fresh context — no recall, no new long-term writes."}</p>
    <button class="btn mp-restore" style="width:100%;margin-top:8px" ${st.backup_available ? "" : "disabled"}>
      ↩ Restore previous memory (${st.backup_available ? st.backup_records : 0} records)
    </button>
    <div class="row">
      <button class="btn mp-cancel">Cancel</button>
      <button class="btn primary mp-apply">Apply</button>
    </div></div>`);
  document.body.appendChild(pop);
  _memoryPop = pop;
  const close = () => { pop.remove(); _memoryPop = null; };
  const useCb = $(".mp-use", pop);
  const hint = $(".mp-hint", pop);
  useCb.addEventListener("change", () => {
    hint.textContent = useCb.checked
      ? "Agents will recall long-term memory and skills when planning."
      : "Tasks run with a fresh context — no recall, no new long-term writes.";
  });
  const saveDefault = $(".mp-save-default", pop);
  saveDefault.checked = typeof st.saved_default === "boolean" ? st.saved_default : false;
  $(".mp-cancel", pop).addEventListener("click", close);
  $(".mp-restore", pop).addEventListener("click", async () => {
    if (!st.backup_available) return;
    if (!confirm("Restore memory from before the last clear?")) return;
    const btn = $(".mp-restore", pop);
    btn.disabled = true;
    try {
      const res = await api("POST", "/ui/api/memory/restore");
      alert(`Restored ${res.records_restored} record(s).`);
      close();
      await loadMemoryBadge();
    } catch (e) {
      alert("Restore failed: " + e.message);
      btn.disabled = false;
    }
  });
  $(".mp-apply", pop).addEventListener("click", async () => {
    const applyBtn = $(".mp-apply", pop);
    applyBtn.textContent = "Applying…"; applyBtn.disabled = true;
    try {
      await api("POST", "/ui/api/memory/settings", {
        use_stored_memory: useCb.checked,
        save_default: saveDefault.checked,
      });
      close();
      await loadMemoryBadge();
    } catch (e) {
      hint.className = "hint mp-hint warn";
      hint.textContent = "Failed: " + e.message;
      applyBtn.textContent = "Apply"; applyBtn.disabled = false;
    }
  });
  const badge = $("#memory-badge");
  const r = badge.getBoundingClientRect();
  pop.style.position = "absolute";
  pop.style.top = (r.bottom + 6) + "px";
  pop.style.right = (window.innerWidth - r.right) + "px";
}

$("#clear-memories").addEventListener("click", clearAllMemories);
$("#restore-memories").addEventListener("click", restoreArchivedMemories);
$("#memory-badge").addEventListener("click", openMemoryPicker);
$("#brain-badge").addEventListener("click", openBrainPicker);
$("#coding-badge").addEventListener("click", openCodingPicker);
$("#keys-badge").addEventListener("click", openKeysModal);
$("#usage-badge").addEventListener("click", openUsageModal);

/* ── Token usage dashboard (per brain, by date) ─────────────────────────── */
let _usageModal = null;

async function openUsageModal() {
  if (_usageModal) return;
  const overlay = el(`<div class="modal-overlay"><div class="modal usage-modal">
    <div class="modal-head"><h3>📊 Token usage</h3><button class="icon-btn modal-close" title="Close">✕</button></div>
    <div class="usage-filter">
      <label>From <input type="date" class="u-start" /></label>
      <label>To <input type="date" class="u-end" /></label>
      <button class="btn u-apply">Apply</button>
      <button class="btn u-clear">All time</button>
    </div>
    <div class="usage-body"><div class="placeholder"><span class="spin">◌</span> Loading…</div></div>
  </div></div>`);
  document.body.appendChild(overlay);
  _usageModal = overlay;
  const close = () => { overlay.remove(); _usageModal = null; };
  $(".modal-close", overlay).addEventListener("click", close);
  overlay.addEventListener("click", (e) => { if (e.target === overlay) close(); });

  const load = async () => {
    const start = $(".u-start", overlay).value, end = $(".u-end", overlay).value;
    const qs = new URLSearchParams();
    if (start) qs.set("start", start);
    if (end) qs.set("end", end);
    let data;
    try { data = await api("GET", "/ui/api/usage" + (qs.toString() ? "?" + qs : "")); }
    catch (e) { $(".usage-body", overlay).innerHTML = `<div class="key-status err">✗ ${esc(e.message)}</div>`; return; }
    renderUsage(overlay, data);
  };
  $(".u-apply", overlay).addEventListener("click", load);
  $(".u-clear", overlay).addEventListener("click", () => {
    $(".u-start", overlay).value = ""; $(".u-end", overlay).value = ""; load();
  });
  load();
}

function renderUsage(overlay, data) {
  const body = $(".usage-body", overlay);
  const t = data.total || {};
  if (!t.total_tokens) { body.innerHTML = `<div class="x-muted" style="padding:16px 0">No usage recorded${data.range && (data.range.start || data.range.end) ? " in this range" : " yet"}.</div>`; return; }
  const n = (x) => (x || 0).toLocaleString();

  const provRows = Object.entries(data.by_provider || {}).map(([prov, v]) =>
    `<tr><td><b>${esc(prov)}</b></td><td>${n(v.total_tokens)}</td>
      <td class="x-muted">${n(v.prompt_tokens)}</td><td class="x-muted">${n(v.completion_tokens)}</td>
      <td>${n(v.calls)}</td></tr>`).join("");

  const dateRows = Object.entries(data.by_date || {}).reverse().map(([d, v]) =>
    `<tr><td>${esc(d)}</td><td>${n(v.total_tokens)}</td><td>${n(v.calls)}</td></tr>`).join("");

  body.innerHTML = `
    <div class="usage-total">${n(t.total_tokens)} <span class="x-muted">tokens · ${n(t.calls)} calls · ${data.days} day${data.days === 1 ? "" : "s"}</span></div>
    <h4>By brain (provider)</h4>
    <table class="usage-table"><thead><tr><th>Brain</th><th>Total</th><th>Prompt</th><th>Completion</th><th>Calls</th></tr></thead>
      <tbody>${provRows}</tbody></table>
    <h4>By date</h4>
    <table class="usage-table"><thead><tr><th>Date</th><th>Total</th><th>Calls</th></tr></thead>
      <tbody>${dateRows}</tbody></table>`;
}

/* ── API keys modal (add / remove keys at runtime) ──────────────────────── */
let _keysModal = null;

async function openKeysModal() {
  if (_keysModal) return;
  let data;
  try { data = await api("GET", "/ui/api/keys"); }
  catch (e) { alert("Failed to load keys: " + e.message); return; }

  const overlay = el(`<div class="modal-overlay"><div class="modal keys-modal">
    <div class="modal-head"><h3>🔑 API keys</h3><button class="icon-btn modal-close" title="Close">✕</button></div>
    <p class="modal-sub">Saved on the server and overlaid on <code>.env</code>. Values are masked;
       saving overwrites, removing reveals any <code>.env</code> value on next restart.</p>
    <div class="keys-body"></div>
  </div></div>`);
  document.body.appendChild(overlay);
  _keysModal = overlay;

  const close = () => { overlay.remove(); _keysModal = null; };
  $(".modal-close", overlay).addEventListener("click", close);
  overlay.addEventListener("click", (e) => { if (e.target === overlay) close(); });
  renderKeys(overlay, data.keys);
}

function renderKeys(overlay, keys) {
  const body = $(".keys-body", overlay);
  body.innerHTML = "";
  const groups = { search: "Search & data sources", llm: "LLM providers" };
  for (const [cat, title] of Object.entries(groups)) {
    const inCat = keys.filter(k => k.category === cat);
    if (!inCat.length) continue;
    const group = el(`<div class="key-group"><h4>${esc(title)}</h4></div>`);
    inCat.forEach(k => group.appendChild(keyRow(overlay, k)));
    body.appendChild(group);
  }
}

function keyRow(overlay, k) {
  const status = k.configured
    ? `<span class="key-set">${esc(k.masked)}</span>`
    : `<span class="key-unset">not set</span>`;
  const tag = k.requires_rebuild ? ` <span class="key-tag">restarts brain</span>` : "";
  const row = el(`<div class="key-row">
    <div class="key-info">
      <div class="key-label">${esc(k.label)} ${status}${tag}</div>
      <div class="key-desc">${esc(k.description)}</div>
    </div>
    <div class="key-actions">
      <input class="key-input" type="${k.secret ? "password" : "text"}"
             placeholder="${k.configured ? "replace…" : "paste value…"}" />
      <button class="btn key-test">Test</button>
      ${k.category === "llm" ? `<button class="btn key-credits">Credits</button>` : ""}
      <button class="btn key-save">Save</button>
      <button class="btn key-del" ${k.configured ? "" : "disabled"}>Remove</button>
    </div>
    <div class="key-status"></div></div>`);

  const input = $(".key-input", row);
  const statusEl = $(".key-status", row);
  const refresh = async () => renderKeys(overlay, (await api("GET", "/ui/api/keys")).keys);

  $(".key-test", row).addEventListener("click", async () => {
    const btn = $(".key-test", row); btn.textContent = "Testing…"; btn.disabled = true;
    statusEl.className = "key-status"; statusEl.textContent = "";
    try {
      const value = input.value.trim() || undefined;  // test typed value, else stored
      const res = await api("POST", `/ui/api/keys/${k.name}/test`, { value });
      statusEl.className = "key-status " + (res.ok ? "ok" : "err");
      statusEl.textContent = (res.ok ? "✓ " : "✗ ") + res.message + (res.detail ? `  (${res.detail})` : "");
    } catch (e) {
      statusEl.className = "key-status err"; statusEl.textContent = "✗ " + e.message;
    }
    btn.textContent = "Test"; btn.disabled = false;
  });

  const creditsBtn = $(".key-credits", row);
  if (creditsBtn) creditsBtn.addEventListener("click", async () => {
    creditsBtn.textContent = "Checking…"; creditsBtn.disabled = true;
    statusEl.className = "key-status"; statusEl.textContent = "";
    try {
      const value = input.value.trim() || undefined;  // check typed value, else stored
      const res = await api("POST", `/ui/api/keys/${k.name}/credits`, { value });
      // supported=false → provider has no balance API (show a neutral hint + dashboard link).
      const cls = !res.supported ? "" : (res.ok ? "ok" : "err");
      const icon = !res.supported ? "ℹ " : (res.ok ? "✓ " : "✗ ");
      const isLink = res.detail && /^https?:\/\//.test(res.detail);
      statusEl.className = "key-status " + cls;
      statusEl.innerHTML = icon + esc(res.message) + (
        isLink ? ` — <a href="${esc(res.detail)}" target="_blank" rel="noopener">open dashboard</a>`
               : (res.detail ? `  (${esc(res.detail)})` : ""));
    } catch (e) {
      statusEl.className = "key-status err"; statusEl.textContent = "✗ " + e.message;
    }
    creditsBtn.textContent = "Credits"; creditsBtn.disabled = false;
  });

  $(".key-save", row).addEventListener("click", async () => {
    const value = input.value.trim();
    if (!value) { input.focus(); return; }
    const btn = $(".key-save", row); btn.textContent = "Saving…"; btn.disabled = true;
    try {
      await api("PUT", `/ui/api/keys/${k.name}`, { value });
      if (k.requires_rebuild) loadBrainBadge();
      await refresh();
    } catch (e) {
      alert("Failed: " + e.message); btn.textContent = "Save"; btn.disabled = false;
    }
  });
  $(".key-del", row).addEventListener("click", async () => {
    if (!confirm(`Remove ${k.label}?`)) return;
    try {
      await api("DELETE", `/ui/api/keys/${k.name}`);
      if (k.requires_rebuild) loadBrainBadge();
      await refresh();
    } catch (e) { alert("Failed: " + e.message); }
  });
  return row;
}

/* ── Coding-agent backend badge + picker (brain / Claude / Codex / Cursor) ─ */
let _codingOpts = null, _codingPop = null;

async function loadCodingBadge() {
  const badge = $("#coding-badge");
  let opts;
  try { opts = await api("GET", "/ui/api/coding/options"); }
  catch { badge.textContent = "code: ?"; return; }
  _codingOpts = opts;
  const label = (opts.labels && opts.labels[opts.current]) || opts.current;
  badge.classList.toggle("warn", opts.current !== "brain" && opts.available[opts.current] === false);
  badge.innerHTML =
    `<span class="b-led"></span><span class="b-prov">code</span>` +
    `<span class="b-model">${esc(opts.current)}</span><span class="caret">▾</span>`;
  badge.title = `Coding agent: ${label}\nClick to use the current brain, Claude Code, Codex, or Cursor.`;
}

async function openCodingPicker() {
  if (_codingPop) { _codingPop.remove(); _codingPop = null; return; }
  let opts = _codingOpts;
  try { if (!opts) opts = await api("GET", "/ui/api/coding/options"); }
  catch { return; }

  const pop = el(`<div class="brain-pop coding-pop">
    <h4>Coding agent</h4>
    <label>Backend (writes &amp; runs the code)</label>
    <select class="cp-backend"></select>
    <label class="cp-save-wrap"><input type="checkbox" class="cp-save-default" />
      <span>Set as default (remember after refresh)</span></label>
    <div class="hint cp-hint"></div>
    <div class="row">
      <button class="btn cp-cancel">Cancel</button>
      <button class="btn primary cp-apply">Apply</button>
    </div></div>`);
  document.body.appendChild(pop);
  _codingPop = pop;

  const sel = $(".cp-backend", pop), hint = $(".cp-hint", pop);
  const saveDefault = $(".cp-save-default", pop);
  saveDefault.checked = Boolean(opts.saved && opts.saved.coding_backend);
  opts.backends.forEach(b => {
    const o = document.createElement("option");
    o.value = b;
    const avail = opts.available[b] !== false;
    o.textContent = ((opts.labels && opts.labels[b]) || b) + (avail ? "" : " — not installed");
    o.disabled = !avail;
    if (b === opts.current) o.selected = true;
    sel.appendChild(o);
  });
  const upd = () => {
    const b = sel.value, avail = opts.available[b] !== false;
    hint.className = "hint cp-hint" + (avail ? "" : " warn");
    hint.textContent = b === "brain"
      ? "Uses the configured LLM through the normal tool-using loop."
      : (!avail
        ? `The ${b} CLI isn't installed or on PATH. Install it (and sign in) to use this backend.`
        : `Delegates the coding step to the ${b} CLI, which writes AND runs the code in the workspace.`);
  };
  upd();
  sel.addEventListener("change", upd);

  $(".cp-cancel", pop).addEventListener("click", () => { pop.remove(); _codingPop = null; });
  $(".cp-apply", pop).addEventListener("click", async () => {
    const backend = sel.value, btn = $(".cp-apply", pop);
    btn.textContent = "Applying…"; btn.disabled = true;
    try {
      await api("POST", "/ui/api/coding", { backend, save_default: saveDefault.checked });
      pop.remove(); _codingPop = null;
      await loadCodingBadge();
    } catch (e) {
      hint.className = "hint cp-hint warn";
      hint.textContent = "Failed: " + e.message;
      btn.textContent = "Apply"; btn.disabled = false;
    }
  });
}

/* ── Active brain badge ────────────────────────────────────────────────── */
function formatCreditsHtml(res) {
  const cls = !res.supported ? "" : (res.ok ? "ok" : "err");
  const icon = !res.supported ? "ℹ " : (res.ok ? "✓ " : "✗ ");
  const isLink = res.detail && /^https?:\/\//.test(res.detail);
  const extra = isLink
    ? ` — <a href="${esc(res.detail)}" target="_blank" rel="noopener">open dashboard</a>`
    : (res.detail ? ` (${esc(res.detail)})` : "");
  return { cls, html: icon + esc(res.message) + extra };
}

async function fetchBrainCredits(provider, apiKey) {
  if (!HOSTED_BRAIN_PROVIDERS.includes(provider)) return null;
  try {
    return await api("POST", "/ui/api/brain/credits", {
      provider,
      api_key: apiKey || undefined,
    });
  } catch {
    return null;
  }
}

function scheduleBrainBalanceRefresh() {
  clearInterval(_brainBalanceTimer);
  _brainBalanceTimer = null;
  const cfg = _brainCfg;
  if (
    cfg && cfg.needs_api_key && cfg.has_api_key
    && HOSTED_BRAIN_PROVIDERS.includes(cfg.provider)
  ) {
    _brainBalanceTimer = setInterval(() => { loadBrainBadge(); }, BRAIN_BALANCE_REFRESH_MS);
  }
}

async function loadBrainBadge() {
  const badge = $("#brain-badge");
  let cfg;
  try { cfg = await api("GET", "/config"); }
  catch { badge.textContent = "brain: unknown"; return; }
  _brainCfg = cfg;  // cache for the Explain header's model badges
  // Warn when the answers won't be "real": offline mock, or a hosted provider w/o a key.
  const warn = cfg.offline_mock || (cfg.needs_api_key && !cfg.has_api_key);
  badge.classList.toggle("warn", warn);
  let note = "";
  if (cfg.offline_mock) note = " · offline (placeholder answers)";
  else if (warn) note = " · ⚠ no API key";
  let balanceNote = "";
  let balanceTitle = "";
  if (cfg.needs_api_key && cfg.has_api_key) {
    const cr = await fetchBrainCredits(cfg.provider);
    if (cr) {
      if (cr.supported && cr.balance) {
        balanceNote = ` · ${cr.balance}`;
        balanceTitle = `\nBalance: ${cr.balance}`;
      } else if (cr.supported) {
        balanceTitle = `\n${cr.message}`;
      }
    }
  }
  badge.innerHTML =
    `<span class="b-led"></span><span class="b-prov">${esc(cfg.provider)}</span>` +
    `<span class="b-model">${esc(cfg.model || "")}${esc(note)}${esc(balanceNote)}</span>`;
  badge.innerHTML += `<span class="caret">▾</span>`;
  badge.title =
    `Brain: ${cfg.provider} (${cfg.model})${balanceTitle}\n` +
    `Embeddings: ${cfg.embedding_provider} (${cfg.embedding_model})\n` +
    `Memory backend: ${cfg.memory_backend}\n` +
    `Click to change the brain & model.`;
  scheduleBrainBalanceRefresh();
}

/* ── Brain picker popover (change provider + model from the GUI) ────────── */
let _brainPop = null;

async function openBrainPicker() {
  if (_brainPop) { _brainPop.remove(); _brainPop = null; return; }
  let opts;
  try { opts = await api("GET", "/ui/api/brain/options"); }
  catch { return; }

  const pop = el(`<div class="brain-pop">
    <h4>Change brain</h4>
    <label>Provider</label>
    <select class="bp-provider"></select>
    <label>Model</label>
    <select class="bp-model"></select>
    <input class="bp-model-custom" placeholder="…or type a custom model name" style="margin-top:6px" />
    <div class="bp-key-wrap"><label>API key (optional, this session only)</label>
      <input class="bp-key" type="password" placeholder="paste key to use without editing .env" /></div>
    <div class="bp-balance"></div>
    <label class="bp-save-wrap"><input type="checkbox" class="bp-save-default" />
      <span>Set as default (remember after refresh)</span></label>
    <div class="hint bp-hint"></div>
    <div class="row">
      <button class="btn bp-cancel">Cancel</button>
      <button class="btn primary bp-apply">Apply</button>
    </div></div>`);
  document.body.appendChild(pop);
  _brainPop = pop;

  const provSel = $(".bp-provider", pop), modelSel = $(".bp-model", pop);
  const keyWrap = $(".bp-key-wrap", pop), hint = $(".bp-hint", pop);
  const balanceEl = $(".bp-balance", pop);
  const keyInput = $(".bp-key", pop);
  let balanceTimer = null;

  const refreshBalance = async () => {
    const prov = provSel.value;
    if (!HOSTED_BRAIN_PROVIDERS.includes(prov)) {
      balanceEl.innerHTML = "";
      return;
    }
    balanceEl.className = "bp-balance loading";
    balanceEl.innerHTML = `<span class="spin">◌</span> Checking remaining balance…`;
    const cr = await fetchBrainCredits(prov, keyInput.value.trim());
    if (!cr) {
      balanceEl.className = "bp-balance err";
      balanceEl.textContent = "Could not check balance.";
      return;
    }
    const fmt = formatCreditsHtml(cr);
    balanceEl.className = "bp-balance " + fmt.cls;
    balanceEl.innerHTML = fmt.html;
  };

  const scheduleBalance = () => {
    clearTimeout(balanceTimer);
    balanceTimer = setTimeout(refreshBalance, 400);
  };
  for (const p of opts.providers) {
    const o = document.createElement("option");
    o.value = p; o.textContent = p; if (p === opts.current.provider) o.selected = true;
    provSel.appendChild(o);
  }
  const customInput = $(".bp-model-custom", pop);
  const fillModels = (prov) => {
    modelSel.innerHTML = "";
    const list = [...(opts.models[prov] || [])];
    // Make the currently-active model selectable even if it's a custom value not
    // in the suggested list (e.g. a hand-typed or local model name).
    if (prov === opts.current.provider && opts.current.model && !list.includes(opts.current.model))
      list.unshift(opts.current.model);
    list.forEach(m => {
      const o = document.createElement("option"); o.value = m; o.textContent = m;
      if (prov === opts.current.provider && m === opts.current.model) o.selected = true;
      modelSel.appendChild(o);
    });
    customInput.value = "";  // switching provider clears any half-typed custom name
    const needsKey = ["deepseek", "openai", "gemini", "glm", "opencode"].includes(prov);
    keyWrap.style.display = needsKey ? "block" : "none";
    hint.className = "hint bp-hint" + (prov === "mock" ? " warn" : "");
    hint.textContent = prov === "mock"
      ? "Mock is offline — answers are placeholders."
      : (prov === "local" ? "Local model via Ollama/OpenAI-compatible server."
        : "Hosted provider — needs an API key (paste above or set it in .env).");
    scheduleBalance();
  };
  fillModels(opts.current.provider);
  provSel.addEventListener("change", () => fillModels(provSel.value));
  keyInput.addEventListener("input", scheduleBalance);
  const saveDefault = $(".bp-save-default", pop);
  saveDefault.checked = Boolean(opts.saved && opts.saved.brain);
  // Picking a listed model must win over a stale custom entry — clear it so the
  // dropdown selection is what gets applied (fixes a half-typed name sticking).
  modelSel.addEventListener("change", () => { customInput.value = ""; });

  $(".bp-cancel", pop).addEventListener("click", () => { pop.remove(); _brainPop = null; });
  $(".bp-apply", pop).addEventListener("click", async () => {
    const provider = provSel.value;
    const model = ($(".bp-model-custom", pop).value.trim()) || modelSel.value;
    const api_key = $(".bp-key", pop).value.trim() || undefined;
    const applyBtn = $(".bp-apply", pop);
    applyBtn.textContent = "Applying…"; applyBtn.disabled = true;
    try {
      await api("POST", "/ui/api/brain", {
        provider, model, api_key, save_default: saveDefault.checked,
      });
      pop.remove(); _brainPop = null;
      await loadBrainBadge();
    } catch (e) {
      hint.className = "hint bp-hint warn";
      hint.textContent = "Failed: " + e.message;
      applyBtn.textContent = "Apply"; applyBtn.disabled = false;
    }
  });
}

// Populate the saved-tasks panel + brain/coding badges on load.
loadSessionList();
loadBrainBadge();
loadCodingBadge();
loadMemoryBadge();
