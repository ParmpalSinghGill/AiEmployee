/* AI Employee Console — dependency-free SPA.
 * Tabs = sessions; each session holds an initial run + follow-ups. We poll each
 * running task and render a live n8n-style flow, logs, and the final result. */

const TERMINAL = new Set(["completed", "failed", "awaiting_approval", "waiting_for_user", "cancelled"]);
// Live statuses keep polling. "cooling" means a provider rate limit made the
// agent sleep before its automatic retry — still live, just visibly paused.
const LIVE = new Set(["running", "pending", "cooling"]);
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
    const live = s.runs.filter(r => LIVE.has(r.status));
    const status = live.some(r => r.status === "cooling")
      ? "cooling" : (live.length ? "running" : (last.status || "completed"));
    const item = el(`<div class="sess-item ${state.sessions.has(s.id) ? "open" : ""}">
      <span class="dot ${status}"></span>
      <div class="meta"><div class="t">${esc(s.title)}</div>
        <div class="sub">${s.runs.length} run${s.runs.length === 1 ? "" : "s"} · ${fmtDate(s.created_at)}${s.saved === false ? " · guest (not saved)" : ""}</div></div>
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

/* Re-import runs still present in the server log (lost to a restart/crash). */
async function recoverRuns() {
  if (!confirm(
    "Recover runs from the server log into Saved tasks?\n\n" +
    "Use this after a restart (or crash) to reopen a run that wasn't saved — " +
    "the console rebuilds it from the structured log so you can inspect its " +
    "Explain/Logs/Result views again. No task is re-executed."
  )) return;
  const btn = $("#recover-runs");
  btn.disabled = true;
  const old = btn.textContent;
  btn.textContent = "…";
  try {
    const res = await api("POST", "/ui/api/sessions/recover");
    alert(res.count
      ? `Recovered ${res.count} run(s) from the log:\n` + res.imported.map(i => `• ${i.title}`).join("\n")
      : `Nothing new to recover (${res.note || "all log runs are already saved"}).`);
    await loadSessionList();
  } catch (e) {
    alert("Recover failed: " + e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = old;
  }
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

function tokenBudget() {
  const el = $("#token-budget");
  const v = el ? parseInt(el.value, 10) : 0;
  return Number.isFinite(v) && v > 0 ? v : null;
}

async function newTask(goal, mode) {
  const body = {
    goal,
    mode: mode || "task",
    auto_approve: isAutoApprove(),
    token_budget: tokenBudget(),
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
  const live = s.runs.filter(r => LIVE.has(r.status));
  if (live.some(r => r.status === "cooling")) return "cooling";
  if (live.length) return "running";
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
  // Expand all / collapse all every nested <details> (LLM calls, tools, verification)
  // and remember the choice so a re-render (live polling) keeps it.
  explain.querySelectorAll("[data-explain-expand]").forEach(btn => {
    btn.addEventListener("click", () => {
      const open = btn.dataset.explainExpand === "all";
      let m = state.explainOpen.get(session.activeRun);
      if (!m) { m = new Map(); state.explainOpen.set(session.activeRun, m); }
      explain.querySelectorAll("details[data-xk]").forEach(d => {
        d.open = open;
        m.set(d.getAttribute("data-xk"), open);
      });
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
  // Parallel execution: how the plan was scheduled, and which steps are in flight.
  let schedule = null;
  const steps = new Map();
  const ensure = (n, agent) => {
    if (!steps.has(n)) steps.set(n, { step: n, agent: agent || "?", tools: new Map(), verified: null, confidence: null, retries: 0, tokens: 0, running: false, started: false });
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
      case "brain.response": {
        // Attribute the agent's own LLM tokens (purpose "agent:<name>") to its step,
        // so the flow can show how much each agent's reasoning cost.
        const purpose = f.purpose || "";
        if (f.step != null && purpose.startsWith("agent:")) {
          const s = ensure(f.step);
          s.tokens = (s.tokens || 0) + (((f.usage || {}).total_tokens) || 0);
        }
        break;
      }
      case "workflow.retry": if (f.step != null) ensure(f.step).retries = f.attempt || 0; break;
      // Parallel mode: these are the only events that say a step is *in flight*
      // right now — a step with no tool calls yet is otherwise invisible.
      case "workflow.parallel_schedule":
        schedule = { steps: f.steps, waves: f.waves, maxParallel: f.max_parallelism, limit: f.limit };
        break;
      case "workflow.parallel_start": {
        const s = ensure(f.step, f.agent); s.running = true; s.started = true; break;
      }
      case "workflow.parallel_done": {
        const s = ensure(f.step); s.running = false; break;
      }
      case "workflow.node": if (f.node === "synthesize") synth = true; if (f.node === "reflect") reflect = true; break;
      case "agent.needs_approval": approval = { ...(approval || {}), operation: f.op, tool: f.tool ?? approval?.tool, args: f.args ?? approval?.args }; break;
    }
  }
  // Backfill agent names the event stream didn't carry (see buildTrace).
  for (const r of (task.result?.steps || [])) {
    const s = steps.get(r.id);
    if (s && (!s.agent || s.agent === "?") && r.agent) s.agent = r.agent;
  }
  return { plan, steps: [...steps.values()].sort((a, b) => a.step - b.step), synth, reflect, approval, schedule };
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

  const liveSteps = TERMINAL.has(task.status) ? [] : fl.steps.filter(s => s.running);
  const liveStat = liveSteps.length
    ? `<div class="stat"><div class="v">${liveSteps.length}</div><div class="k">Running now</div></div>`
    : "";

  const totalRetries = fl.steps.reduce((n, s) => n + (s.retries || 0), 0);
  const totalTokens = fl.steps.reduce((n, s) => n + (s.tokens || 0), 0);

  // Provider rate limit: the agent is not stuck, it is sleeping out the
  // provider's cooldown and will retry automatically. Show when it resumes.
  const cool = task.cooling || null;
  let coolingBar = "";
  if (cool) {
    const until = cool.until ? new Date(cool.until) : null;
    const valid = until && !Number.isNaN(until.getTime());
    const secs = valid ? Math.max(0, Math.round((until.getTime() - Date.now()) / 1000)) : null;
    const when = valid ? until.toLocaleTimeString() : "shortly";
    const who = [cool.provider, cool.agent, cool.step != null ? `step ${cool.step}` : ""]
      .filter(Boolean).join(" · ");
    coolingBar = `<div class="cooling-bar">
      <span class="cooling-spinner"></span>
      <div class="cooling-text">
        <b>Cooling down — rate limit</b>
        <span class="muted">${who ? esc(who) + " · " : ""}retrying at ${esc(when)}${secs != null ? ` (${secs}s)` : ""}</span>
      </div>
    </div>`;
  }

  const summary = `<div class="flow-summary">
    <div class="stat"><div class="v">${fl.steps.length || (fl.plan?.steps || 0)}</div><div class="k">Steps</div></div>
    <div class="stat"><div class="v">${agents.size}</div><div class="k">Agents</div></div>
    <div class="stat" title="Verifier-driven re-runs across all steps"><div class="v">${totalRetries}</div><div class="k">Retries</div></div>
    ${liveStat}
    <div class="stat" title="Tokens spent by the agents' reasoning (excludes planner/verifier/synthesis)"><div class="v">${fmtTok(totalTokens)}</div><div class="k">Tokens</div></div>
    <div class="stat"><div class="v">${tools.size}</div><div class="k">Tools used</div></div>
    <div class="stat"><div class="v">${esc(statusLabel(task.status))}</div><div class="k">Status</div></div>
  </div>`;

  // Name the agents working concurrently — the whole point of parallel mode is
  // seeing that several are live at the same moment.
  let parallelBar = "";
  if (liveSteps.length > 1) {
    const who = liveSteps
      .map(s => `<span class="pchip"><span class="pulse"></span>${esc(s.agent)} <span class="muted">· step ${s.step}</span></span>`)
      .join("");
    parallelBar = `<div class="parallel-bar">
      <span class="pbar-title">⚡ ${liveSteps.length} agents running in parallel</span>
      <span class="pbar-chips">${who}</span></div>`;
  } else if (fl.schedule && fl.schedule.maxParallel > 1 && !TERMINAL.has(task.status)) {
    parallelBar = `<div class="parallel-bar">
      <span class="pbar-title">⚡ Parallel plan</span>
      <span class="pbar-chips"><span class="muted">up to ${fl.schedule.maxParallel} step(s) at once`
      + `${fl.schedule.limit ? `, capped at ${fl.schedule.limit}` : ""}</span></span></div>`;
  }

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
    // In parallel mode several agents are live at once — mark which, so the flow
    // shows concurrency rather than looking like a sequential list.
    const coolingHere = cool && task.status === "cooling" && cool.step != null
      && Number(cool.step) === Number(s.step);
    const live = coolingHere
      ? `<div class="badge cooling"><span class="pulse"></span>cooling down</div>`
      : (s.running && !TERMINAL.has(task.status)
        ? `<div class="badge running"><span class="pulse"></span>running now</div>` : "");
    const metaBits = [];
    if (s.retries) metaBits.push(`retried ×${s.retries}`);
    if (s.tokens) metaBits.push(`🧮 ${fmtTok(s.tokens)} tok`);
    const meta = metaBits.length ? `<div class="node-sub tok">${metaBits.join(" · ")}</div>` : "";
    nodes.push(nodeHtml("step" + (s.running ? " is-running" : ""), `Step ${s.step} · agent`, "▣",
      s.agent, live + chips + badge + meta, "comp:step:" + s.step));
  }
  // Always offer the approve control in-flow while paused — fall back to the
  // result payload (or a generic card) when the approval events were trimmed.
  const pausedNow = task.status === "awaiting_approval" || task.status === "waiting_for_user";
  const apprInfo = pausedNow
    ? (fl.approval || (task.result && task.result.needs_approval) || { operation: "sensitive operation" })
    : null;
  if (apprInfo) nodes.push(approvalNode(apprInfo, task));
  if (fl.synth) nodes.push(nodeHtml("synthesize", "CEO", "★", "Synthesize", `<div class="node-sub">integrate step outputs</div>`, "comp:synthesize"));
  if (fl.reflect) nodes.push(nodeHtml("reflect", "Memory", "↻", "Reflect", `<div class="node-sub">store outcome + learn skill</div>`, "comp:reflect"));
  // The deliverable itself is produced by synthesis, so Result jumps there.
  if (task.status === "completed") nodes.push(nodeHtml("result", "Output", "✓", "Result", `<div class="node-sub">final deliverable ready</div>`, fl.synth ? "comp:synthesize" : ""));

  // Wire the in-flow approve button once it's in the DOM (mirrors the Result tab).
  if (apprInfo && session) {
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

  return coolingBar + summary + parallelBar + `<div class="flow">` + nodes.join(conn()) + `</div>`;
}

function statusLabel(s) {
  return { running: "Running", pending: "Starting", cooling: "Cooling down", completed: "Done", failed: "Failed", awaiting_approval: "Needs approval", waiting_for_user: "Waiting for you", cancelled: "Cancelled" }[s] || s;
}

/* Compact token count: 12400 → "12.4k", 1_500_000 → "1.5M". */
function fmtTok(n) {
  if (n == null || !Number.isFinite(n)) return "0";
  if (n >= 1e6) return (n / 1e6).toFixed(1).replace(/\.0$/, "") + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(1).replace(/\.0$/, "") + "k";
  return String(n);
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
    if (!steps.has(n)) { const c = { kind: "step", step: n, agent: "?", title: "Step " + n, icon: "▣", attempts: [], retries: 0 }; steps.set(n, c); comps.push(c); }
    return steps.get(n);
  };
  // One attempt = one agent.run() for this step. retry 0 is the initial attempt;
  // retry N (1,2,…) is the Nth verifier-driven retry. Each keeps its own input,
  // LLM calls, tools and verification so the Explain view can show them separately.
  const mkAttempt = (s, retry) => {
    let a = s.attempts.find(x => x.retry === retry);
    if (!a) {
      a = { retry, input: null, answer: null, llm: [], tools: [], timeline: [], verify: { input: null, verdict: null, llm: [], timeline: [] } };
      s.attempts.push(a);
      s.attempts.sort((p, q) => p.retry - q.retry);
    }
    return a;
  };
  const mkSynth = () => synth || (comps.push(synth = { kind: "synthesize", title: "Synthesize", icon: "★", agent: "ceo", input: null, output: null, llm: [], timeline: [] }), synth);
  const mkReflect = () => reflect || (comps.push(reflect = { kind: "reflect", title: "Reflect", icon: "↻", llm: [], timeline: [], notes: "Persist the outcome to long-term memory and distill a reusable skill." }), reflect);

  const traceTarget = (f) => {
    if (f.step != null) {
      const s = mkStep(f.step);
      const a = mkAttempt(s, Number(f.retry) || 0);
      return f.phase === "verify" ? { box: a.verify, comp: s } : { box: a, comp: s };
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
        if (f.step != null) { const s = mkStep(f.step); s.agent = f.agent || s.agent; mkAttempt(s, Number(f.retry) || 0).input = { system: f.system, memories: f.memories || [], task: f.task }; }
        else if (f.node === "synthesize") mkSynth().input = { system: f.system, memories: f.memories || [], task: f.task };
        break;
      case "agent.answer":
        if (f.step != null) { mkAttempt(mkStep(f.step), Number(f.retry) || 0).answer = { output: f.output, note: f.note }; }
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
          const a = mkAttempt(mkStep(f.step), Number(f.retry) || 0);
          const tool = { name: f.tool, args: f.args, status: "running" };
          a.tools.push(tool);
          a.timeline.push({ type: "tool", ref: tool });
        }
        break;
      case "tool.result": if (f.step != null) { const ts = mkAttempt(mkStep(f.step), Number(f.retry) || 0).tools; for (let i = ts.length - 1; i >= 0; i--) if (ts[i].name === f.tool && ts[i].status === "running") { Object.assign(ts[i], { status: f.ok ? "ok" : "fail", output: f.output, error: f.error, latency: f.latency_ms }); break; } } break;
      case "verifier.input": if (f.step != null) mkAttempt(mkStep(f.step), Number(f.retry) || 0).verify.input = { task: f.task, criteria: f.criteria, output: f.output }; break;
      case "verifier.verdict": if (f.step != null) mkAttempt(mkStep(f.step), Number(f.retry) || 0).verify.verdict = { passed: f.passed, confidence: f.confidence, issues: f.issues || [], suggestion: f.suggestion }; break;
      case "workflow.node": if (f.node === "synthesize") mkSynth(); if (f.node === "reflect") mkReflect(); break;
      case "approval.granted": comps.push({ kind: "approval", idx: comps.length, title: "Approved", icon: "✅", op: f.op, tool: f.tool, args: f.args }); break;
    }
  }
  // Some steps never emit an event carrying the agent name (it comes from the
  // bound log context). The finished result records it per step, so backfill
  // rather than leaving the card labelled "?".
  for (const r of (task.result?.steps || [])) {
    const c = steps.get(r.id);
    if (!c) continue;
    if ((!c.agent || c.agent === "?") && r.agent) c.agent = r.agent;
    c.retries = Math.max(c.retries || 0, r.retries || 0);
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
    const attempts = c.attempts || [];
    if (!attempts.length) body += _text("…");
    attempts.forEach(a => {
      const ak = ck + ":att:" + a.retry;
      const label = a.retry === 0
        ? `<span class="x-att-tag initial">Attempt 1</span>`
        : `<span class="x-att-tag retry">↺ Attempt ${a.retry + 1} — retry ${a.retry}</span>`;
      let ab = "";
      if (a.input) {
        ab += _field(
          a.retry === 0 ? "Input (task given to agent)" : "Input (retry — continues from previous attempt)",
          _text(a.input.task));
        if (a.input.system) ab += _field("Agent role (system prompt)", _text(a.input.system));
        if (a.input.memories && a.input.memories.length) ab += _field("Recalled knowledge", `<ul class="x-list">${a.input.memories.map(m => `<li>${linkify(m)}</li>`).join("")}</ul>`);
      }
      if (a.llm.length || a.tools.length) {
        ab += `<div class="x-label">Reasoning & actions</div><div class="x-proc">`;
        ab += explainOrder === "timeline"
          ? procBlocksTimeline(a.timeline, ak)
          : procBlocksGrouped(a.llm, a.tools, ak);
        ab += `</div>`;
      }
      if (a.answer) ab += _field("Output (agent's answer)", _text(a.answer.output) + (a.answer.note ? `<div class="x-muted">⚠ ${linkify(a.answer.note)}</div>` : ""));
      if (a.verify.input || a.verify.verdict || a.verify.llm.length) ab += verifyBlock(a.verify, ak);
      body += `<details class="x-attempt${a.retry > 0 ? " retry" : ""}" data-xk="${esc(ak)}" open><summary>${label}</summary><div class="x-attempt-body">${ab || _text("…")}</div></details>`;
    });
  } else if (c.kind === "synthesize") {
    if (c.input) body += _field("Input (team's step outputs)", _text(c.input.task));
    body += explainOrder === "timeline" ? procBlocksTimeline(c.timeline, ck) : procBlocksGrouped(c.llm, [], ck);
    if (c.output) body += _field("Output (final deliverable)", _text(c.output.output));
  } else if (c.kind === "reflect") {
    body += _text(c.notes);
    body += explainOrder === "timeline" ? procBlocksTimeline(c.timeline, ck) : procBlocksGrouped(c.llm, [], ck);
  }

  let sub = "";
  if (c.kind === "step") {
    const retried = Math.max(c.retries || 0, (c.attempts?.length || 1) - 1);
    sub = `agent: ${esc(c.agent)}` + (retried > 0
      ? ` <span class="x-retry-badge" title="This step was retried by the verifier — ${retried + 1} attempts">↺ retried ${retried}×</span>`
      : "");
  }
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
    <div class="x-explain-toolbar-label" style="margin-left:auto">Details</div>
    <div class="x-seg">
      <button type="button" class="x-seg-btn" data-explain-expand="all">⤢ Expand all</button>
      <button type="button" class="x-seg-btn" data-explain-expand="none">⤡ Collapse all</button>
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

function lastVerdict(task, stepId) {
  // The verifier's final verdict for a step (its last attempt) — carries the
  // issues + suggestion when the step did not pass review.
  let verdict = null;
  for (const e of (task.events || [])) {
    const f = e.fields || {};
    if (e.event === "verifier.verdict" && f.step === stepId) verdict = f;
  }
  return verdict || {};
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
  if (task.status === "failed" || task.status === "cancelled") {
    const stopped = task.status === "cancelled";
    const label = stopped ? "Task stopped" : "Task failed";
    const color = stopped ? "var(--muted)" : "var(--red)";
    const bg = stopped ? "rgba(140,140,150,.10)" : "rgba(239,94,107,.12)";
    const detail = linkify(task.error || (stopped ? "stopped" : "unknown error"));
    // Say how much survived, so "Resume" is an obvious continue rather than a redo.
    const done = task.completed_steps || 0;
    const planned = task.planned_steps || 0;
    const progress = planned
      ? ` — ${done} of ${planned} step(s) completed and saved.`
      : "";
    const action = task.can_resume
      ? `<div style="margin-top:8px">
           <button class="btn primary" id="resume-banner-${task.id}">▶ Resume from last checkpoint</button>
         </div>`
      : "";
    if (task.can_resume) {
      setTimeout(() => {
        const btn = document.getElementById(`resume-banner-${task.id}`);
        if (btn) btn.addEventListener("click", () => resumeTask(session.id, task.id));
      }, 0);
    }
    return `<div class="approval-banner" style="border-color:${color};color:${color};background:${bg}">`
      + `${label}: ${detail}${progress}${action}</div>`;
  }
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

  // At-a-glance run stats: how many steps, how many retries, verification split.
  const steps = r.steps || [];
  const totalRetries = steps.reduce((n, s) => n + (s.retries || 0), 0);
  const unverified = steps.filter(s => !s.verified);
  const verifiedCount = steps.length - unverified.length;
  if (steps.length) {
    html += `<div class="result-stats">
      <span class="stat" title="Steps in the plan">📋 <b>${steps.length}</b> step${steps.length === 1 ? "" : "s"}</span>
      <span class="stat" title="Verifier-driven re-runs across all steps">↺ <b>${totalRetries}</b> retr${totalRetries === 1 ? "y" : "ies"}</span>
      <span class="stat ${unverified.length ? "warn" : "ok"}" title="Steps that passed review">
        ✓ <b>${verifiedCount}</b> verified${unverified.length ? ` · <b>${unverified.length}</b> needs work` : ""}</span>
    </div>`;
  }

  // Needs-work explainer: name the unverified steps and what the reviewer found.
  if (unverified.length) {
    const items = unverified.map(st => {
      const v = lastVerdict(task, st.id);
      const issues = (v.issues || []).filter(Boolean);
      const sug = v.suggestion;
      return `<li><b>Step ${esc(st.id)} · ${esc(st.agent || "?")}</b> — ${esc(st.description || "")}
        ${issues.length ? `<ul class="x-list">${issues.map(i => `<li>${linkify(i)}</li>`).join("")}</ul>` : ""}
        ${sug ? `<div class="x-muted">Suggested fix: ${linkify(sug)}</div>` : ""}</li>`;
    }).join("");
    html += `<div class="needs-work-banner">
      <div class="nw-head">⚠ ${unverified.length} step${unverified.length === 1 ? "" : "s"} not verified — needs work</div>
      <ul class="nw-list">${items}</ul>
    </div>`;
  }

  html += `<div id="media-gallery-${esc(task.id)}" class="media-gallery"></div>`;
  html += `<div id="deliverables-${esc(task.id)}"></div>`;

  if (steps.length) {
    html += `<div class="steps-list"><h3>Step outputs</h3>` + steps.map(st => {
      const retried = st.retries || 0;
      const badge = st.verified
        ? `<span class="x-mark ok">✓ verified</span>`
        : `<span class="x-mark fail">⚠ needs work</span>`;
      const retryBadge = retried > 0 ? `<span class="x-retry-badge" title="${retried} verifier-driven re-run(s)">↺ retried ${retried}×</span>` : "";
      return `<div class="step-card"><h4>Step ${esc(st.id)} · ${esc(st.agent)} ${badge} ${retryBadge}</h4>
        <div class="out md">${renderMarkdown((st.output || "").slice(0, 1200))}</div></div>`;
    }).join("") + `</div>`;
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
$("#recover-runs").addEventListener("click", recoverRuns);

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
    <button class="btn mp-skills" style="width:100%;margin-top:8px">
      💡 Manage learned skills…
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
  $(".mp-skills", pop).addEventListener("click", () => { close(); openSkillsModal(); });
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

/* ── Parallel execution toggle ──────────────────────────────────────────── */
let _parallelSettings = null;

async function loadParallelBadge() {
  const badge = $("#parallel-badge");
  if (!badge) return;
  let st;
  try { st = await api("GET", "/ui/api/parallel/settings"); }
  catch { badge.textContent = "parallel: ?"; return; }
  _parallelSettings = st;
  renderParallelBadge(st);
}

function renderParallelBadge(st) {
  const badge = $("#parallel-badge");
  const on = !!st.parallel_steps;
  // "warn" is the dimmed/off styling the memory badge uses for its off state.
  badge.classList.toggle("warn", !on);
  badge.innerHTML =
    `<span class="b-led"></span><span>⚡ Parallel: ${on ? "On" : "Off"}</span>`;
  badge.title = on
    ? `Independent plan steps run at the same time (up to ${st.max_parallel_steps}). `
      + "Steps that depend on others wait, then start the moment those finish. Click to turn off."
    : "Steps run one at a time. Click to run independent steps in parallel.";
}

async function toggleParallel() {
  const badge = $("#parallel-badge");
  const next = !(_parallelSettings && _parallelSettings.parallel_steps);
  badge.disabled = true;
  try {
    const st = await api("POST", "/ui/api/parallel/settings", {
      parallel_steps: next,
      save_default: true,   // persist, so it survives a restart like the other badges
    });
    _parallelSettings = st;
    renderParallelBadge(st);
  } catch (e) {
    badge.title = "Failed to change: " + e.message;
  } finally {
    badge.disabled = false;
  }
}

$("#parallel-badge").addEventListener("click", toggleParallel);

/* ── 🎚 Agent limits modal (iterations & loop extensions per agent) ───────── */
let _limitsModal = null;

async function loadLimitsBadge() {
  const badge = $("#limits-badge");
  if (!badge) return;
  try {
    const st = await api("GET", "/ui/api/agents/limits");
    const overrides = [st.max_iterations_by_name, st.loop_extensions_by_name]
      .filter(v => (v || "").trim()).length;
    badge.innerHTML = `<span class="b-led"></span><span>🎚 Limits${overrides ? ` (${overrides})` : ""}</span>`;
    badge.title = (st.max_iterations
      ? `Every agent gets ${st.max_iterations} tool rounds + ${st.loop_extensions_max} loop window(s).`
      : "Agents use their built-in iteration budgets.")
      + (overrides ? ` ${overrides} per-agent override(s).` : "")
      + " Click to edit.";
  } catch { badge.textContent = "limits: ?"; }
}

async function openLimitsModal() {
  if (_limitsModal) return;
  const overlay = el(`<div class="modal-overlay"><div class="modal limits-modal">
    <div class="modal-head"><h3>🎚 Agent limits</h3><button class="icon-btn modal-close" title="Close">✕</button></div>
    <p class="modal-sub">How many tool rounds (<b>iterations</b>) each agent may run before its
      loop check fires, and how many extra <b>loop windows</b> it may be granted when the check
      judges it still progressing. A per-agent override wins over the global value, which wins
      over each agent's built-in default. Format: <code>coding=24,research=16</code></p>
    <div class="limits-form">
      <label>Global max iterations
        <input type="number" class="l-max" min="0" step="1" />
        <span class="muted">0 = use each agent's built-in default</span></label>
      <label>Per-agent iterations
        <input type="text" class="l-max-by" placeholder="coding=24,research=16" /></label>
      <label>Loop extensions (global)
        <input type="number" class="l-loops" min="0" step="1" /></label>
      <label>Per-agent loops
        <input type="text" class="l-loops-by" placeholder="coding=4" /></label>
      <label class="limits-check"><input type="checkbox" class="l-loopcheck" />
        <span>Enable loop check (grant extra windows while the agent is still progressing)</span></label>
      <div class="limits-sep">Explain view — how much input/output text is kept (0 = everything)</div>
      <label>Task &amp; LLM text — max chars
        <input type="number" class="l-explain" min="0" step="1000" />
        <span class="muted">0 = keep everything (no “… more chars”)</span></label>
      <label>Tool text — max chars
        <input type="number" class="l-explain-tool" min="0" step="1000" />
        <span class="muted">0 = keep all tool args/output</span></label>
    </div>
    <div class="limits-body"></div>
    <div class="row limits-actions">
      <span class="key-status l-status"></span>
      <button class="btn l-cancel">Close</button>
      <button class="btn primary l-save">Save</button>
    </div>
  </div></div>`);
  document.body.appendChild(overlay);
  _limitsModal = overlay;
  const close = () => { overlay.remove(); _limitsModal = null; };
  $(".modal-close", overlay).addEventListener("click", close);
  overlay.addEventListener("click", e => { if (e.target === overlay) close(); });
  $(".l-cancel", overlay).addEventListener("click", close);
  $(".l-save", overlay).addEventListener("click", () => saveLimits(overlay));
  await renderLimits(overlay);
}

async function renderLimits(overlay) {
  const body = $(".limits-body", overlay);
  let st;
  try { st = await api("GET", "/ui/api/agents/limits"); }
  catch (e) { body.innerHTML = `<div class="key-status err">✗ ${esc(e.message)}</div>`; return; }
  $(".l-max", overlay).value = st.max_iterations;
  $(".l-max-by", overlay).value = st.max_iterations_by_name || "";
  $(".l-loops", overlay).value = st.loop_extensions_max;
  $(".l-loops-by", overlay).value = st.loop_extensions_by_name || "";
  $(".l-loopcheck", overlay).checked = !!st.loop_check_enabled;
  $(".l-explain", overlay).value = st.explain_max_chars ?? 0;
  $(".l-explain-tool", overlay).value = st.explain_tool_max_chars ?? 0;
  renderLimitsTable(overlay, st);
}

function renderLimitsTable(overlay, st) {
  const rows = (st.agents || []).map(a => `<tr>
    <td>${esc(a.name)}</td>
    <td>${esc(a.role || "")}</td>
    <td>${a.iterations}</td>
    <td>${a.loops}</td>
    <td>${a.max_rounds}</td></tr>`).join("");
  $(".limits-body", overlay).innerHTML = `
    <div class="key-desc" style="margin:8px 0 6px">Effective budgets — total rounds = iterations × (1 + loops).</div>
    <table class="res-table">
      <thead><tr><th>Agent</th><th>Role</th><th>Iterations</th><th>Loops</th><th>Max rounds</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

async function saveLimits(overlay) {
  const st = $(".l-status", overlay);
  const btn = $(".l-save", overlay);
  btn.disabled = true; st.className = "key-status"; st.textContent = "saving…";
  try {
    const res = await api("POST", "/ui/api/agents/limits", {
      max_iterations: parseInt($(".l-max", overlay).value || "0", 10) || 0,
      max_iterations_by_name: $(".l-max-by", overlay).value,
      loop_extensions_max: parseInt($(".l-loops", overlay).value || "0", 10) || 0,
      loop_extensions_by_name: $(".l-loops-by", overlay).value,
      loop_check_enabled: $(".l-loopcheck", overlay).checked,
      explain_max_chars: parseInt($(".l-explain", overlay).value || "0", 10) || 0,
      explain_tool_max_chars: parseInt($(".l-explain-tool", overlay).value || "0", 10) || 0,
      save_default: true,
    });
    st.className = "key-status ok"; st.textContent = "✓ saved";
    $(".l-max-by", overlay).value = res.max_iterations_by_name || "";
    $(".l-loops-by", overlay).value = res.loop_extensions_by_name || "";
    renderLimitsTable(overlay, res);
    await loadLimitsBadge();
  } catch (e) {
    st.className = "key-status err"; st.textContent = "✗ " + e.message;
  } finally { btn.disabled = false; }
}


$("#clear-memories").addEventListener("click", clearAllMemories);
$("#restore-memories").addEventListener("click", restoreArchivedMemories);
$("#memory-badge").addEventListener("click", openMemoryPicker);
$("#skills-badge").addEventListener("click", openSkillsModal);
$("#brain-badge").addEventListener("click", openBrainPicker);
$("#coding-badge").addEventListener("click", openCodingPicker);
$("#keys-badge").addEventListener("click", openKeysModal);
$("#mcp-badge").addEventListener("click", openMcpModal);
$("#schedules-badge").addEventListener("click", openSchedulesModal);
$("#autonomy-badge").addEventListener("click", openAutonomyModal);
$("#usage-badge").addEventListener("click", openUsageModal);
$("#resources-badge").addEventListener("click", openResourcesModal);
$("#limits-badge").addEventListener("click", openLimitsModal);

/* ── 💡 Learned skills (list / create / edit / delete) ──────────────────── */
let _skillsModal = null;

function skillWhen(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? "" : d.toLocaleString();
}

async function openSkillsModal() {
  if (_skillsModal) return;
  const overlay = el(`<div class="modal-overlay"><div class="modal skills-modal">
    <div class="modal-head"><h3>💡 Learned skills</h3><button class="icon-btn modal-close" title="Close">✕</button></div>
    <p class="modal-sub">Reusable workflows the employee learned after successful runs.
      Future tasks recall them during planning, so edit a bad skill here or delete it to stop the recall.</p>
    <div class="skill-toolbar">
      <button class="btn primary skill-new">＋ New skill</button>
      <button class="btn skill-refresh">⟳ Refresh</button>
      <span class="key-status skill-status"></span>
    </div>
    <div class="skills-body"></div>
  </div></div>`);
  document.body.appendChild(overlay);
  _skillsModal = overlay;
  const close = () => { overlay.remove(); _skillsModal = null; };
  $(".modal-close", overlay).addEventListener("click", close);
  overlay.addEventListener("click", e => { if (e.target === overlay) close(); });
  $(".skill-new", overlay).addEventListener("click", () => showSkillEditor(overlay, null));
  $(".skill-refresh", overlay).addEventListener("click", () => loadSkills(overlay));
  await loadSkills(overlay);
}

async function loadSkills(overlay) {
  const body = $(".skills-body", overlay);
  const status = $(".skill-status", overlay);
  body.innerHTML = `<div class="x-muted" style="padding:14px 0"><span class="spin">◌</span> Loading…</div>`;
  let data;
  try { data = await api("GET", "/ui/api/skills"); }
  catch (e) { body.innerHTML = `<div class="key-status err">✗ ${esc(e.message)}</div>`; return; }

  if (!data.skills.length) {
    body.innerHTML = `<p class="muted">No learned skills yet. They appear here after a successful run
      (the Reflect step distils one), or you can create one with <b>＋ New skill</b>.</p>`;
    return;
  }

  body.innerHTML = data.skills.map(s => {
    const tags = (s.tags || []).map(t => `<span class="skill-tag">${esc(t)}</span>`).join("");
    const meta = [
      skillWhen(s.created_at),
      s.steps.length ? `${s.steps.length} step${s.steps.length === 1 ? "" : "s"}` : "",
    ].filter(Boolean).join(" · ");
    const steps = (s.steps || []).length
      ? `<ol class="skill-steps">${s.steps.map(x => `<li>${esc(x)}</li>`).join("")}</ol>`
      : "";
    return `<div class="skill-card" data-skill-id="${esc(s.id)}">
      <div class="skill-card-head">
        <div class="skill-title">
          <div class="skill-name">${esc(s.name)}</div>
          <div class="skill-meta">${esc(meta)}</div>
          ${tags ? `<div class="skill-tags">${tags}</div>` : ""}
        </div>
        <div class="skill-actions">
          <button class="btn skill-edit" title="Edit this skill">✎ Edit</button>
          <button class="btn skill-del" title="Delete this skill">🗑 Delete</button>
        </div>
      </div>
      ${s.description ? `<div class="skill-desc">${esc(s.description)}</div>` : ""}
      ${steps}
    </div>`;
  }).join("");

  body.querySelectorAll(".skill-card").forEach(card => {
    const skill = data.skills.find(s => s.id === card.dataset.skillId);
    if (!skill) return;
    $(".skill-edit", card).addEventListener("click", () => showSkillEditor(overlay, skill));
    $(".skill-del", card).addEventListener("click", async () => {
      if (!confirm(`Delete the skill “${skill.name}”?\n\nFuture tasks will no longer recall it.`)) return;
      if (status) { status.className = "key-status"; status.textContent = "deleting…"; }
      try {
        await api("DELETE", "/ui/api/skills/" + encodeURIComponent(skill.id));
        if (status) { status.className = "key-status ok"; status.textContent = "✓ deleted"; }
        await loadSkills(overlay);
      } catch (e) {
        if (status) { status.className = "key-status err"; status.textContent = "✗ " + e.message; }
      }
    });
  });
}

function showSkillEditor(overlay, skill) {
  const body = $(".skills-body", overlay);
  body.innerHTML = `<form class="skill-form">
    <label>Name
      <input class="skill-f-name" maxlength="200" required value="${esc(skill?.name || "")}"
             placeholder="e.g. Deploy a Django app" />
    </label>
    <label>When to use / description
      <textarea class="skill-f-desc" rows="3"
        placeholder="What problem does this skill solve?">${esc(skill?.description || "")}</textarea>
    </label>
    <label>Steps — one per line
      <textarea class="skill-f-steps" rows="7"
        placeholder="Inspect the repository&#10;Apply the minimal fix&#10;Run the targeted tests">${esc((skill?.steps || []).join("\n"))}</textarea>
    </label>
    <label>Tags — comma separated
      <input class="skill-f-tags" value="${esc((skill?.tags || []).join(", "))}"
             placeholder="django, testing" />
    </label>
    <div class="skill-form-actions">
      <button type="submit" class="btn primary">${skill ? "Save changes" : "Create skill"}</button>
      <button type="button" class="btn skill-cancel">Cancel</button>
      <span class="key-status skill-status"></span>
    </div>
  </form>`;

  const form = $(".skill-form", body);
  const status = $(".skill-status", form);
  $(".skill-cancel", form).addEventListener("click", () => loadSkills(overlay));
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const name = $(".skill-f-name", form).value.trim();
    if (!name) { status.className = "key-status err"; status.textContent = "A name is required."; return; }
    const payload = {
      name,
      description: $(".skill-f-desc", form).value.trim(),
      steps: $(".skill-f-steps", form).value.split("\n").map(x => x.trim()).filter(Boolean),
      tags: $(".skill-f-tags", form).value.split(",").map(x => x.trim()).filter(Boolean),
    };
    const btn = $("button[type=submit]", form);
    btn.disabled = true;
    status.className = "key-status"; status.textContent = "saving…";
    try {
      if (skill) await api("PUT", "/ui/api/skills/" + encodeURIComponent(skill.id), payload);
      else await api("POST", "/ui/api/skills", payload);
      await loadSkills(overlay);
      const top = $(".skill-status", overlay);
      if (top) { top.className = "key-status ok"; top.textContent = skill ? "✓ saved" : "✓ created"; }
    } catch (err) {
      status.className = "key-status err"; status.textContent = "✗ " + err.message;
      btn.disabled = false;
    }
  });
}

/* ── 🗂 Resources modal (task <-> source links + Isolate) ─────────────────── */
let _resModal = null;

async function openResourcesModal() {
  if (_resModal) return;
  const overlay = el(`<div class="modal-overlay"><div class="modal resources-modal">
    <div class="modal-head"><h3>🗂 Resources</h3><button class="icon-btn modal-close" title="Close">✕</button></div>
    <p class="modal-sub">Every source the employee has, with the tasks linked to it.
      <b>Shared</b> sources are reusable by any task; <b>private</b> ones belong to one task.
      <b>Isolate all tasks</b> copies each shared source into that task's private area — re-downloading
      from the internet when the shared copy is gone — so nothing stays shared afterwards.</p>
    <div class="key-actions" style="margin:0 0 8px">
      <button class="btn primary res-isolate-all" title="Copy every shared source into each task's private area">🔒 Isolate all tasks</button>
      <button class="btn res-purge-all" title="Delete every catalogued resource (catalog + files)">🗑 Delete all</button>
      <button class="btn res-wipe-all" title="Delete EVERYTHING in the workspace, tracked or not (models/, data/, sessions/, …) — full reset">🧨 Wipe workspace</button>
      <button class="btn res-refresh">⟳ Refresh</button>
      <span class="key-status res-status"></span>
    </div>
    <div class="res-body"></div>
  </div></div>`);
  document.body.appendChild(overlay);
  _resModal = overlay;
  const close = () => { overlay.remove(); _resModal = null; };
  $(".modal-close", overlay).addEventListener("click", close);
  overlay.addEventListener("click", e => { if (e.target === overlay) close(); });
  $(".res-refresh", overlay).addEventListener("click", () => renderResources(overlay));
  $(".res-isolate-all", overlay).addEventListener("click", async () => {
    const st = $(".res-status", overlay);
    st.className = "key-status"; st.textContent = "isolating… (may re-download)";
    try {
      const r = await api("POST", "/ui/api/resources/isolate", { all: true, allow_redownload: true });
      st.className = "key-status ok";
      st.textContent = `✓ isolated ${r.isolated.length} · purged ${(r.purged || []).length} · pending ${r.needs_redownload.length}`;
      renderResources(overlay);
    } catch (e) { st.className = "key-status err"; st.textContent = "✗ " + e.message; }
  });
  $(".res-purge-all", overlay).addEventListener("click", async () => {
    if (!confirm("Delete ALL resources — every catalog entry AND the files on disk — for a fresh start?\n\nThis cannot be undone.")) return;
    const st = $(".res-status", overlay);
    st.className = "key-status"; st.textContent = "deleting…";
    try {
      const r = await api("POST", "/ui/api/resources/purge", { purge_bytes: true });
      st.className = "key-status ok";
      st.textContent = `✓ deleted ${r.deleted} resources · ${r.bytes_freed} bytes freed`;
      renderResources(overlay);
    } catch (e) { st.className = "key-status err"; st.textContent = "✗ " + e.message; }
  });
  $(".res-wipe-all", overlay).addEventListener("click", async () => {
    if (!confirm("WIPE THE WORKSPACE?\n\nThis deletes EVERYTHING in the workspace — every downloaded model/dataset, every file a task produced (models/, data/, sessions/, …), tracked or not.\n\nThis cannot be undone.")) return;
    const st = $(".res-status", overlay);
    st.className = "key-status"; st.textContent = "wiping workspace…";
    try {
      const r = await api("POST", "/ui/api/resources/purge", { purge_bytes: true, wipe_workspace: true });
      st.className = "key-status ok";
      st.textContent = `✓ purged ${r.deleted} resources · wiped ${r.workspace?.removed_count ?? 0} entries · ${r.workspace?.bytes_freed ?? 0} bytes`;
      renderResources(overlay);
    } catch (e) { st.className = "key-status err"; st.textContent = "✗ " + e.message; }
  });
  renderResources(overlay);
}

async function renderResources(overlay) {
  const body = $(".res-body", overlay);
  let data;
  try { data = await api("GET", "/ui/api/resources"); }
  catch (e) { body.innerHTML = `<div class="key-status err">✗ ${esc(e.message)}</div>`; return; }
  if (!data.resources.length) {
    body.innerHTML = `<p class="muted">No resources yet — run a task that downloads something.</p>`;
    return;
  }
  const rows = data.resources.map(r => {
    let tasks = (r.tasks || []).map(t => t.task_key || t.task_id || t.session_id).filter(Boolean);
    if (!tasks.length && r.owner_task) tasks = [r.owner_task];
    const sharedWith = r.task_count > 1 ? ` <span class="key-set">· ${r.task_count} tasks</span>` : "";
    const cls = r.visibility === "shared" ? "res-shared" : "res-private";
    return `<tr class="${cls}">
      <td>${esc(r.name)}${sharedWith}</td>
      <td>${esc(r.kind)}</td>
      <td>${esc(r.visibility)}</td>
      <td>${tasks.map(esc).join(", ") || "—"}</td>
      <td>${esc(r.size)} B${r.exists ? "" : " ⚠ missing"}</td>
    </tr>`;
  }).join("");
  body.innerHTML = `
    <div class="key-desc" style="margin:4px 0 8px">${data.count} sources · ${data.shared} shared · ${data.private} private</div>
    <table class="res-table">
      <thead><tr><th>Source</th><th>Kind</th><th>Visibility</th><th>Linked tasks</th><th>Size</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

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

/* ── MCP connectors modal (enable popular MCP servers as tools) ──────────── */
let _mcpModal = null;

async function openMcpModal() {
  if (_mcpModal) return;
  let data;
  try { data = await api("GET", "/ui/api/mcp/options"); }
  catch (e) { alert("Failed to load MCP connectors: " + e.message); return; }

  const overlay = el(`<div class="modal-overlay"><div class="modal mcp-modal">
    <div class="modal-head"><h3>🔌 MCP connectors</h3><button class="icon-btn modal-close" title="Close">✕</button></div>
    <p class="modal-sub">Plug external <b>Model Context Protocol</b> servers into the employee —
       their tools appear to every agent as <code>mcp_*</code> tools. Connecting may take a
       moment the first time (the server package is downloaded).</p>
    <div class="keys-body mcp-body"></div>
  </div></div>`);
  document.body.appendChild(overlay);
  _mcpModal = overlay;

  const close = () => { overlay.remove(); _mcpModal = null; };
  $(".modal-close", overlay).addEventListener("click", close);
  overlay.addEventListener("click", (e) => { if (e.target === overlay) close(); });
  renderMcp(overlay, data);
}

function renderMcp(overlay, data) {
  const body = $(".mcp-body", overlay);
  body.innerHTML = "";
  const rt = data.runtime || {};
  renderGoogleAccounts(body);  // async — fills its own section when loaded
  if (!rt.sdk) {
    body.appendChild(el(`<div class="key-status err" style="padding-top:10px">
      ✗ The <code>mcp</code> Python package is not installed — run <code>pip install mcp</code> and restart.</div>`));
  }
  const groups = [
    ["Enabled", data.connectors.filter(c => c.configured && c.enabled)],
    ["Disabled", data.connectors.filter(c => c.configured && !c.enabled)],
    ["Available", data.connectors.filter(c => !c.configured)],
  ];
  for (const [title, list] of groups) {
    if (!list.length) continue;
    const group = el(`<div class="key-group"><h4>${esc(title)}</h4></div>`);
    list.forEach(c => group.appendChild(mcpRow(overlay, c, rt)));
    body.appendChild(group);
  }
}

function mcpRow(overlay, c, rt) {
  const rtMissing = c.runtime && rt[c.runtime] === false;
  let status;
  if (c.error) status = `<span class="key-unset" style="color:var(--red)">error</span>`;
  else if (c.configured && c.tools.length) status = `<span class="key-set">● ${c.tools.length} tool${c.tools.length === 1 ? "" : "s"}</span>`;
  else if (c.configured) status = `<span class="key-unset">no tools yet</span>`;
  else status = `<span class="key-unset">not connected</span>`;
  const tags =
    (c.require_approval ? ` <span class="key-tag">approval-gated</span>` : "") +
    (c.auth === "oauth" ? ` <span class="key-tag" title="Connect opens the provider's consent page in your browser — no API keys needed">🔐 ${c.authorized ? "authorized" : "browser sign-in"}</span>` : "") +
    (rtMissing ? ` <span class="key-tag" style="color:var(--red);border-color:var(--red)">needs ${esc(c.runtime)}</span>` : "");

  const paramInputs = c.params.map(p => `
    <input class="key-input mcp-param" data-param="${esc(p.name)}"
           type="${p.secret ? "password" : "text"}"
           placeholder="${esc(p.label)}${p.required ? "" : " (optional)"}${p.configured ? " — saved: " + esc(p.masked || "set") : (p.default ? " — default: " + esc(p.default) : "")}"
           title="${esc(p.description || p.label)}" />`).join("");

  const row = el(`<div class="key-row">
    <div class="key-info">
      <div class="key-label">${esc(c.label)} ${status}${tags}</div>
      <div class="key-desc">${esc(c.description)}
        ${c.docs ? ` <a class="x-link" href="${esc(c.docs)}" target="_blank" rel="noopener">docs ↗</a>` : ""}</div>
      ${c.tools.length ? `<div class="mcp-tools">${c.tools.map(t => `<code>${esc(t)}</code>`).join(" ")}</div>` : ""}
    </div>
    <div class="key-actions">
      ${paramInputs}
      <label class="mcp-approve" title="Pause for human approval before any tool of this connector runs">
        <input type="checkbox" class="mcp-approve-box" ${c.require_approval ? "checked" : ""} /> approval
      </label>
      <button class="btn key-test mcp-connect">${c.configured ? "Reconnect" : "Connect"}</button>
      ${c.configured ? `<button class="btn mcp-toggle">${c.enabled ? "Disable" : "Enable"}</button>` : ""}
      <button class="btn key-del mcp-del" ${c.configured ? "" : "disabled"}>Remove</button>
    </div>
    <div class="key-status">${c.error ? "✗ " + esc(c.error) : ""}</div></div>`);
  if (c.error) $(".key-status", row).classList.add("err");

  const statusEl = $(".key-status", row);
  const refresh = async () => renderMcp(overlay, await api("GET", "/ui/api/mcp/options"));

  $(".mcp-connect", row).addEventListener("click", async () => {
    const btn = $(".mcp-connect", row);
    btn.textContent = "Connecting…"; btn.disabled = true;
    statusEl.className = "key-status"; statusEl.textContent = "Starting the server and listing its tools…";
    const params = {};
    row.querySelectorAll(".mcp-param").forEach(i => { params[i.dataset.param] = i.value.trim(); });
    try {
      const res = await api("PUT", `/ui/api/mcp/servers/${c.id}`, {
        params, require_approval: $(".mcp-approve-box", row).checked,
      });
      if (res.ok) { await refresh(); return; }
      if (res.needs_auth) {  // OAuth connector: open the provider's consent page
        const ok = await mcpOauthFlow(c, btn, statusEl);
        if (ok) { await refresh(); return; }
      } else {
        statusEl.className = "key-status err"; statusEl.textContent = "✗ " + res.error;
      }
    } catch (e) {
      statusEl.className = "key-status err"; statusEl.textContent = "✗ " + e.message;
    }
    btn.textContent = c.configured ? "Reconnect" : "Connect"; btn.disabled = false;
  });

  const toggleBtn = $(".mcp-toggle", row);
  if (toggleBtn) toggleBtn.addEventListener("click", async () => {
    toggleBtn.disabled = true;
    try { await api("POST", `/ui/api/mcp/servers/${c.id}/toggle`, { enabled: !c.enabled }); await refresh(); }
    catch (e) { alert("Failed: " + e.message); toggleBtn.disabled = false; }
  });

  $(".mcp-del", row).addEventListener("click", async () => {
    if (!confirm(`Remove the ${c.label} connector and its saved settings?`)) return;
    try { await api("DELETE", `/ui/api/mcp/servers/${c.id}`); await refresh(); }
    catch (e) { alert("Failed: " + e.message); }
  });
  return row;
}

/* ── Google accounts (Gmail & Calendar via the operator's OAuth app) ──────── */
async function renderGoogleAccounts(body) {
  const section = el(`<div class="key-group"><h4>Google accounts (Gmail &amp; Calendar)</h4></div>`);
  body.prepend(section);
  let st;
  try { st = await api("GET", "/ui/api/google/status"); }
  catch { section.remove(); return; }

  if (!st.configured) {
    section.appendChild(el(`<div class="key-row"><div class="key-info">
      <div class="key-desc">One-time operator setup: put your Google Cloud
      <code>client_secret.json</code> at <code>${esc(st.client_file)}</code> and restart.
      After that, anyone can connect their own Gmail/Calendar with one click.</div>
    </div></div>`));
    return;
  }

  for (const a of st.accounts) {
    const row = el(`<div class="key-row">
      <div class="key-info">
        <div class="key-label">${esc(a.email)} <span class="key-set">● connected</span></div>
        <div class="key-desc">since ${esc(a.connected_at || "?")}</div>
      </div>
      <div class="key-actions"><button class="btn key-del">Disconnect</button></div>
      <div class="key-status"></div></div>`);
    $(".key-del", row).addEventListener("click", async () => {
      if (!confirm(`Disconnect ${a.email}? The AI employee loses access to this mailbox.`)) return;
      try { await api("DELETE", `/ui/api/google/accounts/${encodeURIComponent(a.email)}`); row.remove(); }
      catch (e) { alert("Failed: " + e.message); }
    });
    section.appendChild(row);
  }

  const addRow = el(`<div class="key-row">
    <div class="key-info"><div class="key-desc">Connect a Google account — approve on Google's
      consent page; the employee's <code>gmail_*</code>/<code>calendar_*</code> tools then act on that mailbox.
      Sending mail & creating events stay approval-gated.</div></div>
    <div class="key-actions"><button class="btn key-test google-connect">Connect Google account</button></div>
    <div class="key-status"></div></div>`);
  const btn = $(".google-connect", addRow), statusEl = $(".key-status", addRow);
  btn.addEventListener("click", async () => {
    btn.disabled = true; btn.textContent = "Waiting for approval…";
    statusEl.className = "key-status"; statusEl.textContent = "Approve in the Google window that just opened…";
    try {
      const res = await api("POST", "/ui/api/google/authorize", {});
      const popup = window.open(res.auth_url, "google-oauth", "width=620,height=760");
      if (!popup) window.open(res.auth_url, "_blank");
      const before = st.accounts.length, deadline = Date.now() + 3 * 60 * 1000;
      while (Date.now() < deadline) {
        await new Promise(r => setTimeout(r, 2000));
        let cur;
        try { cur = await api("GET", "/ui/api/google/status"); } catch { continue; }
        if (cur.accounts.length > before) {  // re-render the whole section fresh
          section.remove(); renderGoogleAccounts(body); return;
        }
      }
      statusEl.className = "key-status err";
      statusEl.textContent = "✗ Timed out waiting for approval — try again.";
    } catch (e) {
      statusEl.className = "key-status err"; statusEl.textContent = "✗ " + e.message;
    }
    btn.disabled = false; btn.textContent = "Connect Google account";
  });
  section.appendChild(addRow);
}

/* ── ⏰ Schedules modal (recurring tasks, one-shots, Gmail watchers) ───────── */
let _schedModal = null;

async function openSchedulesModal() {
  if (_schedModal) return;
  const overlay = el(`<div class="modal-overlay"><div class="modal sched-modal">
    <div class="modal-head"><h3>⏰ Scheduled tasks &amp; watchers</h3><button class="icon-btn modal-close" title="Close">✕</button></div>
    <p class="modal-sub">The employee works while you're away: run a goal on a schedule, once at a
      set time, or whenever new email matches a Gmail search. Runs appear as normal tasks —
      they pause for approvals and push to your phone if notifications are set up.</p>
    <div class="keys-body sched-body"></div>
  </div></div>`);
  document.body.appendChild(overlay);
  _schedModal = overlay;
  const close = () => { overlay.remove(); _schedModal = null; };
  $(".modal-close", overlay).addEventListener("click", close);
  overlay.addEventListener("click", e => { if (e.target === overlay) close(); });
  renderSchedules(overlay);
}

function schedSummary(s) {
  if (s.type === "once") return `once at ${s.run_at}`;
  if (s.type === "gmail_watch") return `watch Gmail "${s.query}" every ${s.interval_minutes} min`;
  if (s.every_minutes) return `every ${s.every_minutes} min`;
  const days = s.days ? ` on days ${s.days.join(",")}` : " daily";
  return `at ${s.at}${days}`;
}

async function renderSchedules(overlay) {
  const body = $(".sched-body", overlay);
  let data;
  try { data = await api("GET", "/ui/api/schedules"); }
  catch (e) { body.innerHTML = `<div class="key-status err">✗ ${esc(e.message)}</div>`; return; }
  body.innerHTML = "";
  const refresh = () => renderSchedules(overlay);

  const list = el(`<div class="key-group"><h4>Configured</h4></div>`);
  if (!data.schedules.length)
    list.appendChild(el(`<div class="key-row"><div class="key-info"><div class="key-desc">Nothing scheduled yet.</div></div></div>`));
  for (const s of data.schedules) {
    const status = s.enabled
      ? `<span class="key-set">● on</span>` : `<span class="key-unset">off</span>`;
    const err = s.last_error ? `<div class="key-status err">✗ ${esc(s.last_error)}</div>` : "";
    const row = el(`<div class="key-row">
      <div class="key-info">
        <div class="key-label">${esc(s.name)} ${status} <span class="key-tag">${esc(schedSummary(s))}</span>
          ${s.auto_approve ? '<span class="key-tag">auto-approve</span>' : ""}</div>
        <div class="key-desc">${esc(s.goal.slice(0, 160))}</div>
        <div class="key-desc muted">next: ${esc(s.next_run || "—")} · last: ${esc(s.last_run || "never")} · runs: ${s.runs}</div>
      </div>
      <div class="key-actions">
        <button class="btn key-test sch-run">Run now</button>
        <button class="btn sch-toggle">${s.enabled ? "Disable" : "Enable"}</button>
        <button class="btn key-del sch-del">Delete</button>
      </div>${err}</div>`);
    $(".sch-run", row).addEventListener("click", async () => {
      try { await api("POST", `/ui/api/schedules/${s.id}/run_now`); loadSessionList(); refresh(); }
      catch (e) { alert("Failed: " + e.message); }
    });
    $(".sch-toggle", row).addEventListener("click", async () => {
      try { await api("PUT", `/ui/api/schedules/${s.id}`, { enabled: !s.enabled }); refresh(); }
      catch (e) { alert("Failed: " + e.message); }
    });
    $(".sch-del", row).addEventListener("click", async () => {
      if (!confirm(`Delete schedule "${s.name}"?`)) return;
      try { await api("DELETE", `/ui/api/schedules/${s.id}`); refresh(); }
      catch (e) { alert("Failed: " + e.message); }
    });
    list.appendChild(row);
  }
  body.appendChild(list);

  const form = el(`<div class="key-group"><h4>New schedule</h4>
    <div class="sched-form">
      <input class="key-input" id="sch-name" placeholder="Name (optional)" />
      <textarea class="key-input" id="sch-goal" rows="2" placeholder="Goal, e.g. Summarize my unread emails and today's calendar, then email me the briefing"></textarea>
      <div class="sched-form-row">
        <select class="key-input" id="sch-type">
          <option value="daily">Daily at time</option>
          <option value="every">Every N minutes</option>
          <option value="once">Once at date/time</option>
          <option value="gmail">When new Gmail matches…</option>
        </select>
        <input class="key-input" id="sch-at" type="time" value="08:00" />
        <input class="key-input hidden" id="sch-every" type="number" min="1" placeholder="minutes" />
        <input class="key-input hidden" id="sch-once" type="datetime-local" />
        <input class="key-input hidden" id="sch-query" placeholder='Gmail query, e.g. is:unread from:boss@co.com' />
        <label class="mcp-approve" title="Never pause this schedule's runs for approval">
          <input type="checkbox" id="sch-auto" /> auto-approve</label>
        <button class="btn primary" id="sch-add">Add</button>
      </div>
      <div class="key-status" id="sch-status"></div>
    </div></div>`);
  const typeSel = $("#sch-type", form);
  const show = id => {
    for (const f of ["sch-at", "sch-every", "sch-once", "sch-query"])
      $("#" + f, form).classList.toggle("hidden", f !== id);
  };
  typeSel.addEventListener("change", () => {
    show({ daily: "sch-at", every: "sch-every", once: "sch-once", gmail: "sch-query" }[typeSel.value]);
  });
  $("#sch-add", form).addEventListener("click", async () => {
    const t = typeSel.value;
    const payload = {
      name: $("#sch-name", form).value.trim(),
      goal: $("#sch-goal", form).value.trim(),
      auto_approve: $("#sch-auto", form).checked,
      type: t === "gmail" ? "gmail_watch" : t === "once" ? "once" : "recurring",
    };
    if (t === "daily") payload.at = $("#sch-at", form).value;
    if (t === "every") payload.every_minutes = parseInt($("#sch-every", form).value, 10) || null;
    if (t === "once") payload.run_at = $("#sch-once", form).value;
    if (t === "gmail") { payload.query = $("#sch-query", form).value.trim(); payload.interval_minutes = 5; }
    const st = $("#sch-status", form);
    try { await api("POST", "/ui/api/schedules", payload); refresh(); }
    catch (e) { st.className = "key-status err"; st.textContent = "✗ " + e.message; }
  });
  body.appendChild(form);
}

/* ── 🛡 Autonomy modal (approval policies, notifications, budget, audit) ──── */
let _autoModal = null;

async function openAutonomyModal() {
  if (_autoModal) return;
  const overlay = el(`<div class="modal-overlay"><div class="modal autonomy-modal">
    <div class="modal-head"><h3>🛡 Autonomy &amp; safety</h3><button class="icon-btn modal-close" title="Close">✕</button></div>
    <p class="modal-sub">How much the employee may do on its own — and a full trail of what it has done.</p>
    <div class="keys-body autonomy-body"></div>
  </div></div>`);
  document.body.appendChild(overlay);
  _autoModal = overlay;
  const close = () => { overlay.remove(); _autoModal = null; };
  $(".modal-close", overlay).addEventListener("click", close);
  overlay.addEventListener("click", e => { if (e.target === overlay) close(); });
  renderAutonomy(overlay);
}

async function renderAutonomy(overlay) {
  const body = $(".autonomy-body", overlay);
  let st, audit;
  try {
    st = await api("GET", "/ui/api/autonomy");
    audit = await api("GET", "/ui/api/audit?limit=50");
  } catch (e) { body.innerHTML = `<div class="key-status err">✗ ${esc(e.message)}</div>`; return; }
  body.innerHTML = "";

  // Push notifications
  const ntfy = el(`<div class="key-group"><h4>Phone notifications</h4>
    <div class="key-row"><div class="key-info">
      <div class="key-label">ntfy ${st.ntfy.enabled ? '<span class="key-set">● configured</span>' : '<span class="key-unset">off</span>'}</div>
      <div class="key-desc">${st.ntfy.enabled
        ? `Pushes to topic ${esc(st.ntfy.topic)} on ${esc(st.ntfy.server)} when a run needs approval, completes, or fails.`
        : `Set <code>NTFY_TOPIC=&lt;a-long-random-name&gt;</code> in <code>.env</code>, restart, and subscribe to the same topic in the free ntfy app — approvals then reach your phone.`}</div>
    </div>
    <div class="key-actions">${st.ntfy.enabled ? '<button class="btn key-test ntfy-test">Send test</button>' : ""}</div>
    <div class="key-status"></div></div></div>`);
  const testBtn = $(".ntfy-test", ntfy);
  if (testBtn) testBtn.addEventListener("click", async () => {
    const s2 = $(".key-status", ntfy);
    try { await api("POST", "/ui/api/autonomy/notify_test"); s2.className = "key-status ok"; s2.textContent = "✓ sent — check your phone"; }
    catch (e) { s2.className = "key-status err"; s2.textContent = "✗ " + e.message; }
  });
  body.appendChild(ntfy);

  // Approval policies
  const pol = el(`<div class="key-group"><h4>Approval policies</h4>
    <div class="key-desc" style="margin:4px 0 8px">Per operation: keep the human gate, or auto-approve
      within limits (recipient patterns like <code>*@yourco.com</code>, and a daily quota counted from the audit trail).</div>
    <div class="policy-rows"></div>
    <div class="key-actions" style="margin-top:8px"><button class="btn primary pol-save">Save policies</button></div>
    <div class="key-status"></div></div>`);
  const rows = $(".policy-rows", pol);
  for (const key of st.approval_keys) {
    const p = st.policies[key] || {};
    rows.appendChild(el(`<div class="policy-row" data-key="${esc(key)}">
      <code class="policy-key">${esc(key)}</code>
      <select class="key-input pol-mode">
        <option value="gate" ${p.mode !== "auto" ? "selected" : ""}>ask a human</option>
        <option value="auto" ${p.mode === "auto" ? "selected" : ""}>auto-approve</option>
      </select>
      <input class="key-input pol-recipients" placeholder="allowed recipients (comma, * ok)"
             value="${esc((p.allow_recipients || []).join(", "))}" />
      <input class="key-input pol-limit" type="number" min="0" placeholder="daily limit"
             value="${p.daily_limit || ""}" />
    </div>`));
  }
  $(".pol-save", pol).addEventListener("click", async () => {
    const policies = {};
    rows.querySelectorAll(".policy-row").forEach(r => {
      const mode = $(".pol-mode", r).value;
      const recipients = $(".pol-recipients", r).value.split(",").map(s => s.trim()).filter(Boolean);
      const limit = parseInt($(".pol-limit", r).value, 10);
      if (mode === "auto" || recipients.length || limit) {
        policies[r.dataset.key] = { mode };
        if (recipients.length) policies[r.dataset.key].allow_recipients = recipients;
        if (limit) policies[r.dataset.key].daily_limit = limit;
      }
    });
    const s2 = $(".key-status", pol);
    try { await api("PUT", "/ui/api/autonomy/policies", { policies }); s2.className = "key-status ok"; s2.textContent = "✓ saved"; }
    catch (e) { s2.className = "key-status err"; s2.textContent = "✗ " + e.message; }
  });
  body.appendChild(pol);

  // Budget
  body.appendChild(el(`<div class="key-group"><h4>Token budget</h4>
    <div class="key-row"><div class="key-info"><div class="key-desc">
      Global default: <b>${st.task_token_budget ? st.task_token_budget.toLocaleString() + " tokens/task" : "unlimited"}</b>
      (set <code>TASK_TOKEN_BUDGET</code> in <code>.env</code>). Per-run override: the “∞ tokens” box next to
      the Run button. A run that hits its cap stops with a clear budget error.</div></div></div></div>`));

  // Audit trail
  const au = el(`<div class="key-group"><h4>Audit trail — every sensitive action, all tasks</h4>
    <div class="audit-rows"></div></div>`);
  const ar = $(".audit-rows", au);
  if (!audit.actions.length)
    ar.appendChild(el(`<div class="key-desc">Nothing yet — sensitive tool runs (mail sends, calendar writes, shell writes, MCP calls…) will appear here.</div>`));
  for (const a of audit.actions) {
    const argsTxt = Object.entries(a.args || {}).map(([k, v]) => `${k}=${v}`).join("  ").slice(0, 180);
    ar.appendChild(el(`<div class="audit-row ${a.ok ? "" : "audit-fail"}">
      <span class="audit-ts">${esc(a.ts)}</span>
      <code>${esc(a.tool)}</code>
      <span class="key-tag">${esc(a.approved_by)}</span>
      <span class="audit-args">${esc(argsTxt)}</span>
      <span>${a.ok ? "✓" : "✗"}</span></div>`));
  }
  body.appendChild(au);
}

/* Open the provider's consent page in a popup, then poll until the callback
   route has exchanged the code and the connector reports its tools. */
async function mcpOauthFlow(c, btn, statusEl) {
  let auth;
  try { auth = await api("POST", `/ui/api/mcp/servers/${c.id}/authorize`, {}); }
  catch (e) {
    statusEl.className = "key-status err"; statusEl.textContent = "✗ " + e.message;
    return false;
  }
  const popup = window.open(auth.auth_url, "mcp-oauth", "width=620,height=760");
  if (!popup) { window.open(auth.auth_url, "_blank"); }
  btn.textContent = "Waiting for approval…";
  statusEl.className = "key-status";
  statusEl.textContent = `Approve ${c.label} in the window that just opened…`;
  const deadline = Date.now() + 3 * 60 * 1000;
  while (Date.now() < deadline) {
    await new Promise(r => setTimeout(r, 2000));
    let opts;
    try { opts = await api("GET", "/ui/api/mcp/options"); } catch { continue; }
    const cur = (opts.connectors || []).find(x => x.id === c.id) || {};
    if (cur.tools && cur.tools.length) return true;
    if (cur.error) {
      statusEl.className = "key-status err"; statusEl.textContent = "✗ " + cur.error;
      return false;
    }
  }
  statusEl.className = "key-status err";
  statusEl.textContent = "✗ Timed out waiting for approval — click Connect to try again.";
  return false;
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
    <div class="row bp-model-tools">
      <button class="btn bp-test-model" title="Ask the provider whether this model is available to your key">Test model</button>
      <span class="bp-model-status"></span>
    </div>
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

  // "Test model": ask the provider's own server whether this model is callable by
  // the current key. Documentation lists models a key may not be entitled to, so
  // the picker verifies instead of assuming. Results are advisory — a failed probe
  // never blocks Apply.
  const testBtn = $(".bp-test-model", pop), modelStatus = $(".bp-model-status", pop);
  let _modelTestTimer = null;
  const selectedModel = () => (customInput.value.trim() || modelSel.value || "").trim();
  const runModelTest = async () => {
    const provider = provSel.value;
    const model = selectedModel();
    modelStatus.className = "bp-model-status";
    if (!model) { modelStatus.textContent = ""; return; }
    if (provider === "mock") {
      modelStatus.textContent = "Mock is offline — no model list to check.";
      return;
    }
    modelStatus.className = "bp-model-status loading";
    modelStatus.innerHTML = `<span class="spin">◌</span> Testing ${esc(model)} (listing + real request)…`;
    let res;
    try {
      res = await api("POST", "/ui/api/brain/models/check", {
        provider, model, api_key: keyInput.value.trim() || undefined,
      });
    } catch (e) {
      modelStatus.className = "bp-model-status err";
      modelStatus.textContent = "Check failed: " + e.message;
      return;
    }
    // Two distinct facts: is it listed, and will the API actually serve it?
    const listedTxt = res.listed === true ? "listed" : (res.listed === false ? "not listed" : "list n/a");
    if (res.callable === true) {
      modelStatus.className = "bp-model-status ok";
      modelStatus.textContent = `✓ ${res.matched || model} works (${listedTxt}, real request OK)`;
    } else if (res.callable === false) {
      modelStatus.className = "bp-model-status err";
      const why = res.probe && res.probe.http_status ? `HTTP ${res.probe.http_status}` : (res.error || "failed");
      modelStatus.textContent = `✗ ${model} cannot be used — ${why} (${listedTxt})`;
      modelStatus.title = [res.message, res.detail].filter(Boolean).join("\n\n");
    } else {
      modelStatus.className = "bp-model-status warn";
      modelStatus.textContent = res.message || "Could not verify this model.";
    }
  };
  testBtn.addEventListener("click", runModelTest);
  modelSel.addEventListener("change", () => {
    clearTimeout(_modelTestTimer);
    _modelTestTimer = setTimeout(runModelTest, 250);
  });
  customInput.addEventListener("change", runModelTest);

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

/* ── Account (optional — default is a no-account "TestN" guest) ─────────── */
let _me = null, _accountPop = null;

async function loadAccountBadge() {
  const badge = $("#account-badge");
  if (!badge) return null;
  try { _me = await api("GET", "/ui/api/me"); }
  catch { badge.textContent = "👤 ?"; return null; }
  const guest = _me.kind === "guest";
  badge.classList.toggle("warn", guest);
  badge.innerHTML = guest
    ? `<span>👤 ${esc(_me.name)}</span><span class="caret">▾</span>`
    : `<span>👤 ${esc(_me.name || _me.email)}</span><span class="caret">▾</span>`;
  badge.title = guest
    ? `Browsing as guest "${_me.name}" — nothing about you is stored, and your tasks are forgotten when the server restarts. Click to sign in.`
    : `Signed in as ${_me.email}. Your tasks are saved to your account.`;
  return _me;
}

function openAccountPicker() {
  if (_accountPop) { _accountPop.remove(); _accountPop = null; return; }
  const me = _me || { kind: "guest", name: "guest", google_signin_available: false };
  const guest = me.kind === "guest";
  const pop = el(`<div class="brain-pop account-pop">
    <h4>Account</h4>
    ${guest
      ? `<p class="hint">You're using the console <b>without an account</b> as
           <b>${esc(me.name)}</b>. Nothing about you is stored, and your tasks
           disappear when the server restarts.</p>
         <button class="btn primary acc-signin" style="width:100%"
                 ${me.google_signin_available ? "" : "disabled"}>Sign in with Google</button>
         ${me.google_signin_available ? "" :
           `<p class="hint warn">Google sign-in isn't configured — the operator must add
              the OAuth client file (data/google_oauth_client.json).</p>`}`
      : `<p class="hint">Signed in as <b>${esc(me.email)}</b>. Your tasks are saved
           to this account and survive restarts.</p>
         <button class="btn acc-signout" style="width:100%">Sign out</button>`}
    <div class="row"><button class="btn acc-close">Close</button></div></div>`);
  document.body.appendChild(pop);
  _accountPop = pop;
  const close = () => { pop.remove(); _accountPop = null; };
  $(".acc-close", pop).addEventListener("click", close);
  const signin = $(".acc-signin", pop);
  if (signin) signin.addEventListener("click", async () => {
    signin.disabled = true; signin.textContent = "Opening Google…";
    try {
      const res = await api("POST", "/ui/api/auth/google");
      window.open(res.auth_url, "_blank", "width=520,height=680");
      // The popup sets the session cookie on success; poll until identity flips.
      const before = me.id;
      const timer = setInterval(async () => {
        const now = await loadAccountBadge();
        if (now && now.id !== before) {
          clearInterval(timer);
          close();
          loadSessionList();   // the account's saved tasks replace the guest view
        }
      }, 1500);
      setTimeout(() => clearInterval(timer), 180000);
      signin.textContent = "Waiting for Google…";
    } catch (e) {
      signin.disabled = false; signin.textContent = "Sign in with Google";
      alert("Could not start sign-in: " + e.message);
    }
  });
  const signout = $(".acc-signout", pop);
  if (signout) signout.addEventListener("click", async () => {
    try { await api("POST", "/ui/api/auth/logout"); }
    catch (e) { alert("Sign-out failed: " + e.message); return; }
    close();
    location.reload();   // next /me mints a fresh guest (TestN)
  });
  const badge = $("#account-badge");
  const r = badge.getBoundingClientRect();
  pop.style.position = "absolute";
  pop.style.top = (r.bottom + 6) + "px";
  pop.style.right = (window.innerWidth - r.right) + "px";
}

$("#account-badge").addEventListener("click", openAccountPicker);

/* ── Org chart — CEO on top, employees below, live activity + message bubbles ── */
let _orgModal = null;
let _orgTimer = null;
let _orgRoster = null; // cached /agents catalog (static; doesn't change at runtime)

const OC_CATEGORY_ICON = {
  management: "🧭", research: "🔎", engineering: "🛠", knowledge: "📚",
  data: "📊", quality: "✅", ops: "⚙️", comms: "✉️", web: "🌐",
};
function ocIcon(agent) {
  if (agent.name === "ceo") return "★";
  if (agent.name === "planner") return "🧭";
  if (agent.name === "human_approval") return "🖐";
  return OC_CATEGORY_ICON[agent.category] || "🤖";
}

async function openOrgChartModal() {
  if (_orgModal) return;
  const overlay = el(`<div class="modal-overlay"><div class="modal orgchart-modal">
    <div class="modal-head"><h3>🗺 Org chart</h3><button class="icon-btn modal-close" title="Close">✕</button></div>
    <p class="modal-sub">The CEO at the top, every employee underneath. An arrow only appears while
       information is actually moving: <span class="oc-legend-dot dispatch"></span> planner
       dispatching a step, <span class="oc-legend-dot handoff"></span> one agent's output feeding
       another, <span class="oc-legend-dot report"></span> a result reaching the CEO. A pulsing
       border + progress bar marks whoever is working right now, and a bubble travels each arrow
       in the direction the arrowhead points.</p>
    <div class="orgchart-body"></div>
  </div></div>`);
  document.body.appendChild(overlay);
  _orgModal = overlay;
  const close = () => {
    overlay.remove(); _orgModal = null;
    if (_orgTimer) { clearInterval(_orgTimer); _orgTimer = null; }
  };
  $(".modal-close", overlay).addEventListener("click", close);
  overlay.addEventListener("click", (e) => { if (e.target === overlay) close(); });

  const body = $(".orgchart-body", overlay);
  body.innerHTML = `<div class="orgchart-empty">Loading team…</div>`;
  try {
    if (!_orgRoster) _orgRoster = (await api("GET", "/agents")).agents;
  } catch (e) {
    body.innerHTML = `<div class="key-status err">✗ ${esc(e.message)}</div>`;
    return;
  }

  const tick = async () => {
    if (!_orgModal) return;
    let data;
    try { data = await api("GET", "/ui/api/orgchart"); }
    catch (e) { return; } // transient — keep showing the last good frame
    if (!_orgModal) return;
    renderOrgChart(body, _orgRoster, data);
  };
  await tick();
  _orgTimer = setInterval(tick, 1000);
}

// Fixed tiers so the layout is stable frame to frame (cards don't jump around
// as agents start/stop working) — grouped by the same category used elsewhere.
const OC_TIERS = [
  ["management", "Management", ["planner", "human_approval"]],
  ["research", "Research", []],
  ["engineering", "Engineering", []],
  ["knowledge", "Knowledge", []],
  ["data", "Data", []],
  ["quality", "Quality", []],
  ["ops", "Ops", []],
  ["comms", "Comms", []],
  ["web", "Web", []],
];

function renderOrgChart(body, roster, data) {
  const ceo = roster.find(a => a.name === "ceo");
  const others = roster.filter(a => a.name !== "ceo");

  // Live activity, unioned across every currently running task.
  const activeAgents = new Set();
  const progressByAgent = new Map();
  const edges = []; // {from, to, kind}
  let ceoActive = false;
  for (const t of (data.tasks || [])) {
    for (const a of (t.active_agents || [])) {
      activeAgents.add(a);
      progressByAgent.set(a, Math.max(progressByAgent.get(a) || 0, t.progress || 0));
    }
    if (t.ceo_active) ceoActive = true;
    for (const e of (t.edges || [])) edges.push(e);
  }

  const tierBuckets = new Map(OC_TIERS.map(([cat, label]) => [cat, { label, agents: [] }]));
  if (!tierBuckets.has("other")) tierBuckets.set("other", { label: "Other", agents: [] });
  const pinned = new Map(OC_TIERS.flatMap(([cat, , names]) => names.map(n => [n, cat])));
  for (const a of others) {
    const cat = pinned.get(a.name) || a.category || "other";
    if (!tierBuckets.has(cat)) tierBuckets.set(cat, { label: cat, agents: [] });
    tierBuckets.get(cat).agents.push(a);
  }

  const ocCard = (a, isCeo) => {
    const active = isCeo ? ceoActive : activeAgents.has(a.name);
    const progress = isCeo ? 100 : (progressByAgent.get(a.name) || 0);
    const task = (data.tasks || []).find(t =>
      isCeo ? t.ceo_active : (t.active_agents || []).includes(a.name));
    return `<div class="oc-card ${isCeo ? "oc-ceo" : ""} ${active ? "oc-active" : ""}" data-agent="${esc(a.name)}">
      <div class="oc-icon">${ocIcon(a)}</div>
      <div class="oc-name">${esc(a.role || a.name)}</div>
      <div class="oc-role">${esc(a.name)}</div>
      ${active ? `<div class="oc-status"><span class="pulse"></span> working</div>
        <div class="oc-progress"><div class="oc-progress-bar" style="width:${progress}%"></div></div>` : ""}
      ${active && task ? `<div class="oc-goal" title="${esc(task.goal || "")}">${esc(task.goal || "")}</div>` : ""}
    </div>`;
  };

  body.innerHTML = `<div class="orgchart-wrap">
    <div class="oc-ceo-row">${ocCard(ceo, true)}</div>
    <div class="oc-tiers">
      ${[...tierBuckets.values()].filter(b => b.agents.length).map(b => `
        <div class="oc-tier-group">
          <div class="oc-tier-label">${esc(b.label)}</div>
          <div class="oc-tier">${b.agents.map(a => ocCard(a, false)).join("")}</div>
        </div>`).join("")}
    </div>
    <svg class="orgchart-svg"></svg>
  </div>`;

  drawOrgEdges(body, edges);
}

// Draws ONLY edges that represent a real, live information flow right now —
// planner->agent dispatch, agent->agent handoff (a depends_on step reading
// another agent's finished output), and agent->ceo report during synthesis.
// No standing/always-on lines: an agent with nothing flowing to or from it
// has no line at all. Each edge gets an arrowhead for direction and an
// animated bubble travelling from -> to.
function drawOrgEdges(body, edges) {
  const svg = $(".orgchart-svg", body);
  const wrap = $(".orgchart-wrap", body);
  const wrapRect = wrap.getBoundingClientRect();
  svg.setAttribute("width", wrapRect.width);
  svg.setAttribute("height", wrapRect.height);

  const centerOf = (name) => {
    const card = body.querySelector(`.oc-card[data-agent="${CSS.escape(name)}"]`);
    if (!card) return null;
    const r = card.getBoundingClientRect();
    return { x: r.left + r.width / 2 - wrapRect.left, y: r.top + r.height / 2 - wrapRect.top };
  };

  const reduceMotion = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
  let svgHtml = `<defs>
    <marker id="oc-arrow-dispatch" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="var(--blue)"></path>
    </marker>
    <marker id="oc-arrow-handoff" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="var(--accent-2)"></path>
    </marker>
    <marker id="oc-arrow-report" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="var(--green)"></path>
    </marker>
  </defs>`;

  // Shrink each line so it stops at the card edge, not its center, so the
  // arrowhead is visible next to the box instead of hidden under it.
  const shrink = (from, to, px) => {
    const dx = to.x - from.x, dy = to.y - from.y;
    const len = Math.hypot(dx, dy) || 1;
    const t = Math.min(px / len, 0.45);
    return { x: to.x - dx * t, y: to.y - dy * t };
  };

  let bubbleId = 0;
  for (const e of edges) {
    const from = centerOf(e.from === "planner" ? "planner" : e.from);
    const to = centerOf(e.to);
    if (!from || !to) continue;
    const kind = e.kind || "dispatch";
    const end = shrink(from, to, 78);
    const pathId = `oc-path-${bubbleId++}`;
    svgHtml += `<path id="${pathId}" d="M${from.x},${from.y} L${to.x},${to.y}" fill="none" stroke="none"></path>`;
    svgHtml += `<line class="oc-edge active ${esc(kind)}" x1="${from.x}" y1="${from.y}" x2="${end.x}" y2="${end.y}"
      marker-end="url(#oc-arrow-${esc(kind)})"></line>`;
    if (!reduceMotion) {
      svgHtml += `<circle class="oc-bubble ${esc(kind)}" r="4">
        <animateMotion dur="1.1s" repeatCount="indefinite" rotate="auto">
          <mpath href="#${pathId}" xlink:href="#${pathId}"></mpath>
        </animateMotion>
      </circle>`;
    }
  }

  svg.innerHTML = svgHtml;
}

$("#orgchart-badge").addEventListener("click", openOrgChartModal);

// Populate the saved-tasks panel + brain/coding badges on load. The identity
// call runs first so the very first request mints the guest cookie before the
// session list (which is scoped to the current user) is fetched.
loadAccountBadge().finally(loadSessionList);
loadBrainBadge();
loadCodingBadge();
loadMemoryBadge();
loadParallelBadge();
loadLimitsBadge();
