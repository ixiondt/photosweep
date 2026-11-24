"use strict";
// Photo sweep review page. No framework, no CDN: the CSP allows 'self' only.

const TABS = [
  { key: "screenshot", label: "Screenshots", note: "Captures of a phone or computer screen. Most certain first." },
  { key: "meme", label: "Memes", note: "Images made to be shared: captions, reaction images, forwarded graphics." },
  { key: "document", label: "Documents", note: "Receipts, forms, IDs and other paper photographed for reference. Check before deleting: some may be worth keeping." },
  { key: "junk", label: "Junk", note: "Black frames, pocket shots and photos too blurred to make out." },
  { key: "clutter", label: "Clutter", note: "Photos the model, asked separately about each one, says nobody would miss: floors and walls, dark or accidental frames, random objects, scraps of paper, empty car parks. Surest first. People and pets should not be here, so look before deleting, and Keep anything it got wrong." },
  { key: "photo", label: "Photos", note: "What the model thinks is a real photo, least certain first: those are the ones it may have got wrong. Blurry ones are under Blurry and clutter under Clutter instead." },
  { key: "blurry", label: "Blurry", note: "Shaken, out-of-focus or smeared photos. The model is asked about each photo separately; its surest calls come first. Photos it has not looked at yet are judged by a pixel score until it does, and that score mistakes smooth skin, sky and plain walls for blur. Dark and Bright pick out the under- and over-exposed ones. Look before deleting." },
  { key: "dupes", label: "Duplicates", note: "The same picture in more than one file: a WhatsApp copy next to its original, a HEIC next to its JPEG, an IMG_0205_1 re-import. Found by what the picture shows, not by name or bytes. Each group puts its best copy first (most pixels, then a camera in the EXIF, then the larger file), and Select all extras picks every other copy in every Same group, loaded or not. Similar is burst shots and near-repeats: different photos, so there is no bulk select there; pick the ones to go by hand. The one exception is Mirrored only, where Select all mirrored copies picks just the flipped copies." },
  { key: "kept", label: "Kept", note: "Images you marked to keep. They never come back into the other tabs." },
  { key: "error", label: "Errors", note: "Files the scan could not read or label after three tries. Recheck tries again; an empty (0-byte) or garbage file will only fail again, so Delete those." },
];
// The tabs that need a model. With MODEL=none they are hidden, and the page
// is Duplicates, Kept and Errors.
const LABEL_TABS = ["screenshot", "meme", "document", "junk", "clutter", "photo", "blurry"];
function visibleTabs() {
  const off = state.summary && state.summary.labelling === false;
  return off ? TABS.filter((t) => !LABEL_TABS.includes(t.key)) : TABS;
}
// Recheck asks the model again: Errors are relabelled, photos get the clutter
// and blur questions again from the original at a larger size.
const RECHECKABLE = ["photo", "clutter", "blurry", "error"];
// Rechecking a photo asks the vision model again; with none configured only
// Errors can be retried (they are fingerprinted and sorted again).
function canRecheck(key) {
  return RECHECKABLE.includes(key) && (key === "error" || state.summary?.visionModel !== false);
}

const PAGE = 120;
const POLL_MS = 15000;
const DELETE_CHUNK = 500;

const $ = (id) => document.getElementById(id);
// Blurry only: many photos read as blurred because they are under- or
// over-exposed. Measured from the thumbnail's grey levels (exposure.py).
const EXPOSURES = [
  { key: "all", label: "Any" },
  { key: "dark", label: "Dark" },
  { key: "bright", label: "Bright" },
  { key: "normal", label: "Normal" },
];
const SOURCES = [
  { key: "all", label: "All" },
  { key: "camera", label: "Camera" },
  { key: "nocamera", label: "No camera" },
  { key: "messaging", label: "Messaging" },
];

const state = {
  tab: "screenshot", minConf: 0, source: "all", exposure: "all", year: "all", tier: "same", mirrored: false, groups: 0, bulk: false, items: [], total: 0, loading: false, done: false,
  selected: new Set(), anchor: null, summary: null, viewerIndex: -1, gen: 0,
};

// ---------------------------------------------------------------- api ---
async function api(path, body) {
  const opts = body === undefined ? {} : {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-Photosweep": "1" },
    body: JSON.stringify(body),
  };
  const resp = await fetch(path, opts);
  // signed out or the session ran out: back to sign-in (password mode only;
  // in header mode the proxy in front decides, and a reload re-asks it)
  if (resp.status === 401 && state.summary && state.summary.signOut) { location.assign("/login"); }
  let data = null;
  try { data = await resp.json(); } catch { /* non-JSON error page */ }
  if (!resp.ok) {
    const msg = (data && data.message) || `Request failed (${resp.status}).`;
    const err = new Error(msg);
    err.code = data && data.code;
    throw err;
  }
  return data;
}

let toastTimer = null;
function toast(message) {
  const el = $("toast");
  el.textContent = message;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, 6000);
}
function announce(message) { $("live").textContent = message; }

// ------------------------------------------------------------ summary ---
async function refreshSummary() {
  try {
    const s = await api("/api/summary");
    state.summary = s;
    renderSummary();
  } catch (err) {
    $("progress-line").textContent = `Could not load progress: ${err.message}`;
  }
}

function fmt(n) { return Number(n).toLocaleString(); }

function clutterText(item) {
  if (item.clutter_recheck) return "recheck queued";
  if (item.clutter === 1) return `clutter, ${item.clutter_conf}% sure${item.clutter_reason ? `: ${item.clutter_reason}` : ""}`;
  if (item.clutter === 0) return `a keeper, ${item.clutter_conf}% sure${item.clutter_reason ? `: ${item.clutter_reason}` : ""}`;
  if (item.clutter === -1) return "no answer; it stays in Photos";
  return "not checked yet";
}

function blurText(item) {
  if (item.blur_recheck) return "recheck queued";
  const by = item.blur_src === "classifier" ? " (image classifier)" : "";
  if (item.blurry === 1) return `blurry, ${item.blur_conf}% sure${by}`;
  if (item.blurry === 0) return `sharp, ${item.blur_conf}% sure${by}`;
  if (item.blurry === -1) return "no answer; the pixel score decides";
  return "not checked yet; the pixel score decides";
}

// A model outside the owner's network sees every photo it labels. Say so on
// every visit, not once in a settings screen.
function renderModelNote(m) {
  const el = $("model-note");
  if (!m || !m.remote) { el.hidden = true; return; }
  el.hidden = false;
  el.dataset.state = m.allowed ? "sending" : "refused";
  el.textContent = m.allowed
    ? `Photos are being sent to ${m.host} to be labelled. A small copy of each photo leaves this server, and what that service keeps or trains on is set by its terms, not by Photo sweep.`
    : `Labelling is off: the model is set to ${m.host}, outside this network, so photos would leave this server. Set PHOTOS_LEAVE_THIS_SERVER=yes to allow it, or point the model at a server of your own.`;
}

function renderSummary() {
  const s = state.summary;
  if (!s) return;
  const p = s.progress;
  // Fingerprinting (no model) comes first; labelling follows when there is a model.
  const printing = p.fingerprinted < p.total;
  const done = s.labelling ? p.labelled : p.fingerprinted;
  const pct = p.total ? Math.round((done / p.total) * 100) : 0;
  let line;
  if (!s.labelling) {
    line = `${fmt(p.fingerprinted)} of ${fmt(p.total)} fingerprinted (${pct}%) · no model, so duplicates only`;
    line += s.paused ? " · paused" : printing ? "" : " · scan complete";
  } else if (printing) {
    line = `Fingerprinting ${fmt(p.fingerprinted)} of ${fmt(p.total)} for duplicates, then labelling`;
    if (s.paused) line += " · paused";
  } else {
    const verb = s.visionModel ? "labelled" : "sorted on this computer";
    line = `${fmt(p.labelled)} of ${fmt(p.total)} ${verb} (${pct}%)`;
    if (s.paused) line += " · paused";
    else if (p.pending && p.perHour) line += ` · about ${fmt(p.perHour)} an hour · ${p.etaHours} h left`;
    else if (p.pending) line += " · waiting for the model";
    else if (!(s.questions && s.questions.left)) line += " · scan complete";
  }
  const q = s.questions;
  if (s.visionModel && !printing && !p.pending && q && q.left) {
    // labelling is done; the blur and clutter questions (re-asks included) are not
    line += q.rechecks ? ` · re-asking ${fmt(q.rechecks)} for clutter` : " · checking blur and clutter";
    if (!s.paused && q.perHour) line += `, about ${fmt(q.perHour)} an hour, ${q.etaHours} h left`;
  }
  if (p.deleted) line += ` · ${fmt(p.deleted)} deleted`;
  const b = s.blurCheck;
  if (b && b.total && b.checked < b.total) line += ` · blur check ${fmt(b.checked)} of ${fmt(b.total)} photos`;
  const c = s.clutterCheck;
  // re-asks are already on the line above; this is the first pass
  const firstPass = c ? c.total - c.checked - ((s.questions && s.questions.rechecks) || 0) : 0;
  if (c && c.total && firstPass > 0) line += ` · clutter check ${fmt(c.total - firstPass)} of ${fmt(c.total)}`;
  $("progress-line").textContent = line;
  $("progress-bar").style.width = `${pct}%`;
  const pause = $("pause-btn");
  pause.setAttribute("aria-pressed", String(s.paused));
  pause.textContent = s.paused ? "Resume scan" : "Pause scan";
  $("nc-btn").textContent = s.nextcloudReady ? "Nextcloud connected" : "Connect Nextcloud";
  $("nc-btn").hidden = !s.nextcloudSetUp;
  $("signout-btn").hidden = !s.signOut;
  renderModelNote(s.modelRemote);
  renderTiers(s.dupGroups);
  syncDupControls();
  for (const t of TABS) {
    const el = document.querySelector(`[data-tab="${t.key}"] .tab-count`);
    if (el) el.textContent = fmt(s.counts[t.key] || 0);
  }
  $("recheck-btn").hidden = !canRecheck(state.tab);
  const shown = visibleTabs();
  for (const el of document.querySelectorAll(".tab")) el.hidden = !shown.some((t) => t.key === el.dataset.tab);
  if (!shown.some((t) => t.key === state.tab)) switchTab("dupes");
}

// --------------------------------------------------------------- tabs ---
function renderTabs() {
  const wrap = $("tabs");
  wrap.replaceChildren(...TABS.map((t) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "tab";
    b.setAttribute("role", "tab");
    b.dataset.tab = t.key;
    b.setAttribute("aria-selected", String(t.key === state.tab));
    b.setAttribute("aria-controls", "grid");
    const name = document.createElement("span");
    name.textContent = t.label;
    const count = document.createElement("span");
    count.className = "tab-count";
    count.textContent = "–";
    b.append(name, count);
    b.addEventListener("click", () => switchTab(t.key));
    return b;
  }));
  wrap.addEventListener("keydown", (e) => {
    if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
    const tabs = visibleTabs();
    const i = tabs.findIndex((t) => t.key === state.tab);
    const next = tabs[(i + (e.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length];
    switchTab(next.key);
    document.querySelector(`[data-tab="${next.key}"]`).focus();
  });
}

function switchTab(key) {
  state.tab = key;
  for (const el of document.querySelectorAll(".tab")) el.setAttribute("aria-selected", String(el.dataset.tab === key));
  const t = TABS.find((x) => x.key === key);
  $("tab-title").textContent = t.label;
  $("tab-note").textContent = t.note + " Click to select, Shift-click for a range.";
  $("conf-seg").hidden = !usesConfidence(key);
  $("source-seg").hidden = key === "error" || key === "dupes";
  $("exposure-seg").hidden = key !== "blurry";
  $("year-pick").hidden = key === "error" || key === "dupes";
  $("tier-seg").hidden = key !== "dupes";
  syncDupControls();
  $("grid").classList.toggle("grid-groups", key === "dupes");
  $("keep-btn").textContent = key === "kept" ? "Un-keep" : "Keep";
  $("recheck-btn").hidden = !canRecheck(key);
  // Keep acts on labelled files only; Delete also clears unreadable ones from Errors
  $("keep-btn").hidden = key === "error";
  resetGrid();
}

// Confidence is the model's; Kept, Errors and Blurry are not ranked by it.
function usesConfidence(key) { return !["kept", "error", "blurry", "clutter", "dupes"].includes(key); }

// Duplicates: Same offers Select all extras (every group, loaded or not);
// Similar offers no bulk select at all, because its groups are different
// photos. Bulk-selecting them deleted 886 burst frames on 2026-11-21.
function syncDupControls() {
  const dupes = state.tab === "dupes";
  const n = state.summary && state.summary.dupExtras ? state.summary.dupExtras.same : null;
  $("select-shown").hidden = dupes;
  const mirrorMode = dupes && state.tier === "similar" && state.mirrored;
  const m = state.summary && state.summary.dupExtras ? state.summary.dupExtras.mirrored : null;
  $("select-all-extras").hidden = !(dupes && (state.tier === "same" || mirrorMode));
  $("select-all-extras").textContent = mirrorMode
    ? (m != null ? `Select all ${fmt(m)} mirrored ${m === 1 ? "copy" : "copies"}` : "Select all mirrored copies")
    : (n != null ? `Select all ${fmt(n)} extras` : "Select all extras");
  $("mirror-only").hidden = !(dupes && state.tier === "similar");
  $("mirror-only").setAttribute("aria-pressed", String(state.mirrored));
}

async function selectAllExtras() {
  const btn = $("select-all-extras");
  btn.disabled = true;
  try {
    const mirror = state.tier === "similar" && state.mirrored;
    const data = await api(mirror ? "/api/dupes/extras?tier=similar&mirrored=1" : "/api/dupes/extras?tier=same");
    state.selected = new Set(data.ids);
    state.bulk = mirror ? "mirrored" : "same";
    state.anchor = null;
    for (const item of state.items) markTile(item.id, state.selected.has(item.id));
    updateSelection();
    toast(mirror
      ? `Selected ${fmt(data.count)} mirrored copies. Each group keeps its best copy, and burst frames are left alone.`
      : `Selected ${fmt(data.count)} extras across every Same group. Each group keeps its best copy.`);
  } catch (err) {
    toast(err.message);
  } finally {
    btn.disabled = false;
  }
}

function renderTiers(counts) {
  for (const b of document.querySelectorAll("#tier-seg button")) {
    const label = b.dataset.tier === "same" ? "Same" : "Similar";
    b.textContent = counts ? `${label} ${fmt(counts[b.dataset.tier] || 0)}` : label;
    b.setAttribute("aria-pressed", String(b.dataset.tier === state.tier));
  }
}

function renderExposures(counts) {
  for (const b of document.querySelectorAll("#exposure-seg button")) {
    const ex = EXPOSURES.find((x) => x.key === b.dataset.exposure);
    b.textContent = counts ? `${ex.label} ${fmt(counts[ex.key] || 0)}` : ex.label;
    b.setAttribute("aria-pressed", String(ex.key === state.exposure));
  }
}

// The year filter's choices, rebuilt from the first page's counts only: a
// rebuild while scrolling would close the list under the pointer. A chosen
// year with nothing in this tab stays listed, at 0, so the choice holds.
function renderYears(counts) {
  const sel = $("year-select");
  const all = Object.values(counts).reduce((a, b) => a + b, 0);
  const years = Object.keys(counts).filter((k) => k !== "none");
  if (/^\d{4}$/.test(state.year) && !years.includes(state.year)) years.push(state.year);
  years.sort((a, b) => b.localeCompare(a));
  const opts = [["all", `All years · ${fmt(all)}`], ...years.map((y) => [y, `${y} · ${fmt(counts[y] || 0)}`])];
  if (counts.none || state.year === "none") opts.push(["none", `No date · ${fmt(counts.none || 0)}`]);
  sel.replaceChildren(...opts.map(([value, text]) => {
    const o = document.createElement("option");
    o.value = value;
    o.textContent = text;
    return o;
  }));
  sel.value = state.year;
}

function renderSources(counts) {
  for (const b of document.querySelectorAll("#source-seg button")) {
    const src = SOURCES.find((x) => x.key === b.dataset.source);
    b.textContent = counts ? `${src.label} ${fmt(counts[src.key] || 0)}` : src.label;
    b.setAttribute("aria-pressed", String(src.key === state.source));
  }
}

function resetGrid() {
  state.gen += 1;
  // A superseded request returns without clearing `loading` (its finally only
  // clears for the current generation), so the reset must. Without this the
  // observer's boot-time load, overtaken by the first switchTab, left the grid
  // empty for good (2026-11-17).
  state.loading = false;
  state.items = [];
  state.groups = 0;
  state.total = 0;
  state.done = false;
  state.selected.clear();
  state.bulk = false;
  state.anchor = null;
  $("grid").replaceChildren();
  updateSelection();
  loadMore();
}

// --------------------------------------------------------------- grid ---
async function loadMore() {
  if (state.loading || state.done) return;
  state.loading = true;
  const gen = state.gen;
  $("grid").setAttribute("aria-busy", "true");
  try {
    if (state.tab === "dupes") { await loadGroups(gen); return; }
    const minConf = usesConfidence(state.tab) ? state.minConf : 0;
    const source = state.tab === "error" ? "all" : state.source;
    const exposure = state.tab === "blurry" ? state.exposure : "all";
    const year = state.tab === "error" ? "all" : state.year;
    const q = new URLSearchParams({ tab: state.tab, min_conf: minConf, source, exposure, year, offset: state.items.length, limit: PAGE });
    const data = await api(`/api/items?${q}`);
    if (gen !== state.gen) return; // a tab switch superseded this page
    const start = state.items.length;
    state.items.push(...data.items);
    state.total = data.total;
    renderSources(data.sources);
    if (state.tab === "blurry") renderExposures(data.exposures);
    if (start === 0) renderYears(data.years);
    state.done = state.items.length >= data.total || data.items.length === 0;
    const frag = document.createDocumentFragment();
    data.items.forEach((item, k) => frag.append(tile(item, start + k)));
    $("grid").append(frag);
    renderEmpty();
  } catch (err) {
    toast(`Could not load images: ${err.message}`);
  } finally {
    if (gen === state.gen) {
      state.loading = false;
      $("grid").setAttribute("aria-busy", "false");
      // a short first page can leave the sentinel on screen; keep filling
      requestAnimationFrame(() => { if (sentinelVisible() && !state.done) loadMore(); });
    }
  }
}

// Duplicates page by group, so a group never splits across pages. Each item
// carries its group's number (gid) and whether it is that group's best copy.
const GROUP_PAGE = 30;
async function loadGroups(gen) {
  const q = new URLSearchParams({ tier: state.tier, offset: state.groups, limit: GROUP_PAGE });
  if (state.tier === "similar" && state.mirrored) q.set("mirrored", "1");
  const data = await api(`/api/dupes?${q}`);
  if (gen !== state.gen) return;
  const start = state.items.length;
  const base = state.items.length ? state.items[state.items.length - 1].gid + 1 : 0;
  data.groups.forEach((g, k) => g.items.forEach((item) => state.items.push({ ...item, gid: base + k, tier: g.tier })));
  state.groups += data.groups.length;
  state.total = data.total;
  renderTiers({ same: data.counts.same, similar: data.counts.similar });
  $("mirror-only").textContent = `Mirrored only ${fmt(data.counts.mirrored || 0)}`;
  state.done = state.groups >= data.total || data.groups.length === 0;
  $("grid").append(groupBlocks(state.items.slice(start), start));
  renderEmpty();
}

function groupBlocks(items, start) {
  const frag = document.createDocumentFragment();
  let inner = null, gid = null;
  items.forEach((item, k) => {
    if (item.gid !== gid) {
      gid = item.gid;
      const members = items.filter((x) => x.gid === gid);
      const block = document.createElement("li");
      block.className = "dup-group";
      const head = document.createElement("p");
      head.className = "dup-head";
      const n = document.createElement("strong");
      n.textContent = `${members.length} ${item.tier === "same" ? "copies" : "alike"}`;
      head.append(n, ` · ${baseName(members[0].rel)} is the best copy`);
      inner = document.createElement("ul");
      inner.className = "grid dup-tiles";
      block.append(head, inner);
      frag.append(block);
    }
    inner.append(tile(item, start + k));
  });
  return frag;
}

// After a delete or Keep: drop groups with one file left, or with every file
// kept (settled), and redraw. The server applies the same rule, so the next
// page's offset (state.groups) still lines up.
function regroup() {
  const by = new Map();
  for (const item of state.items) (by.get(item.gid) || by.set(item.gid, []).get(item.gid)).push(item);
  const live = [...by.values()].filter((g) => g.length > 1 && !g.every((x) => x.kept));
  const dropped = by.size - live.length;
  state.items = live.flat();
  state.groups = Math.max(0, state.groups - dropped);
  state.total = Math.max(0, state.total - dropped);
  const ids = new Set(state.items.map((x) => x.id));
  for (const id of [...state.selected]) if (!ids.has(id)) state.selected.delete(id);
  $("grid").replaceChildren(groupBlocks(state.items, 0));
  for (const id of state.selected) markTile(id, true);
}

function sentinelVisible() {
  const r = $("sentinel").getBoundingClientRect();
  return r.top < window.innerHeight + 600;
}

function renderEmpty() {
  const empty = state.items.length === 0;
  $("empty").hidden = !empty;
  if (!empty) return;
  const scanning = state.summary && state.summary.progress.pending > 0;
  $("empty-title").textContent = state.tab === "kept" ? "Nothing kept yet" : state.tab === "error" ? "No errors"
    : state.tab === "dupes" ? "No duplicates here" : "Nothing in this tab";
  if (state.tab === "dupes") {
    $("empty-body").textContent = state.tier === "same"
      ? "No two files hold the same picture, or every group is settled. Try Similar for burst shots."
      : "No burst shots or near-repeats left to choose between.";
    return;
  }
  $("empty-body").textContent = state.tab === "kept"
    ? "Select images in another tab and press Keep to hide them from the sweep for good."
    : state.minConf > 0 && usesConfidence(state.tab)
      ? "Nothing reaches that confidence. Try Any."
      : state.year !== "all" && state.tab !== "error"
        ? "Nothing from that year in this tab. Try All years."
      : state.tab === "blurry" && state.exposure !== "all"
        ? "No blurry photos like that. Try Any."
      : state.source !== "all" && state.tab !== "error"
        ? "Nothing from that source in this tab. Try All."
      : scanning ? "The scan is still running; new images appear here as they are labelled." : "The scan found nothing of this kind.";
}

const CHECK_SVG = '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3 8.5l3.2 3L13 4.5" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/></svg>';
const ZOOM_SVG = '<svg viewBox="0 0 20 20" aria-hidden="true"><circle cx="8.5" cy="8.5" r="5.5" fill="none" stroke="currentColor" stroke-width="2"/><path d="M13 13l4.5 4.5" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>';

function baseName(rel) { return rel.slice(rel.lastIndexOf("/") + 1); }
function fmtBytes(n) { return n >= 1e6 ? `${(n / 1e6).toFixed(1)} MB` : `${Math.max(1, Math.round(n / 1e3))} KB`; }
function dupBadge(item) {
  const dims = item.width && item.height ? `${item.width}×${item.height}` : fmtBytes(item.size);
  const flip = item.mirror ? "mirrored · " : "";
  return item.keeper ? `best · ${dims}` : item.kept ? `kept · ${flip}${dims}` : `${flip}${dims}`;
}

function tile(item, index) {
  const li = document.createElement("li");
  li.className = "tile";
  li.dataset.id = item.id;
  const name = baseName(item.rel);

  const hit = document.createElement("button");
  hit.type = "button";
  hit.className = "tile-hit";
  hit.setAttribute("aria-pressed", "false");
  hit.setAttribute("aria-label", `${name}, ${item.category || "error"}${item.confidence != null ? ` ${item.confidence}%` : ""}. Select`);

  const fallback = document.createElement("span");
  fallback.className = "fallback";
  fallback.textContent = item.error ? `${name}: ${item.error}` : name;
  hit.append(fallback);

  if (state.tab !== "error") {
    const img = document.createElement("img");
    img.loading = "lazy";
    img.decoding = "async";
    img.alt = "";
    img.src = `/thumb/${item.id}`;
    img.addEventListener("error", () => img.remove());
    const scrim = document.createElement("span");
    scrim.className = "scrim";
    const badge = document.createElement("span");
    badge.className = "badge";
    // On Blurry the label confidence ("95% a photo") answers the wrong question.
    // On Clutter likewise: the badge is the clutter verdict's.
    badge.textContent = state.tab === "dupes" ? dupBadge(item)
      : state.tab === "clutter" ? `clutter ${item.clutter_conf}%`
      : state.tab !== "blurry" ? `${item.confidence}%`
      : item.blurry === 1 ? `blurry ${item.blur_conf}%` : `pixel ${Math.round(item.sharpness ?? 0)}`;
    if (item.name_hint && state.tab !== "dupes") {
      const h = document.createElement("span");
      h.className = "hint";
      h.textContent = item.name_hint;
      badge.append(h);
    }
    hit.append(img, scrim, badge);
    hit.title = state.tab === "dupes" ? `${item.rel} · ${fmtBytes(item.size)}`
      : (state.tab === "clutter" ? item.clutter_reason : item.reason) || "";
  }
  const check = document.createElement("span");
  check.className = "check";
  check.innerHTML = CHECK_SVG;
  hit.append(check);
  hit.addEventListener("click", (e) => onTileClick(index, e.shiftKey));
  li.append(hit);

  if (state.tab !== "error") {
    const open = document.createElement("button");
    open.type = "button";
    open.className = "tile-open";
    open.setAttribute("aria-label", `Open a larger view of ${name}`);
    open.innerHTML = ZOOM_SVG;
    open.addEventListener("click", () => openViewer(index));
    li.append(open);
  }
  return li;
}

// ---------------------------------------------------------- selection ---
function setSelected(index, on) {
  const item = state.items[index];
  if (!item) return;
  if (on) state.selected.add(item.id); else state.selected.delete(item.id);
  markTile(item.id, on);
}

// By id, not by position: on Duplicates the tiles sit inside group blocks.
function markTile(id, on) {
  const li = $("grid").querySelector(`li.tile[data-id="${id}"]`);
  if (li) {
    li.classList.toggle("selected", on);
    li.querySelector(".tile-hit").setAttribute("aria-pressed", String(on));
  }
}

function onTileClick(index, shift) {
  const item = state.items[index];
  if (shift && state.anchor !== null) {
    const on = !state.selected.has(item.id) || state.selected.has(state.items[state.anchor].id);
    const [a, b] = state.anchor < index ? [state.anchor, index] : [index, state.anchor];
    for (let i = a; i <= b; i++) setSelected(i, on);
  } else {
    setSelected(index, !state.selected.has(item.id));
  }
  state.anchor = index;
  updateSelection();
}

function updateSelection() {
  const n = state.selected.size;
  $("sel-count").textContent = fmt(n);
  $("actionbar").hidden = n === 0;
  announce(n ? `${n} selected` : "");
}

function selectShown() {
  // On Duplicates, everything but each group's best copy and what you kept
  state.items.forEach((item, i) => { if (state.tab !== "dupes" || (!item.keeper && !item.kept)) setSelected(i, true); });
  updateSelection();
}

function clearSelection() {
  state.items.forEach((_, i) => setSelected(i, false));
  state.selected.clear();
  state.bulk = false;
  state.anchor = null;
  updateSelection();
}

function removeIds(ids) {
  const drop = new Set(ids);
  if (state.tab === "dupes") {
    state.items = state.items.filter((x) => !drop.has(x.id));
    for (const id of drop) state.selected.delete(id);
    state.anchor = null;
    regroup();
    updateSelection();
    renderEmpty();
    if (sentinelVisible()) loadMore();
    return;
  }
  const keep = [];
  const grid = $("grid");
  const children = Array.from(grid.children);
  state.items.forEach((item, i) => { if (drop.has(item.id)) children[i].remove(); else keep.push(item); });
  state.items = keep;
  state.total = Math.max(0, state.total - drop.size);
  // indexes changed: rebind click handlers by rebuilding from the kept items
  grid.replaceChildren(...state.items.map((item, i) => {
    const li = tile(item, i);
    if (state.selected.has(item.id)) { li.classList.add("selected"); li.querySelector(".tile-hit").setAttribute("aria-pressed", "true"); }
    return li;
  }));
  for (const id of drop) state.selected.delete(id);
  state.anchor = null;
  updateSelection();
  renderEmpty();
  if (sentinelVisible()) loadMore();
}

// ------------------------------------------------------------ actions ---
async function doRecheck() {
  const ids = Array.from(state.selected);
  const btn = $("recheck-btn");
  btn.disabled = true;
  try {
    let relabel = 0, blur = 0;
    for (let i = 0; i < ids.length; i += DELETE_CHUNK) {
      const r = await api("/api/recheck", { ids: ids.slice(i, i + DELETE_CHUNK) });
      relabel += r.relabel; blur += r.blur;
    }
    if (state.tab === "error") removeIds(ids);  // back in the queue; they leave Errors now
    else { state.selected.clear(); state.anchor = null; updateSelection(); switchTab(state.tab); }
    toast(state.tab === "error"
      ? `Sent ${fmt(relabel)} back to be labelled again. They reappear in their tab once done.`
      : `Rechecking ${fmt(blur)} for clutter and blur, ahead of the queue. Each takes a few seconds; they move tabs when done.`);
    refreshSummary();
  } catch (err) {
    toast(err.message);
  } finally {
    btn.disabled = false;
  }
}

async function doKeep() {
  const ids = Array.from(state.selected);
  const keepFlag = state.tab !== "kept";
  try {
    for (let i = 0; i < ids.length; i += DELETE_CHUNK) {
      await api("/api/keep", { ids: ids.slice(i, i + DELETE_CHUNK), keep: keepFlag });
    }
    if (state.tab === "dupes") {
      const kept = new Set(ids);
      state.items = state.items.map((x) => (kept.has(x.id) ? { ...x, kept: 1 } : x));
      state.selected.clear();
      state.anchor = null;
      if (state.bulk) { state.bulk = false; resetGrid(); } else { regroup(); updateSelection(); renderEmpty(); }
      toast(`Kept ${fmt(ids.length)}. A group leaves this tab once every file in it is kept or gone.`);
      refreshSummary();
      return;
    }
    removeIds(ids);
    toast(keepFlag ? `Kept ${fmt(ids.length)}. They will not come back here.` : `Returned ${fmt(ids.length)} to the sweep.`);
    refreshSummary();
  } catch (err) {
    toast(err.message);
  }
}

function askDelete() {
  if (!state.summary || !state.summary.nextcloudReady) { openNextcloud(); return; }
  $("confirm-count").textContent = fmt(state.selected.size);
  // Deleting every file of a group deletes the picture, not a duplicate of it.
  const warn = $("confirm-warn");
  let whole = 0;
  if (state.tab === "dupes") {
    const by = new Map();
    for (const x of state.items) (by.get(x.gid) || by.set(x.gid, []).get(x.gid)).push(x);
    for (const g of by.values()) if (g.every((x) => state.selected.has(x.id))) whole += 1;
  }
  const note = $("confirm-note");
  note.hidden = !state.bulk;
  // about 3.7 deletes a second, measured from the audit log on 2026-11-21
  const minutes = Math.max(1, Math.round(state.selected.size / 3.7 / 60));
  note.textContent = state.bulk
    ? `Every Same group keeps its best copy: the server checks each file again before deleting it. This takes about ${minutes} min; keep this tab open. Anything not reached stays.`
    : "";
  warn.hidden = whole === 0;
  warn.textContent = whole ? `${fmt(whole)} ${whole === 1 ? "group has" : "groups have"} every copy selected, so that picture would go entirely.` : "";
  $("confirm").showModal();
}

async function doDelete() {
  const ok = $("confirm-ok");
  ok.disabled = true;
  ok.textContent = "Deleting…";
  const ids = Array.from(state.selected);
  const bulk = state.bulk;
  const done = [];
  let failed = 0, skipped = 0;
  try {
    for (let i = 0; i < ids.length; i += DELETE_CHUNK) {
      const chunk = ids.slice(i, i + DELETE_CHUNK);
      const res = await api(bulk ? "/api/dupes/delete-extras" : "/api/delete",
        bulk ? { ids: chunk, kind: bulk } : { ids: chunk });
      const failedIds = new Set([...res.failedIds, ...(res.skippedIds || [])]);
      failed += res.failed;
      skipped += res.skipped || 0;
      for (const id of ids.slice(i, i + DELETE_CHUNK)) if (!failedIds.has(id)) done.push(id);
      ok.textContent = `Deleting… ${fmt(done.length)} of ${fmt(ids.length)}`;
    }
  } catch (err) {
    toast(err.message);
    if (err.code === "NEXTCLOUD_NOT_CONFIGURED") openNextcloud();
  } finally {
    ok.disabled = false;
    ok.textContent = "Move to Deleted files";
    $("confirm").close();
  }
  if (bulk) {
    resetGrid();  // most of what went was never loaded; redraw from the server
  } else if (done.length) {
    removeIds(done);
  }
  if (done.length || failed || skipped) {
    let msg = `Moved ${fmt(done.length)} to Deleted files.`;
    if (skipped) msg += bulk === "mirrored"
      ? ` ${fmt(skipped)} skipped: no longer a mirrored copy (now a best copy, kept, or its pair is gone).`
      : ` ${fmt(skipped)} skipped: no longer an extra (now a best copy, kept, or not in a Same group).`;
    if (failed) msg += ` ${fmt(failed)} could not be deleted${bulk ? "" : " and are still selected"}.`;
    toast(msg);
  }
  refreshSummary();
}

async function togglePause() {
  const paused = !(state.summary && state.summary.paused);
  try {
    await api("/api/pause", { paused });
    await refreshSummary();
  } catch (err) { toast(err.message); }
}

function openNextcloud() {
  $("nc-msg").textContent = "";
  $("nc-password").value = "";
  $("nc-dialog").showModal();
}

async function saveNextcloud() {
  const btn = $("nc-save");
  btn.disabled = true;
  $("nc-msg").textContent = "";
  try {
    await api("/api/nextcloud", { password: $("nc-password").value });
    $("nc-dialog").close();
    toast("Nextcloud connected. Deletes now go to its Deleted files.");
    refreshSummary();
  } catch (err) {
    $("nc-msg").textContent = err.message;
  } finally {
    btn.disabled = false;
    $("nc-password").value = "";
  }
}

// ------------------------------------------------------------- viewer ---
function openViewer(index) {
  state.viewerIndex = index;
  renderViewer();
  if (!$("viewer").open) $("viewer").showModal();
}

function renderViewer() {
  const item = state.items[state.viewerIndex];
  if (!item) { $("viewer").close(); return; }
  const img = $("viewer-img");
  img.src = `/view/${item.id}`;
  img.alt = `Larger view of ${baseName(item.rel)}`;
  // a fingerprinted picture can be opened from Duplicates before it is sorted
  $("viewer-cat").textContent = item.category ? `${item.category} · ${item.confidence}%` : "not sorted yet";
  $("viewer-name").textContent = baseName(item.rel);
  $("viewer-reason").textContent = item.label_src === "classifier" ? "" : item.reason || "";
  const meta = $("viewer-meta");
  const rows = [["Folder", item.rel.includes("/") ? item.rel.slice(0, item.rel.lastIndexOf("/")) : "Pics"]];
  if (item.width && item.height) rows.push(["Size", `${item.width} × ${item.height}`]);
  rows.push(["Camera", item.camera || "none recorded"]);
  if (item.category) rows.push(["Sorted by", item.label_src === "classifier" ? "the image classifier, on this computer" : "the vision model"]);
  if (item.name_hint) rows.push(["Name hint", item.name_hint]);
  if (item.category === "photo") rows.push(["Clutter check", clutterText(item)], ["Blur check", blurText(item)]);
  if (item.sharpness != null) rows.push(["Pixel sharpness", `${Math.round(item.sharpness)} (blurry under ${state.summary?.blurMax ?? "?"})`]);
  if (item.mtime) rows.push(["Modified", item.mtime.slice(0, 10)]);
  if (state.tab === "dupes") {
    const best = state.items.find((x) => x.gid === item.gid && x.keeper);
    rows.push(["File size", fmtBytes(item.size)]);
    rows.push(["Duplicate", item.keeper ? "the best copy in its group"
      : `${item.tier === "same" ? "a copy" : item.mirror ? "a mirrored copy" : "a near-repeat"} of ${best ? baseName(best.rel) : "another file"}`]);
  }
  meta.replaceChildren(...rows.flatMap(([k, v]) => {
    const dt = document.createElement("dt"); dt.textContent = k;
    const dd = document.createElement("dd"); dd.textContent = v;
    return [dt, dd];
  }));
  const on = state.selected.has(item.id);
  const sel = $("viewer-select");
  sel.setAttribute("aria-pressed", String(on));
  sel.textContent = on ? "Selected" : "Select";
  $("viewer-prev").disabled = state.viewerIndex <= 0;
  $("viewer-next").disabled = state.viewerIndex >= state.items.length - 1;
}

function viewerStep(delta) {
  const next = state.viewerIndex + delta;
  if (next < 0 || next >= state.items.length) return;
  state.viewerIndex = next;
  renderViewer();
  if (next >= state.items.length - 5) loadMore();
}

function viewerToggle() {
  const item = state.items[state.viewerIndex];
  if (!item) return;
  setSelected(state.viewerIndex, !state.selected.has(item.id));
  state.anchor = state.viewerIndex;
  updateSelection();
  renderViewer();
}

function closeOnBackdrop(dialog) {
  dialog.addEventListener("click", (e) => {
    const r = dialog.getBoundingClientRect();
    const inside = e.clientX >= r.left && e.clientX <= r.right && e.clientY >= r.top && e.clientY <= r.bottom;
    if (!inside) dialog.close();
  });
}

// Picture size, remembered on this device only.
const SIZE_KEY = "photosweep.size";
function setSize(size) {
  if (!["s", "m", "l"].includes(size)) size = "m";
  document.body.dataset.size = size;
  for (const b of document.querySelectorAll("#size-seg button")) b.setAttribute("aria-pressed", String(b.dataset.size === size));
  try { localStorage.setItem(SIZE_KEY, size); } catch (err) { console.warn("size not saved", err.message); }
}
function savedSize() {
  try { return localStorage.getItem(SIZE_KEY); } catch (err) { console.warn("size not read", err.message); return null; }
}

// ---------------------------------------------------------------- boot ---
function boot() {
  setSize(savedSize());
  for (const b of document.querySelectorAll("#size-seg button")) b.addEventListener("click", () => setSize(b.dataset.size));
  renderTabs();
  renderSources(null);
  renderExposures(null);
  $("year-select").addEventListener("change", (e) => {
    state.year = e.target.value;
    resetGrid();
  });
  for (const b of document.querySelectorAll("#exposure-seg button")) {
    b.addEventListener("click", () => {
      state.exposure = b.dataset.exposure;
      renderExposures(null);
      resetGrid();
    });
  }
  for (const b of document.querySelectorAll("#source-seg button")) {
    b.addEventListener("click", () => {
      state.source = b.dataset.source;
      renderSources(null);
      resetGrid();
    });
  }
  for (const b of document.querySelectorAll("#tier-seg button")) {
    b.addEventListener("click", () => {
      state.tier = b.dataset.tier;
      state.mirrored = false;
      renderTiers(state.summary && state.summary.dupGroups);
      syncDupControls();
      resetGrid();
    });
  }
  for (const b of document.querySelectorAll("#conf-seg button")) {
    b.addEventListener("click", () => {
      state.minConf = Number(b.dataset.conf);
      for (const x of document.querySelectorAll("#conf-seg button")) x.setAttribute("aria-pressed", String(x === b));
      resetGrid();
    });
  }
  $("select-shown").addEventListener("click", selectShown);
  $("select-all-extras").addEventListener("click", selectAllExtras);
  $("mirror-only").addEventListener("click", () => {
    state.mirrored = !state.mirrored;
    syncDupControls();
    resetGrid();
  });
  $("clear-sel").addEventListener("click", clearSelection);
  $("keep-btn").addEventListener("click", doKeep);
  $("recheck-btn").addEventListener("click", doRecheck);
  $("delete-btn").addEventListener("click", askDelete);
  $("confirm-cancel").addEventListener("click", () => $("confirm").close());
  $("confirm-ok").addEventListener("click", doDelete);
  $("pause-btn").addEventListener("click", togglePause);
  $("nc-btn").addEventListener("click", openNextcloud);
  $("signout-btn").addEventListener("click", async () => {
    try { await api("/api/logout", {}); } catch (err) { console.warn("sign out", err.message); }
    location.assign("/login");
  });
  $("nc-cancel").addEventListener("click", () => $("nc-dialog").close());
  $("nc-save").addEventListener("click", saveNextcloud);
  $("nc-password").addEventListener("keydown", (e) => { if (e.key === "Enter") saveNextcloud(); });
  $("viewer-prev").addEventListener("click", () => viewerStep(-1));
  $("viewer-next").addEventListener("click", () => viewerStep(1));
  $("viewer-select").addEventListener("click", viewerToggle);
  $("viewer-close").addEventListener("click", () => $("viewer").close());
  $("viewer").addEventListener("keydown", (e) => {
    if (e.key === "ArrowRight") { e.preventDefault(); viewerStep(1); }
    else if (e.key === "ArrowLeft") { e.preventDefault(); viewerStep(-1); }
    else if (e.key === "x" || e.key === "X") { e.preventDefault(); viewerToggle(); }
  });
  for (const d of ["viewer", "confirm", "nc-dialog"]) closeOnBackdrop($(d));

  new IntersectionObserver((entries) => {
    if (entries.some((e) => e.isIntersecting)) loadMore();
  }, { rootMargin: "600px" }).observe($("sentinel"));

  let timer = null;
  const startPolling = () => { if (!timer) timer = setInterval(refreshSummary, POLL_MS); };
  const stopPolling = () => { clearInterval(timer); timer = null; };
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) stopPolling(); else { refreshSummary(); startPolling(); }
  });
  refreshSummary().then(() => switchTab(state.tab));
  startPolling();
}

document.addEventListener("DOMContentLoaded", boot);
