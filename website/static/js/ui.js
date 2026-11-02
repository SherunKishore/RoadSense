// Shared UI helpers: escaping, icons, badges, formatting, toasts and modals.

export const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

const P = (d) => `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${d}</svg>`;
export const icon = {
  overview: P('<rect x="3" y="3" width="7" height="9" rx="1.5"/><rect x="14" y="3" width="7" height="5" rx="1.5"/><rect x="14" y="12" width="7" height="9" rx="1.5"/><rect x="3" y="16" width="7" height="5" rx="1.5"/>'),
  plus: P('<path d="M12 5v14M5 12h14"/>'),
  camera: P('<path d="M14.5 4h-5L7.5 6.5H4a1 1 0 0 0-1 1V18a1 1 0 0 0 1 1h16a1 1 0 0 0 1-1V7.5a1 1 0 0 0-1-1h-3.5z"/><circle cx="12" cy="12.5" r="3.5"/>'),
  map: P('<path d="M9 4 3.5 6v14L9 18l6 2 5.5-2V4L15 6z"/><path d="M9 4v14M15 6v14"/>'),
  list: P('<path d="M8 6h13M8 12h13M8 18h13"/><circle cx="3.5" cy="6" r="1"/><circle cx="3.5" cy="12" r="1"/><circle cx="3.5" cy="18" r="1"/>'),
  layers: P('<path d="m12 3 9 5-9 5-9-5z"/><path d="m3 13 9 5 9-5"/>'),
  upload: P('<path d="M12 16V4M7 9l5-5 5 5"/><path d="M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3"/>'),
  flask: P('<path d="M9 3h6M10 3v6L4.5 18.5A1.6 1.6 0 0 0 5.9 21h12.2a1.6 1.6 0 0 0 1.4-2.5L14 9V3"/><path d="M7 15h10"/>'),
  info: P('<circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 7.5v.5"/>'),
  users: P('<circle cx="9" cy="8" r="3.5"/><path d="M2.5 20c.8-3.5 3.4-5.5 6.5-5.5s5.7 2 6.5 5.5"/><path d="M16 4.5a3.5 3.5 0 0 1 0 7M18 14.8c1.8.8 3 2.6 3.5 5.2"/>'),
  logout: P('<path d="M15 4h3a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2h-3"/><path d="M10 17l5-5-5-5M15 12H3"/>'),
  sun: P('<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>'),
  moon: P('<path d="M20 14.5A8 8 0 1 1 9.5 4a6.5 6.5 0 0 0 10.5 10.5z"/>'),
  locate: P('<circle cx="12" cy="12" r="7"/><circle cx="12" cy="12" r="2.5"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3"/>'),
  pin: P('<path d="M12 21s-7-6.2-7-11.5a7 7 0 0 1 14 0C19 14.8 12 21 12 21z"/><circle cx="12" cy="9.5" r="2.5"/>'),
  image: P('<rect x="3" y="4" width="18" height="16" rx="2"/><circle cx="8.5" cy="9.5" r="1.8"/><path d="m21 16-5-5L5 20"/>'),
  eye: P('<path d="M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/>'),
  download: P('<path d="M12 4v12M7 11l5 5 5-5"/><path d="M4 20h16"/>'),
  trash: P('<path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3"/>'),
  check: P('<path d="m5 12.5 4.5 4.5L19 7.5"/>'),
  x: P('<path d="M6 6l12 12M18 6 6 18"/>'),
  alert: P('<path d="M12 3.5 2.5 20h19z"/><path d="M12 10v4.5M12 17.5v.5"/>'),
  ext: P('<path d="M14 4h6v6M20 4l-9 9"/><path d="M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5"/>'),
  refresh: P('<path d="M20 12a8 8 0 1 1-2.3-5.6L20 8.5"/><path d="M20 3.5v5h-5"/>'),
  search: P('<circle cx="11" cy="11" r="6.5"/><path d="m20 20-4.2-4.2"/>'),
  printer: P('<path d="M7 9V4h10v5"/><rect x="3" y="9" width="18" height="8" rx="1.5"/><path d="M7 14h10v6H7z"/>'),
  sparkle: P('<path d="M12 3v4M12 17v4M3 12h4M17 12h4M6 6l2.5 2.5M15.5 15.5 18 18M6 18l2.5-2.5M15.5 8.5 18 6"/>'),
  road: P('<path d="M8 21 10.5 3M16 21 13.5 3"/><path d="M12 6v2M12 11v2M12 16v2"/>'),
  gauge: P('<path d="M4 18a8 8 0 1 1 16 0"/><path d="m12 18 4-6"/>'),
  shield: P('<path d="M12 3 4.5 6v6c0 4.5 3.2 7.8 7.5 9 4.3-1.2 7.5-4.5 7.5-9V6z"/><path d="m9 12 2 2 4-4"/>'),
  clock: P('<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>'),
  more: P('<circle cx="5" cy="12" r="1.2"/><circle cx="12" cy="12" r="1.2"/><circle cx="19" cy="12" r="1.2"/>'),
  route: P('<circle cx="6" cy="19" r="2.5"/><circle cx="18" cy="5" r="2.5"/><path d="M8.5 19H17a3.5 3.5 0 0 0 0-7H7a3.5 3.5 0 0 1 0-7h8.5"/>'),
  video: P('<rect x="3" y="6" width="13" height="12" rx="2"/><path d="m16 10 5-3v10l-5-3"/>'),
  key: P('<circle cx="8" cy="15" r="4"/><path d="m11 12 9-9M17 6l3 3"/>'),
};
export const logo = `<svg viewBox="0 0 32 32" aria-hidden="true"><path d="M11 26 14.5 6M21 26 17.5 6" stroke="#141413" stroke-width="2.4" stroke-linecap="round"/><path d="M16 9v2.5M16 15v2.5M16 21v2.5" stroke="#c98500" stroke-width="2.2" stroke-linecap="round"/><circle cx="22.5" cy="11" r="3.4" fill="#d03b3b" stroke="#f0efec" stroke-width="1.5"/></svg>`;

// ---------------------------------------------------------------- state shared by views
export const S = { user: null, info: null };
export const isAdmin = () => S.user?.role === "admin";
export const cls = (id) => S.info?.classes.find((c) => c.id === id || c.name === id);
export const clsLabel = (name) => cls(name)?.label ?? String(name).replace(/^D\d\d_/, "").replace(/_/g, " ");
export const SHORT_TO_NAME = { "longitudinal crack": "D00_Longitudinal_Crack", "transverse crack": "D10_Transverse_Crack", "alligator crack": "D20_Alligator_Crack", "pothole": "D40_Pothole" };
export const damageLabel = (s) => (s ? s.charAt(0).toUpperCase() + s.slice(1) : "—");
export const STATUS_LABEL = { open: "Open", in_progress: "In progress", repaired: "Repaired", rejected: "Rejected" };
export const GPS_LABEL = { exif: "Photo GPS (EXIF)", device: "Device location", map: "Map pin", gpx: "GPX track" };

export function priorityBadge(p, name) {
  if (!p) return `<span class="badge"><span class="dot"></span>No damage</span>`;
  const level = S.info?.priority_levels.find((l) => l.level === p);
  return `<span class="badge ${p}"><span class="dot"></span>${p} ${esc(name || level?.name || "")}</span>`;
}
export const statusTag = (s) => `<span class="status ${esc(s)}"><i></i>${esc(STATUS_LABEL[s] || s)}</span>`;
export const sevTag = (lv) => lv ? `<span class="sev ${esc(lv)}"><i><b></b><b></b><b></b></i>${esc(lv)}</span>` : `<span class="muted">—</span>`;
export const clsTag = (name) => { const c = cls(name); return `<span class="cls"><i style="background:${c?.color || "#888"}"></i>${esc(clsLabel(name))}</span>`; };
export const pct = (v, d = 0) => (v == null ? "—" : `${(v * 100).toFixed(d)}%`);

export function when(iso, withTime = true) {
  if (!iso) return "—";
  const d = new Date(iso);
  if (isNaN(d)) return esc(iso);
  const opts = withTime ? { day: "numeric", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit" } : { day: "numeric", month: "short", year: "numeric" };
  return d.toLocaleString(undefined, opts);
}
export function ago(iso) {
  const s = (Date.now() - new Date(iso).getTime()) / 1000;
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
  if (s < 86400 * 30) return `${Math.floor(s / 86400)} d ago`;
  return when(iso, false);
}
export const coord = (lat, lon) => (lat == null ? "—" : `${(+lat).toFixed(6)}, ${(+lon).toFixed(6)}`);
export function accuracyTag(m) {
  if (m == null) return "";
  const c = m <= 15 ? "acc-good" : m <= 50 ? "acc-ok" : "acc-bad";
  return `<span class="${c} small">± ${Math.round(m)} m</span>`;
}

// ---------------------------------------------------------------- feedback
export function toast(msg, kind = "") {
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.textContent = msg;
  $("#toasts").append(el);
  setTimeout(() => el.remove(), kind === "error" ? 6000 : 3500);
}

export function modal(title, bodyHtml, { wide = false, actions = "" } = {}) {
  const back = document.createElement("div");
  back.className = "modal-back";
  back.innerHTML = `<div class="modal ${wide ? "wide" : ""}" role="dialog" aria-modal="true" aria-label="${esc(title)}">
    <div class="card-h"><h2>${esc(title)}</h2><button class="btn ghost sm" data-close style="margin-left:auto" aria-label="Close">${icon.x}</button></div>
    <div class="card-b">${bodyHtml}</div>${actions ? `<div class="card-f row" style="justify-content:flex-end">${actions}</div>` : ""}</div>`;
  const close = () => { back.remove(); document.removeEventListener("keydown", onKey); };
  const onKey = (e) => { if (e.key === "Escape") close(); };
  back.addEventListener("click", (e) => { if (e.target === back || e.target.closest("[data-close]")) close(); });
  document.addEventListener("keydown", onKey);
  document.body.append(back);
  return { el: back, close };
}

export function confirmBox(title, text, okLabel = "Confirm", danger = false) {
  return new Promise((resolve) => {
    const m = modal(title, `<p class="ink2">${esc(text)}</p>`, {
      actions: `<button class="btn" data-close>Cancel</button><button class="btn ${danger ? "danger" : "primary"}" data-ok>${esc(okLabel)}</button>`,
    });
    m.el.querySelector("[data-ok]").onclick = () => { m.close(); resolve(true); };
    m.el.addEventListener("click", (e) => { if (e.target.closest("[data-close]")) resolve(false); });
  });
}

export function busy(btn, on, label) {
  if (on) { btn.dataset.label = btn.innerHTML; btn.disabled = true; btn.innerHTML = `<span class="spin"></span>${esc(label || "Working…")}`; }
  else { btn.disabled = false; if (btn.dataset.label) btn.innerHTML = btn.dataset.label; }
}

export const emptyState = (title, text, ic = icon.list) => `<div class="empty">${ic}<b>${esc(title)}</b>${esc(text)}</div>`;
