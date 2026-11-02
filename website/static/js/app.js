// RoadSense single-page app: session, layout and hash router.
import { api, setSignedOutHandler } from "./api.js";
import { $, $$, S, esc, icon, logo, toast } from "./ui.js";

const VIEWS = {
  overview: () => import("./views/overview.js"),
  new: () => import("./views/newreport.js"),
  map: () => import("./views/map.js"),
  reports: () => import("./views/reports.js"),
  report: () => import("./views/report.js"),
  surveys: () => import("./views/surveys.js"),
  about: () => import("./views/about.js"),
  users: () => import("./views/users.js"),
  account: () => import("./views/account.js"),
};
const NAV = [
  ["overview", "Overview", icon.overview],
  ["new", "New report", icon.camera],
  ["map", "Damage map", icon.map],
  ["reports", "Reports", icon.list],
  ["surveys", "Surveys", icon.layers],
];
const NAV2 = [["about", "About the model", icon.info]];
let cleanup = null;

function parseHash() {
  const h = location.hash.replace(/^#\/?/, "");
  const [path, qs] = h.split("?");
  const parts = path.split("/").filter(Boolean);
  let view = parts[0] || "overview";
  const params = {};
  if (view === "reports" && parts[1]) { view = "report"; params.id = +parts[1]; }
  return { view, params, query: Object.fromEntries(new URLSearchParams(qs || "")) };
}

function theme() {
  const t = document.documentElement.getAttribute("data-theme");
  return t || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
}
function toggleTheme() {
  const next = theme() === "dark" ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", next);
  try { localStorage.setItem("roadsense-theme", next); } catch (e) { /* not saved, still applied */ }
  shell();
  route();
}

function initials(name) {
  return esc(name.split(/\s+/).map((w) => w[0]).join("").slice(0, 2).toUpperCase());
}

function shell() {
  const u = S.user;
  const navLink = ([k, label, ic]) => `<a href="#/${k}" data-nav="${k}">${ic}<span>${label}</span></a>`;
  $("#root").innerHTML = `
  <div class="app">
    <aside class="sidebar" aria-label="Main navigation">
      <div class="brand"><div class="brand-mark">${logo}</div><div><b>RoadSense</b><span>Road damage survey</span></div></div>
      <nav class="nav">${NAV.map(navLink).join("")}
        <div class="nav-label">Model</div>${NAV2.map(navLink).join("")}
        ${u.role === "admin" ? `<div class="nav-label">Admin</div>${navLink(["users", "Users", icon.users])}` : ""}
      </nav>
      <div class="side-foot">
        <a class="user-card" href="#/account" style="color:inherit;text-decoration:none"><div class="avatar">${initials(u.name)}</div><div><b>${esc(u.name)}</b><span>${esc(u.role)}</span></div></a>
        <button class="side-btn" data-theme-toggle>${theme() === "dark" ? icon.sun : icon.moon}<span>${theme() === "dark" ? "Light" : "Dark"} theme</span></button>
        <button class="side-btn" data-logout>${icon.logout}<span>Sign out</span></button>
      </div>
    </aside>
    <div class="main">
      <header class="mobile-head"><div class="brand"><div class="brand-mark">${logo}</div><b>RoadSense</b></div>
        <button class="side-btn" data-theme-toggle aria-label="Toggle theme">${theme() === "dark" ? icon.sun : icon.moon}</button></header>
      <div class="topbar"><div class="grow"><div class="crumbs" id="crumbs"></div><h1 id="title"></h1></div><div class="actions" id="actions"></div></div>
      <main class="content" id="view"></main>
    </div>
    <nav class="tabbar" aria-label="Main navigation">
      <a href="#/overview" data-nav="overview">${icon.overview}Overview</a>
      <a href="#/map" data-nav="map">${icon.map}Map</a>
      <a href="#/new" data-nav="new" class="new"><span class="i">${icon.camera}</span></a>
      <a href="#/reports" data-nav="reports">${icon.list}Reports</a>
      <a href="#/more" data-nav="more">${icon.more}More</a>
    </nav>
  </div>`;
  $$("[data-theme-toggle]").forEach((b) => (b.onclick = toggleTheme));
  $("[data-logout]").onclick = async () => { await api("/api/logout", { method: "POST" }).catch(() => {}); location.hash = ""; boot(); };
}

function more(ctx) {
  ctx.setTitle("More");
  const items = [["surveys", "Surveys", icon.layers], ...NAV2, ...(S.user.role === "admin" ? [["users", "Users", icon.users]] : []),
    ["account", "Account & password", icon.key]];
  ctx.el.innerHTML = `<div class="card">${items.map(([k, l, ic]) => `<a href="#/${k}" class="row" style="padding:14px 16px;border-bottom:1px solid var(--line);color:inherit">${ic.replace("<svg", '<svg width="18" height="18"')}<span>${l}</span></a>`).join("")}
    <button class="row btn ghost block" style="justify-content:flex-start;height:48px;padding:0 16px" id="mlogout">${icon.logout}Sign out</button></div>`;
  $("#mlogout").onclick = async () => { await api("/api/logout", { method: "POST" }).catch(() => {}); location.hash = ""; boot(); };
}

async function route() {
  if (!S.user) return;
  const { view, params, query } = parseHash();
  if (typeof cleanup === "function") { try { cleanup(); } catch (e) { /* view already gone */ } }
  cleanup = null;
  const navKey = view === "report" ? "reports" : view;
  $$("[data-nav]").forEach((a) => a.classList.toggle("active", a.dataset.nav === navKey));
  const el = $("#view");
  el.innerHTML = "";
  el.className = "content";
  const ctx = {
    el, params, query,
    setTitle(title, crumbs = "", actions = "") {
      $("#title").textContent = title;
      $("#crumbs").innerHTML = crumbs;
      $("#actions").innerHTML = actions;
      document.title = `${title} · RoadSense`;
    },
    full(on) { el.className = on ? "" : "content"; },
  };
  if (view === "more") return more(ctx);
  if (view === "users" && S.user.role !== "admin") { location.hash = "#/overview"; return; }
  const loader = VIEWS[view];
  if (!loader) { location.hash = "#/overview"; return; }
  try {
    const mod = await loader();
    cleanup = await mod.render(ctx);
  } catch (e) {
    console.error(e);
    el.innerHTML = `<div class="alert error">${icon.alert}<div>${esc(e.message || "Something went wrong.")}</div></div>`;
  }
  window.scrollTo(0, 0);
}

async function boot() {
  const s = await api("/api/session").catch((e) => ({ error: e.message }));
  if (s.error) { $("#root").innerHTML = `<div class="auth-form"><div class="alert error">${icon.alert}${esc(s.error)}</div></div>`; return; }
  S.user = s.user;
  if (!S.user) {
    const { renderAuth } = await import("./views/login.js");
    return renderAuth($("#root"), s, async () => boot());
  }
  S.info = await api("/api/info");
  shell();
  if (!location.hash) location.hash = "#/overview";
  route();
}

setSignedOutHandler(() => { if (S.user) { S.user = null; toast("Your session ended. Please sign in again.", "error"); boot(); } });
window.addEventListener("hashchange", route);
boot();
