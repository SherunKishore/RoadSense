// Overview: headline numbers, open work by priority and damage type, reports per day, urgent list and a mini map.
import { api } from "../api.js";
import { baseMap, pinIcon, INDIA } from "../mapkit.js";
import { S, ago, clsLabel, damageLabel, emptyState, esc, icon, priorityBadge, statusTag } from "../ui.js";

const PCOLOR = { P1: "var(--p1)", P2: "var(--p2)", P3: "var(--p3)", P4: "var(--p4)", none: "var(--pn)" };

function hbars(rows, max) {
  return `<div class="hbars">${rows.map((r) => `<div class="hbar" title="${esc(r.label)}: ${r.n}">
    <span class="ink2" style="white-space:nowrap;overflow:hidden;text-overflow:ellipsis">${r.html || esc(r.label)}</span>
    <div class="track"><div class="fill" style="width:${max ? (100 * r.n) / max : 0}%;background:${r.color}"></div></div>
    <span class="v">${r.n}</span></div>`).join("")}</div>`;
}

function columns(el, days) {
  const max = Math.max(1, ...days.map((d) => d.n));
  const W = Math.max(240, Math.round((el.clientWidth || 636) - 36)), H = 180, pad = { l: 28, r: 6, t: 10, b: 22 };
  const bw = (W - pad.l - pad.r) / days.length;
  const barW = Math.min(14, bw - 3);
  const y = (v) => H - pad.b - ((H - pad.t - pad.b) * v) / max;
  const ticks = [...new Set([0, Math.ceil(max / 2), max])];
  const bars = days.map((d, i) => {
    const x = pad.l + i * bw + (bw - barW) / 2, top = y(d.n), h = H - pad.b - top;
    const r = Math.min(4, h);
    const path = d.n ? `M${x},${H - pad.b} V${top + r} Q${x},${top} ${x + r},${top} H${x + barW - r} Q${x + barW},${top} ${x + barW},${top + r} V${H - pad.b} Z` : "";
    return `<g data-i="${i}"><rect x="${pad.l + i * bw}" y="${pad.t}" width="${bw}" height="${H - pad.t - pad.b}" fill="transparent"/>${path ? `<path d="${path}" fill="var(--accent)"/>` : ""}</g>`;
  }).join("");
  const fmt = (iso) => new Date(iso + "T00:00:00").toLocaleDateString(undefined, { day: "numeric", month: "short" });
  el.innerHTML = `<div class="cols"><svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="Reports per day, last 30 days">
    ${ticks.map((t) => `<line x1="${pad.l}" x2="${W - pad.r}" y1="${y(t)}" y2="${y(t)}" stroke="var(--line)" stroke-width="1"/>
      <text x="${pad.l - 6}" y="${y(t) + 4}" text-anchor="end" font-size="11" fill="var(--muted)">${t}</text>`).join("")}
    ${bars}
    <text x="${pad.l}" y="${H - 6}" font-size="11" fill="var(--muted)">${fmt(days[0].date)}</text>
    <text x="${W - pad.r}" y="${H - 6}" font-size="11" fill="var(--muted)" text-anchor="end">Today</text>
  </svg><div class="tip hidden"></div></div>`;
  const svg = el.querySelector("svg"), tip = el.querySelector(".tip");
  svg.addEventListener("mousemove", (e) => {
    const g = e.target.closest("g[data-i]");
    if (!g) return tip.classList.add("hidden");
    const d = days[+g.dataset.i], box = svg.getBoundingClientRect(), host = el.querySelector(".cols").getBoundingClientRect();
    const i = +g.dataset.i;
    tip.innerHTML = `<b>${d.n}</b> report${d.n === 1 ? "" : "s"} · ${fmt(d.date)}`;
    tip.style.left = `${box.left - host.left + ((pad.l + (i + 0.5) * bw) / W) * box.width}px`;
    tip.style.top = `${box.top - host.top + (y(d.n) / H) * box.height}px`;
    tip.classList.remove("hidden");
  });
  svg.addEventListener("mouseleave", () => tip.classList.add("hidden"));
}

export async function render(ctx) {
  ctx.setTitle("Overview", `Welcome back, ${esc(S.user.name.split(" ")[0])}`,
    `<a class="btn" href="#/map">${icon.map}Open map</a><a class="btn primary" href="#/new">${icon.camera}New report</a>`);
  ctx.el.innerHTML = `<div class="stack"><div class="grid g4">${Array(4).fill('<div class="card kpi"><div class="label">&nbsp;</div><div class="value muted">…</div></div>').join("")}</div></div>`;
  const st = await api("/api/stats");
  const k = st.kpi;
  const total = k.total || 0;
  const pr = ["P1", "P2", "P3", "P4"].map((p) => ({ label: `${p} ${S.info.priority_levels.find((l) => l.level === p).name}`, n: st.by_priority[p] || 0, color: PCOLOR[p] }));
  const dm = S.info.classes.map((c) => ({ label: c.label, n: st.by_damage[c.name] || 0, color: "var(--accent)" }));
  ctx.el.innerHTML = `
  <div class="stack" style="gap:16px">
    ${total === 0 ? `<div class="alert info">${icon.info}<div><b>No reports yet.</b> Start with <a href="#/new">New report</a> for a single photo, or <a href="#/surveys">Surveys</a> to upload a batch of geotagged photos or a dash-cam video with its GPX track.</div></div>` : ""}
    <div class="grid g4">
      <div class="card kpi"><div class="label">Open reports</div><div class="value num">${k.open || 0}</div><div class="note">${total} in total · ${k.no_location || 0} without location</div></div>
      <div class="card kpi"><div class="label"><span class="dot" style="background:var(--p1)"></span>Urgent (P1)</div><div class="value num">${k.urgent || 0}</div><div class="note">to repair within 48 hours</div></div>
      <div class="card kpi"><div class="label"><span class="dot" style="background:var(--d40)"></span>Open with potholes</div><div class="value num">${k.potholes_open || 0}</div><div class="note">the main safety risk</div></div>
      <div class="card kpi"><div class="label"><span class="dot" style="background:var(--p4)"></span>Repaired this month</div><div class="value num">${k.repaired_month || 0}</div><div class="note">avg. condition index ${k.condition == null ? "—" : Math.round(k.condition)} / 100</div></div>
    </div>
    <div class="grid g3">
      <div class="card"><div class="card-h"><h2>Open work by priority</h2></div><div class="card-b">${hbars(pr, Math.max(...pr.map((r) => r.n)))}</div></div>
      <div class="card"><div class="card-h"><h2>Open work by damage type</h2></div><div class="card-b">${hbars(dm, Math.max(...dm.map((r) => r.n)))}</div></div>
      <div class="card"><div class="card-h"><h2>Reports per day</h2><span class="sub">last 30 days</span></div><div class="card-b" id="perday"></div></div>
    </div>
    <div class="split">
      <div class="card"><div class="card-h"><h2>Most urgent open work</h2><a class="btn sm ghost" style="margin-left:auto" href="#/reports?status=open,in_progress">View all</a></div>
        ${st.urgent.length ? `<div class="table-wrap"><table class="table"><thead><tr><th>Report</th><th>Priority</th><th>Worst damage</th><th>Location</th><th>Status</th></tr></thead><tbody>
          ${st.urgent.map((r) => `<tr class="click" data-id="${r.id}"><td><b>#${r.id}</b><div class="tiny muted">${ago(r.created_at)}</div></td>
            <td>${priorityBadge(r.priority, r.priority_name)}<div class="tiny muted">${esc(r.deadline || "")}</div></td>
            <td>${esc(damageLabel(r.worst_damage))}<div class="tiny muted">${esc(r.worst_level || "")} severity</div></td>
            <td class="small" style="max-width:260px">${r.address ? esc(r.address) : r.lat != null ? `${(+r.lat).toFixed(5)}, ${(+r.lon).toFixed(5)}` : '<span class="muted">No location</span>'}</td>
            <td>${statusTag(r.status)}</td></tr>`).join("")}</tbody></table></div>`
          : emptyState("Nothing urgent", "No open P1 or P2 reports right now.", icon.check)}
      </div>
      <div class="card"><div class="card-h"><h2>Where the damage is</h2><a class="btn sm ghost" style="margin-left:auto" href="#/map">Full map</a></div>
        <div id="ovmap" style="height:310px;border-radius:0 0 10px 10px"></div></div>
    </div>
  </div>`;
  columns(ctx.el.querySelector("#perday"), st.per_day);
  ctx.el.querySelectorAll("tr[data-id]").forEach((tr) => (tr.onclick = () => (location.hash = `#/reports/${tr.dataset.id}`)));
  const map = baseMap(ctx.el.querySelector("#ovmap"), { scrollWheelZoom: false });
  const pts = await api("/api/map?status=open,in_progress");
  const layer = L.featureGroup().addTo(map);
  pts.points.forEach((p) => L.marker([p.lat, p.lon], { icon: pinIcon(p.priority, p.status) }).addTo(layer)
    .bindTooltip(`#${p.id} ${p.priority || ""} · ${esc(damageLabel(p.worst_damage))}`).on("click", () => (location.hash = `#/reports/${p.id}`)));
  if (pts.points.length) map.fitBounds(layer.getBounds().pad(0.2), { maxZoom: 16 }); else map.setView(INDIA, 4);
  return () => map.remove();
}
