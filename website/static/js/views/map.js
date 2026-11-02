// Damage map: every located report as a priority-coloured pin, clustered, with filters and exports.
import { api } from "../api.js";
import { baseMap, bestPosition, clusterGroup, directionsUrl, INDIA, meIcon, accuracyCircle, pinIcon } from "../mapkit.js";
import { GPS_LABEL, S, STATUS_LABEL, accuracyTag, damageLabel, esc, icon, priorityBadge, statusTag, toast, when } from "../ui.js";

const RANGES = [["", "Any time"], ["7", "Last 7 days"], ["30", "Last 30 days"], ["90", "Last 90 days"]];

export async function render(ctx) {
  ctx.full(true);
  ctx.setTitle("Damage map", "Every report with a location",
    `<a class="btn" id="csv">${icon.download}CSV</a><a class="btn" id="geojson">${icon.download}GeoJSON</a>`);
  const q = ctx.query;
  const f = {
    priority: new Set((q.priority || "P1,P2,P3,P4").split(",")),
    status: new Set((q.status || "open,in_progress").split(",")),
    damage: new Set((q.damage || S.info.classes.map((c) => c.name).join(",")).split(",")),
    days: q.days || "", mine: q.mine === "1", survey: q.survey || "",
  };
  ctx.el.innerHTML = `
  <div class="map-page">
    <aside class="map-filters">
      <div class="stack tight"><div class="label">Priority</div>
        ${["P1", "P2", "P3", "P4"].map((p) => `<label class="check"><input type="checkbox" data-f="priority" value="${p}" ${f.priority.has(p) ? "checked" : ""}>${priorityBadge(p)}</label>`).join("")}
        <label class="check"><input type="checkbox" data-f="priority" value="none" ${f.priority.has("none") ? "checked" : ""}>${priorityBadge(null)}</label></div>
      <div class="stack tight"><div class="label">Status</div>
        ${S.info.statuses.map((s) => `<label class="check"><input type="checkbox" data-f="status" value="${s}" ${f.status.has(s) ? "checked" : ""}>${statusTag(s)}</label>`).join("")}</div>
      <div class="stack tight"><div class="label">Damage type</div>
        ${S.info.classes.map((c) => `<label class="check"><input type="checkbox" data-f="damage" value="${c.name}" ${f.damage.has(c.name) ? "checked" : ""}><span class="cls"><i style="background:${c.color}"></i>${esc(c.label)}</span></label>`).join("")}</div>
      <div class="field"><label for="days">Reported</label><select class="input" id="days">${RANGES.map(([v, l]) => `<option value="${v}" ${f.days === v ? "selected" : ""}>${l}</option>`).join("")}</select></div>
      <label class="check"><input type="checkbox" id="mine" ${f.mine ? "checked" : ""}>Only my reports</label>
      ${f.survey ? `<div class="alert info small">${icon.layers}<div>Showing survey #${esc(f.survey)} only. <a href="#/map">Show all</a></div></div>` : ""}
      <div class="small muted" id="count">Loading…</div>
    </aside>
    <div style="position:relative"><div id="bigmap" class="big-map"></div></div>
  </div>`;
  const map = baseMap(ctx.el.querySelector("#bigmap"));
  map.setView(INDIA, 5);
  const layer = clusterGroup().addTo(map);
  const legend = L.control({ position: "bottomleft" });
  legend.onAdd = () => {
    const d = L.DomUtil.create("div", "map-legend");
    d.innerHTML = S.info.priority_levels.map((l) => `<div><i style="background:var(--${l.level.toLowerCase()})"></i>${l.level} ${esc(l.name)} · ${esc(l.deadline)}</div>`).join("") +
      `<div class="muted">Faded pin = repaired or rejected</div>`;
    return d;
  };
  legend.addTo(map);
  const locate = L.control({ position: "topright" });
  locate.onAdd = () => {
    const b = L.DomUtil.create("button", "btn sm");
    b.innerHTML = `${icon.locate}My location`;
    b.style.boxShadow = "var(--shadow-lg)";
    L.DomEvent.disableClickPropagation(b);
    b.onclick = async () => {
      b.disabled = true;
      try {
        const p = await bestPosition({ timeoutMs: 8000 });
        L.marker([p.lat, p.lon], { icon: meIcon() }).addTo(map).bindTooltip("You are here");
        accuracyCircle(p.lat, p.lon, p.accuracy).addTo(map);
        map.setView([p.lat, p.lon], 16);
      } catch (e) { toast(e.message, "error"); }
      b.disabled = false;
    };
    return b;
  };
  locate.addTo(map);

  const qs = () => {
    const p = new URLSearchParams();
    if (f.priority.size < 5) p.set("priority", [...f.priority].join(",") || "x");
    if (f.status.size < 4) p.set("status", [...f.status].join(",") || "x");
    if (f.damage.size < S.info.classes.length) p.set("damage", [...f.damage].join(",") || "x");
    if (f.days) p.set("since", new Date(Date.now() - f.days * 86400000).toISOString().slice(0, 10));
    if (f.mine) p.set("mine", "1");
    if (f.survey) p.set("survey_id", f.survey);
    return p.toString();
  };
  let first = true;
  const load = async () => {
    const query = qs();
    ctx.el.closest(".main").querySelector("#csv").href = `/api/export.csv?${query}`;
    ctx.el.closest(".main").querySelector("#geojson").href = `/api/export.geojson?${query}`;
    const r = await api(`/api/map?${query}`);
    layer.clearLayers();
    r.points.forEach((p) => {
      const m = L.marker([p.lat, p.lon], { icon: pinIcon(p.priority, p.status), priority: p.priority, title: `Report #${p.id}` });
      m.bindPopup(() => `<div class="pop"><div class="row between"><span class="t">Report #${p.id}</span>${priorityBadge(p.priority, p.priority_name)}</div>
        <img src="/api/reports/${p.id}/image/thumb" alt="Annotated photo of report ${p.id}" loading="lazy">
        <div><b>${esc(damageLabel(p.worst_damage))}</b>${p.worst_level ? ` · ${esc(p.worst_level)} severity` : ""}</div>
        <div class="small ink2">${p.address ? esc(p.address) : `${(+p.lat).toFixed(6)}, ${(+p.lon).toFixed(6)}`}</div>
        <div class="small muted">${statusTag(p.status)} · ${when(p.created_at)} · ${esc(GPS_LABEL[p.gps_source] || "")} ${accuracyTag(p.accuracy_m)}</div>
        <div class="row" style="margin-top:8px"><a class="btn sm primary" href="#/reports/${p.id}">Open report</a><a class="btn sm" href="${directionsUrl(p.lat, p.lon)}" target="_blank" rel="noopener">${icon.route}Directions</a></div></div>`,
        { maxWidth: 280, minWidth: 240 });
      layer.addLayer(m);
    });
    ctx.el.querySelector("#count").innerHTML = `<b>${r.points.length}</b> report${r.points.length === 1 ? "" : "s"} on the map${r.without_location ? ` · <a href="#/reports?has_location=0">${r.without_location} without a location</a>` : ""}`;
    if (r.points.length && (first || f.survey)) map.fitBounds(layer.getBounds().pad(0.15), { maxZoom: 17 });
    first = false;
  };
  ctx.el.querySelectorAll("[data-f]").forEach((cb) => (cb.onchange = () => { const s = f[cb.dataset.f]; cb.checked ? s.add(cb.value) : s.delete(cb.value); load(); }));
  ctx.el.querySelector("#days").onchange = (e) => { f.days = e.target.value; load(); };
  ctx.el.querySelector("#mine").onchange = (e) => { f.mine = e.target.checked; load(); };
  await load();
  setTimeout(() => map.invalidateSize(), 50);
  return () => map.remove();
}

export { STATUS_LABEL };
