// Report detail: photo with boxes / model input / heat map, priority and repairs, location, status workflow, history.
import { api } from "../api.js";
import { accuracyCircle, baseMap, bestPosition, directionsUrl, locIcon, osmUrl, pinIcon } from "../mapkit.js";
import { GPS_LABEL, S, STATUS_LABEL, accuracyTag, busy, confirmBox, coord, esc, icon, isAdmin, modal, priorityBadge, statusTag, toast, when } from "../ui.js";
import { photoViewer, viewerLegend } from "../viewer.js";
import { actionsList, detectionsTable, priorityBanner, qualityAlerts } from "./newreport.js";

const ROAD_LABEL = { unknown: "Not specified", highway: "Highway", arterial: "Main road (arterial)", local: "Local street" };
const EVENT = {
  created: (d) => `Report created${d?.survey_id ? ` by survey #${d.survey_id}` : ""}`,
  status: (d) => `Status: ${STATUS_LABEL[d.from] || d.from} → <b>${STATUS_LABEL[d.to] || d.to}</b>${d.comment ? `<div class="small ink2">“${esc(d.comment)}”</div>` : ""}`,
  note: () => "Note edited",
  location: (d) => `Location ${d.from?.[0] == null ? "set" : "moved"} (${esc(GPS_LABEL[d.source] || d.source)})`,
  road_class: (d) => `Road type: ${esc(ROAD_LABEL[d.from])} → ${esc(ROAD_LABEL[d.to])}`,
  deleted: () => "Deleted",
};

export async function render(ctx) {
  const id = ctx.params.id;
  let r;
  try { r = await api(`/api/reports/${id}`); }
  catch (e) { ctx.setTitle("Report not found"); ctx.el.innerHTML = `<div class="alert error">${icon.alert}<div>${esc(e.message)} <a href="#/reports">Back to reports</a></div></div>`; return; }
  ctx.setTitle(`Report #${r.id}`, `<a href="#/reports">Reports</a> / #${r.id}`,
    `<button class="btn" onclick="window.print()">${icon.printer}Print / PDF</button>
     <a class="btn" href="/api/reports/${r.id}/image/annotated" download="roadsense_report_${r.id}.jpg">${icon.download}Photo</a>
     ${isAdmin() ? `<button class="btn danger" id="del">${icon.trash}Delete</button>` : ""}`);
  const plan = r.plan;
  const dets = r.detections.map((d) => ({ ...d, box_xyxy: d.box, severity: { level: d.severity_level, score: d.severity_score } }));
  ctx.el.innerHTML = `
  <div class="split">
    <div class="stack">
      <div class="card"><div class="card-h"><div class="row wrap">${priorityBadge(r.priority, r.priority_name)}${statusTag(r.status)}</div>
        <span class="sub" style="margin-left:auto">${esc(r.source_name || "")}</span></div>
        <div class="card-b stack"><div id="viewer"></div>${viewerLegend()}${qualityAlerts(r.quality)}
          ${r.media.length ? `<div><div class="label" style="margin-bottom:6px">Also seen in ${r.media.length} other ${r.media.length === 1 ? "photo" : "photos / frames"} at this location</div>
            <div class="strip">${r.media.map((m) => `<img src="/api/media/${m.id}" alt="${esc(m.label || "")}" title="${esc(m.label || "")}" data-media="${m.id}" loading="lazy">`).join("")}</div></div>` : ""}
        </div></div>
      <div class="card"><div class="card-h"><h2>Detected damage</h2><span class="sub">${r.n_detections} on the main photo</span></div>${detectionsTable(dets)}</div>
      <div class="card"><div class="card-h"><h2>History</h2></div><div class="card-b"><div class="timeline">
        ${r.events.map((e) => `<div class="tl"><i></i><div><div>${(EVENT[e.action] || (() => esc(e.action)))(e.detail)}</div><div class="tiny muted">${when(e.at)}${e.user ? ` · ${esc(e.user)}` : ""}</div></div></div>`).join("")}
      </div></div></div>
    </div>
    <div class="stack">
      <div class="card"><div class="card-h"><h2>Repair priority</h2></div><div class="card-b stack">${priorityBanner(plan)}${actionsList(plan)}
        ${plan && plan.seen_in > 1 ? `<div class="small muted">Damage at this location was seen in ${plan.seen_in} photos/frames; priority counts it once.</div>` : ""}</div></div>
      <div class="card"><div class="card-h"><h2>Status</h2></div><div class="card-b stack">
        ${isAdmin() ? `<div class="seg" id="statusseg" style="flex-wrap:wrap">${S.info.statuses.map((s) => `<button data-s="${s}" class="${s === r.status ? "on" : ""}">${STATUS_LABEL[s]}</button>`).join("")}</div>
          <div class="field"><label for="scomment">Comment for the history <span class="muted">(optional)</span></label><input class="input" id="scomment" maxlength="500" placeholder="e.g. patched by ward crew"></div>`
          : `<div>${statusTag(r.status)}</div><div class="small muted">Only admins change the status.</div>`}
      </div></div>
      <div class="card"><div class="card-h"><h2>Location</h2>${r.can_edit ? `<button class="btn sm" id="relocate" style="margin-left:auto">${icon.pin}${r.lat == null ? "Set location" : "Correct"}</button>` : ""}</div>
        <div class="card-b stack">
          ${r.lat != null ? `<div class="coord"><b class="num">${coord(r.lat, r.lon)}</b>${accuracyTag(r.accuracy_m)}</div>
            <div class="small ink2">${esc(r.address || "No street address found")}</div>
            <div class="small muted">Source: ${esc(GPS_LABEL[r.gps_source] || r.gps_source || "—")}</div>
            <div class="mini-map" id="rmap"></div>
            <div class="row wrap no-print"><a class="btn sm" href="${directionsUrl(r.lat, r.lon)}" target="_blank" rel="noopener">${icon.route}Directions</a>
              <a class="btn sm" href="${osmUrl(r.lat, r.lon)}" target="_blank" rel="noopener">${icon.ext}OpenStreetMap</a>
              <a class="btn sm" href="#/map">${icon.map}On the damage map</a></div>
            ${r.nearby.length ? `<div class="alert info">${icon.info}<div>${r.nearby.length} other report${r.nearby.length === 1 ? "" : "s"} within 25 m: ${r.nearby.map((n) => `<a href="#/reports/${n.id}">#${n.id}</a> (${Math.round(n.distance_m)} m, ${esc(n.priority || "no damage")})`).join(", ")}. It may be the same damage.</div></div>` : ""}`
            : `<div class="alert warn">${icon.pin}<div>This report has no location, so it isn't on the map.${r.can_edit ? " Use <b>Set location</b>." : ""}</div></div>`}
        </div></div>
      <div class="card"><div class="card-h"><h2>Details</h2></div><div class="card-b">
        <dl class="dl">
          <dt>Reported</dt><dd>${when(r.created_at)}</dd>
          <dt>By</dt><dd>${esc(r.reporter || "—")}</dd>
          ${r.taken_at ? `<dt>Photo taken</dt><dd>${when(r.taken_at)}</dd>` : ""}
          ${r.survey_id ? `<dt>Survey</dt><dd><a href="#/reports?survey_id=${r.survey_id}">#${r.survey_id}</a></dd>` : ""}
          <dt>Road type</dt><dd>${r.can_edit ? `<select class="input sm" id="road" style="max-width:220px">${S.info.road_classes.map((c) => `<option value="${c}" ${c === r.road_class ? "selected" : ""}>${ROAD_LABEL[c]}</option>`).join("")}</select>` : esc(ROAD_LABEL[r.road_class])}</dd>
          <dt>Condition index</dt><dd>${r.condition_index == null ? "—" : `${Math.round(r.condition_index)} / 100`} <span class="tiny muted">(100 = no visible damage)</span></dd>
          <dt>Photo size</dt><dd class="num">${r.width} × ${r.height}</dd>
        </dl>
        <div class="field" style="margin-top:14px"><label for="note">Note</label>
          ${r.can_edit ? `<textarea class="input" id="note" maxlength="2000">${esc(r.note || "")}</textarea><div class="row"><button class="btn sm" id="savenote">Save note</button></div>`
            : `<div class="ink2">${esc(r.note || "—")}</div>`}</div>
      </div></div>
    </div>
  </div>`;

  await photoViewer(ctx.el.querySelector("#viewer"), {
    src: `/api/reports/${r.id}/image/original`, width: r.width, height: r.height, detections: dets,
    explain: () => api(`/api/reports/${r.id}/explain`, { method: "POST" }),
  });
  ctx.el.querySelectorAll("[data-media]").forEach((img) => (img.onclick = () => modal(img.alt || "Photo", `<img src="${img.src}" alt="" style="width:100%;border-radius:8px">`, { wide: true })));
  let map = null;
  if (r.lat != null) {
    map = baseMap(ctx.el.querySelector("#rmap"), { scrollWheelZoom: false });
    L.marker([r.lat, r.lon], { icon: pinIcon(r.priority, r.status) }).addTo(map);
    if (r.accuracy_m) accuracyCircle(r.lat, r.lon, r.accuracy_m).addTo(map);
    map.setView([r.lat, r.lon], 17);
  }
  const patch = async (body, okMsg) => {
    try { await api(`/api/reports/${r.id}`, { method: "PATCH", body }); toast(okMsg); render(ctx); }
    catch (e) { toast(e.message, "error"); }
  };
  ctx.el.querySelector("#statusseg")?.addEventListener("click", (e) => {
    const b = e.target.closest("[data-s]");
    if (b && b.dataset.s !== r.status) patch({ status: b.dataset.s, comment: ctx.el.querySelector("#scomment").value }, `Status set to ${STATUS_LABEL[b.dataset.s]}.`);
  });
  ctx.el.querySelector("#savenote")?.addEventListener("click", () => patch({ note: ctx.el.querySelector("#note").value }, "Note saved."));
  ctx.el.querySelector("#road")?.addEventListener("change", (e) => patch({ road_class: e.target.value }, "Road type updated."));
  ctx.el.querySelector("#del")?.addEventListener("click", async () => {
    if (!(await confirmBox(`Delete report #${r.id}?`, "It disappears from the map, lists and exports. The history keeps a record that it was deleted.", "Delete", true))) return;
    try { await api(`/api/reports/${r.id}`, { method: "DELETE" }); toast(`Report #${r.id} deleted.`); location.hash = "#/reports"; }
    catch (e) { toast(e.message, "error"); }
  });
  ctx.el.querySelector("#relocate")?.addEventListener("click", () => relocate(r, () => render(ctx)));
  return () => map && map.remove();
}

function relocate(r, done) {
  const m = modal("Set the damage location", `<div class="stack">
    <div class="small ink2">Tap the map or drag the pin to the exact spot. <b>My location</b> uses this device's GPS. Only use it if you are standing at the damage now.</div>
    <div class="mini-map" id="relmap" style="height:380px"></div><div class="coord" id="relinfo"></div></div>`,
  { wide: true, actions: `<button class="btn" id="relme">${icon.locate}My location</button><button class="btn" data-close>Cancel</button><button class="btn primary" id="relsave" disabled>Save location</button>` });
  const map = baseMap(m.el.querySelector("#relmap"));
  let pos = r.lat != null ? { lat: r.lat, lon: r.lon, source: "map", accuracy: null } : null, marker = null, circle = null;
  const set = (lat, lon, source, accuracy = null) => {
    pos = { lat, lon, source, accuracy };
    if (marker) marker.setLatLng([lat, lon]); else { marker = L.marker([lat, lon], { draggable: true, icon: locIcon() }).addTo(map); marker.on("dragend", () => { const p = marker.getLatLng(); set(p.lat, p.lng, "map"); }); }
    if (circle) { circle.remove(); circle = null; }
    if (accuracy) circle = accuracyCircle(lat, lon, accuracy).addTo(map);
    m.el.querySelector("#relinfo").innerHTML = `<b class="num">${coord(lat, lon)}</b> ${accuracyTag(accuracy)} <span class="small muted">${esc(GPS_LABEL[source])}</span>`;
    m.el.querySelector("#relsave").disabled = false;
  };
  if (pos) { map.setView([pos.lat, pos.lon], 18); set(pos.lat, pos.lon, "map"); m.el.querySelector("#relsave").disabled = true; }
  else map.setView([20.6, 78.9], 5);
  map.on("click", (e) => set(e.latlng.lat, e.latlng.lng, "map"));
  setTimeout(() => map.invalidateSize(), 60);
  m.el.querySelector("#relme").onclick = async (e) => {
    busy(e.currentTarget, true, "Locating…");
    try { const p = await bestPosition(); set(p.lat, p.lon, "device", p.accuracy); map.setView([p.lat, p.lon], 18); }
    catch (err) { toast(err.message, "error"); }
    busy(m.el.querySelector("#relme"), false);
  };
  m.el.querySelector("#relsave").onclick = async () => {
    try {
      await api(`/api/reports/${r.id}`, { method: "PATCH", body: { lat: pos.lat, lon: pos.lon, gps_source: pos.source, accuracy_m: pos.accuracy } });
      toast("Location saved."); m.close(); map.remove(); done();
    } catch (err) { toast(err.message, "error"); }
  };
}
