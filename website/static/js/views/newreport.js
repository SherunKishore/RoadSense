// New report: photo -> analysis (ensemble) -> location (photo GPS / device / map pin) -> save.
import { api } from "../api.js";
import { accuracyCircle, baseMap, bestPosition, INDIA, locIcon } from "../mapkit.js";
import { S, accuracyTag, busy, clsTag, coord, damageLabel, esc, icon, pct, sevTag, toast } from "../ui.js";
import { photoViewer, viewerLegend } from "../viewer.js";

const ROAD_LABEL = { unknown: "Not specified", highway: "Highway", arterial: "Main road (arterial)", local: "Local street" };

export function priorityBanner(plan) {
  if (!plan) return `<div class="prio none"><div class="big">—</div><div class="t"><b>No damage detected</b><span>Nothing passed the model's thresholds. If you can see damage, save it with a note.</span></div></div>`;
  return `<div class="prio ${plan.priority}"><div class="big">${plan.priority}</div><div class="t"><b>${esc(plan.priority_name)} priority</b><span>Repair ${esc(plan.deadline)}${plan.override ? ` · because of ${esc(plan.override)}` : ""}</span></div>
    <div class="score">score<b class="num">${Math.round(plan.priority_score)}</b></div></div>`;
}

export function actionsList(plan) {
  if (!plan) return "";
  return `<div class="stack tight">${plan.actions.map((a) => `<div class="row" style="align-items:flex-start;gap:10px">
    <span style="margin-top:2px">${sevTag(a.severity)}</span><div><b>${esc(damageLabel(a.damage))}</b><div class="small ink2">${esc(a.action)}</div></div></div>`).join("")}</div>`;
}

export function detectionsTable(dets) {
  if (!dets.length) return `<div class="empty" style="padding:22px">No damage above the threshold.</div>`;
  return `<div class="table-wrap"><table class="table"><thead><tr><th>Damage</th><th title="Calibrated: how often boxes like this were real damage in testing">Confidence</th><th title="How many of the 3 models found it">Models</th><th>Severity</th></tr></thead><tbody>
    ${dets.map((d) => `<tr><td>${clsTag(d.name)}</td><td class="num">${pct(d.conf)}</td><td class="num">${d.agree ?? "—"}/3</td>
      <td>${sevTag(d.severity?.level || d.severity_level)} <span class="tiny muted num">${Math.round(d.severity?.score ?? d.severity_score ?? 0)}</span></td></tr>`).join("")}
  </tbody></table></div>`;
}

export function qualityAlerts(q) {
  return q?.warnings?.length ? `<div class="alert warn">${icon.alert}<div>${q.warnings.map(esc).join("<br>")}</div></div>` : "";
}

export async function render(ctx) {
  ctx.setTitle("New report", "Analyse a road photo and record the damage");
  let st = { file: null, url: null, res: null, loc: null, map: null, marker: null, circle: null, watchAbort: false, captured: false };

  const start = () => {
    if (st.url) URL.revokeObjectURL(st.url);
    if (st.map) { st.map.remove(); st.map = null; }
    st = { file: null, url: null, res: null, loc: null, map: null, marker: null, circle: null, watchAbort: true, captured: false };
    ctx.el.innerHTML = `
      <div class="split">
        <div class="card"><div class="card-b">
          <label class="drop" id="drop" tabindex="0">
            <input type="file" id="pick" accept="image/jpeg,image/png,image/webp" class="sr-only">
            ${icon.image}<b>Drop a road photo here</b><span class="muted">or click to choose a file · JPEG, PNG or WebP, up to 25 MB</span>
          </label>
          <div class="row wrap" style="margin-top:14px;justify-content:center">
            <label class="btn primary lg" for="cam">${icon.camera}Take photo</label>
            <input type="file" id="cam" accept="image/*" capture="environment" class="sr-only">
            <label class="btn lg" for="pick">${icon.upload}Choose photo</label>
          </div>
        </div></div>
        <div class="stack">
          <div class="card"><div class="card-h"><h2>Tips for accurate results</h2></div><div class="card-b stack tight small ink2">
            <div class="row" style="align-items:flex-start">${icon.road.replace("<svg", '<svg width="16" height="16" style="flex:none;margin-top:2px"')}<span>Point the camera along the road, roughly as a car dash-cam sees it. The models were trained on photos taken this way.</span></div>
            <div class="row" style="align-items:flex-start">${icon.sun.replace("<svg", '<svg width="16" height="16" style="flex:none;margin-top:2px"')}<span>Daylight, no motion blur. RoadSense warns you if a photo is too dark or blurred.</span></div>
            <div class="row" style="align-items:flex-start">${icon.pin.replace("<svg", '<svg width="16" height="16" style="flex:none;margin-top:2px"')}<span>Use <b>Take photo</b> on site: RoadSense then records your phone's GPS. Photos picked from a gallery keep their GPS only if the camera saved it.</span></div>
          </div></div>
          <div class="card"><div class="card-b"><div class="field"><label for="road">Road type</label>
            <select class="input" id="road">${S.info.road_classes.map((r) => `<option value="${r}">${ROAD_LABEL[r]}</option>`).join("")}</select>
            <span class="help">Busier roads get a higher repair priority for the same damage.</span></div></div></div>
        </div>
      </div>`;
    const drop = ctx.el.querySelector("#drop");
    ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
    ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
    drop.addEventListener("drop", (e) => e.dataTransfer.files[0] && analyse(e.dataTransfer.files[0], false));
    ctx.el.querySelector("#pick").onchange = (e) => e.target.files[0] && analyse(e.target.files[0], false);
    ctx.el.querySelector("#cam").onchange = (e) => e.target.files[0] && analyse(e.target.files[0], true);
  };

  async function analyse(file, captured) {
    const road = ctx.el.querySelector("#road")?.value || "unknown";
    st.watchAbort = false;
    st.file = file; st.captured = captured; st.url = URL.createObjectURL(file);
    // a photo taken just now was taken here: start the device fix in parallel with the analysis
    const devicePromise = captured ? bestPosition({ onFix: (f) => { st.liveFix = f; } }).catch((e) => ({ error: e.message })) : null;
    ctx.el.innerHTML = `
      <div class="split">
        <div class="stack">
          <div class="card"><div class="card-h"><h2>${esc(file.name || "Photo")}</h2><span class="sub" id="timing"></span>
            <button class="btn sm ghost" id="again" style="margin-left:auto">${icon.refresh}Another photo</button></div>
            <div class="card-b stack"><div id="viewer"><div class="viewer" style="aspect-ratio:4/3"><div class="busy"><div style="text-align:center"><div class="spin"></div>Analysing with 3 models…</div></div></div></div>
            <div id="legend"></div><div id="quality"></div></div></div>
          <div class="card"><div class="card-h"><h2>Detected damage</h2><span class="sub" id="dcount"></span></div><div id="dets"></div></div>
        </div>
        <div class="stack">
          <div class="card"><div class="card-h"><h2>Repair priority</h2></div><div class="card-b stack" id="prio"><div class="muted">Waiting for the analysis…</div></div></div>
          <div class="card"><div class="card-h"><h2>Location</h2><span class="sub" id="locstate"></span></div><div class="card-b stack" id="loc"></div></div>
          <div class="card"><div class="card-h"><h2>Details</h2></div><div class="card-b stack">
            <div class="field"><label for="road2">Road type</label><select class="input" id="road2">${S.info.road_classes.map((r) => `<option value="${r}" ${r === road ? "selected" : ""}>${ROAD_LABEL[r]}</option>`).join("")}</select></div>
            <div class="field"><label for="note">Note <span class="muted">(optional)</span></label><textarea class="input" id="note" maxlength="2000" placeholder="e.g. near the bus stop, water collects here after rain"></textarea></div>
          </div><div class="card-f row"><span class="small muted grow" id="savehint"></span><button class="btn primary" id="save" disabled>${icon.check}Save report</button></div></div>
        </div>
      </div>`;
    ctx.el.querySelector("#again").onclick = start;
    const fd = new FormData();
    fd.append("file", file);
    fd.append("road_class", road);
    let res;
    try {
      res = await api("/api/analyze", { method: "POST", form: fd });
    } catch (e) {
      ctx.el.querySelector("#viewer").innerHTML = `<div class="alert error">${icon.alert}<div>${esc(e.message)}</div></div>`;
      return;
    }
    st.res = res;
    ctx.el.querySelector("#timing").textContent = `analysed in ${((res.ms.preprocess + res.ms.inference + (res.ms.fusion || 0)) / 1000).toFixed(1)} s on ${S.info.device}`;
    await photoViewer(ctx.el.querySelector("#viewer"), {
      src: st.url, width: res.width, height: res.height, detections: res.detections, uncertain: res.uncertain,
      enhanced: async () => URL.createObjectURL(await api(`/api/analyze/${res.token}/enhanced`)),
      explain: () => api(`/api/analyze/${res.token}/explain`, { method: "POST" }),
    });
    ctx.el.querySelector("#legend").innerHTML = viewerLegend();
    ctx.el.querySelector("#quality").innerHTML = qualityAlerts(res.quality);
    ctx.el.querySelector("#dcount").textContent = `${res.detections.length} found`;
    ctx.el.querySelector("#dets").innerHTML = detectionsTable(res.detections);
    showPlan(res.plan);
    ctx.el.querySelector("#road2").onchange = async (e) => {
      try { showPlan((await api(`/api/analyze/${res.token}/plan`, { method: "POST", body: { road_class: e.target.value } })).plan); }
      catch (err) { toast(err.message, "error"); }
    };
    setupLocation(res, devicePromise);
    const save = ctx.el.querySelector("#save");
    save.disabled = false;
    if (!res.detections.length) save.innerHTML = `${icon.check}Save anyway`;
    save.onclick = () => saveReport(save);
  }

  function showPlan(plan) {
    ctx.el.querySelector("#prio").innerHTML = priorityBanner(plan) + actionsList(plan);
  }

  function setLoc(lat, lon, source, accuracy = null) {
    st.loc = { lat, lon, source, accuracy };
    const m = st.map;
    if (st.marker) st.marker.setLatLng([lat, lon]); else {
      st.marker = L.marker([lat, lon], { draggable: true, autoPan: true, icon: locIcon() }).addTo(m);
      st.marker.on("dragend", () => { const p = st.marker.getLatLng(); setLoc(p.lat, p.lng, "map"); });
    }
    if (st.circle) { st.circle.remove(); st.circle = null; }
    if (accuracy) st.circle = accuracyCircle(lat, lon, accuracy).addTo(m);
    m.setView([lat, lon], Math.max(m.getZoom(), accuracy && accuracy > 100 ? 15 : 17));
    ctx.el.querySelectorAll("[data-src]").forEach((b) => b.classList.toggle("on", b.dataset.src === source));
    const srcLabel = { exif: "from the photo's GPS", device: "from this device", map: "pinned on the map" }[source];
    ctx.el.querySelector("#locinfo").innerHTML = `<div class="coord"><b class="num">${coord(lat, lon)}</b>${accuracyTag(accuracy)}</div>
      <div class="small muted">${srcLabel}. Drag the pin to correct it.</div><div class="small ink2" id="addr">Looking up the address…</div>`;
    ctx.el.querySelector("#locstate").innerHTML = `<span class="badge outline">${icon.check.replace("<svg", '<svg width="12" height="12"')}Set</span>`;
    ctx.el.querySelector("#savehint").textContent = "";
    clearTimeout(st.geoT);
    st.geoT = setTimeout(async () => {
      try {
        const r = await api(`/api/geocode?lat=${lat}&lon=${lon}`);
        const el = ctx.el.querySelector("#addr");
        if (el && st.loc && st.loc.lat === lat) el.textContent = r.address || "No street address found for this spot.";
      } catch (e) { const el = ctx.el.querySelector("#addr"); if (el) el.textContent = "Address lookup unavailable (offline?)."; }
    }, 600);
  }

  function setupLocation(res, devicePromise) {
    const exif = res.exif && res.exif.lat != null ? res.exif : null;
    const box = ctx.el.querySelector("#loc");
    box.innerHTML = `
      <div class="loc-src">
        <button data-src="exif" ${exif ? "" : "disabled"} title="${exif ? "GPS saved in the photo" : "This photo has no GPS data"}">${icon.image}Photo GPS</button>
        <button data-src="device">${icon.locate}My location</button>
        <button data-src="map">${icon.pin}Pin on map</button>
      </div>
      <div id="locinfo" class="stack tight"></div>
      <div class="mini-map" id="minimap"></div>`;
    st.map = baseMap(box.querySelector("#minimap"));
    st.map.setView(INDIA, 4);
    st.map.on("click", (e) => { if (st.pickMode || !st.loc) setLoc(e.latlng.lat, e.latlng.lng, "map"); });
    const info = box.querySelector("#locinfo");
    const noLoc = (msg) => {
      info.innerHTML = `<div class="alert ${msg ? "warn" : "info"}">${icon.pin}<div>${msg ? esc(msg) + " " : ""}${exif ? "" : "This photo has no GPS data. "}Use <b>My location</b> if you are at the spot now, or tap the map to place the pin.</div></div>`;
      ctx.el.querySelector("#locstate").textContent = "not set";
      ctx.el.querySelector("#savehint").textContent = "Reports without a location won't appear on the map.";
    };
    const useDevice = async (promise) => {
      info.innerHTML = `<div class="row small ink2"><span class="spin" style="width:14px;height:14px;border:2px solid currentColor;border-right-color:transparent;border-radius:50%;animation:spin .7s linear infinite"></span><span id="fixmsg">Getting your location…</span></div>`;
      ctx.el.querySelectorAll("[data-src]").forEach((b) => b.classList.toggle("on", b.dataset.src === "device"));
      const r = await (promise || bestPosition({ onFix: (f) => { const el = ctx.el.querySelector("#fixmsg"); if (el) el.textContent = `Improving accuracy… ± ${Math.round(f.accuracy)} m`; } }).catch((e) => ({ error: e.message })));
      if (st.watchAbort) return;
      if (r.error) { toast(r.error, "error"); return noLoc(r.error); }
      setLoc(r.lat, r.lon, "device", r.accuracy);
      if (r.accuracy > 50) toast(`Location is only accurate to ± ${Math.round(r.accuracy)} m. Drag the pin to the exact spot if you can.`);
    };
    box.querySelector('[data-src="exif"]').onclick = () => exif && setLoc(exif.lat, exif.lon, "exif", exif.accuracy_m ?? null);
    box.querySelector('[data-src="device"]').onclick = () => useDevice(null);
    box.querySelector('[data-src="map"]').onclick = () => {
      st.pickMode = true;
      ctx.el.querySelectorAll("[data-src]").forEach((b) => b.classList.toggle("on", b.dataset.src === "map"));
      if (st.loc) { st.loc.source = "map"; }
      info.innerHTML = `<div class="alert info">${icon.pin}<div>Tap the map where the damage is. You can zoom in and drag the pin afterwards.</div></div>`;
    };
    if (exif) setLoc(exif.lat, exif.lon, "exif", exif.accuracy_m ?? null);
    else if (devicePromise) useDevice(devicePromise);
    else noLoc(null);
  }

  async function saveReport(btn) {
    busy(btn, true, "Saving…");
    const body = { token: st.res.token, road_class: ctx.el.querySelector("#road2").value, note: ctx.el.querySelector("#note").value };
    if (st.loc) Object.assign(body, { lat: st.loc.lat, lon: st.loc.lon, gps_source: st.loc.source, accuracy_m: st.loc.accuracy });
    try {
      const r = await api("/api/reports", { method: "POST", body });
      toast(`Report #${r.id} saved.`);
      location.hash = `#/reports/${r.id}`;
    } catch (e) { busy(btn, false); toast(e.message, "error"); }
  }

  start();
  return () => { st.watchAbort = true; if (st.map) st.map.remove(); if (st.url) URL.revokeObjectURL(st.url); };
}
