// Surveys: upload many geotagged photos, or a dash-cam video with its GPX track; processed in the background.
import { api } from "../api.js";
import { S, busy, emptyState, esc, icon, toast, when } from "../ui.js";

const ROAD_LABEL = { unknown: "Not specified", highway: "Highway", arterial: "Main road (arterial)", local: "Local street" };

function surveyRow(s) {
  const st = s.stats || {};
  const state = s.status === "done" ? `<span class="badge P4"><span class="dot"></span>Done</span>` : s.status === "failed" ? `<span class="badge P1"><span class="dot"></span>Failed</span>`
    : `<span class="badge P3"><span class="dot"></span>${s.status === "queued" ? "Queued" : "Running"}</span>`;
  return `<tr><td><b>${esc(s.name)}</b><div class="tiny muted">#${s.id} · ${s.kind === "video" ? "video" : "photos"} · ${when(s.created_at)} · ${esc(s.user || "")}</div></td>
    <td style="min-width:200px">${state}${s.status === "running" ? `<div class="progress" style="margin-top:6px"><div style="width:${Math.round(100 * (s.progress || 0))}%"></div></div>` : ""}
      <div class="tiny muted" style="margin-top:3px">${esc(s.message || "")}</div></td>
    <td class="num small">${st.items != null ? `${st.with_damage}/${st.items} with damage<br>${st.with_gps} geotagged` : "—"}</td>
    <td class="num">${st.reports ?? "—"}${st.reports_without_location ? `<div class="tiny muted">${st.reports_without_location} need a location</div>` : ""}</td>
    <td>${s.status === "done" && st.reports ? `<div class="row"><a class="btn sm" href="#/map?survey=${s.id}&status=open,in_progress,repaired,rejected&priority=P1,P2,P3,P4,none">${icon.map}Map</a><a class="btn sm" href="#/reports?survey_id=${s.id}">${icon.list}List</a></div>` : ""}</td></tr>`;
}

export async function render(ctx) {
  ctx.setTitle("Surveys", "Batch uploads: geotagged photos or dash-cam video + GPX");
  let kind = "photos", timer = null;
  ctx.el.innerHTML = `
  <div class="split">
    <div class="card"><div class="card-h"><h2>New survey</h2></div><div class="card-b stack">
      <div class="seg" id="kind"><button data-k="photos" class="on">${icon.image.replace("<svg", '<svg width="14" height="14"')} Photos</button><button data-k="video">${icon.video.replace("<svg", '<svg width="14" height="14"')} Video + GPX</button></div>
      <form id="f" class="stack">
        <div class="field"><label for="name">Name</label><input class="input" id="name" maxlength="120" placeholder="e.g. Ward 12, MG Road, 21 Oct"></div>
        <div id="photosbox" class="field"><label for="photos">Photos</label><input class="input" type="file" id="photos" multiple accept="image/jpeg,image/png,image/webp" style="padding-top:6px;height:auto">
          <span class="help">Up to 500 photos. Each photo's GPS (EXIF) places it on the map; damage within 25 m is grouped into one repair location.</span></div>
        <div id="videobox" class="stack hidden">
          <div class="field"><label for="video">Dash-cam video</label><input class="input" type="file" id="video" accept="video/mp4,video/quicktime,video/x-msvideo,.mp4,.mov,.avi" style="padding-top:6px;height:auto"><span class="help">MP4 (H.264), MOV or AVI, up to 500 MB.</span></div>
          <div class="field"><label for="gpx">GPS track (GPX) <span class="muted">(optional)</span></label><input class="input" type="file" id="gpx" accept=".gpx,application/gpx+xml" style="padding-top:6px;height:auto">
            <span class="help">From the dash-cam or a GPS logger app. Frame times are matched to the track. Without it, damage is grouped by time and you set locations by hand.</span></div>
          <div class="grid g2">
            <div class="field"><label for="vstart">Video start time (UTC) <span class="muted">(optional)</span></label><input class="input" id="vstart" placeholder="2026-10-21T08:15:00Z"><span class="help">Default: the track's first point.</span></div>
            <div class="field"><label for="stride">Analyse every</label><select class="input" id="stride">${[[1, "frame"], [3, "3rd frame"], [5, "5th frame"], [10, "10th frame"], [15, "15th frame"]].map(([v, l]) => `<option value="${v}" ${v === 5 ? "selected" : ""}>${l}</option>`).join("")}</select><span class="help">At 30 fps, every 5th frame = 6 per second.</span></div>
          </div>
        </div>
        <div class="field"><label for="road">Road type</label><select class="input" id="road">${["unknown", "highway", "arterial", "local"].map((r) => `<option value="${r}">${ROAD_LABEL[r]}</option>`).join("")}</select></div>
        <button class="btn primary" type="submit" id="go">${icon.upload}Upload and analyse</button>
      </form>
    </div></div>
    <div class="card"><div class="card-h"><h2>How surveys work</h2></div><div class="card-b stack tight small ink2">
      <div>1. Files upload to this computer; nothing leaves it except address lookups (OpenStreetMap).</div>
      <div>2. Each photo or sampled frame runs through the 3-model ensemble, then severity scoring.</div>
      <div>3. Detections are grouped into <b>repair locations</b>: within 25 m with GPS, or 5-second windows of video without GPS.</div>
      <div>4. Each location becomes one report, showing its worst photo plus up to 6 other sightings.</div>
      <div class="muted" style="margin-top:6px">Speed on ${esc(S.info.device)}: about 1 photo per second. You can leave this page; the survey keeps running.</div>
    </div></div>
  </div>
  <div class="card" style="margin-top:16px"><div class="card-h"><h2>Recent surveys</h2><button class="btn sm ghost" id="refresh" style="margin-left:auto">${icon.refresh}Refresh</button></div><div id="list"></div></div>`;
  const setKind = (k) => {
    kind = k;
    ctx.el.querySelectorAll("#kind button").forEach((b) => b.classList.toggle("on", b.dataset.k === k));
    ctx.el.querySelector("#photosbox").classList.toggle("hidden", k !== "photos");
    ctx.el.querySelector("#videobox").classList.toggle("hidden", k !== "video");
  };
  ctx.el.querySelector("#kind").onclick = (e) => { const b = e.target.closest("[data-k]"); if (b) setKind(b.dataset.k); };
  const list = async () => {
    const r = await api("/api/surveys");
    ctx.el.querySelector("#list").innerHTML = r.surveys.length ? `<div class="table-wrap"><table class="table"><thead><tr><th>Survey</th><th>Progress</th><th>Analysed</th><th>Repair locations</th><th></th></tr></thead><tbody>${r.surveys.map(surveyRow).join("")}</tbody></table></div>`
      : emptyState("No surveys yet", "Upload a folder of geotagged photos or a dash-cam video above.", icon.layers);
    clearTimeout(timer);
    if (r.surveys.some((s) => s.status === "running" || s.status === "queued")) timer = setTimeout(list, 1500);
  };
  ctx.el.querySelector("#refresh").onclick = list;
  ctx.el.querySelector("#f").onsubmit = async (e) => {
    e.preventDefault();
    const fd = new FormData();
    fd.append("kind", kind);
    fd.append("name", ctx.el.querySelector("#name").value);
    fd.append("road_class", ctx.el.querySelector("#road").value);
    if (kind === "photos") {
      const files = ctx.el.querySelector("#photos").files;
      if (!files.length) return toast("Choose some photos first.", "error");
      [...files].forEach((f) => fd.append("photos", f));
    } else {
      const v = ctx.el.querySelector("#video").files[0];
      if (!v) return toast("Choose a video first.", "error");
      fd.append("video", v);
      const g = ctx.el.querySelector("#gpx").files[0];
      if (g) fd.append("gpx", g);
      fd.append("video_start", ctx.el.querySelector("#vstart").value.trim());
      fd.append("stride", ctx.el.querySelector("#stride").value);
    }
    const btn = ctx.el.querySelector("#go");
    busy(btn, true, "Uploading…");
    try {
      const r = await api("/api/surveys", { method: "POST", form: fd });
      toast(`Survey #${r.id} started.`);
      ctx.el.querySelector("#f").reset(); setKind(kind);
      list();
    } catch (err) { toast(err.message, "error"); }
    busy(btn, false);
  };
  await list();
  return () => clearTimeout(timer);
}
