// About the model: what it detects, honest validation/test numbers, calibration, limitations and photo tips.
import { S, esc, icon, pct } from "../ui.js";

const f3 = (v) => (v == null ? "—" : (+v).toFixed(3));

export async function render(ctx) {
  ctx.setTitle("About the model", "How RoadSense decides, and how accurate it is");
  const m = S.info.model;
  const t = m.test;
  const cal = t?.calibration_check;
  ctx.el.innerHTML = `
  <div class="stack" style="gap:16px">
    <div class="grid g3">
      <div class="card kpi"><div class="label">Photos judged correctly (test)</div><div class="value num">${t ? pct(t.combined.ensemble.image_level.photo_accuracy, 1) : "—"}</div>
        <div class="note">damaged vs. undamaged, ${t ? t.combined.ensemble.image_level.damaged_photos + t.combined.ensemble.image_level.clean_photos : "—"} unseen test photos</div></div>
      <div class="card kpi"><div class="label">Right damage type when it flags a photo</div><div class="value num">${t ? pct(t.combined.ensemble.image_level.right_type_when_flagged, 1) : "—"}</div><div class="note">at least one detected type is correct</div></div>
      <div class="card kpi"><div class="label">Box-level F1 (test)</div><div class="value num">${t ? f3(t.combined.ensemble.at_thresholds.f1) : "—"}</div><div class="note">exact damage outlines are the hard part</div></div>
    </div>
    <div class="card"><div class="card-h"><h2>What runs when you analyse a photo</h2></div><div class="card-b stack">
      <ol style="margin:0;padding-left:20px;display:grid;gap:6px" class="ink2">
        <li><b>Detail enhancement.</b> A guided filter sharpens fine texture (cracks) on the brightness channel at 720 px.</li>
        <li><b>Three YOLO11s detectors vote.</b> ${m.members.map((x) => esc(x)).join("; ")}.</li>
        <li><b>Weighted boxes fusion</b> (IoU ${m.wbf_iou}) averages overlapping boxes. A box only one model sees keeps a third of its score, so agreement counts.</li>
        <li><b>Per-damage thresholds</b> tuned for the best F1 on the validation photos: ${Object.entries(m.thresholds).map(([k, v]) => `${esc(k.slice(0, 3))} ${v}`).join(", ")} (fused score).</li>
        <li><b>Calibrated confidence.</b> The percentage shown is how often boxes with that score were real damage on validation photos.</li>
        <li><b>Severity</b> from damage type, size (corrected for camera perspective) and confidence; then the <b>repair priority</b> P1–P4 with the recommended treatment.</li>
        <li><b>Heat maps</b> (HiResCAM) from the lead model show which pixels drove each detection.</li>
      </ol></div></div>
    <div class="grid g2">
      <div class="card"><div class="card-h"><h2>Is the confidence honest?</h2><span class="sub">test photos</span></div>
        ${cal ? `<div class="table-wrap"><table class="table"><thead><tr><th>Shown confidence</th><th>Boxes</th><th>Average shown</th><th>Actually correct</th></tr></thead><tbody>
          ${cal.bins.map((b) => `<tr><td class="num">${Math.round(b.range[0] * 100)}–${Math.round(b.range[1] * 100)}%</td><td class="num">${b.boxes}</td><td class="num">${pct(b.mean_confidence)}</td><td class="num">${pct(b.actually_correct)}</td></tr>`).join("")}
          </tbody></table></div><div class="card-f small ink2">Average gap between shown and actual: ${pct(cal.expected_calibration_error, 1)} (expected calibration error).</div>` : `<div class="card-b muted">Not available.</div>`}
      </div>
      <div class="card"><div class="card-h"><h2>Limits to keep in mind</h2></div><div class="card-b stack tight small ink2">
        <div>• Trained on photos from a vehicle-mounted phone (RDD2022 India and Japan). Close-ups, drone views or very different roads are less reliable.</div>
        <div>• Transverse cracks are rare in the Indian training data (68 boxes), so they are the least reliable type on Indian roads.</div>
        <div>• Severity and priority are transparent rules (configs/severity.json, configs/priority.json), not measurements. Crack width can't be measured from a photo.</div>
        <div>• Long cracks are sometimes split into several boxes, or several cracks merged into one.</div>
        <div>• An inspector should confirm P1 reports before crews are sent.</div>
      </div></div>
    </div>
    <div class="card"><div class="card-h"><h2>Location (GPS) sources</h2></div><div class="card-b grid g2 small ink2">
      <div><b>Photo GPS (EXIF):</b> the position the camera saved in the photo. Best for photos taken on site with location tagging on. Phone browsers often remove it when a photo is picked from the gallery.</div>
      <div><b>Device location:</b> the phone's live GPS when you use <i>Take photo</i> or <i>My location</i>. RoadSense waits for ±30 m or better (up to 12 s) and stores the accuracy. It needs HTTPS on phones.</div>
      <div><b>Map pin:</b> tap or drag on the map. Use it when GPS is missing or off.</div>
      <div><b>GPX track:</b> for dash-cam videos; each analysed frame is placed by its time along the track.</div>
      <div>Street addresses come from OpenStreetMap (Nominatim). Only the coordinates are sent.</div>
      <div>Reports within 25 m of each other are flagged as possibly the same damage.</div>
    </div></div>
    <div class="small muted">Settings tuned on ${esc(m.tuned || "—")} · models run on ${esc(S.info.device)}.</div>
  </div>`;
}

export { icon };
