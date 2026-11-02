// Photo viewer: draws detection boxes on a canvas, with switchable "model input" and HiResCAM heat-map views.
import { S, clsLabel, esc, icon, toast } from "./ui.js";

const MAX_SIDE = 1600;

export function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("The photo could not be displayed."));
    img.src = src;
  });
}

function colorOf(c) { return S.info.classes.find((k) => k.id === c)?.color || "#888"; }

/** Draw img and boxes (in original-image pixels w x h) onto canvas. */
export function paint(canvas, img, w, h, { dets = [], uncertain = [], gt = [], showLabels = true, scoreKey = "conf" } = {}) {
  const s = Math.min(1, MAX_SIDE / Math.max(w, h));
  canvas.width = Math.round(w * s);
  canvas.height = Math.round(h * s);
  const ctx = canvas.getContext("2d");
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  const lw = Math.max(2, Math.round(canvas.width / 360));
  const fs = Math.max(12, Math.round(canvas.width / 52));
  ctx.font = `600 ${fs}px Inter, system-ui, sans-serif`;
  ctx.textBaseline = "top";
  for (const g of gt) {
    const [x1, y1, x2, y2] = g.box.map((v) => v * s);
    ctx.setLineDash([lw * 3, lw * 2]); ctx.lineWidth = lw; ctx.strokeStyle = "#ffffff";
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    ctx.setLineDash([]);
  }
  const box = (d, dashed) => {
    const [x1, y1, x2, y2] = (d.box_xyxy || d.box).map((v) => v * s);
    const c = colorOf(d.cls);
    ctx.lineWidth = lw; ctx.strokeStyle = c; ctx.setLineDash(dashed ? [lw * 2.5, lw * 2] : []);
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    ctx.setLineDash([]);
    if (!showLabels) return;
    const text = `${clsLabel(d.name)} ${Math.round((d[scoreKey] ?? d.conf) * 100)}%${dashed ? " ?" : ""}`;
    const tw = ctx.measureText(text).width + 10, th = fs + 8;
    const ty = y1 - th >= 0 ? y1 - th : y1;
    ctx.fillStyle = c; ctx.globalAlpha = dashed ? 0.75 : 1;
    ctx.fillRect(x1 - lw / 2, ty, tw, th);
    ctx.globalAlpha = 1; ctx.fillStyle = "#fff";
    ctx.fillText(text, x1 + 5 - lw / 2, ty + 4);
  };
  [...uncertain].forEach((d) => box(d, true));
  [...dets].sort((a, b) => (a[scoreKey] ?? a.conf) - (b[scoreKey] ?? b.conf)).forEach((d) => box(d, false));
}

/**
 * Interactive viewer. opts: {src, width, height, detections, uncertain, enhanced: () => url, explain: async () => res}
 * Returns {el, setGT(gt), redraw()}.
 */
export async function photoViewer(container, opts) {
  const wrap = document.createElement("div");
  wrap.className = "viewer";
  const canvas = document.createElement("canvas");
  canvas.setAttribute("role", "img");
  canvas.setAttribute("aria-label", `Photo with ${opts.detections.length} detected damage${opts.detections.length === 1 ? "" : "s"}`);
  const tools = document.createElement("div");
  tools.className = "tools no-print";
  const modes = [["boxes", "Detections"]];
  if (opts.enhanced) modes.push(["input", "Model input"]);
  if (opts.explain) modes.push(["cam", "Heat map"]);
  tools.innerHTML = `<div class="seg" role="tablist">${modes.map(([k, l], i) => `<button data-m="${k}" class="${i ? "" : "on"}" role="tab">${l}</button>`).join("")}</div>
    ${opts.uncertain?.length ? `<div class="seg"><button data-u title="Boxes just below the model's threshold">Show ${opts.uncertain.length} uncertain</button></div>` : ""}`;
  const busyEl = document.createElement("div");
  busyEl.className = "busy hidden";
  wrap.append(canvas, tools, busyEl);
  container.innerHTML = "";
  container.append(wrap);
  const camPanel = document.createElement("div");
  camPanel.className = "stack tight hidden";
  camPanel.style.marginTop = "10px";
  container.append(camPanel);

  const img = await loadImage(opts.src);
  const w = opts.width || img.naturalWidth, h = opts.height || img.naturalHeight;
  let mode = "boxes", showU = false, gt = [], enhImg = null, cam = null;

  const setBusy = (text) => {
    busyEl.innerHTML = text ? `<div style="text-align:center"><div class="spin"></div>${esc(text)}</div>` : "";
    busyEl.classList.toggle("hidden", !text);
  };
  const redraw = async () => {
    camPanel.classList.add("hidden");
    if (mode === "boxes") return paint(canvas, img, w, h, { dets: opts.detections, uncertain: showU ? opts.uncertain : [], gt, scoreKey: opts.scoreKey });
    if (mode === "input") {
      if (!enhImg) { setBusy("Preparing the model's view…"); try { enhImg = await loadImage(await opts.enhanced()); } finally { setBusy(null); } }
      return paint(canvas, enhImg, w, h, { dets: opts.detections, gt, scoreKey: opts.scoreKey });
    }
    if (!cam) {
      setBusy("Computing heat map (HiResCAM)…");
      try { cam = await opts.explain(); cam.img = await loadImage(cam.combined); } catch (e) { setBusy(null); toast(e.message, "error"); mode = "boxes"; syncTabs(); return redraw(); }
      setBusy(null);
    }
    paint(canvas, cam.img, w, h, { dets: [] });
    camPanel.classList.remove("hidden");
    camPanel.innerHTML = cam.detections.length
      ? `<div class="small ink2">Heat map from <b>${esc(cam.model)}</b>, the ensemble's lead model: red areas pushed its decision most. ${cam.detections.map((d) =>
        `<span class="chip" style="margin:2px">${esc(clsLabel(d.name))} (lead-model score ${d.conf.toFixed(2)}) · ${d.peak_in_box ? "hottest spot inside the box" : "hottest spot outside the box"} · ${Math.round(d.cam_in_box * 100)}% of heat in box</span>`).join("")}</div>`
      : `<div class="small muted">The lead model found nothing above its threshold here, so there is nothing to explain.</div>`;
  };
  const syncTabs = () => tools.querySelectorAll("[data-m]").forEach((b) => b.classList.toggle("on", b.dataset.m === mode));
  tools.addEventListener("click", (e) => {
    const m = e.target.closest("[data-m]");
    if (m) { mode = m.dataset.m; syncTabs(); redraw(); }
    const u = e.target.closest("[data-u]");
    if (u) { showU = !showU; u.classList.toggle("on", showU); u.textContent = `${showU ? "Hide" : "Show"} ${opts.uncertain.length} uncertain`; if (mode !== "boxes") { mode = "boxes"; syncTabs(); } redraw(); }
  });
  await redraw();
  return { el: wrap, setGT(g) { gt = g || []; redraw(); }, redraw };
}

export const viewerLegend = (withGT = false) => `<div class="legend">${S.info.classes.map((c) =>
  `<span><span class="key" style="border-color:${c.color};background:${c.color}22"></span>${esc(c.label)}</span>`).join("")}
  ${withGT ? `<span><span class="key gt"></span>Ground truth</span>` : ""}</div>`;

export { icon };
