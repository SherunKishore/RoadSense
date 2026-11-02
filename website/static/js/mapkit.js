// Leaflet helpers: OpenStreetMap base layer, priority pins, clusters and a location picker.
import { esc } from "./ui.js";

export const INDIA = [20.6, 78.9];

export function baseMap(el, opts = {}) {
  const map = L.map(el, { zoomControl: true, attributionControl: true, ...opts });
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19, attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
  }).addTo(map);
  return map;
}

export function pinIcon(priority, status) {
  const p = priority || "none";
  const done = status === "repaired" || status === "rejected";
  const size = p === "P1" ? 26 : 22;
  return L.divIcon({ className: "", html: `<div class="pin ${p}${done ? " done" : ""}">${p === "none" ? "" : p.slice(1)}</div>`,
    iconSize: [size, size], iconAnchor: [size / 2, size / 2], popupAnchor: [0, -size / 2] });
}

export function clusterGroup() {
  return L.markerClusterGroup({
    showCoverageOnHover: false, maxClusterRadius: 44, spiderfyOnMaxZoom: true,
    iconCreateFunction(c) {
      const kids = c.getAllChildMarkers();
      const n = kids.length, p1 = kids.some((m) => m.options.priority === "P1");
      const s = n < 10 ? 34 : n < 100 ? 40 : 46;
      return L.divIcon({ className: "", html: `<div class="cluster${p1 ? " has-p1" : ""}" style="width:${s}px;height:${s}px">${n}</div>`, iconSize: [s, s] });
    },
  });
}

export function accuracyCircle(lat, lon, m) {
  return L.circle([lat, lon], { radius: m, color: "#2a78d6", weight: 1, fillColor: "#2a78d6", fillOpacity: 0.12 });
}

/** The draggable "damage is here" pin used when setting a location (drawn locally; no external marker images). */
export const locIcon = () => L.divIcon({ className: "", iconSize: [30, 40], iconAnchor: [15, 38],
  html: `<svg class="loc-pin" viewBox="0 0 30 40" width="30" height="40" aria-hidden="true"><path d="M15 39C15 39 2 23.5 2 14a13 13 0 0 1 26 0c0 9.5-13 25-13 25z" fill="#111110" stroke="#fff" stroke-width="2"/><circle cx="15" cy="14" r="5" fill="#fab219"/></svg>` });

export const meIcon = () => L.divIcon({ className: "", html: '<div class="me-dot"></div>', iconSize: [16, 16], iconAnchor: [8, 8] });

export const directionsUrl = (lat, lon) => `https://www.google.com/maps/dir/?api=1&destination=${lat},${lon}`;
export const osmUrl = (lat, lon) => `https://www.openstreetmap.org/?mlat=${lat}&mlon=${lon}#map=19/${lat}/${lon}`;

/**
 * Watch the device position until accuracy <= goodM or timeoutMs, reporting each fix.
 * Resolves with the best fix {lat, lon, accuracy}. Requires HTTPS or localhost.
 */
export function bestPosition({ goodM = 30, timeoutMs = 12000, onFix } = {}) {
  return new Promise((resolve, reject) => {
    if (!("geolocation" in navigator)) return reject(new Error("This browser has no location support."));
    if (!window.isSecureContext) return reject(new Error("Location needs a secure connection. Open RoadSense with https:// (phone) or on http://localhost (this computer)."));
    let best = null, done = false;
    const finish = () => {
      if (done) return; done = true;
      navigator.geolocation.clearWatch(id); clearTimeout(t);
      best ? resolve(best) : reject(new Error("No location fix yet. Move outdoors or pick the spot on the map."));
    };
    const id = navigator.geolocation.watchPosition((p) => {
      const fix = { lat: p.coords.latitude, lon: p.coords.longitude, accuracy: p.coords.accuracy };
      if (!best || fix.accuracy < best.accuracy) { best = fix; onFix && onFix(fix); }
      if (fix.accuracy <= goodM) finish();
    }, (e) => {
      if (done) return;
      done = true; navigator.geolocation.clearWatch(id); clearTimeout(t);
      reject(new Error(e.code === 1 ? "Location permission was denied. Allow location for this site, or pick the spot on the map."
        : e.code === 3 ? "Location timed out. Pick the spot on the map instead." : "Location is unavailable here. Pick the spot on the map."));
    }, { enableHighAccuracy: true, maximumAge: 0, timeout: timeoutMs });
    const t = setTimeout(finish, timeoutMs);
  });
}

export const popupHtml = (inner) => `<div class="pop">${inner}</div>`;
export { esc };
