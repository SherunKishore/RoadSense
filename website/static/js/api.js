// Fetch wrapper: JSON in/out, the CSRF header on writes, and a single place for "signed out" handling.

export class ApiError extends Error {
  constructor(message, status) { super(message); this.status = status; }
}

let onSignedOut = () => {};
export const setSignedOutHandler = (fn) => { onSignedOut = fn; };

export async function api(path, { method = "GET", body, form, raw, signal } = {}) {
  const headers = { "X-RoadSense": "1" };
  let payload;
  if (form) payload = form;
  else if (raw) { payload = raw; headers["Content-Type"] = "application/octet-stream"; }
  else if (body !== undefined) { payload = JSON.stringify(body); headers["Content-Type"] = "application/json"; }
  let res;
  try {
    res = await fetch(path, { method, headers, body: payload, credentials: "same-origin", signal });
  } catch (e) {
    if (e.name === "AbortError") throw e;
    throw new ApiError("Can't reach the RoadSense server. Is it still running?", 0);
  }
  const type = res.headers.get("content-type") || "";
  const data = type.includes("json") ? await res.json() : await res.blob();
  if (!res.ok) {
    if (res.status === 401 && !path.startsWith("/api/login") && !path.startsWith("/api/setup")) onSignedOut();
    throw new ApiError((data && data.error) || `Request failed (${res.status})`, res.status);
  }
  return data;
}
