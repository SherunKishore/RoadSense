// Sign-in screen, and first-run creation of the admin account (only offered on the computer running RoadSense).
import { api } from "../api.js";
import { busy, esc, icon, logo } from "../ui.js";

export function renderAuth(root, session, done) {
  const setup = session.setup_needed;
  root.innerHTML = `
  <div class="auth">
    <section class="auth-art">
      <div class="brand" style="padding:0"><div class="brand-mark">${logo}</div><div><b style="font-size:16px">RoadSense</b><span>Road damage survey</span></div></div>
      <div>
        <h2>Find road damage early, map it, and fix the worst first.</h2>
        <p>Photograph a road and RoadSense detects cracks and potholes, rates how severe they are, sets a repair priority and puts the damage on the map.</p>
        <div class="feat">
          <div>${icon.camera}<span><b>Four damage types.</b> Longitudinal, transverse and alligator cracks, and potholes, found by three AI models voting together.</span></div>
          <div>${icon.pin}<span><b>GPS from the photo or the phone.</b> Every report records where its location came from and how accurate it is.</span></div>
          <div>${icon.clock}<span><b>Repair priorities.</b> P1 within 48 hours down to routine maintenance, with the recommended treatment.</span></div>
          <div>${icon.eye}<span><b>Explainable.</b> Heat maps show which part of the photo each detection is based on.</span></div>
        </div>
      </div>
      <div class="small" style="color:#8f8d86">Final-year project · YOLO11 ensemble trained on RDD2022 India + Japan</div>
      <svg class="road" viewBox="0 0 200 200" fill="none" stroke="#fff" stroke-width="6"><path d="M60 200 95 0M140 200 105 0"/></svg>
    </section>
    <section class="auth-form">
      <div class="card">
        <div class="card-b stack" style="padding:28px">
          <div><h1 style="font-size:21px">${setup ? "Create the admin account" : "Sign in"}</h1>
          <p class="muted" style="margin-top:4px">${setup ? "No accounts exist yet. This first account can add inspectors later." : "Use the account your RoadSense admin gave you."}</p></div>
          ${setup && !session.setup_allowed ? `<div class="alert warn">${icon.alert}<div>Create the first admin from the computer running RoadSense (open <b>http://localhost:8000</b> there), or run <span class="mono">python website/manage.py create-user --role admin</span>.</div></div>` : `
          <form class="stack" id="f" autocomplete="on">
            ${setup ? `<div class="field"><label for="name">Your name</label><input class="input" id="name" required maxlength="80" autocomplete="name"></div>` : ""}
            <div class="field"><label for="username">Username</label><input class="input" id="username" required autocomplete="username" autocapitalize="none" spellcheck="false"></div>
            <div class="field"><label for="password">Password</label><input class="input" id="password" type="password" required autocomplete="${setup ? "new-password" : "current-password"}">
              ${setup ? `<span class="help">At least 8 characters, with letters and numbers.</span>` : ""}</div>
            <div id="msg"></div>
            <button class="btn primary lg block" type="submit">${setup ? "Create account" : "Sign in"}</button>
          </form>`}
        </div>
      </div>
    </section>
  </div>`;
  const f = root.querySelector("#f");
  if (!f) return;
  f.onsubmit = async (e) => {
    e.preventDefault();
    const btn = f.querySelector("button[type=submit]");
    busy(btn, true, setup ? "Creating…" : "Signing in…");
    const body = { username: f.username.value.trim(), password: f.password.value };
    if (setup) body.name = f.querySelector("#name").value.trim();
    try {
      await api(setup ? "/api/setup" : "/api/login", { method: "POST", body });
      done();
    } catch (err) {
      busy(btn, false);
      f.querySelector("#msg").innerHTML = `<div class="alert error">${icon.alert}<div>${esc(err.message)}</div></div>`;
    }
  };
  f.querySelector("input").focus();
}
