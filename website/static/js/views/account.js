// Account: who you are, and change your own password.
import { api } from "../api.js";
import { S, busy, esc, icon, toast } from "../ui.js";

export async function render(ctx) {
  ctx.setTitle("Account", "Your profile and password");
  ctx.el.innerHTML = `<div class="grid g2" style="max-width:900px">
    <div class="card"><div class="card-h"><h2>Profile</h2></div><div class="card-b"><dl class="dl">
      <dt>Name</dt><dd>${esc(S.user.name)}</dd><dt>Username</dt><dd class="mono">${esc(S.user.username)}</dd>
      <dt>Role</dt><dd style="text-transform:capitalize">${esc(S.user.role)}</dd></dl></div></div>
    <div class="card"><div class="card-h"><h2>Change password</h2></div><form class="card-b stack" id="pf">
      <div class="field"><label for="cur">Current password</label><input class="input" id="cur" type="password" required autocomplete="current-password"></div>
      <div class="field"><label for="new1">New password</label><input class="input" id="new1" type="password" required autocomplete="new-password"><span class="help">At least 8 characters with letters and numbers.</span></div>
      <div class="field"><label for="new2">Repeat new password</label><input class="input" id="new2" type="password" required autocomplete="new-password"></div>
      <div><button class="btn primary" type="submit">${icon.key}Change password</button></div></form></div></div>`;
  ctx.el.querySelector("#pf").onsubmit = async (e) => {
    e.preventDefault();
    const f = e.target, n1 = f.querySelector("#new1").value;
    if (n1 !== f.querySelector("#new2").value) return toast("The new passwords don't match.", "error");
    const b = f.querySelector("button"); busy(b, true, "Saving…");
    try { await api("/api/me/password", { method: "POST", body: { current: f.querySelector("#cur").value, new: n1 } }); toast("Password changed."); f.reset(); }
    catch (err) { toast(err.message, "error"); }
    busy(b, false);
  };
}
