// Users (admin only): add accounts, change roles, reset passwords, deactivate.
import { api } from "../api.js";
import { S, ago, busy, confirmBox, esc, icon, modal, toast } from "../ui.js";

export async function render(ctx) {
  ctx.setTitle("Users", "Accounts and roles", `<button class="btn primary" id="add">${icon.plus}Add user</button>`);
  const load = async () => {
    const r = await api("/api/users");
    ctx.el.innerHTML = `<div class="card"><div class="table-wrap"><table class="table"><thead><tr><th>Name</th><th>Username</th><th>Role</th><th>Reports</th><th>Last sign-in</th><th>Status</th><th></th></tr></thead><tbody>
      ${r.users.map((u) => `<tr><td><b>${esc(u.name)}</b>${u.id === S.user.id ? ' <span class="badge outline">you</span>' : ""}</td><td class="mono">${esc(u.username)}</td>
        <td><select class="input sm" data-role="${u.id}" ${u.id === S.user.id ? "disabled" : ""} style="width:auto"><option value="inspector" ${u.role === "inspector" ? "selected" : ""}>Inspector</option><option value="admin" ${u.role === "admin" ? "selected" : ""}>Admin</option></select></td>
        <td class="num">${u.reports}</td><td class="small ink2">${u.last_login ? ago(u.last_login) : "never"}</td>
        <td>${u.active ? '<span class="status repaired"><i></i>Active</span>' : '<span class="status rejected"><i></i>Deactivated</span>'}</td>
        <td><div class="row" style="justify-content:flex-end"><button class="btn sm" data-pw="${u.id}">${icon.key}Reset password</button>
          ${u.id === S.user.id ? "" : `<button class="btn sm ${u.active ? "danger" : ""}" data-act="${u.id}" data-on="${u.active ? 0 : 1}">${u.active ? "Deactivate" : "Reactivate"}</button>`}</div></td></tr>`).join("")}
      </tbody></table></div><div class="card-f small ink2"><b>Inspectors</b> analyse photos, create reports and surveys, and edit their own reports' notes and locations. <b>Admins</b> can also change status, delete reports and manage users.</div></div>`;
    ctx.el.querySelectorAll("[data-role]").forEach((s) => (s.onchange = async () => {
      try { await api(`/api/users/${s.dataset.role}`, { method: "PATCH", body: { role: s.value } }); toast("Role updated."); } catch (e) { toast(e.message, "error"); load(); }
    }));
    ctx.el.querySelectorAll("[data-act]").forEach((b) => (b.onclick = async () => {
      const on = b.dataset.on === "1";
      if (!on && !(await confirmBox("Deactivate this account?", "They are signed out and can't sign in until reactivated. Their reports stay.", "Deactivate", true))) return;
      try { await api(`/api/users/${b.dataset.act}`, { method: "PATCH", body: { active: on } }); load(); } catch (e) { toast(e.message, "error"); }
    }));
    ctx.el.querySelectorAll("[data-pw]").forEach((b) => (b.onclick = () => {
      const m = modal("Reset password", `<div class="field"><label for="npw">New password</label><input class="input" id="npw" type="password" autocomplete="new-password"><span class="help">At least 8 characters with letters and numbers. Tell the user in person; they are signed out everywhere.</span></div>`,
        { actions: `<button class="btn" data-close>Cancel</button><button class="btn primary" id="okpw">Set password</button>` });
      m.el.querySelector("#okpw").onclick = async () => {
        try { await api(`/api/users/${b.dataset.pw}`, { method: "PATCH", body: { password: m.el.querySelector("#npw").value } }); toast("Password changed."); m.close(); }
        catch (e) { toast(e.message, "error"); }
      };
    }));
  };
  ctx.el.closest(".main").querySelector("#add").onclick = () => {
    const m = modal("Add user", `<form class="stack" id="uf">
      <div class="field"><label for="un">Full name</label><input class="input" id="un" required maxlength="80"></div>
      <div class="field"><label for="uu">Username</label><input class="input" id="uu" required autocapitalize="none" spellcheck="false"><span class="help">3–32 letters, digits, dots, dashes or underscores.</span></div>
      <div class="field"><label for="ur">Role</label><select class="input" id="ur"><option value="inspector">Inspector</option><option value="admin">Admin</option></select></div>
      <div class="field"><label for="up">Password</label><input class="input" id="up" type="password" required autocomplete="new-password"><span class="help">At least 8 characters with letters and numbers.</span></div></form>`,
    { actions: `<button class="btn" data-close>Cancel</button><button class="btn primary" id="ok">Add user</button>` });
    m.el.querySelector("#ok").onclick = async (e) => {
      busy(e.currentTarget, true, "Adding…");
      try {
        await api("/api/users", { method: "POST", body: { name: m.el.querySelector("#un").value, username: m.el.querySelector("#uu").value, role: m.el.querySelector("#ur").value, password: m.el.querySelector("#up").value } });
        toast("User added."); m.close(); load();
      } catch (err) { busy(m.el.querySelector("#ok"), false); toast(err.message, "error"); }
    };
  };
  await load();
}
