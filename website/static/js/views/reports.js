// Reports list: search, filters, sorting, pagination and exports.
import { api } from "../api.js";
import { S, ago, damageLabel, emptyState, esc, icon, priorityBadge, sevTag, statusTag } from "../ui.js";

export async function render(ctx) {
  const q = ctx.query;
  const st = { q: q.q || "", status: q.status || "", priority: q.priority || "", damage: q.damage || "", sort: q.sort || "priority",
    page: +(q.page || 1), mine: q.mine === "1", has_location: q.has_location || "", survey_id: q.survey_id || "" };
  ctx.setTitle("Reports", "All damage reports", `<a class="btn" id="csv">${icon.download}Export CSV</a><a class="btn primary" href="#/new">${icon.camera}New report</a>`);
  const opt = (v, l, cur) => `<option value="${v}" ${cur === v ? "selected" : ""}>${l}</option>`;
  ctx.el.innerHTML = `
  <div class="card">
    <div class="filters-bar">
      <div class="row" style="position:relative;flex:1;min-width:200px;max-width:340px">
        <input class="input" id="q" placeholder="Search address, note, file or #id" value="${esc(st.q)}" style="padding-left:34px">
        <span style="position:absolute;left:10px;top:9px;width:16px;color:var(--muted)">${icon.search}</span></div>
      <select class="input" id="status">${opt("", "Any status", st.status)}${opt("open,in_progress", "Open + in progress", st.status)}${S.info.statuses.map((s) => opt(s, { open: "Open", in_progress: "In progress", repaired: "Repaired", rejected: "Rejected" }[s], st.status)).join("")}</select>
      <select class="input" id="priority">${opt("", "Any priority", st.priority)}${["P1", "P2", "P3", "P4"].map((p) => opt(p, `${p} ${S.info.priority_levels.find((l) => l.level === p).name}`, st.priority)).join("")}${opt("none", "No damage", st.priority)}</select>
      <select class="input" id="damage">${opt("", "Any damage", st.damage)}${S.info.classes.map((c) => opt(c.name, c.label, st.damage)).join("")}</select>
      <select class="input" id="has_location">${opt("", "Any location", st.has_location)}${opt("1", "With location", st.has_location)}${opt("0", "Without location", st.has_location)}</select>
      <label class="check"><input type="checkbox" id="mine" ${st.mine ? "checked" : ""}>Mine</label>
      <select class="input" id="sort" style="margin-left:auto">${opt("priority", "Most urgent first", st.sort)}${opt("newest", "Newest first", st.sort)}${opt("oldest", "Oldest first", st.sort)}</select>
    </div>
    ${st.survey_id ? `<div class="alert info" style="margin:12px 14px 0">${icon.layers}<div>Survey #${esc(st.survey_id)} only. <a href="#/reports">Show all reports</a></div></div>` : ""}
    <div id="tbl"><div class="empty">Loading…</div></div>
  </div>`;
  const sync = () => {
    const p = new URLSearchParams();
    for (const k of ["q", "status", "priority", "damage", "has_location", "survey_id"]) if (st[k]) p.set(k, st[k]);
    if (st.sort !== "priority") p.set("sort", st.sort);
    if (st.mine) p.set("mine", "1");
    if (st.page > 1) p.set("page", st.page);
    return p;
  };
  const load = async () => {
    const p = sync();
    history.replaceState(null, "", `#/reports${p.toString() ? "?" + p : ""}`);
    ctx.el.closest(".main").querySelector("#csv").href = `/api/export.csv?${p}`;
    p.set("per_page", 25);
    const r = await api(`/api/reports?${p}`);
    const pages = Math.max(1, Math.ceil(r.total / 25));
    ctx.el.querySelector("#tbl").innerHTML = r.rows.length ? `<div class="table-wrap"><table class="table"><thead><tr>
        <th></th><th>Report</th><th>Priority</th><th>Damage</th><th>Severity</th><th>Location</th><th>Status</th><th>Reported by</th></tr></thead><tbody>
      ${r.rows.map((x) => `<tr class="click" data-id="${x.id}">
        <td style="width:76px"><img class="thumb" src="/api/reports/${x.id}/image/thumb" alt="" loading="lazy"></td>
        <td><b>#${x.id}</b><div class="tiny muted">${ago(x.created_at)}</div></td>
        <td>${priorityBadge(x.priority, x.priority_name)}<div class="tiny muted">${esc(x.deadline || "")}</div></td>
        <td>${esc(damageLabel(x.worst_damage))}<div class="tiny muted">${x.damage_types.length > 1 ? `+${x.damage_types.length - 1} more type${x.damage_types.length > 2 ? "s" : ""}` : `${x.n_detections} detection${x.n_detections === 1 ? "" : "s"}`}</div></td>
        <td>${sevTag(x.worst_level)}</td>
        <td class="small" style="max-width:280px">${x.address ? esc(x.address) : x.lat != null ? `<span class="num">${(+x.lat).toFixed(5)}, ${(+x.lon).toFixed(5)}</span>` : `<span class="badge outline">${icon.pin.replace("<svg", '<svg width="12" height="12"')}No location</span>`}</td>
        <td>${statusTag(x.status)}</td><td class="small ink2">${esc(x.reporter || "—")}</td></tr>`).join("")}
      </tbody></table></div>
      <div class="pager"><span>${(st.page - 1) * 25 + 1}–${Math.min(st.page * 25, r.total)} of ${r.total}</span>
        <button class="btn sm" id="prev" ${st.page <= 1 ? "disabled" : ""}>Previous</button><button class="btn sm" id="next" ${st.page >= pages ? "disabled" : ""}>Next</button></div>`
      : emptyState("No reports match", "Change the filters, or add a report with New report.");
    ctx.el.querySelectorAll("tr[data-id]").forEach((tr) => (tr.onclick = () => (location.hash = `#/reports/${tr.dataset.id}`)));
    const prev = ctx.el.querySelector("#prev"), next = ctx.el.querySelector("#next");
    if (prev) prev.onclick = () => { st.page--; load(); };
    if (next) next.onclick = () => { st.page++; load(); };
  };
  for (const k of ["status", "priority", "damage", "sort", "has_location"]) ctx.el.querySelector(`#${k}`).onchange = (e) => { st[k] = e.target.value; st.page = 1; load(); };
  ctx.el.querySelector("#mine").onchange = (e) => { st.mine = e.target.checked; st.page = 1; load(); };
  let t;
  ctx.el.querySelector("#q").oninput = (e) => { clearTimeout(t); t = setTimeout(() => { st.q = e.target.value.trim(); st.page = 1; load(); }, 300); };
  await load();
}
