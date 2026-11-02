// Applies the saved theme before first paint (light / dark / system), so the page never flashes the wrong theme.
(function () {
  try {
    var t = localStorage.getItem("roadsense-theme");
    if (t === "light" || t === "dark") document.documentElement.setAttribute("data-theme", t);
  } catch (e) { /* storage blocked: follow the system setting */ }
})();
