// In-page dialogs in place of the browser's confirm() / alert() / prompt(). Embedded browsers (the desktop
// app's web view, the Claude app's browser pane) may suppress those: a suppressed confirm() returns false at
// once, so "Delete project…" did nothing at all (2026-11-10). Used by app.html and review.html.
//   if (await Dialog.ask("Delete it?", "Move to Trash", { danger: true })) ...
//   await Dialog.tell("Couldn't split: ...");
//   const v = await Dialog.input("Where is it now?", "/path");   // null if cancelled
// Enter answers OK (Cancel, if that has the focus), Esc cancels, Tab stays inside the dialog; while a dialog is open
// no key reaches the page's own shortcuts. Colours come from theme.css (both pages load it).
(() => {
  const css = document.createElement("style");
  css.textContent = `
  .pa-dlg-back { position: fixed; inset: 0; z-index: 1000; background: rgba(7,6,13,.6); display: flex; align-items: center; justify-content: center; padding: 16px; }
  .pa-dlg { background: var(--panel); color: var(--text); border: 1px solid var(--line); border-radius: 14px;
            box-shadow: 0 12px 40px rgba(0,0,0,.6); max-width: 520px; width: 100%; padding: 20px 20px 16px; font: 14px/1.5 var(--body); }
  .pa-dlg p { margin: 0 0 14px; white-space: pre-line; overflow-wrap: anywhere; }
  .pa-dlg input { width: 100%; box-sizing: border-box; margin: 0 0 14px; padding: 7px 9px; border-radius: 8px; border: 1px solid var(--line);
                  background: var(--bg); color: inherit; font: inherit; }
  .pa-dlg .pa-row { display: flex; gap: 8px; justify-content: flex-end; }
  .pa-dlg button { font: inherit; font-weight: 500; border-radius: 10px; border: 1px solid var(--line2); background: transparent; color: inherit; padding: 8px 16px; cursor: pointer; }
  .pa-dlg button.pa-ok { background: var(--accent); border-color: var(--accent); color: var(--ink); font-weight: 700; }
  .pa-dlg button.pa-ok.pa-danger { background: var(--bad); border-color: var(--bad); color: #1A0505; }`;
  document.head.append(css);

  function open({ text, ok = t("common.ok"), cancel = t("common.cancel"), input = null, danger = false }) {
    return new Promise((done) => {
      const back = document.createElement("div");
      back.className = "pa-dlg-back";
      back.innerHTML = `<div class="pa-dlg" role="dialog" aria-modal="true" tabindex="-1"><p></p>${input !== null ? "<input>" : ""}
        <div class="pa-row">${cancel ? `<button class="pa-cancel"></button>` : ""}<button class="pa-ok${danger ? " pa-danger" : ""}"></button></div></div>`;
      back.querySelector("p").textContent = text;
      back.querySelector(".pa-ok").textContent = ok;
      if (cancel) back.querySelector(".pa-cancel").textContent = cancel;
      const field = back.querySelector("input");
      if (field) field.value = input;
      const finish = (yes) => {
        window.removeEventListener("keydown", keys, true);
        back.remove();
        done(input !== null ? (yes ? field.value : null) : yes);
      };
      const keys = (e) => {  // the page's shortcuts never see a key while the dialog is up
        e.stopImmediatePropagation();
        const stops = [...back.querySelectorAll("input, button")], at = stops.indexOf(document.activeElement);
        if (e.key === "Escape") { e.preventDefault(); finish(false); }
        else if (e.key === "Tab") {  // round the dialog's own field and buttons, never out to the page behind it
          e.preventDefault();
          stops[(at + (e.shiftKey ? -1 : 1) + stops.length) % stops.length].focus();
        }
        else if (e.key === "Enter") {  // a focused Cancel means cancel; focus outside the dialog answers nothing
          e.preventDefault();
          if (back.contains(document.activeElement)) finish(document.activeElement !== back.querySelector(".pa-cancel"));
        }
      };
      window.addEventListener("keydown", keys, true);
      back.querySelector(".pa-ok").onclick = () => finish(true);
      if (cancel) back.querySelector(".pa-cancel").onclick = () => finish(false);
      back.addEventListener("mousedown", (e) => { if (e.target === back) finish(false); });
      document.body.append(back);
      (field || back.querySelector(danger && cancel ? ".pa-cancel" : ".pa-ok")).focus();
      if (field) field.select();
    });
  }
  window.Dialog = {
    ask: (text, ok = t("common.ok"), { danger = false, cancel = t("common.cancel") } = {}) => open({ text, ok, cancel, danger }),
    tell: (text) => open({ text, ok: t("common.ok"), cancel: null }).then(() => undefined),
    input: (text, value = "", ok = t("common.ok")) => open({ text, ok, input: value ?? "" }),
  };
})();
