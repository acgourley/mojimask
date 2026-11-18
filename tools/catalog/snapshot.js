// The screen catalog's capture of a screen as HTML (tools/catalog/run.py evaluates this in the app's page): the page
// as it stands, frames and all, frozen into one HTML file that renders without the app's server or scripts.
//
//   frames      each visible frame (Decisions, People) inlined as a srcdoc frame, captured the same way
//   pictures    what scripts draw (canvases: the scan's drawing, the Decisions crops; a video's frame) becomes a
//               picture; pictures the server makes (/api/..., /p/...) are inlined as data: URLs; the app's own files
//               (fonts, assets) point at __WEB__/..., which run.py points at a copy of them
//   styles      every stylesheet inlined, in order
//   state       form values kept; scroll positions kept as data-scroll="top,left" (the viewer restores them)
//   scripts     removed, and on* handlers
//   text        each piece of text that came from strings.js marked with its key (data-k, as strings_record.js's
//               __tKeyOf finds it), the words as shown (data-t) and whether values were filled into them
//               (data-filled), so the viewer can swap in other wording and see it reflow; tooltips' keys as data-tip-k
async () => {
  const dataURL = (blob) => new Promise((r) => { const f = new FileReader(); f.onload = () => r(f.result); f.onerror = () => r(""); f.readAsDataURL(blob); });
  const cache = (window.__snapCache ||= new Map());  // kept while the page is: the next size and step reuse it
  const inline = async (url, base) => {  // a server-made picture as a data: URL
    const abs = new URL(url, base).href;
    if (!cache.has(abs)) cache.set(abs, fetch(abs).then((r) => (r.ok ? r.blob() : null)).then((b) => (b ? dataURL(b) : "")).catch(() => ""));
    return cache.get(abs);
  };
  const isStatic = (u) => /^\/?(fonts|assets)\//.test(u);
  const toWeb = (u) => "__WEB__/" + u.replace(/^\//, "");
  async function fixUrls(css, base) {  // url(...) in styles: the app's files -> __WEB__, server pictures -> data:
    const out = [];
    let last = 0;
    for (const m of css.matchAll(/url\((['"]?)([^'")]+)\1\)/g)) {
      out.push(css.slice(last, m.index));
      const u = m[2];
      out.push(u.startsWith("data:") ? m[0] : isStatic(u) ? `url("${toWeb(u)}")` : `url("${await inline(u, base)}")`);
      last = m.index + m[0].length;
    }
    out.push(css.slice(last));
    return out.join("");
  }

  async function snap(win) {
    const doc = win.document, keyOf = win.__tKeyOf || (() => null), filled = win.__tFilled || new Set(), base = doc.baseURI;
    const live = [...doc.documentElement.querySelectorAll("*")];
    const clone = doc.documentElement.cloneNode(true);
    const cl = [...clone.querySelectorAll("*")];
    for (let i = 0; i < live.length; i++) {
      const a = live[i], b = cl[i];
      if (!b) continue;
      for (const at of [...b.attributes]) if (/^on/i.test(at.name)) b.removeAttribute(at.name);
      if (a.scrollTop || a.scrollLeft) b.setAttribute("data-scroll", `${a.scrollTop},${a.scrollLeft}`);
      const st = b.getAttribute("style");
      if (st && st.includes("url(")) b.setAttribute("style", await fixUrls(st, base));
      const tag = a.tagName;
      if (tag === "CANVAS" || tag === "VIDEO") {
        const img = doc.createElement("img");
        try {
          if (tag === "CANVAS") img.src = a.toDataURL();
          else { const c = doc.createElement("canvas"); c.width = a.videoWidth; c.height = a.videoHeight; c.getContext("2d").drawImage(a, 0, 0); img.src = c.toDataURL("image/jpeg", 0.85); }
        } catch (e) { /* tainted or empty: an empty picture of the same size */ }
        for (const at of ["class", "style", "id", "title"]) if (b.getAttribute(at) != null) img.setAttribute(at, b.getAttribute(at));
        const r = a.getBoundingClientRect();
        if (!b.getAttribute("style")) img.setAttribute("style", `width:${r.width}px;height:${r.height}px`);
        img.setAttribute("data-was", tag.toLowerCase());
        b.replaceWith(img);
      } else if (tag === "IMG") {
        const u = a.getAttribute("src") || "";
        if (u && !u.startsWith("data:")) b.setAttribute("src", isStatic(u) ? toWeb(u) : await inline(u, base));
        b.removeAttribute("srcset");
      } else if (tag === "IFRAME") {
        let inner = "";
        try { if (a.getClientRects().length && a.contentDocument) inner = await snap(a.contentWindow); } catch (e) { /* another origin */ }
        b.removeAttribute("src");
        b.setAttribute("srcdoc", inner);
      } else if (tag === "INPUT") {
        if (a.type === "checkbox" || a.type === "radio") { if (a.checked) b.setAttribute("checked", ""); else b.removeAttribute("checked"); }
        else b.setAttribute("value", a.value);
      } else if (tag === "TEXTAREA") b.textContent = a.value;
      else if (tag === "SELECT") { [...b.options].forEach((o, j) => { if (a.options[j]?.selected) o.setAttribute("selected", ""); else o.removeAttribute("selected"); }); }
    }
    for (const s of [...clone.querySelectorAll("script, link[rel=stylesheet], style, link[rel=preload], link[rel=modulepreload]")]) s.remove();
    let css = "";
    for (const sh of doc.styleSheets) { try { css += [...sh.cssRules].map((r) => r.cssText).join("\n") + "\n"; } catch (e) { /* another origin */ } }
    const style = doc.createElement("style");
    style.textContent = await fixUrls(css, base);
    (clone.querySelector("head") || clone).prepend(style);
    // strings.js text, marked with its key
    const w = doc.createTreeWalker(clone.querySelector("body") || clone, NodeFilter.SHOW_TEXT);
    const nodes = [];
    for (let n; (n = w.nextNode());) nodes.push(n);
    for (const n of nodes) {
      const text = n.data.replace(/\s+/g, " ").trim(), p = n.parentElement;
      if (!text || !p || /^(STYLE|SCRIPT)$/.test(p.tagName)) continue;
      const k = keyOf(text);
      if (!k) continue;
      let el = p;
      if (p.childNodes.length !== 1) { el = doc.createElement("span"); n.replaceWith(el); el.append(n); }
      el.setAttribute("data-k", k); el.setAttribute("data-t", text);
      if (filled.has(text)) el.setAttribute("data-filled", "");
    }
    for (const el of clone.querySelectorAll("[title]")) {  // tooltips too
      const k = keyOf(el.getAttribute("title").trim());
      if (k) el.setAttribute("data-tip-k", k);
    }
    return "<!doctype html>\n" + clone.outerHTML;
  }
  return snap(window);
}
