// The screen catalog's measurement (tools/catalog/run.py evaluates this in the app's page; its viewer, in a captured
// screen as you edit its words): every piece of text on the
// screen, in reading order through the app's frames, with the strings.js key it came from (window.__tKeyOf: in the
// app, strings_record.js's; in the viewer, the keys the capture marked) and how it fits: its lines, whether it's cut off by
// its box, whether it's below the fold, where it is on the screen (box), and about how many more characters fit before it wraps, is cut off or pushes
// something out of its box (tried for real, by lengthening it and putting it back at once). Tooltips are listed too.
(win) => {
  // ---- measuring a screen: every visible piece of text, in reading order, through the frames
  function lineCount(rects) {
    const tops = [];
    for (const r of rects) if (!tops.some((t) => Math.abs(t - r.top) < 3)) tops.push(r.top);
    return tops.length;
  }
  const beyond = (r, b) => r.right > b.right + 1 || r.left < b.left - 1 || r.bottom > b.bottom + 1 || r.top < b.top - 1;
  const ownBox = (el, win) => { let b = el; while (b.parentElement && win.getComputedStyle(b).display.startsWith("inline") && b.tagName !== "BUTTON") b = b.parentElement; return b; };
  // Is the text cut off? true: by its own box (it sticks out of it where overflow is hidden, or an ellipsis / line clamp
  // shortens it); "edge": its whole box is partly outside a box that hides overflow (a card at a strip's edge);
  // "away": wholly outside such a box (not on screen at all); "fold": scrolled out of view in a list or pane (on the
  // screen once you scroll); false: whole.
  function clippedBy(el, rects, win) {
    const own = ownBox(el, win), ob = own.getBoundingClientRect(), sticks = rects.some((r) => beyond(r, ob));
    for (let a = el; a && a !== win.document.documentElement; a = a.parentElement) {
      const cs = win.getComputedStyle(a);
      if (cs.textOverflow === "ellipsis" && a.scrollWidth > a.clientWidth + 1) return true;
      if (cs.webkitLineClamp && cs.webkitLineClamp !== "none" && a.scrollHeight > a.clientHeight + 1) return true;
      if (!/hidden|clip|auto|scroll/.test(cs.overflowX + cs.overflowY)) continue;
      const ar = a.getBoundingClientRect();
      if (!rects.some((r) => beyond(r, ar))) continue;
      if (/auto|scroll/.test(cs.overflowX + cs.overflowY)) return "fold";  // in a list or pane, scrolled out of view
      if (!rects.some((r) => r.right > ar.left && r.left < ar.right && r.bottom > ar.top && r.top < ar.bottom)) return "away";
      return sticks ? true : "edge";
    }
    return false;
  }
  // Overflowing sideways, anywhere from the text's box up: a nowrap button pushing past its row, a row past the page.
  function overflowing(el, win) {
    const o = [];
    for (let a = el; a; a = a.parentElement) o.push(a.scrollWidth > a.clientWidth + 1);
    return o;
  }
  // About how many more characters fit before the text wraps onto another line, is cut off, or pushes something out of
  // its box: tried for real, by lengthening the text a little at a time (and putting it back at once).
  function room(n, win) {
    const el = n.parentElement, orig = n.data, doc = win.document;
    // boxes around the text sized by a picture (a card under its two crops): the text may not make them wider
    const sized = [];
    for (let a = el, i = 0; a && i < 5; a = a.parentElement, i++) {
      const w = a.getBoundingClientRect().width, ms = [...a.querySelectorAll("img, canvas, video")].map((m) => m.getBoundingClientRect());
      // its pictures, side by side, span its width (one picture, or a card's two crops in a row)
      if (ms.length && Math.max(...ms.map((r) => r.right)) - Math.min(...ms.map((r) => r.left)) >= w - 4) sized.push([a, w]);
    }
    const state = () => {
      const rg = doc.createRange(); rg.selectNodeContents(n);
      const rects = [...rg.getClientRects()].filter((r) => r.width > 0 && r.height > 0);
      return { lines: lineCount(rects), cut: rects.length ? clippedBy(el, rects, win) === true : false, over: overflowing(el, win) };
    };
    const s0 = state();
    const fits = (k) => {
      n.data = orig + "n".repeat(k);
      const s = state();
      return s.lines === s0.lines && !s.cut && s.over.every((v, i) => !v || s0.over[i])
        && sized.every(([a, w]) => a.getBoundingClientRect().width <= w + 1);
    };
    let lo = 0, hi = 1;
    try {
      while (hi <= 256 && fits(hi)) { lo = hi; hi *= 2; }
      if (hi > 256) return 256;
      while (hi - lo > 1) { const m = (lo + hi) >> 1; if (fits(m)) lo = m; else hi = m; }
      return lo;
    } finally { n.data = orig; }
  }
  // where it is on the screen: [left, top, width, height] in the window's points (the screenshot's, at 1x)
  const boxOf = (rects, ox, oy) => {
    const l = Math.min(...rects.map((r) => r.left)), t = Math.min(...rects.map((r) => r.top));
    const r = Math.max(...rects.map((x) => x.right)), b = Math.max(...rects.map((x) => x.bottom));
    return [Math.round(ox + l), Math.round(oy + t), Math.round(r - l), Math.round(b - t)];
  };
  function measure(win, out = [], where = "app", ox = 0, oy = 0) {  // ox, oy: the frame's place in the window
    const doc = win.document, keyOf = win.__tKeyOf || (() => null);
    const walk = doc.createTreeWalker(doc.body, NodeFilter.SHOW_ELEMENT | NodeFilter.SHOW_TEXT);
    for (let n = walk.currentNode; n; n = walk.nextNode()) {
      if (n.nodeType === 1) {
        const el = n;
        if (el.tagName === "IFRAME") {
          const r = el.getBoundingClientRect();
          if (el.getClientRects().length) try { measure(el.contentWindow, out, el.id || "frame", ox + r.left + el.clientLeft, oy + r.top + el.clientTop); } catch (e) { /* another origin */ }
          continue;
        }
        if (el.tagName === "SCRIPT" || el.tagName === "STYLE") continue;
        const tip = el.getAttribute("title");
        if (tip && el.getClientRects().length && el.checkVisibility?.({ visibilityProperty: true }) !== false)
          out.push({ kind: "tooltip", where, key: keyOf(tip.trim()), text: tip.trim(), box: boxOf([el.getBoundingClientRect()], ox, oy) });
        continue;
      }
      const text = n.data.replace(/\s+/g, " ").trim(), el = n.parentElement;
      if (!text || !el || /^(SCRIPT|STYLE|OPTION)$/.test(el.tagName)) continue;
      if (el.checkVisibility && !el.checkVisibility({ visibilityProperty: true, opacityProperty: true })) continue;
      const rg = doc.createRange(); rg.selectNodeContents(n);
      const rects = [...rg.getClientRects()].filter((r) => r.width > 0 && r.height > 0);
      if (!rects.length || !rects.some((r) => r.right > 0 && r.left < win.innerWidth)) continue;
      const cut = clippedBy(el, rects, win);
      if (cut === "away") continue;
      const fold = cut === "fold" || !rects.some((r) => r.bottom > 0 && r.top < win.innerHeight);  // needs scrolling to see
      const lines = lineCount(rects);
      out.push({ kind: "text", where, key: keyOf(text), text, lines, cut: cut === true, edge: cut === "edge", fold, box: boxOf(rects, ox, oy),
                 room: cut === true || cut === "edge" ? null : room(n, win) });
    }
    return out;
  }
  return measure(win || window);
}
