// The project's edits have ONE owner: the review server (review_server.py, "the edits document").
// A page (Review in app.html, the editor in review.html, a second window) keeps a working copy, changes it
// however it likes, and push() sends only what differs from the last copy the server confirmed, as small
// ops on paths. The server applies them to ITS copy, so a page holding a stale copy can never overwrite what
// another page saved (the old whole-file save let the last writer win). Every push is one step on the
// server's undo history, shared by every page.
//
//   const sync = EditSync.connect("api/");        // or "/p/<project>/api/" from the app
//   let edits = sync.adopt(await (await fetch("api/init")).json());
//   edits.not_face["142"] = "you";
//   const r = await sync.push(edits, "not a face");  // r.edits: the server's edits plus anything changed
//   edits = r.edits;                                  //   locally while the push was in flight
(function () {
  // lists that are sets: two pages adding different items both keep theirs
  const SETS = new Set(["deleted", "not_real", "dismissed", "shown", "not_same"]);
  const isSet = (p) => (p.length === 1 && SETS.has(p[0])) || (p.length === 3 && p[0] === "people" && p[2] === "pieces");
  const plain = (v) => v !== null && typeof v === "object" && !Array.isArray(v);
  const J = (v) => JSON.stringify(v);
  const clone = (v) => (v === undefined ? undefined : JSON.parse(J(v)));

  // ops that turn a into b: objects compare key by key, set-like lists item by item, anything else whole
  function diff(a, b, path = [], ops = []) {
    for (const k of new Set([...Object.keys(a || {}), ...Object.keys(b || {})])) {
      const va = a ? a[k] : undefined, vb = b ? b[k] : undefined, p = [...path, String(k)];
      if (J(va) === J(vb)) continue;
      if (vb === undefined) ops.push({ op: "del", path: p });
      else if (isSet(p) && Array.isArray(vb) && (va === undefined || Array.isArray(va))) {
        const A = new Set((va || []).map(J)), B = new Set(vb.map(J));
        for (const x of vb) if (!A.has(J(x))) { ops.push({ op: "add", path: p, value: x }); A.add(J(x)); }
        for (const x of va || []) if (!B.has(J(x))) { ops.push({ op: "remove", path: p, value: x }); B.add(J(x)); }
      } else if (plain(va) && plain(vb)) diff(va, vb, p, ops);
      else ops.push({ op: "set", path: p, value: clone(vb) });
    }
    return ops;
  }
  // the same op semantics as review_server.py _apply_op (used to replay local changes onto the server's copy)
  function apply(doc, op) {
    let parent = doc;
    for (const k of op.path.slice(0, -1)) {
      if (!plain(parent[k])) { if (op.op === "del" || op.op === "remove") return; parent[k] = {}; }
      parent = parent[k];
    }
    const k = op.path[op.path.length - 1];
    if (op.op === "set") parent[k] = clone(op.value);
    else if (op.op === "del") delete parent[k];
    else {
      const lst = Array.isArray(parent[k]) ? parent[k] : [], key = J(op.value), have = lst.some((x) => J(x) === key);
      if (op.op === "add" && !have) parent[k] = [...lst, clone(op.value)];
      if (op.op === "remove" && have) parent[k] = lst.filter((x) => J(x) !== key);
    }
  }

  function connect(base) {
    const s = { rev: 0, confirmed: {}, queue: Promise.resolve() };
    const check = async (r) => {
      const j = await r.json();
      if (!r.ok || j.error) throw new Error(j.error || t("sync.err.failed", { status: r.status }));
      return j;
    };
    const get = async (path) => check(await fetch(base + path));
    const post = async (path, body) => check(await fetch(base + path, { method: "POST", headers: { "Content-Type": "application/json" }, body: J(body || {}) }));
    // one request at a time, in order; a failed one rejects only its own caller, never the ones after it
    const run = (fn) => { const p = s.queue.then(fn); s.queue = p.catch(() => {}); return p; };
    // take the server's reply as the confirmed copy; the page's copy becomes it, plus whatever the page
    // changed since `sent` (its copy when the request went out), replayed on top
    const settle = (reply, working, sent) => {
      const later = working ? diff(sent, working) : [];
      s.rev = reply.rev; s.confirmed = clone(reply.edits);
      const next = clone(reply.edits);
      for (const op of later) apply(next, op);
      return { ...reply, edits: next };
    };
    // start from a server reply (/api/init or /api/edits); returns the page's working copy
    s.adopt = (reply) => settle(reply, null, null).edits;
    // send what the working copy changed; resolves to the reply with .edits = the page's new working copy.
    // The server applies the ops to its current copy as they come (no conflict check against s.rev, by design:
    // each op says only what this page changed). opts.undoable = false: bookkeeping that ⌘Z skips.
    s.push = (working, label = "", opts = {}) => run(async () => {
      const sent = clone(working), ops = diff(s.confirmed, sent);
      return settle(await post("ops", { ops, label, ...opts }), working, sent);
    });
    // ⌘Z / ⇧⌘Z: the server takes back its last change (or puts back the last one taken back), whichever page made it
    s.undo = (working) => run(async () => {
      const sent = clone(working);
      return settle(await post("undo"), working, sent);
    });
    s.redo = (working) => run(async () => {
      const sent = clone(working);
      return settle(await post("redo"), working, sent);
    });
    // catch up with changes made elsewhere (another page or window); null if nothing changed
    s.pull = async (working) => {
      const { rev } = await get("rev");
      if (rev === s.rev) return null;
      return run(async () => {
        const sent = clone(working), reply = await get("edits");
        return settle(reply, working, sent);
      });
    };
    // after a failed save: drop the page's unsaved changes and start again from the server's copy
    s.reset = () => run(async () => settle(await get("edits"), null, null));
    return s;
  }
  window.EditSync = { connect };
})();
