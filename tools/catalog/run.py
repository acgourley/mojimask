"""The screen catalog: scripted paths through the app, as a person would click them, the screen captured after every
step.

    .venv/bin/python tools/catalog/run.py                  # every path (tools/catalog/paths.py), today's wording
    .venv/bin/python tools/catalog/run.py same_no people   # only these paths
    .venv/bin/python tools/catalog/run.py --drafts         # with the draft wording (out/strings_draft.json)
    .venv/bin/python tools/catalog/run.py --list           # the paths, without running them

It is two things at once:
  a test     each path must get through (every step finds what it waits for and clicks); the run ends 1 if any
             doesn't, with a screenshot of where it stopped
  a viewer   out/catalog/<run>/index.html (viewer.html; opens from disk): every path's steps in order, each at a
             13-inch MacBook Air's window (1280 x 740) and the first-launch size (1280 x 880), as its screenshot
             (exactly what WebKit drew) or as live HTML (snapshot.js: the page frozen with each strings.js string
             marked, so other words can be tried in place and measured again), with every piece of text on it, its
             key and how it fits (measure.js: lines, characters to spare, cut off, below the fold); and every key the
             run reached, with the steps it shows on. What to look at when writing or translating the app's words.
A step that changes nothing on screen isn't captured again. --trace also keeps Playwright's own trace of each path.

Paths name what to click by strings.js key (ui.click("decisions.a.different_people")), not by its words, so a path
keeps working when the wording changes. Which key each piece of text came from is recorded as the pages run:
strings_record.js, appended to strings.js as WebKit loads it (the app itself has nothing for the catalog), which also
lays the drafts over today's wording with --drafts. Everything runs in its own app server (as people run it: no
developer mode; on a free port) and its own projects folder (out/catalog/<run>/projects), each path on
a fresh copy of a scanned jazz project (no answers, opening on the result screen like a new project), in WebKit: the
engine the Mac app's window uses, so text wraps where it will there.

Only jazz: the screenshots show the clip's faces, and test_clips/jazz.mp4 is the one public-domain clip (AGENTS.md).
A project whose video isn't that file is refused.

Needs Playwright and its WebKit (a download, once):
    uv pip install --python .venv/bin/python -r tools/catalog/requirements.txt
    .venv/bin/python -m playwright install webkit
"""
import argparse
import contextlib
import filecmp
import glob
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from mojimask.projects import copy_scan  # noqa: E402

JAZZ = os.path.join(ROOT, "test_clips", "jazz.mp4")
SIZES = [(1280, 740), (1280, 880)]  # a 13-inch Air's window as the app opens it; the first-launch size (desktop.py)
MEASURE = open(os.path.join(HERE, "measure.js")).read()
RECORD = open(os.path.join(HERE, "strings_record.js")).read()
SNAPSHOT = open(os.path.join(HERE, "snapshot.js")).read()


# ---- the project every path starts from
def is_jazz(proj):
    """Whether a project's video is test_clips/jazz.mp4 (or a byte-for-byte copy of it)."""
    try:
        video = json.load(open(os.path.join(proj, "meta.json")))["video"]
        return os.path.isfile(video) and filecmp.cmp(video, JAZZ, shallow=False)
    except (OSError, ValueError, KeyError):
        return False


def find_base():
    """The newest scanned jazz project in out/projects made by this scan version (not a copy)."""
    from mojimask.scan import newalgo
    best = None
    for p in glob.glob(os.path.join(ROOT, "out", "projects", "*")):
        if "_fresh" in p or not os.path.exists(os.path.join(p, "proto_tracks.json")):
            continue
        try:
            if json.load(open(os.path.join(p, "proto_tracks.json"))).get("scan_version") != newalgo.SCAN_VERSION:
                continue
        except (OSError, ValueError):
            continue
        if is_jazz(p) and (best is None or os.path.getmtime(os.path.join(p, "meta.json")) > os.path.getmtime(os.path.join(best, "meta.json"))):
            best = p
    return best


# ---- the app server the run uses
class Server:
    def __init__(self, projects):
        s = socket.socket(); s.bind(("127.0.0.1", 0)); self.port = s.getsockname()[1]; s.close()
        env = {**os.environ, "MOJIMASK_PROJECTS": projects}
        for k in ("MOJIMASK_APP", "MOJIMASK_DEV"):  # the app as people run it
            env.pop(k, None)
        self.log = open(os.path.join(os.path.dirname(projects), "app_server.log"), "w")
        self.proc = subprocess.Popen([sys.executable, "app_server.py", "--port", str(self.port)], cwd=ROOT, env=env,
                                     stdout=self.log, stderr=subprocess.STDOUT)
        self.url = f"http://127.0.0.1:{self.port}"
        for _ in range(100):
            try:
                urllib.request.urlopen(self.url + "/api/app", timeout=1); return
            except OSError:
                time.sleep(0.2)
        raise RuntimeError("the app server didn't start: see " + self.log.name)

    def api(self, path, body=None):
        req = urllib.request.Request(self.url + path, data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="GET" if body is None else "POST")
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.loads(r.read() or b"{}")

    def stop(self):
        self.proc.send_signal(signal.SIGINT)  # it stops its project servers on the way out
        try:
            self.proc.wait(20)
        except subprocess.TimeoutExpired:
            self.proc.kill()


# ---- what a path does
class Failed(Exception):
    pass


class UI:
    """A path's hands and eyes: find and click things by their strings.js key, answer through the app's API (as the
    Decisions page does, for what a path only needs done, not seen), and capture the screen after every step."""

    def __init__(self, page, server, project, label, out, name, steps):
        self.page, self.server, self.project, self.label = page, server, project, label
        self.out, self.name, self.steps = out, name, steps
        self.sig, self.capture, self.size = None, True, SIZES[0]

    # what a key reads as in the app's page (strings.js's t(), the drafts laid over it if the run uses them)
    def words(self, key):
        return self.page.evaluate("(k) => t(k)", key)

    # the visible element holding a piece of text that reads exactly so (a heading with a count after it, a button
    # with its key hint: the words' own text node), within an element showing `within` if given
    FIND = """([text, within, nth, tip]) => {
      const vis = (el) => { const r = el.getBoundingClientRect(); return r.width > 0 && r.height > 0 && getComputedStyle(el).visibility !== "hidden"; };
      const reads = (n, t) => {  // exactly the words, or the words and a count ("Answered 10")
        const d = n.data.replace(/\\s+/g, " ").trim();
        return (d === t || (d.startsWith(t + " ") && /^[\\d,]+$/.test(d.slice(t.length + 1)))) && n.parentElement && vis(n.parentElement);
      };
      const all = (root, t, anchor) => {  // in the page's order (anchor: also a caption that goes on, "Green box · frame 68")
        if (tip && !anchor) return [...root.querySelectorAll("[title]")].filter((el) => el.title.trim() === t && vis(el));
        const out = [], w = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
        for (let n; (n = w.nextNode());) if (reads(n, t) || (anchor && n.data.trim().startsWith(t + " \u00b7 ") && vis(n.parentElement))) out.push(n.parentElement);
        return out;
      };
      if (!within) return all(document.body, text)[nth] || null;
      // from the box showing `within` outwards: the nearest one that also shows the text
      for (let a = all(document.body, within, true)[0]; a; a = a.parentElement) { const el = all(a, text)[nth]; if (el) return el; }
      return null;
    }"""
    # what the screen shows, to tell a step that changed nothing (numbers aside: a playing clip's frame count); with
    # the tooltips, as two rows can read alike and differ only there (two rules' answers)
    SIG = """() => {
      const tips = (d) => [...d.querySelectorAll("[title]")].filter((el) => el.getClientRects().length).map((el) => el.title).join("|");
      const text = (w) => { try { return w.document.body.innerText + tips(w.document) + [...w.document.querySelectorAll("iframe")]
        .filter((f) => f.getClientRects().length).map((f) => text(f.contentWindow)).join("\\n"); } catch (e) { return ""; } };
      return text(window).replace(/\\d+/g, "#");
    }"""

    def find(self, key, timeout=30000, within=None, nth=0, tip=False, text=None):
        """The visible element showing exactly the words of key (the nth such, in the page's order), in any of the
        app's frames. within: inside the nearest box that also shows these words, e.g. a project's name. tip: the
        element whose tooltip reads so (a picture). text: these words instead of the key's (several keys' words
        put together, as a list row's "No · rule")."""
        text = text if text is not None else self.words(key)
        end = time.time() + timeout / 1000
        while time.time() < end:
            for fr in self.page.frames:
                try:
                    el = fr.evaluate_handle(self.FIND, [text, within, nth, tip]).as_element()
                    if el:
                        return el, text
                except Exception:  # noqa: BLE001  (a frame navigating away meanwhile)
                    pass
            time.sleep(0.2)
        raise Failed(f"nothing on screen {'has the tooltip' if tip else 'reads'} {key} ({text!r})" + (f", #{nth + 1}" if nth else ""))

    def wait(self, key, timeout=30000, **kw):
        _, text = self.find(key, timeout, **kw)
        self.step(f"Shows \u201c{text}\u201d")

    def click(self, key, timeout=30000, **kw):
        end = time.time() + timeout / 1000
        while True:  # found, then redrawn before the click landed (a button whose label is refreshed): find it again
            el, text = self.find(key, max(1000, (end - time.time()) * 1000), **kw)
            try:
                el.click(timeout=5000)
                break
            except Exception as e:  # noqa: BLE001
                if "not attached" not in str(e) or time.time() > end:
                    raise
                time.sleep(0.3)
        self.settle()
        self.step(f"Clicked \u201c{text}\u201d" if not kw.get("tip") else f"Clicked the picture \u201c{text}\u201d")

    def press(self, keyname, on):
        """A key pressed on a page: `on` is a key whose words are on that page (the key goes to the frame showing
        them, as a person's would after clicking there)."""
        el, _ = self.find(on)
        el.evaluate("(el) => el.ownerDocument.defaultView.focus()")
        self.page.keyboard.press(keyname)
        self.settle()
        self.step(f"Pressed {keyname}")

    def goto(self, hash=""):
        self.page.goto(f"{self.server.url}/{'#' + hash if hash else ''}")
        self.settle()
        self.step("Opened the app")

    def open_project(self):
        """From the Projects list, as a person would: Open on the project's card."""
        self.goto()
        self.click("projects.open", within=self.label)

    def open_decisions(self):
        """The project opened from the Projects list, then Refine the covers: the Decisions step."""
        self.open_project()
        self.click("result.refine", timeout=60000)

    # answering through the project's API (the Decisions page's own way): what a path needs done before it starts,
    # not seen; the viewer shows the note in its place
    def _answer(self, id_, answer):
        self.server.api(f"/p/{self.project}/api/decisions", {"id": id_, "answer": answer})

    def _aside(self, note):
        self.steps.append({"label": note + " (through the app's API, not on screen)", "aside": True})

    def answer(self, kinds, answer):
        """Every open question of these kinds answered so, until none is left: one answer can open others."""
        for _ in range(500):
            ds = self.server.api(f"/p/{self.project}/api/decisions")["decisions"]
            todo = [d for d in ds if d["kind"] in kinds and not d.get("by")]
            if not todo:
                return
            for d in todo:
                self._answer(d["id"], answer)
        raise Failed(f"questions of kinds {kinds} kept opening")

    def answered(self, answers, note):
        """These questions answered (id -> answer), in this order."""
        for id_, a in answers.items():
            self._answer(id_, a)
        self._aside(note)

    def faces_answered(self):
        """Every is-it-a-face question answered as a face (part of their face, for an overlap): what's left are the
        who's-who questions."""
        self.answer(("track", "blip"), "a face")
        self.answer(("overlap",), "same face")
        self._aside("Answered every is-it-a-face question")

    def settle(self, most=3000):
        """After an action: until the screen changes (at most `most` ms), then a moment for it to finish."""
        end = time.time() + most / 1000
        while time.time() < end and self.page.evaluate(self.SIG) == self.sig:
            time.sleep(0.15)
        self.page.wait_for_timeout(500)

    @contextlib.contextmanager
    def quietly(self, note):
        """Steps done but not captured: getting to where a path starts, already shown by another path. They still
        have to get through (the test); the viewer shows the note in their place."""
        self.capture = False
        try:
            yield
        finally:
            self.capture = True
            self.steps.append({"label": note, "aside": True})

    def shot(self, name, note="", wait=0):
        """Name this screen (and say what to look at), after `wait` ms for it to finish drawing: the step just
        captured, or, if the screen has changed since (an animation going on), a new step."""
        if wait:
            self.page.wait_for_timeout(wait)
        self.step("A moment later", name=name, note=note)

    def step(self, label, name=None, note=""):
        """Capture the screen after a step, at each size: a screenshot, the page as HTML (snapshot.js) and its text,
        measured (measure.js). A step that changed nothing on screen isn't captured again; naming it names the last."""
        if not self.capture:
            return
        sig = self.page.evaluate(self.SIG)
        last = next((s for s in reversed(self.steps) if not s.get("aside")), None)
        if sig == self.sig and last is not None:
            if name:
                last.update(name=name, note=note or last.get("note", ""))
            return
        self.sig = sig
        n = sum(not s.get("aside") for s in self.steps) + 1
        entry = {"n": n, "label": label, "name": name, "note": note, "sizes": {}}
        for w, h in SIZES:
            z = f"{w}x{h}"
            self.resize((w, h), 500)
            stem = f"{self.name}/{n:02d}-{z}"
            self.page.screenshot(path=os.path.join(self.out, stem + ".png"))
            html_ = self.page.evaluate(SNAPSHOT)
            with open(os.path.join(self.out, stem + ".snap.js"), "w") as f:
                f.write(f"CATALOG_SNAP({json.dumps(stem)}, {json.dumps(html_)});\n")
            entry["sizes"][z] = {"png": stem + ".png", "snap": stem + ".snap.js", "text": self.page.evaluate(MEASURE)}
        self.resize(SIZES[0], 300)
        self.sig = self.page.evaluate(self.SIG)  # as captured (it may have finished loading meanwhile)
        self.steps.append(entry)

    def resize(self, size, wait):
        """The window at this size, and `wait` ms for the page to lay itself out again (nothing if it's already so)."""
        if size != self.size:
            self.page.set_viewport_size({"width": size[0], "height": size[1]})
            self.page.wait_for_timeout(wait)
            self.size = size


# ---- the viewer: a static page, opened from disk (index.html, data.js, the captures, a copy of the app's fonts)
def write_viewer(out, results, drafts):
    shutil.copy(os.path.join(HERE, "viewer.html"), os.path.join(out, "index.html"))
    with open(os.path.join(out, "measure.js"), "w") as f:
        f.write("window.CATALOG_MEASURE = " + MEASURE.strip() + ";\n")
    web = os.path.join(out, "web")
    for d in ("fonts", os.path.join("assets", "gallery")):
        if not os.path.exists(os.path.join(web, d)):
            shutil.copytree(os.path.join(ROOT, "web", d), os.path.join(web, d))
    info = {"when": time.strftime("%Y-%m-%d %H:%M"), "wording": f"draft ({drafts})" if drafts else "today's",
            "sizes": [f"{w}x{h}" for w, h in SIZES], "paths": results}
    with open(os.path.join(out, "data.js"), "w") as f:
        f.write("window.CATALOG = " + json.dumps(info) + ";\n")


# ---- running
def main():
    import paths as P
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("only", nargs="*", help="run only these paths (their names: --list)")
    ap.add_argument("--drafts", nargs="?", const=os.path.join(ROOT, "out", "strings_draft.json"),
                    help="with the draft wording (default out/strings_draft.json)")
    ap.add_argument("--base", help="the scanned jazz project to copy (default: the newest in out/projects)")
    ap.add_argument("--out", help="where the run goes (default out/catalog/<date-time>)")
    ap.add_argument("--headed", action="store_true", help="show the browser while it runs")
    ap.add_argument("--trace", action="store_true",
                    help="also record Playwright traces (<path>/trace.zip, for `.venv/bin/python -m playwright show-trace`): "
                         "every action, for finding why a path fails; ~40 MB a path")
    ap.add_argument("--keep-projects", action="store_true", help="keep the run's project copies (<run>/projects)")
    ap.add_argument("--list", action="store_true", help="list the paths and stop")
    a = ap.parse_args()
    todo = [p for p in P.PATHS if not a.only or p.__name__ in a.only]
    if a.list or not todo:
        for p in P.PATHS:
            print(f"{p.__name__:20} {p.title}")
        return 0 if a.list else 2
    base = os.path.abspath(a.base) if a.base else find_base()
    if not base or not is_jazz(base):
        sys.exit("No scanned jazz project to copy: scan test_clips/jazz.mp4 in the app first (or pass --base). "
                 "Only jazz: its frames are the public-domain ones.")
    out = os.path.abspath(a.out or os.path.join(ROOT, "out", "catalog", time.strftime("%Y%m%d-%H%M%S")))
    projects = os.path.join(out, "projects")
    os.makedirs(projects)
    print(f"base: {base}\nout:  {out}")
    from playwright.sync_api import sync_playwright
    record = RECORD.replace("__DRAFTS__", json.dumps(json.load(open(a.drafts)) if a.drafts else {}))

    def strings_js(route):  # every page's strings.js (the framed pages' too), with strings_record.js after it
        r = route.fetch()
        route.fulfill(response=r, body=r.text() + "\n" + record)
    server = Server(projects)
    results = []
    try:
        with sync_playwright() as pw:
            browser = pw.webkit.launch(headless=not a.headed)
            for p in todo:
                name = p.__name__
                os.makedirs(os.path.join(out, name))
                pid, label = f"catalog_{name}", f"jazz (catalog: {name})"
                copy_scan(base, os.path.join(projects, pid), label)
                steps = []
                r = {"id": name, "title": p.title, "doc": (p.__doc__ or "").strip(), "steps": steps, "ok": True}
                ctx = browser.new_context(viewport={"width": SIZES[0][0], "height": SIZES[0][1]}, device_scale_factor=2)
                if a.trace:
                    ctx.tracing.start(screenshots=True, snapshots=True)  # every action, for `playwright show-trace`
                ctx.route("**/strings.js", strings_js)
                page = ctx.new_page()
                ui = UI(page, server, pid, label, out, name, steps)
                t0 = time.time()
                try:
                    p(ui)
                except Exception as err:  # noqa: BLE001  (a step that didn't get through: the path fails, the run goes on)
                    r.update(ok=False, error="".join(traceback.format_exception_only(type(err), err)).strip())
                    try:
                        page.screenshot(path=os.path.join(out, name, "failed.png")); r["failed_png"] = f"{name}/failed.png"
                    except Exception:  # noqa: BLE001
                        pass
                if a.trace:
                    ctx.tracing.stop(path=os.path.join(out, name, "trace.zip"))
                ctx.close()
                results.append(r)
                n = sum(not s.get("aside") for s in steps)
                print(f"{'ok    ' if r['ok'] else 'FAILED'} {name:20} {n} steps, {time.time() - t0:.0f} s" + ("" if r["ok"] else f"\n       {r['error']}"))
            browser.close()
    finally:
        server.stop()
        if not a.keep_projects:  # scratch: copies of the base project the paths answered in
            shutil.rmtree(projects, ignore_errors=True)
        write_viewer(out, results, a.drafts)
        print(f"open:  {os.path.join(out, 'index.html')}")
    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
