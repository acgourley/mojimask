"""mojimask for Mac: the engine and its local web UI in a native window.

    .venv/bin/python desktop.py          # from a checkout
    ./packaging/build_mac.sh              # -> dist/Mojimask.app (PyInstaller)

The window is macOS's own WebKit view (pywebview) showing app.html from a local server on 127.0.0.1
(app_server). Nothing leaves the computer. The page gets a small native API: Open and Save dialogs and
Show in Finder; files dropped on the window are used from where they are (their real path comes from
macOS through pywebview). Per-project review servers are child processes; inside the bundle they are this
same executable started with --review-server.

Self-test (MOJIMASK_SELFTEST=1, see selftest()): drives the real window -- the Choose video and Export buttons,
the real Open and Save panels, a drop -- and writes a JSON report, then quits.
"""
import json
import os
import sys
import threading
import time

from mojimask import about


def run_review_server(args):
    sys.argv = ["review_server.py", *args]
    import runpy
    runpy.run_module("mojimask.review_server", run_name="__main__")


VIDEO_EXT = (".mov", ".mp4", ".m4v")
PHOTO_EXT = (".jpg", ".jpeg", ".png", ".heic", ".heif", ".tif", ".tiff", ".webp")  # (video_io reads these: a photo is a clip of one frame)
OPEN_TYPES = ("Videos and photos (" + ";".join(f"*{e};*{e.upper()}" for e in VIDEO_EXT + PHOTO_EXT) + ")",)
PHOTO_OUT = (".jpg", ".jpeg", ".png")  # what an exported photo can be saved as


class Api:
    """Called from app.html as window.pywebview.api.<name>(...); every method returns JSON-able values."""

    def __init__(self):
        self.window = None
        self.test_pick = None  # self-test only: the file a person would pick in the Open panel

    def choose_video(self):
        import webview
        if self.test_pick:
            path, self.test_pick = self.test_pick, None
            return path
        r = self.window.create_file_dialog(webview.FileDialog.OPEN, file_types=OPEN_TYPES)
        return r[0] if r else None

    def locate_video(self, name):
        return self.choose_video()

    def save_video(self, name, directory=""):
        import webview
        photo = name.lower().endswith(PHOTO_OUT)  # the page names a photo's export .jpg
        if directory and not os.path.isdir(directory):
            directory = ""
        r = self.window.create_file_dialog(webview.FileDialog.SAVE, save_filename=name,
                                           directory=directory or os.path.expanduser("~/Pictures" if photo else "~/Movies"))
        if not r:
            return None
        path = r if isinstance(r, str) else r[0]
        exts, default = (PHOTO_OUT, ".jpg") if photo else (VIDEO_EXT, ".mp4")
        return path if path.lower().endswith(exts) else path + default

    def reveal(self, path):
        if os.environ.get("MOJIMASK_SELFTEST"):
            return True  # don't pop Finder windows during the self-test
        from AppKit import NSWorkspace
        from Foundation import NSURL
        if path and os.path.exists(path):
            NSWorkspace.sharedWorkspace().activateFileViewerSelectingURLs_([NSURL.fileURLWithPath_(path)])
        return True


def force_dark():
    """The app is dark only (its pages say color-scheme: dark): the window, its title bar, menus and the Open / Save
    panels follow, even when macOS is set to light."""
    try:
        from AppKit import NSApplication, NSAppearance, NSAppearanceNameDarkAqua
        from PyObjCTools import AppHelper
        AppHelper.callAfter(lambda: NSApplication.sharedApplication().setAppearance_(
            NSAppearance.appearanceNamed_(NSAppearanceNameDarkAqua)))
    except Exception:  # pragma: no cover  (not macOS)
        pass


WINDOW_SIZE = (1280, 880)  # the first launch's size, in points, when the screen has room
WINDOW_MIN = (980, 680)    # the pages' own minimum: the Edit step's video, timeline and side panel; Decisions' crops
WINDOW_FIT = 0.92          # at most this share of the screen's free area (without the menu bar and the Dock)
WINDOW_KEY = "Mojimask main"  # the window's place, remembered by AppKit in the app's defaults


def place_window(window):
    """Where the window opens, before it is first shown (so it doesn't jump): where it was left last time (AppKit's
    frame autosave, which also pulls it back onto a screen that has since gone or shrunk), or else WINDOW_SIZE,
    shrunk to fit the free area of the screen the pointer is on, centred there. Points, not pixels: a Retina
    screen draws the same window with twice the pixels, so resolution plays no part."""
    try:
        import AppKit
    except ImportError:  # pragma: no cover  (not macOS)
        return
    w = window.native
    screens = list(AppKit.NSScreen.screens() or [])
    if not screens:
        return
    free = screens[0].visibleFrame()
    at = AppKit.NSEvent.mouseLocation()
    for s in screens:
        if AppKit.NSPointInRect(at, s.frame()):
            free = s.visibleFrame()
            break
    lo = (min(WINDOW_MIN[0], free.size.width), min(WINDOW_MIN[1], free.size.height))
    w.setMinSize_(lo)  # (a screen smaller than the minimum still gets the whole window)
    if not w.setFrameUsingName_(WINDOW_KEY):
        width = max(lo[0], min(WINDOW_SIZE[0], free.size.width * WINDOW_FIT))
        height = max(lo[1], min(WINDOW_SIZE[1], free.size.height * WINDOW_FIT))
        x = free.origin.x + (free.size.width - width) / 2
        y = free.origin.y + (free.size.height - height) / 2
        w.setFrame_display_(AppKit.NSMakeRect(round(x), round(y), round(width), round(height)), False)
    w.setFrameAutosaveName_(WINDOW_KEY)


def unify_titlebar(window):
    """The page's own bar is the window's title bar (app.html: body.native): the content runs under a transparent
    title bar, and an empty toolbar makes that bar 52 points tall with the traffic lights centred in it, the height
    the page's bar has. Dragging the page's bar moves the window (pywebview's drag region).

    pywebview paints the title bar's container an opaque window colour (for windows with a frame, "so that it does not
    change with the window color"): over the page that is a grey strip hiding its bar. It's made clear again here."""
    try:
        import AppKit
        from PyObjCTools import AppHelper

        def go():
            w = window.native
            w.setStyleMask_(w.styleMask() | AppKit.NSWindowStyleMaskFullSizeContentView)
            w.setTitlebarAppearsTransparent_(True)
            w.setTitleVisibility_(AppKit.NSWindowTitleHidden)
            bar = AppKit.NSToolbar.alloc().initWithIdentifier_("mojimask")
            bar.setShowsBaselineSeparator_(False)
            w.setToolbar_(bar)
            if hasattr(w, "setToolbarStyle_"):
                w.setToolbarStyle_(AppKit.NSWindowToolbarStyleUnified)
            container = w.standardWindowButton_(AppKit.NSWindowCloseButton).superview().superview()  # (pywebview's own way in)
            if container.respondsToSelector_("setBackgroundColor:"):
                container.setBackgroundColor_(None)  # (no colour; clearColor also hides it, but then clicks there stop
                                                     # reaching the page: its bar's buttons and dragging)
        if AppKit.NSThread.isMainThread():
            go()
        else:
            AppHelper.callAfter(go)
    except Exception:  # pragma: no cover  (not macOS)
        pass


def menu_bar(window):
    """The Mac menu bar, the usual one: Mojimask (About, Hide, Quit), File (Open…, Close Window), Edit (Undo, Redo,
    Cut, Copy, Paste, Select All), View (Full Screen), Window (Minimize, Zoom) and Help (the project's page, about.URL, and Licences). Undo, Redo, Open and
    Licences are the page's (app.html window.menuCommand); the rest are the standard AppKit actions. ⌘Z and ⇧⌘Z go
    to the page first and reach these items only when it doesn't take them itself (WKWebView's order)."""
    try:
        import AppKit
        import objc
        from PyObjCTools import AppHelper
    except ImportError:  # pragma: no cover  (not macOS)
        return

    class MenuTarget(AppKit.NSObject):
        @objc.typedSelector(b"v@:@")
        def github_(self, item):
            AppKit.NSWorkspace.sharedWorkspace().openURL_(AppKit.NSURL.URLWithString_(about.URL))

        @objc.typedSelector(b"v@:@")
        def command_(self, item):
            cmd = item.representedObject()  # (evaluate_js waits on the main thread: call it from another)
            threading.Thread(target=window.evaluate_js, args=(f"window.menuCommand && menuCommand({json.dumps(cmd)})",),
                             daemon=True).start()

    target = menu_bar.target = MenuTarget.alloc().init()  # (kept: a menu item holds its target weakly)
    cmd_ = AppKit.NSEventModifierFlagCommand

    def build():
        app, bar = AppKit.NSApp, AppKit.NSMenu.alloc().init()

        def menu(title, items):
            m = AppKit.NSMenu.alloc().initWithTitle_(title)
            for it in items:
                if it is None:
                    m.addItem_(AppKit.NSMenuItem.separatorItem())
                    continue
                title_, action, key, *mods = it
                page = not action.endswith(":")  # a page command, not an action
                mi = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(title_, "command:" if page else action, key)
                if page:
                    mi.setRepresentedObject_(action)
                if page or target.respondsToSelector_(action):  # ours (the rest: AppKit's, along the responder chain)
                    mi.setTarget_(target)
                if mods:
                    mi.setKeyEquivalentModifierMask_(mods[0])
                m.addItem_(mi)
            top = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(title, None, "")
            top.setSubmenu_(m)
            bar.addItem_(top)
            return m

        n = about.NAME  # (brand.js; the first menu's title is the bundle's name in the built app, set by macOS)
        menu(n, [(f"About {n}", "orderFrontStandardAboutPanel:", ""), None,
                 (f"Hide {n}", "hide:", "h"),
                 ("Hide Others", "hideOtherApplications:", "h", cmd_ | AppKit.NSEventModifierFlagOption),
                 ("Show All", "unhideAllApplications:", ""), None,
                 (f"Quit {n}", "terminate:", "q")])
        menu("File", [("Open Video or Photo…", "open", "o"), None, ("Close Window", "performClose:", "w")])
        menu("Edit", [("Undo", "undo", "z"), ("Redo", "redo", "Z"), None,
                      ("Cut", "cut:", "x"), ("Copy", "copy:", "c"), ("Paste", "paste:", "v"),
                      ("Select All", "selectAll:", "a")])
        menu("View", [("Enter Full Screen", "toggleFullScreen:", "f", cmd_ | AppKit.NSEventModifierFlagControl)])
        windows = menu("Window", [("Minimize", "performMiniaturize:", "m"), ("Zoom", "performZoom:", ""), None,
                                  ("Bring All to Front", "arrangeInFront:", "")])
        help_ = menu("Help", [(f"{n} on GitHub", "github:", ""), ("Licences", "licences", "")])
        app.setMainMenu_(bar)
        app.setWindowsMenu_(windows)
        app.setHelpMenu_(help_)

    AppHelper.callAfter(build)


def watch_drops(window):
    """Dropped files: macOS gives pywebview their real path (pywebviewFullPath) when a Python listener is
    attached; hand it to the page, which imports the video where it is (no copy). No usable path (not a video or
    photo, or no path given): onNativeDrop(null), so the page knows now instead of waiting out its timer."""
    from webview.dom import DOMEventHandler

    def on_drop(e):
        files = (e.get("dataTransfer") or {}).get("files") or []
        path = files[0].get("pywebviewFullPath") if files else None
        ok = path and path.lower().endswith(VIDEO_EXT + PHOTO_EXT)
        window.evaluate_js(f"onNativeDrop({json.dumps(path) if ok else 'null'})")

    window.dom.document.events.dragover += DOMEventHandler(lambda e: None, prevent_default=True)
    window.dom.document.events.drop += DOMEventHandler(on_drop, prevent_default=True)


# Everything the bundle must carry. The build runs `<app> --check` on the finished .app, so a module
# PyInstaller missed (one imported by name at runtime) or a page left out of the bundle fails the build,
# not a scan halfway through on someone's Mac. (packaging/build_mac.sh bundles the models CHECK_MODELS names.)
CHECK_MODULES = ("mojimask.about", "app_server", "mojimask.review_server", "mojimask.splash", "mojimask.pipeline", "mojimask.onnx_core", "mojimask.track", "mojimask.formats", "mojimask.video_io", "mojimask.reid",
                 "mojimask.appearance", "mojimask.pose_util", "mojimask.lighting", "mojimask.palettes", "mojimask.portrait", "mojimask.facts", "mojimask.projects", "decisions_cli",
                 "mojimask.coords", "mojimask.head_pose", "mojimask.depth", "mojimask.scan.newalgo", "mojimask.scan.faces", "mojimask.gpu_draw", "AVFoundation", "CoreMedia", "Quartz", "Metal",
                 "AppKit", "WebKit", "webview.platforms.cocoa")
CHECK_FILES = ("web/brand.js", "web/app.html", "web/review.html", "web/decisions.html", "web/people.html", "web/palettes.json", "web/emoji_eyes.json",
               "web/decide_viz.js", "web/pose_ui.js", "web/light_ui.js", "web/emoji_set_ui.js", "web/edits_sync.js", "web/dialog_ui.js", "web/icons.js",
               "web/theme.css",
               "web/strings.js", "web/page_util.js", "web/depth_ui.js", "test_clips/jazz.mp4",
               "web/assets/gallery/mona.jpg", "web/assets/gallery/cavalier.jpg", "web/assets/gallery/pearl.jpg", "web/assets/icon_art.jpg",
               "web/fonts/Nunito.woff2", "web/fonts/Inter.woff2", "web/fonts/Nunito-OFL.txt", "web/fonts/Inter-OFL.txt",
               "LICENSE", "THIRD_PARTY_NOTICES.md", "web/licenses.html", "licenses/Apache-2.0.txt", "licenses/AGPL-3.0.txt",
               "licenses/GPL-3.0.txt", "licenses/Python-PSF-2.0.txt")
CHECK_MODELS = ("yolov5m-face", "retinaface-r50", "yolo11x-pose", "sface-2021dec", "osnet-ain-x1_0-msmt17")
# The Mac app has depth on (partial occlusion, and the editor's Depth view) unless MOJIMASK_DEPTH says otherwise; the build
# bundles this model's file (depth.MODELS[APP_DEPTH]["path"]) and --check looks for it. Apache-2.0 (models/README.md).
APP_DEPTH = "anything-small"


def check():
    """--check: import every engine module, look for every page and model; print a JSON report and exit 1 on
    any gap. Opens no window. review_server is only imported by name: its module body starts a server."""
    import importlib
    import importlib.util
    here = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    os.environ.setdefault("MOJIMASK_PROJECTS", os.path.join(os.environ.get("TMPDIR", "/tmp"), "pa-check-projects"))
    bad = {}
    for m in CHECK_MODULES:
        try:
            if m == "mojimask.review_server":
                if importlib.util.find_spec(m) is None:
                    raise ImportError("not bundled")
            else:
                importlib.import_module(m)
        except Exception as e:  # noqa: BLE001
            bad[m] = f"{type(e).__name__}: {e}"
    missing = [f for f in CHECK_FILES if not os.path.exists(os.path.join(here, f))]
    missing += [f"models/onnx/{m}.onnx" for m in CHECK_MODELS if not os.path.exists(os.path.join(here, "models", "onnx", f"{m}.onnx"))]
    try:
        from mojimask import depth
        dpath = depth.MODELS[APP_DEPTH]["path"]
        missing += [dpath] if not os.path.exists(os.path.join(here, dpath)) else []
    except Exception as e:  # noqa: BLE001
        bad["depth"] = f"{type(e).__name__}: {e}"
    try:
        import onnxruntime as ort
        coreml = "CoreMLExecutionProvider" in ort.get_available_providers()
    except Exception as e:  # noqa: BLE001
        bad["onnxruntime"], coreml = f"{type(e).__name__}: {e}", False
    ok = not bad and not missing and coreml
    print(json.dumps({"ok": ok, "import_errors": bad, "missing_files": missing, "coreml": coreml}, indent=2))
    sys.exit(0 if ok else 1)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--review-server":
        return run_review_server(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "--decisions":  # decisions_cli.py from the bundled app (not advertised)
        sys.argv = ["decisions_cli.py", *sys.argv[2:]]
        import decisions_cli
        return decisions_cli.main()
    if len(sys.argv) > 1 and sys.argv[1] == "--check":
        return check()
    os.environ.setdefault("MOJIMASK_APP", "1")
    os.environ.setdefault("MOJIMASK_DEPTH", APP_DEPTH)  # project servers and scans inherit it; MOJIMASK_DEPTH=off turns depth off
    import webview
    from mojimask import splash
    # only the bar itself drags the window, not the buttons and links in it (app.html marks the parts that do)
    webview.settings["DRAG_REGION_DIRECT_TARGET_ONLY"] = True
    api = Api()
    # The window opens at once on the icon and wordmark (splash.py); the engine is imported and its server started
    # behind it, then the splash fades to black and the app's page takes its place.
    window = api.window = webview.create_window(about.NAME, html=splash.html(), js_api=api,
                                                width=WINDOW_SIZE[0], height=WINDOW_SIZE[1], min_size=WINDOW_MIN,
                                                background_color="#0D0B17", text_select=True)
    # Before the window first appears (on the main thread): the title bar first, since its toolbar changes the
    # window's height, then its size and place.
    window.events.before_show += lambda: (unify_titlebar(window), place_window(window))
    window.events.loaded += lambda: watch_drops(window)
    window.events.shown += force_dark
    window.events.shown += lambda: menu_bar(window)
    up = {}

    def boot():
        try:
            import app_server
            up["app_server"] = app_server
            up["srv"], port = app_server.serve(int(os.environ.get("MOJIMASK_PORT", 0)))  # MOJIMASK_PORT: fixed port for testing
        except Exception:  # noqa: BLE001 -- say so in the window rather than sit on the opening page
            boot_failed(window)
            return
        window.evaluate_js("document.body.classList.add('out')")
        time.sleep(splash.FADE_S)
        window.load_url(f"http://127.0.0.1:{port}/")
        if os.environ.get("MOJIMASK_SELFTEST"):
            selftest(window, api, port)
    try:
        webview.start(boot)
    finally:
        if "app_server" in up:
            up["app_server"].shutdown()
        if "srv" in up:
            up["srv"].shutdown()


def boot_failed(window):
    """The engine didn't start: log why (~/Library/Logs/Mojimask/boot.log) and show it in the window."""
    import html
    import traceback
    trace = traceback.format_exc()
    log = os.path.expanduser("~/Library/Logs/Mojimask/boot.log")
    try:
        os.makedirs(os.path.dirname(log), exist_ok=True)
        with open(log, "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n{trace}\n")
    except OSError:
        log = None
    print(trace, file=sys.stderr, flush=True)
    last = html.escape(trace.strip().splitlines()[-1])
    where = f"The details are in {html.escape(log)}." if log else ""
    window.load_html("<!doctype html><meta charset=utf-8><body style='font:15px -apple-system,sans-serif;padding:48px;"
                     "background:#0D0B17;color:#ececec'><h2 style='font-weight:600'>"
                     f"{html.escape(about.NAME)} couldn't start</h2>"
                     f"<p>{last}</p><p style='color:#9a98a6'>{where}</p>")


# ------------------------------------------------------------------ self-test (MOJIMASK_SELFTEST=1)
class _Panels:
    """Watches for the app-modal Open/Save panel on the main thread and ends it like a person would:
    'cancel', or 'save' (the Save panel returns its folder + the proposed file name). Records what it saw."""

    def __init__(self):
        import AppKit
        import Foundation
        from PyObjCTools import AppHelper
        self.AppKit, self.action, self.seen, self.done = AppKit, None, [], threading.Event()
        owner = self

        class Tick(Foundation.NSObject):
            def fire_(self, timer):
                owner._tick()
        self._tick_obj = Tick.alloc().init()

        def install():
            t = Foundation.NSTimer.timerWithTimeInterval_target_selector_userInfo_repeats_(0.25, self._tick_obj, "fire:", None, True)
            Foundation.NSRunLoop.currentRunLoop().addTimer_forMode_(t, AppKit.NSModalPanelRunLoopMode)
        AppHelper.callAfter(install)

    def expect(self, action, directory=None):
        self.action, self.directory, self.wait_ticks = action, directory, 4
        self.done.clear()

    def _tick(self):
        w = self.AppKit.NSApp.modalWindow()
        if w is None or self.action is None:
            return
        self.wait_ticks -= 1
        if self.wait_ticks > 0:
            return  # let the panel finish appearing, as a person would
        info = {"panel": type(w).__name__.replace("NSKVONotifying_", ""), "directory": w.directoryURL().path() if w.directoryURL() else None}
        if "Save" in info["panel"]:
            info["name"] = w.nameFieldStringValue()
            if self.directory:
                w.setDirectoryURL_(__import__("Foundation").NSURL.fileURLWithPath_(self.directory))
        self.seen.append({**info, "action": self.action})
        code = self.AppKit.NSModalResponseOK if self.action == "save" else self.AppKit.NSModalResponseCancel
        self.action = None
        self.AppKit.NSApp.stopModalWithCode_(code)
        self.done.set()


def selftest(window, api, port):
    import urllib.request
    report = {"steps": [], "ok": False}
    video = os.environ["MOJIMASK_SELFTEST_VIDEO"]
    drop_video = os.environ.get("MOJIMASK_SELFTEST_DROP", video)
    out_dir = os.environ["MOJIMASK_SELFTEST_OUT"]
    base = f"http://127.0.0.1:{port}"

    def get(path):
        with urllib.request.urlopen(base + path, timeout=60) as r:
            return json.loads(r.read())

    def js(code):
        return window.evaluate_js(code)

    def step(name, **kw):
        report["steps"].append({"step": name, "t": round(time.time() - t0, 1), **kw})
        print("SELFTEST", name, kw, flush=True)

    def until(cond, timeout, every=1.0):
        end = time.time() + timeout
        while time.time() < end:
            v = cond()
            if v:
                return v
            time.sleep(every)
        raise TimeoutError

    t0 = time.time()
    try:
        panels = _Panels()
        until(lambda: js("typeof NATIVE !== 'undefined' && NATIVE !== null"), 30, 0.5)
        step("window loaded; native bridge ready")

        # 1. Choose video... opens the real Open panel (here: cancelled -> nothing imported)
        before = len(get("/api/projects"))
        panels.expect("cancel")
        js('document.getElementById("pick").click()')
        panels.done.wait(30)
        time.sleep(1.5)
        step("Open panel shown from Choose video, cancelled", panel=panels.seen[-1], projects_unchanged=len(get("/api/projects")) == before)

        # 2. Choose video... again; the pick a person makes in that panel arrives through the same bridge call
        api.test_pick = video
        js('document.getElementById("pick").click()')
        pid = until(lambda: next((p["id"] for p in get("/api/projects") if p.get("video") == video), None), 60)
        step("video chosen -> project created (referenced, not copied)", project=pid)
        # after the scan: the result screen covers the faces, plays the clip and offers Export or Refine
        until(lambda: js('CUR === "result" && !document.getElementById("r-export").disabled'), 1200, 2)
        step("scan finished; covered result shown", style=js('styleChoice'),
             refine=js('document.getElementById("r-refine").textContent'))
        js('document.getElementById("r-refine").click()')  # Refine -> the open questions
        until(lambda: js('CUR') == "decisions", 60, 1)
        step("Refine opened", screen=js('CUR'))
        js('go("edit")')
        until(lambda: js('CUR') == "edit", 120, 1)
        until(lambda: js('(() => { try { const d = document.getElementById("editor").contentDocument; return !!(d && d.querySelector("#timeline")); } catch (e) { return false; } })()'), 60, 1)
        step("full editor on screen")

        # 3. Export: vertical, whole frame; Export... opens the real Save panel; Save
        js('go("export")')
        until(lambda: js('FORMATS !== null'), 30, 0.5)
        js('document.querySelector(\'#formats .style[data-k="vertical"]\').click(); document.getElementById("fit-fit").click();'
           'document.getElementById("tstart").value = "1"; document.getElementById("tend").value = "4"; drawFormats();')
        panels.expect("save")  # Save as proposed: the original's folder + "<name>_mojimask_9x16.mp4"
        js('document.getElementById("export").click()')
        panels.done.wait(60)
        step("Save panel shown from Export, Save pressed", panel=panels.seen[-1])
        status = until(lambda: (lambda s: s if ("Saved" in s or "failed" in s) else None)(js('document.getElementById("exportstatus").textContent')), 600, 1)
        saved = get(f"/p/{pid}/api/render").get("path") or ""
        step("export finished", status=status, file=saved, exists=os.path.exists(saved), bytes=os.path.getsize(saved) if os.path.exists(saved) else 0,
             where_save_panel_pointed=panels.seen[-1].get("directory"))
        if os.path.exists(saved) and os.path.dirname(saved) != os.path.abspath(out_dir):
            os.replace(saved, os.path.join(out_dir, os.path.basename(saved)))  # keep the test's file out of the user's folders
            saved = os.path.join(out_dir, os.path.basename(saved))

        # 4. Drop a video on the start screen. macOS's drag session can't be synthesised without
        # Accessibility permission, so this enters where pywebview's native drop handler records the
        # dropped file's path, then delivers the DOM drop the window would.
        from webview.dom import _dnd_state
        js('show("start"); loadRecent();')
        time.sleep(1)
        _dnd_state["paths"].append((os.path.basename(drop_video), drop_video))
        js(f'(() => {{ const dt = new DataTransfer(); dt.items.add(new File([new Uint8Array(8)], {json.dumps(os.path.basename(drop_video))}, {{type: "video/mp4"}}));'
           f' document.getElementById("drop").dispatchEvent(new DragEvent("drop", {{dataTransfer: dt, bubbles: true, cancelable: true}})); }})()')
        p2 = until(lambda: next((p for p in get("/api/projects") if p.get("video") == drop_video and p["id"] != pid), None), 60)
        step("dropped video -> project created", project=p2["id"], copied=p2["copied"], video=p2["video"])
        report["ok"] = bool(os.path.exists(saved) and not p2["copied"] and "Saved" in status)
    except Exception as e:  # noqa: BLE001 -- report whatever went wrong
        import traceback
        report["error"] = f"{type(e).__name__}: {e}"
        report["trace"] = traceback.format_exc()
    finally:
        json.dump(report, open(os.environ.get("MOJIMASK_SELFTEST_REPORT", "selftest_report.json"), "w"), indent=2)
        time.sleep(1)
        window.destroy()


if __name__ == "__main__":
    main()
