"""What the two local servers (app_server.py, review_server.py) let in.

They listen on 127.0.0.1 only, but a web page open in the user's browser can still reach 127.0.0.1: by posting to it
(a "simple" cross-site POST needs no preflight) or by DNS rebinding (a site whose name later points at 127.0.0.1, so the
browser treats our server as that site and lets the page read its answers: people's frames). Two checks stop both:

  Host    must name this server, as 127.0.0.1:<port> or localhost:<port> (a rebound page sends its own name)
  Origin  when the browser sends one (every cross-site POST, any fetch from another page), it must be a local origin;
          scripts with no Origin (decisions_cli.py, the app server's proxy) are let through

and the static fallback serves only the files the pages load, never the rest of the folder (projects, .py, .git).
"""
import re

LOCAL_HOSTS = ("127.0.0.1", "localhost")
STATIC = re.compile(r"/[A-Za-z0-9_-]+\.(html|js|css|json)|/fonts/[A-Za-z0-9_-]+\.woff2|/assets/gallery/[a-z_]+\.jpg"
                    r"|/THIRD_PARTY_NOTICES\.md|/LICENSE")  # licenses.html shows these two


def _host_port(value):
    m = re.fullmatch(r"(?:[a-z]+://)?\[?([A-Za-z0-9.:-]+?)\]?(?::(\d+))?/?", value or "")
    return (m.group(1).lower(), int(m.group(2)) if m.group(2) else None) if m else (None, None)


def refusal(handler):
    """None when the request may go on, else why not (for a 403)."""
    port = handler.server.server_address[1]
    host, hport = _host_port(handler.headers.get("Host"))
    if host not in LOCAL_HOSTS or hport != port:
        return "wrong host"
    origin = handler.headers.get("Origin")
    if origin is not None and origin != "null" and _host_port(origin)[0] not in LOCAL_HOSTS:
        return "cross-site request"
    return None


def static_ok(path):
    """Whether a static file at this URL path (no query) may be served."""
    return ".." not in path and STATIC.fullmatch(path) is not None



def own_code(path):
    """The app's own pages, scripts, styles and data files: never cached, so a new version of the app (or a restarted
    server) can't run with an old copy of one of them beside new ones."""
    p = path.split("?")[0]
    return p == "/" or p.endswith((".html", ".js", ".css", ".json"))

class GuardMixin:
    """Put before SimpleHTTPRequestHandler: refuses requests that fail refusal() before any do_* method runs."""

    def parse_request(self):
        if not super().parse_request():
            return False
        why = refusal(self)
        if why:
            self.send_error(403, why)
            return False
        return True
