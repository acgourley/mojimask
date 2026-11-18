"""The native app's Python sidecar: app_server.py (unchanged) on a port the app picks, its project servers on ports
from the same range, and everything stopped when the app goes away.

    python sidecar.py ROOT PORT LO HI [--dev]      (MOJIMASK_PROJECTS set by the app)

The app holds this process's stdin: when the app quits or dies the pipe closes, and this stops app_server's project
servers (app_server.shutdown) before exiting. SIGTERM does the same.
"""
import os
import signal
import socket
import sys
import threading
import time

root, port, lo, hi = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
dev = "--dev" in sys.argv
os.chdir(root)
sys.path.insert(0, root)
if dev:
    os.environ["MOJIMASK_DEV"] = "1"  # project servers (child processes) inherit it
import app_server as A  # noqa: E402

_taken = set()
_lock = threading.Lock()


def free_port():
    """A free port in [lo, hi) other than the app server's own (the native app's sidecar range, 9300-9399: native/README.md)."""
    with _lock:
        for p in range(lo, hi):
            if p == port or p in _taken:
                continue
            s = socket.socket()
            try:
                s.bind(("127.0.0.1", p))
            except OSError:
                continue
            finally:
                s.close()
            _taken.add(p)
            return p
    raise RuntimeError(f"no free port in {lo}-{hi}")


A.free_port = free_port
# the project servers: review_server.py plus the native app's draw records (native/tools/sidecar/project_server.py)
_HELPER = os.path.join(root, "native", "tools", "sidecar", "project_server.py")
if os.path.exists(_HELPER):
    A.review_command = lambda args: [sys.executable, _HELPER, *args]
A.DEV = dev
srv, got = A.serve(port)
print(f"sidecar: app_server on {got} (projects: {A.PROJ_ROOT})", flush=True)

_done = threading.Event()


def stop(*_):
    if _done.is_set():
        return
    _done.set()
    try:
        A.shutdown()
    finally:
        os._exit(0)


signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)


def watch_parent():
    try:
        while sys.stdin.read(1 << 16):
            pass
    except OSError:
        pass
    stop()


threading.Thread(target=watch_parent, daemon=True).start()
while True:
    time.sleep(3600)
