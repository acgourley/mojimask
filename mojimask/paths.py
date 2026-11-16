"""Where things are: the checkout's root (the bundle's, in the built app, which keeps the same layout) and the pages."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB = os.path.join(ROOT, "web")  # pages, scripts, styles, fonts and pictures, served as the app's static files
TRUTH = os.path.join(ROOT, "tools", "truth")  # censuses kept in git: public-domain clips only (jazz)


def truth(name, root=ROOT):
    """A clip's census: tools/truth/NAME.json when it's kept in git, else out/truth/NAME.json under `root` (a census
    of real footage, git-ignored)."""
    kept = os.path.join(TRUTH, f"{name}.json")
    return kept if os.path.exists(kept) else os.path.join(root, "out", "truth", f"{name}.json")
