// The bundle's CFBundleExecutable (Contents/MacOS/Mojimask): sets up the environment the bundled libraries need,
// then execs the real program in the same process (no child, so the Dock, the menu bar and signals see one app).
//
//   Mojimask [args]          -> Contents/MacOS/mojimask-app [args]   (the Qt host)
//   Mojimask --check [DIR]   -> /bin/sh Contents/Resources/check.sh [DIR]   (scan + export jazz, JSON report)
//   Mojimask --cli ARGS      -> Contents/MacOS/mm-cli ARGS            (the store and engine without a window)
//
// Environment (each one only when not already set, so a developer can override it):
// - GStreamer: GST_PLUGIN_SYSTEM_PATH(_1_0) = Contents/PlugIns/gstreamer (only the bundled plugins, never
//   Homebrew's compiled-in /opt/homebrew/lib/gstreamer-1.0), GST_PLUGIN_PATH(_1_0) = "" , GST_PLUGIN_SCANNER(_1_0) =
//   Contents/MacOS/gst-plugin-scanner, GST_REGISTRY(_1_0) = ~/Library/Caches/org.mojimask.app/gst-registry.bin
//   (a cache, rebuilt when a plugin file changes), GIO_MODULE_DIR = Contents/Frameworks/gio-modules (none: glib must not load Homebrew's).
// - The app: MM_ROOT = Contents/Resources (web/, models/, test_clips/ as in a checkout), MOJIMASK_MODELS =
//   Contents/Resources/models.
//
// main.cpp could do the same at the top of main() (make_app.sh's header); this keeps app/ untouched.
#include <errno.h>
#include <libgen.h>
#include <limits.h>
#include <mach-o/dyld.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static void set_default(const char *k, const char *v) { setenv(k, v, 0); }

static void both(const char *k, const char *v) {
    char k1[128];
    snprintf(k1, sizeof k1, "%s_1_0", k);
    set_default(k1, v);
    set_default(k, v);
}

int main(int argc, char **argv) {
    char exe[PATH_MAX], real[PATH_MAX];
    uint32_t n = sizeof exe;
    if (_NSGetExecutablePath(exe, &n) != 0 || !realpath(exe, real)) {
        fprintf(stderr, "Mojimask: can't find myself\n");
        return 2;
    }
    char macos[PATH_MAX], contents[PATH_MAX], p[PATH_MAX];
    strlcpy(macos, dirname(real), sizeof macos);
    strlcpy(p, macos, sizeof p);
    strlcpy(contents, dirname(p), sizeof contents);

#define PATHF(...) (snprintf(p, sizeof p, __VA_ARGS__), p)
    both("GST_PLUGIN_SYSTEM_PATH", PATHF("%s/PlugIns/gstreamer", contents));
    both("GST_PLUGIN_PATH", "");
    both("GST_PLUGIN_SCANNER", PATHF("%s/gst-plugin-scanner", macos));
    set_default("GIO_MODULE_DIR", PATHF("%s/Frameworks/gio-modules", contents));
    const char *home = getenv("HOME");
    if (home && !getenv("GST_REGISTRY")) {
        char dir[PATH_MAX];
        snprintf(dir, sizeof dir, "%s/Library/Caches/org.mojimask.app", home);
        mkdir(dir, 0755);
        both("GST_REGISTRY", PATHF("%s/gst-registry.bin", dir));
    }
    set_default("MM_ROOT", PATHF("%s/Resources", contents));
    set_default("MOJIMASK_MODELS", PATHF("%s/Resources/models", contents));

    if (argc > 1 && strcmp(argv[1], "--check") == 0) {
        char script[PATH_MAX];
        snprintf(script, sizeof script, "%s/Resources/check.sh", contents);
        char *args[] = {"/bin/sh", script, argc > 2 ? argv[2] : NULL, NULL};
        execv("/bin/sh", args);
    } else if (argc > 1 && strcmp(argv[1], "--cli") == 0) {
        char cli[PATH_MAX];
        snprintf(cli, sizeof cli, "%s/mm-cli", macos);
        argv[1] = cli;
        execv(cli, argv + 1);
    } else {
        char app[PATH_MAX];
        snprintf(app, sizeof app, "%s/mojimask-app", macos);
        argv[0] = app;
        execv(app, argv);
    }
    fprintf(stderr, "Mojimask: exec failed: %s\n", strerror(errno));
    return 2;
}
