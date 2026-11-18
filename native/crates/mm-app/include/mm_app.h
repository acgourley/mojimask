/* mm_app.h: the C ABI between the Qt host (native/app) and the Rust store (native/crates/mm-app).
 *
 * The ABI is deliberately small and stable: adding a feature never touches it. New commands and state fields
 * travel as JSON through mm_app_call / mm_app_state. Keep in step with crates/mm-app/src/ffi.rs.
 *
 * Threads: every function may be called from any thread, except mm_app_render (the host's render thread).
 * The callbacks are called from any thread (worker threads included): hop to your own thread before touching UI.
 * Strings returned by mm_app_state / mm_app_call are owned by the caller: free them with mm_app_string_free.
 */
#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

#define MM_APP_ABI 1

typedef struct MmApp MmApp;

/* `changes` is a JSON array of the state paths that changed ("editor.t", "export"), and "@render" when the
 * video item has a new picture to draw (call update() on it). */
typedef void (*MmNotifyFn)(void *ctx, const char *changes);
/* level: 0 debug, 1 info, 2 warning, 3 error */
typedef void (*MmLogFn)(void *ctx, int32_t level, const char *message);

typedef struct {
    int32_t shown_t;     /* frame now in the texture, -1 none */
    int32_t new_frame;   /* a newly decoded frame was drawn by this call */
    int32_t composited;  /* the texture was (re)drawn by this call */
    double composite_ms; /* encode + submit, CPU side */
    double decode_ms;    /* for a new frame: its decode time */
    double latency_ms;   /* for a new frame: request -> composited */
} MmRenderInfo;

uint32_t mm_app_abi(void);

/* config: {"root": repo root, "state_file": path, "restore": bool, "dev": bool} */
MmApp *mm_app_new(const char *config_json, MmNotifyFn notify, MmLogFn log, void *ctx);
void mm_app_free(MmApp *app);

/* The state tree, or the subtree at a dotted path ("editor.video.width"), as JSON ("null" if absent). */
char *mm_app_state(MmApp *app, const char *path);
/* Run a command: returns {"ok": true, "result": ...} or {"ok": false, "error": "..."}. */
char *mm_app_call(MmApp *app, const char *cmd, const char *args_json);
void mm_app_string_free(char *s);

/* Render thread. Draw the current frame into the host's texture `texture` (Metal: id<MTLTexture>, RGBA8, sized as
 * state editor.video.width x height) on the host's device and queue (Metal: id<MTLDevice>, id<MTLCommandQueue>).
 * Returns 0, or -1 with a message in err. */
int32_t mm_app_render(MmApp *app, void *device, void *queue, void *texture, MmRenderInfo *out, char *err, int32_t errlen);
/* Render thread: drop GPU objects made on the host's device (before the device goes away). */
void mm_app_release_gpu(MmApp *app);

#ifdef __cplusplus
}
#endif
