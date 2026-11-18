// The Rust store's C ABI as a table of function pointers: filled by dlopen()ing libmm_app.dylib (dev: a Rust
// rebuild needs no C++ relink) or, with MM_STATIC, from the statically linked library.
#pragma once
#include "mm_app.h"
#include <QString>

struct MmApi {
    uint32_t (*abi)(void);
    MmApp *(*create)(const char *, MmNotifyFn, MmLogFn, void *);
    void (*destroy)(MmApp *);
    char *(*state)(MmApp *, const char *);
    char *(*call)(MmApp *, const char *, const char *);
    void (*string_free)(char *);
    int32_t (*render)(MmApp *, void *, void *, void *, MmRenderInfo *, char *, int32_t);
    void (*release_gpu)(MmApp *);
    QString path; // where it was loaded from

    // Load the API (lib: the dylib's path in dev; ignored with MM_STATIC). Returns an error message or "".
    QString load(const QString &lib);
};

MmApi &mmApi();
