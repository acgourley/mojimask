#include "mmapi.h"
#include <dlfcn.h>

MmApi &mmApi() {
    static MmApi a{};
    return a;
}

#ifdef MM_STATIC
QString MmApi::load(const QString &) {
    abi = mm_app_abi;
    create = mm_app_new;
    destroy = mm_app_free;
    state = mm_app_state;
    call = mm_app_call;
    string_free = mm_app_string_free;
    render = mm_app_render;
    release_gpu = mm_app_release_gpu;
    path = QStringLiteral("(static)");
    return {};
}
#else
QString MmApi::load(const QString &lib) {
    void *h = dlopen(lib.toUtf8().constData(), RTLD_NOW | RTLD_LOCAL);
    if (!h) return QStringLiteral("can't load %1: %2").arg(lib, QString::fromUtf8(dlerror()));
    auto sym = [&](const char *n) { return dlsym(h, n); };
    abi = reinterpret_cast<decltype(abi)>(sym("mm_app_abi"));
    create = reinterpret_cast<decltype(create)>(sym("mm_app_new"));
    destroy = reinterpret_cast<decltype(destroy)>(sym("mm_app_free"));
    state = reinterpret_cast<decltype(state)>(sym("mm_app_state"));
    call = reinterpret_cast<decltype(call)>(sym("mm_app_call"));
    string_free = reinterpret_cast<decltype(string_free)>(sym("mm_app_string_free"));
    render = reinterpret_cast<decltype(render)>(sym("mm_app_render"));
    release_gpu = reinterpret_cast<decltype(release_gpu)>(sym("mm_app_release_gpu"));
    if (!abi || !create || !destroy || !state || !call || !string_free || !render || !release_gpu)
        return QStringLiteral("%1 lacks a mm_app_* symbol").arg(lib);
    if (abi() != MM_APP_ABI) return QStringLiteral("%1 has ABI %2, the host wants %3").arg(lib).arg(abi()).arg(MM_APP_ABI);
    path = lib;
    return {};
}
#endif
