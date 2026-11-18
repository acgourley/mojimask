// The Qt host: start-up and flags, the QML engine, loading the Rust store (libmm_app), live QML reload, devtools.
//
//   Mojimask [--dev] [--video CLIP] [--data DUMP] [--restore] [--lib libmm_app.dylib] [--root REPO]
//
//   --dev      QML from native/app/qml on disk, reloaded when a file changes (store state survives); the developer
//              hooks on 127.0.0.1:9200-9299 (out/native/devtools.json)
//   --restore  reopen what the store saved in out/native/state.json (mmctl restart passes it)
//   --lib      the store dylib to load (default native/target/debug/libmm_app.dylib)
//   --projects the projects folder (default out/native/projects; the Mac app's own in a bundle)
//   --no-sidecar  don't start an engine (the Editor alone, on a PoC dump)
//   --check [DIR]  the self-test: the native engine scans test_clips/jazz.mp4 into DIR (default: a temporary folder,
//              removed), the store exports it with its covers; prints one JSON line ending "ok": true (exit 0) or false
//
// Where files are (mm_core::paths): the root (web/, models/, test_clips/: read-only; the bundle's Contents/Resources)
// and the data folder (state.json, projects: out/native/ in a checkout, ~/Library/Application Support/Mojimask bundled).
// Inside a bundle, the environment the bundled libraries need is set here before anything loads (each variable only
// when not already set): GStreamer's plugin path, scanner and registry, GIO's modules, MM_ROOT, MOJIMASK_MODELS.
#include "devtools.h"
#include "mmapi.h"
#include "store.h"
#include "videoitem.h"
#include <QDateTime>
#include <QDir>
#include <QDirIterator>
#include <QFileInfo>
#include <QFileSystemWatcher>
#include <QGuiApplication>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QProcess>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include <QQuickStyle>
#include <QQuickWindow>
#include <QTimer>
#include <cstdio>
#include <cstdlib>
#include <climits>
#include <mach-o/dyld.h>
#include <sys/stat.h>
#include <unistd.h>

void macStyleWindow(QWindow *w);

static void messageHandler(QtMsgType type, const QMessageLogContext &ctx, const QString &msg) {
    int level = type == QtDebugMsg ? 0 : type == QtInfoMsg ? 1 : type == QtWarningMsg ? 2 : 3;
    QString src = ctx.category && QString(ctx.category) != "default" ? QString(ctx.category) : QStringLiteral("qt");
    Devtools::appendLog(level, src, msg);
    fprintf(stderr, "[%s] %s\n", qPrintable(src), qPrintable(msg));
}

static QString findRoot(const QString &given) {
    if (!given.isEmpty()) return QDir(given).absolutePath();
    if (qEnvironmentVariableIsSet("MM_ROOT")) return qEnvironmentVariable("MM_ROOT");
    for (QString start : {QDir::currentPath(), QCoreApplication::applicationDirPath()}) {
        QDir d(start);
        for (int i = 0; i < 12; ++i) {
            if (d.exists("native/ARCHITECTURE.md")) return d.absolutePath();
            if (!d.cdUp()) break;
        }
    }
    return QDir::currentPath();
}

/// Inside Mojimask.app (Contents/Resources/web next to MacOS/): the bundled libraries' environment (launcher.c's).
static void bundleEnv() {
    char exe[PATH_MAX], real[PATH_MAX];
    uint32_t n = sizeof exe;
    if (_NSGetExecutablePath(exe, &n) != 0 || !realpath(exe, real)) return;
    const QString macos = QFileInfo(QString::fromUtf8(real)).absolutePath();
    const QString contents = QFileInfo(macos).absolutePath();
    if (!QFileInfo::exists(contents + "/Resources/web")) return;
    auto setDefault = [](const char *k, const QString &v) {
        if (!qEnvironmentVariableIsSet(k)) qputenv(k, v.toUtf8());
    };
    auto both = [&](const QString &k, const QString &v) {
        setDefault(qPrintable(k + "_1_0"), v);
        setDefault(qPrintable(k), v);
    };
    both("GST_PLUGIN_SYSTEM_PATH", contents + "/PlugIns/gstreamer");
    both("GST_PLUGIN_PATH", "");
    both("GST_PLUGIN_SCANNER", macos + "/gst-plugin-scanner");
    setDefault("GIO_MODULE_DIR", contents + "/Frameworks/gio-modules");
    const QString caches = QDir::homePath() + "/Library/Caches/org.mojimask.app";
    QDir().mkpath(caches);
    both("GST_REGISTRY", caches + "/gst-registry.bin");
    setDefault("MM_ROOT", contents + "/Resources");
    setDefault("MOJIMASK_MODELS", contents + "/Resources/models");
}

/// The data folder (mm_core::paths::data_dir): out/native in a checkout, Application Support bundled.
static QString dataDir(const QString &root) {
    if (QFileInfo::exists(root + "/native")) return root + "/out/native";
    return QDir::homePath() + "/Library/Application Support/Mojimask";
}

static double epochMs() { return double(QDateTime::currentMSecsSinceEpoch()); }

int main(int argc, char **argv) {
    bundleEnv();
    qputenv("QSG_RHI_BACKEND", "metal");
    QQuickWindow::setGraphicsApi(QSGRendererInterface::Metal);
    QQuickStyle::setStyle("Basic"); // our own controls on Theme.qml's tokens, not the macOS style
    QGuiApplication app(argc, argv);
    app.setApplicationName("Mojimask");
    app.setOrganizationDomain("mojimask.org");
    qInstallMessageHandler(messageHandler);

    QStringList a = app.arguments();
    auto flag = [&](const QString &k) { return a.contains(k); };
    auto opt = [&](const QString &k) {
        int i = a.indexOf(k);
        return i >= 0 && i + 1 < a.size() ? a[i + 1] : QString();
    };
    const bool dev = flag("--dev") || qEnvironmentVariable("MOJIMASK_DEV") == "1";
    const QString root = findRoot(opt("--root"));
    QString lib = opt("--lib");
    if (lib.isEmpty()) lib = root + "/native/target/debug/libmm_app.dylib";
    QString err = mmApi().load(lib);
    if (!err.isEmpty()) {
        fprintf(stderr, "%s\n", qPrintable(err));
        return 2;
    }
    if (flag("--check")) {
        // the self-test, headless: the native engine and the store's export, one JSON line
        int i = a.indexOf("--check");
        const QString dir = i + 1 < a.size() && !a[i + 1].startsWith("--") ? a[i + 1] : QString();
        Store store(QJsonObject{{"root", root}, {"state_file", ""}, {"restore", false}, {"dev", false}, {"sidecar", false}});
        QJsonObject r = store.callJson("check", QJsonObject{{"dir", dir}});
        QJsonObject rep = r.value("result").toObject();
        if (rep.isEmpty()) rep = QJsonObject{{"error", r.value("error").toString()}, {"ok", false}};
        // "ok" last
        QJsonValue ok = rep.take("ok");
        QByteArray line = QJsonDocument(rep).toJson(QJsonDocument::Compact);
        line.chop(1);
        line += QByteArray(rep.isEmpty() ? "" : ", ") + "\"ok\": " + (ok.toBool() ? "true" : "false") + "}";
        printf("%s\n", line.constData());
        fflush(stdout);
        _exit(ok.toBool() ? 0 : 1);
    }
    const QString data = dataDir(root);
    QDir().mkpath(data);
    const QString stateFile = data + "/state.json";
    const bool restore = flag("--restore") && QFileInfo::exists(stateFile);
    Store store(QJsonObject{{"root", root}, {"state_file", stateFile}, {"restore", restore}, {"dev", dev},
                             {"sidecar", !flag("--no-sidecar")}, {"projects", opt("--projects")}});
    Store::log(1, "host", QStringLiteral("store from %1, root %2%3").arg(mmApi().path, root, restore ? ", restored" : ""));
    if (!restore && !opt("--video").isEmpty()) {
        QJsonObject r = store.callJson("open", QJsonObject{{"video", opt("--video")}, {"data", opt("--data")}});
        if (!r.value("ok").toBool()) Store::log(3, "host", r.value("error").toString());
    }

    qmlRegisterType<VideoItem>("Mojimask.Native", 1, 0, "VideoItem");
    QQmlApplicationEngine engine;
    engine.rootContext()->setContextProperty("store", &store);
    engine.rootContext()->setContextProperty("devMode", dev);
    engine.rootContext()->setContextProperty("qmlReloaded", false);   // true after a live reload (Splash.qml skips)
    // the app's fonts and pictures (mm_core::paths::web): native/web in a checkout, Resources/web in the bundle
    const QString web = QDir(root + "/native/web").exists() ? root + "/native/web" : root + "/web";
    engine.rootContext()->setContextProperty("fontsDir", QUrl::fromLocalFile(web + "/fonts/"));
    engine.rootContext()->setContextProperty("webDir", QUrl::fromLocalFile(web + "/"));
    // the Projects page's "Licences": the notices (the bundle's are made from the build, mm-pack notices; the licence texts
    // are in licenses/ beside them)
    engine.rootContext()->setContextProperty("noticesUrl", QUrl::fromLocalFile(root + "/THIRD_PARTY_NOTICES.md"));
    const QString qmlDir = dev ? root + "/native/app/qml" : QString();
    const QUrl mainUrl = dev ? QUrl::fromLocalFile(qmlDir + "/Main.qml") : QUrl("qrc:/qml/Main.qml");

    Devtools *tools = new Devtools(&engine, &store, root, &app);
    // every new window (start, reload): the Mac title bar, frame timing, and drop our GPU objects with its device
    QObject::connect(&engine, &QQmlApplicationEngine::objectCreated, &app, [&](QObject *o, const QUrl &) {
        auto *w = qobject_cast<QQuickWindow *>(o);
        if (!w) return;
        tools->attachWindow(w);
        QObject::connect(w, &QQuickWindow::sceneGraphInvalidated, w, [&store] {
            if (store.app()) mmApi().release_gpu(store.app());
        }, Qt::DirectConnection);
        QTimer::singleShot(0, w, [w] { macStyleWindow(w); });
    });
    engine.load(mainUrl);
    if (engine.rootObjects().isEmpty()) return 1;

    // reload QML: drop the window, load Main.qml again; the store (and so the open video, frame, selection) stays
    auto reload = [&](double savedMs) -> bool {
        const double noticed = epochMs();
        QRect geo;
        if (QQuickWindow *w = tools->window()) geo = w->geometry();
        for (QObject *o : engine.rootObjects()) delete o;
        engine.clearComponentCache();
        engine.rootContext()->setContextProperty("qmlReloaded", true);
        engine.load(mainUrl);
        QQuickWindow *w = tools->window();
        double loaded = epochMs();
        if (!w) {
            Store::log(3, "host", "QML reload failed");
            return false;
        }
        if (geo.isValid()) w->setGeometry(geo);
        auto c = std::make_shared<QMetaObject::Connection>();
        *c = QObject::connect(w, &QQuickWindow::frameSwapped, &app, [=] {
            QObject::disconnect(*c);
            double frame = epochMs();
            if (savedMs > 0) {
                tools->noteReload(frame - savedMs, loaded - savedMs);
                Store::log(1, "host", QStringLiteral("QML reload: saved -> noticed %1 ms, -> loaded %2 ms, -> first frame %3 ms")
                                          .arg(noticed - savedMs, 0, 'f', 0).arg(loaded - savedMs, 0, 'f', 0).arg(frame - savedMs, 0, 'f', 0));
            } else {
                Store::log(1, "host", "QML reloaded");
            }
        }, Qt::QueuedConnection);
        return true;
    };
    tools->reload = [&] { return reload(0); };
    tools->restart = [&] {
        QStringList args = QCoreApplication::arguments().mid(1);
        if (!args.contains("--restore")) args << "--restore";
        store.callJson("save_state", QJsonValue());
        Store::log(1, "host", "restarting: " + args.join(' '));
        QProcess::startDetached(QCoreApplication::applicationFilePath(), args, QDir::currentPath());
        qApp->quit();
    };

    if (dev) {
        // live reload: watch every QML file and folder (editors often replace a file rather than write it)
        auto *watch = new QFileSystemWatcher(&app);
        auto rewatch = [watch, qmlDir] {
            QStringList paths{qmlDir};
            QDirIterator it(qmlDir, QDir::Dirs | QDir::Files | QDir::NoDotAndDotDot, QDirIterator::Subdirectories);
            while (it.hasNext()) {
                QString p = it.next();
                if (QFileInfo(p).isDir() || p.endsWith(".qml") || p.endsWith("qmldir")) paths << p;
            }
            // re-add every file: an editor that saves by replacing the file leaves the watch on the old one
            if (!watch->files().isEmpty()) watch->removePaths(watch->files());
            if (!watch->directories().isEmpty()) watch->removePaths(watch->directories());
            watch->addPaths(paths);
        };
        rewatch();
        auto pending = std::make_shared<bool>(false);
        auto onChange = [&, rewatch, pending, qmlDir](const QString &) {
            if (*pending) return;
            *pending = true;
            // a few ms to let an editor finish writing (and coalesce the file + folder notes)
            QTimer::singleShot(5, &app, [&, rewatch, pending, qmlDir] {
                *pending = false;
                rewatch();
                double newest = 0;
                QDirIterator it(qmlDir, {"*.qml", "qmldir"}, QDir::Files, QDirIterator::Subdirectories);
                while (it.hasNext()) newest = qMax(newest, double(QFileInfo(it.next()).lastModified().toMSecsSinceEpoch()));
                reload(newest);
            });
        };
        QObject::connect(watch, &QFileSystemWatcher::fileChanged, &app, onChange);
        QObject::connect(watch, &QFileSystemWatcher::directoryChanged, &app, onChange);
        tools->start(9200, 9299);
    }
    int rc = app.exec();
    if (dev) { // remove devtools.json if it's still ours (after a restart it's the new process's)
        QFile f(root + "/out/native/devtools.json");
        if (f.open(QIODevice::ReadOnly) && QJsonDocument::fromJson(f.readAll()).object().value("pid").toInteger() == app.applicationPid()) {
            f.close();
            f.remove();
        }
    }
    return rc;
}
