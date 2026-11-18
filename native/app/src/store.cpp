#include "store.h"
#include <atomic>
#include "devtools.h"
#include <QCoreApplication>
#include <QJsonArray>
#include <QJsonDocument>
#include <QQmlEngine>
#include <QTimer>

static Store *g_store = nullptr;
// a command or worker changed the state since the last re-read (read-only commands leave it alone)
static std::atomic<bool> g_dirty{true};

static void notifyCb(void *, const char *changes) {
    // any thread: hop to the GUI thread
    QStringList paths;
    for (const auto &v : QJsonDocument::fromJson(changes).array()) paths << v.toString();
    for (const auto &p : paths)
        if (p != QStringLiteral("@render")) g_dirty = true;
    QMetaObject::invokeMethod(qApp, [paths] {
        if (g_store) g_store->onNotify(paths);
    }, Qt::QueuedConnection);
}

static void logCb(void *, int32_t level, const char *message) { Store::log(level, QStringLiteral("rust"), QString::fromUtf8(message)); }

void Store::log(int level, const QString &source, const QString &msg) { Devtools::appendLog(level, source, msg); }

Store::Store(const QJsonObject &config, QObject *parent) : QObject(parent) {
    g_store = this;
    QByteArray cfg = QJsonDocument(config).toJson(QJsonDocument::Compact);
    m_app = mmApi().create(cfg.constData(), notifyCb, logCb, this);
    refresh();
}

Store::~Store() {
    if (m_app) mmApi().destroy(m_app);
    m_app = nullptr;
    g_store = nullptr;
}

Store *Store::instance() { return g_store; }

void Store::onNotify(const QStringList &paths) {
    if (paths.contains(QStringLiteral("@render"))) emit renderRequested();
    bool data = false;
    for (const auto &p : paths)
        if (p != QStringLiteral("@render")) data = true;
    if (!data || m_pending) return;
    m_pending = true;
    // coalesce a burst of notes into one re-read
    QTimer::singleShot(0, this, [this] {
        m_pending = false;
        refresh();
    });
}

void Store::refresh() {
    g_dirty = false;
    char *s = mmApi().state(m_app, "");
    QJsonDocument d = QJsonDocument::fromJson(s);
    mmApi().string_free(s);
    m_state = d.object().toVariantMap();
    m_version++;
    emit stateChanged();
}

QJsonValue Store::stateJson(const QString &path) const {
    char *s = mmApi().state(m_app, path.toUtf8().constData());
    QByteArray b(s);
    mmApi().string_free(s);
    // wrap so scalars parse too
    QJsonDocument d = QJsonDocument::fromJson("[" + b + "]");
    return d.array().isEmpty() ? QJsonValue() : d.array().at(0);
}

QJsonObject Store::callJson(const QString &cmd, const QJsonValue &args) {
    QByteArray a = args.isUndefined() || args.isNull() ? QByteArray("{}")
                   : args.isObject() ? QJsonDocument(args.toObject()).toJson(QJsonDocument::Compact)
                   : QJsonDocument(QJsonArray{args}).toJson(QJsonDocument::Compact);
    char *r = mmApi().call(m_app, cmd.toUtf8().constData(), a.constData());
    QJsonObject o = QJsonDocument::fromJson(r).object();
    mmApi().string_free(r);
    // what the command changed is visible to the next binding or /eval at once (the change note follows anyway);
    // a command that changed nothing (loop_frames, viz_events) doesn't re-read, so a binding may call it
    if (g_dirty) refresh();
    return o;
}

QVariant Store::call(const QString &cmd, const QVariant &args) {
    QJsonValue a = QJsonValue::fromVariant(args);
    QJsonObject r = callJson(cmd, a);
    if (!r.value("ok").toBool()) {
        Store::log(2, QStringLiteral("store"), QStringLiteral("call %1 failed: %2").arg(cmd, r.value("error").toString()));
        return QVariant();
    }
    return r.value("result").toVariant();
}
