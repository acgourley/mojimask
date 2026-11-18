// The store as QML sees it: `store.state` (the Rust state tree: one property per top-level section) and
// `store.call(cmd, args)`. Rust owns the state; this only mirrors it. On a change note from Rust (any thread) it
// re-reads, on the GUI thread, the sections the note names, and sets those that changed: a binding on
// `store.state.editor` re-evaluates when the editor changes, not when the People page does. (Until 2026-11-18
// `state` was one QVariantMap: every note re-read all ~270 KB and every binding on any section re-ran and converted
// the whole map; playing a 4K party clip that was ~17 ms of the GUI thread per frame, half its time.) "@render"
// also asks the video items to draw.
#pragma once
#include "mmapi.h"
#include <QJsonObject>
#include <QJsonValue>
#include <QObject>
#include <QQmlPropertyMap>
#include <QSet>
#include <QVariantMap>

class Store : public QObject {
    Q_OBJECT
    Q_PROPERTY(QQmlPropertyMap *state READ state CONSTANT)
public:
    explicit Store(const QJsonObject &config, QObject *parent = nullptr);
    ~Store() override;
    static Store *instance();

    QQmlPropertyMap *state() const { return m_state; }
    // Run a command; returns its result, or throws a QML error / logs on failure (QML).
    Q_INVOKABLE QVariant call(const QString &cmd, const QVariant &args = QVariant());
    // {ok, result | error} for the devtools
    QJsonObject callJson(const QString &cmd, const QJsonValue &args);
    QJsonValue stateJson(const QString &path = QString()) const;
    MmApp *app() const { return m_app; }
    quint64 version() const { return m_version; }

    // log ring (Rust's lines arrive here too); the devtools serve it
    static void log(int level, const QString &source, const QString &msg);

    void onNotify(const QStringList &paths);

signals:
    void stateChanged();
    void renderRequested();

private:
    // re-read every section (keys empty) or those named; set the ones that changed
    void refresh(const QSet<QString> &keys = {});
    MmApp *m_app = nullptr;
    QQmlPropertyMap *m_state = nullptr;
    // each section as last read, to tell which changed
    QJsonObject m_json;
    // sections named by change notes since the last re-read ("" = all)
    QSet<QString> m_keys;
    bool m_pending = false;
    quint64 m_version = 0;
};
