// The store as QML sees it: `store.state` (the Rust state tree as a QVariantMap) and `store.call(cmd, args)`.
// Rust owns the state; this only mirrors it. On a change note from Rust (any thread) it re-reads the state on the
// GUI thread and emits stateChanged; "@render" also asks the video items to draw.
#pragma once
#include "mmapi.h"
#include <QJsonObject>
#include <QJsonValue>
#include <QObject>
#include <QVariantMap>

class Store : public QObject {
    Q_OBJECT
    Q_PROPERTY(QVariantMap state READ state NOTIFY stateChanged)
public:
    explicit Store(const QJsonObject &config, QObject *parent = nullptr);
    ~Store() override;
    static Store *instance();

    QVariantMap state() const { return m_state; }
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
    void refresh();
    MmApp *m_app = nullptr;
    QVariantMap m_state;
    bool m_pending = false;
    quint64 m_version = 0;
};
