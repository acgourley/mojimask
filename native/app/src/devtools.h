// The developer hooks (only with --dev): an HTTP server on 127.0.0.1 (ports 9200-9299) that lets an agent read the
// state, call commands, read the QML item tree, send real mouse and key events, evaluate JS, grab pictures, reload
// QML, read logs and timings, quit and restart. Its port, pid and arguments go to out/native/devtools.json.
// native/tools/mmctl wraps it. Routes: native/ARCHITECTURE.md, "Agent hooks".
//
// Privacy (AGENTS.md): /grab paints every VideoItem's area grey unless the open clip is public domain
// (state editor.video.public: test_clips/jazz.mp4 or out/poc/clips/). Fails closed: unknown means grey.
#pragma once
#include <QElapsedTimer>
#include <QImage>
#include <QJsonObject>
#include <QMutex>
#include <QObject>
#include <QStringList>
#include <QVector>
#include <functional>

class QQmlApplicationEngine;
class QQuickItem;
class QQuickWindow;
class QHttpServer;
class Store;

class Devtools : public QObject {
    Q_OBJECT
public:
    Devtools(QQmlApplicationEngine *engine, Store *store, const QString &root, QObject *parent = nullptr);
    bool start(int from = 9200, int to = 9299);
    int port() const { return m_port; }

    // the log ring, kept with or without --dev (thread-safe)
    static void appendLog(int level, const QString &source, const QString &msg);
    static QJsonObject logsSince(qint64 since, int max = 2000);

    // timings the host measures
    void attachWindow(QQuickWindow *w);
    void noteReload(double savedToFrameMs, double savedToLoadedMs);

    std::function<bool()> reload;  // reload QML now; false if it failed
    std::function<void()> restart; // relaunch with the same arguments

    QQuickWindow *window() const;
    QImage grab(const QString &itemName, QString *err);

private:
    QJsonObject tree(QQuickItem *it, int depth) const;
    QJsonObject itemInfo(QQuickItem *it) const;
    QList<QQuickItem *> find(const QString &name, const QString &text) const;
    QQuickItem *findOne(const QJsonObject &spec, QString *err) const;
    bool pointOf(const QJsonObject &spec, QPointF *p, QString *err) const;
    QJsonValue eval(const QString &js, QString *err) const;
    QJsonObject perf();
    void writeInfo();

    QQmlApplicationEngine *m_engine;
    Store *m_store;
    QString m_root;
    QHttpServer *m_server = nullptr;
    int m_port = 0;
    QMutex m_mu;
    QVector<qint64> m_swaps; // ns, render thread
    QVector<double> m_reloads;
    double m_lastReloadLoadedMs = -1;
};
