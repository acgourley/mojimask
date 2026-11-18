#include "devtools.h"
#include "store.h"
#include "videoitem.h"
#include <QBuffer>
#include <QCoreApplication>
#include <QDateTime>
#include <QDir>
#include <QEventLoop>
#include <QFile>
#include <QHttpServer>
#include <QHttpServerRequest>
#include <QHttpServerResponse>
#include <QJsonArray>
#include <QJsonDocument>
#include <QKeySequence>
#include <QPainter>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include <QQmlExpression>
#include <QQuickItem>
#include <QQuickWindow>
#include <QRegularExpression>
#include <QTcpServer>
#include <QTest>
#include <QTimer>
#include <QUrlQuery>
#include <QJSValue>
#include <chrono>
#include <cmath>

static qint64 nowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------------------------------------------- logs
struct LogLine {
    qint64 n;
    qint64 ms;
    int level;
    QString src, msg;
};
static QMutex g_logMu;
static QVector<LogLine> g_log;
static qint64 g_logN = 0;

void Devtools::appendLog(int level, const QString &source, const QString &msg) {
    {
        QMutexLocker l(&g_logMu);
        g_log.append({++g_logN, QDateTime::currentMSecsSinceEpoch(), level, source, msg});
        if (g_log.size() > 5000) g_log.remove(0, 1000);
    }
    if (source == QLatin1String("rust")) fprintf(stderr, "[rust] %s\n", qPrintable(msg));
}

QJsonObject Devtools::logsSince(qint64 since, int max) {
    QMutexLocker l(&g_logMu);
    QJsonArray a;
    static const char *lv[] = {"debug", "info", "warning", "error"};
    for (const auto &x : g_log) {
        if (x.n <= since) continue;
        a.append(QJsonObject{{"n", x.n}, {"ms", x.ms}, {"level", lv[qBound(0, x.level, 3)]}, {"source", x.src}, {"message", x.msg}});
        if (a.size() >= max) break;
    }
    return QJsonObject{{"next", g_logN}, {"lines", a}};
}

// ---------------------------------------------------------------------------------------------------- helpers
static QJsonObject ok(const QJsonValue &v = QJsonValue(true)) { return QJsonObject{{"ok", true}, {"result", v}}; }
static QHttpServerResponse fail(const QString &e, QHttpServerResponse::StatusCode c = QHttpServerResponse::StatusCode::BadRequest) {
    return QHttpServerResponse(QJsonObject{{"ok", false}, {"error", e}}, c);
}
static QJsonObject body(const QHttpServerRequest &r) { return QJsonDocument::fromJson(r.body()).object(); }

static QString typeName(const QObject *o) {
    QString n = QString::fromLatin1(o->metaObject()->className());
    n.remove(QRegularExpression("_QMLTYPE_\\d+$|_QML_\\d+$"));
    if (n.startsWith("QQuick")) n = n.mid(6);
    return n;
}

static QJsonValue toJson(const QVariant &v0) {
    QVariant v = v0;
    if (v.userType() == qMetaTypeId<QJSValue>()) v = v.value<QJSValue>().toVariant();
    QJsonValue j = QJsonValue::fromVariant(v);
    if (j.isNull() && v.isValid() && !v.isNull()) return v.toString();
    return j;
}

static void walk(QQuickItem *it, const std::function<void(QQuickItem *)> &f) {
    f(it);
    for (QQuickItem *c : it->childItems()) walk(c, f);
}

Devtools::Devtools(QQmlApplicationEngine *engine, Store *store, const QString &root, QObject *parent)
    : QObject(parent), m_engine(engine), m_store(store), m_root(root) {}

QQuickWindow *Devtools::window() const {
    const auto roots = m_engine->rootObjects();
    for (int i = roots.size() - 1; i >= 0; --i)
        if (auto *w = qobject_cast<QQuickWindow *>(roots[i])) return w;
    return nullptr;
}

void Devtools::attachWindow(QQuickWindow *w) {
    connect(w, &QQuickWindow::frameSwapped, this, [this] {
        QMutexLocker l(&m_mu);
        m_swaps.append(nowNs());
        if (m_swaps.size() > 600) m_swaps.remove(0, 100);
    }, Qt::DirectConnection);
}

void Devtools::noteReload(double savedToFrameMs, double savedToLoadedMs) {
    m_reloads.append(savedToFrameMs);
    m_lastReloadLoadedMs = savedToLoadedMs;
}

QJsonObject Devtools::itemInfo(QQuickItem *it) const {
    QRectF r = it->mapRectToScene(QRectF(0, 0, it->width(), it->height()));
    QJsonObject o{{"type", typeName(it)},
                  {"name", it->objectName()},
                  {"rect", QJsonArray{std::round(r.x() * 10) / 10, std::round(r.y() * 10) / 10, std::round(r.width() * 10) / 10,
                                      std::round(r.height() * 10) / 10}},
                  {"visible", it->isVisible()},
                  {"enabled", it->isEnabled()},
                  {"focus", it->hasActiveFocus()}};
    const QMetaObject *mo = it->metaObject();
    for (const char *p : {"text", "value", "checked", "from", "to", "placeholderText", "title"}) {
        int i = mo->indexOfProperty(p);
        if (i < 0) continue;
        QVariant v = mo->property(i).read(it);
        if (!v.isValid()) continue;
        if (v.userType() == QMetaType::QString && v.toString().isEmpty()) continue;
        o[p] = toJson(v);
    }
    return o;
}

QJsonObject Devtools::tree(QQuickItem *it, int depth) const {
    QJsonObject o = itemInfo(it);
    if (depth != 0) {
        QJsonArray kids;
        for (QQuickItem *c : it->childItems()) kids.append(tree(c, depth - 1));
        if (!kids.isEmpty()) o["children"] = kids;
    }
    return o;
}

QList<QQuickItem *> Devtools::find(const QString &name, const QString &text) const {
    QList<QQuickItem *> exact, partial;
    QQuickWindow *w = window();
    if (!w) return {};
    walk(w->contentItem(), [&](QQuickItem *it) {
        if (!name.isEmpty()) {
            if (it->objectName() == name) exact << it;
            return;
        }
        QVariant t = it->property("text");
        if (!t.isValid()) return;
        QString s = t.toString();
        if (s == text) exact << it;
        else if (!text.isEmpty() && s.contains(text, Qt::CaseInsensitive)) partial << it;
    });
    // visible ones first
    auto vis = [](QList<QQuickItem *> l) {
        std::stable_sort(l.begin(), l.end(), [](QQuickItem *a, QQuickItem *b) { return a->isVisible() > b->isVisible(); });
        return l;
    };
    return vis(exact) + vis(partial);
}

QQuickItem *Devtools::findOne(const QJsonObject &spec, QString *err) const {
    QString name = spec.value("name").toString(), text = spec.value("text").toString();
    if (name.isEmpty() && text.isEmpty()) {
        *err = "give name or text";
        return nullptr;
    }
    auto l = find(name, text);
    if (l.isEmpty()) {
        *err = QStringLiteral("no item with %1 '%2'").arg(name.isEmpty() ? "text" : "name", name.isEmpty() ? text : name);
        return nullptr;
    }
    // a Text inside a button: click the button (the nearest ancestor with an objectName, if the Text has none)
    QQuickItem *it = l.first();
    if (it->objectName().isEmpty())
        for (QQuickItem *p = it->parentItem(); p; p = p->parentItem())
            if (!p->objectName().isEmpty()) { it = p; break; }
    return it;
}

bool Devtools::pointOf(const QJsonObject &spec, QPointF *p, QString *err) const {
    if (spec.contains("x") && spec.contains("y") && !spec.contains("name") && !spec.contains("text")) {
        *p = QPointF(spec.value("x").toDouble(), spec.value("y").toDouble());
        return true;
    }
    QQuickItem *it = findOne(spec, err);
    if (!it) return false;
    if (!it->isVisible()) {
        *err = QStringLiteral("item '%1' is not visible").arg(it->objectName());
        return false;
    }
    QRectF r = it->mapRectToScene(QRectF(0, 0, it->width(), it->height()));
    // optional offset inside the item, as fractions (fx, fy) of its size
    double fx = spec.value("fx").toDouble(0.5), fy = spec.value("fy").toDouble(0.5);
    *p = QPointF(r.x() + r.width() * fx, r.y() + r.height() * fy);
    return true;
}

QJsonValue Devtools::eval(const QString &js, QString *err) const {
    QQuickWindow *w = window();
    QQmlExpression e(m_engine->rootContext(), w, js);
    QVariant v = e.evaluate();
    if (e.hasError()) {
        *err = e.error().toString();
        return QJsonValue();
    }
    return toJson(v);
}

QImage Devtools::grab(const QString &itemName, QString *err) {
    QQuickWindow *w = window();
    if (!w) {
        *err = "no window";
        return {};
    }
    QImage img = w->grabWindow(); // Qt renders the scene again and reads it back (works with the screen locked)
    const double dpr = img.devicePixelRatio() > 0 ? img.devicePixelRatio() : w->effectiveDevicePixelRatio();
    // privacy: grey out every video unless the open clip is public domain
    // and every picture of a project's video (QML items with mmPicture: crops, previews, thumbnails, the scan's faces)
    // unless that project's video is public domain (mmPublic, worked out by the store from mm_core::privacy)
    bool isPublic = m_store->stateJson("editor.video.public").toBool(false);
    {
        QPainter p(&img);
        walk(w->contentItem(), [&](QQuickItem *it) {
            if (!it->isVisible()) return;
            const bool video = qobject_cast<VideoItem *>(it) != nullptr;
            const bool picture = it->property("mmPicture").toBool();
            if (!(video && !isPublic) && !(picture && !it->property("mmPublic").toBool())) return;
            QRectF r = it->mapRectToScene(QRectF(0, 0, it->width(), it->height()));
            // QPainter on a QImage with a dpr uses logical coordinates
            p.fillRect(r, QColor(128, 128, 128));
        });
        // popups and dialogs live in the overlay, outside contentItem
        if (QQuickItem *ov = w->contentItem()->parentItem()) {
            walk(ov, [&](QQuickItem *it) {
                if (!it->isVisible() || !it->property("mmPicture").toBool() || it->property("mmPublic").toBool()) return;
                p.fillRect(it->mapRectToScene(QRectF(0, 0, it->width(), it->height())), QColor(128, 128, 128));
            });
        }
    }
    if (!itemName.isEmpty()) {
        auto l = find(itemName, QString());
        if (l.isEmpty()) {
            *err = QStringLiteral("no item named '%1'").arg(itemName);
            return {};
        }
        QRectF r = l.first()->mapRectToScene(QRectF(0, 0, l.first()->width(), l.first()->height()));
        QRect px(std::floor(r.x() * dpr), std::floor(r.y() * dpr), std::round(r.width() * dpr), std::round(r.height() * dpr));
        img = img.copy(px.intersected(img.rect()));
    }
    return img;
}

static QJsonValue pctl(QVector<double> v, double p) {
    if (v.isEmpty()) return QJsonValue();
    std::sort(v.begin(), v.end());
    return std::round(v[qRound((v.size() - 1) * p)] * 100) / 100;
}

QJsonObject Devtools::perf() {
    QVector<double> gaps;
    {
        QMutexLocker l(&m_mu);
        for (int i = 1; i < m_swaps.size(); ++i) gaps.append((m_swaps[i] - m_swaps[i - 1]) * 1e-6);
    }
    QJsonArray reloads;
    for (double r : m_reloads) reloads.append(r);
    return QJsonObject{
        {"frame_ms", QJsonObject{{"n", gaps.size()}, {"median", pctl(gaps, 0.5)}, {"p95", pctl(gaps, 0.95)}}},
        {"video_item", QJsonObject{{"renders", VideoItem::renders.load()}, {"composites", VideoItem::composites.load()},
                                   {"last_composite_ms", VideoItem::lastCompositeMs.load()}}},
        {"qml_reload_saved_to_frame_ms", reloads},
        {"qml_reload_saved_to_loaded_ms_last", m_lastReloadLoadedMs},
        {"store", m_store->callJson("perf", QJsonValue()).value("result")},
    };
}

void Devtools::writeInfo() {
    QStringList args = QCoreApplication::arguments();
    QJsonObject o{{"port", m_port},
                  {"pid", QCoreApplication::applicationPid()},
                  {"url", QStringLiteral("http://127.0.0.1:%1").arg(m_port)},
                  {"exe", QCoreApplication::applicationFilePath()},
                  {"args", QJsonArray::fromStringList(args.mid(1))},
                  {"cwd", QDir::currentPath()},
                  {"started", QDateTime::currentDateTime().toString(Qt::ISODateWithMs)}};
    QDir().mkpath(m_root + "/out/native");
    QFile f(m_root + "/out/native/devtools.json");
    if (f.open(QIODevice::WriteOnly)) f.write(QJsonDocument(o).toJson());
}

// ---------------------------------------------------------------------------------------------------- routes
bool Devtools::start(int from, int to) {
    m_server = new QHttpServer(this);
    using M = QHttpServerRequest::Method;
    const auto any = M::Get | M::Post;
    auto *s = m_server;

    s->route("/", any, [] {
        return QHttpServerResponse(QJsonObject{{"routes", QJsonArray{"/state", "/call", "/tree", "/find", "/click", "/drag", "/key", "/type",
                                                                     "/eval", "/grab", "/reload", "/logs", "/wait", "/perf", "/quit", "/restart"}}});
    });
    s->route("/state", any, [this](const QHttpServerRequest &r) {
        return QHttpServerResponse(ok(m_store->stateJson(r.query().queryItemValue("path"))));
    });
    s->route("/call", any, [this](const QHttpServerRequest &r) {
        QJsonObject b = body(r);
        QString cmd = b.value("cmd").toString(r.query().queryItemValue("cmd"));
        if (cmd.isEmpty()) return fail("give cmd");
        QJsonObject res = m_store->callJson(cmd, b.value("args"));
        return QHttpServerResponse(res, res.value("ok").toBool() ? QHttpServerResponse::StatusCode::Ok
                                                                 : QHttpServerResponse::StatusCode::BadRequest);
    });
    s->route("/tree", any, [this](const QHttpServerRequest &r) {
        QQuickWindow *w = window();
        if (!w) return fail("no window", QHttpServerResponse::StatusCode::ServiceUnavailable);
        QString rootName = r.query().queryItemValue("root");
        int depth = r.query().hasQueryItem("depth") ? r.query().queryItemValue("depth").toInt() : -1;
        QQuickItem *root = w->contentItem();
        if (!rootName.isEmpty()) {
            auto l = find(rootName, QString());
            if (l.isEmpty()) return fail("no item named " + rootName);
            root = l.first();
        }
        QJsonObject t = tree(root, depth);
        return QHttpServerResponse(ok(QJsonObject{{"window", QJsonObject{{"width", w->width()}, {"height", w->height()},
                                                                         {"dpr", w->effectiveDevicePixelRatio()}, {"title", w->title()}}},
                                                  {"tree", t}}));
    });
    s->route("/find", any, [this](const QHttpServerRequest &r) {
        QJsonObject b = body(r);
        QString name = b.value("name").toString(r.query().queryItemValue("name", QUrl::FullyDecoded));
        QString text = b.value("text").toString(r.query().queryItemValue("text", QUrl::FullyDecoded));
        QJsonArray a;
        for (QQuickItem *it : find(name, text)) a.append(itemInfo(it));
        return QHttpServerResponse(ok(a));
    });
    s->route("/click", any, [this](const QHttpServerRequest &r) {
        QJsonObject b = body(r);
        QPointF p;
        QString err;
        if (!pointOf(b, &p, &err)) return fail(err);
        QQuickWindow *w = window();
        Qt::MouseButton btn = b.value("button").toString() == "right" ? Qt::RightButton : Qt::LeftButton;
        if (b.value("double").toBool()) QTest::mouseDClick(w, btn, Qt::NoModifier, p.toPoint());
        else QTest::mouseClick(w, btn, Qt::NoModifier, p.toPoint());
        return QHttpServerResponse(ok(QJsonObject{{"x", p.x()}, {"y", p.y()}}));
    });
    s->route("/drag", any, [this](const QHttpServerRequest &r) {
        QJsonObject b = body(r);
        QPointF a, z;
        QString err;
        if (!pointOf(b.value("from").toObject(), &a, &err)) return fail("from: " + err);
        if (b.contains("to")) {
            if (!pointOf(b.value("to").toObject(), &z, &err)) return fail("to: " + err);
        } else {
            z = a + QPointF(b.value("dx").toDouble(), b.value("dy").toDouble());
        }
        int steps = qMax(1, b.value("steps").toInt(10));
        QQuickWindow *w = window();
        QTest::mousePress(w, Qt::LeftButton, Qt::NoModifier, a.toPoint());
        for (int i = 1; i <= steps; ++i) {
            QPointF q = a + (z - a) * (double(i) / steps);
            QTest::mouseMove(w, q.toPoint(), 8);
        }
        QTest::mouseRelease(w, Qt::LeftButton, Qt::NoModifier, z.toPoint());
        return QHttpServerResponse(ok(QJsonObject{{"from", QJsonArray{a.x(), a.y()}}, {"to", QJsonArray{z.x(), z.y()}}, {"steps", steps}}));
    });
    s->route("/key", any, [this](const QHttpServerRequest &r) {
        QString k = body(r).value("key").toString(r.query().queryItemValue("key"));
        QKeySequence ks(k);
        if (ks.isEmpty()) return fail("can't read key '" + k + "' (e.g. Space, Right, Ctrl+E, A)");
        QKeyCombination kc = ks[0];
        QTest::keyClick(window(), kc.key(), kc.keyboardModifiers());
        return QHttpServerResponse(ok(k));
    });
    s->route("/type", any, [this](const QHttpServerRequest &r) {
        QString t = body(r).value("text").toString();
        for (QChar c : t) {
            if (c.unicode() < 128) QTest::keyClick(window(), char(c.unicode()));
            else QTest::sendKeyEvent(QTest::Click, window(), Qt::Key_unknown, QString(c), Qt::NoModifier);
        }
        return QHttpServerResponse(ok(t.size()));
    });
    s->route("/eval", any, [this](const QHttpServerRequest &r) {
        QString js = body(r).value("js").toString(r.query().queryItemValue("js", QUrl::FullyDecoded));
        QString err;
        QJsonValue v = eval(js, &err);
        if (!err.isEmpty()) return fail(err);
        return QHttpServerResponse(ok(v));
    });
    s->route("/grab", any, [this](const QHttpServerRequest &r) {
        QString err;
        QImage img = grab(r.query().queryItemValue("name"), &err);
        if (img.isNull()) return fail(err.isEmpty() ? QStringLiteral("grab failed") : err);
        QByteArray png;
        QBuffer b(&png);
        b.open(QIODevice::WriteOnly);
        img.save(&b, "PNG");
        return QHttpServerResponse("image/png", png);
    });
    s->route("/reload", any, [this] {
        bool good = reload && reload();
        return good ? QHttpServerResponse(ok()) : fail("reload failed (see /logs)");
    });
    s->route("/logs", any, [](const QHttpServerRequest &r) {
        return QHttpServerResponse(ok(logsSince(r.query().queryItemValue("since").toLongLong())));
    });
    s->route("/wait", any, [this](const QHttpServerRequest &r) {
        QString js = r.query().queryItemValue("js", QUrl::FullyDecoded);
        if (js.isEmpty()) js = body(r).value("js").toString();
        int timeout = r.query().hasQueryItem("timeout") ? r.query().queryItemValue("timeout").toInt() : 10000;
        QElapsedTimer t;
        t.start();
        QString err;
        QJsonValue v;
        // a nested event loop: the app keeps running (and rendering) while we poll
        while (true) {
            err.clear();
            v = eval(js, &err);
            bool truthy = err.isEmpty() && !(v.isNull() || v.isUndefined() || v == QJsonValue(false) || v == QJsonValue(0) || v == QJsonValue(""));
            if (truthy) return QHttpServerResponse(ok(QJsonObject{{"value", v}, {"waited_ms", double(t.elapsed())}}));
            if (t.elapsed() > timeout) break;
            QEventLoop loop;
            QTimer::singleShot(10, &loop, &QEventLoop::quit);
            loop.exec();
        }
        return QHttpServerResponse(QJsonObject{{"ok", false}, {"error", err.isEmpty() ? QStringLiteral("timed out") : err},
                                               {"value", v}, {"waited_ms", double(t.elapsed())}},
                                   QHttpServerResponse::StatusCode::RequestTimeout);
    });
    s->route("/perf", any, [this] { return QHttpServerResponse(ok(perf())); });
    s->route("/quit", any, [] {
        QTimer::singleShot(50, qApp, [] { qApp->quit(); });
        return QHttpServerResponse(ok());
    });
    s->route("/restart", any, [this] {
        QTimer::singleShot(50, this, [this] { if (restart) restart(); });
        return QHttpServerResponse(ok());
    });

    for (int p = from; p <= to; ++p) {
        auto *tcp = new QTcpServer(this);
        if (tcp->listen(QHostAddress::LocalHost, p) && s->bind(tcp)) {
            m_port = p;
            writeInfo();
            appendLog(1, "devtools", QStringLiteral("devtools on http://127.0.0.1:%1").arg(p));
            return true;
        }
        delete tcp;
    }
    appendLog(3, "devtools", QStringLiteral("no free port in %1-%2").arg(from).arg(to));
    return false;
}
