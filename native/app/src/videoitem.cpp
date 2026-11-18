#include "videoitem.h"
#include "mmapi.h"
#include "store.h"
#include <rhi/qrhi.h>

std::atomic<int> VideoItem::renders{0}, VideoItem::composites{0};
std::atomic<double> VideoItem::lastCompositeMs{0};

// Runs on Qt's render thread.
class VideoRenderer : public QQuickRhiItemRenderer {
public:
    void initialize(QRhiCommandBuffer *) override {}
    void synchronize(QQuickRhiItem *it) override { item = static_cast<VideoItem *>(it); }
    void render(QRhiCommandBuffer *) override {
        Store *s = Store::instance();
        if (!s || !s->app()) return;
        QRhi *r = rhi();
        QString err;
        MmRenderInfo info{};
        if (r->backend() != QRhi::Metal) {
            err = QStringLiteral("Qt Quick is not on Metal");
        } else {
            auto *nh = static_cast<const QRhiMetalNativeHandles *>(r->nativeHandles());
            QRhiTexture::NativeTexture nt = colorTexture()->nativeTexture();
            char buf[1024] = {0};
            // Rust encodes + commits on Qt's own MTLCommandQueue before Qt commits this frame: Metal keeps the order
            if (mmApi().render(s->app(), nh->dev, nh->cmdQueue, reinterpret_cast<void *>(nt.object), &info, buf, sizeof buf) != 0)
                err = QString::fromUtf8(buf);
        }
        VideoItem::renders++;
        if (info.composited) {
            VideoItem::composites++;
            VideoItem::lastCompositeMs = info.composite_ms;
        }
        if (err != lastErr) {
            lastErr = err;
            QPointer<VideoItem> it = item;
            QMetaObject::invokeMethod(item, [it, err] { if (it) it->setRenderError(err); }, Qt::QueuedConnection);
        }
    }
    VideoItem *item = nullptr;
    QString lastErr;
};

VideoItem::VideoItem(QQuickItem *parent) : QQuickRhiItem(parent) {
    setColorBufferFormat(TextureFormat::RGBA8);
    if (Store *s = Store::instance()) connect(s, &Store::renderRequested, this, [this] { update(); });
}

VideoItem::~VideoItem() = default;

QQuickRhiItemRenderer *VideoItem::createRenderer() { return new VideoRenderer; }

void VideoItem::setTextureWidth(int w) {
    if (w == m_tw) return;
    m_tw = w;
    applySize();
    emit textureSizeChanged();
}

void VideoItem::setTextureHeight(int h) {
    if (h == m_th) return;
    m_th = h;
    applySize();
    emit textureSizeChanged();
}

void VideoItem::applySize() {
    // the item's texture is the video's own size: Rust composites into it at full resolution
    if (m_tw > 0 && m_th > 0) {
        setFixedColorBufferWidth(m_tw);
        setFixedColorBufferHeight(m_th);
    }
    update();
}

void VideoItem::setRenderError(const QString &e) {
    if (e == m_err) return;
    m_err = e;
    if (!e.isEmpty()) Store::log(3, QStringLiteral("video"), e);
    emit renderErrorChanged();
}
