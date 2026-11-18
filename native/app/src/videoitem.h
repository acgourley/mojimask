// The video item: a QQuickRhiItem whose texture Rust draws into. On Qt's render thread it hands Qt's MTLDevice,
// command queue and the item's colour texture to mm_app_render; Rust composites there with wgpu on Qt's own queue.
// It has no logic: its texture size is bound in QML to the video's size (store.state.editor.video), and it redraws
// when the store says "@render".
#pragma once
#include <QQuickRhiItem>
#include <atomic>

class VideoItem : public QQuickRhiItem {
    Q_OBJECT
    Q_PROPERTY(int textureWidth READ textureWidth WRITE setTextureWidth NOTIFY textureSizeChanged)
    Q_PROPERTY(int textureHeight READ textureHeight WRITE setTextureHeight NOTIFY textureSizeChanged)
    Q_PROPERTY(QString renderError READ renderError NOTIFY renderErrorChanged)
public:
    explicit VideoItem(QQuickItem *parent = nullptr);
    ~VideoItem() override;
    QQuickRhiItemRenderer *createRenderer() override;

    int textureWidth() const { return m_tw; }
    int textureHeight() const { return m_th; }
    void setTextureWidth(int w);
    void setTextureHeight(int h);
    QString renderError() const { return m_err; }
    void setRenderError(const QString &e);

    // render() calls since start, composites, last composite time (for /perf)
    static std::atomic<int> renders, composites;
    static std::atomic<double> lastCompositeMs;

signals:
    void textureSizeChanged();
    void renderErrorChanged();

private:
    void applySize();
    int m_tw = 0, m_th = 0;
    QString m_err;
};
