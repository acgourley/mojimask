"""Cut the default portrait (web/assets/portrait_mona.png) out of the start page's Mona Lisa (public domain):
her head and hair, traced by hand, upscaled 4x with a feathered edge. Also writes a preview on white.
  .venv/bin/python tools/mona_portrait.py web/assets/gallery/mona.jpg web/assets/portrait_mona.png preview.png
"""
import cv2, numpy as np, sys
src, out, prev = sys.argv[1:4]
im = cv2.imread(src)
K = 4
pts = [(175,58),(195,59),(213,65),(230,77),(243,93),(253,112),(259,135),(262,160),(263,190),(262,215),(257,234),
       (246,243),(228,240),(212,232),(196,228),(178,228),(160,232),(140,240),(122,240),(114,226),(113,200),(115,170),
       (120,150),(125,130),(131,110),(141,88),(152,72),(162,63)]
x0, y0, x1, y1 = 100, 50, 275, 258
crop = cv2.resize(im[y0:y1, x0:x1], None, fx=K, fy=K, interpolation=cv2.INTER_LANCZOS4)
m = np.zeros(crop.shape[:2], np.uint8)
cv2.fillPoly(m, [np.array([((x-x0)*K, (y-y0)*K) for x, y in pts], np.int32)], 255, cv2.LINE_AA)
m = cv2.GaussianBlur(m, (0, 0), 1.5 * K)
m = np.clip((m.astype(np.float32) - 40) * 255 / 175, 0, 255).astype(np.uint8)  # soft but not hazy
cv2.imwrite(out, np.dstack([crop, m]))
bg = np.full_like(crop, 255); a = m[..., None] / 255.0
cv2.imwrite(prev, (crop * a + bg * (1 - a)).astype(np.uint8))
print(crop.shape)
