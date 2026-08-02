"""레일(62.3cm) 기반 국소 스케일 — 박스 위치에서 컨베이어 폭으로 cm/px (원근 부분보정)."""
import numpy as np
RAIL_CM = 62.3


def local_cmperpx(corridor, axis, center, pt):
    ys, xs = np.where(corridor > 0)
    if len(xs) < 30:
        return None
    pts = np.stack([xs, ys], 1).astype(np.float32)
    perp = np.array([-axis[1], axis[0]], np.float32)
    along_all = (pts - center) @ axis
    a0 = float((np.asarray(pt) - center) @ axis)
    span = np.percentile(along_all, 95) - np.percentile(along_all, 5)
    band = pts[np.abs(along_all - a0) < max(8.0, span * 0.06)]
    if len(band) < 15:
        band = pts
    proj = (band - center) @ perp
    wpx = np.percentile(proj, 95) - np.percentile(proj, 5)
    if wpx < 5:
        return None
    return RAIL_CM / wpx
