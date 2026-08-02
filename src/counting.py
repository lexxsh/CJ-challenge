"""
Gate-temporal 박스 카운팅 (고전 CV, 빠름).

아이디어: 컨베이어를 가로지르는 'gate'(진행축 수직 선분)를 박스가 크게 보이는 구간에 둔다.
박스가 일렬로 gate를 통과하므로, gate의 시간축 신호에서 박스 수를 센다.
  - occupancy(t): gate 위 전경 비율 -> 점유 구간(run) = 박스 그룹
  - appearance(t): gate 위 전경 픽셀 평균색 -> 점유 중 급변 = 맞닿은 새 박스(seam)
공간적으로 병합돼도 '시간축 분리 + 외관 변화'로 분리 카운트한다.
"""
import numpy as np
import cv2
from . import video_io as vio
from . import conveyor as cv_geo


def corridor_axis(corridor):
    """corridor 점들의 PCA 주축(진행축). optical flow 없이 빠름."""
    ys, xs = np.where(corridor > 0)
    if len(xs) < 50:
        return np.array([1.0, 0.0]), np.array([640.0, 360.0])
    pts = np.stack([xs, ys], 1).astype(np.float32)
    c = pts.mean(0)
    u, s, vt = np.linalg.svd(pts - c, full_matrices=False)
    axis = vt[0] / (np.linalg.norm(vt[0]) + 1e-9)
    return axis, c


def _gate_line(corridor, axis, center, along_frac=0.62, half_span=0.6):
    """
    gate 선분 정의. along_frac: 진행축 사영 분위수 위치(박스가 큰 쪽 가깝게).
    반환: gate 픽셀좌표 배열 (K,2)
    """
    ys, xs = np.where(corridor > 0)
    pts = np.stack([xs, ys], 1).astype(np.float32)
    perp = np.array([-axis[1], axis[0]], np.float32)
    along = (pts - center) @ axis
    # '큰 쪽' = corridor 폭이 넓은 끝. along 양끝 각각의 폭 비교
    a_lo, a_hi = np.percentile(along, [8, 92])
    def width_at(a0):
        sel = pts[np.abs(along - a0) < (a_hi - a_lo) * 0.08]
        if len(sel) < 10:
            return 0
        p = (sel - center) @ perp
        return np.percentile(p, 95) - np.percentile(p, 5)
    big_end = a_hi if width_at(a_hi) >= width_at(a_lo) else a_lo
    small_end = a_lo if big_end == a_hi else a_hi
    a_gate = small_end + (big_end - small_end) * along_frac
    gate_center = center + axis * a_gate
    sel = pts[np.abs(along - a_gate) < (a_hi - a_lo) * 0.06]
    if len(sel) < 10:
        wpx = 80.0
    else:
        p = (sel - center) @ perp
        wpx = (np.percentile(p, 95) - np.percentile(p, 5))
    half = wpx * half_span
    ts = np.linspace(-half, half, max(20, int(2 * half)))
    line = gate_center[None, :] + ts[:, None] * perp[None, :]
    return line.astype(np.int32), gate_center, wpx


def count_boxes(frames, bg, corridor=None, thr=40, debug=False):
    if corridor is None:
        corridor = cv_geo.corridor_mask(frames, bg)
    axis, center = corridor_axis(corridor)
    line, gate_center, wpx = _gate_line(corridor, axis, center)
    H, W = bg.shape[:2]
    xs = np.clip(line[:, 0], 0, W - 1)
    ys = np.clip(line[:, 1], 0, H - 1)

    occ, app = [], []
    for fr in frames:
        d = np.abs(fr[ys, xs].astype(np.int16) - bg[ys, xs].astype(np.int16)).sum(1)
        fgm = d > thr
        occ.append(float(fgm.mean()))
        if fgm.sum() >= 3:
            app.append(fr[ys, xs][fgm].mean(0))
        else:
            app.append(np.array([np.nan, np.nan, np.nan]))
    occ = np.array(occ)
    app = np.array(app)

    # 점유 신호 평활 + 임계
    occ_s = cv2.GaussianBlur(occ.reshape(-1, 1), (1, 7), 0).ravel()
    on = occ_s > 0.18
    # run 검출
    runs = []
    i = 0
    while i < len(on):
        if on[i]:
            j = i
            while j < len(on) and on[j]:
                j += 1
            if j - i >= 2:
                runs.append((i, j))
            i = j
        else:
            i += 1

    # 각 run 내부에서 appearance 급변으로 추가 분할
    total = 0
    splits_dbg = []
    for (a, b) in runs:
        seg = app[a:b]
        valid = ~np.isnan(seg[:, 0])
        if valid.sum() < 3:
            total += 1
            continue
        # 연속 프레임 색차
        diffs = np.zeros(b - a)
        prev = None
        for k in range(b - a):
            if valid[k]:
                if prev is not None:
                    diffs[k] = np.abs(seg[k] - prev).sum()
                prev = seg[k]
        # seam = 큰 색변화 피크 (불응기 적용)
        n_split = 0
        thr_app = max(40.0, np.nanpercentile(diffs[diffs > 0], 85) if (diffs > 0).any() else 40.0)
        last = -10
        for k in range(b - a):
            if diffs[k] > thr_app and (k - last) > 3:
                n_split += 1
                last = k
        boxes_in_run = 1 + n_split
        total += boxes_in_run
        splits_dbg.append((a, b, boxes_in_run))

    dbg = {"axis": axis, "gate_center": gate_center, "gate_wpx": round(wpx, 1),
           "n_runs": len(runs), "runs": splits_dbg, "occ": occ, "line": line}
    return total, (dbg if debug else None)
