"""
컨베이어 기하 + 메트릭 스케일 추정.

핵심: 레일(컨베이어) 폭 = 62.3cm 고정. 이를 화면에서 측정해 cm/px 스케일을 잡는다.
v0: 박스가 휩쓰는 corridor(전경 union)의 진행방향 수직 폭 ≈ 벨트 폭으로 근사.
    (데이터상 최대 박스폭 65.6cm ≈ 벨트 62.3cm 이므로 corridor 폭 ≈ 벨트폭 가정 타당)
다음 단계: 배경 plate에서 레일 에지 Hough 검출 + 소실점 호모그래피로 대체(정밀).
"""
import cv2
import numpy as np

RAIL_CM = 62.3


def motion_axis(frames, bg, step=3, thr=40):
    """전경 영역의 dominant optical-flow 방향(단위벡터) 추정."""
    vecs = []
    for i in range(0, len(frames) - step, step):
        a = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        b = cv2.cvtColor(frames[i + step], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 21, 3, 5, 1.2, 0)
        d = np.abs(frames[i].astype(np.int16) - bg.astype(np.int16)).sum(2)
        m = d > thr
        if m.sum() < 300:
            continue
        fx = flow[..., 0][m].mean()
        fy = flow[..., 1][m].mean()
        if fx * fx + fy * fy > 0.25:  # 유의미한 운동만
            vecs.append((fx, fy))
    if not vecs:
        return np.array([1.0, 0.0]), 0.0
    v = np.mean(vecs, 0)
    speed = float(np.linalg.norm(v))
    n = v / (np.linalg.norm(v) + 1e-9)
    return n, speed


def corridor_mask(frames, bg, thr=40, sample=80):
    """박스가 지나간 영역의 union 마스크."""
    idx = np.linspace(0, len(frames) - 1, min(sample, len(frames))).astype(int)
    acc = np.zeros(bg.shape[:2], np.uint16)
    for i in idx:
        d = np.abs(frames[i].astype(np.int16) - bg.astype(np.int16)).sum(2)
        acc += (d > thr).astype(np.uint16)
    # 여러 프레임에서 반복적으로 전경인 곳 = 벨트 위 경로
    band = (acc >= max(2, len(idx) // 12)).astype(np.uint8) * 255
    band = cv2.morphologyEx(band, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    band = cv2.morphologyEx(band, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    return band


def estimate_scale(corridor, axis):
    """
    corridor의 '진행방향 수직' 폭(px)을 측정 -> cm_per_px = 62.3 / width_px.
    perp = axis에 수직인 방향. corridor 점들을 perp축에 사영해 폭 추정(로버스트).
    반환: (cm_per_px, gate_center(x,y), perp_unit, width_px)
    """
    ys, xs = np.where(corridor > 0)
    if len(xs) < 50:
        return None
    pts = np.stack([xs, ys], 1).astype(np.float32)
    center = pts.mean(0)
    perp = np.array([-axis[1], axis[0]], np.float32)
    # corridor 중앙 부근(진행축 사영의 중앙 40%)에서 폭 측정 -> 원근 영향 완화
    along = (pts - center) @ axis
    lo, hi = np.percentile(along, [30, 70])
    sel = pts[(along >= lo) & (along <= hi)]
    if len(sel) < 30:
        sel = pts
    proj = (sel - center) @ perp
    width_px = float(np.percentile(proj, 97) - np.percentile(proj, 3))
    if width_px < 5:
        return None
    cm_per_px = RAIL_CM / width_px
    return cm_per_px, center, perp, width_px
