"""크롭 경계 결정 — **학습·추론이 반드시 이 함수를 공유해야 함** (train/serve skew 방지).

3단 폴백:
  1순위 벨트 seg bbox + 마진   (150영상 실측: seg 100% 성공, 면적 22% -> 2.13배 확대)
  2순위 검은 띠 크롭            (seg 실패/이상 시. 1042x720, 1.23배)
  3순위 원본                    (둘 다 실패)

근거(150영상 실측):
  - 검은 띠: mp4 디코딩 시 픽셀 정확히 0 vs 콘텐츠 최소 7.3 -> 임계 4로 완전 분리
    (우리 JPEG 재저장본만 아티팩트로 최대 3 -> 그래도 4 미만)
  - GT 박스가 벨트 bbox 밖으로 최대 128px 초과 -> 마진 150px 면 안전
"""
import numpy as np

THR = 4.0          # 검은 띠 밝기 임계
MIN_BAR = 0.02     # 이 비율 미만 띠는 무시
MARGIN = 150       # 벨트 bbox 마진 (실측 최대 초과 128px + 여유)
MIN_BELT_FRAC = 0.03   # 벨트 bbox가 프레임의 이 비율 미만이면 seg 이상으로 간주


def black_bar_bounds(frames, thr=THR, min_bar=MIN_BAR):
    """검은 띠 제거 경계. frames: 이미지 또는 리스트. -> (x1,y1,x2,y2) exclusive"""
    a = np.asarray(frames)
    if a.ndim == 3:
        a = a[None]
    m = a.astype(np.float32).mean(-1).max(0)     # 프레임 최대 -> 띠는 여전히 0
    H, W = m.shape
    cols = np.where(m.max(0) > thr)[0]
    rows = np.where(m.max(1) > thr)[0]
    if len(cols) == 0 or len(rows) == 0:
        return 0, 0, W, H
    x1, x2 = int(cols.min()), int(cols.max())+1
    y1, y2 = int(rows.min()), int(rows.max())+1
    if (x1 + (W-x2)) < min_bar*W:
        x1, x2 = 0, W
    if (y1 + (H-y2)) < min_bar*H:
        y1, y2 = 0, H
    return x1, y1, x2, y2


def belt_bounds(frames, belt_seg, margin=MARGIN, min_frac=MIN_BELT_FRAC):
    """벨트 seg bbox + 마진. 실패/이상이면 None.
    belt_seg: geo_infer.SegONNX (belt_seg.onnx). frames 여러 장의 union (박스 가림 보완)."""
    a = frames if isinstance(frames, (list, tuple)) else [frames]
    H, W = a[0].shape[:2]
    bm = np.zeros((H, W), bool)
    for im in a[:4]:
        for m, _, _ in belt_seg(im):
            bm |= m
    if not bm.any():
        return None
    ys, xs = np.where(bm)
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1
    if (x2-x1)*(y2-y1) < min_frac*W*H:      # 비정상적으로 작으면 seg 이상
        return None
    return (max(x1-margin, 0), max(y1-margin, 0), min(x2+margin, W), min(y2+margin, H))


def crop_bounds(frames, belt_seg=None, margin=MARGIN):
    """최종 크롭 경계 (3단 폴백). -> ((x1,y1,x2,y2), mode)"""
    if belt_seg is not None:
        b = belt_bounds(frames, belt_seg, margin)
        if b is not None:
            return b, "belt"
    b = black_bar_bounds(frames)
    a = np.asarray(frames)
    H, W = (a.shape[1:3] if a.ndim == 4 else a.shape[:2])
    if (b[2]-b[0]) < W or (b[3]-b[1]) < H:
        return b, "blackbar"
    return (0, 0, W, H), "full"


def crop(img, b):
    x1, y1, x2, y2 = b
    return img[y1:y2, x1:x2]
