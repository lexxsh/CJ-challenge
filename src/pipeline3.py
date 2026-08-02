"""추론 파이프라인 v3 = 검출기 3종 median + 기존 size 로직(pipeline2와 동일).

검출: 최종 RF-DETR 3종(crop/bar/full, 150영상 검수라벨 학습)을 각자 **학습과 동일한 크롭 규칙**으로
      돌려 트랙을 얻고, 영상별 count 의 **중앙값**에 해당하는 모델의 트랙을 채택.
      (final_test_eval 의 median 은 count 숫자 중앙값이라 치수를 못 냄 -> 트랙까지 고르도록 구현)
size: pipeline2 와 동일 — CNN(tight×4 + rail×2, ep8) × (1-alpha) + 기하스택 × alpha(0.6)

★ 크롭 규칙은 학습과 반드시 일치 (train/serve skew 방지):
    crop -> blackbar.crop_bounds(frames, belt_seg)[0]   (벨트 세그 기반)
    bar  -> blackbar.black_bar_bounds(frames)
    full -> 원본 전체
  검출 박스는 크롭 좌표로 나오므로 **원점(x0,y0)을 더해 전체프레임 좌표로 환산** 후 추적/사이징.

⚠️ 정직성 한계: 이 3종은 test 50 을 학습에 포함 -> test 50 수치(median 0.100)는 in-sample.
   히든 test 에선 처음 보는 영상이므로 그 값이 재현되지 않음. fold OOF 기준 정직값은 0.233~0.280.
"""
import numpy as np
import cv2
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
from . import video_io as vio
from .conveyor import corridor_mask
from .counting import corridor_axis
from .ocsort import OCSort, KalmanBoxTracker
from .pipeline2 import size_tracks


def bounds_for(ds, frames, belt_seg):
    """학습 데이터셋과 동일한 크롭 규칙 (final_test_eval.bounds_for 와 동일)."""
    from .blackbar import crop_bounds, black_bar_bounds
    H, W = frames[0].shape[:2]
    if ds == "crop":
        return crop_bounds(frames, belt_seg)[0]
    if ds == "bar":
        return black_bar_bounds(frames)
    return (0, 0, W, H)


def _detect_track(frames, det, ds, belt_seg, corr, W, H, iou_thr, max_age, min_hits):
    """한 검출기로 검출+추적. 박스는 전체프레임 좌표로 환산. -> (tracks, frame_dets)"""
    x0, y0, x1, y1 = bounds_for(ds, frames, belt_seg)
    fa = W*H
    KalmanBoxTracker.count = 0
    trk = OCSort(iou_thr=iou_thr, max_age=max_age, min_hits=min_hits, det_thresh=0.0, byte_iou=0.2)
    frame_dets = {}
    for t, fr in enumerate(frames):
        dets = []
        for d in det(fr[y0:y1, x0:x1]):
            bx1, by1, bx2, by2 = [float(v) for v in d["box"]]
            bx1 += x0; bx2 += x0; by1 += y0; by2 += y0      # ★ 크롭 -> 전체프레임 좌표
            if (bx2-bx1)*(by2-by1) > 0.33*fa:
                continue
            cx = min(max(int((bx1+bx2)//2), 0), W-1); cy = min(max(int((by1+by2)//2), 0), H-1)
            if corr[cy, cx] == 0:
                continue
            dets.append({"box": [bx1, by1, bx2, by2], "score": float(d.get("score", 1.0))})
        frame_dets[t] = [d["box"] for d in dets]      # rail_crop 이 프레임별 박스를 씀
        trk.step(dets, t)
    return trk.result(), frame_dets


def process_video(path, dets3, tight_sizers, rail_sizers, geo, belt_seg=None, frame_stride=3,
                  best_k=3, min_hits=2, max_age=20, iou_thr=0.15, alpha_geo=0.6, debug=False,
                  zoom_seg=True):
    """dets3: [(ds, detector), ...] — ds ∈ {crop,bar,full}"""
    frames, total, fps = vio.read_frames(path, stride=frame_stride)
    if len(frames) < 3:
        return {"objects": []}
    bg = vio.background_plate(frames)
    corridor = corridor_mask(frames, bg)
    corr = cv2.dilate(corridor, np.ones((61, 61), np.uint8))
    axis, center = corridor_axis(corridor)
    H, W = bg.shape[:2]
    _ys = np.where(corridor > 0)[0]
    belt_bot = int(_ys.max()) if len(_ys) else H

    # ---- 검출기 3종: 각자 학습과 동일한 크롭으로 -> 트랙 ----
    cands = []
    for ds, det in dets3:
        tr, fd = _detect_track(frames, det, ds, belt_seg, corr, W, H, iou_thr, max_age, min_hits)
        cands.append((ds, tr, fd))
    counts = [len(c[1]) for c in cands]

    # ---- count 중앙값에 해당하는 모델의 트랙 채택 ----
    med = int(np.median(counts))
    pick = min(range(len(cands)), key=lambda i: (abs(counts[i]-med), i))
    tracks, frame_dets = cands[pick][1], cands[pick][2]
    if debug:
        print(f"    count {dict((cands[i][0], counts[i]) for i in range(len(cands)))} "
              f"-> median {med} -> {cands[pick][0]} 채택", flush=True)

    gcrop = bounds_for("crop", frames, belt_seg) if (zoom_seg and belt_seg is not None) else None
    objs, _ = size_tracks(tracks, frames, bg, corridor, corr, axis, center, belt_bot, W, H,
                          frame_dets, tight_sizers, rail_sizers, geo, best_k, alpha_geo,
                          geo_crop=gcrop, zoom_seg=zoom_seg and gcrop is not None)
    return {"objects": objs, "counts": dict((cands[i][0], counts[i]) for i in range(len(cands))),
            "picked": cands[pick][0]}
