"""최종 추론 파이프라인: 검출/추적은 v14와 동일(count 보존) + size만 교체.
size = CNN(tight×4 + rail×2, ep8) × 0.4  +  기하스택(seg+belt+DA3 -> RF) × 0.6
"""
import numpy as np
import cv2
import sys, os
from . import video_io as vio
from .conveyor import corridor_mask
from .counting import corridor_axis
from .railscale import local_cmperpx
from .ocsort import OCSort, KalmanBoxTracker
from .pipeline import crop_from_box, rail_crop
from .sizer2 import geo_feat


def process_video(path, det, tight_sizers, rail_sizers, geo, frame_stride=3, best_k=3,
                  min_hits=2, max_age=20, iou_thr=0.15, alpha_geo=0.6, debug=False):
    frames, total, fps = vio.read_frames(path, stride=frame_stride)
    if len(frames) < 3:
        return {"objects": []}
    bg = vio.background_plate(frames)
    corridor = corridor_mask(frames, bg)
    corr = cv2.dilate(corridor, np.ones((61, 61), np.uint8))
    axis, center = corridor_axis(corridor)
    H, W = bg.shape[:2]; fa = H*W
    _ys = np.where(corridor > 0)[0]; belt_bot = int(_ys.max()) if len(_ys) else H
    frame_dets = {}

    # ---- 검출 + OC-SORT (v14와 완전 동일 -> count 불변) ----
    KalmanBoxTracker.count = 0
    trk = OCSort(iou_thr=iou_thr, max_age=max_age, min_hits=min_hits, det_thresh=0.0, byte_iou=0.2)
    for t, fr in enumerate(frames):
        dets = []
        for d in det(fr):
            x1, y1, x2, y2 = [float(v) for v in d["box"]]
            if (x2-x1)*(y2-y1) > 0.33*fa:
                continue
            cx = min(max(int((x1+x2)//2), 0), W-1); cy = min(max(int((y1+y2)//2), 0), H-1)
            if corr[cy, cx] == 0:
                continue
            dets.append({"box": [x1, y1, x2, y2], "score": float(d.get("score", 1.0))})
        frame_dets[t] = [d["box"] for d in dets]
        trk.step(dets, t)
    tracks = trk.result()

    objs, geo_ctx = size_tracks(tracks, frames, bg, corridor, corr, axis, center, belt_bot, W, H,
                                frame_dets, tight_sizers, rail_sizers, geo, best_k, alpha_geo)
    out = {"objects": objs}
    if debug:
        out["debug"] = {"n_tracks": len(tracks), "n_sized": len(objs), "n_geo_frames": geo_ctx}
    return out


def size_tracks(tracks, frames, bg, corridor, corr, axis, center, belt_bot, W, H,
                frame_dets, tight_sizers, rail_sizers, geo, best_k=3, alpha_geo=0.6,
                geo_crop=None, zoom_seg=False):
    """트랙 -> 치수. pipeline3(검출기 3종 median)와 공유하는 단일 사이징 경로.
    ★ 여기 로직은 리더보드 12.097 을 만든 것 — 동작을 바꾸지 말 것."""
    def vis_qual(td):
        x1, y1, x2, y2 = td[1]["box"]
        full = not (x1 <= 2 or y1 <= 2 or x2 >= W-2 or y2 >= H-2)
        return (1 if full else 0, (x2-x1)*(y2-y1))

    geo_ctx = {}      # 프레임별 기하 컨텍스트 캐시 (seg/belt/DA3는 프레임당 1회)
    objs = []
    for tr in tracks:
        obs = sorted(tr.obs, key=vis_qual, reverse=True)
        full = [o for o in obs if vis_qual(o)[0] == 1]
        use = (full or obs)[:best_k]
        cts, crs, gs = [], [], []
        for (t, d) in use:
            cr = crop_from_box(frames[t], d["box"], size=224)
            if cr is None:
                continue
            bx = d["box"]
            cmpp = local_cmperpx(corridor, axis, center, ((bx[0]+bx[2])/2, (bx[1]+bx[3])/2)) or 0.2
            gs.append(geo_feat(bx, cmpp, H))
            cts.append(cr)
            rc = rail_crop(frames[t], bg, bx, frame_dets.get(t, []), 62.3/max(cmpp, 1e-6),
                           size=224, belt_bottom=belt_bot)
            crs.append(rc if rc is not None else cr)
        if not cts:
            continue
        parts = [np.median(s(cts, gs), 0) for s in tight_sizers]
        parts += [np.median(s(crs, gs), 0) for s in rail_sizers]
        cnn = np.mean(parts, 0)

        # 기하: 트랙의 best 프레임 1개
        gdim = None
        if geo is not None:
            t0, d0 = use[0]
            if t0 not in geo_ctx:
                # ★ seg 줌: 벨트 크롭에서 seg_box 실행 -> 작은 박스 마스크 정확 -> z_box 개선
                #   정직 OOF 검증: 10.7377 -> 10.5432 (공통 851박스, -0.19). depth 줌은 노이즈라 제외
                geo_ctx[t0] = geo.frame_ctx(frames[t0], crop=geo_crop, zoom_seg=zoom_seg)
            gdim = geo.dims(geo_ctx[t0], d0["box"])
        pred = np.sort(cnn)[::-1] if gdim is None else \
            (1-alpha_geo)*np.sort(cnn)[::-1] + alpha_geo*np.asarray(gdim)
        w, d_, h = pred
        objs.append({"size_cm": {"w": float(w), "d": float(d_), "h": float(h)}})
    return objs, len(geo_ctx)
