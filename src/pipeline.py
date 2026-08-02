"""
최종 추론 파이프라인 (ONNX, Docker 호환).
검출(det.onnx) -> corridor/배경차분 필터 -> 추적(count) -> 박스별 풀뷰 best 크롭
-> 치수회귀(size.onnx) + 레일스케일 -> 다프레임 median -> objects[{w,d,h}]
"""
import numpy as np
import cv2
from . import video_io as vio
from .conveyor import corridor_mask
from .counting import corridor_axis
from .railscale import local_cmperpx
from .detector import YoloDet
from .sizer import Sizer, crop_box
from .ocsort import OCSort, KalmanBoxTracker


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]); ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0., ix2-ix1), max(0., iy2-iy1); inter = iw*ih
    ua = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/ua if ua > 0 else 0.


def crop_from_box(frame, box, ctx=0.1, size=192):
    x1, y1, x2, y2 = box
    cx, cy = (x1+x2)/2, (y1+y2)/2
    half = max(x2-x1, y2-y1)*(1+ctx)/2
    X1, Y1, X2, Y2 = int(cx-half), int(cy-half), int(cx+half), int(cy+half)
    H, W = frame.shape[:2]
    crop = frame[max(0, Y1):min(H, Y2), max(0, X1):min(W, X2)]
    if crop.size == 0:
        return None
    crop = cv2.copyMakeBorder(crop, max(0, -Y1), max(0, Y2-H), max(0, -X1), max(0, X2-W),
                              cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return cv2.resize(crop, (size, size))


def rail_crop(frame, bg, box, others, rail_px, size=224, k=1.35, belt_bottom=None):
    """레일폭 프레이밍 + 이웃 배경제거 (size3_rail 학습 크롭과 동일). 앙상블 rail모델용."""
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = box; cx, cy = (x1+x2)/2, (y1+y2)/2
    half = max(x2-x1, y2-y1, rail_px*k)/2
    f = frame.copy()
    for ob in others:
        if _iou(box, ob[:4]) > 0.45:
            continue
        ox1, oy1, ox2, oy2 = max(0, int(ob[0])), max(0, int(ob[1])), min(W, int(ob[2])), min(H, int(ob[3]))
        if ox2 > ox1 and oy2 > oy1:
            f[oy1:oy2, ox1:ox2] = bg[oy1:oy2, ox1:ox2]
    tx1, ty1, tx2, ty2 = max(0, int(x1)), max(0, int(y1)), min(W, int(x2)), min(H, int(y2))
    f[ty1:ty2, tx1:tx2] = frame[ty1:ty2, tx1:tx2]
    X1, Y1, X2, Y2 = int(cx-half), int(cy-half), int(cx+half), int(cy+half)
    if belt_bottom is not None and Y2 > belt_bottom:
        d = Y2 - int(belt_bottom); Y1 -= d; Y2 -= d
    crop = f[max(0, Y1):min(H, Y2), max(0, X1):min(W, X2)]
    if crop.size == 0:
        return None
    crop = cv2.copyMakeBorder(crop, max(0, -Y1), max(0, Y2-H), max(0, -X1), max(0, X2-W), cv2.BORDER_REPLICATE)
    return cv2.resize(crop, (size, size))


def process_video(path, det, sizer: Sizer, frame_stride=3, best_k=3,
                  min_hits=3, max_age=30, iou_thr=0.25, debug=False, sizer2=None, seq_sizer=None,
                  sizer4=None, depther=None, tight_extra=None, rail_extra=None, depth_extra=None,
                  mview=None, mv_alpha=0.3, depther2=None, depth2_sizer=None, depth2_extra=None,
                  byte_track=False, track_high_conf=0.70, byte_iou=0.2,
                  drop_short_fp=False, temporal_agg="median"):
    frames, total, fps = vio.read_frames(path, stride=frame_stride)
    if len(frames) < 3:
        return {"objects": []}
    bg = vio.background_plate(frames); bgi = bg.astype(np.int16)
    corridor = corridor_mask(frames, bg)
    corr = cv2.dilate(corridor, np.ones((61, 61), np.uint8))   # 관대(작은/가장자리 박스 보존, count_eval와 동일)
    axis, center = corridor_axis(corridor)
    H, W = bg.shape[:2]; fa = H * W
    _ys = np.where(corridor > 0)[0]; belt_bot = int(_ys.max()) if len(_ys) else H   # rail크롭 클램프용
    frame_dets = {}   # rail크롭 이웃제거용 (프레임별 박스)

    # 검출 + OC-SORT 추적 (모션+OCR 갭복구+퇴장확정 -> 영상 고유 박스 카운트)
    KalmanBoxTracker.count = 0
    trk = OCSort(iou_thr=iou_thr, max_age=max_age, min_hits=min_hits,
                 det_thresh=(track_high_conf if byte_track else 0.0), byte_iou=byte_iou)
    for t, fr in enumerate(frames):
        dets = []
        for d in det(fr):   # 완화 필터: corridor+area만 (fg 강제X)
            x1, y1, x2, y2 = [float(v) for v in d["box"]]
            if (x2-x1)*(y2-y1) > 0.33*fa:
                continue
            cx = min(max(int((x1+x2)//2), 0), W-1); cy = min(max(int((y1+y2)//2), 0), H-1)
            if corr[cy, cx] == 0:
                continue
            dets.append({"box": [x1, y1, x2, y2], "score": float(d.get("score", 1.0))})
        frame_dets[t] = [d["box"] for d in dets]
        trk.step(dets, t)
    tracks = trk.result()   # finished(퇴장확정)+alive(확정) = 고유 박스
    if drop_short_fp:
        kept = []
        for tr in tracks:
            scores = [float(d.get("score", 1.0)) for _, d in tr.obs]
            if len(tr.obs) <= 2 and scores and float(np.mean(scores)) < 0.80:
                continue
            kept.append(tr)
        tracks = kept

    # 박스별 풀뷰 best 크롭 -> per-track 3-way 앙상블 (tight median + rail median + seq 어텐션)
    is_v2 = sizer.__class__.__name__ == "Sizer2"
    crop_sz = getattr(sizer, "imgsz", 192)
    rail_sz = getattr(sizer2, "imgsz", 224) if sizer2 is not None else 224
    from .sizer2 import geo_feat
    def vis_qual(td):
        x1, y1, x2, y2 = td[1]["box"]
        full = not (x1 <= 2 or y1 <= 2 or x2 >= W-2 or y2 >= H-2)
        return (1 if full else 0, (x2-x1)*(y2-y1))
    def _temporal_reduce(vals):
        vals = np.asarray(vals, dtype=float)
        if len(vals) == 1:
            return vals[0]
        if temporal_agg == "mean":
            return np.mean(vals, 0)
        if temporal_agg == "trim":
            if len(vals) <= 3:
                return np.median(vals, 0)
            med = np.median(vals, 0)
            dist = np.abs(vals - med).mean(1)
            keep = np.argsort(dist)[:max(2, len(vals)-1)]
            return np.mean(vals[keep], 0)
        if temporal_agg == "qmean":
            med = np.median(vals, 0)
            dist = np.abs(vals - med).mean(1)
            w = 1.0 / (dist + 1e-3)
            return (vals * w[:, None]).sum(0) / w.sum()
        return np.median(vals, 0)

    objs = []
    track_preds = []   # (ens_pred, mv_pred|None)
    for tr in tracks:
        obs = sorted(tr.obs, key=vis_qual, reverse=True)
        full = [o for o in obs if vis_qual(o)[0] == 1]
        use = (full or obs)[:best_k]
        cts, crs, gs = [], [], []
        for (t, d) in use:
            cr = crop_from_box(frames[t], d["box"], size=crop_sz)
            if cr is None:
                continue
            bx = d["box"]; cmpp = local_cmperpx(corridor, axis, center, ((bx[0]+bx[2])/2, (bx[1]+bx[3])/2)) or 0.2
            gs.append(geo_feat(bx, cmpp, H) if is_v2 else np.hypot(bx[2]-bx[0], bx[3]-bx[1])*cmpp)
            cts.append(cr)
            if sizer2 is not None:
                rc = rail_crop(frames[t], bg, bx, frame_dets.get(t, []), 62.3/max(cmpp, 1e-6),
                               size=rail_sz, belt_bottom=belt_bot)
                crs.append(rc if rc is not None else cr)
        if not cts:
            continue
        # ablation으로 선정된 핵심 5멤버 딥앙상블 (base/vm/convnext × tight/rail/depth)
        parts = [_temporal_reduce(sizer(cts, gs))]   # base_t (tight)
        for tm in (tight_extra or []):
            parts.append(_temporal_reduce(tm(cts, gs)))   # vm_t, cvx_t (tight, 다른레시피)
        if sizer2 is not None:
            parts.append(_temporal_reduce(sizer2(crs, gs)))   # rail
            for rm in (rail_extra or []):
                parts.append(_temporal_reduce(rm(crs, gs)))   # rail 다른레시피
        if seq_sizer is not None:
            parts.append(np.asarray(seq_sizer(cts, gs)))   # seq
        if sizer4 is not None and depther is not None:
            dcs = [depther.depth_crop(c) for c in cts]
            parts.append(_temporal_reduce(sizer4(dcs, gs)))   # depth (DA2-Small)
            for dm in (depth_extra or []):
                parts.append(_temporal_reduce(dm(dcs, gs)))
        if depth2_sizer is not None and depther2 is not None:
            dcs2 = [depther2.depth_crop(c) for c in cts]
            parts.append(_temporal_reduce(depth2_sizer(dcs2, gs)))   # depthB (DA2-Base)
            for dm in (depth2_extra or []):
                parts.append(_temporal_reduce(dm(dcs2, gs)))
        ens = np.mean(parts, 0)
        # 멀티뷰: 원근 다양한 3뷰 (full-visible을 cy 오름차순 -> far/mid/near)
        mv_pred = None
        if mview is not None:
            pool = sorted(full or obs, key=lambda o: (o[1]["box"][1]+o[1]["box"][3])/2)   # cy 오름차순
            if len(pool) >= 2:
                sel = [pool[0], pool[len(pool)//2], pool[-1]]
                mcs, mgs = [], []
                for (t, d) in sel:
                    cr = crop_from_box(frames[t], d["box"], size=224)
                    if cr is None:
                        break
                    bx = d["box"]; cmpp = local_cmperpx(corridor, axis, center, ((bx[0]+bx[2])/2, (bx[1]+bx[3])/2)) or 0.2
                    mcs.append(cr); mgs.append(geo_feat(bx, cmpp, H))
                if len(mcs) == 3:
                    mv_pred = np.asarray(mview(mcs, mgs))
        track_preds.append((ens, mv_pred))
    # 예측부피 상위 1/3 트랙만 멀티뷰 블렌드 (입력 기반 게이트)
    if track_preds:
        vols = np.array([p[0].prod() for p in track_preds])
        thr = np.quantile(vols, 2/3) if len(vols) >= 3 else np.inf
        for ens, mv_pred in track_preds:
            if mview is not None and mv_alpha > 0 and mv_pred is not None and ens.prod() >= thr:
                ens = (1-mv_alpha)*ens + mv_alpha*mv_pred
            w, d_, h = ens
            objs.append({"size_cm": {"w": float(w), "d": float(d_), "h": float(h)}})
    out = {"objects": objs}
    if debug:
        out["debug"] = {"n_tracks": len(tracks), "n_sized": len(objs)}
    return out
