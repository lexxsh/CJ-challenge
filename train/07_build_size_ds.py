"""
[사이즈 데이터셋 재매칭 빌드] RF-DETR 트랙(vdump) -> 트랙별 현 size.onnx 예측 -> 트랙↔GT Hungarian 매칭(축순열-min)
-> 올바른 GT 라벨로 크롭 저장. 부피랭크(bbox면적, 원근오염) 대신 모델예측 매칭 = D4 라벨노이즈(2.2cm) 제거 목표.
사용: source .venv_train/bin/activate && source scripts/gpu_env.sh && \
      python train/07_build_size_ds.py --dump explore/vdump_rfdetr_all.json --out datasets/size3
이후: python train/10_train_size_final.py --data datasets/size_tight --out checkpoints/f_tight_s1.onnx
"""
import sys, os, json, math, argparse, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scipy.optimize import linear_sum_assignment
from src import video_io as vio
from src.conveyor import corridor_mask
from src.counting import corridor_axis
from src.railscale import local_cmperpx
from src.ocsort import OCSort, KalmanBoxTracker
from src.sizer2 import Sizer2, crop_from_box, geo_feat
from src.geomap import build_rail_homography_precise, build_rail_homography

_RAIL_GOLD = None
def gold_H(vid):
    """rail_gold.json(수동 4코너)로 정확한 벨트 top-down 호모그래피 (train용)."""
    global _RAIL_GOLD
    if _RAIL_GOLD is None:
        p = "explore/rail_gold.json"
        _RAIL_GOLD = json.load(open(p)) if os.path.exists(p) else {}
    if vid not in _RAIL_GOLD:
        return None
    src = np.array(_RAIL_GOLD[vid], np.float32)
    railpx = (np.linalg.norm(src[0]-src[1])+np.linalg.norm(src[3]-src[2]))/2
    L = (np.linalg.norm(src[3]-src[0])+np.linalg.norm(src[2]-src[1]))/2 * (62.3/railpx)
    dst = np.array([[-62.3/2, 0], [62.3/2, 0], [62.3/2, L], [-62.3/2, L]], np.float32)
    return cv2.getPerspectiveTransform(src, dst)


def bev_crop(frame, bx, H, size=224, pad=0.25):
    """박스 bbox를 BEV(top-down)로 워프해 박스만 격리 crop. tight크롭처럼 박스 채우되 시점=top-down."""
    if H is None:
        return None
    x1, y1, x2, y2 = bx
    cn = np.array([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]], np.float32)
    w = cv2.perspectiveTransform(cn, H)[0]            # bbox 4코너 -> BEV cm좌표
    cx, cy = w[:, 0].mean(), w[:, 1].mean()
    half = max(np.ptp(w[:, 0]), np.ptp(w[:, 1])) * (0.5 + pad)   # 박스 크기에 맞춘 정사각 윈도우(+패딩)
    x0, y0 = cx-half, cy-half; s = size/(2*half)
    S = np.array([[s, 0, -s*x0], [0, s, -s*y0], [0, 0, 1.0]])
    return cv2.warpPerspective(frame, S @ H, (size, size), flags=cv2.INTER_LINEAR)
VDIR = "assignment1/dataset/train"


def pair_cost(p, g):   # 축순열-min = 정렬 두 벡터 L1 (정렬이 canonical)
    return float(np.abs(np.sort(p)[::-1] - np.sort(g)[::-1]).sum())


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]); ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0., ix2-ix1), max(0., iy2-iy1); inter = iw*ih
    ua = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/ua if ua > 0 else 0.


def rail_crop(frame, bg, box, others, rail_px, size=224, k=1.35, tight=False, belt_bottom=None):
    """이웃 박스를 배경판으로 제거. tight=True면 박스 타이트 프레이밍(디테일 유지),
    아니면 레일폭 담는 넓은 프레이밍(스케일 시각기준). 둘 다 이웃제거는 동일.
    belt_bottom: 주면 크롭 하단이 벨트 하단을 넘을 때 위로 시프트(벨트밖 어두운영역 제외)."""
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = box; cx, cy = (x1+x2)/2, (y1+y2)/2
    half = (max(x2-x1, y2-y1)*1.12/2) if tight else (max(x2-x1, y2-y1, rail_px*k)/2)
    f = frame.copy()
    for ob in others:                       # 이웃 박스 -> 배경판으로(레일 보이게)
        if _iou(box, ob[:4]) > 0.45:        # 타겟 자신
            continue
        ox1, oy1, ox2, oy2 = max(0, int(ob[0])), max(0, int(ob[1])), min(W, int(ob[2])), min(H, int(ob[3]))
        if ox2 > ox1 and oy2 > oy1:
            f[oy1:oy2, ox1:ox2] = bg[oy1:oy2, ox1:ox2]
    tx1, ty1, tx2, ty2 = max(0, int(x1)), max(0, int(y1)), min(W, int(x2)), min(H, int(y2))
    f[ty1:ty2, tx1:tx2] = frame[ty1:ty2, tx1:tx2]   # 타겟 원복(이웃 겹침 보호)
    X1, Y1, X2, Y2 = int(cx-half), int(cy-half), int(cx+half), int(cy+half)
    if belt_bottom is not None and Y2 > belt_bottom:   # 클램프: 벨트밖(아래 어두움) 대신 위로 시프트
        d = Y2 - int(belt_bottom); Y1 -= d; Y2 -= d
    crop = f[max(0, Y1):min(H, Y2), max(0, X1):min(W, X2)]
    if crop.size == 0:
        return None
    crop = cv2.copyMakeBorder(crop, max(0, -Y1), max(0, Y2-H), max(0, -X1), max(0, X2-W), cv2.BORDER_REPLICATE)
    return cv2.resize(crop, (size, size))


def _belt_resid(d, corrb, boxes_t, H, W):
    """벨트 평면(robust) 적합 -> 박스=양(+) 잔차(rpos), 벨트 잔차 std(sd) 반환."""
    belt = corrb.copy()
    for b in boxes_t:
        x1, y1, x2, y2 = b[:4]
        belt[max(0, int(y1)):min(H, int(y2)), max(0, int(x1)):min(W, int(x2))] = False
    ys, xs = np.where(belt)
    if len(xs) < 100:
        return None, 1.0
    Yg, Xg = np.mgrid[0:H, 0:W]; sel = belt.copy()
    for _ in range(3):
        yy, xx = np.where(sel)
        A = np.stack([xx, yy, np.ones_like(xx)], 1).astype(np.float32)
        coef, *_ = np.linalg.lstsq(A, d[sel], rcond=None)
        plane = coef[0]*Xg + coef[1]*Yg + coef[2]
        s = (d-plane)[sel].std()+1e-6
        sel = belt & (np.abs(d-plane) < 2*s)
    resid = d - plane
    sgn = np.sign(np.median(resid[corrb][resid[corrb] != 0])) if corrb.sum() else 1.0
    return resid*sgn, float(resid[belt].std()+1e-6)


def _depth_box_feats(rpos, sd, box, corrb, H, W):
    """박스 영역 벨트위 솟음 통계(벨트 sigma 단위, 차원무관): [p90, mean, std]."""
    if rpos is None:
        return [0.0, 0.0, 0.0]
    x1, y1, x2, y2 = [int(v) for v in box]
    m = np.zeros((H, W), bool); m[max(0, y1):min(H, y2), max(0, x1):min(W, x2)] = True
    m &= corrb
    rp = rpos[m]
    if rp.size < 10:
        return [0.0, 0.0, 0.0]
    return [float(np.percentile(rp, 90)/sd), float(rp.mean()/sd), float(rp.std()/sd)]


def _traj_raw(obs, corridor, axis, center):
    """트랙 전체 obs -> (s위치, bbox폭cm, bbox높이cm, cmpp) 배열."""
    ss, bwc, bhc, cm = [], [], [], []
    for (t, d) in obs:
        bx = d["box"]; ctr = ((bx[0]+bx[2])/2, (bx[1]+bx[3])/2)
        cmpp = local_cmperpx(corridor, axis, center, ctr) or 0.2
        s = (ctr[0]-center[0])*axis[0] + (ctr[1]-center[1])*axis[1]
        ss.append(s); bwc.append((bx[2]-bx[0])*cmpp); bhc.append((bx[3]-bx[1])*cmpp); cm.append(cmpp)
    return tuple(np.array(z) for z in (ss, bwc, bhc, cm))


def _traj_feats(raw, srng):
    """위치별 외형크기 곡선 -> 20 요약피처 (traj_test 검증판)."""
    ss, bwc, bhc, cm = raw
    sn = (ss - ss.min())/max(1.0, srng)

    def st(x):
        return [x.mean(), np.median(x), x.min(), x.max(), np.percentile(x, 10), np.percentile(x, 90), x.std()]
    slb = np.polyfit(sn, bwc, 1)[0] if len(sn) > 2 else 0.0
    slh = np.polyfit(sn, bhc, 1)[0] if len(sn) > 2 else 0.0
    return [float(x) for x in st(bwc) + st(bhc) + [slb, slh, cm.mean(), cm.std(), float(len(sn)), float(sn.max()-sn.min())]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="explore/vdump_rfdetr_all.json"); ap.add_argument("--size", default="checkpoints/size.onnx")
    ap.add_argument("--depth", action="store_true"); ap.add_argument("--depth_w", default="checkpoints/depth_da2s.onnx")
    ap.add_argument("--limit", type=int, default=0)   # >0이면 앞 N영상만 (스모크 테스트)
    ap.add_argument("--out", default="datasets/size3")
    ap.add_argument("--conf", type=float, default=0.70); ap.add_argument("--min_hits", type=int, default=2); ap.add_argument("--max_age", type=int, default=20)
    ap.add_argument("--best_k", type=int, default=4)   # 트랙당 저장 크롭 수
    ap.add_argument("--crop", default="rail", choices=["rail", "tight", "tight_clean", "bev"])
    # rail=레일포함+이웃제거 / tight=기존(이웃노이즈 있음) / tight_clean=타이트+이웃제거(디테일+깨끗)
    a = ap.parse_args()
    os.makedirs(f"{a.out}/crops", exist_ok=True)
    dp = json.load(open(a.dump)); stride = dp["stride"]
    GT = json.load(open("assignment1/dataset/train_label.json"))
    gmap = {v["video_id"]: [[o["size_cm"]["w"], o["size_cm"]["d"], o["size_cm"]["h"]] for o in v["objects"]] for v in GT["videos"]}
    sizer = Sizer2(a.size)
    dep = None
    if a.depth:
        from pipeline.depth_check import DepthDA2
        dep = DepthDA2(a.depth_w)
    out_recs = []; nmatch = 0; ntrack = 0
    items = list(dp["videos"].items())
    if a.limit > 0:
        items = items[:a.limit]
    for vi, (vid, v) in enumerate(items):
        if vid not in gmap:
            continue
        frames, _, _ = vio.read_frames(f"{VDIR}/{vid}.mp4", stride=stride)
        bg = vio.background_plate(frames)
        corridor = corridor_mask(frames, bg); corr = cv2.dilate(corridor, np.ones((61, 61), np.uint8))
        axis, center = corridor_axis(corridor); H, W = bg.shape[:2]; fa = H*W
        corrb = corr > 0; dcache = {}   # depth: 프레임별 (rpos, sd) 캐시
        Hbev = None
        if a.crop == "bev":
            Hbev = gold_H(vid)   # rail_gold 수동코너 (정확). test는 rail_seg로 대체 예정
            if Hbev is None:
                Hbev, _, _ = build_rail_homography(corridor)
        _ys = np.where(corridor > 0)[0]; belt_bot = int(_ys.max()) if len(_ys) else H   # 벨트 하단(클램프용)
        # OC-SORT (count-best 파라미터)
        KalmanBoxTracker.count = 0
        trk = OCSort(iou_thr=0.25, max_age=a.max_age, min_hits=a.min_hits)
        n = min(len(frames), len(v["frames"]))
        for t in range(n):
            dets = []
            for x1, y1, x2, y2, s in v["frames"][t]:
                if s < a.conf or (x2-x1)*(y2-y1) > 0.33*fa:
                    continue
                cx = min(max(int((x1+x2)//2), 0), W-1); cy = min(max(int((y1+y2)//2), 0), H-1)
                if corr[cy, cx] == 0:
                    continue
                dets.append({"box": [float(x1), float(y1), float(x2), float(y2)]})
            trk.step(dets, t)
        tracks = trk.result(); ntrack += len(tracks)
        if not tracks:
            continue

        # 트랙별 best 크롭 + geo + 현모델 예측
        def vis_qual(td):
            x1, y1, x2, y2 = td[1]["box"]
            full = not (x1 <= 2 or y1 <= 2 or x2 >= W-2 or y2 >= H-2)
            return (1 if full else 0, (x2-x1)*(y2-y1))
        tdata = []
        for tr in tracks:
            obs = sorted(tr.obs, key=vis_qual, reverse=True)
            use = ([o for o in obs if vis_qual(o)[0] == 1] or obs)[:a.best_k]
            crs_tight, crs_save, geos, metas = [], [], [], []
            for (t, d) in use:
                if t >= len(frames):
                    continue
                bx = d["box"]; ctr = ((bx[0]+bx[2])/2, (bx[1]+bx[3])/2)
                cmpp = local_cmperpx(corridor, axis, center, ctr) or 0.2
                rail_px = 62.3/max(cmpp, 1e-6)
                ct = crop_from_box(frames[t], bx, size=224)   # 타이트(매칭=현 size.onnx 도메인)
                if a.crop == "rail":
                    cs = rail_crop(frames[t], bg, bx, v["frames"][t], rail_px, belt_bottom=belt_bot)
                elif a.crop == "tight_clean":
                    cs = rail_crop(frames[t], bg, bx, v["frames"][t], rail_px, tight=True, belt_bottom=belt_bot)
                elif a.crop == "bev":
                    cs = bev_crop(frames[t], bx, Hbev)
                else:
                    cs = ct
                if ct is None or cs is None:
                    continue
                dfeat = None
                if dep is not None:
                    if t not in dcache and t < len(frames):
                        dcache[t] = _belt_resid(dep(frames[t]), corrb, v["frames"][t], H, W)
                    rpos, sdv = dcache.get(t, (None, 1.0))
                    dfeat = _depth_box_feats(rpos, sdv, bx, corrb, H, W)
                crs_tight.append(ct); crs_save.append(cs); geos.append(geo_feat(bx, cmpp, H))
                metas.append((t, bx, cmpp, dfeat))
            if not crs_tight:
                continue
            pred = np.median(sizer(crs_tight, geos), 0)   # 매칭용 예측은 타이트 크롭으로
            traw = _traj_raw(tr.obs, corridor, axis, center)   # 트랙 전체 obs 궤적
            tdata.append({"crops": crs_save, "geos": geos, "metas": metas, "pred": pred, "traw": traw})
        if not tdata:
            continue
        srng = max(1.0, max(td["traw"][0].max() for td in tdata) - min(td["traw"][0].min() for td in tdata))

        # Hungarian 매칭: 트랙 예측 vs GT (축순열-min 비용)
        G = gmap[vid]
        C = np.array([[pair_cost(td["pred"], g) for g in G] for td in tdata])
        ri, ci = linear_sum_assignment(C)
        for i, j in zip(ri, ci):
            g = G[j]; nmatch += 1
            # margin: 차순위 GT와의 비용차(작을수록 애매한 매칭) + 차순위 GT 치수(검수 후보용)
            others = [(C[i, k], k) for k in range(len(G)) if k != j]
            margin = (min(others)[0] - C[i, j]) if others else 99.0
            alt = G[min(others)[1]] if others else None
            tfeat = _traj_feats(tdata[i]["traw"], srng)   # 트랙 궤적 20피처 (트랙 내 동일)
            for k, (cr, geo, (t, bx, cmpp, dfeat)) in enumerate(zip(tdata[i]["crops"], tdata[i]["geos"], tdata[i]["metas"])):
                nm = f"{vid}_t{i}_f{t}.jpg"; cv2.imwrite(f"{a.out}/crops/{nm}", cr)
                bw, bh = bx[2]-bx[0], bx[3]-bx[1]
                rec = {"crop": nm, "w": g[0], "d": g[1], "h": g[2], "scale": math.hypot(bw, bh)*cmpp,
                       "vid": vid, "bw_px": float(bw), "bh_px": float(bh), "rail_px": 62.3/max(cmpp, 1e-6),
                       "cy_norm": float((bx[1]+bx[3])/2/H), "match_cost": float(C[i, j]),
                       "margin": float(margin), "alt_gt": alt}
                if dfeat is not None:
                    rec["dfeat"] = dfeat
                rec["traj"] = tfeat
                out_recs.append(rec)
        print(f"[{vi+1}/{len(dp['videos'])}] {vid}: track {len(tdata)} / GT {len(G)} -> 매칭 {len(ri)}", flush=True)
    with open(f"{a.out}/labels.jsonl", "w") as f:
        for r in out_recs:
            f.write(json.dumps(r, default=float)+"\n")
    mc = np.array([r["match_cost"] for r in out_recs]); mg = np.array([r["margin"] for r in out_recs])
    print(f"\n총 트랙 {ntrack} / 매칭 {nmatch} / 크롭 {len(out_recs)} -> {a.out}")
    print(f"매칭비용(트랙예측↔배정GT) 평균 {mc.mean():.2f}cm (작을수록 확신)")
    print(f"애매도(margin): margin<2cm = {100*np.mean(mg<2):.0f}% / <1cm = {100*np.mean(mg<1):.0f}% (작을수록 애매=검수후보)")
    print("-> 애매 비율 보고 결정: 소수면 타겟 수동검수 / 다수면 EM반복 or 학습서 제외")


if __name__ == "__main__":
    main()
