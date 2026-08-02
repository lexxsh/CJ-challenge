"""
데이터셋 빌더 (teacher=WildDet3D).
한 번의 영상 순회로:
  - 검출기 학습셋: 프레임별 이미지 + YOLO 2D박스 라벨 (teacher 박스, count용)
  - 치수회귀 학습셋: track별 크롭 + GT매칭 치수 + 레일스케일 (size용)
매칭 검수 viz -> _viz/dataset/.
사용: python build_dataset.py --n 100 --stride 5   (.venv_wd3d, GPU)
"""
import argparse, os, sys, glob, json, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import video_io as vio
from src.conveyor import corridor_mask
from src.counting import corridor_axis
from src.railscale import local_cmperpx
from src.geomap import build_rail_homography, plane_maps
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wd3d_detect import WildDet3DDetector

W_IMG, H_IMG = 1280, 720


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]); ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0., ix2-ix1), max(0., iy2-iy1); inter = iw*ih
    ua = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/ua if ua > 0 else 0.


def crop_from_box(frame, box, ctx=0.1, size=192):  # 타이트 bbox 크롭(옆 박스 distractor 최소화)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="assignment1/dataset/train")
    ap.add_argument("--gt", default="assignment1/dataset/train_label.json")
    ap.add_argument("--det_out", default="datasets/det")
    ap.add_argument("--size_out", default="datasets/size")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--score", type=float, default=0.3)
    ap.add_argument("--max_count_diff", type=int, default=4)
    ap.add_argument("--best_k", type=int, default=3)
    args = ap.parse_args()

    for sp in ("train", "val"):
        os.makedirs(f"{args.det_out}/images/{sp}", exist_ok=True)
        os.makedirs(f"{args.det_out}/labels/{sp}", exist_ok=True)
    os.makedirs(f"{args.size_out}/crops", exist_ok=True)
    os.makedirs(f"{args.size_out}/geo", exist_ok=True)
    VIZ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_viz", "dataset"); os.makedirs(VIZ, exist_ok=True)

    det = WildDet3DDetector(score=args.score)
    G = json.load(open(args.gt))
    cam = {v["video_id"]: v["camera"] for v in G["videos"]}
    gt_sizes = {v["video_id"]: sorted([(o["size_cm"]["w"], o["size_cm"]["d"], o["size_cm"]["h"])
                                       for o in v["objects"]], key=lambda t: -(t[0]*t[1]*t[2])) for v in G["videos"]}
    vids = sorted(glob.glob(os.path.join(args.videos, "*.mp4")))[: args.n]

    size_f = open(f"{args.size_out}/labels.jsonl", "w")
    n_det_img = n_det_box = n_size = used_vid = 0
    for vi, vp in enumerate(vids):
        vid = os.path.splitext(os.path.basename(vp))[0]
        gt = gt_sizes[vid]
        c = cam[vid]
        fx = c["focal_length_mm"]/c["sensor_width_mm"]*W_IMG
        fy = c["focal_length_mm"]/c["sensor_height_mm"]*H_IMG
        K = np.array([[fx, 0, W_IMG/2], [0, fy, H_IMG/2], [0, 0, 1]], np.float32)
        det.set_K(K)
        frames, total, fps = vio.read_frames(vp, stride=args.stride)
        if len(frames) < 3:
            continue
        bg = vio.background_plate(frames)
        corridor = corridor_mask(frames, bg)
        corr = cv2.dilate(corridor, np.ones((41, 41), np.uint8))
        axis, center = corridor_axis(corridor)
        Hrail, _, _ = build_rail_homography(corridor)   # 영상 1회 호모그래피(레일 평면)
        split = "val" if (vi % 8 == 0) else "train"

        # 1) 추적만 먼저 (라벨은 나중에 keep 후)
        bgi = bg.astype(np.int16)
        trks = []
        for t, fr in enumerate(frames):
            fg = np.abs(fr.astype(np.int16) - bgi).sum(2) > 40   # 배경차분 전경
            kept = []
            for d in det(fr):
                x1, y1, x2, y2 = [int(v) for v in d["box"]]
                cx = min(max((x1+x2)//2, 0), W_IMG-1); cy = min(max((y1+y2)//2, 0), H_IMG-1)
                if corr[cy, cx] == 0:
                    continue
                # 박스 영역 전경 비율 — 빈 벨트 FP 제거
                xa, ya = max(0, x1), max(0, y1); xb, yb = min(W_IMG, x2), min(H_IMG, y2)
                if xb <= xa or yb <= ya:
                    continue
                if fg[ya:yb, xa:xb].mean() < 0.15:
                    continue
                kept.append(d)
            used = set()
            for tr in trks:
                best, bi = 0.2, -1
                for j, d in enumerate(kept):
                    if j in used:
                        continue
                    v = iou(tr["box"], d["box"])
                    if v > best:
                        best, bi = v, j
                if bi >= 0:
                    tr["box"] = kept[bi]["box"]; tr["obs"].append((t, kept[bi])); tr["hits"] += 1; used.add(bi)
            for j, d in enumerate(kept):
                if j not in used:
                    trks.append({"box": d["box"], "obs": [(t, d)], "hits": 1})
        tracks = [t for t in trks if t["hits"] >= 3]

        # 2) C-WSL: 알려진 GT개수로 정리 — 품질(hits×평균score) 순 top-GT개수만 keep
        def qual(tr):
            return tr["hits"] * np.mean([d["score"] for (_, d) in tr["obs"]])
        tracks.sort(key=qual, reverse=True)
        tracks = tracks[: len(gt)]   # 과대카운트 트림 (FP/단편 제거)

        # 3) 검출기 라벨: keep된 track의 관측만 (프레임 단위로 모아 저장)
        from collections import defaultdict
        frame_boxes = defaultdict(list)
        for tr in tracks:
            for (t, d) in tr["obs"]:
                frame_boxes[t].append(d["box"])
        for t, boxes in frame_boxes.items():
            stem = f"{vid}_f{t:03d}"
            cv2.imwrite(f"{args.det_out}/images/{split}/{stem}.jpg", frames[t])
            with open(f"{args.det_out}/labels/{split}/{stem}.txt", "w") as f:
                for bx in boxes:
                    cxn = (bx[0]+bx[2])/2/W_IMG; cyn = (bx[1]+bx[3])/2/H_IMG
                    wn = (bx[2]-bx[0])/W_IMG; hn = (bx[3]-bx[1])/H_IMG
                    f.write(f"0 {cxn:.5f} {cyn:.5f} {wn:.5f} {hn:.5f}\n")
            n_det_img += 1; n_det_box += len(boxes)

        if False:
            pass
        else:
            # track별 정렬키(WildDet3D 치수 부피 median) + best 크롭 + 스케일
            def vis_qual(td):
                # 풀뷰(경계 안닿음) 우선, 그다음 면적 큰 순. 잘린 프레임 배제용.
                x1, y1, x2, y2 = td[1]["box"]
                full = not (x1 <= 2 or y1 <= 2 or x2 >= W_IMG-2 or y2 >= H_IMG-2)
                area = (x2-x1)*(y2-y1)
                return (1 if full else 0, area)
            recs = []
            for tr in tracks:
                obs = sorted(tr["obs"], key=vis_qual, reverse=True)
                # 경계에 안 닿은 풀뷰만; 없으면 전체에서 fallback
                full_obs = [o for o in obs if vis_qual(o)[0] == 1]
                use_obs = (full_obs or obs)[: args.best_k]
                vols = [d["dims3d"][0]*d["dims3d"][1]*d["dims3d"][2] for (_, d) in tr["obs"]]
                recs.append({"order": float(np.median(vols)), "obs": use_obs})
            recs.sort(key=lambda r: -r["order"])
            m = min(len(recs), len(gt))
            for i in range(m):
                w, d_, h = gt[i]
                for (t, dd) in recs[i]["obs"]:
                    crop = crop_from_box(frames[t], dd["box"])
                    if crop is None:
                        continue
                    bx = dd["box"]; ctr = ((bx[0]+bx[2])/2, (bx[1]+bx[3])/2)
                    cmpp = local_cmperpx(corridor, axis, center, ctr) or 0.2
                    bw_px = float(bx[2]-bx[0]); bh_px = float(bx[3]-bx[1])
                    diag = np.hypot(bw_px, bh_px)
                    rail_px = 62.3 / cmpp  # 박스 위치의 레일 픽셀폭
                    name = f"{vid}_t{i}_f{t}.jpg"
                    cv2.imwrite(f"{args.size_out}/crops/{name}", crop)
                    # 기하맵: 크롭 영역(타이트박스 ctx=0.1 동일)을 평면좌표/cm-px로 -> 크롭 크기로 리샘플 저장
                    geo_saved = False
                    if Hrail is not None:
                        cx0, cy0 = ctr; half = max(bw_px, bh_px)*(1+0.1)/2
                        reg = (cx0-half, cy0-half, cx0+half, cy0+half)
                        gm = plane_maps(Hrail, reg, (H_IMG, W_IMG))
                        if gm is not None:
                            S = crop.shape[0]
                            X = cv2.resize(gm["X"], (S, S)); Z = cv2.resize(gm["Z"], (S, S)); SC = cv2.resize(gm["scale"], (S, S))
                            np.savez_compressed(f"{args.size_out}/geo/{name}.npz",
                                                X=X.astype(np.float32), Z=Z.astype(np.float32), scale=SC.astype(np.float32))
                            geo_saved = True
                    size_f.write(json.dumps({"crop": name, "w": w, "d": d_, "h": h,
                                             "scale": float(diag*cmpp), "vid": vid,
                                             "bw_px": bw_px, "bh_px": bh_px, "rail_px": float(rail_px),
                                             "cy_norm": float(ctr[1]/H_IMG), "geo": geo_saved}) + "\n")
                    n_size += 1
            used_vid += 1
            # 검수 viz(첫 12영상): rank순 크롭+부여GT
            if vi < 12:
                cell, cols = 150, 6; mm = max(len(recs), len(gt)); rows = (mm+cols-1)//cols
                canvas = np.full((rows*cell, cols*cell, 3), 40, np.uint8)
                for i in range(mm):
                    r2, c2 = i//cols, i % cols; y0, x0 = r2*cell, c2*cell
                    if i < len(recs):
                        t0, dd0 = recs[i]["obs"][0]; cr = crop_from_box(frames[t0], dd0["box"], size=128)
                        if cr is not None:
                            canvas[y0:y0+128, x0:x0+128] = cr
                    g = f"GT {gt[i][0]:.0f}x{gt[i][1]:.0f}x{gt[i][2]:.0f}" if i < len(gt) else "GT-"
                    cv2.putText(canvas, f"#{i} {g}", (x0+2, y0+140), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 255, 255), 1)
                cv2.imwrite(f"{VIZ}/{vid}.jpg", canvas)
        print(f"[{vi+1}/{len(vids)}] {vid}: det_trk={len(tracks)} gt={len(gt)} | det_img={n_det_img} size={n_size}", flush=True)

    size_f.close()
    with open(f"{args.det_out}/data.yaml", "w") as f:
        f.write(f"path: {os.path.abspath(args.det_out)}\ntrain: images/train\nval: images/val\nnames:\n  0: box\n")
    print(f"\n완료: 검출 이미지={n_det_img}(박스 {n_det_box}) | 치수 크롭={n_size} | size매칭 영상={used_vid}")
    print(f"-> {args.det_out}/ , {args.size_out}/ , 검수 {VIZ}/")


if __name__ == "__main__":
    main()
