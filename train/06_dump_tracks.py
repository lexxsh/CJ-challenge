"""3종 median 검출기로 train 100영상 vdump 생성 — size 학습 데이터를 **추론과 동일한 박스**로.

왜: 리더보드 12.097 -> 10.937 은 검출기 교체(RF-DETR 3종 median)로 얻음. 그런데 size CNN/기하는
  여전히 **옛 배포본 검출기 박스**로 만든 크롭에 학습돼 있음 -> train/serve 왜곡 + 개선된 박스를 못 받아먹음.
  이 덤프로 크롭·기하피처를 다시 만들어 size 를 새 검출기에 맞춘다.

로직은 pipeline3.process_video 의 검출부와 동일:
  각 모델(crop/bar/full)을 **학습과 동일한 크롭 규칙**으로 추론 -> 전체프레임 좌표 환산 -> 추적 ->
  영상별 트랙수의 중앙값에 해당하는 모델을 골라 **그 모델의 프레임별 박스**를 출력.

포맷: {stride, videos:{vid:{wh:[W,H], frames:[[[x1,y1,x2,y2,score],...],...]}}}  (build_size_ds 입력)
환경: .venv_train (ONNX CUDA). ★ source scripts/gpu_env.sh 필수

사용(GPU 3장 분산):
  CUDA_VISIBLE_DEVICES=1 python train/06_dump_tracks.py --shard 0 --nshard 3 &
  CUDA_VISIBLE_DEVICES=2 python train/06_dump_tracks.py --shard 1 --nshard 3 &
  CUDA_VISIBLE_DEVICES=3 python train/06_dump_tracks.py --shard 2 --nshard 3 &
  python train/06_dump_tracks.py --merge
"""
import argparse, glob, json, os, sys
import numpy as np
import cv2
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, R)
from src import video_io as vio
from src.conveyor import corridor_mask
from src.ocsort import OCSort, KalmanBoxTracker
from src.blackbar import crop_bounds, black_bar_bounds

ap = argparse.ArgumentParser()
ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1)
ap.add_argument("--conf", type=float, default=0.65)      # 배포와 동일
ap.add_argument("--resolution", type=int, default=1024)
ap.add_argument("--stride", type=int, default=3)
ap.add_argument("--out", default="explore/vdump_med.json")
ap.add_argument("--merge", action="store_true")
a = ap.parse_args()
PART = f"{a.out}.part{a.shard}"

if a.merge:
    out = {"stride": a.stride, "videos": {}}
    for p in sorted(glob.glob(f"{a.out}.part*")):
        out["videos"].update(json.load(open(p))["videos"])
    json.dump(out, open(a.out, "w"))
    print(f"[merge] {len(out['videos'])}영상 -> {a.out}")
    sys.exit()

from src.detector_rfdetr import RfdetrDet
from src.geo_infer import SegONNX
PROV = ["CUDAExecutionProvider", "CPUExecutionProvider"]
DETS = [(ds, RfdetrDet(f"{R}/runs_final/onnx_{ds}/rfdetr-medium.onnx", conf=a.conf,
                       resolution=a.resolution)) for ds in ["crop", "bar", "full"]]
BELT = SegONNX(f"{R}/checkpoints/belt_seg.onnx", 960, PROV, conf=0.25)


def bounds_for(ds, frames):
    H, W = frames[0].shape[:2]
    if ds == "crop":
        return crop_bounds(frames, BELT)[0]
    if ds == "bar":
        return black_bar_bounds(frames)
    return (0, 0, W, H)


def detect_track(frames, det, ds, corr, W, H):
    x0, y0, x1, y1 = bounds_for(ds, frames)
    fa = W*H
    KalmanBoxTracker.count = 0
    trk = OCSort(iou_thr=0.15, max_age=20, min_hits=2, det_thresh=0.0, byte_iou=0.2)
    fr_boxes = []
    for t, fr in enumerate(frames):
        dets = []
        for d in det(fr[y0:y1, x0:x1]):
            b1, c1, b2, c2 = [float(v) for v in d["box"]]
            b1 += x0; b2 += x0; c1 += y0; c2 += y0        # 크롭 -> 전체프레임
            if (b2-b1)*(c2-c1) > 0.33*fa:
                continue
            cx = min(max(int((b1+b2)//2), 0), W-1); cy = min(max(int((c1+c2)//2), 0), H-1)
            if corr[cy, cx] == 0:
                continue
            dets.append({"box": [b1, c1, b2, c2], "score": float(d.get("score", 1.0))})
        fr_boxes.append([[*d["box"], d["score"]] for d in dets])
        trk.step(dets, t)
    return len(trk.result()), fr_boxes


vids = sorted(glob.glob(f"{R}/assignment1/dataset/train/*.mp4"))
vids = [v for i, v in enumerate(vids) if i % a.nshard == a.shard]
out = json.load(open(PART)) if os.path.exists(PART) else {"stride": a.stride, "videos": {}}
print(f"[shard {a.shard}/{a.nshard}] {len(vids)}영상 (완료 {len(out['videos'])})", flush=True)

for i, vp in enumerate(vids):
    vid = os.path.splitext(os.path.basename(vp))[0]
    if vid in out["videos"]:
        continue
    frames, _, _ = vio.read_frames(vp, stride=a.stride)
    bg = vio.background_plate(frames)
    corr = cv2.dilate(corridor_mask(frames, bg), np.ones((61, 61), np.uint8))
    H, W = frames[0].shape[:2]
    cands = [(ds, *detect_track(frames, det, ds, corr, W, H)) for ds, det in DETS]
    counts = [c[1] for c in cands]
    med = int(np.median(counts))
    pick = min(range(len(cands)), key=lambda k: (abs(counts[k]-med), k))   # 배포와 동일한 채택 규칙
    out["videos"][vid] = {"wh": [W, H], "frames": cands[pick][2]}
    if i % 5 == 0:
        json.dump(out, open(PART, "w"))
        print(f"  {i+1}/{len(vids)} {vid}: {dict((c[0], c[1]) for c in cands)} -> {cands[pick][0]}", flush=True)
json.dump(out, open(PART, "w"))
print(f"[shard {a.shard}] 완료 {len(out['videos'])}영상 -> {PART}", flush=True)
