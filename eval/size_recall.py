"""크기 구간별 recall 측정 — 작은 박스 손실가중의 판정자.

count 로 판정하면 노이즈가 커서(90영상×0/±1) 개념 검증이 안 됨.
대신 fold val 이미지에서 GT 박스를 크기별로 나눠 검출률을 직접 잼.

기준선 (RF-DETR conf0.7 / YOLO26m ep36 conf0.7, fold1·2·3):
  크기      GT수   RF-DETR  YOLO26m
  0-50      661     66.0%    65.8%   <- 여기가 오르면 성공
  50-70     998     87.5%    85.3%
  70-90    1025     94.6%    88.7%
  90-120   1157     95.9%    93.3%
  120-160   917     98.3%    96.5%
  160+      689     99.1%    96.7%
  전체              91.3%    88.6%

사용: python pipeline/size_recall.py --weights runs/detect/runs_smallw/f1_a1_m4/weights/last.pt --fold 1
"""
import argparse, os, glob
import numpy as np
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True)
ap.add_argument("--fold", type=int, default=1)
ap.add_argument("--ds", default="fold")
ap.add_argument("--conf", type=float, default=0.7)
ap.add_argument("--imgsz", type=int, default=1280)
ap.add_argument("--device", default="0")
a = ap.parse_args()


def iou(g, d):
    ix1, iy1 = max(g[0], d[0]), max(g[1], d[1])
    ix2, iy2 = min(g[2], d[2]), min(g[3], d[3])
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw*ih
    ua = (g[2]-g[0])*(g[3]-g[1]) + (d[2]-d[0])*(d[3]-d[1]) - inter
    return inter/ua if ua > 0 else 0


paths = [x for x in open(f"datasets/{a.ds}{a.fold}/val.txt").read().split("\n") if x]
from ultralytics import YOLO
m = YOLO(a.weights)
rows = []
for i, ip in enumerate(paths):
    lp = ip.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
    if not os.path.exists(lp):
        continue
    im = cv2.imread(ip)
    if im is None:
        continue
    H, W = im.shape[:2]
    gbs = []
    for ln in open(lp):
        t = ln.split()
        if len(t) != 5:
            continue
        cx, cy, w, h = map(float, t[1:])
        gbs.append([(cx-w/2)*W, (cy-h/2)*H, (cx+w/2)*W, (cy+h/2)*H])
    if not gbs:
        continue
    r = m.predict(im, imgsz=a.imgsz, conf=a.conf, device=a.device, verbose=False)[0]
    dets = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
    for g in gbs:
        s = np.sqrt((g[2]-g[0])*(g[3]-g[1]))
        rows.append((s, any(iou(g, d) > 0.5 for d in dets)))
    if i % 100 == 0:
        print(f"  {i}/{len(paths)}", flush=True)

A = np.array(rows)
print(f"\n=== {os.path.basename(os.path.dirname(os.path.dirname(a.weights)))} "
      f"(fold{a.fold} val, conf{a.conf}) — GT {len(A)}개 ===")
print(f"{'크기':>10s} {'GT수':>6s} {'recall':>8s}")
for lo, hi in [(0, 50), (50, 70), (70, 90), (90, 120), (120, 160), (160, 999)]:
    msk = (A[:, 0] >= lo) & (A[:, 0] < hi)
    if msk.sum() < 5:
        continue
    print(f"{f'{lo}-{hi}':>10s} {int(msk.sum()):6d} {A[msk, 1].mean()*100:7.1f}%")
print(f"{'전체':>10s} {len(A):6d} {A[:, 1].mean()*100:7.1f}%")
