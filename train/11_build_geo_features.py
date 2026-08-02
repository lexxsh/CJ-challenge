"""학습 피처를 '추론과 동일한 코드 경로'(geo_infer.GeoSizer.feats)로 재생성 -> train/serve skew=0.
이전 build_features2는 ultralytics(rect 736x1280) 경로라 ONNX(square letterbox) 추론과 불일치했음.
출력: features3.json  (VDUMP 바꾸면 OUT 도 같이 바꿀 것)

★ 크롭 누수 (2026-07-16): 기본 vdump 는 **그 영상을 학습한 검출기**로 뜬 것 -> 기하 피처
  (box_px/rail_px/z_box)도 배포 때보다 정확 -> OOF 낙관. CNN 크롭과 **같은 누수를 공유**함.
  누수 없는 피처: VDUMP=explore/vdump_oof.json (pipeline/dump_vdump_oof.py 로 생성)

사용: VDUMP=explore/vdump_oof.json OUT=features3_oof.json python train/11_build_geo_features.py
"""
import json, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import video_io as vio
from src.geo_infer import GeoSizer
from collections import defaultdict

NVID = int(os.environ.get("NVID", "100"))
VDUMP = os.environ.get("VDUMP", "explore/vdump_rfdetr_all.json")
OUT = os.environ.get("OUT", "features3.json")
# ★ 줌: 검출기에서 full->belt-crop 줌이 -1.16 을 만든 그 메커니즘을 기하스택에 적용
ZOOM_SEG = os.environ.get("ZOOM_SEG", "0") == "1"
ZOOM_DEPTH = os.environ.get("ZOOM_DEPTH", "0") == "1"
PROV = ["CUDAExecutionProvider", "CPUExecutionProvider"]
GS = GeoSizer("checkpoints/seg_box.onnx", "checkpoints/belt_seg.onnx",
              "checkpoints/da3_metric_large.onnx",
              [f"checkpoints/rf_{n}.onnx" for n in ["long", "mid", "short"]], PROV)
print(f"[피처빌드] vdump={VDUMP} -> {OUT} | zoom_seg={ZOOM_SEG} zoom_depth={ZOOM_DEPTH}", flush=True)
VM = json.load(open(VDUMP))["videos"]
FEATS = ["maj_cm", "min_cm", "bw_cm", "bh_cm", "area_cm2", "cy_norm",
         "ratio", "maj_x_ratio", "min_x_ratio", "zspan", "zbox"]


def find_bbox(vid, fi, bw, bh, cy_norm, H):
    if fi >= len(VM[vid]["frames"]):
        return None
    cyt = cy_norm*H; best = None; bd = 1e9
    for b in VM[vid]["frames"][fi]:
        w, h = b[2]-b[0], b[3]-b[1]; cy = (b[1]+b[3])/2
        dd = abs(w-bw)+abs(h-bh)+abs(cy-cyt)*0.5
        if dd < bd:
            bd = dd; best = b
    return best[:4] if best and bd < 60 else None


recs = [json.loads(l) for l in open("datasets/size3_tight/labels.jsonl")]
bytk = defaultdict(list)
for r in recs:
    bytk[r["crop"].rsplit("_f", 1)[0]].append(r)
reps = [max(rs, key=lambda r: r["bw_px"]) for rs in bytk.values()]
vids = sorted({r["vid"] for r in reps})[:NVID]
byv = defaultdict(list)
for r in reps:
    if r["vid"] in set(vids):
        byv[r["vid"]].append(r)
print(f"트랙 {sum(len(v) for v in byv.values())} / {len(vids)}영상", flush=True)

rows = []
for vi, vid in enumerate(vids):
    try:
        frames, _, _ = vio.read_frames(f"assignment1/dataset/train/{vid}.mp4", stride=3)
    except Exception:
        continue
    byframe = defaultdict(list)
    for r in byv[vid]:
        byframe[int(r["crop"].rsplit("_f", 1)[1].split(".")[0])].append(r)
    cbnd = None
    if ZOOM_SEG or ZOOM_DEPTH:
        from src.blackbar import crop_bounds
        cbnd = crop_bounds(frames, GS.belt)[0]      # 검출기 crop 모드와 동일한 벨트 경계
    for fi, rs in byframe.items():
        if fi >= len(frames):
            continue
        fr = frames[fi]
        ctx = GS.frame_ctx(fr, crop=cbnd, zoom_seg=ZOOM_SEG, zoom_depth=ZOOM_DEPTH)   # ★ 추론과 동일
        if ctx is None:
            continue
        H = fr.shape[0]
        for r in rs:
            bb = find_bbox(vid, fi, r["bw_px"], r["bh_px"], r["cy_norm"], H)
            if bb is None:
                continue
            f = GS.feats(ctx, [float(v) for v in bb])   # ★ 추론과 동일
            if f is None:
                continue
            rows.append({"vid": vid, "w": r["w"], "d": r["d"], "h": r["h"],
                         "bw_px_key": r["bw_px"],
                         **{k: float(v) for k, v in zip(FEATS, f)}})
    if vi % 10 == 0:
        print(f"  {vi}/{len(vids)}, {len(rows)}박스", flush=True)

json.dump(rows, open(OUT, "w"))
Y = np.array([sorted([r["w"], r["d"], r["h"]], reverse=True) for r in rows])
vol = Y.prod(1)
print(f"\n완료 {len(rows)}박스 -> features3.json")
for k in ["maj_cm", "ratio", "zbox"]:
    v = np.array([r[k] for r in rows])
    print(f"  {k:8s} GT부피 상관 r={np.corrcoef(v, vol)[0,1]:+.3f}")
print("  [features2(ultralytics경로) 대비] maj_cm r=+0.763, ratio r=-0.397")
print("=== DONE ===")
