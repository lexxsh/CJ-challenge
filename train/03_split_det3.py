"""검수 완료된 train+test 라벨(explore/detds_all) -> det 학습셋.

크롭: blackbar.crop_bounds() 3단 폴백 (벨트seg+마진 -> 검은띠 -> 원본). **추론도 같은 함수 필수.**
분할: 영상 단위(프레임 중복 누수 방지) + 혼잡도 층화 + 출처(train/test) 층화.

사용:
  # 크롭본 (기본)
  CUDA_VISIBLE_DEVICES=6 python train/03_split_det3.py --out datasets/det3_crop
  # 원본본 (비교용)
  python train/03_split_det3.py --out datasets/det3_full --mode full
  # 혼합본 = 크롭 + 원본 (같은 val 영상 유지)
  python train/03_split_det3.py --out datasets/det3_mix --mode mix
"""
import os, glob, json, random, argparse, sys
import numpy as np
import cv2
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.blackbar import crop_bounds, black_bar_bounds, MARGIN

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="explore/detds_all")
ap.add_argument("--out", default="datasets/det3_crop")
ap.add_argument("--mode", default="crop", choices=["crop", "bar", "full", "mix"])
# crop=벨트seg+마진(1.78배) / bar=검은띠만 제거(1.23배, 위험0) / full=원본 그대로(1.0배) / mix=crop+full
ap.add_argument("--nval", type=int, default=30)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--margin", type=int, default=MARGIN)
a = ap.parse_args()
random.seed(a.seed)

byv = defaultdict(list)
for p in sorted(glob.glob(f"{a.src}/labels/*.txt")):
    nm = os.path.basename(p)[:-4]
    byv[nm.rsplit("_f", 1)[0]].append(nm)
vids = sorted(byv)

# ---- 크롭 경계 (영상별) ----
BND, MODE = {}, {}
if a.mode in ("crop", "bar", "mix"):
    belt = None
    if a.mode in ("crop", "mix"):
        from src.geo_infer import SegONNX
        belt = SegONNX("checkpoints/belt_seg.onnx", 960,
                       ["CUDAExecutionProvider", "CPUExecutionProvider"], conf=0.25)
    for i, v in enumerate(vids):
        ims = [cv2.imread(f"{a.src}/images/{nm}.jpg") for nm in sorted(byv[v])]
        ims = [im for im in ims if im is not None]
        if a.mode == "bar":          # 검은 띠만 제거 (벨트 seg 미사용 -> 위험 0)
            BND[v] = black_bar_bounds(ims); MODE[v] = "blackbar"
        else:
            BND[v], MODE[v] = crop_bounds(ims, belt, a.margin)
        if i % 30 == 0:
            print(f"  크롭경계 {i}/{len(vids)} ...", flush=True)
    from collections import Counter
    print(f"  크롭 모드: {dict(Counter(MODE.values()))}", flush=True)

crowd = {v: max(sum(1 for ln in open(f"{a.src}/labels/{nm}.txt") if ln.strip()) for nm in byv[v])
         for v in vids}
bin_of = lambda c: 0 if c <= 4 else 1 if c <= 7 else 2 if c <= 11 else 3
BN = ["≤4", "5-7", "8-11", "12+"]

# ---- 층화 분할 (seed 고정 -> 모든 --mode 가 같은 val 사용) ----
groups = defaultdict(list)
for v in vids:
    groups[(bin_of(crowd[v]), v.split("_")[0])].append(v)
val = []
frac = a.nval/len(vids)
for k in sorted(groups):
    g = sorted(groups[k]); random.shuffle(g)
    val += g[:int(round(len(g)*frac))]
val = set(val); train = [v for v in vids if v not in val]

for sp in ["train", "val"]:
    os.makedirs(f"{a.out}/images/{sp}", exist_ok=True)
    os.makedirs(f"{a.out}/labels/{sp}", exist_ok=True)


def emit(nm, im, b, sp, suffix=""):
    """크롭 적용 + 라벨 보정(클램프) 후 저장. 반환 박스 수."""
    H0, W0 = im.shape[:2]
    x0, y0, x1c, y1c = b
    im2 = im[y0:y1c, x0:x1c]
    H, W = im2.shape[:2]
    out = []
    for ln in open(f"{a.src}/labels/{nm}.txt"):
        t = ln.split()
        if len(t) != 5:
            continue
        cx, cy, w, h = map(float, t[1:])
        X1, X2 = (cx-w/2)*W0-x0, (cx+w/2)*W0-x0
        Y1, Y2 = (cy-h/2)*H0-y0, (cy+h/2)*H0-y0
        X1, X2 = max(0., min(W, X1)), max(0., min(W, X2))
        Y1, Y2 = max(0., min(H, Y1)), max(0., min(H, Y2))
        if X2-X1 < 3 or Y2-Y1 < 3:
            continue
        out.append(f"0 {(X1+X2)/2/W:.6f} {(Y1+Y2)/2/H:.6f} {(X2-X1)/W:.6f} {(Y2-Y1)/H:.6f}")
    cv2.imwrite(f"{a.out}/images/{sp}/{nm}{suffix}.jpg", im2, [cv2.IMWRITE_JPEG_QUALITY, 95])
    open(f"{a.out}/labels/{sp}/{nm}{suffix}.txt", "w").write("\n".join(out))
    return len(out)


cnt = {"train": [0, 0], "val": [0, 0]}
for v in vids:
    sp = "val" if v in val else "train"
    for nm in sorted(byv[v]):
        im = cv2.imread(f"{a.src}/images/{nm}.jpg")
        if im is None:
            continue
        H0, W0 = im.shape[:2]
        if a.mode in ("crop", "bar", "mix"):
            n = emit(nm, im, BND[v], sp)
            cnt[sp][0] += 1; cnt[sp][1] += n
        if a.mode in ("full", "mix"):
            sfx = "_full" if a.mode == "mix" else ""
            n = emit(nm, im, (0, 0, W0, H0), sp, sfx)
            cnt[sp][0] += 1; cnt[sp][1] += n

open(f"{a.out}/data.yaml", "w").write(
    f"path: {os.path.abspath(a.out)}\ntrain: images/train\nval: images/val\nnames:\n  0: box\n")
json.dump({"mode": a.mode, "margin": a.margin, "seed": a.seed,
           "val_videos": sorted(val), "train_videos": sorted(train),
           "bounds": {v: list(BND[v]) for v in vids} if BND else None,
           "crop_mode": MODE or None}, open(f"{a.out}/split.json", "w"), indent=1)

print(f"\n=== {a.out}  (mode={a.mode}, margin={a.margin}, seed={a.seed}) ===")
print(f"  train {len(train)}영상 / {cnt['train'][0]}장 / {cnt['train'][1]}박스")
print(f"  val   {len(val)}영상 / {cnt['val'][0]}장 / {cnt['val'][1]}박스")
if BND:
    ws = np.array([BND[v][2]-BND[v][0] for v in vids]); hs = np.array([BND[v][3]-BND[v][1] for v in vids])
    print(f"  크롭 크기: {ws.min()}x{hs.min()} ~ {ws.max()}x{hs.max()} (중앙 {np.median(ws):.0f}x{np.median(hs):.0f})")
    print(f"  면적비 중앙 {100*np.median(ws*hs)/(1280*720):.1f}% -> 확대 {np.sqrt((1280*720)/np.median(ws*hs)):.2f}배")
print(f"\n{'구간':>6s} | {'train':>12s} | {'val':>12s}")
for bi in range(4):
    t = sum(1 for v in train if bin_of(crowd[v]) == bi); u = sum(1 for v in val if bin_of(crowd[v]) == bi)
    print(f"{BN[bi]:>6s} | {t:4d} ({t/len(train)*100:4.1f}%) | {u:4d} ({u/max(len(val),1)*100:4.1f}%)")
for pre in ["train", "test"]:
    print(f"  {pre:5s}: train셋 {sum(1 for v in train if v.startswith(pre)):3d} | val셋 {sum(1 for v in val if v.startswith(pre)):2d}")
print(f"\n⚠️ 추론도 blackbar.crop_bounds() 동일 사용 필수")
