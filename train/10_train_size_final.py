"""배포용 최종 CNN 학습: 전체 train, 정확히 8에폭, 선택누수 없음(에폭선택 안 함) -> ONNX.
정직한 OOF에서 ep8이 최적으로 확정됐으므로 홀드아웃 없이 전 데이터를 씀.
"""
import argparse, os, sys, json, numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from size_net2 import SizeNet2
from train_size2_enh import DS, geo_vec

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="datasets/size3_tight")
ap.add_argument("--out", required=True)
ap.add_argument("--backbone", default="resnet18")
ap.add_argument("--epochs", type=int, default=8)
ap.add_argument("--batch", type=int, default=48)
ap.add_argument("--imgsz", type=int, default=224)
ap.add_argument("--wd", type=float, default=5e-4)
ap.add_argument("--erase", type=float, default=0.4)
ap.add_argument("--distmatch", type=float, default=0.5)
ap.add_argument("--dm_tail", type=float, default=2.0)
a = ap.parse_args()

recs = [json.loads(l) for l in open(os.path.join(a.data, "labels.jsonl"))]
md = np.array([sorted([r["w"], r["d"], r["h"]], reverse=True) for r in recs]).mean(0)
GD = len(geo_vec(recs[0]))
dev = "cuda" if torch.cuda.is_available() else "cpu"
net = SizeNet2(mean_dims=tuple(md), backbone=a.backbone, geo_dim=GD).to(dev)
dl = DataLoader(DS(a.data, recs, True, a.imgsz, a.erase), batch_size=a.batch,
                shuffle=True, num_workers=4, drop_last=True)
opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=a.wd)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
lossf = nn.L1Loss()
print(f"[{a.out}] {len(recs)}크롭 전체 학습, {a.epochs}에폭, {a.backbone}", flush=True)
for ep in range(a.epochs):
    net.train(); tot = 0; n = 0
    for img, g, y in dl:
        img, g, y = img.to(dev), g.to(dev), y.to(dev)
        pred = net(img, g); loss = lossf(pred, y)
        if pred.size(0) > 4 and a.distmatch > 0:
            ps, _ = pred.sort(0, descending=True); ts = y.sort(0, descending=True)[0]
            dl_ = (ps-ts).abs()
            if a.dm_tail > 0:
                nn_ = ts.size(0)
                rw = (1.0 + torch.linspace(1., 0., nn_, device=y.device)*a.dm_tail)[:, None]
                dl_ = dl_*rw
            loss = loss + a.distmatch*dl_.mean()
        opt.zero_grad(); loss.backward(); opt.step()
        tot += float(loss)*img.size(0); n += img.size(0)
    sched.step()
    print(f"  ep{ep+1}/{a.epochs} loss={tot/n:.3f}", flush=True)
net.eval().cpu()
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
dummy = (torch.zeros(1, 3, a.imgsz, a.imgsz), torch.zeros(1, GD))
torch.onnx.export(net, dummy, a.out, opset_version=19,
                  input_names=["crop", "geo"], output_names=["dims"],
                  dynamic_axes={"crop": {0: "n"}, "geo": {0: "n"}, "dims": {0: "n"}})
print(f"저장: {a.out}\n=== DONE ===", flush=True)
