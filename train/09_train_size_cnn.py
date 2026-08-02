"""향상 트릭 실험 트레이너: EMA / MixUp / Random Erasing / SmoothL1 토글.
baseline(train_size2)와 동일 구조, 트릭만 추가. held-out val MAE 비교용.
"""
import argparse, os, sys, json, math, copy, numpy as np, cv2
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from size_net2 import SizeNet2, GEO_DIM


_REL = False
def geo_vec(r):
    rail = max(r.get("rail_px", 1.0), 1.0); bw, bh = r.get("bw_px", 0.0), r.get("bh_px", 0.0)
    v = [bw/rail, bh/rail, math.hypot(bw, bh)/rail, math.log(r.get("scale", 1.0)+1.0), r.get("cy_norm", 0.5)]
    if "rel_area" in r: v.append(r["rel_area"])  # 데이터에 있으면 자동
    if "focal" in r: v.append(r["focal"])          # 카메라 focal (oracle/예측)
    return v


class DS(Dataset):
    TOPMASK = False
    OCC = 0.0
    def __init__(self, root, recs, train=True, imgsz=224, erase=0.0):
        self.root, self.recs, self.train, self.imgsz, self.erase = root, recs, train, imgsz, erase
    def __len__(self): return len(self.recs)
    def __getitem__(self, i):
        r = self.recs[i]
        img = cv2.imread(os.path.join(self.root, "crops", r["crop"]))[:, :, ::-1]
        if img.shape[0] != self.imgsz: img = cv2.resize(img, (self.imgsz, self.imgsz))
        img = img.astype(np.float32)/255.0; g = geo_vec(r)
        msk = None
        if DS.TOPMASK:
            mp = f"datasets/size3_topmask/masks/{r['crop']}"
            m = cv2.imread(mp, 0)
            msk = (cv2.resize(m, (self.imgsz, self.imgsz)).astype(np.float32)/255.0) if m is not None else np.zeros((self.imgsz, self.imgsz), np.float32)
        if self.train:
            if np.random.rand() < 0.5:
                img = img[:, ::-1].copy()
                if msk is not None: msk = msk[:, ::-1].copy()
            ang = np.random.uniform(-15, 15); h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), ang, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderValue=(0.447, 0.447, 0.447))
            if msk is not None: msk = cv2.warpAffine(msk, M, (w, h), borderValue=0.0)   # 마스크 동기 회전
            img = np.clip(img*np.random.uniform(0.7, 1.3)+np.random.uniform(-0.08, 0.08), 0, 1)
            jit = np.random.uniform(0.9, 1.1)
            g = [g[0]*jit, g[1]*jit, g[2]*jit, g[3]+math.log(jit), g[4]] + list(g[5:])
            if self.erase > 0 and np.random.rand() < self.erase:   # Random Erasing (가림, RGB만)
                eh, ew = np.random.randint(h//8, h//3), np.random.randint(w//8, w//3)
                y0, x0 = np.random.randint(0, h-eh), np.random.randint(0, w-ew)
                img[y0:y0+eh, x0:x0+ew] = 0.447
            if DS.OCC > 0 and np.random.rand() < DS.OCC:   # occlusion-paste: 다른 크롭 일부를 가장자리에서 겹침(실제 이웃가림)
                oc = cv2.imread(os.path.join(self.root, "crops", self.recs[np.random.randint(len(self.recs))]["crop"]))[:, :, ::-1].astype(np.float32)/255.0
                oc = cv2.resize(oc, (w, h)); ph = np.random.randint(h//4, h//2)
                if np.random.rand() < 0.5: img[:ph] = oc[h-ph:]      # 위 가림
                else: img[h-ph:] = oc[:ph]                            # 아래 가림
        img = (img-np.array([0.485, 0.456, 0.406]))/np.array([0.229, 0.224, 0.225])
        img = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32)
        if DS.TOPMASK:
            img = torch.cat([img, torch.tensor(msk, dtype=torch.float32)[None]], 0)   # 4번째 채널=윗면마스크
        if getattr(DS,"NO_GEO",False): g=[0.0]*len(g)
        dims = sorted([r["w"], r["d"], r["h"]], reverse=True)
        return img, torch.tensor(g, dtype=torch.float32), torch.tensor(dims, dtype=torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/size3_tight"); ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=48); ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--out", default="")
    ap.add_argument("--ema", type=float, default=0.0)       # EMA decay (0=끄기, 0.999 추천)
    ap.add_argument("--mixup", type=float, default=0.0)     # MixUp alpha (0=끄기, 0.2 추천)
    ap.add_argument("--erase", type=float, default=0.0)     # Random Erasing prob
    ap.add_argument("--smoothl1", action="store_true")     # L1 -> SmoothL1(Huber)
    ap.add_argument("--wd", type=float, default=5e-4)      # weight decay
    ap.add_argument("--dropout", type=float, default=-1)   # head dropout override (-1=기본0.4)
    ap.add_argument("--cutmix", type=float, default=0.0)   # CutMix alpha
    ap.add_argument("--swa", action="store_true")          # 마지막 25% epoch 가중치평균
    ap.add_argument("--varmatch", type=float, default=0.0)  # 분산보존: 배치 pred std가 target std 따르게 (de-shrinkage)
    ap.add_argument("--distmatch", type=float, default=0.0)  # 분포매칭: 정렬-pred가 정렬-target 따르게 (tail de-shrink)
    ap.add_argument("--dm_tail", type=float, default=0.0)  # distmatch를 큰박스에 가중(tail집중, 값=가중지수)
    ap.add_argument("--dm_ax", default="")  # 축별 dm 가중 "long,mid,short" 예 "2.0,1.0,0.7"
    ap.add_argument("--ratiomatch", type=float, default=0.0)  # 형상비율 분포매칭 (aspect 압축 교정)
    ap.add_argument("--shapeloss", type=float, default=0.0)  # per-sample log비율 L1 (형상 직접 페널티)
    ap.add_argument("--varmatch_ax", default="")            # 축별 varmatch "0.2,0.4,0.6" (long,mid,short)
    ap.add_argument("--sizeweight", type=float, default=0.0) # 손실을 (vol/meanvol)^p 로 가중 (큰박스 upweight)
    ap.add_argument("--backbone", default="resnet18")
    ap.add_argument("--topmask", action="store_true")
    ap.add_argument("--occ", type=float, default=0.0)  # occlusion-paste 확률(이웃박스 크롭 붙임)
    ap.add_argument("--relfeat", action="store_true")  # 상대크기(인터박스) feature 추가
    ap.add_argument("--labels", default="")  # 라벨파일 override (재매칭 labels_v2 등)
    ap.add_argument("--no_geo", action="store_true")  # geo 특징 0으로 (중요도 측정)       # deep ensemble 다양성용 백본
    ap.add_argument("--volq_lo", type=float, default=0.0)   # train 볼륨 하한 percentile (큰박스 전문가=0.5)
    ap.add_argument("--volq_hi", type=float, default=1.0)   # train 볼륨 상한 percentile (작은박스 전문가=0.5)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    DS.TOPMASK = args.topmask
    DS.NO_GEO = args.no_geo
    DS.OCC = args.occ
    global _REL; _REL = args.relfeat

    recs = [json.loads(l) for l in open(os.path.join(args.data, args.labels or ("labels_rel.jsonl" if args.relfeat else "labels.jsonl")))]
    GD = len(geo_vec(recs[0]))   # geo 차원 자동 (rel_area/focal 포함시 +)
    vids = sorted({r["vid"] for r in recs}); val_vids = set(vids[::8])
    tr = [r for r in recs if r["vid"] not in val_vids]; va = [r for r in recs if r["vid"] in val_vids]
    if args.volq_lo > 0 or args.volq_hi < 1:   # 볼륨 percentile로 train 필터 (전문가 모델)
        tv = np.array([r["w"]*r["d"]*r["h"] for r in tr]); lo, hi = np.quantile(tv, [args.volq_lo, args.volq_hi])
        tr = [r for r in tr if lo <= r["w"]*r["d"]*r["h"] <= hi]
        print(f"볼륨필터 [{args.volq_lo},{args.volq_hi}] -> train {len(tr)}")
    md = np.array([sorted([r["w"], r["d"], r["h"]], reverse=True) for r in tr]).mean(0)
    dev = args.device if torch.cuda.is_available() else "cpu"
    net = SizeNet2(mean_dims=tuple(md), backbone=args.backbone, geo_dim=GD).to(dev)
    if args.topmask:   # resnet conv1 3->4 채널 (RGB 가중 복사 + 4번째=0)
        old = net.backbone.conv1
        new = nn.Conv2d(4, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False).to(dev)
        with torch.no_grad():
            new.weight[:, :3] = old.weight; new.weight[:, 3:] = 0
        net.backbone.conv1 = new
    if args.dropout >= 0:
        for m in net.head.modules():
            if isinstance(m, nn.Dropout): m.p = args.dropout
    ema = copy.deepcopy(net) if args.ema > 0 else None
    swa_sd = None; swa_n = 0
    dl_tr = DataLoader(DS(args.data, tr, True, args.imgsz, args.erase), batch_size=args.batch, shuffle=True, num_workers=4, drop_last=True)
    dl_va = DataLoader(DS(args.data, va, False, args.imgsz), batch_size=args.batch, num_workers=4)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    lossf = nn.SmoothL1Loss() if args.smoothl1 else nn.L1Loss()
    best = 1e9; best_sd = None; best_std = (0, 0); best_big = 0.0
    for ep in range(args.epochs):
        net.train()
        for img, g, y in dl_tr:
            img, g, y = img.to(dev), g.to(dev), y.to(dev)
            if args.mixup > 0:
                lam = np.random.beta(args.mixup, args.mixup); idx = torch.randperm(img.size(0), device=dev)
                img = lam*img + (1-lam)*img[idx]; g = lam*g + (1-lam)*g[idx]
                pred = net(img, g); loss = lam*lossf(pred, y) + (1-lam)*lossf(pred, y[idx])
            elif args.cutmix > 0:
                lam = np.random.beta(args.cutmix, args.cutmix); idx = torch.randperm(img.size(0), device=dev)
                H, W = img.shape[2:]; rh, rw = int(H*math.sqrt(1-lam)), int(W*math.sqrt(1-lam))
                cy, cx = np.random.randint(H), np.random.randint(W)
                y0, y1 = max(cy-rh//2, 0), min(cy+rh//2, H); x0, x1 = max(cx-rw//2, 0), min(cx+rw//2, W)
                img[:, :, y0:y1, x0:x1] = img[idx, :, y0:y1, x0:x1]
                lam2 = 1-((y1-y0)*(x1-x0)/(H*W)); g = lam2*g + (1-lam2)*g[idx]
                pred = net(img, g); loss = lam2*lossf(pred, y) + (1-lam2)*lossf(pred, y[idx])
            else:
                pred = net(img, g)
                if args.sizeweight > 0:   # 큰박스 손실 가중 (regression-to-mean 완화)
                    vol = y.prod(1); w_ = (vol / vol.mean()).clamp(0.3, 3.0) ** args.sizeweight
                    loss = ((pred - y).abs().mean(1) * w_).mean()
                else:
                    loss = lossf(pred, y)
                if pred.size(0) > 4:   # 분산보존 (전역 or 축별)
                    if args.varmatch_ax:
                        va = torch.tensor([float(x) for x in args.varmatch_ax.split(",")], device=dev)
                        loss = loss + (va * (pred.std(0) - y.std(0)).abs()).mean()
                    elif args.varmatch > 0:
                        loss = loss + args.varmatch * (pred.std(0) - y.std(0)).abs().mean()
                    if args.distmatch > 0:   # 분포매칭: 배치 내 축별 순서통계(정렬) 매칭 -> tail de-shrink
                        ps, pi = pred.sort(0, descending=True); ts = y.sort(0, descending=True)[0]
                        dloss = (ps - ts).abs()
                        if args.dm_tail > 0:   # 큰박스(상위순위)에 가중 -> tail만 de-shrink
                            n = ts.size(0); rankw = (torch.linspace(1.0, 0.0, n, device=ts.device)**0 + (torch.linspace(1.0,0.0,n,device=ts.device)*args.dm_tail))[:,None]
                            dloss = dloss * rankw
                        if args.dm_ax:
                            axw = torch.tensor([float(x) for x in args.dm_ax.split(",")], device=dloss.device)
                            dloss = dloss * axw[None, :]   # 축별 가중 (long,mid,short)
                        loss = loss + args.distmatch * dloss.mean()
                    if args.shapeloss > 0:   # 박스별 형상(log비율) 직접 페널티
                        rp = pred.sort(1, descending=True)[0].clamp(min=1.0).log()
                        rt = y.clamp(min=1.0).log()
                        loss = loss + args.shapeloss * ((rp[:,0]-rp[:,2])-(rt[:,0]-rt[:,2])).abs().mean()
                    if args.ratiomatch > 0:   # 형상(비율) 분포매칭: log비율 순서통계 매칭 -> aspect 압축 교정
                        rp = pred.sort(1, descending=True)[0].clamp(min=1.0).log()
                        rt = y.clamp(min=1.0).log()
                        pr = torch.stack([rp[:,0]-rp[:,2], rp[:,0]-rp[:,1], rp[:,1]-rp[:,2]],1)
                        tr_ = torch.stack([rt[:,0]-rt[:,2], rt[:,0]-rt[:,1], rt[:,1]-rt[:,2]],1)
                        loss = loss + args.ratiomatch * (pr.sort(0)[0] - tr_.sort(0)[0]).abs().mean()
            opt.zero_grad(); loss.backward(); opt.step()
            if ema is not None:
                with torch.no_grad():
                    for e, p in zip(ema.parameters(), net.parameters()): e.mul_(args.ema).add_(p, alpha=1-args.ema)
                    for e, p in zip(ema.buffers(), net.buffers()): e.copy_(p)
        sched.step()
        if args.swa and ep >= int(args.epochs*0.75):   # 마지막 25% 가중치 누적평균
            sd = net.state_dict()
            if swa_sd is None: swa_sd = {k: v.cpu().clone().float() for k, v in sd.items()}
            else:
                for k in swa_sd: swa_sd[k].add_(sd[k].cpu().float())
            swa_n += 1
        evalnet = ema if ema is not None else net
        evalnet.eval(); errs = []
        with torch.no_grad():
            for img, g, y in dl_va:
                errs.append((evalnet(img.to(dev), g.to(dev)).cpu()-y).abs().mean(1).numpy())
        vmae = float(np.concatenate(errs).mean())
        if vmae < best:
            best = vmae; best_sd = {k: v.cpu().clone() for k, v in evalnet.state_dict().items()}
            # val 예측/GT std 기록 (de-shrinkage 확인)
            with torch.no_grad():
                pp = np.concatenate([evalnet(img.to(dev), g.to(dev)).cpu().numpy() for img, g, y in dl_va])
                yy = np.concatenate([y.numpy() for _, _, y in dl_va])
            best_std = (pp.std(0).round(2), yy.std(0).round(2))
            vol = yy.prod(1); big = vol >= np.quantile(vol, 0.67)
            best_big = round(float(np.abs(pp[big] - yy[big]).mean()), 3)
    if args.swa and swa_sd is not None:   # SWA: 평균 가중치 + BN 재보정 후 평가
        for k in swa_sd: swa_sd[k] = swa_sd[k]/swa_n
        net.load_state_dict({k: v.to(dev) for k, v in swa_sd.items()})
        net.train()
        with torch.no_grad():
            for img, g, y in dl_tr: net(img.to(dev), g.to(dev))   # BN 통계 갱신
        net.eval(); errs = []
        with torch.no_grad():
            for img, g, y in dl_va: errs.append((net(img.to(dev), g.to(dev)).cpu()-y).abs().mean(1).numpy())
        sv = float(np.concatenate(errs).mean())
        if sv < best: best = sv; best_sd = {k: v.cpu().clone() for k, v in net.state_dict().items()}
        print(f"[{args.tag}] SWA val={sv:.3f}", flush=True)
    print(f"[{args.tag}] best val MAE={best:.3f} 큰박스MAE={best_big:.3f} pred_std={best_std[0]} GT_std={best_std[1]}", flush=True)
    if args.out and best_sd is not None:
        net.load_state_dict(best_sd); net.eval().cpu()
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        if args.out.endswith(".onnx"):
            dummy = (torch.zeros(1, 4 if args.topmask else 3, args.imgsz, args.imgsz), torch.zeros(1, GD))
            torch.onnx.export(net, dummy, args.out, opset_version=19,
                              input_names=["crop", "geo"], output_names=["dims"],
                              dynamic_axes={"crop": {0: "n"}, "geo": {0: "n"}, "dims": {0: "n"}})
        else:
            torch.save(best_sd, args.out)
        print(f"[{args.tag}] saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
