"""정직한 CV harness — v14 과적합의 3중 방법론 결함을 제거.
  (1) 선택누수 제거: outer test fold는 학습·에폭선택 어디에도 안 씀 (inner val 따로)
  (2) 선택기준: val MAE(분산수축 유발) 대신 --select scorer (공식 부피순위 지표)
  (3) 전 영상 OOF → 공식 scorer = 진짜 out-of-sample 실력

★ 4번째 결함(2026-07-16 발견): **크롭 누수**.
  기존 크롭은 그 영상을 학습한 검출기로 만들어짐 -> 박스가 배포 때보다 정확 -> OOF 가 낙관적.
  증거: 기하스택 이득을 예측 -1.77 vs 실제 리더보드 -0.09 (20배 과대예측).
  대책: --folds_json 으로 검출기와 fold 정의를 통일하고, 크롭도 fold별 OOF 검출기로 재빌드
        (pipeline/dump_vdump_oof.py -> build_size_ds). 3층(크롭·라벨·평가)이 한 분리로 정렬됨.

⚠️ 학습 노이즈: base 5시드 std ~0.06 -> 0.15 이상 차이만 실재로 읽을 것. 단일 실행 판정 금지.

사용: python eval/cv_honest_oof.py --tag v14base --erase 0.4 --distmatch 0.5 --dm_tail 2.0
"""
import argparse, os, sys, json, math, numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from size_net2 import SizeNet2
from train_size2_enh import DS, geo_vec
from eval import scorer as SC
from collections import defaultdict


def track_of(r):
    return r["crop"].rsplit("_f", 1)[0]


def to_videos(recs, preds):
    """per-frame 예측 -> 트랙별 median -> 영상별 objects."""
    byt = defaultdict(list)
    for r, p in zip(recs, preds):
        byt[(r["vid"], track_of(r))].append(p)
    out = defaultdict(list)
    for (vid, tk), ps in byt.items():
        m = np.median(np.stack(ps), 0)
        out[vid].append({"size_cm": {"w": float(m[0]), "d": float(m[1]), "h": float(m[2])}})
    return dict(out)


_FULLGT = None


def gt_videos(recs, full=False):
    """full=False: 매칭된 트랙의 GT만 -> **count 가 항상 정답**이 되어 count 비용이 안 보임.
    full=True : train_label.json 의 **전 물체** -> 미검출이 scorer 의 앞쪽 (0,0,0) 패딩으로 벌점.
                배포(hidden test)와 동일 조건. count 비용(실측 1.506)이 지표에 반영됨.
    """
    if full:
        global _FULLGT
        if _FULLGT is None:
            _FULLGT = {v["video_id"]: [{"size_cm": dict(o["size_cm"])} for o in v["objects"]]
                       for v in json.load(open("assignment1/dataset/train_label.json"))["videos"]}
        return {v: _FULLGT[v] for v in {r["vid"] for r in recs}}
    byt = {}
    for r in recs:
        byt[(r["vid"], track_of(r))] = r
    out = defaultdict(list)
    for (vid, tk), r in byt.items():
        out[vid].append({"size_cm": {"w": r["w"], "d": r["d"], "h": r["h"]}})
    return dict(out)


def predict(net, dl, dev):
    net.eval(); ps = []
    with torch.no_grad():
        for img, g, y in dl:
            ps.append(net(img.to(dev), g.to(dev)).cpu().numpy())
    return np.concatenate(ps)


def train_fold(args, tr, inval, test, dev):
    md = np.array([sorted([r["w"], r["d"], r["h"]], reverse=True) for r in tr]).mean(0)
    GD = len(geo_vec(tr[0]))
    net = SizeNet2(mean_dims=tuple(md), backbone=args.backbone, geo_dim=GD).to(dev)
    dl_tr = DataLoader(DS(args.data, tr, True, args.imgsz, args.erase), batch_size=args.batch,
                       shuffle=True, num_workers=4, drop_last=True)
    dl_iv = DataLoader(DS(args.data, inval, False, args.imgsz), batch_size=args.batch, num_workers=4)
    dl_te = DataLoader(DS(args.data, test, False, args.imgsz), batch_size=args.batch, num_workers=4)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    lossf = nn.L1Loss()
    iv_gt = gt_videos(inval)
    best = 1e9; best_sd = None
    for ep in range(args.epochs):
        net.train()
        for img, g, y in dl_tr:
            img, g, y = img.to(dev), g.to(dev), y.to(dev)
            pred = net(img, g)
            loss = lossf(pred, y)
            if pred.size(0) > 4:
                if args.varmatch > 0:
                    loss = loss + args.varmatch*(pred.std(0)-y.std(0)).abs().mean()
                if args.distmatch > 0:
                    ps_, _ = pred.sort(0, descending=True); ts = y.sort(0, descending=True)[0]
                    dl_ = (ps_-ts).abs()
                    if args.dm_tail > 0:
                        n = ts.size(0)
                        rw = (1.0 + torch.linspace(1., 0., n, device=y.device)*args.dm_tail)[:, None]
                        dl_ = dl_*rw
                    loss = loss + args.distmatch*dl_.mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        # --- 에폭 선택: inner val (test fold 절대 미사용) ---
        P = predict(net, dl_iv, dev)
        if args.select == "scorer":
            crit = SC.score(to_videos(inval, P), iv_gt, per_side_reduce="sum")[0]
        else:
            Y = np.array([sorted([r["w"], r["d"], r["h"]], reverse=True) for r in inval])
            crit = float(np.abs(P-Y).mean())
        if crit < best:
            best = crit; best_sd = {k: v.cpu().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_sd)
    return predict(net, dl_te, dev), best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/size3_tight")
    ap.add_argument("--tag", default="cv")
    ap.add_argument("--backbone", default="resnet18")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--erase", type=float, default=0.4)
    ap.add_argument("--distmatch", type=float, default=0.5)
    ap.add_argument("--dm_tail", type=float, default=2.0)
    ap.add_argument("--varmatch", type=float, default=0.0)
    ap.add_argument("--nfolds", type=int, default=5)
    ap.add_argument("--folds_json", default="datasets/folds.json",
                    help="검출기와 동일한 fold 정의. 빈 문자열이면 구 stride 분리(재현용)")
    ap.add_argument("--full_gt", action="store_true",
                    help="GT를 train_label.json 전 물체로 -> 미검출이 벌점(배포 조건). 기본은 매칭트랙만(count 숨김)")
    ap.add_argument("--select", default="scorer", choices=["scorer", "mae"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(os.path.join(args.data, "labels.jsonl"))]
    vids = sorted({r["vid"] for r in recs})
    dev = args.device if torch.cuda.is_available() else "cpu"

    # ★ fold 정의를 folds.json 으로 통일 (검출기 fold 와 동일).
    #   이래야 fold k 영상의 크롭이 fold k 검출기(그 영상 미학습)에서 나오고,
    #   같은 영상이 size 모델의 outer test 가 됨 -> 크롭·평가 3층이 한 분리로 정렬.
    #   (구 방식 vids[k::nfolds] 는 이름순 stride 라 검출 fold 와 어긋남)
    FOLDS = None
    if args.folds_json:
        FD = json.load(open(args.folds_json))
        FOLDS = {k: [v for v in FD["folds"][str(k)] if v in set(vids)] for k in range(args.nfolds)}
        n = sum(len(v) for v in FOLDS.values())
        if n != len(vids):
            sys.exit(f"★ fold 정의 불일치: folds.json 이 {n}영상만 덮음 (데이터셋 {len(vids)}영상)")
    print(f"[{args.tag}] {len(recs)}크롭 / {len(vids)}영상 / {args.nfolds}-fold"
          f" / split={'folds.json' if FOLDS else 'stride(구)'} / select={args.select}", flush=True)

    oof_recs = []; oof_preds = []
    for k in range(args.nfolds):
        test_v = set(FOLDS[k]) if FOLDS else set(vids[k::args.nfolds])
        rest = [v for v in vids if v not in test_v]
        inval_v = set(rest[::6])                       # inner val (에폭선택 전용)
        tr = [r for r in recs if r["vid"] not in test_v and r["vid"] not in inval_v]
        iv = [r for r in recs if r["vid"] in inval_v]
        te = [r for r in recs if r["vid"] in test_v]
        P, crit = train_fold(args, tr, iv, te, dev)
        oof_recs += te; oof_preds.append(P)
        s = SC.score(to_videos(te, P), gt_videos(te, args.full_gt), per_side_reduce="sum")[0]
        print(f"  fold{k}: test {len(test_v)}영상 | inner-val {args.select}={crit:.4f} | test scorer={s:.4f}", flush=True)
    OP = np.concatenate(oof_preds)
    final = SC.score(to_videos(oof_recs, OP), gt_videos(oof_recs, args.full_gt), per_side_reduce="sum")[0]
    Y = np.array([sorted([r["w"], r["d"], r["h"]], reverse=True) for r in oof_recs])
    print(f"\n[{args.tag}] === 정직한 OOF scorer = {final:.4f} ===  (축별 MAE {np.abs(OP-Y).mean(0).round(2)})", flush=True)
    os.makedirs("oof", exist_ok=True)
    json.dump({"tag": args.tag, "scorer": final,
               "preds": [{"vid": r["vid"], "crop": r["crop"], "p": [float(x) for x in p]}
                         for r, p in zip(oof_recs, OP)]},
              open(f"oof/{args.tag}.json", "w"))
    print("=== DONE ===")


if __name__ == "__main__":
    main()
