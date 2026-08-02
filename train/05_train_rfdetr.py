"""RF-DETR fold 학습 — YOLO 와 **같은 fold 정의**로 공정 비교.

목적: YOLO26m+crop 이 기존 RF-DETR(test50 0.240)보다 나쁜 원인이
      아키텍처인지 데이터/크롭인지 가르기.

fold 파일리스트(datasets/{ds}{k}/train.txt) -> COCO 변환 -> RFDETR 학습.
환경: .venv_rfdetr  (torch 2.6+cu124. .venv_train 과 충돌하므로 분리)

사용:
  source .venv_rfdetr/bin/activate
  CUDA_VISIBLE_DEVICES=0 python train/05_train_rfdetr.py --fold 0 --ds fold \
      --size medium --epochs 40 --out runs_fold/rf_f0_crop
"""
import sys, os, json, argparse, cv2

CATS = [{"id": 0, "name": "box-det", "supercategory": "none"},
        {"id": 1, "name": "box", "supercategory": "box-det"}]


def img2label(p):
    return p.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"


def build(ds, fold, out):
    """fold 파일리스트 -> COCO. train=train.txt, valid/test=val.txt"""
    for cd, src in [("train", "train.txt"), ("valid", "val.txt"), ("test", "val.txt")]:
        os.makedirs(f"{out}/{cd}", exist_ok=True)
        paths = [x for x in open(f"datasets/{ds}{fold}/{src}").read().split("\n") if x]
        coco = {"images": [], "annotations": [], "categories": CATS}
        iid = aid = 0
        for ip in paths:
            nm = os.path.splitext(os.path.basename(ip))[0]
            im = cv2.imread(ip)
            if im is None:
                continue
            H, W = im.shape[:2]
            cv2.imwrite(f"{out}/{cd}/{nm}.jpg", im)
            coco["images"].append({"id": iid, "file_name": f"{nm}.jpg", "width": W, "height": H})
            lp = img2label(ip)
            if os.path.exists(lp):
                for ln in open(lp):
                    t = ln.split()
                    if len(t) != 5:
                        continue
                    cx, cy, w, h = map(float, t[1:])
                    bw, bh = w*W, h*H
                    coco["annotations"].append({"id": aid, "image_id": iid, "category_id": 1,
                                                "bbox": [cx*W-bw/2, cy*H-bh/2, bw, bh],
                                                "area": bw*bh, "iscrowd": 0})
                    aid += 1
            iid += 1
        json.dump(coco, open(f"{out}/{cd}/_annotations.coco.json", "w"))
        print(f"  COCO {cd:5s}: {iid}장 / {aid}박스", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--ds", default="fold")          # fold(crop) / barfold / fullfold
    ap.add_argument("--size", default="medium", choices=["medium", "large"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--resolution", type=int, default=1008)   # 56 배수
    ap.add_argument("--out", default="")
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    out = a.out or f"runs_fold/rf_f{a.fold}_{a.ds}"
    DS = f"datasets/{a.ds}{a.fold}_coco"

    if a.rebuild or not os.path.exists(f"{DS}/train/_annotations.coco.json"):
        print(f"[COCO 변환] datasets/{a.ds}{a.fold} -> {DS}", flush=True)
        build(a.ds, a.fold, DS)
    else:
        print(f"COCO 존재 -> 변환 생략 ({DS})", flush=True)

    if os.path.exists(f"{out}/checkpoint_best_total.pth"):
        sys.exit(f"[ERR] {out} 에 checkpoint 존재 -> 덮어쓰기 방지")
    os.makedirs(out, exist_ok=True)
    from rfdetr import RFDETRMedium, RFDETRLarge
    M = RFDETRMedium if a.size == "medium" else RFDETRLarge
    print(f"[RF-DETR {a.size}] fold{a.fold} ds={a.ds} res={a.resolution} ep={a.epochs} -> {out}", flush=True)
    model = M(resolution=a.resolution)
    model.train(dataset_dir=DS, epochs=a.epochs, batch_size=a.batch,
                grad_accum_steps=a.accum, lr=1e-4, output_dir=out)
    print(f"[완료] {out}")


if __name__ == "__main__":
    main()
