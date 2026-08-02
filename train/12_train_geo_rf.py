"""기하스택 회귀기 (1등 정식화) 학습 — **소실된 배포 학습기 재작성**.

배포본(checkpoints/rf_{long,mid,short}.onnx, 07-11 22:02)을 만든 스크립트가 레포에 없었음
(이전 세션 heredoc 추정). ONNX 역추출로 하이퍼파라미터 복원:
  트리 300개 / 피처 11개 / 트리당 노드 최대 103
  -> depth5(최대 63노드)면 불가, depth6(최대 127노드)면 가능 -> **RF(n=300, max_depth=6)**
  geo_meta.json 의 FEATS·blend_alpha_geo=0.6, STATUS 의 "RF-d6" 과 일치.
(train_gbm.py 는 gbm_*.onnx 를 뱉는 별개의 죽은 스크립트 — 배포본이 아님)

정식화: S = 62.3 × (box_px/rail_px) × (z_box/z_rail) × 학습잔차
  z_box 를 **seg 마스크 픽셀만**으로 뽑는 게 결정적 (bbox 중앙값은 벨트가 섞여 신호가 죽음)

★ 크롭 누수: 기본 features3.json 은 **그 영상을 학습한 검출기** 트랙에서 나옴 -> OOF 낙관.
  누수 없는 판: --feats features3_oof.json (VDUMP=explore/vdump_oof.json 로 빌드)
  --cv 는 folds.json(검출기와 동일 분리)로 GroupKFold -> 3층 정렬.

사용:
  python train/12_train_geo_rf.py --cv                      # 정직한 OOF scorer
  python train/12_train_geo_rf.py --final --out checkpoints   # 배포 ONNX
"""
import argparse, json, os, sys
import numpy as np
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval import scorer as SC

FEATS = ["maj_cm", "min_cm", "bw_cm", "bh_cm", "area_cm2", "cy_norm",
         "ratio", "maj_x_ratio", "min_x_ratio", "zspan", "zbox"]   # geo_meta.json 과 동일 순서 = 단일 진실

ap = argparse.ArgumentParser()
ap.add_argument("--feats", default="features3.json")
ap.add_argument("--folds_json", default="datasets/folds.json")
ap.add_argument("--nfolds", type=int, default=5)
ap.add_argument("--trees", type=int, default=300)     # ONNX 역추출
ap.add_argument("--depth", type=int, default=6)       # ONNX 역추출
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--target", default="abs", choices=["abs", "ratio"],
                help="abs=절대cm 직접회귀(현행) / ratio=형상비만 회귀하고 스케일은 기하(maj_cm)로 복원 (1등 단서)")
ap.add_argument("--scale_key", default="maj_cm", help="ratio 모드의 스케일 기준 피처")
ap.add_argument("--out", default="checkpoints")
ap.add_argument("--full_gt", action="store_true",
                help="GT를 train_label.json 전 물체로 -> 미검출 벌점(배포 조건). 기본은 매칭박스만(count 숨김)")
ap.add_argument("--cv", action="store_true"); ap.add_argument("--final", action="store_true")
a = ap.parse_args()

A = json.load(open(a.feats))
X = np.array([[float(r[f]) for f in FEATS] for r in A], np.float32)
Y = np.array([sorted([r["w"], r["d"], r["h"]], reverse=True) for r in A], np.float32)  # long/mid/short
V = [r["vid"] for r in A]

# ★ shape/scale 분리 (1등 체크포인트의 'ratio' 단서).
#   maj_cm 은 이미 기하(레일 62.3 × 픽셀비 × depth비)로 복원된 **스케일**.
#   절대 cm 를 직접 회귀하면 RF 가 스케일 오차 + 형상 오차를 동시에 떠안음 (long축 MAE 5.44 가 그 증상).
#   대신 Y/maj_cm(형상비)만 회귀하고 예측 때 maj_cm 을 곱해 스케일을 되돌림.
SCALE = np.maximum(np.array([float(r[a.scale_key]) for r in A], np.float32), 1e-3)
if a.target == "ratio":
    Yt = Y / SCALE[:, None]
else:
    Yt = Y
print(f"[기하스택] {a.feats}: {len(A)}박스 / {len(set(V))}영상 / 피처 {X.shape[1]}개 "
      f"| RF(n={a.trees}, depth={a.depth}) | target={a.target}", flush=True)


def to_videos(vids, P):
    out = defaultdict(list)
    for v, p in zip(vids, P):
        out[v].append({"size_cm": {"w": float(p[0]), "d": float(p[1]), "h": float(p[2])}})
    return dict(out)


_FULLGT = None


def gt_videos(vids, Yv):
    """--full_gt: train_label.json 전 물체를 GT 로 -> 미검출이 scorer 앞쪽 (0,0,0) 패딩으로 벌점.
    기본(매칭된 박스만)은 count 가 항상 정답이 되어 count 비용(실측 1.506)이 지표에서 사라짐.
    """
    if a.full_gt:
        global _FULLGT
        if _FULLGT is None:
            _FULLGT = {v["video_id"]: [{"size_cm": dict(o["size_cm"])} for o in v["objects"]]
                       for v in json.load(open("assignment1/dataset/train_label.json"))["videos"]}
        return {v: _FULLGT[v] for v in set(vids)}
    out = defaultdict(list)
    for v, y in zip(vids, Yv):
        out[v].append({"size_cm": {"w": float(y[0]), "d": float(y[1]), "h": float(y[2])}})
    return dict(out)


def build():
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(n_estimators=a.trees, max_depth=a.depth,
                                 random_state=a.seed, n_jobs=-1)


if a.cv:
    FD = json.load(open(a.folds_json))
    fold_of = {v: k for k in range(a.nfolds) for v in FD["folds"][str(k)]}
    miss = sorted({v for v in V if v not in fold_of})
    if miss:
        sys.exit(f"★ folds.json 에 없는 영상 {len(miss)}개: {miss[:3]}")
    OP = np.zeros_like(Y)
    for k in range(a.nfolds):
        te = np.array([fold_of[v] == k for v in V])
        if not te.any():
            continue
        for ax in range(3):
            m = build(); m.fit(X[~te], Yt[~te, ax])
            OP[te, ax] = m.predict(X[te]) * (SCALE[te] if a.target == "ratio" else 1.0)
        s = SC.score(to_videos([v for v, t in zip(V, te) if t], OP[te]),
                     gt_videos([v for v, t in zip(V, te) if t], Y[te]), per_side_reduce="sum")[0]
        print(f"  fold{k}: {int(te.sum())}박스 / {len({v for v,t in zip(V,te) if t})}영상 | test scorer {s:.4f}", flush=True)
    final = SC.score(to_videos(V, OP), gt_videos(V, Y), per_side_reduce="sum")[0]
    print(f"\n[기하스택] === 정직한 OOF scorer = {final:.4f} ===  (축별 MAE {np.abs(OP-Y).mean(0).round(2)})")
    tag = (os.path.basename(a.feats).replace(".json", "")
           + ("_fullgt" if a.full_gt else "") + ("" if a.target == "abs" else f"_{a.target}"))
    os.makedirs("oof", exist_ok=True)
    p = f"oof/geo_{tag}.json"
    json.dump({"tag": f"geo_{tag}", "scorer": float(final),
               "preds": [{"vid": v, "p": [float(x) for x in p_]} for v, p_ in zip(V, OP)]},
              open(p, "w"))
    print(f"  -> {p}")

if a.final:
    from skl2onnx import to_onnx
    import onnxruntime as ort
    os.makedirs(a.out, exist_ok=True)
    for ax, nm in enumerate(["long", "mid", "short"]):
        m = build(); m.fit(X, Yt[:, ax])
        onx = to_onnx(m, X[:1], target_opset={"": 19, "ai.onnx.ml": 3})
        p = f"{a.out}/rf_{nm}.onnx"
        open(p, "wb").write(onx.SerializeToString())
        s = ort.InferenceSession(p, providers=["CPUExecutionProvider"])
        po = s.run(None, {s.get_inputs()[0].name: X[:200]})[0].ravel()
        print(f"  {nm:6s}: {os.path.getsize(p)/1e3:.0f}KB  ONNX-sklearn 최대오차 "
              f"{np.abs(po-m.predict(X[:200])).max():.2e}  train MAE {np.abs(m.predict(X)-Y[:,ax]).mean():.3f}")
    json.dump({"feats": FEATS, "blend_alpha_geo": 0.6, "target": a.target,
           "scale_key": a.scale_key}, open(f"{a.out}/geo_meta.json", "w"))
    print(f"  -> {a.out}/rf_*.onnx + geo_meta.json")

if not (a.cv or a.final):
    sys.exit("--cv 또는 --final 중 하나를 지정")
