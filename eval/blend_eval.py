"""배포에 충실한 혼합 평가 — CNN×(1-a) + 기하스택×a 를 **배포 파이프라인과 같은 조건**으로 채점.

기존 하네스(cv_train.gt_videos / combine_eval)의 2대 결함을 제거:
  ① count 숨김: GT 를 '매칭된 트랙'에서 만들어 트랙 9개면 GT 도 9개 -> count 가 항상 정답.
     실제 리더보드에선 count 오류가 1.506점(실측). scorer 는 앞쪽 (0,0,0) 패딩 -> 최대박스와 매칭 + 순위 밀림.
     => GT 는 train_label.json 의 **전 물체**를 쓴다.
  ② 기하 폴백 무시: 기하 피처는 seg 마스크를 못 찾으면 생성 실패(918 GT 중 865만 존재).
     combine_eval 은 그런 박스를 **평가에서 제외** -> 기하스택이 자기가 잘하는 박스에서만 채점받음.
     배포는 기하 없으면 **CNN 단독으로 폴백**해서 그 박스도 치수를 냄.
     => 기하 없는 트랙은 CNN 값을 그대로 쓴다.

이 둘을 고치면 OOF 가 배포와 같은 것을 재게 됨.
(실측: 기하 단독 OOF 가 count 숨김 10.825 -> 반영 15.857. 결함의 크기가 0.1 이 아니라 5점 규모)

조인: features 의 bw_px_key <-> labels.jsonl 의 트랙 대표 bw_px (combine_eval 방식과 동일)

사용:
  python eval/blend_eval.py --cnn ep8_s1 ep8_s2 --cnn2 ep8_rail_s1 ep8_rail_s2 \
      --geo oof/geo_features3.json --data datasets/size3_tight
"""
import argparse, json, os, sys
import numpy as np
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval import scorer as SC

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="datasets/size3_tight")
ap.add_argument("--cnn", nargs="*", default=["ep8_s1", "ep8_s2", "ep8_s3", "ep8_s4"], help="CNN OOF 태그(시드평균)")
ap.add_argument("--cnn2", nargs="*", default=["ep8_rail_s1", "ep8_rail_s2"], help="두번째 뷰(있으면 평균)")
ap.add_argument("--geo", default="oof/geo_features3.json")
ap.add_argument("--feats", default="features3.json")
ap.add_argument("--alphas", type=float, nargs="*", default=[0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0])
ap.add_argument("--hide_count", action="store_true", help="(진단용) 구 방식대로 count 숨김")
a = ap.parse_args()


def track_of(c):
    return c.rsplit("_f", 1)[0]


recs = [json.loads(l) for l in open(f"{a.data}/labels.jsonl")]


def load(tag):
    p = tag if tag.endswith(".json") else f"oof/{tag}.json"
    if not os.path.exists(p):
        return None
    return {(x["vid"], x["crop"]): np.array(x["p"]) for x in json.load(open(p))["preds"]}


def member(tags):
    ms = [m for m in (load(t) for t in tags) if m]
    if not ms:
        return None
    keys = set(ms[0])
    for m in ms[1:]:
        keys &= set(m)
    return {k: np.mean([m[k] for m in ms], 0) for k in keys}


# ---- CNN: 시드평균 -> (뷰평균) -> 트랙 median ----
v1 = member(a.cnn)
if v1 is None:
    sys.exit(f"CNN OOF 없음: {a.cnn}")
v2 = member(a.cnn2) if a.cnn2 else None
if v2:
    ks = set(v1) & set(v2)
    cnn_crop = {k: (v1[k]+v2[k])/2 for k in ks}
else:
    cnn_crop = v1
tk_ = defaultdict(list)
for (vid, crop), p in cnn_crop.items():
    tk_[(vid, track_of(crop))].append(p)
CNN = {k: np.median(np.stack(v), 0) for k, v in tk_.items()}
print(f"CNN 트랙 {len(CNN)}개 (뷰 {'2' if v2 else '1'}, 시드 {len(a.cnn)})")

# ---- 기하: bw_px_key 로 트랙 조인 ----
bytk = defaultdict(list)
for r in recs:
    bytk[(r["vid"], track_of(r["crop"]))].append(r)
key2tk = {}
for tk, rs in bytk.items():
    rep = max(rs, key=lambda r: r["bw_px"])
    key2tk[(rep["vid"], round(rep["bw_px"], 2))] = tk

F = json.load(open(a.feats))
GP = json.load(open(a.geo))["preds"]
if len(GP) != len(F):
    sys.exit(f"★ 기하 OOF({len(GP)})와 피처({len(F)}) 행수 불일치 — 같은 --feats 로 만든 것인지 확인")
GEO = {}
for r, g in zip(F, GP):
    tk = key2tk.get((r["vid"], round(r["bw_px_key"], 2)))
    if tk and tk in CNN:
        GEO[tk] = np.array(g["p"])
print(f"기하 조인 {len(GEO)}트랙  (CNN 트랙 중 {len(GEO)/max(len(CNN),1)*100:.0f}%)")
print(f"  -> 기하 없는 {len(CNN)-len(GEO)}트랙은 **CNN 단독 폴백** (배포와 동일)")

# ---- GT: 전 물체 (count 반영) ----
FULL = {v["video_id"]: [{"size_cm": dict(o["size_cm"])} for o in v["objects"]]
        for v in json.load(open("assignment1/dataset/train_label.json"))["videos"]}
VIDS = sorted({k[0] for k in CNN})
if a.hide_count:
    gtv = defaultdict(list)
    for tk, rs in bytk.items():
        if tk in CNN:
            r = rs[0]
            gtv[tk[0]].append({"size_cm": {"w": r["w"], "d": r["d"], "h": r["h"]}})
    GTV = dict(gtv)
    print("  [진단모드] count 숨김 — 매칭 트랙만 GT")
else:
    GTV = {v: FULL[v] for v in VIDS}
    ngt = sum(len(FULL[v]) for v in VIDS)
    print(f"GT {ngt}물체 / 트랙 {len(CNN)}개 -> 미검출 {ngt-len(CNN)}개가 벌점 대상 (배포 조건)")


def sc(alpha):
    pv = defaultdict(list)
    for tk, c in CNN.items():
        p = c if tk not in GEO else c*(1-alpha) + GEO[tk]*alpha    # 기하 없으면 CNN 폴백
        pv[tk[0]].append({"size_cm": {"w": float(p[0]), "d": float(p[1]), "h": float(p[2])}})
    return SC.score(dict(pv), GTV, per_side_reduce="sum")[0]


print(f"\n{'alpha(기하 비중)':>16s} {'scorer':>9s}")
best = None
for al in a.alphas:
    s = sc(al)
    m = ""
    if al == 0.0:
        m = "  <- CNN 단독"
    if al == 0.6:
        m = "  <- 배포값"
    if al == 1.0:
        m = "  <- 기하 단독"
    print(f"{al:16.1f} {s:9.4f}{m}")
    if best is None or s < best[1]:
        best = (al, s)
print(f"\n★ 최적 alpha={best[0]} -> {best[1]:.4f}   (CNN 단독 대비 {best[1]-sc(0.0):+.4f})")
