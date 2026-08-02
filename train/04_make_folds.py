"""150영상 5-fold 분할 (혼잡도 × 출처 층화).

목적: 최적 에폭을 **count 기준**으로 찾기 위함 (mAP 아님 — mAP50은 이미 포화).
  - 각 영상이 4/5 fold 에서 train  -> 어려운 영상도 학습에 80% 포함
  - 각 영상이 정확히 1번 out-of-sample 평가 -> OOF n=150 (val 29의 탐지한계 0.125 -> ~0.06)
  - fold 마다 난이도 분포 동일 (혼잡한 영상이 한 fold에 몰리지 않게)
최종 모델은 여기서 찾은 에폭으로 **150영상 전부** 재학습 (val 없음 = 선택누수 0).
"""
import os, glob, json, random, argparse
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="explore/detds_all")
ap.add_argument("--out", default="datasets/folds.json")
ap.add_argument("--k", type=int, default=5)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
random.seed(a.seed)

byv = defaultdict(list)
for p in sorted(glob.glob(f"{a.src}/labels/*.txt")):
    nm = os.path.basename(p)[:-4]
    byv[nm.rsplit("_f", 1)[0]].append(nm)
vids = sorted(byv)
crowd = {v: max(sum(1 for ln in open(f"{a.src}/labels/{nm}.txt") if ln.strip()) for nm in byv[v])
         for v in vids}


def bin_of(c):
    return 0 if c <= 4 else 1 if c <= 7 else 2 if c <= 11 else 3


BN = ["<=4", "5-7", "8-11", "12+"]

# (혼잡도 × 출처) 그룹별 라운드로빈. 오프셋을 그룹간 이어붙여 나머지가 한쪽 fold에 쏠리지 않게.
groups = defaultdict(list)
for v in vids:
    groups[(bin_of(crowd[v]), v.split("_")[0])].append(v)
fold = {}
off = 0
for k in sorted(groups):
    g = sorted(groups[k]); random.shuffle(g)
    for i, v in enumerate(g):
        fold[v] = (off + i) % a.k
    off = (off + len(g)) % a.k        # 다음 그룹은 이어서 배정 -> fold 크기 균등

F = {i: sorted(v for v in vids if fold[v] == i) for i in range(a.k)}
json.dump({"k": a.k, "seed": a.seed, "fold": fold,
           "folds": {str(i): F[i] for i in range(a.k)}}, open(a.out, "w"), indent=1)

print(f"=== {a.k}-fold / {len(vids)}영상 (seed={a.seed}) -> {a.out} ===\n")
print(f"{'fold':>5s} {'영상':>4s} | " + " ".join(f"{b:>5s}" for b in BN) + " | train_ test_")
for i in range(a.k):
    c = [sum(1 for v in F[i] if bin_of(crowd[v]) == b) for b in range(4)]
    nt = sum(1 for v in F[i] if v.startswith("train")); ne = len(F[i])-nt
    print(f"{i:5d} {len(F[i]):4d} | " + " ".join(f"{x:5d}" for x in c) + f" | {nt:5d} {ne:5d}")
tot = [sum(1 for v in vids if bin_of(crowd[v]) == b) for b in range(4)]
print(f"{'ALL':>5s} {len(vids):4d} | " + " ".join(f"{x:5d}" for x in tot))
print(f"\n각 fold 학습량: {len(vids)-len(F[0])}영상 ({100*(len(vids)-len(F[0]))/len(vids):.0f}%)")
print(f"OOF 평가: 전 {len(vids)}영상 -> 탐지한계 val29 0.125 대비 ~2배 예민")
