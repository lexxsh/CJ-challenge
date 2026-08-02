"""OC-SORT 좁은 확인 — 새 검출기(RF-DETR res1008)에서 기존 파라미터가 아직 최적인지만 검증.

**넓은 격자 스윕을 하지 않는 이유**: 90영상 count MAE 의 노이즈가 ~0.06~0.07 인데
파라미터 조합을 수십 개 훑으면 개선이 아니라 **노이즈 최저점**을 고르게 됨.
전례: 수동 튜닝으로 13.814 -> 13.855 (악화). 크기별 conf 임계도 12조합 중 최고 0.211 -> 선택편향으로 폐기.

따라서: 현재 설정(conf0.7 / iou0.15 / age20 / mh2)에서 **축 하나씩만** 흔들고,
부트스트랩 신뢰구간으로 '노이즈를 넘는 차이'인지 판정. 넘지 않으면 현행 유지.

사용: python eval/ocsort_check.py
"""
import sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cross_eval import track_count, gt_counts

BASE = dict(conf=0.7, iou_thr=0.15, max_age=20, min_hits=2)
FOLDS = [1, 2, 3]

per = {}
for k in FOLDS:
    per.update(json.load(open(f"runs_fold/epdump/f{k}_rfdetr_crop_ep10.json")))
G = gt_counts()
V = sorted(per)
print(f"=== OC-SORT 좁은 확인 | RF-DETR res1008 | fold{FOLDS} {len(V)}영상 OOF ===")


def err(cfg):
    return np.array([track_count(per[v], cfg["conf"], cfg["iou_thr"],
                                 cfg["max_age"], cfg["min_hits"]) - G[v] for v in V])


rng = np.random.default_rng(0)
BI = rng.integers(0, len(V), size=(2000, len(V)))    # 고정 부트스트랩 인덱스 (설정 간 공유)


def stat(e):
    m = np.abs(e).mean()
    bs = np.abs(e)[BI].mean(1)
    return m, bs


e0 = err(BASE); m0, bs0 = stat(e0)
print(f"\n현행 {BASE}")
print(f"  MAE {m0:.3f}  정확 {(e0==0).sum()}/{len(V)}  순Δ {e0.sum():+d}")
print(f"  부트스트랩 95%CI [{np.percentile(bs0,2.5):.3f}, {np.percentile(bs0,97.5):.3f}]")

AXES = {
    "iou_thr":  [0.10, 0.20, 0.30],
    "max_age":  [10, 15, 30, 40],
    "min_hits": [1, 3],
    "conf":     [0.65, 0.75],
}
print(f"\n{'축':>9s} {'값':>6s} {'MAE':>7s} {'Δ':>7s} {'Δ의 95%CI':>18s}  판정")
wins = []
for ax, vals in AXES.items():
    for v in vals:
        cfg = dict(BASE); cfg[ax] = v
        e = err(cfg); m, bs = stat(e)
        d = m - m0
        # 쌍대 부트스트랩: 같은 리샘플에서의 차이 분포 -> 영상 간 변동 상쇄
        db = bs - bs0
        lo, hi = np.percentile(db, 2.5), np.percentile(db, 97.5)
        sig = "개선" if hi < 0 else ("악화" if lo > 0 else "노이즈")
        print(f"{ax:>9s} {str(v):>6s} {m:7.3f} {d:+7.3f} {f'[{lo:+.3f}, {hi:+.3f}]':>18s}  {sig}")
        if hi < 0:
            wins.append((ax, v, m, lo, hi))

print()
if wins:
    print("★ 노이즈를 넘는 개선 후보:")
    for ax, v, m, lo, hi in wins:
        print(f"   {ax}={v} -> MAE {m:.3f} (Δ 95%CI [{lo:+.3f}, {hi:+.3f}])")
    print("   -> 단일 축이고 CI 가 0 아래이므로 채택 검토 가능. 단 다른 fold 로 재확인할 것.")
else:
    print("★ 모든 축에서 개선 없음 (전부 노이즈 범위) -> 현행 설정 유지가 정답.")
    print("  기존 파라미터가 새 검출기에서도 최적 근처임을 확인. count 트랙 종결.")
