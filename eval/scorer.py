"""
공식 평가지표 로컬 재현 채점기.

채점 순서(과제 명세):
  1. 정답 박스를 부피(w*d*h) 내림차순 정렬
  2. 예측 박스도 부피 내림차순 정렬
  3. 짧은 쪽 목록의 '앞부분'을 (0,0,0)으로 padding 하여 길이 맞춤
  4. rank 순서대로 pairwise 매칭
  5. 각 pair: 세 변(w,d,h) 절대오차를, 축 permutation(6가지) 중 최소가 되도록 매핑
       per_pair = (|dw|+|dd|+|dh|)  (min permutation)
  6. video MAE = mean over pairs of per_pair  (= '세 변 오차 합산 후 pair 평균')
  7. final score = mean over videos of video MAE

명세 문구가 '세 변의 MAE'와 '세 변 오차 합산'을 함께 써서, 변 3개를 합으로 볼지
평균으로 볼지 모호하다. 둘은 상수배(×3)라 '개선 방향/순위'는 동일하므로
per_side_reduce 로 선택 가능하게 두고 두 값 모두 출력한다. 기본 'sum'.
"""
from itertools import permutations
import json
import numpy as np

_PERMS = list(permutations(range(3)))  # 6 axis permutations


def _to_arr(objs):
    """objects 리스트 -> (N,3) ndarray of (w,d,h)."""
    if not objs:
        return np.zeros((0, 3), dtype=float)
    out = []
    for o in objs:
        s = o["size_cm"]
        out.append([float(s["w"]), float(s["d"]), float(s["h"])])
    return np.asarray(out, dtype=float)


def _sort_by_volume_desc(a):
    if len(a) == 0:
        return a
    vol = a[:, 0] * a[:, 1] * a[:, 2]
    return a[np.argsort(-vol)]


def _pair_error(pred_row, gt_row):
    """한 pair의 변 절대오차 합(축 permutation 최소)."""
    best = None
    for p in _PERMS:
        e = np.abs(pred_row[list(p)] - gt_row).sum()
        if best is None or e < best:
            best = e
    return best


def video_mae(pred_objs, gt_objs, per_side_reduce="sum"):
    """한 영상 점수. per_side_reduce: 'sum'(변3개 합) 또는 'mean'(/3)."""
    pred = _sort_by_volume_desc(_to_arr(pred_objs))
    gt = _sort_by_volume_desc(_to_arr(gt_objs))
    n = max(len(pred), len(gt))
    if n == 0:
        return 0.0
    # 앞부분 zero-padding
    def front_pad(a):
        if len(a) < n:
            pad = np.zeros((n - len(a), 3), dtype=float)
            return np.vstack([pad, a])
        return a
    pred, gt = front_pad(pred), front_pad(gt)
    errs = [_pair_error(pred[i], gt[i]) for i in range(n)]
    per_pair = np.asarray(errs, dtype=float)
    if per_side_reduce == "mean":
        per_pair = per_pair / 3.0
    return float(per_pair.mean())


def score(pred_videos, gt_videos, per_side_reduce="sum", verbose=False):
    """
    pred_videos, gt_videos: {video_id: [objects...]} 또는 result.json 형식 dict.
    반환: (final_score, per_video dict)
    """
    pred_map = _as_map(pred_videos)
    gt_map = _as_map(gt_videos)
    per_video = {}
    for vid, gt_objs in gt_map.items():
        pred_objs = pred_map.get(vid, [])
        per_video[vid] = video_mae(pred_objs, gt_objs, per_side_reduce)
    missing = [v for v in gt_map if v not in pred_map]
    if verbose and missing:
        print(f"[scorer] 예측 누락 영상 {len(missing)}개 (빈 예측으로 처리): {missing[:5]}...")
    final = float(np.mean(list(per_video.values()))) if per_video else 0.0
    return final, per_video


def size_diagnostic(pred_videos, gt_videos):
    """
    size 품질만 분리 측정 (count 영향 제거).
    영상별: pred/gt를 부피 내림차순 정렬 후 '겹치는 상위 n개'만 pairwise(축 permutation 최소),
    매칭된 박스의 변당 절대오차 평균. count 과부족 박스는 제외.
    반환: dict(per_side_mae, n_matched, count_pred_total, count_gt_total)
    """
    pmap, gmap = _as_map(pred_videos), _as_map(gt_videos)
    errs, nmatch, cp, cg = [], 0, 0, 0
    for vid, gt in gmap.items():
        P = _sort_by_volume_desc(_to_arr(pmap.get(vid, [])))
        G = _sort_by_volume_desc(_to_arr(gt))
        cp += len(P); cg += len(G)
        n = min(len(P), len(G))
        for i in range(n):
            best = min(np.abs(P[i][list(p)] - G[i]).sum() for p in _PERMS)
            errs.append(best / 3.0)  # 변당 평균
            nmatch += 1
    return {"per_side_mae": float(np.mean(errs)) if errs else None,
            "n_matched": nmatch, "count_pred_total": cp, "count_gt_total": cg}


def _as_map(x):
    if isinstance(x, dict) and "videos" in x:
        x = x["videos"]
    if isinstance(x, list):
        return {v["video_id"]: v.get("objects", []) for v in x}
    if isinstance(x, dict):
        return x
    raise TypeError("지원하지 않는 형식")


def load(path):
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="로컬 채점기")
    ap.add_argument("--pred", required=True, help="result.json")
    ap.add_argument("--gt", required=True, help="train_label.json 형식")
    ap.add_argument("--reduce", default="sum", choices=["sum", "mean"])
    args = ap.parse_args()
    final, pv = score(load(args.pred), load(args.gt), args.reduce, verbose=True)
    both = {}
    for r in ("sum", "mean"):
        f, _ = score(load(args.pred), load(args.gt), r)
        both[r] = f
    print(f"Final score (reduce={args.reduce}): {final:.4f}")
    print(f"  [참고] sum={both['sum']:.4f}  mean(/3)={both['mean']:.4f}")
    worst = sorted(pv.items(), key=lambda kv: -kv[1])[:5]
    print("worst videos:", [(k, round(v, 2)) for k, v in worst])
