"""
OC-SORT 완전 구현 (noahcao/OC_SORT, MIT 기반 재현). 순수 numpy/scipy/filterpy -> ONNX 무관, Docker 호환.

핵심 3단계 + 소멸/출력 로직:
  1) First association: IoU + OCM(속도방향 일관성, inertia)
  2) BYTE: (옵션) 저신뢰 검출을 미매칭 track에 2차 매칭
  3) OCR: 미매칭 track의 '마지막 관측'으로 미매칭 검출 재연결 (잠깐 끊긴 track 복구)
  소멸: time_since_update > max_age 인 track 제거
  출력/활성: time_since_update < 1 AND (hit_streak >= min_hits OR 초반 프레임)
박스별 obs 누적 -> 사이즈 추정에 사용.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_batch(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    w = np.clip(x2-x1, 0, None); h = np.clip(y2-y1, 0, None); inter = w*h
    aa = (a[:, 2]-a[:, 0])*(a[:, 3]-a[:, 1]); ab = (b[:, 2]-b[:, 0])*(b[:, 3]-b[:, 1])
    return inter/(aa[:, None]+ab[None, :]-inter+1e-9)


def speed_direction(b1, b2):
    """두 박스 중심 사이 단위방향 (OCM용)."""
    cx1, cy1 = (b1[0]+b1[2])/2, (b1[1]+b1[3])/2
    cx2, cy2 = (b2[0]+b2[2])/2, (b2[1]+b2[3])/2
    n = np.sqrt((cy2-cy1)**2 + (cx2-cx1)**2) + 1e-6
    return np.array([(cy2-cy1)/n, (cx2-cx1)/n])


def _to_z(b):
    w = b[2]-b[0]; h = b[3]-b[1]
    return np.array([b[0]+w/2, b[1]+h/2, w*h, w/(h+1e-6)]).reshape(4, 1)


def _to_bbox(x):
    w = np.sqrt(abs(x[2]*x[3])); h = x[2]/(w+1e-6)
    return np.array([x[0]-w/2, x[1]-h/2, x[0]+w/2, x[1]+h/2]).ravel()


class KalmanBoxTracker:
    count = 0

    def __init__(self, det, t, delta_t=3):
        from filterpy.kalman import KalmanFilter
        box = det["box"]
        kf = KalmanFilter(dim_x=7, dim_z=4)
        kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],
                         [0,0,0,1,0,0,0],[0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]], float)
        kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]], float)
        kf.R[2:, 2:] *= 10.; kf.P[4:, 4:] *= 1000.; kf.P *= 10.
        kf.Q[-1, -1] *= 0.01; kf.Q[4:, 4:] *= 0.01
        kf.x[:4] = _to_z(box)
        self.kf = kf
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count; KalmanBoxTracker.count += 1
        self.hits = 1; self.hit_streak = 1; self.age = 0
        self.last_observation = np.asarray(box, float)
        self.observations = {t: np.asarray(box, float)}   # frame -> box (속도계산용)
        self.velocity = None
        self.delta_t = delta_t
        self.obs = [(t, det)]   # 사이즈 추정용 관측(원 det 보존)

    def update(self, det, t):
        if det is None:
            self.time_since_update += 1; self.hit_streak = 0
            return
        box = np.asarray(det["box"], float)
        # OCM: delta_t 전 관측과의 속도방향
        prev = None
        for dt in range(self.delta_t, 0, -1):
            if t-dt in self.observations:
                prev = self.observations[t-dt]; break
        if prev is None:
            prev = self.last_observation
        self.velocity = speed_direction(prev, box)
        self.last_observation = box; self.observations[t] = box
        self.time_since_update = 0; self.hits += 1; self.hit_streak += 1
        self.kf.update(_to_z(box))
        self.obs.append((t, det))

    def predict(self):
        self.kf.predict(); self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return _to_bbox(self.kf.x[:4].ravel())

    @property
    def state_box(self):
        return _to_bbox(self.kf.x[:4].ravel())


def _associate(dets, trks, vels, last_boxes, iou_thr, inertia):
    """1차 매칭: IoU + OCM(속도 일관성). 반환 matches, um_d, um_t."""
    if len(trks) == 0 or len(dets) == 0:
        return [], list(range(len(dets))), list(range(len(trks)))
    iou = iou_batch(trks, dets)
    # OCM: track 속도방향 vs (track마지막관측->각 검출) 방향의 일치도
    cost_ocm = np.zeros_like(iou)
    for i in range(len(trks)):
        if vels[i] is None:
            continue
        for j in range(len(dets)):
            d = speed_direction(last_boxes[i], dets[j])
            cos = np.clip((d*vels[i]).sum(), -1, 1)
            cost_ocm[i, j] = (np.arccos(cos)/np.pi)  # 0(동일)~1(반대)
    score = iou - inertia*cost_ocm   # IoU 높고 방향 일치할수록 ↑
    ri, ci = linear_sum_assignment(-score)
    matches, um_d, um_t = [], [], []
    md, mt = set(), set()
    for r, c in zip(ri, ci):
        if iou[r, c] >= iou_thr:
            matches.append((c, r)); md.add(c); mt.add(r)   # (det_idx, trk_idx)
    um_d = [j for j in range(len(dets)) if j not in md]
    um_t = [i for i in range(len(trks)) if i not in mt]
    return matches, um_d, um_t


class OCSort:
    def __init__(self, iou_thr=0.3, max_age=30, min_hits=3, delta_t=3, inertia=0.2,
                 det_thresh=0.0, byte_iou=0.2):
        self.iou_thr, self.max_age, self.min_hits = iou_thr, max_age, min_hits
        self.delta_t, self.inertia = delta_t, inertia
        self.det_thresh = det_thresh
        self.byte_iou = byte_iou
        self.trackers = []
        self.finished = []   # 소멸된(퇴장) track 중 확정된 것 누적 (count용)
        self.frame = 0

    def step(self, dets, t):
        self.frame += 1
        scores = np.array([d.get("score", 1.0) for d in dets], float) if dets else np.zeros(0)
        hi = [i for i in range(len(dets)) if scores[i] >= self.det_thresh]
        lo = [i for i in range(len(dets)) if scores[i] < self.det_thresh]
        hdets = [dets[i] for i in hi]
        hboxes = np.array([d["box"] for d in hdets], float) if hdets else np.zeros((0, 4))
        preds = []
        for trk in self.trackers:
            preds.append(trk.predict())
        preds = np.array(preds) if preds else np.zeros((0, 4))
        vels = [trk.velocity for trk in self.trackers]
        last_boxes = [trk.last_observation for trk in self.trackers]

        matches, um_d, um_t = _associate(hboxes, preds, vels, last_boxes, self.iou_thr, self.inertia)
        for di, ti in matches:
            self.trackers[ti].update(hdets[di], t)

        # OCR: 미매칭 track의 마지막관측 vs 미매칭 고신뢰 검출 재연결
        if um_t and um_d:
            last = np.array([self.trackers[i].last_observation for i in um_t])
            cand = hboxes[um_d]
            iou2 = iou_batch(last, cand)
            ri, ci = linear_sum_assignment(-iou2)
            dt_, dd_ = set(), set()
            for r, c in zip(ri, ci):
                if iou2[r, c] >= self.iou_thr:
                    ti, di = um_t[r], um_d[c]
                    self.trackers[ti].update(hdets[di], t)
                    dt_.add(ti); dd_.add(di)
            um_t = [i for i in um_t if i not in dt_]; um_d = [j for j in um_d if j not in dd_]

        # BYTE-style 2차 매칭: 저신뢰 검출은 기존 track 유지에만 사용하고 신규 track은 만들지 않는다.
        if um_t and lo:
            lboxes = np.array([dets[i]["box"] for i in lo], float)
            tb = preds[um_t]
            iou3 = iou_batch(tb, lboxes)
            ri, ci = linear_sum_assignment(-iou3)
            dt_ = set()
            for r, c in zip(ri, ci):
                if iou3[r, c] >= self.byte_iou:
                    self.trackers[um_t[r]].update(dets[lo[c]], t)
                    dt_.add(um_t[r])
            um_t = [i for i in um_t if i not in dt_]

        # 미매칭 track: 관측 없음 처리(hit_streak 끊김)는 predict에서 반영됨
        # 신규 track은 고신뢰 미매칭 검출만 사용한다.
        for di in um_d:
            self.trackers.append(KalmanBoxTracker(hdets[di], t, self.delta_t))
        # 소멸: time_since_update > max_age. 확정된 건 finished에 누적(퇴장 박스도 카운트)
        alive = []
        for tr in self.trackers:
            if tr.time_since_update > self.max_age:
                if tr.hits >= self.min_hits:
                    self.finished.append(tr)
            else:
                alive.append(tr)
        self.trackers = alive

    def result(self):
        """전체 영상 unique 박스 = 퇴장 확정(finished) + 현재 확정(alive). count + obs."""
        out = list(self.finished)
        for tr in self.trackers:
            if tr.hits >= self.min_hits:
                out.append(tr)
        return out
