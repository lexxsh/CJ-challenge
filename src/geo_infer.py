"""배포용 기하 스택 (ONNX + numpy만 — 컨테이너에 ultralytics/sklearn 없음).
YOLO-seg ONNX(NMS내장) numpy 후처리 + 벨트seg + DA3METRIC -> 피처 -> RF ONNX -> w,d,h
build_features2.py(학습)와 피처 정의가 반드시 동일해야 함.
"""
import numpy as np, cv2, onnxruntime as ort

MEAN = np.array([0.485, 0.456, 0.406]); STD = np.array([0.229, 0.224, 0.225])
DA3_W, DA3_H = 504, 280


def _sess(p, prov):
    so = ort.SessionOptions(); so.log_severity_level = 4
    return ort.InferenceSession(p, so, providers=prov)


def _letterbox(im, new):
    h, w = im.shape[:2]; r = min(new/h, new/w)
    nh, nw = int(round(h*r)), int(round(w*r))
    out = np.full((new, new, 3), 114, np.uint8)
    top, left = (new-nh)//2, (new-nw)//2
    out[top:top+nh, left:left+nw] = cv2.resize(im, (nw, nh))
    return out, r, left, top, nw, nh


class SegONNX:
    """YOLOv8/11-seg ONNX (출력: [1,300,38] NMS내장 + [1,32,mh,mw] protos)."""
    def __init__(self, path, imgsz, prov, conf=0.25):
        self.s = _sess(path, prov); self.imgsz = imgsz; self.conf = conf
        self.inp = self.s.get_inputs()[0].name

    def __call__(self, frame):
        H, W = frame.shape[:2]
        im, r, dx, dy, nw, nh = _letterbox(frame, self.imgsz)
        x = (im[:, :, ::-1].astype(np.float32)/255.).transpose(2, 0, 1)[None]
        d0, protos = self.s.run(None, {self.inp: x})
        d0 = d0[0]; protos = protos[0]                      # [300,38], [32,mh,mw]
        keep = d0[:, 4] > self.conf
        d0 = d0[keep]
        if len(d0) == 0:
            return []
        boxes_lb = d0[:, :4]                                # letterbox 좌표
        coef = d0[:, 6:]                                    # [n,32]
        C, mh, mw = protos.shape
        m = 1/(1+np.exp(-(coef @ protos.reshape(C, -1))))   # [n, mh*mw]
        m = m.reshape(-1, mh, mw)
        out = []
        for i in range(len(d0)):
            mk = cv2.resize(m[i], (self.imgsz, self.imgsz))
            # bbox 밖은 0 (ultralytics crop_mask와 동일)
            bx = boxes_lb[i]
            cm = np.zeros_like(mk)
            x1, y1, x2, y2 = [int(v) for v in bx]
            x1, y1 = max(x1, 0), max(y1, 0); x2, y2 = min(x2, self.imgsz), min(y2, self.imgsz)
            cm[y1:y2, x1:x2] = mk[y1:y2, x1:x2]
            cm = cm[dy:dy+nh, dx:dx+nw]                     # 패딩 제거
            cm = cv2.resize(cm, (W, H)) > 0.5
            bo = np.array([(bx[0]-dx)/r, (bx[1]-dy)/r, (bx[2]-dx)/r, (bx[3]-dy)/r])
            out.append((cm, bo, float(d0[i, 4])))
        return out


class DA3:
    def __init__(self, path, prov):
        self.s = _sess(path, prov); self.inp = self.s.get_inputs()[0].name

    def __call__(self, frame):
        H, W = frame.shape[:2]
        img = cv2.resize(frame[:, :, ::-1], (DA3_W, DA3_H)).astype(np.float32)/255.
        img = (img-MEAN)/STD
        d = self.s.run(None, {self.inp: img.transpose(2, 0, 1)[None].astype(np.float32)})[0].squeeze()
        return cv2.resize(d, (W, H))


class GeoSizer:
    """트랙 박스 -> (seg마스크, 벨트, depth) -> 피처 -> RF -> [long,mid,short] cm."""
    def __init__(self, seg_p, belt_p, da3_p, rf_paths, prov):
        self.seg = SegONNX(seg_p, 1280, prov)
        self.belt = SegONNX(belt_p, 960, prov)
        self.da3 = DA3(da3_p, prov)
        self.rf = [_sess(p, ["CPUExecutionProvider"]) for p in rf_paths]

    def frame_ctx(self, frame, crop=None, zoom_seg=False, zoom_depth=False):
        """프레임 1회 계산: 벨트마스크, depth, 박스마스크들.

        ★ 줌 옵션 (crop=(x0,y0,x1,y1), 보통 벨트 크롭):
          검출기에서 full->belt-crop 줌이 리더보드 -1.16 을 만든 그 메커니즘을 기하스택에도 적용.
          - zoom_seg  : seg_box 를 크롭에서 실행 (1280 입력에 박스가 더 크게 담김)
          - zoom_depth: DA3 를 크롭에서 실행 (입력 504x280 고정 -> 전체프레임은 0.39배 축소라 특히 거침)
          출력은 항상 **전체프레임 좌표**로 되돌리므로 feats/dims 는 그대로 동작.
        """
        bl = self.belt(frame)
        if not bl:
            return None
        bm = np.zeros(frame.shape[:2], bool)
        for m, _, _ in bl:
            bm |= m
        H, W = frame.shape[:2]

        if crop is not None and (zoom_seg or zoom_depth):
            x0, y0, x1, y1 = [int(v) for v in crop]
            x0, y0 = max(x0, 0), max(y0, 0); x1, y1 = min(x1, W), min(y1, H)
            sub = frame[y0:y1, x0:x1]
            if sub.shape[0] < 16 or sub.shape[1] < 16:
                crop = None

        if crop is None or not zoom_seg:
            boxes = self.seg(frame)
        else:
            boxes = []
            for m, bo, sc in self.seg(sub):
                fm = np.zeros((H, W), bool); fm[y0:y1, x0:x1] = m      # 크롭 마스크 -> 전체프레임
                boxes.append((fm, np.array([bo[0]+x0, bo[1]+y0, bo[2]+x0, bo[3]+y0]), sc))

        if crop is None or not zoom_depth:
            depth = self.da3(frame)
        else:
            dsub = self.da3(sub)                                        # 크롭에서 depth
            depth = np.zeros((H, W), np.float32); depth[y0:y1, x0:x1] = dsub
        return {"belt": bm, "depth": depth, "boxes": boxes}

    def feats(self, ctx, box):
        """★ 학습(build_features3)과 추론이 공유하는 단일 피처 함수 (train/serve skew=0).
        반환: [maj_cm, min_cm, bw_cm, bh_cm, area_cm2, cy_norm, ratio, maj*r, min*r, zspan, zbox]"""
        if ctx is None or not ctx["boxes"]:
            return None
        bw, bh = box[2]-box[0], box[3]-box[1]; bcy = (box[1]+box[3])/2
        best = None; bd = 1e9
        for m, bo, _ in ctx["boxes"]:
            w, h = bo[2]-bo[0], bo[3]-bo[1]; cy = (bo[1]+bo[3])/2
            dd = abs(w-bw)+abs(h-bh)+abs(cy-bcy)*0.5
            if dd < bd:
                bd = dd; best = m
        if best is None or bd > 80 or best.sum() < 30:
            return None
        depth, belt = ctx["depth"], ctx["belt"]
        H = depth.shape[0]
        ys, xs = np.where(best)
        P = np.stack([xs, ys], 1).astype(np.float32); Cc = P-P.mean(0)
        _, _, Vt = np.linalg.svd(Cc, full_matrices=False)
        pr = Cc @ Vt.T
        maj = float(np.percentile(pr[:, 0], 97)-np.percentile(pr[:, 0], 3))
        mnr = float(np.percentile(pr[:, 1], 97)-np.percentile(pr[:, 1], 3))
        cy = float(ys.mean())
        # 벨트폭(=62.3cm) at 박스 행
        y0, y1 = max(int(cy-20), 0), min(int(cy+20), H)
        ws = [belt[y].sum() for y in range(y0, y1) if belt[y].sum() > 5]
        if not ws:
            return None
        belt_px = float(np.median(ws))
        if belt_px < 10:
            return None
        cmpp = 62.3/belt_px
        zb = float(np.median(depth[best]))
        zb10 = float(np.percentile(depth[best], 10)); zb90 = float(np.percentile(depth[best], 90))
        band = belt.copy(); band[:y0] = False; band[y1:] = False; band &= ~best
        if band.sum() < 30:
            return None
        zr = float(np.median(depth[band]))
        if zr <= 0 or zb <= 0:
            return None
        ratio = zb/zr
        maj_cm, min_cm = maj*cmpp, mnr*cmpp
        return np.array([maj_cm, min_cm, float(xs.max()-xs.min())*cmpp,
                         float(ys.max()-ys.min())*cmpp, float(len(xs))*cmpp**2, cy/H,
                         ratio, maj_cm*ratio, min_cm*ratio, zb90-zb10, zb], np.float32)

    def dims(self, ctx, box):
        """기하 치수 [long,mid,short] cm. 실패 시 None."""
        f = self.feats(ctx, box)
        if f is None:
            return None
        x = f[None].astype(np.float32)
        return np.array([float(s.run(None, {s.get_inputs()[0].name: x})[0].ravel()[0])
                         for s in self.rf])
