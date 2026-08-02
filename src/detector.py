"""
YOLO11 detection ONNX 추론 (ultralytics 없이 onnxruntime + 직접 디코드).
출력 contract: [1, 4+nc, 8400] (xywh + class scores). 단일 'box' 클래스.
"""
import numpy as np
import cv2

try:
    import onnxruntime as ort
except Exception:
    ort = None


def letterbox(img, new_shape=1280, color=(114, 114, 114)):
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    out = np.full((new_shape, new_shape, 3), color, np.uint8)
    top, left = (new_shape - nh) // 2, (new_shape - nw) // 2
    out[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh))
    return out, r, left, top


def _nms(boxes, scores, iou_thr=0.6):
    try:
        import torch
        from torchvision.ops import nms
        keep = nms(torch.as_tensor(boxes, dtype=torch.float32),
                   torch.as_tensor(scores, dtype=torch.float32), iou_thr)
        return keep.cpu().numpy()
    except Exception:
        idx = scores.argsort()[::-1]; keep = []
        while len(idx):
            i = idx[0]; keep.append(i)
            if len(idx) == 1:
                break
            xx1 = np.maximum(boxes[i, 0], boxes[idx[1:], 0]); yy1 = np.maximum(boxes[i, 1], boxes[idx[1:], 1])
            xx2 = np.minimum(boxes[i, 2], boxes[idx[1:], 2]); yy2 = np.minimum(boxes[i, 3], boxes[idx[1:], 3])
            w = np.maximum(0, xx2 - xx1); h = np.maximum(0, yy2 - yy1); inter = w * h
            ai = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            ao = (boxes[idx[1:], 2] - boxes[idx[1:], 0]) * (boxes[idx[1:], 3] - boxes[idx[1:], 1])
            ov = inter / (ai + ao - inter + 1e-9)
            idx = idx[1:][ov < iou_thr]
        return np.array(keep, dtype=int)


class YoloDet:
    def __init__(self, onnx_path, imgsz=1280, conf=0.25, iou=0.6,
                 providers=("CUDAExecutionProvider", "CPUExecutionProvider")):
        assert ort is not None
        so = ort.SessionOptions()
        so.intra_op_num_threads = 4   # affinity 경고 방지(클러스터 노드)
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(onnx_path, sess_options=so, providers=list(providers))
        self.inp = self.sess.get_inputs()[0].name
        self.out = [o.name for o in self.sess.get_outputs()]
        self.imgsz, self.conf, self.iou = imgsz, conf, iou

    def __call__(self, img):
        H, W = img.shape[:2]
        lb, r, padx, pady = letterbox(img, self.imgsz)
        blob = np.ascontiguousarray(lb[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0)
        pred = self.sess.run(self.out, {self.inp: blob})[0][0].transpose(1, 0)  # [N, 4+nc]
        nc = pred.shape[1] - 4
        conf = pred[:, 4:4 + nc].max(1)
        m = conf >= self.conf
        if m.sum() == 0:
            return []
        xywh = pred[m, :4]; conf = conf[m]
        xy, wh = xywh[:, :2], xywh[:, 2:]
        xyxy = np.concatenate([xy - wh / 2, xy + wh / 2], 1)
        keep = _nms(xyxy, conf, self.iou)
        xyxy, conf = xyxy[keep], conf[keep]
        xyxy[:, [0, 2]] = ((xyxy[:, [0, 2]] - padx) / r).clip(0, W)
        xyxy[:, [1, 3]] = ((xyxy[:, [1, 3]] - pady) / r).clip(0, H)
        return [{"box": xyxy[i], "score": float(conf[i])} for i in range(len(xyxy))]
