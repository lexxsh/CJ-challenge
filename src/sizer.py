"""
Dimension-head ONNX 추론 래퍼 (Docker: onnxruntime, ultralytics 불필요).
박스 크롭 + 레일대비 스케일 -> (정렬된 w,d,h) cm.
"""
import numpy as np
import cv2
import math

try:
    import onnxruntime as ort
except Exception:
    ort = None

_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


def crop_box(frame, mask, ctx=0.1, size=192):  # 타이트 크롭(학습과 동일)
    ys, xs = np.where(mask)
    if len(xs) < 20:
        return None
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(x2 - x1, y2 - y1) * (1 + ctx) / 2
    X1, Y1, X2, Y2 = int(cx - half), int(cy - half), int(cx + half), int(cy + half)
    H, W = frame.shape[:2]
    crop = frame[max(0, Y1):min(H, Y2), max(0, X1):min(W, X2)]
    if crop.size == 0:
        return None
    crop = cv2.copyMakeBorder(crop, max(0, -Y1), max(0, Y2 - H), max(0, -X1), max(0, X2 - W),
                              cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return cv2.resize(crop, (size, size))


class Sizer:
    def __init__(self, onnx_path, providers=("CUDAExecutionProvider", "CPUExecutionProvider")):
        assert ort is not None
        so = ort.SessionOptions(); so.intra_op_num_threads = 4; so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(onnx_path, sess_options=so, providers=list(providers))
        self.ins = [i.name for i in self.sess.get_inputs()]

    def __call__(self, crops, scales):
        """crops: list[HxWx3 BGR 192], scales: list[float] -> (N,3) dims."""
        blobs = []
        for c in crops:
            x = c[:, :, ::-1].astype(np.float32) / 255.0
            x = (x - _MEAN) / _STD
            blobs.append(x.transpose(2, 0, 1))
        crop_in = np.stack(blobs).astype(np.float32)
        scale_in = np.array([[math.log(max(s, 1e-3) + 1.0)] for s in scales], np.float32)
        out = self.sess.run(["dims"], {self.ins[0]: crop_in, self.ins[1]: scale_in})[0]
        return out  # (N,3) 정렬 치수
