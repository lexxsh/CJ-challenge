"""
Dimension-head v2 ONNX 추론 래퍼 (geo 멀티특징 입력). Docker: onnxruntime.
geo: [bw/rail, bh/rail, diag/rail, log(diag*cmpp+1), cy_norm]
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


def crop_from_box(frame, box, ctx=0.1, size=224):
    x1, y1, x2, y2 = box
    cx, cy = (x1+x2)/2, (y1+y2)/2
    half = max(x2-x1, y2-y1)*(1+ctx)/2
    X1, Y1, X2, Y2 = int(cx-half), int(cy-half), int(cx+half), int(cy+half)
    H, W = frame.shape[:2]
    crop = frame[max(0, Y1):min(H, Y2), max(0, X1):min(W, X2)]
    if crop.size == 0:
        return None
    crop = cv2.copyMakeBorder(crop, max(0, -Y1), max(0, Y2-H), max(0, -X1), max(0, X2-W),
                              cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return cv2.resize(crop, (size, size))


def geo_feat(box, cmpp, H_img):
    bw = float(box[2]-box[0]); bh = float(box[3]-box[1])
    diag = math.hypot(bw, bh)
    rail = max(62.3/max(cmpp, 1e-6), 1.0)
    cy = (box[1]+box[3])/2/H_img
    return [bw/rail, bh/rail, diag/rail, math.log(diag*cmpp+1.0), cy]


class Sizer2:
    def __init__(self, onnx_path, imgsz=224, providers=("CUDAExecutionProvider", "CPUExecutionProvider")):
        assert ort is not None
        so = ort.SessionOptions(); so.intra_op_num_threads = 4; so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(onnx_path, sess_options=so, providers=list(providers))
        self.ins = [i.name for i in self.sess.get_inputs()]
        self.imgsz = imgsz

    def __call__(self, crops, geos):
        blobs = []
        for c in crops:
            if c.shape[0] != self.imgsz:
                c = cv2.resize(c, (self.imgsz, self.imgsz))
            x = c[:, :, ::-1].astype(np.float32)/255.0
            blobs.append(((x-_MEAN)/_STD).transpose(2, 0, 1))
        crop_in = np.stack(blobs).astype(np.float32)
        geo_in = np.asarray(geos, np.float32)
        return self.sess.run(["dims"], {self.ins[0]: crop_in, self.ins[1]: geo_in})[0]


class SizerSeq:
    """멀티프레임 어텐션 size (size_seq.onnx). __call__(crops, geos) -> dims(3,) (트랙 1개)."""
    def __init__(self, onnx_path, imgsz=224, providers=("CUDAExecutionProvider", "CPUExecutionProvider")):
        assert ort is not None
        so = ort.SessionOptions(); so.intra_op_num_threads = 4; so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(onnx_path, sess_options=so, providers=list(providers))
        self.imgsz = imgsz

    def __call__(self, crops, geos):
        blobs = []
        for c in crops:
            if c.shape[0] != self.imgsz:
                c = cv2.resize(c, (self.imgsz, self.imgsz))
            x = c[:, :, ::-1].astype(np.float32)/255.0
            blobs.append(((x-_MEAN)/_STD).transpose(2, 0, 1))
        crops_in = np.stack(blobs)[None].astype(np.float32)          # [1,K,3,S,S]
        geos_in = np.asarray(geos, np.float32)[None]                  # [1,K,G]
        mask_in = np.ones((1, len(crops)), np.float32)               # [1,K]
        return self.sess.run(["dims"], {"crops": crops_in, "geos": geos_in, "mask": mask_in})[0][0]
