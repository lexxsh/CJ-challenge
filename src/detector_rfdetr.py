"""
RF-DETR 검출 ONNX 추론 (rfdetr 패키지 없이 onnxruntime + numpy 전/후처리). Docker 배포용.
출력 contract: YoloDet과 동일 -> [{"box": np.array([x1,y1,x2,y2]), "score": float}] (원본 픽셀).

전처리(rfdetr.predict와 일치): BGR->RGB -> /255 -> 정사각 resize(res,res) bilinear -> (x-mean)/std(ImageNet) -> CHW.
후처리: sigmoid(logits) -> Q*C flatten -> top-300 -> query=idx//C -> cxcywh->xyxy(정규화) -> 원본[W,H,W,H] 곱 -> conf 필터.
NMS 없음(DETR set prediction). custom op 없음(opset17 표준).
"""
import numpy as np
import cv2

try:
    import onnxruntime as ort
except Exception:
    ort = None

_MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
_STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)


class RfdetrDet:
    def __init__(self, onnx_path, resolution=1024, conf=0.5, num_select=300,
                 providers=("CUDAExecutionProvider", "CPUExecutionProvider")):
        assert ort is not None, "onnxruntime 필요"
        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        so.inter_op_num_threads = 1
        # cudnn 알고리즘을 결정적으로 고정 (기본 EXHAUSTIVE는 타이밍 벤치마크라
        # 프로세스/메모리 상태에 따라 수치가 흔들려 경계 conf 검출이 플립됨 -> count 비재현)
        provs = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}) if p == "CUDAExecutionProvider" else p
                 for p in providers]
        self.sess = ort.InferenceSession(onnx_path, sess_options=so, providers=provs)
        self.inp = self.sess.get_inputs()[0].name
        self.out = [o.name for o in self.sess.get_outputs()]   # ["dets", "labels"]
        self.res, self.conf, self.num_select = resolution, conf, num_select

    def __call__(self, img):
        H, W = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        sq = cv2.resize(rgb, (self.res, self.res), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        blob = (sq.transpose(2, 0, 1) - _MEAN) / _STD
        blob = np.ascontiguousarray(blob[None].astype(np.float32))
        dets, labels = self.sess.run(self.out, {self.inp: blob})
        boxes = dets[0]                       # [Q,4] cxcywh, 정규화
        logits = labels[0]                    # [Q,C] raw logit
        prob = 1.0 / (1.0 + np.exp(-logits))  # sigmoid
        C = prob.shape[1]
        flat = prob.reshape(-1)               # [Q*C]
        k = min(self.num_select, flat.shape[0])
        idx = np.argpartition(-flat, k - 1)[:k]   # top-k (정렬X, 집합만)
        scores = flat[idx]
        q = idx // C                          # query 인덱스 (class = idx % C 는 미사용: 단일 box)
        sel = boxes[q]                        # [k,4] cxcywh
        cx, cy, bw, bh = sel[:, 0], sel[:, 1], sel[:, 2], sel[:, 3]
        xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
        xyxy = xyxy * np.array([W, H, W, H], np.float32)   # 원본 픽셀(축별 복원)
        m = scores >= self.conf
        xyxy, scores = xyxy[m], scores[m]
        xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, W)
        xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, H)
        order = scores.argsort()[::-1]
        return [{"box": xyxy[i], "score": float(scores[i])} for i in order]
