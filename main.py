"""제출 진입점.

    python main.py --input {video_folder}

폴더 내 모든 .mp4 를 추론해 result.json 을 main.py 위치에 생성.
onnxruntime-gpu 만 사용, 네트워크/외부 다운로드 없음.

구성
  검출: RF-DETR 3종(crop/bar/full) — 각자 학습과 동일한 크롭 규칙으로 추론 후
        영상별 트랙 수의 중앙값에 해당하는 모델의 트랙을 채택
  추적: OC-SORT (conf 0.65 / iou 0.15 / max_age 20 / min_hits 2)
  치수: CNN(tight×4 + rail×2) × 0.2 + 기하스택(seg_box + belt_seg + DA3 -> RF) × 0.8
        seg_box 는 벨트 크롭에서 실행(줌) — 작은 박스 마스크 정확도 -> z_box 개선
        기하스택 정식화: S = 62.3 × (box_px/rail_px) × (z_box/z_rail) × 학습잔차
"""
import argparse, glob, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from src.detector_rfdetr import RfdetrDet
from src.sizer2 import Sizer2
from src.geo_infer import GeoSizer, SegONNX
from src import pipeline2, pipeline3

PROV = ["CUDAExecutionProvider", "CPUExecutionProvider"]
CK = os.path.join(HERE, "checkpoints")   # .gitignore 대상 — README 의 재현 절차로 생성


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", dest="video_dir", required=True,
                    help="추론할 .mp4 들이 있는 폴더")
    ap.add_argument("--out", default=os.path.join(HERE, "result.json"))
    ap.add_argument("--conf", type=float, default=0.65)
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--best_k", type=int, default=3)
    ap.add_argument("--min_hits", type=int, default=2)
    ap.add_argument("--max_age", type=int, default=20)
    ap.add_argument("--iou_thr", type=float, default=0.15)
    ap.add_argument("--alpha_geo", type=float, default=0.8)   # seg 줌 기하의 정직 OOF 최적 (11.3753 @0.8)
    ap.add_argument("--single", action="store_true", help="검출기 1종만 사용(디버그용)")
    a = ap.parse_args()

    # ---- 검출기 ----
    dets3 = None
    det = None
    if a.single:
        det = RfdetrDet(os.path.join(CK, "det_crop.onnx"), conf=a.conf, resolution=a.resolution)
        print("[det] 단일 (crop)", flush=True)
    else:
        dets3 = [(ds, RfdetrDet(os.path.join(CK, f"det_{ds}.onnx"), conf=a.conf,
                                resolution=a.resolution))
                 for ds in ["crop", "bar", "full"]
                 if os.path.exists(os.path.join(CK, f"det_{ds}.onnx"))]
        if not dets3:
            sys.exit("검출기 ONNX 없음: checkpoints/det_{crop,bar,full}.onnx")
        print(f"[det] {len(dets3)}종 median: {[d[0] for d in dets3]} (conf {a.conf})", flush=True)

    # ---- 치수 ----
    tight = [Sizer2(os.path.join(CK, f"f_tight_s{i}.onnx"), providers=PROV) for i in [1, 2, 3, 4]
             if os.path.exists(os.path.join(CK, f"f_tight_s{i}.onnx"))]
    rail = [Sizer2(os.path.join(CK, f"f_rail_s{i}.onnx"), providers=PROV) for i in [1, 2]
            if os.path.exists(os.path.join(CK, f"f_rail_s{i}.onnx"))]
    belt_seg = SegONNX(os.path.join(CK, "belt_seg.onnx"), 960, PROV, conf=0.25) if dets3 else None
    geo = GeoSizer(os.path.join(CK, "seg_box.onnx"), os.path.join(CK, "belt_seg.onnx"),
                   os.path.join(CK, "da3_metric_large.onnx"),
                   [os.path.join(CK, f"rf_{n}.onnx") for n in ["long", "mid", "short"]], PROV)
    print(f"[size] CNN tight×{len(tight)} + rail×{len(rail)} + 기하스택(alpha={a.alpha_geo})", flush=True)

    vids = sorted(glob.glob(os.path.join(a.video_dir, "*.mp4")))
    print(f"[input] {a.video_dir}: {len(vids)}개 영상", flush=True)
    out = {"videos": []}
    for i, vp in enumerate(vids):
        vid = os.path.splitext(os.path.basename(vp))[0]
        try:
            if dets3 is not None:
                res = pipeline3.process_video(vp, dets3, tight, rail, geo, belt_seg=belt_seg,
                                              frame_stride=a.stride, best_k=a.best_k,
                                              min_hits=a.min_hits, max_age=a.max_age,
                                              iou_thr=a.iou_thr, alpha_geo=a.alpha_geo)
            else:
                res = pipeline2.process_video(vp, det, tight, rail, geo, frame_stride=a.stride,
                                              best_k=a.best_k, min_hits=a.min_hits,
                                              max_age=a.max_age, iou_thr=a.iou_thr,
                                              alpha_geo=a.alpha_geo)
            objs = res["objects"]
        except Exception as e:
            print(f"[WARN] {vid}: {e}", file=sys.stderr)
            objs = []
        out["videos"].append({"video_id": vid, "objects": objs})
        print(f"[{i+1}/{len(vids)}] {vid}: {len(objs)} boxes", flush=True)
    json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=2)
    print(f"저장: {a.out}", flush=True)


if __name__ == "__main__":
    main()
