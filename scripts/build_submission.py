"""대회 제출용 ZIP 빌드.

규격 (대회 공지):
  demo.zip
  ├── main.py           진입점 — `python main.py --input {folder}`
  ├── src/              추가 소스
  ├── train_src/        학습 코드 + README.md + requirements.txt  (AI 모델 사용 시 필수)
  ├── checkpoints/      ONNX 만 (.pt/.pth 금지)
  └── README.md

⚠️ `--input_dir` 은 비표준으로 지적받음. 빌드 후 패키지 전체에 그 문자열이 없는지 검사하고
   하나라도 있으면 빌드를 실패시킨다.
⚠️ 압축 해제 후 **격리 실행 스모크**를 반드시 할 것 — 과거 `src/pipeline.py` 가 끌어오는
   `detector.py`/`sizer.py` 누락으로 import 단계에서 죽은 적이 있다.

사용: python scripts/build_submission.py --out submission.zip
"""
import argparse, os, shutil, sys, zipfile

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.join(R, "submission.zip"))
ap.add_argument("--stage", default=os.path.join(R, "build_submit"))
ap.add_argument("--ck", default=os.path.join(R, "checkpoints"))
a = ap.parse_args()

S = a.stage
if os.path.exists(S):
    shutil.rmtree(S)
os.makedirs(f"{S}/src"); os.makedirs(f"{S}/checkpoints"); os.makedirs(f"{S}/train_src")

shutil.copy(f"{R}/main.py", f"{S}/main.py")
for f in sorted(os.listdir(f"{R}/src")):
    if f.endswith(".py"):
        shutil.copy(f"{R}/src/{f}", f"{S}/src/{f}")

CK = ["det_crop", "det_bar", "det_full", "f_tight_s1", "f_tight_s2", "f_tight_s3", "f_tight_s4",
      "f_rail_s1", "f_rail_s2", "seg_box", "belt_seg", "da3_metric_large",
      "rf_long", "rf_mid", "rf_short"]
miss = [n for n in CK if not os.path.exists(f"{a.ck}/{n}.onnx")]
if miss:
    sys.exit(f"★ 체크포인트 누락 ({a.ck}): {miss}")
for n in CK:
    shutil.copy(f"{a.ck}/{n}.onnx", f"{S}/checkpoints/{n}.onnx")

for f in sorted(os.listdir(f"{R}/train")):
    shutil.copy(f"{R}/train/{f}", f"{S}/train_src/{f}")
shutil.copy(f"{R}/train/requirements.txt", f"{S}/train_src/training-requirements.txt")
shutil.copy(f"{R}/README.md", f"{S}/README.md")

bad = []
for root, _, fs in os.walk(S):
    for f in fs:
        if f.endswith((".py", ".md", ".txt")):
            p = os.path.join(root, f)
            if "input_dir" in open(p, encoding="utf-8", errors="ignore").read():
                bad.append(os.path.relpath(p, S))
if bad:
    sys.exit(f"★ 'input_dir' 잔존: {bad}")

with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
    for root, _, fs in os.walk(S):
        for f in fs:
            p = os.path.join(root, f)
            z.write(p, os.path.relpath(p, S))
print(f"[완료] {a.out}  ({os.path.getsize(a.out)/1e6:.0f}MB)")
print("  인터페이스 검사 통과 — python main.py --input {folder}")
print("  ★ 다음: 압축 해제 후 격리 실행 스모크를 반드시 할 것")
