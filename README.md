# CCTV 영상 기반 화물 크기 추정

컨베이어 CCTV 영상만으로 화물 박스의 **가로·세로·높이(cm)** 를 추정합니다.
카메라 intrinsic 이 주어지지 않는 조건에서, **레일 폭 62.3cm** 를 유일한 절대 기준으로 스케일을 복원합니다.

> **대회 최종 2위** — 최종 채점은 비공개 Hidden Test 150영상.
> 공개 리더보드 진행: `12.187` → `12.097`(기하스택 도입) → `10.937`(검출기 교체)
> ⚠️ 마지막 −1.16 은 실력 향상이 아니라 **공개 test 50영상을 검출기 학습에 포함**해서 생긴 값입니다.
> 자세한 내용은 [docs/findings.md](docs/findings.md) — 이 레포에서 가장 값진 문서입니다.

## 문제

| | |
|---|---|
| 입력 | 합성(Blender) 컨베이어 CCTV 영상 (10초, 1280×720) |
| 출력 | 영상별 박스 목록의 w/d/h (cm) |
| 채점 | 부피 내림차순 정렬 → 앞쪽 (0,0,0) 패딩 → rank 매칭 → 축 permutation 최소 오차 |
| 제약 | test 에 카메라 intrinsic 없음 · ONNX 만 · 네트워크 차단 · 추가 pip install 금지 |

**채점 방식의 함정**: 박스 개수를 하나 틀리면 앞쪽 패딩 때문에 **최대 박스와 잘못 매칭되고 전체 순위가 밀립니다.**
오답 영상 1개당 평균 7.9점 손해 — 그래서 치수 정확도만큼 **개수(count)** 가 중요합니다.

## 파이프라인

```
영상 ─┬─ RF-DETR ×3 (belt-crop / black-bar / full-frame)
      │      └─ 각자 학습과 동일한 크롭 규칙으로 추론 → 전체프레임 좌표로 환산
      ├─ OC-SORT 추적 → 영상별 트랙 수의 중앙값에 해당하는 모델을 채택
      └─ 트랙별 치수
            ├─ CNN ×6   (tight ×4 + rail ×2)          ── 외형 학습
            └─ 기하스택 (seg + belt + DA3 → RandomForest) ── 물리 측정
                  최종 = CNN × 0.2 + 기하 × 0.8
```

**기하스택 정식화**

```
S = 62.3 × (box_px / rail_px) × (z_box / z_rail) × 학습잔차
```

- 레일 폭(62.3cm)에 대한 **픽셀 비율**이라 초점거리가 소거됩니다 → intrinsic-free
- `z_box` 는 **seg 마스크 픽셀만**으로 산출합니다. bbox 중앙값을 쓰면 벨트가 섞여 신호가 죽습니다
- seg 는 **벨트 크롭에서 실행**합니다(줌). 작은 박스의 마스크 정확도가 올라갑니다

**CNN 도 intrinsic-free**: ResNet18 특징 + 레일 기준 기하벡터
`[bw/rail, bh/rail, diag/rail, log(diag·cmpp+1), cy_norm]` → 평균치수 + 잔차

## 사용

```bash
pip install -r requirements.txt
python main.py --input {video_folder}     # → result.json
```

`checkpoints/` 의 ONNX 15개가 필요합니다(레포에서 제외 — 아래 재현 절차 참고).

```
checkpoints/
  det_crop.onnx  det_bar.onnx  det_full.onnx      # RF-DETR 검출기 3종
  f_tight_s1~4.onnx  f_rail_s1~2.onnx             # 치수 CNN 6개
  seg_box.onnx  belt_seg.onnx                     # 세그멘테이션
  da3_metric_large.onnx                           # Depth Anything V3 (metric)
  rf_long.onnx  rf_mid.onnx  rf_short.onnx        # 기하스택 회귀기
```

## 재현

`train/` 의 스크립트를 번호 순서대로 실행합니다. 상세는 [train/README.md](train/README.md).

```
01 자동 2D 라벨          →  02 크롭 규칙  →  03 데이터셋(crop/bar/full)  →  04 영상단위 5-fold
05 RF-DETR ×3 학습       →  det_{crop,bar,full}.onnx
06 트랙 덤프             →  07 크롭 + GT 헝가리안 매칭
08~10 치수 CNN           →  f_tight_s1~4.onnx, f_rail_s1~2.onnx
11~12 기하 피처 + RF     →  rf_{long,mid,short}.onnx
```

2D 박스 라벨은 자동 생성 후 **150영상 × 10프레임 = 1500장(9071 박스)을 사람이 검수**했습니다.
자동 라벨과 검수 라벨은 기준이 달라 **섞어 학습하면 성능이 떨어집니다**(실측 확인) — 검출기는 검수 라벨만 씁니다.

`seg_box` / `belt_seg` / `da3_metric_large` 는 외부 사전학습 모델을 파인튜닝하거나 그대로 쓴 것입니다.

## 검증

이 프로젝트에서 가장 중요했던 부분입니다. 리더보드는 제출 횟수 제한(3회/일)이 있고
공개 test 를 학습에 포함한 뒤로는 신호가 되지 못해, **로컬 정직 OOF 가 유일한 나침반**이었습니다.

```bash
python eval/cv_honest_oof.py --data datasets/size3_tight --full_gt   # 영상단위 GroupKFold(5)
python eval/blend_eval.py                                            # 배포 구성 그대로 채점
python eval/scorer.py                                                # 공식 지표 재현
```

- `cv_honest_oof.py` — outer test fold 는 학습·에폭선택 어디에도 쓰지 않습니다(inner val 별도).
  `--full_gt` 는 GT 를 전 물체로 놓아 **미검출을 벌점에 반영**합니다(배포 조건과 일치).
- `blend_eval.py` — **CNN+기하 혼합 + 기하 없는 트랙은 CNN 폴백**까지 배포 그대로 재현합니다.
  성분을 단독으로 재면 반복해서 틀린 결론이 나왔습니다([docs/findings.md](docs/findings.md) 참고).

**노이즈 기준**: 시드 std ~0.06, 50영상 count ±0.1, 90영상 count ±0.07 → **0.15 미만 차이는 읽지 않습니다.**

## 성능

정직 OOF(train 100영상, 미검출 벌점 포함, 낮을수록 좋음):

| 구성 | scorer |
|---|---|
| 신호 0 기준선 (모든 박스에 GT 평균) | 19.140 |
| CNN 단독 | 12.06 |
| 기하스택 단독 | 11.76 |
| **CNN + 기하 혼합** | **11.39** |

count (검출+추적, 낮을수록 좋음):

| | MAE | 정확 |
|---|---|---|
| fold OOF (90영상) | 0.233 | 70/90 |
| test 50영상 (out-of-sample) | 0.240 | 39/50 |

## 구조

```
main.py            추론 진입점
src/               추론 모듈 (검출·추적·치수·기하스택·크롭규칙)
train/             학습 12단계 + README
eval/              정직 OOF 하네스, 공식 scorer, 혼합 평가
scripts/           제출 패키지 빌드
docs/findings.md   실험 기록 — 무엇이 통했고 무엇이 통하지 않았나
```

## 알게 된 것

전체는 [docs/findings.md](docs/findings.md) 에 있습니다. 요약하면:

- **우리 치수 오차는 보정 가능한 편향이 아니라 정보 부족입니다.** 예측의 분산 수축비가
  상관계수와 거의 같은데(0.79/0.76/0.71 vs r=0.746/0.739/0.671), 이는 회귀 감쇠 이론상
  **MSE 최적 상태**입니다. 그래서 분산 매칭·분위 매핑 같은 보정은 전부 점수를 악화시켰습니다.
- **성분 단독 지표를 믿으면 안 됩니다.** 기하스택 단독 −0.19 개선이 혼합 후에는 0 이었습니다.
  CNN 이 이미 갖고 있는 정보였기 때문입니다.
- **작은 박스가 count 병목입니다.** 0-50px recall 66%(다른 구간 95~99%), YOLO 와 RF-DETR 이
  똑같은 66% → 아키텍처가 아니라 학습 신호/정보의 문제입니다.
