# 학습 — 재현 순서

평가 Docker 와 무관한 별도 환경에서 학습했습니다. 파일은 실행 순서대로 번호를 붙였습니다.

## 환경

```bash
pip install -r train/requirements.txt        # 치수 CNN · 기하 회귀기
# 검출기(RF-DETR)는 별도 환경: torch 2.6.0+cu124 + rfdetr
```

스크립트는 **레포 루트에서 실행**하는 것을 전제로 상대경로(`datasets/...`)를 씁니다.

## 데이터

- 제공 train 100영상 + public test 50영상 (총 150영상)
- **치수 정답(size_cm)은 train 100영상에만** 존재 → 치수 학습은 train 100영상만 사용
- 2D 박스 라벨은 150영상 모두 자체 생성

---

## 1. 2D 박스 라벨링

**`01_label_auto.py`** — teacher 모델로 프레임별 2D 박스 자동 라벨 (train 100영상, stride 5, 3244장)

이후 **150영상 × 10프레임 = 1500장(9071 박스)을 사람이 직접 검수·수정**했습니다.

> ⚠️ 자동 라벨과 검수 라벨은 **기준이 달라 섞으면 성능이 떨어집니다**(실측 확인).
> 두 데이터를 합쳐 학습했더니 test 50 count MAE 가 0.240/0.280 → **0.520 으로 악화**했습니다.
> 검출기는 검수 라벨만 씁니다.

## 2. 검출기 데이터셋

**`02_blackbar.py`** — 크롭 경계 규칙. **학습과 추론이 이 함수를 공유**해 train/serve skew 를 없앱니다.

| 모드 | 규칙 |
|---|---|
| `crop` | 벨트 세그 기반 경계 (+margin 150) |
| `bar` | 레터박스 검은 띠 제거 |
| `full` | 원본 전체 |

**`03_split_det3.py`** — 위 3가지 크롭으로 데이터셋 생성 (박스 손실 없음)
**`04_make_folds.py`** — **영상 단위** 5-fold 정의 (crowding × source 층화).
검출기·치수·평가가 **이 정의 하나를 공유**합니다. 프레임 단위로 나누면 같은 영상이 학습과 평가에 동시에 들어가 누수가 생깁니다.

## 3. 검출기 학습 → `det_{crop,bar,full}.onnx`

**`05_train_rfdetr.py`** — RF-DETR(medium) 3종을 각 크롭 도메인으로 학습

```bash
python train/05_train_rfdetr.py --fold 0 --ds merged --size medium \
    --epochs 10 --resolution 1008 --batch 4 --accum 4 --out runs/rf_crop
```

- 해상도 1008, batch 4 × accum 4, 10 epoch (mAP 기준 **ep4~5 에서 포화** 확인)
- 추론용 ONNX 는 32 배수인 **1024** 로 내보냅니다 (RF-DETR predict 제약)

## 4. 치수 데이터셋

**`06_dump_tracks.py`** — 검출기 3종을 각자 크롭 규칙으로 추론 → OC-SORT →
영상별 트랙 수의 중앙값에 해당하는 모델의 박스를 채택 (**추론 파이프라인과 동일한 규칙**)

**`07_build_size_ds.py`** — 트랙별 크롭 저장 + 트랙↔GT 헝가리안 매칭(축순열 최소)

| 뷰 | 내용 |
|---|---|
| `tight` | 박스에 맞춘 크롭 |
| `rail` | 레일이 함께 보이는 크롭 (스케일 기준 제공) |

## 5. 치수 CNN → `f_tight_s1~4.onnx`, `f_rail_s1~2.onnx`

**`08_size_net.py`** — 모델 정의. ResNet18(ImageNet) + 기하벡터 MLP → 평균치수 + **잔차**

```
크롭 224² ─→ ResNet18 ─→ 512 ┐
                             ├─→ 256 ─→ Dropout(0.4) ─→ 3
기하벡터 5 ─→ MLP ─→ 64      ┘        출력 = mean_dims + residual
```

기하벡터 `[bw/rail, bh/rail, diag/rail, log(diag·cmpp+1), cy_norm]` 가 **intrinsic-free 의 핵심**입니다.
전부 레일 기준 비율이라 초점거리가 소거되고, 절대 스케일은 레일(62.3cm)이 물리적 자 역할을 합니다.

**`09_train_size_cnn.py`** / **`10_train_size_final.py`**

```bash
python train/10_train_size_final.py --data datasets/size_tight \
    --out checkpoints/f_tight_s1.onnx --epochs 8 \
    --erase 0.4 --distmatch 0.5 --dm_tail 2.0
```

tight 시드 4개 + rail 시드 2개 = 6 멤버. 8 epoch 은 정직 OOF 로 확정한 값입니다(ep40 은 과적합).

## 6. 기하스택 → `rf_{long,mid,short}.onnx`

```
S = 62.3 × (box_px / rail_px) × (z_box / z_rail) × 학습잔차
```

**`11_build_geo_features.py`** — 박스별 11 피처 생성

```bash
ZOOM_SEG=1 VDUMP=vdump.json OUT=features.json python train/11_build_geo_features.py
```

- `ZOOM_SEG=1` — seg_box 를 **벨트 크롭에서 실행**. 작은 박스 마스크 정확도가 올라갑니다
- `z_box` 는 seg 마스크 픽셀만으로 산출 (bbox 중앙값은 벨트가 섞여 신호가 죽음)
- 피처 함수는 `src/geo_infer.py` 에 있고 **학습과 추론이 공유**합니다

**`12_train_geo_rf.py`** — RandomForest(300 trees, max_depth 6) 3축 → ONNX

```bash
python train/12_train_geo_rf.py --cv    --feats features.json   # 정직 OOF 확인
python train/12_train_geo_rf.py --final --feats features.json --out checkpoints
```

## 최종 배포 구성

```
검출  RF-DETR 3종 median (conf 0.65)
추적  OC-SORT (iou 0.15 / max_age 20 / min_hits 2)
치수  CNN(tight×4 + rail×2) × 0.2 + 기하스택 × 0.8
```
