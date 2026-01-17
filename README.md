# GRU_k-step 파이프라인 정리

본 프로젝트는 ICU 환자의 시계열 데이터를 이용해 **현재 시점 기준 향후 H시간 내 이벤트 발생 위험도**를 예측하는 모델 학습/평가 파이프라인입니다.  
모델은 미래 정보를 직접 사용하지 않고, 과거 관측된 시계열만으로 위험도를 계산합니다.

## 파이프라인 개요
1) 데이터 로딩  
- `data/train_processed.parquet`, `data/valid_processed.parquet`, `data/test_processed.parquet` 로드
- 스키마 검증: `stay_id`, `t`, `event` 및 feature 컬럼 존재 확인

2) 라벨 준비 및 검열 처리  
- 라벨 정의  
  - `_future_label`: 현재 시점 `t` 이후 `(t, t+H]` 구간 내 `event == 1` 발생 여부  
  - `_label_observable`: 미래 H구간을 끝까지 관측했는지 여부  
- 학습/평가에는 `_label_observable == 1` 인 행만 사용  
- `drop_after_event=True`면 이벤트 최초 발생 시점 이후 행 제거(누수 방지)

3) 전처리  
- 결측치 처리: train 평균값으로 impute  
- 표준화: train 평균/표준편차 기준 standardize  
- stay_id별 시퀀스 구성 (시간순 정렬)
- 시퀀스 길이가 `max_len`을 초과하면 앞부분만 사용

4) 모델 학습  
- 모델: `Timewise GRU` (시점별 위험도 로짓 출력)  
- Loss: `BCEWithLogits` + class imbalance 대응 `pos_weight`  
- padding 구간은 mask로 loss 제외  
- AdamW, early stopping 적용

5) 평가  
- Row-level: AUC/AP (모든 valid/test 행)  
- Stay-level: `cutoff_hours` 내 row 점수를 `max/mean/last`로 집계  
- threshold는 valid에서 `target_recall` 이상인 지점 중 precision 최대 선택  
- test는 valid에서 선택한 threshold로 precision/recall/f1 계산

---

## 산출물 (어떤 걸 알 수 있나)

### 모델 및 메타데이터 (artifacts/)
- `artifacts/kstep_gru_state.pt`  
  - 모델 가중치 (서비스 배포 시 이 파일 사용)
- `artifacts/kstep_gru_meta.json`  
  - 사용된 feature 목록  
  - impute/standardize 통계  
  - 학습 설정 및 성능 요약  
  - threshold 및 precision/recall/f1
- `artifacts/training_history.json`  
  - epoch별 loss/metric 기록

### 시각화/리포트 (output/)
- `row_metrics.csv` / `stay_metrics.csv`  
  - row/stay 성능 지표 요약
- `row_metrics_table.png` / `stay_metrics_table.png`  
  - PPT에 바로 넣을 수 있는 테이블 이미지
- `training_loss.png` / `valid_row_auc_ap.png` / `valid_stay_auc_ap.png` / `stay_metrics.png`  
  - 학습곡선 및 성능 시각화

이를 통해 **모델 성능(ROC-AUC/AP/F1/Recall/Precision)**, **threshold**, **전처리 통계**를 확인할 수 있습니다.

---

## 실행 방법
```bash
python main.py
```

환경변수 `DATA_DIR`를 설정하면 다른 데이터 디렉토리를 사용할 수 있습니다.  
예: `DATA_DIR=/path/to/data python main.py`

---

## 주요 설정 (src/config.py)
- `data.horizon_hours`: 미래 라벨 H  
- `data.cutoff_hours`: stay-level 집계 cutoff  
- `data.agg_mode`: stay 집계 방식 (`max`, `mean`, `last`)  
- `sequence.max_len`: 시퀀스 최대 길이  
- `data.use_precomputed_labels`: 데이터에 저장된 `_future_label`, `_label_observable` 사용 여부  
- `data.recompute_labels`: event 기반 재계산 여부

---

## 서비스 적용 시 주의사항
- 실서비스는 **현재 시점의 row_score**를 위험도로 사용  
- 전처리는 반드시 **train 기준 통계**를 사용 (impute/standardize)  
- feature 컬럼 순서/구성은 `kstep_gru_meta.json`의 `feature_cols`와 동일해야 함
