# SP4SCA
Selective Prediction for Single-Cell Annotation under Biological Shift.

## V1 pipeline (paper-oriented)
`V1` hiện có pipeline tối giản đúng trục selective prediction:
- Train được base annotator trên reference split (`train_ref`).
- Tính 3 họ score: `confidence`, `support`, `shift` (+ `combined`).
- Calibrate threshold `tau` trên `calib_query` theo target risk `alpha`.
- Report đầy đủ:
  - risk-coverage curve
  - coverage@risk và risk@coverage
  - worst-group risk + coverage disparity
  - OOD AUROC/AUPRC (held-out cell types qua `is_ood`)

## Cấu trúc code V1
- `src/data/`: preprocess + split contract (`split`, `is_ood`).
- `src/models/`: embedding (`scvi`/`pca`) + classifier MLP + prediction.
- `src/scores/`: confidence/support/shift/normalize/combine.
- `src/calibration/`: threshold scan + weighted calibration utilities.
- `src/eval/`: selective metrics, group metrics, OOD metrics, plotting.
- `scripts/run_v1_pipeline.py`: runner end-to-end theo YAML config.

## Cài dependencies
```bash
pip install -r requirements.txt
```

## Quickstart (demo smoke test)
```bash
./.venv/bin/python scripts/make_demo_data.py --out-dir data --ref-name demo_ref.h5ad --qry-name demo_qry.h5ad
MPLCONFIGDIR=/tmp/matplotlib-codex ./.venv/bin/python scripts/run_v1_pipeline.py --config configs/v1_demo_pca.yaml
```

## Chạy bản thật (cross-donor)
```bash
MPLCONFIGDIR=/tmp/matplotlib-codex ./.venv/bin/python scripts/run_v1_pipeline.py --config configs/v1_cross_donor.yaml
```

Các config có sẵn:
- `configs/v1_cross_donor.yaml`
- `configs/v1_cross_donor_weighted.yaml`
- `configs/v1_cross_platform.yaml`
- `configs/v1_heldout_celltype.yaml`
- `configs/v1_demo_pca.yaml`

## Artifacts chính cho paper
Sau khi chạy, xem:
- `figures/<run_name>/risk_coverage_main.png`
- `figures/<run_name>/subgroup_risk_<group_key>.png`
- `figures/<run_name>/ood_combined_roc.png`
- `figures/<run_name>/ood_combined_pr.png`

Và bảng số liệu:
- `outputs/<run_name>/calibration/calibration_summary.csv`
- `outputs/<run_name>/metrics/operating_points.csv`
- `outputs/<run_name>/metrics/target_metrics.csv`
- `outputs/<run_name>/metrics/subgroup_summary.csv`
- `outputs/<run_name>/metrics/group_metrics_combined.csv`
- `outputs/<run_name>/metrics/ood_metrics.csv`

## Legacy pipeline
Các script cũ vẫn giữ nguyên để tương thích:
- `scripts/run_pipeline.py`
- `scripts/eval_selective.py`
- `scripts/train_scvi_probe.py`
