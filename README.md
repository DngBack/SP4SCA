# SP4SCA
Selective Prediction for Single-Cell Annotation under Biological Shift: Why Marginal Reliability Is Not Enough

## Những phần đã triển khai

- Pipeline end-to-end bằng config YAML: preprocess -> train scVI + linear probe -> eval selective.
- Preprocess theo đúng spec tuần đầu:
  - giữ `adata.layers["counts"]` cho scVI
  - intersect gene ref/query bằng inner join
  - QC: `min_genes >= 200`, `min_cells per gene >= 10`, `pct_counts_mt < 20`
  - HVG: `n_top_genes=3000`, `flavor="seurat_v3"`, ưu tiên `study_id` rồi `batch_id`
- Backbone: `scVI (NB likelihood) + LogisticRegression(class_weight="balanced")`.
- Rejector scores: `score_conf` (max softmax) và `score_neg_entropy`.
- Metrics/figure:
  - risk-coverage curve
  - `metrics.csv`
  - confusion matrix trên accepted set
  - sanity checks cơ bản.

## Cấu trúc scripts

- `scripts/preprocess.py`: preprocess ref/query theo chuẩn pipeline.
- `scripts/train_scvi_probe.py`: train scVI và linear probe.
- `scripts/eval_selective.py`: predict query + selective metrics/plots.
- `scripts/run_pipeline.py`: chạy full pipeline từ `config.yaml`.
- `scripts/make_demo_data.py`: tạo dữ liệu synthetic để smoke test.
- `scripts/list_pancreas_datasets.py`: liệt kê dataset pancreas trên CELLxGENE Census.
- `scripts/download_source_h5ad.py`: tải 1 file source `.h5ad` theo `dataset_id`.
- `scripts/download_pancreas_pair.py`: tải nhanh cặp `pancreas_ref.h5ad` và `pancreas_qry.h5ad`.

## Cài dependencies

```bash
pip install -r requirements.txt
```

## Pipeline quickstart

1. Set dataset paths trong `config.yaml`:
- `data.ref_h5ad`
- `data.qry_h5ad`

2. Nếu chưa có `.h5ad`, có 2 lựa chọn:

Tạo dữ liệu demo (synthetic):

```bash
python3 scripts/make_demo_data.py --out-dir data
```

Hoặc tải dữ liệu thật pancreas từ CELLxGENE:

```bash
python3 scripts/list_pancreas_datasets.py --keyword pancreas --top-k 30
python3 scripts/download_pancreas_pair.py --overwrite
```

3. Chạy pipeline:

```bash
./run.sh
```

Hoặc với config khác:

```bash
./run.sh path/to/another_config.yaml
```

## Outputs

- `figures/pancreas_risk_coverage.png`
- `outputs/pancreas/preprocessed.h5ad`
- `outputs/models/pancreas_scvi/scvi_model/`
- `outputs/models/pancreas_scvi/linear_probe.joblib`
- `outputs/models/pancreas_scvi/train_summary.csv`
- `outputs/metrics/pancreas/predictions_query.csv`
- `outputs/metrics/pancreas/curve_confidence.csv`
- `outputs/metrics/pancreas/curve_neg_entropy.csv`
- `outputs/metrics/pancreas/metrics.csv`
- `outputs/metrics/pancreas/confusion_matrix_accepted.csv`
- `outputs/metrics/pancreas/sanity_warnings.csv` (nếu có cảnh báo)

## Troubleshooting

- `FileNotFoundError` cho `data/pancreas_ref.h5ad` hoặc `data/pancreas_qry.h5ad`:
  - kiểm tra lại `data.ref_h5ad` và `data.qry_h5ad` trong `config.yaml`
  - hoặc chạy script download/tạo demo data ở trên.

- Lỗi do placeholder `<DATASET_ID>` trong shell:
  - không dùng dấu `< >` khi chạy lệnh
  - dùng ID thật, ví dụ: `--dataset-id 31f657dc-1875-4c4b-a5ca-ce63b3ef3a82`.
