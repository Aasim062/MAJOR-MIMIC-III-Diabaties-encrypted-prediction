# Pipeline Summary (Phases 0–3)

## Phase 0 – Cohort Extraction
- Extracted adult ICU stays with diabetes from core MIMIC-III tables (ADMISSIONS, PATIENTS, ICUSTAYS, DIAGNOSES_ICD, etc.).
- Joined IDs across tables, filtered age and diabetes criteria, and produced the base cohort that feeds Phase 1.
- Artifacts: cohort CSVs per source table (see project root CSV folders) used as inputs to feature_engineering.py.
- Performance/quality: final cohort size = 50,702 admissions (from console log).

## Phase 1 – Feature Engineering
- Engineered structured features from vitals, labs, diagnoses, and demographics; applied missing-value handling and normalization.
- Validation (final run):
	- Features: 47,204 admissions × 60 features
	- Missing after imputation: 0
	- Target distribution: 11.01% mortality (5,199 cases)
	- Checks: 5/5 passed
- Outputs: feature_engineering_report.txt (utf-8), X_features.csv, y_labels.csv, feature_names.txt, scaler_params.json in data/processed/.
- Performance/quality: feature dimension 60; missingness 0; stats per feature in feature_engineering_report.txt.

## Phase 2 – Splitting & Hospital Assignment
- Stratified train/val/test split with class balance; scaler fit on train only and applied to val/test.
- Hospital-balanced assignments (A/B/C) created for each split and saved to assignment_train.csv, assignment_val.csv, assignment_test.csv.
- Saved arrays: X_train.npy, y_train.npy, X_val.npy, y_val.npy, X_test.npy, y_test.npy, plus scaler.pkl for reproducible normalization.
- Split stats: split_stats.txt documents sample counts, mortality rates, and hospital distribution per split (captured below).

## Phase 3 – Training (MLP-3)
- Model: 64→128→64→1 MLP, ReLU + Dropout, Adam optimizer, weight decay, CosineAnnealingLR, grad clipping, class-weighted BCEWithLogitsLoss.
- Numerical stability: sigmoid uses scipy.expit to avoid overflow warnings.
- Per-hospital training: assignment masks slice train/val/test per hospital (A/B/C); weights saved with hospital suffix when --hospital ALL.
- Logs: mlp_train_log.txt (epoch history); console prints final metrics.
- Outputs: models/mlp_best_model.pt (single) or models/mlp_best_model_A/B/C.pt (per-hospital), plus logs.
- Single-model run (latest):
	- Best ValAUC: 0.9003 (checkpoint saved)
	- Test: AUC 0.8762, Acc 0.8902, F1 0.5223, Threshold 0.670, Confusion [[7838, 564], [473, 567]]
- Per-hospital run (latest, --hospital ALL):
	- Hospital A: ValAUC 0.8897; Test AUC 0.8633, Acc 0.8929, F1 0.5080, Threshold 0.56, Confusion [[2637, 164], [173, 174]]
	- Hospital B: ValAUC 0.8723; Test AUC 0.8525, Acc 0.8777, F1 0.5006, Threshold 0.24, Confusion [[2569, 232], [153, 193]]
	- Hospital C: ValAUC 0.8989; Test AUC 0.8668, Acc 0.8990, F1 0.5268, Threshold 0.65, Confusion [[2652, 148], [170, 177]]
- Training dynamics: see mlp_train_log.txt for epoch-wise loss/ValAUC/ValF1 (300 epochs per run in logs).

## Key Artifacts (paths)
- data/processed/phase2: X_train.npy, X_val.npy, X_test.npy, y_train.npy, y_val.npy, y_test.npy
- data/processed/phase2: assignment_train.csv, assignment_val.csv, assignment_test.csv (hospital labels), scaler.pkl, split_stats.txt
- data/processed/: feature_engineering_report.txt, X_features.csv, y_labels.csv, feature_names.txt, scaler_params.json
- reports/pipeline_summary.md (this file)
- models/: mlp_best_model.pt (single) or mlp_best_model_A/B/C.pt (per-hospital)
- phase_3_train.py: training code with per-hospital masking and model saving
- mlp_train_log.txt: training history (one line per epoch)

## Where to read metrics
- Phase 1 metrics: feature_engineering_report.txt (captured above)
- Phase 2 metrics: split_stats.txt (captured in Filled Metrics section)
- Phase 3 metrics: console output and mlp_train_log.txt (captured in Filled Metrics section)

## Performance Measures (math) and Where to Store Values
- Confusion matrix: TP, FP, FN, TN counts for each evaluation.
- Accuracy: $\text{Acc} = \frac{TP + TN}{TP + TN + FP + FN}$
- Precision: $\text{Prec} = \frac{TP}{TP + FP}$
- Recall (Sensitivity): $\text{Rec} = \frac{TP}{TP + FN}$
- F1: $\text{F1} = 2 \cdot \frac{\text{Prec} \cdot \text{Rec}}{\text{Prec} + \text{Rec}}$
- ROC AUC: area under the ROC curve (reported by sklearn.roc_auc_score).
- Threshold: probability cutoff chosen to maximize F1 on the validation set.

## Detailed Outputs and Expected Locations
- Phase 1 feature report: produced by feature_engineering.py (utf-8 text) — add feature count and missingness summaries here when known.
- Phase 2 artifacts: data/processed/phase2/
	- X_train.npy, X_val.npy, X_test.npy, y_train.npy, y_val.npy, y_test.npy
	- assignment_train.csv, assignment_val.csv, assignment_test.csv (hospital labels)
	- scaler.pkl, split_stats.txt (contains counts/mortality/hospital distribution) — copy values below when available.
- Phase 3 artifacts: models/ after run; mlp_train_log.txt (epoch history); console/log output prints final metrics.
	- Single-model run: models/mlp_best_model.pt
	- Per-hospital run (--hospital ALL): models/mlp_best_model_A.pt, models/mlp_best_model_B.pt, models/mlp_best_model_C.pt

## Filled Metrics (latest runs)
- Split stats (from split_stats.txt):
	- Train: samples = 33,042; mortality = 11.01% (3,639/33,042); hospital counts = {A: 11,014, B: 11,014, C: 11,014}
	- Val:   samples = 4,720; mortality = 11.02% (520/4,720); hospital counts = {A: 1,574, B: 1,573, C: 1,573}
	- Test:  samples = 9,442; mortality = 11.01% (1,040/9,442); hospital counts = {A: 3,148, B: 3,147, C: 3,147}
- Phase 3 validation (best epoch, single-model run): Best ValAUC = 0.9003 (checkpoint saved). Val F1/threshold correspond to that epoch (see mlp_train_log.txt).
- Phase 3 test metrics (per hospital, --hospital ALL):
	- Hospital A: AUC = 0.8633, Acc = 0.8929, F1 = 0.5080, Threshold = 0.56, Confusion = [[2637, 164], [173, 174]]
	- Hospital B: AUC = 0.8525, Acc = 0.8777, F1 = 0.5006, Threshold = 0.24, Confusion = [[2569, 232], [153, 193]]
	- Hospital C: AUC = 0.8668, Acc = 0.8990, F1 = 0.5268, Threshold = 0.65, Confusion = [[2652, 148], [170, 177]]
- Phase 3 test metrics (single model): AUC = 0.8762, Acc = 0.8902, F1 = 0.5223, Threshold = 0.670, Confusion = [[7838, 564], [473, 567]]

## How to extract the numbers quickly
- Phase 2: open split_stats.txt and copy the counts/mortality/hospital distribution into the template above.
- Phase 3: rerun training or view console output; the final block prints AUC/Acc/F1/Threshold/Confusion per hospital. Copy those into the template. Per-epoch trends are in mlp_train_log.txt.
