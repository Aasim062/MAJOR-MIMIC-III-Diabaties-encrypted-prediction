"""
PHASE 2: DATA PREPROCESSING & SPLITTING + HOSPITAL ASSIGNMENT
— mean/median/zero imputation, normalization, stratified split, save as .npy

Inputs:  
    X_features.csv, y_labels.csv from Phase 1

Outputs: 
    data/processed/phase2/
        - X_train.npy, y_train.npy, X_val.npy, y_val.npy, X_test.npy, y_test.npy
        - scaler.pkl
        - assignment_train.csv, assignment_val.csv, assignment_test.csv
        - split_stats.txt
"""

import argparse
import os
import numpy as np
import pandas as pd
import pickle

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold

def parse_args():
    parser = argparse.ArgumentParser(description="Phase 2 Preprocessing")
    parser.add_argument("--features", default="data/processed/X_features.csv")
    parser.add_argument("--labels", default="data/processed/y_labels.csv")
    parser.add_argument("--output_dir", default="data/processed/phase2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_hospitals", type=int, default=3)
    parser.add_argument("--train_frac", type=float, default=0.7)
    parser.add_argument("--val_frac", type=float, default=0.1)
    return parser.parse_args()

def main():
    args = parse_args()
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # ----- 1. Load Data -----
    X = pd.read_csv(args.features, index_col=0)
    y = pd.read_csv(args.labels, index_col=0).squeeze()
    y = y.loc[X.index]

    # Identify feature groups from column names (assumes convention!)
    vitals = [c for c in X if "heart_rate" in c or "bp_" in c or "respiratory" in c or "temperature" in c or "spo2" in c or "glucose_bedside" in c]
    labs = [c for c in X if c in 
            ["creatinine","bun","sodium","potassium","chloride","co2",
             "wbc","hemoglobin","hematocrit","platelets","ph","pco2","po2","hco3",
             "glucose_lab","alt","ast","bilirubin_total","lactate","albumin","magnesium","hba1c"]]
    binary_flags = [
        c for c in X if (
            any(flag in c for flag in [
                "insulin", "antibiotics", "vasopressors", "mech", "diuretics",
                "ace_inhibitors", "beta_blockers", "corticosteroids",
                "ckd_flag","chf_flag","copd_flag","sepsis_flag","hypertension_flag","anemia_flag","malignancy_flag",
                "readmission_flag","gender"
            ])
            or (
                X[c].dropna().value_counts().index.difference([0,1]).empty and X[c].max() <= 1.0
            )
        )
    ]  # heuristic
    # Everything else: treat as continuous 
    other_continuous = sorted(list(set(X.columns) - set(vitals) - set(labs) - set(binary_flags)))
    continuous = sorted(set(vitals) | set(other_continuous) | set(labs))

    # ----- 2. Imputation -----
    # Binary: fillna 0
    X[binary_flags] = X[binary_flags].fillna(0)

    # Labs: median imputation
    if labs:
        lab_imputer = SimpleImputer(strategy="median")
        X[labs] = lab_imputer.fit_transform(X[labs])
    # Vitals: mean imputation
    vital_imputer = SimpleImputer(strategy="mean")
    if vitals:
        X[vitals] = vital_imputer.fit_transform(X[vitals])
    # Other continuous: mean imputation
    if other_continuous:
        other_imputer = SimpleImputer(strategy="mean")
        X[other_continuous] = other_imputer.fit_transform(X[other_continuous])

    # Optional: Forward fill for time series (you'd do this on DF with time axis before aggregation to single row)
    # For typical tabular X as above, already aggregated mean/median, so this step is a NOP.

    # ----- 3. Stratified Split -----
    idx_all = X.index
    y_all = y.loc[idx_all]
    assert len(y_all) == len(X)

    idx_train, idx_temp, y_train, y_temp = train_test_split(
        idx_all, y_all, test_size=1.0-args.train_frac, stratify=y_all, random_state=args.seed)
    relative_val_frac = args.val_frac / (1 - args.train_frac)
    idx_val, idx_test, y_val, y_test = train_test_split(
        idx_temp, y_temp, test_size=1.0-relative_val_frac, stratify=y_temp, random_state=args.seed)

    def Xy(idx):
        return X.loc[idx].values.astype(np.float32), y.loc[idx].values.astype(np.int64)

    X_train, y_train = Xy(idx_train)
    X_val,   y_val   = Xy(idx_val)
    X_test,  y_test  = Xy(idx_test)

    # ----- 4. Normalization (Scaler fit ONLY on train) -----
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_sc = scaler.transform(X_train)
    X_val_sc   = scaler.transform(X_val)
    X_test_sc  = scaler.transform(X_test)

    # ----- 5. Hospital Assignment (stratified inside each split) -----
    def assign_hospitals(y_local, n_hospitals, seed):
        """Class-balanced hospital assignment; returns array[A/B/C...] for each sample in passed y_local."""
        y_local = pd.Series(y_local)
        skf = StratifiedKFold(n_splits=n_hospitals, shuffle=True, random_state=seed)
        hospital_arr = np.empty(len(y_local), dtype="U1")
        for i, (_, test_idx) in enumerate(skf.split(np.zeros(len(y_local)), y_local)):
            hospital_label = chr(65 + i)  # "A".."Z"
            hospital_arr[test_idx] = hospital_label
        return hospital_arr
    hosp_assign = {
        "train": assign_hospitals(y_train, args.n_hospitals, args.seed),
        "val":   assign_hospitals(y_val, args.n_hospitals, args.seed),
        "test":  assign_hospitals(y_test, args.n_hospitals, args.seed),
    }

    # ----- 6. Save Files -----
    np.save(os.path.join(args.output_dir, "X_train.npy"), X_train_sc)
    np.save(os.path.join(args.output_dir, "X_val.npy"),   X_val_sc)
    np.save(os.path.join(args.output_dir, "X_test.npy"),  X_test_sc)
    np.save(os.path.join(args.output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(args.output_dir, "y_val.npy"),   y_val)
    np.save(os.path.join(args.output_dir, "y_test.npy"),  y_test)
    with open(os.path.join(args.output_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    # Assignments (for reproducibility/federated FL simulation)
    pd.Series(hosp_assign["train"], index=idx_train).to_csv(os.path.join(args.output_dir, "assignment_train.csv"), header=["hospital"])
    pd.Series(hosp_assign["val"],   index=idx_val).to_csv(os.path.join(args.output_dir, "assignment_val.csv"), header=["hospital"])
    pd.Series(hosp_assign["test"],  index=idx_test).to_csv(os.path.join(args.output_dir, "assignment_test.csv"), header=["hospital"])

    # Report
    with open(os.path.join(args.output_dir, "split_stats.txt"), "w") as f:
        for name, Xsp, ysp, assign in [
            ("TRAIN", X_train, y_train, hosp_assign["train"]),
            ("VAL",   X_val,   y_val,   hosp_assign["val"]),
            ("TEST",  X_test,  y_test,  hosp_assign["test"]),
        ]:
            f.write(f"{name} SAMPLES: {Xsp.shape[0]:>6}\n")
            f.write(f"{name} Mortality: {100*ysp.mean():.2f}% ({ysp.sum()}/{len(ysp)})\n")
            valc = pd.Series(assign).value_counts().sort_index()
            f.write(f"{name} Hospital distribution: {valc.to_dict()}\n\n")
    print("Preprocessing, splitting, and hospital assignment COMPLETE.")

if __name__ == "__main__":
    main()