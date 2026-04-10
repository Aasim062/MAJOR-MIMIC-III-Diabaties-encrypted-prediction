import argparse
import os
from datetime import datetime
from typing import Iterable, List

import pandas as pd

# Prefer GPU when available (used by downstream torch phases; harmless if torch is absent)
try:  # GPU-preference hook
    import torch

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(DEVICE)
    GPU_AVAILABLE = DEVICE.type == "cuda"
except ImportError:
    DEVICE = None
    GPU_AVAILABLE = False


REQUIRED_FILES = [
    "PATIENTS.csv",
    "ADMISSIONS.csv",
    "ICUSTAYS.csv",
    "DIAGNOSES_ICD.csv",
    "D_ICD_DIAGNOSES.csv",
]


DIABETES_PREFIXES = (
    "250",  # Diabetes mellitus
    "249",  # Secondary diabetes mellitus
    "E10",  # Type 1
    "E11",  # Type 2
    "E13",  # Other specified diabetes mellitus
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 0: Extract MIMIC-III ICU cohort and diabetes flag",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--raw_dir",
        default=os.path.join("data", "raw"),
        help="Directory containing raw MIMIC-III CSV files",
    )
    parser.add_argument(
        "--output",
        default=os.path.join("data", "processed", "cohort.csv"),
        help="Path to write the extracted cohort CSV",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=250_000,
        help="Chunk size for streaming large tables (DIAGNOSES_ICD)",
    )
    parser.add_argument(
        "--min_age",
        type=float,
        default=16.0,
        help="Minimum age at admission to keep (years)",
    )
    return parser.parse_args()


def assert_required_files(raw_dir: str) -> None:
    missing: List[str] = []
    for fname in REQUIRED_FILES:
        if not os.path.exists(os.path.join(raw_dir, fname)):
            missing.append(fname)
    if missing:
        missing_str = ", ".join(missing)
        raise FileNotFoundError(
            f"Missing required files in {raw_dir}: {missing_str}. "
            "Place MIMIC-III CSVs in data/raw/."
        )


def read_patients(path: str) -> pd.DataFrame:
    # Handle possible schema variants (GENDER vs SEX)
    raw = pd.read_csv(path)
    colmap = {
        "SUBJECT_ID": "subject_id",
        "DOB": "dob",
        "DOD": "dod",
    }
    if "GENDER" in raw.columns:
        colmap["GENDER"] = "gender"
    elif "SEX" in raw.columns:  # fallback
        colmap["SEX"] = "gender"
    else:
        raw["gender"] = pd.NA
    df = raw[list(colmap.keys())].rename(columns=colmap)
    date_cols = [c for c in ["dob", "dod"] if c in df.columns]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def read_admissions(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        usecols=["SUBJECT_ID", "HADM_ID", "ADMITTIME", "DISCHTIME", "HOSPITAL_EXPIRE_FLAG"],
        parse_dates=["ADMITTIME", "DISCHTIME"],
    )
    return df.rename(
        columns={
            "SUBJECT_ID": "subject_id",
            "HADM_ID": "hadm_id",
            "ADMITTIME": "admittime",
            "DISCHTIME": "dischtime",
            "HOSPITAL_EXPIRE_FLAG": "hospital_expire_flag",
        }
    )


def read_icustays(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        usecols=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "INTIME", "OUTTIME", "LOS"],
        parse_dates=["INTIME", "OUTTIME"],
    )
    return df.rename(
        columns={
            "SUBJECT_ID": "subject_id",
            "HADM_ID": "hadm_id",
            "ICUSTAY_ID": "icustay_id",
            "INTIME": "intime",
            "OUTTIME": "outtime",
            "LOS": "los_days",
        }
    )


def collect_diabetes_subjects(path: str, chunksize: int) -> pd.DataFrame:
    matches: List[int] = []
    for chunk in pd.read_csv(
        path,
        usecols=["SUBJECT_ID", "ICD9_CODE"],
        dtype={"SUBJECT_ID": "int32", "ICD9_CODE": "string"},
        chunksize=chunksize,
    ):
        mask = chunk["ICD9_CODE"].str.startswith(DIABETES_PREFIXES, na=False)
        if mask.any():
            matches.extend(chunk.loc[mask, "SUBJECT_ID"].astype(int).tolist())
    diabetes_subjects = pd.DataFrame({"subject_id": pd.Series(matches, dtype="int32")})
    diabetes_subjects = diabetes_subjects.drop_duplicates().assign(diabetes_flag=1)
    return diabetes_subjects


def compute_age_at_admission(row: pd.Series) -> float:
    """Safe age computation that avoids pandas overflow and nonsensical dates."""
    dob = row["dob"]
    admit = row["admittime"]
    if pd.isna(dob) or pd.isna(admit):
        return float("nan")
    try:
        delta_days = (admit.to_pydatetime() - dob.to_pydatetime()).days
    except OverflowError:
        return float("nan")
    # Filter out negative or implausible ages (>120y) before conversion to years
    if delta_days < 0 or delta_days > 120 * 365:
        return float("nan")
    return round(delta_days / 365.25, 2)


def build_cohort(raw_dir: str, chunksize: int, min_age: float) -> pd.DataFrame:
    patients = read_patients(os.path.join(raw_dir, "PATIENTS.csv"))
    admissions = read_admissions(os.path.join(raw_dir, "ADMISSIONS.csv"))
    icu = read_icustays(os.path.join(raw_dir, "ICUSTAYS.csv"))
    diabetes = collect_diabetes_subjects(
        os.path.join(raw_dir, "DIAGNOSES_ICD.csv"), chunksize=chunksize
    )

    cohort = (
        icu.merge(admissions, on=["subject_id", "hadm_id"], how="left")
        .merge(patients, on="subject_id", how="left")
        .merge(diabetes, on="subject_id", how="left")
    )

    cohort["diabetes_flag"] = cohort["diabetes_flag"].fillna(0).astype(int)
    cohort["age"] = cohort.apply(compute_age_at_admission, axis=1)
    if "gender" not in cohort.columns:
        cohort["gender"] = pd.NA

    cohort = cohort.loc[cohort["age"] >= min_age].copy()

    # Keep a lean set of columns for downstream phases.
    cohort = cohort[
        [
            "subject_id",
            "hadm_id",
            "icustay_id",
            "intime",
            "outtime",
            "los_days",
            "age",
            "gender",
            "hospital_expire_flag",
            "diabetes_flag",
        ]
    ]

    cohort = cohort.sort_values(["subject_id", "hadm_id", "icustay_id"]).reset_index(drop=True)
    return cohort


def save_cohort(df: pd.DataFrame, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)


def main() -> None:
    args = parse_args()
    assert_required_files(args.raw_dir)

    print("[Phase 0] Starting cohort extraction...")
    print(f"Raw directory: {args.raw_dir}")
    print(f"Output file:   {args.output}")
    print(f"Chunk size:    {args.chunksize}")
    if GPU_AVAILABLE:
        print(f"Device:        CUDA ({DEVICE})")
    elif DEVICE is not None:
        print(f"Device:        CPU ({DEVICE})")
    else:
        print("Device:        CPU (torch not installed)")

    cohort = build_cohort(args.raw_dir, args.chunksize, args.min_age)
    save_cohort(cohort, args.output)

    print(f"[Phase 0] Done. Rows: {len(cohort):,}. Saved to {args.output}.")


if __name__ == "__main__":
    main()
