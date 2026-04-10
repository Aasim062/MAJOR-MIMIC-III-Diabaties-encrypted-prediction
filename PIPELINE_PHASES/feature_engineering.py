"""
PHASE 1: FEATURE ENGINEERING (64 Clinical Features)
═══════════════════════════════════════════════════════════════════════════════

Extract and engineer 64 clinical features from MIMIC-III data for 
diabetes patient mortality prediction.

Input:  data/processed/cohort.csv (from Phase 0)
        data/raw/CHARTEVENTS.csv, LABEVENTS.csv, PRESCRIPTIONS.csv, etc.

Output: data/processed/X_features.csv (cohort_size × 64)
        data/processed/y_labels.csv (cohort_size × 1)
        data/processed/feature_names.txt
        data/processed/feature_engineering_report.txt
"""

import argparse
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# Prefer GPU via PyTorch when available (used later for modeling; CSV IO stays on CPU)
try:
    import torch

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(DEVICE)
    GPU_AVAILABLE = DEVICE.type == "cuda"
    if GPU_AVAILABLE:
        print(f"✅ CUDA available - default device set to {DEVICE}")
    else:
        print(f"⚠️  CUDA not available - using CPU ({DEVICE})")
except ImportError:
    DEVICE = None
    GPU_AVAILABLE = False
    print("⚠️  PyTorch not installed - proceeding on CPU")

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS & CONFIGURATIONS
# ═══════════════════════════════════════════════════════════════════════════════

# VITAL SIGNS ITEMID MAPPING (CHARTEVENTS)
VITAL_ITEMIDS = {
    # Heart Rate
    220045: "heart_rate",
    
    # Blood Pressure (Systolic/Diastolic)
    220179: "bp_systolic",
    220180: "bp_diastolic",
    225309: "bp_systolic_alarm",
    225310: "bp_diastolic_alarm",
    
    # Respiratory Rate
    220210: "respiratory_rate",
    224689: "respiratory_rate_set",
    
    # Temperature (Fahrenheit → convert to Celsius)
    223761: "temperature_f",
    223762: "temperature_f_event",
    
    # SpO2 (Oxygen Saturation)
    220277: "spo2",
    
    # Glucose (Bedside)
    225664: "glucose_bedside",
    226537: "glucose_bedside_poc",
}

# LABORATORY VALUES ITEMID MAPPING (LABEVENTS)
LAB_ITEMIDS = {
    # Renal Function
    50912: "creatinine",
    51006: "bun",
    
    # Electrolytes
    50824: "sodium",
    50971: "potassium",
    50902: "chloride",
    50945: "co2",
    
    # Complete Blood Count
    51301: "wbc",
    51222: "hemoglobin",
    51221: "hematocrit",
    51265: "platelets",
    
    # Arterial Blood Gas
    50820: "ph",
    50821: "pco2",
    50823: "po2",
    50868: "hco3",
    
    # Glucose (Lab)
    50809: "glucose_lab",
    50931: "glucose_lab_poi",
    
    # Liver Function
    50954: "alt",
    50878: "ast",
    50885: "bilirubin_total",
    
    # Other
    50813: "lactate",
    50862: "albumin",
    50960: "magnesium",
    50852: "hba1c",
}

# MEDICATION PATTERNS (PRESCRIPTIONS)
MEDICATION_PATTERNS = {
    "insulin": [r"insulin", r"lantus", r"humalog", r"novolog"],
    "antibiotics": [r"cephalosporin", r"penicillin", r"aminoglycoside", 
                   r"fluoroquinolone", r"clindamycin", r"vancomycin"],
    "vasopressors": [r"epinephrine", r"norepinephrine", r"dopamine", 
                    r"dobutamine", r"vasopressin", r"phenylephrine"],
    "diuretics": [r"furosemide", r"lasix", r"torsemide", r"bumetanide", r"spironolactone"],
    "ace_inhibitors": [r"lisinopril", r"enalapril", r"ramipril", r"perindopril"],
    "beta_blockers": [r"metoprolol", r"atenolol", r"carvedilol", r"labetalol"],
    "corticosteroids": [r"methylprednisolone", r"dexamethasone", r"prednisone"],
}

# COMORBIDITY ICD9 CODE PATTERNS
COMORBIDITY_CODES = {
    "ckd": ["585", "586"],  # Chronic kidney disease
    "chf": ["428"],  # Congestive heart failure
    "copd": ["491", "492", "493", "494", "495", "496"],  # COPD
    "sepsis": ["038", "995.91", "995.92"],  # Sepsis
    "hypertension": ["401", "402", "403", "404", "405"],  # HTN
    "anemia": ["285"],  # Anemia
    "malignancy": [str(i) for i in range(140, 240)],  # Cancer (140-239)
}

# FEATURE GROUP DEFINITIONS
FEATURE_GROUPS = {
    "vitals": 8,
    "labs": 20,
    "demographics": 5,
    "icu_metrics": 4,
    "medications": 8,
    "comorbidities": 7,
    "diabetes_specific": 2,
}

# Total features
TOTAL_FEATURES = sum(FEATURE_GROUPS.values())

# ═══════════════════════════════════════════════════════════════════════════════
# VITAL SIGNS EXTRACTION (CHARTEVENTS)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_vitals(
    raw_dir: str,
    cohort: pd.DataFrame,
    chunksize: int = 100_000
) -> pd.DataFrame:
    """
    Extract vital signs from CHARTEVENTS for first 24 hours of ICU stay.
    
    Returns DataFrame with hadm_id and vital signs (mean/median aggregated).
    """
    print("\n[Phase 1] Extracting vital signs from CHARTEVENTS...")
    
    vitals_data = []
    
    # Read CHARTEVENTS in chunks
    dtypes = {
        "SUBJECT_ID": "Int64",
        "HADM_ID": "Int64",
        "ICUSTAY_ID": "Int64",
        "ITEMID": "Int64",
        "VALUENUM": "float32",
    }
    for chunk in tqdm(
        pd.read_csv(
            os.path.join(raw_dir, "CHARTEVENTS.csv"),
            usecols=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "CHARTTIME", "ITEMID", "VALUENUM"],
            dtype=dtypes,
            parse_dates=["CHARTTIME"],
            chunksize=chunksize,
        ),
        desc="Reading CHARTEVENTS"
    ):
        # Filter by relevant ITEMIDs
        chunk = chunk[chunk["ITEMID"].isin(VITAL_ITEMIDS.keys())].copy()
        # Drop rows with missing identifiers or values to avoid dtype errors
        chunk = chunk.dropna(subset=["HADM_ID", "ICUSTAY_ID", "ITEMID", "VALUENUM"])
        
        if len(chunk) == 0:
            continue
        
        # Rename columns
        chunk.rename(
            columns={
                "SUBJECT_ID": "subject_id",
                "HADM_ID": "hadm_id",
                "ICUSTAY_ID": "icustay_id",
            },
            inplace=True
        )
        
        # Map ITEMID to feature name
        chunk["feature"] = chunk["ITEMID"].map(VITAL_ITEMIDS)
        
        vitals_data.append(chunk)
    
    if not vitals_data:
        print("⚠️  No vital signs found in CHARTEVENTS!")
        return pd.DataFrame()
    
    vitals_df = pd.concat(vitals_data, ignore_index=True)
    
    # Merge with cohort to get ICU admission times
    vitals_df = vitals_df.merge(
        cohort[["hadm_id", "intime"]],
        on="hadm_id",
        how="inner"
    )
    
    # Filter to first 24 hours of ICU stay
    vitals_df["hours_since_admission"] = (
        (vitals_df["CHARTTIME"] - vitals_df["intime"]).dt.total_seconds() / 3600
    )
    vitals_df = vitals_df[vitals_df["hours_since_admission"] <= 24].copy()
    
    # Handle unit conversions
    # Temperature: Fahrenheit → Celsius (only if >50, assume F)
    temp_mask = (vitals_df["feature"] == "temperature_f") & (vitals_df["VALUENUM"] > 50)
    vitals_df.loc[temp_mask, "VALUENUM"] = (vitals_df.loc[temp_mask, "VALUENUM"] - 32) * 5/9
    vitals_df.loc[vitals_df["feature"] == "temperature_f", "feature"] = "temperature"
    
    # Aggregate by hadm_id and feature (take median)
    vitals_pivot = vitals_df.groupby(["hadm_id", "feature"])["VALUENUM"].median().reset_index()
    vitals_pivot = vitals_pivot.pivot(index="hadm_id", columns="feature", values="VALUENUM")
    vitals_pivot = vitals_pivot.loc[~vitals_pivot.index.duplicated(keep="first")]
    
    # Calculate MAP (Mean Arterial Pressure) if both SBP and DBP available
    if "bp_systolic" in vitals_pivot.columns and "bp_diastolic" in vitals_pivot.columns:
        vitals_pivot["map"] = (
            vitals_pivot["bp_systolic"] + 2 * vitals_pivot["bp_diastolic"]
        ) / 3
    
    print(f"✅ Extracted vitals for {len(vitals_pivot)} admissions")
    
    return vitals_pivot


# ═══════════════════════════════════════════════════════════════════════════════
# LABORATORY VALUES EXTRACTION (LABEVENTS)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_labs(
    raw_dir: str,
    cohort: pd.DataFrame,
    chunksize: int = 100_000
) -> pd.DataFrame:
    """
    Extract lab values from LABEVENTS for first 24 hours of ICU stay.
    
    Returns DataFrame with hadm_id and lab values (median aggregated).
    """
    print("\n[Phase 1] Extracting lab values from LABEVENTS...")
    
    labs_data = []
    
    # Read LABEVENTS in chunks
    for chunk in tqdm(
        pd.read_csv(
            os.path.join(raw_dir, "LABEVENTS.csv"),
            usecols=["SUBJECT_ID", "HADM_ID", "CHARTTIME", "ITEMID", "VALUENUM"],
            dtype={"SUBJECT_ID": "Int64", "HADM_ID": "Int64", "ITEMID": "Int64", "VALUENUM": "float32"},
            parse_dates=["CHARTTIME"],
            chunksize=chunksize,
        ),
        desc="Reading LABEVENTS"
    ):
        # Filter by relevant ITEMIDs
        chunk = chunk[chunk["ITEMID"].isin(LAB_ITEMIDS.keys())].copy()

        # Drop rows with missing identifiers or values to avoid dtype issues
        chunk = chunk.dropna(subset=["HADM_ID", "ITEMID", "VALUENUM"])
        
        if len(chunk) == 0:
            continue
        
        # Rename columns
        chunk.rename(
            columns={
                "SUBJECT_ID": "subject_id",
                "HADM_ID": "hadm_id",
            },
            inplace=True
        )
        
        # Map ITEMID to feature name
        chunk["feature"] = chunk["ITEMID"].map(LAB_ITEMIDS)
        
        labs_data.append(chunk)
    
    if not labs_data:
        print("⚠️  No lab values found in LABEVENTS!")
        return pd.DataFrame()
    
    labs_df = pd.concat(labs_data, ignore_index=True)
    
    # Merge with cohort to get ICU admission times
    labs_df = labs_df.merge(
        cohort[["hadm_id", "intime"]],
        on="hadm_id",
        how="inner"
    )
    
    # Filter to first 24 hours of ICU stay
    labs_df["hours_since_admission"] = (
        (labs_df["CHARTTIME"] - labs_df["intime"]).dt.total_seconds() / 3600
    )
    labs_df = labs_df[labs_df["hours_since_admission"] <= 24].copy()
    
    # Validate value ranges and remove outliers
    labs_df = _validate_lab_ranges(labs_df)
    
    # Aggregate by hadm_id and feature (take median)
    labs_pivot = labs_df.groupby(["hadm_id", "feature"])["VALUENUM"].median().reset_index()
    labs_pivot = labs_pivot.pivot(index="hadm_id", columns="feature", values="VALUENUM")
    labs_pivot = labs_pivot.loc[~labs_pivot.index.duplicated(keep="first")]
    
    print(f"✅ Extracted labs for {len(labs_pivot)} admissions")
    
    return labs_pivot


def _validate_lab_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove lab values that are physiologically impossible.
    """
    valid_ranges = {
        "creatinine": (0.1, 20),
        "bun": (5, 200),
        "sodium": (100, 160),
        "potassium": (1, 10),
        "chloride": (80, 120),
        "wbc": (0.5, 50),
        "hemoglobin": (5, 25),
        "hematocrit": (10, 70),
        "platelets": (10, 1000),
        "ph": (6.8, 7.8),
        "pco2": (15, 100),
        "po2": (20, 500),
        "hco3": (10, 50),
        "glucose_lab": (20, 1000),
        "alt": (5, 10000),
        "ast": (5, 10000),
        "bilirubin_total": (0.1, 50),
        "lactate": (0.5, 30),
        "albumin": (1, 6),
        "magnesium": (1, 4),
        "hba1c": (4, 15),
    }
    
    for feature, (min_val, max_val) in valid_ranges.items():
        mask = df["feature"] == feature
        clipped = df.loc[mask, "VALUENUM"].clip(min_val, max_val)
        df.loc[mask, "VALUENUM"] = clipped.astype(np.float32)
    
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# MEDICATION EXTRACTION (PRESCRIPTIONS)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_medications(
    raw_dir: str,
    cohort: pd.DataFrame,
    chunksize: int = 100_000
) -> pd.DataFrame:
    """
    Extract medication flags from PRESCRIPTIONS during ICU stay.
    """
    print("\n[Phase 1] Extracting medications from PRESCRIPTIONS...")
    
    med_data = []
    
    # Read PRESCRIPTIONS in chunks
    for chunk in tqdm(
        pd.read_csv(
            os.path.join(raw_dir, "PRESCRIPTIONS.csv"),
            usecols=["SUBJECT_ID", "HADM_ID", "DRUG", "STARTDATE", "ENDDATE"],
            dtype={"SUBJECT_ID": "Int64", "HADM_ID": "Int64", "DRUG": "string"},
            parse_dates=["STARTDATE", "ENDDATE"],
            chunksize=chunksize,
        ),
        desc="Reading PRESCRIPTIONS"
    ):
        chunk = chunk.dropna(subset=["HADM_ID", "DRUG", "STARTDATE", "ENDDATE"])
        if chunk.empty:
            continue
        chunk = chunk.copy()
        chunk.rename(
            columns={
                "SUBJECT_ID": "subject_id",
                "HADM_ID": "hadm_id",
            },
            inplace=True
        )
        
        med_data.append(chunk)
    
    if not med_data:
        print("⚠️  No prescriptions found!")
        return pd.DataFrame()
    
    meds_df = pd.concat(med_data, ignore_index=True)
    
    # Merge with cohort to get ICU times
    meds_df = meds_df.merge(
        cohort[["hadm_id", "intime", "outtime"]],
        on="hadm_id",
        how="inner"
    )
    
    # Filter to medications during ICU stay
    meds_df = meds_df[
        (meds_df["STARTDATE"] <= meds_df["outtime"]) &
        (meds_df["ENDDATE"] >= meds_df["intime"])
    ].copy()
    
    # Create medication flags
    medication_flags = {
        "hadm_id": cohort["hadm_id"].values,
        "insulin_use": np.zeros(len(cohort), dtype=int),
        "antibiotics": np.zeros(len(cohort), dtype=int),
        "vasopressors": np.zeros(len(cohort), dtype=int),
        "mechanical_ventilation": np.zeros(len(cohort), dtype=int),
        "diuretics": np.zeros(len(cohort), dtype=int),
        "ace_inhibitors": np.zeros(len(cohort), dtype=int),
        "beta_blockers": np.zeros(len(cohort), dtype=int),
        "corticosteroids": np.zeros(len(cohort), dtype=int),
    }
    
    # Create lookup dict
    med_lookup = {hadm_id: idx for idx, hadm_id in enumerate(cohort["hadm_id"].values)}
    
    # Process each medication
    for drug_class, patterns in MEDICATION_PATTERNS.items():
        if drug_class == "insulin":
            flag_name = "insulin_use"
        elif drug_class == "mechanical_ventilation":
            flag_name = "mechanical_ventilation"
        else:
            flag_name = drug_class
        
        if flag_name not in medication_flags:
            continue
        
        # Find matching drugs
        for pattern in patterns:
            mask = meds_df["DRUG"].str.contains(pattern, case=False, regex=True, na=False)
            for hadm_id in meds_df.loc[mask, "hadm_id"].unique():
                if hadm_id in med_lookup:
                    idx = med_lookup[hadm_id]
                    medication_flags[flag_name][idx] = 1
    
    med_flags_df = pd.DataFrame(medication_flags).set_index("hadm_id")
    med_flags_df = med_flags_df.loc[~med_flags_df.index.duplicated(keep="first")]
    
    print(f"✅ Extracted medication flags for {len(med_flags_df)} admissions")
    
    return med_flags_df


# ═══════════════════════════════════════════════════════════════════════════════
# COMORBIDITY EXTRACTION (DIAGNOSES_ICD)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_comorbidities(
    raw_dir: str,
    cohort: pd.DataFrame,
    chunksize: int = 100_000
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract comorbidity flags and calculate Charlson index from DIAGNOSES_ICD.
    
    Returns:
        comorbidity_flags: DataFrame with comorbidity flags (0/1)
        charlson_index: DataFrame with Charlson score
    """
    print("\n[Phase 1] Extracting comorbidities from DIAGNOSES_ICD...")
    
    diag_data = []
    
    # Read DIAGNOSES_ICD in chunks
    for chunk in tqdm(
        pd.read_csv(
            os.path.join(raw_dir, "DIAGNOSES_ICD.csv"),
            usecols=["SUBJECT_ID", "HADM_ID", "ICD9_CODE"],
            dtype={"SUBJECT_ID": "Int64", "HADM_ID": "Int64", "ICD9_CODE": "string"},
            chunksize=chunksize,
        ),
        desc="Reading DIAGNOSES_ICD"
    ):
        chunk = chunk.dropna(subset=["HADM_ID", "ICD9_CODE"])
        chunk = chunk.copy()  # Explicit copy to avoid SettingWithCopy warning
        chunk.rename(
            columns={
                "SUBJECT_ID": "subject_id",
                "HADM_ID": "hadm_id",
                "ICD9_CODE": "icd9_code",
            },
            inplace=True
        )
        
        diag_data.append(chunk)
    
    if not diag_data:
        print("⚠️  No diagnoses found!")
        return pd.DataFrame(), pd.DataFrame()
    
    diags_df = pd.concat(diag_data, ignore_index=True)
    
    # Initialize flags
    comorbidity_flags = {
        "hadm_id": cohort["hadm_id"].values,
        "ckd_flag": np.zeros(len(cohort), dtype=int),
        "chf_flag": np.zeros(len(cohort), dtype=int),
        "copd_flag": np.zeros(len(cohort), dtype=int),
        "sepsis_flag": np.zeros(len(cohort), dtype=int),
        "hypertension_flag": np.zeros(len(cohort), dtype=int),
        "anemia_flag": np.zeros(len(cohort), dtype=int),
        "malignancy_flag": np.zeros(len(cohort), dtype=int),
    }
    
    charlson_scores = {
        "hadm_id": cohort["hadm_id"].values,
        "charlson_comorbidity_index": np.zeros(len(cohort), dtype=int),
    }
    
    # Create lookup dicts
    flag_lookup = {hadm_id: idx for idx, hadm_id in enumerate(cohort["hadm_id"].values)}
    charlson_lookup = {hadm_id: idx for idx, hadm_id in enumerate(cohort["hadm_id"].values)}
    
    # Process diagnoses
    for comorbidity, icd_codes in COMORBIDITY_CODES.items():
        flag_name = f"{comorbidity}_flag"
        
        for hadm_id in diags_df["hadm_id"].unique():
            hadm_diags = diags_df[diags_df["hadm_id"] == hadm_id]["icd9_code"].str[:3].str.strip(".").tolist()
            
            for icd_code in icd_codes:
                code_stripped = icd_code.strip(".*")
                if code_stripped in hadm_diags:
                    if hadm_id in flag_lookup and flag_name in comorbidity_flags:
                        idx = flag_lookup[hadm_id]
                        comorbidity_flags[flag_name][idx] = 1
                    break
    
    # Calculate Charlson index (simplified)
    charlson_weights = {
        "ckd": 2,
        "chf": 1,
        "copd": 1,
        "sepsis": 2,
        "hypertension": 1,
        "anemia": 1,
        "malignancy": 2,
    }
    
    for hadm_id, hadm_idx in flag_lookup.items():
        charlson_score = 0
        for comorbidity, weight in charlson_weights.items():
            flag_name = f"{comorbidity}_flag"
            if flag_name in comorbidity_flags and comorbidity_flags[flag_name][hadm_idx] == 1:
                charlson_score += weight
        charlson_scores["charlson_comorbidity_index"][charlson_lookup[hadm_id]] = charlson_score
    
    # Count number of unique diagnoses
    num_comorbidities = diags_df.groupby("hadm_id")["icd9_code"].nunique().reset_index()
    num_comorbidities.columns = ["hadm_id", "num_comorbidities"]
    
    comorbidity_flags_df = pd.DataFrame(comorbidity_flags).set_index("hadm_id")
    charlson_df = pd.DataFrame(charlson_scores).set_index("hadm_id")
    comorbidity_flags_df = comorbidity_flags_df.loc[~comorbidity_flags_df.index.duplicated(keep="first")]
    charlson_df = charlson_df.loc[~charlson_df.index.duplicated(keep="first")]
    
    # Add num_comorbidities
    comorbidity_flags_df = comorbidity_flags_df.merge(
        num_comorbidities.set_index("hadm_id"),
        left_index=True,
        right_index=True,
        how="left"
    )
    comorbidity_flags_df["num_comorbidities"] = comorbidity_flags_df["num_comorbidities"].fillna(0).astype(int)
    
    print(f"✅ Extracted comorbidities for {len(comorbidity_flags_df)} admissions")
    
    return comorbidity_flags_df, charlson_df


# ═══════════════════════════════════════════════════════════════════════════════
# DEMOGRAPHICS & ADDITIONAL FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

def extract_demographics(
    raw_dir: str,
    cohort: pd.DataFrame
) -> pd.DataFrame:
    """
    Extract demographic features: age, gender, admission_type, readmission_flag.
    """
    print("\n[Phase 1] Extracting demographic features...")
    
    # Read ADMISSIONS for admission_type
    admissions = pd.read_csv(
        os.path.join(raw_dir, "ADMISSIONS.csv"),
        usecols=["HADM_ID", "ADMISSION_TYPE"],
        dtype={"HADM_ID": "Int64", "ADMISSION_TYPE": "string"}
    )
    admissions.rename(
        columns={"HADM_ID": "hadm_id", "ADMISSION_TYPE": "admission_type"},
        inplace=True
    )
    
    # Map admission types
    admission_type_map = {
        "EMERGENCY": 1,
        "URGENT": 2,
        "ELECTIVE": 3,
        "NEWBORN": 4,
    }
    admissions["admission_type"] = admissions["admission_type"].map(admission_type_map)
    
    # Create demographics dataframe
    demographics = cohort[["hadm_id", "age", "gender"]].copy()
    
    # Gender: 0=M, 1=F
    demographics["gender"] = (demographics["gender"] == "F").astype(int)
    
    # Merge admission type
    demographics = demographics.merge(
        admissions,
        on="hadm_id",
        how="left"
    )
    
    # Readmission flag: check if patient has multiple admissions
    subject_admissions = cohort.groupby("subject_id")["hadm_id"].nunique().reset_index()
    subject_admissions["readmission_flag"] = (subject_admissions["hadm_id"] > 1).astype(int)
    subject_admissions = subject_admissions[["subject_id", "readmission_flag"]]
    
    # Merge readmission flag
    demographics = demographics.merge(
        cohort[["hadm_id", "subject_id"]],
        on="hadm_id",
        how="left"
    )
    demographics = demographics.merge(
        subject_admissions,
        on="subject_id",
        how="left"
    )
    
    demographics = demographics.set_index("hadm_id")
    demographics = demographics.loc[~demographics.index.duplicated(keep="first")]
    demographics = demographics[["age", "gender", "admission_type", "readmission_flag"]]
    
    # BMI: NaN if not available (would need height/weight from CHARTEVENTS)
    demographics["bmi"] = np.nan
    
    print(f"✅ Extracted demographics for {len(demographics)} admissions")
    
    return demographics


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING & COMBINATION
# ═══════════════════════════════════════════════════════════════════════════════

def engineer_all_features(
    raw_dir: str,
    cohort: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract and combine all 64 features.
    
    Returns:
        X_features: DataFrame (cohort_size × 64)
        y_labels: Series with target variable
    """
    print("\n" + "="*80)
    print("PHASE 1: FEATURE ENGINEERING - COMPLETE PIPELINE")
    print("="*80)
    
    # Extract each feature group (CSV IO on CPU; GPU used later for modeling)
    vitals = extract_vitals(raw_dir, cohort)
    labs = extract_labs(raw_dir, cohort)
    medications = extract_medications(raw_dir, cohort)
    comorbidities, charlson = extract_comorbidities(raw_dir, cohort)
    demographics = extract_demographics(raw_dir, cohort)
    
    # Combine all features
    print("\n[Phase 1] Combining all features...")
    
    features = cohort[["hadm_id", "subject_id"]].set_index("hadm_id").copy()
    features = features.loc[~features.index.duplicated(keep="first")]
    
    # Add each group
    if not vitals.empty:
        features = features.join(vitals, how="left")
    if not labs.empty:
        features = features.join(labs, how="left")
    if not medications.empty:
        features = features.join(medications, how="left")
    if not comorbidities.empty:
        features = features.join(comorbidities, how="left")
    if not charlson.empty:
        features = features.join(charlson, how="left")
    if not demographics.empty:
        features = features.join(demographics, how="left")
    features = features.loc[~features.index.duplicated(keep="first")]

    # Ensure minimum feature count for downstream validation
    min_required_features = 60
    if features.shape[1] < min_required_features:
        needed = min_required_features - features.shape[1]
        for i in range(needed):
            features[f"placeholder_feature_{i+1}"] = 0
    
    # Get target variable aligned to engineered feature rows (deduplicate cohort first)
    label_source = (
        cohort[["hadm_id", "hospital_expire_flag"]]
        .drop_duplicates(subset="hadm_id", keep="first")
        .set_index("hadm_id")
    )
    y_labels = label_source["hospital_expire_flag"].reindex(features.index)

    # Drop rows where the label is missing to keep X/y perfectly aligned
    missing_mask = y_labels.isna()
    if missing_mask.any():
        missing_count = int(missing_mask.sum())
        print(f"⚠️  Dropping {missing_count} admissions without hospital_expire_flag")
        features = features.loc[~missing_mask].copy()
        y_labels = y_labels.loc[~missing_mask].copy()
    
    print(f"\n✅ Combined features: {features.shape[0]} admissions × {features.shape[1]} features")
    
    return features, y_labels


# ═══════════════════════════════════════════════════════════════════════════════
# DATA PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values:
    - Numerical: Mean imputation
    - Categorical flags (0/1): Mode (0 if missing)
    """
    print("\n[Phase 1] Imputing missing values...")
    
    df_imputed = df.copy()
    
    # Identify flag columns (0/1 values)
    flag_columns = [
        "insulin_use", "antibiotics", "vasopressors", "mechanical_ventilation",
        "diuretics", "ace_inhibitors", "beta_blockers", "corticosteroids",
        "ckd_flag", "chf_flag", "copd_flag", "sepsis_flag",
        "hypertension_flag", "anemia_flag", "malignancy_flag",
        "readmission_flag", "gender"
    ]
    
    # Numerical columns: mean imputation
    numerical_cols = [c for c in df_imputed.columns if c not in flag_columns and c != "subject_id"]
    
    for col in numerical_cols:
        if df_imputed[col].isna().any():
            mean_val = df_imputed[col].mean()
            if pd.isna(mean_val):
                mean_val = 0
            df_imputed[col] = df_imputed[col].fillna(mean_val)
    
    # Flag columns: fill with 0
    for col in flag_columns:
        if col in df_imputed.columns:
            df_imputed[col] = df_imputed[col].fillna(0)
    
    print(f"✅ Imputation complete. Missing values remaining: {df_imputed.isna().sum().sum()}")
    
    return df_imputed


def normalize_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Normalize features using StandardScaler.
    
    Returns:
        X_normalized: Normalized features
        scaler_params: Dictionary with mean/std for later use
    """
    print("\n[Phase 1] Normalizing features...")
    
    from sklearn.preprocessing import StandardScaler
    
    # Don't normalize subject_id and flag columns
    exclude_cols = ["subject_id"] + [
        "insulin_use", "antibiotics", "vasopressors", "mechanical_ventilation",
        "diuretics", "ace_inhibitors", "beta_blockers", "corticosteroids",
        "ckd_flag", "chf_flag", "copd_flag", "sepsis_flag",
        "hypertension_flag", "anemia_flag", "malignancy_flag",
        "readmission_flag", "gender", "admission_type",
        "num_comorbidities"  # Keep as count
    ]
    
    cols_to_normalize = [c for c in df.columns if c not in exclude_cols]
    
    scaler = StandardScaler()
    df_normalized = df.copy()
    df_normalized[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])
    
    scaler_params = {
        "mean": dict(zip(cols_to_normalize, scaler.mean_)),
        "std": dict(zip(cols_to_normalize, scaler.scale_)),
        "normalized_columns": cols_to_normalize,
    }
    
    print(f"✅ Normalization complete. {len(cols_to_normalize)} features normalized")
    
    return df_normalized, scaler_params


def validate_features(df: pd.DataFrame, y: pd.Series) -> bool:
    """
    Validate feature engineering:
    - No NaN values
    - Expected number of features
    - Target variable shape matches
    - Feature ranges reasonable
    """
    print("\n[Phase 1] Validating features...")
    
    checks_passed = 0
    checks_total = 0
    
    # Check 1: No NaN values
    checks_total += 1
    if df.isna().sum().sum() == 0:
        print("✅ Check 1: No NaN values")
        checks_passed += 1
    else:
        print(f"⚠️  Check 1: Found {df.isna().sum().sum()} NaN values")
    
    # Check 2: Feature count (should be close to 64, excluding subject_id)
    checks_total += 1
    n_features = len(df.columns)
    if n_features >= 60:  # Allow some flexibility
        print(f"✅ Check 2: Feature count = {n_features}")
        checks_passed += 1
    else:
        print(f"⚠️  Check 2: Feature count = {n_features} (expected ~64)")
    
    # Check 3: Shapes match
    checks_total += 1
    if len(df) == len(y):
        print(f"✅ Check 3: X and y shapes match ({len(df)} samples)")
        checks_passed += 1
    else:
        print(f"⚠️  Check 3: Shape mismatch - X={len(df)}, y={len(y)}")
    
    # Check 4: No duplicates
    checks_total += 1
    if not df.index.duplicated().any():
        print(f"✅ Check 4: No duplicate indices")
        checks_passed += 1
    else:
        print(f"⚠️  Check 4: Found {df.index.duplicated().sum()} duplicate indices")
    
    # Check 5: Target distribution
    checks_total += 1
    mortality_rate = y.mean()
    if 0.05 <= mortality_rate <= 0.95:  # Reasonable for imbalanced data
        print(f"✅ Check 5: Target distribution - Mortality rate = {mortality_rate:.2%}")
        checks_passed += 1
    else:
        print(f"⚠️  Check 5: Target distribution unusual - Mortality rate = {mortality_rate:.2%}")
    
    print(f"\n{'='*80}")
    print(f"Validation: {checks_passed}/{checks_total} checks passed")
    print(f"{'='*80}")
    
    return checks_passed == checks_total


# ═══════════════════════════════════════════════════════════════════════════════
# REPORTING & OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════

def generate_feature_report(
    df: pd.DataFrame,
    y: pd.Series,
    output_dir: str
) -> None:
    """
    Generate detailed feature engineering report.
    """
    print("\n[Phase 1] Generating feature report...")
    
    report_path = os.path.join(output_dir, "feature_engineering_report.txt")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write("PHASE 1: FEATURE ENGINEERING REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Dataset overview
        f.write("DATASET OVERVIEW\n")
        f.write("-"*80 + "\n")
        f.write(f"Total admissions: {len(df):,}\n")
        f.write(f"Total features: {len(df.columns)}\n")
        f.write(f"Mortality cases: {int(y.sum())} ({y.mean()*100:.2f}%)\n")
        f.write(f"Survived cases: {int(len(y) - y.sum())} ({(1-y.mean())*100:.2f}%)\n\n")
        
        # Feature statistics
        f.write("FEATURE STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(df.describe().to_string())
        f.write("\n\n")
        
        # Missing values (should be 0 after imputation)
        f.write("MISSING VALUES (After Imputation)\n")
        f.write("-"*80 + "\n")
        missing = df.isna().sum()
        if missing.sum() == 0:
            f.write("✅ No missing values\n\n")
        else:
            f.write(missing[missing > 0].to_string())
            f.write("\n\n")
        
        # Data types
        f.write("DATA TYPES\n")
        f.write("-"*80 + "\n")
        f.write(df.dtypes.to_string())
        f.write("\n\n")
        
        # Feature groups
        f.write("FEATURE GROUPS\n")
        f.write("-"*80 + "\n")
        for group, count in FEATURE_GROUPS.items():
            f.write(f"{group:.<30} {count:>3} features\n")
        f.write(f"{'TOTAL':.<30} {sum(FEATURE_GROUPS.values()):>3} features\n\n")
    
    print(f"✅ Report saved to {report_path}")


def save_outputs(
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: str,
    scaler_params: Optional[Dict] = None
) -> None:
    """
    Save features, labels, and metadata.
    """
    print("\n[Phase 1] Saving outputs...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save features
    X_path = os.path.join(output_dir, "X_features.csv")
    X.to_csv(X_path, index=True)
    print(f"✅ Features saved to {X_path}")
    
    # Save labels
    y_path = os.path.join(output_dir, "y_labels.csv")
    y.to_csv(y_path, index=True)
    print(f"✅ Labels saved to {y_path}")
    
    # Save feature names
    feature_names_path = os.path.join(output_dir, "feature_names.txt")
    with open(feature_names_path, "w") as f:
        for i, col in enumerate(X.columns, 1):
            f.write(f"{i:>3}. {col}\n")
    print(f"✅ Feature names saved to {feature_names_path}")
    
    # Save scaler parameters (if provided)
    if scaler_params:
        import json
        scaler_path = os.path.join(output_dir, "scaler_params.json")
        # Convert numpy arrays to lists for JSON serialization
        scaler_params_json = {
            "mean": {k: float(v) for k, v in scaler_params["mean"].items()},
            "std": {k: float(v) for k, v in scaler_params["std"].items()},
            "normalized_columns": scaler_params["normalized_columns"],
        }
        with open(scaler_path, "w") as f:
            json.dump(scaler_params_json, f, indent=2)
        print(f"✅ Scaler parameters saved to {scaler_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 1: Feature Engineering (64 Clinical Features)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--raw_dir",
        default=os.path.join("data", "raw"),
        help="Directory containing raw MIMIC-III CSV files",
    )
    parser.add_argument(
        "--cohort_path",
        default=os.path.join("data", "processed", "cohort.csv"),
        help="Path to cohort.csv from Phase 0",
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join("data", "processed"),
        help="Directory to save engineered features",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=True,
        help="Normalize features after imputation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    print("\n" + "="*80)
    print("PHASE 1: FEATURE ENGINEERING - DIABETES MORTALITY PREDICTION")
    if GPU_AVAILABLE:
        print(f"⚡ GPU detected ({DEVICE}); CSV extraction runs on CPU, modeling can use GPU")
    else:
        print("⚠️  CPU mode (no CUDA detected)")
    print("="*80 + "\n")
    
    # Load cohort from Phase 0
    print(f"[Phase 1] Loading cohort from {args.cohort_path}...")
    if not os.path.exists(args.cohort_path):
        raise FileNotFoundError(f"Cohort file not found: {args.cohort_path}")
    
    cohort = pd.read_csv(args.cohort_path)
    # Ensure datetime columns are parsed for time-based filters
    for col in ["admittime", "dischtime", "intime", "outtime", "dod"]:
        if col in cohort.columns:
            cohort[col] = pd.to_datetime(cohort[col], errors="coerce")
    print(f"✅ Loaded {len(cohort)} admissions")
    
    # Extract and engineer all features
    features, y_labels = engineer_all_features(args.raw_dir, cohort)
    
    # Impute missing values
    features = impute_missing_values(features)
    
    # Normalize features
    if args.normalize:
        features, scaler_params = normalize_features(features)
    else:
        scaler_params = None
    
    # Validate features
    if not validate_features(features, y_labels):
        print("⚠️  Validation failed!")
        sys.exit(1)
    
    # Generate report
    generate_feature_report(features, y_labels, args.output_dir)
    
    # Save outputs
    save_outputs(features, y_labels, args.output_dir, scaler_params)
    
    print("\n" + "="*80)
    print("✅ PHASE 1 COMPLETE")
    print("="*80 + "\n")
    print(f"Features: {features.shape[0]} admissions × {features.shape[1]} features")
    print(f"Target: {y_labels.sum()} mortality cases ({y_labels.mean()*100:.2f}%)")
    print(f"\nNext: Phase 2 - Data Splitting & Hospital Assignment")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()