# Data Audit Report: UCI Z-Alizadeh Sani Extension Dataset

## 1. Overview and Dataset Dimensions
- **Source File**: `data/raw/extention of Z-Alizadeh sani dataset.xlsx`
- **Total Records (Patients)**: 303
- **Total Columns**: 59
- **Duplicate Rows**: 0
- **Missing Values**: 0 across all 59 columns (clean, complete clinical dataset)

## 2. UCI 59 Columns vs. Published 54 Features Accounting
The UCI repository file contains 59 columns. Published studies on this dataset cite 54 features.
The breakdown is:
- **4 Target Columns**:
  - `Cath`: Overall CAD status (Stenosis ≥ 50% in at least one major vessel)
  - `LAD`: Left Anterior Descending artery stenosis
  - `LCX`: Left Circumflex artery stenosis
  - `RCA`: Right Coronary Artery stenosis
- **1 Constant / Zero-Variance Column (Dropped)**:
  - `Exertional CP`: Value is `'N'` for 100% of rows (303/303). Offers 0 predictive information.
- **54 Active Input Features**:
  - 5 Demographic features
  - 25 Symptoms, Physical Exam, and Clinical History features
  - 7 Electrocardiogram (ECG) features
  - 14 Laboratory Blood Test features
  - 3 Echocardiographic features
  - **Total**: 4 + 1 + 54 = 59 columns.

## 3. Target Distribution and Logic Verification
### Target Class Balances:
- **LAD**:
  - `Stenotic`: 177 (58.42%)
  - `Normal`: 126 (41.58%)
- **LCX**:
  - `Stenotic`: 119 (39.27%)
  - `Normal`: 184 (60.73%)
- **RCA**:
  - `Stenotic`: 114 (37.62%)
  - `Normal`: 189 (62.38%)
- **Cath (Overall CAD)**:
  - Raw UCI labels: `CAD`: 216 (71.29%), `Normal`: 87 (28.71%)
  - Target Consistency Audit:
    In the raw dataset, exactly 1 row exhibits a logical discrepancy where `Cath != (LAD | LCX | RCA)`:
    - **Pandas DataFrame index**: `93`
    - **Spreadsheet row** (Excel 1-indexed, header = row 1): `95`
    - **Patient attributes** (no patient ID present in raw data): Age: 65, Sex: 'Male', Weight: 73 kg, Length: 165 cm
    - **Raw target values**: `Cath='Normal'`, `LAD='Stenotic'`, `LCX='Normal'`, `RCA='Normal'`
    - **Alignment rationale**: Aligned to match the dataset's own definition (CAD = ≥1 stenotic vessel); the data cannot show which label is wrong (whether LAD was false-positive or Cath was false-negative). In derived training data, `Cath_aligned = Cath | LAD | LCX | RCA`.
    After alignment:
    `Cath`: 217 CAD (71.62%), 86 Normal (28.38%).
    `Cath == (LAD | LCX | RCA)` consistency: **100% (0 mismatches across 303 patients)**.

### Sensitivity Analysis: Recorded vs. Aligned Cath (LogisticRegression 5-Fold × 3 Repeats CV)

| Target Labeling | ROC-AUC | PR-AUC | F1-Score | Recall (Sens.) | Specificity | Brier Score | Accuracy |
|---|---|---|---|---|---|---|---|
| **Recorded Cath** (`align_row_93=False`, 216 CAD / 87 Normal) | 0.926 ± 0.030 | 0.969 ± 0.013 | 0.906 ± 0.026 | 0.926 ± 0.038 | 0.705 ± 0.120 | 0.100 ± 0.017 | 0.862 ± 0.039 |
| **Aligned Cath** (`align_row_93=True`, 217 CAD / 86 Normal) | 0.929 ± 0.024 | 0.971 ± 0.011 | 0.910 ± 0.021 | 0.934 ± 0.031 | 0.698 ± 0.083 | 0.098 ± 0.012 | 0.867 ± 0.032 |

The single-patient alignment improves label consistency to 100% with negligible variation in cross-validated performance (ROC-AUC shift: +0.003, Brier shift: -0.002).

## 4. Column-by-Column Inventory and Roles

| Index | Column Name | Raw Dtype | Unique Values | Role | Reason / Processing |
|---|---|---|---|---|---|
| 0 | Age | int64 | 46 | Feature (Demographic) | Continuous [30 - 86], years |
| 1 | Weight | int64 | 54 | Feature (Demographic) | Continuous [48 - 120], kg |
| 2 | Length | int64 | 44 | Feature (Demographic) | Continuous [140 - 188], cm (Height) |
| 3 | Sex | str | 2 ('Male', 'Fmale') | Feature (Demographic) | Binary categorical ('Male' vs 'Fmale') |
| 4 | BMI | float64 | 263 | Feature (Demographic) | Continuous [18.1 - 40.9], kg/m² |
| 5 | DM | int64 | 2 (0, 1) | Feature (History) | Binary (Diabetes Mellitus) |
| 6 | HTN | int64 | 2 (0, 1) | Feature (History) | Binary (Hypertension) |
| 7 | Current Smoker | int64 | 2 (0, 1) | Feature (History) | Binary |
| 8 | EX-Smoker | int64 | 2 (0, 1) | Feature (History) | Binary |
| 9 | FH | int64 | 2 (0, 1) | Feature (History) | Binary (Family History) |
| 10 | Obesity | str | 2 ('Y', 'N') | Feature (History) | Binary ('Y'/'N') |
| 11 | CRF | str | 2 ('Y', 'N') | Feature (History) | Binary (Chronic Renal Failure) |
| 12 | CVA | str | 2 ('Y', 'N') | Feature (History) | Binary (Cerebrovascular Accident) |
| 13 | Airway disease | str | 2 ('Y', 'N') | Feature (History) | Binary (COPD / Asthma) |
| 14 | Thyroid Disease | str | 2 ('Y', 'N') | Feature (History) | Binary |
| 15 | CHF | str | 2 ('Y', 'N') | Feature (History) | Binary (Congestive Heart Failure) |
| 16 | DLP | str | 2 ('Y', 'N') | Feature (History) | Binary (Dyslipidemia) |
| 17 | BP | int64 | 17 | Feature (Vitals) | Continuous [90 - 190], mmHg |
| 18 | PR | int64 | 21 | Feature (Vitals) | Continuous [50 - 110], bpm |
| 19 | Edema | int64 | 2 (0, 1) | Feature (Exam) | Binary (Peripheral edema) |
| 20 | Weak Peripheral Pulse | str | 2 ('Y', 'N') | Feature (Exam) | Binary |
| 21 | Lung rales | str | 2 ('Y', 'N') | Feature (Exam) | Binary |
| 22 | Systolic Murmur | str | 2 ('Y', 'N') | Feature (Exam) | Binary |
| 23 | Diastolic Murmur | str | 2 ('Y', 'N') | Feature (Exam) | Binary |
| 24 | Typical Chest Pain | int64 | 2 (0, 1) | Feature (Symptoms) | Binary |
| 25 | Dyspnea | str | 2 ('Y', 'N') | Feature (Symptoms) | Binary |
| 26 | Function Class | int64 | 4 (0, 1, 2, 3) | Feature (Symptoms) | Ordinal (NYHA class 0-3) |
| 27 | Atypical | str | 2 ('Y', 'N') | Feature (Symptoms) | Binary (Atypical chest pain) |
| 28 | Nonanginal | str | 2 ('Y', 'N') | Feature (Symptoms) | Binary (Non-anginal chest pain) |
| 29 | Exertional CP | str | 1 ('N') | **Drop** | Zero variance (constant across 303 rows) |
| 30 | LowTH Ang | str | 2 ('Y', 'N') | Feature (Symptoms) | Binary (Low threshold angina) |
| 31 | Q Wave | int64 | 2 (0, 1) | Feature (ECG) | Binary (Pathological Q wave) |
| 32 | St Elevation | int64 | 2 (0, 1) | Feature (ECG) | Binary |
| 33 | St Depression | int64 | 2 (0, 1) | Feature (ECG) | Binary |
| 34 | Tinversion | int64 | 2 (0, 1) | Feature (ECG) | Binary (T-wave inversion) |
| 35 | LVH | str | 2 ('Y', 'N') | Feature (ECG) | Binary (Left Ventricular Hypertrophy) |
| 36 | Poor R Progression | str | 2 ('Y', 'N') | Feature (ECG) | Binary |
| 37 | BBB | str | 3 ('N', 'LBBB', 'RBBB') | Feature (ECG) | Nominal categorical (Bundle Branch Block) |
| 38 | FBS | int64 | 113 | Feature (Lab) | Continuous [62 - 400], mg/dL (Fasting Blood Sugar) |
| 39 | CR | float64 | 18 | Feature (Lab) | Continuous [0.5 - 2.2], mg/dL (Creatinine) |
| 40 | TG | int64 | 147 | Feature (Lab) | Continuous [37 - 1050], mg/dL (Triglycerides) |
| 41 | LDL | int64 | 110 | Feature (Lab) | Continuous [18 - 232], mg/dL (Low-Density Lipoprotein) |
| 42 | HDL | float64 | 47 | Feature (Lab) | Continuous [15.9 - 111], mg/dL (High-Density Lipoprotein) |
| 43 | BUN | int64 | 33 | Feature (Lab) | Continuous [6 - 52], mg/dL (Blood Urea Nitrogen) |
| 44 | ESR | int64 | 58 | Feature (Lab) | Continuous [1 - 90], mm/hr (Erythrocyte Sedimentation Rate) |
| 45 | HB | float64 | 66 | Feature (Lab) | Continuous [8.9 - 17.6], g/dL (Hemoglobin) |
| 46 | K | float64 | 27 | Feature (Lab) | Continuous [3.0 - 6.6], mEq/L (Potassium) |
| 47 | Na | int64 | 25 | Feature (Lab) | Continuous [128 - 156], mEq/L (Sodium) |
| 48 | WBC | int64 | 78 | Feature (Lab) | Continuous [3700 - 18000], /µL (White Blood Cell Count) |
| 49 | Lymph | int64 | 50 | Feature (Lab) | Continuous [7 - 60], % (Lymphocyte count) |
| 50 | Neut | int64 | 52 | Feature (Lab) | Continuous [32 - 89], % (Neutrophil count) |
| 51 | PLT | int64 | 135 | Feature (Lab) | Continuous [25 - 742], 1000/µL (Platelets) |
| 52 | EF-TTE | int64 | 11 | Feature (Echo) | Continuous [15 - 60], % (Ejection Fraction) |
| 53 | Region RWMA | int64 | 5 (0, 1, 2, 3, 4) | Feature (Echo) | Ordinal/Count [0 - 4] (Number of Wall Motion Abnormality Regions) |
| 54 | VHD | str | 4 ('N', 'mild', 'Moderate', 'Severe') | Feature (Echo) | Ordinal categorical (Valvular Heart Disease) |
| 55 | LAD | str | 2 ('Stenotic', 'Normal') | **Target** | Stenosis in Left Anterior Descending |
| 56 | LCX | str | 2 ('Stenotic', 'Normal') | **Target** | Stenosis in Left Circumflex |
| 57 | RCA | str | 2 ('Stenotic', 'Normal') | **Target** | Stenosis in Right Coronary Artery |
| 58 | Cath | str | 2 ('CAD', 'Normal') | **Target** | Overall Coronary Artery Disease diagnosis |

## 5. Leakage Prevention
Per Invariant 1 and Track requirements:
`LAD`, `LCX`, `RCA`, and `Cath` are strictly excluded from input feature matrix `X` for all models. No target column or catheterization finding is ever present as a model input.
