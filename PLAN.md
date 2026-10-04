# PLAN.md — CAD risk prediction + interactive 3D coronary viewer

## 0. Decisions and assumptions
1. **TimesFM is not in the product pipeline**: TimesFM is designed for ordered time series. The dataset is 303 patients × one static snapshot. Using it introduces spurious temporal structure on 303 rows. Handled in report under "Considered and Rejected".
2. **Core tools**: scikit-learn, XGBoost, SHAP; FastAPI; React + Three.js via React Three Fiber.
3. **Training**: Local CPU training completes within seconds for 303 rows with repeated stratified CV. Colab launcher scripts provided in `scripts/`.
4. **Targets**: overall CAD = `Cath`; vessels = `LAD`, `LCX`, `RCA`. All four excluded from X for all models to prevent leakage.
5. **Validation**: Repeated Stratified K-Fold CV (5-fold × 3 repeats) for robust, un-leaked evaluation; final model fit on entire dataset.
6. **Interpretability**: Fast, exact tree/linear SHAP explanations served dynamically per patient.
7. **Clinical Safety**: Clear, persistent disclaimer banner and footer across the UI.

## 1. Directory Structure
```
cad/
├─ CLAUDE.md
├─ PLAN.md
├─ data/raw/
├─ reports/
├─ ml/
│  ├─ schema.json
│  ├─ pipeline.py
│  ├─ train.py
│  ├─ explain.py
│  ├─ requirements.txt
│  ├─ artifacts/
│  └─ tests/
├─ api/main.py
├─ web/
├─ scripts/
├─ docs/
└─ Makefile
```
