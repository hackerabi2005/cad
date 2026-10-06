# Logical Risk Coherence Evaluation: P(CAD) vs max(P_vessel)

## 1. Background and Motivation
Coronary artery disease (CAD) is defined as significant stenosis ($\ge 50\%$) in at least one major coronary artery (LAD, LCX, or RCA).
Logically, the event $\text{CAD} = \text{LAD} \lor \text{LCX} \lor \text{RCA}$ implies $P(\text{CAD}) \ge \max(P_{\text{LAD}}, P_{\text{LCX}}, P_{\text{RCA}})$.

However, because separate machine learning models are trained for each clinical target (Logistic Regression for Cath and RCA; Random Forest for LAD; XGBoost for LCX), tree-based probability distributions can be compressed or calibrated differently from linear models. Applying a naive $\max()$ post-processing rule could theoretically allow a weaker model (e.g., LAD AUC $\approx 0.85$ or LCX AUC $\approx 0.73$) to override the well-calibrated, high-performing overall CAD model (Cath AUC $\approx 0.93$).

Per FIX_PLAN Decision 3 and Fix F2, we empirically evaluate out-of-fold (OOF) cross-validation predictions to determine whether to retain the coherence rule:
**Retain `max()` only if displayed Brier $\le$ raw Brier $+ 0.01$ and AUC drop $\le 0.01$.**

---

## 2. Quantitative Evaluation on Out-of-Fold Predictions (15 Folds RSKF)

| Metric | Raw CAD Model (`LogisticRegression`) | Coherent CAD (`max(CAD, LAD, LCX, RCA)`) | Delta (Coherent - Raw) | Threshold Criterion | Decision |
|---|---|---|---|---|---|
| **ROC-AUC** | **0.9302** | **0.9291** | **-0.0012** | AUC drop $\le 0.01$ | **PASS** |
| **Brier Score** | **0.0967** | **0.1017** | **+0.0050** | Displayed Brier $\le$ Raw $+ 0.01$ | **PASS** |
| **Patients Changed** | — | **21.5% (65 / 303)** | — | Informational | — |
| **Median Shift (Changed)** | — | **+0.0781** | — | Informational | — |

---

## 3. Reliability and Calibration Analysis
- Looking at the calibration curves across targets:
  - Cath (Logistic Regression) exhibits strong probabilistic calibration with a Brier score of $0.098$.
  - Enforcing $P(\text{CAD}) \leftarrow \max(P(\text{CAD}), \max(P_{\text{vessel}}))$ shifts probabilities upwards for 65 out of 303 patients where a vessel model predicted a higher local stenosis likelihood than the systemic CAD classifier.
  - The Brier score increases by only $0.0050$ (from $0.0967$ to $0.1017$), staying well within the $+0.010$ budget.
  - The ROC-AUC decreases by an almost negligible $0.0012$ (from $0.9302$ to $0.9291$).

---

## 4. Final Policy Decision
1. **Retain `max()` Coherence Rule**:
   The mathematical criteria (Brier increase $\le 0.01$, AUC drop $\le 0.01$) are satisfied. Displayed CAD risk respects anatomical logic without degrading predictive discrimination.
2. **Transparent API Contract**:
   The API provides both `raw_prob` (the pure CAD logistic regression output) and `prob` (the coherent displayed score), along with a boolean `coherence_adjusted` flag.
3. **Thresholding on Displayed Score**:
   The clinical operating threshold for CAD is calibrated directly on the displayed score (0.611) to maintain $\ge 90\%$ sensitivity on the final displayed probability.
4. **No Discrete Label Rule**:
   We explicitly reject an "any vessel High $\implies$ overall High" label override. Vessel models use sensitivity-first thresholds ($\sim 0.21$–$0.47$) tuned to capture subtle localized disease; applying discrete OR label overriding would sharply degrade specificity, flagging excessive false positives.
