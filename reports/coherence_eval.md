# Logical Risk Coherence Evaluation: P(CAD) vs max(P_vessel)

## 1. Background and Motivation
Coronary artery disease (CAD) is defined as significant stenosis ($\ge 50\%$) in at least one major coronary artery (LAD, LCX, or RCA).
Logically, the event $\text{CAD} = \text{LAD} \lor \text{LCX} \lor \text{RCA}$ implies $P(\text{CAD}) \ge \max(P_{\text{LAD}}, P_{\text{LCX}}, P_{\text{RCA}})$.

However, while LogitEnsemble is trained across all four clinical targets, local branch models and the systemic CAD classifier are optimized independently. Applying a naive $\max()$ post-processing rule could theoretically allow a branch model (e.g., LAD AUC $\approx 0.84$ or LCX AUC $\approx 0.73$) to override the well-calibrated, high-performing overall CAD model (Cath AUC $\approx 0.93$).

To validate this design decision, we empirically evaluate out-of-fold (OOF) cross-validation predictions to determine whether to retain the coherence rule:
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
4. **Logical Status Harmonization (Clinical Disjunctive Rule)**:
   To eliminate clinical contradictions where an individual vessel is flagged as Stenotic while overall CAD is marked Low Risk (e.g., a patient where $P_{\text{LCX}} \ge 0.248$ while $P_{\text{CAD}} < 0.646$), the diagnostic status engine evaluates:
   $$\text{CAD Flagged} \iff (P(\text{CAD})_{\text{coherent}} \ge \text{cutoff}_{\text{CAD}}) \lor (\text{LAD Flagged}) \lor (\text{LCX Flagged}) \lor (\text{RCA Flagged})$$

   ### Definition of Discordant States & Empirical Frequencies (303 Out-of-Fold Patients)
   We distinguish two distinct directional states of discordance:

   1. **Direction 1: Vessel-Positive / CAD-Negative Contradiction**
      - *Definition*: At least one vessel model is flagged ($\ge \text{cutoff}_V$), but the baseline CAD model was below threshold ($< \text{cutoff}_{\text{CAD}}$).
      - *Baseline rate*: **10.89% (33 / 303 patients)** were contradictory before status harmonization.
      - *Harmonized rate*: **0.0% (0 / 303 patients)**. This contradiction is eliminated **strictly by construction** through the disjunctive rule ($A \lor B \implies A \text{ is flagged if } B \text{ is flagged}$).
      - *Trade-off*: Harmonization ensures clinical consistency; when lowering thresholds for screening, sensitivity rises (94.5% $\to$ 96.3%) while specificity drops (70.9% $\to$ 57.0%). The disjunctive rule buys consistency, not standalone classifier accuracy.

   2. **Direction 2: CAD-Positive / Zero-Vessels-Flagged Discordance (Reverse Discordance)**
      - *Definition*: Overall CAD model flags elevated risk ($\ge \text{cutoff}_{\text{CAD}}$), but no single epicardial vessel crosses its respective branch cutoff.
      - *Occurrence rate*:
        - **Balanced Mode (0.50 cutoff)**: 33 / 303 patients (10.9%)
        - **Screening Mode**: 24 / 303 patients (7.9%)
      - *Clinical Interpretation*: Occurs when diffuse systemic features (advanced age, severe dyslipidemia, systemic hypertension) drive global ischemic probability above cutoff, but branch-specific signals (localized ECG changes or wall motion abnormalities) are insufficient to localize stenosis to LAD, LCX, or RCA. The workstation explicitly informs the clinician: *"Overall model above cutoff; no vessel model above its cutoff; location not determined."*
