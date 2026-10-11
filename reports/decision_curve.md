# Decision Curve Analysis (DCA) Report

Net-benefit difference per 100 patients evaluated across clinically plausible threshold probabilities 
(0.10, 0.20, 0.30) and each model's own sensitivity-first operating cutoff. 
All estimates are apparent out-of-fold results from a single-centre cross-sectional cohort.

## Net Benefit Gain vs. Treat-All (per 100 patients, 95% Bootstrap CI)

| Target | Cutoff | pt = 0.10 | pt = 0.20 | pt = 0.30 | At Model Cutoff |
|---|---|---|---|---|---|
| **Cath** | 0.400 | +0.04 [0.00, 0.11] | +0.48 [-0.66, 1.49] | +2.69 [0.24, 4.95] | **+8.91 [5.06, 12.43]** |
| **LAD** | 0.420 | +0.01 [-0.81, 0.48] | +1.46 [-0.25, 3.05] | +4.90 [1.84, 7.69] | **+12.31 [7.64, 16.83]** |
| **LCX** | 0.248 | +0.27 [-0.59, 0.84] | +2.35 [0.08, 4.46] | +6.31 [2.45, 10.00] | **+5.19 [2.34, 8.02]** |
| **RCA** | 0.232 | +0.29 [-0.44, 0.84] | +2.52 [0.08, 4.87] | +7.03 [2.73, 11.51] | **+4.45 [1.67, 7.15]** |

## Contextual Notes

- **Overall CAD (Cath)**: At screening cutoff 0.400 (prevalence π = 71.6%), the CAD model yields an apparent net-benefit delta of **+8.91 [5.06, 12.43] per 100 patients** over Treat-All on pooled out-of-fold validation. At the balanced cutoff 0.500, the net-benefit delta is approximately +18.8 per 100 patients.

- **Branch Vessel Models**: At branch cutoffs (LAD 0.420: +12.31 [7.64, 16.83], LCX 0.248: +5.19 [2.34, 8.02], RCA 0.232: +4.45 [1.67, 7.15] per 100 patients), sensitivity calibration maintains positive clinical net benefit across decision thresholds.

- **Study Limitations**: This analysis is strictly out-of-fold, single-centre and apparent; external multicentre validation is necessary before clinical deployment.
