# Decision Curve Analysis (DCA) Report

Net-benefit difference per 100 patients evaluated across clinically plausible threshold probabilities 
(0.10, 0.20, 0.30) and each model's own sensitivity-first operating cutoff. 
All estimates are apparent out-of-fold results from a single-centre cross-sectional cohort.

## Net Benefit Gain vs. Treat-All (per 100 patients, 95% Bootstrap CI)

| Target | Cutoff | pt = 0.10 | pt = 0.20 | pt = 0.30 | At Model Cutoff |
|---|---|---|---|---|---|
| **Cath** | 0.611 | +0.18 [0.04, 0.37] | +1.24 [0.17, 2.15] | +2.64 [0.38, 4.76] | **+29.73 [21.39, 38.17]** |
| **LAD** | 0.469 | +-0.00 [0.00, 0.00] | +0.08 [-0.00, 0.25] | +2.52 [0.71, 4.34] | **+15.87 [10.34, 21.15]** |
| **LCX** | 0.217 | +0.20 [-0.92, 1.03] | +1.32 [-1.24, 3.88] | +6.87 [2.97, 10.85] | **+1.71 [-1.08, 4.62]** |
| **RCA** | 0.213 | +0.59 [-0.40, 1.47] | +1.85 [-0.91, 4.29] | +6.20 [2.07, 10.56] | **+2.13 [-0.83, 4.74]** |

## Contextual Notes

- **Overall CAD (Cath)**: At cutoff 0.611, the CAD model yields a net-benefit delta of approximately +29.6 per 100 patients over Treat-All.

- **Branch Vessel Models**: At branch cutoffs (LCX 0.217, RCA 0.213), the net-benefit delta is modest (+1.7 to +2.2 per 100 patients, corresponding to approx +0.02 to +0.03 in standard net benefit units) due to high sensitivity calibration.

- **Study Limitations**: This analysis is strictly out-of-fold, single-centre and apparent; external multicentre validation is necessary before clinical deployment.
