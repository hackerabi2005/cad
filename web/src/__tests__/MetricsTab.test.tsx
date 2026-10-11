import React from 'react';
import { describe, it, expect } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { MetricsTab } from '../components/MetricsTab';
import { ModelMetricsResponse } from '../types';

describe('MetricsTab Component', () => {
  const mockMetricsData: ModelMetricsResponse = {
    metrics: {
      Cath: {
        LogisticRegression: {
          roc_auc_mean: 0.929,
          roc_auc_std: 0.024,
          pr_auc_mean: 0.941,
          f1_mean: 0.895,
          recall_mean: 0.912,
          specificity_mean: 0.812,
          brier_mean: 0.098,
        },
      },
      LAD: {
        RandomForest: {
          roc_auc_mean: 0.846,
          roc_auc_std: 0.049,
          pr_auc_mean: 0.880,
          f1_mean: 0.821,
          recall_mean: 0.850,
          specificity_mean: 0.760,
          brier_mean: 0.145,
        },
      },
    },
    shap_global: {},
    model_card: {
      selected_models: {
        Cath: 'LogisticRegression',
        LAD: 'RandomForest',
      },
      operating_thresholds: {
        Cath: {
          cutoff: 0.611,
          sensitivity: 0.903,
          specificity: 0.826,
          cv_roc_auc: 0.929,
          cv_brier: 0.098,
          high_sensitivity_threshold: 0.611,
        },
        LAD: {
          cutoff: 0.469,
          sensitivity: 0.904,
          specificity: 0.603,
          cv_roc_auc: 0.846,
          cv_brier: 0.145,
          high_sensitivity_threshold: 0.469,
        },
      },
    },
  };

  it('renders loading state when metricsData is undefined', () => {
    render(<MetricsTab metricsData={undefined} />);
    expect(screen.getByText(/Loading validation metrics/i)).toBeInTheDocument();
  });

  it('renders validation metrics for Cath target and switches to LAD', () => {
    render(<MetricsTab metricsData={mockMetricsData} />);
    expect(screen.getByText(/Rigorous ML Validation & Calibration/i)).toBeInTheDocument();
    expect(screen.getByText('0.929')).toBeInTheDocument();

    const ladBtn = screen.getByRole('button', { name: 'LAD' });
    fireEvent.click(ladBtn);
    expect(screen.getByText('0.846')).toBeInTheDocument();
  });

  it('renders multi-vessel burden matrix and toggles operating cutoff modes', () => {
    render(<MetricsTab metricsData={mockMetricsData} />);
    expect(screen.getByText(/Multi-Vessel Burden: Flagged Vessels/i)).toBeInTheDocument();
    expect(screen.getByText(/0 Vessels \(Normal\)/i)).toBeInTheDocument();
    expect(screen.getByText(/3 Vessels \(Triple\)/i)).toBeInTheDocument();

    // Toggle to screening mode
    const screeningBtn = screen.getByRole('button', { name: /Screening Cutoffs/i });
    fireEvent.click(screeningBtn);
    expect(screen.getByText(/Triple vessel sensitivity:/i)).toBeInTheDocument();
  });

  it('renders diagnostic curves with calibration slope, intercept and brier score', () => {
    const metricsWithCurves: ModelMetricsResponse = {
      ...mockMetricsData,
      eval_curves: {
        roc: { Cath: { fpr: [0, 0.2, 1], tpr: [0, 0.9, 1], thresholds: [1, 0.5, 0], auc: 0.932 } },
        pr: { Cath: { precision: [0.9, 0.8, 0.7], recall: [0.1, 0.8, 1], thresholds: [0.9, 0.5, 0], pr_auc: 0.973 } },
        calibration: { Cath: { prob_true: [0.1, 0.5, 0.9], prob_pred: [0.12, 0.48, 0.92], brier: 0.096, slope: 1.16, intercept: -0.15 } as any },
        confusion_matrices: {},
        decision_curve: {},
      },
    };

    render(<MetricsTab metricsData={metricsWithCurves} />);
    const curvesTab = screen.getByRole('button', { name: /Interactive Diagnostic Curves/i });
    fireEvent.click(curvesTab);

    expect(screen.getByText(/Calibration Curve \(Reliability\)/i)).toBeInTheDocument();
    expect(screen.getByText(/Slope:/i)).toBeInTheDocument();
    expect(screen.getByText('1.16')).toBeInTheDocument();
    expect(screen.getByText('-0.15')).toBeInTheDocument();
  });
});
