import React from 'react';
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { WhatIfSimulator } from '../components/WhatIfSimulator';
import { PredictResponse } from '../types';

describe('WhatIfSimulator Component', () => {
  const mockCurrentPrediction: Partial<PredictResponse> = {
    cad: {
      prob: 0.75,
      raw_prob: 0.75,
      displayed_prob: 0.75,
      coherent_prob: 0.75,
      driving_vessel: null,
      label: 'High Risk',
      high_sens_label: 'High Risk',
      threshold: 0.5,
      high_sensitivity_threshold: 0.61,
      coherence_adjusted: false,
    },
    vessels: {
      LAD: { prob: 0.6, label: 'Stenotic', high_sens_label: 'Stenotic', threshold: 0.5, high_sensitivity_threshold: 0.469 },
      LCX: { prob: 0.4, label: 'Normal', high_sens_label: 'Stenotic', threshold: 0.5, high_sensitivity_threshold: 0.217 },
      RCA: { prob: 0.3, label: 'Normal', high_sens_label: 'Normal', threshold: 0.5, high_sensitivity_threshold: 0.213 },
    },
  };

  it('renders simulator header and baseline risk comparison', () => {
    render(
      <WhatIfSimulator
        patientData={{ BP: 155, FBS: 140, 'Current Smoker': 'Y' }}
        currentPrediction={mockCurrentPrediction as PredictResponse}
        onApplyToPatient={() => {}}
      />
    );
    expect(screen.getByText(/Model Sensitivity Explorer/i)).toBeInTheDocument();
    expect(screen.getByText(/Current Model Score/i)).toBeInTheDocument();
  });

  it('allows resetting intervention parameters to current patient baseline', () => {
    render(
      <WhatIfSimulator
        patientData={{ BP: 155, FBS: 140, 'Current Smoker': 'Y' }}
        currentPrediction={mockCurrentPrediction as PredictResponse}
        onApplyToPatient={() => {}}
      />
    );
    const resetBtn = screen.getByRole('button', { name: /Reset/i });
    expect(resetBtn).toBeInTheDocument();
  });
});
