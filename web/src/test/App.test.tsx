import React from 'react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { App } from '../App';

describe('App Main Component', () => {
  const mockSchema = {
    Age: {
      group: 'demographic',
      encoding: 'numeric',
      unit: 'years',
      label: 'Age',
      min: 30,
      max: 90,
      median: 60,
    },
    Sex: {
      group: 'demographic',
      encoding: 'nominal',
      unit: 'category',
      label: 'Sex',
      categories: ['Male', 'Female'],
      mode: 'Male',
    },
  };

  const mockSamples = [
    {
      id: 'patient-1',
      name: 'Sample Patient 1',
      description: 'Typical presentation',
      data: { Age: 55, Sex: 'Male' },
    },
  ];

  const mockMetrics = {
    schema_version: '2.0.0',
    targets: {
      Cath: {
        raw_cv_auc: 0.932,
        coherent_cv_auc: 0.930,
        brier_score: 0.098,
        accuracy: 0.88,
        f1: 0.91,
        sensitivity: 0.903,
        specificity: 0.826,
        threshold: 0.50,
        high_sensitivity_threshold: 0.611,
      },
    },
  };

  const mockPredictResponse = {
    cad: {
      prob: 0.65,
      raw_prob: 0.65,
      coherent_prob: 0.65,
      label: 'High Risk',
      high_sens_label: 'High Risk',
      threshold: 0.50,
      high_sensitivity_threshold: 0.611,
      nested_sensitivity: 0.903,
      coherence_adjusted: false,
    },
    vessels: {
      LAD: {
        prob: 0.40,
        label: 'Normal',
        high_sens_label: 'Normal',
        threshold: 0.50,
        high_sensitivity_threshold: 0.469,
      },
      LCX: {
        prob: 0.20,
        label: 'Normal',
        high_sens_label: 'Normal',
        threshold: 0.50,
        high_sensitivity_threshold: 0.217,
      },
      RCA: {
        prob: 0.15,
        label: 'Normal',
        high_sens_label: 'Normal',
        threshold: 0.50,
        high_sensitivity_threshold: 0.213,
      },
    },
    explain: {},
    cohort_defaults_applied: [],
    disclaimer: 'Safety disclaimer',
  };

  beforeEach(() => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockImplementation((url: string) => {
        if (url.includes('/api/schema')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve(mockSchema),
          });
        }
        if (url.includes('/api/samples')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve(mockSamples),
          });
        }
        if (url.includes('/api/metrics')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve(mockMetrics),
          });
        }
        if (url.includes('/api/predict')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve(mockPredictResponse),
          });
        }
        return Promise.reject(new Error('Unknown endpoint ' + url));
      })
    );
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('loads schema and sample data on initial mount', async () => {
    render(<App />);

    await waitFor(() => {
      expect(screen.getByText('Cardio3D AI')).toBeInTheDocument();
    });

    expect(globalThis.fetch).toHaveBeenCalledWith('/api/schema');
    expect(globalThis.fetch).toHaveBeenCalledWith('/api/samples');
    expect(globalThis.fetch).toHaveBeenCalledWith('/api/metrics');
  });

  it('opens AboutModelModal when clicking the About Model header button (G8)', async () => {
    render(<App />);

    await waitFor(() => {
      expect(screen.getByTestId('about-model-chip')).toBeInTheDocument();
    });

    const aboutBtn = screen.getByTestId('about-model-chip');
    fireEvent.click(aboutBtn);

    // Modal dialog is displayed
    expect(screen.getByRole('dialog')).toBeInTheDocument();
    expect(screen.getByText(/About Cardio3D AI & Clinical Safety/i)).toBeInTheDocument();

    // Closing the modal removes it
    const closeBtn = screen.getByRole('button', { name: /Close dialog/i });
    fireEvent.click(closeBtn);
    expect(screen.queryByRole('dialog')).toBeNull();
  });

  it('renders API error message banner when backend connection fails (A5)', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockRejectedValue(new Error('Network connection refused'))
    );

    render(<App />);

    await waitFor(() => {
      expect(screen.getByText(/Unable to connect to ML backend/i)).toBeInTheDocument();
    });
  });

  it('initializes operating mode from URL query parameters (mode=screening)', async () => {
    // Set URL search to ?mode=screening
    window.history.replaceState(null, '', '?mode=screening');

    render(<App />);

    await waitFor(() => {
      expect(screen.getByText('Cardio3D AI')).toBeInTheDocument();
    });

    // Risk overview should indicate screening mode
    await waitFor(() => {
      expect(screen.getByText(/Screening Mode/i)).toBeInTheDocument();
    });

    // Clean up URL
    window.history.replaceState(null, '', '/');
  });
});
