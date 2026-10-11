import React from 'react';
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { RiskOverview } from '../components/RiskOverview';
import { CadPrediction } from '../types';

describe('RiskOverview Component', () => {
  const baseCad: CadPrediction = {
    prob: 0.72,
    raw_prob: 0.58,
    displayed_prob: 0.72,
    coherent_prob: 0.72,
    driving_vessel: 'LAD',
    coherence_adjusted: true,
    label: 'High Risk',
    high_sens_label: 'High Risk',
    threshold: 0.50,
    high_sensitivity_threshold: 0.61,
    nested_sensitivity: 0.903,
    similar_cohort: {
      n: 42,
      observed_rate: 0.74,
      wilson_interval: [0.59, 0.85],
      summary: '42 similar patients in cohort',
    },
  };

  it('renders loading state when cad is undefined (A9 neutral state)', () => {
    render(<RiskOverview cad={undefined} />);
    expect(screen.getByText(/Awaiting risk calculation/i)).toBeInTheDocument();
  });

  it('renders raw model score and coherent risk score distinctly (A1)', () => {
    render(<RiskOverview cad={baseCad} mode="balanced" />);

    // Raw score should show 58.0%
    expect(screen.getByText('58.0%')).toBeInTheDocument();
    // Coherent score should show 72.0%
    expect(screen.getByText('72.0%')).toBeInTheDocument();
  });

  it('renders driving vessel badge when coherence_adjusted is true (A1)', () => {
    render(<RiskOverview cad={baseCad} mode="balanced" />);

    // Badge showing Max (LAD)
    const badge = screen.getByText(/Max \(LAD\)/i);
    expect(badge).toBeInTheDocument();
  });

  it('does not render driving vessel badge when coherence_adjusted is false (A1)', () => {
    const unadjustedCad: CadPrediction = {
      ...baseCad,
      raw_prob: 0.85,
      coherent_prob: 0.85,
      coherence_adjusted: false,
      driving_vessel: null,
    };
    render(<RiskOverview cad={unadjustedCad} mode="balanced" />);
    expect(screen.queryByText(/Max \(/i)).toBeNull();
  });

  it('renders default (0.50) mode operating threshold and label (A2)', () => {
    render(<RiskOverview cad={baseCad} mode="balanced" />);

    expect(screen.getByText(/Default \(0\.50\) Mode/i)).toBeInTheDocument();
    expect(screen.getByText(/Operating Threshold: 0.50/i)).toBeInTheDocument();
    expect(screen.getByText('High Risk')).toBeInTheDocument();
  });

  it('renders screening mode high-sensitivity cutoff and label (A2)', () => {
    const screeningCad: CadPrediction = {
      ...baseCad,
      prob: 0.45,
      raw_prob: 0.45,
      coherent_prob: 0.45,
      label: 'Low Risk',
      high_sens_label: 'High Risk',
      threshold: 0.50,
      high_sensitivity_threshold: 0.40,
    };

    const { rerender } = render(<RiskOverview cad={screeningCad} mode="balanced" />);
    expect(screen.getByText('Low Risk')).toBeInTheDocument();

    rerender(<RiskOverview cad={screeningCad} mode="screening" />);
    expect(screen.getByText(/Screening Mode/i)).toBeInTheDocument();
    expect(screen.getByText(/Screening Cutoff: 0.40/i)).toBeInTheDocument();
    expect(screen.getByText('High Risk')).toBeInTheDocument();
  });

  it('calls onToggleMode handler when mode toggle button is clicked (A2)', () => {
    const toggleMock = vi.fn();
    render(<RiskOverview cad={baseCad} mode="balanced" onToggleMode={toggleMock} />);

    const button = screen.getByRole('button', { name: /mode: default/i });
    fireEvent.click(button);
    expect(toggleMock).toHaveBeenCalledTimes(1);
  });

  it('displays nested CV sensitivity and similar cohort statistics (A1)', () => {
    render(<RiskOverview cad={baseCad} mode="balanced" />);

    expect(screen.getByText(/Nested-CV Sens: 90.3%/i)).toBeInTheDocument();
    expect(screen.getByText(/42 similar patients/i)).toBeInTheDocument();
  });
});
