import React from 'react';
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { RiskOverview } from '../components/RiskOverview';
import { CadPrediction } from '../types';

describe('RiskOverview Component', () => {
  const mockCadBalanced: CadPrediction = {
    prob: 0.65,
    raw_prob: 0.52,
    displayed_prob: 0.65,
    coherent_prob: 0.65,
    driving_vessel: 'LAD',
    label: 'High Risk',
    high_sens_label: 'High Risk',
    threshold: 0.50,
    high_sensitivity_threshold: 0.61,
    nested_sensitivity: 0.903,
    coherence_adjusted: true,
    similar_cohort: {
      n: 45,
      observed_rate: 0.68,
      wilson_interval: [0.54, 0.80],
      summary: 'Among 45 cohort patients with similar risk, 68% had confirmed stenosis.',
    },
  };

  it('renders awaiting calculation placeholder when cad is undefined', () => {
    render(<RiskOverview cad={undefined} />);
    expect(screen.getByText(/Awaiting risk calculation/i)).toBeInTheDocument();
  });

  it('displays formatted percentage and active label in balanced mode', () => {
    render(<RiskOverview cad={mockCadBalanced} mode="balanced" />);
    expect(screen.getByText('65%')).toBeInTheDocument();
    expect(screen.getByText('High Risk')).toBeInTheDocument();
    expect(screen.getByText(/Default \(0\.50\) Mode/i)).toBeInTheDocument();
    expect(screen.getByText(/Operating Threshold: 0.50/i)).toBeInTheDocument();
  });

  it('displays separate raw model score and coherent risk with driving vessel badge', () => {
    render(<RiskOverview cad={mockCadBalanced} mode="balanced" />);
    expect(screen.getByText(/Raw Model/i)).toBeInTheDocument();
    expect(screen.getByText('52.0%')).toBeInTheDocument();
    expect(screen.getByText(/Coherent Risk/i)).toBeInTheDocument();
    expect(screen.getByText(/Max \(LAD\)/i)).toBeInTheDocument();
  });

  it('switches to screening mode cutoffs and labels correctly', () => {
    const mockCadFlip: CadPrediction = {
      ...mockCadBalanced,
      prob: 0.55,
      coherent_prob: 0.55,
      raw_prob: 0.55,
      coherence_adjusted: false,
      driving_vessel: null,
      label: 'High Risk',
      high_sens_label: 'Normal', // flips below 0.61 cutoff
    };

    const { rerender } = render(<RiskOverview cad={mockCadFlip} mode="balanced" />);
    expect(screen.getByText('High Risk')).toBeInTheDocument();

    rerender(<RiskOverview cad={mockCadFlip} mode="screening" />);
    expect(screen.getByText('Normal')).toBeInTheDocument();
    expect(screen.getByText(/Screening Cutoff: 0.61/i)).toBeInTheDocument();
  });

  it('calls onToggleMode when mode button is clicked', () => {
    const handleToggle = vi.fn();
    render(<RiskOverview cad={mockCadBalanced} mode="balanced" onToggleMode={handleToggle} />);
    const button = screen.getByTitle(/Click to toggle between/i);
    fireEvent.click(button);
    expect(handleToggle).toHaveBeenCalledTimes(1);
  });

  it('displays similar cohort empirical reference and Wilson interval', () => {
    render(<RiskOverview cad={mockCadBalanced} mode="balanced" />);
    expect(screen.getByText(/Observed in 45 similar patients: 68% \(95% CI 54%–80%\)/i)).toBeInTheDocument();
  });

  it('renders reverse contradiction banner when CAD is flagged but no vessel is flagged', () => {
    const reverseCad: CadPrediction = {
      ...mockCadBalanced,
      coherence_adjusted: false,
      flagged_vessels_count: 0,
      label: 'High Risk',
    };
    render(<RiskOverview cad={reverseCad} mode="balanced" />);
    expect(
      screen.getByText(/Overall model above cutoff; no vessel model above its cutoff; location not determined/i)
    ).toBeInTheDocument();
  });

  it('displays exact 95% CIs on mode toggle button', () => {
    render(<RiskOverview cad={mockCadBalanced} mode="balanced" onToggleMode={() => {}} />);
    expect(screen.getByText(/Sens: 94\.5% · Spec: 70\.9%/i)).toBeInTheDocument();
  });
});
