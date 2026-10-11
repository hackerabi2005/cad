import React from 'react';
import { describe, it, expect } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { ShapWaterfall } from '../components/ShapWaterfall';
import { TargetExplanation } from '../types';

describe('ShapWaterfall Component', () => {
  const mockExplanation: TargetExplanation = {
    base_value: 0.25,
    raw_score: 1.15,
    reconstructed_score: 1.15,
    additive_error: 0.00001,
    shap_unit: 'log-odds',
    unit_label: 'Log-odds margin',
    features: [
      {
        feature: 'Typical Chest Pain',
        label: 'Typical Anginal Chest Pain',
        group: 'symptoms-exam',
        unit: '',
        value: 1,
        shap: 0.45,
        pct: 40.0,
        is_imputed: false,
      },
      {
        feature: 'Age',
        label: 'Patient Age',
        group: 'demographic',
        unit: 'yrs',
        value: 62,
        shap: 0.25,
        pct: 22.2,
        is_imputed: false,
      },
      {
        feature: 'HTN',
        label: 'Hypertension History',
        group: 'symptoms-exam',
        unit: '',
        value: 1,
        shap: 0.15,
        pct: 13.3,
        is_imputed: true,
      },
      {
        feature: 'EF-TTE',
        label: 'Ejection Fraction',
        group: 'echo',
        unit: '%',
        value: 55,
        shap: -0.20,
        pct: 17.8,
        is_imputed: false,
      },
    ],
  };

  const mockExplanations = {
    Cath: mockExplanation,
  };

  it('renders base value, model margin, and unit label', () => {
    render(<ShapWaterfall explanations={mockExplanations} activeTarget="Cath" onSelectTarget={() => {}} />);
    // Switch to log-odds view mode
    const logOddsBtn = screen.getByRole('button', { name: /Log-Odds Margin/i });
    fireEvent.click(logOddsBtn);
    expect(screen.getByText(/Base Margin:/i)).toBeInTheDocument();
    expect(screen.getByText('0.25')).toBeInTheDocument();
    expect(screen.getByText(/Score:/i)).toBeInTheDocument();
    expect(screen.getByText('1.15')).toBeInTheDocument();
  });

  it('displays the non-causal associational disclaimer banner', () => {
    render(<ShapWaterfall explanations={mockExplanations} activeTarget="Cath" onSelectTarget={() => {}} />);
    expect(screen.getByText(/associational attribution, not causal/i)).toBeInTheDocument();
  });

  it('renders assumed chip for omitted/imputed features', () => {
    render(<ShapWaterfall explanations={mockExplanations} activeTarget="Cath" onSelectTarget={() => {}} />);
    expect(screen.getAllByText('cohort median').length).toBeGreaterThan(0);
  });

  it('renders risk elevating and protective feature contributions', () => {
    render(<ShapWaterfall explanations={mockExplanations} activeTarget="Cath" onSelectTarget={() => {}} />);
    expect(screen.getAllByText(/Typical Anginal Chest Pain/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/Ejection Fraction/i).length).toBeGreaterThan(0);
    // Switch to log-odds margin mode to check formatted SHAP values
    const logOddsBtn = screen.getByRole('button', { name: /Log-Odds Margin/i });
    fireEvent.click(logOddsBtn);
    expect(screen.getAllByText(/\+0\.450/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/-0\.200/i).length).toBeGreaterThan(0);
  });

  it('toggles between Waterfall, Physiology, Domains and All 54 views', () => {
    render(<ShapWaterfall explanations={mockExplanations} activeTarget="Cath" onSelectTarget={() => {}} />);
    const all54Btn = screen.getByRole('button', { name: /All 54/i });
    fireEvent.click(all54Btn);
    expect(all54Btn).toBeInTheDocument();

    const physioBtn = screen.getByRole('button', { name: /Physiology/i });
    fireEvent.click(physioBtn);
    expect(physioBtn).toBeInTheDocument();

    const domainsBtn = screen.getByRole('button', { name: /Domains/i });
    fireEvent.click(domainsBtn);
    expect(domainsBtn).toBeInTheDocument();
  });

  it('renders coherence adjustment callout and row when Cath risk is raised to match driving vessel', () => {
    render(
      <ShapWaterfall
        explanations={mockExplanations}
        activeTarget="Cath"
        onSelectTarget={() => {}}
        cadCoherenceInfo={{
          adjusted: true,
          driving_vessel: 'LAD',
          raw_prob: 0.45,
          displayed_prob: 0.78,
        }}
      />
    );
    expect(screen.getAllByText(/Raised to match LAD by max\(\) coherence rule/i).length).toBeGreaterThan(0);
  });
});
