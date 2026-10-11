import React from 'react';
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { ShapWaterfall } from '../components/ShapWaterfall';
import { TargetExplanation, SchemaRegistry } from '../types';

describe('ShapWaterfall Component', () => {
  const dummyExplanation: Record<string, TargetExplanation> = {
    Cath: {
      base_value: 0.15,
      raw_score: 1.45,
      reconstructed_score: 1.45,
      additive_error: 1.2e-15,
      shap_unit: 'log-odds',
      features: [
        {
          feature: 'Typical_Chest_Pain',
          label: 'Typical Chest Pain',
          group: 'symptoms-exam',
          unit: 'binary',
          value: 'Y',
          shap: 0.72,
          pct: 45.2,
        },
        {
          feature: 'Age',
          label: 'Age',
          group: 'demographic',
          unit: 'years',
          value: 62,
          shap: 0.42,
          pct: 26.4,
        },
        {
          feature: 'EF-TTE',
          label: 'Ejection Fraction',
          group: 'echo',
          unit: '%',
          value: 45,
          shap: 0.16,
          pct: 10.1,
        },
      ],
    },
  };

  const dummySchema: SchemaRegistry = {
    'EF-TTE': {
      group: 'echo',
      encoding: 'numeric',
      unit: '%',
      label: 'Ejection Fraction',
      min: 15,
      max: 70,
      ref_low: 50,
      ref_high: 70,
      source: 'ASE/EACVI 2015 chamber quantification guidelines',
      reviewed: false,
    },
    'Age': {
      group: 'demographic',
      encoding: 'numeric',
      unit: 'years',
      label: 'Age',
      min: 30,
      max: 90,
    },
  };

  it('renders target title and feature attributions', () => {
    render(
      <ShapWaterfall
        explanations={dummyExplanation}
        activeTarget="Cath"
        onSelectTarget={() => {}}
        schema={dummySchema}
      />
    );

    expect(screen.getAllByText(/Overall CAD Diagnosis Model/i)[0]).toBeInTheDocument();
    expect(screen.getAllByText(/Typical Chest Pain/i)[0]).toBeInTheDocument();
    expect(screen.getAllByText('Age')[0]).toBeInTheDocument();
  });

  it('toggles between probability path and log-odds display', () => {
    render(
      <ShapWaterfall
        explanations={dummyExplanation}
        activeTarget="Cath"
        onSelectTarget={() => {}}
        schema={dummySchema}
      />
    );

    const logOddsBtn = screen.getByRole('button', { name: /Log-Odds/i });
    expect(logOddsBtn).toBeInTheDocument();
    fireEvent.click(logOddsBtn);

    const probPathBtn = screen.getByRole('button', { name: /Probability Path/i });
    expect(probPathBtn).toBeInTheDocument();
    fireEvent.click(probPathBtn);
  });

  it('renders clinical physiology panel with reference ranges and pending review badge', () => {
    render(
      <ShapWaterfall
        explanations={dummyExplanation}
        activeTarget="Cath"
        onSelectTarget={() => {}}
        schema={dummySchema}
      />
    );

    // Switch to Physiology view mode
    const physioBtn = screen.getByRole('button', { name: /Physiology/i });
    expect(physioBtn).toBeInTheDocument();
    fireEvent.click(physioBtn);

    // Should render ASE/EACVI reference guideline
    expect(screen.getByText(/ASE\/EACVI 2015 chamber quantification guidelines/i)).toBeInTheDocument();

    // Should render "Pending clinical review" badge
    expect(screen.getByText(/Pending clinical review/i)).toBeInTheDocument();

    // Should indicate abnormal status for EF-TTE (value 45 vs ref_low 50)
    expect(screen.getByText(/Below normal/i)).toBeInTheDocument();
  });
});
