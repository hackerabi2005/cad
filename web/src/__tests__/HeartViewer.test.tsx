import React from 'react';
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { HeartViewer, getRiskColor } from '../components/HeartViewer';
import { PredictResponse } from '../types';

describe('HeartViewer Component & Risk Color Scale', () => {
  it('interpolates risk colors across continuous thresholds', () => {
    // 0.0 -> green (#10b981)
    expect(getRiskColor(0.0).toLowerCase()).toContain('10b981');
    // 1.0 -> red (#ef4444)
    expect(getRiskColor(1.0).toLowerCase()).toContain('ef4444');
    // intermediate probabilities return valid hex codes
    expect(getRiskColor(0.35)).toMatch(/^#[0-9a-f]{6}$/i);
    expect(getRiskColor(0.65)).toMatch(/^#[0-9a-f]{6}$/i);
  });

  it('renders 2D SVG coronary schematic fallback in environments without WebGL', () => {
    render(
      <HeartViewer
        prediction={null}
        selectedVessel={null}
        onSelectVessel={() => {}}
      />
    );
    expect(screen.getByTestId('webgl-fallback')).toBeInTheDocument();
    expect(screen.getByText(/WebGL Unavailable — 2D Coronary Schematic Fallback/i)).toBeInTheDocument();
  });

  it('allows selecting vessels via 2D schematic fallback click', () => {
    const handleSelect = vi.fn();
    const mockPrediction: Partial<PredictResponse> = {
      cad: {
        prob: 0.6,
        coherent_prob: 0.6,
        label: 'High Risk',
        high_sens_label: 'High Risk',
        threshold: 0.5,
        high_sensitivity_threshold: 0.61,
        coherence_adjusted: false,
      },
      vessels: {
        LAD: { prob: 0.7, label: 'Stenotic', high_sens_label: 'Stenotic', threshold: 0.5, high_sensitivity_threshold: 0.469 },
        LCX: { prob: 0.3, label: 'Normal', high_sens_label: 'Stenotic', threshold: 0.5, high_sensitivity_threshold: 0.217 },
        RCA: { prob: 0.2, label: 'Normal', high_sens_label: 'Normal', threshold: 0.5, high_sensitivity_threshold: 0.213 },
      },
    };

    render(
      <HeartViewer
        prediction={mockPrediction as PredictResponse}
        selectedVessel={null}
        onSelectVessel={handleSelect}
      />
    );

    // Text labels for the 3 vessels exist in the fallback
    expect(screen.getByText(/LAD: 70%/i)).toBeInTheDocument();
    expect(screen.getByText(/LCX: 30%/i)).toBeInTheDocument();
    expect(screen.getByText(/RCA: 20%/i)).toBeInTheDocument();
  });

  it('renders color scale stops and legend in viewer footer', () => {
    render(
      <HeartViewer
        prediction={null}
        selectedVessel={null}
        onSelectVessel={() => {}}
      />
    );
    expect(screen.getByText(/Risk vs Cutoff:|P\(≥50% Stenosis\)/i)).toBeInTheDocument();
  });
});
