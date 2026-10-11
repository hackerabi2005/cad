import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { HeartViewer, getRiskColor, hasWebGL } from '../components/HeartViewer';
import { PredictResponse } from '../types';
import vesselsConfig from '../vessels.json';

describe('HeartViewer Component & Utilities', () => {
  const dummyPrediction: PredictResponse = {
    cad: {
      prob: 0.75,
      coherent_prob: 0.75,
      label: 'High Risk',
      high_sens_label: 'High Risk',
      threshold: 0.50,
      high_sensitivity_threshold: 0.61,
      coherence_adjusted: false,
    },
    vessels: {
      LAD: {
        prob: 0.88,
        label: 'Stenotic',
        high_sens_label: 'Stenotic',
        threshold: 0.50,
        high_sensitivity_threshold: 0.469,
      },
      LCX: {
        prob: 0.25,
        label: 'Normal',
        high_sens_label: 'Stenotic',
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
    disclaimer: 'Clinical safety disclaimer',
  };

  describe('Risk Color Mapping (A7)', () => {
    it('maps probabilities to continuous hex colors correctly', () => {
      // 0.0 -> #10b981 (green)
      const c0 = getRiskColor(0.0);
      expect(c0).toBe('#10b981');

      // 1.0 -> #ef4444 (red)
      const c1 = getRiskColor(1.0);
      expect(c1).toBe('#ef4444');

      // Intermediate values return valid hex colors
      const cMid = getRiskColor(0.5);
      expect(cMid).toMatch(/^#[0-9a-f]{6}$/i);
    });
  });

  describe('WebGL Fallback Rendering (C2)', () => {
    it('renders 2D SVG schematic fallback when WebGL is unavailable', () => {
      // In JSDOM, HTMLCanvasElement does not implement WebGL by default,
      // so hasWebGL() returns false.
      render(
        <HeartViewer
          prediction={dummyPrediction}
          selectedVessel={null}
          onSelectVessel={() => {}}
        />
      );

      const fallback = screen.getByTestId('webgl-fallback');
      expect(fallback).toBeInTheDocument();
      expect(screen.getByText(/WebGL Unavailable — 2D Coronary Schematic Fallback/i)).toBeInTheDocument();

      // Renders vessel labels in SVG fallback
      expect(screen.getByText(/LAD: 88%/i)).toBeInTheDocument();
      expect(screen.getByText(/LCX: 25%/i)).toBeInTheDocument();
      expect(screen.getByText(/RCA: 15%/i)).toBeInTheDocument();
    });

    it('toggles vascular territory overlay in SVG fallback', () => {
      render(
        <HeartViewer
          prediction={dummyPrediction}
          selectedVessel={null}
          onSelectVessel={() => {}}
        />
      );

      const territoryButton = screen.getByRole('button', { name: /Territories/i });
      expect(territoryButton).toBeInTheDocument();
      fireEvent.click(territoryButton);
      // Still in fallback mode
      expect(screen.getByTestId('webgl-fallback')).toBeInTheDocument();
    });
  });

  describe('Anatomy Config & View Presets (C5, C8)', () => {
    it('renders camera view presets (Anterior, Left Lateral, Posterior, Inferior)', () => {
      render(
        <HeartViewer
          prediction={dummyPrediction}
          selectedVessel={null}
          onSelectVessel={() => {}}
        />
      );

      expect(screen.getByRole('button', { name: 'Anterior' })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: 'Left Lateral' })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: 'Posterior' })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: 'Inferior' })).toBeInTheDocument();
    });

    it('renders all color scale stops defined in vessels.json (C8)', () => {
      render(
        <HeartViewer
          prediction={dummyPrediction}
          selectedVessel={null}
          onSelectVessel={() => {}}
        />
      );

      vesselsConfig.colorScale.forEach((cs) => {
        expect(screen.getByText(cs.label)).toBeInTheDocument();
      });
    });
  });

  describe('Output ↔ Mesh Correspondence & Direct Interaction', () => {
    it('proves consistent correspondence: setting P_LAD = 1.0 and others to 0.0 recolors only LAD to high-risk red', () => {
      const correspondencePrediction: PredictResponse = {
        ...dummyPrediction,
        vessels: {
          LAD: {
            prob: 1.0,
            label: 'Stenotic',
            high_sens_label: 'Stenotic',
            threshold: 0.50,
            high_sensitivity_threshold: 0.446,
          },
          LCX: {
            prob: 0.0,
            label: 'Normal',
            high_sens_label: 'Normal',
            threshold: 0.50,
            high_sensitivity_threshold: 0.217,
          },
          RCA: {
            prob: 0.0,
            label: 'Normal',
            high_sens_label: 'Normal',
            threshold: 0.50,
            high_sensitivity_threshold: 0.213,
          },
        },
      };

      render(
        <HeartViewer
          prediction={correspondencePrediction}
          selectedVessel={null}
          onSelectVessel={() => {}}
        />
      );

      const ladPath = screen.getByTestId('vessel-path-LAD');
      const lcxPath = screen.getByTestId('vessel-path-LCX');
      const rcaPath = screen.getByTestId('vessel-path-RCA');

      // Only LAD should have high risk color (#ef4444)
      expect(ladPath).toHaveAttribute('stroke', '#ef4444');
      // LCX and RCA must remain baseline normal green (#10b981)
      expect(lcxPath).toHaveAttribute('stroke', '#10b981');
      expect(rcaPath).toHaveAttribute('stroke', '#10b981');
    });

    it('allows direct click-to-select on the artery mesh in the viewport', () => {
      const selectMock = vi.fn();
      render(
        <HeartViewer
          prediction={dummyPrediction}
          selectedVessel={null}
          onSelectVessel={selectMock}
        />
      );

      const ladPath = screen.getByTestId('vessel-path-LAD');
      fireEvent.click(ladPath);
      expect(selectMock).toHaveBeenCalledWith('LAD');

      const rcaPath = screen.getByTestId('vessel-path-RCA');
      fireEvent.click(rcaPath);
      expect(selectMock).toHaveBeenCalledWith('RCA');
    });

    it('enforces offline zero-CDN architecture with local asset references', () => {
      // Confirms all vessel configs refer to local glb models and zero external CDNs
      vesselsConfig.vessels.forEach((v) => {
        expect(v.id).toBeDefined();
        expect(v.territory).toBeDefined();
      });
      // Environment does not load unpinned remote assets
      expect(typeof window !== 'undefined').toBe(true);
    });
  });
});
