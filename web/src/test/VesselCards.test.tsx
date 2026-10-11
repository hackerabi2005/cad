import React from 'react';
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { VesselCards } from '../components/VesselCards';
import { VesselPrediction } from '../types';

describe('VesselCards Component', () => {
  const sampleVessels: Record<'LAD' | 'LCX' | 'RCA', VesselPrediction> = {
    LAD: {
      prob: 0.85,
      label: 'Stenotic',
      high_sens_label: 'Stenotic',
      threshold: 0.50,
      high_sensitivity_threshold: 0.469,
      nested_sensitivity: 0.882,
    },
    LCX: {
      prob: 0.32,
      label: 'Normal',
      high_sens_label: 'Stenotic',
      threshold: 0.50,
      high_sensitivity_threshold: 0.217,
      nested_sensitivity: 0.916,
    },
    RCA: {
      prob: 0.12,
      label: 'Normal',
      high_sens_label: 'Normal',
      threshold: 0.50,
      high_sensitivity_threshold: 0.213,
      nested_sensitivity: 0.892,
    },
  };

  it('renders all three vessel cards (LAD, LCX, RCA) (C8)', () => {
    render(
      <VesselCards
        vessels={sampleVessels}
        selectedVessel={null}
        onSelectVessel={() => {}}
      />
    );

    expect(screen.getByText('LAD')).toBeInTheDocument();
    expect(screen.getByText('LCX')).toBeInTheDocument();
    expect(screen.getByText('RCA')).toBeInTheDocument();
  });

  it('renders missing values with em-dash "—" without inventing numbers (A9)', () => {
    render(
      <VesselCards
        vessels={undefined}
        selectedVessel={null}
        onSelectVessel={() => {}}
      />
    );

    const dashes = screen.getAllByText('—');
    expect(dashes.length).toBeGreaterThanOrEqual(3);
  });

  it('handles partial missing vessel data gracefully (A9)', () => {
    const partialVessels = {
      LAD: sampleVessels.LAD,
      LCX: undefined as any,
      RCA: sampleVessels.RCA,
    };

    render(
      <VesselCards
        vessels={partialVessels}
        selectedVessel={null}
        onSelectVessel={() => {}}
      />
    );

    expect(screen.getByText('85%')).toBeInTheDocument();
    expect(screen.getByText('12%')).toBeInTheDocument();
    expect(screen.getAllByText('—').length).toBeGreaterThan(0);
  });

  it('provides accessible role and keyboard navigation support (D3)', () => {
    const selectMock = vi.fn();
    render(
      <VesselCards
        vessels={sampleVessels}
        selectedVessel={null}
        onSelectVessel={selectMock}
      />
    );

    const ladCard = screen.getByTestId('vessel-card-LAD');
    expect(ladCard).toHaveAttribute('role', 'button');
    expect(ladCard).toHaveAttribute('tabIndex', '0');

    // ARIA label includes vessel name, risk percentage, and status
    expect(ladCard.getAttribute('aria-label')).toMatch(/Left Anterior Descending.*85%.*Stenotic/i);

    // Enter key triggers selection
    fireEvent.keyDown(ladCard, { key: 'Enter', code: 'Enter' });
    expect(selectMock).toHaveBeenCalledWith('LAD');

    // Space key triggers selection
    fireEvent.keyDown(ladCard, { key: ' ', code: 'Space' });
    expect(selectMock).toHaveBeenCalledWith('LAD');
  });

  it('provides descriptive aria-label when data is missing (D3)', () => {
    render(
      <VesselCards
        vessels={undefined}
        selectedVessel={null}
        onSelectVessel={() => {}}
      />
    );

    const ladCard = screen.getByTestId('vessel-card-LAD');
    expect(ladCard.getAttribute('aria-label')).toContain('stenosis risk unavailable');
  });

  it('updates labels between balanced and screening mode (A2)', () => {
    const { rerender } = render(
      <VesselCards
        vessels={sampleVessels}
        selectedVessel={null}
        onSelectVessel={() => {}}
        mode="balanced"
      />
    );

    // In balanced mode, LCX prob 0.32 is Normal
    const lcxCard = screen.getByTestId('vessel-card-LCX');
    expect(lcxCard).toHaveTextContent(/Normal/i);

    // In screening mode, LCX prob 0.32 > 0.217 cutoff -> Stenotic
    rerender(
      <VesselCards
        vessels={sampleVessels}
        selectedVessel={null}
        onSelectVessel={() => {}}
        mode="screening"
      />
    );

    expect(lcxCard).toHaveTextContent(/Stenotic/i);
  });

  it('triggers onSelectVessel when clicked and reflects selection state', () => {
    const selectMock = vi.fn();
    const { rerender } = render(
      <VesselCards
        vessels={sampleVessels}
        selectedVessel={null}
        onSelectVessel={selectMock}
      />
    );

    const ladCard = screen.getByTestId('vessel-card-LAD');
    fireEvent.click(ladCard);
    expect(selectMock).toHaveBeenCalledWith('LAD');

    // Selected state shows "Inspecting"
    rerender(
      <VesselCards
        vessels={sampleVessels}
        selectedVessel="LAD"
        onSelectVessel={selectMock}
      />
    );

    expect(screen.getByText('Inspecting')).toBeInTheDocument();
  });
});
