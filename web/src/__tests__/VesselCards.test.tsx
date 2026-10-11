import React from 'react';
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { VesselCards } from '../components/VesselCards';
import { VesselPrediction } from '../types';

describe('VesselCards Component', () => {
  const mockVessels: Record<'LAD' | 'LCX' | 'RCA', VesselPrediction> = {
    LAD: {
      prob: 0.72,
      label: 'Stenotic',
      high_sens_label: 'Stenotic',
      threshold: 0.50,
      high_sensitivity_threshold: 0.469,
      nested_sensitivity: 0.882,
    },
    LCX: {
      prob: 0.35,
      label: 'Normal',
      high_sens_label: 'Stenotic', // flips at 0.217 screening cutoff
      threshold: 0.50,
      high_sensitivity_threshold: 0.217,
      nested_sensitivity: 0.916,
    },
    RCA: {
      prob: 0.15,
      label: 'Normal',
      high_sens_label: 'Normal',
      threshold: 0.50,
      high_sensitivity_threshold: 0.213,
      nested_sensitivity: 0.892,
    },
  };

  it('renders all 3 major vessels correctly', () => {
    render(
      <VesselCards
        vessels={mockVessels}
        selectedVessel={null}
        onSelectVessel={() => {}}
      />
    );
    expect(screen.getByText('LAD')).toBeInTheDocument();
    expect(screen.getByText('LCX')).toBeInTheDocument();
    expect(screen.getByText('RCA')).toBeInTheDocument();
  });

  it('switches vessel labels between Balanced and Screening mode', () => {
    const { rerender } = render(
      <VesselCards
        vessels={mockVessels}
        selectedVessel={null}
        onSelectVessel={() => {}}
        mode="balanced"
      />
    );
    // In balanced mode, LCX prob 0.35 is Normal
    const lcxCardBalanced = screen.getByTestId('vessel-card-LCX');
    expect(lcxCardBalanced).toHaveTextContent('Normal');

    rerender(
      <VesselCards
        vessels={mockVessels}
        selectedVessel={null}
        onSelectVessel={() => {}}
        mode="screening"
      />
    );
    // In screening mode, LCX prob 0.35 exceeds 0.217 cutoff -> Stenotic
    const lcxCardScreening = screen.getByTestId('vessel-card-LCX');
    expect(lcxCardScreening).toHaveTextContent('Stenotic');
  });

  it('handles missing vessel data gracefully with dashes', () => {
    render(
      <VesselCards
        vessels={undefined}
        selectedVessel={null}
        onSelectVessel={() => {}}
      />
    );
    const ladCard = screen.getByTestId('vessel-card-LAD');
    expect(ladCard).toHaveTextContent('—');
  });

  it('fires onSelectVessel callback when card is clicked', () => {
    const handleSelect = vi.fn();
    render(
      <VesselCards
        vessels={mockVessels}
        selectedVessel={null}
        onSelectVessel={handleSelect}
      />
    );
    fireEvent.click(screen.getByTestId('vessel-card-LAD'));
    expect(handleSelect).toHaveBeenCalledWith('LAD');
  });

  it('supports keyboard accessibility via Enter and Space keys', () => {
    const handleSelect = vi.fn();
    render(
      <VesselCards
        vessels={mockVessels}
        selectedVessel={null}
        onSelectVessel={handleSelect}
      />
    );
    const rcaCard = screen.getByTestId('vessel-card-RCA');
    fireEvent.keyDown(rcaCard, { key: 'Enter' });
    expect(handleSelect).toHaveBeenCalledWith('RCA');

    fireEvent.keyDown(rcaCard, { key: ' ' });
    expect(handleSelect).toHaveBeenCalledWith('RCA');
  });
});
