import React from 'react';
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { AboutModelModal } from '../components/AboutModelModal';

describe('AboutModelModal Component (G8)', () => {
  it('renders clinical safety disclaimer and intended use notice', () => {
    render(<AboutModelModal onClose={() => {}} />);

    expect(screen.getByText(/About Cardio3D AI & Clinical Safety/i)).toBeInTheDocument();
    expect(screen.getByText(/Clinical Intended Use & Safety Disclaimer/i)).toBeInTheDocument();
    expect(screen.getByText(/not a substitute for formal diagnostic imaging/i)).toBeInTheDocument();
  });

  it('renders model architecture and exact SHAP additivity specifications', () => {
    render(<AboutModelModal onClose={() => {}} />);

    expect(screen.getByText(/LogitEnsemble/i)).toBeInTheDocument();
    expect(screen.getByText(/Exact analytical log-odds decomposition/i)).toBeInTheDocument();
    expect(screen.getByText(/P\(CAD\)_coherent = max\(P\(CAD\), P\(LAD\), P\(LCX\), P\(RCA\)\)/i)).toBeInTheDocument();
  });

  it('calls onClose when close button is clicked', () => {
    const closeMock = vi.fn();
    render(<AboutModelModal onClose={closeMock} />);

    const closeBtn = screen.getByRole('button', { name: /Close dialog/i });
    fireEvent.click(closeBtn);
    expect(closeMock).toHaveBeenCalledTimes(1);
  });

  it('calls onClose when Escape key is pressed', () => {
    const closeMock = vi.fn();
    render(<AboutModelModal onClose={closeMock} />);

    fireEvent.keyDown(window, { key: 'Escape', code: 'Escape' });
    expect(closeMock).toHaveBeenCalledTimes(1);
  });
});
