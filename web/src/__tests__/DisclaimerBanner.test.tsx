import React from 'react';
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { DisclaimerHeaderBanner, DisclaimerFooter } from '../components/DisclaimerBanner';

describe('Disclaimer Components', () => {
  it('renders sticky header disclaimer with required text', () => {
    render(<DisclaimerHeaderBanner />);
    const banner = screen.getByText(/Decision support \/ educational (use|purposes) only/i);
    expect(banner).toBeInTheDocument();
    expect(document.getElementById('safety-disclaimer-banner')).toHaveClass('sticky');
  });

  it('renders footer with safety notice and dataset / mesh attribution', () => {
    render(<DisclaimerFooter />);
    expect(screen.getByText(/Clinical Safety Notice/i)).toBeInTheDocument();
    expect(screen.getByText(/UCI Extension of Z-Alizadeh Sani/i)).toBeInTheDocument();
    expect(screen.getByText(/BodyParts3D © DBCLS/i)).toBeInTheDocument();
  });
});
