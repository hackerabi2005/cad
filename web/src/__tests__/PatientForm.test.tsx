import React from 'react';
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { PatientForm } from '../components/PatientForm';
import { SchemaRegistry, SamplePatient } from '../types';

describe('PatientForm Component', () => {
  const mockSchema: SchemaRegistry = {
    Age: {
      group: 'demographic',
      encoding: 'numeric',
      unit: 'yrs',
      label: 'Patient Age',
      min: 30,
      max: 86,
      median: 58,
    },
    Sex: {
      group: 'demographic',
      encoding: 'binary',
      unit: '',
      label: 'Biological Sex',
      categories: ['Female', 'Male'],
      mode: 'Male',
    },
    BP: {
      group: 'symptoms-exam',
      encoding: 'numeric',
      unit: 'mmHg',
      label: 'Systolic BP',
      min: 90,
      max: 190,
      median: 130,
    },
  };

  const mockSamples: SamplePatient[] = [
    {
      id: 'sample-low-risk',
      name: 'Patient A — Low Risk Profile',
      description: 'Predicted CAD prob: 2.6%',
      data: { Age: 42, Sex: 'Female', BP: 110 },
    },
    {
      id: 'sample-high-risk',
      name: 'Patient C — High Risk Stenosis',
      description: 'Predicted CAD prob: 99.9%',
      data: { Age: 68, Sex: 'Male', BP: 160 },
    },
  ];

  it('renders preset patient buttons and handles preset loading', () => {
    const handleLoadSample = vi.fn();
    render(
      <PatientForm
        schema={mockSchema}
        patientData={{ Age: 58, Sex: 'Male', BP: 130 }}
        samples={mockSamples}
        onChangeField={() => {}}
        onLoadSample={handleLoadSample}
        onReset={() => {}}
        cohortDefaults={[]}
      />
    );

    const highRiskBtn = screen.getByRole('button', { name: /Patient C — High Risk Stenosis/i });
    expect(highRiskBtn).toBeInTheDocument();
    fireEvent.click(highRiskBtn);
    expect(handleLoadSample).toHaveBeenCalledWith(mockSamples[1]);
  });

  it('renders form category tabs and switches active tab', () => {
    render(
      <PatientForm
        schema={mockSchema}
        patientData={{ Age: 58, Sex: 'Male', BP: 130 }}
        samples={mockSamples}
        onChangeField={() => {}}
        onLoadSample={() => {}}
        onReset={() => {}}
        cohortDefaults={[]}
      />
    );

    expect(screen.getByText('Patient Age')).toBeInTheDocument();
    const vitalsTab = screen.getByRole('button', { name: /Vitals & Symptoms/i });
    fireEvent.click(vitalsTab);
    expect(screen.getByText('Systolic BP')).toBeInTheDocument();
  });

  it('invokes onChangeField when input value changes', () => {
    const handleChange = vi.fn();
    render(
      <PatientForm
        schema={mockSchema}
        patientData={{ Age: 58, Sex: 'Male', BP: 130 }}
        samples={mockSamples}
        onChangeField={handleChange}
        onLoadSample={() => {}}
        onReset={() => {}}
        cohortDefaults={[]}
      />
    );

    const ageInputs = screen.getAllByDisplayValue('58');
    fireEvent.change(ageInputs[0], { target: { value: '62' } });
    expect(handleChange).toHaveBeenCalledWith('Age', 62);
  });
});
