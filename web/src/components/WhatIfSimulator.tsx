import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Sliders, Sparkles, ArrowRight, RefreshCw, Check, Activity, Info } from 'lucide-react';
import { PredictResponse } from '../types';
import vesselsConfig from '../vessels.json';
import { VesselsConfig } from '../types';
import { getRiskColor } from './HeartViewer';

interface WhatIfSimulatorProps {
  patientData: Record<string, any>;
  currentPrediction: PredictResponse | null;
  onApplyToPatient: (newData: Record<string, any>) => void;
}

export function WhatIfSimulator({
  patientData,
  currentPrediction,
  onApplyToPatient,
}: WhatIfSimulatorProps) {
  const [simData, setSimData] = useState<Record<string, any>>({ ...patientData });
  const [simPrediction, setSimPrediction] = useState<PredictResponse | null>(null);
  const [isSimulating, setIsSimulating] = useState<boolean>(false);
  const [appliedSuccess, setAppliedSuccess] = useState<boolean>(false);
  const debounceRef = useRef<any>(null);

  // Sync simData when patientData changes
  useEffect(() => {
    setSimData({ ...patientData });
  }, [patientData]);

  // Run simulation prediction against fast-path /api/whatif (skips SHAP decomposition)
  const runSimPrediction = useCallback(async (data: Record<string, any>) => {
    try {
      setIsSimulating(true);
      const res = await fetch('/api/whatif', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ baseline: patientData, modified: data }),
      });
      if (res.ok) {
        const result = await res.json();
        setSimPrediction(result.modified_prediction);
      }
    } catch (e) {
      console.error('Simulation error:', e);
    } finally {
      setIsSimulating(false);
    }
  }, [patientData]);

  // Debounced prediction trigger
  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => {
      runSimPrediction(simData);
    }, 250);
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, [simData, runSimPrediction]);

  const updateField = (field: string, val: any) => {
    setSimData((prev) => {
      const next = { ...prev, [field]: val };

      // Keep linked fields in sync: BMI with Weight & Height (Length in cm)
      if (field === 'BMI') {
        const len = Number(next.Length || 170);
        if (len > 0) {
          next.Weight = Math.round(Number(val) * Math.pow(len / 100, 2) * 10) / 10;
        }
        // One-way coupling: set when crossing up, never cleared when sliding down
        if (Number(val) > 25) {
          next.Obesity = 'Y';
        }
      } else if (field === 'Weight' || field === 'Length') {
        const w = Number(next.Weight);
        const l = Number(next.Length);
        if (w > 0 && l > 0) {
          next.BMI = Math.round((w / Math.pow(l / 100, 2)) * 10) / 10;
          if (next.BMI > 25) {
            next.Obesity = 'Y';
          }
        }
      }

      // Historical diagnostic flags (HTN, DLP) remain fixed: empirical audit showed rules match
      // dataset flags only 74.3% and 53.1% of the time, so flags are never auto-flipped from measurements.
      // Obesity (BMI > 25) maintains clinical definition concordance (99.7% agreement).

      // Keep linked fields in sync: Quit smoking (current=0, ex-smoker=1)
      if (field === 'Current Smoker') {
        if (Number(val) === 0 && Number(prev['Current Smoker']) === 1) {
          next['EX-Smoker'] = 1;
        } else if (Number(val) === 1) {
          next['EX-Smoker'] = 0;
        }
      }

      return next;
    });
    setAppliedSuccess(false);
  };

  // Example Values Presets (continuous modifiable factors; historical diagnosis flags HTN/DLP/DM kept fixed)
  const applyExampleA = () => {
    setSimData((prev) => ({
      ...prev,
      BP: 118,
      FBS: 92,
      LDL: 68,
      'Current Smoker': 0,
      'EX-Smoker': prev['Current Smoker'] === 1 ? 1 : prev['EX-Smoker'],
    }));
  };

  const applyExampleB = () => {
    setSimData((prev) => {
      const next: Record<string, any> = {
        ...prev,
        BMI: 23.5,
        BP: 124,
        'Current Smoker': 0,
        'EX-Smoker': prev['Current Smoker'] === 1 ? 1 : prev['EX-Smoker'],
        FBS: 95,
      };
      const len = Number(next.Length || 170);
      if (len > 0) next.Weight = Math.round(23.5 * Math.pow(len / 100, 2) * 10) / 10;
      return next;
    });
  };

  const resetToCurrent = () => {
    setSimData({ ...patientData });
  };

  const handleApply = () => {
    onApplyToPatient(simData);
    setAppliedSuccess(true);
    setTimeout(() => setAppliedSuccess(false), 2500);
  };

  const baselineProb = currentPrediction?.cad.prob ?? 0.5;
  const simulatedProb = simPrediction?.cad.prob ?? baselineProb;
  const riskDelta = simulatedProb - baselineProb;
  const riskDeltaPercent = (riskDelta * 100).toFixed(1);
  const relativeShift =
    baselineProb > 0 ? (((simulatedProb - baselineProb) / baselineProb) * 100).toFixed(0) : '0';

  const config = vesselsConfig as VesselsConfig;

  return (
    <div className="p-5 rounded-2xl bg-slate-900/80 border border-slate-800/80 shadow-xl space-y-5 animate-fade-in">
      {/* Title & Presets Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-slate-800">
        <div>
          <span className="text-xs font-semibold uppercase tracking-wider text-cyan-400">
            Interactive Input Sensitivity
          </span>
          <h3 className="text-lg font-bold text-white font-display flex items-center gap-2">
            Model Sensitivity Explorer
            <span className="text-xs px-2 py-0.5 rounded-full bg-cyan-500/10 text-cyan-300 border border-cyan-500/30">
              Live Interactive
            </span>
          </h3>
          <p className="text-xs text-slate-400 mt-1 max-w-2xl leading-relaxed">
            <strong className="text-amber-300">Model sensitivity, not predicted treatment effect:</strong> Shows observational changes in model risk score across single-variable modifications. Cross-sectional diagnostic models cannot estimate clinical treatment or interventional effects.
          </p>
        </div>

        {/* Example Values Presets */}
        <div className="flex flex-wrap items-center gap-1.5 self-start sm:self-center">
          <button
            onClick={applyExampleA}
            className="flex items-center gap-1 px-2.5 py-1.5 rounded-lg bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 text-xs font-semibold hover:bg-emerald-500/30 transition-colors"
            title="Example values: BP 118, LDL 68, FBS 92, non-smoker"
          >
            <Sparkles className="w-3.5 h-3.5" />
            <span>Example values A</span>
          </button>
          <button
            onClick={applyExampleB}
            className="flex items-center gap-1 px-2.5 py-1.5 rounded-lg bg-sky-500/20 text-sky-300 border border-sky-500/30 text-xs font-semibold hover:bg-sky-500/30 transition-colors"
            title="Example values: BMI 23.5, BP 124, FBS 95, non-smoker"
          >
            <Activity className="w-3.5 h-3.5" />
            <span>Example values B</span>
          </button>
          <button
            onClick={resetToCurrent}
            className="flex items-center gap-1 px-2.5 py-1.5 rounded-lg bg-slate-800/80 text-slate-300 border border-slate-700/60 text-xs font-medium hover:bg-slate-700 transition-colors"
            title="Reset to current inputs"
          >
            <RefreshCw className="w-3 h-3" />
            <span>Reset</span>
          </button>
        </div>
      </div>

      {/* Primary Comparative Model Score Bar */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 p-4 rounded-xl bg-slate-950/70 border border-slate-800">
        {/* Baseline Score Card */}
        <div className="p-3 rounded-lg bg-slate-900/80 border border-slate-800/80 text-center">
          <span className="text-[10px] uppercase font-bold text-slate-400 block mb-1">
            Current Model Score
          </span>
          <div
            className="text-2xl font-black font-mono"
            style={{ color: getRiskColor(baselineProb) }}
          >
            {(baselineProb * 100).toFixed(1)}%
          </div>
          <span className="text-[10px] text-slate-500 font-mono mt-0.5 block">
            Category: {currentPrediction?.cad.label ?? 'CAD'}
          </span>
        </div>

        {/* Arrow & Delta Summary */}
        <div className="flex flex-col items-center justify-center text-center p-2">
          <div className="flex items-center gap-2 mb-1">
            <span className="text-xs text-slate-400">Change in Model Score</span>
            <ArrowRight className="w-4 h-4 text-cyan-400" />
          </div>
          <div
            className={`text-2xl font-black font-mono flex items-center gap-1 ${
              riskDelta < 0 ? 'text-emerald-400' : riskDelta > 0 ? 'text-red-400' : 'text-slate-400'
            }`}
          >
            {riskDelta > 0 ? `+${riskDeltaPercent} pp` : `${riskDeltaPercent} pp`}
          </div>
          <span className="text-[10px] text-slate-400">
            {riskDelta !== 0
              ? `Relative shift: ${Number(relativeShift) > 0 ? '+' : ''}${relativeShift}%`
              : 'No modification'}
          </span>
        </div>

        {/* Adjusted Simulated Score Card */}
        <div className="p-3 rounded-lg bg-slate-900/80 border border-cyan-500/40 text-center">
          <span className="text-[10px] uppercase font-bold text-cyan-400 block mb-1">
            Simulated Model Score
          </span>
          <div
            className="text-2xl font-black font-mono"
            style={{ color: getRiskColor(simulatedProb) }}
          >
            {(simulatedProb * 100).toFixed(1)}%
          </div>
          <span className="text-[10px] text-slate-400 font-mono mt-0.5 block">
            Category: {simPrediction?.cad.label ?? 'Simulating...'}
          </span>
        </div>
      </div>

      {/* Target Vessels Impact Comparison */}
      <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800 space-y-2">
        <span className="text-[10px] uppercase font-bold text-slate-400 tracking-wider block">
          Coronary Artery Branch Score Deltas
        </span>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          {config.vessels.map((v) => {
            const vKey = v.id as 'LAD' | 'LCX' | 'RCA';
            const baseVProb = currentPrediction?.vessels[vKey]?.prob ?? 0.5;
            const simVProb = simPrediction?.vessels[vKey]?.prob ?? baseVProb;
            const deltaV = simVProb - baseVProb;
            const deltaVStr = (deltaV * 100).toFixed(0);

            return (
              <div
                key={v.id}
                className="p-3 rounded-lg bg-slate-900/90 border border-slate-800 flex items-center justify-between"
              >
                <div>
                  <span className="font-bold text-white font-mono text-sm block">{v.abbreviation}</span>
                  <span className="text-[10px] text-slate-400">
                    {(baseVProb * 100).toFixed(0)}% → <strong className="text-white">{(simVProb * 100).toFixed(0)}%</strong>
                  </span>
                </div>
                <div
                  className={`text-xs font-mono font-bold px-2 py-0.5 rounded ${
                    deltaV < 0
                      ? 'bg-emerald-500/20 text-emerald-300'
                      : deltaV > 0
                      ? 'bg-red-500/20 text-red-300'
                      : 'bg-slate-800 text-slate-400'
                  }`}
                >
                  {deltaV > 0 ? `+${deltaVStr} pp` : `${deltaVStr} pp`}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Modifiable Input Factor Sliders */}
      <div className="space-y-4">
        <span className="text-xs font-semibold uppercase tracking-wider text-slate-400 block">
          Adjust Continuous Clinical Inputs
        </span>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Systolic BP */}
          <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800 space-y-1.5">
            <div className="flex items-center justify-between text-xs">
              <span className="text-slate-300 font-medium">Resting Systolic BP</span>
              <span className="font-mono font-bold text-cyan-300">{simData.BP ?? 130} mmHg</span>
            </div>
            <input
              type="range"
              min="90"
              max="200"
              step="1"
              value={simData.BP ?? 130}
              onChange={(e) => updateField('BP', parseFloat(e.target.value))}
              className="w-full h-1.5 bg-slate-800 rounded-lg cursor-pointer accent-cyan-400"
            />
            <div className="flex justify-between text-[10px] text-slate-500 font-mono">
              <span>90 mmHg</span>
              <span>120</span>
              <span>140</span>
              <span>200</span>
            </div>
          </div>

          {/* Fasting Blood Sugar */}
          <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800 space-y-1.5">
            <div className="flex items-center justify-between text-xs">
              <span className="text-slate-300 font-medium">Fasting Blood Sugar (FBS)</span>
              <span className="font-mono font-bold text-cyan-300">{simData.FBS ?? 100} mg/dL</span>
            </div>
            <input
              type="range"
              min="70"
              max="250"
              step="1"
              value={simData.FBS ?? 100}
              onChange={(e) => updateField('FBS', parseFloat(e.target.value))}
              className="w-full h-1.5 bg-slate-800 rounded-lg cursor-pointer accent-cyan-400"
            />
            <div className="flex justify-between text-[10px] text-slate-500 font-mono">
              <span>70</span>
              <span>100</span>
              <span>126</span>
              <span>250</span>
            </div>
          </div>

          {/* LDL Cholesterol */}
          <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800 space-y-1.5">
            <div className="flex items-center justify-between text-xs">
              <span className="text-slate-300 font-medium">Serum LDL Cholesterol</span>
              <span className="font-mono font-bold text-cyan-300">{simData.LDL ?? 120} mg/dL</span>
            </div>
            <input
              type="range"
              min="40"
              max="250"
              step="1"
              value={simData.LDL ?? 120}
              onChange={(e) => updateField('LDL', parseFloat(e.target.value))}
              className="w-full h-1.5 bg-slate-800 rounded-lg cursor-pointer accent-cyan-400"
            />
            <div className="flex justify-between text-[10px] text-slate-500 font-mono">
              <span>40</span>
              <span>70</span>
              <span>100</span>
              <span>250</span>
            </div>
          </div>

          {/* Body Mass Index */}
          <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800 space-y-1.5">
            <div className="flex items-center justify-between text-xs">
              <span className="text-slate-300 font-medium">Body Mass Index (BMI)</span>
              <span className="font-mono font-bold text-cyan-300">{Number(simData.BMI ?? 27.2).toFixed(1)} kg/m²</span>
            </div>
            <input
              type="range"
              min="18.5"
              max="40"
              step="0.5"
              value={simData.BMI ?? 27.2}
              onChange={(e) => updateField('BMI', parseFloat(e.target.value))}
              className="w-full h-1.5 bg-slate-800 rounded-lg cursor-pointer accent-cyan-400"
            />
            <div className="flex justify-between text-[10px] text-slate-500 font-mono">
              <span>18.5</span>
              <span>25.0</span>
              <span>30.0</span>
              <span>40</span>
            </div>
          </div>
        </div>

        {/* Categorical Modifiers (Smoking only - Typical Chest Pain is a symptom, removed) */}
        <div className="flex flex-wrap items-center gap-3 pt-2">
          <label className="flex items-center gap-2 p-2.5 rounded-xl bg-slate-950/70 border border-slate-800 text-xs text-slate-300 cursor-pointer hover:border-slate-700">
            <input
              type="checkbox"
              checked={Boolean(simData['Current Smoker'])}
              onChange={(e) => updateField('Current Smoker', e.target.checked ? 1 : 0)}
              className="rounded bg-slate-800 border-slate-700 text-cyan-500 focus:ring-0"
            />
            <span>Active Cigarette Smoker</span>
          </label>
        </div>
      </div>

      {/* Apply Values to Active Patient Button */}
      <div className="pt-3 border-t border-slate-800 flex items-center justify-between">
        <span className="text-[11px] text-slate-400">
          Save explorer parameters to active patient form for full multi-target analysis.
        </span>

        <button
          onClick={handleApply}
          className="flex items-center gap-2 px-4 py-2 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold text-xs shadow-lg shadow-cyan-500/20 transition-all"
        >
          {appliedSuccess ? (
            <>
              <Check className="w-4 h-4 text-slate-950" />
              <span>Applied Successfully!</span>
            </>
          ) : (
            <>
              <Sliders className="w-4 h-4" />
              <span>Apply Explored Values to Patient</span>
            </>
          )}
        </button>
      </div>
    </div>
  );
}
