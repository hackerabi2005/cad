import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Sliders, Sparkles, TrendingDown, ArrowRight, RefreshCw, Check, Activity, Heart, ShieldAlert } from 'lucide-react';
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

  // Run simulation prediction against /api/predict
  const runSimPrediction = useCallback(async (data: Record<string, any>) => {
    try {
      setIsSimulating(true);
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      });
      if (res.ok) {
        const result: PredictResponse = await res.json();
        setSimPrediction(result);
      }
    } catch (e) {
      console.error('Simulation error:', e);
    } finally {
      setIsSimulating(false);
    }
  }, []);

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
    setSimData((prev) => ({ ...prev, [field]: val }));
    setAppliedSuccess(false);
  };

  // Quick Clinical Presets
  const applyIntensiveGDMT = () => {
    setSimData((prev) => ({
      ...prev,
      BP: 118,
      FBS: 92,
      LDL: 68,
      'Current Smoker': 0,
      DM: 0,
    }));
  };

  const applyLifestyle = () => {
    setSimData((prev) => ({
      ...prev,
      BMI: 23.5,
      BP: 124,
      'Current Smoker': 0,
      FBS: 95,
    }));
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
  const relativeReduction =
    baselineProb > 0 ? (((baselineProb - simulatedProb) / baselineProb) * 100).toFixed(0) : '0';

  const config = vesselsConfig as VesselsConfig;

  return (
    <div className="p-5 rounded-2xl bg-slate-900/80 border border-slate-800/80 shadow-xl space-y-5 animate-fade-in">
      {/* Title & Presets Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-slate-800">
        <div>
          <span className="text-xs font-semibold uppercase tracking-wider text-cyan-400">
            Interactive Counterfactual Modeling
          </span>
          <h3 className="text-lg font-bold text-white font-display flex items-center gap-2">
            What-If Risk Reduction Simulator
            <span className="text-xs px-2 py-0.5 rounded-full bg-cyan-500/10 text-cyan-300 border border-cyan-500/30">
              Live Real-Time
            </span>
          </h3>
        </div>

        {/* Quick Clinical Intervention Actions */}
        <div className="flex flex-wrap items-center gap-1.5">
          <button
            onClick={applyIntensiveGDMT}
            className="flex items-center gap-1 px-2.5 py-1.5 rounded-lg bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 text-xs font-semibold hover:bg-emerald-500/30 transition-colors"
            title="Intensive medical therapy (BP <120, LDL <70, FBS <100, Stop Smoking)"
          >
            <Sparkles className="w-3.5 h-3.5" />
            <span>Target GDMT</span>
          </button>
          <button
            onClick={applyLifestyle}
            className="flex items-center gap-1 px-2.5 py-1.5 rounded-lg bg-sky-500/20 text-sky-300 border border-sky-500/30 text-xs font-semibold hover:bg-sky-500/30 transition-colors"
            title="Lifestyle optimization (BMI 23.5, BP 124, Non-smoker)"
          >
            <Activity className="w-3.5 h-3.5" />
            <span>Lifestyle Target</span>
          </button>
          <button
            onClick={resetToCurrent}
            className="flex items-center gap-1 px-2.5 py-1.5 rounded-lg bg-slate-800/80 text-slate-300 border border-slate-700/60 text-xs font-medium hover:bg-slate-700 transition-colors"
            title="Reset simulator values to current patient profile"
          >
            <RefreshCw className="w-3 h-3" />
            <span>Reset</span>
          </button>
        </div>
      </div>

      {/* Primary Comparative Risk Delta Bar */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 p-4 rounded-xl bg-slate-950/70 border border-slate-800">
        {/* Baseline Risk Card */}
        <div className="p-3 rounded-lg bg-slate-900/80 border border-slate-800/80 text-center">
          <span className="text-[10px] uppercase font-bold text-slate-400 block mb-1">
            Current Patient Risk
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
            <span className="text-xs text-slate-400">Simulated Delta</span>
            <ArrowRight className="w-4 h-4 text-cyan-400" />
          </div>
          <div
            className={`text-2xl font-black font-mono flex items-center gap-1 ${
              riskDelta < 0 ? 'text-emerald-400' : riskDelta > 0 ? 'text-red-400' : 'text-slate-400'
            }`}
          >
            {riskDelta > 0 ? `+${riskDeltaPercent}%` : `${riskDeltaPercent}%`}
          </div>
          <span className="text-[10px] text-slate-400">
            {riskDelta < 0
              ? `Relative reduction: ~${relativeReduction}%`
              : riskDelta > 0
              ? 'Increased stenosis probability'
              : 'No modification'}
          </span>
        </div>

        {/* Counterfactual Simulated Risk Card */}
        <div className="p-3 rounded-lg bg-slate-900/80 border border-cyan-500/40 text-center">
          <span className="text-[10px] uppercase font-bold text-cyan-400 block mb-1">
            Counterfactual Risk
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
          Coronary Artery Branch Occlusion Deltas
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
                  {deltaV > 0 ? `+${deltaVStr}%` : `${deltaVStr}%`}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Modifiable Risk Factor Sliders */}
      <div className="space-y-4">
        <span className="text-xs font-semibold uppercase tracking-wider text-slate-400 block">
          Adjust Modifiable Clinical Targets
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
              <span>90 mmHg (Ideal)</span>
              <span>120</span>
              <span>140 (HTN)</span>
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
              <span>70 (Normal)</span>
              <span>100 (Impaired)</span>
              <span>126 (Diabetic)</span>
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
              <span>&lt;55 (Target High-Risk)</span>
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
              <span>18.5 (Normal)</span>
              <span>25.0 (Overweight)</span>
              <span>30.0 (Obese)</span>
              <span>40</span>
            </div>
          </div>
        </div>

        {/* Categorical Modifiers (Smoking, Typical Angina) */}
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

          <label className="flex items-center gap-2 p-2.5 rounded-xl bg-slate-950/70 border border-slate-800 text-xs text-slate-300 cursor-pointer hover:border-slate-700">
            <input
              type="checkbox"
              checked={Boolean(simData['Typical Chest Pain'])}
              onChange={(e) => updateField('Typical Chest Pain', e.target.checked ? 1 : 0)}
              className="rounded bg-slate-800 border-slate-700 text-cyan-500 focus:ring-0"
            />
            <span>Typical Angina Symptoms</span>
          </label>
        </div>
      </div>

      {/* Apply Counterfactual to Active Patient Button */}
      <div className="pt-3 border-t border-slate-800 flex items-center justify-between">
        <span className="text-[11px] text-slate-400">
          Save counterfactual parameters to active patient form for full multi-target analysis.
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
              <span>Apply Counterfactual to Patient</span>
            </>
          )}
        </button>
      </div>
    </div>
  );
}
