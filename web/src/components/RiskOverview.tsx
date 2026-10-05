import React from 'react';
import { Activity, ShieldAlert, ShieldCheck, HelpCircle } from 'lucide-react';
import { CadPrediction } from '../types';
import { getRiskColor } from './HeartViewer';

interface RiskOverviewProps {
  cad: CadPrediction | undefined;
  isLoading?: boolean;
}

export function RiskOverview({ cad, isLoading }: RiskOverviewProps) {
  if (!cad) {
    return (
      <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 animate-pulse flex items-center justify-center h-48">
        <span className="text-slate-500 text-sm">Awaiting risk calculation...</span>
      </div>
    );
  }

  const prob = cad.coherent_prob;
  const percentage = Math.round(prob * 100);
  const displayPercentage = percentage >= 100 ? '>99%' : `${percentage}%`;
  const riskColor = getRiskColor(prob);
  const isHighRisk = cad.label === 'High Risk';
  const isHighSensHigh = cad.high_sens_label === 'High Risk';

  // SVG Gauge calculations
  const radius = 54;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (percentage / 100) * circumference;

  return (
    <div className="p-5 rounded-2xl bg-gradient-to-br from-slate-900/90 to-slate-950/90 border border-slate-800/80 shadow-xl relative overflow-hidden">
      {/* Background radial glow */}
      <div
        className="absolute -right-10 -top-10 w-40 h-40 rounded-full blur-3xl opacity-20 pointer-events-none"
        style={{ backgroundColor: riskColor }}
      />

      <div className="flex items-start justify-between mb-4">
        <div>
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
            Clinical Assessment
          </span>
          <h2 className="text-xl font-bold text-white font-display flex items-center gap-2 mt-0.5">
            Overall CAD Status
            <span
              className={`text-xs px-2.5 py-0.5 rounded-full font-sans font-bold flex items-center gap-1 border ${
                isHighRisk
                  ? 'bg-red-500/15 text-red-400 border-red-500/30'
                  : 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30'
              }`}
            >
              {isHighRisk ? <ShieldAlert className="w-3.5 h-3.5" /> : <ShieldCheck className="w-3.5 h-3.5" />}
              {cad.label}
            </span>
          </h2>
        </div>

        <div className="flex flex-col items-end">
          <span className="text-[11px] text-slate-400 font-mono">Operating Threshold: 0.50</span>
          <span className="text-[11px] text-amber-400/90 font-mono">
            Nested-CV Sens: {cad.nested_sensitivity ? `${(cad.nested_sensitivity * 100).toFixed(1)}%` : '90.9%'} (cutoff {cad.high_sensitivity_threshold.toFixed(2)})
          </span>
        </div>
      </div>

      <div className="flex flex-col sm:flex-row items-center gap-6">
        {/* Circular Gauge */}
        <div className="relative w-32 h-32 flex-shrink-0 flex items-center justify-center" style={{ width: '128px', height: '128px' }}>
          <svg className="w-32 h-32 transform -rotate-90" width="128" height="128" style={{ width: '128px', height: '128px', display: 'block' }} viewBox="0 0 128 128">
            <circle
              cx="64"
              cy="64"
              r={radius}
              stroke="currentColor"
              strokeWidth="9"
              className="text-slate-800"
              fill="transparent"
            />
            <circle
              cx="64"
              cy="64"
              r={radius}
              stroke={riskColor}
              strokeWidth="9"
              strokeDasharray={circumference}
              strokeDashoffset={strokeDashoffset}
              strokeLinecap="round"
              fill="transparent"
              className="transition-all duration-700 ease-out"
            />
          </svg>
          <div className="absolute flex flex-col items-center justify-center text-center">
            <span className="text-3xl font-extrabold font-display" style={{ color: riskColor }}>
              {displayPercentage}
            </span>
            <span className="text-[10px] text-slate-400 uppercase tracking-wider font-semibold">
              CAD Risk
            </span>
          </div>
        </div>

        {/* Narrative & Coherence Notes */}
        <div className="flex-1 space-y-2.5">
          <p className="text-sm text-slate-300 leading-relaxed">
            {isHighRisk
              ? 'Model indicates high likelihood of clinically significant coronary stenosis (≥50% diameter reduction in ≥1 major vessel).'
              : 'Model indicates low baseline likelihood of significant coronary stenosis based on clinical, lab, and ECG findings.'}
          </p>

          <div className="grid grid-cols-2 gap-2 text-xs pt-1">
            <div className="p-2 rounded-lg bg-slate-800/60 border border-slate-700/50">
              <span className="text-[10px] text-slate-400 block">Raw Model Score:</span>
              <span className="font-mono font-semibold text-slate-200">
                {cad.prob >= 0.995 ? '>99%' : `${(cad.prob * 100).toFixed(1)}%`}
              </span>
            </div>

            <div className="p-2 rounded-lg bg-slate-800/60 border border-slate-700/50">
              <span className="text-[10px] text-slate-400 block">Coherent Risk Score:</span>
              <span className="font-mono font-semibold text-slate-200 flex items-center gap-1">
                {cad.coherent_prob >= 0.995 ? '>99%' : `${(cad.coherent_prob * 100).toFixed(1)}%`}
                {cad.coherence_adjusted && (
                  <span className="text-[9px] px-1 py-0.2 rounded bg-amber-500/20 text-amber-300 border border-amber-500/30">
                    Max(Vessels)
                  </span>
                )}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
