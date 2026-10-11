import React from 'react';
import { Activity, ShieldAlert, ShieldCheck } from 'lucide-react';
import { CadPrediction } from '../types';
import { getRiskColor } from './HeartViewer';

interface RiskOverviewProps {
  cad: CadPrediction | undefined;
  isLoading?: boolean;
  mode?: 'balanced' | 'screening';
  onToggleMode?: () => void;
}

export function RiskOverview({ cad, isLoading, mode = 'balanced', onToggleMode }: RiskOverviewProps) {
  if (!cad) {
    return (
      <div className="p-5 rounded-2xl bg-slate-900/60 border border-slate-800 animate-pulse flex items-center justify-center h-48">
        <span className="text-slate-500 text-sm">Awaiting risk calculation...</span>
      </div>
    );
  }

  const prob = cad.coherent_prob;
  const percentage = Math.round(prob * 100);
  const displayPercentage = prob >= 0.99 ? '>99%' : (prob <= 0.01 ? '<1%' : `${percentage}%`);
  const riskColor = getRiskColor(prob);
  const isScreening = mode === 'screening';
  const activeLabel = isScreening ? cad.high_sens_label : cad.label;
  const isHighRisk = activeLabel === 'High Risk';
  const activeCutoff = isScreening ? cad.high_sensitivity_threshold : cad.threshold;
  const flaggedBranches = isScreening
    ? (cad.flagged_vessels_screening ?? [])
    : (cad.flagged_vessels_balanced ?? []);
  const branchCount = flaggedBranches.length;
  const isReverseContradiction = isHighRisk && branchCount === 0;

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
            Clinical Assessment ({isScreening ? 'Screening Mode' : 'Default (0.50) Mode'})
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
              {activeLabel}
            </span>
          </h2>
        </div>

        <div className="flex flex-col items-end gap-1">
          {onToggleMode && (
            <button
              onClick={onToggleMode}
              className="text-[10px] px-2.5 py-1 rounded-md border border-cyan-500/40 bg-cyan-500/15 text-cyan-300 font-mono font-medium hover:bg-cyan-500/25 transition-all flex items-center gap-1.5"
              title={`Click to toggle between Default (0.50) and Screening modes (${
                isScreening
                  ? 'Screening mode (0.40 cutoff): Cath model achieves Sens 95.9% (208/217), Spec 62.8% (54/86). Multivessel Consistency Rule (OR flag) achieves Sens 96.3% (209/217), Spec 57.0% (49/86) — gaining 1 true positive for 5 false alarms to ensure zero flagged vessels receive Low Risk CAD.'
                  : 'Default mode (0.50 cutoff): Balanced standard cutoff — Sens: 94.5% (205/217), Spec: 70.9% (61/86).'
              })`}
            >
              <span>Mode: <strong className="uppercase">{isScreening ? 'Screening' : 'Default (0.50)'}</strong></span>
              <span className="text-[9px] text-cyan-400/80 font-mono border-l border-cyan-500/30 pl-1.5">
                {isScreening ? 'Cath: 95.9%/62.8% · Flag: 96.3%/57.0%' : 'Sens: 94.5% · Spec: 70.9%'}
              </span>
            </button>
          )}
          <span className="text-[11px] text-slate-400 font-mono">
            {isScreening ? `Screening Cutoff: ${activeCutoff.toFixed(2)}` : 'Operating Threshold: 0.50'}
          </span>
          <span className="text-[11px] text-amber-400/90 font-mono">
            Nested-CV Sens: {cad.nested_sensitivity ? `${(cad.nested_sensitivity * 100).toFixed(1)}%` : '—'}
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
            {cad.coherence_adjusted ? (
              <span
                className="text-[8px] text-amber-300 font-mono tracking-tight px-1 font-semibold"
                title={`Cath model alone: ${((cad.raw_prob ?? cad.prob) * 100).toFixed(1)}%; raised to ${cad.driving_vessel || 'vessel'}’s ${(cad.coherent_prob * 100).toFixed(1)}%.`}
              >
                Cath alone: {((cad.raw_prob ?? cad.prob) * 100).toFixed(1)}%
              </span>
            ) : cad.similar_cohort ? (
              <span className="text-[8px] text-slate-400 font-mono tracking-tight px-1 text-center" title={`Wilson 95%: ${(cad.similar_cohort.wilson_interval[0] * 100).toFixed(0)}%–${(cad.similar_cohort.wilson_interval[1] * 100).toFixed(0)}%`}>
                n={cad.similar_cohort.n} ({(cad.similar_cohort.observed_rate * 100).toFixed(0)}%)
              </span>
            ) : (
              <span className="text-[8px] text-slate-500 font-mono tracking-tight">
                Empirical ref
              </span>
            )}
          </div>
        </div>

        {/* Narrative & Coherence Notes */}
        <div className="flex-1 space-y-2.5">
          <p className="text-sm text-slate-300 leading-relaxed">
            {isHighRisk
              ? 'Model indicates high likelihood of clinically significant coronary stenosis (≥50% diameter reduction in ≥1 major vessel).'
              : 'Model indicates low baseline likelihood of significant coronary stenosis based on clinical, lab, and ECG findings.'}
          </p>

          <div className="grid grid-cols-4 gap-2 text-xs pt-1">
            <div className="p-2 rounded-lg bg-slate-800/60 border border-slate-700/50">
              <span className="text-[10px] text-slate-400 block">Raw Model:</span>
              <span className="font-mono font-semibold text-slate-200">
                {(cad.raw_prob ?? cad.prob) >= 0.99 ? '>99%' : `${((cad.raw_prob ?? cad.prob) * 100).toFixed(1)}%`}
              </span>
            </div>

            <div className="p-2 rounded-lg bg-slate-800/60 border border-slate-700/50">
              <span className="text-[10px] text-slate-400 block">Coherent Risk:</span>
              <span className="font-mono font-semibold text-slate-200 flex items-center gap-1">
                {cad.coherent_prob >= 0.99 ? '>99%' : `${(cad.coherent_prob * 100).toFixed(1)}%`}
                {cad.coherence_adjusted && (
                  <span
                    className="text-[9px] px-1 py-0.2 rounded bg-amber-500/20 text-amber-300 border border-amber-500/30"
                    title={cad.driving_vessel ? `Adjusted to match ${cad.driving_vessel} by max-rule` : 'Raised by coherence rule'}
                  >
                    Max ({cad.driving_vessel ?? 'Rule'})
                  </span>
                )}
              </span>
            </div>

            <div className="p-2 rounded-lg bg-slate-800/60 border border-slate-700/50">
              <span className="text-[10px] text-slate-400 block">Vessels above cutoff:</span>
              <span className="font-mono font-semibold text-amber-300 text-[11px] block">
                {(cad.flagged_vessels_count ?? cad.diseased_vessels_count ?? branchCount) !== undefined
                  ? `${cad.flagged_vessels_count ?? cad.diseased_vessels_count ?? branchCount} / 3`
                  : '—'}
              </span>
            </div>

            <div className="p-2 rounded-lg bg-slate-800/60 border border-slate-700/50">
              <span className="text-[10px] text-slate-400 block">Similar Cohort:</span>
              <span className="font-mono font-semibold text-cyan-300 text-[11px] block" title={cad.similar_cohort ? `Observed rate in 50 nearest patients: 95% CI ${(cad.similar_cohort.wilson_interval[0] * 100).toFixed(0)}%–${(cad.similar_cohort.wilson_interval[1] * 100).toFixed(0)}%` : undefined}>
                {cad.similar_cohort
                  ? `${(cad.similar_cohort.observed_rate * 100).toFixed(0)}% (n=${cad.similar_cohort.n})`
                  : '—'}
              </span>
            </div>
          </div>

          {cad.flagged_by_vessel && (
            <div className="text-[11px] px-2.5 py-1 rounded-lg bg-amber-500/15 border border-amber-500/30 text-amber-300 flex items-center gap-1.5">
              <ShieldAlert className="w-3.5 h-3.5 text-amber-400 flex-shrink-0" />
              <span>Overall status flagged positive via <strong>{(cad.flagged_vessels_screening ?? cad.flagged_vessels_balanced)?.join(', ')}</strong> stenosis (logical disjunctive rule).</span>
            </div>
          )}

          {isHighRisk && !cad.flagged_by_vessel && branchCount === 0 && (
            <div className="text-[11px] px-2.5 py-1 rounded-lg bg-cyan-500/15 border border-cyan-500/30 text-cyan-300 flex items-center gap-1.5">
              <ShieldAlert className="w-3.5 h-3.5 text-cyan-400 flex-shrink-0" />
              <span>Overall model above cutoff; no vessel model above its cutoff; location not determined (no single vessel crosses its cutoff, so the model cannot localize focal stenosis).</span>
            </div>
          )}

          {cad.plain_language_summary && (
            <p className="text-xs text-cyan-300/90 bg-cyan-950/30 p-2 rounded-lg border border-cyan-500/20 leading-relaxed font-sans">
              <strong>Clinical Assessment:</strong> {cad.plain_language_summary}
            </p>
          )}

          {cad.similar_cohort && (
            <p className="text-[11px] text-slate-400 font-mono mt-1" title="Observed positive rate among 50 nearest reference cohort patients (Wilson score 95% CI); not model uncertainty.">
              Observed in {cad.similar_cohort.n} similar patients: {(cad.similar_cohort.observed_rate * 100).toFixed(0)}% (95% CI {(cad.similar_cohort.wilson_interval[0] * 100).toFixed(0)}%–{(cad.similar_cohort.wilson_interval[1] * 100).toFixed(0)}%)
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
