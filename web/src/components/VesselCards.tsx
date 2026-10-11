import React from 'react';
import { Target, CheckCircle2, AlertOctagon, ChevronRight, Shield } from 'lucide-react';
import vesselsConfig from '../vessels.json';
import { VesselsConfig, VesselPrediction } from '../types';
import { getVesselRiskColor } from './HeartViewer';

interface VesselCardsProps {
  vessels: Record<'LAD' | 'LCX' | 'RCA', VesselPrediction> | undefined;
  selectedVessel: string | null;
  onSelectVessel: (vesselId: string) => void;
  mode?: 'balanced' | 'screening';
}

const VESSEL_RELIABILITY_META: Record<
  'LAD' | 'LCX' | 'RCA',
  {
    ppvScreening: string;
    ppvBalanced: string;
    badge: string;
    badgeClass: string;
    prevalence: string;
    auc: string;
  }
> = {
  LAD: {
    ppvScreening: '76%',
    ppvBalanced: '79%',
    badge: 'Good Reliability',
    badgeClass: 'bg-emerald-500/15 text-emerald-300 border-emerald-500/30',
    prevalence: '58.4%',
    auc: '0.84',
  },
  LCX: {
    ppvScreening: '51%',
    ppvBalanced: '62%',
    badge: 'Mod. Reliability',
    badgeClass: 'bg-amber-500/15 text-amber-300 border-amber-500/30',
    prevalence: '39.3%',
    auc: '0.72',
  },
  RCA: {
    ppvScreening: '49%',
    ppvBalanced: '66%',
    badge: 'Mod. Reliability',
    badgeClass: 'bg-amber-500/15 text-amber-300 border-amber-500/30',
    prevalence: '37.6%',
    auc: '0.74',
  },
};

export function VesselCards({ vessels, selectedVessel, onSelectVessel, mode = 'balanced' }: VesselCardsProps) {
  const config = vesselsConfig as VesselsConfig;
  const isScreening = mode === 'screening';

  return (
    <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
      {config.vessels.map((vDef) => {
        const vKey = vDef.id as 'LAD' | 'LCX' | 'RCA';
        const vData = vessels ? vessels[vKey] : undefined;
        const hasData = vData !== undefined && vData.prob !== undefined;
        const prob = hasData ? vData.prob : 0;
        const percentage = hasData ? Math.round(prob * 100) : null;
        const activeCutoff = isScreening
          ? (vData?.high_sensitivity_threshold ?? 0.42)
          : (vData?.threshold ?? 0.50);
        const color = hasData ? getVesselRiskColor(prob, activeCutoff) : '#64748b';
        const isSelected = selectedVessel === vDef.id;
        const displayLabel = hasData
          ? (isScreening ? (vData.high_sens_label ?? 'Normal') : (vData.label ?? 'Normal'))
          : '—';
        const isStenotic = displayLabel === 'Stenotic';
        const relMeta = VESSEL_RELIABILITY_META[vKey];

        return (
          <div
            key={vDef.id}
            id={`vessel-card-${vDef.id}`}
            data-testid={`vessel-card-${vDef.id}`}
            tabIndex={0}
            role="button"
            aria-label={percentage !== null ? `${vDef.name} stenosis risk ${percentage}%, classified as ${displayLabel}` : `${vDef.name} stenosis risk unavailable`}
            onClick={() => onSelectVessel(vDef.id)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                onSelectVessel(vDef.id);
              }
            }}
            className={`p-3.5 rounded-xl cursor-pointer transition-all duration-200 border relative overflow-hidden group focus:outline-none focus:ring-2 focus:ring-cyan-400 ${
              isSelected
                ? 'bg-slate-800/95 border-cyan-400 shadow-lg shadow-cyan-500/10 scale-[1.02]'
                : 'bg-slate-900/70 border-slate-800/80 hover:bg-slate-800/70 hover:border-slate-700'
            }`}
          >
            {/* Top row: Name, Risk Dot, & Stenotic Status */}
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <span
                  className="w-3 h-3 rounded-full"
                  style={{ backgroundColor: color }}
                />
                <span className="font-bold text-white font-display text-base tracking-wide">
                  {vDef.abbreviation}
                </span>
              </div>

              <div className="flex items-center gap-1.5">
                <span
                  className={`text-[9px] px-1.5 py-0.5 rounded font-mono border ${relMeta.badgeClass}`}
                  title={`PPV ${isScreening ? relMeta.ppvScreening : relMeta.ppvBalanced} at cutoff ${activeCutoff.toFixed(2)} in this referral cohort (prevalence ${relMeta.prevalence}, AUC ${relMeta.auc})`}
                >
                  {isScreening ? `Screening PPV: ${relMeta.ppvScreening}` : `Default PPV: ${relMeta.ppvBalanced}`}
                </span>
                <span
                  className={`text-[10px] px-2 py-0.5 rounded-full font-bold uppercase tracking-wider border ${
                    isStenotic
                      ? 'bg-red-500/15 text-red-400 border-red-500/30'
                      : 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30'
                  }`}
                >
                  {displayLabel}
                </span>
              </div>
            </div>

            {/* Probability Number & Progress bar */}
            <div className="my-2">
              <div className="flex items-baseline justify-between mb-1">
                <span className="text-[11px] text-slate-400">P(Stenosis ≥ 50%)</span>
                <span className="text-xl font-extrabold font-mono" style={{ color }}>
                  {hasData ? (prob >= 0.99 ? '>99%' : (prob <= 0.01 ? '<1%' : `${percentage}%`)) : '—'}
                </span>
              </div>

              <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                <div
                  className="h-full rounded-full transition-all duration-500"
                  style={{ width: `${percentage ?? 0}%`, backgroundColor: color }}
                />
              </div>
            </div>

            {/* Territory subtitle & reliability note */}
            <div className="flex items-center justify-between text-[10px] text-slate-400 mb-2">
              <span className="truncate" title={vDef.territory}>{vDef.territory}</span>
              <span className="font-mono text-slate-500 flex-shrink-0 ml-1">{relMeta.badge}</span>
            </div>

            {/* Action Footer */}
            <div className="pt-2 border-t border-slate-800/80 flex items-center justify-between text-[11px]">
              <span className="text-slate-400 font-mono text-[10px]">
                Sens: {vData?.nested_sensitivity ? `${(vData.nested_sensitivity * 100).toFixed(1)}%` : '—'} (cutoff {activeCutoff.toFixed(2)})
              </span>
              <span
                className={`flex items-center gap-0.5 font-medium transition-colors ${
                  isSelected ? 'text-cyan-400 font-semibold' : 'text-slate-400 group-hover:text-slate-200'
                }`}
              >
                {isSelected ? 'Inspecting' : 'Explain'}
                <ChevronRight className="w-3 h-3" />
              </span>
            </div>
          </div>
        );
      })}
    </div>
  );
}
