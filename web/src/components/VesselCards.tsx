import React from 'react';
import { Target, CheckCircle2, AlertOctagon, ChevronRight } from 'lucide-react';
import vesselsConfig from '../vessels.json';
import { VesselsConfig, VesselPrediction } from '../types';
import { getRiskColor } from './HeartViewer';

interface VesselCardsProps {
  vessels: Record<'LAD' | 'LCX' | 'RCA', VesselPrediction> | undefined;
  selectedVessel: string | null;
  onSelectVessel: (vesselId: string) => void;
}

export function VesselCards({ vessels, selectedVessel, onSelectVessel }: VesselCardsProps) {
  const config = vesselsConfig as VesselsConfig;

  return (
    <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
      {config.vessels.map((vDef) => {
        const vKey = vDef.id as 'LAD' | 'LCX' | 'RCA';
        const vData = vessels ? vessels[vKey] : undefined;
        const prob = vData?.prob ?? 0;
        const percentage = Math.round(prob * 100);
        const color = getRiskColor(prob);
        const isSelected = selectedVessel === vDef.id;
        const isStenotic = vData?.label === 'Stenotic';

        return (
          <div
            key={vDef.id}
            id={`vessel-card-${vDef.id}`}
            data-testid={`vessel-card-${vDef.id}`}
            onClick={() => onSelectVessel(vDef.id)}
            className={`p-4 rounded-xl cursor-pointer transition-all duration-200 border relative overflow-hidden group ${
              isSelected
                ? 'bg-slate-800/95 border-cyan-400 shadow-lg shadow-cyan-500/10 scale-[1.02]'
                : 'bg-slate-900/70 border-slate-800/80 hover:bg-slate-800/70 hover:border-slate-700'
            }`}
          >
            {/* Top row: Name & Selection Indicator */}
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

              <span
                className={`text-[10px] px-2 py-0.5 rounded-full font-bold uppercase tracking-wider border ${
                  isStenotic
                    ? 'bg-red-500/15 text-red-400 border-red-500/30'
                    : 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30'
                }`}
              >
                {vData?.label ?? 'Normal'}
              </span>
            </div>

            {/* Probability Number & Progress bar */}
            <div className="my-2">
              <div className="flex items-baseline justify-between mb-1">
                <span className="text-[11px] text-slate-400">Stenosis Risk</span>
                <span className="text-xl font-extrabold font-mono" style={{ color }}>
                  {percentage}%
                </span>
              </div>

              <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                <div
                  className="h-full rounded-full transition-all duration-500"
                  style={{ width: `${percentage}%`, backgroundColor: color }}
                />
              </div>
            </div>

            {/* Territory subtitle */}
            <p className="text-[11px] text-slate-400 line-clamp-1 mt-1 mb-2">
              {vDef.territory}
            </p>

            {/* Action Footer */}
            <div className="pt-2 border-t border-slate-800/80 flex items-center justify-between text-[11px]">
              <span className="text-slate-500 font-mono text-[10px]">
                Sens-Thresh: {vData?.high_sensitivity_threshold ?? 0.35}
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
