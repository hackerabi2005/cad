import React from 'react';
import { X, Droplet, ShieldAlert } from 'lucide-react';
import vesselsConfig from '../vessels.json';
import { VesselsConfig } from '../types';
import { getRiskColor } from './HeartViewer';

interface LumenCrossSectionProps {
  vesselId: string;
  prob: number;
  onClose: () => void;
  onSelectVessel?: (vesselId: string) => void;
}

export function LumenCrossSection({
  vesselId,
  prob,
  onClose,
  onSelectVessel,
}: LumenCrossSectionProps) {
  const config = vesselsConfig as VesselsConfig;
  const vesselMeta = config.vessels.find((v) => v.id === vesselId) ?? {
    id: vesselId,
    name: vesselId,
    abbreviation: vesselId,
    territory: 'Coronary artery territory',
    description: 'Coronary perfusion vessel',
  };

  // Fixed illustrative narrowing (normal vs. ≥ 50%)
  const isStenotic = prob >= 0.50;
  const riskColor = getRiskColor(prob);
  const predictedLabel = isStenotic ? 'Likely Stenotic (≥ 50%)' : 'Likely Normal (< 50%)';

  // SVG dimensions & geometry
  const cx = 100;
  const cy = 100;
  const outerRadius = 78; // External elastic lamina (arterial wall)
  const mediaRadius = 70; // Tunica media
  const normalLumenRadius = 58; // Healthy lumen radius

  // Fixed illustrative patent radius: normal vs >= 50%
  const patentRadius = isStenotic ? 28 : normalLumenRadius;

  return (
    <div
      className="p-4 rounded-2xl bg-slate-900/95 border border-cyan-500/40 shadow-2xl backdrop-blur-xl text-slate-100 flex flex-col gap-4 animate-fade-in"
      style={{ maxWidth: '440px', width: '100%' }}
    >
      {/* Header with Vessel Selector */}
      <div className="flex items-center justify-between border-b border-slate-800 pb-2.5">
        <div className="flex items-center gap-2">
          <Droplet className="w-4 h-4 text-rose-400" />
          <h3 className="text-sm font-bold tracking-tight text-white flex items-center gap-1.5">
            Luminal Stenosis Schematic
            <span
              className="text-xs px-2 py-0.5 rounded-full font-mono font-bold"
              style={{ backgroundColor: `${riskColor}22`, color: riskColor, border: `1px solid ${riskColor}66` }}
            >
              {vesselMeta.abbreviation}
            </span>
          </h3>
        </div>

        <div className="flex items-center gap-1">
          {config.vessels.map((v) => (
            <button
              key={v.id}
              onClick={() => onSelectVessel && onSelectVessel(v.id)}
              className={`px-2 py-0.5 rounded text-[10px] font-bold font-mono transition-colors ${
                v.id === vesselId
                  ? 'bg-cyan-500 text-slate-950 shadow-sm'
                  : 'bg-slate-800/80 text-slate-400 hover:text-white'
              }`}
            >
              {v.abbreviation}
            </button>
          ))}
          <button
            onClick={onClose}
            className="p-1 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors ml-1"
            title="Close Schematic"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Anatomical Territory */}
      <div className="text-[11px] text-slate-300 bg-slate-950/60 p-2.5 rounded-xl border border-slate-800/80">
        <span className="text-slate-400 font-semibold block text-[10px] uppercase tracking-wider mb-0.5">
          Perfusion Territory:
        </span>
        {vesselMeta.territory}
      </div>

      {/* Radial Cross-Section Schematic */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 items-center">
        <div className="flex flex-col items-center">
          <svg width="180" height="180" viewBox="0 0 200 200" className="drop-shadow-md select-none">
            <defs>
              {/* Plaque lipid core gradient */}
              <linearGradient id="plaqueGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#f59e0b" />
                <stop offset="60%" stopColor="#d97706" />
                <stop offset="100%" stopColor="#b45309" />
              </linearGradient>
              {/* Patent blood flow gradient */}
              <radialGradient id="bloodFlowGrad" cx="50%" cy="50%" r="50%">
                <stop offset="0%" stopColor="#ef4444" />
                <stop offset="70%" stopColor="#b91c1c" />
                <stop offset="100%" stopColor="#7f1d1d" />
              </radialGradient>
            </defs>

            {/* External adventitial ring */}
            <circle cx={cx} cy={cy} r={outerRadius} fill="#1e293b" stroke="#334155" strokeWidth="3" />

            {/* Media muscle ring */}
            <circle cx={cx} cy={cy} r={mediaRadius} fill="#0f172a" stroke="#475569" strokeWidth="2" strokeDasharray="4 2" />

            {/* Healthy reference boundary (dashed green) */}
            <circle
              cx={cx}
              cy={cy}
              r={normalLumenRadius}
              fill="none"
              stroke="#10b981"
              strokeWidth="1.5"
              strokeDasharray="3 3"
              opacity="0.5"
            />

            {/* Atherosclerotic Plaque Crescent - only if narrowing >= 50% */}
            {isStenotic && (
              <path
                d={`
                  M ${cx - normalLumenRadius * 0.95} ${cy}
                  A ${normalLumenRadius} ${normalLumenRadius} 0 0 1 ${cx + normalLumenRadius * 0.95} ${cy}
                  Q ${cx} ${cy - patentRadius * 0.3} ${cx - normalLumenRadius * 0.95} ${cy}
                  Z
                `}
                fill="url(#plaqueGrad)"
                stroke="#f59e0b"
                strokeWidth="1"
                opacity="0.9"
              />
            )}

            {/* Residual Patent Lumen (blood channel) */}
            <circle
              cx={cx}
              cy={isStenotic ? cy + (normalLumenRadius - patentRadius) * 0.4 : cy}
              r={patentRadius}
              fill="url(#bloodFlowGrad)"
              stroke={riskColor}
              strokeWidth="2.5"
            />

            {/* Center Blood Velocity Arrow / Pulse */}
            <circle
              cx={cx}
              cy={isStenotic ? cy + (normalLumenRadius - patentRadius) * 0.4 : cy}
              r={patentRadius * 0.35}
              fill="#fca5a5"
              opacity="0.7"
            />

            {/* Callout text labels */}
            <text x="100" y="24" fill="#94a3b8" fontSize="9" textAnchor="middle" fontFamily="monospace">
              External Adventitia
            </text>
            <text x="100" y="190" fill={riskColor} fontSize="10" textAnchor="middle" fontWeight="bold" fontFamily="monospace">
              {isStenotic ? 'Illustrative Narrowed Lumen' : 'Illustrative Normal Lumen'}
            </text>
          </svg>
          <span className="text-[10px] text-slate-400 mt-1 font-mono">
            {isStenotic ? 'Fixed Narrowing (≥ 50%)' : 'Normal Lumen (< 50%)'}
          </span>
        </div>

        {/* Model Probability & Classification (No Invented Severity) */}
        <div className="space-y-2.5 text-xs">
          <div className="p-2 rounded-lg bg-slate-950/70 border border-slate-800">
            <span className="text-slate-400 text-[10px] uppercase font-bold block">Model Probability</span>
            <div className="flex items-baseline gap-2 mt-0.5">
              <span className="text-lg font-extrabold font-mono" style={{ color: riskColor }}>
                {(prob * 100).toFixed(1)}%
              </span>
              <span className="text-[10px] text-slate-400">P(Stenosis ≥ 50%)</span>
            </div>
          </div>

          <div className="p-2 rounded-lg bg-slate-950/70 border border-slate-800">
            <span className="text-slate-400 text-[10px] uppercase font-bold block">Model Prediction</span>
            <div className="text-[12px] font-bold mt-0.5 font-mono" style={{ color: riskColor }}>
              {predictedLabel}
            </div>
            <span className="text-[10px] text-slate-400">Binary cutoff: 50%</span>
          </div>

          <div className="p-2 rounded-lg bg-slate-950/70 border border-slate-800">
            <span className="text-slate-400 text-[10px] uppercase font-bold block">Target Vessel</span>
            <div className="text-[11px] font-semibold text-slate-200 mt-0.5">
              {vesselMeta.name} ({vesselMeta.abbreviation})
            </div>
          </div>
        </div>
      </div>

      {/* Mandatory Disclaimer Caption */}
      <div className="p-2.5 rounded-xl bg-slate-950/80 border border-amber-500/30 text-[11px] flex items-start gap-2">
        <ShieldAlert className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" />
        <div>
          <span className="font-semibold text-amber-300 block">
            Schematic only — not patient imaging; stenosis severity is not predicted
          </span>
          <span className="text-slate-400 text-[10px]">
            The underlying clinical dataset records binary stenosis (≥ 50%). Continuous percentage narrowing and physiological flow indices are not predicted.
          </span>
        </div>
      </div>
    </div>
  );
}
