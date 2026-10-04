import React, { useState } from 'react';
import { TrendingUp, TrendingDown, HelpCircle, BarChart3, ListFilter, Cpu } from 'lucide-react';
import { TargetExplanation, FeatureContribution } from '../types';

interface ShapWaterfallProps {
  explanations: Record<string, TargetExplanation> | undefined;
  activeTarget: string; // 'Cath' | 'LAD' | 'LCX' | 'RCA'
  onSelectTarget: (target: string) => void;
}

export function ShapWaterfall({
  explanations,
  activeTarget,
  onSelectTarget,
}: ShapWaterfallProps) {
  const [viewMode, setViewMode] = useState<'chart' | 'table'>('chart');

  const exp = explanations ? explanations[activeTarget] : undefined;

  const targetTitles: Record<string, string> = {
    Cath: 'Overall CAD Diagnosis Model',
    LAD: 'Left Anterior Descending (LAD) Stenosis Model',
    LCX: 'Left Circumflex (LCX) Stenosis Model',
    RCA: 'Right Coronary Artery (RCA) Stenosis Model',
  };

  if (!exp) {
    return (
      <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 animate-pulse flex items-center justify-center h-64">
        <span className="text-slate-500 text-sm">Computing SHAP attributions...</span>
      </div>
    );
  }

  // Top 10 features + other aggregated
  const top10 = exp.features.slice(0, 10);
  const remaining = exp.features.slice(10);
  const remainingShap = remaining.reduce((acc, f) => acc + f.shap, 0);
  const remainingAbsPct = remaining.reduce((acc, f) => acc + f.pct, 0);

  // Identify top risk-elevating and risk-lowering drivers for natural language summary
  const topRiskElevating = exp.features
    .filter((f) => f.shap > 0)
    .slice(0, 3)
    .map((f) => `${f.label} (${f.value ?? 'Abnormal'})`);

  const topRiskLowering = exp.features
    .filter((f) => f.shap < 0)
    .slice(0, 2)
    .map((f) => `${f.label} (${f.value ?? 'Normal'})`);

  // Max absolute SHAP for bar scaling
  const maxAbsShap = Math.max(...exp.features.map((f) => Math.abs(f.shap)), 0.001);

  return (
    <div className="p-5 rounded-2xl bg-slate-900/80 border border-slate-800/80 shadow-xl space-y-4">
      {/* Header and Target Selector Tabs */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-slate-800">
        <div>
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
            Explainable AI Attribution
          </span>
          <h3 className="text-lg font-bold text-white font-display">
            {targetTitles[activeTarget] ?? activeTarget}
          </h3>
        </div>

        {/* Target Buttons */}
        <div className="flex items-center gap-1 p-1 bg-slate-950/80 rounded-xl border border-slate-800">
          {(['Cath', 'LAD', 'LCX', 'RCA'] as const).map((t) => (
            <button
              key={t}
              id={`btn-shap-${t}`}
              data-testid={`btn-shap-${t}`}
              onClick={() => onSelectTarget(t)}
              className={`px-2.5 py-1 rounded-lg text-xs font-semibold transition-all ${
                activeTarget === t
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              {t === 'Cath' ? 'Overall CAD' : t}
            </button>
          ))}
        </div>
      </div>

      {/* Natural Language Clinical Summary Banner */}
      <div className="p-3 rounded-xl bg-slate-950/70 border border-slate-800/80 text-xs leading-relaxed text-slate-300 flex items-start gap-2.5">
        <Cpu className="w-4 h-4 text-cyan-400 flex-shrink-0 mt-0.5" />
        <div>
          <span className="font-semibold text-white">Clinical Model Explanation: </span>
          {topRiskElevating.length > 0 ? (
            <span>
              Primary factors elevating risk: <strong className="text-red-300">{topRiskElevating.join(', ')}</strong>.
            </span>
          ) : (
            <span>No strong risk-elevating factors detected.</span>
          )}
          {topRiskLowering.length > 0 && (
            <span className="ml-1">
              Protective / mitigating factors: <strong className="text-emerald-300">{topRiskLowering.join(', ')}</strong>.
            </span>
          )}
        </div>
      </div>

      {/* View Mode Toggle & Model Base Value */}
      <div className="flex items-center justify-between text-xs text-slate-400">
        <div className="flex items-center gap-2">
          <span className="font-mono text-[11px]">
            Base Value: <strong className="text-slate-200">{exp.base_value}</strong>
          </span>
          <span className="text-slate-600">|</span>
          <span className="font-mono text-[11px]">
            Model Score: <strong className="text-slate-200">{exp.raw_score}</strong>
          </span>
        </div>

        <div className="flex items-center gap-1 bg-slate-800/60 p-0.5 rounded-lg border border-slate-700/50">
          <button
            id="btn-shap-chart-view"
            data-testid="btn-shap-chart-view"
            onClick={() => setViewMode('chart')}
            className={`px-2 py-0.5 rounded text-[11px] font-medium flex items-center gap-1 ${
              viewMode === 'chart' ? 'bg-cyan-500/20 text-cyan-300 font-semibold' : 'text-slate-400'
            }`}
          >
            <BarChart3 className="w-3 h-3" />
            Bars
          </button>
          <button
            id="btn-shap-table-view"
            data-testid="btn-shap-table-view"
            onClick={() => setViewMode('table')}
            className={`px-2 py-0.5 rounded text-[11px] font-medium flex items-center gap-1 ${
              viewMode === 'table' ? 'bg-cyan-500/20 text-cyan-300 font-semibold' : 'text-slate-400'
            }`}
          >
            <ListFilter className="w-3 h-3" />
            Table
          </button>
        </div>
      </div>

      {/* Waterfall / Horizontal Signed Bar Chart View */}
      {viewMode === 'chart' ? (
        <div className="space-y-2 pt-1 max-h-[380px] overflow-y-auto pr-1">
          {top10.map((f) => {
            const isRisk = f.shap > 0;
            const barWidth = Math.max(3, Math.round((Math.abs(f.shap) / maxAbsShap) * 100));

            return (
              <div key={f.feature} className="group text-xs">
                <div className="flex items-center justify-between mb-0.5">
                  <div className="flex items-center gap-1.5 truncate max-w-[70%]">
                    {isRisk ? (
                      <TrendingUp className="w-3.5 h-3.5 text-red-400 flex-shrink-0" />
                    ) : (
                      <TrendingDown className="w-3.5 h-3.5 text-emerald-400 flex-shrink-0" />
                    )}
                    <span className="text-slate-200 font-medium truncate" title={f.label}>
                      {f.label}
                    </span>
                    {f.value !== null && (
                      <span className="text-[10px] text-slate-400 font-mono">
                        ({f.value} {f.unit})
                      </span>
                    )}
                  </div>

                  <div className="flex items-center gap-2 font-mono text-[11px]">
                    <span className={isRisk ? 'text-red-400 font-bold' : 'text-emerald-400 font-bold'}>
                      {f.shap > 0 ? `+${f.shap.toFixed(3)}` : f.shap.toFixed(3)}
                    </span>
                    <span className="text-slate-500 text-[10px] w-9 text-right">{f.pct.toFixed(0)}%</span>
                  </div>
                </div>

                {/* Signed Diverging Bar */}
                <div className="w-full bg-slate-950 h-2 rounded-full overflow-hidden flex relative">
                  {/* Center vertical reference line */}
                  <div className="absolute left-1/2 top-0 bottom-0 w-px bg-slate-700 z-10" />

                  {isRisk ? (
                    // Bar extends right from center (50%)
                    <div className="w-full h-full flex">
                      <div className="w-1/2 h-full" />
                      <div className="w-1/2 h-full flex items-center">
                        <div
                          className="h-full rounded-r-full bg-gradient-to-r from-red-500 to-rose-400"
                          style={{ width: `${barWidth}%` }}
                        />
                      </div>
                    </div>
                  ) : (
                    // Bar extends left from center (50%)
                    <div className="w-full h-full flex">
                      <div className="w-1/2 h-full flex items-center justify-end">
                        <div
                          className="h-full rounded-l-full bg-gradient-to-l from-emerald-500 to-teal-400"
                          style={{ width: `${barWidth}%` }}
                        />
                      </div>
                      <div className="w-1/2 h-full" />
                    </div>
                  )}
                </div>
              </div>
            );
          })}

          {/* Aggregated Other Features Row */}
          {remaining.length > 0 && (
            <div className="pt-2 border-t border-slate-800 text-xs">
              <div className="flex items-center justify-between mb-0.5 text-slate-400">
                <span className="italic">Other {remaining.length} physiological features combined</span>
                <span className="font-mono text-[11px] font-semibold text-slate-300">
                  {remainingShap > 0 ? `+${remainingShap.toFixed(3)}` : remainingShap.toFixed(3)}
                  <span className="text-slate-500 text-[10px] ml-2">({remainingAbsPct.toFixed(0)}%)</span>
                </span>
              </div>
            </div>
          )}
        </div>
      ) : (
        /* Detailed Measurement vs Attribution Table */
        <div className="max-h-[380px] overflow-y-auto rounded-xl border border-slate-800">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-950/80 text-slate-400 uppercase text-[10px] tracking-wider border-b border-slate-800 sticky top-0">
              <tr>
                <th className="py-2 px-3">Physiological Feature</th>
                <th className="py-2 px-2">Patient Value</th>
                <th className="py-2 px-2 text-right">SHAP Contribution</th>
                <th className="py-2 px-3 text-right">Impact %</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-850 font-mono">
              {exp.features.map((f) => (
                <tr key={f.feature} className="hover:bg-slate-800/40">
                  <td className="py-2 px-3 font-sans text-slate-200">
                    <span className="font-medium">{f.label}</span>
                    <span className="text-[10px] text-slate-500 block">{f.group}</span>
                  </td>
                  <td className="py-2 px-2 text-slate-300">
                    {f.value !== null ? `${f.value} ${f.unit}` : '—'}
                  </td>
                  <td
                    className={`py-2 px-2 text-right font-bold ${
                      f.shap > 0 ? 'text-red-400' : 'text-emerald-400'
                    }`}
                  >
                    {f.shap > 0 ? `+${f.shap.toFixed(4)}` : f.shap.toFixed(4)}
                  </td>
                  <td className="py-2 px-3 text-right text-slate-400">{f.pct.toFixed(1)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
