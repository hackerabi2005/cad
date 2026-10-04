import React, { useState } from 'react';
import { Layers, Globe, Filter } from 'lucide-react';
import { GlobalShapItem } from '../types';

interface GlobalImportanceProps {
  globalShap: Record<string, GlobalShapItem[]> | undefined;
}

export function GlobalImportance({ globalShap }: GlobalImportanceProps) {
  const [selectedTarget, setSelectedTarget] = useState<string>('Cath');
  const [selectedGroup, setSelectedGroup] = useState<string>('all');

  if (!globalShap) {
    return (
      <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 animate-pulse flex items-center justify-center h-64">
        <span className="text-slate-500 text-sm">Loading global SHAP importances...</span>
      </div>
    );
  }

  const items = globalShap[selectedTarget] ?? [];
  const filtered =
    selectedGroup === 'all'
      ? items
      : items.filter((item) => item.group.toLowerCase() === selectedGroup.toLowerCase());

  const groups = [
    { id: 'all', label: 'All Categories' },
    { id: 'symptoms-exam', label: 'Symptoms & Vitals' },
    { id: 'ECG', label: 'ECG' },
    { id: 'lab', label: 'Lab Tests' },
    { id: 'echo', label: 'Echo' },
    { id: 'demographic', label: 'Demographics' },
  ];

  return (
    <div className="p-5 rounded-2xl bg-slate-900/80 border border-slate-800/80 shadow-xl space-y-4">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-slate-800">
        <div>
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
            Population-Level Insights
          </span>
          <h3 className="text-lg font-bold text-white font-display flex items-center gap-2">
            Global Feature Importance (Mean |SHAP|)
          </h3>
        </div>

        {/* Target Selector */}
        <div className="flex items-center gap-1 p-1 bg-slate-950/80 rounded-xl border border-slate-800">
          {(['Cath', 'LAD', 'LCX', 'RCA'] as const).map((t) => (
            <button
              key={t}
              onClick={() => setSelectedTarget(t)}
              className={`px-2.5 py-1 rounded-lg text-xs font-semibold transition-all ${
                selectedTarget === t
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              {t === 'Cath' ? 'Overall CAD' : t}
            </button>
          ))}
        </div>
      </div>

      {/* Group Category Filter Pills */}
      <div className="flex flex-wrap gap-1.5 text-xs">
        {groups.map((g) => (
          <button
            key={g.id}
            onClick={() => setSelectedGroup(g.id)}
            className={`px-2.5 py-1 rounded-lg transition-all ${
              selectedGroup === g.id
                ? 'bg-slate-700 text-white font-semibold border border-slate-600'
                : 'bg-slate-950/60 text-slate-400 border border-slate-800 hover:text-slate-200'
            }`}
          >
            {g.label}
          </button>
        ))}
      </div>

      {/* Ranked Importance List */}
      <div className="space-y-2.5 max-h-[380px] overflow-y-auto pr-1">
        {filtered.slice(0, 15).map((item, idx) => {
          return (
            <div key={item.feature} className="p-2.5 rounded-xl bg-slate-950/60 border border-slate-800/80 text-xs">
              <div className="flex items-center justify-between mb-1.5">
                <div className="flex items-center gap-2">
                  <span className="w-5 h-5 rounded-full bg-slate-800 text-slate-300 text-[10px] font-mono flex items-center justify-center font-bold">
                    {idx + 1}
                  </span>
                  <span className="font-semibold text-slate-200">{item.label}</span>
                  <span className="text-[10px] text-slate-500 uppercase tracking-wider font-mono">
                    [{item.group}]
                  </span>
                </div>

                <div className="flex items-center gap-2 font-mono">
                  <span className="text-cyan-400 font-bold">{item.mean_abs_shap.toFixed(4)}</span>
                  <span className="text-slate-500 text-[10px]">({item.importance_pct}%)</span>
                </div>
              </div>

              {/* Progress bar */}
              <div className="w-full bg-slate-900 h-1.5 rounded-full overflow-hidden">
                <div
                  className="h-full rounded-full bg-gradient-to-r from-cyan-500 to-blue-500"
                  style={{ width: `${Math.min(100, item.importance_pct * 4)}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
