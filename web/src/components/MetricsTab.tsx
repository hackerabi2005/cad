import React, { useState } from 'react';
import { Award, ShieldCheck, CheckCircle, BarChart2 } from 'lucide-react';
import { ModelMetricsResponse } from '../types';

interface MetricsTabProps {
  metricsData: ModelMetricsResponse | undefined;
}

export function MetricsTab({ metricsData }: MetricsTabProps) {
  const [selectedTarget, setSelectedTarget] = useState<string>('Cath');

  if (!metricsData) {
    return (
      <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 animate-pulse flex items-center justify-center h-64">
        <span className="text-slate-500 text-sm">Loading validation metrics...</span>
      </div>
    );
  }

  const { metrics, model_card } = metricsData;
  const targetMetrics = metrics[selectedTarget] ?? {};
  const selectedModelName = model_card?.selected_models?.[selectedTarget] ?? 'LogisticRegression';
  const thresholdMeta = model_card?.operating_thresholds?.[selectedTarget];

  return (
    <div className="p-5 rounded-2xl bg-slate-900/80 border border-slate-800/80 shadow-xl space-y-4">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-slate-800">
        <div>
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
            Rigorous ML Validation
          </span>
          <h3 className="text-lg font-bold text-white font-display flex items-center gap-2">
            Repeated Stratified 5-Fold Cross-Validation (15 Folds)
          </h3>
        </div>

        {/* Target Buttons */}
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

      {/* Model Selected Highlight Card */}
      <div className="p-3.5 rounded-xl bg-gradient-to-r from-cyan-950/40 to-slate-900/60 border border-cyan-500/30 text-xs flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div className="flex items-center gap-2.5">
          <Award className="w-5 h-5 text-cyan-400 flex-shrink-0" />
          <div>
            <span className="text-slate-400 block text-[11px]">Production Model Chosen:</span>
            <span className="font-bold text-white text-sm font-display">{selectedModelName}</span>
          </div>
        </div>

        {thresholdMeta && (
          <div className="flex flex-wrap items-center gap-3 font-mono text-[11px]">
            <div className="bg-slate-900/80 px-2.5 py-1 rounded-lg border border-slate-800">
              <span className="text-slate-400">CV ROC-AUC:</span>{' '}
              <strong className="text-cyan-300">{thresholdMeta.cv_roc_auc?.toFixed(3)}</strong>
            </div>
            <div className="bg-slate-900/80 px-2.5 py-1 rounded-lg border border-slate-800">
              <span className="text-slate-400">Brier Score:</span>{' '}
              <strong className="text-emerald-300">{thresholdMeta.cv_brier?.toFixed(3)}</strong>
            </div>
            <div className="bg-slate-900/80 px-2.5 py-1 rounded-lg border border-slate-800">
              <span className="text-slate-400">High-Sens Thresh:</span>{' '}
              <strong className="text-amber-300">{thresholdMeta.high_sensitivity_threshold}</strong>
            </div>
          </div>
        )}
      </div>

      {/* Benchmarking Table */}
      <div className="overflow-x-auto rounded-xl border border-slate-800 max-h-[320px] overflow-y-auto">
        <table className="w-full text-left text-xs">
          <thead className="bg-slate-950/80 text-slate-400 uppercase text-[10px] tracking-wider border-b border-slate-800 sticky top-0">
            <tr>
              <th className="py-2.5 px-3">Candidate Model</th>
              <th className="py-2.5 px-2">ROC-AUC (Mean ± SD)</th>
              <th className="py-2.5 px-2">PR-AUC</th>
              <th className="py-2.5 px-2">F1-Score</th>
              <th className="py-2.5 px-2">Recall (Sens.)</th>
              <th className="py-2.5 px-2">Specificity</th>
              <th className="py-2.5 px-3">Brier Score</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-850 font-mono">
            {Object.entries(targetMetrics).map(([mName, m]) => {
              const isSelected = mName === selectedModelName;
              return (
                <tr
                  key={mName}
                  className={`transition-colors ${
                    isSelected ? 'bg-cyan-500/10 font-semibold text-white' : 'hover:bg-slate-800/40 text-slate-300'
                  }`}
                >
                  <td className="py-2.5 px-3 font-sans flex items-center gap-1.5">
                    {isSelected && <CheckCircle className="w-3.5 h-3.5 text-cyan-400" />}
                    <span>{mName}</span>
                  </td>
                  <td className="py-2.5 px-2 text-cyan-300">
                    {m.roc_auc_mean?.toFixed(3)} ± {m.roc_auc_std?.toFixed(3)}
                  </td>
                  <td className="py-2.5 px-2">{m.pr_auc_mean?.toFixed(3)}</td>
                  <td className="py-2.5 px-2">{m.f1_mean?.toFixed(3)}</td>
                  <td className="py-2.5 px-2 text-emerald-300">{m.recall_mean?.toFixed(3)}</td>
                  <td className="py-2.5 px-2">{m.specificity_mean?.toFixed(3)}</td>
                  <td className="py-2.5 px-3 text-slate-400">{m.brier_mean?.toFixed(3)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Model Selection Policy Notes */}
      <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800/80 text-[11px] text-slate-400 space-y-1">
        <p>
          <strong className="text-slate-300">Selection Rule:</strong> Simplest model within 1 Standard Error of maximum CV ROC-AUC, prioritizing linear/tree models with analytical, exact SHAP additivity.
        </p>
        <p>
          <strong className="text-slate-300">Leakage Audit:</strong> LAD, LCX, RCA, and Cath are strictly excluded from input matrices. Label-shuffled CV ROC-AUC confirmed ~0.50 (chance level).
        </p>
      </div>
    </div>
  );
}
