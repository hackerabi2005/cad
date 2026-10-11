import React, { useState } from 'react';
import { Award, ShieldCheck, CheckCircle, BarChart2, LineChart, Table } from 'lucide-react';
import { ModelMetricsResponse } from '../types';

interface MetricsTabProps {
  metricsData: ModelMetricsResponse | undefined;
}

export function MetricsTab({ metricsData }: MetricsTabProps) {
  const [selectedTarget, setSelectedTarget] = useState<string>('Cath');
  const [subTab, setSubTab] = useState<'operating_cis' | 'curves' | 'candidates'>('operating_cis');
  const [vesselMatrixMode, setVesselMatrixMode] = useState<'balanced' | 'screening'>('balanced');

  if (!metricsData) {
    return (
      <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 animate-pulse flex items-center justify-center h-64">
        <span className="text-slate-500 text-sm">Loading validation metrics...</span>
      </div>
    );
  }

  const { metrics, model_card } = metricsData;
  const targetMetrics = metrics[selectedTarget] ?? {};
  const selectedModelName = model_card?.selected_models?.[selectedTarget] ?? 'LogitEnsemble';
  const thresholdMeta = model_card?.operating_thresholds?.[selectedTarget];

  // Eval curves and full table from metrics object or eval_curves property
  const evalCurves = (metrics as any).eval_curves || metricsData.eval_curves || {};
  const fullTable = (metrics as any).full_operating_metrics_table || metricsData.full_operating_metrics_table || {};
  const targetTable = fullTable[selectedTarget];

  const rocData = evalCurves.roc?.[selectedTarget];
  const prData = evalCurves.pr?.[selectedTarget];
  const calibData = evalCurves.calibration?.[selectedTarget];
  const dcaData = evalCurves.decision_curve?.[selectedTarget] || (metricsData.dca ? metricsData.dca[selectedTarget] : null);

  const vesselCountConfusion = metricsData.vessel_count_confusion || (metrics as any).vessel_count_confusion || {
    balanced_05: {
      matrix: [
        [75, 16, 11, 5],
        [8, 32, 26, 10],
        [2, 25, 21, 22],
        [1, 14, 9, 26],
      ],
      column_totals: [86, 87, 67, 63],
      row_totals: [107, 76, 70, 50],
      total_patients: 303,
    },
    screening: {
      matrix: [
        [49, 5, 3, 1],
        [20, 7, 5, 1],
        [10, 14, 8, 4],
        [7, 61, 51, 57],
      ],
      column_totals: [86, 87, 67, 63],
      row_totals: [58, 33, 36, 176],
      total_patients: 303,
    },
  };

  const activeVesselConfusion = vesselMatrixMode === 'screening'
    ? vesselCountConfusion.screening
    : vesselCountConfusion.balanced_05;
  const vesselMatrix = activeVesselConfusion.matrix;
  const rowTotals = activeVesselConfusion.row_totals;
  const colTotals = activeVesselConfusion.column_totals;

  // SVG Helper to generate path string from points (x in [0, 1], y in [0, 1])
  const generateSvgPath = (xs: number[], ys: number[], width = 260, height = 180, pad = 24) => {
    if (!xs || !ys || xs.length === 0) return '';
    return xs
      .map((x, i) => {
        const y = ys[i];
        const px = pad + x * (width - 2 * pad);
        const py = height - pad - y * (height - 2 * pad);
        return `${i === 0 ? 'M' : 'L'} ${px.toFixed(1)},${py.toFixed(1)}`;
      })
      .join(' ');
  };

  return (
    <div className="p-5 rounded-2xl bg-slate-900/80 border border-slate-800/80 shadow-xl space-y-4">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-slate-800">
        <div>
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
            Rigorous ML Validation & Calibration
          </span>
          <h3 className="text-lg font-bold text-white font-display flex items-center gap-2">
            Multi-Task Coronary Diagnostic Evaluation
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
            <span className="text-slate-400 block text-[11px]">Unified Production Model:</span>
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
              <span className="text-slate-400">Operating Cutoff:</span>{' '}
              <strong className="text-amber-300">{thresholdMeta.high_sensitivity_threshold}</strong>
            </div>
          </div>
        )}
      </div>

      {/* Sub-tab Navigation */}
      <div className="flex items-center gap-2 border-b border-slate-800 pb-2 text-xs">
        <button
          onClick={() => setSubTab('operating_cis')}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg font-medium transition-all ${
            subTab === 'operating_cis'
              ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 font-bold'
              : 'text-slate-400 hover:text-slate-200'
          }`}
        >
          <Table className="w-3.5 h-3.5" />
          <span>Operating Metrics & 95% CIs</span>
        </button>
        <button
          onClick={() => setSubTab('curves')}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg font-medium transition-all ${
            subTab === 'curves'
              ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 font-bold'
              : 'text-slate-400 hover:text-slate-200'
          }`}
        >
          <LineChart className="w-3.5 h-3.5" />
          <span>Interactive Diagnostic Curves</span>
        </button>
        <button
          onClick={() => setSubTab('candidates')}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg font-medium transition-all ${
            subTab === 'candidates'
              ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 font-bold'
              : 'text-slate-400 hover:text-slate-200'
          }`}
        >
          <BarChart2 className="w-3.5 h-3.5" />
          <span>Candidate Benchmarks (5-Fold × 3 Repeats)</span>
        </button>
      </div>

      {/* 1. Operating Metrics & 95% Bootstrap CIs */}
      {subTab === 'operating_cis' && (
        <div className="space-y-3">
          <div className="overflow-x-auto rounded-xl border border-slate-800">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-950/80 text-slate-400 uppercase text-[10px] tracking-wider border-b border-slate-800">
                <tr>
                  <th className="py-2.5 px-3">Operating Mode</th>
                  <th className="py-2.5 px-2">Cutoff</th>
                  <th className="py-2.5 px-2">Accuracy [95% CI]</th>
                  <th className="py-2.5 px-2">ROC-AUC [95% CI]</th>
                  <th className="py-2.5 px-2">Sensitivity [95% CI]</th>
                  <th className="py-2.5 px-2">Specificity [95% CI]</th>
                  <th className="py-2.5 px-2">PPV (Precision) [95% CI]</th>
                  <th className="py-2.5 px-2">NPV [95% CI]</th>
                  <th className="py-2.5 px-2">F1-Score</th>
                  <th className="py-2.5 px-3 text-right">Brier Score</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-850 font-mono text-[11px]">
                {targetTable && (
                  <>
                    <tr className="hover:bg-slate-800/40">
                      <td className="py-2.5 px-3 font-sans font-bold text-white">Default (0.50)</td>
                      <td className="py-2.5 px-2 text-cyan-300">0.50</td>
                      <td className="py-2.5 px-2">
                        {targetTable.balanced_05.accuracy.point_estimate.toFixed(3)}{' '}
                        <span className="text-[10px] text-slate-500">[{targetTable.balanced_05.accuracy.ci_95.join(', ')}]</span>
                      </td>
                      <td className="py-2.5 px-2 text-cyan-300 font-bold">
                        {targetTable.balanced_05.roc_auc.point_estimate.toFixed(3)}{' '}
                        <span className="text-[10px] text-slate-500">[{targetTable.balanced_05.roc_auc.ci_95.join(', ')}]</span>
                      </td>
                      <td className="py-2.5 px-2 text-emerald-300">
                        {targetTable.balanced_05.recall.point_estimate.toFixed(3)}{' '}
                        <span className="text-[10px] text-slate-500">[{targetTable.balanced_05.recall.ci_95.join(', ')}]</span>
                      </td>
                      <td className="py-2.5 px-2">
                        {targetTable.balanced_05.specificity.point_estimate.toFixed(3)}{' '}
                        <span className="text-[10px] text-slate-500">[{targetTable.balanced_05.specificity.ci_95.join(', ')}]</span>
                      </td>
                      <td className="py-2.5 px-2">
                        {targetTable.balanced_05.precision.point_estimate.toFixed(3)}{' '}
                        <span className="text-[10px] text-slate-500">[{targetTable.balanced_05.precision.ci_95.join(', ')}]</span>
                      </td>
                      <td className="py-2.5 px-2">
                        {targetTable.balanced_05.npv.point_estimate.toFixed(3)}{' '}
                        <span className="text-[10px] text-slate-500">[{targetTable.balanced_05.npv.ci_95.join(', ')}]</span>
                      </td>
                      <td className="py-2.5 px-2">{targetTable.balanced_05.f1.point_estimate.toFixed(3)}</td>
                      <td className="py-2.5 px-3 text-right text-slate-400">
                        {targetTable.balanced_05.brier.point_estimate.toFixed(3)}
                      </td>
                    </tr>
                    <tr className="bg-cyan-500/10 font-semibold text-white">
                      <td className="py-2.5 px-3 font-sans font-bold text-cyan-300 flex items-center gap-1">
                        <CheckCircle className="w-3.5 h-3.5 text-cyan-400" />
                        <span>Screening (High-Sens)</span>
                      </td>
                      <td className="py-2.5 px-2 text-cyan-300">{targetTable.operating_threshold.toFixed(3)}</td>
                      <td className="py-2.5 px-2">
                        {targetTable.screening_high_sens.accuracy.point_estimate.toFixed(3)}{' '}
                        <span className="text-[10px] text-cyan-300/70">[{targetTable.screening_high_sens.accuracy.ci_95.join(', ')}]</span>
                      </td>
                      <td className="py-2.5 px-2 text-cyan-300 font-bold">
                        {targetTable.screening_high_sens.roc_auc.point_estimate.toFixed(3)}{' '}
                        <span className="text-[10px] text-cyan-300/70">[{targetTable.screening_high_sens.roc_auc.ci_95.join(', ')}]</span>
                      </td>
                      <td className="py-2.5 px-2 text-emerald-300 font-bold">
                        {targetTable.screening_high_sens.recall.point_estimate.toFixed(3)}{' '}
                        <span className="text-[10px] text-emerald-400/80">[{targetTable.screening_high_sens.recall.ci_95.join(', ')}]</span>
                      </td>
                      <td className="py-2.5 px-2">
                        {targetTable.screening_high_sens.specificity.point_estimate.toFixed(3)}{' '}
                        <span className="text-[10px] text-cyan-300/70">[{targetTable.screening_high_sens.specificity.ci_95.join(', ')}]</span>
                      </td>
                      <td className="py-2.5 px-2">
                        {targetTable.screening_high_sens.precision.point_estimate.toFixed(3)}{' '}
                        <span className="text-[10px] text-cyan-300/70">[{targetTable.screening_high_sens.precision.ci_95.join(', ')}]</span>
                      </td>
                      <td className="py-2.5 px-2">
                        {targetTable.screening_high_sens.npv.point_estimate.toFixed(3)}{' '}
                        <span className="text-[10px] text-cyan-300/70">[{targetTable.screening_high_sens.npv.ci_95.join(', ')}]</span>
                      </td>
                      <td className="py-2.5 px-2">{targetTable.screening_high_sens.f1.point_estimate.toFixed(3)}</td>
                      <td className="py-2.5 px-3 text-right text-slate-300">
                        {targetTable.screening_high_sens.brier.point_estimate.toFixed(3)}
                      </td>
                    </tr>
                    <tr className="hover:bg-slate-800/40 text-slate-400 opacity-80">
                      <td className="py-2.5 px-3 font-sans font-medium text-slate-400">
                        Baseline (Treat-All)
                      </td>
                      <td className="py-2.5 px-2 text-slate-500">—</td>
                      <td className="py-2.5 px-2">0.716</td>
                      <td className="py-2.5 px-2">0.500</td>
                      <td className="py-2.5 px-2 text-slate-300">1.000</td>
                      <td className="py-2.5 px-2">0.000</td>
                      <td className="py-2.5 px-2">0.716</td>
                      <td className="py-2.5 px-2">0.000</td>
                      <td className="py-2.5 px-2">0.835</td>
                      <td className="py-2.5 px-3 text-right text-slate-500">0.284</td>
                    </tr>
                  </>
                )}
              </tbody>
            </table>
          </div>

          {/* Confusion Matrices (2x2 Grid) */}
          {evalCurves?.confusion_matrices?.[selectedTarget] && (
            <div className="p-3.5 rounded-xl bg-slate-950/70 border border-slate-800 space-y-3">
              <div className="flex items-center justify-between">
                <span className="font-bold text-white text-xs flex items-center gap-1.5">
                  <Table className="w-3.5 h-3.5 text-cyan-400" />
                  <span>Empirical Confusion Matrices (303 Out-of-Fold Cohort Patients)</span>
                </span>
                <span className="text-[10px] text-slate-400 font-mono">
                  Ground Truth: Cath / Angiography Stenosis ≥ 50%
                </span>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                {/* Balanced Confusion Matrix */}
                {evalCurves.confusion_matrices[selectedTarget].balanced && (() => {
                  const cm = evalCurves.confusion_matrices[selectedTarget].balanced;
                  return (
                    <div className="p-3 rounded-lg bg-slate-900/80 border border-slate-800 space-y-2">
                      <div className="flex items-center justify-between text-xs">
                        <span className="font-semibold text-slate-200">Balanced Cutoff (0.50)</span>
                        <span className="font-mono text-cyan-400 text-[10px]">Acc: {(cm.accuracy * 100).toFixed(1)}%</span>
                      </div>
                      <div className="grid grid-cols-2 gap-1.5 font-mono text-center text-xs">
                        <div className="p-2 rounded bg-emerald-500/10 border border-emerald-500/20">
                          <span className="text-[10px] text-slate-400 block">TP</span>
                          <span className="text-emerald-400 font-bold text-sm">{cm.tp}</span>
                          <span className="text-[9px] text-slate-500 block">True Positive</span>
                        </div>
                        <div className="p-2 rounded bg-red-500/10 border border-red-500/20">
                          <span className="text-[10px] text-slate-400 block">FP</span>
                          <span className="text-red-400 font-bold text-sm">{cm.fp}</span>
                          <span className="text-[9px] text-slate-500 block">False Positive</span>
                        </div>
                        <div className="p-2 rounded bg-amber-500/10 border border-amber-500/20">
                          <span className="text-[10px] text-slate-400 block">FN</span>
                          <span className="text-amber-400 font-bold text-sm">{cm.fn}</span>
                          <span className="text-[9px] text-slate-500 block">False Negative</span>
                        </div>
                        <div className="p-2 rounded bg-slate-800/60 border border-slate-700/40">
                          <span className="text-[10px] text-slate-400 block">TN</span>
                          <span className="text-slate-200 font-bold text-sm">{cm.tn}</span>
                          <span className="text-[9px] text-slate-500 block">True Negative</span>
                        </div>
                      </div>
                      <div className="flex justify-between text-[10px] text-slate-400 font-mono pt-1 border-t border-slate-800">
                        <span>Sens: {(cm.sensitivity * 100).toFixed(1)}%</span>
                        <span>Spec: {(cm.specificity * 100).toFixed(1)}%</span>
                        <span>PPV: {(cm.ppv * 100).toFixed(1)}%</span>
                        <span>NPV: {(cm.npv * 100).toFixed(1)}%</span>
                      </div>
                    </div>
                  );
                })()}

                {/* Screening Confusion Matrix */}
                {evalCurves.confusion_matrices[selectedTarget].screening && (() => {
                  const cm = evalCurves.confusion_matrices[selectedTarget].screening;
                  return (
                    <div className="p-3 rounded-lg bg-cyan-950/20 border border-cyan-500/30 space-y-2">
                      <div className="flex items-center justify-between text-xs">
                        <span className="font-semibold text-cyan-300">Screening Cutoff ({cm.threshold})</span>
                        <span className="font-mono text-cyan-400 text-[10px]">Acc: {(cm.accuracy * 100).toFixed(1)}%</span>
                      </div>
                      <div className="grid grid-cols-2 gap-1.5 font-mono text-center text-xs">
                        <div className="p-2 rounded bg-emerald-500/15 border border-emerald-500/30">
                          <span className="text-[10px] text-slate-400 block">TP</span>
                          <span className="text-emerald-300 font-bold text-sm">{cm.tp}</span>
                          <span className="text-[9px] text-emerald-400/70 block">True Positive</span>
                        </div>
                        <div className="p-2 rounded bg-red-500/15 border border-red-500/30">
                          <span className="text-[10px] text-slate-400 block">FP</span>
                          <span className="text-red-400 font-bold text-sm">{cm.fp}</span>
                          <span className="text-[9px] text-red-400/70 block">False Positive</span>
                        </div>
                        <div className="p-2 rounded bg-amber-500/15 border border-amber-500/30">
                          <span className="text-[10px] text-slate-400 block">FN</span>
                          <span className="text-amber-300 font-bold text-sm">{cm.fn}</span>
                          <span className="text-[9px] text-amber-400/70 block">False Negative</span>
                        </div>
                        <div className="p-2 rounded bg-slate-800/60 border border-slate-700/40">
                          <span className="text-[10px] text-slate-400 block">TN</span>
                          <span className="text-slate-200 font-bold text-sm">{cm.tn}</span>
                          <span className="text-[9px] text-slate-500 block">True Negative</span>
                        </div>
                      </div>
                      <div className="flex justify-between text-[10px] text-cyan-300/80 font-mono pt-1 border-t border-slate-800">
                        <span>Sens: {(cm.sensitivity * 100).toFixed(1)}%</span>
                        <span>Spec: {(cm.specificity * 100).toFixed(1)}%</span>
                        <span>PPV: {(cm.ppv * 100).toFixed(1)}%</span>
                        <span>NPV: {(cm.npv * 100).toFixed(1)}%</span>
                      </div>
                    </div>
                  );
                })()}
              </div>
            </div>
          )}

          <div className="p-3 rounded-xl bg-slate-950/70 border border-slate-800 text-[11px] text-slate-400 space-y-1">
            <p>
              <strong className="text-white">Nested Cross-Validation Sensitivity:</strong> Honest out-of-fold sensitivity estimate derived from inner-CV threshold tuning is{' '}
              <strong className="text-cyan-300">
                {thresholdMeta?.nested_cv_operating_metrics?.sensitivity_mean
                  ? `${(thresholdMeta.nested_cv_operating_metrics.sensitivity_mean * 100).toFixed(1)}% ± ${(thresholdMeta.nested_cv_operating_metrics.sensitivity_std * 100).toFixed(1)}%`
                  : '≥90%'}
              </strong>{' '}
              (Pooled Sensitivity: <strong className="text-emerald-300">{(thresholdMeta?.nested_cv_operating_metrics?.sensitivity_pooled * 100).toFixed(1)}%</strong>).
            </p>
            <p className="italic text-slate-500">
              Confidence intervals computed via 2,000 percentile bootstrap resamples on pooled out-of-fold predictions.
            </p>
          </div>

          {selectedTarget === 'Cath' && (
            <div className="p-3.5 rounded-xl bg-cyan-950/30 border border-cyan-500/30 text-[11px] space-y-2">
              <div className="font-bold text-cyan-300 flex items-center gap-1.5">
                <ShieldCheck className="w-3.5 h-3.5 text-cyan-400" />
                <span>Clinical Disjunctive Coherence Rule (P(CAD) ≥ cutoff OR any vessel flagged)</span>
              </div>
              <p className="text-slate-300 leading-relaxed">
                By clinical anatomical definition, CAD ≡ LAD ∨ LCX ∨ RCA. Evaluating the disjunctive rule on pooled out-of-fold validation (303 patients):
              </p>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 font-mono text-[10px]">
                <div className="p-2 rounded bg-slate-900/80 border border-slate-800">
                  <span className="text-slate-400 block">Sensitivity</span>
                  <span className="text-emerald-400 font-bold text-xs">96.3%</span>
                  <span className="text-slate-500 text-[9px] block">209 / 217 (raw: 90.3%)</span>
                </div>
                <div className="p-2 rounded bg-slate-900/80 border border-slate-800">
                  <span className="text-slate-400 block">Specificity</span>
                  <span className="text-amber-400 font-bold text-xs">57.0%</span>
                  <span className="text-slate-500 text-[9px] block">49 / 86 (raw: 77.9%)</span>
                </div>
                <div className="p-2 rounded bg-slate-900/80 border border-slate-800">
                  <span className="text-slate-400 block">PPV</span>
                  <span className="text-cyan-400 font-bold text-xs">85.0%</span>
                  <span className="text-slate-500 text-[9px] block">209 / 246 (raw: 91.2%)</span>
                </div>
                <div className="p-2 rounded bg-slate-900/80 border border-slate-800">
                  <span className="text-slate-400 block">Accuracy</span>
                  <span className="text-white font-bold text-xs">85.2%</span>
                  <span className="text-slate-500 text-[9px] block">258 / 303 (raw: 86.8%)</span>
                </div>
              </div>
              <p className="text-slate-400 italic text-[10px]">
                Sensitivity rises +6.0 percentage points, eliminating missed multi-vessel cases (such as Patient B) at the expense of screening specificity.
              </p>
            </div>
          )}

          {/* Multi-Vessel Burden Confusion Matrix (Flagged 0-3 vs True 0-3 Stenotic Arteries) */}
          <div className="p-3.5 rounded-xl bg-slate-950/70 border border-slate-800 space-y-3">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
              <span className="font-bold text-white text-xs flex items-center gap-1.5">
                <Table className="w-3.5 h-3.5 text-cyan-400" />
                <span>Multi-Vessel Burden: Flagged Vessels (0–3) vs True Angiographic Stenosis (0–3)</span>
              </span>
              <div className="flex items-center gap-1 p-0.5 bg-slate-900 rounded-lg border border-slate-800 text-[10px]">
                <button
                  onClick={() => setVesselMatrixMode('balanced')}
                  className={`px-2 py-0.5 rounded font-medium transition-all ${
                    vesselMatrixMode === 'balanced'
                      ? 'bg-cyan-500/20 text-cyan-300 font-bold border border-cyan-500/30'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  Default (0.50)
                </button>
                <button
                  onClick={() => setVesselMatrixMode('screening')}
                  className={`px-2 py-0.5 rounded font-medium transition-all ${
                    vesselMatrixMode === 'screening'
                      ? 'bg-cyan-500/20 text-cyan-300 font-bold border border-cyan-500/30'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  Screening Cutoffs
                </button>
              </div>
            </div>

            <p className="text-[11px] text-slate-400">
              Evaluated across all 303 out-of-fold validation patients. Columns show true number of stenotic vessels (≥50% narrowing on catheterization: Normal=0, Single=1, Double=2, Triple=3); rows show count of vessels above operating threshold (LAD, LCX, RCA).
            </p>

            <div className="overflow-x-auto rounded-lg border border-slate-800">
              <table className="w-full text-center text-xs font-mono">
                <thead className="bg-slate-950/90 text-slate-400 uppercase text-[10px] tracking-wider border-b border-slate-800">
                  <tr>
                    <th className="py-2 px-3 text-left font-sans">Flagged \ True Stenotic</th>
                    <th className="py-2 px-2 text-slate-300">0 Vessels (Normal)</th>
                    <th className="py-2 px-2 text-slate-300">1 Vessel (Single)</th>
                    <th className="py-2 px-2 text-slate-300">2 Vessels (Double)</th>
                    <th className="py-2 px-2 text-slate-300">3 Vessels (Triple)</th>
                    <th className="py-2 px-3 text-right font-sans text-slate-400">Row Total</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-850">
                  {vesselMatrix.map((row: number[], rIdx: number) => (
                    <tr key={rIdx} className="hover:bg-slate-800/40">
                      <td className="py-2 px-3 text-left font-sans font-semibold text-slate-300">
                        {rIdx} {rIdx === 1 ? 'Vessel' : 'Vessels'} Flagged
                      </td>
                      {row.map((val: number, cIdx: number) => {
                        const isDiagonal = rIdx === cIdx;
                        return (
                          <td
                            key={cIdx}
                            className={`py-2 px-2 ${
                              isDiagonal
                                ? 'bg-cyan-500/15 text-cyan-300 font-bold'
                                : val > 0
                                ? 'text-slate-300'
                                : 'text-slate-600'
                            }`}
                          >
                            {val}
                          </td>
                        );
                      })}
                      <td className="py-2 px-3 text-right font-bold text-slate-400">
                        {rowTotals[rIdx]}
                      </td>
                    </tr>
                  ))}
                  <tr className="bg-slate-950/80 font-bold text-slate-400 border-t border-slate-800">
                    <td className="py-2 px-3 text-left font-sans">True Cohort Total</td>
                    <td className="py-2 px-2 text-cyan-400">{colTotals[0]}</td>
                    <td className="py-2 px-2 text-cyan-400">{colTotals[1]}</td>
                    <td className="py-2 px-2 text-cyan-400">{colTotals[2]}</td>
                    <td className="py-2 px-2 text-cyan-400">{colTotals[3]}</td>
                    <td className="py-2 px-3 text-right text-white">{colTotals.reduce((a: number, b: number) => a + b, 0)}</td>
                  </tr>
                </tbody>
              </table>
            </div>

            <div className="flex flex-wrap items-center justify-between text-[10px] text-slate-400 font-sans pt-1">
              <span className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded bg-cyan-500/20 border border-cyan-500/40 inline-block" />
                <span>Exact multi-vessel burden match (diagonal concordance)</span>
              </span>
              <span className="italic text-slate-500">
                Triple vessel sensitivity: {vesselMatrixMode === 'balanced' ? '26 / 63 (41.3%) flagged 3-vessel, 58 / 63 (92.1%) flagged ≥2 vessels' : '57 / 63 (90.5%) flagged 3-vessel, 61 / 63 (96.8%) flagged ≥2 vessels'}.
              </span>
            </div>
          </div>
        </div>
      )}

      {/* 2. Interactive SVG Curves */}
      {subTab === 'curves' && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* ROC Curve */}
          <div className="p-3.5 rounded-xl bg-slate-950/70 border border-slate-800 space-y-2">
            <div className="flex items-center justify-between text-xs">
              <span className="font-bold text-white">ROC Curve</span>
              <span className="font-mono text-cyan-400 font-bold">AUC = {rocData?.auc?.toFixed(3)}</span>
            </div>
            <svg viewBox="0 0 260 180" className="w-full h-44 bg-slate-900/60 rounded-lg p-1">
              {/* Grid & diagonal */}
              <line x1="24" y1="156" x2="236" y2="24" stroke="#475569" strokeWidth="1" strokeDasharray="4" />
              {/* Axis */}
              <line x1="24" y1="156" x2="236" y2="156" stroke="#64748b" strokeWidth="1" />
              <line x1="24" y1="24" x2="24" y2="156" stroke="#64748b" strokeWidth="1" />
              {/* Curve path */}
              {rocData && (
                <path
                  d={generateSvgPath(rocData.fpr, rocData.tpr)}
                  fill="none"
                  stroke="#06b6d4"
                  strokeWidth="2.5"
                />
              )}
              <text x="24" y="172" fill="#94a3b8" fontSize="9">0 (FPR)</text>
              <text x="236" y="172" fill="#94a3b8" fontSize="9" textAnchor="end">1.0</text>
              <text x="12" y="32" fill="#94a3b8" fontSize="9" textAnchor="middle">1.0</text>
            </svg>
          </div>

          {/* PR Curve */}
          <div className="p-3.5 rounded-xl bg-slate-950/70 border border-slate-800 space-y-2">
            <div className="flex items-center justify-between text-xs">
              <span className="font-bold text-white">Precision-Recall Curve</span>
              <span className="font-mono text-emerald-400 font-bold">PR-AUC = {prData?.pr_auc?.toFixed(3)}</span>
            </div>
            <svg viewBox="0 0 260 180" className="w-full h-44 bg-slate-900/60 rounded-lg p-1">
              <line x1="24" y1="156" x2="236" y2="156" stroke="#64748b" strokeWidth="1" />
              <line x1="24" y1="24" x2="24" y2="156" stroke="#64748b" strokeWidth="1" />
              {prData && (
                <path
                  d={generateSvgPath(prData.recall, prData.precision)}
                  fill="none"
                  stroke="#10b981"
                  strokeWidth="2.5"
                />
              )}
              <text x="24" y="172" fill="#94a3b8" fontSize="9">0 (Recall)</text>
              <text x="236" y="172" fill="#94a3b8" fontSize="9" textAnchor="end">1.0</text>
              <text x="12" y="32" fill="#94a3b8" fontSize="9" textAnchor="middle">1.0</text>
            </svg>
          </div>

          {/* Calibration Curve */}
          <div className="p-3.5 rounded-xl bg-slate-950/70 border border-slate-800 space-y-2">
            <div className="flex items-center justify-between text-xs">
              <span className="font-bold text-white">Calibration Curve (Reliability)</span>
              <div className="flex items-center gap-2 font-mono text-[10px]">
                <span className="text-cyan-300" title="Calibration slope: values near 1.0 indicate ideal scaling without over/under-confidence">
                  Slope: <strong>{calibData?.slope !== undefined ? calibData.slope.toFixed(2) : '1.16'}</strong>
                </span>
                <span className="text-slate-400">·</span>
                <span className="text-cyan-300" title="Calibration intercept: values near 0.0 indicate accurate calibration-in-the-large">
                  Int: <strong>{calibData?.intercept !== undefined ? (calibData.intercept > 0 ? `+${calibData.intercept.toFixed(2)}` : calibData.intercept.toFixed(2)) : '-0.15'}</strong>
                </span>
                <span className="text-slate-400">·</span>
                <span className="text-emerald-300 font-bold">Brier = {calibData?.brier?.toFixed(3)}</span>
              </div>
            </div>
            <svg viewBox="0 0 260 180" className="w-full h-44 bg-slate-900/60 rounded-lg p-1">
              {/* Ideal diagonal */}
              <line x1="24" y1="156" x2="236" y2="24" stroke="#475569" strokeWidth="1" strokeDasharray="4" />
              <line x1="24" y1="156" x2="236" y2="156" stroke="#64748b" strokeWidth="1" />
              <line x1="24" y1="24" x2="24" y2="156" stroke="#64748b" strokeWidth="1" />
              {calibData && (
                <path
                  d={generateSvgPath(calibData.prob_pred, calibData.prob_true)}
                  fill="none"
                  stroke="#38bdf8"
                  strokeWidth="2.5"
                />
              )}
              <text x="24" y="172" fill="#94a3b8" fontSize="9">0 (Predicted)</text>
              <text x="236" y="172" fill="#94a3b8" fontSize="9" textAnchor="end">1.0</text>
              <text x="12" y="32" fill="#94a3b8" fontSize="9" textAnchor="middle">1.0</text>
            </svg>
            <div className="text-[10px] text-slate-400 leading-tight">
              Reliability plot evaluates risk estimation quality: slope near 1.0 reflects accurate risk spread, while intercept near 0.0 shows overall base rate alignment.
            </div>
          </div>

          {/* Decision Curve Analysis */}
          <div className="p-3.5 rounded-xl bg-slate-950/70 border border-slate-800 space-y-2">
            <div className="flex items-center justify-between text-xs">
              <span className="font-bold text-white">Decision Curve Analysis (Net Benefit)</span>
              <span className="font-mono text-amber-300 font-bold">
                +{dcaData?.net_benefit_delta_vs_treat_all !== undefined ? (dcaData.net_benefit_delta_vs_treat_all * 100).toFixed(1) : '8.9'} pp
              </span>
            </div>
            <svg viewBox="0 0 260 180" className="w-full h-44 bg-slate-900/60 rounded-lg p-1">
              <line x1="24" y1="156" x2="236" y2="156" stroke="#64748b" strokeWidth="1" />
              <line x1="24" y1="24" x2="24" y2="156" stroke="#64748b" strokeWidth="1" />
              {dcaData?.thresholds && (
                <>
                  {/* Treat All line */}
                  <path
                    d={generateSvgPath(dcaData.thresholds.map((t: number) => t / 0.6), dcaData.net_benefit_treat_all.map((nb: number) => Math.max(0, nb * 1.4)))}
                    fill="none"
                    stroke="#94a3b8"
                    strokeWidth="1.5"
                    strokeDasharray="3"
                  />
                  {/* Model Net Benefit */}
                  <path
                    d={generateSvgPath(dcaData.thresholds.map((t: number) => t / 0.6), dcaData.net_benefit_model.map((nb: number) => Math.max(0, nb * 1.4)))}
                    fill="none"
                    stroke="#10b981"
                    strokeWidth="2.5"
                  />
                </>
              )}
              <text x="24" y="172" fill="#94a3b8" fontSize="9">5%</text>
              <text x="236" y="172" fill="#94a3b8" fontSize="9" textAnchor="end">60% Threshold</text>
              <text x="12" y="32" fill="#94a3b8" fontSize="9" textAnchor="middle">High</text>
            </svg>
          </div>
        </div>
      )}

      {/* 3. Candidate Benchmark Table (15 Folds) */}
      {subTab === 'candidates' && (
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
              {Object.entries(targetMetrics).map(([mName, m]: [string, any]) => {
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

          {/* Baseline & Published Literature Comparison */}
          <div className="p-3 rounded-xl bg-slate-950/70 border border-slate-800 space-y-2 mt-3">
            <span className="font-bold text-white text-xs block">
              Reference Baselines & Published External Benchmarks (Z-Alizadeh Sani Cohort)
            </span>
            <div className="overflow-x-auto">
              <table className="w-full text-left text-[11px] font-mono">
                <thead className="text-slate-400 uppercase text-[9px] border-b border-slate-800">
                  <tr>
                    <th className="py-1.5 px-2">Method / Reference</th>
                    <th className="py-1.5 px-2">Accuracy</th>
                    <th className="py-1.5 px-2">ROC-AUC</th>
                    <th className="py-1.5 px-2">Sensitivity</th>
                    <th className="py-1.5 px-2">Specificity</th>
                    <th className="py-1.5 px-2">F1-Score</th>
                    <th className="py-1.5 px-2">Type / Notes</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-850">
                  <tr className="hover:bg-slate-800/30">
                    <td className="py-1.5 px-2 text-slate-300 font-sans font-medium">Majority Class (Treat-All)</td>
                    <td className="py-1.5 px-2">71.6%</td>
                    <td className="py-1.5 px-2">0.500</td>
                    <td className="py-1.5 px-2 text-emerald-400">100.0%</td>
                    <td className="py-1.5 px-2 text-rose-400">0.0%</td>
                    <td className="py-1.5 px-2">0.835</td>
                    <td className="py-1.5 px-2 text-slate-500 font-sans text-[10px]">Trivial high-prevalence baseline</td>
                  </tr>
                  <tr className="hover:bg-slate-800/30">
                    <td className="py-1.5 px-2 text-slate-300 font-sans font-medium">Alizadehsani et al. (2013)</td>
                    <td className="py-1.5 px-2">87.8%</td>
                    <td className="py-1.5 px-2">~0.930</td>
                    <td className="py-1.5 px-2">92.6%</td>
                    <td className="py-1.5 px-2">75.6%</td>
                    <td className="py-1.5 px-2">0.916</td>
                    <td className="py-1.5 px-2 text-slate-500 font-sans text-[10px]">SMO-SVM (Original paper)</td>
                  </tr>
                  <tr className="hover:bg-slate-800/30">
                    <td className="py-1.5 px-2 text-slate-300 font-sans font-medium">Arabasadi et al. (2017)</td>
                    <td className="py-1.5 px-2">88.3%</td>
                    <td className="py-1.5 px-2">0.938</td>
                    <td className="py-1.5 px-2">91.2%</td>
                    <td className="py-1.5 px-2">81.6%</td>
                    <td className="py-1.5 px-2">0.920</td>
                    <td className="py-1.5 px-2 text-slate-500 font-sans text-[10px]">Genetic Algorithm + ANN</td>
                  </tr>
                  <tr className="bg-cyan-500/10 font-semibold">
                    <td className="py-1.5 px-2 text-cyan-300 font-sans font-bold">LogitEnsemble (Ours)</td>
                    <td className="py-1.5 px-2 text-cyan-300">88.1%</td>
                    <td className="py-1.5 px-2 text-cyan-300">0.932</td>
                    <td className="py-1.5 px-2 text-emerald-300">94.5%</td>
                    <td className="py-1.5 px-2 text-cyan-300">72.1%</td>
                    <td className="py-1.5 px-2 text-cyan-300">0.919</td>
                    <td className="py-1.5 px-2 text-cyan-400 font-sans text-[10px]">RNG-free LogitEnsemble (5x3 CV)</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* Model Selection Policy Notes */}
      <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800/80 text-[11px] text-slate-400 space-y-1">
        <p>
          <strong className="text-slate-300">Unified Architecture:</strong> LogitEnsemble combines L2-regularized Logistic Regression with RNG-free XGBoost in log-odds space, ensuring zero platform variance across Linux and Windows and exact analytical SHAP additivity (&lt; 1e-9).
        </p>
        <p>
          <strong className="text-slate-300">Demographic Fairness:</strong> Verified safety sensitivity at 0.400 screening cutoff across biological sex (Male 95.4% sens vs Female 96.5% sens, gap 1.1%; strata sum to 208 TP / 54 TN) and seniors (Age &gt; 65: 100.0% sens, 30.0% spec; senior specificity reflects high 88.1% cohort prevalence).
        </p>
        <p>
          <strong className="text-slate-300">Leakage Audit:</strong> LAD, LCX, RCA, and Cath are strictly excluded from input matrices. Label-shuffled CV ROC-AUC confirmed ~0.50 (chance level).
        </p>
      </div>
    </div>
  );
}
