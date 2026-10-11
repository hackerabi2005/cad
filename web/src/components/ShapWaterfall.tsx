import React, { useState } from 'react';
import {
  TrendingUp,
  TrendingDown,
  BarChart3,
  ListFilter,
  Layers,
  Activity,
  Droplet,
  Zap,
  Heart,
  ShieldAlert,
  HelpCircle,
  Stethoscope,
} from 'lucide-react';
import { TargetExplanation, FeatureContribution, SchemaRegistry } from '../types';

interface ShapWaterfallProps {
  explanations: Record<string, TargetExplanation> | undefined;
  activeTarget: string; // 'Cath' | 'LAD' | 'LCX' | 'RCA'
  onSelectTarget: (target: string) => void;
  schema?: SchemaRegistry;
  cadCoherenceInfo?: {
    adjusted: boolean;
    driving_vessel?: string | null;
    raw_prob?: number;
    displayed_prob?: number;
  };
}

function sigmoid(z: number): number {
  return 1.0 / (1.0 + Math.exp(-z));
}

export function ShapWaterfall({
  explanations,
  activeTarget,
  onSelectTarget,
  schema = {},
  cadCoherenceInfo,
}: ShapWaterfallProps) {
  const [viewMode, setViewMode] = useState<'chart' | 'table' | 'domains' | 'physiology'>('chart');
  const [unitMode, setUnitMode] = useState<'prob_path' | 'log_odds'>('prob_path');

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

  // Helper to format feature value into human-readable clinical string
  const formatFeatureValue = (f: FeatureContribution) => {
    if (f.value === null || f.value === undefined) return 'Assumed Default';
    if (f.value === 'N' || f.value === 0) return 'Absent / No';
    if (f.value === 'Y' || f.value === 1) return 'Present / Yes';
    if (typeof f.value === 'number') {
      return Number.isInteger(f.value) ? String(f.value) : f.value.toFixed(1);
    }
    return String(f.value);
  };

  // Cumulative probability path calculation
  const baseProb = sigmoid(exp.base_value);
  const finalProb = sigmoid(exp.raw_score);
  const totalMarginDiff = exp.raw_score - exp.base_value;
  const totalProbDeltaPp = (finalProb - baseProb) * 100;

  // Proportional percentage points rescaling: ensures feature attributions sum to P - P_0 exactly
  let runningMargin = exp.base_value;
  let runningProb = sigmoid(runningMargin);
  const probPathSteps = top10.map((f) => {
    const prevProb = runningProb;
    runningMargin += f.shap;
    runningProb = sigmoid(runningMargin);
    const deltaPp = Math.abs(totalMarginDiff) > 1e-6
      ? (f.shap / totalMarginDiff) * totalProbDeltaPp
      : 0;
    return {
      feature: f,
      prevProb,
      newProb: runningProb,
      deltaPp,
    };
  });

  // Identify top risk-elevating and risk-lowering drivers
  const topRiskElevating = exp.features
    .filter((f) => f.shap > 0)
    .slice(0, 3)
    .map((f) => `${f.label} [${formatFeatureValue(f)}]`);

  const topRiskLowering = exp.features
    .filter((f) => f.shap < 0)
    .slice(0, 2)
    .map((f) => `${f.label} [${formatFeatureValue(f)}]`);

  // Max absolute SHAP for bar scaling
  const maxAbsShap = Math.max(...exp.features.map((f) => Math.abs(f.shap)), 0.001);
  const maxAbsDeltaPp = Math.max(...probPathSteps.map((s) => Math.abs(s.deltaPp)), 0.1);

  // 5 Clinical Domain Groupings aligned directly with schema.json (group field)
  const domainDefinitions = [
    {
      id: 'demographic',
      name: 'Demographics & Vitals',
      icon: Heart,
      desc: 'Age, biological sex, BMI, weight, height',
    },
    {
      id: 'symptoms-exam',
      name: 'Symptoms & Physical Exam',
      icon: Activity,
      desc: 'Chest pain characteristics, cardiovascular history, murmurs, and functional status',
    },
    {
      id: 'lab',
      name: 'Laboratory & Blood Chemistry',
      icon: Droplet,
      desc: 'Lipids (LDL, HDL, TG, DLP), glucose (FBS), renal panel (CR, BUN), electrolytes, and hematology',
    },
    {
      id: 'ECG',
      name: 'Electrocardiogram (ECG)',
      icon: Zap,
      desc: 'ST deviations (St Elevation, St Depression), T-wave inversion, Q-wave, LVH, rhythm',
    },
    {
      id: 'echo',
      name: 'Cardiac Structure & Echo',
      icon: Stethoscope,
      desc: 'Left ventricular ejection fraction (EF-TTE), regional wall motion (RWMA), and valvular disease',
    },
  ];

  // Partition features directly into the 5 canonical schema domains (0 unassigned)
  const domainFeatureBuckets: Record<string, FeatureContribution[]> = {
    demographic: [],
    'symptoms-exam': [],
    lab: [],
    ECG: [],
    echo: [],
    other: [],
  };

  exp.features.forEach((f) => {
    const rawGrp = f.group || schema[f.feature]?.group || '';
    const normGrp = rawGrp.toLowerCase();
    if (normGrp === 'demographic' || normGrp === 'demographics') {
      domainFeatureBuckets.demographic.push(f);
    } else if (normGrp === 'symptoms-exam' || normGrp === 'symptoms_exam' || normGrp === 'symptoms') {
      domainFeatureBuckets['symptoms-exam'].push(f);
    } else if (normGrp === 'lab' || normGrp === 'labs') {
      domainFeatureBuckets.lab.push(f);
    } else if (normGrp === 'ecg') {
      domainFeatureBuckets.ECG.push(f);
    } else if (normGrp === 'echo') {
      domainFeatureBuckets.echo.push(f);
    } else {
      domainFeatureBuckets.other.push(f);
    }
  });

  const domainGroups = [
    ...domainDefinitions.map((d) => {
      const features = domainFeatureBuckets[d.id];
      const netShap = features.reduce((acc, f) => acc + f.shap, 0);
      const absSum = features.reduce((acc, f) => acc + Math.abs(f.shap), 0);
      return {
        ...d,
        features,
        netShap,
        absSum,
      };
    }),
    ...(domainFeatureBuckets.other.length > 0
      ? [
          {
            id: 'other',
            name: 'Other Clinical Factors',
            icon: Layers,
            desc: 'Additional biomarkers, inflammatory markers or extended variables',
            features: domainFeatureBuckets.other,
            netShap: domainFeatureBuckets.other.reduce((acc, f) => acc + f.shap, 0),
            absSum: domainFeatureBuckets.other.reduce((acc, f) => acc + Math.abs(f.shap), 0),
          },
        ]
      : []),
  ];

  // Features with clinical reference ranges for Physiology Panel
  const physiologyFeatures = exp.features
    .map((f) => {
      const meta = schema[f.feature];
      if (!meta || meta.ref_low === undefined || meta.ref_high === undefined) return null;
      const numVal = typeof f.value === 'number' ? f.value : null;
      let status: 'below' | 'normal' | 'above' = 'normal';
      if (numVal !== null) {
        if (numVal < meta.ref_low) status = 'below';
        else if (numVal > meta.ref_high) status = 'above';
      }
      return {
        feature: f.feature,
        label: f.label,
        value: f.value,
        unit: meta.unit || f.unit,
        ref_low: meta.ref_low,
        ref_high: meta.ref_high,
        source: meta.source || 'Standard Reference Range',
        reviewed: meta.reviewed ?? false,
        status,
        shap: f.shap,
        pct: f.pct,
        is_imputed: f.is_imputed,
      };
    })
    .filter(Boolean) as any[];

  return (
    <div className="p-5 rounded-2xl bg-slate-900/80 border border-slate-800/80 shadow-xl space-y-4">
      {/* Header with Title and Controls */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-slate-800">
        <div>
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
            Explainable AI (Additive SHAP)
          </span>
          <h3 className="text-base font-bold text-white font-display flex items-center gap-2">
            {targetTitles[activeTarget]}
          </h3>
        </div>

        {/* View Mode Tabs */}
        <div className="flex flex-wrap items-center gap-1 p-1 bg-slate-950/80 rounded-xl border border-slate-800 text-xs">
          <button
            onClick={() => setViewMode('chart')}
            className={`flex items-center gap-1 px-2.5 py-1 rounded-lg font-medium transition-all ${
              viewMode === 'chart'
                ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <BarChart3 className="w-3 h-3" />
            <span>Waterfall</span>
          </button>
          <button
            onClick={() => setViewMode('physiology')}
            className={`flex items-center gap-1 px-2.5 py-1 rounded-lg font-medium transition-all ${
              viewMode === 'physiology'
                ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Stethoscope className="w-3 h-3" />
            <span>Physiology</span>
          </button>
          <button
            onClick={() => setViewMode('domains')}
            className={`flex items-center gap-1 px-2.5 py-1 rounded-lg font-medium transition-all ${
              viewMode === 'domains'
                ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Layers className="w-3 h-3" />
            <span>Domains</span>
          </button>
          <button
            id="btn-shap-table-view"
            onClick={() => setViewMode('table')}
            className={`flex items-center gap-1 px-2.5 py-1 rounded-lg font-medium transition-all ${
              viewMode === 'table'
                ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <ListFilter className="w-3 h-3" />
            <span>All 54</span>
          </button>
        </div>
      </div>

      {/* Target Selector Buttons */}
      <div className="flex items-center gap-2 overflow-x-auto pb-1 text-xs">
        <span className="text-slate-500 font-medium text-[11px] whitespace-nowrap">Focus:</span>
        {(['Cath', 'LAD', 'LCX', 'RCA'] as const).map((t) => (
          <button
            key={t}
            onClick={() => onSelectTarget(t)}
            className={`px-3 py-1 rounded-lg font-semibold transition-all ${
              activeTarget === t
                ? 'bg-cyan-500/25 text-cyan-300 border border-cyan-500/50 shadow-sm font-bold'
                : 'bg-slate-950/60 text-slate-400 border border-slate-800 hover:bg-slate-800/60'
            }`}
          >
            {t === 'Cath' ? 'Overall CAD' : t}
          </button>
        ))}
      </div>

      {/* 1. Waterfall Chart View */}
      {viewMode === 'chart' && (
        <div className="space-y-3">
          {/* Plain-Language Clinical Summary Banner */}
          <div className="p-3 rounded-xl bg-slate-950/80 border border-slate-800 text-xs">
            <span className="text-[10px] text-slate-400 font-semibold uppercase tracking-wider block mb-1">
              Plain-Language Clinical Explanation:
            </span>
            <p className="text-slate-200 leading-relaxed font-sans">
              Driven mainly by{' '}
              <strong className="text-red-300">
                {topRiskElevating.length > 0 ? topRiskElevating.join(', ') : 'baseline clinical profile'}
              </strong>
              {topRiskLowering.length > 0 && (
                <>
                  ; partly offset by{' '}
                  <strong className="text-emerald-300">{topRiskLowering.join(', ')}</strong>
                </>
              )}
              .
            </p>
          </div>

          {/* Coherence Adjustment Callout */}
          {activeTarget === 'Cath' && cadCoherenceInfo?.adjusted && (
            <div className="p-3 rounded-xl bg-amber-500/15 border border-amber-500/30 text-amber-200 text-xs flex items-start gap-2.5">
              <ShieldAlert className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" />
              <div>
                <span className="font-bold text-amber-300 block text-xs">
                  Adjusted to match {cadCoherenceInfo.driving_vessel ?? 'vessel'}
                </span>
                <span className="text-[11px] text-slate-300 leading-relaxed">
                  Overall CAD risk was elevated to{' '}
                  <strong className="text-white">{Math.round((cadCoherenceInfo.displayed_prob ?? 0) * 100)}%</strong>{' '}
                  (raw Cath model: {Math.round((cadCoherenceInfo.raw_prob ?? 0) * 100)}%) to maintain anatomical consistency with the flagged {cadCoherenceInfo.driving_vessel} model. The waterfall below details the raw systemic Cath model's features.
                </span>
              </div>
            </div>
          )}

          {/* Unit Toggle: Cumulative Probability Path vs Log-Odds */}
          <div className="flex items-center justify-between p-2 rounded-xl bg-slate-950/70 border border-slate-800 text-[11px]">
            <div className="flex items-center gap-2">
              <span className="text-slate-400">Attribution Unit:</span>
              <div className="flex items-center gap-1 p-0.5 bg-slate-900 rounded-lg border border-slate-800">
                <button
                  onClick={() => setUnitMode('prob_path')}
                  className={`px-2 py-0.5 rounded text-[10px] font-medium transition-all ${
                    unitMode === 'prob_path'
                      ? 'bg-cyan-500/25 text-cyan-300 font-bold border border-cyan-500/40'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  Probability Path (pp)
                </button>
                <button
                  onClick={() => setUnitMode('log_odds')}
                  className={`px-2 py-0.5 rounded text-[10px] font-medium transition-all ${
                    unitMode === 'log_odds'
                      ? 'bg-cyan-500/25 text-cyan-300 font-bold border border-cyan-500/40'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  Log-Odds Margin
                </button>
              </div>
            </div>

            <div className="font-mono text-slate-300 text-[10px]">
              {unitMode === 'prob_path' ? (
                <span>
                  Base: <strong className="text-cyan-400">{(baseProb * 100).toFixed(1)}%</strong> &rarr; Final:{' '}
                  <strong className="text-emerald-400">{(finalProb * 100).toFixed(1)}%</strong>
                </span>
              ) : (
                <span>
                  Base Margin: <strong className="text-cyan-400">{exp.base_value.toFixed(2)}</strong> &rarr; Score:{' '}
                  <strong className="text-emerald-400">{exp.raw_score.toFixed(2)}</strong>
                </span>
              )}
            </div>
          </div>

          {/* Feature List */}
          <div className="space-y-2 pt-1 max-h-[340px] overflow-y-auto pr-1">
            {unitMode === 'prob_path'
              ? probPathSteps.map(({ feature: f, prevProb, newProb, deltaPp }) => {
                  const isRisk = deltaPp > 0;
                  const barWidth = Math.min((Math.abs(deltaPp) / maxAbsDeltaPp) * 100, 100);
                  const valStr = f.is_imputed ? 'cohort median' : `${f.value !== null ? f.value : 'N/A'}${f.unit ? ' ' + f.unit : ''}`;
                  return (
                    <div key={f.feature} className="group text-xs">
                      <div className="flex items-center justify-between mb-0.5">
                        <div className="flex items-center gap-1.5 truncate max-w-[70%]">
                          {isRisk ? (
                            <TrendingUp className="w-3.5 h-3.5 text-red-400 flex-shrink-0" />
                          ) : (
                            <TrendingDown className="w-3.5 h-3.5 text-emerald-400 flex-shrink-0" />
                          )}
                          <span className="text-slate-200 font-medium truncate" title={`${f.label} = ${valStr}`}>
                            <span className="text-white font-semibold">{f.label}</span>
                            <span className="text-slate-400 font-mono text-[11px]"> = {valStr}</span>
                          </span>
                          {f.is_imputed && (
                            <span className="text-[9px] px-1.5 py-0.2 rounded bg-slate-800 text-amber-300/80 border border-slate-700 font-mono">
                              cohort median
                            </span>
                          )}
                        </div>

                        <div className="flex items-center gap-1.5 font-mono text-[11px]">
                          <span className="text-slate-400">&rarr;</span>
                          <span className={isRisk ? 'text-red-400 font-bold' : 'text-emerald-400 font-bold'}>
                            {deltaPp > 0 ? `+${deltaPp.toFixed(1)} pp` : `${deltaPp.toFixed(1)} pp`}
                          </span>
                          <span className="text-slate-500 text-[10px]">
                            ({(newProb * 100).toFixed(0)}%)
                          </span>
                        </div>
                      </div>

                      {/* Running Diverging Bar */}
                      <div className="w-full bg-slate-950 h-2 rounded-full overflow-hidden flex relative">
                        <div className="absolute left-1/2 top-0 bottom-0 w-px bg-slate-700 z-10" />
                        {isRisk ? (
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
                })
              : top10.map((f) => {
                  const isRisk = f.shap > 0;
                  const barWidth = Math.min((Math.abs(f.shap) / maxAbsShap) * 100, 100);
                  const valStr = f.is_imputed ? 'cohort median' : `${f.value !== null ? f.value : 'N/A'}${f.unit ? ' ' + f.unit : ''}`;
                  return (
                    <div key={f.feature} className="group text-xs">
                      <div className="flex items-center justify-between mb-0.5">
                        <div className="flex items-center gap-1.5 truncate max-w-[70%]">
                          {isRisk ? (
                            <TrendingUp className="w-3.5 h-3.5 text-red-400 flex-shrink-0" />
                          ) : (
                            <TrendingDown className="w-3.5 h-3.5 text-emerald-400 flex-shrink-0" />
                          )}
                          <span className="text-slate-200 font-medium truncate" title={`${f.label} = ${valStr}`}>
                            <span className="text-white font-semibold">{f.label}</span>
                            <span className="text-slate-400 font-mono text-[11px]"> = {valStr}</span>
                          </span>
                          {f.is_imputed && (
                            <span className="text-[9px] px-1.5 py-0.2 rounded bg-slate-800 text-amber-300/80 border border-slate-700 font-mono">
                              cohort median
                            </span>
                          )}
                        </div>

                        <div className="flex items-center gap-1.5 font-mono text-[11px]">
                          <span className="text-slate-400">&rarr;</span>
                          <span className={isRisk ? 'text-red-400 font-bold' : 'text-emerald-400 font-bold'}>
                            {f.shap > 0 ? `+${f.shap.toFixed(3)}` : f.shap.toFixed(3)}
                          </span>
                          <span className="text-slate-500 text-[10px] w-9 text-right">{f.pct.toFixed(0)}%</span>
                        </div>
                      </div>

                      {/* Log-Odds Bar */}
                      <div className="w-full bg-slate-950 h-2 rounded-full overflow-hidden flex relative">
                        <div className="absolute left-1/2 top-0 bottom-0 w-px bg-slate-700 z-10" />
                        {isRisk ? (
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

            {activeTarget === 'Cath' && cadCoherenceInfo?.adjusted && (
              <div className="pt-2 mt-1 border-t border-amber-500/30 text-xs bg-amber-500/10 p-2 rounded-lg">
                <div className="flex items-center justify-between text-amber-300">
                  <span className="font-semibold flex items-center gap-1.5">
                    <ShieldAlert className="w-3.5 h-3.5 text-amber-400" />
                    Raised to match {cadCoherenceInfo.driving_vessel ?? 'vessel'} by max() coherence rule
                  </span>
                  <span className="font-mono text-[11px] font-bold">
                    {Math.round((cadCoherenceInfo.raw_prob ?? 0) * 100)}% &rarr; {Math.round((cadCoherenceInfo.displayed_prob ?? 0) * 100)}%
                  </span>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* 2. Clinical Physiology Panel (D2) */}
      {viewMode === 'physiology' && (
        <div className="space-y-2.5 max-h-[380px] overflow-y-auto pr-1">
          <div className="flex items-center justify-between p-2 rounded-xl bg-slate-950/70 border border-slate-800 text-[11px]">
            <span className="text-slate-300">Clinical Reference Ranges & Status Flags</span>
            <span className="px-2 py-0.5 rounded-full bg-amber-500/20 text-amber-300 border border-amber-500/40 text-[10px] font-semibold">
              Pending clinical review
            </span>
          </div>

          <div className="overflow-x-auto rounded-xl border border-slate-800">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-950/80 text-slate-400 uppercase text-[10px] tracking-wider border-b border-slate-800">
                <tr>
                  <th className="py-2 px-3">Biomarker</th>
                  <th className="py-2 px-2">Patient Value</th>
                  <th className="py-2 px-2">Reference Range</th>
                  <th className="py-2 px-2">Status</th>
                  <th className="py-2 px-3 text-right">Risk Impact</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 font-mono text-[11px]">
                {physiologyFeatures.map((pf) => {
                  const isRisk = pf.shap > 0;
                  return (
                    <tr key={pf.feature} className="hover:bg-slate-800/30">
                      <td className="py-2 px-3 font-sans">
                        <span className="font-medium text-slate-200">{pf.label}</span>
                        <span className="text-[10px] text-slate-500 block">{pf.source}</span>
                      </td>
                      <td className="py-2 px-2 text-white font-bold">
                        {pf.value !== null ? `${pf.value} ${pf.unit}` : '—'}
                      </td>
                      <td className="py-2 px-2 text-slate-400">
                        {pf.ref_low} – {pf.ref_high} {pf.unit}
                      </td>
                      <td className="py-2 px-2 font-sans">
                        {pf.status === 'below' && (
                          <span className="px-1.5 py-0.5 rounded bg-cyan-500/20 text-cyan-300 border border-cyan-500/30 text-[10px]">
                            Below Normal
                          </span>
                        )}
                        {pf.status === 'normal' && (
                          <span className="px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 text-[10px]">
                            Normal
                          </span>
                        )}
                        {pf.status === 'above' && (
                          <span className="px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-300 border border-amber-500/30 text-[10px]">
                            Above Normal
                          </span>
                        )}
                      </td>
                      <td className="py-2 px-3 text-right">
                        <span className={isRisk ? 'text-red-400 font-bold' : 'text-emerald-400 font-bold'}>
                          {pf.shap > 0 ? `+${(pf.pct).toFixed(0)}%` : `-${(pf.pct).toFixed(0)}%`}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* 3. Physiological Factor Breakdown (Domains) View */}
      {viewMode === 'domains' && (
        <div className="space-y-3 pt-1 max-h-[380px] overflow-y-auto pr-1">
          {domainGroups.map((dg) => {
            const Icon = dg.icon;
            const isNetRisk = dg.netShap > 0;
            return (
              <div
                key={dg.id}
                className="p-3 rounded-xl bg-slate-950/70 border border-slate-800/80 space-y-2 hover:border-slate-700 transition-colors"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <div className="p-1.5 rounded-lg bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
                      <Icon className="w-4 h-4" />
                    </div>
                    <div>
                      <h4 className="font-bold text-white text-xs">{dg.name}</h4>
                      <p className="text-[10px] text-slate-400">{dg.desc}</p>
                    </div>
                  </div>

                  <div className="text-right font-mono">
                    <span
                      className={`text-xs font-bold ${
                        isNetRisk ? 'text-red-400' : dg.netShap < 0 ? 'text-emerald-400' : 'text-slate-400'
                      }`}
                    >
                      {dg.netShap > 0 ? `+${dg.netShap.toFixed(3)}` : dg.netShap.toFixed(3)}
                    </span>
                    <span className="text-[10px] text-slate-500 block">Net impact</span>
                  </div>
                </div>

                <div className="flex flex-wrap gap-1.5 pt-1">
                  {dg.features.slice(0, 5).map((f) => (
                    <span
                      key={f.feature}
                      className="text-[11px] px-2 py-0.5 rounded bg-slate-900 border border-slate-800 text-slate-300"
                    >
                      {f.label}: <strong className="font-mono">{formatFeatureValue(f)}</strong>
                    </span>
                  ))}
                  {dg.features.length > 5 && (
                    <span className="text-[11px] px-1.5 py-0.5 rounded bg-slate-900/50 text-slate-500">
                      +{dg.features.length - 5} more
                    </span>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* 4. Complete Table View (All 54 Features) */}
      {viewMode === 'table' && (
        <div className="overflow-x-auto rounded-xl border border-slate-800 max-h-[380px] overflow-y-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-950/80 text-slate-400 uppercase text-[10px] tracking-wider border-b border-slate-800 sticky top-0">
              <tr>
                <th className="py-2.5 px-3">Rank & Feature</th>
                <th className="py-2.5 px-2">Patient Value</th>
                <th className="py-2.5 px-2">Margin (SHAP)</th>
                <th className="py-2.5 px-3 text-right">Relative Impact</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-850 font-mono text-[11px]">
              {exp.features.map((f, idx) => {
                const isRisk = f.shap > 0;
                return (
                  <tr key={f.feature} className="hover:bg-slate-800/40 transition-colors">
                    <td className="py-2 px-3 font-sans flex items-center gap-1.5">
                      <span className="text-slate-500 text-[10px] w-4">{idx + 1}.</span>
                      <span className="font-medium text-slate-200">{f.label}</span>
                      {f.is_imputed && (
                        <span className="text-[9px] px-1 py-0.2 rounded bg-slate-800 text-amber-300 border border-slate-700">
                          cohort median
                        </span>
                      )}
                    </td>
                    <td className="py-2 px-2 text-slate-300">
                      {f.is_imputed ? 'cohort median' : f.value !== null ? `${f.value} ${f.unit}` : '—'}
                    </td>
                    <td className="py-2 px-2">
                      <span className={isRisk ? 'text-red-400 font-bold' : 'text-emerald-400 font-bold'}>
                        {f.shap > 0 ? `+${f.shap.toFixed(4)}` : f.shap.toFixed(4)}
                      </span>
                    </td>
                    <td className="py-2 px-3 text-right text-slate-300">{f.pct.toFixed(1)}%</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* Clinical Narrative Summary */}
      <div className="p-3.5 rounded-xl bg-slate-950/60 border border-slate-800/80 text-xs text-slate-300 space-y-1.5">
        <div className="flex items-center gap-1.5 font-bold text-white text-[11px] uppercase tracking-wider">
          <span>Key Findings for {targetTitles[activeTarget]}</span>
        </div>
        <p className="leading-relaxed text-[11px]">
          {topRiskElevating.length > 0 && (
            <>
              <strong className="text-red-400 font-medium">Primary Risk Drivers:</strong>{' '}
              {topRiskElevating.join(', ')}.
            </>
          )}{' '}
          {topRiskLowering.length > 0 && (
            <>
              <strong className="text-emerald-400 font-medium">Mitigating Factors:</strong>{' '}
              {topRiskLowering.join(', ')}.
            </>
          )}
        </p>
        <p className="text-[10px] text-slate-500 italic pt-1 border-t border-slate-800/60">
          Note: Feature attributions reflect cohort statistical patterns (associational attribution, not causal mechanisms).
        </p>
      </div>
    </div>
  );
}
