import React, { useState } from 'react';
import { UserCheck, Sparkles, Sliders, RefreshCw, ChevronDown, ChevronUp } from 'lucide-react';
import { SchemaRegistry, SamplePatient } from '../types';

interface PatientFormProps {
  schema: SchemaRegistry;
  patientData: Record<string, any>;
  samples: SamplePatient[];
  onChangeField: (field: string, value: any) => void;
  onLoadSample: (sample: SamplePatient) => void;
  onReset: () => void;
  cohortDefaults: string[];
}

export function PatientForm({
  schema,
  patientData,
  samples,
  onChangeField,
  onLoadSample,
  onReset,
  cohortDefaults,
}: PatientFormProps) {
  const [activeGroup, setActiveGroup] = useState<string>('demographic');

  const groups = [
    { id: 'demographic', label: 'Demographics' },
    { id: 'symptoms-exam', label: 'Vitals & Symptoms' },
    { id: 'ECG', label: 'Electrocardiogram (ECG)' },
    { id: 'lab', label: 'Laboratory Blood' },
    { id: 'echo', label: 'Echocardiography' },
  ];

  // Group features by category
  const groupedFeatures: Record<string, string[]> = {
    demographic: [],
    'symptoms-exam': [],
    ECG: [],
    lab: [],
    echo: [],
  };

  Object.entries(schema).forEach(([col, meta]) => {
    if (groupedFeatures[meta.group]) {
      groupedFeatures[meta.group].push(col);
    }
  });

  return (
    <div className="p-5 rounded-2xl bg-slate-900/80 border border-slate-800/80 shadow-xl space-y-4">
      {/* Header and Presets Bar */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-slate-800">
        <div>
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
            Patient Physiological Input
          </span>
          <h3 className="text-lg font-bold text-white font-display flex items-center gap-2">
            Clinical Parameters
            <span className="text-[11px] font-sans font-normal text-slate-400">
              (Live Debounced Prediction)
            </span>
          </h3>
        </div>

        <button
          onClick={onReset}
          className="text-xs text-slate-400 hover:text-slate-200 flex items-center gap-1 self-start sm:self-auto py-1 px-2.5 rounded-lg border border-slate-800 hover:bg-slate-800/60"
        >
          <RefreshCw className="w-3 h-3" />
          Cohort Medians
        </button>
      </div>

      {/* Preset Buttons */}
      <div>
        <span className="text-[11px] text-slate-400 block mb-1.5 font-medium flex items-center gap-1">
          <Sparkles className="w-3.5 h-3.5 text-amber-400" />
          Load Real Patient Presets:
        </span>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
          {samples.map((s, idx) => {
            const presetColors = [
              'hover:border-emerald-500/50 hover:bg-emerald-500/10 text-emerald-300',
              'hover:border-amber-500/50 hover:bg-amber-500/10 text-amber-300',
              'hover:border-red-500/50 hover:bg-red-500/10 text-red-300',
            ];
            return (
              <button
                key={s.id}
                onClick={() => onLoadSample(s)}
                className={`p-2 rounded-xl text-left border border-slate-800 bg-slate-950/60 transition-all text-xs ${presetColors[idx % 3]}`}
              >
                <span className="font-bold block">{s.name}</span>
                <span className="text-[10px] text-slate-400 line-clamp-1 mt-0.5">{s.description}</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Group Navigation Tabs */}
      <div className="flex flex-wrap gap-1 p-1 bg-slate-950/80 rounded-xl border border-slate-800 text-xs">
        {groups.map((g) => (
          <button
            key={g.id}
            onClick={() => setActiveGroup(g.id)}
            className={`px-3 py-1.5 rounded-lg font-medium transition-all ${
              activeGroup === g.id
                ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm font-semibold'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            {g.label}
          </button>
        ))}
      </div>

      {/* Form Fields for Active Group */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 max-h-[380px] overflow-y-auto pr-1">
        {groupedFeatures[activeGroup]?.map((featKey) => {
          const meta = schema[featKey];
          const val = patientData[featKey];
          const isCohortDefault = cohortDefaults.includes(featKey);

          return (
            <div
              key={featKey}
              className="p-3 rounded-xl bg-slate-950/60 border border-slate-800/80 hover:border-slate-700/80 transition-colors"
            >
              <div className="flex items-start justify-between gap-1 mb-1.5">
                <label className="text-xs font-medium text-slate-200 block truncate" title={meta.label}>
                  {meta.label}
                </label>
                {meta.unit && (
                  <span className="text-[10px] text-slate-500 font-mono flex-shrink-0">
                    {meta.unit}
                  </span>
                )}
              </div>

              {/* Render Field depending on encoding */}
              {meta.encoding === 'numeric' || meta.encoding === 'ordinal' ? (
                <div className="space-y-1.5">
                  <div className="flex items-center justify-between">
                    <input
                      type="number"
                      min={meta.min}
                      max={meta.max}
                      step={featKey === 'CR' || featKey === 'K' || featKey === 'HB' ? '0.1' : '1'}
                      value={val ?? meta.median ?? ''}
                      onChange={(e) => {
                        const parsed = parseFloat(e.target.value);
                        onChangeField(featKey, isNaN(parsed) ? null : parsed);
                      }}
                      className="w-24 px-2 py-1 bg-slate-900 border border-slate-700 rounded-lg text-xs font-mono text-white focus:outline-none focus:border-cyan-400"
                    />
                    <span className="text-[10px] text-slate-500 font-mono">
                      Range: [{meta.min} - {meta.max}]
                    </span>
                  </div>

                  {meta.min !== undefined && meta.max !== undefined && (
                    <input
                      type="range"
                      min={meta.min}
                      max={meta.max}
                      step={featKey === 'CR' || featKey === 'K' || featKey === 'HB' ? '0.1' : '1'}
                      value={val ?? meta.median ?? meta.min}
                      onChange={(e) => onChangeField(featKey, parseFloat(e.target.value))}
                      className="w-full h-1 bg-slate-800 rounded-lg cursor-pointer accent-cyan-400"
                    />
                  )}
                </div>
              ) : meta.encoding === 'binary' ? (
                <div className="flex items-center gap-1.5 pt-1">
                  {meta.categories?.map((cat) => {
                    const isSelected = String(val) === String(cat);
                    const labelDisplay =
                      cat === 1 || cat === 'Y' || cat === 'Male'
                        ? cat === 'Male'
                          ? 'Male'
                          : 'Yes'
                        : cat === 'Fmale'
                        ? 'Female'
                        : 'No';

                    return (
                      <button
                        key={String(cat)}
                        type="button"
                        onClick={() => onChangeField(featKey, cat)}
                        className={`flex-1 py-1 px-2 rounded-lg text-xs font-medium transition-all ${
                          isSelected
                            ? 'bg-cyan-500/25 text-cyan-300 border border-cyan-500/50 font-bold shadow-sm'
                            : 'bg-slate-900/80 text-slate-400 border border-slate-800 hover:text-slate-200'
                        }`}
                      >
                        {labelDisplay}
                      </button>
                    );
                  })}
                </div>
              ) : (
                /* Nominal Categorical (e.g. BBB, VHD) */
                <select
                  value={val ?? meta.categories?.[0] ?? ''}
                  onChange={(e) => onChangeField(featKey, e.target.value)}
                  className="w-full px-2.5 py-1.5 bg-slate-900 border border-slate-700 rounded-lg text-xs text-slate-200 focus:outline-none focus:border-cyan-400"
                >
                  {meta.categories?.map((cat) => (
                    <option key={String(cat)} value={cat}>
                      {String(cat)}
                    </option>
                  ))}
                </select>
              )}

              {/* Indicator if cohort default was applied */}
              {isCohortDefault && (
                <span className="text-[9px] text-slate-500 italic block mt-1">
                  Default (Median: {meta.median ?? meta.categories?.[0]})
                </span>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
