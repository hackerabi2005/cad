import React, { useState, useRef } from 'react';
import { UserCheck, Sparkles, Sliders, RefreshCw, ChevronDown, ChevronUp, Upload, CheckCircle2, AlertCircle } from 'lucide-react';
import { SchemaRegistry, SamplePatient } from '../types';

interface PatientFormProps {
  schema: SchemaRegistry;
  patientData: Record<string, any>;
  samples: SamplePatient[];
  onChangeField: (field: string, value: any) => void;
  onLoadSample: (sample: SamplePatient) => void;
  onReset: () => void;
  onBulkLoad?: (data: Record<string, any>) => void;
  cohortDefaults: string[];
}

export function PatientForm({
  schema,
  patientData,
  samples,
  onChangeField,
  onLoadSample,
  onReset,
  onBulkLoad,
  cohortDefaults,
}: PatientFormProps) {
  const [activeGroup, setActiveGroup] = useState<string>('demographic');
  const [importStatus, setImportStatus] = useState<{ type: 'success' | 'error'; message: string } | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

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

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const text = event.target?.result as string;
        if (!text) throw new Error('File is empty');

        let importedRecord: Record<string, any> = {};
        const filename = file.name.toLowerCase();

        if (filename.endsWith('.json')) {
          let parsed = JSON.parse(text);
          if (Array.isArray(parsed) && parsed.length > 0) {
            parsed = parsed[0];
          }
          if (parsed && typeof parsed === 'object' && parsed.data && typeof parsed.data === 'object') {
            parsed = parsed.data;
          }
          if (!parsed || typeof parsed !== 'object') {
            throw new Error('JSON does not contain a valid key-value object');
          }
          importedRecord = parsed;
        } else if (filename.endsWith('.csv')) {
          const lines = text.trim().split(/\r?\n/).filter((l) => l.trim().length > 0);
          if (lines.length < 2) {
            throw new Error('CSV must contain a header row and at least one data row');
          }
          const sep = lines[0].includes('\t') ? '\t' : lines[0].includes(';') ? ';' : ',';
          const headers = lines[0].split(sep).map((h) => h.trim().replace(/^["']|["']$/g, ''));
          const values = lines[1].split(sep).map((v) => v.trim().replace(/^["']|["']$/g, ''));

          headers.forEach((h, idx) => {
            if (values[idx] !== undefined && values[idx] !== '') {
              importedRecord[h] = values[idx];
            }
          });
        } else {
          throw new Error('Unsupported format. Please select a .json or .csv file');
        }

        // Clean & cast against schema with clamped boundary detection
        const schemaKeys = Object.keys(schema);
        const schemaLowerMap = new Map(schemaKeys.map((k) => [k.toLowerCase(), k]));
        const validImport: Record<string, any> = {};
        const matchedFields: string[] = [];
        const unmatchedColumns: string[] = [];
        const clampedFields: string[] = [];

        Object.entries(importedRecord).forEach(([k, rawVal]) => {
          const actualKey = schema[k] ? k : schemaLowerMap.get(k.toLowerCase());
          if (!actualKey) {
            unmatchedColumns.push(k);
            return;
          }
          const meta = schema[actualKey];
          let val = rawVal;
          if (meta.encoding === 'numeric' || meta.encoding === 'ordinal') {
            if (typeof val === 'string') {
              const num = parseFloat(val);
              val = isNaN(num) ? val : num;
            }
            if (typeof val === 'number') {
              const pmin = meta.phys_min ?? meta.min;
              const pmax = meta.phys_max ?? meta.max;
              if (pmin !== undefined && val < pmin) {
                clampedFields.push(`${actualKey} (< ${pmin})`);
                val = pmin;
              } else if (pmax !== undefined && val > pmax) {
                clampedFields.push(`${actualKey} (> ${pmax})`);
                val = pmax;
              }
            }
          } else if (meta.encoding === 'binary') {
            if (val === '1' || val === 1 || val === true || val === 'true' || val === 'Y' || val === 'Yes') {
              val = meta.categories?.includes(1) ? 1 : meta.categories?.includes('Y') ? 'Y' : val;
            } else if (val === '0' || val === 0 || val === false || val === 'false' || val === 'N' || val === 'No') {
              val = meta.categories?.includes(0) ? 0 : meta.categories?.includes('N') ? 'N' : val;
            }
          }
          validImport[actualKey] = val;
          matchedFields.push(actualKey);
        });

        if (matchedFields.length === 0) {
          throw new Error('No matching clinical parameters found in file');
        }

        if (onBulkLoad) {
          onBulkLoad(validImport);
        } else {
          Object.entries(validImport).forEach(([k, v]) => onChangeField(k, v));
        }

        const clampedMsg = clampedFields.length > 0 ? `, ${clampedFields.length} values clamped to physiological limits` : '';
        const unmatchedMsg = unmatchedColumns.length > 0 ? `, ${unmatchedColumns.length} columns unmatched` : '';
        setImportStatus({
          type: 'success',
          message: `Imported ${matchedFields.length} of 54 fields${clampedMsg}${unmatchedMsg} from ${file.name}`,
        });
      } catch (err: any) {
        setImportStatus({
          type: 'error',
          message: err.message || 'Error parsing file',
        });
      } finally {
        if (fileInputRef.current) fileInputRef.current.value = '';
      }
    };

    reader.onerror = () => {
      setImportStatus({ type: 'error', message: 'Failed to read file from disk' });
    };

    reader.readAsText(file);
  };

  const [revealCath, setRevealCath] = useState(false);

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
          <div className="flex items-center gap-2 mt-1">
            {(() => {
              const totalCount = Object.keys(schema).length || 54;
              const providedCount = Object.keys(schema).reduce(
                (c, k) => (patientData[k] !== undefined && patientData[k] !== null ? c + 1 : c),
                0
              );
              const imputedCount = totalCount - providedCount;
              const reliability = providedCount >= 35 ? 'High' : providedCount >= 20 ? 'Moderate' : 'Low';
              const colorCls = providedCount >= 35
                ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
                : providedCount >= 20
                ? 'bg-amber-500/10 text-amber-400 border-amber-500/30'
                : 'bg-rose-500/10 text-rose-400 border-rose-500/30';
              return (
                <span className={`text-[10px] px-2 py-0.5 rounded-full border font-mono ${colorCls}`}>
                  Inputs provided: {providedCount} of {totalCount} ({imputedCount > 0 ? `${imputedCount} imputed to cohort median/mode` : 'all provided'}) · {reliability} Reliability
                </span>
              );
            })()}
          </div>
        </div>

        <div className="flex items-center gap-2 self-start sm:self-auto">
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileUpload}
            accept=".json,.csv"
            className="hidden"
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            className="text-xs text-cyan-400 hover:text-cyan-300 flex items-center gap-1.5 py-1 px-2.5 rounded-lg border border-cyan-500/30 hover:bg-cyan-500/10 transition-colors"
            title="Import patient parameters from a CSV or JSON file"
          >
            <Upload className="w-3 h-3" />
            Import CSV/JSON
          </button>
          <button
            onClick={onReset}
            className="text-xs text-slate-400 hover:text-slate-200 flex items-center gap-1 py-1 px-2.5 rounded-lg border border-slate-800 hover:bg-slate-800/60 transition-colors"
            title="Reset all inputs to cohort-typical median and mode values"
          >
            <RefreshCw className="w-3 h-3" />
            Cohort-Typical Patient
          </button>
        </div>
      </div>

      {/* Import Status Alert Banner */}
      {importStatus && (
        <div
          className={`p-2.5 rounded-xl border text-xs flex items-center justify-between animate-fade-in ${
            importStatus.type === 'success'
              ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-300'
              : 'bg-rose-500/10 border-rose-500/30 text-rose-300'
          }`}
        >
          <div className="flex items-center gap-2">
            {importStatus.type === 'success' ? (
              <CheckCircle2 className="w-4 h-4 text-emerald-400 flex-shrink-0" />
            ) : (
              <AlertCircle className="w-4 h-4 text-rose-400 flex-shrink-0" />
            )}
            <span>{importStatus.message}</span>
          </div>
          <button
            onClick={() => setImportStatus(null)}
            className="text-[10px] uppercase font-bold text-slate-400 hover:text-white ml-2"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Preset Buttons */}
      <div>
        <div className="flex items-center justify-between mb-1.5">
          <span className="text-[11px] text-slate-400 font-medium flex items-center gap-1">
            <Sparkles className="w-3.5 h-3.5 text-amber-400" />
            Load Real Cohort Patients (Out-of-Fold Evaluated):
          </span>
          <button
            onClick={() => setRevealCath(!revealCath)}
            className={`text-[10px] px-2 py-0.5 rounded-md border font-mono font-medium transition-colors ${
              revealCath
                ? 'bg-cyan-500/20 text-cyan-300 border-cyan-500/40'
                : 'bg-slate-800/80 text-slate-400 border-slate-700/60 hover:text-white'
            }`}
          >
            {revealCath ? 'Hide Actual Cath' : 'Reveal Actual Cath Result'}
          </button>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2">
          {samples.map((s, idx) => {
            const presetColors = [
              'hover:border-emerald-500/50 hover:bg-emerald-500/10 text-emerald-300',
              'hover:border-amber-500/50 hover:bg-amber-500/10 text-amber-300',
              'hover:border-red-500/50 hover:bg-red-500/10 text-red-300',
              'hover:border-purple-500/50 hover:bg-purple-500/10 text-purple-300',
            ];
            const cathLabel = s.ground_truth_cath === 1 ? 'CAD Confirmed (≥50% Stenosis)' : 'No ≥50% stenosis (Normal)';
            return (
              <div key={s.id} className="flex flex-col gap-1">
                <button
                  onClick={() => onLoadSample(s)}
                  className={`p-2 rounded-xl text-left border border-slate-800 bg-slate-950/60 transition-all text-xs w-full ${presetColors[idx % 4]}`}
                >
                  <span className="font-bold block">{s.name}</span>
                  <span className="text-[10px] text-slate-400 line-clamp-1 mt-0.5">{s.description}</span>
                </button>
                {revealCath && (
                  <div className="p-2 rounded-lg bg-slate-950/90 border border-slate-800 text-[10px] space-y-0.5 animate-fade-in font-mono">
                    <span className="text-slate-400 block">Actual Angiography:</span>
                    <span className={`font-bold block ${s.ground_truth_cath === 1 ? 'text-red-400' : 'text-emerald-400'}`}>
                      {cathLabel}
                    </span>
                    <span className="text-slate-500 block text-[9px]">
                      Out-of-Fold P(CAD): {s.cv_estimated_prob !== undefined ? `${(s.cv_estimated_prob * 100).toFixed(1)}%` : '—'}
                    </span>
                    <span className="text-amber-400/80 block text-[8px]">
                      Training cohort preset (live score is in-sample)
                    </span>
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {revealCath && (
          <div className="mt-2.5 p-3 rounded-xl bg-slate-950/90 border border-slate-800 text-[11px] text-slate-300 space-y-2">
            <div>
              <span className="font-bold text-amber-300 block mb-0.5">Patient B Case Study — Educational False Positive:</span>
              <p className="text-slate-300 text-[11px] leading-relaxed">
                In Patient B, the model flagged elevated risk (OOF P≈50%), but angiography found <strong>no ≥50% stenosis</strong>. Possible explanations include non-obstructive plaque or microvascular dysfunction, which this dataset cannot determine.
              </p>
            </div>
            <div className="pt-2 border-t border-slate-800/80">
              <span className="font-bold text-purple-300 block mb-0.5">Patient D Case Study — Educational False Negative (Silent / Atypical LCX):</span>
              <p className="text-slate-300 text-[11px] leading-relaxed">
                In Patient D (47yo female, Row 20), typical chest pain is absent, ECG is normal, and EF is preserved (50%), yielding a low OOF predicted CAD risk of <strong>16.1%</strong>. However, catheterization revealed <strong>confirmed LCX stenosis ≥50%</strong>. This demonstrates the critical clinical reality: young female patients with non-focal symptoms and isolated circumflex lesions can slip past non-invasive risk models, underscoring why AI is decision support, not a replacement for anatomical imaging.
              </p>
            </div>
          </div>
        )}
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
                      step={featKey === 'CR' || featKey === 'K' || featKey === 'HB' || featKey === 'BMI' ? '0.1' : '1'}
                      value={
                        val !== undefined && val !== null
                          ? typeof val === 'number' && !Number.isInteger(val)
                            ? Number(val.toFixed(1))
                            : val
                          : ''
                      }
                      onChange={(e) => {
                        const parsed = parseFloat(e.target.value);
                        onChangeField(featKey, isNaN(parsed) ? null : parsed);
                      }}
                      className="w-24 px-2 py-1 bg-slate-900 border border-slate-700 rounded-lg text-xs font-mono text-white focus:outline-none focus:border-cyan-400"
                    />
                    <span className="text-[10px] text-slate-500 font-mono">
                      Range: [{meta.min !== undefined ? (Number.isInteger(meta.min) ? meta.min : meta.min.toFixed(1)) : ''} - {meta.max !== undefined ? (Number.isInteger(meta.max) ? meta.max : meta.max.toFixed(1)) : ''}]
                    </span>
                  </div>

                  {meta.min !== undefined && meta.max !== undefined && (
                    <input
                      type="range"
                      min={meta.min}
                      max={meta.max}
                      step={featKey === 'CR' || featKey === 'K' || featKey === 'HB' || featKey === 'BMI' ? '0.1' : '1'}
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
                  Default (Median/Mode: {meta.median ?? meta.mode ?? meta.categories?.[0]})
                </span>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
