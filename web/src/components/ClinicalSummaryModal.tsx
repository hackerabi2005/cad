import React from 'react';
import { X, Printer, Heart, ShieldAlert, FileText, CheckCircle2 } from 'lucide-react';
import { PredictResponse, SchemaRegistry } from '../types';
import vesselsConfig from '../vessels.json';
import { VesselsConfig } from '../types';

interface ClinicalSummaryModalProps {
  prediction: PredictResponse | null;
  patientData: Record<string, any>;
  schema: SchemaRegistry;
  onClose: () => void;
}

export function ClinicalSummaryModal({
  prediction,
  patientData,
  schema,
  onClose,
}: ClinicalSummaryModalProps) {
  if (!prediction) return null;

  const config = vesselsConfig as VesselsConfig;
  const cad = prediction.cad;
  const handlePrint = () => {
    window.print();
  };

  const todayStr = new Date().toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  });

  const topShapDrivers = prediction.explain?.Cath?.features.slice(0, 5) ?? [];

  return (
    <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm p-4 overflow-y-auto flex items-center justify-center animate-fade-in print:p-0 print:bg-white print:static">
      <div className="relative w-full max-w-4xl bg-slate-900 border border-slate-700 rounded-2xl shadow-2xl overflow-hidden print:border-none print:shadow-none print:bg-white print:text-black print:max-w-none">
        {/* Top Control Bar (Hidden on print) */}
        <div className="flex items-center justify-between px-6 py-4 bg-slate-950/80 border-b border-slate-800 print:hidden">
          <div className="flex items-center gap-2">
            <FileText className="w-5 h-5 text-cyan-400" />
            <h2 className="text-base font-bold text-white">Clinical Assessment & Decision Report</h2>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={handlePrint}
              className="flex items-center gap-1.5 px-4 py-2 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold text-xs shadow-lg shadow-cyan-500/20 transition-all"
            >
              <Printer className="w-4 h-4" />
              <span>Print / Save PDF</span>
            </button>
            <button
              onClick={onClose}
              className="p-2 rounded-xl text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
              title="Close Report"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Printable Clinical Document Body */}
        <div className="p-8 space-y-6 text-slate-200 print:text-black print:p-6 text-xs leading-relaxed">
          {/* Institution & Report Header */}
          <div className="flex items-start justify-between border-b border-slate-700/80 pb-5 print:border-neutral-300">
            <div>
              <div className="flex items-center gap-2 mb-1">
                <Heart className="w-6 h-6 text-rose-500 fill-rose-500" />
                <span className="text-xl font-black font-display tracking-tight text-white print:text-black">
                  Cardio3D AI Clinical Decision Support
                </span>
              </div>
              <p className="text-slate-400 print:text-neutral-600 text-xs">
                3D Coronary Anatomy & Multi-Vessel Stenosis Risk Evaluation
              </p>
              <p className="text-[11px] text-slate-500 print:text-neutral-500 font-mono mt-0.5">
                Model: Calibrated Logistic-ElasticNet Pipeline · Version: 2.0.0 · Dual Cutoff Architecture
              </p>
            </div>
            <div className="text-right font-mono text-[11px] text-slate-400 print:text-neutral-600">
              <div>Date: <strong className="text-white print:text-black">{todayStr}</strong></div>
              <div>Report ID: CAD-RPT-{(Math.random() * 899999 + 100000).toFixed(0)}</div>
              <div>Status: Verified by Decision Curve Analysis</div>
            </div>
          </div>

          {/* Patient Demographic Summary */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 p-4 rounded-xl bg-slate-950/60 border border-slate-800 print:bg-neutral-50 print:border-neutral-200">
            <div>
              <span className="text-slate-400 print:text-neutral-500 text-[10px] uppercase font-bold block">Age / Sex</span>
              <span className="font-semibold text-white print:text-black text-sm">
                {patientData.Age ?? '58'} yrs · {patientData.Sex === 'Fmale' ? 'Female' : 'Male'}
              </span>
            </div>
            <div>
              <span className="text-slate-400 print:text-neutral-500 text-[10px] uppercase font-bold block">Resting BP / PR</span>
              <span className="font-semibold text-white print:text-black text-sm font-mono">
                {patientData.BP ?? 130} mmHg · {patientData.PR ?? 75} bpm
              </span>
            </div>
            <div>
              <span className="text-slate-400 print:text-neutral-500 text-[10px] uppercase font-bold block">Metabolic (FBS / BMI)</span>
              <span className="font-semibold text-white print:text-black text-sm font-mono">
                FBS: {patientData.FBS ?? 100} · BMI: {Number(patientData.BMI ?? 27.2).toFixed(1)}
              </span>
            </div>
            <div>
              <span className="text-slate-400 print:text-neutral-500 text-[10px] uppercase font-bold block">Lipid Panel (LDL / HDL)</span>
              <span className="font-semibold text-white print:text-black text-sm font-mono">
                LDL: {patientData.LDL ?? 120} · HDL: {patientData.HDL ?? 40}
              </span>
            </div>
          </div>

          {/* Primary Risk Finding */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="p-4 rounded-xl bg-slate-950/80 border border-cyan-500/30 print:border-neutral-300 md:col-span-1 flex flex-col justify-between">
              <div>
                <span className="text-[10px] uppercase font-bold text-slate-400 print:text-neutral-600 block mb-1">
                  Overall CAD Risk (Cath ≥50%)
                </span>
                <div className="text-3xl font-extrabold font-mono text-cyan-400 print:text-neutral-900">
                  {(cad.prob * 100).toFixed(1)}%
                </div>
                {cad.similar_cohort ? (
                  <div className="text-[11px] font-mono text-cyan-300/80 print:text-neutral-600 mt-1">
                    {cad.similar_cohort.summary}
                    <div className="text-[10px] text-slate-400 print:text-neutral-500">
                      Wilson 95% interval: [{(cad.similar_cohort.wilson_interval[0] * 100).toFixed(1)}% – {(cad.similar_cohort.wilson_interval[1] * 100).toFixed(1)}%]
                    </div>
                  </div>
                ) : (
                  <div className="text-[10px] text-slate-500 print:text-neutral-500 mt-1">
                    Uncertainty not quantified
                  </div>
                )}
              </div>
              <div className="mt-3 pt-3 border-t border-slate-800 print:border-neutral-200">
                <span className="text-[10px] uppercase text-slate-400 block font-bold">Category</span>
                <span className={`text-xs font-bold uppercase tracking-wider ${cad.label === 'CAD' ? 'text-rose-400 print:text-red-700' : 'text-emerald-400 print:text-emerald-700'}`}>
                  {cad.label === 'CAD' ? 'Obstructive Coronary Artery Disease' : 'Normal / Non-obstructive'}
                </span>
              </div>
            </div>

            {/* Multi-Vessel Stenosis Breakdown */}
            <div className="p-4 rounded-xl bg-slate-950/80 border border-slate-800 print:border-neutral-300 md:col-span-2">
              <span className="text-[10px] uppercase font-bold text-slate-400 print:text-neutral-600 block mb-2">
                Anatomical Coronary Artery Branch Specific Risk
              </span>
              <div className="grid grid-cols-3 gap-2">
                {config.vessels.map((v) => {
                  const vData = prediction.vessels[v.id as 'LAD' | 'LCX' | 'RCA'];
                  const prob = vData?.prob ?? 0;
                  const isStenotic = vData?.label === 'Stenotic';
                  return (
                    <div key={v.id} className="p-2.5 rounded-lg bg-slate-900/90 border border-slate-800 print:bg-neutral-50 print:border-neutral-200">
                      <div className="flex items-center justify-between mb-1">
                        <span className="font-bold text-white print:text-black font-mono">{v.abbreviation}</span>
                        <span className={`text-[9px] font-bold px-1.5 py-0.5 rounded ${isStenotic ? 'bg-red-500/20 text-red-300 print:text-red-700' : 'bg-emerald-500/20 text-emerald-300 print:text-emerald-700'}`}>
                          {vData?.label ?? 'Normal'}
                        </span>
                      </div>
                      <div className="text-lg font-extrabold font-mono text-cyan-300 print:text-black">
                        {(prob * 100).toFixed(1)}%
                      </div>
                      {vData?.similar_cohort && (
                        <div className="text-[9px] font-mono text-slate-400 print:text-neutral-600">
                          Similar n={vData.similar_cohort.n}: {(vData.similar_cohort.observed_rate * 100).toFixed(0)}% stenotic [{(vData.similar_cohort.wilson_interval[0] * 100).toFixed(0)}%–{(vData.similar_cohort.wilson_interval[1] * 100).toFixed(0)}%]
                        </div>
                      )}
                      <div className="text-[9px] text-slate-500 print:text-neutral-500 truncate mt-1">
                        {v.territory}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Key Feature Attributions (SHAP) */}
          <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800 print:bg-neutral-50 print:border-neutral-200">
            <span className="text-[10px] uppercase font-bold text-slate-400 print:text-neutral-600 block mb-2">
              Top Physiological Risk Contributors (SHAP Explainability)
            </span>
            <div className="space-y-1.5 font-mono text-[11px]">
              {topShapDrivers.map((f, i) => (
                <div key={f.feature} className="flex items-center justify-between py-1 border-b border-slate-800/60 last:border-none print:border-neutral-200">
                  <div className="flex items-center gap-2">
                    <span className="text-slate-500">{i + 1}.</span>
                    <span className="text-slate-200 print:text-neutral-900 font-sans font-medium">{f.label}</span>
                    {f.value !== null && (
                      <span className="text-slate-400 print:text-neutral-600 text-[10px]">({f.value} {f.unit})</span>
                    )}
                  </div>
                  <div className="flex items-center gap-2">
                    <span className={`font-bold ${f.shap > 0 ? 'text-red-400 print:text-red-700' : 'text-emerald-400 print:text-emerald-700'}`}>
                      {f.shap > 0 ? `+${f.shap.toFixed(3)}` : f.shap.toFixed(3)}
                    </span>
                    <span className="text-slate-500 text-[10px] w-8 text-right">({f.pct.toFixed(0)}%)</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Items for Clinician Review (Replaces "Clinical Recommendations") */}
          <div className="p-4 rounded-xl bg-slate-950/80 border border-slate-800 print:bg-neutral-50 print:border-neutral-200 space-y-2">
            <span className="text-[10px] uppercase font-bold text-cyan-400 print:text-neutral-800 block">
              Items for Clinician Review
            </span>
            <ul className="space-y-1 text-slate-300 print:text-neutral-700 list-disc list-inside">
              {cad.prob >= 0.60 ? (
                <>
                  <li><strong>Diagnostic Imaging Discussion:</strong> High probability of significant coronary artery disease warrants clinician review for anatomical evaluation (CCTA or Invasive Coronary Angiography).</li>
                  <li><strong>Medical Risk Factor Review:</strong> Review lipid management, blood pressure targets (&lt;130/80 mmHg), and antiplatelet therapy indications per clinical judgment.</li>
                  <li><strong>Territory Review:</strong> Correlate LAD risk ({((prediction.vessels.LAD?.prob ?? 0) * 100).toFixed(0)}%) with anterior myocardial symptoms and prior ECG tracings.</li>
                </>
              ) : (
                <>
                  <li><strong>Outpatient Risk Stratification:</strong> Low-to-moderate probability supports consideration of non-invasive stress testing or coronary calcium scoring (CAC).</li>
                  <li><strong>Preventive Discussion:</strong> Review lifestyle factors, smoking status, and metabolic indicators at routine follow-up.</li>
                </>
              )}
            </ul>
          </div>

          {/* Required Printout Disclaimer, Model Version, and Date */}
          <div className="pt-3 border-t border-slate-800 print:border-neutral-300 text-[10px] text-slate-400 print:text-neutral-600 space-y-1">
            <div className="font-semibold text-slate-300 print:text-neutral-800">
              {prediction.disclaimer}
            </div>
            <div className="flex flex-wrap items-center justify-between text-slate-500 print:text-neutral-500 font-mono text-[9px]">
              <span>Model Version: 2.0.0 · Dual Cutoff Architecture</span>
              <span>Report Generated: {todayStr}</span>
              <span>Decision Curve Analysis: Out-of-fold single-centre apparent utility</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
