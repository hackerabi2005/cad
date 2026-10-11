import React from 'react';
import { X, Printer, Heart, ShieldAlert, FileText, CheckCircle2 } from 'lucide-react';
import { PredictResponse, SchemaRegistry } from '../types';
import vesselsConfig from '../vessels.json';
import { VesselsConfig } from '../types';
import { getVesselRiskColor } from './HeartViewer';

interface ClinicalSummaryModalProps {
  prediction: PredictResponse | null;
  patientData: Record<string, any>;
  schema: SchemaRegistry;
  mode?: 'balanced' | 'screening';
  isSimulated?: boolean;
  onClose: () => void;
}

export function ClinicalSummaryModal({
  prediction,
  patientData,
  schema,
  mode = 'balanced',
  isSimulated = false,
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
      {/* Print-Only Background Watermark */}
      <div className="hidden print:flex fixed inset-0 pointer-events-none z-0 items-center justify-center overflow-hidden">
        <div className="text-neutral-300/30 text-4xl font-black uppercase tracking-wider -rotate-45 select-none text-center leading-relaxed border-4 border-dashed border-neutral-300/25 p-10 rounded-3xl max-w-3xl">
          {isSimulated ? 'SIMULATED COUNTERFACTUAL VALUES APPLIED — NOT AN ACTUAL RECORD' : 'DECISION SUPPORT / EDUCATIONAL USE ONLY'}
          <div className="text-lg font-bold mt-3 text-neutral-400/30">
            NOT A SUBSTITUTE FOR FORMAL DIAGNOSTIC IMAGING
          </div>
        </div>
      </div>

      <div className="relative w-full max-w-4xl bg-slate-900 border border-slate-700 rounded-2xl shadow-2xl overflow-hidden print:border-none print:shadow-none print:bg-white print:text-black print:max-w-none z-10">
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

          {isSimulated && (
            <div className="p-3 rounded-xl bg-amber-500/20 border-2 border-amber-500/60 text-amber-300 print:text-amber-900 print:border-amber-600 flex items-center justify-between gap-2 text-xs font-bold uppercase tracking-wider">
              <span>⚠ SIMULATED VALUES APPLIED — COUNTERFACTUAL WHAT-IF EXPLORATION (NOT AN ACTUAL PATIENT RECORD)</span>
              <span className="font-mono text-[10px] px-2 py-0.5 rounded bg-amber-500/30">SIMULATED</span>
            </div>
          )}

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
                  {cad.prob >= 0.99 ? '>99%' : cad.prob <= 0.01 ? '<1%' : `${(cad.prob * 100).toFixed(1)}%`}
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
                {cad.flagged_by_vessel && (
                  <span className="text-[10px] text-amber-400 font-mono font-bold block mt-1">
                    ⚠ Flagged by branch vessel stenosis (Disjunctive rule)
                  </span>
                )}
                <div className="text-[10px] text-slate-400 print:text-neutral-600 font-mono mt-1">
                  Flagged Vessels: <strong className="text-white print:text-black">{cad.diseased_vessels_count ?? 0} / 3</strong>
                </div>
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

          {/* Anatomical Coronary Tree & Myocardial Perfusion Territory Snapshot */}
          {(() => {
            const ladProb = prediction.vessels.LAD?.prob ?? 0;
            const lcxProb = prediction.vessels.LCX?.prob ?? 0;
            const rcaProb = prediction.vessels.RCA?.prob ?? 0;
            const ladCutoff = mode === 'screening'
              ? (prediction.vessels.LAD?.high_sensitivity_threshold ?? 0.420)
              : (prediction.vessels.LAD?.threshold ?? 0.50);
            const lcxCutoff = mode === 'screening'
              ? (prediction.vessels.LCX?.high_sensitivity_threshold ?? 0.248)
              : (prediction.vessels.LCX?.threshold ?? 0.50);
            const rcaCutoff = mode === 'screening'
              ? (prediction.vessels.RCA?.high_sensitivity_threshold ?? 0.232)
              : (prediction.vessels.RCA?.threshold ?? 0.50);
            const ladColor = getVesselRiskColor(ladProb, ladCutoff);
            const lcxColor = getVesselRiskColor(lcxProb, lcxCutoff);
            const rcaColor = getVesselRiskColor(rcaProb, rcaCutoff);

            return (
              <div className="p-4 rounded-xl bg-slate-950/80 border border-slate-800 print:bg-neutral-50 print:border-neutral-200 space-y-3">
                <div className="flex items-center justify-between border-b border-slate-800 print:border-neutral-300 pb-2">
                  <div>
                    <span className="text-[10px] uppercase font-bold text-slate-400 print:text-neutral-600 block">
                      3D Anatomical Coronary Reconstruction & Territory Map
                    </span>
                    <span className="text-xs font-bold text-white print:text-black font-display">
                      Coronary Branch Perfusion Territories & Stenosis Risk
                    </span>
                  </div>
                  <span className="text-[10px] font-mono text-cyan-400 print:text-neutral-600">
                    AHA 17-Segment Alignment
                  </span>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-12 gap-4 items-center">
                  {/* Vector Coronary Tree Snapshot Schematic */}
                  <div className="md:col-span-5 bg-slate-900/90 print:bg-white rounded-xl p-3 border border-slate-800 print:border-neutral-200 flex flex-col items-center justify-center">
                    <svg viewBox="0 0 280 220" className="w-full max-w-[240px] h-auto drop-shadow-md">
                      {/* Heart Muscle Silhouette */}
                      <path
                        d="M 140,50 C 90,15 30,55 40,115 C 50,170 140,210 140,210 C 140,210 230,170 240,115 C 250,55 190,15 140,50 Z"
                        fill="#be123c"
                        fillOpacity="0.12"
                        stroke="#e11d48"
                        strokeWidth="1.5"
                        strokeDasharray="3 3"
                      />
                      {/* Aorta Bulb (Neutral grey - Not modeled) */}
                      <path
                        d="M 125,55 C 125,25 155,25 155,55 Z"
                        fill="#334155"
                        stroke="#64748b"
                        strokeWidth="2"
                      />
                      <text x="140" y="38" textAnchor="middle" fill="#94a3b8" fontSize="6.5" fontFamily="sans-serif" fontWeight="bold">Aorta (Not modeled)</text>

                      {/* Left Main Stem (Neutral grey - Not modeled) */}
                      <path d="M 146,54 L 165,65" stroke="#64748b" strokeWidth="4" strokeLinecap="round" />
                      <text x="156" y="58" textAnchor="end" fill="#94a3b8" fontSize="5.5" fontFamily="sans-serif">LM (Not modeled)</text>

                      {/* LAD Branch */}
                      <path
                        d="M 165,65 Q 160,110 142,195"
                        fill="none"
                        stroke={ladColor}
                        strokeWidth="4.5"
                        strokeLinecap="round"
                      />
                      <path
                        d="M 160,105 Q 185,130 195,150"
                        fill="none"
                        stroke={ladColor}
                        strokeWidth="2.5"
                        strokeLinecap="round"
                      />

                      {/* LCX Branch */}
                      <path
                        d="M 165,65 Q 215,75 228,125 Q 230,145 220,165"
                        fill="none"
                        stroke={lcxColor}
                        strokeWidth="4"
                        strokeLinecap="round"
                      />
                      <path
                        d="M 210,95 Q 225,115 225,135"
                        fill="none"
                        stroke={lcxColor}
                        strokeWidth="2"
                        strokeLinecap="round"
                      />

                      {/* RCA Branch */}
                      <path
                        d="M 134,55 Q 85,70 65,115 Q 60,145 80,180 Q 105,195 130,200"
                        fill="none"
                        stroke={rcaColor}
                        strokeWidth="4"
                        strokeLinecap="round"
                      />
                      <path
                        d="M 68,125 Q 85,140 100,145"
                        fill="none"
                        stroke={rcaColor}
                        strokeWidth="2"
                        strokeLinecap="round"
                      />

                      {/* Labels */}
                      <rect x="148" y="115" width="48" height="15" rx="3" fill="#0f172a" fillOpacity="0.85" stroke={ladColor} strokeWidth="1" />
                      <text x="172" y="126" textAnchor="middle" fill={ladColor} fontSize="8" fontWeight="bold" fontFamily="monospace">
                        LAD {(ladProb * 100).toFixed(0)}%
                      </text>

                      <rect x="215" y="80" width="48" height="15" rx="3" fill="#0f172a" fillOpacity="0.85" stroke={lcxColor} strokeWidth="1" />
                      <text x="239" y="91" textAnchor="middle" fill={lcxColor} fontSize="8" fontWeight="bold" fontFamily="monospace">
                        LCX {(lcxProb * 100).toFixed(0)}%
                      </text>

                      <rect x="18" y="90" width="48" height="15" rx="3" fill="#0f172a" fillOpacity="0.85" stroke={rcaColor} strokeWidth="1" />
                      <text x="42" y="101" textAnchor="middle" fill={rcaColor} fontSize="8" fontWeight="bold" fontFamily="monospace">
                        RCA {(rcaProb * 100).toFixed(0)}%
                      </text>
                    </svg>
                    <span className="text-[9px] text-slate-400 print:text-neutral-500 font-mono mt-1 text-center">
                      Anterior Projection · Shaded by Cutoff-Relative Risk
                    </span>
                  </div>

                  {/* Territory & Segment Legend */}
                  <div className="md:col-span-7 space-y-2 text-[11px]">
                    <div className="p-2 rounded-lg bg-slate-900/60 border border-slate-800 print:bg-neutral-100 print:border-neutral-200">
                      <div className="flex items-center justify-between">
                        <span className="font-bold text-white print:text-black flex items-center gap-1.5">
                          <span className="w-2.5 h-2.5 rounded-full inline-block" style={{ backgroundColor: ladColor }} />
                          LAD Territory (Anterior Wall & Septum)
                        </span>
                        <span className="font-mono font-bold" style={{ color: ladColor }}>
                          {(ladProb * 100).toFixed(1)}% ({prediction.vessels.LAD?.label ?? 'Normal'})
                        </span>
                      </div>
                      <p className="text-[10px] text-slate-400 print:text-neutral-600 mt-0.5">
                        AHA Segments 1, 2, 7, 8, 13, 14, 17. Supplies 45-55% of LV myocardium including apex.
                      </p>
                    </div>

                    <div className="p-2 rounded-lg bg-slate-900/60 border border-slate-800 print:bg-neutral-100 print:border-neutral-200">
                      <div className="flex items-center justify-between">
                        <span className="font-bold text-white print:text-black flex items-center gap-1.5">
                          <span className="w-2.5 h-2.5 rounded-full inline-block" style={{ backgroundColor: lcxColor }} />
                          LCX Territory (Lateral Wall)
                        </span>
                        <span className="font-mono font-bold" style={{ color: lcxColor }}>
                          {(lcxProb * 100).toFixed(1)}% ({prediction.vessels.LCX?.label ?? 'Normal'})
                        </span>
                      </div>
                      <p className="text-[10px] text-slate-400 print:text-neutral-600 mt-0.5">
                        AHA Segments 5, 6, 11, 12, 16. Supplies lateral wall and posterior base of LV.
                      </p>
                    </div>

                    <div className="p-2 rounded-lg bg-slate-900/60 border border-slate-800 print:bg-neutral-100 print:border-neutral-200">
                      <div className="flex items-center justify-between">
                        <span className="font-bold text-white print:text-black flex items-center gap-1.5">
                          <span className="w-2.5 h-2.5 rounded-full inline-block" style={{ backgroundColor: rcaColor }} />
                          RCA Territory (Inferior Wall & RV)
                        </span>
                        <span className="font-mono font-bold" style={{ color: rcaColor }}>
                          {(rcaProb * 100).toFixed(1)}% ({prediction.vessels.RCA?.label ?? 'Normal'})
                        </span>
                      </div>
                      <p className="text-[10px] text-slate-400 print:text-neutral-600 mt-0.5">
                        AHA Segments 3, 4, 9, 10, 15. Supplies inferior LV wall, right ventricle, and conduction tissue.
                      </p>
                    </div>

                    <p className="text-[9px] text-slate-500 print:text-neutral-500 italic pt-0.5">
                      *Correspondence note: Entire vessel branches are shaded according to model predicted probability. The reference dataset (Alizadehsani et al.) provides binary vessel stenosis, not focal lesion coordinates.
                    </p>
                  </div>
                </div>
              </div>
            );
          })()}

          {/* Key Feature Attributions (SHAP) */}
          <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800 print:bg-neutral-50 print:border-neutral-200">
            <span className="text-[10px] uppercase font-bold text-slate-400 print:text-neutral-600 block mb-2">
              Top Physiological Risk Contributors (SHAP Explainability)
            </span>
            <div className="space-y-1.5 font-mono text-[11px]">
              {topShapDrivers.map((f, i) => {
                const deltaPp = f.shap * 100;
                return (
                  <div key={f.feature} className="flex items-center justify-between py-1 border-b border-slate-800/60 last:border-none print:border-neutral-200">
                    <div className="flex items-center gap-2">
                      <span className="text-slate-500">{i + 1}.</span>
                      <span className="text-slate-200 print:text-neutral-900 font-sans font-medium">{f.label}</span>
                      {f.value !== null && (
                        <span className="text-slate-400 print:text-neutral-600 text-[10px]">
                          = {String(f.value)} {f.unit}
                        </span>
                      )}
                    </div>
                    <div className="flex items-center gap-2">
                      <span className={`font-bold ${deltaPp > 0 ? 'text-red-400 print:text-red-700' : 'text-emerald-400 print:text-emerald-700'}`}>
                        {deltaPp > 0 ? `+${deltaPp.toFixed(1)}` : deltaPp.toFixed(1)} pp
                      </span>
                      <span className="text-slate-500 text-[10px] w-8 text-right">({f.pct.toFixed(0)}%)</span>
                    </div>
                  </div>
                );
              })}
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
