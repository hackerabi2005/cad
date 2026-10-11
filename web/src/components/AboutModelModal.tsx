import React, { useEffect } from 'react';
import { X, ShieldAlert, Cpu, Database, CheckCircle2, Sliders, Layers, Heart } from 'lucide-react';

interface AboutModelModalProps {
  onClose: () => void;
}

export function AboutModelModal({ onClose }: AboutModelModalProps) {
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="about-model-title"
      className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm p-4 overflow-y-auto flex items-center justify-center animate-fade-in"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="relative w-full max-w-2xl bg-slate-900 border border-slate-700 rounded-2xl shadow-2xl overflow-hidden text-slate-100">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 bg-slate-950/80 border-b border-slate-800">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-amber-500/20 border border-amber-500/40 flex items-center justify-center text-amber-300">
              <ShieldAlert className="w-4 h-4" />
            </div>
            <div>
              <h2 id="about-model-title" className="text-base font-bold text-white">
                About Cardio3D AI & Clinical Safety
              </h2>
              <p className="text-[11px] text-slate-400">
                Model specifications, clinical validation, and intended use
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-xl text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
            title="Close dialog"
            aria-label="Close dialog"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content Body */}
        <div className="p-6 space-y-5 max-h-[75vh] overflow-y-auto text-xs leading-relaxed text-slate-300">
          {/* Clinical Safety Disclaimer Banner */}
          <div className="p-3.5 rounded-xl bg-amber-500/10 border border-amber-500/30 text-amber-200">
            <h3 className="font-bold text-amber-300 text-sm mb-1 flex items-center gap-1.5">
              <ShieldAlert className="w-4 h-4" />
              Clinical Intended Use & Safety Disclaimer
            </h3>
            <p>
              Cardio3D AI is designed for <strong>decision support / educational purposes only — not a substitute for formal diagnostic imaging</strong>. Formal bedside clinical evaluation, invasive coronary angiography, or formal cardiac imaging (CCTA/SPECT) remain mandatory.
            </p>
          </div>

          {/* Model Architecture */}
          <div className="space-y-2">
            <h3 className="font-bold text-white text-sm flex items-center gap-1.5">
              <Cpu className="w-4 h-4 text-cyan-400" />
              Model Architecture & Explainability
            </h3>
            <p>
              All 4 coronary targets (Overall CAD, LAD, LCX, RCA) utilize a unified <strong>LogitEnsemble</strong> combining calibrated L2 Logistic Regression with non-stochastic XGBoost trees in log-odds space.
            </p>
            <ul className="list-disc list-inside space-y-1 text-slate-400 ml-1">
              <li><strong>SHAP Additivity:</strong> Exact analytical log-odds decomposition combining LinearSHAP and TreeSHAP with guaranteed mathematical precision (|error| &lt; 1e-9).</li>
              <li><strong>Cross-Validation Performance:</strong> Out-of-fold ROC-AUC 0.932 for overall CAD, 0.840 for LAD, 0.724 for LCX, and 0.741 for RCA evaluated across 5-fold × 3 repeats (3×5-fold repeated stratified CV).</li>
            </ul>
          </div>

          {/* Operating Modes */}
          <div className="space-y-2">
            <h3 className="font-bold text-white text-sm flex items-center gap-1.5">
              <Sliders className="w-4 h-4 text-emerald-400" />
              Operating Decision Modes
            </h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-1">
              <div className="p-3 rounded-xl bg-slate-800/60 border border-slate-700/60">
                <span className="font-bold text-slate-200 block text-xs">Default (0.50) Mode</span>
                <p className="text-[11px] text-slate-400 mt-1">
                  Standard statistical threshold (Sens 94.5% [95% CI 90.6%–96.8%], Spec 70.9% [95% CI 60.6%–79.5%], PPV 89.1%). Best for standard risk profiling.
                </p>
              </div>
              <div className="p-3 rounded-xl bg-slate-800/60 border border-cyan-500/30 bg-cyan-950/20">
                <span className="font-bold text-cyan-300 block text-xs">Screening Mode (~90%+ Sensitivity)</span>
                <p className="text-[11px] text-slate-400 mt-1">
                  High-sensitivity cutoffs (CAD 0.400, LAD 0.420, LCX 0.248, RCA 0.232; Sens 96.3% [95% CI 92.9%–98.1%], Spec 57.0% [95% CI 46.4%–66.9%]) tuned on out-of-fold validation to prevent missed coronary stenosis in triage.
                </p>
              </div>
            </div>
          </div>

          {/* Logical Coherence Rule */}
          <div className="space-y-2">
            <h3 className="font-bold text-white text-sm flex items-center gap-1.5">
              <Layers className="w-4 h-4 text-indigo-400" />
              Logical Risk Coherence Policy
            </h3>
            <p>
              By definition, significant stenosis (≥50%) in any major vessel constitutes CAD (CAD = LAD ∨ LCX ∨ RCA). The system enforces both probability and discrete flag coherence:
            </p>
            <div className="p-2.5 rounded-lg bg-slate-950/80 font-mono text-[11px] text-indigo-300 border border-indigo-500/20 space-y-1">
              <div>P(CAD)_coherent = max(P(CAD), P(LAD), P(LCX), P(RCA))</div>
              <div className="text-cyan-300">Flag(CAD) = (P(CAD) ≥ cutoff) ∨ Flag(LAD) ∨ Flag(LCX) ∨ Flag(RCA)</div>
            </div>
            <p className="text-slate-400">
              Applying the disjunctive rule eliminates contradiction where a vessel is flagged positive while overall CAD is marked Low Risk (0.0% discordance), elevating screening sensitivity to 96.3% (specificity 57.0%).
            </p>
          </div>

          {/* Cohort Limitations */}
          <div className="space-y-2">
            <h3 className="font-bold text-white text-sm flex items-center gap-1.5">
              <ShieldAlert className="w-4 h-4 text-rose-400" />
              Cohort Limitations & Generalizability
            </h3>
            <ul className="list-disc list-inside space-y-1 text-slate-400 ml-1">
              <li><strong>Single Center:</strong> Recruited from a single tertiary cardiovascular center (Shaheed Rajaei Cardiovascular, Medical and Research Center in Tehran). External multi-center validation is required.</li>
              <li><strong>High CAD Prevalence:</strong> 71.6% of patients in this cohort had angiographically confirmed CAD (217 CAD, 86 Normal), reflecting high pre-test clinical referral probability.</li>
              <li><strong>Referral Bias:</strong> All 303 patients underwent invasive coronary angiography; asymptomatic screening cohorts are not represented.</li>
              <li><strong>Branch Vessel Limits:</strong> LCX/RCA ROC-AUC (~0.73) represents the physiological limit of non-invasive clinical biomarkers without localized imaging.</li>
            </ul>
          </div>

          {/* Cohort Provenance & Licensing */}
          <div className="space-y-2">
            <h3 className="font-bold text-white text-sm flex items-center gap-1.5">
              <Database className="w-4 h-4 text-violet-400" />
              Data Provenance & Open Science Attribution
            </h3>
            <p>
              Trained on the UCI Z-Alizadeh Sani coronary artery disease dataset (n=303 Iranian patients, 54 physiological biomarkers from Shaheed Rajaei Cardiovascular, Medical and Research Center in Tehran). The published baseline split is 216 CAD and 87 Normal (71.3%). Row 93 consensus alignment: In raw records, Row 93 had Cath = Normal despite LAD = Stenotic; adhering to the clinical definition that stenosis in any major epicardial vessel constitutes CAD aligns this to CAD = 1, resulting in 217 CAD and 86 Normal (71.6% prevalence).
            </p>
            <div className="text-[11px] text-slate-400 pt-1 border-t border-slate-800 space-y-1">
              <p>• <strong>Dataset Citation:</strong> Alizadehsani, Roshanzamir, Sani (2017), <em>Extension of Z-Alizadeh Sani dataset</em>, UCI Machine Learning Repository (CC BY 4.0, DOI: 10.17632/bgf5czvpg2.1).</p>
              <p>• <strong>3D Anatomical Meshes:</strong> BodyParts3D, © The Database Center for Life Science licensed under <strong>CC BY-SA 2.1 JP</strong>. Node mappings: LAD (anterior interventricular branch), LCX (circumflex branch), RCA (right coronary artery), Aorta (ascending aorta), Heart_Muscle (chamber myocardium).</p>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end px-6 py-3 bg-slate-950/80 border-t border-slate-800">
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
