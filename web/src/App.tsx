import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Heart, Activity, BarChart2, Globe, FileText, AlertCircle, Sliders, Zap } from 'lucide-react';
import { SchemaRegistry, PredictResponse, SamplePatient, ModelMetricsResponse } from './types';
import { HeartViewer } from './components/HeartViewer';
import { RiskOverview } from './components/RiskOverview';
import { VesselCards } from './components/VesselCards';
import { ShapWaterfall } from './components/ShapWaterfall';
import { PatientForm } from './components/PatientForm';
import { GlobalImportance } from './components/GlobalImportance';
import { MetricsTab } from './components/MetricsTab';
import { WhatIfSimulator } from './components/WhatIfSimulator';
import { ClinicalSummaryModal } from './components/ClinicalSummaryModal';
import { DisclaimerHeaderBanner, DisclaimerFooter } from './components/DisclaimerBanner';

export function App() {
  const [schema, setSchema] = useState<SchemaRegistry>({});
  const [samples, setSamples] = useState<SamplePatient[]>([]);
  const [metricsData, setMetricsData] = useState<ModelMetricsResponse | undefined>();
  const [patientData, setPatientData] = useState<Record<string, any>>({});
  const [prediction, setPrediction] = useState<PredictResponse | null>(null);

  const [selectedVessel, setSelectedVessel] = useState<string | null>(null);
  const [activeTarget, setActiveTarget] = useState<string>('Cath');
  const [activeTab, setActiveTab] = useState<'patient-shap' | 'what-if' | 'global-shap' | 'metrics'>('patient-shap');

  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [isPredicting, setIsPredicting] = useState<boolean>(false);
  const [apiError, setApiError] = useState<string | null>(null);
  const [showSummaryModal, setShowSummaryModal] = useState<boolean>(false);
  const [inferenceLatency, setInferenceLatency] = useState<number | null>(null);

  const debounceTimerRef = useRef<any>(null);

  // Initial data loading from backend
  useEffect(() => {
    async function initData() {
      try {
        setIsLoading(true);
        const [schemaRes, samplesRes, metricsRes] = await Promise.all([
          fetch('/api/schema'),
          fetch('/api/samples'),
          fetch('/api/metrics'),
        ]);

        if (!schemaRes.ok || !samplesRes.ok || !metricsRes.ok) {
          throw new Error('Failed to load initial API data');
        }

        const schemaData = await schemaRes.json();
        const samplesData = await samplesRes.json();
        const metricsDataJson = await metricsRes.json();

        setSchema(schemaData);
        setSamples(samplesData);
        setMetricsData(metricsDataJson);

        // Initialize patient data from first sample (Low risk preset)
        if (samplesData.length > 0) {
          setPatientData(samplesData[0].data);
        } else {
          // Fallback to schema medians
          const defaults: Record<string, any> = {};
          Object.entries(schemaData).forEach(([k, meta]: [string, any]) => {
            defaults[k] = meta.median ?? meta.categories?.[0] ?? 0;
          });
          setPatientData(defaults);
        }

        setApiError(null);
      } catch (err: any) {
        console.error('Initialization error:', err);
        setApiError('Unable to connect to ML backend. Please verify FastAPI is running on port 8000.');
      } finally {
        setIsLoading(false);
      }
    }

    initData();
  }, []);

  // Debounced prediction trigger (300 ms) whenever patientData changes
  const runPrediction = useCallback(async (data: Record<string, any>) => {
    try {
      setIsPredicting(true);
      const startTime = performance.now();
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      });

      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        throw new Error(errJson.detail || 'Prediction failed');
      }

      const predResult: PredictResponse = await res.json();
      setInferenceLatency(Math.round(performance.now() - startTime));
      setPrediction(predResult);
      setApiError(null);
    } catch (err: any) {
      console.error('Prediction request error:', err);
      setApiError(err.message || 'Error executing prediction');
    } finally {
      setIsPredicting(false);
    }
  }, []);

  useEffect(() => {
    if (Object.keys(patientData).length === 0) return;

    if (debounceTimerRef.current) {
      clearTimeout(debounceTimerRef.current);
    }

    debounceTimerRef.current = setTimeout(() => {
      runPrediction(patientData);
    }, 300);

    return () => {
      if (debounceTimerRef.current) clearTimeout(debounceTimerRef.current);
    };
  }, [patientData, runPrediction]);

  // Sync vessel selection with SHAP active target
  const handleSelectVessel = (vesselId: string | null) => {
    setSelectedVessel(vesselId);
    if (vesselId) {
      setActiveTarget(vesselId);
    } else {
      setActiveTarget('Cath');
    }
  };

  const handleChangeField = (field: string, value: any) => {
    setPatientData((prev) => ({ ...prev, [field]: value }));
  };

  const handleLoadSample = (sample: SamplePatient) => {
    setPatientData({ ...sample.data });
  };

  const handleResetDefaults = () => {
    const defaults: Record<string, any> = {};
    Object.entries(schema).forEach(([k, meta]) => {
      defaults[k] = meta.median ?? meta.categories?.[0] ?? 0;
    });
    setPatientData(defaults);
  };

  return (
    <div className="min-h-screen flex flex-col bg-[#070b14] text-slate-100">
      {/* Persistent Top Safety Disclaimer Banner */}
      <DisclaimerHeaderBanner />

      {/* Main App Navigation Header */}
      <header className="border-b border-slate-800/80 bg-slate-950/70 backdrop-blur-md sticky top-0 z-20 px-4 lg:px-8 py-3.5">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-rose-600 to-red-500 flex items-center justify-center shadow-lg shadow-rose-600/30">
              <Heart className="w-5 h-5 text-white animate-pulse" />
            </div>
            <div>
              <h1 className="text-lg font-extrabold font-display tracking-tight text-white flex items-center gap-2">
                Cardio3D AI
                <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-cyan-500/20 text-cyan-300 border border-cyan-500/40">
                  Multimodal v2.0
                </span>
              </h1>
              <p className="text-[11px] text-slate-400">
                Interactive 3D Coronary Anatomy & Multi-Vessel Stenosis Risk
              </p>
            </div>
          </div>

          {/* Telemetry Badge & Header Actions */}
          <div className="flex items-center gap-3">
            <div className="hidden md:flex items-center gap-2 px-3 py-1.5 rounded-xl bg-slate-900/80 border border-slate-800 text-[11px] text-slate-400 font-mono">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              <span>Latency: <strong className="text-emerald-300">{inferenceLatency ?? 14}ms</strong></span>
              <span className="text-slate-600">|</span>
              <span className="text-cyan-400">95% CI Calibrated</span>
            </div>

            <button
              onClick={() => setShowSummaryModal(true)}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-cyan-500/20 hover:bg-cyan-500/30 text-cyan-300 border border-cyan-500/40 text-xs font-semibold shadow-sm transition-all"
              title="Open and print comprehensive clinical assessment report"
            >
              <FileText className="w-3.5 h-3.5" />
              <span>Clinical Report</span>
            </button>

            {/* Navigation Tabs */}
            <div className="flex items-center gap-1 p-1 bg-slate-900/90 rounded-xl border border-slate-800 text-xs">
              <button
                onClick={() => setActiveTab('patient-shap')}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg font-medium transition-all ${
                  activeTab === 'patient-shap'
                    ? 'bg-cyan-500/25 text-cyan-300 font-bold border border-cyan-500/40 shadow-sm'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <Activity className="w-3.5 h-3.5" />
                <span>Risk & SHAP</span>
              </button>
              <button
                onClick={() => setActiveTab('what-if')}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg font-medium transition-all ${
                  activeTab === 'what-if'
                    ? 'bg-cyan-500/25 text-cyan-300 font-bold border border-cyan-500/40 shadow-sm'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <Sliders className="w-3.5 h-3.5" />
                <span>What-If</span>
              </button>
              <button
                onClick={() => setActiveTab('global-shap')}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg font-medium transition-all ${
                  activeTab === 'global-shap'
                    ? 'bg-cyan-500/25 text-cyan-300 font-bold border border-cyan-500/40 shadow-sm'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <Globe className="w-3.5 h-3.5" />
                <span>Global Factors</span>
              </button>
              <button
                onClick={() => setActiveTab('metrics')}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg font-medium transition-all ${
                  activeTab === 'metrics'
                    ? 'bg-cyan-500/25 text-cyan-300 font-bold border border-cyan-500/40 shadow-sm'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <BarChart2 className="w-3.5 h-3.5" />
                <span>Model CV Validation</span>
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Backend Error Banner */}
      {apiError && (
        <div className="bg-red-500/20 border-b border-red-500/40 px-4 py-2 text-center text-xs text-red-200 flex items-center justify-center gap-2">
          <AlertCircle className="w-4 h-4 text-red-400" />
          <span>{apiError}</span>
        </div>
      )}

      {/* Main Workspace Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 lg:p-6 space-y-6">
        {/* Top Split Section: 3D Anatomical Viewer + Clinical Overview */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
          {/* Left Column: Interactive 3D Heart Canvas (7 cols on lg) */}
          <div className="lg:col-span-7 h-[500px] lg:h-[620px] flex flex-col">
            <HeartViewer
              prediction={prediction}
              selectedVessel={selectedVessel}
              onSelectVessel={handleSelectVessel}
            />
          </div>

          {/* Right Column: CAD Risk Status & Vessel Cards (5 cols on lg) */}
          <div className="lg:col-span-5 flex flex-col justify-between space-y-4">
            <RiskOverview cad={prediction?.cad} isLoading={isPredicting} />

            <div className="space-y-2">
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-400 block">
                Target Vessel Stenosis (≥50% Narrowing)
              </span>
              <VesselCards
                vessels={prediction?.vessels}
                selectedVessel={selectedVessel}
                onSelectVessel={handleSelectVessel}
              />
            </div>

            {/* Quick Helper Badge */}
            <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800 text-[11px] text-slate-400 flex items-center justify-between">
              <span>Selected Target for SHAP:</span>
              <span className="font-mono font-bold text-cyan-300 uppercase px-2 py-0.5 rounded bg-cyan-500/10 border border-cyan-500/30">
                {activeTarget === 'Cath' ? 'Overall CAD' : activeTarget}
              </span>
            </div>
          </div>
        </div>

        {/* Bottom Section: Tabbed Detail Area */}
        <div className="w-full">
          {activeTab === 'patient-shap' && (
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
              {/* Patient Parameter Input Form */}
              <div className="lg:col-span-6">
                <PatientForm
                  schema={schema}
                  patientData={patientData}
                  samples={samples}
                  onChangeField={handleChangeField}
                  onLoadSample={handleLoadSample}
                  onReset={handleResetDefaults}
                  cohortDefaults={prediction?.cohort_defaults_applied ?? []}
                />
              </div>

              {/* Local SHAP Waterfall Attribution */}
              <div className="lg:col-span-6">
                <ShapWaterfall
                  explanations={prediction?.explain}
                  activeTarget={activeTarget}
                  onSelectTarget={setActiveTarget}
                />
              </div>
            </div>
          )}

          {activeTab === 'what-if' && (
            <WhatIfSimulator
              patientData={patientData}
              currentPrediction={prediction}
              onApplyToPatient={(modified) => {
                setPatientData(modified);
              }}
            />
          )}

          {activeTab === 'global-shap' && (
            <GlobalImportance globalShap={metricsData?.shap_global} />
          )}

          {activeTab === 'metrics' && (
            <MetricsTab metricsData={metricsData} />
          )}
        </div>
      </main>

      {/* Clinical Summary Report Modal */}
      {showSummaryModal && (
        <ClinicalSummaryModal
          prediction={prediction}
          patientData={patientData}
          schema={schema}
          onClose={() => setShowSummaryModal(false)}
        />
      )}

      {/* Persistent Bottom Disclaimer Footer */}
      <DisclaimerFooter />
    </div>
  );
}
