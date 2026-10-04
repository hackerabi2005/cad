export interface SchemaItem {
  group: 'demographic' | 'symptoms-exam' | 'ECG' | 'lab' | 'echo';
  encoding: 'numeric' | 'binary' | 'ordinal' | 'nominal';
  unit: string;
  label: string;
  min?: number;
  max?: number;
  mean?: number;
  median?: number;
  categories?: (string | number)[];
}

export type SchemaRegistry = Record<string, SchemaItem>;

export interface FeatureContribution {
  feature: string;
  label: string;
  group: string;
  unit: string;
  value: number | string | null;
  shap: number;
  pct: number;
}

export interface TargetExplanation {
  base_value: number;
  raw_score: number;
  reconstructed_score: number;
  additive_error: number;
  features: FeatureContribution[];
}

export interface CadPrediction {
  prob: number;
  coherent_prob: number;
  label: string;
  high_sens_label: string;
  threshold: number;
  high_sensitivity_threshold: number;
  coherence_adjusted: boolean;
}

export interface VesselPrediction {
  prob: number;
  label: string;
  high_sens_label: string;
  threshold: number;
  high_sensitivity_threshold: number;
}

export interface PredictResponse {
  cad: CadPrediction;
  vessels: Record<'LAD' | 'LCX' | 'RCA', VesselPrediction>;
  explain: Record<string, TargetExplanation>;
  cohort_defaults_applied: string[];
  disclaimer: string;
}

export interface SamplePatient {
  id: string;
  name: string;
  description: string;
  data: Record<string, any>;
}

export interface GlobalShapItem {
  feature: string;
  label: string;
  group: string;
  unit: string;
  mean_abs_shap: number;
  importance_pct: number;
}

export interface ModelMetricsResponse {
  metrics: Record<string, Record<string, Record<string, number>>>;
  shap_global: Record<string, GlobalShapItem[]>;
  model_card: Record<string, any>;
}

export interface VesselDef {
  id: string;
  target: string;
  nodeName: string;
  name: string;
  abbreviation: string;
  territory: string;
  description: string;
  badgePosition: [number, number, number];
}

export interface AnatomyDef {
  id: string;
  nodeName: string;
  name: string;
  defaultColor: string;
  defaultOpacity: number;
}

export interface ColorStop {
  stop: number;
  color: string;
  label: string;
}

export interface VesselsConfig {
  vessels: VesselDef[];
  anatomy: AnatomyDef[];
  colorScale: ColorStop[];
}
