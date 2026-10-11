export interface SchemaItem {
  group: 'demographic' | 'symptoms-exam' | 'ECG' | 'lab' | 'echo';
  encoding: 'numeric' | 'binary' | 'ordinal' | 'nominal';
  unit: string;
  label: string;
  min?: number;
  max?: number;
  mean?: number;
  median?: number;
  mode?: string | number;
  categories?: (string | number)[];
  ref_low?: number;
  ref_high?: number;
  phys_min?: number;
  phys_max?: number;
  source?: string;
  reviewed?: boolean;
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
  is_imputed?: boolean;
}

export interface TargetExplanation {
  base_value: number;
  raw_score: number;
  reconstructed_score: number;
  additive_error: number;
  features: FeatureContribution[];
  shap_unit?: 'probability' | 'log-odds';
  unit_label?: string;
}

export interface SimilarCohortInfo {
  n: number;
  observed_rate: number;
  wilson_interval: [number, number];
  summary: string;
}

export interface CadPrediction {
  prob: number;
  raw_prob?: number;
  displayed_prob?: number;
  driving_vessel?: 'LAD' | 'LCX' | 'RCA' | null;
  coherent_prob: number;
  similar_cohort?: SimilarCohortInfo;
  label: string;
  high_sens_label: string;
  threshold: number;
  high_sensitivity_threshold: number;
  nested_sensitivity?: number;
  coherence_adjusted: boolean;
  flagged_by_vessel?: boolean;
  flagged_vessels_screening?: string[];
  flagged_vessels_balanced?: string[];
  diseased_vessels_count?: number;
  flagged_vessels_count?: number;
  diffuse_or_non_focal?: boolean;
  diffuse_or_non_focal_screening?: boolean;
  diffuse_or_non_focal_balanced?: boolean;
  diffuse_cad_message?: string;
  plain_language_summary?: string;
}

export interface VesselPrediction {
  prob: number;
  similar_cohort?: SimilarCohortInfo;
  label: string;
  high_sens_label: string;
  threshold: number;
  high_sensitivity_threshold: number;
  cutoff_applied?: number;
  nested_sensitivity?: number;
  ppv?: number;
}

export interface CompletenessGroup {
  provided: number;
  total: number;
}

export interface CompletenessInfo {
  provided: number;
  total: number;
  by_group: Record<string, CompletenessGroup>;
}

export interface PredictResponse {
  cad: CadPrediction;
  vessels: Record<'LAD' | 'LCX' | 'RCA', VesselPrediction>;
  explain: Record<string, TargetExplanation>;
  cohort_defaults_applied: string[];
  missing_features?: string[];
  imputed_values?: Record<string, any>;
  completeness?: CompletenessInfo;
  reliability?: 'high' | 'moderate' | 'low';
  warnings?: string[];
  disclaimer: string;
}

export interface SamplePatient {
  id: string;
  name: string;
  description: string;
  data: Record<string, any>;
  ground_truth_cath?: number;
  cv_estimated_prob?: number;
  seen_in_training?: boolean;
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
  dca?: Record<string, any>;
  eval_curves?: {
    roc: Record<string, { fpr: number[]; tpr: number[]; thresholds: number[]; auc: number }>;
    pr: Record<string, { precision: number[]; recall: number[]; thresholds: number[]; pr_auc: number }>;
    calibration: Record<string, { prob_true: number[]; prob_pred: number[]; brier: number }>;
    confusion_matrices: Record<string, Record<string, any>>;
    decision_curve: Record<string, any>;
  };
  full_operating_metrics_table?: Record<string, any>;
  vessel_count_confusion?: {
    balanced_05: {
      matrix: number[][];
      column_totals: number[];
      row_totals: number[];
      total_patients: number;
    };
    screening: {
      matrix: number[][];
      column_totals: number[];
      row_totals: number[];
      total_patients: number;
    };
  };
}

export interface WhatIfResponse {
  baseline_cad_prob: number;
  modified_cad_prob: number;
  delta_cad_prob: number;
  vessel_deltas: Record<string, number>;
  baseline_prediction: PredictResponse;
  modified_prediction: PredictResponse;
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
