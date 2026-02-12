export interface RunInfo {
  run_id: string;
  started_at: number;
  epochs: number;
  num_training_points: number;
  num_eval_points: number;
  has_hash_analysis: boolean;
  // Enhanced fields from runs table
  status?: "running" | "completed" | "failed";
  ended_at?: number;
  checkpoint_count?: number;
  best_val_loss?: number;
  best_checkpoint?: {
    id: number;
    epoch: number;
    val_loss: number;
    path: string;
  };
}

export interface EpochSummary {
  epoch: number;
  start_step: number;
  end_step: number;
  num_steps: number;
  started_at: number;
  ended_at: number;
  checkpoint?: {
    id: number;
    path: string;
    val_loss: number | null;
    size_mb: number | null;
  };
  eval?: {
    map_i2t: number | null;
    map_t2i: number | null;
    val_loss_total: number | null;
  };
}

export interface Checkpoint {
  id: number;
  run_id: string;
  epoch: number;
  step: number | null;
  path: string;
  filename: string;
  size_mb: number | null;
  val_loss: number | null;
  created_at: number;
  hparams_json?: string;
  run_status?: string;
}

export interface TrainingMetric {
  run_id?: string;
  step: number;
  epoch: number;
  loss_total: number;
  loss_contrastive: number;
  loss_quantization: number;
  loss_balance: number;
  loss_consistency: number;
  loss_ortho: number;
  loss_lcs: number;
  loss_distillation: number;
  lr: number;
  temperature: number | null;
}

export interface EvalMetric {
  run_id?: string;
  epoch: number;
  step: number | null;
  map_i2t: number | null;
  map_t2i: number | null;
  backbone_map_i2t: number | null;
  backbone_map_t2i: number | null;
  p1: number | null;
  p5: number | null;
  p10: number | null;
  backbone_p1: number | null;
  backbone_p5: number | null;
  backbone_p10: number | null;
  bit_entropy: number | null;
  quant_error: number | null;
  val_loss_total: number | null;
  val_loss_contrastive: number | null;
  val_loss_quantization: number | null;
  val_loss_balance: number | null;
  val_loss_consistency: number | null;
  val_loss_ortho: number | null;
  val_loss_lcs: number | null;
  val_loss_distillation: number | null;
}

export interface SystemMetric {
  gpu_util: number;
  gpu_mem_used: number;
  gpu_mem_total: number;
  gpu_temp: number;
  gpu_name: string;
  cpu_util: number;
  ram_used: number;
  ram_total: number;
}

export interface TrainingStatus {
  run_id?: string;
  epoch: number;
  step: number;
  total_epochs: number;
  total_steps: number;
  is_training: boolean;
  config: Record<string, unknown> | null;
}

export interface HashAnalysisSample {
  image_id: number;
  caption: string;
  thumbnail: string;
}

export interface HashAnalysisData {
  epoch: number;
  step: number;
  bit_activations: Record<string, number[]>;
  samples: HashAnalysisSample[];
  sample_img_codes: number[][];
  sample_txt_codes: number[][];
  similarity_matrix: number[][];
  bit: number;
  augmentation?: AugmentationAnalysis;
}

// --- Augmentation robustness types ---

export interface AugmentationSample {
  image_id: number;
  weak_mean_sim: number;
  weak_min_sim: number;
  strong_mean_sim: number;
  strong_min_sim: number;
  weak_code: number[];
  strong_code: number[];
  weak_thumbnail: string;
  strong_thumbnail: string;
}

export interface AugmentationAnalysis {
  samples: AugmentationSample[];
  weak_mean_overall: number;
  strong_mean_overall: number;
  weak_bit_stability: number[];
  strong_bit_stability: number[];
  bit: number;
  n_augs: number;
}

export interface CheckpointInfo {
  path: string;
  name: string;
  run_dir: string;
  size_mb: number;
  modified: string;
  epoch: number | null;
  step: number | null;
  val_loss: number | null;
  map_i2t?: number | null;
  map_t2i?: number | null;
  p1?: number | null;
  p5?: number | null;
  p10?: number | null;
}

// --- Search types ---

export interface SearchResult {
  rank: number;
  image_id: number;
  caption: string;
  thumbnail: string;
  score: number;
  distance: number | null;
}

export interface SearchResponse {
  query_type: string;
  mode: string;
  results: SearchResult[];
}

export interface IndexStatus {
  loaded: boolean;
  num_items: number;
  bit_list: number[];
  index_path: string;
}

// --- Inference types ---

export interface InferenceHashCode {
  bits: number;
  binary: number[];
  continuous: number[];
}

export interface InferenceComparison {
  bits: number;
  hamming: number;
  max_distance: number;
  similarity: number;
}

export interface ModelStatus {
  loaded: boolean;
  backbone_only?: boolean;
  checkpoint: string;
  model_name: string;
  bit_list: number[];
  hparams: Record<string, unknown>;
}

export interface WSMessage {
  type: "training" | "eval" | "system" | "status" | "hash_analysis";
  data:
    | TrainingMetric
    | EvalMetric
    | SystemMetric
    | TrainingStatus
    | HashAnalysisData;
}

// --- Optuna types ---

export interface OptunaTrial {
  number: number;
  value: number | null;
  state: "COMPLETE" | "PRUNED" | "FAIL" | "RUNNING" | "WAITING";
  params: Record<string, number>;
  user_attrs: Record<string, number>;
  duration_seconds: number | null;
}

export interface OptunaStudySummary {
  name: string;
  direction: string;
  n_trials: number;
  n_complete: number;
  n_pruned: number;
  n_fail: number;
  n_running: number;
  best_value: number | null;
  best_trial_number: number | null;
  param_importances: Record<string, number>;
  best_trial: OptunaTrial | null;
}
