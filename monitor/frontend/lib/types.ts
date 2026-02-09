export interface TrainingMetric {
  step: number;
  epoch: number;
  loss_total: number;
  loss_contrastive: number;
  loss_quantization: number;
  loss_balance: number;
  loss_consistency: number;
  lr: number;
}

export interface EvalMetric {
  epoch: number;
  map_i2t: number | null;
  map_t2i: number | null;
  map_i2i: number | null;
  map_t2t: number | null;
  p1: number | null;
  p5: number | null;
  p10: number | null;
  bit_entropy: number | null;
  quant_error: number | null;
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
  epoch: number;
  step: number;
  total_epochs: number;
  total_steps: number;
  is_training: boolean;
  config: Record<string, unknown> | null;
}

export interface WSMessage {
  type: "training" | "eval" | "system" | "status";
  data: TrainingMetric | EvalMetric | SystemMetric | TrainingStatus;
}
