"use client";

import { Activity, Clock, Zap } from "lucide-react";
import type { TrainingMetric, TrainingStatus as TStatus } from "@/lib/types";

interface Props {
  status: TStatus | null;
  latestMetric: TrainingMetric | undefined;
}

export default function TrainingStatus({ status, latestMetric }: Props) {
  if (!status) {
    return (
      <div className="rounded-xl bg-gray-900 p-4 border border-gray-800">
        <p className="text-gray-500 text-sm">Waiting for training to start...</p>
      </div>
    );
  }

  const epochProgress = status.total_epochs
    ? (status.epoch / status.total_epochs) * 100
    : 0;
  const stepProgress = status.total_steps
    ? (status.step / status.total_steps) * 100
    : 0;

  return (
    <div className="rounded-xl bg-gray-900 p-4 border border-gray-800">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Activity className="w-4 h-4 text-blue-400" />
          <h2 className="text-sm font-semibold text-gray-200">Training Progress</h2>
        </div>
        <span
          className={`text-xs px-2 py-0.5 rounded-full ${
            status.is_training
              ? "bg-green-900/50 text-green-400"
              : "bg-gray-800 text-gray-500"
          }`}
        >
          {status.is_training ? "Training" : "Idle"}
        </span>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Epoch */}
        <div>
          <p className="text-xs text-gray-500 mb-1">Epoch</p>
          <p className="text-lg font-mono text-gray-200">
            {status.epoch}
            <span className="text-gray-600">/{status.total_epochs}</span>
          </p>
          <div className="mt-1 h-1.5 bg-gray-800 rounded-full overflow-hidden">
            <div
              className="h-full bg-blue-500 rounded-full transition-all"
              style={{ width: `${epochProgress}%` }}
            />
          </div>
        </div>

        {/* Step */}
        <div>
          <p className="text-xs text-gray-500 mb-1">Step</p>
          <p className="text-lg font-mono text-gray-200">
            {status.step.toLocaleString()}
            <span className="text-gray-600">
              /{status.total_steps.toLocaleString()}
            </span>
          </p>
          <div className="mt-1 h-1.5 bg-gray-800 rounded-full overflow-hidden">
            <div
              className="h-full bg-purple-500 rounded-full transition-all"
              style={{ width: `${stepProgress}%` }}
            />
          </div>
        </div>

        {/* Learning Rate */}
        <div>
          <div className="flex items-center gap-1">
            <Zap className="w-3 h-3 text-yellow-400" />
            <p className="text-xs text-gray-500">Learning Rate</p>
          </div>
          <p className="text-lg font-mono text-gray-200 mt-1">
            {latestMetric ? latestMetric.lr.toExponential(2) : "—"}
          </p>
        </div>

        {/* Loss */}
        <div>
          <div className="flex items-center gap-1">
            <Clock className="w-3 h-3 text-orange-400" />
            <p className="text-xs text-gray-500">Total Loss</p>
          </div>
          <p className="text-lg font-mono text-gray-200 mt-1">
            {latestMetric ? latestMetric.loss_total.toFixed(4) : "—"}
          </p>
        </div>
      </div>
    </div>
  );
}
