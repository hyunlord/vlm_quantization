"use client";

import { useMemo } from "react";
import { Activity, TrendingUp, TrendingDown, Minus } from "lucide-react";
import type { EvalMetric } from "@/lib/types";

interface BackboneComparisonProps {
  evalData: EvalMetric[];
  selectedEpoch?: number;
  selectedStep?: number;
}

function formatPercent(value: number | null | undefined): string {
  if (value == null) return "—";
  return `${(value * 100).toFixed(1)}%`;
}

function TrendIcon({ hash, backbone }: { hash: number | null; backbone: number | null }) {
  if (hash == null || backbone == null) return <Minus className="w-3 h-3 text-gray-500" />;
  const diff = hash - backbone;
  if (diff > 0.01) return <TrendingUp className="w-3 h-3 text-emerald-400" />;
  if (diff < -0.01) return <TrendingDown className="w-3 h-3 text-red-400" />;
  return <Minus className="w-3 h-3 text-gray-500" />;
}

export default function BackboneComparison({ evalData, selectedEpoch, selectedStep }: BackboneComparisonProps) {
  // Get the eval data for selected step/epoch or latest
  const currentEval = useMemo(() => {
    if (evalData.length === 0) return null;
    // Prefer step-based match (unique), fall back to epoch, then latest
    if (selectedStep != null) {
      const byStep = evalData.find((e) => e.step === selectedStep);
      if (byStep) return byStep;
    }
    if (selectedEpoch != null) {
      // With val_check_interval < 1, multiple evals share the same epoch.
      // Return the LAST one for that epoch (most recent).
      const matches = evalData.filter((e) => e.epoch === selectedEpoch);
      if (matches.length > 0) return matches[matches.length - 1];
    }
    return evalData[evalData.length - 1];
  }, [evalData, selectedEpoch, selectedStep]);

  // Get baseline (epoch 0) for comparison
  const baselineEval = useMemo(() => {
    return evalData.find((e) => e.epoch === 0) ?? null;
  }, [evalData]);

  if (!currentEval) {
    return (
      <div className="rounded-xl bg-gray-900 border border-gray-800 p-4">
        <div className="flex items-center gap-2 mb-3">
          <Activity className="w-4 h-4 text-gray-500" />
          <span className="text-sm font-medium text-gray-300">Backbone vs Hash Comparison</span>
        </div>
        <p className="text-xs text-gray-600 text-center py-4">
          No evaluation data available yet
        </p>
      </div>
    );
  }

  const metrics = [
    {
      label: "mAP (I→T)",
      hash: currentEval.map_i2t,
      backbone: currentEval.backbone_map_i2t,
      baseline: baselineEval?.map_i2t,
    },
    {
      label: "mAP (T→I)",
      hash: currentEval.map_t2i,
      backbone: currentEval.backbone_map_t2i,
      baseline: baselineEval?.map_t2i,
    },
    {
      label: "P@1",
      hash: currentEval.p1,
      backbone: currentEval.backbone_p1,
      baseline: baselineEval?.p1,
    },
    {
      label: "P@5",
      hash: currentEval.p5,
      backbone: currentEval.backbone_p5,
      baseline: baselineEval?.p5,
    },
    {
      label: "P@10",
      hash: currentEval.p10,
      backbone: currentEval.backbone_p10,
      baseline: baselineEval?.p10,
    },
  ];

  return (
    <div className="rounded-xl bg-gray-900 border border-gray-800 p-4">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Activity className="w-4 h-4 text-purple-400" />
          <span className="text-sm font-medium text-gray-300">Backbone vs Hash Comparison</span>
        </div>
        <span className="text-xs text-gray-500">
          Epoch {currentEval.epoch}
        </span>
      </div>

      {/* Legend */}
      <div className="flex items-center gap-4 mb-3 text-[10px]">
        <div className="flex items-center gap-1.5">
          <div className="w-2 h-2 rounded-full bg-blue-500" />
          <span className="text-gray-400">Backbone (SigLIP)</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-2 h-2 rounded-full bg-purple-500" />
          <span className="text-gray-400">Hash Layer</span>
        </div>
        {baselineEval && (
          <div className="flex items-center gap-1.5">
            <div className="w-2 h-2 rounded-full bg-gray-600" />
            <span className="text-gray-500">Baseline (Epoch 0)</span>
          </div>
        )}
      </div>

      {/* Metrics Grid */}
      <div className="space-y-2">
        {metrics.map((m) => {
          const hashValue = m.hash ?? 0;
          const backboneValue = m.backbone ?? 0;
          const maxValue = Math.max(hashValue, backboneValue, 0.01);

          return (
            <div key={m.label} className="space-y-1">
              <div className="flex items-center justify-between">
                <span className="text-xs text-gray-400 w-20">{m.label}</span>
                <div className="flex items-center gap-3 text-xs">
                  <span className="text-blue-400 font-mono w-14 text-right">
                    {formatPercent(m.backbone)}
                  </span>
                  <TrendIcon hash={m.hash} backbone={m.backbone} />
                  <span className="text-purple-400 font-mono w-14 text-right">
                    {formatPercent(m.hash)}
                  </span>
                </div>
              </div>

              {/* Bar comparison */}
              <div className="flex gap-1 h-2">
                {/* Backbone bar */}
                <div className="flex-1 bg-gray-800 rounded-sm overflow-hidden">
                  <div
                    className="h-full bg-blue-500/60 rounded-sm transition-all"
                    style={{ width: `${(backboneValue / maxValue) * 100}%` }}
                  />
                </div>
                {/* Hash bar */}
                <div className="flex-1 bg-gray-800 rounded-sm overflow-hidden">
                  <div
                    className="h-full bg-purple-500/60 rounded-sm transition-all"
                    style={{ width: `${(hashValue / maxValue) * 100}%` }}
                  />
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Summary */}
      <div className="mt-4 pt-3 border-t border-gray-800">
        <div className="grid grid-cols-2 gap-3 text-xs">
          <div className="bg-blue-900/20 rounded-lg p-2.5 border border-blue-800/30">
            <p className="text-blue-400 font-medium mb-1">Backbone (Continuous)</p>
            <p className="text-gray-400">
              mAP: {formatPercent(currentEval.backbone_map_i2t)} / {formatPercent(currentEval.backbone_map_t2i)}
            </p>
            <p className="text-[10px] text-gray-500 mt-1">
              1152-dim cosine similarity
            </p>
          </div>
          <div className="bg-purple-900/20 rounded-lg p-2.5 border border-purple-800/30">
            <p className="text-purple-400 font-medium mb-1">Hash (Binary)</p>
            <p className="text-gray-400">
              mAP: {formatPercent(currentEval.map_i2t)} / {formatPercent(currentEval.map_t2i)}
            </p>
            <p className="text-[10px] text-gray-500 mt-1">
              Hamming distance retrieval
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
