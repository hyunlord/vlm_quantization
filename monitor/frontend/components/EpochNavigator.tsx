"use client";

import { useMemo } from "react";
import { CheckCircle, Circle, Download, Play } from "lucide-react";
import { useRunContext } from "@/contexts/RunContext";

export default function EpochNavigator() {
  const {
    selectedRunId,
    epochSummaries,
    selectedEvalEpoch,
    selectEvalEpoch,
    selectedCheckpointId,
    selectCheckpoint,
    loadCheckpointForInference,
  } = useRunContext();

  const bestEpoch = useMemo(() => {
    const withCkpt = epochSummaries.filter((e) => e.checkpoint?.val_loss != null);
    if (withCkpt.length === 0) return null;
    return withCkpt.reduce((a, b) =>
      (a.checkpoint?.val_loss ?? Infinity) < (b.checkpoint?.val_loss ?? Infinity) ? a : b
    );
  }, [epochSummaries]);

  if (!selectedRunId) {
    return (
      <div className="rounded-xl bg-gray-900 border border-gray-800 p-3">
        <div className="flex items-center gap-1.5 mb-2">
          <Circle className="w-3.5 h-3.5 text-gray-500" />
          <span className="text-xs font-medium text-gray-400">Epochs</span>
        </div>
        <p className="text-xs text-gray-600 py-2">Select a run to view epochs</p>
      </div>
    );
  }

  if (epochSummaries.length === 0) {
    return (
      <div className="rounded-xl bg-gray-900 border border-gray-800 p-3">
        <div className="flex items-center gap-1.5 mb-2">
          <Circle className="w-3.5 h-3.5 text-gray-500" />
          <span className="text-xs font-medium text-gray-400">Epochs</span>
        </div>
        <p className="text-xs text-gray-600 py-2">No epochs recorded yet</p>
      </div>
    );
  }

  return (
    <div className="rounded-xl bg-gray-900 border border-gray-800 p-3">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-1.5">
          <Circle className="w-3.5 h-3.5 text-gray-500" />
          <span className="text-xs font-medium text-gray-400">
            Epochs ({epochSummaries.length})
          </span>
        </div>
        {bestEpoch && (
          <span className="text-[10px] text-yellow-500">
            Best: Epoch {bestEpoch.epoch}
          </span>
        )}
      </div>

      <div className="space-y-1 max-h-[250px] overflow-y-auto">
        {epochSummaries.map((epoch) => {
          const isSelected = selectedEvalEpoch === epoch.epoch;
          const hasCkpt = !!epoch.checkpoint;
          const isBest = bestEpoch?.epoch === epoch.epoch;
          const ckptSelected = epoch.checkpoint?.id === selectedCheckpointId;

          return (
            <div
              key={epoch.epoch}
              className={`rounded-lg px-2.5 py-2 text-xs transition-colors cursor-pointer ${
                isSelected
                  ? "bg-blue-900/30 border border-blue-800/50"
                  : isBest
                    ? "bg-yellow-900/20 border border-yellow-800/30 hover:bg-yellow-900/30"
                    : "bg-gray-800/50 hover:bg-gray-800 border border-transparent"
              }`}
              onClick={() => selectEvalEpoch(epoch.epoch)}
            >
              <div className="flex items-center justify-between mb-1">
                <div className="flex items-center gap-1.5">
                  {hasCkpt ? (
                    <CheckCircle
                      className={`w-3 h-3 ${isBest ? "text-yellow-500" : "text-green-500"}`}
                    />
                  ) : (
                    <Circle className="w-3 h-3 text-gray-600" />
                  )}
                  <span className={`font-medium ${isSelected ? "text-blue-300" : "text-gray-300"}`}>
                    Epoch {epoch.epoch}
                  </span>
                </div>
                <span className="text-[10px] text-gray-500">
                  Steps {epoch.start_step}-{epoch.end_step}
                </span>
              </div>

              {/* Metrics row */}
              <div className="flex items-center gap-3 text-[10px] text-gray-500 ml-4">
                {epoch.eval?.val_loss_total != null && (
                  <span>Loss: {epoch.eval.val_loss_total.toFixed(4)}</span>
                )}
                {epoch.eval?.map_i2t != null && (
                  <span>mAP: {(epoch.eval.map_i2t * 100).toFixed(1)}%</span>
                )}
              </div>

              {/* Checkpoint actions */}
              {hasCkpt && epoch.checkpoint && (
                <div className="flex items-center gap-2 mt-1.5 ml-4">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      selectCheckpoint(epoch.checkpoint!.id);
                    }}
                    className={`flex items-center gap-1 px-2 py-0.5 rounded text-[10px] transition-colors ${
                      ckptSelected
                        ? "bg-blue-600 text-white"
                        : "bg-gray-700 text-gray-300 hover:bg-gray-600"
                    }`}
                  >
                    <Download className="w-2.5 h-2.5" />
                    <span>{epoch.checkpoint.size_mb?.toFixed(0) ?? "?"}MB</span>
                  </button>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      loadCheckpointForInference(epoch.checkpoint!.path);
                    }}
                    className="flex items-center gap-1 px-2 py-0.5 rounded bg-green-700/50 text-green-300 hover:bg-green-700 text-[10px] transition-colors"
                  >
                    <Play className="w-2.5 h-2.5" />
                    <span>Load</span>
                  </button>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
