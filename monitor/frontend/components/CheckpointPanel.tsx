"use client";

import { useMemo } from "react";
import Link from "next/link";
import { Download, ExternalLink, Play, Star } from "lucide-react";
import { useRunContext } from "@/contexts/RunContext";

export default function CheckpointPanel() {
  const {
    selectedRunId,
    checkpointsByRun,
    runCheckpoints: dbCheckpoints,
    selectedCheckpointId,
    selectCheckpoint,
    loadCheckpointForInference,
  } = useRunContext();

  // Use file-based checkpoints (legacy) or DB checkpoints
  const fileCheckpoints = selectedRunId
    ? checkpointsByRun[selectedRunId] ?? []
    : [];

  // Prefer DB checkpoints if available, fall back to file-based
  const hasDbCheckpoints = dbCheckpoints.length > 0;

  const bestPath = useMemo(() => {
    const source = hasDbCheckpoints
      ? dbCheckpoints.map((c) => ({ val_loss: c.val_loss, path: c.path }))
      : fileCheckpoints.map((c) => ({ val_loss: c.val_loss, path: c.path }));
    const withLoss = source.filter((c) => c.val_loss != null);
    if (withLoss.length === 0) return null;
    return withLoss.reduce((a, b) =>
      (a.val_loss ?? Infinity) < (b.val_loss ?? Infinity) ? a : b,
    ).path;
  }, [hasDbCheckpoints, dbCheckpoints, fileCheckpoints]);

  return (
    <div className="rounded-xl bg-gray-900 border border-gray-800 p-3">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-1.5">
          <Download className="w-3.5 h-3.5 text-gray-500" />
          <span className="text-xs font-medium text-gray-400">
            Checkpoints
          </span>
        </div>
        {(hasDbCheckpoints || fileCheckpoints.length > 0) && (
          <Link
            href={`/inference${selectedRunId ? `?run=${selectedRunId}` : ""}`}
            className="flex items-center gap-1 text-[10px] text-blue-400 hover:text-blue-300 transition-colors"
          >
            <span>Open Explorer</span>
            <ExternalLink className="w-2.5 h-2.5" />
          </Link>
        )}
      </div>

      {!hasDbCheckpoints && fileCheckpoints.length === 0 ? (
        <p className="text-xs text-gray-600 py-2">
          {selectedRunId
            ? "No checkpoints for this run"
            : "Select a run to view checkpoints"}
        </p>
      ) : hasDbCheckpoints ? (
        /* DB-based checkpoints with selection support */
        <div className="space-y-1 max-h-[200px] overflow-y-auto">
          {dbCheckpoints.map((ckpt) => {
            const isBest = ckpt.path === bestPath;
            const isSelected = ckpt.id === selectedCheckpointId;
            return (
              <div
                key={ckpt.id}
                className={`rounded-lg px-2.5 py-1.5 text-xs transition-colors cursor-pointer ${
                  isSelected
                    ? "bg-blue-900/30 border border-blue-700/50"
                    : isBest
                      ? "bg-yellow-900/20 border border-yellow-800/40 hover:bg-yellow-900/30"
                      : "bg-gray-800/50 hover:bg-gray-800 border border-transparent"
                }`}
                onClick={() => selectCheckpoint(ckpt.id)}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-1.5 min-w-0">
                    {isBest && (
                      <Star className="w-3 h-3 text-yellow-500 shrink-0 fill-yellow-500" />
                    )}
                    <span className={`truncate ${isSelected ? "text-blue-300" : "text-gray-300"}`}>
                      Epoch {ckpt.epoch}{ckpt.step != null ? ` (step ${ckpt.step})` : ""}
                    </span>
                  </div>
                  <div className="flex items-center gap-2 shrink-0">
                    {ckpt.val_loss != null && (
                      <span className="text-[10px] text-gray-500">
                        loss {ckpt.val_loss.toFixed(4)}
                      </span>
                    )}
                    <span className="text-[10px] text-gray-500">
                      {ckpt.size_mb?.toFixed(0) ?? "?"}MB
                    </span>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        loadCheckpointForInference(ckpt.path);
                      }}
                      className="flex items-center gap-0.5 px-1.5 py-0.5 rounded bg-green-700/50 text-green-300 hover:bg-green-700 text-[10px] transition-colors"
                    >
                      <Play className="w-2 h-2" />
                    </button>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      ) : (
        /* File-based checkpoints (legacy fallback) */
        <div className="space-y-1 max-h-[200px] overflow-y-auto">
          {fileCheckpoints.map((ckpt) => {
            const isBest = ckpt.path === bestPath;
            return (
              <Link
                key={ckpt.path}
                href={`/inference?load=${encodeURIComponent(ckpt.path)}`}
                className={`block rounded-lg px-2.5 py-1.5 text-xs transition-colors ${
                  isBest
                    ? "bg-yellow-900/20 border border-yellow-800/40 hover:bg-yellow-900/30"
                    : "bg-gray-800/50 hover:bg-gray-800"
                }`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-1.5 min-w-0">
                    {isBest && (
                      <Star className="w-3 h-3 text-yellow-500 shrink-0 fill-yellow-500" />
                    )}
                    <span className="text-gray-300 truncate">
                      {ckpt.epoch != null ? `Epoch ${ckpt.epoch}` : ckpt.name}
                    </span>
                  </div>
                  <div className="flex items-center gap-2 shrink-0 text-[10px] text-gray-500">
                    {ckpt.val_loss != null && (
                      <span>loss {ckpt.val_loss.toFixed(4)}</span>
                    )}
                    <span>{ckpt.size_mb.toFixed(0)}MB</span>
                  </div>
                </div>
              </Link>
            );
          })}
        </div>
      )}
    </div>
  );
}
