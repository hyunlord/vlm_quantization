"use client";

import { useMemo } from "react";
import Link from "next/link";
import { Download, ExternalLink, Star } from "lucide-react";
import { useRunContext } from "@/contexts/RunContext";

export default function CheckpointPanel() {
  const { selectedRunId, checkpointsByRun } = useRunContext();

  const runCheckpoints = selectedRunId
    ? checkpointsByRun[selectedRunId] ?? []
    : [];

  const bestPath = useMemo(() => {
    const withLoss = runCheckpoints.filter((c) => c.val_loss != null);
    if (withLoss.length === 0) return null;
    return withLoss.reduce((a, b) =>
      (a.val_loss ?? Infinity) < (b.val_loss ?? Infinity) ? a : b,
    ).path;
  }, [runCheckpoints]);

  return (
    <div className="rounded-xl bg-gray-900 border border-gray-800 p-3">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-1.5">
          <Download className="w-3.5 h-3.5 text-gray-500" />
          <span className="text-xs font-medium text-gray-400">
            Checkpoints
          </span>
        </div>
        {runCheckpoints.length > 0 && (
          <Link
            href={`/inference${selectedRunId ? `?run=${selectedRunId}` : ""}`}
            className="flex items-center gap-1 text-[10px] text-blue-400 hover:text-blue-300 transition-colors"
          >
            <span>Open Explorer</span>
            <ExternalLink className="w-2.5 h-2.5" />
          </Link>
        )}
      </div>

      {runCheckpoints.length === 0 ? (
        <p className="text-xs text-gray-600 py-2">
          {selectedRunId
            ? "No checkpoints for this run"
            : "Select a run to view checkpoints"}
        </p>
      ) : (
        <div className="space-y-1 max-h-[200px] overflow-y-auto">
          {runCheckpoints.map((ckpt) => {
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
