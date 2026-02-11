"use client";

import { useRunContext } from "@/contexts/RunContext";

function formatRunId(runId: string): string {
  // Parse YYYYMMDD_HHMMSS format (server time, assumed UTC)
  // Convert to KST (UTC+9) for display
  const m = runId.match(/^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})$/);
  if (!m) return runId;
  // Create date assuming UTC, then format in KST
  const utcDate = new Date(Date.UTC(
    parseInt(m[1]), parseInt(m[2]) - 1, parseInt(m[3]),
    parseInt(m[4]), parseInt(m[5]), parseInt(m[6])
  ));
  return utcDate.toLocaleString("ko-KR", {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
    timeZone: "Asia/Seoul",
  });
}

export default function RunSelector() {
  const {
    runs,
    selectedRunId,
    selectRun,
    evalPoints,
    selectedEvalEpoch,
    selectEvalEpoch,
    checkpointsByRun,
  } = useRunContext();

  if (runs.length === 0) return null;

  return (
    <div className="flex items-center gap-2">
      {/* Run selector */}
      <select
        value={selectedRunId ?? ""}
        onChange={(e) => selectRun(e.target.value || null)}
        className="bg-gray-800 border border-gray-700 text-gray-300 text-xs rounded px-2 py-1 focus:outline-none focus:border-blue-500"
      >
        {runs.map((run) => {
          const ckptCount = checkpointsByRun[run.run_id]?.length ?? 0;
          return (
            <option key={run.run_id} value={run.run_id}>
              {formatRunId(run.run_id)} — {run.num_eval_points} evals
              {ckptCount > 0 ? `, ${ckptCount} ckpts` : ""}
            </option>
          );
        })}
      </select>

      {/* Eval epoch selector */}
      {evalPoints.length > 0 && (
        <select
          value={selectedEvalEpoch ?? ""}
          onChange={(e) => {
            const v = e.target.value;
            selectEvalEpoch(v ? Number(v) : null);
          }}
          className="bg-gray-800 border border-gray-700 text-gray-300 text-xs rounded px-2 py-1 focus:outline-none focus:border-blue-500"
        >
          {evalPoints.map((ep) => (
            <option key={`${ep.epoch}-${ep.step}`} value={ep.epoch}>
              Epoch {ep.epoch}
              {ep.step != null ? `, Step ${ep.step}` : ""}
            </option>
          ))}
        </select>
      )}
    </div>
  );
}
