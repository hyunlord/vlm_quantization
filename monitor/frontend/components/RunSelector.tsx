"use client";

import { useRunContext } from "@/contexts/RunContext";

function formatRunId(runId: string): string {
  // Parse YYYYMMDD_HHMMSS format (server local time from datetime.now())
  const m = runId.match(/^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})$/);
  if (!m) return runId;
  const year = m[1];
  const month = parseInt(m[2]);
  const day = parseInt(m[3]);
  const hour = m[4];
  const minute = m[5];
  return `${year}년 ${month}월 ${day}일 ${hour}:${minute}`;
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

      {/* Eval point selector (step-based for uniqueness) */}
      {evalPoints.length > 0 && (
        <select
          value={evalPoints.find((ep) => ep.epoch === selectedEvalEpoch)?.step ?? ""}
          onChange={(e) => {
            const step = Number(e.target.value);
            const ep = evalPoints.find((p) => (p.step ?? p.epoch) === step);
            if (ep) selectEvalEpoch(ep.epoch);
          }}
          className="bg-gray-800 border border-gray-700 text-gray-300 text-xs rounded px-2 py-1 focus:outline-none focus:border-blue-500"
        >
          {evalPoints.map((ep) => (
            <option key={`${ep.epoch}-${ep.step}`} value={ep.step ?? ep.epoch}>
              Epoch {ep.epoch}, Step {ep.step ?? 0}
            </option>
          ))}
        </select>
      )}
    </div>
  );
}
