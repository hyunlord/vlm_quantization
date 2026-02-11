"use client";

import { useRunContext } from "@/contexts/RunContext";

function formatRunId(runId: string): string {
  const m = runId.match(/^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})$/);
  if (!m) return runId;
  const months = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
  ];
  return `${months[parseInt(m[2]) - 1]} ${parseInt(m[3])}, ${m[1]} ${m[4]}:${m[5]}`;
}

export default function RunSelector() {
  const {
    runs,
    selectedRunId,
    selectRun,
    evalPoints,
    selectedEvalEpoch,
    selectEvalEpoch,
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
        {runs.map((run) => (
          <option key={run.run_id} value={run.run_id}>
            {formatRunId(run.run_id)} ({run.num_eval_points} evals)
          </option>
        ))}
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
