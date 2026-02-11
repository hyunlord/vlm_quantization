"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";
import type { Checkpoint, CheckpointInfo, EpochSummary, EvalMetric, RunInfo } from "@/lib/types";

interface RunContextType {
  runs: RunInfo[];
  selectedRunId: string | null;
  selectRun: (id: string | null) => void;
  evalPoints: EvalMetric[];
  selectedEvalEpoch: number | null;
  selectEvalEpoch: (epoch: number | null) => void;
  refreshRuns: () => Promise<void>;
  checkpoints: CheckpointInfo[];
  checkpointsByRun: Record<string, CheckpointInfo[]>;
  refreshCheckpoints: () => Promise<void>;
  // New: epoch/checkpoint hierarchy
  epochSummaries: EpochSummary[];
  selectedCheckpointId: number | null;
  selectCheckpoint: (id: number | null) => void;
  runCheckpoints: Checkpoint[];
  loadCheckpointForInference: (path: string) => Promise<void>;
}

const RunContext = createContext<RunContextType>({
  runs: [],
  selectedRunId: null,
  selectRun: () => {},
  evalPoints: [],
  selectedEvalEpoch: null,
  selectEvalEpoch: () => {},
  refreshRuns: async () => {},
  checkpoints: [],
  checkpointsByRun: {},
  refreshCheckpoints: async () => {},
  // New: epoch/checkpoint hierarchy
  epochSummaries: [],
  selectedCheckpointId: null,
  selectCheckpoint: () => {},
  runCheckpoints: [],
  loadCheckpointForInference: async () => {},
});

export function useRunContext() {
  return useContext(RunContext);
}

export function RunProvider({ children }: { children: React.ReactNode }) {
  const [runs, setRuns] = useState<RunInfo[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [evalPoints, setEvalPoints] = useState<EvalMetric[]>([]);
  const [selectedEvalEpoch, setSelectedEvalEpoch] = useState<number | null>(
    null,
  );
  const [checkpoints, setCheckpoints] = useState<CheckpointInfo[]>([]);
  // New state for epoch/checkpoint hierarchy
  const [epochSummaries, setEpochSummaries] = useState<EpochSummary[]>([]);
  const [selectedCheckpointId, setSelectedCheckpointId] = useState<number | null>(null);
  const [runCheckpoints, setRunCheckpoints] = useState<Checkpoint[]>([]);

  const checkpointsByRun = useMemo(() => {
    const groups: Record<string, CheckpointInfo[]> = {};
    for (const ckpt of checkpoints) {
      (groups[ckpt.run_dir] ??= []).push(ckpt);
    }
    return groups;
  }, [checkpoints]);

  const refreshCheckpoints = useCallback(async () => {
    try {
      const res = await fetch("/api/inference/checkpoints");
      const data = await res.json();
      if (data.checkpoints?.length) setCheckpoints(data.checkpoints);
      else setCheckpoints([]);
    } catch {
      // ignore
    }
  }, []);

  const refreshRuns = useCallback(async () => {
    try {
      const res = await fetch("/api/runs");
      const data = await res.json();
      const fetched: RunInfo[] = data.runs ?? [];
      setRuns(fetched);
      // Auto-select latest run if none selected
      if (fetched.length > 0 && !selectedRunId) {
        setSelectedRunId(fetched[0].run_id);
      }
    } catch {
      // ignore
    }
  }, [selectedRunId]);

  // Load runs and checkpoints on mount
  useEffect(() => {
    refreshRuns();
    refreshCheckpoints();
    // Refresh every 30s to detect new runs/checkpoints
    const id = setInterval(() => {
      refreshRuns();
      refreshCheckpoints();
    }, 30000);
    return () => clearInterval(id);
  }, [refreshRuns, refreshCheckpoints]);

  // Load eval points when run changes + refresh periodically
  const refreshEvalPoints = useCallback(() => {
    if (!selectedRunId) return;
    fetch(`/api/metrics/history?run_id=${encodeURIComponent(selectedRunId)}`)
      .then((r) => r.json())
      .then((data) => {
        const evals: EvalMetric[] = data.eval ?? [];
        setEvalPoints((prev) => {
          // Only update if data actually changed (new entries)
          if (evals.length !== prev.length) {
            // Auto-select latest eval epoch when new data arrives
            if (evals.length > 0) {
              setSelectedEvalEpoch(evals[evals.length - 1].epoch);
            }
            return evals;
          }
          return prev;
        });
      })
      .catch(() => {});
  }, [selectedRunId]);

  useEffect(() => {
    if (!selectedRunId) {
      setEvalPoints([]);
      setSelectedEvalEpoch(null);
      return;
    }
    // Initial fetch
    refreshEvalPoints();
    // Refresh every 15s to pick up new validation results
    const id = setInterval(refreshEvalPoints, 15000);
    return () => clearInterval(id);
  }, [selectedRunId, refreshEvalPoints]);

  // Load epoch summaries and run checkpoints when run changes
  useEffect(() => {
    if (!selectedRunId) {
      setEpochSummaries([]);
      setRunCheckpoints([]);
      setSelectedCheckpointId(null);
      return;
    }
    // Fetch epoch summaries
    fetch(`/api/runs/${encodeURIComponent(selectedRunId)}/epochs`)
      .then((r) => r.json())
      .then((data) => {
        const epochs: EpochSummary[] = data.epochs ?? [];
        setEpochSummaries(epochs);
        // Auto-select best checkpoint if available
        const withCkpt = epochs.filter((e) => e.checkpoint);
        if (withCkpt.length > 0) {
          // Find epoch with lowest val_loss
          const best = withCkpt.reduce((a, b) =>
            (a.checkpoint?.val_loss ?? Infinity) < (b.checkpoint?.val_loss ?? Infinity) ? a : b
          );
          if (best.checkpoint) {
            setSelectedCheckpointId(best.checkpoint.id);
          }
        }
      })
      .catch(() => setEpochSummaries([]));

    // Fetch checkpoints for this run
    fetch(`/api/runs/${encodeURIComponent(selectedRunId)}/checkpoints`)
      .then((r) => r.json())
      .then((data) => {
        setRunCheckpoints(data.checkpoints ?? []);
      })
      .catch(() => setRunCheckpoints([]));
  }, [selectedRunId]);

  const selectRun = useCallback((id: string | null) => {
    setSelectedRunId(id);
    setSelectedEvalEpoch(null);
    setSelectedCheckpointId(null);
  }, []);

  const selectCheckpoint = useCallback((id: number | null) => {
    setSelectedCheckpointId(id);
    // Also update selectedEvalEpoch to match checkpoint's epoch
    if (id !== null) {
      const ckpt = runCheckpoints.find((c) => c.id === id);
      if (ckpt) {
        setSelectedEvalEpoch(ckpt.epoch);
      }
    }
  }, [runCheckpoints]);

  const loadCheckpointForInference = useCallback(async (path: string) => {
    try {
      await fetch("/api/inference/load", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ checkpoint_path: path }),
      });
    } catch {
      // ignore
    }
  }, []);

  return (
    <RunContext.Provider
      value={{
        runs,
        selectedRunId,
        selectRun,
        evalPoints,
        selectedEvalEpoch,
        selectEvalEpoch: setSelectedEvalEpoch,
        refreshRuns,
        checkpoints,
        checkpointsByRun,
        refreshCheckpoints,
        // New: epoch/checkpoint hierarchy
        epochSummaries,
        selectedCheckpointId,
        selectCheckpoint,
        runCheckpoints,
        loadCheckpointForInference,
      }}
    >
      {children}
    </RunContext.Provider>
  );
}
