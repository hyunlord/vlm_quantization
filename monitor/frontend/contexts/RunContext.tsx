"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";
import type { CheckpointInfo, EvalMetric, RunInfo } from "@/lib/types";

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

  // Load eval points when run changes
  useEffect(() => {
    if (!selectedRunId) {
      setEvalPoints([]);
      setSelectedEvalEpoch(null);
      return;
    }
    fetch(`/api/metrics/history?run_id=${encodeURIComponent(selectedRunId)}`)
      .then((r) => r.json())
      .then((data) => {
        const evals: EvalMetric[] = data.eval ?? [];
        setEvalPoints(evals);
        // Auto-select latest eval epoch
        if (evals.length > 0) {
          setSelectedEvalEpoch(evals[evals.length - 1].epoch);
        } else {
          setSelectedEvalEpoch(null);
        }
      })
      .catch(() => {
        setEvalPoints([]);
        setSelectedEvalEpoch(null);
      });
  }, [selectedRunId]);

  const selectRun = useCallback((id: string | null) => {
    setSelectedRunId(id);
    setSelectedEvalEpoch(null);
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
      }}
    >
      {children}
    </RunContext.Provider>
  );
}
