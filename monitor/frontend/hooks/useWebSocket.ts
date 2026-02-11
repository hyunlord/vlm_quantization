"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type {
  EvalMetric,
  HashAnalysisData,
  SystemMetric,
  TrainingMetric,
  TrainingStatus,
  WSMessage,
} from "@/lib/types";

const MAX_TRAINING_POINTS = 2000;

export function useWebSocket(url: string, runId?: string | null) {
  const [isConnected, setIsConnected] = useState(false);
  const [systemData, setSystemData] = useState<SystemMetric | null>(null);
  const [trainingData, setTrainingData] = useState<TrainingMetric[]>([]);
  const [evalData, setEvalData] = useState<EvalMetric[]>([]);
  const [status, setStatus] = useState<TrainingStatus | null>(null);
  const [hashAnalysis, setHashAnalysis] = useState<HashAnalysisData | null>(
    null,
  );
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeout = useRef<NodeJS.Timeout>(undefined);
  const runIdRef = useRef(runId);
  runIdRef.current = runId;

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    let ws: WebSocket;
    try {
      ws = new WebSocket(url);
    } catch {
      // WebSocket blocked (e.g. mixed content ws:// from https://)
      reconnectTimeout.current = setTimeout(connect, 5000);
      return;
    }
    wsRef.current = ws;

    ws.onopen = () => setIsConnected(true);

    ws.onmessage = (event) => {
      try {
        const msg: WSMessage = JSON.parse(event.data);

        switch (msg.type) {
          case "system":
            setSystemData(msg.data as SystemMetric);
            break;
          case "training": {
            const tm = msg.data as TrainingMetric;
            // Filter by run_id if set
            if (runIdRef.current && tm.run_id && tm.run_id !== runIdRef.current)
              break;
            setTrainingData((prev) => {
              const next = [...prev, tm];
              return next.length > MAX_TRAINING_POINTS
                ? next.slice(-MAX_TRAINING_POINTS)
                : next;
            });
            break;
          }
          case "eval": {
            const em = msg.data as EvalMetric;
            if (runIdRef.current && em.run_id && em.run_id !== runIdRef.current)
              break;
            setEvalData((prev) => [...prev, em]);
            break;
          }
          case "status": {
            const s = msg.data as TrainingStatus;
            setStatus(s);
            break;
          }
          case "hash_analysis":
            setHashAnalysis(msg.data as HashAnalysisData);
            break;
        }
      } catch {
        // Ignore malformed messages
      }
    };

    ws.onclose = () => {
      setIsConnected(false);
      reconnectTimeout.current = setTimeout(connect, 3000);
    };

    ws.onerror = () => ws.close();
  }, [url]);

  useEffect(() => {
    connect();
    return () => {
      clearTimeout(reconnectTimeout.current);
      wsRef.current?.close();
    };
  }, [connect]);

  // Load historical data when runId changes
  useEffect(() => {
    const qs = runId ? `?run_id=${encodeURIComponent(runId)}` : "";
    fetch(`/api/metrics/history${qs}`)
      .then((r) => r.json())
      .then((data) => {
        if (data.training?.length) setTrainingData(data.training);
        else setTrainingData([]);
        if (data.eval?.length) setEvalData(data.eval);
        else setEvalData([]);
      })
      .catch(() => {});

    fetch("/api/training/status")
      .then((r) => r.json())
      .then((data) => setStatus(data))
      .catch(() => {});
  }, [runId]);

  // Load hash analysis on mount
  useEffect(() => {
    fetch("/api/metrics/hash_analysis")
      .then((r) => r.json())
      .then((data) => {
        if (data.hash_analysis) setHashAnalysis(data.hash_analysis);
      })
      .catch(() => {});
  }, []);

  return {
    isConnected,
    systemData,
    trainingData,
    evalData,
    status,
    hashAnalysis,
  };
}
