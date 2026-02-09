"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type {
  EvalMetric,
  SystemMetric,
  TrainingMetric,
  TrainingStatus,
  WSMessage,
} from "@/lib/types";

const MAX_TRAINING_POINTS = 2000;

export function useWebSocket(url: string) {
  const [isConnected, setIsConnected] = useState(false);
  const [systemData, setSystemData] = useState<SystemMetric | null>(null);
  const [trainingData, setTrainingData] = useState<TrainingMetric[]>([]);
  const [evalData, setEvalData] = useState<EvalMetric[]>([]);
  const [status, setStatus] = useState<TrainingStatus | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeout = useRef<NodeJS.Timeout>(undefined);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => setIsConnected(true);

    ws.onmessage = (event) => {
      const msg: WSMessage = JSON.parse(event.data);

      switch (msg.type) {
        case "system":
          setSystemData(msg.data as SystemMetric);
          break;
        case "training":
          setTrainingData((prev) => {
            const next = [...prev, msg.data as TrainingMetric];
            return next.length > MAX_TRAINING_POINTS
              ? next.slice(-MAX_TRAINING_POINTS)
              : next;
          });
          break;
        case "eval":
          setEvalData((prev) => [...prev, msg.data as EvalMetric]);
          break;
        case "status":
          setStatus(msg.data as TrainingStatus);
          break;
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

  // Load historical data on mount
  useEffect(() => {
    fetch("/api/metrics/history")
      .then((r) => r.json())
      .then((data) => {
        if (data.training?.length) setTrainingData(data.training);
        if (data.eval?.length) setEvalData(data.eval);
      })
      .catch(() => {});

    fetch("/api/training/status")
      .then((r) => r.json())
      .then((data) => setStatus(data))
      .catch(() => {});
  }, []);

  return { isConnected, systemData, trainingData, evalData, status };
}
