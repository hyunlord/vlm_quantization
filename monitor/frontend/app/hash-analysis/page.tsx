"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { ArrowLeft } from "lucide-react";
import { useWebSocket } from "@/hooks/useWebSocket";
import { useRunContext } from "@/contexts/RunContext";
import RunSelector from "@/components/RunSelector";
import BitBalanceChart from "@/components/BitBalanceChart";
import SampleGallery from "@/components/SampleGallery";
import SimilarityHeatmap from "@/components/SimilarityHeatmap";
import type { HashAnalysisData } from "@/lib/types";

interface SnapshotEntry {
  id: number;
  epoch: number;
  step: number;
  timestamp: number;
}

function getWsUrl() {
  if (typeof window === "undefined") return "ws://localhost:8000/ws";
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}/ws`;
}

const LIVE_VALUE = "live";

export default function HashAnalysisPage() {
  const { selectedRunId } = useRunContext();
  const { hashAnalysis: liveData, isConnected } = useWebSocket(getWsUrl(), selectedRunId);

  const [snapshots, setSnapshots] = useState<SnapshotEntry[]>([]);
  const [selectedId, setSelectedId] = useState<string>(LIVE_VALUE);
  const [displayData, setDisplayData] = useState<HashAnalysisData | null>(null);
  const [loading, setLoading] = useState(false);

  // Load snapshot list on mount and when run changes
  useEffect(() => {
    const qs = selectedRunId
      ? `?run_id=${encodeURIComponent(selectedRunId)}`
      : "";
    fetch(`/api/metrics/hash_analysis/list${qs}`)
      .then((r) => r.json())
      .then((data) => {
        if (data.snapshots?.length) setSnapshots(data.snapshots);
        else setSnapshots([]);
      })
      .catch(() => {});
  }, [selectedRunId]);

  // When new live data arrives via WebSocket, append to snapshot list
  useEffect(() => {
    if (!liveData) return;
    setSnapshots((prev) => {
      const exists = prev.some(
        (s) => s.epoch === liveData.epoch && s.step === liveData.step,
      );
      if (exists) return prev;
      // Append placeholder entry (id will be assigned by DB, use -1)
      return [
        ...prev,
        {
          id: -1,
          epoch: liveData.epoch,
          step: liveData.step,
          timestamp: Date.now() / 1000,
        },
      ];
    });
  }, [liveData]);

  // Fetch specific snapshot when dropdown changes
  const fetchSnapshot = useCallback(async (id: string) => {
    if (id === LIVE_VALUE || id === "-1") return;
    setLoading(true);
    try {
      const res = await fetch(
        `/api/metrics/hash_analysis?id=${encodeURIComponent(id)}`,
      );
      const data = await res.json();
      if (data.hash_analysis) {
        setDisplayData(data.hash_analysis);
      }
    } catch {
      // ignore
    } finally {
      setLoading(false);
    }
  }, []);

  // Determine what to display
  const isLive = selectedId === LIVE_VALUE;
  const currentData = isLive ? liveData : displayData;

  const handleSelect = (value: string) => {
    setSelectedId(value);
    if (value === LIVE_VALUE) {
      setDisplayData(null);
    } else {
      fetchSnapshot(value);
    }
  };

  // Reload snapshot list when switching away from live (to pick up new IDs)
  useEffect(() => {
    if (!isLive) {
      const qs = selectedRunId
        ? `?run_id=${encodeURIComponent(selectedRunId)}`
        : "";
      fetch(`/api/metrics/hash_analysis/list${qs}`)
        .then((r) => r.json())
        .then((data) => {
          if (data.snapshots?.length) setSnapshots(data.snapshots);
        })
        .catch(() => {});
    }
  }, [isLive, selectedRunId]);

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 p-4 md:p-6">
      <div className="max-w-[1200px] mx-auto space-y-4">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Link
              href="/"
              className="text-gray-500 hover:text-gray-300 transition-colors"
            >
              <ArrowLeft size={18} />
            </Link>
            <h1 className="text-lg font-semibold">Hash Analysis</h1>
            <RunSelector />

            {/* Epoch Selector */}
            <select
              value={selectedId}
              onChange={(e) => handleSelect(e.target.value)}
              className="bg-gray-800 border border-gray-700 text-gray-300 text-xs rounded px-2 py-1 focus:outline-none focus:border-blue-500"
            >
              <option value={LIVE_VALUE}>
                Latest (Live)
                {liveData
                  ? ` — Epoch ${liveData.epoch}, Step ${liveData.step}`
                  : ""}
              </option>
              {[...snapshots].reverse().map((s, i) => (
                <option
                  key={`${s.id}-${s.epoch}-${s.step}`}
                  value={s.id === -1 ? "-1" : String(s.id)}
                  disabled={s.id === -1}
                >
                  Epoch {s.epoch}, Step {s.step}
                  {i === 0 && s.id !== -1 ? " (newest)" : ""}
                </option>
              ))}
            </select>

            {loading && (
              <span className="text-xs text-gray-500 animate-pulse">
                Loading...
              </span>
            )}
          </div>
          <div className="flex items-center gap-2">
            <div
              className={`w-2 h-2 rounded-full ${isConnected ? "bg-emerald-400" : "bg-red-400"}`}
            />
            <span className="text-xs text-gray-500">
              {isConnected ? "Connected" : "Disconnected"}
            </span>
          </div>
        </div>

        {!currentData ? (
          <div className="rounded-xl bg-gray-900 p-8 border border-gray-800 text-center">
            <p className="text-gray-500">
              {snapshots.length === 0
                ? "Waiting for validation epoch to complete..."
                : "Select a snapshot from the dropdown above."}
            </p>
            <p className="text-xs text-gray-600 mt-2">
              Hash analysis data is collected at the end of each validation
              epoch.
            </p>
          </div>
        ) : (
          <>
            {/* Bit Balance */}
            <BitBalanceChart
              bitActivations={currentData.bit_activations ?? {}}
            />

            {/* Sample Gallery */}
            <SampleGallery
              samples={currentData.samples ?? []}
              imgCodes={currentData.sample_img_codes ?? []}
              txtCodes={currentData.sample_txt_codes ?? []}
              similarityMatrix={currentData.similarity_matrix ?? []}
            />

            {/* Similarity Heatmap */}
            <SimilarityHeatmap
              matrix={currentData.similarity_matrix ?? []}
              samples={currentData.samples ?? []}
              bit={currentData.bit ?? 64}
            />
          </>
        )}
      </div>
    </div>
  );
}
