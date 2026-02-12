"use client";

import Link from "next/link";
import { BarChart3, Database, FlaskConical, Layers, Search, Wifi, WifiOff } from "lucide-react";
import { useWebSocket } from "@/hooks/useWebSocket";
import { useRunContext } from "@/contexts/RunContext";
import RunSelector from "@/components/RunSelector";
import TrainingStatus from "@/components/TrainingStatus";
import SystemPanel from "@/components/SystemPanel";
import LossChart from "@/components/LossChart";
import MetricsChart from "@/components/MetricsChart";
import HashQuality from "@/components/HashQuality";
import CheckpointPanel from "@/components/CheckpointPanel";
import EpochNavigator from "@/components/EpochNavigator";

function getWsUrl() {
  if (typeof window === "undefined") return "ws://localhost:8000/ws";
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}/ws`;
}

export default function Dashboard() {
  const { selectedRunId } = useRunContext();
  const { isConnected, systemData, trainingData, evalData, status } =
    useWebSocket(getWsUrl(), selectedRunId);

  const latestMetric = trainingData[trainingData.length - 1];

  return (
    <div className="min-h-screen p-4 max-w-[1400px] mx-auto space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <h1 className="text-lg font-bold text-gray-200">
            VLM Quantization Monitor
          </h1>
          <RunSelector />
        </div>
        <div className="flex items-center gap-3">
          <Link
            href="/hash-analysis"
            className="flex items-center gap-1.5 text-xs text-gray-400 hover:text-purple-400 transition-colors"
          >
            <BarChart3 className="w-3.5 h-3.5" />
            <span>Hash Analysis</span>
          </Link>
          <Link
            href="/augmentation"
            className="flex items-center gap-1.5 text-xs text-gray-400 hover:text-cyan-400 transition-colors"
          >
            <Layers className="w-3.5 h-3.5" />
            <span>Augmentation</span>
          </Link>
          <Link
            href="/inference"
            className="flex items-center gap-1.5 text-xs text-gray-400 hover:text-blue-400 transition-colors"
          >
            <Search className="w-3.5 h-3.5" />
            <span>Hash Explorer</span>
          </Link>
          <Link
            href="/search"
            className="flex items-center gap-1.5 text-xs text-gray-400 hover:text-emerald-400 transition-colors"
          >
            <Database className="w-3.5 h-3.5" />
            <span>Search</span>
          </Link>
          <Link
            href="/optuna"
            className="flex items-center gap-1.5 text-xs text-gray-400 hover:text-amber-400 transition-colors"
          >
            <FlaskConical className="w-3.5 h-3.5" />
            <span>Optuna</span>
          </Link>
          {isConnected ? (
            <div className="flex items-center gap-1.5 text-emerald-400 text-xs">
              <Wifi className="w-3.5 h-3.5" />
              <span>Connected</span>
            </div>
          ) : (
            <div className="flex items-center gap-1.5 text-red-400 text-xs">
              <WifiOff className="w-3.5 h-3.5" />
              <span>Disconnected</span>
            </div>
          )}
        </div>
      </div>

      {/* Training Status Bar */}
      <TrainingStatus status={status} latestMetric={latestMetric} />

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-[280px_1fr] gap-4">
        {/* Sidebar: System + Epochs + Checkpoints */}
        <div className="space-y-4">
          <SystemPanel data={systemData} />
          <EpochNavigator />
          <CheckpointPanel />
        </div>

        {/* Main content */}
        <div className="space-y-4">
          {/* Loss Curves */}
          <LossChart data={trainingData} evalData={evalData} />

          {/* Bottom row */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <MetricsChart data={evalData} />
            <HashQuality data={evalData} />
          </div>
        </div>
      </div>
    </div>
  );
}
