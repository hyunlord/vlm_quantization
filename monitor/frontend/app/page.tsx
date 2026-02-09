"use client";

import Link from "next/link";
import { Search, Wifi, WifiOff } from "lucide-react";
import { useWebSocket } from "@/hooks/useWebSocket";
import TrainingStatus from "@/components/TrainingStatus";
import SystemPanel from "@/components/SystemPanel";
import LossChart from "@/components/LossChart";
import MetricsChart from "@/components/MetricsChart";
import HashQuality from "@/components/HashQuality";

function getWsUrl() {
  if (typeof window === "undefined") return "ws://localhost:8000/ws";
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}/ws`;
}

export default function Dashboard() {
  const { isConnected, systemData, trainingData, evalData, status } =
    useWebSocket(getWsUrl());

  const latestMetric = trainingData[trainingData.length - 1];

  return (
    <div className="min-h-screen p-4 max-w-[1400px] mx-auto space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-lg font-bold text-gray-200">
          VLM Quantization Monitor
        </h1>
        <div className="flex items-center gap-3">
          <Link
            href="/inference"
            className="flex items-center gap-1.5 text-xs text-gray-400 hover:text-blue-400 transition-colors"
          >
            <Search className="w-3.5 h-3.5" />
            <span>Hash Explorer</span>
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
        {/* Sidebar: System */}
        <SystemPanel data={systemData} />

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
