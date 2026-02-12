"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { ArrowLeft, Layers } from "lucide-react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import { useWebSocket } from "@/hooks/useWebSocket";
import { useRunContext } from "@/contexts/RunContext";
import RunSelector from "@/components/RunSelector";
import type {
  AugmentationAnalysis,
  AugmentationSample,
  HashAnalysisData,
  HashAnalysisSample,
} from "@/lib/types";

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

function simColor(sim: number): string {
  if (sim >= 0.85) return "text-emerald-400";
  if (sim >= 0.7) return "text-yellow-400";
  return "text-red-400";
}

function pct(v: number): string {
  return (v * 100).toFixed(1);
}

// --- Summary Card ---
function SummaryCard({
  label,
  value,
  sub,
  color,
}: {
  label: string;
  value: string;
  sub?: string;
  color: string;
}) {
  return (
    <div className="rounded-lg bg-gray-800/60 border border-gray-700/50 p-3">
      <div className="text-[10px] text-gray-500 uppercase tracking-wider mb-1">
        {label}
      </div>
      <div className={`text-xl font-mono font-bold ${color}`}>{value}</div>
      {sub && <div className="text-[10px] text-gray-500 mt-0.5">{sub}</div>}
    </div>
  );
}

// --- Bit Stability Chart ---
function BitStabilityChart({
  weakStability,
  strongStability,
  bit,
}: {
  weakStability: number[];
  strongStability: number[];
  bit: number;
}) {
  const data = weakStability.map((w, i) => ({
    bit: i,
    weak: w,
    strong: strongStability[i] ?? 0,
  }));

  return (
    <div className="rounded-xl bg-gray-900 p-4 border border-gray-800">
      <h2 className="text-sm font-semibold text-gray-200 mb-1">
        Per-Bit Stability Under Augmentation
      </h2>
      <p className="text-[10px] text-gray-500 mb-3">
        How often each bit position stays the same when the image is augmented.
        Higher = more robust. {bit}-bit codes, averaged over 3 augmentations per
        sample.
      </p>
      <ResponsiveContainer width="100%" height={200}>
        <AreaChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis
            dataKey="bit"
            tick={{ fontSize: 9, fill: "#6b7280" }}
            label={{
              value: "Bit Position",
              position: "insideBottom",
              offset: -2,
              style: { fontSize: 10, fill: "#6b7280" },
            }}
          />
          <YAxis
            domain={[0.4, 1]}
            tick={{ fontSize: 9, fill: "#6b7280" }}
            tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "#1f2937",
              border: "1px solid #374151",
              borderRadius: "8px",
              fontSize: "11px",
            }}
            formatter={(v) => [
              typeof v === "number" ? `${(v * 100).toFixed(1)}%` : String(v ?? ""),
              "",
            ]}
          />
          <Legend
            wrapperStyle={{ fontSize: "10px" }}
            iconType="circle"
            iconSize={8}
          />
          <Area
            type="monotone"
            dataKey="weak"
            name="Weak Aug"
            stroke="#06b6d4"
            fill="#06b6d4"
            fillOpacity={0.15}
            strokeWidth={1.5}
          />
          <Area
            type="monotone"
            dataKey="strong"
            name="Strong Aug"
            stroke="#f97316"
            fill="#f97316"
            fillOpacity={0.15}
            strokeWidth={1.5}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

// --- Sample Comparison Gallery ---
function AugSampleGallery({
  augSamples,
  origSamples,
  origCodes,
}: {
  augSamples: AugmentationSample[];
  origSamples: HashAnalysisSample[];
  origCodes: number[][];
}) {
  return (
    <div className="rounded-xl bg-gray-900 p-4 border border-gray-800">
      <h2 className="text-sm font-semibold text-gray-200 mb-1">
        Sample Augmentation Comparison
      </h2>
      <p className="text-[10px] text-gray-500 mb-3">
        Original vs. augmented images and their hash code similarity. Each
        sample shows original, weak augmentation, and strong augmentation side by
        side.
      </p>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
        {augSamples.map((aug, i) => {
          const orig = origSamples[i];
          const origCode = origCodes[i] ?? [];
          const weakMatch =
            origCode.length > 0
              ? origCode.filter((b, j) => b === aug.weak_code[j]).length
              : 0;
          const strongMatch =
            origCode.length > 0
              ? origCode.filter((b, j) => b === aug.strong_code[j]).length
              : 0;

          return (
            <div
              key={aug.image_id}
              className="rounded-lg bg-gray-800/60 border border-gray-700/50 overflow-hidden"
            >
              {/* Three thumbnails side by side */}
              <div className="grid grid-cols-3 gap-px bg-gray-700/30">
                <div className="relative">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={orig?.thumbnail}
                    alt="Original"
                    className="w-full aspect-square object-cover"
                  />
                  <span className="absolute bottom-0 left-0 right-0 bg-black/60 text-[8px] text-center text-gray-300 py-0.5">
                    Original
                  </span>
                </div>
                <div className="relative">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={aug.weak_thumbnail}
                    alt="Weak aug"
                    className="w-full aspect-square object-cover"
                  />
                  <span className="absolute bottom-0 left-0 right-0 bg-cyan-900/70 text-[8px] text-center text-cyan-300 py-0.5">
                    Weak
                  </span>
                </div>
                <div className="relative">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={aug.strong_thumbnail}
                    alt="Strong aug"
                    className="w-full aspect-square object-cover"
                  />
                  <span className="absolute bottom-0 left-0 right-0 bg-orange-900/70 text-[8px] text-center text-orange-300 py-0.5">
                    Strong
                  </span>
                </div>
              </div>

              <div className="p-2 space-y-1.5">
                {/* Caption */}
                <p
                  className="text-[10px] text-gray-400 line-clamp-2 leading-tight"
                  title={orig?.caption}
                >
                  {orig?.caption}
                </p>

                {/* Similarity bars */}
                <div className="space-y-1">
                  <div className="flex items-center gap-1.5">
                    <span className="text-[9px] text-cyan-400 w-10 shrink-0">
                      Weak
                    </span>
                    <div className="flex-1 h-1.5 bg-gray-700 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-cyan-500 rounded-full"
                        style={{
                          width: `${aug.weak_mean_sim * 100}%`,
                        }}
                      />
                    </div>
                    <span
                      className={`text-[9px] font-mono w-10 text-right ${simColor(aug.weak_mean_sim)}`}
                    >
                      {pct(aug.weak_mean_sim)}%
                    </span>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <span className="text-[9px] text-orange-400 w-10 shrink-0">
                      Strong
                    </span>
                    <div className="flex-1 h-1.5 bg-gray-700 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-orange-500 rounded-full"
                        style={{
                          width: `${aug.strong_mean_sim * 100}%`,
                        }}
                      />
                    </div>
                    <span
                      className={`text-[9px] font-mono w-10 text-right ${simColor(aug.strong_mean_sim)}`}
                    >
                      {pct(aug.strong_mean_sim)}%
                    </span>
                  </div>
                </div>

                {/* Bit match info */}
                <div className="flex items-center justify-between text-[9px] text-gray-600">
                  <span>ID: {aug.image_id}</span>
                  <span>
                    {weakMatch}/{origCode.length} |{" "}
                    {strongMatch}/{origCode.length}
                  </span>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// --- Distribution Summary ---
function DistributionSummary({
  augSamples,
}: {
  augSamples: AugmentationSample[];
}) {
  if (augSamples.length === 0) return null;

  // Bucket similarities into histogram bins
  const bins = [
    { label: "<60%", min: 0, max: 0.6 },
    { label: "60-70%", min: 0.6, max: 0.7 },
    { label: "70-80%", min: 0.7, max: 0.8 },
    { label: "80-90%", min: 0.8, max: 0.9 },
    { label: "90-100%", min: 0.9, max: 1.01 },
  ];

  const weakDist = bins.map((b) => ({
    ...b,
    count: augSamples.filter(
      (s) => s.weak_mean_sim >= b.min && s.weak_mean_sim < b.max,
    ).length,
  }));
  const strongDist = bins.map((b) => ({
    ...b,
    count: augSamples.filter(
      (s) => s.strong_mean_sim >= b.min && s.strong_mean_sim < b.max,
    ).length,
  }));

  return (
    <div className="rounded-xl bg-gray-900 p-4 border border-gray-800">
      <h2 className="text-sm font-semibold text-gray-200 mb-3">
        Similarity Distribution
      </h2>
      <div className="grid grid-cols-2 gap-6">
        {/* Weak Distribution */}
        <div>
          <h3 className="text-[10px] text-cyan-400 uppercase tracking-wider mb-2">
            Weak Augmentation
          </h3>
          <div className="space-y-1">
            {weakDist.map((b) => (
              <div key={b.label} className="flex items-center gap-2">
                <span className="text-[9px] text-gray-500 w-12 text-right">
                  {b.label}
                </span>
                <div className="flex-1 h-3 bg-gray-800 rounded overflow-hidden">
                  <div
                    className="h-full bg-cyan-600/60 rounded"
                    style={{
                      width: `${(b.count / augSamples.length) * 100}%`,
                    }}
                  />
                </div>
                <span className="text-[9px] text-gray-500 w-4">{b.count}</span>
              </div>
            ))}
          </div>
        </div>
        {/* Strong Distribution */}
        <div>
          <h3 className="text-[10px] text-orange-400 uppercase tracking-wider mb-2">
            Strong Augmentation
          </h3>
          <div className="space-y-1">
            {strongDist.map((b) => (
              <div key={b.label} className="flex items-center gap-2">
                <span className="text-[9px] text-gray-500 w-12 text-right">
                  {b.label}
                </span>
                <div className="flex-1 h-3 bg-gray-800 rounded overflow-hidden">
                  <div
                    className="h-full bg-orange-600/60 rounded"
                    style={{
                      width: `${(b.count / augSamples.length) * 100}%`,
                    }}
                  />
                </div>
                <span className="text-[9px] text-gray-500 w-4">{b.count}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

// === Main Page ===
export default function AugmentationPage() {
  const { selectedRunId } = useRunContext();
  const { hashAnalysis: liveData, isConnected } = useWebSocket(
    getWsUrl(),
    selectedRunId,
  );

  const [snapshots, setSnapshots] = useState<SnapshotEntry[]>([]);
  const [selectedId, setSelectedId] = useState<string>(LIVE_VALUE);
  const [displayData, setDisplayData] = useState<HashAnalysisData | null>(null);
  const [loading, setLoading] = useState(false);

  // Load snapshot list
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

  // Append live snapshot
  useEffect(() => {
    if (!liveData) return;
    setSnapshots((prev) => {
      const exists = prev.some(
        (s) => s.epoch === liveData.epoch && s.step === liveData.step,
      );
      if (exists) return prev;
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

  // Fetch specific snapshot
  const fetchSnapshot = useCallback(async (id: string) => {
    if (id === LIVE_VALUE || id === "-1") return;
    setLoading(true);
    try {
      const res = await fetch(
        `/api/metrics/hash_analysis?id=${encodeURIComponent(id)}`,
      );
      const data = await res.json();
      if (data.hash_analysis) setDisplayData(data.hash_analysis);
    } catch {
      /* ignore */
    } finally {
      setLoading(false);
    }
  }, []);

  const isLive = selectedId === LIVE_VALUE;
  const currentData = isLive ? liveData : displayData;
  const augData: AugmentationAnalysis | undefined = currentData?.augmentation;

  const handleSelect = (value: string) => {
    setSelectedId(value);
    if (value === LIVE_VALUE) setDisplayData(null);
    else fetchSnapshot(value);
  };

  // Reload snapshot list when switching from live
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
            <Layers className="w-4 h-4 text-cyan-400" />
            <h1 className="text-lg font-semibold">Augmentation Robustness</h1>
            <RunSelector />

            {/* Snapshot Selector */}
            <select
              value={selectedId}
              onChange={(e) => handleSelect(e.target.value)}
              className="bg-gray-800 border border-gray-700 text-gray-300 text-xs rounded px-2 py-1 focus:outline-none focus:border-cyan-500"
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

        {/* No data state */}
        {!currentData ? (
          <div className="rounded-xl bg-gray-900 p-8 border border-gray-800 text-center">
            <p className="text-gray-500">
              Waiting for validation epoch with augmentation analysis...
            </p>
            <p className="text-xs text-gray-600 mt-2">
              Augmentation robustness data is computed during validation by
              comparing original and augmented image hash codes.
            </p>
          </div>
        ) : !augData ? (
          <div className="rounded-xl bg-gray-900 p-8 border border-gray-800 text-center">
            <p className="text-gray-500">
              No augmentation analysis in this snapshot.
            </p>
            <p className="text-xs text-gray-600 mt-2">
              This feature requires the latest callback version. Augmentation
              analysis is computed automatically during validation.
            </p>
          </div>
        ) : (
          <>
            {/* Summary Cards */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <SummaryCard
                label="Weak Aug Similarity"
                value={`${pct(augData.weak_mean_overall)}%`}
                sub="Avg across all samples"
                color={simColor(augData.weak_mean_overall)}
              />
              <SummaryCard
                label="Strong Aug Similarity"
                value={`${pct(augData.strong_mean_overall)}%`}
                sub="Avg across all samples"
                color={simColor(augData.strong_mean_overall)}
              />
              <SummaryCard
                label="Bit Level"
                value={`${augData.bit}-bit`}
                sub={`${augData.n_augs} augmentations each`}
                color="text-gray-200"
              />
              <SummaryCard
                label="Samples"
                value={String(augData.samples.length)}
                sub="Fixed validation images"
                color="text-gray-200"
              />
            </div>

            {/* Bit Stability Chart */}
            <BitStabilityChart
              weakStability={augData.weak_bit_stability}
              strongStability={augData.strong_bit_stability}
              bit={augData.bit}
            />

            {/* Sample Gallery */}
            <AugSampleGallery
              augSamples={augData.samples}
              origSamples={currentData.samples ?? []}
              origCodes={currentData.sample_img_codes ?? []}
            />

            {/* Distribution Summary */}
            <DistributionSummary augSamples={augData.samples} />
          </>
        )}
      </div>
    </div>
  );
}
