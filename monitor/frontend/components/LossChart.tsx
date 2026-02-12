"use client";

import { useMemo, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Brush,
} from "recharts";
import type { EvalMetric, TrainingMetric } from "@/lib/types";

interface Props {
  data: TrainingMetric[];
  evalData?: EvalMetric[];
}

const LOSS_LINES = [
  { key: "total", color: "#3b82f6", label: "Total" },
  { key: "contrastive", color: "#8b5cf6", label: "Contrastive" },
  { key: "quantization", color: "#f59e0b", label: "Quantization" },
  { key: "balance", color: "#10b981", label: "Balance" },
  { key: "consistency", color: "#ec4899", label: "Consistency" },
  { key: "ortho", color: "#06b6d4", label: "Ortho" },
  { key: "lcs", color: "#f97316", label: "LCS" },
  { key: "distillation", color: "#a855f7", label: "Distill" },
] as const;

function formatStep(v: number) {
  return v >= 1000 ? `${(v / 1000).toFixed(1)}k` : String(v);
}

function formatY(v: number) {
  return v < 0.01 ? v.toExponential(0) : v.toFixed(2);
}

const tooltipStyle = {
  backgroundColor: "#111827",
  border: "1px solid #374151",
  borderRadius: "8px",
  fontSize: "11px",
};

export default function LossChart({ data, evalData }: Props) {
  const [visible, setVisible] = useState<Record<string, boolean>>(() =>
    Object.fromEntries(LOSS_LINES.map((l) => [l.key, true])),
  );
  const [logScale, setLogScale] = useState(false);

  const toggle = (key: string) =>
    setVisible((prev) => ({ ...prev, [key]: !prev[key] }));

  // --- Training chart data ---
  const trainData = useMemo(
    () =>
      data.map((d) => ({
        step: d.step,
        total: d.loss_total,
        contrastive: d.loss_contrastive,
        quantization: d.loss_quantization,
        balance: d.loss_balance,
        consistency: d.loss_consistency,
        ortho: d.loss_ortho,
        lcs: d.loss_lcs,
        distillation: d.loss_distillation,
      })),
    [data],
  );

  // --- Validation chart data ---
  const valData = useMemo(() => {
    if (!evalData?.length) return [];
    return evalData
      .filter((e) => e.val_loss_total != null)
      .map((e) => ({
        step: e.step ?? e.epoch,
        total: e.val_loss_total,
        contrastive: e.val_loss_contrastive,
        quantization: e.val_loss_quantization,
        balance: e.val_loss_balance,
        consistency: e.val_loss_consistency,
        ortho: e.val_loss_ortho,
        lcs: e.val_loss_lcs,
        distillation: e.val_loss_distillation,
      }));
  }, [evalData]);

  // --- Y-axis domains ---
  function computeYDomain(
    chartData: Record<string, number | null | undefined>[],
  ) {
    if (logScale) return [0.001, "auto"] as const;
    if (chartData.length === 0) return [0, "auto"] as const;
    const activeKeys = LOSS_LINES.filter((l) => visible[l.key]).map(
      (l) => l.key,
    );
    if (activeKeys.length === 0) return [0, "auto"] as const;
    let max = 0;
    for (const d of chartData) {
      for (const k of activeKeys) {
        const v = d[k];
        if (v != null && typeof v === "number" && v > max) max = v;
      }
    }
    return [0, max > 0 ? max * 1.1 : "auto"] as const;
  }

  const trainYDomain = useMemo(
    () => computeYDomain(trainData),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [trainData, visible, logScale],
  );
  const valYDomain = useMemo(
    () => computeYDomain(valData),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [valData, visible, logScale],
  );

  return (
    <div className="rounded-xl bg-gray-900 p-4 border border-gray-800 space-y-4">
      {/* Header + toggles */}
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold text-gray-200">Loss Curves</h2>
        <div className="flex gap-1.5 flex-wrap items-center">
          <button
            onClick={() => setLogScale((p) => !p)}
            className={`text-[10px] px-1.5 py-0.5 rounded border transition-all mr-1 ${
              logScale
                ? "border-white text-white opacity-100"
                : "border-gray-700 text-gray-500 opacity-60"
            }`}
          >
            Log
          </button>
          {LOSS_LINES.map((l) => (
            <button
              key={l.key}
              onClick={() => toggle(l.key)}
              className={`text-[10px] px-1.5 py-0.5 rounded border transition-all ${
                visible[l.key]
                  ? "border-current opacity-100"
                  : "border-gray-700 opacity-40"
              }`}
              style={{ color: l.color }}
            >
              {l.label}
            </button>
          ))}
        </div>
      </div>

      {/* Training losses */}
      <div>
        <p className="text-[10px] text-gray-500 mb-1">Training</p>
        {data.length === 0 ? (
          <div className="h-48 flex items-center justify-center text-gray-600 text-sm">
            Waiting for training data...
          </div>
        ) : (
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={trainData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
              <XAxis
                dataKey="step"
                type="number"
                domain={["dataMin", "dataMax"]}
                tick={{ fontSize: 10, fill: "#6b7280" }}
                stroke="#374151"
                tickFormatter={formatStep}
              />
              <YAxis
                scale={logScale ? "log" : "auto"}
                domain={trainYDomain}
                allowDataOverflow
                tick={{ fontSize: 10, fill: "#6b7280" }}
                stroke="#374151"
                tickFormatter={formatY}
              />
              <Tooltip
                contentStyle={tooltipStyle}
                labelStyle={{ color: "#9ca3af" }}
                labelFormatter={(v) => `Step ${formatStep(v as number)}`}
              />
              <Legend
                wrapperStyle={{ fontSize: "11px" }}
                onClick={(e) => toggle(e.dataKey as string)}
              />
              {LOSS_LINES.map((l) =>
                visible[l.key] ? (
                  <Line
                    key={l.key}
                    type="monotone"
                    dataKey={l.key}
                    stroke={l.color}
                    dot={false}
                    strokeWidth={l.key === "total" ? 2 : 1.2}
                    name={l.label}
                    isAnimationActive={false}
                  />
                ) : null,
              )}
              <Brush
                dataKey="step"
                height={20}
                stroke="#374151"
                fill="#111827"
                travellerWidth={8}
              />
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>

      {/* Validation losses */}
      {valData.length > 0 && (
        <div>
          <p className="text-[10px] text-gray-500 mb-1">Validation</p>
          <ResponsiveContainer width="100%" height={180}>
            <LineChart data={valData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
              <XAxis
                dataKey="step"
                type="number"
                domain={["dataMin", "dataMax"]}
                tick={{ fontSize: 10, fill: "#6b7280" }}
                stroke="#374151"
                tickFormatter={formatStep}
              />
              <YAxis
                scale={logScale ? "log" : "auto"}
                domain={valYDomain}
                allowDataOverflow
                tick={{ fontSize: 10, fill: "#6b7280" }}
                stroke="#374151"
                tickFormatter={formatY}
              />
              <Tooltip
                contentStyle={tooltipStyle}
                labelStyle={{ color: "#9ca3af" }}
                labelFormatter={(v) => `Step ${formatStep(v as number)}`}
              />
              {LOSS_LINES.map((l) =>
                visible[l.key] ? (
                  <Line
                    key={l.key}
                    type="monotone"
                    dataKey={l.key}
                    stroke={l.color}
                    dot={{ r: 3, fill: l.color }}
                    strokeWidth={l.key === "total" ? 2 : 1.2}
                    connectNulls
                    name={l.label}
                    isAnimationActive={false}
                  />
                ) : null,
              )}
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
