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

const TRAIN_LINES = [
  { key: "loss_total", color: "#3b82f6", label: "Total" },
  { key: "loss_contrastive", color: "#8b5cf6", label: "Contrastive" },
  { key: "loss_quantization", color: "#f59e0b", label: "Quantization" },
  { key: "loss_balance", color: "#10b981", label: "Balance" },
  { key: "loss_consistency", color: "#ec4899", label: "Consistency" },
  { key: "loss_ortho", color: "#06b6d4", label: "Ortho" },
  { key: "loss_lcs", color: "#f97316", label: "LCS" },
] as const;

const VAL_LINES = [
  { key: "val_loss_total", color: "#3b82f6", label: "Val Total" },
  { key: "val_loss_contrastive", color: "#8b5cf6", label: "Val Contrastive" },
  { key: "val_loss_quantization", color: "#f59e0b", label: "Val Quantization" },
  { key: "val_loss_balance", color: "#10b981", label: "Val Balance" },
  { key: "val_loss_consistency", color: "#ec4899", label: "Val Consistency" },
  { key: "val_loss_ortho", color: "#06b6d4", label: "Val Ortho" },
  { key: "val_loss_lcs", color: "#f97316", label: "Val LCS" },
] as const;

export default function LossChart({ data, evalData }: Props) {
  const [visible, setVisible] = useState<Record<string, boolean>>(() => ({
    ...Object.fromEntries(TRAIN_LINES.map((l) => [l.key, true])),
    ...Object.fromEntries(VAL_LINES.map((l) => [l.key, true])),
  }));
  const [logScale, setLogScale] = useState(false);
  const [showVal, setShowVal] = useState(true);

  const toggle = (key: string) =>
    setVisible((prev) => ({ ...prev, [key]: !prev[key] }));

  // Merge training data with eval (val loss) data at matching steps
  type ChartPoint = Record<string, number | null | undefined>;
  const chartData = useMemo((): ChartPoint[] => {
    if (!evalData?.length || !showVal) return data as unknown as ChartPoint[];

    // Build step -> val losses map
    const valMap = new Map<number, EvalMetric>();
    for (const e of evalData) {
      if (e.step != null) valMap.set(e.step, e);
    }

    if (valMap.size === 0) return data as unknown as ChartPoint[];

    const valFields = (val: EvalMetric) => ({
      val_loss_total: val.val_loss_total,
      val_loss_contrastive: val.val_loss_contrastive,
      val_loss_quantization: val.val_loss_quantization,
      val_loss_balance: val.val_loss_balance,
      val_loss_consistency: val.val_loss_consistency,
      val_loss_ortho: val.val_loss_ortho,
      val_loss_lcs: val.val_loss_lcs,
    });

    // Clone training data and merge val losses at matching steps
    const stepSet = new Set(data.map((d) => d.step));
    const merged: ChartPoint[] = data.map((d) => {
      const val = valMap.get(d.step);
      if (!val) return { ...d };
      return { ...d, ...valFields(val) };
    });

    // If eval step doesn't match any training step, insert as standalone point
    for (const [step, val] of valMap) {
      if (stepSet.has(step)) continue;
      merged.push({ step, epoch: val.epoch, ...valFields(val) });
    }

    merged.sort((a, b) => (a.step ?? 0) - (b.step ?? 0));
    return merged;
  }, [data, evalData, showVal]);

  const hasValData = (evalData?.length ?? 0) > 0 &&
    evalData!.some((e) => e.val_loss_total != null);

  const yDomain = useMemo(() => {
    if (logScale) return [0.001, "auto"] as const;
    if (chartData.length === 0) return [0, "auto"] as const;

    const allKeys = [
      ...TRAIN_LINES.filter((l) => visible[l.key]).map((l) => l.key),
      ...(showVal && hasValData
        ? VAL_LINES.filter((l) => visible[l.key]).map((l) => l.key)
        : []),
    ];
    if (allKeys.length === 0) return [0, "auto"] as const;

    let max = 0;
    for (const d of chartData) {
      for (const k of allKeys) {
        const v = (d as unknown as Record<string, number>)[k];
        if (v != null && v > max) max = v;
      }
    }
    return [0, max > 0 ? max * 1.1 : "auto"] as const;
  }, [chartData, visible, logScale, showVal, hasValData]);

  return (
    <div className="rounded-xl bg-gray-900 p-4 border border-gray-800 h-full">
      <div className="flex items-center justify-between mb-3">
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
          {hasValData && (
            <button
              onClick={() => setShowVal((p) => !p)}
              className={`text-[10px] px-1.5 py-0.5 rounded border transition-all mr-1 ${
                showVal
                  ? "border-orange-500 text-orange-400 opacity-100"
                  : "border-gray-700 text-gray-500 opacity-60"
              }`}
            >
              Val
            </button>
          )}
          {TRAIN_LINES.map((l) => (
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

      {data.length === 0 ? (
        <div className="h-64 flex items-center justify-center text-gray-600 text-sm">
          Waiting for training data...
        </div>
      ) : (
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
            <XAxis
              dataKey="step"
              type="number"
              domain={["dataMin", "dataMax"]}
              tick={{ fontSize: 10, fill: "#6b7280" }}
              stroke="#374151"
              tickFormatter={(v: number) =>
                v >= 1000 ? `${(v / 1000).toFixed(1)}k` : String(v)
              }
            />
            <YAxis
              scale={logScale ? "log" : "auto"}
              domain={yDomain}
              allowDataOverflow
              tick={{ fontSize: 10, fill: "#6b7280" }}
              stroke="#374151"
              tickFormatter={(v: number) =>
                v < 0.01 ? v.toExponential(0) : v.toFixed(2)
              }
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "#111827",
                border: "1px solid #374151",
                borderRadius: "8px",
                fontSize: "11px",
              }}
              labelStyle={{ color: "#9ca3af" }}
            />
            <Legend
              wrapperStyle={{ fontSize: "11px" }}
              onClick={(e) => toggle(e.dataKey as string)}
            />
            {/* Training loss lines (solid) */}
            {TRAIN_LINES.map((l) =>
              visible[l.key] ? (
                <Line
                  key={l.key}
                  type="monotone"
                  dataKey={l.key}
                  stroke={l.color}
                  dot={false}
                  strokeWidth={l.key === "loss_total" ? 2 : 1.2}
                  name={l.label}
                  isAnimationActive={false}
                />
              ) : null
            )}
            {/* Validation loss lines (dashed, with dots at epoch boundaries) */}
            {showVal &&
              hasValData &&
              VAL_LINES.map((l) =>
                visible[l.key] ? (
                  <Line
                    key={l.key}
                    type="monotone"
                    dataKey={l.key}
                    stroke={l.color}
                    dot={{ r: 3, fill: l.color }}
                    strokeWidth={l.key === "val_loss_total" ? 2 : 1.2}
                    strokeDasharray="6 3"
                    connectNulls
                    name={l.label}
                    isAnimationActive={false}
                  />
                ) : null
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
  );
}
