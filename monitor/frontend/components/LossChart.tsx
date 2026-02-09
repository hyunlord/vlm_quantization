"use client";

import { useState } from "react";
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
import type { TrainingMetric } from "@/lib/types";

interface Props {
  data: TrainingMetric[];
}

const LOSS_LINES = [
  { key: "loss_total", color: "#3b82f6", label: "Total" },
  { key: "loss_contrastive", color: "#8b5cf6", label: "Contrastive" },
  { key: "loss_quantization", color: "#f59e0b", label: "Quantization" },
  { key: "loss_balance", color: "#10b981", label: "Balance" },
  { key: "loss_consistency", color: "#ec4899", label: "Consistency" },
] as const;

export default function LossChart({ data }: Props) {
  const [visible, setVisible] = useState<Record<string, boolean>>(() =>
    Object.fromEntries(LOSS_LINES.map((l) => [l.key, true]))
  );

  const toggle = (key: string) =>
    setVisible((prev) => ({ ...prev, [key]: !prev[key] }));

  return (
    <div className="rounded-xl bg-gray-900 p-4 border border-gray-800 h-full">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-sm font-semibold text-gray-200">Loss Curves</h2>
        <div className="flex gap-1.5 flex-wrap">
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

      {data.length === 0 ? (
        <div className="h-64 flex items-center justify-center text-gray-600 text-sm">
          Waiting for training data...
        </div>
      ) : (
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
            <XAxis
              dataKey="step"
              tick={{ fontSize: 10, fill: "#6b7280" }}
              stroke="#374151"
            />
            <YAxis
              tick={{ fontSize: 10, fill: "#6b7280" }}
              stroke="#374151"
              tickFormatter={(v: number) => v.toFixed(2)}
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
            {LOSS_LINES.map((l) =>
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
