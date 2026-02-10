"use client";

import { useMemo, useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
  Cell,
} from "recharts";

interface Props {
  bitActivations: Record<string, number[]>;
}

function balanceColor(rate: number): string {
  const dist = Math.abs(rate - 0.5) * 2; // 0 = perfect, 1 = worst
  if (dist < 0.2) return "#22c55e"; // green
  if (dist < 0.4) return "#84cc16"; // lime
  if (dist < 0.6) return "#eab308"; // yellow
  if (dist < 0.8) return "#f97316"; // orange
  return "#ef4444"; // red
}

export default function BitBalanceChart({ bitActivations }: Props) {
  const bitLevels = useMemo(
    () =>
      Object.keys(bitActivations)
        .map((k) => parseInt(k.replace("activation_", ""), 10))
        .sort((a, b) => a - b),
    [bitActivations],
  );

  const [selectedBit, setSelectedBit] = useState<number>(
    bitLevels.includes(64) ? 64 : bitLevels[0] ?? 64,
  );

  const chartData = useMemo(() => {
    const rates = bitActivations[`activation_${selectedBit}`];
    if (!rates) return [];
    return rates.map((rate, i) => ({ bit: i, rate }));
  }, [bitActivations, selectedBit]);

  return (
    <div className="rounded-xl bg-gray-900 p-4 border border-gray-800">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-sm font-semibold text-gray-200">Bit Balance</h2>
        <div className="flex gap-1.5">
          {bitLevels.map((b) => (
            <button
              key={b}
              onClick={() => setSelectedBit(b)}
              className={`text-[10px] px-2 py-0.5 rounded border transition-all ${
                selectedBit === b
                  ? "border-blue-500 text-blue-400 bg-blue-900/30"
                  : "border-gray-700 text-gray-500"
              }`}
            >
              {b}-bit
            </button>
          ))}
        </div>
      </div>

      <p className="text-[10px] text-gray-500 mb-2">
        Per-bit activation rate (ideal = 0.50). Red bars indicate dead or
        saturated bits.
      </p>

      {chartData.length === 0 ? (
        <div className="h-48 flex items-center justify-center text-gray-600 text-sm">
          Waiting for validation data...
        </div>
      ) : (
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={chartData} barCategoryGap={0}>
            <CartesianGrid
              strokeDasharray="3 3"
              stroke="#1f2937"
              vertical={false}
            />
            <XAxis
              dataKey="bit"
              tick={{ fontSize: 9, fill: "#6b7280" }}
              stroke="#374151"
              interval={selectedBit <= 32 ? 0 : "preserveStartEnd"}
              label={{
                value: "Bit Position",
                position: "insideBottom",
                offset: -2,
                fontSize: 10,
                fill: "#6b7280",
              }}
            />
            <YAxis
              domain={[0, 1]}
              ticks={[0, 0.25, 0.5, 0.75, 1.0]}
              tick={{ fontSize: 9, fill: "#6b7280" }}
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
              labelFormatter={(label) => `Bit ${label}`}
              formatter={(value) =>
                typeof value === "number"
                  ? [value.toFixed(4), "Activation"]
                  : [String(value), "Activation"]
              }
            />
            <ReferenceLine
              y={0.5}
              stroke="#22c55e"
              strokeDasharray="6 3"
              strokeOpacity={0.6}
              label={{
                value: "0.50",
                position: "right",
                fontSize: 9,
                fill: "#22c55e",
              }}
            />
            <Bar dataKey="rate" maxBarSize={12}>
              {chartData.map((entry, i) => (
                <Cell key={i} fill={balanceColor(entry.rate)} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}
