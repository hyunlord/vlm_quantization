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
  BarChart,
  Bar,
} from "recharts";
import type { EvalMetric } from "@/lib/types";

interface Props {
  data: EvalMetric[];
}

const DIRECTIONS = ["I2T", "T2I"] as const;
const DIR_COLORS: Record<string, string> = {
  I2T: "#3b82f6",
  T2I: "#8b5cf6",
};
const BACKBONE_COLORS: Record<string, string> = {
  I2T: "#60a5fa",
  T2I: "#a78bfa",
};

export default function MetricsChart({ data }: Props) {
  const [activeTab, setActiveTab] = useState<"map" | "precision">("map");

  const mapData = data.map((d) => ({
    step: d.step ?? d.epoch,
    epoch: d.epoch,
    I2T: d.map_i2t,
    T2I: d.map_t2i,
    "Backbone I2T": d.backbone_map_i2t,
    "Backbone T2I": d.backbone_map_t2i,
  }));

  const hasBackbone = data.some(
    (d) => d.backbone_map_i2t != null || d.backbone_map_t2i != null
  );

  const latest = data[data.length - 1];
  const precisionData = latest
    ? [
        { name: "P@1", value: latest.p1 ?? null },
        { name: "P@5", value: latest.p5 ?? null },
        { name: "P@10", value: latest.p10 ?? null },
      ]
    : [];

  const hasPrecision = precisionData.some((d) => d.value !== null);

  return (
    <div className="rounded-xl bg-gray-900 p-4 border border-gray-800 h-full">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-sm font-semibold text-gray-200">
          Retrieval Metrics
        </h2>
        <div className="flex gap-1">
          {(["map", "precision"] as const).map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`text-xs px-2 py-0.5 rounded ${
                activeTab === tab
                  ? "bg-blue-900/50 text-blue-400"
                  : "text-gray-500 hover:text-gray-300"
              }`}
            >
              {tab === "map" ? "mAP" : "P@k"}
            </button>
          ))}
        </div>
      </div>

      {data.length === 0 ? (
        <div className="h-48 flex items-center justify-center text-gray-600 text-sm">
          Waiting for evaluation data...
        </div>
      ) : activeTab === "map" ? (
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={mapData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
            <XAxis
              dataKey="step"
              tick={{ fontSize: 10, fill: "#6b7280" }}
              stroke="#374151"
              label={{
                value: "Step",
                position: "insideBottom",
                offset: -2,
                fontSize: 10,
                fill: "#6b7280",
              }}
            />
            <YAxis
              domain={[0, 1]}
              tick={{ fontSize: 10, fill: "#6b7280" }}
              stroke="#374151"
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "#111827",
                border: "1px solid #374151",
                borderRadius: "8px",
                fontSize: "11px",
              }}
              labelFormatter={(step) => {
                const point = mapData.find((d) => d.step === step);
                return point
                  ? `Epoch ${point.epoch}, Step ${step}`
                  : `Step ${step}`;
              }}
            />
            <Legend wrapperStyle={{ fontSize: "11px" }} />
            {DIRECTIONS.map((dir) => (
              <Line
                key={dir}
                type="monotone"
                dataKey={dir}
                stroke={DIR_COLORS[dir]}
                dot={{ r: 3 }}
                strokeWidth={1.5}
                name={`Hash ${dir}`}
                isAnimationActive={false}
              />
            ))}
            {hasBackbone &&
              DIRECTIONS.map((dir) => (
                <Line
                  key={`backbone-${dir}`}
                  type="monotone"
                  dataKey={`Backbone ${dir}`}
                  stroke={BACKBONE_COLORS[dir]}
                  dot={{ r: 2 }}
                  strokeWidth={1.5}
                  strokeDasharray="6 3"
                  name={`Backbone ${dir}`}
                  isAnimationActive={false}
                />
              ))}
          </LineChart>
        </ResponsiveContainer>
      ) : !hasPrecision ? (
        <div className="h-48 flex items-center justify-center text-gray-600 text-sm">
          Waiting for P@K data...
        </div>
      ) : (
        <ResponsiveContainer width="100%" height={200}>
          <BarChart
            data={precisionData.map((d) => ({ ...d, value: d.value ?? 0 }))}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
            <XAxis
              dataKey="name"
              tick={{ fontSize: 10, fill: "#6b7280" }}
              stroke="#374151"
            />
            <YAxis
              domain={[0, 1]}
              tick={{ fontSize: 10, fill: "#6b7280" }}
              stroke="#374151"
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "#111827",
                border: "1px solid #374151",
                borderRadius: "8px",
                fontSize: "11px",
              }}
              formatter={(value) =>
                typeof value === "number"
                  ? [value.toFixed(4), "Precision"]
                  : [String(value), "Precision"]
              }
            />
            <Bar dataKey="value" fill="#3b82f6" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}
