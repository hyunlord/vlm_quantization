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

const DIRECTIONS = ["I2T", "T2I", "I2I", "T2T"] as const;
const DIR_COLORS: Record<string, string> = {
  I2T: "#3b82f6",
  T2I: "#8b5cf6",
  I2I: "#10b981",
  T2T: "#f59e0b",
};

export default function MetricsChart({ data }: Props) {
  const [activeTab, setActiveTab] = useState<"map" | "precision">("map");

  const mapData = data.map((d) => ({
    epoch: d.epoch,
    I2T: d.map_i2t,
    T2I: d.map_t2i,
    I2I: d.map_i2i,
    T2T: d.map_t2t,
  }));

  const latest = data[data.length - 1];
  const precisionData = latest
    ? [
        { name: "P@1", value: latest.p1 ?? 0 },
        { name: "P@5", value: latest.p5 ?? 0 },
        { name: "P@10", value: latest.p10 ?? 0 },
      ]
    : [];

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
              {tab === "map" ? "mAP@50" : "P@k"}
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
              dataKey="epoch"
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
                name={`${dir} mAP@50`}
                isAnimationActive={false}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      ) : (
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={precisionData}>
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
            />
            <Bar dataKey="value" fill="#3b82f6" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}
