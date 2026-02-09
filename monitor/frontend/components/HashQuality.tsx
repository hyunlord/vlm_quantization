"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import type { EvalMetric } from "@/lib/types";

interface Props {
  data: EvalMetric[];
}

export default function HashQuality({ data }: Props) {
  const entropyData = data
    .filter((d) => d.bit_entropy !== null)
    .map((d) => ({ epoch: d.epoch, entropy: d.bit_entropy }));

  const quantData = data
    .filter((d) => d.quant_error !== null)
    .map((d) => ({ epoch: d.epoch, error: d.quant_error }));

  const latestEntropy = entropyData[entropyData.length - 1]?.entropy ?? null;
  const latestError = quantData[quantData.length - 1]?.error ?? null;

  return (
    <div className="rounded-xl bg-gray-900 p-4 border border-gray-800 h-full">
      <h2 className="text-sm font-semibold text-gray-200 mb-3">
        Hash Quality
      </h2>

      {data.length === 0 ? (
        <div className="h-48 flex items-center justify-center text-gray-600 text-sm">
          Waiting for evaluation data...
        </div>
      ) : (
        <div className="space-y-4">
          {/* Summary */}
          <div className="grid grid-cols-2 gap-3">
            <div className="bg-gray-800/50 rounded-lg p-2.5">
              <p className="text-xs text-gray-500">Bit Entropy</p>
              <p className="text-lg font-mono text-gray-200">
                {latestEntropy !== null ? latestEntropy.toFixed(4) : "—"}
              </p>
              <p className="text-[10px] text-gray-600">ideal: 1.0</p>
            </div>
            <div className="bg-gray-800/50 rounded-lg p-2.5">
              <p className="text-xs text-gray-500">Quant Error</p>
              <p className="text-lg font-mono text-gray-200">
                {latestError !== null ? latestError.toFixed(4) : "—"}
              </p>
              <p className="text-[10px] text-gray-600">ideal: 0.0</p>
            </div>
          </div>

          {/* Entropy chart */}
          {entropyData.length > 0 && (
            <div>
              <p className="text-xs text-gray-500 mb-1">
                Bit Entropy per Epoch
              </p>
              <ResponsiveContainer width="100%" height={100}>
                <LineChart data={entropyData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                  <XAxis
                    dataKey="epoch"
                    tick={{ fontSize: 9, fill: "#6b7280" }}
                    stroke="#374151"
                  />
                  <YAxis
                    domain={[0, 1]}
                    tick={{ fontSize: 9, fill: "#6b7280" }}
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
                  <ReferenceLine
                    y={1}
                    stroke="#10b981"
                    strokeDasharray="3 3"
                    strokeOpacity={0.5}
                  />
                  <Line
                    type="monotone"
                    dataKey="entropy"
                    stroke="#10b981"
                    dot={{ r: 2 }}
                    strokeWidth={1.5}
                    isAnimationActive={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Quantization error chart */}
          {quantData.length > 0 && (
            <div>
              <p className="text-xs text-gray-500 mb-1">
                Quantization Error per Epoch
              </p>
              <ResponsiveContainer width="100%" height={100}>
                <LineChart data={quantData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                  <XAxis
                    dataKey="epoch"
                    tick={{ fontSize: 9, fill: "#6b7280" }}
                    stroke="#374151"
                  />
                  <YAxis
                    tick={{ fontSize: 9, fill: "#6b7280" }}
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
                  <ReferenceLine
                    y={0}
                    stroke="#10b981"
                    strokeDasharray="3 3"
                    strokeOpacity={0.5}
                  />
                  <Line
                    type="monotone"
                    dataKey="error"
                    stroke="#f59e0b"
                    dot={{ r: 2 }}
                    strokeWidth={1.5}
                    isAnimationActive={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
