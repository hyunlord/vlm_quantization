"use client";

import { Cpu, Thermometer, Monitor } from "lucide-react";
import type { SystemMetric } from "@/lib/types";

interface Props {
  data: SystemMetric | null;
}

function GaugeRing({
  value,
  label,
  unit,
  icon,
  colorFn,
}: {
  value: number;
  label: string;
  unit: string;
  icon: React.ReactNode;
  colorFn: (v: number) => string;
}) {
  const circumference = 2 * Math.PI * 36;
  const offset = circumference - (value / 100) * circumference;
  const color = colorFn(value);

  return (
    <div className="flex flex-col items-center">
      <div className="relative w-24 h-24">
        <svg className="w-full h-full -rotate-90" viewBox="0 0 80 80">
          <circle
            cx="40" cy="40" r="36"
            fill="none" stroke="#1f2937" strokeWidth="6"
          />
          <circle
            cx="40" cy="40" r="36"
            fill="none" stroke={color} strokeWidth="6"
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            className="transition-all duration-500"
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          {icon}
          <span className="text-sm font-mono font-bold text-gray-200 mt-0.5">
            {value.toFixed(0)}{unit}
          </span>
        </div>
      </div>
      <p className="text-xs text-gray-500 mt-1">{label}</p>
    </div>
  );
}

function MemoryBar({
  used,
  total,
  label,
}: {
  used: number;
  total: number;
  label: string;
}) {
  const pct = total > 0 ? (used / total) * 100 : 0;
  const color =
    pct > 90 ? "bg-red-500" : pct > 70 ? "bg-yellow-500" : "bg-emerald-500";

  return (
    <div>
      <div className="flex justify-between text-xs mb-1">
        <span className="text-gray-500">{label}</span>
        <span className="text-gray-400 font-mono">
          {used.toFixed(1)} / {total.toFixed(1)} GB
        </span>
      </div>
      <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-500 ${color}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

const utilColor = (v: number) =>
  v > 95 ? "#ef4444" : v > 80 ? "#eab308" : "#10b981";

export default function SystemPanel({ data }: Props) {
  if (!data) {
    return (
      <div className="rounded-xl bg-gray-900 p-4 border border-gray-800 h-full">
        <h2 className="text-sm font-semibold text-gray-200 mb-3">System</h2>
        <p className="text-gray-500 text-sm">No data</p>
      </div>
    );
  }

  return (
    <div className="rounded-xl bg-gray-900 p-4 border border-gray-800 h-full">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-sm font-semibold text-gray-200">System</h2>
        {data.gpu_name && (
          <span className="text-xs text-gray-600 font-mono truncate max-w-[140px]">
            {data.gpu_name}
          </span>
        )}
      </div>

      {/* Gauges */}
      <div className="grid grid-cols-2 gap-3 mb-4">
        <GaugeRing
          value={data.gpu_util}
          label="GPU"
          unit="%"
          icon={<Monitor className="w-3.5 h-3.5 text-gray-500" />}
          colorFn={utilColor}
        />
        <GaugeRing
          value={data.cpu_util}
          label="CPU"
          unit="%"
          icon={<Cpu className="w-3.5 h-3.5 text-gray-500" />}
          colorFn={utilColor}
        />
      </div>

      {/* Memory bars */}
      <div className="space-y-3">
        <MemoryBar
          used={data.gpu_mem_used}
          total={data.gpu_mem_total}
          label="GPU Memory"
        />
        <MemoryBar used={data.ram_used} total={data.ram_total} label="RAM" />
      </div>

      {/* Temperature */}
      {data.gpu_temp > 0 && (
        <div className="mt-3 flex items-center gap-2 text-xs">
          <Thermometer className="w-3.5 h-3.5 text-gray-500" />
          <span className="text-gray-500">GPU Temp:</span>
          <span
            className={`font-mono font-bold ${
              data.gpu_temp > 85
                ? "text-red-400"
                : data.gpu_temp > 70
                  ? "text-yellow-400"
                  : "text-emerald-400"
            }`}
          >
            {data.gpu_temp}°C
          </span>
        </div>
      )}
    </div>
  );
}
