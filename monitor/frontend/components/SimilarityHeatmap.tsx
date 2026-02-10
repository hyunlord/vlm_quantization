"use client";

import { useState } from "react";
import type { HashAnalysisSample } from "@/lib/types";

interface Props {
  matrix: number[][];
  samples: HashAnalysisSample[];
  bit: number;
}

function heatColor(value: number): string {
  // 0.0 → dark blue, 0.5 → gray, 1.0 → bright green
  if (value >= 0.9) return "bg-green-400";
  if (value >= 0.8) return "bg-green-500";
  if (value >= 0.7) return "bg-green-600";
  if (value >= 0.6) return "bg-green-700";
  if (value >= 0.5) return "bg-emerald-800";
  if (value >= 0.4) return "bg-gray-700";
  if (value >= 0.3) return "bg-gray-800";
  return "bg-gray-900";
}

function textColor(value: number): string {
  if (value >= 0.7) return "text-gray-900";
  return "text-gray-300";
}

export default function SimilarityHeatmap({ matrix, samples, bit }: Props) {
  const [hover, setHover] = useState<{
    r: number;
    c: number;
  } | null>(null);

  const n = matrix.length;

  return (
    <div className="rounded-xl bg-gray-900 p-4 border border-gray-800">
      <div className="flex items-center justify-between mb-1">
        <h2 className="text-sm font-semibold text-gray-200">
          Cross-Modal Similarity Matrix
        </h2>
        <span className="text-[10px] text-gray-500">{bit}-bit Hamming</span>
      </div>
      <p className="text-[10px] text-gray-500 mb-3">
        Image (rows) vs Text (columns). Diagonal = matched pairs — should be
        brightest.
      </p>

      {n === 0 ? (
        <div className="h-32 flex items-center justify-center text-gray-600 text-sm">
          Waiting for validation data...
        </div>
      ) : (
        <div className="overflow-x-auto">
          {/* Tooltip */}
          {hover && (
            <div className="text-[10px] text-gray-400 mb-2 h-4">
              Image {samples[hover.r]?.image_id ?? hover.r} &harr; Text &ldquo;
              {(samples[hover.c]?.caption ?? "").slice(0, 40)}
              ...&rdquo; ={" "}
              <span className="text-white font-mono">
                {((matrix[hover.r]?.[hover.c] ?? 0) * 100).toFixed(1)}%
              </span>
            </div>
          )}

          <div
            className="inline-grid gap-px"
            style={{
              gridTemplateColumns: `auto repeat(${n}, 1fr)`,
              gridTemplateRows: `auto repeat(${n}, 1fr)`,
            }}
          >
            {/* Top-left empty cell */}
            <div />

            {/* Column headers (text) */}
            {samples.slice(0, n).map((s, c) => (
              <div
                key={`col-${c}`}
                className="text-[8px] text-gray-500 text-center px-0.5 truncate w-12"
                title={s.caption}
              >
                T{c}
              </div>
            ))}

            {/* Rows */}
            {matrix.map((row, r) => (
              <>
                {/* Row header (image) */}
                <div
                  key={`row-${r}`}
                  className="text-[8px] text-gray-500 flex items-center pr-1"
                >
                  I{r}
                </div>

                {/* Cells */}
                {row.slice(0, n).map((val, c) => (
                  <div
                    key={`${r}-${c}`}
                    className={`w-12 h-10 flex items-center justify-center rounded-sm cursor-default transition-all ${heatColor(val)} ${textColor(val)} ${
                      r === c
                        ? "ring-1 ring-yellow-500/50"
                        : ""
                    } ${
                      hover?.r === r || hover?.c === c
                        ? "opacity-100"
                        : hover
                          ? "opacity-60"
                          : "opacity-100"
                    }`}
                    onMouseEnter={() => setHover({ r, c })}
                    onMouseLeave={() => setHover(null)}
                  >
                    <span className="text-[10px] font-mono">
                      {(val * 100).toFixed(0)}
                    </span>
                  </div>
                ))}
              </>
            ))}
          </div>

          {/* Legend */}
          <div className="flex items-center gap-2 mt-3 text-[9px] text-gray-500">
            <span>Low</span>
            <div className="flex gap-px">
              {["bg-gray-900", "bg-gray-800", "bg-gray-700", "bg-emerald-800", "bg-green-700", "bg-green-600", "bg-green-500", "bg-green-400"].map(
                (bg) => (
                  <div key={bg} className={`w-4 h-2 rounded-sm ${bg}`} />
                ),
              )}
            </div>
            <span>High</span>
            <span className="ml-2 text-yellow-500/70">
              [ ] = diagonal (matched pair)
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
