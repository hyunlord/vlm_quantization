"use client";

import HashBitmap from "./HashBitmap";

interface HashCode {
  bits: number;
  binary: number[];
  continuous: number[];
}

interface Comparison {
  bits: number;
  hamming: number;
  max_distance: number;
  similarity: number;
}

interface Props {
  codesA: HashCode[];
  codesB: HashCode[];
  comparisons: Comparison[];
}

export default function HashComparison({
  codesA,
  codesB,
  comparisons,
}: Props) {
  return (
    <div className="space-y-3">
      {comparisons.map((comp, i) => {
        const a = codesA[i];
        const b = codesB[i];

        // Find differing bits
        const diffBits = new Set<number>();
        if (a && b) {
          for (let j = 0; j < a.binary.length; j++) {
            if (a.binary[j] !== b.binary[j]) diffBits.add(j);
          }
        }

        const simPct = (comp.similarity * 100).toFixed(1);
        const simColor =
          comp.similarity > 0.8
            ? "text-emerald-400"
            : comp.similarity > 0.6
              ? "text-yellow-400"
              : "text-red-400";

        return (
          <div
            key={comp.bits}
            className="rounded-lg bg-gray-800/50 border border-gray-700 p-3"
          >
            {/* Header */}
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-semibold text-gray-300">
                {comp.bits}-bit
              </span>
              <div className="flex items-center gap-3 text-xs">
                <span className="text-gray-400">
                  Hamming:{" "}
                  <span className="text-gray-200 font-mono">
                    {comp.hamming}
                  </span>
                  <span className="text-gray-600">/{comp.max_distance}</span>
                </span>
                <span className={`font-semibold ${simColor}`}>
                  {simPct}%
                </span>
              </div>
            </div>

            {/* Bitmaps side by side */}
            {a && b && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                <HashBitmap binary={a.binary} highlights={diffBits} label="A" />
                <HashBitmap binary={b.binary} highlights={diffBits} label="B" />
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
