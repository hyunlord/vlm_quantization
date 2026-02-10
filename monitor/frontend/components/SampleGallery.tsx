"use client";

import type { HashAnalysisSample } from "@/lib/types";

interface Props {
  samples: HashAnalysisSample[];
  imgCodes: number[][];
  txtCodes: number[][];
  similarityMatrix: number[][];
}

function simColor(sim: number): string {
  if (sim >= 0.8) return "text-green-400";
  if (sim >= 0.6) return "text-yellow-400";
  return "text-red-400";
}

function simBg(sim: number): string {
  if (sim >= 0.8) return "bg-green-900/40";
  if (sim >= 0.6) return "bg-yellow-900/40";
  return "bg-red-900/40";
}

export default function SampleGallery({
  samples,
  imgCodes,
  txtCodes,
  similarityMatrix,
}: Props) {
  return (
    <div className="rounded-xl bg-gray-900 p-4 border border-gray-800">
      <h2 className="text-sm font-semibold text-gray-200 mb-1">
        Sample Quality Check
      </h2>
      <p className="text-[10px] text-gray-500 mb-3">
        Fixed 8 validation samples — image vs. matched text hash similarity
        (diagonal of similarity matrix).
      </p>

      {samples.length === 0 ? (
        <div className="h-32 flex items-center justify-center text-gray-600 text-sm">
          Waiting for validation data...
        </div>
      ) : (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {samples.map((sample, i) => {
            const diagSim = similarityMatrix[i]?.[i] ?? 0;
            const imgCode = imgCodes[i] ?? [];
            const txtCode = txtCodes[i] ?? [];
            const matching =
              imgCode.length > 0
                ? imgCode.filter((b, j) => b === txtCode[j]).length
                : 0;

            return (
              <div
                key={sample.image_id}
                className="rounded-lg bg-gray-800/60 border border-gray-700/50 overflow-hidden"
              >
                {/* Thumbnail */}
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={sample.thumbnail}
                  alt={`COCO ${sample.image_id}`}
                  className="w-full aspect-square object-cover"
                />

                <div className="p-2 space-y-1.5">
                  {/* Caption */}
                  <p
                    className="text-[10px] text-gray-400 line-clamp-2 leading-tight"
                    title={sample.caption}
                  >
                    {sample.caption}
                  </p>

                  {/* Similarity badge */}
                  <div className="flex items-center justify-between">
                    <span className="text-[9px] text-gray-600">
                      ID: {sample.image_id}
                    </span>
                    <span
                      className={`text-[10px] font-mono px-1.5 py-0.5 rounded ${simBg(diagSim)} ${simColor(diagSim)}`}
                    >
                      {(diagSim * 100).toFixed(0)}%
                    </span>
                  </div>

                  {/* Bit match info */}
                  <div className="text-[9px] text-gray-600">
                    {matching}/{imgCode.length} bits match
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
