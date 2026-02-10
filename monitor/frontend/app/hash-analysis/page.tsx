"use client";

import Link from "next/link";
import { ArrowLeft } from "lucide-react";
import { useWebSocket } from "@/hooks/useWebSocket";
import BitBalanceChart from "@/components/BitBalanceChart";
import SampleGallery from "@/components/SampleGallery";
import SimilarityHeatmap from "@/components/SimilarityHeatmap";

function getWsUrl() {
  if (typeof window === "undefined") return "ws://localhost:8000/ws";
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}/ws`;
}

export default function HashAnalysisPage() {
  const { hashAnalysis, isConnected } = useWebSocket(getWsUrl());

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 p-4 md:p-6">
      <div className="max-w-[1200px] mx-auto space-y-4">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Link
              href="/"
              className="text-gray-500 hover:text-gray-300 transition-colors"
            >
              <ArrowLeft size={18} />
            </Link>
            <h1 className="text-lg font-semibold">Hash Analysis</h1>
            {hashAnalysis && (
              <span className="text-xs text-gray-500">
                Epoch {hashAnalysis.epoch} / Step {hashAnalysis.step}
              </span>
            )}
          </div>
          <div className="flex items-center gap-2">
            <div
              className={`w-2 h-2 rounded-full ${isConnected ? "bg-emerald-400" : "bg-red-400"}`}
            />
            <span className="text-xs text-gray-500">
              {isConnected ? "Connected" : "Disconnected"}
            </span>
          </div>
        </div>

        {!hashAnalysis ? (
          <div className="rounded-xl bg-gray-900 p-8 border border-gray-800 text-center">
            <p className="text-gray-500">
              Waiting for validation epoch to complete...
            </p>
            <p className="text-xs text-gray-600 mt-2">
              Hash analysis data is collected at the end of each validation
              epoch.
            </p>
          </div>
        ) : (
          <>
            {/* Bit Balance */}
            <BitBalanceChart
              bitActivations={hashAnalysis.bit_activations ?? {}}
            />

            {/* Sample Gallery */}
            <SampleGallery
              samples={hashAnalysis.samples ?? []}
              imgCodes={hashAnalysis.sample_img_codes ?? []}
              txtCodes={hashAnalysis.sample_txt_codes ?? []}
              similarityMatrix={hashAnalysis.similarity_matrix ?? []}
            />

            {/* Similarity Heatmap */}
            <SimilarityHeatmap
              matrix={hashAnalysis.similarity_matrix ?? []}
              samples={hashAnalysis.samples ?? []}
              bit={hashAnalysis.bit ?? 64}
            />
          </>
        )}
      </div>
    </div>
  );
}
