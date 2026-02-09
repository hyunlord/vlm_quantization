"use client";

import { ArrowLeft, FolderSearch, Loader2, RefreshCw, Server } from "lucide-react";
import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import HashComparison from "@/components/inference/HashComparison";
import InputPanel from "@/components/inference/InputPanel";

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

interface ModelStatus {
  loaded: boolean;
  checkpoint: string;
  model_name: string;
  bit_list: number[];
}

interface Checkpoint {
  path: string;
  name: string;
  run_dir: string;
  size_mb: number;
  modified: string;
}

export default function InferencePage() {
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);
  const [loadingModel, setLoadingModel] = useState(false);
  const [loadingPath, setLoadingPath] = useState<string | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [showBrowser, setShowBrowser] = useState(true);

  // Checkpoint browser
  const [ckptDir, setCkptDir] = useState("");
  const [scanning, setScanning] = useState(false);
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([]);
  const [hasScanned, setHasScanned] = useState(false);

  const [codesA, setCodesA] = useState<HashCode[] | null>(null);
  const [codesB, setCodesB] = useState<HashCode[] | null>(null);
  const [comparisons, setComparisons] = useState<Comparison[] | null>(null);
  const [comparing, setComparing] = useState(false);

  // Check model status on mount
  useEffect(() => {
    fetch("/api/inference/status")
      .then((r) => r.json())
      .then((data) => {
        setModelStatus(data);
        if (data.loaded) setShowBrowser(false);
      })
      .catch(() => {});
  }, []);

  const scanCheckpoints = async () => {
    if (!ckptDir.trim()) return;
    setScanning(true);
    setHasScanned(true);
    try {
      const res = await fetch(
        `/api/inference/checkpoints?directory=${encodeURIComponent(ckptDir.trim())}`
      );
      const data = await res.json();
      setCheckpoints(data.checkpoints || []);
    } catch {
      setCheckpoints([]);
    } finally {
      setScanning(false);
    }
  };

  const loadCheckpoint = async (path: string) => {
    setLoadingModel(true);
    setLoadingPath(path);
    setLoadError(null);
    try {
      const res = await fetch("/api/inference/load", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ checkpoint_path: path }),
      });
      const data = await res.json();
      if (data.error) {
        setLoadError(data.error);
      } else {
        setModelStatus(data);
        setShowBrowser(false);
        setCodesA(null);
        setCodesB(null);
        setComparisons(null);
      }
    } catch (e) {
      setLoadError(e instanceof Error ? e.message : "Load failed");
    } finally {
      setLoadingModel(false);
      setLoadingPath(null);
    }
  };

  const compare = useCallback(async () => {
    if (!codesA || !codesB) return;
    setComparing(true);
    try {
      const res = await fetch("/api/inference/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ codes_a: codesA, codes_b: codesB }),
      });
      const data = await res.json();
      setComparisons(data.comparisons);
    } catch {
      // silently handle
    } finally {
      setComparing(false);
    }
  }, [codesA, codesB]);

  // Auto-compare when both codes are available
  useEffect(() => {
    if (codesA && codesB) {
      compare();
    }
  }, [codesA, codesB, compare]);

  const isModelLoaded = modelStatus?.loaded ?? false;

  // Group checkpoints by run_dir
  const grouped = checkpoints.reduce<Record<string, Checkpoint[]>>(
    (acc, ckpt) => {
      (acc[ckpt.run_dir] ??= []).push(ckpt);
      return acc;
    },
    {}
  );

  const timeAgo = (iso: string) => {
    const diff = Date.now() - new Date(iso).getTime();
    const mins = Math.floor(diff / 60000);
    if (mins < 60) return `${mins}m ago`;
    const hrs = Math.floor(mins / 60);
    if (hrs < 24) return `${hrs}h ago`;
    return `${Math.floor(hrs / 24)}d ago`;
  };

  return (
    <div className="min-h-screen p-4 max-w-[1200px] mx-auto space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Link
            href="/"
            className="text-gray-500 hover:text-gray-300 transition-colors"
          >
            <ArrowLeft className="w-4 h-4" />
          </Link>
          <h1 className="text-lg font-bold text-gray-200">
            VLM Hash Explorer
          </h1>
        </div>
        <div className="flex items-center gap-2">
          <Server className="w-3.5 h-3.5 text-gray-500" />
          <span
            className={`text-xs px-2 py-0.5 rounded-full ${
              isModelLoaded
                ? "bg-emerald-900/50 text-emerald-400"
                : "bg-gray-800 text-gray-500"
            }`}
          >
            {isModelLoaded ? "Model Loaded" : "No Model"}
          </span>
        </div>
      </div>

      {/* Model info bar (when loaded and browser hidden) */}
      {isModelLoaded && modelStatus && !showBrowser && (
        <div className="rounded-xl bg-gray-900 border border-gray-800 p-3">
          <div className="flex items-center justify-between">
            <div className="text-xs space-y-0.5">
              <div className="text-gray-400">
                {modelStatus.model_name} | bits: [
                {modelStatus.bit_list.join(", ")}]
              </div>
              <div className="text-gray-600 truncate max-w-[500px]">
                {modelStatus.checkpoint}
              </div>
            </div>
            <button
              onClick={() => setShowBrowser(true)}
              className="text-xs px-3 py-1.5 rounded-lg border border-gray-700
                         text-gray-400 hover:text-gray-200 hover:border-gray-500
                         transition-colors flex items-center gap-1.5"
            >
              <RefreshCw className="w-3 h-3" />
              Change Model
            </button>
          </div>
        </div>
      )}

      {/* Checkpoint browser */}
      {showBrowser && (
        <div className="rounded-xl bg-gray-900 border border-gray-800 p-4 space-y-3">
          <div className="flex items-center gap-2 mb-1">
            <FolderSearch className="w-4 h-4 text-gray-500" />
            <span className="text-sm font-medium text-gray-300">
              Checkpoint Browser
            </span>
          </div>

          {/* Directory input + Scan */}
          <div className="flex gap-2">
            <input
              value={ckptDir}
              onChange={(e) => setCkptDir(e.target.value)}
              placeholder="/path/to/checkpoints"
              onKeyDown={(e) => e.key === "Enter" && scanCheckpoints()}
              className="flex-1 bg-gray-800 border border-gray-700 rounded-lg px-3 py-2
                         text-sm text-gray-200 placeholder-gray-600
                         focus:outline-none focus:border-gray-500"
            />
            <button
              onClick={scanCheckpoints}
              disabled={scanning || !ckptDir.trim()}
              className="px-4 py-2 rounded-lg text-xs font-medium
                         bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700
                         disabled:text-gray-500 text-white transition-colors
                         flex items-center gap-1.5"
            >
              {scanning && <Loader2 className="w-3 h-3 animate-spin" />}
              Scan
            </button>
          </div>

          {/* Checkpoint list grouped by run_dir */}
          {checkpoints.length > 0 && (
            <div className="space-y-1 max-h-[300px] overflow-y-auto">
              {Object.entries(grouped).map(([runDir, ckpts]) => (
                <div key={runDir}>
                  <div className="text-[10px] text-gray-500 uppercase tracking-wider px-2 py-1.5 sticky top-0 bg-gray-900 border-b border-gray-800">
                    {runDir}
                  </div>
                  {ckpts.map((ckpt) => {
                    const isCurrent =
                      modelStatus?.checkpoint === ckpt.path;
                    const isLoading = loadingPath === ckpt.path;
                    return (
                      <button
                        key={ckpt.path}
                        onClick={() =>
                          !isCurrent && loadCheckpoint(ckpt.path)
                        }
                        disabled={loadingModel}
                        className={`w-full text-left px-3 py-2 rounded-lg text-xs transition-colors
                          flex items-center justify-between gap-2 ${
                            isCurrent
                              ? "bg-emerald-900/30 border border-emerald-800/50"
                              : "hover:bg-gray-800 border border-transparent"
                          }`}
                      >
                        <div className="flex items-center gap-2 min-w-0">
                          {isLoading ? (
                            <Loader2 className="w-3 h-3 animate-spin text-blue-400 shrink-0" />
                          ) : isCurrent ? (
                            <span className="text-emerald-400 text-[10px] shrink-0">
                              ●
                            </span>
                          ) : null}
                          <span
                            className={`truncate ${isCurrent ? "text-emerald-300" : "text-gray-300"}`}
                          >
                            {ckpt.name}
                          </span>
                        </div>
                        <div className="flex items-center gap-3 text-gray-600 shrink-0">
                          <span>{ckpt.size_mb} MB</span>
                          <span>{timeAgo(ckpt.modified)}</span>
                        </div>
                      </button>
                    );
                  })}
                </div>
              ))}
            </div>
          )}

          {/* Empty state after scan */}
          {hasScanned && !scanning && checkpoints.length === 0 && (
            <p className="text-xs text-gray-600 text-center py-2">
              No .ckpt files found in this directory.
            </p>
          )}

          {/* Error */}
          {loadError && (
            <p className="text-xs text-red-400">{loadError}</p>
          )}
        </div>
      )}

      {/* Input panels */}
      {isModelLoaded && (
        <>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <InputPanel label="Input A" onEncode={setCodesA} />
            <InputPanel label="Input B" onEncode={setCodesB} />
          </div>

          {/* Compare button (manual trigger when auto-compare hasn't run) */}
          {codesA && codesB && !comparisons && (
            <div className="flex justify-center">
              <button
                onClick={compare}
                disabled={comparing}
                className="px-6 py-2 rounded-lg text-sm font-medium
                           bg-purple-600 hover:bg-purple-500 disabled:bg-gray-700
                           text-white transition-colors flex items-center gap-2"
              >
                {comparing && (
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                )}
                Compare Hash Codes
              </button>
            </div>
          )}

          {/* Results */}
          {comparisons && codesA && codesB && (
            <div className="rounded-xl bg-gray-900 border border-gray-800 p-4">
              <h2 className="text-sm font-semibold text-gray-200 mb-3">
                Hash Code Comparison
              </h2>
              <HashComparison
                codesA={codesA}
                codesB={codesB}
                comparisons={comparisons}
              />
            </div>
          )}

          {/* Hint when only one side encoded */}
          {(codesA && !codesB) || (!codesA && codesB) ? (
            <p className="text-center text-xs text-gray-600">
              Encode both inputs to compare hash codes
            </p>
          ) : null}
        </>
      )}
    </div>
  );
}
