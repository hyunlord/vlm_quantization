"use client";

import {
  ArrowLeft,
  ChevronDown,
  ChevronRight,
  ChevronUp,
  Download,
  FolderSearch,
  Loader2,
  RefreshCw,
  Server,
  Star,
} from "lucide-react";
import Link from "next/link";
import { Suspense, useCallback, useEffect, useMemo, useState } from "react";
import { useSearchParams } from "next/navigation";
import { useRunContext } from "@/contexts/RunContext";
import RunSelector from "@/components/RunSelector";
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
  backbone_only?: boolean;
  checkpoint: string;
  model_name: string;
  bit_list: number[];
  hparams: Record<string, unknown>;
}

interface Checkpoint {
  path: string;
  name: string;
  run_dir: string;
  size_mb: number;
  modified: string;
  epoch: number | null;
  val_loss: number | null;
}

const DEFAULT_CKPT_DIR =
  "/content/drive/MyDrive/vlm_quantization/checkpoints";

// --- Helpers ---

function timeAgo(iso: string) {
  const diff = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  return `${Math.floor(hrs / 24)}d ago`;
}

function formatDate(iso: string) {
  const d = new Date(iso);
  return (
    d.toLocaleDateString("en", { month: "short", day: "numeric" }) +
    " " +
    d.toLocaleTimeString("en", {
      hour: "2-digit",
      minute: "2-digit",
      hour12: false,
    })
  );
}

function formatRunDir(runDir: string) {
  // Parse YYYYMMDD_HHMMSS format
  const m = runDir.match(/^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})$/);
  if (m) {
    const months = [
      "Jan", "Feb", "Mar", "Apr", "May", "Jun",
      "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    return `${months[parseInt(m[2]) - 1]} ${parseInt(m[3])}, ${m[1]} ${m[4]}:${m[5]}`;
  }
  return runDir;
}

function formatHparamValue(v: unknown): string {
  if (Array.isArray(v)) return JSON.stringify(v);
  if (typeof v === "number") {
    if (v < 0.001 && v > 0) return v.toExponential(1);
    return String(v);
  }
  if (typeof v === "string") return v.split("/").pop() ?? v;
  return String(v);
}

// --- Compact hparams display for run preview ---

function RunHparams({ hp }: { hp: Record<string, unknown> }) {
  if (!hp || Object.keys(hp).length === 0) {
    return (
      <p className="text-[10px] text-gray-600 px-3 py-2">
        No hyperparameters found
      </p>
    );
  }
  const rows = ([
    ["backbone", hp.model_name],
    ["bits", hp.bit_list],
    ["hidden", hp.hidden_dim],
    ["freeze", hp.freeze_backbone],
    ["hash_lr", hp.hash_lr],
    ["backbone_lr", hp.backbone_lr],
    ["warmup", hp.warmup_steps],
    ["wd", hp.weight_decay],
    ["temp", hp.temperature],
    ["ema", hp.ema_decay],
    ["c_w", hp.contrastive_weight],
    ["q_w", hp.quantization_weight],
    ["b_w", hp.balance_weight],
    ["o_w", hp.ortho_weight],
    ["cons_w", hp.consistency_weight],
    ["lcs_w", hp.lcs_weight],
  ] as [string, unknown][]).filter(([, v]) => v != null);

  return (
    <div className="flex flex-wrap gap-x-3 gap-y-0.5 px-3 py-2 bg-gray-800/30 text-[10px] border-b border-gray-800/50">
      {rows.map(([k, v]) => (
        <span key={k}>
          <span className="text-gray-500">{k}:</span>{" "}
          <span className="text-gray-400 font-mono">
            {formatHparamValue(v)}
          </span>
        </span>
      ))}
    </div>
  );
}

// ========== Main Component ==========

export default function InferencePageWrapper() {
  return (
    <Suspense>
      <InferencePage />
    </Suspense>
  );
}

function InferencePage() {
  const { selectedRunId, selectedCheckpointId, runCheckpoints, selectCheckpoint } = useRunContext();
  const searchParams = useSearchParams();
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);
  const [loadingModel, setLoadingModel] = useState(false);
  const [loadingPath, setLoadingPath] = useState<string | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [showBrowser, setShowBrowser] = useState(true);
  const [showHparams, setShowHparams] = useState(false);

  // Checkpoint browser
  const [ckptDir, setCkptDir] = useState(DEFAULT_CKPT_DIR);
  const [scanning, setScanning] = useState(false);
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([]);
  const [hasScanned, setHasScanned] = useState(false);

  // Expandable run details
  const [expandedRun, setExpandedRun] = useState<string | null>(null);
  const [runHparams, setRunHparams] = useState<
    Record<string, Record<string, unknown>>
  >({});
  const [fetchingHparams, setFetchingHparams] = useState<string | null>(null);

  const [codesA, setCodesA] = useState<HashCode[] | null>(null);
  const [codesB, setCodesB] = useState<HashCode[] | null>(null);
  const [comparisons, setComparisons] = useState<Comparison[] | null>(null);
  const [comparing, setComparing] = useState(false);

  // Backbone embeddings
  const [embA, setEmbA] = useState<number[] | null>(null);
  const [embB, setEmbB] = useState<number[] | null>(null);
  const [backboneSim, setBackboneSim] = useState<number | null>(null);

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

  // Try to get checkpoint_dir from training status config
  useEffect(() => {
    fetch("/api/training/status")
      .then((r) => r.json())
      .then((data) => {
        const dir = data?.config?.training?.checkpoint_dir;
        if (dir && typeof dir === "string") {
          setCkptDir(dir);
        }
      })
      .catch(() => {});
  }, []);

  const scanCheckpoints = useCallback(async (dir?: string) => {
    setScanning(true);
    setHasScanned(true);
    setRunHparams({});
    setExpandedRun(null);
    try {
      // If explicit dir provided, pass it; otherwise let server use CHECKPOINT_DIR env var
      const qs = dir
        ? `?directory=${encodeURIComponent(dir.trim())}`
        : "";
      const res = await fetch(`/api/inference/checkpoints${qs}`);
      const data = await res.json();
      setCheckpoints(data.checkpoints || []);
    } catch {
      setCheckpoints([]);
    } finally {
      setScanning(false);
    }
  }, []);

  // Auto-scan on mount (uses server-configured CHECKPOINT_DIR)
  useEffect(() => {
    scanCheckpoints();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Auto-expand the group matching selectedRunId
  useEffect(() => {
    if (selectedRunId && checkpoints.length > 0 && !expandedRun) {
      const match = checkpoints.find((c) => c.run_dir === selectedRunId);
      if (match) {
        setExpandedRun(selectedRunId);
        // Also fetch hparams for the expanded run
        if (!runHparams[selectedRunId]) {
          fetch(
            `/api/inference/checkpoint-info?path=${encodeURIComponent(match.path)}`,
          )
            .then((r) => r.json())
            .then((data) => {
              setRunHparams((prev) => ({
                ...prev,
                [selectedRunId]: data.hparams || {},
              }));
            })
            .catch(() => {});
        }
      }
    }
  }, [selectedRunId, checkpoints, expandedRun, runHparams]);

  const toggleRunExpand = async (runDir: string, firstCkptPath: string) => {
    if (expandedRun === runDir) {
      setExpandedRun(null);
      return;
    }
    setExpandedRun(runDir);
    if (!runHparams[runDir]) {
      setFetchingHparams(runDir);
      try {
        const res = await fetch(
          `/api/inference/checkpoint-info?path=${encodeURIComponent(firstCkptPath)}`,
        );
        const data = await res.json();
        setRunHparams((prev) => ({
          ...prev,
          [runDir]: data.hparams || {},
        }));
      } catch {
        setRunHparams((prev) => ({ ...prev, [runDir]: {} }));
      } finally {
        setFetchingHparams(null);
      }
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
        setShowHparams(true);
        setCodesA(null);
        setCodesB(null);
        setComparisons(null);
        // Sync with context: find matching checkpoint and select it
        const matchingCkpt = runCheckpoints.find((c) => c.path === path);
        if (matchingCkpt && matchingCkpt.id !== selectedCheckpointId) {
          selectCheckpoint(matchingCkpt.id);
        }
      }
    } catch (e) {
      setLoadError(e instanceof Error ? e.message : "Load failed");
    } finally {
      setLoadingModel(false);
      setLoadingPath(null);
    }
  };

  // Handle ?load= query param OR auto-load from context checkpoint selection
  useEffect(() => {
    const loadPath = searchParams.get("load");
    if (loadPath && !modelStatus?.loaded && !loadingModel) {
      loadCheckpoint(loadPath);
      return;
    }
    // Auto-load from context if a checkpoint is selected but not yet loaded
    if (selectedCheckpointId && runCheckpoints.length > 0 && !modelStatus?.loaded && !loadingModel) {
      const ckpt = runCheckpoints.find((c) => c.id === selectedCheckpointId);
      if (ckpt && ckpt.path) {
        loadCheckpoint(ckpt.path);
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [searchParams, selectedCheckpointId, runCheckpoints]);

  const loadBackboneOnly = async () => {
    setLoadingModel(true);
    setLoadError(null);
    try {
      const res = await fetch("/api/inference/load-backbone", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_name: "google/siglip2-so400m-patch14-384" }),
      });
      const data = await res.json();
      if (data.error) {
        setLoadError(data.error);
      } else {
        setModelStatus(data);
        setShowBrowser(false);
        setShowHparams(false);
        setCodesA(null);
        setCodesB(null);
        setComparisons(null);
        setEmbA(null);
        setEmbB(null);
        setBackboneSim(null);
      }
    } catch (e) {
      setLoadError(e instanceof Error ? e.message : "Load failed");
    } finally {
      setLoadingModel(false);
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

  // Auto-compare backbone embeddings
  useEffect(() => {
    if (embA && embB) {
      fetch("/api/inference/compare-backbone", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ embedding_a: embA, embedding_b: embB }),
      })
        .then((r) => r.json())
        .then((data) => {
          if (data.cosine_similarity != null) {
            setBackboneSim(data.cosine_similarity);
          }
        })
        .catch(() => {});
    }
  }, [embA, embB]);

  const isModelLoaded = modelStatus?.loaded ?? false;
  const isBackboneOnly = modelStatus?.backbone_only ?? false;

  // Group checkpoints by run_dir
  const grouped = useMemo(() => {
    const groups = checkpoints.reduce<Record<string, Checkpoint[]>>(
      (acc, ckpt) => {
        (acc[ckpt.run_dir] ??= []).push(ckpt);
        return acc;
      },
      {},
    );
    return Object.entries(groups);
  }, [checkpoints]);

  // Find best checkpoint per run (lowest val_loss)
  const bestPerRun = useMemo(() => {
    const bests = new Set<string>();
    for (const [, ckpts] of grouped) {
      const withLoss = ckpts.filter((c) => c.val_loss != null);
      if (withLoss.length > 0) {
        const best = withLoss.reduce((a, b) =>
          (a.val_loss ?? Infinity) < (b.val_loss ?? Infinity) ? a : b,
        );
        bests.add(best.path);
      }
    }
    return bests;
  }, [grouped]);

  // Format hparams for display in 3 groups (loaded model)
  const hparamGroups = useMemo(() => {
    const hp = modelStatus?.hparams ?? {};
    if (Object.keys(hp).length === 0) return null;
    return {
      model: [
        ["backbone", hp.model_name],
        ["bits", JSON.stringify(hp.bit_list)],
        ["hidden_dim", hp.hidden_dim],
        ["dropout", hp.dropout],
        ["freeze", String(hp.freeze_backbone)],
      ].filter(([, v]) => v != null),
      training: [
        ["hash_lr", hp.hash_lr],
        ["backbone_lr", hp.backbone_lr],
        ["weight_decay", hp.weight_decay],
        ["warmup_steps", hp.warmup_steps],
      ].filter(([, v]) => v != null),
      loss: [
        ["contrastive", hp.contrastive_weight],
        ["ortho", hp.ortho_weight],
        ["quantization", hp.quantization_weight],
        ["balance", hp.balance_weight],
        ["consistency", hp.consistency_weight],
        ["lcs", hp.lcs_weight],
        ["temperature", hp.temperature],
      ].filter(([, v]) => v != null),
    };
  }, [modelStatus]);

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
              <ArrowLeft className="w-4 h-4" />
            </Link>
            <h1 className="text-lg font-bold text-gray-200">
              VLM Hash Explorer
            </h1>
            <RunSelector />
          </div>
          <div className="flex items-center gap-2">
            <Server className="w-3.5 h-3.5 text-gray-500" />
            <span
              className={`text-xs px-2 py-0.5 rounded-full ${
                isModelLoaded
                  ? isBackboneOnly
                    ? "bg-blue-900/50 text-blue-400"
                    : "bg-emerald-900/50 text-emerald-400"
                  : "bg-gray-800 text-gray-500"
              }`}
            >
              {isModelLoaded
                ? isBackboneOnly
                  ? "Backbone Only"
                  : "Hash Model"
                : "No Model"}
            </span>
          </div>
        </div>

        {/* Checkpoint browser */}
        {showBrowser && (
          <div className="rounded-xl bg-gray-900 border border-gray-800 p-4 space-y-3">
            <div className="flex items-center gap-2 mb-1">
              <FolderSearch className="w-4 h-4 text-gray-500" />
              <span className="text-sm font-medium text-gray-300">
                Select Checkpoint
              </span>
            </div>

            {/* Directory input + Scan */}
            <div className="flex gap-2">
              <input
                value={ckptDir}
                onChange={(e) => setCkptDir(e.target.value)}
                placeholder="Auto-detected from server (override here)"
                onKeyDown={(e) =>
                  e.key === "Enter" && scanCheckpoints(ckptDir)
                }
                className="flex-1 bg-gray-800 border border-gray-700 rounded-lg px-3 py-2
                           text-sm text-gray-200 placeholder-gray-600
                           focus:outline-none focus:border-gray-500"
              />
              <button
                onClick={() => scanCheckpoints(ckptDir)}
                disabled={scanning || !ckptDir.trim()}
                className="px-4 py-2 rounded-lg text-xs font-medium
                           bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700
                           disabled:text-gray-500 text-white transition-colors
                           flex items-center gap-1.5"
              >
                {scanning ? (
                  <Loader2 className="w-3 h-3 animate-spin" />
                ) : (
                  <RefreshCw className="w-3 h-3" />
                )}
                Scan
              </button>
            </div>

            {/* Checkpoint list grouped by run_dir */}
            {checkpoints.length > 0 && (
              <div className="space-y-2 max-h-[450px] overflow-y-auto">
                {grouped.map(([runDir, ckpts]) => {
                  const isExpanded = expandedRun === runDir;
                  const isFetching = fetchingHparams === runDir;
                  const hp = runHparams[runDir];

                  return (
                    <div
                      key={runDir}
                      className="rounded-lg border border-gray-800 overflow-hidden"
                    >
                      {/* Run group header — clickable to expand */}
                      <button
                        onClick={() =>
                          toggleRunExpand(runDir, ckpts[0].path)
                        }
                        className="w-full text-left px-3 py-2 bg-gray-800/50
                                   flex items-center justify-between gap-2
                                   hover:bg-gray-800/80 transition-colors"
                      >
                        <div className="flex items-center gap-2 min-w-0">
                          {isFetching ? (
                            <Loader2 className="w-3 h-3 animate-spin text-gray-500 shrink-0" />
                          ) : isExpanded ? (
                            <ChevronDown className="w-3 h-3 text-gray-500 shrink-0" />
                          ) : (
                            <ChevronRight className="w-3 h-3 text-gray-500 shrink-0" />
                          )}
                          <span className="text-[11px] text-gray-300 font-medium">
                            {formatRunDir(runDir)}
                          </span>
                        </div>
                        <span className="text-[10px] text-gray-600 shrink-0">
                          {ckpts.length} checkpoint
                          {ckpts.length !== 1 ? "s" : ""}
                        </span>
                      </button>

                      {/* Hparams preview (when expanded) */}
                      {isExpanded && hp && <RunHparams hp={hp} />}
                      {isExpanded && isFetching && (
                        <div className="flex items-center gap-2 px-3 py-2 bg-gray-800/30 text-[10px] text-gray-500">
                          <Loader2 className="w-3 h-3 animate-spin" />
                          Loading config...
                        </div>
                      )}

                      {/* Checkpoint rows */}
                      <div className="divide-y divide-gray-800/50">
                        {ckpts.map((ckpt) => {
                          const isCurrent =
                            modelStatus?.checkpoint === ckpt.path;
                          const isLoading = loadingPath === ckpt.path;
                          const isBest = bestPerRun.has(ckpt.path);
                          return (
                            <div
                              key={ckpt.path}
                              className={`w-full px-3 py-2.5 text-xs
                                flex items-center justify-between gap-2 ${
                                  isCurrent
                                    ? "bg-emerald-900/20"
                                    : ""
                                }`}
                            >
                              <div className="flex items-center gap-2 min-w-0">
                                {isBest ? (
                                  <Star className="w-3 h-3 text-yellow-500 shrink-0 fill-yellow-500" />
                                ) : isCurrent ? (
                                  <span className="text-emerald-400 text-[10px] shrink-0">
                                    ●
                                  </span>
                                ) : (
                                  <span className="w-3 shrink-0" />
                                )}

                                <span
                                  className={`font-mono ${isCurrent ? "text-emerald-300" : "text-gray-300"}`}
                                >
                                  {ckpt.epoch != null
                                    ? `Epoch ${ckpt.epoch}`
                                    : ckpt.name}
                                </span>

                                {ckpt.val_loss != null && (
                                  <span className="text-gray-500">
                                    loss:{" "}
                                    <span
                                      className={
                                        isBest
                                          ? "text-yellow-400 font-medium"
                                          : "text-gray-400"
                                      }
                                    >
                                      {ckpt.val_loss.toFixed(4)}
                                    </span>
                                  </span>
                                )}
                              </div>

                              <div className="flex items-center gap-3 shrink-0">
                                <span className="text-gray-600 text-[10px]">
                                  {ckpt.size_mb} MB
                                </span>
                                <span
                                  className="text-gray-600 text-[10px] w-[85px] text-right"
                                  title={new Date(ckpt.modified).toLocaleString()}
                                >
                                  {formatDate(ckpt.modified)}
                                </span>
                                <span className="text-gray-700 text-[10px] w-[42px] text-right">
                                  {timeAgo(ckpt.modified)}
                                </span>
                                {isCurrent ? (
                                  <span className="text-emerald-400 text-[10px] font-medium w-12 text-center">
                                    Active
                                  </span>
                                ) : (
                                  <button
                                    onClick={() => loadCheckpoint(ckpt.path)}
                                    disabled={loadingModel}
                                    className="px-2.5 py-1 rounded text-[10px] font-medium w-12
                                               bg-blue-600/80 hover:bg-blue-500 disabled:bg-gray-700
                                               disabled:text-gray-500 text-white transition-colors
                                               flex items-center justify-center gap-1"
                                  >
                                    {isLoading ? (
                                      <Loader2 className="w-3 h-3 animate-spin" />
                                    ) : (
                                      <>
                                        <Download className="w-2.5 h-2.5" />
                                        Load
                                      </>
                                    )}
                                  </button>
                                )}
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  );
                })}
              </div>
            )}

            {/* Empty state after scan */}
            {hasScanned && !scanning && checkpoints.length === 0 && (
              <p className="text-xs text-gray-600 text-center py-4">
                No .ckpt files found in this directory.
              </p>
            )}

            {/* Backbone-only option */}
            <div className="pt-2 border-t border-gray-800">
              <button
                onClick={loadBackboneOnly}
                disabled={loadingModel}
                className="w-full py-2.5 rounded-lg text-xs font-medium
                           border border-blue-600/50 text-blue-400 hover:bg-blue-600/10
                           disabled:border-gray-700 disabled:text-gray-500
                           transition-colors flex items-center justify-center gap-2"
              >
                {loadingModel && !loadingPath ? (
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                ) : (
                  <Server className="w-3.5 h-3.5" />
                )}
                Load Backbone Only (no hash layers)
              </button>
              <p className="text-[10px] text-gray-600 text-center mt-1">
                Compare with raw SigLIP2 cosine similarity as baseline
              </p>
            </div>

            {/* Error */}
            {loadError && (
              <p className="text-xs text-red-400">{loadError}</p>
            )}
          </div>
        )}

        {/* Model info panel (when loaded and browser hidden) */}
        {isModelLoaded && modelStatus && !showBrowser && (
          <div className="rounded-xl bg-gray-900 border border-gray-800 p-4">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <div className={`w-2 h-2 rounded-full ${isBackboneOnly ? "bg-blue-400" : "bg-emerald-400"}`} />
                <span className="text-sm font-medium text-gray-200">
                  {modelStatus.model_name.split("/").pop()}
                </span>
                {isBackboneOnly ? (
                  <span className="text-xs text-blue-400">Backbone Only</span>
                ) : (
                  <span className="text-xs text-gray-500">
                    bits: [{modelStatus.bit_list.join(", ")}]
                  </span>
                )}
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setShowHparams(!showHparams)}
                  className="text-[10px] text-gray-500 hover:text-gray-300 transition-colors
                             flex items-center gap-1"
                >
                  {showHparams ? (
                    <ChevronUp className="w-3 h-3" />
                  ) : (
                    <ChevronDown className="w-3 h-3" />
                  )}
                  Config
                </button>
                <button
                  onClick={() => setShowBrowser(true)}
                  className="text-xs px-3 py-1.5 rounded-lg border border-gray-700
                             text-gray-400 hover:text-gray-200 hover:border-gray-500
                             transition-colors flex items-center gap-1.5"
                >
                  <RefreshCw className="w-3 h-3" />
                  Change
                </button>
              </div>
            </div>

            <p className="text-[10px] text-gray-600 truncate mb-2">
              {modelStatus.checkpoint}
            </p>

            {/* Hparams detail */}
            {showHparams && hparamGroups && (
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mt-3 pt-3 border-t border-gray-800">
                {/* Model params */}
                <div>
                  <h4 className="text-[10px] uppercase tracking-wider text-gray-500 mb-1.5">
                    Model
                  </h4>
                  <div className="space-y-1">
                    {hparamGroups.model.map(([k, v]) => (
                      <div
                        key={k as string}
                        className="flex justify-between text-[11px]"
                      >
                        <span className="text-gray-500">{k as string}</span>
                        <span className="text-gray-300 font-mono">
                          {String(v)}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Training params */}
                <div>
                  <h4 className="text-[10px] uppercase tracking-wider text-gray-500 mb-1.5">
                    Training
                  </h4>
                  <div className="space-y-1">
                    {hparamGroups.training.map(([k, v]) => (
                      <div
                        key={k as string}
                        className="flex justify-between text-[11px]"
                      >
                        <span className="text-gray-500">{k as string}</span>
                        <span className="text-gray-300 font-mono">
                          {String(v)}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Loss weights */}
                <div>
                  <h4 className="text-[10px] uppercase tracking-wider text-gray-500 mb-1.5">
                    Loss Weights
                  </h4>
                  <div className="space-y-1">
                    {hparamGroups.loss.map(([k, v]) => (
                      <div
                        key={k as string}
                        className="flex justify-between text-[11px]"
                      >
                        <span className="text-gray-500">{k as string}</span>
                        <span className="text-gray-300 font-mono">
                          {String(v)}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Input panels */}
        {isModelLoaded ? (
          <>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <InputPanel
                label="Input A"
                onEncode={setCodesA}
                onBackboneEncode={setEmbA}
                backboneOnly={isBackboneOnly}
              />
              <InputPanel
                label="Input B"
                onEncode={setCodesB}
                onBackboneEncode={setEmbB}
                backboneOnly={isBackboneOnly}
              />
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

            {/* Backbone similarity (shown when available) */}
            {backboneSim != null && (
              <div className="rounded-xl bg-gray-900 border border-gray-800 p-4">
                <h2 className="text-sm font-semibold text-gray-200 mb-3">
                  Backbone Comparison (Cosine Similarity)
                </h2>
                <div className="flex items-center justify-between rounded-lg bg-gray-800/50 border border-gray-700 p-3">
                  <span className="text-xs text-gray-400">
                    SigLIP2 1152-dim embedding
                  </span>
                  <div className="flex items-center gap-3">
                    <span className="text-xs text-gray-400">Cosine:</span>
                    <span
                      className={`text-lg font-bold font-mono ${
                        backboneSim > 0.8
                          ? "text-emerald-400"
                          : backboneSim > 0.6
                            ? "text-yellow-400"
                            : "text-red-400"
                      }`}
                    >
                      {(backboneSim * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              </div>
            )}

            {/* Hash code results */}
            {comparisons && codesA && codesB && (
              <div className="rounded-xl bg-gray-900 border border-gray-800 p-4">
                <h2 className="text-sm font-semibold text-gray-200 mb-3">
                  Hash Code Comparison (Hamming Distance)
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
                Encode both inputs to compare
              </p>
            ) : null}
          </>
        ) : (
          !showBrowser && (
            <div className="rounded-xl bg-gray-900 border border-gray-800 p-8 text-center">
              <p className="text-gray-500 text-sm">
                Select a checkpoint above to start encoding
              </p>
            </div>
          )
        )}
      </div>
    </div>
  );
}
