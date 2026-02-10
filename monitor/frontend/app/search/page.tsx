"use client";

import {
  ArrowLeft,
  Database,
  Globe,
  ImageIcon,
  Loader2,
  Search,
  Type,
  Upload,
} from "lucide-react";
import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import type { IndexStatus, SearchResult } from "@/lib/types";

type InputMode = "upload" | "url" | "text";
type SearchMode = "backbone" | "hash";

function getApiBase() {
  if (typeof window === "undefined") return "";
  return "";
}

export default function SearchPage() {
  // Index state
  const [indexPath, setIndexPath] = useState("");
  const [indexStatus, setIndexStatus] = useState<IndexStatus | null>(null);
  const [loadingIndex, setLoadingIndex] = useState(false);

  // Model status
  const [modelLoaded, setModelLoaded] = useState(false);
  const [backboneOnly, setBackboneOnly] = useState(false);

  // Query state
  const [inputMode, setInputMode] = useState<InputMode>("text");
  const [searchMode, setSearchMode] = useState<SearchMode>("backbone");
  const [text, setText] = useState("");
  const [imageUrl, setImageUrl] = useState("");
  const [preview, setPreview] = useState<string | null>(null);
  const [bit, setBit] = useState(64);
  const [topK, setTopK] = useState(20);
  const [searching, setSearching] = useState(false);

  // Results
  const [results, setResults] = useState<SearchResult[] | null>(null);
  const [queryType, setQueryType] = useState<string>("");
  const [resultMode, setResultMode] = useState<string>("");
  const [error, setError] = useState<string | null>(null);

  const fileRef = useRef<HTMLInputElement>(null);

  // Check index and model status on mount
  useEffect(() => {
    const base = getApiBase();
    fetch(`${base}/api/search/status`)
      .then((r) => r.json())
      .then((data) => {
        if (data.loaded) setIndexStatus(data);
      })
      .catch(() => {});
    fetch(`${base}/api/inference/status`)
      .then((r) => r.json())
      .then((data) => {
        setModelLoaded(data.loaded ?? false);
        setBackboneOnly(data.backbone_only ?? false);
      })
      .catch(() => {});
  }, []);

  const loadIndex = async () => {
    if (!indexPath.trim()) return;
    setLoadingIndex(true);
    setError(null);
    try {
      const res = await fetch(`${getApiBase()}/api/search/load-index`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ index_path: indexPath.trim() }),
      });
      const data = await res.json();
      if (data.error) {
        setError(data.error);
      } else {
        setIndexStatus(data);
        if (data.bit_list?.length > 0 && !data.bit_list.includes(bit)) {
          setBit(data.bit_list[data.bit_list.length - 1]);
        }
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load index");
    } finally {
      setLoadingIndex(false);
    }
  };

  const handleFile = useCallback((file: File) => {
    if (!file.type.startsWith("image/")) return;
    const reader = new FileReader();
    reader.onload = (e) => {
      setPreview(e.target?.result as string);
      setError(null);
    };
    reader.readAsDataURL(file);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile],
  );

  const doSearch = async () => {
    setSearching(true);
    setError(null);
    setResults(null);
    try {
      const body: Record<string, unknown> = {
        mode: searchMode,
        bit,
        top_k: topK,
      };

      if (inputMode === "upload" && preview) {
        body.image_base64 = preview;
      } else if (inputMode === "url" && imageUrl.trim()) {
        body.image_url = imageUrl.trim();
      } else if (inputMode === "text" && text.trim()) {
        body.text = text.trim();
      } else {
        setError("Please provide an input");
        setSearching(false);
        return;
      }

      const res = await fetch(`${getApiBase()}/api/search/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      if (data.error) {
        setError(data.error);
      } else {
        setResults(data.results);
        setQueryType(data.query_type);
        setResultMode(data.mode);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Search failed");
    } finally {
      setSearching(false);
    }
  };

  const inputTabs: { key: InputMode; icon: typeof Upload; label: string }[] = [
    { key: "upload", icon: Upload, label: "Upload" },
    { key: "url", icon: Globe, label: "URL" },
    { key: "text", icon: Type, label: "Text" },
  ];

  const canSearch =
    indexStatus?.loaded &&
    modelLoaded &&
    ((inputMode === "upload" && preview) ||
      (inputMode === "url" && imageUrl.trim()) ||
      (inputMode === "text" && text.trim()));

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 p-4 md:p-6">
      <div className="max-w-[1400px] mx-auto space-y-4">
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
              VLM Hash Search
            </h1>
          </div>
          <div className="flex items-center gap-2">
            <Link
              href="/inference"
              className="flex items-center gap-1.5 text-xs text-gray-400 hover:text-blue-400 transition-colors"
            >
              <Search className="w-3.5 h-3.5" />
              <span>Hash Explorer</span>
            </Link>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-[320px_1fr] gap-4">
          {/* Left panel: Controls */}
          <div className="space-y-4">
            {/* Index panel */}
            <div className="rounded-xl bg-gray-900 border border-gray-800 p-4 space-y-3">
              <div className="flex items-center gap-2">
                <Database className="w-4 h-4 text-gray-500" />
                <span className="text-sm font-medium text-gray-300">
                  Search Index
                </span>
              </div>

              {indexStatus?.loaded ? (
                <div className="space-y-1.5">
                  <div className="flex items-center gap-2">
                    <div className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                    <span className="text-xs text-emerald-400">Loaded</span>
                  </div>
                  <div className="text-[10px] text-gray-500 space-y-0.5">
                    <p>{indexStatus.num_items.toLocaleString()} items</p>
                    <p>bits: [{indexStatus.bit_list.join(", ")}]</p>
                    <p className="truncate" title={indexStatus.index_path}>
                      {indexStatus.index_path.split("/").pop()}
                    </p>
                  </div>
                </div>
              ) : (
                <>
                  <input
                    value={indexPath}
                    onChange={(e) => setIndexPath(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && loadIndex()}
                    placeholder="/path/to/search_index.pt"
                    className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2
                               text-xs text-gray-200 placeholder-gray-600
                               focus:outline-none focus:border-gray-500"
                  />
                  <button
                    onClick={loadIndex}
                    disabled={loadingIndex || !indexPath.trim()}
                    className="w-full py-2 rounded-lg text-xs font-medium
                               bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700
                               disabled:text-gray-500 text-white transition-colors
                               flex items-center justify-center gap-2"
                  >
                    {loadingIndex ? (
                      <Loader2 className="w-3 h-3 animate-spin" />
                    ) : (
                      <Database className="w-3 h-3" />
                    )}
                    Load Index
                  </button>
                </>
              )}

              {/* Model status */}
              <div className="pt-2 border-t border-gray-800">
                <div className="flex items-center gap-2">
                  <div
                    className={`w-1.5 h-1.5 rounded-full ${modelLoaded ? "bg-emerald-400" : "bg-gray-600"}`}
                  />
                  <span className="text-[10px] text-gray-500">
                    Model:{" "}
                    {modelLoaded
                      ? backboneOnly
                        ? "Backbone Only"
                        : "Hash Model"
                      : "Not loaded"}
                  </span>
                </div>
                {!modelLoaded && (
                  <p className="text-[10px] text-gray-600 mt-1">
                    Load a model from{" "}
                    <Link
                      href="/inference"
                      className="text-blue-400 hover:underline"
                    >
                      Hash Explorer
                    </Link>
                  </p>
                )}
              </div>
            </div>

            {/* Query panel */}
            <div className="rounded-xl bg-gray-900 border border-gray-800 p-4 space-y-3">
              <span className="text-sm font-medium text-gray-300">Query</span>

              {/* Input mode tabs */}
              <div className="flex gap-0.5 bg-gray-800 rounded-lg p-0.5">
                {inputTabs.map((tab) => {
                  const Icon = tab.icon;
                  return (
                    <button
                      key={tab.key}
                      onClick={() => {
                        setInputMode(tab.key);
                        setError(null);
                      }}
                      className={`flex-1 flex items-center justify-center gap-1 text-[11px] px-2 py-1.5 rounded-md transition-colors ${
                        inputMode === tab.key
                          ? "bg-blue-600 text-white"
                          : "text-gray-500 hover:text-gray-300"
                      }`}
                    >
                      <Icon className="w-3 h-3" />
                      {tab.label}
                    </button>
                  );
                })}
              </div>

              {/* Upload */}
              {inputMode === "upload" && (
                <div
                  onDrop={handleDrop}
                  onDragOver={(e) => e.preventDefault()}
                  onClick={() => fileRef.current?.click()}
                  className="border-2 border-dashed border-gray-700 rounded-lg p-3 cursor-pointer
                             hover:border-gray-500 transition-colors min-h-[100px]
                             flex flex-col items-center justify-center gap-1.5"
                >
                  {preview ? (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img
                      src={preview}
                      alt="Preview"
                      className="max-h-[80px] rounded object-contain"
                    />
                  ) : (
                    <>
                      <ImageIcon className="w-6 h-6 text-gray-600" />
                      <p className="text-[10px] text-gray-500">
                        Drop or click
                      </p>
                    </>
                  )}
                  <input
                    ref={fileRef}
                    type="file"
                    accept="image/*"
                    className="hidden"
                    onChange={(e) => {
                      const file = e.target.files?.[0];
                      if (file) handleFile(file);
                    }}
                  />
                </div>
              )}

              {/* URL */}
              {inputMode === "url" && (
                <input
                  value={imageUrl}
                  onChange={(e) => setImageUrl(e.target.value)}
                  placeholder="https://example.com/image.jpg"
                  className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2
                             text-xs text-gray-200 placeholder-gray-600
                             focus:outline-none focus:border-gray-500"
                />
              )}

              {/* Text */}
              {inputMode === "text" && (
                <textarea
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  placeholder="Describe what you're looking for..."
                  rows={3}
                  className="w-full bg-gray-800 border border-gray-700 rounded-lg p-3
                             text-xs text-gray-200 placeholder-gray-600 resize-none
                             focus:outline-none focus:border-gray-500"
                />
              )}

              {/* Search mode */}
              <div className="space-y-2">
                <span className="text-[10px] uppercase tracking-wider text-gray-500">
                  Search Mode
                </span>
                <div className="flex gap-2">
                  <button
                    onClick={() => setSearchMode("backbone")}
                    className={`flex-1 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                      searchMode === "backbone"
                        ? "bg-blue-600 text-white"
                        : "bg-gray-800 text-gray-500 hover:text-gray-300"
                    }`}
                  >
                    Backbone (Cosine)
                  </button>
                  <button
                    onClick={() => setSearchMode("hash")}
                    disabled={backboneOnly}
                    className={`flex-1 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                      searchMode === "hash"
                        ? "bg-purple-600 text-white"
                        : "bg-gray-800 text-gray-500 hover:text-gray-300"
                    } disabled:opacity-30 disabled:cursor-not-allowed`}
                  >
                    Hash (Hamming)
                  </button>
                </div>
              </div>

              {/* Bit level (hash mode only) */}
              {searchMode === "hash" && indexStatus?.bit_list && (
                <div className="space-y-1.5">
                  <span className="text-[10px] uppercase tracking-wider text-gray-500">
                    Bit Level
                  </span>
                  <div className="flex gap-1.5">
                    {indexStatus.bit_list.map((b) => (
                      <button
                        key={b}
                        onClick={() => setBit(b)}
                        className={`flex-1 py-1 rounded text-[11px] font-mono transition-colors ${
                          bit === b
                            ? "bg-purple-600 text-white"
                            : "bg-gray-800 text-gray-500 hover:text-gray-300"
                        }`}
                      >
                        {b}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {/* Top-K */}
              <div className="space-y-1.5">
                <div className="flex items-center justify-between">
                  <span className="text-[10px] uppercase tracking-wider text-gray-500">
                    Top-K
                  </span>
                  <span className="text-[10px] text-gray-400 font-mono">
                    {topK}
                  </span>
                </div>
                <input
                  type="range"
                  min={5}
                  max={50}
                  step={5}
                  value={topK}
                  onChange={(e) => setTopK(Number(e.target.value))}
                  className="w-full accent-blue-500"
                />
              </div>

              {/* Search button */}
              <button
                onClick={doSearch}
                disabled={!canSearch || searching}
                className="w-full py-2.5 rounded-lg text-xs font-medium
                           bg-emerald-600 hover:bg-emerald-500 disabled:bg-gray-700
                           disabled:text-gray-500 text-white transition-colors
                           flex items-center justify-center gap-2"
              >
                {searching ? (
                  <>
                    <Loader2 className="w-3.5 h-3.5 animate-spin" />
                    Searching...
                  </>
                ) : (
                  <>
                    <Search className="w-3.5 h-3.5" />
                    Search
                  </>
                )}
              </button>

              {error && <p className="text-xs text-red-400">{error}</p>}
            </div>
          </div>

          {/* Right panel: Results */}
          <div className="rounded-xl bg-gray-900 border border-gray-800 p-4 min-h-[400px]">
            {results ? (
              <>
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-sm font-semibold text-gray-200">
                    Results
                  </h2>
                  <span className="text-[10px] text-gray-500">
                    {queryType} query | {resultMode} mode |{" "}
                    {results.length} results
                  </span>
                </div>

                {results.length === 0 ? (
                  <p className="text-xs text-gray-500 text-center py-8">
                    No results found.
                  </p>
                ) : (
                  <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 xl:grid-cols-5 gap-3">
                    {results.map((r) => (
                      <div
                        key={`${r.image_id}-${r.rank}`}
                        className="rounded-lg bg-gray-800/50 border border-gray-700/50 overflow-hidden
                                   hover:border-gray-600 transition-colors"
                      >
                        {/* Thumbnail */}
                        {r.thumbnail ? (
                          // eslint-disable-next-line @next/next/no-img-element
                          <img
                            src={`data:image/jpeg;base64,${r.thumbnail}`}
                            alt={r.caption}
                            className="w-full aspect-square object-cover bg-gray-800"
                          />
                        ) : (
                          <div className="w-full aspect-square bg-gray-800 flex items-center justify-center">
                            <ImageIcon className="w-6 h-6 text-gray-700" />
                          </div>
                        )}

                        {/* Info */}
                        <div className="p-2 space-y-1">
                          <p className="text-[10px] text-gray-400 line-clamp-2 leading-relaxed">
                            {r.caption}
                          </p>
                          <div className="flex items-center justify-between">
                            <span className="text-[10px] text-gray-600">
                              #{r.rank}
                            </span>
                            <span
                              className={`text-[11px] font-mono font-medium ${
                                r.score > 0.8
                                  ? "text-emerald-400"
                                  : r.score > 0.6
                                    ? "text-yellow-400"
                                    : "text-red-400"
                              }`}
                            >
                              {(r.score * 100).toFixed(1)}%
                            </span>
                          </div>
                          {r.distance != null && (
                            <span className="text-[10px] text-gray-600">
                              hamming: {r.distance}
                            </span>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </>
            ) : (
              <div className="flex flex-col items-center justify-center h-full min-h-[300px] gap-3">
                <Search className="w-8 h-8 text-gray-700" />
                <p className="text-xs text-gray-600 text-center max-w-[250px]">
                  {!indexStatus?.loaded
                    ? "Load a search index to get started"
                    : !modelLoaded
                      ? "Load a model from Hash Explorer"
                      : "Enter a query and click Search"}
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
