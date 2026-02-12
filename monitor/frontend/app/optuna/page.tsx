"use client";

import { ArrowLeft, Calendar, FlaskConical, Loader2, Trophy, Wifi, WifiOff } from "lucide-react";
import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Scatter,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { OptunaTrial, OptunaStudyListItem, OptunaStudySummary } from "@/lib/types";

const STATE_COLORS: Record<string, string> = {
  COMPLETE: "#34d399",
  PRUNED: "#fbbf24",
  FAIL: "#f87171",
  RUNNING: "#60a5fa",
  WAITING: "#9ca3af",
};

/** Format ISO datetime to concise local string: "2026-02-12 16:23" */
function formatDateTime(iso: string | null): string {
  if (!iso) return "—";
  const d = new Date(iso);
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

/** Format ISO datetime to short time: "16:23:05" */
function formatTime(iso: string | null): string {
  if (!iso) return "—";
  const d = new Date(iso);
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}

/** Relative time ago: "2h ago", "3d ago" */
function timeAgo(iso: string | null): string {
  if (!iso) return "";
  const diff = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  return `${days}d ago`;
}

export default function OptunaPage() {
  const [status, setStatus] = useState<{
    available: boolean;
    studies: string[];
    studies_detail?: OptunaStudyListItem[];
  } | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedStudy, setSelectedStudy] = useState<string>("");
  const [studyData, setStudyData] = useState<OptunaStudySummary | null>(null);
  const [trials, setTrials] = useState<OptunaTrial[]>([]);
  const [loadingStudy, setLoadingStudy] = useState(false);
  const [sortCol, setSortCol] = useState<"number" | "value" | "state" | "duration_seconds">("number");
  const [sortAsc, setSortAsc] = useState(true);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const prevTrialCount = useRef(0);
  const [isLive, setIsLive] = useState(false);

  // Fetch Optuna status on mount
  useEffect(() => {
    setLoading(true);
    fetch(`/api/optuna/status`)
      .then((r) => r.json())
      .then((data) => {
        setStatus(data);
        // Auto-select the most recent study (first in studies_detail, sorted by datetime desc)
        if (data.studies_detail?.length > 0) {
          setSelectedStudy(data.studies_detail[0].name);
        } else if (data.studies?.length === 1) {
          setSelectedStudy(data.studies[0]);
        }
      })
      .catch(() => setStatus({ available: false, studies: [] }))
      .finally(() => setLoading(false));
  }, []);

  // Fetch study details + trials
  const fetchStudy = useCallback(
    async (name: string) => {
      if (!name) return;
      setLoadingStudy(true);
      try {
        const [studyRes, trialsRes] = await Promise.all([
          fetch(`/api/optuna/studies/${name}`).then((r) => r.json()),
          fetch(`/api/optuna/studies/${name}/trials`).then((r) => r.json()),
        ]);
        if (!studyRes.error) setStudyData(studyRes);
        if (trialsRes.trials) {
          setTrials(trialsRes.trials);
          // Detect live changes
          if (trialsRes.trials.length !== prevTrialCount.current) {
            setIsLive(trialsRes.trials.length > prevTrialCount.current && prevTrialCount.current > 0);
            prevTrialCount.current = trialsRes.trials.length;
          }
        }
      } catch {
        /* ignore */
      } finally {
        setLoadingStudy(false);
      }
    },
    [],
  );

  useEffect(() => {
    if (selectedStudy) fetchStudy(selectedStudy);
  }, [selectedStudy, fetchStudy]);

  // Poll every 10s
  useEffect(() => {
    if (pollRef.current) clearInterval(pollRef.current);
    if (!selectedStudy) return;
    pollRef.current = setInterval(() => fetchStudy(selectedStudy), 10000);
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [selectedStudy, fetchStudy]);

  // Derived data
  const completedTrials = trials.filter((t) => t.state === "COMPLETE");
  const bestSoFar = completedTrials.reduce<{ number: number; value: number }[]>((acc, t) => {
    if (t.value == null) return acc;
    const prev = acc.length > 0 ? acc[acc.length - 1].value : Infinity;
    const best = Math.min(prev, t.value);
    acc.push({ number: t.number, value: best });
    return acc;
  }, []);

  // Trial chart data (all trials)
  const trialChartData = trials
    .filter((t) => t.value != null)
    .map((t) => ({
      number: t.number,
      value: t.value,
      state: t.state,
    }));

  // Sorted trials for table
  const sortedTrials = [...trials].sort((a, b) => {
    const aVal = a[sortCol] ?? -Infinity;
    const bVal = b[sortCol] ?? -Infinity;
    if (aVal < bVal) return sortAsc ? -1 : 1;
    if (aVal > bVal) return sortAsc ? 1 : -1;
    return 0;
  });

  const handleSort = (col: typeof sortCol) => {
    if (sortCol === col) {
      setSortAsc(!sortAsc);
    } else {
      setSortCol(col);
      setSortAsc(col === "number");
    }
  };

  // Importance bar data
  const importanceData = studyData?.param_importances
    ? Object.entries(studyData.param_importances)
        .sort(([, a], [, b]) => b - a)
        .map(([name, value]) => ({ name, value: Math.round(value * 100) }))
    : [];

  // Study list for selector (prefer detailed list with timestamps)
  const studyList: OptunaStudyListItem[] =
    status?.studies_detail ??
    (status?.studies?.map((s) => ({ name: s, datetime_start: null, n_trials: 0 })) || []);

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-950 text-gray-100 flex items-center justify-center">
        <Loader2 className="w-6 h-6 animate-spin text-gray-600" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 p-4 md:p-6">
      <div className="max-w-[1400px] mx-auto space-y-4">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Link href="/" className="text-gray-500 hover:text-gray-300 transition-colors">
              <ArrowLeft className="w-4 h-4" />
            </Link>
            <FlaskConical className="w-4 h-4 text-amber-400" />
            <h1 className="text-lg font-bold text-gray-200">
              Optuna Hyperparameter Search
            </h1>
          </div>
          <div className="flex items-center gap-3">
            <Link
              href="/"
              className="flex items-center gap-1.5 text-xs text-gray-400 hover:text-blue-400 transition-colors"
            >
              Dashboard
            </Link>
            {status?.available ? (
              <div className="flex items-center gap-1.5 text-emerald-400 text-xs">
                <Wifi className="w-3.5 h-3.5" />
                <span>{isLive ? "Live" : "Connected"}</span>
              </div>
            ) : (
              <div className="flex items-center gap-1.5 text-red-400 text-xs">
                <WifiOff className="w-3.5 h-3.5" />
                <span>No DB</span>
              </div>
            )}
          </div>
        </div>

        {/* No DB state */}
        {!status?.available && (
          <div className="rounded-xl bg-gray-900 border border-gray-800 p-8 text-center">
            <FlaskConical className="w-8 h-8 text-gray-700 mx-auto mb-3" />
            <p className="text-sm text-gray-500">
              No Optuna database found. Run <code className="text-amber-400">optuna_search.py</code> to start a study.
            </p>
          </div>
        )}

        {/* Study selector — always visible when studies exist */}
        {status?.available && studyList.length > 0 && (
          <div className="rounded-xl bg-gray-900 border border-gray-800 p-3">
            <div className="flex items-center gap-2 mb-2">
              <Calendar className="w-3.5 h-3.5 text-gray-500" />
              <span className="text-[10px] text-gray-500 uppercase tracking-wider">Select Study</span>
              <span className="text-[10px] text-gray-600">({studyList.length} studies)</span>
            </div>
            <div className="grid gap-1.5">
              {studyList.map((s) => (
                <button
                  key={s.name}
                  onClick={() => setSelectedStudy(s.name)}
                  className={`flex items-center justify-between px-3 py-2 rounded-lg text-left transition-all ${
                    selectedStudy === s.name
                      ? "bg-amber-500/10 border border-amber-500/30 text-amber-200"
                      : "bg-gray-800/50 border border-transparent hover:bg-gray-800 hover:border-gray-700 text-gray-300"
                  }`}
                >
                  <div className="flex items-center gap-3 min-w-0">
                    <FlaskConical className={`w-3.5 h-3.5 flex-shrink-0 ${
                      selectedStudy === s.name ? "text-amber-400" : "text-gray-600"
                    }`} />
                    <div className="min-w-0">
                      <div className="text-xs font-medium truncate">{s.name}</div>
                      {s.datetime_start && (
                        <div className="text-[10px] text-gray-500 mt-0.5">
                          {formatDateTime(s.datetime_start)}
                          <span className="text-gray-600 ml-1.5">({timeAgo(s.datetime_start)})</span>
                        </div>
                      )}
                    </div>
                  </div>
                  <div className="flex items-center gap-2 flex-shrink-0 ml-3">
                    <span className="text-[10px] text-gray-500">{s.n_trials} trials</span>
                  </div>
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Study content */}
        {status?.available && selectedStudy && (
          <>
            {loadingStudy && !studyData ? (
              <div className="flex items-center justify-center py-12">
                <Loader2 className="w-5 h-5 animate-spin text-gray-600" />
              </div>
            ) : studyData ? (
              <>
                {/* Summary Cards */}
                <div className="grid grid-cols-2 md:grid-cols-6 gap-3">
                  <SummaryCard label="Total Trials" value={studyData.n_trials} />
                  <SummaryCard label="Completed" value={studyData.n_complete} color="text-emerald-400" />
                  <SummaryCard label="Pruned" value={studyData.n_pruned} color="text-amber-400" />
                  <SummaryCard
                    label="Best Value"
                    value={studyData.best_trial?.value != null ? studyData.best_trial.value.toFixed(4) : "—"}
                    color="text-blue-400"
                  />
                  <SummaryCard
                    label="Best Trial"
                    value={studyData.best_trial?.number != null ? `#${studyData.best_trial.number}` : "—"}
                    color="text-purple-400"
                  />
                  <SummaryCard
                    label="Started"
                    value={studyData.datetime_start ? formatDateTime(studyData.datetime_start) : "—"}
                    color="text-gray-400"
                    small
                  />
                </div>

                {/* Charts row */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                  {/* Trial History */}
                  <div className="rounded-xl bg-gray-900 border border-gray-800 p-4">
                    <h3 className="text-xs font-medium text-gray-400 mb-3">Trial History</h3>
                    {trialChartData.length > 0 ? (
                      <ResponsiveContainer width="100%" height={240}>
                        <ComposedChart data={trialChartData} margin={{ top: 5, right: 10, bottom: 5, left: 10 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                          <XAxis
                            dataKey="number"
                            type="number"
                            tick={{ fontSize: 10, fill: "#6b7280" }}
                            label={{ value: "Trial", position: "insideBottom", offset: -2, fontSize: 10, fill: "#6b7280" }}
                          />
                          <YAxis
                            type="number"
                            tick={{ fontSize: 10, fill: "#6b7280" }}
                            label={{ value: "val/total", angle: -90, position: "insideLeft", fontSize: 10, fill: "#6b7280" }}
                          />
                          <Tooltip
                            contentStyle={{ backgroundColor: "#111827", border: "1px solid #374151", borderRadius: 8, fontSize: 11 }}
                            labelFormatter={(v) => `Trial #${v}`}
                            formatter={(v) => [typeof v === "number" ? v.toFixed(4) : String(v ?? ""), "value"]}
                          />
                          <Scatter dataKey="value" fill="#34d399">
                            {trialChartData.map((entry, i) => (
                              <Cell key={i} fill={STATE_COLORS[entry.state] || "#9ca3af"} />
                            ))}
                          </Scatter>
                          {bestSoFar.length > 1 && (
                            <Line
                              data={bestSoFar}
                              dataKey="value"
                              stroke="#60a5fa"
                              strokeWidth={2}
                              dot={false}
                              type="stepAfter"
                              isAnimationActive={false}
                            />
                          )}
                        </ComposedChart>
                      </ResponsiveContainer>
                    ) : (
                      <div className="flex items-center justify-center h-[240px] text-gray-600 text-xs">
                        No completed trials yet
                      </div>
                    )}
                    {/* Legend */}
                    <div className="flex items-center gap-3 mt-2 justify-center">
                      {Object.entries(STATE_COLORS).map(([state, color]) => (
                        <div key={state} className="flex items-center gap-1">
                          <div className="w-2 h-2 rounded-full" style={{ backgroundColor: color }} />
                          <span className="text-[10px] text-gray-500">{state}</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Parameter Importance */}
                  <div className="rounded-xl bg-gray-900 border border-gray-800 p-4">
                    <h3 className="text-xs font-medium text-gray-400 mb-3">Parameter Importance</h3>
                    {importanceData.length > 0 ? (
                      <ResponsiveContainer width="100%" height={240}>
                        <BarChart data={importanceData} layout="vertical" margin={{ top: 5, right: 20, bottom: 5, left: 80 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                          <XAxis
                            type="number"
                            tick={{ fontSize: 10, fill: "#6b7280" }}
                            domain={[0, 100]}
                            tickFormatter={(v) => `${v}%`}
                          />
                          <YAxis
                            dataKey="name"
                            type="category"
                            tick={{ fontSize: 10, fill: "#9ca3af" }}
                            width={75}
                          />
                          <Tooltip
                            contentStyle={{ backgroundColor: "#111827", border: "1px solid #374151", borderRadius: 8, fontSize: 11 }}
                            formatter={(v) => [`${v}%`, "Importance"]}
                          />
                          <Bar dataKey="value" fill="#818cf8" radius={[0, 4, 4, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    ) : (
                      <div className="flex items-center justify-center h-[240px] text-gray-600 text-xs">
                        Need 2+ completed trials
                      </div>
                    )}
                  </div>
                </div>

                {/* Best Trial Detail */}
                {studyData.best_trial && (
                  <div className="rounded-xl bg-gray-900 border border-gray-800 p-4">
                    <div className="flex items-center gap-2 mb-3">
                      <Trophy className="w-4 h-4 text-amber-400" />
                      <h3 className="text-xs font-medium text-gray-400">
                        Best Trial #{studyData.best_trial.number}
                      </h3>
                      <span className="text-xs text-blue-400 font-mono ml-auto">
                        val/total: {studyData.best_trial.value?.toFixed(4)}
                      </span>
                    </div>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                      {Object.entries(studyData.best_trial.params).map(([key, val]) => (
                        <div key={key} className="bg-gray-800/50 rounded-lg p-2.5">
                          <div className="text-[10px] text-gray-500 mb-0.5">{key}</div>
                          <div className="text-xs text-gray-200 font-mono">
                            {typeof val === "number"
                              ? val < 0.001
                                ? val.toExponential(2)
                                : val % 1 === 0
                                  ? val
                                  : val.toFixed(4)
                              : String(val)}
                          </div>
                        </div>
                      ))}
                    </div>
                    {studyData.best_trial.user_attrs &&
                      Object.keys(studyData.best_trial.user_attrs).length > 0 && (
                        <div className="mt-3 pt-3 border-t border-gray-800">
                          <div className="text-[10px] text-gray-500 mb-2">Metrics</div>
                          <div className="grid grid-cols-3 md:grid-cols-6 gap-2">
                            {Object.entries(studyData.best_trial.user_attrs).map(
                              ([key, val]) => (
                                <div key={key} className="text-center">
                                  <div className="text-[10px] text-gray-500">{key}</div>
                                  <div className="text-xs text-gray-300 font-mono">
                                    {typeof val === "number" ? val.toFixed(4) : String(val)}
                                  </div>
                                </div>
                              ),
                            )}
                          </div>
                        </div>
                      )}
                  </div>
                )}

                {/* Trials Table */}
                <div className="rounded-xl bg-gray-900 border border-gray-800 p-4">
                  <h3 className="text-xs font-medium text-gray-400 mb-3">
                    All Trials ({trials.length})
                  </h3>
                  <div className="overflow-x-auto">
                    <table className="w-full text-xs">
                      <thead>
                        <tr className="text-gray-500 border-b border-gray-800">
                          <Th label="#" col="number" sortCol={sortCol} sortAsc={sortAsc} onClick={handleSort} />
                          <Th label="Value" col="value" sortCol={sortCol} sortAsc={sortAsc} onClick={handleSort} />
                          <Th label="State" col="state" sortCol={sortCol} sortAsc={sortAsc} onClick={handleSort} />
                          <th className="text-left py-2 px-2 font-medium whitespace-nowrap">Started</th>
                          <Th label="Duration" col="duration_seconds" sortCol={sortCol} sortAsc={sortAsc} onClick={handleSort} />
                          {/* Param columns from first trial */}
                          {trials[0] &&
                            Object.keys(trials[0].params).map((p) => (
                              <th key={p} className="text-left py-2 px-2 font-medium whitespace-nowrap">
                                {p}
                              </th>
                            ))}
                        </tr>
                      </thead>
                      <tbody>
                        {sortedTrials.map((t) => (
                          <tr
                            key={t.number}
                            className={`border-b border-gray-800/50 hover:bg-gray-800/30 ${
                              studyData.best_trial?.number === t.number ? "bg-amber-500/5" : ""
                            }`}
                          >
                            <td className="py-1.5 px-2 text-gray-300">
                              {studyData.best_trial?.number === t.number && (
                                <Trophy className="w-3 h-3 text-amber-400 inline mr-1" />
                              )}
                              {t.number}
                            </td>
                            <td className="py-1.5 px-2 font-mono text-gray-200">
                              {t.value != null ? t.value.toFixed(4) : "—"}
                            </td>
                            <td className="py-1.5 px-2">
                              <span
                                className="inline-block w-2 h-2 rounded-full mr-1.5"
                                style={{ backgroundColor: STATE_COLORS[t.state] || "#9ca3af" }}
                              />
                              <span className="text-gray-400">{t.state}</span>
                            </td>
                            <td className="py-1.5 px-2 text-gray-500 whitespace-nowrap">
                              {formatTime(t.datetime_start)}
                            </td>
                            <td className="py-1.5 px-2 text-gray-400">
                              {t.duration_seconds != null ? `${t.duration_seconds.toFixed(0)}s` : "—"}
                            </td>
                            {Object.values(t.params).map((v, i) => (
                              <td key={i} className="py-1.5 px-2 font-mono text-gray-400">
                                {typeof v === "number"
                                  ? v < 0.001
                                    ? v.toExponential(2)
                                    : v % 1 === 0
                                      ? v
                                      : v.toFixed(4)
                                  : String(v)}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </>
            ) : null}
          </>
        )}

        {/* Study available but none selected */}
        {status?.available && !selectedStudy && studyList.length > 0 && (
          <div className="rounded-xl bg-gray-900 border border-gray-800 p-8 text-center">
            <p className="text-sm text-gray-500">Select a study to view results.</p>
          </div>
        )}
      </div>
    </div>
  );
}

// --- Sub-components ---

function SummaryCard({
  label,
  value,
  color = "text-gray-200",
  small = false,
}: {
  label: string;
  value: string | number;
  color?: string;
  small?: boolean;
}) {
  return (
    <div className="rounded-xl bg-gray-900 border border-gray-800 p-3">
      <div className="text-[10px] text-gray-500 mb-1">{label}</div>
      <div className={`${small ? "text-xs" : "text-lg"} font-bold font-mono ${color}`}>{value}</div>
    </div>
  );
}

function Th({
  label,
  col,
  sortCol,
  sortAsc,
  onClick,
}: {
  label: string;
  col: "number" | "value" | "state" | "duration_seconds";
  sortCol: string;
  sortAsc: boolean;
  onClick: (col: "number" | "value" | "state" | "duration_seconds") => void;
}) {
  return (
    <th
      className="text-left py-2 px-2 font-medium cursor-pointer hover:text-gray-300 whitespace-nowrap select-none"
      onClick={() => onClick(col)}
    >
      {label}
      {sortCol === col && <span className="ml-1">{sortAsc ? "\u25B2" : "\u25BC"}</span>}
    </th>
  );
}
