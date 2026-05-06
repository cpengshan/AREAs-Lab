import React, { useMemo, useState } from "react";

type LoadedRun = {
  dataset: string;
  strategy: string;
  experimentId: string;
  outputJson: any | null;
  evalJson: any | null;
  error: string | null;
};

const DEFAULT_DATASETS = [
    "alexfabbri_multi_news",
    "santoshtyss_uk_legislation",
    "starmpcc_Asclepius-Synthetic-Clinical-Notes",
    "thu-coai_esconv",
    "ccdv_arxiv-summarization",
    "ccdv_govreport-summarization",
    "ccdv_mediasum",
    "ccdv_patent-classification",
    "ccdv_pubmed-summarization",
    "mrSoul7766_ECTSum",
    "HuggingFaceFW_fineweb-edu",
    "HuggingFaceH4_MATH-500",
    "danidanou_Reuters_Financial_News",
    "Pavithree_eli5",
    "FiscalNote_billsum",
    "Harley-ml_lesswrong",
    "jlohding_sp500-edgar-10k",
    "kritsadaK_EDGAR-CORPUS-Financial-Summarization",
    "macadeliccc_US-SupremeCourtVerdicts",
    "rohitsaxena_MovieSum",
    "v1ctor10_section1_annual_reports",
];
const DEFAULT_STRATEGIES = ["zero_shot", "zero_shot_with_samples_reason", "user_interaction", "hybrid"];
const DEFAULT_EXPERIMENTS = Array.from({ length: 30 }, (_, i) => `Experiment${i + 1}`);
const METRIC_KEYS = [
  "precision",
  "recall",
  "f1",
] as const;

type MetricKey = (typeof METRIC_KEYS)[number];

async function fetchJson(url: string) {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to load ${url} (${res.status})`);
  }
  return res.json();
}

// user_interaction output has an extra habit-key level: persona → habit → task
// Flatten it to persona → task so the rest of the UI works uniformly.
function normalizeOutputJson(json: any): any {
  if (!json || typeof json !== "object") return json;
  const result: any = {};
  for (const [personaKey, personaVal] of Object.entries(json)) {
    if (personaKey === "args" || typeof personaVal !== "object" || personaVal === null) {
      result[personaKey] = personaVal;
      continue;
    }
    const firstVal = Object.values(personaVal as object)[0];
    const isHabitLevel =
      firstVal !== null &&
      typeof firstVal === "object" &&
      !Array.isArray(firstVal) &&
      !("messages" in (firstVal as object)) &&
      !("scores" in (firstVal as object));

    if (isHabitLevel) {
      // merge all habit buckets into a single task map
      const merged: any = {};
      for (const habitVal of Object.values(personaVal as object)) {
        Object.assign(merged, habitVal);
      }
      result[personaKey] = merged;
    } else {
      result[personaKey] = personaVal;
    }
  }
  return result;
}

function formatScore(v: any) {
  if (v === undefined || v === null || Number.isNaN(Number(v))) return "—";
  return Number(v).toFixed(4);
}

function avg(arr: number[]) {
  return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : null;
}

function getTaskIds(run: LoadedRun | null, personaId: string) {
  if (!run || !personaId) return [];
  const outputTasks = Object.keys(run.outputJson?.[personaId] || {});
  const evalTasks = Object.keys(run.evalJson?.[personaId] || {});
  return Array.from(new Set([...outputTasks, ...evalTasks])).sort((a, b) => {
    const an = Number(String(a).replace(/\D/g, ""));
    const bn = Number(String(b).replace(/\D/g, ""));
    return an - bn;
  });
}

function getPersonaIds(runs: LoadedRun[]) {
  const ids = new Set<string>();
  for (const run of runs) {
    Object.keys(run.outputJson || {}).forEach((k) => {
      if (k !== "args") ids.add(k);
    });
    Object.keys(run.evalJson || {}).forEach((k) => {
      if (k !== "args") ids.add(k);
    });
  }
  return Array.from(ids).sort((a, b) => Number(a) - Number(b));
}

function getGroundTruthUnits(runs: LoadedRun[], personaId: string, taskId: string) {
  for (const run of runs) {
    const gt = run.evalJson?.[personaId]?.[taskId]?.ground_truth_units;
    if (gt?.length) return gt as string[];
  }
  return [];
}

function buildMatchMap(evalTask: any) {
  const map: Record<string, string[]> = {};
  const pairs = evalTask?.matched_pairs || [];
  for (const p of pairs) {
    map[p.ground_truth] = p.predicted || [];
  }
  return map;
}

function getRunAverageScores(run: LoadedRun) {
  const values: Record<MetricKey, number[]> = {
    precision: [],
    recall: [],
    f1: [],
  };

  const evalJson = run.evalJson || {};
  for (const personaKey of Object.keys(evalJson)) {
    if (personaKey === "args") continue;
    const personaBlock = evalJson[personaKey];
    if (!personaBlock || typeof personaBlock !== "object") continue;

    for (const taskKey of Object.keys(personaBlock)) {
      const scores = personaBlock?.[taskKey]?.scores;
      if (!scores) continue;

      for (const key of METRIC_KEYS) {
        const v = scores[key];
        if (v !== undefined && v !== null && !Number.isNaN(Number(v))) {
          values[key].push(Number(v));
        }
      }
    }
  }

  return {
    precision: avg(values.precision),
    recall: avg(values.recall),
    f1: avg(values.f1),
  };
}

const SUBCATEGORIES = ["user_specified", "data_specified"] as const;
type Subcategory = (typeof SUBCATEGORIES)[number];

function getRunPredComposition(run: LoadedRun) {
  const counts: Record<Subcategory, number[]> = { user_specified: [], data_specified: [] };

  const evalJson = run.evalJson || {};
  for (const personaKey of Object.keys(evalJson)) {
    if (personaKey === "args") continue;
    const personaBlock = evalJson[personaKey];
    if (!personaBlock || typeof personaBlock !== "object") continue;
    for (const taskKey of Object.keys(personaBlock)) {
      const subScores = personaBlock?.[taskKey]?.subcategory_scores;
      if (!subScores) continue;
      for (const cat of SUBCATEGORIES) {
        const n = subScores[cat]?.n_pred;
        if (n !== undefined && n !== null) counts[cat].push(Number(n));
      }
    }
  }

  const totalUs = counts.user_specified.reduce((a, b) => a + b, 0);
  const totalDs = counts.data_specified.reduce((a, b) => a + b, 0);
  const total = totalUs + totalDs;
  const nTasks = Math.max(counts.user_specified.length, counts.data_specified.length);

  return {
    user_specified: { total: totalUs, avg: nTasks ? totalUs / nTasks : null, ratio: total ? totalUs / total : null },
    data_specified: { total: totalDs, avg: nTasks ? totalDs / nTasks : null, ratio: total ? totalDs / total : null },
    nTasks,
  };
}

function getRunAverageSubcategoryScores(run: LoadedRun) {
  const values: Record<Subcategory, Record<MetricKey, number[]>> = {
    user_specified: { precision: [], recall: [], f1: [] },
    data_specified: { precision: [], recall: [], f1: [] },
  };

  const evalJson = run.evalJson || {};
  for (const personaKey of Object.keys(evalJson)) {
    if (personaKey === "args") continue;
    const personaBlock = evalJson[personaKey];
    if (!personaBlock || typeof personaBlock !== "object") continue;

    for (const taskKey of Object.keys(personaBlock)) {
      const subScores = personaBlock?.[taskKey]?.subcategory_scores;
      if (!subScores) continue;

      for (const cat of SUBCATEGORIES) {
        const catScores = subScores[cat];
        if (!catScores) continue;
        for (const key of METRIC_KEYS) {
          const v = catScores[key];
          if (v !== undefined && v !== null && !Number.isNaN(Number(v))) {
            values[cat][key].push(Number(v));
          }
        }
      }
    }
  }

  const result: Record<Subcategory, Record<MetricKey, number | null>> = {} as any;
  for (const cat of SUBCATEGORIES) {
    result[cat] = {
      precision: avg(values[cat].precision),
      recall: avg(values[cat].recall),
      f1: avg(values[cat].f1),
    };
  }
  return result;
}


function panelStyle(): React.CSSProperties {
  return {
    border: "1px solid #ddd",
    padding: 16,
    marginBottom: 16,
    borderRadius: 10,
    background: "white",
  };
}

function sectionTitleStyle(): React.CSSProperties {
  return {
    marginTop: 0,
    marginBottom: 12,
    fontSize: 20,
  };
}

function smallLabelStyle(): React.CSSProperties {
  return {
    display: "block",
    fontSize: 12,
    fontWeight: 600,
    color: "#555",
    marginBottom: 6,
  };
}

function inputStyle(): React.CSSProperties {
  return {
    width: "100%",
    padding: "10px 12px",
    borderRadius: 8,
    border: "1px solid #ccc",
    boxSizing: "border-box",
  };
}

function buttonStyle(): React.CSSProperties {
  return {
    padding: "10px 14px",
    borderRadius: 8,
    border: "1px solid #2563eb",
    background: "#2563eb",
    color: "white",
    fontWeight: 600,
    cursor: "pointer",
  };
}

function secondaryButtonStyle(): React.CSSProperties {
  return {
    padding: "10px 14px",
    borderRadius: 8,
    border: "1px solid #ccc",
    background: "#f8f8f8",
    color: "#222",
    fontWeight: 600,
    cursor: "pointer",
  };
}

function MessageView({ messages }: { messages: any[] }) {
  const filteredMessages = (messages || []).filter((m: any, idx: number) => {
    if (idx === 0) return true;
    return m.role === "mimic_user";
  });

  return (
    <div
      style={{
        maxHeight: 520,
        overflow: "auto",
        border: "1px solid #eee",
        borderRadius: 10,
        padding: 12,
      }}
    >
      {filteredMessages.length === 0 ? (
        <div style={{ color: "#666" }}>No messages found.</div>
      ) : (
        filteredMessages.map((m: any, idx: number) => {
          const isFirstTurn = idx === 0;

          return (
            <div
              key={idx}
              style={{
                marginBottom: 16,
                padding: 12,
                borderRadius: 10,
                background: "#fafafa",
                border: "1px solid #eee",
              }}
            >
              <div
                style={{
                  fontSize: 12,
                  color: "#666",
                  marginBottom: 10,
                  fontWeight: 600,
                }}
              >
                {isFirstTurn ? "Turn 1" : `Mimic User Turn ${idx + 1}`}
              </div>

              {isFirstTurn ? (
                <div>
                  <div
                    style={{
                      fontSize: 12,
                      fontWeight: 700,
                      color: "#444",
                      marginBottom: 4,
                    }}
                  >
                    Output
                  </div>
                  <div
                    style={{
                      whiteSpace: "pre-wrap",
                      lineHeight: 1.5,
                      background: "#f6f8fa",
                      border: "1px solid #e5e7eb",
                      borderRadius: 8,
                      padding: 10,
                    }}
                  >
                    {m.output && String(m.output).trim() !== ""
                      ? m.output
                      : "(empty output)"}
                  </div>
                </div>
              ) : (
                <>
                  <div style={{ marginBottom: 10 }}>
                    <div
                      style={{
                        fontSize: 12,
                        fontWeight: 700,
                        color: "#444",
                        marginBottom: 4,
                      }}
                    >
                      Input
                    </div>
                    <div
                      style={{
                        whiteSpace: "pre-wrap",
                        lineHeight: 1.5,
                        background: "#eef6ff",
                        border: "1px solid #dbeafe",
                        borderRadius: 8,
                        padding: 10,
                      }}
                    >
                      {m.input && String(m.input).trim() !== ""
                        ? m.input
                        : "(empty input)"}
                    </div>
                  </div>

                  <div>
                    <div
                      style={{
                        fontSize: 12,
                        fontWeight: 700,
                        color: "#444",
                        marginBottom: 4,
                      }}
                    >
                      Output
                    </div>
                    <div
                      style={{
                        whiteSpace: "pre-wrap",
                        lineHeight: 1.5,
                        background: "#f6f8fa",
                        border: "1px solid #e5e7eb",
                        borderRadius: 8,
                        padding: 10,
                      }}
                    >
                      {m.output && String(m.output).trim() !== ""
                        ? m.output
                        : "(empty output)"}
                    </div>
                  </div>
                </>
              )}
            </div>
          );
        })
      )}
    </div>
  );
}

function ConversationHistoryView({ history }: { history: any[] }) {
  if (!history || history.length === 0) return <div style={{ color: "#666" }}>No conversation history.</div>;
  return (
    <div style={{ maxHeight: 520, overflow: "auto", border: "1px solid #eee", borderRadius: 10, padding: 12 }}>
      {history.map((entry: any, idx: number) => {
        const isAgent = entry.role === "agent";
        return (
          <div
            key={idx}
            style={{
              marginBottom: 12,
              display: "flex",
              flexDirection: isAgent ? "row-reverse" : "row",
              gap: 8,
            }}
          >
            <div style={{ flex: "0 0 auto", fontSize: 11, fontWeight: 700, color: isAgent ? "#2563eb" : "#16a34a", paddingTop: 4, minWidth: 60, textAlign: isAgent ? "right" : "left" }}>
              {isAgent ? "Agent" : "User"}
              <div style={{ fontWeight: 400, color: "#999" }}>T{entry.turn}</div>
            </div>
            <div style={{ flex: 1 }}>
              <div style={{
                whiteSpace: "pre-wrap", lineHeight: 1.5, padding: "10px 12px", borderRadius: 10, fontSize: 14,
                background: isAgent ? "#eff6ff" : "#f0fdf4",
                border: isAgent ? "1px solid #bfdbfe" : "1px solid #bbf7d0",
              }}>
                {entry.content}
              </div>
              {entry.thought && (
                <div style={{ marginTop: 4, fontSize: 12, color: "#888", fontStyle: "italic", paddingLeft: 4 }}>
                  💭 {entry.thought}
                </div>
              )}
              {entry.timestamp && (
                <div style={{ marginTop: 2, fontSize: 11, color: "#bbb", paddingLeft: 4 }}>{entry.timestamp}</div>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function ReflectionHistoryView({ messages }: { messages: any[] }) {
  const reflections = (messages || []).filter((m: any) => m.role === "aunu_agent" && m.action === "reflect");
  if (reflections.length === 0) return <div style={{ color: "#666" }}>No reflections found.</div>;
  return (
    <div style={{ maxHeight: 520, overflow: "auto", border: "1px solid #eee", borderRadius: 10, padding: 12 }}>
      {reflections.map((m: any, idx: number) => (
        <div key={idx} style={{ marginBottom: 16, padding: 12, borderRadius: 10, background: "#fafafa", border: "1px solid #eee" }}>
          <div style={{ fontSize: 12, color: "#666", marginBottom: 8, fontWeight: 600 }}>
            Reflection #{idx + 1}
            {m.start_time && <span style={{ fontWeight: 400, marginLeft: 8 }}>{m.start_time}</span>}
          </div>
          {m.identified_ambiguity && (
            <div style={{ marginBottom: 8 }}>
              <div style={{ fontSize: 12, fontWeight: 700, color: "#7c3aed", marginBottom: 4 }}>Identified ambiguity</div>
              <div style={{ whiteSpace: "pre-wrap", lineHeight: 1.5, background: "#f5f3ff", border: "1px solid #ddd6fe", borderRadius: 8, padding: 10 }}>
                {m.identified_ambiguity}
              </div>
            </div>
          )}
          <div>
            <div style={{ fontSize: 12, fontWeight: 700, color: "#444", marginBottom: 4 }}>Raw output</div>
            <div style={{ whiteSpace: "pre-wrap", lineHeight: 1.5, background: "#f6f8fa", border: "1px solid #e5e7eb", borderRadius: 8, padding: 10 }}>
              {m.output && String(m.output).trim() !== "" ? m.output : "(empty)"}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

function FormatReflectionHistoryView({ history }: { history: any[] }) {
  if (!history || history.length === 0) return <div style={{ color: "#666" }}>No format reflections found.</div>;
  return (
    <div style={{ maxHeight: 520, overflow: "auto", border: "1px solid #eee", borderRadius: 10, padding: 12 }}>
      {history.map((entry: any, idx: number) => (
        <div key={idx} style={{ marginBottom: 16, padding: 12, borderRadius: 10, background: "#fafafa", border: "1px solid #eee" }}>
          <div style={{ fontSize: 12, color: "#666", marginBottom: 8, fontWeight: 600 }}>
            Format Reflection Turn {entry.turn ?? idx + 1}
          </div>
          {entry.format_gaps && (
            <div style={{ marginBottom: 8 }}>
              <div style={{ fontSize: 12, fontWeight: 700, color: "#b45309", marginBottom: 4 }}>Format gaps</div>
              <div style={{ whiteSpace: "pre-wrap", lineHeight: 1.5, background: "#fffbeb", border: "1px solid #fde68a", borderRadius: 8, padding: 10 }}>
                {typeof entry.format_gaps === "string" ? entry.format_gaps : JSON.stringify(entry.format_gaps, null, 2)}
              </div>
            </div>
          )}
          {entry.alignment_issues && (
            <div style={{ marginBottom: 8 }}>
              <div style={{ fontSize: 12, fontWeight: 700, color: "#b91c1c", marginBottom: 4 }}>Alignment issues</div>
              <div style={{ whiteSpace: "pre-wrap", lineHeight: 1.5, background: "#fef2f2", border: "1px solid #fecaca", borderRadius: 8, padding: 10 }}>
                {typeof entry.alignment_issues === "string" ? entry.alignment_issues : JSON.stringify(entry.alignment_issues, null, 2)}
              </div>
            </div>
          )}
          {entry.rewritten_requirement && (
            <div>
              <div style={{ fontSize: 12, fontWeight: 700, color: "#166534", marginBottom: 4 }}>Rewritten requirement</div>
              <div style={{ whiteSpace: "pre-wrap", lineHeight: 1.5, background: "#f0fdf4", border: "1px solid #bbf7d0", borderRadius: 8, padding: 10 }}>
                {entry.rewritten_requirement}
              </div>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

function DataInspectionHistoryView({ history }: { history: any[] }) {
  const [openIdx, setOpenIdx] = React.useState<number | null>(null);
  if (!history || history.length === 0) return <div style={{ color: "#666" }}>No data inspection history.</div>;
  return (
    <div style={{ maxHeight: 560, overflow: "auto", border: "1px solid #eee", borderRadius: 10, padding: 12 }}>
      {history.map((entry: any, idx: number) => (
        <div key={idx} style={{ marginBottom: 12, border: "1px solid #e5e7eb", borderRadius: 10, overflow: "hidden" }}>
          <div
            onClick={() => setOpenIdx(openIdx === idx ? null : idx)}
            style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "10px 14px", background: "#f6f8fa", cursor: "pointer", userSelect: "none" }}
          >
            <div style={{ fontWeight: 700, fontSize: 14 }}>
              Inspection #{entry.inspection_idx} · Step {entry.step} · {entry.n_samples} sample{entry.n_samples !== 1 ? "s" : ""}
            </div>
            <div style={{ display: "flex", gap: 16, alignItems: "center" }}>
              <div style={{ fontSize: 12, color: "#666" }}>
                rows: [{(entry.row_indices || []).join(", ")}]
              </div>
              {entry.query && <div style={{ fontSize: 12, color: "#2563eb" }}>"{entry.query}"</div>}
              <div style={{ fontSize: 12, color: "#999" }}>{openIdx === idx ? "▲" : "▼"}</div>
            </div>
          </div>
          {openIdx === idx && (
            <div style={{ padding: 14, background: "white" }}>
              {entry.timestamp && <div style={{ fontSize: 11, color: "#bbb", marginBottom: 8 }}>{entry.timestamp}</div>}
              {(entry.samples || []).map((sample: any, si: number) => (
                <div key={si} style={{ marginBottom: 10, padding: 10, background: "#fafafa", border: "1px solid #eee", borderRadius: 8 }}>
                  <div style={{ fontSize: 12, fontWeight: 700, color: "#555", marginBottom: 6 }}>
                    Row {sample._row_idx ?? si}
                  </div>
                  {Object.entries(sample).filter(([k]) => k !== "_row_idx").map(([k, v]) => (
                    <div key={k} style={{ marginBottom: 6 }}>
                      <span style={{ fontSize: 12, fontWeight: 600, color: "#444" }}>{k}: </span>
                      <span style={{ fontSize: 13, whiteSpace: "pre-wrap", lineHeight: 1.4 }}>{String(v).slice(0, 400)}{String(v).length > 400 ? "…" : ""}</span>
                    </div>
                  ))}
                </div>
              ))}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

function MatchedPairsView({ pairs }: { pairs: any[] }) {
  return (
    <div
      style={{
        maxHeight: 420,
        overflow: "auto",
        border: "1px solid #eee",
        borderRadius: 10,
        padding: 12,
      }}
    >
      {pairs.length === 0 ? (
        <div style={{ color: "#666" }}>No matched pairs.</div>
      ) : (
        pairs.map((p, idx) => (
          <div
            key={idx}
            style={{
              marginBottom: 14,
              padding: 12,
              border: "1px solid #eee",
              borderRadius: 10,
            }}
          >
            <div style={{ marginBottom: 8 }}>
              <strong>Ground truth</strong>
              <div style={{ marginTop: 6, whiteSpace: "pre-wrap" }}>{p.ground_truth}</div>
            </div>
            <div>
              <strong>Predicted</strong>
              <ul style={{ marginTop: 6, paddingLeft: 20 }}>
                {(p.predicted || []).map((x: string, i: number) => (
                  <li key={i} style={{ marginBottom: 4 }}>
                    {x}
                  </li>
                ))}
              </ul>
            </div>
          </div>
        ))
      )}
    </div>
  );
}

function UnitsView({ title, items }: { title: string; items: string[] }) {
  return (
    <div style={{ ...panelStyle(), minHeight: 320 }}>
      <h3 style={{ marginTop: 0 }}>{title}</h3>
      <div style={{ maxHeight: 360, overflow: "auto" }}>
        {items.length === 0 ? (
          <div style={{ color: "#666" }}>No items.</div>
        ) : (
          <ul style={{ paddingLeft: 20, margin: 0 }}>
            {items.map((u, idx) => (
              <li key={idx} style={{ marginBottom: 8, lineHeight: 1.45 }}>
                {u}
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

function SelectedPairMetricsPanel({
  runs,
  personaId,
  taskId,
}: {
  runs: LoadedRun[];
  personaId: string;
  taskId: string;
}) {
  return (
    <div style={panelStyle()}>
      <h2 style={sectionTitleStyle()}>Selected pair metrics</h2>
      {!personaId || !taskId ? (
        <div style={{ color: "#666" }}>Select a persona_id and task_id.</div>
      ) : runs.length === 0 ? (
        <div style={{ color: "#666" }}>Load experiments first.</div>
      ) : (
        <table style={{ width: "100%", borderCollapse: "collapse" }}>
          <thead>
            <tr style={{ background: "#f6f8fa" }}>
              <th style={{ textAlign: "left", padding: 10, borderBottom: "1px solid #ddd" }}>
                Strategy
              </th>
              <th style={{ textAlign: "left", padding: 10, borderBottom: "1px solid #ddd" }}>
                Experiment
              </th>
              <th style={{ textAlign: "left", padding: 10, borderBottom: "1px solid #ddd" }}>
                Precision
              </th>
              <th style={{ textAlign: "left", padding: 10, borderBottom: "1px solid #ddd" }}>
                Recall
              </th>
              <th style={{ textAlign: "left", padding: 10, borderBottom: "1px solid #ddd" }}>
                F1
              </th>
            </tr>
          </thead>
          <tbody>
            {runs.map((run) => {
              const scores = run.evalJson?.[personaId]?.[taskId]?.scores || {};
              return (
                <tr key={`${run.strategy}-${run.experimentId}-${personaId}-${taskId}`}>
                  <td style={{ padding: 10, borderBottom: "1px solid #eee" }}>{run.strategy}</td>
                  <td style={{ padding: 10, borderBottom: "1px solid #eee" }}>{run.experimentId}</td>
                  <td style={{ padding: 10, borderBottom: "1px solid #eee" }}>
                    {formatScore(scores.precision)}
                  </td>
                  <td style={{ padding: 10, borderBottom: "1px solid #eee" }}>
                    {formatScore(scores.recall)}
                  </td>
                  <td style={{ padding: 10, borderBottom: "1px solid #eee" }}>
                    {formatScore(scores.f1)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </div>
  );
}

function GroundTruthMatrix({
  runs,
  personaId,
  taskId,
}: {
  runs: LoadedRun[];
  personaId: string;
  taskId: string;
}) {
  const gtUnits = getGroundTruthUnits(runs, personaId, taskId);

  return (
    <div style={panelStyle()}>
      <h2 style={sectionTitleStyle()}>Ground truth unit comparison</h2>

      {!personaId || !taskId ? (
        <div style={{ color: "#666" }}>Select persona_id and task_id.</div>
      ) : gtUnits.length === 0 ? (
        <div style={{ color: "#666" }}>No ground-truth units found for this selection.</div>
      ) : (
        <div style={{ overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", minWidth: 1100 }}>
            <thead>
              <tr style={{ background: "#f6f8fa" }}>
                <th
                  style={{
                    textAlign: "left",
                    padding: 12,
                    borderBottom: "1px solid #ddd",
                    width: "28%",
                    verticalAlign: "top",
                  }}
                >
                  Ground truth unit
                </th>

                {runs.map((r) => (
                  <th
                    key={`${r.strategy}-${r.experimentId}`}
                    style={{
                      textAlign: "left",
                      padding: 12,
                      borderBottom: "1px solid #ddd",
                      verticalAlign: "top",
                    }}
                  >
                    <div style={{ fontWeight: 700 }}>{r.strategy}</div>
                    <div style={{ fontSize: 12, color: "#666", marginTop: 4 }}>
                      {r.experimentId}
                    </div>
                  </th>
                ))}
              </tr>
            </thead>

            <tbody>
              {gtUnits.map((gt, i) => (
                <tr key={i}>
                  <td
                    style={{
                      padding: 12,
                      borderBottom: "1px solid #eee",
                      verticalAlign: "top",
                      background: "#fafbfc",
                      lineHeight: 1.5,
                      fontWeight: 500,
                    }}
                  >
                    {gt}
                  </td>

                  {runs.map((r) => {
                    const evalTask = r.evalJson?.[personaId]?.[taskId];
                    const matchMap = buildMatchMap(evalTask);
                    const matched = matchMap[gt] || [];

                    return (
                      <td
                        key={`${r.strategy}-${r.experimentId}-${gt}`}
                        style={{
                          padding: 12,
                          borderBottom: "1px solid #eee",
                          verticalAlign: "top",
                        }}
                      >
                        {matched.length > 0 ? (
                          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                            {matched.map((m, idx) => (
                              <div
                                key={idx}
                                style={{
                                  background: "#dcfce7",
                                  border: "1px solid #22c55e",
                                  color: "#166534",
                                  borderRadius: 8,
                                  padding: "8px 10px",
                                  fontSize: 14,
                                }}
                              >
                                {m}
                              </div>
                            ))}
                          </div>
                        ) : (
                          <div
                            style={{
                              background: "#f3f4f6",
                              border: "1px solid #e5e7eb",
                              color: "#9ca3af",
                              borderRadius: 8,
                              padding: "8px 10px",
                              fontSize: 14,
                              fontStyle: "italic",
                            }}
                          >
                            unmatched
                          </div>
                        )}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function StrategyRunPanel({
  run,
  personaId,
  taskId,
}: {
  run: LoadedRun;
  personaId: string;
  taskId: string;
}) {
  const outputTask = run.outputJson?.[personaId]?.[taskId];
  const evalTask = run.evalJson?.[personaId]?.[taskId];
  const messages = outputTask?.messages || [];
  const counts = evalTask?.counts || {};
  const conversationHistory = outputTask?.conversation_history || [];
  const dataInspectionHistory = outputTask?.data_inspection_history || [];
  const formatReflectionHistory = outputTask?.format_reflection_history || [];

  const predictedRequirement =
    messages.filter((m: any) => m.role === "mimic_user").slice(-1)[0]?.output ||
    messages.slice(-1)[0]?.output ||
    "";

  return (
    <div style={panelStyle()}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
        <h2 style={{ margin: 0, fontSize: 20 }}>
          {run.strategy} · {run.experimentId}
        </h2>
        <div style={{ fontSize: 12, color: "#666" }}>{run.dataset}</div>
      </div>

      {run.error ? (
        <div style={{ color: "#b42318", marginBottom: 12 }}>{run.error}</div>
      ) : null}

      {!outputTask && !evalTask ? (
        <div style={{ color: "#666" }}>No data for selected persona/task.</div>
      ) : (
        <>
          <div style={panelStyle()}>
            <h3 style={{ marginTop: 0 }}>Predicted task requirement</h3>
            <div style={{ whiteSpace: "pre-wrap", lineHeight: 1.5 }}>
              {predictedRequirement || "No predicted requirement found."}
            </div>
          </div>

          <div style={panelStyle()}>
            <h3 style={{ marginTop: 0 }}>Counts</h3>
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <tbody>
                {Object.entries(counts).map(([k, v]) => (
                  <tr key={k}>
                    <td style={{ borderBottom: "1px solid #eee", padding: "8px 4px", fontWeight: 600 }}>{k}</td>
                    <td style={{ borderBottom: "1px solid #eee", padding: "8px 4px" }}>{String(v)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div style={panelStyle()}>
            <h3 style={{ marginTop: 0 }}>Conversation trajectory</h3>
            <MessageView messages={messages} />
          </div>

          {messages.some((m: any) => m.role === "aunu_agent" && m.action === "reflect") && (
            <div style={panelStyle()}>
              <h3 style={{ marginTop: 0 }}>Reflection history</h3>
              <ReflectionHistoryView messages={messages} />
            </div>
          )}

          {conversationHistory.length > 0 && (
            <div style={panelStyle()}>
              <h3 style={{ marginTop: 0 }}>Conversation history</h3>
              <ConversationHistoryView history={conversationHistory} />
            </div>
          )}

          {dataInspectionHistory.length > 0 && (
            <div style={panelStyle()}>
              <h3 style={{ marginTop: 0 }}>Data inspection history</h3>
              <DataInspectionHistoryView history={dataInspectionHistory} />
            </div>
          )}

          {formatReflectionHistory.length > 0 && (
            <div style={panelStyle()}>
              <h3 style={{ marginTop: 0 }}>Format reflection history</h3>
              <FormatReflectionHistoryView history={formatReflectionHistory} />
            </div>
          )}

          <div style={panelStyle()}>
            <h3 style={{ marginTop: 0 }}>Matched pairs</h3>
            <MatchedPairsView pairs={evalTask?.matched_pairs || []} />
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
            <UnitsView title="Missing units" items={evalTask?.missing_units || []} />
            <UnitsView title="Hallucinated units" items={evalTask?.hallucinated_units || []} />
          </div>
        </>
      )}
    </div>
  );
}

function FormatReflectTab({
  runs,
  baseUrl,
}: {
  runs: LoadedRun[];
  baseUrl: string;
}) {
  const [frPersonaId, setFrPersonaId] = React.useState("");
  const [frTaskId, setFrTaskId] = React.useState("");
  const [frExperimentIds, setFrExperimentIds] = React.useState<Record<string, string>>({});
  const [frRuns, setFrRuns] = React.useState<LoadedRun[]>([]);
  const [frLoading, setFrLoading] = React.useState(false);

  // When the parent loads runs, sync experiment ids and load format-reflect runs
  React.useEffect(() => {
    if (runs.length === 0) return;
    const ids: Record<string, string> = {};
    for (const r of runs) ids[r.strategy] = r.experimentId;
    setFrExperimentIds(ids);
  }, [runs]);

  const allStrategies = runs.map((r) => r.strategy);

  async function loadFrRuns() {
    if (runs.length === 0) return;
    setFrLoading(true);
    const next: LoadedRun[] = [];
    for (const run of runs) {
      const expId = frExperimentIds[run.strategy] || run.experimentId;
      const outputUrl = `${baseUrl}/${run.dataset}/${run.strategy}/${expId}/output.json`;
      try {
        const rawOutput = await fetchJson(outputUrl);
        next.push({ ...run, experimentId: expId, outputJson: normalizeOutputJson(rawOutput), evalJson: run.evalJson });
      } catch {
        next.push({ ...run, experimentId: expId, outputJson: null, evalJson: null, error: `Failed to load ${run.strategy}/${expId}` });
      }
    }
    setFrRuns(next);
    const personas = getPersonaIds(next);
    const p = personas[0] || "";
    setFrPersonaId(p);
    const firstLoaded = next.find((r) => r.outputJson) || null;
    setFrTaskId(getTaskIds(firstLoaded, p)[0] || "");
    setFrLoading(false);
  }

  // Auto-load when experiment ids change and we already have runs
  const activeRuns = frRuns.length > 0 ? frRuns : runs;

  const frPersonaIds = useMemo(() => getPersonaIds(activeRuns), [activeRuns]);
  const frTaskIds = useMemo(() => {
    const firstLoaded = activeRuns.find((r) => r.outputJson) || null;
    return getTaskIds(firstLoaded, frPersonaId);
  }, [activeRuns, frPersonaId]);

  // Auto-select first persona/task when they become available
  React.useEffect(() => {
    if (!frPersonaId && frPersonaIds.length > 0) setFrPersonaId(frPersonaIds[0]);
  }, [frPersonaIds]);
  React.useEffect(() => {
    if (!frTaskId && frTaskIds.length > 0) setFrTaskId(frTaskIds[0]);
  }, [frTaskIds]);

  const runsWithData = activeRuns.filter((r) => {
    const history = r.outputJson?.[frPersonaId]?.[frTaskId]?.format_reflection_history;
    return Array.isArray(history) && history.length > 0;
  });

  return (
    <div>
      <div style={panelStyle()}>
        <h2 style={sectionTitleStyle()}>Format Reflect & Rewrite</h2>
        <p style={{ color: "#57606a", marginTop: 0, marginBottom: 12 }}>
          Shows the <code>format_reflection_history</code> for each loaded run — format gaps, alignment issues, and the rewritten requirement per turn.
        </p>

        {allStrategies.length > 0 && (
          <div style={{ marginBottom: 12 }}>
            <div style={smallLabelStyle()}>Experiment per strategy</div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(180px, 1fr))", gap: 10, marginBottom: 10 }}>
              {allStrategies.map((strategy) => (
                <div key={strategy}>
                  <label style={smallLabelStyle()}>{strategy}</label>
                  <select
                    style={inputStyle()}
                    value={frExperimentIds[strategy] || DEFAULT_EXPERIMENTS[0]}
                    onChange={(e) =>
                      setFrExperimentIds((prev) => ({ ...prev, [strategy]: e.target.value }))
                    }
                  >
                    {DEFAULT_EXPERIMENTS.map((x) => (
                      <option key={x} value={x}>{x}</option>
                    ))}
                  </select>
                </div>
              ))}
            </div>
            <button style={buttonStyle()} onClick={loadFrRuns} disabled={frLoading}>
              {frLoading ? "Loading..." : "Load for this tab"}
            </button>
          </div>
        )}

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginTop: 12 }}>
          <div>
            <label style={smallLabelStyle()}>persona_id</label>
            <select
              style={inputStyle()}
              value={frPersonaId}
              onChange={(e) => {
                setFrPersonaId(e.target.value);
                const firstLoaded = activeRuns.find((r) => r.outputJson) || null;
                setFrTaskId(getTaskIds(firstLoaded, e.target.value)[0] || "");
              }}
            >
              <option value="">Select persona</option>
              {frPersonaIds.map((p) => (
                <option key={p} value={p}>{p}</option>
              ))}
            </select>
          </div>
          <div>
            <label style={smallLabelStyle()}>task_id</label>
            <select
              style={inputStyle()}
              value={frTaskId}
              onChange={(e) => setFrTaskId(e.target.value)}
            >
              <option value="">Select task</option>
              {frTaskIds.map((t) => (
                <option key={t} value={t}>{t}</option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {runs.length === 0 ? (
        <div style={{ ...panelStyle(), color: "#666" }}>Load experiments first.</div>
      ) : !frPersonaId || !frTaskId ? (
        <div style={{ ...panelStyle(), color: "#666" }}>Select a persona and task.</div>
      ) : runsWithData.length === 0 ? (
        <div style={{ ...panelStyle(), color: "#666" }}>No format reflection history found for this selection.</div>
      ) : (
        runsWithData.map((run) => {
          const history: any[] = run.outputJson?.[frPersonaId]?.[frTaskId]?.format_reflection_history || [];
          const rewriteHistory: any[] = run.outputJson?.[frPersonaId]?.[frTaskId]?.rewrite_history || [];
          const finalReq = run.outputJson?.[frPersonaId]?.[frTaskId]?.task_requirement_final || "";
          return (
            <div key={`${run.strategy}-${run.experimentId}`} style={panelStyle()}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
                <h2 style={{ margin: 0, fontSize: 20 }}>
                  {run.strategy} · {run.experimentId}
                </h2>
                <div style={{ fontSize: 12, color: "#666" }}>{run.dataset}</div>
              </div>

              {finalReq && (
                <div style={{ marginBottom: 16 }}>
                  <div style={{ fontSize: 13, fontWeight: 700, color: "#166534", marginBottom: 6 }}>Final task requirement</div>
                  <div style={{ whiteSpace: "pre-wrap", lineHeight: 1.5, background: "#f0fdf4", border: "1px solid #bbf7d0", borderRadius: 8, padding: 12 }}>
                    {finalReq}
                  </div>
                </div>
              )}

              <div style={{ fontSize: 13, fontWeight: 700, color: "#444", marginBottom: 8 }}>
                Reflection history ({history.length} turn{history.length !== 1 ? "s" : ""})
              </div>

              {history.map((entry: any, idx: number) => (
                <div key={idx} style={{ marginBottom: 20, border: "1px solid #e5e7eb", borderRadius: 10, overflow: "hidden" }}>
                  <div style={{ background: "#f6f8fa", padding: "10px 14px", fontWeight: 700, fontSize: 14, borderBottom: "1px solid #e5e7eb" }}>
                    Turn {entry.turn ?? idx + 1}
                  </div>
                  <div style={{ padding: 14 }}>
                    {entry.format_gaps && (
                      <div style={{ marginBottom: 12 }}>
                        <div style={{ fontSize: 12, fontWeight: 700, color: "#b45309", marginBottom: 4 }}>Format gaps</div>
                        <div style={{ whiteSpace: "pre-wrap", lineHeight: 1.5, background: "#fffbeb", border: "1px solid #fde68a", borderRadius: 8, padding: 10 }}>
                          {typeof entry.format_gaps === "string" ? entry.format_gaps : JSON.stringify(entry.format_gaps, null, 2)}
                        </div>
                      </div>
                    )}
                    {entry.alignment_issues && (
                      <div style={{ marginBottom: 12 }}>
                        <div style={{ fontSize: 12, fontWeight: 700, color: "#b91c1c", marginBottom: 4 }}>Alignment issues</div>
                        <div style={{ whiteSpace: "pre-wrap", lineHeight: 1.5, background: "#fef2f2", border: "1px solid #fecaca", borderRadius: 8, padding: 10 }}>
                          {typeof entry.alignment_issues === "string" ? entry.alignment_issues : JSON.stringify(entry.alignment_issues, null, 2)}
                        </div>
                      </div>
                    )}
                    {entry.rewritten_requirement && (
                      <div>
                        <div style={{ fontSize: 12, fontWeight: 700, color: "#166534", marginBottom: 4 }}>Rewritten requirement</div>
                        <div style={{ whiteSpace: "pre-wrap", lineHeight: 1.5, background: "#f0fdf4", border: "1px solid #bbf7d0", borderRadius: 8, padding: 10 }}>
                          {entry.rewritten_requirement}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              ))}

              {rewriteHistory.length > 0 && (
                <>
                  <div style={{ fontSize: 13, fontWeight: 700, color: "#444", marginTop: 20, marginBottom: 8 }}>
                    Rewrite history ({rewriteHistory.length} rewrite{rewriteHistory.length !== 1 ? "s" : ""})
                  </div>
                  {rewriteHistory.map((entry: any, idx: number) => {
                    const scores = entry.scores || {};
                    const subScores = entry.subcategory_scores || {};
                    return (
                      <div key={idx} style={{ marginBottom: 20, border: "1px solid #dbeafe", borderRadius: 10, overflow: "hidden" }}>
                        <div style={{ background: "#eff6ff", padding: "10px 14px", fontWeight: 700, fontSize: 14, borderBottom: "1px solid #dbeafe" }}>
                          Rewrite at turn {entry.turn ?? idx + 1}
                        </div>
                        <div style={{ padding: 14 }}>
                          {/* Overall scores */}
                          <div style={{ marginBottom: 12 }}>
                            <div style={{ fontSize: 12, fontWeight: 700, color: "#1d4ed8", marginBottom: 6 }}>Overall scores</div>
                            <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
                              {["precision", "recall", "f1"].map((k) => (
                                scores[k] !== undefined && (
                                  <div key={k} style={{ background: "#dbeafe", borderRadius: 6, padding: "4px 10px", fontSize: 13 }}>
                                    <span style={{ fontWeight: 600 }}>{k}</span>: {scores[k]}
                                  </div>
                                )
                              ))}
                            </div>
                          </div>
                          {/* Subcategory scores */}
                          {Object.keys(subScores).length > 0 && (
                            <div style={{ marginBottom: 12 }}>
                              <div style={{ fontSize: 12, fontWeight: 700, color: "#6d28d9", marginBottom: 6 }}>Subcategory scores</div>
                              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
                                {Object.entries(subScores).map(([cat, catScores]: [string, any]) => (
                                  <div key={cat} style={{ background: "#f5f3ff", border: "1px solid #ddd6fe", borderRadius: 8, padding: "8px 12px" }}>
                                    <div style={{ fontWeight: 700, fontSize: 12, color: "#5b21b6", marginBottom: 4 }}>{cat}</div>
                                    <div style={{ fontSize: 12, color: "#444" }}>
                                      n_gt: {catScores.n_gt} · n_pred: {catScores.n_pred} · tp: {catScores.tp}
                                    </div>
                                    <div style={{ display: "flex", gap: 8, marginTop: 4, flexWrap: "wrap" }}>
                                      {["precision", "recall", "f1"].map((k) => (
                                        catScores[k] !== undefined && (
                                          <div key={k} style={{ background: "#ede9fe", borderRadius: 4, padding: "2px 8px", fontSize: 12 }}>
                                            <span style={{ fontWeight: 600 }}>{k}</span>: {catScores[k]}
                                          </div>
                                        )
                                      ))}
                                    </div>
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}
                          {/* Rewritten requirement */}
                          {entry.requirement && (
                            <div>
                              <div style={{ fontSize: 12, fontWeight: 700, color: "#166534", marginBottom: 4 }}>Rewritten requirement</div>
                              <div style={{ whiteSpace: "pre-wrap", lineHeight: 1.5, background: "#f0fdf4", border: "1px solid #bbf7d0", borderRadius: 8, padding: 10 }}>
                                {entry.requirement}
                              </div>
                            </div>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </>
              )}
            </div>
          );
        })
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Hybrid Trace Tab
// ---------------------------------------------------------------------------

function TextBlock({ text, bg, border, maxHeight = 200 }: { text: string; bg: string; border: string; maxHeight?: number }) {
  return (
    <div style={{ whiteSpace: "pre-wrap", lineHeight: 1.6, background: bg, border: `1px solid ${border}`, borderRadius: 8, padding: 10, fontSize: 13, maxHeight, overflow: "auto" }}>
      {text && text.trim() !== "" ? text : "(empty)"}
    </div>
  );
}

function HybridIterationTrace({ taskData }: { taskData: any }) {
  const messages: any[] = taskData?.messages || [];
  const routerHistory: any[] = taskData?.router_history || [];

  // Extract elevator pitch: prefer stored field, fall back to parsing the first router's input prompt
  const elevatorPitch: string = (() => {
    if (taskData?.elevator_pitch) return taskData.elevator_pitch;
    const firstRouter = messages.find((m: any) => m.role === "aunu_agent" && m.action === "router");
    if (firstRouter?.input) {
      const match = firstRouter.input.match(/Initial User Requirement\s*\n([\s\S]*?)(?:\n\n##|\n##|\n---|\n#\s)/);
      if (match) return match[1].trim();
    }
    return "";
  })();
  // Build initial requirement (zero-shot draft expanded from elevator pitch)
  const initReq: string =
    taskData?.zero_shot_draft ||
    messages.find((m: any) => m.action === "zero_shot_draft")?.output ||
    elevatorPitch ||
    "(initial task requirement)";

  // Group messages into per-turn buckets using router messages as boundaries.
  // Each bucket: { routerMsg, bodyMsgs[] }
  type TurnBucket = { routerMsg: any; bodyMsgs: any[] };
  const buckets: TurnBucket[] = [];
  let current: TurnBucket | null = null;
  for (const m of messages) {
    if (m.role === "aunu_agent" && m.action === "router") {
      if (current) buckets.push(current);
      current = { routerMsg: m, bodyMsgs: [] };
    } else if (current) {
      current.bodyMsgs.push(m);
    }
  }
  if (current) buckets.push(current);

  // Track evolving task requirement for display in "requirement given to router"
  let currentReq = initReq;
  const reqSnapshot: string[] = buckets.map((b) => {
    const snap = currentReq;
    const dataMsg = b.bodyMsgs.find((m) => m.action === "zero_shot_with_samples_reason");
    if (dataMsg?.output) {
      try {
        const text = dataMsg.output.trim().replace(/^```json\n?/, "").replace(/\n?```$/, "");
        const parsed = JSON.parse(text);
        if (parsed.final_task_requirement) currentReq = parsed.final_task_requirement;
        else currentReq = dataMsg.output;
      } catch { currentReq = dataMsg.output; }
    }
    return snap;
  });

  const actionColor: Record<string, string> = {
    data_interaction: "#1d4ed8",
    user_interaction: "#16a34a",
    reflection_synthesis: "#7c3aed",
  };
  const actionBg: Record<string, string> = {
    data_interaction: "#dbeafe",
    user_interaction: "#dcfce7",
    reflection_synthesis: "#ede9fe",
  };

  const labelStyle = (color: string): React.CSSProperties => ({
    fontSize: 11, fontWeight: 700, color, textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 5,
  });

  return (
    <div>
      {/* Elevator pitch (raw user input) */}
      {elevatorPitch && (
        <div style={{ marginBottom: 12, border: "1px solid #d1fae5", borderRadius: 10, overflow: "hidden" }}>
          <div style={{ background: "#ecfdf5", padding: "10px 14px", fontWeight: 700, fontSize: 14, borderBottom: "1px solid #d1fae5", display: "flex", alignItems: "center", gap: 8 }}>
            <span style={{ background: "#059669", color: "white", borderRadius: 4, padding: "2px 8px", fontSize: 12 }}>ELEVATOR PITCH</span>
            User's original request
          </div>
          <div style={{ padding: 14 }}>
            <TextBlock text={elevatorPitch} bg="#f0fdf4" border="#d1fae5" maxHeight={150} />
          </div>
        </div>
      )}

      {/* Initial zero-shot draft */}
      <div style={{ marginBottom: 20, border: "1px solid #e5e7eb", borderRadius: 10, overflow: "hidden" }}>
        <div style={{ background: "#f6f8fa", padding: "10px 14px", fontWeight: 700, fontSize: 14, borderBottom: "1px solid #e5e7eb", display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ background: "#374151", color: "white", borderRadius: 4, padding: "2px 8px", fontSize: 12 }}>INIT</span>
          Initial task requirement (zero-shot draft)
        </div>
        <div style={{ padding: 14 }}>
          <TextBlock text={initReq} bg="#fafafa" border="#e5e7eb" maxHeight={250} />
        </div>
      </div>

      {buckets.length === 0 && (
        <div style={{ color: "#666", fontStyle: "italic" }}>No router messages found.</div>
      )}

      {buckets.map((bucket, idx) => {
        const rh = routerHistory[idx];
        const action: string = rh?.action || "unknown";
        const routerOutput = bucket.routerMsg?.output || "";

        // Parse router JSON for reason / target_gap (fall back to router_history entry)
        let reason = rh?.reason || "";
        let targetGap = rh?.target_gap || "";
        try {
          const p = JSON.parse(routerOutput);
          if (p.reason) reason = p.reason;
          if (p.target_gap) targetGap = p.target_gap;
        } catch { /* use router_history values */ }

        const askMsg = bucket.bodyMsgs.find((m) => m.action === "ask_user");
        const mimicMsg = bucket.bodyMsgs.find((m) => m.role === "mimic_user" && m.action === "respond");
        const dataMsg = bucket.bodyMsgs.find((m) => m.action === "zero_shot_with_samples_reason");
        const guidelineMsg = bucket.bodyMsgs.find((m) => m.action === "guideline_update");
        const finishMsg = bucket.bodyMsgs.find((m) => m.action === "finish");

        return (
          <div key={idx} style={{ marginBottom: 20, border: "1px solid #e5e7eb", borderRadius: 10, overflow: "hidden" }}>
            {/* Turn header */}
            <div style={{ background: "#f6f8fa", padding: "10px 14px", fontWeight: 700, fontSize: 14, borderBottom: "1px solid #e5e7eb", display: "flex", alignItems: "center", gap: 8 }}>
              <span style={{ background: "#6b7280", color: "white", borderRadius: 4, padding: "2px 8px", fontSize: 12 }}>T{rh?.turn ?? idx}</span>
              <span style={{ background: actionBg[action] || "#f3f4f6", color: actionColor[action] || "#374151", borderRadius: 4, padding: "2px 8px", fontSize: 12, fontWeight: 700 }}>
                {action}
              </span>
            </div>

            <div style={{ padding: 14, display: "flex", flexDirection: "column", gap: 16 }}>

              {/* ① Task requirement given to router */}
              <div>
                <div style={labelStyle("#6b7280")}>① Task requirement given to router</div>
                <TextBlock text={reqSnapshot[idx]} bg="#fafafa" border="#e5e7eb" />
              </div>

              {/* ② Router decision */}
              <div>
                <div style={labelStyle(actionColor[action] || "#374151")}>② Router decision → {action}</div>
                <div style={{ background: actionBg[action] || "#f3f4f6", border: `1px solid ${actionColor[action] || "#ccc"}33`, borderRadius: 8, padding: 10 }}>
                  {reason && <div style={{ fontSize: 13, marginBottom: targetGap ? 6 : 0 }}><span style={{ fontWeight: 700 }}>Reason: </span>{reason}</div>}
                  {targetGap && <div style={{ fontSize: 13, color: "#555" }}><span style={{ fontWeight: 700 }}>Target gap: </span>{targetGap}</div>}
                </div>
              </div>

              {/* ③ User interaction */}
              {askMsg && (
                <div>
                  <div style={labelStyle("#1d4ed8")}>③ Message sent to user</div>
                  <TextBlock text={(() => {
                    try {
                      const cleaned = askMsg.output.trim().replace(/^```json\n?/, "").replace(/\n?```$/, "").replace(/\\n/g, "\n");
                      const p = JSON.parse(cleaned);
                      return p.question || askMsg.output;
                    } catch { return askMsg.output; }
                  })()} bg="#eff6ff" border="#bfdbfe" />
                </div>
              )}
              {mimicMsg && (
                <div>
                  <div style={labelStyle("#15803d")}>④ User's reply</div>
                  <TextBlock text={mimicMsg.output} bg="#f0fdf4" border="#bbf7d0" />
                  {mimicMsg.thought && (
                    <div style={{ marginTop: 6, fontSize: 12, color: "#6b7280", fontStyle: "italic", paddingLeft: 4 }}>
                      💭 <span style={{ fontWeight: 600 }}>Thought: </span>{mimicMsg.thought}
                    </div>
                  )}
                </div>
              )}

              {/* ③ Data interaction */}
              {dataMsg && (
                <div>
                  <div style={labelStyle("#1d4ed8")}>③ Data inspection — refined requirement</div>
                  <TextBlock text={(() => {
                    try {
                      const text = dataMsg.output.trim().replace(/^```json\n?/, "").replace(/\n?```$/, "");
                      const p = JSON.parse(text);
                      return p.final_task_requirement || dataMsg.output;
                    } catch { return dataMsg.output; }
                  })()} bg="#eff6ff" border="#bfdbfe" maxHeight={300} />
                </div>
              )}

              {/* Guideline update */}
              {guidelineMsg && (
                <div>
                  <div style={labelStyle("#92400e")}>⑤ Guideline update</div>
                  <TextBlock text={guidelineMsg.output} bg="#fffbeb" border="#fde68a" maxHeight={250} />
                </div>
              )}

              {/* Synthesis */}
              {finishMsg && (
                <div>
                  <div style={labelStyle("#7c3aed")}>③ Final synthesized requirement</div>
                  <TextBlock text={finishMsg.output} bg="#faf5ff" border="#ddd6fe" maxHeight={350} />
                </div>
              )}

            </div>
          </div>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------------------
// User Interaction Trace Tab
// ---------------------------------------------------------------------------

function UserInteractionTrace({ taskData }: { taskData: any }) {
  const messages: any[] = taskData?.messages || [];

  const zeroDraftMsg = messages.find((m: any) => m.action === "zero_shot_draft");
  const finishMsg = messages.find((m: any) => m.action === "finish");

  // Group into conversation turns: each turn = ask_user + mimic_user respond
  type ConvTurn = { askMsg: any; mimicMsg: any | null };
  const turns: ConvTurn[] = [];
  let pendingAsk: any = null;
  for (const m of messages) {
    if (m.action === "ask_user") {
      pendingAsk = m;
    } else if (m.role === "mimic_user" && m.action === "respond" && pendingAsk) {
      turns.push({ askMsg: pendingAsk, mimicMsg: m });
      pendingAsk = null;
    }
  }
  // ask_user with no reply yet (edge case)
  if (pendingAsk) turns.push({ askMsg: pendingAsk, mimicMsg: null });

  const labelStyle = (color: string): React.CSSProperties => ({
    fontSize: 11, fontWeight: 700, color, textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 5,
  });

  return (
    <div>
      {/* Zero-shot draft */}
      {zeroDraftMsg && (
        <div style={{ marginBottom: 20, border: "1px solid #e5e7eb", borderRadius: 10, overflow: "hidden" }}>
          <div style={{ background: "#f6f8fa", padding: "10px 14px", fontWeight: 700, fontSize: 14, borderBottom: "1px solid #e5e7eb", display: "flex", alignItems: "center", gap: 8 }}>
            <span style={{ background: "#374151", color: "white", borderRadius: 4, padding: "2px 8px", fontSize: 12 }}>INIT</span>
            Zero-shot draft
          </div>
          <div style={{ padding: 14 }}>
            <TextBlock text={zeroDraftMsg.output} bg="#fafafa" border="#e5e7eb" maxHeight={250} />
          </div>
        </div>
      )}

      {turns.length === 0 && (
        <div style={{ color: "#666", fontStyle: "italic" }}>No conversation turns found.</div>
      )}

      {turns.map(({ askMsg, mimicMsg }, idx) => (
        <div key={idx} style={{ marginBottom: 20, border: "1px solid #e5e7eb", borderRadius: 10, overflow: "hidden" }}>
          <div style={{ background: "#f6f8fa", padding: "10px 14px", fontWeight: 700, fontSize: 14, borderBottom: "1px solid #e5e7eb", display: "flex", alignItems: "center", gap: 8 }}>
            <span style={{ background: "#6b7280", color: "white", borderRadius: 4, padding: "2px 8px", fontSize: 12 }}>Turn {idx + 1}</span>
            <span style={{ background: "#dcfce7", color: "#16a34a", borderRadius: 4, padding: "2px 8px", fontSize: 12, fontWeight: 700 }}>user_interaction</span>
          </div>
          <div style={{ padding: 14, display: "flex", flexDirection: "column", gap: 14 }}>
            <div>
              <div style={labelStyle("#1d4ed8")}>① Message sent to user</div>
              <TextBlock text={askMsg.output} bg="#eff6ff" border="#bfdbfe" />
            </div>
            {mimicMsg ? (
              <div>
                <div style={labelStyle("#15803d")}>② User's reply</div>
                <TextBlock text={mimicMsg.output} bg="#f0fdf4" border="#bbf7d0" />
                {mimicMsg.thought && (
                  <div style={{ marginTop: 6, fontSize: 12, color: "#6b7280", fontStyle: "italic", paddingLeft: 4 }}>
                    💭 <span style={{ fontWeight: 600 }}>Thought: </span>{mimicMsg.thought}
                  </div>
                )}
              </div>
            ) : (
              <div style={{ color: "#999", fontStyle: "italic", fontSize: 13 }}>No user reply recorded.</div>
            )}
          </div>
        </div>
      ))}

      {/* Final requirement */}
      {finishMsg && (
        <div style={{ marginBottom: 20, border: "1px solid #ddd6fe", borderRadius: 10, overflow: "hidden" }}>
          <div style={{ background: "#faf5ff", padding: "10px 14px", fontWeight: 700, fontSize: 14, borderBottom: "1px solid #ddd6fe", display: "flex", alignItems: "center", gap: 8 }}>
            <span style={{ background: "#7c3aed", color: "white", borderRadius: 4, padding: "2px 8px", fontSize: 12 }}>FINISH</span>
            Final synthesized requirement
          </div>
          <div style={{ padding: 14 }}>
            <TextBlock text={finishMsg.output} bg="#faf5ff" border="#ddd6fe" maxHeight={350} />
          </div>
        </div>
      )}
    </div>
  );
}

function UserInteractionTraceTab({ baseUrl }: { baseUrl: string }) {
  const [dataset, setDataset] = React.useState(DEFAULT_DATASETS[0]);
  const [experimentId, setExperimentId] = React.useState(DEFAULT_EXPERIMENTS[0]);
  const [run, setRun] = React.useState<LoadedRun | null>(null);
  const [loading, setLoading] = React.useState(false);
  const [loadError, setLoadError] = React.useState<string | null>(null);
  const [personaId, setPersonaId] = React.useState("");
  const [taskId, setTaskId] = React.useState("");

  async function load() {
    setLoading(true);
    setLoadError(null);
    setRun(null);
    setPersonaId("");
    setTaskId("");
    try {
      const outputUrl = `${baseUrl}/${dataset}/user_interaction/${experimentId}/output.json`;
      const evalUrl = `${baseUrl}/${dataset}/user_interaction/${experimentId}/eval_results.json`;
      const [rawOutput, rawEval] = await Promise.all([
        fetchJson(outputUrl),
        fetchJson(evalUrl).catch(() => null),
      ]);
      const outputJson = normalizeOutputJson(rawOutput);
      const evalJson = normalizeOutputJson(rawEval);
      const loaded: LoadedRun = { dataset, strategy: "user_interaction", experimentId, outputJson, evalJson, error: null };
      setRun(loaded);
      const personas = getPersonaIds([loaded]);
      const p = personas[0] || "";
      setPersonaId(p);
      setTaskId(getTaskIds(loaded, p)[0] || "");
    } catch (err: any) {
      setLoadError(err?.message || "Failed to load user_interaction results.");
    } finally {
      setLoading(false);
    }
  }

  const personaIds = useMemo(() => (run ? getPersonaIds([run]) : []), [run]);
  const taskIds = useMemo(() => getTaskIds(run, personaId), [run, personaId]);

  const overallScores = useMemo(() => (run ? getRunAverageScores(run) : null), [run]);
  const subScores = useMemo(() => (run ? getRunAverageSubcategoryScores(run) : null), [run]);

  const taskData = run?.outputJson?.[personaId]?.[taskId] ?? null;
  const evalTask = run?.evalJson?.[personaId]?.[taskId] ?? null;
  const taskScores = evalTask?.scores || {};

  return (
    <div>
      <div style={panelStyle()}>
        <h2 style={sectionTitleStyle()}>User Interaction — dataset & experiment</h2>
        <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr auto", gap: 12, alignItems: "flex-end" }}>
          <div>
            <label style={smallLabelStyle()}>Dataset</label>
            <select style={inputStyle()} value={dataset} onChange={(e) => setDataset(e.target.value)}>
              {DEFAULT_DATASETS.map((d) => <option key={d} value={d}>{d}</option>)}
            </select>
          </div>
          <div>
            <label style={smallLabelStyle()}>Experiment</label>
            <select style={inputStyle()} value={experimentId} onChange={(e) => setExperimentId(e.target.value)}>
              {DEFAULT_EXPERIMENTS.map((x) => <option key={x} value={x}>{x}</option>)}
            </select>
          </div>
          <button style={buttonStyle()} onClick={load} disabled={loading}>
            {loading ? "Loading..." : "Load"}
          </button>
        </div>
        {loadError && <div style={{ color: "#b42318", marginTop: 10 }}>{loadError}</div>}
      </div>

      {run && (
        <>
          <div style={panelStyle()}>
            <h2 style={sectionTitleStyle()}>Overall scores</h2>
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead>
                <tr style={{ background: "#f6f8fa" }}>
                  <th style={{ textAlign: "left", padding: 10, borderBottom: "1px solid #ddd" }}>Metric</th>
                  <th style={{ textAlign: "left", padding: 10, borderBottom: "1px solid #ddd" }}>Overall</th>
                  <th style={{ textAlign: "left", padding: 10, borderBottom: "1px solid #ddd", borderLeft: "2px solid #c7d2fe", background: "#eef2ff", color: "#3730a3" }}>user_specified P</th>
                  <th style={{ textAlign: "left", padding: 10, borderBottom: "1px solid #ddd", background: "#eef2ff", color: "#3730a3" }}>user_specified R</th>
                  <th style={{ textAlign: "left", padding: 10, borderBottom: "1px solid #ddd", background: "#eef2ff", color: "#3730a3" }}>user_specified F1</th>
                  <th style={{ textAlign: "left", padding: 10, borderBottom: "1px solid #ddd", borderLeft: "2px solid #bbf7d0", background: "#f0fdf4", color: "#166534" }}>data_specified P</th>
                  <th style={{ textAlign: "left", padding: 10, borderBottom: "1px solid #ddd", background: "#f0fdf4", color: "#166534" }}>data_specified R</th>
                  <th style={{ textAlign: "left", padding: 10, borderBottom: "1px solid #ddd", background: "#f0fdf4", color: "#166534" }}>data_specified F1</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td style={{ padding: 10, borderBottom: "1px solid #eee", fontWeight: 700 }}>Avg (all tasks)</td>
                  <td style={{ padding: 10, borderBottom: "1px solid #eee" }}>
                    P: {formatScore(overallScores?.precision)} / R: {formatScore(overallScores?.recall)} / <strong>F1: {formatScore(overallScores?.f1)}</strong>
                  </td>
                  <td style={{ padding: 10, borderBottom: "1px solid #eee", borderLeft: "2px solid #c7d2fe", background: "#f5f7ff" }}>{formatScore(subScores?.user_specified.precision)}</td>
                  <td style={{ padding: 10, borderBottom: "1px solid #eee", background: "#f5f7ff" }}>{formatScore(subScores?.user_specified.recall)}</td>
                  <td style={{ padding: 10, borderBottom: "1px solid #eee", background: "#f5f7ff", fontWeight: 700 }}>{formatScore(subScores?.user_specified.f1)}</td>
                  <td style={{ padding: 10, borderBottom: "1px solid #eee", borderLeft: "2px solid #bbf7d0", background: "#f0fdf4" }}>{formatScore(subScores?.data_specified.precision)}</td>
                  <td style={{ padding: 10, borderBottom: "1px solid #eee", background: "#f0fdf4" }}>{formatScore(subScores?.data_specified.recall)}</td>
                  <td style={{ padding: 10, borderBottom: "1px solid #eee", background: "#f0fdf4", fontWeight: 700 }}>{formatScore(subScores?.data_specified.f1)}</td>
                </tr>
              </tbody>
            </table>
          </div>

          <div style={panelStyle()}>
            <h2 style={sectionTitleStyle()}>Select persona & task</h2>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
              <div>
                <label style={smallLabelStyle()}>persona_id</label>
                <select style={inputStyle()} value={personaId} onChange={(e) => { setPersonaId(e.target.value); setTaskId(getTaskIds(run, e.target.value)[0] || ""); }}>
                  <option value="">Select persona</option>
                  {personaIds.map((p) => <option key={p} value={p}>{p}</option>)}
                </select>
              </div>
              <div>
                <label style={smallLabelStyle()}>task_id</label>
                <select style={inputStyle()} value={taskId} onChange={(e) => setTaskId(e.target.value)}>
                  <option value="">Select task</option>
                  {taskIds.map((t) => <option key={t} value={t}>{t}</option>)}
                </select>
              </div>
            </div>
            {personaId && taskId && (
              <div style={{ marginTop: 14, display: "flex", flexDirection: "column", gap: 10 }}>
                <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
                  {["precision", "recall", "f1"].map((k) => (
                    taskScores[k] !== undefined && (
                      <div key={k} style={{ background: "#f6f8fa", border: "1px solid #e5e7eb", borderRadius: 8, padding: "8px 14px", fontSize: 14 }}>
                        <span style={{ fontWeight: 700, textTransform: "capitalize" }}>{k}</span>: {formatScore(taskScores[k])}
                      </div>
                    )
                  ))}
                </div>
                {evalTask?.subcategory_scores && (
                  <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
                    {(["user_specified", "data_specified"] as const).map((cat) => {
                      const s = evalTask.subcategory_scores[cat];
                      if (!s) return null;
                      const color = cat === "user_specified" ? { bg: "#eef2ff", border: "#c7d2fe", label: "#3730a3" } : { bg: "#f0fdf4", border: "#bbf7d0", label: "#166534" };
                      return (
                        <div key={cat} style={{ background: color.bg, border: `1px solid ${color.border}`, borderRadius: 8, padding: "8px 14px", fontSize: 13 }}>
                          <span style={{ fontWeight: 700, color: color.label }}>{cat}</span>
                          {" — "}P: {formatScore(s.precision)} / R: {formatScore(s.recall)} / <strong>F1: {formatScore(s.f1)}</strong>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            )}
          </div>

          {personaId && taskId ? (
            <div style={panelStyle()}>
              <h2 style={sectionTitleStyle()}>Conversation trace — persona {personaId}, task {taskId}</h2>
              {taskData ? (
                <UserInteractionTrace taskData={taskData} />
              ) : (
                <div style={{ color: "#666" }}>No data for this persona/task.</div>
              )}
            </div>
          ) : (
            <div style={{ ...panelStyle(), color: "#666" }}>Select a persona and task to view the conversation trace.</div>
          )}
        </>
      )}
    </div>
  );
}

function HybridTraceTab({ baseUrl }: { baseUrl: string }) {
  const [dataset, setDataset] = React.useState(DEFAULT_DATASETS[0]);
  const [experimentId, setExperimentId] = React.useState(DEFAULT_EXPERIMENTS[0]);
  const [run, setRun] = React.useState<LoadedRun | null>(null);
  const [loading, setLoading] = React.useState(false);
  const [loadError, setLoadError] = React.useState<string | null>(null);
  const [personaId, setPersonaId] = React.useState("");
  const [taskId, setTaskId] = React.useState("");

  async function load() {
    setLoading(true);
    setLoadError(null);
    setRun(null);
    setPersonaId("");
    setTaskId("");
    try {
      const outputUrl = `${baseUrl}/${dataset}/hybrid/${experimentId}/output.json`;
      const evalUrl = `${baseUrl}/${dataset}/hybrid/${experimentId}/eval_results.json`;
      const [rawOutput, rawEval] = await Promise.all([
        fetchJson(outputUrl),
        fetchJson(evalUrl).catch(() => null),
      ]);
      const outputJson = normalizeOutputJson(rawOutput);
      const evalJson = normalizeOutputJson(rawEval);
      const loaded: LoadedRun = { dataset, strategy: "hybrid", experimentId, outputJson, evalJson, error: null };
      setRun(loaded);
      const personas = getPersonaIds([loaded]);
      const p = personas[0] || "";
      setPersonaId(p);
      setTaskId(getTaskIds(loaded, p)[0] || "");
    } catch (err: any) {
      setLoadError(err?.message || "Failed to load hybrid results.");
    } finally {
      setLoading(false);
    }
  }

  const personaIds = useMemo(() => (run ? getPersonaIds([run]) : []), [run]);
  const taskIds = useMemo(() => getTaskIds(run, personaId), [run, personaId]);

  const overallScores = useMemo(() => (run ? getRunAverageScores(run) : null), [run]);
  const subScores = useMemo(() => (run ? getRunAverageSubcategoryScores(run) : null), [run]);

  const taskData = run?.outputJson?.[personaId]?.[taskId] ?? null;
  const evalTask = run?.evalJson?.[personaId]?.[taskId] ?? null;
  const taskScores = evalTask?.scores || {};

  return (
    <div>
      {/* Selectors */}
      <div style={panelStyle()}>
        <h2 style={sectionTitleStyle()}>Hybrid — dataset & experiment</h2>
        <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr auto", gap: 12, alignItems: "flex-end" }}>
          <div>
            <label style={smallLabelStyle()}>Dataset</label>
            <select style={inputStyle()} value={dataset} onChange={(e) => setDataset(e.target.value)}>
              {DEFAULT_DATASETS.map((d) => <option key={d} value={d}>{d}</option>)}
            </select>
          </div>
          <div>
            <label style={smallLabelStyle()}>Experiment</label>
            <select style={inputStyle()} value={experimentId} onChange={(e) => setExperimentId(e.target.value)}>
              {DEFAULT_EXPERIMENTS.map((x) => <option key={x} value={x}>{x}</option>)}
            </select>
          </div>
          <button style={buttonStyle()} onClick={load} disabled={loading}>
            {loading ? "Loading..." : "Load"}
          </button>
        </div>
        {loadError && <div style={{ color: "#b42318", marginTop: 10 }}>{loadError}</div>}
      </div>

      {run && (
        <>
          {/* Overall scores */}
          <div style={panelStyle()}>
            <h2 style={sectionTitleStyle()}>Overall scores</h2>
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead>
                <tr style={{ background: "#f6f8fa" }}>
                  <th style={{ textAlign: "left", padding: 10, borderBottom: "1px solid #ddd" }}>Metric</th>
                  <th style={{ textAlign: "left", padding: 10, borderBottom: "1px solid #ddd" }}>Overall</th>
                  <th style={{ textAlign: "left", padding: 10, borderBottom: "1px solid #ddd", borderLeft: "2px solid #c7d2fe", background: "#eef2ff", color: "#3730a3" }}>user_specified P</th>
                  <th style={{ textAlign: "left", padding: 10, borderBottom: "1px solid #ddd", background: "#eef2ff", color: "#3730a3" }}>user_specified R</th>
                  <th style={{ textAlign: "left", padding: 10, borderBottom: "1px solid #ddd", background: "#eef2ff", color: "#3730a3" }}>user_specified F1</th>
                  <th style={{ textAlign: "left", padding: 10, borderBottom: "1px solid #ddd", borderLeft: "2px solid #bbf7d0", background: "#f0fdf4", color: "#166534" }}>data_specified P</th>
                  <th style={{ textAlign: "left", padding: 10, borderBottom: "1px solid #ddd", background: "#f0fdf4", color: "#166534" }}>data_specified R</th>
                  <th style={{ textAlign: "left", padding: 10, borderBottom: "1px solid #ddd", background: "#f0fdf4", color: "#166534" }}>data_specified F1</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td style={{ padding: 10, borderBottom: "1px solid #eee", fontWeight: 700 }}>Avg (all tasks)</td>
                  <td style={{ padding: 10, borderBottom: "1px solid #eee" }}>
                    P: {formatScore(overallScores?.precision)} / R: {formatScore(overallScores?.recall)} / <strong>F1: {formatScore(overallScores?.f1)}</strong>
                  </td>
                  <td style={{ padding: 10, borderBottom: "1px solid #eee", borderLeft: "2px solid #c7d2fe", background: "#f5f7ff" }}>{formatScore(subScores?.user_specified.precision)}</td>
                  <td style={{ padding: 10, borderBottom: "1px solid #eee", background: "#f5f7ff" }}>{formatScore(subScores?.user_specified.recall)}</td>
                  <td style={{ padding: 10, borderBottom: "1px solid #eee", background: "#f5f7ff", fontWeight: 700 }}>{formatScore(subScores?.user_specified.f1)}</td>
                  <td style={{ padding: 10, borderBottom: "1px solid #eee", borderLeft: "2px solid #bbf7d0", background: "#f0fdf4" }}>{formatScore(subScores?.data_specified.precision)}</td>
                  <td style={{ padding: 10, borderBottom: "1px solid #eee", background: "#f0fdf4" }}>{formatScore(subScores?.data_specified.recall)}</td>
                  <td style={{ padding: 10, borderBottom: "1px solid #eee", background: "#f0fdf4", fontWeight: 700 }}>{formatScore(subScores?.data_specified.f1)}</td>
                </tr>
              </tbody>
            </table>
          </div>

          {/* Persona + task selectors */}
          <div style={panelStyle()}>
            <h2 style={sectionTitleStyle()}>Select persona & task</h2>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
              <div>
                <label style={smallLabelStyle()}>persona_id</label>
                <select style={inputStyle()} value={personaId} onChange={(e) => { setPersonaId(e.target.value); setTaskId(getTaskIds(run, e.target.value)[0] || ""); }}>
                  <option value="">Select persona</option>
                  {personaIds.map((p) => <option key={p} value={p}>{p}</option>)}
                </select>
              </div>
              <div>
                <label style={smallLabelStyle()}>task_id</label>
                <select style={inputStyle()} value={taskId} onChange={(e) => setTaskId(e.target.value)}>
                  <option value="">Select task</option>
                  {taskIds.map((t) => <option key={t} value={t}>{t}</option>)}
                </select>
              </div>
            </div>

            {/* Per-task scores */}
            {personaId && taskId && (
              <div style={{ marginTop: 14, display: "flex", flexDirection: "column", gap: 10 }}>
                <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
                  {["precision", "recall", "f1"].map((k) => (
                    taskScores[k] !== undefined && (
                      <div key={k} style={{ background: "#f6f8fa", border: "1px solid #e5e7eb", borderRadius: 8, padding: "8px 14px", fontSize: 14 }}>
                        <span style={{ fontWeight: 700, textTransform: "capitalize" }}>{k}</span>: {formatScore(taskScores[k])}
                      </div>
                    )
                  ))}
                </div>
                {evalTask?.subcategory_scores && (
                  <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
                    {(["user_specified", "data_specified"] as const).map((cat) => {
                      const s = evalTask.subcategory_scores[cat];
                      if (!s) return null;
                      const color = cat === "user_specified" ? { bg: "#eef2ff", border: "#c7d2fe", label: "#3730a3" } : { bg: "#f0fdf4", border: "#bbf7d0", label: "#166534" };
                      return (
                        <div key={cat} style={{ background: color.bg, border: `1px solid ${color.border}`, borderRadius: 8, padding: "8px 14px", fontSize: 13 }}>
                          <span style={{ fontWeight: 700, color: color.label }}>{cat}</span>
                          {" — "}P: {formatScore(s.precision)} / R: {formatScore(s.recall)} / <strong>F1: {formatScore(s.f1)}</strong>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Iteration trace */}
          {personaId && taskId ? (
            <div style={panelStyle()}>
              <h2 style={sectionTitleStyle()}>Iteration trace — persona {personaId}, task {taskId}</h2>
              {taskData ? (
                <HybridIterationTrace taskData={taskData} />
              ) : (
                <div style={{ color: "#666" }}>No data for this persona/task.</div>
              )}
            </div>
          ) : (
            <div style={{ ...panelStyle(), color: "#666" }}>Select a persona and task to view the iteration trace.</div>
          )}
        </>
      )}
    </div>
  );
}

export default function App() {
  const [activeTab, setActiveTab] = useState<"main" | "format-reflect" | "hybrid-trace" | "user-interaction-trace">("main");
  const [baseUrl, setBaseUrl] = useState("/agent/results");
  const [dataset, setDataset] = useState(DEFAULT_DATASETS[0]);
  const [selectedStrategies, setSelectedStrategies] = useState<string[]>([
    "zero_shot",
    "zero_shot_with_samples_reason",
    "user_interaction",
    "hybrid",
  ]);
  const [strategyExperimentIds, setStrategyExperimentIds] = useState<Record<string, string>>({
    zero_shot: DEFAULT_EXPERIMENTS[0],
    zero_shot_with_samples_reason: DEFAULT_EXPERIMENTS[0],
    user_interaction: DEFAULT_EXPERIMENTS[0],
    hybrid: DEFAULT_EXPERIMENTS[0],
  });
  const [runs, setRuns] = useState<LoadedRun[]>([]);
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);

  const personaIds = useMemo(() => getPersonaIds(runs), [runs]);
  const [personaId, setPersonaId] = useState("");
  const taskIds = useMemo(() => {
    const firstLoaded = runs.find((r) => r.outputJson || r.evalJson) || null;
    return getTaskIds(firstLoaded, personaId);
  }, [runs, personaId]);
  const [taskId, setTaskId] = useState("");

  async function loadRuns() {
    setLoading(true);
    setLoadError(null);

    try {
      const nextRuns: LoadedRun[] = [];

      for (const strategy of selectedStrategies) {
        const experimentId = strategyExperimentIds[strategy] || DEFAULT_EXPERIMENTS[0];
        const outputUrl = `${baseUrl}/${dataset}/${strategy}/${experimentId}/output.json`;
        const evalUrl = `${baseUrl}/${dataset}/${strategy}/${experimentId}/eval_results.json`;

        try {
          const [rawOutput, rawEval] = await Promise.all([
            fetchJson(outputUrl),
            fetchJson(evalUrl).catch(() => null),
          ]);
          const outputJson = normalizeOutputJson(rawOutput);
          const evalJson = normalizeOutputJson(rawEval);

          nextRuns.push({
            dataset,
            strategy,
            experimentId,
            outputJson,
            evalJson,
            error: null,
          });
        } catch (err: any) {
          nextRuns.push({
            dataset,
            strategy,
            experimentId,
            outputJson: null,
            evalJson: null,
            error: err?.message || `Failed to load ${strategy}`,
          });
        }
      }

      setRuns(nextRuns);

      const nextPersona = getPersonaIds(nextRuns)[0] || "";
      setPersonaId(nextPersona);

      const firstLoaded = nextRuns.find((r) => r.outputJson || r.evalJson) || null;
      const nextTask = getTaskIds(firstLoaded, nextPersona)[0] || "";
      setTaskId(nextTask);
    } catch (err: any) {
      setLoadError(err?.message || "Unexpected error while loading runs.");
    } finally {
      setLoading(false);
    }
  }

  function toggleStrategy(strategy: string) {
    setSelectedStrategies((prev) =>
      prev.includes(strategy) ? prev.filter((s) => s !== strategy) : [...prev, strategy]
    );
  }

  return (
    <div
      style={{
        padding: 20,
        fontFamily: "Arial, sans-serif",
        background: "#f6f8fa",
        minHeight: "100vh",
        color: "#24292f",
      }}
    >
      <div style={{ maxWidth: 1700, margin: "0 auto" }}>
        <h1 style={{ marginTop: 0 }}>AUNU Results Visualization UI</h1>

        <div style={{ display: "flex", gap: 4, marginBottom: 20, borderBottom: "2px solid #e5e7eb" }}>
          {(["main", "format-reflect", "hybrid-trace", "user-interaction-trace"] as const).map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              style={{
                padding: "10px 20px",
                border: "none",
                borderBottom: activeTab === tab ? "2px solid #2563eb" : "2px solid transparent",
                background: "none",
                cursor: "pointer",
                fontWeight: activeTab === tab ? 700 : 400,
                color: activeTab === tab ? "#2563eb" : "#57606a",
                fontSize: 15,
                marginBottom: -2,
              }}
            >
              {tab === "main" ? "Main" : tab === "format-reflect" ? "Format Reflect & Rewrite" : tab === "hybrid-trace" ? "Hybrid Trace" : "User Interaction Trace"}
            </button>
          ))}
        </div>

        {activeTab === "main" && <>
        <p style={{ color: "#57606a", lineHeight: 1.6 }}>
          This UI follows your results hierarchy:
          <br />
          <code>dataset / strategy / Experiment{"{id}"} / output.json + eval_results.json</code>
          <br />
          It links trajectory and evaluation by shared <code>persona_id</code> and <code>task_id</code>.
        </p>

        <div style={panelStyle()}>
          <h2 style={sectionTitleStyle()}>Load configuration</h2>

          <div
            style={{
              display: "grid",
              gridTemplateColumns: "2fr 1fr",
              gap: 12,
              marginBottom: 12,
            }}
          >
            <div>
              <label style={smallLabelStyle()}>Base URL serving results</label>
              <input
                style={inputStyle()}
                value={baseUrl}
                onChange={(e) => setBaseUrl(e.target.value)}
                placeholder="/results"
              />
            </div>

            <div>
              <label style={smallLabelStyle()}>Dataset</label>
              <select
                style={inputStyle()}
                value={dataset}
                onChange={(e) => setDataset(e.target.value)}
              >
                {DEFAULT_DATASETS.map((d) => (
                  <option key={d} value={d}>
                    {d}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div style={{ marginBottom: 12 }}>
            <div style={smallLabelStyle()}>Experiment per strategy</div>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(6, minmax(0, 1fr))",
                gap: 12,
              }}
            >
              {DEFAULT_STRATEGIES.map((strategy) => (
                <div key={strategy}>
                  <label style={smallLabelStyle()}>{strategy}</label>
                  <select
                    style={inputStyle()}
                    value={strategyExperimentIds[strategy] || DEFAULT_EXPERIMENTS[0]}
                    onChange={(e) =>
                      setStrategyExperimentIds((prev) => ({
                        ...prev,
                        [strategy]: e.target.value,
                      }))
                    }
                  >
                    {DEFAULT_EXPERIMENTS.map((x) => (
                      <option key={x} value={x}>
                        {x}
                      </option>
                    ))}
                  </select>
                </div>
              ))}
            </div>
          </div>

          <div style={{ marginBottom: 12 }}>
            <div style={smallLabelStyle()}>Strategies to compare</div>
            <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
              {DEFAULT_STRATEGIES.map((strategy) => (
                <label key={strategy} style={{ display: "flex", alignItems: "center", gap: 6 }}>
                  <input
                    type="checkbox"
                    checked={selectedStrategies.includes(strategy)}
                    onChange={() => toggleStrategy(strategy)}
                  />
                  {strategy}
                </label>
              ))}
            </div>
          </div>

          <div style={{ display: "flex", gap: 10 }}>
            <button
              style={buttonStyle()}
              onClick={loadRuns}
              disabled={loading || selectedStrategies.length === 0}
            >
              {loading ? "Loading..." : "Load experiment"}
            </button>
            <button
              style={secondaryButtonStyle()}
              onClick={() => {
                setRuns([]);
                setPersonaId("");
                setTaskId("");
                setLoadError(null);
              }}
            >
              Clear
            </button>
          </div>

          {loadError ? <div style={{ color: "#b42318", marginTop: 12 }}>{loadError}</div> : null}
        </div>

        <div style={panelStyle()}>
          <h2 style={sectionTitleStyle()}>Quick score comparison</h2>
          {runs.length === 0 ? (
            <div style={{ color: "#666" }}>Load experiments first.</div>
          ) : (
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead>
                <tr style={{ background: "#f6f8fa" }}>
                  <th style={{ textAlign: "left", padding: 10, borderBottom: "1px solid #ddd" }}>Strategy</th>
                  <th style={{ textAlign: "left", padding: 10, borderBottom: "1px solid #ddd" }}>Experiment</th>
                  <th style={{ textAlign: "left", padding: 10, borderBottom: "1px solid #ddd" }}>Avg Precision</th>
                  <th style={{ textAlign: "left", padding: 10, borderBottom: "1px solid #ddd" }}>Avg Recall</th>
                  <th style={{ textAlign: "left", padding: 10, borderBottom: "1px solid #ddd" }}>Avg F1</th>
                </tr>
              </thead>
              <tbody>
                {runs.map((run) => {
                  const avgScores = getRunAverageScores(run);
                  return (
                    <tr key={`${run.strategy}-${run.experimentId}`}>
                      <td style={{ padding: 10, borderBottom: "1px solid #eee" }}>{run.strategy}</td>
                      <td style={{ padding: 10, borderBottom: "1px solid #eee" }}>{run.experimentId}</td>
                      <td style={{ padding: 10, borderBottom: "1px solid #eee" }}>{formatScore(avgScores.precision)}</td>
                      <td style={{ padding: 10, borderBottom: "1px solid #eee" }}>{formatScore(avgScores.recall)}</td>
                      <td style={{ padding: 10, borderBottom: "1px solid #eee" }}>{formatScore(avgScores.f1)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </div>

        <div style={panelStyle()}>
          <h2 style={sectionTitleStyle()}>Subcategory score comparison</h2>
          {runs.length === 0 ? (
            <div style={{ color: "#666" }}>Load experiments first.</div>
          ) : (
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead>
                <tr style={{ background: "#f6f8fa" }}>
                  <th style={{ textAlign: "left", padding: 10, borderBottom: "1px solid #ddd" }} rowSpan={2}>Strategy</th>
                  <th style={{ textAlign: "left", padding: 10, borderBottom: "1px solid #ddd" }} rowSpan={2}>Experiment</th>
                  <th style={{ textAlign: "center", padding: 10, borderBottom: "1px solid #ddd", borderLeft: "2px solid #c7d2fe", background: "#eef2ff", color: "#3730a3" }} colSpan={5}>user_specified</th>
                  <th style={{ textAlign: "center", padding: 10, borderBottom: "1px solid #ddd", borderLeft: "2px solid #bbf7d0", background: "#f0fdf4", color: "#166534" }} colSpan={5}>data_specified</th>
                </tr>
                <tr style={{ background: "#f6f8fa" }}>
                  {(["Avg Pred", "Pred Ratio", "Precision", "Recall", "F1"] as const).map((m) => (
                    <th key={`us-${m}`} style={{ textAlign: "left", padding: "6px 10px", borderBottom: "1px solid #ddd", borderLeft: m === "Avg Pred" ? "2px solid #c7d2fe" : undefined, fontSize: 12, color: "#555" }}>{m}</th>
                  ))}
                  {(["Avg Pred", "Pred Ratio", "Precision", "Recall", "F1"] as const).map((m) => (
                    <th key={`ds-${m}`} style={{ textAlign: "left", padding: "6px 10px", borderBottom: "1px solid #ddd", borderLeft: m === "Avg Pred" ? "2px solid #bbf7d0" : undefined, fontSize: 12, color: "#555" }}>{m}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {runs.map((run) => {
                  const sub = getRunAverageSubcategoryScores(run);
                  const comp = getRunPredComposition(run);
                  const hasData = SUBCATEGORIES.some((cat) =>
                    METRIC_KEYS.some((k) => sub[cat][k] !== null)
                  );
                  const fmtRatio = (v: number | null) =>
                    v === null ? "—" : `${(v * 100).toFixed(1)}%`;
                  const fmtAvg = (v: number | null) =>
                    v === null ? "—" : v.toFixed(1);
                  return (
                    <tr key={`${run.strategy}-${run.experimentId}`}>
                      <td style={{ padding: 10, borderBottom: "1px solid #eee" }}>{run.strategy}</td>
                      <td style={{ padding: 10, borderBottom: "1px solid #eee" }}>{run.experimentId}</td>
                      {hasData ? (
                        <>
                          <td style={{ padding: 10, borderBottom: "1px solid #eee", borderLeft: "2px solid #c7d2fe", background: "#f5f7ff", fontSize: 13 }}>{fmtAvg(comp.user_specified.avg)}</td>
                          <td style={{ padding: 10, borderBottom: "1px solid #eee", background: "#f5f7ff", fontSize: 13 }}>{fmtRatio(comp.user_specified.ratio)}</td>
                          <td style={{ padding: 10, borderBottom: "1px solid #eee", background: "#f5f7ff" }}>{formatScore(sub.user_specified.precision)}</td>
                          <td style={{ padding: 10, borderBottom: "1px solid #eee", background: "#f5f7ff" }}>{formatScore(sub.user_specified.recall)}</td>
                          <td style={{ padding: 10, borderBottom: "1px solid #eee", background: "#f5f7ff", fontWeight: 600 }}>{formatScore(sub.user_specified.f1)}</td>
                          <td style={{ padding: 10, borderBottom: "1px solid #eee", borderLeft: "2px solid #bbf7d0", background: "#f0fdf4", fontSize: 13 }}>{fmtAvg(comp.data_specified.avg)}</td>
                          <td style={{ padding: 10, borderBottom: "1px solid #eee", background: "#f0fdf4", fontSize: 13 }}>{fmtRatio(comp.data_specified.ratio)}</td>
                          <td style={{ padding: 10, borderBottom: "1px solid #eee", background: "#f0fdf4" }}>{formatScore(sub.data_specified.precision)}</td>
                          <td style={{ padding: 10, borderBottom: "1px solid #eee", background: "#f0fdf4" }}>{formatScore(sub.data_specified.recall)}</td>
                          <td style={{ padding: 10, borderBottom: "1px solid #eee", background: "#f0fdf4", fontWeight: 600 }}>{formatScore(sub.data_specified.f1)}</td>
                        </>
                      ) : (
                        <td colSpan={10} style={{ padding: 10, borderBottom: "1px solid #eee", color: "#999", fontStyle: "italic" }}>no subcategory scores</td>
                      )}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </div>


        <div style={panelStyle()}>
          <h2 style={sectionTitleStyle()}>Global selection</h2>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
            <div>
              <label style={smallLabelStyle()}>persona_id</label>
              <select
                style={inputStyle()}
                value={personaId}
                onChange={(e) => {
                  const nextPersona = e.target.value;
                  setPersonaId(nextPersona);
                  const firstLoaded = runs.find((r) => r.outputJson || r.evalJson) || null;
                  const nextTasks = getTaskIds(firstLoaded, nextPersona);
                  setTaskId(nextTasks[0] || "");
                }}
              >
                <option value="">Select persona</option>
                {personaIds.map((p) => (
                  <option key={p} value={p}>
                    {p}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label style={smallLabelStyle()}>task_id</label>
              <select
                style={inputStyle()}
                value={taskId}
                onChange={(e) => setTaskId(e.target.value)}
              >
                <option value="">Select task</option>
                {taskIds.map((t) => (
                  <option key={t} value={t}>
                    {t}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </div>

        <SelectedPairMetricsPanel runs={runs} personaId={personaId} taskId={taskId} />

        <GroundTruthMatrix runs={runs} personaId={personaId} taskId={taskId} />

        {runs.map((run) => (
          <StrategyRunPanel
            key={`${run.dataset}-${run.strategy}-${run.experimentId}`}
            run={run}
            personaId={personaId}
            taskId={taskId}
          />
        ))}
        </>}

        {activeTab === "format-reflect" && (
          <FormatReflectTab runs={runs} baseUrl={baseUrl} />
        )}

        {activeTab === "hybrid-trace" && (
          <HybridTraceTab baseUrl={baseUrl} />
        )}

        {activeTab === "user-interaction-trace" && (
          <UserInteractionTraceTab baseUrl={baseUrl} />
        )}
      </div>
    </div>
  );
}