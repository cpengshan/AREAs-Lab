import React, { useMemo, useState } from "react";

type LoadedRun = {
  dataset: string;
  strategy: string;
  experimentId: string;
  outputJson: any | null;
  evalJson: any | null;
  error: string | null;
};

const DEFAULT_DATASETS = ["alexfabbri_multi_news", "ccdv_arxiv-summarization", "ccdv_govreport-summarization", "ccdv_patent-classification", "ccdv_pubmed-summarization"];
const DEFAULT_STRATEGIES = ["zero_shot", "persona", "user", "data", "mix"];
const DEFAULT_EXPERIMENTS = Array.from({ length: 30 }, (_, i) => `Experiment${i + 1}`);
const METRIC_KEYS = [
  "completeness",
  "alignment",
  "faithfulness",
  "constraint_preservation",
] as const;

type MetricKey = (typeof METRIC_KEYS)[number];

async function fetchJson(url: string) {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to load ${url} (${res.status})`);
  }
  return res.json();
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
    completeness: [],
    alignment: [],
    faithfulness: [],
    constraint_preservation: [],
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
    completeness: avg(values.completeness),
    alignment: avg(values.alignment),
    faithfulness: avg(values.faithfulness),
    constraint_preservation: avg(values.constraint_preservation),
  };
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
                Completeness
              </th>
              <th style={{ textAlign: "left", padding: 10, borderBottom: "1px solid #ddd" }}>
                Alignment
              </th>
              <th style={{ textAlign: "left", padding: 10, borderBottom: "1px solid #ddd" }}>
                Faithfulness
              </th>
              <th style={{ textAlign: "left", padding: 10, borderBottom: "1px solid #ddd" }}>
                Constraint
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
                    {formatScore(scores.completeness)}
                  </td>
                  <td style={{ padding: 10, borderBottom: "1px solid #eee" }}>
                    {formatScore(scores.alignment)}
                  </td>
                  <td style={{ padding: 10, borderBottom: "1px solid #eee" }}>
                    {formatScore(scores.faithfulness)}
                  </td>
                  <td style={{ padding: 10, borderBottom: "1px solid #eee" }}>
                    {formatScore(scores.constraint_preservation)}
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

export default function App() {
  const [baseUrl, setBaseUrl] = useState("/results");
  const [dataset, setDataset] = useState(DEFAULT_DATASETS[0]);
  const [selectedStrategies, setSelectedStrategies] = useState<string[]>([
    "zero_shot",
    "persona",
    "user",
    "data",
    "mix",
  ]);
  const [strategyExperimentIds, setStrategyExperimentIds] = useState<Record<string, string>>({
    zero_shot: DEFAULT_EXPERIMENTS[0],
    persona: DEFAULT_EXPERIMENTS[0],
    user: DEFAULT_EXPERIMENTS[0],
    data: DEFAULT_EXPERIMENTS[0],
    mix: DEFAULT_EXPERIMENTS[0],
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
          const [outputJson, evalJson] = await Promise.all([
            fetchJson(outputUrl),
            fetchJson(evalUrl),
          ]);

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
                gridTemplateColumns: "repeat(5, minmax(0, 1fr))",
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
                  <th style={{ textAlign: "left", padding: 10, borderBottom: "1px solid #ddd" }}>Avg Completeness</th>
                  <th style={{ textAlign: "left", padding: 10, borderBottom: "1px solid #ddd" }}>Avg Alignment</th>
                  <th style={{ textAlign: "left", padding: 10, borderBottom: "1px solid #ddd" }}>Avg Faithfulness</th>
                  <th style={{ textAlign: "left", padding: 10, borderBottom: "1px solid #ddd" }}>Avg Constraint</th>
                </tr>
              </thead>
              <tbody>
                {runs.map((run) => {
                  const avgScores = getRunAverageScores(run);
                  return (
                    <tr key={`${run.strategy}-${run.experimentId}`}>
                      <td style={{ padding: 10, borderBottom: "1px solid #eee" }}>{run.strategy}</td>
                      <td style={{ padding: 10, borderBottom: "1px solid #eee" }}>{run.experimentId}</td>
                      <td style={{ padding: 10, borderBottom: "1px solid #eee" }}>{formatScore(avgScores.completeness)}</td>
                      <td style={{ padding: 10, borderBottom: "1px solid #eee" }}>{formatScore(avgScores.alignment)}</td>
                      <td style={{ padding: 10, borderBottom: "1px solid #eee" }}>{formatScore(avgScores.faithfulness)}</td>
                      <td style={{ padding: 10, borderBottom: "1px solid #eee" }}>
                        {formatScore(avgScores.constraint_preservation)}
                      </td>
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
      </div>
    </div>
  );
}