import { useCallback, useEffect, useState, type CSSProperties } from "react";

import { api, type EvaluationRunOut, type MetricsOut } from "../api/client";

const thStyle: CSSProperties = {
  textAlign: "left",
  borderBottom: "1px solid #ddd",
  padding: "0.5rem",
};
const tdStyle: CSSProperties = { borderBottom: "1px solid #eee", padding: "0.5rem" };
const sectionStyle: CSSProperties = { marginBottom: "2rem" };
const hintStyle: CSSProperties = { color: "#666", fontSize: "0.85rem" };

function formatMs(value: number): string {
  return `${value.toFixed(0)} ms`;
}

export function AdminPage() {
  const [metrics, setMetrics] = useState<MetricsOut | null>(null);
  const [evaluations, setEvaluations] = useState<EvaluationRunOut[]>([]);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const [metricsData, evaluationsData] = await Promise.all([
        api.getAdminMetrics(),
        api.getAdminEvaluations(),
      ]);
      setMetrics(metricsData);
      setEvaluations(evaluationsData);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load admin data");
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  return (
    <div>
      <h2>Admin</h2>
      {error && <p style={{ color: "crimson" }}>{error}</p>}

      <section style={sectionStyle}>
        <h3>Request latency</h3>
        <p style={hintStyle}>
          p50/p95 total latency per endpoint, computed from real logged requests (see{" "}
          <code>RequestMetric</code>).
        </p>
        <table style={{ width: "100%", borderCollapse: "collapse" }}>
          <thead>
            <tr>
              <th style={thStyle}>Endpoint</th>
              <th style={thStyle}>Count</th>
              <th style={thStyle}>p50</th>
              <th style={thStyle}>p95</th>
            </tr>
          </thead>
          <tbody>
            {metrics?.endpoints.map((e) => (
              <tr key={e.endpoint}>
                <td style={tdStyle}>{e.endpoint}</td>
                <td style={tdStyle}>{e.count}</td>
                <td style={tdStyle}>{formatMs(e.p50_total_ms)}</td>
                <td style={tdStyle}>{formatMs(e.p95_total_ms)}</td>
              </tr>
            ))}
            {metrics && metrics.endpoints.length === 0 && (
              <tr>
                <td style={tdStyle} colSpan={4}>
                  No requests logged yet.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </section>

      <section style={sectionStyle}>
        <h3>Recent requests</h3>
        <table style={{ width: "100%", borderCollapse: "collapse" }}>
          <thead>
            <tr>
              <th style={thStyle}>Endpoint</th>
              <th style={thStyle}>Total</th>
              <th style={thStyle}>Stages</th>
              <th style={thStyle}>When</th>
            </tr>
          </thead>
          <tbody>
            {metrics?.recent.map((r, i) => (
              <tr key={i}>
                <td style={tdStyle}>{r.endpoint}</td>
                <td style={tdStyle}>{formatMs(r.total_ms)}</td>
                <td style={tdStyle}>
                  {Object.entries(r.stage_latencies_ms)
                    .map(([stage, ms]) => `${stage}: ${formatMs(ms)}`)
                    .join(", ")}
                </td>
                <td style={tdStyle}>{new Date(r.created_at).toLocaleString()}</td>
              </tr>
            ))}
            {metrics && metrics.recent.length === 0 && (
              <tr>
                <td style={tdStyle} colSpan={4}>
                  No requests logged yet.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </section>

      <section>
        <h3>Evaluation runs</h3>
        <p style={hintStyle}>
          Results from <code>eval/runners/*.py</code> — see <code>docs/EVALUATION.md</code> for the
          full write-up.
        </p>
        <table style={{ width: "100%", borderCollapse: "collapse" }}>
          <thead>
            <tr>
              <th style={thStyle}>Name</th>
              <th style={thStyle}>Status</th>
              <th style={thStyle}>Questions</th>
              <th style={thStyle}>Mean metrics</th>
            </tr>
          </thead>
          <tbody>
            {evaluations.map((run) => (
              <tr key={run.id}>
                <td style={tdStyle}>{run.name}</td>
                <td style={tdStyle}>{run.status}</td>
                <td style={tdStyle}>{run.result_count}</td>
                <td style={tdStyle}>
                  {Object.entries(run.mean_metrics)
                    .map(([key, value]) => `${key}: ${value.toFixed(3)}`)
                    .join(", ")}
                </td>
              </tr>
            ))}
            {evaluations.length === 0 && (
              <tr>
                <td style={tdStyle} colSpan={4}>
                  No evaluation runs recorded yet.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </section>
    </div>
  );
}
