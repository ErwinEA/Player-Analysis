import type { PlayerMetrics } from "@/types/playerMetrics";
import styles from "./PlayerMetricsPanel.module.css";

type PlayerMetricsPanelProps = {
  metrics: PlayerMetrics;
  isLoading: boolean;
  hasResult: boolean;
  errorMessage?: string | null;
  duplicateJerseyWarning?: string | null;
  metricsWarning?: string | null;
};

type MetricKey = keyof PlayerMetrics;

const METRIC_CONFIG: { key: MetricKey; label: string; unit?: string }[] = [
  { key: "goals", label: "Goals" },
  { key: "shots", label: "Shots" },
  { key: "passes", label: "Passes" },
  { key: "distanceCoveredKm", label: "Distance covered", unit: "km" },
];

function formatMetricValue(
  key: MetricKey,
  value: number | null,
  unit?: string,
): string {
  if (value === null || !Number.isFinite(value)) return "—";
  if (key === "distanceCoveredKm") {
    return `${value.toFixed(2)} ${unit ?? "km"}`;
  }
  return unit ? `${value} ${unit}` : String(value);
}

function allMetricsEmpty(metrics: PlayerMetrics): boolean {
  return Object.values(metrics).every((v) => v === null);
}

export function PlayerMetricsPanel({
  metrics,
  isLoading,
  hasResult,
  errorMessage = null,
  duplicateJerseyWarning,
  metricsWarning,
}: PlayerMetricsPanelProps) {
  const pendingBackend =
    hasResult && !isLoading && allMetricsEmpty(metrics) && !metricsWarning;

  return (
    <section
      className={styles.panel}
      aria-labelledby="player-metrics-heading"
      aria-busy={isLoading}
    >
      <h2 id="player-metrics-heading" className={styles.heading}>
        Player metrics
      </h2>

      {metricsWarning && (
        <p className={styles.warning} role="alert">
          {metricsWarning}
        </p>
      )}

      {duplicateJerseyWarning && (
        <p className={styles.warning} role="alert">
          {duplicateJerseyWarning}
        </p>
      )}

      {errorMessage && (
        <p className={styles.error} role="alert">
          <span className="srOnly">Error: </span>
          {errorMessage}
        </p>
      )}

      {isLoading && (
        <p className={styles.loading} aria-live="polite">
          Calculating metrics…
        </p>
      )}

      {!isLoading && pendingBackend && (
        <p className={styles.pendingNote}>
          Metrics will appear here once the stats engine is connected.
        </p>
      )}

      {!isLoading && !errorMessage && (
        <dl className={styles.grid}>
          {METRIC_CONFIG.map(({ key, label, unit }) => {
            const value = metrics[key];
            const display = formatMetricValue(key, value, unit);
            return (
              <div key={key} className={styles.metric}>
                <dt className={styles.metricLabel}>{label}</dt>
                <dd className={styles.metricValue}>
                  {display}
                  {value === null && (
                    <span className="srOnly">, no data yet</span>
                  )}
                </dd>
              </div>
            );
          })}
        </dl>
      )}
    </section>
  );
}
