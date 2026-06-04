import type { ComponentType } from "react";
import type { PlayerMetrics } from "@/types/playerMetrics";
import styles from "./PlayerMetricsPanel.module.css";
import {
  DriveIcon,
  GoalIcon,
  LocationIcon,
  PassIcon,
  RulerIcon,
  TargetIcon,
} from "./icons";

type PlayerMetricsPanelProps = {
  metrics: PlayerMetrics;
  isLoading: boolean;
  hasResult: boolean;
  errorMessage?: string | null;
  duplicateJerseyWarning?: string | null;
  metricsWarning?: string | null;
};

type MetricKey = keyof PlayerMetrics;

type MetricConfig = {
  key: MetricKey;
  label: string;
  caption: string;
  unit?: string;
  Icon: ComponentType<{ className?: string }>;
};

const METRIC_CONFIG: MetricConfig[] = [
  { key: "goals", label: "Goals", caption: "expected", Icon: GoalIcon },
  { key: "shots", label: "Shots", caption: "on target", Icon: TargetIcon },
  { key: "passes", label: "Passes", caption: "accuracy", Icon: PassIcon },
  { key: "drives", label: "Drives", caption: "successful", Icon: DriveIcon },
  {
    key: "driveContactM",
    label: "Drive distance",
    caption: "ball contact",
    unit: "m",
    Icon: RulerIcon,
  },
  {
    key: "distanceCoveredKm",
    label: "Distance covered",
    caption: "km total",
    unit: "km",
    Icon: LocationIcon,
  },
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
  if (key === "driveContactM") {
    if (value >= 1000) {
      return `${(value / 1000).toFixed(2)} km`;
    }
    return `${Math.round(value)} ${unit ?? "m"}`;
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
      <header className={styles.header}>
        <h2 id="player-metrics-heading" className={styles.heading}>
          Player Metrics
        </h2>
        <p className={styles.subtitle}>Real-time statistics from video analysis</p>
      </header>

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
          {METRIC_CONFIG.map(({ key, label, caption, unit, Icon }) => {
            const value = metrics[key];
            const display = formatMetricValue(key, value, unit);
            const hasValue = value !== null && Number.isFinite(value);
            return (
              <div key={key} className={styles.metric}>
                <div className={styles.metricTop}>
                  <dt className={styles.metricLabel}>{label}</dt>
                  <span className={styles.metricIcon} aria-hidden="true">
                    <Icon className={styles.metricIconSvg} />
                  </span>
                </div>
                <dd className={styles.metricValue}>
                  {display}
                  {!hasValue && <span className="srOnly">, no data yet</span>}
                </dd>
                <span className={styles.metricCaption}>{caption}</span>
                <span className={styles.track} aria-hidden="true">
                  <span
                    className={`${styles.trackFill} ${hasValue ? styles.trackFillActive : ""}`}
                  />
                </span>
              </div>
            );
          })}
        </dl>
      )}
    </section>
  );
}
