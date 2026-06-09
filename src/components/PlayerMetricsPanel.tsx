"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { ComponentType } from "react";
import type { BadmintonMetrics, PlayerMetrics } from "@/types/playerMetrics";
import type { Sport } from "@/types/sport";
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
  sport: Sport;
  metrics: PlayerMetrics;
  badmintonMetrics: BadmintonMetrics;
  isLoading: boolean;
  hasResult: boolean;
  errorMessage?: string | null;
  duplicateJerseyWarning?: string | null;
  metricsWarning?: string | null;
};

type FootballMetricKey = keyof PlayerMetrics;
type BadmintonMetricKey = keyof BadmintonMetrics;

type MetricConfig<K extends string> = {
  key: K;
  label: string;
  caption: string;
  unit?: string;
  Icon: ComponentType<{ className?: string }>;
};

const FOOTBALL_METRIC_CONFIG: MetricConfig<FootballMetricKey>[] = [
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

const BADMINTON_METRIC_CONFIG: MetricConfig<BadmintonMetricKey>[] = [
  { key: "rallyWins", label: "Rally wins", caption: "rallies won", Icon: GoalIcon },
  { key: "winRatePct", label: "Win rate", caption: "rallies won %", unit: "%", Icon: TargetIcon },
  {
    key: "courtCoverageKm",
    label: "Court coverage",
    caption: "km moved",
    unit: "km",
    Icon: LocationIcon,
  },
  {
    key: "avgRallyDurationS",
    label: "Avg rally duration",
    caption: "seconds",
    unit: "s",
    Icon: RulerIcon,
  },
  { key: "winners", label: "Winners", caption: "unreturnable", Icon: DriveIcon },
  {
    key: "movementSpeedMs",
    label: "Movement speed",
    caption: "avg during rallies",
    unit: "m/s",
    Icon: PassIcon,
  },
];

function formatFootballValue(
  key: FootballMetricKey,
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

function formatBadmintonValue(
  key: BadmintonMetricKey,
  value: number | null,
  unit?: string,
): string {
  if (value === null || !Number.isFinite(value)) return "—";
  if (key === "courtCoverageKm") {
    return `${value.toFixed(2)} ${unit ?? "km"}`;
  }
  if (key === "winRatePct") {
    return `${value.toFixed(1)}${unit ?? "%"}`;
  }
  if (key === "avgRallyDurationS" || key === "movementSpeedMs") {
    return `${value.toFixed(2)} ${unit ?? ""}`.trim();
  }
  return unit ? `${value} ${unit}` : String(value);
}

function countPopulated(metrics: Record<string, number | null>): number {
  return Object.values(metrics).filter(
    (value) => value !== null && Number.isFinite(value),
  ).length;
}

function buildCompletionMessage(
  sport: Sport,
  metrics: PlayerMetrics,
  badmintonMetrics: BadmintonMetrics,
): string {
  const isBadminton = sport === "badminton";
  const active = isBadminton ? badmintonMetrics : metrics;
  const populated = countPopulated(active as Record<string, number | null>);
  const total = isBadminton
    ? BADMINTON_METRIC_CONFIG.length
    : FOOTBALL_METRIC_CONFIG.length;
  if (populated === 0) {
    return "Analysis complete. No player metrics available yet.";
  }
  return `Analysis complete. ${populated} of ${total} player metrics available.`;
}

export function PlayerMetricsPanel({
  sport,
  metrics,
  badmintonMetrics,
  isLoading,
  hasResult,
  errorMessage = null,
  duplicateJerseyWarning,
  metricsWarning,
}: PlayerMetricsPanelProps) {
  const isBadminton = sport === "badminton";
  const activeMetrics = isBadminton ? badmintonMetrics : metrics;
  const pendingBackend =
    hasResult &&
    !isLoading &&
    !metricsWarning &&
    countPopulated(activeMetrics as Record<string, number | null>) === 0;

  const [completionMessage, setCompletionMessage] = useState("");
  const wasLoadingRef = useRef(false);

  useEffect(() => {
    if (isLoading) {
      wasLoadingRef.current = true;
      return;
    }
    if (wasLoadingRef.current && hasResult && !errorMessage) {
      setCompletionMessage(
        buildCompletionMessage(sport, metrics, badmintonMetrics),
      );
      wasLoadingRef.current = false;
    }
  }, [isLoading, hasResult, errorMessage, sport, metrics, badmintonMetrics]);

  const metricConfig = useMemo(
    () => (isBadminton ? BADMINTON_METRIC_CONFIG : FOOTBALL_METRIC_CONFIG),
    [isBadminton],
  );

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
        <p className={styles.subtitle}>
          {isBadminton
            ? "Badminton statistics from video analysis"
            : "Real-time statistics from video analysis"}
        </p>
      </header>

      <p
        className="srOnly"
        role="status"
        aria-live="polite"
        aria-atomic="true"
      >
        {completionMessage}
      </p>

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
        <p className={styles.loading} role="status" aria-live="polite">
          Calculating metrics…
        </p>
      )}

      {!isLoading && pendingBackend && (
        <p className={styles.pendingNote} role="status">
          Metrics will appear here once analysis completes with court calibration.
        </p>
      )}

      {!isLoading && !errorMessage && (
        <ul className={styles.grid} role="list" aria-label="Player metrics">
          {metricConfig.map(({ key, label, caption, unit, Icon }) => {
            const value = (activeMetrics as Record<string, number | null>)[key];
            const display = isBadminton
              ? formatBadmintonValue(key as BadmintonMetricKey, value, unit)
              : formatFootballValue(key as FootballMetricKey, value, unit);
            const hasValue = value !== null && Number.isFinite(value);
            return (
              <li key={key} className={styles.metric} role="listitem">
                <div className={styles.metricTop}>
                  <span className={styles.metricLabel}>{label}</span>
                  <span className={styles.metricIcon} aria-hidden="true">
                    <Icon className={styles.metricIconSvg} />
                  </span>
                </div>
                <p className={styles.metricValue}>
                  <span className="srOnly">{label}: </span>
                  {display}
                  {!hasValue && <span className="srOnly">, no data yet</span>}
                </p>
                <span className={styles.metricCaption} aria-hidden="true">
                  {caption}
                </span>
                <span className={styles.track} aria-hidden="true">
                  <span
                    className={`${styles.trackFill} ${hasValue ? styles.trackFillActive : ""}`}
                  />
                </span>
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
}
