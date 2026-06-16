"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { ComponentType } from "react";
import type { HeatmapZoneSummary } from "@/types/analysis";
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
  zoneSummary?: HeatmapZoneSummary | null;
  badmintonStatsUnavailableReason?: string | null;
  isLoading: boolean;
  hasResult: boolean;
  errorMessage?: string | null;
  duplicateJerseyWarning?: string | null;
  metricsWarning?: string | null;
};

type FootballMetricKey = keyof PlayerMetrics;
type BadmintonScalarKey =
  | "totalRallies"
  | "avgRallyDurationS"
  | "longestRallyS";

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

const BADMINTON_RALLY_CONFIG: MetricConfig<BadmintonScalarKey>[] = [
  {
    key: "totalRallies",
    label: "Total rallies",
    caption: "heuristic count",
    Icon: GoalIcon,
  },
  {
    key: "avgRallyDurationS",
    label: "Avg rally duration",
    caption: "mean seconds",
    unit: "s",
    Icon: RulerIcon,
  },
  {
    key: "longestRallyS",
    label: "Longest rally",
    caption: "peak seconds",
    unit: "s",
    Icon: TargetIcon,
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

function formatBadmintonScalar(
  key: BadmintonScalarKey,
  value: number | null,
  unit?: string,
): string {
  if (value === null || !Number.isFinite(value)) return "—";
  if (key === "avgRallyDurationS" || key === "longestRallyS") {
    return `${value.toFixed(2)} ${unit ?? "s"}`.trim();
  }
  return unit ? `${value} ${unit}` : String(value);
}

function countFootballPopulated(metrics: PlayerMetrics): number {
  return Object.values(metrics).filter(
    (value) => value !== null && Number.isFinite(value),
  ).length;
}

function countBadmintonPopulated(metrics: BadmintonMetrics): number {
  let count = 0;
  if (metrics.totalRallies !== null) count += 1;
  if (metrics.avgRallyDurationS !== null) count += 1;
  if (metrics.longestRallyS !== null) count += 1;
  return count;
}

function buildCompletionMessage(
  sport: Sport,
  metrics: PlayerMetrics,
  badmintonMetrics: BadmintonMetrics,
): string {
  const isBadminton = sport === "badminton";
  const populated = isBadminton
    ? countBadmintonPopulated(badmintonMetrics)
    : countFootballPopulated(metrics);
  const total = isBadminton
    ? BADMINTON_RALLY_CONFIG.length
    : FOOTBALL_METRIC_CONFIG.length;
  if (populated === 0) {
    return "Analysis complete. No player metrics available yet.";
  }
  return `Analysis complete. ${populated} of ${total} player metrics available.`;
}

function hasBadmintonScalar(
  metrics: BadmintonMetrics,
  key: BadmintonScalarKey,
): boolean {
  const value = metrics[key];
  return value !== null && Number.isFinite(value);
}

function rallyEmptyCaption(reason: string | null | undefined): string | null {
  if (reason === "rally_detection_pending") {
    return "No rallies detected in clip";
  }
  if (reason === "rallies_replay_filtered") {
    return "Excluded as replay/highlight footage";
  }
  if (reason === "no_shuttle_weights") {
    return "Shuttle weights missing";
  }
  if (reason === "no_calibration") {
    return "Needs court calibration";
  }
  if (reason === "no_lock") {
    return "Needs player lock";
  }
  return null;
}

function renderMetricCard(
  key: string,
  label: string,
  caption: string,
  display: string,
  hasValue: boolean,
  Icon: ComponentType<{ className?: string }>,
  emptyCaption?: string | null,
) {
  const showCaption = hasValue ? caption : (emptyCaption ?? caption);
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
        {showCaption}
      </span>
      <span className={styles.track} aria-hidden="true">
        <span
          className={`${styles.trackFill} ${hasValue ? styles.trackFillActive : ""}`}
        />
      </span>
    </li>
  );
}

export function PlayerMetricsPanel({
  sport,
  metrics,
  badmintonMetrics,
  zoneSummary = null,
  badmintonStatsUnavailableReason = null,
  isLoading,
  hasResult,
  errorMessage = null,
  duplicateJerseyWarning,
  metricsWarning,
}: PlayerMetricsPanelProps) {
  const isBadminton = sport === "badminton";
  const rallyEmpty = rallyEmptyCaption(badmintonStatsUnavailableReason);
  const showHeuristicBanner =
    isBadminton &&
    hasResult &&
    !isLoading &&
    (badmintonMetrics.totalRallies !== null ||
      badmintonMetrics.avgRallyDurationS !== null ||
      badmintonMetrics.longestRallyS !== null);

  const pendingBackend =
    hasResult &&
    !isLoading &&
    !metricsWarning &&
    (isBadminton
      ? countBadmintonPopulated(badmintonMetrics) === 0
      : countFootballPopulated(metrics) === 0);

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

  const footballConfig = useMemo(() => FOOTBALL_METRIC_CONFIG, []);

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

      {showHeuristicBanner && (
        <p className={styles.warning} role="note">
          Heuristic estimates — not official scoring.
        </p>
      )}

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
          {isBadminton
            ? BADMINTON_RALLY_CONFIG.map((item) => {
                const value = badmintonMetrics[item.key];
                const display = formatBadmintonScalar(
                  item.key,
                  value,
                  item.unit,
                );
                const hasValue = hasBadmintonScalar(badmintonMetrics, item.key);
                return renderMetricCard(
                  item.key,
                  item.label,
                  item.caption,
                  display,
                  hasValue,
                  item.Icon,
                  rallyEmpty,
                );
              })
            : footballConfig.map(({ key, label, caption, unit, Icon }) => {
                const value = metrics[key];
                const display = formatFootballValue(key, value, unit);
                const hasValue = value !== null && Number.isFinite(value);
                return renderMetricCard(
                  key,
                  label,
                  caption,
                  display,
                  hasValue,
                  Icon,
                );
              })}
        </ul>
      )}

      {isBadminton && zoneSummary && !isLoading && (
        <div className={styles.pendingNote} role="note">
          <strong>Court zones: </strong>
          defensive {zoneSummary.defensive_third_pct.toFixed(0)}%, middle{" "}
          {zoneSummary.middle_third_pct.toFixed(0)}%, attacking{" "}
          {zoneSummary.attacking_third_pct.toFixed(0)}% — left{" "}
          {zoneSummary.left_pct.toFixed(0)}%, center{" "}
          {zoneSummary.center_pct.toFixed(0)}%, right{" "}
          {zoneSummary.right_pct.toFixed(0)}%
        </div>
      )}
    </section>
  );
}
