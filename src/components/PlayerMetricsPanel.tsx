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

type BadmintonMetricItem =
  | (MetricConfig<BadmintonScalarKey> & { kind: "scalar" })
  | { kind: "serveReturn"; label: string; caption: string; Icon: ComponentType<{ className?: string }> }
  | { kind: "inOut"; label: string; caption: string; Icon: ComponentType<{ className?: string }> };

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

const BADMINTON_METRIC_CONFIG: BadmintonMetricItem[] = [
  {
    kind: "scalar",
    key: "totalRallies",
    label: "Total rallies",
    caption: "full match",
    Icon: GoalIcon,
  },
  {
    kind: "scalar",
    key: "avgRallyDurationS",
    label: "Avg rally duration",
    caption: "mean seconds",
    unit: "s",
    Icon: RulerIcon,
  },
  {
    kind: "scalar",
    key: "longestRallyS",
    label: "Longest rally",
    caption: "peak seconds",
    unit: "s",
    Icon: TargetIcon,
  },
  {
    kind: "serveReturn",
    label: "Points on serve vs return",
    caption: "won points split",
    Icon: PassIcon,
  },
  {
    kind: "inOut",
    label: "In / out calls",
    caption: "points won vs lost",
    Icon: DriveIcon,
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
  if (
    key === "avgRallyDurationS" ||
    key === "longestRallyS"
  ) {
    return `${value.toFixed(2)} ${unit ?? "s"}`.trim();
  }
  return unit ? `${value} ${unit}` : String(value);
}

function formatServeReturn(metrics: BadmintonMetrics): string {
  const total = metrics.totalPointsWon;
  const serve = metrics.pointsWonOnServe;
  const ret = metrics.pointsWonOnReturn;
  if (
    total === null ||
    serve === null ||
    ret === null ||
    !Number.isFinite(total)
  ) {
    return "—";
  }
  return `${serve}/${total} serve, ${ret}/${total} return`;
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
  if (metrics.totalPointsWon !== null) count += 1;
  if (metrics.pointsIn !== null || metrics.pointsOut !== null) count += 1;
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
    ? BADMINTON_METRIC_CONFIG.length
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

function hasServeReturn(metrics: BadmintonMetrics): boolean {
  return (
    metrics.totalPointsWon !== null &&
    metrics.pointsWonOnServe !== null &&
    metrics.pointsWonOnReturn !== null
  );
}

function hasInOut(metrics: BadmintonMetrics): boolean {
  return (
    (metrics.pointsIn !== null && Number.isFinite(metrics.pointsIn)) ||
    (metrics.pointsOut !== null && Number.isFinite(metrics.pointsOut))
  );
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
            ? BADMINTON_METRIC_CONFIG.map((item) => {
                const { label, caption, Icon } = item;
                if (item.kind === "scalar") {
                  const value = badmintonMetrics[item.key];
                  const display = formatBadmintonScalar(
                    item.key,
                    value,
                    item.unit,
                  );
                  const hasValue = hasBadmintonScalar(badmintonMetrics, item.key);
                  return (
                    <li key={item.key} className={styles.metric} role="listitem">
                      <div className={styles.metricTop}>
                        <span className={styles.metricLabel}>{label}</span>
                        <span className={styles.metricIcon} aria-hidden="true">
                          <Icon className={styles.metricIconSvg} />
                        </span>
                      </div>
                      <p className={styles.metricValue}>
                        <span className="srOnly">{label}: </span>
                        {display}
                        {!hasValue && (
                          <span className="srOnly">, no data yet</span>
                        )}
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
                }

                if (item.kind === "serveReturn") {
                  const display = formatServeReturn(badmintonMetrics);
                  const hasValue = hasServeReturn(badmintonMetrics);
                  return (
                    <li
                      key="serveReturn"
                      className={styles.metric}
                      role="listitem"
                    >
                      <div className={styles.metricTop}>
                        <span className={styles.metricLabel}>{label}</span>
                        <span className={styles.metricIcon} aria-hidden="true">
                          <Icon className={styles.metricIconSvg} />
                        </span>
                      </div>
                      <p className={styles.metricValue}>
                        <span className="srOnly">{label}: </span>
                        {display}
                        {!hasValue && (
                          <span className="srOnly">, no data yet</span>
                        )}
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
                }

                const inVal = badmintonMetrics.pointsIn;
                const outVal = badmintonMetrics.pointsOut;
                const hasValue = hasInOut(badmintonMetrics);
                return (
                  <li key="inOut" className={styles.metric} role="listitem">
                    <div className={styles.metricTop}>
                      <span className={styles.metricLabel}>{label}</span>
                      <span className={styles.metricIcon} aria-hidden="true">
                        <Icon className={styles.metricIconSvg} />
                      </span>
                    </div>
                    <div
                      className={styles.inOutSplit}
                      aria-label={
                        hasValue
                          ? `In ${inVal ?? 0}, out ${outVal ?? 0}`
                          : "In out calls unavailable"
                      }
                    >
                      <span className={styles.inOutSide}>
                        <span className={styles.inOutLabel}>In</span>
                        <span className={styles.inOutValueIn}>
                          {inVal ?? "—"}
                        </span>
                      </span>
                      <span className={styles.inOutDivider} aria-hidden="true" />
                      <span className={styles.inOutSide}>
                        <span className={styles.inOutLabel}>Out</span>
                        <span className={styles.inOutValueOut}>
                          {outVal ?? "—"}
                        </span>
                      </span>
                    </div>
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
              })
            : footballConfig.map(({ key, label, caption, unit, Icon }) => {
                const value = metrics[key];
                const display = formatFootballValue(key, value, unit);
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
                      {!hasValue && (
                        <span className="srOnly">, no data yet</span>
                      )}
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
