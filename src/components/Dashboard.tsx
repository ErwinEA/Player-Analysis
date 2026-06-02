"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { analyzeVideo, checkHealth, type MobileSamHealth } from "@/lib/api";
import type { AnalyzeResponse, MovementStats } from "@/types/analysis";
import {
  EMPTY_METRICS,
  type GameplayAnalysis,
  type PlayerMetrics,
} from "@/types/playerMetrics";
import styles from "./Dashboard.module.css";
import { Sidebar, type PlayerDetails } from "./Sidebar";
import { LegacyVideoPanel } from "./LegacyVideoPanel";
import { HeatMapPanel } from "./HeatMapPanel";
import { PlayerMetricsPanel } from "./PlayerMetricsPanel";
import { GameplayAnalysisPanel } from "./GameplayAnalysisPanel";
import { PitchCalibrationModal } from "./PitchCalibrationModal";
import { calibrationKeyFromFilename } from "@/lib/videoFrame";

const INITIAL_DETAILS: PlayerDetails = {
  name: "",
  jerseyNumber: 0,
  primaryJerseyColor: "",
  secondaryJerseyColor: "",
  teamName: "",
};

type ApiStatus = "checking" | "online" | "offline";
type AnalyzeState = "idle" | "loading" | "done" | "error";

function isDetailsValid(details: PlayerDetails): boolean {
  return details.jerseyNumber >= 1 && details.jerseyNumber <= 99;
}

function metricsFromMovement(
  movement: MovementStats | null | undefined,
): PlayerMetrics {
  if (!movement || movement.units !== "meters") {
    return EMPTY_METRICS;
  }
  return {
    ...EMPTY_METRICS,
    distanceCoveredKm: Math.round((movement.distance_m / 1000) * 100) / 100,
  };
}

function isConfidentPlayerLock(result: AnalyzeResponse): boolean {
  const method = result.target.method;
  return (
    result.target.track_id != null &&
    method != null &&
    method !== "none" &&
    method !== "color" &&
    result.heatmap_source !== "fallback_track"
  );
}

function metricsFromEvents(
  result: AnalyzeResponse,
  base: PlayerMetrics,
): PlayerMetrics {
  const ec = result.event_counts;
  if (!ec) return base;
  return {
    ...base,
    goals: ec.Goal,
    shots: ec.Shot,
    passes: ec.Pass,
    drives: ec.Drive,
  };
}

function eventsWarning(result: AnalyzeResponse): string | null {
  const reason = result.events_unavailable_reason;
  if (!reason) return null;
  if (reason === "no_ball_weights") {
    return "Ball event stats unavailable: add yolov8n_ball.pt (see backend/weights/README.md).";
  }
  if (reason === "no_calibration") {
    return "Ball event stats need pitch calibration. Calibrate the pitch, then re-run Analyze.";
  }
  if (reason === "no_lock") {
    return "Ball event stats need a confident player lock.";
  }
  return null;
}

function metricsForResult(result: AnalyzeResponse): {
  metrics: PlayerMetrics;
  warning: string | null;
} {
  let metrics = metricsFromMovement(result.movement);
  metrics = metricsFromEvents(result, metrics);

  const eventWarn = eventsWarning(result);
  if (result.calibration_skipped_reason === "positions_out_of_bounds") {
    return {
      metrics,
      warning:
        "Pitch calibration is off: player positions fall outside the field. Recalibrate, then re-run Analyze.",
    };
  }
  if (isConfidentPlayerLock(result)) {
    return { metrics, warning: eventWarn };
  }
  const method = result.target.method;
  const lockWarning =
    method === "none" || result.target.track_id == null
      ? "Could not lock onto this jersey number in the video. Distance and heat map may not match your player."
      : "Player lock was uncertain. Treat distance and heat map as approximate.";
  return { metrics, warning: eventWarn ?? lockWarning };
}

function getAnalyzeBlockers(
  apiStatus: ApiStatus,
  video: File | null,
  details: PlayerDetails,
): string[] {
  const blockers: string[] = [];
  if (apiStatus === "checking") blockers.push("Checking API…");
  if (apiStatus === "offline") {
    blockers.push("API offline — start backend on port 8000");
  }
  if (apiStatus !== "online") return blockers;
  if (!video) blockers.push("Upload a video");
  if (details.jerseyNumber <= 0) blockers.push("Enter jersey number (1–99)");
  else if (details.jerseyNumber > 99) {
    blockers.push("Jersey number must be 99 or less");
  }
  return blockers;
}

export function Dashboard() {
  const [details, setDetails] = useState<PlayerDetails>(INITIAL_DETAILS);
  const [video, setVideo] = useState<File | null>(null);
  const [apiStatus, setApiStatus] = useState<ApiStatus>("checking");
  const [analyzeState, setAnalyzeState] = useState<AnalyzeState>("idle");
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [metrics, setMetrics] = useState<PlayerMetrics>(EMPTY_METRICS);
  const [metricsWarning, setMetricsWarning] = useState<string | null>(null);
  const [gameplayAnalysis, setGameplayAnalysis] =
    useState<GameplayAnalysis | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [calibrationOpen, setCalibrationOpen] = useState(false);
  const [legacyCalibrationReady, setLegacyCalibrationReady] = useState(false);
  const [pitchTemplateKey, setPitchTemplateKey] = useState(0);
  const [mobileSamHealth, setMobileSamHealth] = useState<MobileSamHealth | null>(
    null,
  );
  const analyzeRunRef = useRef(0);

  const legacyCalibrationKey = video
    ? calibrationKeyFromFilename(video.name)
    : null;

  const refreshApiStatus = useCallback(async () => {
    setApiStatus("checking");
    try {
      const health = await checkHealth();
      setMobileSamHealth(health.mobile_sam);
      setApiStatus("online");
    } catch {
      setMobileSamHealth(null);
      setApiStatus("offline");
    }
  }, []);

  useEffect(() => {
    refreshApiStatus();
  }, [refreshApiStatus]);

  useEffect(() => {
    if (apiStatus !== "offline") return;
    const id = window.setInterval(() => {
      void refreshApiStatus();
    }, 10000);
    return () => window.clearInterval(id);
  }, [apiStatus, refreshApiStatus]);

  useEffect(() => {
    analyzeRunRef.current += 1;
    setResult(null);
    setMetrics(EMPTY_METRICS);
    setMetricsWarning(null);
    setGameplayAnalysis(null);
    setErrorMessage(null);
    setAnalyzeState("idle");
  }, [
    video,
    details.name,
    details.jerseyNumber,
    details.primaryJerseyColor,
    details.secondaryJerseyColor,
    details.teamName,
  ]);

  const blockers = getAnalyzeBlockers(apiStatus, video, details);

  const canAnalyze =
    apiStatus === "online" &&
    blockers.length === 0 &&
    analyzeState !== "loading" &&
    isDetailsValid(details);

  const handleAnalyze = useCallback(async () => {
    if (!canAnalyze || !video) return;

    const runId = ++analyzeRunRef.current;
    setAnalyzeState("loading");
    setErrorMessage(null);
    setResult(null);
    setMetrics(EMPTY_METRICS);
    setMetricsWarning(null);
    setGameplayAnalysis(null);

    try {
      const response = await analyzeVideo(video, details);
      if (runId !== analyzeRunRef.current) return;
      setResult(response);
      const { metrics: nextMetrics, warning } = metricsForResult(response);
      setMetrics(nextMetrics);
      setMetricsWarning(warning);
      if (response.event_counts && response.provenance === "inferred") {
        const ec = response.event_counts;
        setGameplayAnalysis({
          summary: `Inferred: ${ec.Pass} passes, ${ec.Shot} shots, ${ec.Goal} goals, ${ec.Drive} drives.`,
        });
      } else {
        setGameplayAnalysis(null);
      }
      setAnalyzeState("done");
    } catch (err) {
      if (runId !== analyzeRunRef.current) return;
      setErrorMessage(err instanceof Error ? err.message : "Analysis failed");
      setAnalyzeState("error");
    }
  }, [canAnalyze, video, details]);

  const statusLabel =
    apiStatus === "checking"
      ? "Checking API…"
      : apiStatus === "online"
        ? "API online"
        : "API offline";

  const isLoading = analyzeState === "loading";
  const hasResult = result !== null;
  const insightError = analyzeState === "error" ? errorMessage : null;

  return (
    <div className={styles.root}>
      <a
        id="skip-to-main"
        href="#main-content"
        className="skipLink"
        tabIndex={calibrationOpen ? -1 : undefined}
        aria-hidden={calibrationOpen ? true : undefined}
      >
        Skip to main content
      </a>
      <header className={styles.headerShell}>
        <div className={styles.headerBar}>
          <h1 className={styles.headerTitle}>Player Analysis</h1>
          <div className={styles.headerControls}>
            <div className={styles.apiStatusGroup}>
              <div
                className={`${styles.statusBadge} ${styles[`status_${apiStatus}`]}`}
              >
                <span className={styles.statusDot} aria-hidden="true" />
                <span className={styles.statusText} role="status">
                  {statusLabel}
                </span>
              </div>
              {apiStatus === "offline" && (
                <button
                  type="button"
                  className={styles.retryApiButton}
                  onClick={() => void refreshApiStatus()}
                >
                  Retry
                </button>
              )}
            </div>
            <button
              type="button"
              className={styles.analyzeButton}
              onClick={handleAnalyze}
              disabled={!canAnalyze}
              aria-describedby={
                blockers.length > 0 ? "analyze-hint" : undefined
              }
              aria-busy={isLoading}
            >
              {isLoading ? "Analyzing…" : "Analyze"}
            </button>
          </div>
        </div>
        {blockers.length > 0 && (
          <div className={styles.headerHintBar}>
            <p
              id="analyze-hint"
              className={styles.analyzeHint}
              role="status"
              aria-live="polite"
            >
              <span className={styles.analyzeHintLabel}>Required:</span>
              {blockers.join(" · ")}
            </p>
          </div>
        )}
      </header>
      {insightError && (
        <p className={styles.globalError} role="alert">
          Analysis failed: {insightError}
        </p>
      )}
      <div className={styles.body}>
        <Sidebar details={details} onChange={setDetails} />
        <main id="main-content" className={styles.main}>
          <LegacyVideoPanel
            file={video}
            onFileChange={setVideo}
            onCalibrate={() => setCalibrationOpen(true)}
            calibrationReady={legacyCalibrationReady}
            onCalibrationStatusChange={setLegacyCalibrationReady}
            trackingRows={
              result?.heatmap_source === "locked_target" &&
              result.target.method !== "none" &&
              result.target.method !== "color"
                ? result.rows
                : null
            }
            videoFps={result?.video.fps ?? 30}
            analyzeComplete={analyzeState === "done" && result != null}
            jerseyNumber={details.jerseyNumber}
            masksUnavailableReason={
              result?.masks_available === false
                ? result.masks_unavailable_reason
                : null
            }
            mobileSamHealth={mobileSamHealth}
          />
          <div className={styles.insightsRow}>
            <HeatMapPanel
              isLoading={isLoading}
              hasResult={hasResult}
              heatmap={result?.heatmap}
              heatmapSource={result?.heatmap_source}
              calibrationSkippedReason={result?.calibration_skipped_reason}
              frameCap={result?.video.frame_cap ?? 5000}
              calibrationName={
                result?.calibration_name ?? legacyCalibrationKey
              }
              pitchTemplateKey={pitchTemplateKey}
            />
            <PlayerMetricsPanel
              metrics={metrics}
              isLoading={isLoading}
              hasResult={hasResult}
              metricsWarning={metricsWarning}
            />
          </div>
          <GameplayAnalysisPanel
            analysis={gameplayAnalysis}
            isLoading={isLoading}
            hasResult={hasResult}
          />
        </main>
      </div>
      <PitchCalibrationModal
        open={calibrationOpen}
        calibrationName={legacyCalibrationKey ?? "upload"}
        videoFile={video}
        frameIndex={100}
        onClose={() => setCalibrationOpen(false)}
        onSaved={() => {
          setPitchTemplateKey((k) => k + 1);
          setLegacyCalibrationReady(true);
        }}
      />
    </div>
  );
}
