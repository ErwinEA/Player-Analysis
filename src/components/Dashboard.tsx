"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  analyzeVideo,
  checkHealth,
  fetchInsights,
  getApiBaseUrl,
  type MobileSamHealth,
} from "@/lib/api";
import {
  buildInsightsPayload,
  hasInsightsInput,
} from "@/lib/insightsPayload";
import type { AnalyzeResponse, MovementStats } from "@/types/analysis";
import {
  EMPTY_METRICS,
  type GameplayAnalysis,
  type PlayerMetrics,
} from "@/types/playerMetrics";
import styles from "./Dashboard.module.css";
import { Sidebar, type PlayerDetails } from "./Sidebar";
import { UploadAnalyzePanel } from "./UploadAnalyzePanel";
import { HeatMapPanel } from "./HeatMapPanel";
import { PlayerMetricsPanel } from "./PlayerMetricsPanel";
import { GameplayAnalysisPanel } from "./GameplayAnalysisPanel";
import { PitchCalibrationModal } from "./PitchCalibrationModal";
import { calibrationKeyFromFilename } from "@/lib/videoFrame";
import { AnalyzeIcon, LogoIcon } from "./icons";

const INITIAL_DETAILS: PlayerDetails = {
  name: "",
  jerseyNumber: 0,
  primaryJerseyColor: "",
  secondaryJerseyColor: "",
  teamName: "",
};

type ApiStatus = "checking" | "online" | "offline";
type AnalyzeState = "idle" | "loading" | "done" | "error";
type InsightsState = "idle" | "loading" | "done" | "unavailable";

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
  // Backend sets event_counts and provenance=inferred together; match gameplay summary gating.
  if (!ec || result.provenance !== "inferred") return base;
  const driveContactM =
    result.drive_contact_m != null && Number.isFinite(result.drive_contact_m)
      ? result.drive_contact_m
      : null;
  return {
    ...base,
    goals: ec.Goal,
    shots: ec.Shot,
    passes: ec.Pass,
    drives: ec.Drive,
    driveContactM,
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
  const [insightsState, setInsightsState] = useState<InsightsState>("idle");
  const [insightsUnavailableReason, setInsightsUnavailableReason] = useState<
    string | null
  >(null);
  const [insightsError, setInsightsError] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [calibrationOpen, setCalibrationOpen] = useState(false);
  const [uploadCalibrationReady, setUploadCalibrationReady] = useState(false);
  const [pitchTemplateKey, setPitchTemplateKey] = useState(0);
  const [mobileSamHealth, setMobileSamHealth] = useState<MobileSamHealth | null>(
    null,
  );
  const analyzeRunRef = useRef(0);
  const insightsRunRef = useRef(0);

  const videoCalibrationKey = video
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
    setInsightsState("idle");
    setInsightsUnavailableReason(null);
    setInsightsError(null);
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
    setInsightsState("idle");
    setInsightsUnavailableReason(null);
    setInsightsError(null);

    try {
      const renderVideo =
        process.env.NEXT_PUBLIC_RENDER_ANALYZE_VIDEO === "1" ||
        process.env.NEXT_PUBLIC_RENDER_ANALYZE_VIDEO === "true";
      const response = await analyzeVideo(video, details, {
        calibrationName:
          uploadCalibrationReady && videoCalibrationKey
            ? videoCalibrationKey
            : null,
        renderVideo,
      });
      if (runId !== analyzeRunRef.current) return;
      setResult(response);
      const { metrics: nextMetrics, warning } = metricsForResult(response);
      setMetrics(nextMetrics);
      setMetricsWarning(warning);
      setAnalyzeState("done");

      if (hasInsightsInput(response)) {
        const insightsRunId = ++insightsRunRef.current;
        setInsightsState("loading");
        try {
          const insightsResponse = await fetchInsights(
            buildInsightsPayload(response, details),
          );
          if (
            insightsRunId !== insightsRunRef.current ||
            runId !== analyzeRunRef.current
          ) {
            return;
          }
          if (insightsResponse.summary) {
            setGameplayAnalysis({ summary: insightsResponse.summary });
            setInsightsUnavailableReason(null);
            setInsightsState("done");
          } else {
            setGameplayAnalysis(null);
            setInsightsUnavailableReason(
              insightsResponse.unavailable_reason ??
                "AI insights unavailable — is Ollama running? (`ollama serve`)",
            );
            setInsightsState("unavailable");
          }
        } catch (insightsErr) {
          if (
            insightsRunId !== insightsRunRef.current ||
            runId !== analyzeRunRef.current
          ) {
            return;
          }
          setGameplayAnalysis(null);
          setInsightsError(
            insightsErr instanceof Error
              ? insightsErr.message
              : "Failed to fetch AI insights",
          );
          setInsightsState("unavailable");
        }
      }
    } catch (err) {
      if (runId !== analyzeRunRef.current) return;
      setErrorMessage(err instanceof Error ? err.message : "Analysis failed");
      setAnalyzeState("error");
    }
  }, [
    canAnalyze,
    video,
    details,
    uploadCalibrationReady,
    videoCalibrationKey,
  ]);

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
          <div className={styles.brand}>
            <span className={styles.brandMark} aria-hidden="true">
              <LogoIcon className={styles.brandMarkIcon} />
            </span>
            <div className={styles.brandText}>
              <h1 className={styles.headerTitle}>Player Analysis</h1>
              <span className={styles.brandSubtitle}>Pro Scout Intelligence</span>
            </div>
          </div>
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
            <div
              className={styles.analyzeAction}
              tabIndex={blockers.length > 0 ? 0 : undefined}
              role={blockers.length > 0 ? "group" : undefined}
              aria-label={
                blockers.length > 0 ? "Analyze unavailable" : undefined
              }
              aria-describedby={
                blockers.length > 0 ? "analyze-hint" : undefined
              }
            >
              <button
                type="button"
                className={styles.analyzeButton}
                onClick={handleAnalyze}
                disabled={!canAnalyze}
                aria-busy={isLoading}
              >
                <AnalyzeIcon className={styles.analyzeButtonIcon} />
                {isLoading ? "Analyzing…" : "Analyze"}
              </button>
            </div>
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
          <UploadAnalyzePanel
            file={video}
            onFileChange={setVideo}
            onCalibrate={() => setCalibrationOpen(true)}
            calibrationReady={uploadCalibrationReady}
            onCalibrationStatusChange={setUploadCalibrationReady}
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
            annotatedVideoUrl={
              result?.video_url
                ? `${getApiBaseUrl()}${result.video_url}`
                : null
            }
            annotatedVideoUnavailableReason={
              result?.video_unavailable_reason ?? null
            }
            maskFrameOffset={
              result?.video_url ? (result.video.frame_start ?? 0) : 0
            }
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
                result?.calibration_name ?? videoCalibrationKey
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
            isInsightsLoading={insightsState === "loading"}
            errorMessage={insightsError}
            unavailableReason={insightsUnavailableReason}
            hasResult={hasResult}
          />
        </main>
      </div>
      <footer className={styles.footer}>
        <div className={styles.footerBrand}>
          <span className={styles.footerMark} aria-hidden="true">
            N
          </span>
          <span className={styles.footerBrandText}>Neural Sports Analytics</span>
        </div>
        <nav className={styles.footerLinks} aria-label="Footer">
          <a className={styles.footerLink} href="#">
            Privacy
          </a>
          <a className={styles.footerLink} href="#">
            Terms
          </a>
          <a className={styles.footerLink} href="#">
            Documentation
          </a>
          <span className={styles.footerVersion}>v2.4.1</span>
        </nav>
      </footer>
      <PitchCalibrationModal
        open={calibrationOpen}
        calibrationName={videoCalibrationKey ?? "upload"}
        videoFile={video}
        frameIndex={100}
        onClose={() => setCalibrationOpen(false)}
        onSaved={() => {
          setPitchTemplateKey((k) => k + 1);
          setUploadCalibrationReady(true);
        }}
      />
    </div>
  );
}
