"use client";

import { useEffect, useMemo, useState } from "react";
import { getPitchTemplateUrl } from "@/lib/api";
import { BADMINTON_COURT_IMAGE_PATH } from "@/lib/courtConfig";
import type { HeatmapResult } from "@/types/analysis";
import type { Sport } from "@/types/sport";
import styles from "./HeatMapPanel.module.css";

type HeatMapPanelProps = {
  sport: Sport;
  isLoading: boolean;
  hasResult: boolean;
  heatmap: HeatmapResult | null | undefined;
  heatmapSource?: string | null;
  calibrationSkippedReason?: string | null;
  calibrationName?: string | null;
  frameCap?: number | null;
  pitchTemplateKey?: number;
  errorMessage?: string | null;
};

function getStatusText(
  sport: Sport,
  isLoading: boolean,
  hasResult: boolean,
  hasHeatmap: boolean,
  heatmapSource: string | null | undefined,
  pitchLoaded: boolean,
): string {
  const surface = sport === "badminton" ? "court" : "pitch";
  if (isLoading) return "Generating heat map…";
  if (hasResult && hasHeatmap && heatmapSource === "fallback_track") {
    return "Demo heat map (best-guess track — player not locked)";
  }
  if (hasResult && hasHeatmap) return "Player position heat map";
  if (hasResult) {
    return `No heat map — use a video with matching ${surface} calibration`;
  }
  if (pitchLoaded) return `${surface === "court" ? "Court" : "Pitch"} top view`;
  return "Heat map will appear after analysis";
}

function heatmapDataUrl(heatmap: HeatmapResult): string {
  return `data:image/png;base64,${heatmap.image_png_base64}`;
}

export function HeatMapPanel({
  sport,
  isLoading,
  hasResult,
  heatmap,
  heatmapSource,
  calibrationSkippedReason,
  calibrationName,
  frameCap,
  pitchTemplateKey = 0,
  errorMessage = null,
}: HeatMapPanelProps) {
  const [pitchSrc, setPitchSrc] = useState<string | null>(null);
  const [pitchError, setPitchError] = useState<string | null>(null);

  const displaySrc = useMemo(() => {
    if (heatmap?.image_png_base64) {
      return heatmapDataUrl(heatmap);
    }
    return pitchSrc;
  }, [heatmap, pitchSrc]);

  useEffect(() => {
    if (heatmap?.image_png_base64) {
      setPitchSrc((currentPitchSrc) => {
        if (currentPitchSrc?.startsWith("blob:")) {
          URL.revokeObjectURL(currentPitchSrc);
        }
        return null;
      });
      return;
    }

    if (sport === "badminton") {
      setPitchSrc(`${BADMINTON_COURT_IMAGE_PATH}?v=${pitchTemplateKey}`);
      setPitchError(null);
      return;
    }

    let cancelled = false;
    const url = `${getPitchTemplateUrl(calibrationName ?? undefined, sport)}&v=${pitchTemplateKey}`;

    async function loadPitch() {
      try {
        const res = await fetch(url);
        if (!res.ok) {
          throw new Error(`Pitch template unavailable (${res.status})`);
        }
        const blob = await res.blob();
        if (cancelled) return;
        setPitchSrc(URL.createObjectURL(blob));
        setPitchError(null);
      } catch (err) {
        if (!cancelled) {
          setPitchError(err instanceof Error ? err.message : "Could not load pitch template");
        }
      }
    }

    loadPitch();
    return () => {
      cancelled = true;
    };
  }, [heatmap, calibrationName, pitchTemplateKey, sport]);

  useEffect(() => {
    return () => {
      if (pitchSrc?.startsWith("blob:")) URL.revokeObjectURL(pitchSrc);
    };
  }, [pitchSrc]);

  const hasHeatmap = Boolean(heatmap?.image_png_base64);
  const surface = sport === "badminton" ? "court" : "pitch";
  const statusText = getStatusText(
    sport,
    isLoading,
    hasResult,
    hasHeatmap,
    heatmapSource,
    displaySrc !== null,
  );
  const heatmapStatusId = "heat-map-status";
  const heatmapLegendId = "heat-map-intensity-legend";
  const intensitySummary = hasHeatmap
    ? "Color intensity: cooler blue areas mean less time spent; warmer red areas mean more time spent. Scale runs from low (blue) through green and yellow to high (red)."
    : null;
  const heatmapAlt =
    hasHeatmap && heatmapSource === "fallback_track"
      ? `Demo heat map: ${heatmap?.sample_count ?? 0} samples on a ${heatmap?.grid_cols ?? 0} by ${heatmap?.grid_rows ?? 0} grid. Warmer red areas indicate more time spent; target player was not locked.`
      : hasHeatmap && heatmap
        ? `Player position heat map: ${heatmap.sample_count} samples on a ${heatmap.grid_cols} by ${heatmap.grid_rows} grid. Warmer red areas indicate more time spent.`
        : `Top-down ${surface} diagram`;
  const emptyLabel =
    pitchError && !displaySrc
      ? `${surface === "court" ? "Court" : "Pitch"} diagram unavailable`
      : statusText;

  return (
    <section
      className={styles.panel}
      aria-labelledby="heat-map-heading"
      aria-busy={isLoading}
    >
      <header className={styles.header}>
        <div className={styles.headerText}>
          <h2 id="heat-map-heading" className={styles.heading}>
            Position Heat Map
          </h2>
          <p
            className={styles.subtitle}
            role="status"
            aria-live="polite"
            aria-atomic="true"
          >
            {statusText}
          </p>
        </div>
        <span className={styles.dots} aria-hidden="true">
          <span className={`${styles.dot} ${styles.dotRed}`} />
          <span className={`${styles.dot} ${styles.dotAmber}`} />
          <span className={`${styles.dot} ${styles.dotGreen}`} />
        </span>
      </header>

      {(errorMessage || pitchError) && (
        <p className={styles.error} role="alert">
          <span className="srOnly">Error: </span>
          {errorMessage ?? pitchError}
        </p>
      )}

      <p
        id={heatmapStatusId}
        className={styles.meta}
        role="status"
        aria-live="polite"
      >
          {frameCap != null && frameCap > 0 && frameCap < 10_000_000 && (
            <span>First {frameCap} frames · </span>
          )}
          {heatmapSource === "fallback_track" && (
            <span className={styles.warn}>
              <span className="srOnly">Warning: </span>
              Best-guess track ·{" "}
            </span>
          )}
          {hasHeatmap && heatmap ? (
            <span>
              {heatmap.sample_count} positions · {heatmap.grid_cols}×
              {heatmap.grid_rows} grid · warmer/darker = more time spent
            </span>
          ) : hasResult ? (
            <span>
              {calibrationSkippedReason === "positions_out_of_bounds"
                ? `${surface === "court" ? "Court" : "Pitch"} calibration does not map players onto the ${surface} — recalibrate`
                : calibrationSkippedReason === "court_dimension_mismatch"
                  ? "Court calibration uses football dimensions — recalibrate in Badminton mode"
                  : calibrationSkippedReason === "size_mismatch"
                    ? `${surface === "court" ? "Court" : "Pitch"} calibration size mismatch`
                    : `No calibrated ${surface} overlay`}
            </span>
          ) : (
            <span>Runs after analyze</span>
          )}
        </p>

      {hasHeatmap && intensitySummary && (
        <p id={heatmapLegendId} className="srOnly">
          {intensitySummary}
        </p>
      )}

      <div className={styles.canvas}>
        {isLoading && (
          <p className={styles.loadingOverlay} aria-live="polite">
            Running detection on up to {frameCap ?? 5000} frames…
          </p>
        )}
        {displaySrc && !isLoading ? (
          <>
            {/* eslint-disable-next-line @next/next/no-img-element -- Uses runtime blob:/data: URLs from fetch/base64; next/image optimization is not applicable here. */}
            <img
              key={
                hasHeatmap ? heatmap?.image_png_base64.slice(0, 32) : "template"
              }
              src={displaySrc}
              alt={heatmapAlt}
              aria-describedby={
                hasHeatmap
                  ? `${heatmapStatusId} ${heatmapLegendId}`
                  : heatmapStatusId
              }
              className={styles.pitchImage}
            />
            {hasHeatmap && (
              <span className={styles.legend} aria-hidden="true">
                <span className={styles.legendLabel}>Low</span>
                <span className={styles.legendBar} />
                <span className={styles.legendLabel}>High</span>
              </span>
            )}
          </>
        ) : !isLoading ? (
          <p className={styles.label} aria-live="polite" aria-atomic="true">
            {emptyLabel}
          </p>
        ) : null}
      </div>
    </section>
  );
}
