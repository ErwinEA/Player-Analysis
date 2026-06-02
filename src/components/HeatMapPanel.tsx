"use client";

import { useEffect, useMemo, useState } from "react";
import { getPitchTemplateUrl } from "@/lib/api";
import type { HeatmapResult } from "@/types/analysis";
import styles from "./HeatMapPanel.module.css";

type HeatMapPanelProps = {
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
  isLoading: boolean,
  hasResult: boolean,
  hasHeatmap: boolean,
  heatmapSource: string | null | undefined,
  pitchLoaded: boolean,
): string {
  if (isLoading) return "Generating heat map…";
  if (hasResult && hasHeatmap && heatmapSource === "fallback_track") {
    return "Demo heat map (best-guess track — player not locked)";
  }
  if (hasResult && hasHeatmap) return "Player position heat map";
  if (hasResult) {
    return "No heat map — use a video with matching pitch calibration (e.g. testmatch2)";
  }
  if (pitchLoaded) return "Pitch top view";
  return "Heat map will appear after analysis";
}

function heatmapDataUrl(heatmap: HeatmapResult): string {
  return `data:image/png;base64,${heatmap.image_png_base64}`;
}

export function HeatMapPanel({
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
        if (currentPitchSrc) URL.revokeObjectURL(currentPitchSrc);
        return null;
      });
      return;
    }

    let cancelled = false;
    const url = `${getPitchTemplateUrl(calibrationName ?? undefined)}&v=${pitchTemplateKey}`;

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
  }, [heatmap, calibrationName, pitchTemplateKey]);

  useEffect(() => {
    return () => {
      if (pitchSrc) URL.revokeObjectURL(pitchSrc);
    };
  }, [pitchSrc]);

  const hasHeatmap = Boolean(heatmap?.image_png_base64);
  const statusText = getStatusText(
    isLoading,
    hasResult,
    hasHeatmap,
    heatmapSource,
    displaySrc !== null,
  );
  const heatmapStatusId = "heat-map-status";
  const heatmapAlt =
    hasHeatmap && heatmapSource === "fallback_track"
      ? "Demo heat map on football pitch; target player was not locked"
      : hasHeatmap
        ? "Player position heat map on football pitch"
        : "Top-down football pitch diagram";
  const emptyLabel =
    pitchError && !displaySrc ? "Pitch diagram unavailable" : statusText;

  return (
    <section
      className={styles.panel}
      aria-labelledby="heat-map-heading"
      aria-busy={isLoading}
    >
      <h2 id="heat-map-heading" className={styles.heading}>
        Heat map
      </h2>

      {(errorMessage || pitchError) && (
        <p className={styles.error} role="alert">
          <span className="srOnly">Error: </span>
          {errorMessage ?? pitchError}
        </p>
      )}

      <p id={heatmapStatusId} className={styles.meta}>
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
              {heatmap.sample_count} positions · {heatmap.grid_cols}×{heatmap.grid_rows} grid
            </span>
          ) : hasResult ? (
            <span>
              {calibrationSkippedReason === "positions_out_of_bounds"
                ? "Pitch calibration does not map players onto the field — recalibrate"
                : calibrationSkippedReason === "size_mismatch"
                  ? "Pitch calibration size mismatch"
                  : "No calibrated pitch overlay"}
            </span>
          ) : (
            <span>Runs after analyze</span>
          )}
        </p>

      <div className={styles.canvas}>
        {isLoading && (
          <p className={styles.loadingOverlay} aria-live="polite">
            Running detection on up to {frameCap ?? 5000} frames…
          </p>
        )}
        {displaySrc && !isLoading ? (
          // eslint-disable-next-line @next/next/no-img-element -- Uses runtime blob:/data: URLs from fetch/base64; next/image optimization is not applicable here.
          <img
            key={hasHeatmap ? heatmap?.image_png_base64.slice(0, 32) : "template"}
            src={displaySrc}
            alt={heatmapAlt}
            aria-describedby={heatmapStatusId}
            className={styles.pitchImage}
          />
        ) : !isLoading ? (
          <p className={styles.label} aria-live="polite" aria-atomic="true">
            {emptyLabel}
          </p>
        ) : null}
      </div>
    </section>
  );
}
