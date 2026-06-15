"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  INVALID_VIDEO_MESSAGE,
  isValidVideoFile,
  VIDEO_ACCEPT,
} from "@/lib/videoFile";
import {
  decodeMaskRle,
  drawMaskOnCanvasScaled,
  drawShuttleHighlight,
  rowAtPlaybackTime,
  shuttleAtPlaybackTime,
} from "@/lib/maskRle";
import type { MobileSamHealth } from "@/lib/api";
import type { Row, ShuttleSample } from "@/types/analysis";
import styles from "./VideoUploadPanel.module.css";
import { UploadCloudIcon, VideoIcon } from "./icons";

const VIDEO_FORMATS = ["MP4", "MOV"];

const UPLOAD_HINT_ID = "video-upload-format-hint";

type VideoUploadPanelProps = {
  file: File | null;
  onFileChange: (file: File | null) => void;
  trackingRows?: Row[] | null;
  videoFps?: number;
  analyzeComplete?: boolean;
  playerLabel?: string;
  masksUnavailableReason?: string | null;
  mobileSamHealth?: MobileSamHealth | null;
  annotatedVideoUrl?: string | null;
  annotatedVideoUnavailableReason?: string | null;
  maskFrameOffset?: number;
  shuttleSamples?: ShuttleSample[] | null;
};

export function VideoUploadPanel({
  file,
  onFileChange,
  trackingRows = null,
  videoFps = 30,
  analyzeComplete = false,
  playerLabel,
  masksUnavailableReason = null,
  mobileSamHealth = null,
  annotatedVideoUrl = null,
  annotatedVideoUnavailableReason = null,
  maskFrameOffset = 0,
  shuttleSamples = null,
}: VideoUploadPanelProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number | null>(null);
  const lastPaintedFrameRef = useRef<number | null>(null);
  const lastAnnouncedStatusRef = useRef("");
  const lastAnnouncedFrameRef = useRef<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [showMask, setShowMask] = useState(true);
  const [showShuttle, setShowShuttle] = useState(true);
  const [maskStatus, setMaskStatus] = useState("");
  const [isDragging, setIsDragging] = useState(false);

  const playbackUrl = annotatedVideoUrl ?? previewUrl;
  const showingAnnotated = Boolean(annotatedVideoUrl);

  const maskRows = useMemo(
    () => (trackingRows ?? []).filter((r) => r.mask_rle),
    [trackingRows],
  );

  const hasMasks = maskRows.length > 0;
  const hasShuttleSamples = Boolean(shuttleSamples && shuttleSamples.length > 0);
  // Shuttle is baked into the annotated clip; live canvas only when scrubbing the upload.
  const showLiveShuttle = hasShuttleSamples && showShuttle && !showingAnnotated;
  const hasOverlay = (hasMasks && showMask) || showLiveShuttle;
  const hasTrackingRows = Boolean(trackingRows && trackingRows.length > 0);
  const samStatus = mobileSamHealth?.status;
  const maskUnavailable =
    analyzeComplete &&
    hasTrackingRows &&
    !hasMasks &&
    samStatus !== "loading" &&
    samStatus !== "error";

  useEffect(() => {
    if (!hasMasks) return;
    setShowMask(true);
  }, [hasMasks]);

  useEffect(() => {
    if (!hasShuttleSamples) return;
    setShowShuttle(true);
  }, [hasShuttleSamples]);

  useEffect(() => {
    if (!file) {
      setPreviewUrl(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const announceMaskStatus = useCallback((message: string) => {
    if (lastAnnouncedStatusRef.current === message) return;
    lastAnnouncedStatusRef.current = message;
    setMaskStatus(message);
  }, []);

  const paintMask = useCallback(
    (options?: { announce?: boolean }) => {
      const announce = options?.announce ?? false;

      const maybeAnnounce = (message: string, frame?: number | null) => {
        if (!announce) return;
        if (frame !== undefined && frame !== null) {
          if (
            frame === lastAnnouncedFrameRef.current &&
            lastAnnouncedStatusRef.current === message
          ) {
            return;
          }
          lastAnnouncedFrameRef.current = frame;
        }
        announceMaskStatus(message);
      };

      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (!video || !canvas || !hasOverlay) {
        maybeAnnounce(
          hasOverlay
            ? "Video overlay: waiting for video…"
            : "Video overlay: hidden",
        );
        return;
      }

      const vw = video.videoWidth;
      const vh = video.videoHeight;
      if (!vw || !vh) return;

      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      if (canvas.width !== vw || canvas.height !== vh) {
        canvas.width = vw;
        canvas.height = vh;
      }

      const frame =
        Math.round(video.currentTime * videoFps) + maskFrameOffset;
      const row =
        showMask && hasMasks
          ? rowAtPlaybackTime(
              video.currentTime,
              videoFps,
              maskRows,
              maskFrameOffset,
            )
          : undefined;
      const shuttle =
        showLiveShuttle && shuttleSamples
          ? shuttleAtPlaybackTime(
              video.currentTime,
              videoFps,
              shuttleSamples,
              maskFrameOffset,
            )
          : undefined;

      if (!row?.mask_rle && !shuttle) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        lastPaintedFrameRef.current = null;
        const parts: string[] = [];
        if (showMask && hasMasks) parts.push("no player mask");
        if (showLiveShuttle) parts.push("no shuttle");
        maybeAnnounce(
          `Video overlay at ${video.currentTime.toFixed(1)}s (${parts.join(", ")})` +
            (playerLabel ? ` — ${playerLabel}` : ""),
        );
        return;
      }

      const paintFrame = row?.frame ?? shuttle?.frame ?? frame;
      if (lastPaintedFrameRef.current === paintFrame && !announce) {
        return;
      }
      lastPaintedFrameRef.current = paintFrame;

      ctx.clearRect(0, 0, vw, vh);

      if (row?.mask_rle) {
        const mh = Math.round(row.mask_rle.size[0]);
        const mw = Math.round(row.mask_rle.size[1]);
        const mask = decodeMaskRle(row.mask_rle);
        if (mask.length === mw * mh) {
          drawMaskOnCanvasScaled(ctx, mask, mw, mh, vw, vh);
        }
      }

      if (shuttle) {
        drawShuttleHighlight(ctx, shuttle.cx, shuttle.cy);
      }

      const statusParts: string[] = [];
      if (row?.mask_rle) {
        const src = row.segment_source ?? "sam";
        statusParts.push(`mask frame ${row.frame} (${src})`);
      }
      if (shuttle) {
        statusParts.push(`shuttle frame ${shuttle.frame}`);
      }
      const statusMessage =
        `Video overlay at ${video.currentTime.toFixed(1)}s: ${statusParts.join(", ")}` +
        (playerLabel ? ` — ${playerLabel}` : "");
      maybeAnnounce(statusMessage, paintFrame);
    },
    [
      announceMaskStatus,
      maskRows,
      showMask,
      showLiveShuttle,
      shuttleSamples,
      videoFps,
      hasMasks,
      hasOverlay,
      playerLabel,
      maskFrameOffset,
    ],
  );

  const schedulePaint = useCallback(
    (options?: { announce?: boolean }) => {
      if (rafRef.current !== null) return;
      rafRef.current = requestAnimationFrame(() => {
        rafRef.current = null;
        paintMask(options);
      });
    },
    [paintMask],
  );

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    const onTimeUpdate = () => schedulePaint();
    const onSeeked = () => schedulePaint({ announce: true });
    const onLoaded = () => schedulePaint({ announce: true });
    video.addEventListener("timeupdate", onTimeUpdate);
    video.addEventListener("seeked", onSeeked);
    video.addEventListener("loadeddata", onLoaded);
    video.addEventListener("loadedmetadata", onLoaded);

    const ro =
      typeof ResizeObserver !== "undefined"
        ? new ResizeObserver(() => schedulePaint())
        : null;
    ro?.observe(video);

    schedulePaint({ announce: true });
    return () => {
      video.removeEventListener("timeupdate", onTimeUpdate);
      video.removeEventListener("seeked", onSeeked);
      video.removeEventListener("loadeddata", onLoaded);
      video.removeEventListener("loadedmetadata", onLoaded);
      ro?.disconnect();
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
    };
  }, [schedulePaint, playbackUrl, hasOverlay]);

  useEffect(() => {
    lastPaintedFrameRef.current = null;
    lastAnnouncedFrameRef.current = null;
    lastAnnouncedStatusRef.current = "";
    schedulePaint({ announce: true });
  }, [schedulePaint, trackingRows, showMask, showShuttle, shuttleSamples]);

  const openFilePicker = useCallback(() => {
    inputRef.current?.click();
  }, []);

  const acceptFile = useCallback(
    (selected: File | null | undefined) => {
      if (!selected) return;

      if (!isValidVideoFile(selected)) {
        setError(INVALID_VIDEO_MESSAGE);
        onFileChange(null);
        return;
      }

      setError(null);
      onFileChange(selected);
    },
    [onFileChange],
  );

  const handleFileChange = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const selected = event.target.files?.[0];
      event.target.value = "";
      acceptFile(selected);
    },
    [acceptFile],
  );

  const handleDrop = useCallback(
    (event: React.DragEvent<HTMLButtonElement>) => {
      event.preventDefault();
      setIsDragging(false);
      acceptFile(event.dataTransfer.files?.[0]);
    },
    [acceptFile],
  );

  const handleDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    setIsDragging(false);
  }, []);

  const describedBy = [
    file ? "video-filename" : null,
    error ? "video-error" : null,
    !file ? UPLOAD_HINT_ID : null,
  ]
    .filter(Boolean)
    .join(" ");

  const fileInputId = "upload-video-file-input";

  return (
    <section className={styles.wrapper} aria-labelledby="video-upload-heading">
      <header className={styles.cardHeader}>
        <span className={styles.cardHeaderTitle}>
          <span className={styles.cardHeaderIcon} aria-hidden="true">
            <VideoIcon className={styles.cardHeaderIconSvg} />
          </span>
          <h2 id="video-upload-heading" className={styles.cardTitle}>
            Video Upload
          </h2>
        </span>
        <span className={styles.cardHeaderMeta}>
          {file ? file.name : "No video selected"}
        </span>
      </header>
      <input
        id={fileInputId}
        ref={inputRef}
        type="file"
        accept={VIDEO_ACCEPT}
        className={styles.hiddenInput}
        onChange={handleFileChange}
        tabIndex={-1}
      />

      {playbackUrl ? (
        <div className={styles.previewContainer}>
          <figure className={styles.videoFigure}>
            <div className={styles.videoWrap}>
              <video
                key={playbackUrl}
                ref={videoRef}
                className={styles.preview}
                src={playbackUrl}
                controls
                aria-describedby={
                  hasMasks ? "video-mask-status video-filename" : "video-filename"
                }
              />
              {hasOverlay && (
                <canvas
                  ref={canvasRef}
                  className={styles.maskCanvas}
                  role={maskStatus ? "img" : undefined}
                  aria-hidden={!maskStatus}
                  aria-labelledby={maskStatus ? "video-mask-status" : undefined}
                />
              )}
            </div>
            <figcaption id="video-filename" className={styles.fileName}>
              {showingAnnotated ? "Annotated analysis video" : file?.name}
            </figcaption>
          </figure>
          {annotatedVideoUnavailableReason && (
            <p className={styles.maskHint} role="status" aria-live="polite">
              {annotatedVideoUnavailableReason}
            </p>
          )}
          {showingAnnotated && (
            <p className={styles.maskHint} role="status">
              Track ellipses and shuttle marker are baked into this clip.
              {hasMasks && showMask
                ? " Player mask overlay still draws on top while you scrub."
                : ""}
            </p>
          )}
          <p
            id="video-mask-status"
            className={styles.maskStatus}
            role="status"
            aria-live="polite"
          >
            {maskStatus}
          </p>
          {samStatus === "loading" && (
            <p className={styles.maskHintLoading} role="status" aria-live="polite">
              MobileSAM is loading in the backend (first health check or analyze may
              take a moment).
            </p>
          )}
          {samStatus === "error" && (
            <p className={styles.maskHint} role="status" aria-live="polite">
              {mobileSamHealth?.unavailable_reason ??
                "MobileSAM failed to load — install timm and mobile_sam in the backend venv (see backend/weights/README.md), restart the API, then re-run Analyze."}
            </p>
          )}
          {maskUnavailable && (
            <p className={styles.maskHint} role="status" aria-live="polite">
              {masksUnavailableReason ??
                "Player mask overlay unavailable for this run — check player lock and re-run Analyze."}
            </p>
          )}
          {hasMasks && (
            <label className={styles.maskToggle}>
              <input
                type="checkbox"
                checked={showMask}
                onChange={(e) => setShowMask(e.target.checked)}
              />
              Show player mask
            </label>
          )}
          {hasShuttleSamples && !showingAnnotated && (
            <label className={styles.maskToggle}>
              <input
                type="checkbox"
                checked={showShuttle}
                onChange={(e) => setShowShuttle(e.target.checked)}
              />
              Show shuttle highlight
            </label>
          )}
          <button
            type="button"
            className={styles.changeButton}
            onClick={openFilePicker}
            aria-describedby={describedBy || undefined}
          >
            Change video
          </button>
        </div>
      ) : (
        <button
          type="button"
          className={`${styles.panel} ${isDragging ? styles.panelDragging : ""}`}
          onClick={openFilePicker}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragEnter={handleDragOver}
          onDragLeave={handleDragLeave}
          aria-describedby={describedBy || undefined}
        >
          <span className={styles.dropIcon} aria-hidden="true">
            <UploadCloudIcon className={styles.dropIconSvg} />
          </span>
          <span className={styles.dropTitle}>Drop your match video here</span>
          <span className={styles.dropSubtitle}>
            or click to browse from your device
          </span>
          <span className={styles.formatRow} aria-hidden="true">
            {VIDEO_FORMATS.map((fmt) => (
              <span key={fmt} className={styles.formatPill}>
                {fmt}
              </span>
            ))}
          </span>
          <span id={UPLOAD_HINT_ID} className="srOnly">
            MP4 or MOV only
          </span>
        </button>
      )}

      {error && (
        <p id="video-error" className={styles.error} role="alert">
          {error}
        </p>
      )}
    </section>
  );
}
