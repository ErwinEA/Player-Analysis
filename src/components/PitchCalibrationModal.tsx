"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  fetchPitchFrame,
  previewPitchCalibration,
  saveLegacyPitchCalibration,
  savePitchCalibration,
  type PitchCalibrationPreviewResponse,
  type PitchFrameResponse,
} from "@/lib/api";
import {
  extractVideoFrameAtTime,
  frameIndexToSeekSeconds,
  getVideoDuration,
  seekSecondsToFrameIndex,
  suggestedCalibrationSeekSeconds,
} from "@/lib/videoFrame";
import {
  boundaryPointLabel,
  decagonToQuad,
  legacyCornersToBoundary,
  MAX_BOUNDARY_POINTS,
  MIN_BOUNDARY_POINTS,
} from "@/lib/pitchCalibration";
import { courtDimensionsForSport } from "@/lib/courtConfig";
import type { Sport } from "@/types/sport";
import styles from "./PitchCalibrationModal.module.css";

const RETICLE_STEP = 8;
const RETICLE_STEP_FAST = 24;

type PitchCalibrationModalProps = {
  open: boolean;
  sport?: Sport;
  calibrationName?: string;
  frameIndex?: number;
  /** Uploaded video: pick frame from this file and save calibration with the video. */
  videoFile?: File | null;
  onClose: () => void;
  onSaved: (name: string) => void;
};

type Point = { x: number; y: number };

function getFocusableElements(root: HTMLElement): HTMLElement[] {
  return Array.from(
    root.querySelectorAll<HTMLElement>(
      'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), canvas[tabindex="0"]',
    ),
  );
}

export function PitchCalibrationModal({
  open,
  sport = "football",
  calibrationName = "testmatch2",
  frameIndex = 100,
  videoFile,
  onClose,
  onSaved,
}: PitchCalibrationModalProps) {
  const surface = sport === "badminton" ? "court" : "pitch";
  const surfaceTitle = surface === "court" ? "Court" : "Pitch";
  const uploadMode = Boolean(videoFile);
  const dialogRef = useRef<HTMLDivElement>(null);
  const closeBtnRef = useRef<HTMLButtonElement>(null);
  const previouslyFocusedRef = useRef<HTMLElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const [frameData, setFrameData] = useState<PitchFrameResponse | null>(null);
  const [points, setPoints] = useState<Point[]>([]);
  const [reticle, setReticle] = useState<Point>({ x: 0, y: 0 });
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [reticleAnnouncement, setReticleAnnouncement] = useState("");
  const [step, setStep] = useState<"place" | "preview">("place");
  const [previewData, setPreviewData] =
    useState<PitchCalibrationPreviewResponse | null>(null);
  const [netLineY, setNetLineY] = useState<number | null>(null);
  const [netLineAdjusted, setNetLineAdjusted] = useState(false);
  const [previewing, setPreviewing] = useState(false);
  const [seekSeconds, setSeekSeconds] = useState(4);
  const [videoDuration, setVideoDuration] = useState(0);
  const activeFrameIndex = uploadMode
    ? seekSecondsToFrameIndex(seekSeconds)
    : frameIndex;

  const placeCornerAt = useCallback((x: number, y: number) => {
    if (step !== "place") return;
    setPoints((prev) => {
      if (prev.length >= MAX_BOUNDARY_POINTS) return prev;
      return [...prev, { x, y }];
    });
  }, [step]);

  const derivedQuad = useMemo(
    () =>
      points.length >= MIN_BOUNDARY_POINTS ? decagonToQuad(points) : [],
    [points],
  );

  const redraw = useCallback(() => {
    const canvas = canvasRef.current;
    const img = imageRef.current;
    if (!canvas || !img || !img.complete) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    points.forEach((p, i) => {
      ctx.fillStyle = "#ffc800";
      ctx.beginPath();
      ctx.arc(p.x, p.y, 7, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "#fff";
      ctx.font = "13px sans-serif";
      const label = boundaryPointLabel(i);
      ctx.fillText(`${i + 1}: ${label}`, p.x + 10, p.y - 8);
    });
    if (points.length >= 2) {
      ctx.strokeStyle = "#ffc800";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(points[0].x, points[0].y);
      for (let i = 1; i < points.length; i++) {
        ctx.lineTo(points[i].x, points[i].y);
      }
      if (points.length >= 3) ctx.closePath();
      ctx.stroke();
    }
    if (derivedQuad.length === 4) {
      const colors = ["#00ffff", "#ffa500", "#ff0000", "#ff00ff"];
      ctx.setLineDash([6, 4]);
      ctx.strokeStyle = "#00ff00";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(derivedQuad[0].x, derivedQuad[0].y);
      for (let i = 1; i < 4; i++) {
        ctx.lineTo(derivedQuad[i].x, derivedQuad[i].y);
      }
      ctx.closePath();
      ctx.stroke();
      ctx.setLineDash([]);
      derivedQuad.forEach((p, i) => {
        ctx.fillStyle = colors[i];
        ctx.beginPath();
        ctx.arc(p.x, p.y, 6, 0, Math.PI * 2);
        ctx.fill();
      });
    }
    if (step === "place" && points.length < MAX_BOUNDARY_POINTS) {
      ctx.strokeStyle = "#ffff00";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(reticle.x, reticle.y, 10, 0, Math.PI * 2);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(reticle.x - 14, reticle.y);
      ctx.lineTo(reticle.x + 14, reticle.y);
      ctx.moveTo(reticle.x, reticle.y - 14);
      ctx.lineTo(reticle.x, reticle.y + 14);
      ctx.stroke();
    }
  }, [points, reticle, derivedQuad, step]);

  const applyFrameData = useCallback((data: PitchFrameResponse) => {
    setFrameData(data);
    setReticle({ x: data.width / 2, y: data.height / 2 });
    const boundaryLen = data.image_boundary_points?.length ?? 0;
    if (
      boundaryLen >= MIN_BOUNDARY_POINTS &&
      boundaryLen <= MAX_BOUNDARY_POINTS
    ) {
      setPoints(
        data.image_boundary_points!.map(([x, y]) => ({
          x: Number(x),
          y: Number(y),
        })),
      );
    } else if (data.image_corners?.length === 4) {
      setPoints(
        legacyCornersToBoundary(
          data.image_corners.map(([x, y]) => ({
            x: Number(x),
            y: Number(y),
          })),
        ),
      );
    } else {
      setPoints([]);
    }
  }, []);

  const loadFrame = useCallback(
    async (seek: number) => {
      if (videoFile) {
        return extractVideoFrameAtTime(videoFile, seek, calibrationName);
      }
      return fetchPitchFrame(calibrationName, frameIndex);
    },
    [calibrationName, frameIndex, videoFile],
  );

  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    setPoints([]);
    setStep("place");
    setPreviewData(null);

    const run = async () => {
      try {
        let seek = frameIndexToSeekSeconds(frameIndex);
        if (uploadMode && videoFile) {
          const duration = await getVideoDuration(videoFile);
          if (cancelled) return;
          seek = suggestedCalibrationSeekSeconds(duration);
          setVideoDuration(duration);
          setSeekSeconds(seek);
        }
        const data = await loadFrame(seek);
        if (cancelled) return;
        applyFrameData(data);
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Failed to load frame");
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    };

    void run();
    return () => {
      cancelled = true;
    };
  }, [
    open,
    calibrationName,
    frameIndex,
    videoFile,
    uploadMode,
    loadFrame,
    applyFrameData,
  ]);

  const handleSeekChange = useCallback(
    (seconds: number) => {
      setSeekSeconds(seconds);
      if (!uploadMode || !videoFile) return;
      setLoading(true);
      setError(null);
      void loadFrame(seconds)
        .then(applyFrameData)
        .catch((err) => {
          setError(err instanceof Error ? err.message : "Failed to load frame");
        })
        .finally(() => setLoading(false));
    },
    [uploadMode, videoFile, loadFrame, applyFrameData],
  );

  useEffect(() => {
    if (!frameData) return;
    const img = new Image();
    img.onload = () => {
      imageRef.current = img;
      const canvas = canvasRef.current;
      if (canvas) {
        canvas.width = frameData.width;
        canvas.height = frameData.height;
      }
      redraw();
    };
    img.src = `data:image/jpeg;base64,${frameData.image_jpeg_base64}`;
  }, [frameData, redraw]);

  useEffect(() => {
    redraw();
  }, [points, reticle, redraw]);

  useEffect(() => {
    if (!open || !frameData || step !== "place" || points.length >= MAX_BOUNDARY_POINTS) {
      setReticleAnnouncement("");
      return;
    }
    setReticleAnnouncement(
      `Crosshair at ${Math.round(reticle.x)}, ${Math.round(reticle.y)}`,
    );
  }, [open, frameData, points.length, reticle, step]);

  const canvasAriaLabel = useMemo(() => {
    const placed = points
      .map(
        (p, i) =>
          `${boundaryPointLabel(i)} at ${Math.round(p.x)}, ${Math.round(p.y)}`,
      )
      .join("; ");
    if (step === "preview") {
      const conf = previewData
        ? `Confidence ${Math.round(previewData.confidence * 100)} percent.`
        : "";
      return `${surfaceTitle} calibration preview. ${conf}${placed ? ` Points: ${placed}.` : ""}`;
    }
    if (points.length >= MAX_BOUNDARY_POINTS) {
      return `${surfaceTitle} frame. Maximum ${MAX_BOUNDARY_POINTS} points placed.${placed ? ` ${placed}.` : ""}`;
    }
    const crosshair = `Crosshair at ${Math.round(reticle.x)}, ${Math.round(reticle.y)}`;
    const next = boundaryPointLabel(points.length);
    return (
      `${surfaceTitle} boundary placement. Arrow keys move crosshair; Enter or Space places a point; Backspace removes the last. ` +
      `${crosshair}. ${points.length} points placed (minimum ${MIN_BOUNDARY_POINTS}, maximum ${MAX_BOUNDARY_POINTS}); next: ${next}.${placed ? ` Placed: ${placed}.` : ""}`
    );
  }, [points, reticle, step, previewData, surfaceTitle]);

  useEffect(() => {
    if (!open) return;

    previouslyFocusedRef.current = document.activeElement as HTMLElement | null;

    const main = document.getElementById("main-content");
    const header = document.querySelector("header");
    const sidebar = document.querySelector("aside");
    main?.setAttribute("inert", "");
    header?.setAttribute("inert", "");
    sidebar?.setAttribute("inert", "");

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        onClose();
        return;
      }
      if (e.key !== "Tab") return;
      const dialog = dialogRef.current;
      if (!dialog) return;
      const focusables = getFocusableElements(dialog);
      if (focusables.length === 0) return;
      const first = focusables[0];
      const last = focusables[focusables.length - 1];
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    };

    document.addEventListener("keydown", handleKeyDown);
    closeBtnRef.current?.focus();

    return () => {
      document.removeEventListener("keydown", handleKeyDown);
      main?.removeAttribute("inert");
      header?.removeAttribute("inert");
      sidebar?.removeAttribute("inert");
      previouslyFocusedRef.current?.focus();
    };
  }, [open, onClose]);

  useEffect(() => {
    if (!open || loading || !frameData) return;
    canvasRef.current?.focus();
  }, [open, loading, frameData]);

  const clientToCanvas = useCallback(
    (clientX: number, clientY: number): Point | null => {
      const canvas = canvasRef.current;
      if (!canvas) return null;
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;
      return {
        x: (clientX - rect.left) * scaleX,
        y: (clientY - rect.top) * scaleY,
      };
    },
    [],
  );

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const pt = clientToCanvas(e.clientX, e.clientY);
    if (!pt) return;
    setReticle(pt);
    placeCornerAt(pt.x, pt.y);
  };

  const clampReticle = useCallback(
    (x: number, y: number): Point => {
      const w = frameData?.width ?? 0;
      const h = frameData?.height ?? 0;
      return {
        x: Math.max(0, Math.min(w, x)),
        y: Math.max(0, Math.min(h, y)),
      };
    },
    [frameData],
  );

  const handleCanvasKeyDown = (e: React.KeyboardEvent<HTMLCanvasElement>) => {
    if (!frameData) return;
    const moveStep = e.shiftKey ? RETICLE_STEP_FAST : RETICLE_STEP;

    switch (e.key) {
      case "ArrowLeft":
        e.preventDefault();
        setReticle((r) => clampReticle(r.x - moveStep, r.y));
        break;
      case "ArrowRight":
        e.preventDefault();
        setReticle((r) => clampReticle(r.x + moveStep, r.y));
        break;
      case "ArrowUp":
        e.preventDefault();
        setReticle((r) => clampReticle(r.x, r.y - moveStep));
        break;
      case "ArrowDown":
        e.preventDefault();
        setReticle((r) => clampReticle(r.x, r.y + moveStep));
        break;
      case "Enter":
      case " ":
        e.preventDefault();
        if (step === "place" && points.length < MAX_BOUNDARY_POINTS) {
          placeCornerAt(reticle.x, reticle.y);
        }
        break;
      case "Backspace":
        e.preventDefault();
        setPoints((prev) => prev.slice(0, -1));
        break;
      default:
        break;
    }
  };

  const handleReset = () => {
    setPoints([]);
    setStep("place");
    setPreviewData(null);
    if (frameData) {
      setReticle({ x: frameData.width / 2, y: frameData.height / 2 });
    }
  };

  const boundaryPayload = useMemo(
    () => points.map((p) => [p.x, p.y]),
    [points],
  );

  const courtDims = useMemo(
    () => courtDimensionsForSport(sport),
    [sport],
  );

  const calibrationCourtFields = useMemo(
    () =>
      sport === "badminton"
        ? {
            pitch_length_m: courtDims.pitch_length_m,
            pitch_width_m: courtDims.pitch_width_m,
          }
        : {},
    [sport, courtDims],
  );

  const handlePreview = async () => {
    if (points.length < MIN_BOUNDARY_POINTS) return;
    setPreviewing(true);
    setError(null);
    try {
      const previewPayload: Parameters<typeof previewPitchCalibration>[0] = {
        name: calibrationName,
        frame_index: activeFrameIndex,
        image_boundary_points: boundaryPayload,
        ...calibrationCourtFields,
      };
      if (uploadMode && frameData) {
        previewPayload.image_width = frameData.width;
        previewPayload.image_height = frameData.height;
      }
      const result = await previewPitchCalibration(previewPayload);
      setPreviewData(result);
      setNetLineAdjusted(false);
      setNetLineY(null);
      setStep("preview");
      const confPct = Math.round(result.confidence * 100);
      setReticleAnnouncement(
        `Preview ready. Confidence ${confPct} percent. Probes ${result.probe_count} of ${result.probe_total} on ${surface}.`,
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "Preview failed");
    } finally {
      setPreviewing(false);
    }
  };

  const netLineFields =
    sport === "badminton" && netLineAdjusted && netLineY != null
      ? { net_line_y_override: netLineY }
      : {};

  const handleSave = async () => {
    if (points.length < MIN_BOUNDARY_POINTS) return;
    if (step !== "preview") return;
    setSaving(true);
    setError(null);
    try {
      if (uploadMode && videoFile) {
        await saveLegacyPitchCalibration(videoFile, {
          name: calibrationName,
          frame_index: activeFrameIndex,
          image_boundary_points: boundaryPayload,
          ...calibrationCourtFields,
          ...netLineFields,
        });
      } else {
        await savePitchCalibration({
          name: calibrationName,
          frame_index: activeFrameIndex,
          image_boundary_points: boundaryPayload,
          ...calibrationCourtFields,
          ...netLineFields,
        });
      }
      onSaved(calibrationName);
      onClose();
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Save failed";
      setError(
        message === "Failed to fetch"
          ? "Could not reach the server while saving. Check that the backend is running and you have enough free disk space for the video upload."
          : message,
      );
    } finally {
      setSaving(false);
    }
  };

  if (!open) return null;

  const nextBoundaryLabel =
    step === "place" && points.length < MAX_BOUNDARY_POINTS
      ? boundaryPointLabel(points.length)
      : null;

  const canPreview =
    step === "place" &&
    points.length >= MIN_BOUNDARY_POINTS &&
    !previewing &&
    !loading &&
    frameData != null;

  return (
    <div className={styles.backdrop} role="presentation" onClick={onClose}>
      <div
        ref={dialogRef}
        className={styles.dialog}
        role="dialog"
        aria-labelledby="pitch-calibration-title"
        aria-describedby="pitch-calibration-hint pitch-calibration-progress"
        aria-modal="true"
        onClick={(e) => e.stopPropagation()}
      >
        <header className={styles.header}>
          <h2 id="pitch-calibration-title">{surfaceTitle} calibration</h2>
          <button
            ref={closeBtnRef}
            type="button"
            className={styles.closeBtn}
            onClick={onClose}
          >
            Close
          </button>
        </header>
        <p id="pitch-calibration-hint" className={styles.hint}>
          Place {MIN_BOUNDARY_POINTS}–{MAX_BOUNDARY_POINTS} points clockwise
          around the visible {surface} outline (yellow). A green quadrilateral is
          derived once you have at least {MIN_BOUNDARY_POINTS} points. Use arrow
          keys to move the crosshair, then Enter or Space to place each point.
          Backspace removes the last point.
          {uploadMode
            ? ` Choose a wide, unobstructed view of the ${surface} before marking points.`
            : ` Frame ${activeFrameIndex} from the server default video.`}
          {sport === "badminton"
            ? " Save calibration while Badminton mode is selected so metre stats use the singles court (13.4 m × 6.1 m)."
            : ""}
        </p>
        {uploadMode && videoDuration > 0 && (
          <div className={styles.framePicker}>
            <label htmlFor="calibration-frame-seek" className={styles.frameLabel}>
              Video position (frame ~{activeFrameIndex})
            </label>
            <input
              id="calibration-frame-seek"
              type="range"
              className={styles.frameSlider}
              min={0}
              max={videoDuration}
              step={0.1}
              value={seekSeconds}
              onChange={(e) => handleSeekChange(Number(e.target.value))}
              aria-valuetext={`${seekSeconds.toFixed(1)} seconds`}
            />
            <span className={styles.frameTime}>
              {seekSeconds.toFixed(1)}s / {videoDuration.toFixed(1)}s
            </span>
          </div>
        )}
        <p
          id="pitch-calibration-progress"
          className={styles.progress}
          aria-live="polite"
        >
          {points.length} point{points.length === 1 ? "" : "s"} placed (min{" "}
          {MIN_BOUNDARY_POINTS}, max {MAX_BOUNDARY_POINTS})
          {nextBoundaryLabel ? ` — next: ${nextBoundaryLabel}` : ""}
          {derivedQuad.length === 4 ? " — warp quad ready" : ""}
        </p>
        {step === "preview" && previewData && (
          <div
            className={styles.previewPanel}
            role="region"
            aria-label="Calibration preview"
          >
            <p className={styles.previewMetric}>
              Confidence:{" "}
              <strong>{Math.round(previewData.confidence * 100)}%</strong>
            </p>
            <p className={styles.previewMetric}>
              Frame probes on {surface}:{" "}
              <strong>
                {previewData.probe_count}/{previewData.probe_total}
              </strong>
            </p>
            <p className={styles.previewMetric}>
              Grid coverage: <strong>{previewData.coverage_pct}%</strong>
            </p>
            {previewData.warnings.length > 0 && (
              <ul className={styles.warningList}>
                {previewData.warnings.map((w) => (
                  <li key={w}>{w}</li>
                ))}
              </ul>
            )}
            {sport === "badminton" && frameData && (
              <div className={styles.previewMetric}>
                {netLineAdjusted && netLineY != null ? (
                  <label>
                    Net line (image Y):{" "}
                    <input
                      type="range"
                      min={0}
                      max={frameData.height}
                      value={netLineY}
                      onChange={(e) => setNetLineY(Number(e.target.value))}
                    />
                    <strong>{netLineY}px</strong>
                  </label>
                ) : (
                  <button
                    type="button"
                    onClick={() => {
                      setNetLineAdjusted(true);
                      setNetLineY(Math.round(frameData.height / 2));
                    }}
                  >
                    Adjust net line (optional)
                  </button>
                )}
              </div>
            )}
          </div>
        )}
        <p id="pitch-calibration-reticle" className="srOnly" aria-live="polite">
          {reticleAnnouncement}
        </p>
        {points.length > 0 && (
          <ol
            className="srOnly"
            aria-label="Placed boundary points"
            aria-live="polite"
          >
            {points.map((p, i) => (
              <li key={`${i}-${p.x}-${p.y}`}>
                {boundaryPointLabel(i)}: {Math.round(p.x)}, {Math.round(p.y)}
              </li>
            ))}
          </ol>
        )}
        {error && (
          <p className={styles.error} role="alert">
            {error}
          </p>
        )}
        {loading && (
          <p className={styles.status} aria-live="polite">
            Loading frame…
          </p>
        )}
        {!loading && frameData && (
          <div className={styles.canvasWrap}>
            <canvas
              ref={canvasRef}
              tabIndex={0}
              className={styles.canvas}
              role="application"
              aria-label={canvasAriaLabel}
              aria-describedby="pitch-calibration-progress"
              onClick={handleCanvasClick}
              onKeyDown={handleCanvasKeyDown}
            />
          </div>
        )}
        <footer className={styles.footer}>
          <button
            type="button"
            onClick={handleReset}
            disabled={saving || previewing}
          >
            Reset points
          </button>
          {step === "place" && (
            <button
              type="button"
              className={styles.primary}
              onClick={() => void handlePreview()}
              disabled={!canPreview}
              aria-describedby="pitch-calibration-progress pitch-calibration-hint"
            >
              {previewing ? "Validating…" : "Validate & preview"}
            </button>
          )}
          {step === "preview" && (
            <>
              <button
                type="button"
                onClick={() => {
                  setStep("place");
                  setPreviewData(null);
                }}
                disabled={saving}
              >
                Adjust points
              </button>
              <button
                type="button"
                className={styles.primary}
                onClick={() => void handleSave()}
                disabled={saving}
              >
                {saving ? "Saving…" : "Save calibration"}
              </button>
            </>
          )}
        </footer>
      </div>
    </div>
  );
}
