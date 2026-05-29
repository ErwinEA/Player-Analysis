"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  fetchPitchFrame,
  saveLegacyPitchCalibration,
  savePitchCalibration,
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
  BOUNDARY_LABELS,
  BOUNDARY_POINT_COUNT,
  decagonToQuad,
  legacyCornersToBoundary,
} from "@/lib/pitchCalibration";
import styles from "./PitchCalibrationModal.module.css";

const RETICLE_STEP = 8;
const RETICLE_STEP_FAST = 24;

type PitchCalibrationModalProps = {
  open: boolean;
  calibrationName?: string;
  frameIndex?: number;
  /** Legacy upload: pick frame from this file and save calibration with the video. */
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
  calibrationName = "testmatch2",
  frameIndex = 100,
  videoFile,
  onClose,
  onSaved,
}: PitchCalibrationModalProps) {
  const legacyMode = Boolean(videoFile);
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
  const [seekSeconds, setSeekSeconds] = useState(4);
  const [videoDuration, setVideoDuration] = useState(0);
  const activeFrameIndex = legacyMode
    ? seekSecondsToFrameIndex(seekSeconds)
    : frameIndex;

  const placeCornerAt = useCallback((x: number, y: number) => {
    setPoints((prev) => {
      if (prev.length >= BOUNDARY_POINT_COUNT) return prev;
      return [...prev, { x, y }];
    });
  }, []);

  const derivedQuad = useMemo(
    () =>
      points.length === BOUNDARY_POINT_COUNT ? decagonToQuad(points) : [],
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
      const label = BOUNDARY_LABELS[i] ?? `Point ${i + 1}`;
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
    if (points.length < BOUNDARY_POINT_COUNT) {
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
  }, [points, reticle, derivedQuad]);

  const applyFrameData = useCallback((data: PitchFrameResponse) => {
    setFrameData(data);
    setReticle({ x: data.width / 2, y: data.height / 2 });
    if (data.image_boundary_points?.length === BOUNDARY_POINT_COUNT) {
      setPoints(
        data.image_boundary_points.map(([x, y]) => ({
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

    const run = async () => {
      try {
        let seek = frameIndexToSeekSeconds(frameIndex);
        if (legacyMode && videoFile) {
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
    legacyMode,
    loadFrame,
    applyFrameData,
  ]);

  const handleSeekChange = useCallback(
    (seconds: number) => {
      setSeekSeconds(seconds);
      if (!legacyMode || !videoFile) return;
      setLoading(true);
      setError(null);
      void loadFrame(seconds)
        .then(applyFrameData)
        .catch((err) => {
          setError(err instanceof Error ? err.message : "Failed to load frame");
        })
        .finally(() => setLoading(false));
    },
    [legacyMode, videoFile, loadFrame, applyFrameData],
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
    if (!open || !frameData || points.length >= BOUNDARY_POINT_COUNT) {
      setReticleAnnouncement("");
      return;
    }
    setReticleAnnouncement(
      `Crosshair at ${Math.round(reticle.x)}, ${Math.round(reticle.y)}`,
    );
  }, [open, frameData, points.length, reticle]);

  const canvasAriaLabel = useMemo(() => {
    const placed = points
      .map(
        (p, i) =>
          `${BOUNDARY_LABELS[i] ?? `Point ${i + 1}`} at ${Math.round(p.x)}, ${Math.round(p.y)}`,
      )
      .join("; ");
    if (points.length >= BOUNDARY_POINT_COUNT) {
      return `Pitch frame. All ${BOUNDARY_POINT_COUNT} boundary points placed.${placed ? ` ${placed}.` : ""}`;
    }
    const crosshair = `Crosshair at ${Math.round(reticle.x)}, ${Math.round(reticle.y)}`;
    const next = BOUNDARY_LABELS[points.length] ?? "next point";
    return (
      `Pitch boundary placement. Arrow keys move crosshair; Enter or Space places a point; Backspace removes the last. ` +
      `${crosshair}. ${points.length} of ${BOUNDARY_POINT_COUNT} placed; next: ${next}.${placed ? ` Placed: ${placed}.` : ""}`
    );
  }, [points, reticle]);

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
    const step = e.shiftKey ? RETICLE_STEP_FAST : RETICLE_STEP;

    switch (e.key) {
      case "ArrowLeft":
        e.preventDefault();
        setReticle((r) => clampReticle(r.x - step, r.y));
        break;
      case "ArrowRight":
        e.preventDefault();
        setReticle((r) => clampReticle(r.x + step, r.y));
        break;
      case "ArrowUp":
        e.preventDefault();
        setReticle((r) => clampReticle(r.x, r.y - step));
        break;
      case "ArrowDown":
        e.preventDefault();
        setReticle((r) => clampReticle(r.x, r.y + step));
        break;
      case "Enter":
      case " ":
        e.preventDefault();
        if (points.length < BOUNDARY_POINT_COUNT) {
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
    if (frameData) {
      setReticle({ x: frameData.width / 2, y: frameData.height / 2 });
    }
  };

  const handleSave = async () => {
    if (points.length !== BOUNDARY_POINT_COUNT) return;
    const boundary = points.map((p) => [p.x, p.y]);
    setSaving(true);
    setError(null);
    try {
      if (legacyMode && videoFile) {
        await saveLegacyPitchCalibration(videoFile, {
          name: calibrationName,
          frame_index: activeFrameIndex,
          image_boundary_points: boundary,
        });
      } else {
        await savePitchCalibration({
          name: calibrationName,
          frame_index: activeFrameIndex,
          image_boundary_points: boundary,
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
    points.length < BOUNDARY_POINT_COUNT
      ? BOUNDARY_LABELS[points.length]
      : null;

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
          <h2 id="pitch-calibration-title">Pitch calibration</h2>
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
          Place {BOUNDARY_POINT_COUNT} points clockwise around the visible pitch
          outline (yellow). A green quadrilateral is derived from your outline
          for the warp. Use arrow keys to move the crosshair, then Enter or
          Space to place each point. Backspace removes the last point.
          {legacyMode
            ? " Choose a wide, unobstructed view of the pitch before marking points."
            : ` Frame ${activeFrameIndex} from the server default video.`}
        </p>
        {legacyMode && videoDuration > 0 && (
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
          {points.length} of {BOUNDARY_POINT_COUNT} boundary points placed
          {nextBoundaryLabel ? ` — next: ${nextBoundaryLabel}` : ""}
          {derivedQuad.length === 4 ? " — warp quad ready" : ""}
        </p>
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
                {BOUNDARY_LABELS[i]}: {Math.round(p.x)}, {Math.round(p.y)}
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
          <button type="button" onClick={handleReset} disabled={saving}>
            Reset points
          </button>
          <button
            type="button"
            className={styles.primary}
            onClick={handleSave}
            disabled={points.length !== BOUNDARY_POINT_COUNT || saving}
          >
            {saving ? "Saving…" : "Save calibration"}
          </button>
        </footer>
      </div>
    </div>
  );
}
