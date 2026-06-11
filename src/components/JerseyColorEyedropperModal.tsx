"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { HexColor } from "@/lib/colorPick";
import { sampleColorFromCanvas } from "@/lib/colorPick";
import {
  extractVideoFrameAtTime,
  getVideoDuration,
  suggestedCalibrationSeekSeconds,
} from "@/lib/videoFrame";
import styles from "./JerseyColorEyedropperModal.module.css";

type JerseyColorEyedropperModalProps = {
  open: boolean;
  videoFile: File;
  targetLabel: string;
  onClose: () => void;
  onPick: (color: HexColor) => void;
};

type Point = { x: number; y: number };

function getFocusableElements(root: HTMLElement): HTMLElement[] {
  return Array.from(
    root.querySelectorAll<HTMLElement>(
      'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), canvas[tabindex="0"]',
    ),
  );
}

export function JerseyColorEyedropperModal({
  open,
  videoFile,
  targetLabel,
  onClose,
  onPick,
}: JerseyColorEyedropperModalProps) {
  const dialogRef = useRef<HTMLDivElement>(null);
  const closeBtnRef = useRef<HTMLButtonElement>(null);
  const previouslyFocusedRef = useRef<HTMLElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const loadFrameIdRef = useRef(0);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [seekSeconds, setSeekSeconds] = useState(4);
  const [videoDuration, setVideoDuration] = useState(0);
  const [pickedColor, setPickedColor] = useState<HexColor | null>(null);
  const [marker, setMarker] = useState<Point | null>(null);
  const [frameSize, setFrameSize] = useState({ width: 0, height: 0 });

  const drawMarker = useCallback(
    (ctx: CanvasRenderingContext2D, pt: Point) => {
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, 10, 0, Math.PI * 2);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(pt.x - 14, pt.y);
      ctx.lineTo(pt.x + 14, pt.y);
      ctx.moveTo(pt.x, pt.y - 14);
      ctx.lineTo(pt.x, pt.y + 14);
      ctx.stroke();
    },
    [],
  );

  const redraw = useCallback(
    (markerOverride?: Point | null) => {
      const canvas = canvasRef.current;
      const img = imageRef.current;
      if (!canvas || !img || !img.complete) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      const pt = markerOverride === undefined ? marker : markerOverride;
      if (pt) drawMarker(ctx, pt);
    },
    [marker, drawMarker],
  );

  const loadFrame = useCallback(
    async (seek: number) => {
      const loadId = ++loadFrameIdRef.current;
      setLoading(true);
      setError(null);
      setPickedColor(null);
      setMarker(null);
      try {
        const data = await extractVideoFrameAtTime(
          videoFile,
          seek,
          "eyedropper",
        );
        if (loadId !== loadFrameIdRef.current) return;
        const img = new Image();
        await new Promise<void>((resolve, reject) => {
          img.onload = () => resolve();
          img.onerror = () => reject(new Error("Could not decode frame"));
          img.src = `data:image/jpeg;base64,${data.image_jpeg_base64}`;
        });
        if (loadId !== loadFrameIdRef.current) return;
        imageRef.current = img;
        setFrameSize({ width: data.width, height: data.height });
        const canvas = canvasRef.current;
        if (canvas) {
          canvas.width = data.width;
          canvas.height = data.height;
        }
        redraw();
      } catch (err) {
        if (loadId !== loadFrameIdRef.current) return;
        setError(
          err instanceof Error ? err.message : "Could not load video frame",
        );
      } finally {
        if (loadId === loadFrameIdRef.current) {
          setLoading(false);
        }
      }
    },
    [videoFile, redraw],
  );

  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    void (async () => {
      try {
        const duration = await getVideoDuration(videoFile);
        if (cancelled) return;
        setVideoDuration(duration);
        const seek = suggestedCalibrationSeekSeconds(duration);
        setSeekSeconds(seek);
        await loadFrame(seek);
      } catch (err) {
        if (!cancelled) {
          setError(
            err instanceof Error ? err.message : "Could not read video",
          );
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [open, videoFile, loadFrame]);

  useEffect(() => {
    redraw();
  }, [redraw, frameSize]);

  useEffect(() => {
    if (!open) return;
    previouslyFocusedRef.current =
      document.activeElement instanceof HTMLElement
        ? document.activeElement
        : null;
    const main = document.querySelector("main");
    const header = document.querySelector("header");
    main?.setAttribute("inert", "");
    header?.setAttribute("inert", "");

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        onClose();
        return;
      }
      if (e.key !== "Tab" || !dialogRef.current) return;
      const focusable = getFocusableElements(dialogRef.current);
      if (focusable.length === 0) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
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
      previouslyFocusedRef.current?.focus();
    };
  }, [open, onClose]);

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

  const pickAt = useCallback(
    (pt: Point) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      const hex = sampleColorFromCanvas(
        ctx,
        pt.x,
        pt.y,
        canvas.width,
        canvas.height,
      );
      setMarker(pt);
      setPickedColor(hex);
      redraw(pt);
    },
    [redraw],
  );

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const pt = clientToCanvas(e.clientX, e.clientY);
    if (!pt) return;
    pickAt(pt);
  };

  const handleSeekChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const next = Number(e.target.value);
    setSeekSeconds(next);
    void loadFrame(next);
  };

  const handleConfirm = () => {
    if (!pickedColor) return;
    onPick(pickedColor);
    onClose();
  };

  if (!open) return null;

  const maxSeek = Math.max(0, videoDuration - 0.05);

  return (
    <div
      className={styles.backdrop}
      role="presentation"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div
        ref={dialogRef}
        className={styles.dialog}
        role="dialog"
        aria-modal="true"
        aria-labelledby="jersey-eyedropper-title"
      >
        <div className={styles.header}>
          <h2 id="jersey-eyedropper-title">Pick {targetLabel}</h2>
          <button
            ref={closeBtnRef}
            type="button"
            className={styles.closeBtn}
            onClick={onClose}
          >
            Close
          </button>
        </div>
        <p className={styles.hint}>
          Scrub to a clear view of the player&apos;s shirt, then click the
          jersey to sample its color from the video.
        </p>

        <div className={styles.framePicker}>
          <span className={styles.frameLabel}>Frame position</span>
          <input
            type="range"
            className={styles.frameSlider}
            min={0}
            max={maxSeek}
            step={0.1}
            value={Math.min(seekSeconds, maxSeek)}
            onChange={handleSeekChange}
            disabled={loading || videoDuration <= 0}
            aria-label="Video seek position"
          />
          <span className={styles.frameTime}>
            {seekSeconds.toFixed(1)}s / {videoDuration.toFixed(1)}s
          </span>
        </div>

        {loading && <p className={styles.status}>Loading frame…</p>}
        {error && (
          <p className={styles.error} role="alert">
            {error}
          </p>
        )}

        {!loading && !error && frameSize.width > 0 && (
          <div className={styles.canvasWrap}>
            <canvas
              ref={canvasRef}
              className={styles.canvas}
              tabIndex={0}
              aria-label="Video frame — click to pick shirt color"
              onClick={handleCanvasClick}
            />
          </div>
        )}

        {pickedColor && (
          <div className={styles.preview}>
            <span
              className={styles.previewSwatch}
              style={{ backgroundColor: pickedColor }}
              aria-hidden="true"
            />
            <p className={styles.previewText}>
              Sampled color: <strong>{pickedColor}</strong>
            </p>
          </div>
        )}

        <div className={styles.footer}>
          <button type="button" onClick={onClose}>
            Cancel
          </button>
          <button
            type="button"
            className={styles.primary}
            disabled={!pickedColor}
            onClick={handleConfirm}
          >
            Use this color
          </button>
        </div>
      </div>
    </div>
  );
}
