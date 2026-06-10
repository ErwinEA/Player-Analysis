"use client";

import { useEffect, useRef, useState } from "react";
import { hasPitchCalibration } from "@/lib/api";
import { calibrationKeyFromFilename } from "@/lib/videoFrame";
import type { MobileSamHealth } from "@/lib/api";
import type { Row, ShuttleSample } from "@/types/analysis";
import type { Sport } from "@/types/sport";
import { VideoUploadPanel } from "./VideoUploadPanel";
import styles from "./UploadAnalyzePanel.module.css";

type UploadAnalyzePanelProps = {
  sport: Sport;
  file: File | null;
  onFileChange: (file: File | null) => void;
  onCalibrate: () => void;
  calibrationReady: boolean;
  onCalibrationStatusChange: (ready: boolean) => void;
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

export function UploadAnalyzePanel({
  sport,
  file,
  onFileChange,
  onCalibrate,
  calibrationReady,
  onCalibrationStatusChange,
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
}: UploadAnalyzePanelProps) {
  const [checkingCalibration, setCheckingCalibration] = useState(false);
  const calibrationCheckRef = useRef(0);

  const calibrationKey = file ? calibrationKeyFromFilename(file.name) : null;

  useEffect(() => {
    if (!calibrationKey) {
      onCalibrationStatusChange(false);
      return;
    }
    const checkId = ++calibrationCheckRef.current;
    setCheckingCalibration(true);
    void (async () => {
      try {
        const exists = await hasPitchCalibration(calibrationKey);
        if (checkId !== calibrationCheckRef.current) return;
        onCalibrationStatusChange(exists);
      } catch {
        if (checkId !== calibrationCheckRef.current) return;
        onCalibrationStatusChange(false);
      } finally {
        if (checkId === calibrationCheckRef.current) {
          setCheckingCalibration(false);
        }
      }
    })();
  }, [calibrationKey, onCalibrationStatusChange]);

  const handleFileChange = (next: File | null) => {
    onFileChange(next);
    onCalibrationStatusChange(false);
  };

  return (
    <div className={styles.wrapper}>
      <VideoUploadPanel
        file={file}
        onFileChange={handleFileChange}
        trackingRows={trackingRows}
        videoFps={videoFps}
        analyzeComplete={analyzeComplete}
        playerLabel={playerLabel}
        masksUnavailableReason={masksUnavailableReason}
        mobileSamHealth={mobileSamHealth}
        annotatedVideoUrl={annotatedVideoUrl}
        annotatedVideoUnavailableReason={annotatedVideoUnavailableReason}
        maskFrameOffset={maskFrameOffset}
        shuttleSamples={shuttleSamples}
      />
      {file && (
        <section
          className={styles.calibrationPrompt}
          aria-labelledby="upload-calibration-heading"
        >
          <h2 id="upload-calibration-heading" className={styles.promptTitle}>
            {sport === "badminton"
              ? "Court layout for heat map"
              : "Pitch layout for heat map"}
          </h2>
          {checkingCalibration ? (
            <p
              id="upload-calibration-status"
              className={styles.promptText}
              role="status"
            >
              Checking calibration…
            </p>
          ) : calibrationReady ? (
            <p
              id="upload-calibration-status"
              className={styles.promptText}
              role="status"
            >
              {sport === "badminton" ? "Court" : "Pitch"} corners saved for{" "}
              <code>{calibrationKey}</code>. Re-run analysis to refresh the heat
              map, or recalibrate if the camera angle changed.
            </p>
          ) : (
            <p
              id="upload-calibration-status"
              className={styles.promptText}
              role="status"
              aria-live="polite"
            >
              Before analyzing, open calibration and mark points around the{" "}
              {sport === "badminton" ? "court" : "pitch"} outline on a frame
              with a clear wide view of the{" "}
              {sport === "badminton" ? "court" : "field"}. Validate before
              save to check homography quality.
            </p>
          )}
          <button
            type="button"
            className={styles.calibrateButton}
            onClick={onCalibrate}
            aria-describedby="upload-calibration-status"
          >
            {calibrationReady
              ? sport === "badminton"
                ? "Recalibrate court"
                : "Recalibrate pitch"
              : sport === "badminton"
                ? "Calibrate court layout"
                : "Calibrate pitch layout"}
          </button>
        </section>
      )}
    </div>
  );
}
