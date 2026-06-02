"use client";

import { useCallback, useEffect, useState } from "react";
import { hasPitchCalibration } from "@/lib/api";
import { calibrationKeyFromFilename } from "@/lib/videoFrame";
import type { MobileSamHealth } from "@/lib/api";
import type { Row } from "@/types/analysis";
import { VideoUploadPanel } from "./VideoUploadPanel";
import styles from "./UploadAnalyzePanel.module.css";

type UploadAnalyzePanelProps = {
  file: File | null;
  onFileChange: (file: File | null) => void;
  onCalibrate: () => void;
  calibrationReady: boolean;
  onCalibrationStatusChange: (ready: boolean) => void;
  trackingRows?: Row[] | null;
  videoFps?: number;
  analyzeComplete?: boolean;
  jerseyNumber?: number;
  masksUnavailableReason?: string | null;
  mobileSamHealth?: MobileSamHealth | null;
};

export function UploadAnalyzePanel({
  file,
  onFileChange,
  onCalibrate,
  calibrationReady,
  onCalibrationStatusChange,
  trackingRows = null,
  videoFps = 30,
  analyzeComplete = false,
  jerseyNumber = 0,
  masksUnavailableReason = null,
  mobileSamHealth = null,
}: UploadAnalyzePanelProps) {
  const [checkingCalibration, setCheckingCalibration] = useState(false);

  const calibrationKey = file ? calibrationKeyFromFilename(file.name) : null;

  const checkCalibration = useCallback(async () => {
    if (!calibrationKey) {
      onCalibrationStatusChange(false);
      return;
    }
    setCheckingCalibration(true);
    try {
      const exists = await hasPitchCalibration(calibrationKey);
      onCalibrationStatusChange(exists);
    } catch {
      onCalibrationStatusChange(false);
    } finally {
      setCheckingCalibration(false);
    }
  }, [calibrationKey, onCalibrationStatusChange]);

  useEffect(() => {
    void checkCalibration();
  }, [checkCalibration]);

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
        jerseyNumber={jerseyNumber}
        masksUnavailableReason={masksUnavailableReason}
        mobileSamHealth={mobileSamHealth}
      />
      {file && (
        <section
          className={styles.calibrationPrompt}
          aria-labelledby="upload-calibration-heading"
        >
          <h2 id="upload-calibration-heading" className={styles.promptTitle}>
            Pitch layout for heat map
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
              Pitch corners saved for <code>{calibrationKey}</code>. Re-run
              analysis to refresh the heat map, or recalibrate if the camera
              angle changed.
            </p>
          ) : (
            <p
              id="upload-calibration-status"
              className={styles.promptText}
              role="status"
              aria-live="polite"
            >
              Before analyzing, open calibration and mark points around the
              pitch outline on a frame with a clear wide view of the field.
              Validate before save to check homography quality.
            </p>
          )}
          <button
            type="button"
            className={styles.calibrateButton}
            onClick={onCalibrate}
            aria-describedby="upload-calibration-status"
          >
            {calibrationReady ? "Recalibrate pitch" : "Calibrate pitch layout"}
          </button>
        </section>
      )}
    </div>
  );
}
