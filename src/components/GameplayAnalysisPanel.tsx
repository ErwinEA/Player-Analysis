import type { GameplayAnalysis } from "@/types/playerMetrics";
import styles from "./GameplayAnalysisPanel.module.css";

type GameplayAnalysisPanelProps = {
  analysis: GameplayAnalysis | null;
  isLoading: boolean;
  errorMessage?: string | null;
  hasResult: boolean;
};

export function GameplayAnalysisPanel({
  analysis,
  isLoading,
  errorMessage = null,
  hasResult,
}: GameplayAnalysisPanelProps) {
  return (
    <section
      className={styles.panel}
      aria-labelledby="gameplay-analysis-heading"
    >
      <header className={styles.header}>
        <h2 id="gameplay-analysis-heading" className={styles.heading}>
          Gameplay analysis
        </h2>
        <span className={styles.badge} aria-hidden="true">
          AI insight
        </span>
      </header>

      {errorMessage && (
        <p className={styles.error} role="alert">
          <span className="srOnly">Error: </span>
          {errorMessage}
        </p>
      )}

      {isLoading && (
        <p className={styles.placeholder} aria-live="polite">
          Reviewing player stats and generating gameplay insights…
        </p>
      )}

      {!isLoading && !errorMessage && analysis?.summary && (
        <div className={styles.content} aria-live="polite" aria-atomic="true">
          <p className={styles.summary}>{analysis.summary}</p>
        </div>
      )}

      {!isLoading && !errorMessage && !analysis?.summary && (
        <p className={styles.placeholder}>
          {hasResult
            ? "Gameplay insights will appear here once the analysis engine is connected."
            : "Run an analysis to receive an AI-generated breakdown of the player's performance in the video."}
        </p>
      )}
    </section>
  );
}
