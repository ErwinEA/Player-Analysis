import type { GameplayAnalysis } from "@/types/playerMetrics";
import styles from "./GameplayAnalysisPanel.module.css";
import { LightbulbIcon, SparkleIcon } from "./icons";

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
        <div className={styles.headerText}>
          <h2 id="gameplay-analysis-heading" className={styles.heading}>
            Gameplay Analysis
          </h2>
          <p className={styles.subtitle}>
            AI-generated breakdown of player performance
          </p>
        </div>
        <span className={styles.badge}>
          <SparkleIcon className={styles.badgeIcon} />
          AI Insight
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
        <div className={styles.emptyState}>
          <span className={styles.emptyIcon} aria-hidden="true">
            <LightbulbIcon className={styles.emptyIconSvg} />
          </span>
          <p className={styles.placeholder}>
            {hasResult
              ? "Gameplay insights will appear here once the analysis engine is connected."
              : "Run an analysis to receive an AI-generated breakdown of the player's performance in the video."}
          </p>
        </div>
      )}
    </section>
  );
}
