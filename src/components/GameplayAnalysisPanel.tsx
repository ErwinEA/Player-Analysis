import { parseInsightBlocks } from "@/lib/parseInsightSummary";
import type { GameplayAnalysis } from "@/types/playerMetrics";
import styles from "./GameplayAnalysisPanel.module.css";
import { LightbulbIcon, SparkleIcon } from "./icons";

type GameplayAnalysisPanelProps = {
  analysis: GameplayAnalysis | null;
  isLoading: boolean;
  isInsightsLoading?: boolean;
  errorMessage?: string | null;
  unavailableReason?: string | null;
  hasResult: boolean;
};

export function GameplayAnalysisPanel({
  analysis,
  isLoading,
  isInsightsLoading = false,
  errorMessage = null,
  unavailableReason = null,
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

      {isInsightsLoading && (
        <p className={styles.placeholder} aria-live="polite">
          Generating AI insights…
        </p>
      )}

      {!isLoading && !isInsightsLoading && !errorMessage && analysis?.summary && (
        <div className={styles.content} aria-live="polite" aria-atomic="true">
          <div className={styles.summaryBlocks}>
            {parseInsightBlocks(analysis.summary).map((block) => (
              <p
                key={block}
                className={
                  block.startsWith("Caveat:")
                    ? styles.summaryCaveat
                    : styles.summaryBlock
                }
              >
                {block}
              </p>
            ))}
          </div>
        </div>
      )}

      {!isLoading &&
        !isInsightsLoading &&
        !errorMessage &&
        !analysis?.summary &&
        unavailableReason && (
          <p className={styles.placeholder} role="status">
            {unavailableReason}
          </p>
        )}

      {!isLoading &&
        !isInsightsLoading &&
        !errorMessage &&
        !analysis?.summary &&
        !unavailableReason && (
        <div className={styles.emptyState}>
          <span className={styles.emptyIcon} aria-hidden="true">
            <LightbulbIcon className={styles.emptyIconSvg} />
          </span>
          <p className={styles.placeholder}>
            {hasResult
              ? "No AI insights were generated for this run."
              : "Run an analysis to receive an AI-generated breakdown of the player's performance in the video."}
          </p>
        </div>
      )}
    </section>
  );
}
