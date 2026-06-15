"use client";

import { useRadioGroupKeyboard } from "@/hooks/useRadioGroupKeyboard";
import type { Sport } from "@/types/sport";
import { SPORT_LABELS } from "@/types/sport";
import styles from "./SportToggle.module.css";

type SportToggleProps = {
  sport: Sport;
  onChange: (sport: Sport) => void;
  disabled?: boolean;
};

const OPTIONS: Sport[] = ["football", "badminton"];

export function SportToggle({ sport, onChange, disabled = false }: SportToggleProps) {
  const { onKeyDown, getTabIndex, setButtonRef } = useRadioGroupKeyboard(
    OPTIONS,
    sport,
    onChange,
  );

  return (
    <div
      className={styles.group}
      role="radiogroup"
      aria-label="Sport"
      onKeyDown={onKeyDown}
    >
      {OPTIONS.map((option, index) => {
        const selected = sport === option;
        return (
          <button
            key={option}
            ref={setButtonRef(index)}
            type="button"
            role="radio"
            aria-checked={selected}
            tabIndex={getTabIndex(option)}
            className={`${styles.option} ${selected ? styles.optionSelected : ""}`}
            onClick={() => onChange(option)}
            disabled={disabled}
          >
            {SPORT_LABELS[option]}
          </button>
        );
      })}
    </div>
  );
}
