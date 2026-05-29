"use client";

import { useCallback } from "react";
import styles from "./Sidebar.module.css";

export type HexColor = `#${string}`;

export type PlayerDetails = {
  name: string;
  jerseyNumber: number;
  primaryJerseyColor: HexColor | "";
  secondaryJerseyColor: HexColor | "";
  teamName: string;
};

type SidebarProps = {
  details: PlayerDetails;
  onChange: (details: PlayerDetails) => void;
};

const JERSEY_HINT_ID = "player-jersey-hint";

export function Sidebar({ details, onChange }: SidebarProps) {
  type StringField = "name" | "teamName";

  const update = useCallback(
    (field: StringField, value: string) => {
      onChange({ ...details, [field]: value });
    },
    [details, onChange],
  );

  const handleJerseyNumberChange = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const digitsOnly = event.target.value.replace(/\D/g, "");
      if (digitsOnly === "") {
        onChange({ ...details, jerseyNumber: 0 });
        return;
      }
      const parsed = Math.min(99, Number(digitsOnly));
      onChange({ ...details, jerseyNumber: parsed });
    },
    [details, onChange],
  );

  const jerseyInvalid =
    details.jerseyNumber <= 0 || details.jerseyNumber > 99;

  return (
    <aside className={styles.sidebar} aria-label="Player details">
      <div className={styles.card}>
        <label className={styles.fieldLabel} htmlFor="player-jersey">
          Jersey number
          <span className={styles.requiredMark} aria-hidden="true">
            {" "}
            (required)
          </span>
        </label>
        <input
          id="player-jersey"
          className={styles.field}
          type="text"
          inputMode="numeric"
          pattern="[0-9]*"
          value={details.jerseyNumber === 0 ? "" : details.jerseyNumber}
          onChange={handleJerseyNumberChange}
          placeholder="jersey no."
          aria-required="true"
          aria-invalid={jerseyInvalid}
          aria-describedby={JERSEY_HINT_ID}
        />
        <p id={JERSEY_HINT_ID} className={styles.fieldHint}>
          Used to find the player in the video. Name and kit colors are optional
          helpers when the number is hard to read.
        </p>

        <label className={styles.fieldLabel} htmlFor="player-name">
          Name <span className={styles.optionalMark}>(optional)</span>
        </label>
        <input
          id="player-name"
          className={styles.field}
          type="text"
          value={details.name}
          onChange={(e) => update("name", e.target.value)}
          placeholder="name"
        />

        <p className={styles.sectionHint}>Kit colors (optional)</p>
        <ColorField
          label="primary jersey color"
          value={details.primaryJerseyColor}
          onChange={(value) =>
            onChange({ ...details, primaryJerseyColor: value })
          }
        />
        <ColorField
          label="secondary jersey color"
          value={details.secondaryJerseyColor}
          onChange={(value) =>
            onChange({ ...details, secondaryJerseyColor: value })
          }
        />

        <label className={styles.fieldLabel} htmlFor="player-team">
          Team name <span className={styles.optionalMark}>(optional)</span>
        </label>
        <input
          id="player-team"
          className={styles.field}
          type="text"
          value={details.teamName}
          onChange={(e) => update("teamName", e.target.value)}
          placeholder="team name"
        />
      </div>
    </aside>
  );
}

type ColorFieldProps = {
  label: string;
  value: HexColor | "";
  onChange: (value: HexColor | "") => void;
};

function ColorField({ label, value, onChange }: ColorFieldProps) {
  const hasColor = value !== "";
  const swatchStyle: React.CSSProperties = hasColor
    ? { backgroundColor: value }
    : { backgroundColor: "#e5e5e5" };
  const pickerValue = hasColor ? value : "#808080";

  return (
    <div className={styles.colorField}>
      <span className={styles.colorLabel} id={`${label}-label`}>
        {label}
      </span>
      <span className={styles.colorHex}>
        {hasColor ? value.toLowerCase() : "not set"}
      </span>
      <span className={styles.colorSwatch} style={swatchStyle} aria-hidden />
      <input
        type="color"
        className={styles.hiddenColorInput}
        value={pickerValue}
        onChange={(e) => onChange(e.target.value as HexColor)}
        aria-labelledby={`${label}-label`}
      />
      {hasColor && (
        <button
          type="button"
          className={styles.clearColorButton}
          onClick={() => onChange("")}
        >
          Clear
        </button>
      )}
    </div>
  );
}
