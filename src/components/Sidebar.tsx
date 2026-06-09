"use client";

import { useCallback } from "react";
import styles from "./Sidebar.module.css";
import { InfoIcon, UserIcon } from "./icons";

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
const JERSEY_ERROR_ID = "player-jersey-error";

const KIT_PALETTE: { hex: HexColor; name: string }[] = [
  { hex: "#dc2626", name: "Red" },
  { hex: "#2563eb", name: "Blue" },
  { hex: "#059669", name: "Green" },
  { hex: "#eab308", name: "Gold" },
  { hex: "#1f2937", name: "Black" },
  { hex: "#ffffff", name: "White" },
  { hex: "#9333ea", name: "Purple" },
  { hex: "#ea580c", name: "Orange" },
];

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

  const jerseyInvalid = details.jerseyNumber <= 0 || details.jerseyNumber > 99;
  const jerseyErrorMessage = jerseyInvalid
    ? details.jerseyNumber <= 0
      ? "Enter a jersey number from 1 to 99."
      : "Jersey number must be 99 or less."
    : null;

  return (
    <aside className={styles.sidebar} aria-label="Player configuration">
      <section className={styles.card}>
        <header className={styles.cardHeader}>
          <span className={styles.cardHeaderIcon} aria-hidden="true">
            <UserIcon className={styles.cardHeaderIconSvg} />
          </span>
          <h2 className={styles.cardTitle}>Player Configuration</h2>
        </header>

        <div className={styles.field}>
          <label className={styles.fieldLabel} htmlFor="player-jersey">
            Jersey number
            <span className={styles.requiredMark} aria-hidden="true">
              {" "}
              *
            </span>
          </label>
          <div className={styles.inputWrap}>
            <input
              id="player-jersey"
              className={styles.input}
              type="text"
              inputMode="numeric"
              pattern="[0-9]*"
              value={details.jerseyNumber === 0 ? "" : details.jerseyNumber}
              onChange={handleJerseyNumberChange}
              placeholder="e.g. 10"
              aria-required="true"
              aria-invalid={jerseyInvalid}
              aria-describedby={
                jerseyInvalid
                  ? `${JERSEY_HINT_ID} ${JERSEY_ERROR_ID}`
                  : JERSEY_HINT_ID
              }
              aria-errormessage={jerseyInvalid ? JERSEY_ERROR_ID : undefined}
            />
            <span className={styles.inputSuffix} aria-hidden="true">
              #
            </span>
          </div>
          <p id={JERSEY_HINT_ID} className={styles.fieldHint}>
            Used to identify the player in video. Name and kit colors are
            optional helpers when the number is hard to read.
          </p>
          {jerseyInvalid && jerseyErrorMessage && (
            <p id={JERSEY_ERROR_ID} className={styles.fieldError} role="alert">
              {jerseyErrorMessage}
            </p>
          )}
        </div>

        <div className={styles.field}>
          <label className={styles.fieldLabel} htmlFor="player-name">
            Player name <span className={styles.optionalMark}>(optional)</span>
          </label>
          <div className={styles.inputWrap}>
            <input
              id="player-name"
              className={styles.input}
              type="text"
              value={details.name}
              onChange={(e) => update("name", e.target.value)}
              placeholder="e.g. Marcus Rashford"
            />
          </div>
        </div>

        <div className={styles.field}>
          <p className={styles.fieldLabel}>Kit colors</p>
          <ColorPalette
            label="Primary jersey color"
            value={details.primaryJerseyColor}
            onChange={(value) =>
              onChange({ ...details, primaryJerseyColor: value })
            }
          />
          <ColorPalette
            label="Secondary / trim color"
            value={details.secondaryJerseyColor}
            onChange={(value) =>
              onChange({ ...details, secondaryJerseyColor: value })
            }
          />
        </div>

        <div className={styles.field}>
          <label className={styles.fieldLabel} htmlFor="player-team">
            Team name <span className={styles.optionalMark}>(optional)</span>
          </label>
          <div className={styles.inputWrap}>
            <input
              id="player-team"
              className={styles.input}
              type="text"
              value={details.teamName}
              onChange={(e) => update("teamName", e.target.value)}
              placeholder="e.g. Manchester United"
            />
          </div>
        </div>
      </section>

      <aside className={styles.tip} aria-label="Pro tip">
        <span className={styles.tipIcon} aria-hidden="true">
          <InfoIcon className={styles.tipIconSvg} />
        </span>
        <div className={styles.tipBody}>
          <p className={styles.tipTitle}>Pro Tip</p>
          <p className={styles.tipText}>
            For best results, use video with clear jersey numbers and consistent
            lighting. 1080p or higher recommended.
          </p>
        </div>
      </aside>
    </aside>
  );
}

type ColorPaletteProps = {
  label: string;
  value: HexColor | "";
  onChange: (value: HexColor | "") => void;
};

function ColorPalette({ label, value, onChange }: ColorPaletteProps) {
  const groupLabelId = `${label.replace(/\W+/g, "-")}-label`;
  return (
    <div className={styles.colorBlock}>
      <span className={styles.colorLabel} id={groupLabelId}>
        {label}
      </span>
      <div
        className={styles.swatchRow}
        role="group"
        aria-labelledby={groupLabelId}
      >
        {KIT_PALETTE.map(({ hex, name }) => {
          const selected = value.toLowerCase() === hex.toLowerCase();
          return (
            <button
              key={hex}
              type="button"
              className={`${styles.swatch} ${selected ? styles.swatchSelected : ""}`}
              style={{ backgroundColor: hex }}
              aria-pressed={selected}
              aria-label={`${name}${selected ? " (selected)" : ""}`}
              onClick={() => onChange(selected ? "" : hex)}
            />
          );
        })}
      </div>
    </div>
  );
}
