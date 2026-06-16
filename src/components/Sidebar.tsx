"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { useRadioGroupKeyboard } from "@/hooks/useRadioGroupKeyboard";
import type { CourtSide, Sport } from "@/types/sport";
import styles from "./Sidebar.module.css";
import { EyedropperIcon, InfoIcon, UserIcon } from "./icons";
import { JerseyColorEyedropperModal } from "./JerseyColorEyedropperModal";

export type HexColor = `#${string}`;

export type PlayerDetails = {
  sport: Sport;
  name: string;
  jerseyNumber: number;
  courtSide: CourtSide | "";
  primaryJerseyColor: HexColor | "";
  secondaryJerseyColor: HexColor | "";
  teamName: string;
};

type SidebarProps = {
  sport: Sport;
  details: PlayerDetails;
  onChange: (details: PlayerDetails) => void;
  videoFile?: File | null;
};

type EyedropperTarget = "primary" | "secondary" | null;

const JERSEY_HINT_ID = "player-jersey-hint";
const JERSEY_ERROR_ID = "player-jersey-error";
const COURT_SIDE_HINT_ID = "player-court-side-hint";
const COURT_SIDE_ERROR_ID = "player-court-side-error";
const SHIRT_COLOR_HINT_ID = "player-shirt-color-hint";
const SHIRT_COLOR_ERROR_ID = "player-shirt-color-error";

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

const COURT_SIDE_OPTIONS: { value: CourtSide; label: string }[] = [
  { value: "near", label: "Near court" },
  { value: "far", label: "Far court" },
];

const COURT_SIDE_VALUES = COURT_SIDE_OPTIONS.map((o) => o.value);
const COLOR_VALUES = KIT_PALETTE.map((o) => o.hex);

function isLightSwatch(hex: HexColor): boolean {
  return hex.toLowerCase() === "#ffffff";
}

export function Sidebar({
  sport,
  details,
  onChange,
  videoFile = null,
}: SidebarProps) {
  const [jerseyTouched, setJerseyTouched] = useState(false);
  const [courtSideTouched, setCourtSideTouched] = useState(false);
  const [shirtColorTouched, setShirtColorTouched] = useState(false);
  const [eyedropperTarget, setEyedropperTarget] =
    useState<EyedropperTarget>(null);

  useEffect(() => {
    setJerseyTouched(false);
    setCourtSideTouched(false);
    setShirtColorTouched(false);
  }, [sport]);

  const { onKeyDown: onCourtSideKeyDown, getTabIndex: getCourtSideTabIndex, setButtonRef: setCourtSideRef } =
    useRadioGroupKeyboard(COURT_SIDE_VALUES, details.courtSide, (value) => {
      setCourtSideTouched(true);
      onChange({ ...details, courtSide: value });
    });

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
  const showJerseyError = jerseyTouched && jerseyInvalid;

  const courtSideInvalid = !details.courtSide;
  const shirtColorInvalid = !details.primaryJerseyColor;
  const showCourtSideError = courtSideTouched && courtSideInvalid;
  const showShirtColorError = shirtColorTouched && shirtColorInvalid;

  const handleEyedropperPick = useCallback(
    (color: HexColor) => {
      if (eyedropperTarget === "primary") {
        setShirtColorTouched(true);
        onChange({ ...details, primaryJerseyColor: color });
      } else if (eyedropperTarget === "secondary") {
        onChange({ ...details, secondaryJerseyColor: color });
      }
      setEyedropperTarget(null);
    },
    [details, eyedropperTarget, onChange],
  );

  const eyedropperLabel =
    eyedropperTarget === "secondary"
      ? "secondary shirt color"
      : "primary shirt color";

  return (
    <aside className={styles.sidebar} aria-label="Player configuration">
      <section className={styles.card}>
        <header className={styles.cardHeader}>
          <span className={styles.cardHeaderIcon} aria-hidden="true">
            <UserIcon className={styles.cardHeaderIconSvg} />
          </span>
          <h2 className={styles.cardTitle}>
            {sport === "badminton" ? "Player Selection" : "Player Configuration"}
          </h2>
        </header>

        {sport === "football" ? (
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
                onBlur={() => setJerseyTouched(true)}
                placeholder="e.g. 10"
                aria-required="true"
                aria-invalid={showJerseyError}
                aria-describedby={
                  showJerseyError
                    ? `${JERSEY_HINT_ID} ${JERSEY_ERROR_ID}`
                    : JERSEY_HINT_ID
                }
                aria-errormessage={showJerseyError ? JERSEY_ERROR_ID : undefined}
              />
              <span className={styles.inputSuffix} aria-hidden="true">
                #
              </span>
            </div>
            <p id={JERSEY_HINT_ID} className={styles.fieldHint}>
              Used to identify the player in video. Name and kit colors are
              optional helpers when the number is hard to read.
            </p>
            {showJerseyError && jerseyErrorMessage && (
              <p id={JERSEY_ERROR_ID} className={styles.fieldError} role="alert">
                {jerseyErrorMessage}
              </p>
            )}
          </div>
        ) : (
          <>
            <div className={styles.field}>
              <p className={styles.fieldLabel} id="player-court-side-label">
                Court side
                <span className={styles.requiredMark} aria-hidden="true">
                  {" "}
                  *
                </span>
              </p>
              <div
                className={styles.courtSideRow}
                role="radiogroup"
                aria-labelledby="player-court-side-label"
                aria-required="true"
                aria-invalid={showCourtSideError}
                aria-describedby={
                  showCourtSideError
                    ? `${COURT_SIDE_HINT_ID} ${COURT_SIDE_ERROR_ID}`
                    : COURT_SIDE_HINT_ID
                }
                onKeyDown={onCourtSideKeyDown}
              >
                {COURT_SIDE_OPTIONS.map(({ value, label }, index) => {
                  const selected = details.courtSide === value;
                  return (
                    <button
                      key={value}
                      ref={setCourtSideRef(index)}
                      type="button"
                      role="radio"
                      aria-checked={selected}
                      tabIndex={getCourtSideTabIndex(value)}
                      className={`${styles.courtSideOption} ${selected ? styles.courtSideSelected : ""}`}
                      onClick={() => {
                        setCourtSideTouched(true);
                        onChange({ ...details, courtSide: value });
                      }}
                    >
                      {label}
                    </button>
                  );
                })}
              </div>
              <p id={COURT_SIDE_HINT_ID} className={styles.fieldHint}>
                Pick the side of the court where your player starts. Helps
                disambiguate when both players wear similar kits.
              </p>
              {showCourtSideError && (
                <p id={COURT_SIDE_ERROR_ID} className={styles.fieldError} role="alert">
                  Select near or far court.
                </p>
              )}
            </div>

            <fieldset className={`${styles.field} ${styles.fieldset}`}>
              <legend className={styles.fieldsetLegend}>
                Shirt color
                <span className={styles.requiredMark} aria-hidden="true">
                  {" "}
                  *
                </span>
              </legend>
              <ColorPalette
                label="Primary shirt color"
                value={details.primaryJerseyColor}
                onChange={(value) => {
                  setShirtColorTouched(true);
                  onChange({ ...details, primaryJerseyColor: value });
                }}
                mode="radio"
                required
                invalid={showShirtColorError}
                describedBy={
                  showShirtColorError
                    ? `${SHIRT_COLOR_HINT_ID} ${SHIRT_COLOR_ERROR_ID}`
                    : SHIRT_COLOR_HINT_ID
                }
                videoFile={videoFile}
                onEyedropper={() => setEyedropperTarget("primary")}
              />
              <ColorPalette
                label="Secondary / trim color"
                value={details.secondaryJerseyColor}
                onChange={(value) =>
                  onChange({ ...details, secondaryJerseyColor: value })
                }
                mode="toggle"
                videoFile={videoFile}
                onEyedropper={() => setEyedropperTarget("secondary")}
              />
              <p id={SHIRT_COLOR_HINT_ID} className={styles.fieldHint}>
                Shirt color is the main way we lock onto your player in
                badminton footage. Use the eyedropper to sample the real shirt
                color from your uploaded video.
              </p>
              {showShirtColorError && (
                <p id={SHIRT_COLOR_ERROR_ID} className={styles.fieldError} role="alert">
                  Pick a primary shirt color.
                </p>
              )}
            </fieldset>
          </>
        )}

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
              placeholder={
                sport === "badminton" ? "e.g. Lee Zii Jia" : "e.g. Marcus Rashford"
              }
            />
          </div>
        </div>

        {sport === "football" && (
          <div className={styles.field}>
            <p className={styles.fieldLabel}>Kit colors</p>
            <ColorPalette
              label="Primary jersey color"
              value={details.primaryJerseyColor}
              onChange={(value) =>
                onChange({ ...details, primaryJerseyColor: value })
              }
              mode="toggle"
              videoFile={videoFile}
              onEyedropper={() => setEyedropperTarget("primary")}
            />
            <ColorPalette
              label="Secondary / trim color"
              value={details.secondaryJerseyColor}
              onChange={(value) =>
                onChange({ ...details, secondaryJerseyColor: value })
              }
              mode="toggle"
              videoFile={videoFile}
              onEyedropper={() => setEyedropperTarget("secondary")}
            />
          </div>
        )}

        <div className={styles.field}>
          <label className={styles.fieldLabel} htmlFor="player-team">
            {sport === "badminton" ? "Club / country" : "Team name"}{" "}
            <span className={styles.optionalMark}>(optional)</span>
          </label>
          <div className={styles.inputWrap}>
            <input
              id="player-team"
              className={styles.input}
              type="text"
              value={details.teamName}
              onChange={(e) => update("teamName", e.target.value)}
              placeholder={
                sport === "badminton" ? "e.g. Malaysia" : "e.g. Manchester United"
              }
            />
          </div>
        </div>
      </section>

      <section className={styles.tip} aria-label="Pro tip">
        <span className={styles.tipIcon} aria-hidden="true">
          <InfoIcon className={styles.tipIconSvg} />
        </span>
        <div className={styles.tipBody}>
          <p className={styles.tipTitle}>Pro Tip</p>
          <p className={styles.tipText}>
            {sport === "badminton"
              ? "Use a fixed wide-angle view of the full singles court (13.4 m × 6.1 m). Calibrate court lines before analyzing for metre-based movement stats."
              : "For best results, use video with clear jersey numbers and consistent lighting. 1080p or higher recommended."}
          </p>
        </div>
      </section>

      {videoFile && eyedropperTarget && (
        <JerseyColorEyedropperModal
          open
          videoFile={videoFile}
          targetLabel={eyedropperLabel}
          onClose={() => setEyedropperTarget(null)}
          onPick={handleEyedropperPick}
        />
      )}
    </aside>
  );
}

type ColorPaletteProps = {
  label: string;
  value: HexColor | "";
  onChange: (value: HexColor | "") => void;
  mode?: "radio" | "toggle";
  required?: boolean;
  invalid?: boolean;
  describedBy?: string;
  videoFile?: File | null;
  onEyedropper?: () => void;
};

function ColorPalette({
  label,
  value,
  onChange,
  mode = "toggle",
  required = false,
  invalid = false,
  describedBy,
  videoFile = null,
  onEyedropper,
}: ColorPaletteProps) {
  const groupLabelId = `${label.replace(/\W+/g, "-")}-label`;
  const isRadio = mode === "radio";
  const paletteHexes = useMemo(
    () => new Set(KIT_PALETTE.map((entry) => entry.hex.toLowerCase())),
    [],
  );
  const isCustomColor = Boolean(
    value && !paletteHexes.has(value.toLowerCase()),
  );
  const { onKeyDown, getTabIndex, setButtonRef } = useRadioGroupKeyboard(
    COLOR_VALUES,
    value,
    (next) => onChange(next),
  );

  const handleSelect = (hex: HexColor, selected: boolean) => {
    if (isRadio) {
      if (!selected) onChange(hex);
      return;
    }
    onChange(selected ? "" : hex);
  };

  const eyedropperDisabled = !videoFile;

  return (
    <div className={styles.colorBlock}>
      <span className={styles.colorLabel} id={groupLabelId}>
        {label}
        {isCustomColor && value ? (
          <span className={styles.optionalMark}> ({value})</span>
        ) : null}
      </span>
      <div
        className={styles.swatchRow}
        role={isRadio ? "radiogroup" : "group"}
        aria-labelledby={groupLabelId}
        aria-describedby={describedBy}
        aria-required={required || undefined}
        aria-invalid={invalid || undefined}
        onKeyDown={isRadio ? onKeyDown : undefined}
      >
        {KIT_PALETTE.map(({ hex, name }, index) => {
          const selected = value.toLowerCase() === hex.toLowerCase();
          return (
            <button
              key={hex}
              ref={isRadio ? setButtonRef(index) : undefined}
              type="button"
              role={isRadio ? "radio" : undefined}
              aria-checked={isRadio ? selected : undefined}
              aria-pressed={!isRadio ? selected : undefined}
              tabIndex={isRadio ? getTabIndex(hex) : undefined}
              className={`${styles.swatch} ${selected ? styles.swatchSelected : ""} ${isLightSwatch(hex) ? styles.swatchLight : ""}`}
              style={{ backgroundColor: hex }}
              aria-label={`${name}${selected ? " (selected)" : ""}`}
              onClick={() => handleSelect(hex, selected)}
            />
          );
        })}
        {isCustomColor && value ? (
          <button
            type="button"
            role={isRadio ? "radio" : undefined}
            aria-checked={isRadio ? true : undefined}
            aria-pressed={!isRadio ? true : undefined}
            className={`${styles.swatch} ${styles.swatchSelected} ${styles.customSwatch} ${isLightSwatch(value) ? styles.swatchLight : ""}`}
            style={{ backgroundColor: value }}
            aria-label={`Custom sampled color ${value} (selected)`}
            onClick={() => onChange(value)}
          />
        ) : null}
        <button
          type="button"
          className={styles.eyedropperBtn}
          disabled={eyedropperDisabled}
          aria-label={
            eyedropperDisabled
              ? "Upload a video to sample shirt color"
              : `Sample ${label.toLowerCase()} from video`
          }
          title={
            eyedropperDisabled
              ? "Upload a video first"
              : "Pick color from video"
          }
          onClick={() => onEyedropper?.()}
        >
          <EyedropperIcon className={styles.eyedropperIcon} />
        </button>
      </div>
    </div>
  );
}
