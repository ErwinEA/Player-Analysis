# Badminton Mode — Fix Checklist (v2)

Execution checklist with file paths. Ordered **P0 → P4**. Check boxes as you ship.

**Assumptions:** No local model training; TrackNetV2 out of scope.

---

## Priority 0 — Label / trust (no ML)

> Do first. Stops the UI and LLM from claiming things the pipeline does not compute.

### P0.1 — Rename or remove misleading metric cards

| Task | Files | Notes |
|------|-------|-------|
| [ ] Remove or relabel **"In / out calls"** (actually `points_in` / `points_out` = won/lost) | `src/components/PlayerMetricsPanel.tsx` (`BADMINTON_METRIC_CONFIG`, `kind: "inOut"`, labels ~L99–100, render ~L379–410) | Prefer **remove** until real line-call logic exists; or rename e.g. **"Points won / lost (estimated)"** |
| [ ] Remove or relabel **"Points on serve vs return"** (fake alternating serve rotation) | `src/components/PlayerMetricsPanel.tsx` (`kind: "serveReturn"`, ~L93–96, ~L350–377) | Prefer **remove**; backend still computes in `badminton_stats.py` — OK to keep API fields, hide UI |
| [ ] Add banner on rally block: **"Heuristic estimates — not official scoring"** | `src/components/PlayerMetricsPanel.tsx` (header or `metricsWarning` area) | Applies if any of `total_rallies`, durations, points remain visible |
| [ ] Map API fields in frontend (if labels change only) | `src/lib/badmintonMetrics.ts`, `src/types/playerMetrics.ts`, `src/types/analysis.ts` | `points_in` / `points_out` naming is fine in JSON; fix **display** only |

**Verify:** `npm run lint`; update tests that assert old labels:
- `tests/e2e-badminton-analyze.spec.ts` (~L75)
- `tests/dashboard.spec.ts` (~L64–66)

### P0.2 — Fix LLM facts packet (same lies as UI)

| Task | Files | Notes |
|------|-------|-------|
| [ ] Relabel or omit serve/return and in/out in facts text | `backend/app/pipeline/insights.py` (`_format_badminton_stats`, ~L115–139) | Mirror UI honesty |
| [ ] Add heuristic caveats to badminton system prompt | `backend/app/pipeline/insights.py` (badminton prompt block, ~L226+) | e.g. rally count may include replays; no line calls |
| [ ] Pass warnings into facts packet if not already | `src/lib/insightsPayload.ts` (`collectBadmintonWarnings`), `backend/app/pipeline/insights.py` (`build_facts_packet`) | |

**Verify:** `backend/tests/test_insights.py` (`test_build_facts_packet_badminton_*`, ~L149+)

### P0.3 — Fix court dimension copy (singles vs doubles)

| Task | Files | Notes |
|------|-------|-------|
| [ ] Change pro tip **17 m × 8.5 m** → **13.4 m × 6.1 m (singles)** | `src/components/Sidebar.tsx` (~L370) | Must match backend template |
| [ ] Confirm template dimensions documented consistently | `src/lib/courtConfig.ts`, `backend/app/pipeline/sports/config.py` | Already 13.4 × 6.1 — no change unless adding doubles later |

**Verify:** Grep repo for `17 m` / `8.5` — should be gone or marked doubles-future.

---

## Priority 1 — Wiring / copy (cheapest real wins)

### P1.1 — Show movement stats in badminton metrics panel

| Task | Files | Notes |
|------|-------|-------|
| [ ] Add cards: distance covered, max/avg speed (and sprints if desired) | `src/components/PlayerMetricsPanel.tsx` | Reuse `metrics` prop + `formatFootballValue` patterns or badminton-specific formatters |
| [ ] Ensure Dashboard passes `metrics` for badminton (already does) | `src/components/Dashboard.tsx` (`metricsFromMovement`, ~L64–72, ~L148, `PlayerMetricsPanel` ~L555–561) | `movement` only fills when `units === "meters"` |

**Backend (read-only):** `backend/app/pipeline/run.py` (`compute_movement_stats_from_rows`, ~L1567–1572), `backend/app/pipeline/movement_stats.py`

**Verify:** Analyze badminton clip with calibration → distance card not `—`.

### P1.2 — Show zone summary in badminton UI

| Task | Files | Notes |
|------|-------|-------|
| [ ] Pass `heatmap?.zone_summary` into metrics or heatmap panel | `src/components/Dashboard.tsx`, `src/components/PlayerMetricsPanel.tsx` and/or `src/components/HeatMapPanel.tsx` | Zone data is **not** in `movement` — it's on `AnalyzeResponse.heatmap` |
| [ ] Types | `src/types/analysis.ts` (`HeatmapResult`, `zone_summary`) | |

**Backend:** `backend/app/pipeline/heatmap.py` (`build_zone_summary`, `zone_summary` on result)

### P1.3 — Improve `rally_detection_pending` UX

| Task | Files | Notes |
|------|-------|-------|
| [ ] Replace generic warning copy | `src/lib/badmintonMetrics.ts` (`badmintonMetricsWarning`, ~L40–41) | Use: *"No rallies detected — check shuttle visibility, calibration, and clip type (highlights often fail)."* |
| [ ] Per-metric empty captions on rally cards when `badminton_stats_unavailable_reason === "rally_detection_pending"` | `src/components/PlayerMetricsPanel.tsx`, `src/components/Dashboard.tsx` (pass `reason` or derive from result) | Stat cards currently show `—` with no context |

**Backend reason enum:** `backend/app/pipeline/badminton_stats.py` (~L76), surfaced on `AnalyzeResponse.badminton_stats_unavailable_reason` in `backend/app/schemas.py`

### P1.4 — Rally END overlay label

| Task | Files | Notes |
|------|-------|-------|
| [ ] Add **"Rally ended"** text near flash or replace flash | `backend/app/pipeline/frame_overlay.py` (`draw_overlay_badminton`, `draw_rally_end_flash`, ~L168–210) | Only when `render_video=true` |
| [ ] Optional: document in weights/README | `backend/weights/README.md` | |

**Verify:** `backend/tests/test_frame_overlay.py`

---

## Priority 2 — Shuttle filters (highest ROI algorithm, no training)

> Do **court mask + jump filter** before any adaptive Kalman work.

### P2.1 — Court-bounds gate for shuttle detections

| Task | Files | Notes |
|------|-------|-------|
| [ ] Reject detections outside court polygon / metre bounds (+ margin) | `backend/app/pipeline/badminton/shuttle_detector.py` (`ShuttleDetector.detect`) **or** filter in `backend/app/pipeline/run.py` before Kalman/rally | Use `PitchCalibration.pixel_to_meters` + `is_on_pitch()` |
| [ ] Helpers | `backend/app/pipeline/pitch_homography.py` (`is_on_pitch`, ~L179+; `pixel_to_meters`) | Skip filter when no calibration (fallback: image center half or no filter + warn) |
| [ ] Wire calibration into shuttle path | `backend/app/pipeline/run.py` (shuttle block ~L1235–1291) | `calibration` already in scope |

**Precedent:** `backend/app/pipeline/ball_events/ball_detector.py` (`BALL_MAX_SPEED_M_S`, outlier rejection)

**Verify:** New tests in `backend/tests/test_shuttle_detector.py`; manual clip with ceiling lights.

### P2.2 — Velocity / jump plausibility filter

| Task | Files | Notes |
|------|-------|-------|
| [ ] Reject detection if displacement vs last accepted > max px/frame (or m/s if calibrated) | `backend/app/pipeline/badminton/shuttle_detector.py` or small helper module e.g. `backend/app/pipeline/badminton/shuttle_filter.py` | Stateful: needs last good position + frame index |
| [ ] Env tunables | Document in `backend/README.md`, `backend/weights/README.md` | e.g. `SHUTTLE_MAX_SPEED_PX` |
| [ ] Apply **before** `ShuttleKalman.update` in run loop | `backend/app/pipeline/run.py` (~L1243–1263) | |

**Verify:** `backend/tests/test_shuttle_detector.py` or `backend/tests/test_rally_tracker.py` with injected positions

### P2.3 — [2.5] Adaptive Kalman (after P2.1 + P2.2)

| Task | Files | Notes |
|------|-------|-------|
| [ ] Velocity-dependent process noise / coast cap | `backend/app/pipeline/badminton/shuttle_detector.py` (`ShuttleKalman`, ~L80–136) | Mainly improves **overlay** smoothness |
| [ ] Keep rally path on raw-first policy | `backend/app/pipeline/run.py` (`SHUTTLE_RALLY_KALMAN_GAP`, ~L632–635, ~L1274–1288) | Don't let adaptive Kalman undermine rally raw preference |

---

## Priority 3 — Structural logic (revised diagnoses)

### P3.1 — Min-gap merge for fast exchanges

| Task | Files | Notes |
|------|-------|-------|
| [ ] If new LIVE within N ms of previous END, merge rallies (one `RallyEvent`) | `backend/app/pipeline/badminton/rally_tracker.py` (`RallyTracker`, `_emit_rally`, state transitions ~L164–194) | Env: e.g. `SHUTTLE_RALLY_MERGE_MS` (1–2 s) |
| [ ] Don't confuse with existing `post_land_cooldown_frames` | `rally_tracker.py` (~L94–96, ~L164–169) | Cooldown blocks IDLE→LIVE only |

**Verify:** `backend/tests/test_rally_tracker.py` — add fast-exchange merge test

### P3.2 — `resolve_target_bbox`: court-side + primary-only color before ReID

| Task | Files | Notes |
|------|-------|-------|
| [ ] Allow color fallback with **primary only** (secondary optional) | `backend/app/pipeline/segmentation.py` (`resolve_target_bbox`, ~L697–711; `_resolve_by_kit_color`) | Badminton often has empty `secondaryJerseyColor` |
| [ ] Add court-side filter for badminton re-acquire | `backend/app/pipeline/segmentation.py` + `backend/app/pipeline/sports/badminton_utils.py` (`row_on_court_side`, `net_line_y_px`) | Pass `court_side`, `net_y_px` from `run.py` |
| [ ] Reorder: locked ID → court-side → color → last bbox → ReID tiebreaker | `segmentation.py` (`resolve_target_bbox`, ~L661+) | Mirror football `has_target_identification` idea |
| [ ] Pass badminton context from pipeline | `backend/app/pipeline/run.py` (`resolve_target_bbox` calls ~L945+, ~L970+) | `details.courtSide`, `is_badminton` |

**Reference:** `backend/app/pipeline/ball_events/identification.py` (`has_target_identification`)

**Verify:** `backend/tests/test_resolve_target_bbox.py` (add badminton cases)

### P3.3 — HSV + temporal color sampling

| Task | Files | Notes |
|------|-------|-------|
| [ ] HSV scoring + optional temporal median over frames | `backend/app/pipeline/color.py` (`jersey_color_score`, `hex_pixel_match_score`, ~L27–58) | Keep football RGB path behind flag or migrate both |
| [ ] Accumulate color before lock; stop competing tracks after lock | `backend/app/pipeline/matcher.py`, `backend/app/pipeline/run.py` (badminton lock block ~L880–924) | Pair with P3.2 |

**Verify:** `backend/tests/test_matcher_classifier_vote.py` or new `test_color_hsv.py`

### P3.4 — Court-side lock: single visible player

| Task | Files | Notes |
|------|-------|-------|
| [ ] When `len(mature_tracks) == 1`, allow lock if on chosen half + color threshold | `backend/app/pipeline/sports/badminton_utils.py` (`disambiguate_by_court_side`, ~L33–61; `disambiguate_with_fallback`, ~L82–100) | Today `min_tracks=2` hard-fails |
| [ ] New method hint e.g. `court_side_single` + UI warning | `backend/app/pipeline/matcher.py` (`apply_court_side_lock`), `src/lib/badmintonMetrics.ts` | Lower confidence than two-player disambiguation |
| [ ] Optional: baseline-Y weighting among candidates | `badminton_utils.py` | Active player often deeper on court |

**Verify:** `backend/tests/test_badminton_utils.py`

### P3.5 — Net line override (medium UI + API)

| Task | Files | Notes |
|------|-------|-------|
| [ ] Schema: `net_line_y_override` on saved calibration | `backend/app/pipeline/pitch_homography.py` (`PitchCalibration`, save/load), `backend/app/schemas.py` (pitch APIs) |
| [ ] `net_line_y_px()` reads override first | `backend/app/pipeline/sports/badminton_utils.py` (~L20–25) | |
| [ ] Draggable horizontal line on calibration preview | `src/components/PitchCalibrationModal.tsx`, `src/lib/pitchCalibration.ts`, `src/lib/api.ts` | |
| [ ] API save/preview accepts override | `backend/app/pitch_api.py`, `backend/app/main.py` (`/api/pitch/calibration/*`) | |

**Verify:** `backend/tests/test_pitch_homography.py`, `backend/tests/test_pitch_calibration_api.py`

### P3.6 — Rally duration outlier clamp

| Task | Files | Notes |
|------|-------|-------|
| [ ] Exclude or flag rallies &lt; 1 s or &gt; 2 min from avg/longest | `backend/app/pipeline/badminton_stats.py` (`build_badminton_stats`, `_rally_duration_s`) | Env-tunable thresholds |

**Verify:** `backend/tests/test_badminton_stats.py`

### P3.7 — Scene-cut interrupted merge (**DEFER**)

| Task | Files | Notes |
|------|-------|-------|
| [ ] **Do not ship** until P4.4 `replay_suspect` exists | `backend/app/pipeline/run.py` (scene cut ~L686–728), `rally_tracker.py` | Mark `interrupted` instead of finalize; merge only with replay guard |

---

## Priority 4 — Larger / optional

### P4.1 — SAM default-off for badminton + bbox fallbacks

| Task | Files | Notes |
|------|-------|-------|
| [ ] Skip segmenter / default `SAM_ENABLED=0` on badminton path | `backend/app/pipeline/run.py` (`is_badminton`, segmenter init), `backend/app/pipeline/segmentation.py` | Foot proxy already: `movement_stats.foot_position_pixels` |
| [ ] Bbox ellipse when no `mask_rle` in overlay | `backend/app/pipeline/frame_overlay.py`, frontend `src/lib/maskRle.ts` / `UploadAnalyzePanel` | `draw_foot_ellipse` exists |
| [ ] Document | `backend/weights/README.md`, `UNIFIED_SYSTEM_GUIDE.md` | |

**Verify:** `backend/tests/test_run_pipeline_sport_gates.py`, mask-related tests

### P4.2 — BoT-SORT for badminton (document only)

| Task | Files | Notes |
|------|-------|-------|
| [ ] Confirm `TRACKER_TYPE=botsort` in dev docs for badminton | `scripts/dev_backend.sh` (~L12), `backend/README.md`, `backend/weights/README.md` | Already implemented in `backend/app/pipeline/tracker.py` |

### P4.3 — Higher YOLO `imgsz` (experiment)

| Task | Files | Notes |
|------|-------|-------|
| [ ] Env toggle e.g. `YOLO_IMGSZ` / badminton-only override | `backend/app/pipeline/detector.py`, `backend/app/pipeline/badminton/shuttle_detector.py` | Measure latency on M2 before defaulting |

### P4.4 — Discard OOB-heavy epochs for heatmap

| Task | Files | Notes |
|------|-------|-------|
| [ ] Skip epochs where &gt;X% foot samples are off court | `backend/app/pipeline/run.py` (epoch / `best_epoch` logic ~L660–665, ~L1463+), `backend/app/pipeline/heatmap.py` | Pattern from football ball events multi-epoch |
| [ ] Badminton court check | `backend/app/pipeline/sports/badminton_utils.py` (`filter_positions_to_court_half`) | |

**Verify:** `backend/tests/test_heatmap.py`

### P4.5 — Replay / highlight detection

| Task | Files | Notes |
|------|-------|-------|
| [ ] Flag epochs `replay_suspect` when scene cuts burst in short window | `backend/app/pipeline/scene_cut.py`, `backend/app/pipeline/run.py` | Exclude from rally stats, not from whole analyze |
| [ ] Exclude suspect epochs in `build_badminton_stats` | `backend/app/pipeline/badminton_stats.py` | Pass filtered `rally_events` |

**Then:** unblock P3.7 scene-cut merge with guard

### P4.6 — Async overlay render

| Task | Files | Notes |
|------|-------|-------|
| [ ] Return stats immediately; queue MP4 job | `backend/app/main.py` (`POST /api/analyze`, `_finalize_analyze_response`, ~L92+), new job store module |
| [ ] Poll endpoint e.g. `GET /api/analyze/rendered/{job_id}` | `backend/app/main.py`, `backend/app/video_store.py` | |
| [ ] Frontend polling / status UI | `src/lib/api.ts`, `src/components/UploadAnalyzePanel.tsx`, `src/components/Dashboard.tsx` | `NEXT_PUBLIC_RENDER_ANALYZE_VIDEO` |

### P4.7 — Ollama per-field confidence (if not done in P0.2)

| Task | Files | Notes |
|------|-------|-------|
| [ ] Annotate each stat line with confidence in facts packet | `backend/app/pipeline/insights.py` | |

### P4.8 — Short E2E fixtures

| Task | Files | Notes |
|------|-------|-------|
| [ ] Scene cut mid-rally fixture | `tests/e2e-badminton-*.spec.ts`, `testing video/` (add short clips) | Assert warnings / no crash, not exact rally count |
| [ ] Similar-kit / color ambiguity fixture | same | |
| [ ] Update label assertions after P0 | `tests/e2e-badminton-analyze.spec.ts`, `tests/dashboard.spec.ts` | |

### P4.9 — Weights onboarding (rescoped from v1)

| Task | Files | Notes |
|------|-------|-------|
| [ ] Script: check presence + print fetch steps (not full auto-download) | `backend/scripts/check_weights.sh` (new), `backend/weights/README.md` | Custom weights: `jersey_number_b0.pt`, `osnet_x1_0_soccernet.pth`, `yolov8m_shuttlecock.pt` |
| [ ] Existing MobileSAM installer | `backend/scripts/install_mobile_sam.sh` | |

---

## Suggested sprint order (single developer)

1. **P0** — labels, insights, sidebar copy, test label updates  
2. **P1** — movement + zone wiring, rally-empty UX, overlay label  
3. **P2.1 + P2.2** — court mask + jump filter + tests  
4. **P3.2 + P3.4** — re-acquire order + single-track court-side  
5. **P3.1 + P3.6** — rally merge + duration clamp  
6. **P3.3** — HSV color  
7. **P4.1** — SAM off badminton  
8. **P3.5** — net line override (when calibration UX bandwidth)  
9. **P4.4 → P4.5 → P3.7** — replay suspect, then scene-cut merge  
10. **P4.6+** — async render, imgsz experiment  

---

## Quick verify commands

```bash
# Backend
source backend/.venv/bin/activate
python -m pytest backend/tests -q

# Frontend
npm run lint
npm run build

# E2E (needs backend + test video)
npm run test:e2e:badminton
```

---

## File index (all touched areas)

| Area | Primary files |
|------|----------------|
| UI metrics / labels | `src/components/PlayerMetricsPanel.tsx`, `src/lib/badmintonMetrics.ts`, `src/components/Sidebar.tsx` |
| Dashboard wiring | `src/components/Dashboard.tsx`, `src/components/HeatMapPanel.tsx` |
| Calibration UI | `src/components/PitchCalibrationModal.tsx`, `src/lib/pitchCalibration.ts`, `src/lib/courtConfig.ts` |
| API client | `src/lib/api.ts`, `src/lib/insightsPayload.ts` |
| Pipeline orchestration | `backend/app/pipeline/run.py` |
| Rally FSM | `backend/app/pipeline/badminton/rally_tracker.py` |
| Shuttle detect/filter | `backend/app/pipeline/badminton/shuttle_detector.py` |
| Player lock / color | `backend/app/pipeline/sports/badminton_utils.py`, `backend/app/pipeline/matcher.py`, `backend/app/pipeline/color.py` |
| Re-acquire / SAM | `backend/app/pipeline/segmentation.py` |
| Stats | `backend/app/pipeline/badminton_stats.py` |
| Homography | `backend/app/pipeline/pitch_homography.py`, `backend/app/pitch_api.py` |
| Overlay video | `backend/app/pipeline/frame_overlay.py`, `backend/app/main.py`, `backend/app/video_store.py` |
| Insights / LLM | `backend/app/pipeline/insights.py` |
| Schemas | `backend/app/schemas.py` |
| Tests | `backend/tests/test_rally_tracker.py`, `test_badminton_stats.py`, `test_badminton_utils.py`, `test_shuttle_detector.py`, `test_insights.py`, `test_frame_overlay.py`, `tests/e2e-badminton-analyze.spec.ts` |

---

*Generated from Badminton Fix Plan v2 (codebase-validated). Last updated: 2026-06-15.*
