# Pitch homography calibration

Manual pitch-corner annotation and homography for **pixel → meter** mapping. Powers movement stats and the player position heatmap on the top-down pitch diagram.

## Quick start (sample video)

```bash
# From repo root — interactive corner clicks (TL, TR, BR, BL)
python backend/tools/annotate_pitch.py \
  --video /path/to/testmatch2.mp4 \
  --frame 100 \
  --name testmatch2
```

Non-interactive (JSON corners):

```bash
python backend/tools/annotate_pitch.py \
  --video /path/to/testmatch2.mp4 \
  --frame 100 \
  --name testmatch2 \
  --corners '[[115,425],[905,495],[1265,705],[35,705]]'
```

## Outputs

Under `backend/data/pitch_calibration/`:

| File | Purpose |
|------|---------|
| `{name}.json` | Homography matrix, corners, pitch dimensions |
| `{name}_topview_template.png` | 2D pitch diagram for UI |
| `{name}_corners.jpg` | Annotated frame (verification) |
| `{name}_topview_warp.jpg` | Warped frame (verification) |

## API

- `GET /api/pitch/template?name=testmatch2` — PNG top-down pitch (template only)
- `GET /api/pitch/calibration?name=testmatch2` — calibration metadata
- `GET /api/videos/default` — stream server-hosted demo video
- `GET /api/pitch/frame?name=testmatch2&frame=100` — JPEG frame for corner UI
- `POST /api/pitch/calibration/preview` — validate outline (4–20 points) without saving; returns confidence, probes, coverage
- `POST /api/pitch/calibration` — save outline (4–20 boundary points) + regenerate template; soft validation (warnings + confidence)
- `POST /api/analyze/default` — analyze demo video without upload

Install demo video: `python backend/scripts/install_default_video.py`
- `POST /api/analyze` — includes `heatmap` (PNG base64 overlay) when a target is locked and calibration exists

Analyze and CLI resolve calibration as `PITCH_CALIBRATION_NAME` (default `testmatch2`). The UI may pass `calibration_name` after you save corners for a specific upload (tries that JSON first, then the default). Analyze does **not** auto-pick `{video_stem}.json` from the filename alone.

## Outline calibration (UI)

1. Place **4–20** points clockwise on the visible pitch outline.
2. Click **Validate & preview** — server returns confidence (0–100%), frame probes on-pitch (e.g. 3/5), and grid coverage %.
3. **Save** even when confidence is low (narrow shots); warnings explain limitations.

Hard reject (HTTP 400): fewer than 4 points, degenerate quad, or invalid geometry. Probe/coverage failures are warnings only.

JSON may include optional `mode`, `confidence`, and `coverage_pct` metadata (backward-compatible with older files).

## Python usage

```python
from backend.app.pipeline.pitch_homography import load_calibration

cal = load_calibration("backend/data/pitch_calibration/testmatch2.json")
x_m, y_m = cal.pixel_to_meters(640, 360)  # bbox foot point → meters
```

Corner order: **top_left, top_right, bottom_right, bottom_left** (image coordinates, y-down).

For narrow broadcast angles you do not need a perfect on-screen rectangle — click the best visible boundary points and extrapolate off-frame corners if needed. The tool no longer requires a convex quadrilateral; only near-collinear points are rejected.

Pitch model: FIFA **105 m × 68 m** rectangle; origin at top-left corner in meter space.
