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
- `POST /api/pitch/calibration` — save four corners + regenerate template
- `POST /api/analyze/default` — analyze demo video without upload

Install demo video: `python backend/scripts/install_default_video.py`
- `POST /api/analyze` — includes `heatmap` (PNG base64 overlay) when a target is locked and calibration exists

Calibration is resolved as `{video_stem}.json` first, then `PITCH_CALIBRATION_NAME` (default `testmatch2`).

## Python usage

```python
from backend.app.pipeline.pitch_homography import load_calibration

cal = load_calibration("backend/data/pitch_calibration/testmatch2.json")
x_m, y_m = cal.pixel_to_meters(640, 360)  # bbox foot point → meters
```

Corner order: **top_left, top_right, bottom_right, bottom_left** (image coordinates, y-down).

For narrow broadcast angles you do not need a perfect on-screen rectangle — click the best visible boundary points and extrapolate off-frame corners if needed. The tool no longer requires a convex quadrilateral; only near-collinear points are rejected.

Pitch model: FIFA **105 m × 68 m** rectangle; origin at top-left corner in meter space.
