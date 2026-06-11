# Player Analysis — Tracking Backend

Python service that runs **YOLOv8** (person detection) + **ByteTrack** (multi-object tracking) + **EfficientNet-B0 jersey classifier** (primary) with **EasyOCR** fallback for jersey numbers and names on a match video. It identifies the user's chosen player via a **3-stage fallback** and **locks** that track for the rest of the clip, then returns a single batch JSON of tracked positions.

> Note: PaddleOCR was the original choice, but PaddlePaddle has no wheels for Python 3.13/3.14 on macOS arm64. We use **EasyOCR** (PyTorch-based) for broad Python version support. Models are downloaded automatically on first run.

## Prerequisites

- Python 3.10+ (tested up to 3.14 with EasyOCR; PaddleOCR was dropped due to missing wheels)
- macOS / Linux / Windows (CPU). For NVIDIA GPU, set `OCR_GPU=1` and use a CUDA-enabled PyTorch build for Ultralytics + EasyOCR.

## Install

From the **project root** (`player analysis/`, sibling to `src/`):

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cd ..
```

On first run, Ultralytics downloads `yolov8n.pt` automatically (unless you set `YOLO_WEIGHTS` to another path).

**Model checkpoints (masks, jersey, ReID, ball):** see the [happy path weights checklist](weights/README.md#happy-path-checklist-full-post-apianalyze) in [weights/README.md](weights/README.md). Files are gitignored; copy or train them per machine.

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `YOLO_WEIGHTS` | `yolov8n.pt` | YOLOv8 weights file (`yolov8m.pt`, or a football fine-tuned `.pt`) |
| `DET_CONF` | `0.1` | YOLO confidence threshold (ByteTrack uses low-conf detections) |
| `TRACKER_TYPE` | `bytetrack` | Tracker backend: `bytetrack` or `botsort` (BoT-SORT adds camera-motion compensation) |
| `TRACK_BUFFER` | `30` (bytetrack), `120` (botsort) | Frames a lost track stays alive before its ID is dropped |
| `GMC_METHOD` | `sparseOptFlow` | BoT-SORT camera motion comp: `sparseOptFlow`/`orb`/`ecc`/`none` (`ecc` needs opencv-contrib) |
| `TRACK_LOW_THRESH` | `0.1` | ByteTrack second-stage threshold; warn if `DET_CONF` is higher |
| `TRACK_MIN_AGE` | `3` | Frames before a track is analyzed for OCR / included in output |
| `OCR_EVERY` | `5` | Run OCR every N frames (OCR is slow) |
| `MAX_FRAMES` | `6000` | Cap frames through detection/tracking (max 6000; invalid/`0` → 6000) |
| `MAX_UPLOAD_MB` | `4096` | Max legacy video upload size in megabytes (4 GB) |
| `CORS_ALLOW_LOCAL_REGEX` | `1` | Allow localhost/127.0.0.1 on any port (set `0` in production) |
| `OCR_GPU` | `0` | Set to `1` to run EasyOCR on a CUDA GPU |
| `DEBUG_OCR` | `0` | Set to `1` to save jersey crops + `ocr.log` under `DEBUG_OCR_DIR` |
| `DEBUG_OCR_DIR` | `debug_ocr` | Output folder for OCR debug (project root by default) |
| `COLOR_TOLERANCE` | `10` | +/- units per RGB channel allowed when matching pixels to the user's hex color |
| `COLOR_LOCK_THRESHOLD` | `0.15` | Mean per-frame pixel-match fraction needed to lock by color |
| `COLOR_MATCH_MIN` | `0.05` | Floor: best color track still wins if above this when nothing stronger matches |
| `NUMBER_MIN_CONF` | `0.40` | EasyOCR confidence floor to count a digit reading as a vote |
| `CLASSIFIER_VOTE_WEIGHT` | `1.0` | Scale on EfficientNet target-jersey confidence when accumulating lock votes |
| `CLASSIFIER_MIN_CONF` | `0.5` | Min classifier confidence to count a target read (and to override disagreeing OCR) |
| `NUMBER_LOCK_VOTES` | `1.5` | Cumulative number-OCR confidence required to lock by number alone |
| `LOCK_CONSECUTIVE_FRAMES` | `2` | Consecutive target-jersey OCR observations (same `track_id`, within `OCR_EVERY` spacing) before `number` / `number+name` / `reid` lock (color unchanged) |
| `NAME_MIN_CONF` | `0.35` | EasyOCR confidence floor to count a name reading as a vote |
| `NAME_MIN_RATIO` | `0.6` | Min fuzzy similarity (0-1) between OCR'd name and the user's name |
| `OCR_MIN_CROP_HEIGHT` | `64` | Upscale jersey crops shorter than this before OCR |
| `JERSEY_WEIGHTS` | `detection_test/weights/jersey_number_b0.pt` | EfficientNet jersey classifier checkpoint |
| `JERSEY_CLS_MIN_CONF` | `0.75` | Min classifier confidence for a hard number read |
| `JERSEY_TARGET_MIN_CONF` | `0.55` | Min prob on target jersey class (also requires agreement with argmax) |
| `JERSEY_SOFT_MIN_CONF` | `0.25` | Soft classifier accept before OCR fallback |
| `JERSEY_CLS_DEVICE` | `auto` | `cuda`, `mps`, `cpu`, or `auto` for classifier inference |
| `INFERENCE_DEVICE` | `auto` | `mps`, `cuda`, `cpu`, or `auto` for YOLO + ReID (overridden by `YOLO_DEVICE` for detectors) |
| `YOLO_DEVICE` | unset | Optional override for person/ball YOLO only |
| `SAM_ENABLED` | `1` | Per-frame MobileSAM on locked player (`0` = bbox-only) |
| `SAM_WEIGHTS` | `backend/weights/mobile_sam.pt` | MobileSAM checkpoint |
| `SAM_DEVICE` | `auto` | `mps`, `cuda`, `cpu`, or `auto` (no CPU fallback unless `SAM_ALLOW_CPU_FALLBACK=1`) |
| `SAM_ALLOW_CPU_FALLBACK` | unset | Set `1` to retry MobileSAM on CPU after MPS/CUDA failure |
| `SAM_WARMUP_FRAMES` | `3` | Frames on locked track after identity lock before SAM |
| `SAM_STRIDE` | `1` | Run MobileSAM every N visible frames after warmup (`3` ≈ 3× less SAM work) |
| `REID_FALLBACK_THRESH` | `0.65` | ReID match when ByteTrack loses lock `track_id` |
| `SCENE_CUT_THRESH` | `0.65` | Grayscale-histogram correlation below this marks a hard broadcast cut |
| `SCENE_CUT_SUPPRESS_FRAMES` | `45` | Frames after a cut where ball events are suppressed while the shot/lock stabilizes |
| `SCENE_CUT_CLEAR_VOTES` | `0` | On cut, also clear accumulated identity votes (set `1` if relock keeps picking the wrong track) |
| `POSSESS_REQUIRE_ON_PITCH` | `1` | Require both ball and target foot to project on the pitch before counting possession (metres mode) |
| `POSSESS_RADIUS_M` | `2.0` | Ball-to-target-foot distance (metres) counting as possession when target is visible |
| `GAP_POSSESS_RADIUS_M` | `2.5` | Sticky possession radius (metres) while the target is briefly occluded |
| `EPOCH_MIN_VISIBLE_FRAMES` | `5` | Min on-pitch visible frames for a lock epoch to contribute events / be a heatmap candidate |
| `BALL_MAX_SPEED_M_S` | `36` | Physical ceiling on ball motion; detections implying a larger frame-to-frame jump are rejected as far-field false positives instead of poisoning the Kalman velocity |
| `BALL_OUTLIER_MAX_FRAMES` | `5` | Consecutive rejected detections before the ball filter re-acquires at the new location (handles genuine long passes/relocations) |
| `BBOX_MIN_ASPECT` | `0.85` | Min height/width for a candidate to count as an upright player (rejects banners/graphics) |
| `BALL_EVENTS_SUMMARY_LOG` | unset | Optional path for the JSONL ball-event diagnostic summary (disabled when unset) |

After identity lock, the pipeline waits **3 consecutive frames** on the locked `track_id`, then runs **MobileSAM every frame** (box prompt, center-point retry). Rows include full-res `mask_rle`, normalized `foot_x`/`foot_y` for the heat map, and mask-tight OCR/ReID crops. See [backend/weights/README.md](weights/README.md) for `mobile_sam.pt` and package install.

Example:

```bash
export YOLO_WEIGHTS=yolov8n.pt
export OCR_EVERY=5
export DET_CONF=0.1
export DEBUG_OCR=1
export COLOR_TOLERANCE=10
```

## Identification stages (in order)

The pipeline checks each candidate track in priority order. The **first** stage that succeeds locks `target.track_id` for the rest of the video — no further OCR or color work runs on other tracks.

1. **`number+name`** — EasyOCR finds the user's jersey number on the back, *and* OCR'd name (letters above the number) fuzzy-matches `details.name` (similarity >= `NAME_MIN_RATIO`).
2. **`number`** — Cumulative jersey-number vote weight on a single track reaches `NUMBER_LOCK_VOTES` (classifier + OCR combined per frame when `jersey_number_b0.pt` is loaded).
3. **`color`** — At end of video, picks the track whose torso+shorts pixels most often fall within `COLOR_TOLERANCE` units of the user's primary or secondary hex color. Used only if 1 and 2 both failed.

If none of the three pass their thresholds, `target.method` is `"none"` and `player` stays `null` on all rows.

### OCR debug

With `DEBUG_OCR=1`, each OCR attempt writes:

- `debug_ocr/crops/f000005_t3_n12.jpg` — full kit crop (torso + shorts) sent for debugging
- `debug_ocr/ocr.log` — one JSON line per attempt: raw number/name reads, parsed values, color score, crop path

Inspect crops to see if numbers/names are visible; inspect `ocr.log` for what EasyOCR actually returned.

### Reading the JSON

- **`rows` are sparse** — one row per **confirmed track per frame**, not every video frame. You will usually **not** see `frame: 0` or `frame: 5` unless a player track was confirmed on that exact frame.
- **`track` in the 100s–400s is normal** — ByteTrack assigns incrementing IDs across the whole video (not jersey numbers).
- **OCR runs per track** every `OCR_EVERY` frames after that track appears (not on global frame 0/5 only).
- **`ocr_conf`** is the last successful OCR confidence for that track (carried across rows until the next OCR attempt).

## Run the API server

From the **project root** (so `backend` is importable):

```bash
source backend/.venv/bin/activate
uvicorn backend.app.main:app --reload --port 8000
```

Health check: `GET http://localhost:8000/health`

## API

### Badminton mode

Set `"sport": "badminton"` in `details` with `courtSide` (`"near"` or `"far"`) and `primaryJerseyColor`. Jersey OCR and ball events are skipped; the pipeline locks the target by **court side** (with shirt-color fallback) and returns movement stats in metres when calibration uses the singles court (**13.4 m × 6.1 m**). Save court calibration while Badminton is selected in the UI so dimensions are stored correctly. Rally metrics (rally wins, win rate, duration, winners) need `yolov8n_shuttle.pt` (see [weights/README.md](weights/README.md#shuttle-detector-yolov8-badminton)); without it they stay null with `badminton_stats_unavailable_reason: no_shuttle_weights`.

### `POST /api/analyze`

Uses the same `run_pipeline()` as `python -m backend.cli` (see [CLI](#cli-smoke-test-without-http)).

**Pitch calibration:** the dashboard sends `calibration_name` (video stem) only after court corners are saved for that upload. The CLI uses `calibration_keys_for_run()` with `PITCH_CALIBRATION_NAME` (default `testmatch2`) and optional `{upload_stem}.json` when the file exists — pass the same stem via env or align keys so HTTP and CLI runs match.

Multipart form:

- `video` — MP4 or MOV file
- `details` — JSON string matching frontend `PlayerDetails`:
- `calibration_name` — optional; safe name of saved pitch calibration (legacy flow after calibrate)

```json
{
  "name": "Player Name",
  "jerseyNumber": 10,
  "primaryJerseyColor": "#ffffff",
  "secondaryJerseyColor": "#000000",
  "teamName": "Home"
}
```

### Response

```json
{
  "video": { "fps": 30.0, "width": 1920, "height": 1080, "frames": 1500, "duration_s": 50.0 },
  "target": {
    "jersey": 10,
    "track_id": 7,
    "confidence": 1.83,
    "method": "number+name",
    "locked_at_frame": 412
  },
  "rows": [
    {
      "t": 12.34,
      "frame": 370,
      "track": 7,
      "player": 10,
      "x": 0.52,
      "y": 0.61,
      "bbox": [100, 200, 150, 400],
      "ocr_conf": 0.88
    }
  ]
}
```

- `x`, `y` — bbox center, normalized to `[0, 1]` (video width/height).
- `track` — ByteTrack track id.
- `player` — user's jersey number on rows belonging to the locked target track; `null` on every other track.
- `target.method` — which stage produced the lock: `number+name` | `number` | `color` | `none`.
- `target.locked_at_frame` — frame index where the lock was first acquired (or `null` if color fallback locked at the end).

### curl example

```bash
curl -X POST http://localhost:8000/api/analyze \
  -F "video=@/path/to/match.mp4" \
  -F 'details={"name":"Player","jerseyNumber":10,"primaryJerseyColor":"#ffffff","secondaryJerseyColor":"#000000","teamName":"Home"}'
```

CORS is enabled for `http://localhost:3000` (Next.js dev server).

## CLI (smoke test without HTTP)

From the project root:

```bash
source backend/.venv/bin/activate
python -m backend.cli /path/to/match.mp4 backend/sample-details.json --pretty
```

A sample details file is at [backend/sample-details.json](sample-details.json).

### Annotated analyze video (API)

`POST /api/analyze` accepts optional form field `render_video=true`. When set, the pipeline writes an MP4 with track ellipses, lock highlight, and ball marker to `backend/data/rendered_videos/`. The JSON response includes `video_url` (relative path under `/api/videos/rendered/…`) when the file is browser-playable.

**Requires `ffmpeg` on the server** (`brew install ffmpeg`). Without it, analyze still succeeds but `video_unavailable_reason` explains why no clip was attached.

| Variable | Default | Description |
|----------|---------|-------------|
| `RENDERED_VIDEO_MAX_AGE_HOURS` | `5` | TTL for rendered MP4s (cleaned on API startup) |

Frontend: set `NEXT_PUBLIC_RENDER_ANALYZE_VIDEO=1` in `.env.local` to request rendering from the dashboard.

`scripts/dev_backend.sh` sets `MAX_FRAMES=6000`; raw `uvicorn` uses the same default unless you export `MAX_FRAMES`.

### Mask overlay video (debug)

Writes an MP4 with a green semi-transparent mask composited on every frame that has `mask_rle` in the pipeline output. Uses `run_pipeline(..., include_masks_in_response=True)` so masks are kept even when the HTTP API would strip them (e.g. uncertain lock).

```bash
source backend/.venv/bin/activate
python -m backend.tools.render_mask_overlay_video \\
  /path/to/match.mp4 backend/sample-details.json -o /tmp/mask_overlay.mp4

# Pipeline cap vs output clip (do not use --max-frames for pipeline length)
python -m backend.tools.render_mask_overlay_video \\
  /path/to/match.mp4 backend/data/lozano-details.json -o /tmp/lozano_masks.mp4 \\
  --pipeline-max-frames 1100 --frame-start 970 --max-frames 130

# Quick SAM-only visual check (no player lock; largest person on anchor frame)
python backend/scripts/render_sam_demo_clip.py \\
  /path/to/match.mp4 -o /tmp/sam_demo.mp4 --anchor-frame 400 --num-frames 90

# Optional: rows from a saved `python -m backend.cli ...` JSON (only works if that JSON still contains mask_rle)
python -m backend.tools.render_mask_overlay_video \\
  /path/to/match.mp4 backend/sample-details.json -o /tmp/masked.mp4 \\
  --analyze-json /tmp/analyze.json
```

If the script prints `No rows with mask_rle`, MobileSAM did not run or failed to load (`timm`, weights, `SAM_ENABLED=1`). Check backend logs when running the full pipeline.

### What to verify

1. Command exits with code `0`.
2. JSON includes `video.frames` > 0 and a non-empty `rows` array when people are visible.
3. When the target jersey number is clearly visible for several seconds, `target.track_id` should be non-null and `target.confidence` > 0.
4. Filter heat-map points with `rows.filter(r => r.track === target.track_id)`.

## Project layout

```
backend/
├── app/
│   ├── main.py           # FastAPI /api/analyze
│   ├── schemas.py        # Pydantic models
│   └── pipeline/
│       ├── detector.py   # YOLOv8
│       ├── tracker.py    # ByteTrack
│       ├── ocr.py        # EasyOCR (number + name crops)
│       ├── color.py      # Per-pixel RGB tolerance matcher
│       ├── matcher.py    # 3-stage identifier + persistent track lock
│       ├── bbox.py       # Bbox clipping / region cropping helpers
│       └── run.py        # Orchestrator
├── cli.py
├── requirements.txt
└── sample-details.json
```

### ReID weights

Place **`osnet_x1_0_soccernet.pth`** in `detection_test/weights/` (see [backend/weights/README.md](weights/README.md)). The legacy analyzer uses OSNet embeddings to lock the target player by appearance when jersey OCR is ambiguous.

## Next step (frontend)

Wire the existing Next.js dashboard to call this API from [src/components/Dashboard.tsx](../src/components/Dashboard.tsx):

1. Lift video `File` state from `VideoUploadPanel` into `Dashboard` (same pattern as `PlayerDetails`).
2. On "Run analysis", `POST` multipart to `http://localhost:8000/api/analyze` with `video` + `JSON.stringify(details)`.
3. Use `response.target.track_id` to filter `rows` for the heat map.

## Notes

- OCR on jersey numbers + names works best on clips where the player is **facing away** from the camera so the name/number on the back is legible.
- Once a track is locked, OCR and color scoring are **skipped** for the rest of the video — output rows are produced quickly from that point on.
- For production, consider swapping `yolov8n.pt` for a football-specific fine-tuned model via `YOLO_WEIGHTS`.
- Processing is synchronous (one request at a time); long videos may take minutes on CPU.
