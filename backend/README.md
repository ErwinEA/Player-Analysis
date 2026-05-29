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

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `YOLO_WEIGHTS` | `yolov8n.pt` | YOLOv8 weights file (`yolov8m.pt`, or a football fine-tuned `.pt`) |
| `DET_CONF` | `0.1` | YOLO confidence threshold (ByteTrack uses low-conf detections) |
| `TRACK_LOW_THRESH` | `0.1` | ByteTrack second-stage threshold; warn if `DET_CONF` is higher |
| `TRACK_MIN_AGE` | `3` | Frames before a track is analyzed for OCR / included in output |
| `OCR_EVERY` | `5` | Run OCR every N frames (OCR is slow) |
| `MAX_FRAMES` | `5000` | Cap frames through detection/tracking (max 5000; invalid/`0` → 5000) |
| `MAX_UPLOAD_MB` | `4096` | Max legacy video upload size in megabytes (4 GB) |
| `CORS_ALLOW_LOCAL_REGEX` | `1` | Allow localhost/127.0.0.1 on any port (set `0` in production) |
| `OCR_GPU` | `0` | Set to `1` to run EasyOCR on a CUDA GPU |
| `DEBUG_OCR` | `0` | Set to `1` to save jersey crops + `ocr.log` under `DEBUG_OCR_DIR` |
| `DEBUG_OCR_DIR` | `debug_ocr` | Output folder for OCR debug (project root by default) |
| `COLOR_TOLERANCE` | `10` | +/- units per RGB channel allowed when matching pixels to the user's hex color |
| `COLOR_LOCK_THRESHOLD` | `0.15` | Mean per-frame pixel-match fraction needed to lock by color |
| `COLOR_MATCH_MIN` | `0.05` | Floor: best color track still wins if above this when nothing stronger matches |
| `NUMBER_MIN_CONF` | `0.40` | EasyOCR confidence floor to count a digit reading as a vote |
| `NUMBER_LOCK_VOTES` | `1.5` | Cumulative number-OCR confidence required to lock by number alone |
| `NAME_MIN_CONF` | `0.35` | EasyOCR confidence floor to count a name reading as a vote |
| `NAME_MIN_RATIO` | `0.6` | Min fuzzy similarity (0-1) between OCR'd name and the user's name |
| `OCR_MIN_CROP_HEIGHT` | `64` | Upscale jersey crops shorter than this before OCR |
| `JERSEY_WEIGHTS` | `detetction_test/weights/jersey_number_b0.pt` | EfficientNet jersey classifier checkpoint |
| `JERSEY_CLS_MIN_CONF` | `0.75` | Min classifier confidence for a hard number read |
| `JERSEY_TARGET_MIN_CONF` | `0.55` | Min prob on target jersey class (also requires agreement with argmax) |
| `JERSEY_SOFT_MIN_CONF` | `0.25` | Soft classifier accept before OCR fallback |
| `JERSEY_CLS_DEVICE` | `auto` | `cuda`, `mps`, `cpu`, or `auto` for classifier inference |
| `SAM_ENABLED` | `1` | Per-frame MobileSAM on locked player (`0` = bbox-only) |
| `SAM_WEIGHTS` | `backend/weights/mobile_sam.pt` | MobileSAM checkpoint |
| `SAM_DEVICE` | `auto` | `mps`, `cuda`, `cpu`, or `auto` |
| `SAM_WARMUP_FRAMES` | `3` | Frames on locked track after identity lock before SAM |
| `REID_FALLBACK_THRESH` | `0.65` | ReID match when ByteTrack loses lock `track_id` |

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
2. **`number`** — Cumulative OCR confidence for the user's jersey number on a single track reaches `NUMBER_LOCK_VOTES`.
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

### `POST /api/analyze`

Multipart form:

- `video` — MP4 or MOV file
- `details` — JSON string matching frontend `PlayerDetails`:

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

### Mask overlay video (debug)

Writes an MP4 with a green semi-transparent mask composited on every frame that has `mask_rle` in the pipeline output. Uses `run_pipeline(..., include_masks_in_response=True)` so masks are kept even when the HTTP API would strip them (e.g. uncertain lock).

```bash
source backend/.venv/bin/activate
python -m backend.tools.render_mask_overlay_video \\
  /path/to/match.mp4 backend/sample-details.json -o /tmp/mask_overlay.mp4

# Optional: cap length (still reads input until cap)
python -m backend.tools.render_mask_overlay_video \\
  /path/to/match.mp4 backend/sample-details.json -o /tmp/short.mp4 --max-frames 300

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

Place **`osnet_x1_0_soccernet.pth`** in `detetction_test/weights/` (see [backend/weights/README.md](weights/README.md)). The legacy analyzer uses OSNet embeddings to lock the target player by appearance when jersey OCR is ambiguous.

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
