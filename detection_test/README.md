# Detection Test

Isolated sandbox for soccer-tuned detection, ByteTrack tracking, hybrid jersey identification, and optional ball detection. Does not affect the main backend until validated.

See [WEIGHTS.md](WEIGHTS.md) for model weight paths and env vars.

## Prerequisites

- Python 3.10+ (macOS / Linux / Windows)
- Optional: CUDA GPU for OSNet training and `REID_GPU=1` / `OCR_GPU=1`

## Virtual environment

```bash
cd detetction_test
python3 -m venv test
source test/bin/activate
pip install -r requirements.txt
./scripts/install_torchreid.sh
```

`torchreid` (OSNet ReID) cannot be installed with a plain `pip install git+...` on current Python/pip — see script comments. Do **not** use `pip install torchreid` from PyPI (old 0.2.x API, incompatible with this project).

## Model weights

Drop trained files into `weights/` (gitignored):

| File | Purpose |
|------|---------|
| `yolo_soccer.pt` | Soccer person detector (else falls back to `../yolov8m.pt`) |
| `osnet_x1_0_soccernet.pth` | OSNet ReID (backend player lock via appearance) |
| `jersey_number_b0.pt` | EfficientNet jersey classifier (else EasyOCR digits) |
| `yolov8n_ball.pt` | Ball detector for `--detect-ball` |

## End-to-end pipeline (one command)

Runs **YOLOv8 → geometry filter → ByteTrack → jersey labels → `pipeline.json`**, with optional target-player lock.

```bash
python run_pipeline.py /path/to/video.mp4 \
  --target-number 23 \
  --target-name "Player" \
  --primary-color "#1F3D31" \
  --secondary-color "#000000"
```

Quick test on the first 300 frames:

```bash
python run_pipeline.py video.mp4 --max-frames 300 \
  --target-number 23 --target-name "Player" \
  --primary-color "#1F3D31" --secondary-color "#000000"
```

Output:

```
output/{video_stem}/
├── frames/
├── detections.json          # raw tracks (also written by step 1)
├── detections_labeled.json  # tracks + jersey_number per detection
├── identified.json          # optional target lock segments
└── pipeline.json            # unified JSON (all stages)
```

Use `--no-kit-gate` to label every valid bbox (ignore team colors). Omit `--target-number` to skip the lock stage.

## Step 1 — YOLO detection + ByteTrack tracking

```bash
python run_detect.py /path/to/video.mp4
```

Soccer YOLO with tuned thresholds:

```bash
python run_detect.py video.mp4 \
  --weights weights/yolo_soccer.pt \
  --conf 0.1 \
  --track-buffer 30
```

Optional ball detection (sandbox; adds `balls[]` per frame):

```bash
python run_detect.py video.mp4 --detect-ball --ball-weights weights/yolov8n_ball.pt
```

| Flag | Default | Description |
|------|---------|-------------|
| `--weights` | `weights/yolo_soccer.pt` or `../yolov8m.pt` | YOLO person detector |
| `--conf` | `0.1` | YOLO confidence (ByteTrack uses low-conf detections) |
| `--track-buffer` | `30` | Frames to keep lost tracks alive |
| `--track-high-thresh` | `0.25` | First-stage association threshold |
| `--track-low-thresh` | `0.1` | Second-stage threshold for low-conf detections |
| `--new-track-thresh` | `0.15` | Minimum detection score to start a new track |
| `--max-frames` | all | Process only first N frames (quick tests) |
| `--detect-ball` | off | Enable ball YOLO |
| `--output-dir` | `output` | Output root |

Output:

```
output/{video_stem}/
├── frames/
└── detections.json   # people[] per frame; optional balls[]
```

### Visualize tracks

Draw colored bbox overlays and export an MP4:

```bash
python visualize_tracks.py output/testmatch2/detections.json --report
```

Output: `tracked_overlay.mp4` next to `detections.json`. Use `--max-frames 300` for a short preview.

## Step 2 — Player identification + re-lock

With face photo:

```bash
python run_identify.py output/matchtest/detections.json \
  --photo /path/to/face.jpg \
  --name "Player Name" \
  --number 10 \
  --primary-color "#ffffff" \
  --secondary-color "#000000"
```

Jersey + color only (no photo):

```bash
python run_identify.py output/matchtest/detections.json \
  --name "Player Name" \
  --number 10 \
  --primary-color "#ffffff" \
  --secondary-color "#000000"
```

Jersey numbers use **EfficientNet classifier** when `weights/jersey_number_b0.pt` exists, with **EasyOCR fallback**. Names use **EasyOCR** only.

Output: `identified.json` with `locked_segments[]`, `rows[]`, `summary`.

## OSNet training (SoccerNet-ReID)

Requires SoccerNet credentials when data is ready. See [training/reid_osnet/README.md](training/reid_osnet/README.md).

```bash
export SOCCERNET_PASS=SoccerNet   # reid zips; NDA password for broadcast videos
cd training/reid_osnet
python download_soccernet.py --tasks reid --split train --unpack
python train_osnet.py
python eval_osnet.py
```

For tracking + video crop extraction (large download), see [training/reid_osnet/README.md](training/reid_osnet/README.md).

## Jersey number classifier (SoccerNet jersey-2023)

Train EfficientNet-B0 on tracklet thumbnails (1–99 + `unknown`). See [training/jersey_number/README.md](training/jersey_number/README.md).

```bash
export SOCCERNET_PASS=your_nda_password
cd training/jersey_number
python download_jersey.py --unpack
python prepare_manifest.py
python train_jersey.py
python eval_jersey.py
```

Weights export to `weights/jersey_number_b0.pt` and are used by `run_identify.py`.

## Environment variables

| Variable | Description |
|----------|-------------|
| `YOLO_WEIGHTS` | Override person detector path |
| `YOLO_CLASS_FILTER` | `auto` / `coco` / `none` |
| `REID_WEIGHTS` | OSNet checkpoint (training / future ReID features) |
| `JERSEY_WEIGHTS` | EfficientNet jersey classifier |
| `JERSEY_CLS_MIN_CONF` | Classifier threshold (default 0.75) |
| `BALL_WEIGHTS` | Ball YOLO weights |
| `SOCCERNET_DATA` | SoccerNet download root |

## Validation checklist

### Detection

1. Boxes cover players on sample frames vs `detections.json`.
2. Track `id` stable ~50 frames when player stays visible.
3. Compare ID switches on same clip with different `--track-buffer` / `--conf` values.
4. With `--detect-ball`, spot-check `balls[]` on frames where ball is visible.

### Identification

1. **With photo:** locks within ~30 frames, `method` includes `face`.
2. **Without photo:** locks via `number` + `color` when back visible.
3. **Re-lock:** two `locked_segments` after ~60 frame absence.
4. **Wrong photo:** empty segments, `gave_up: true`.

**Disk usage:** full frame export is large (15k frames ≈ several GB).
