# Model weights (backend)

Place trained checkpoints here or under `detection_test/weights/` (preferred for shared tooling).

## Happy path checklist (full `POST /api/analyze`)

Use on a **new machine** after `pip install -r backend/requirements.txt`. Checkpoints are gitignored (not in clone). Paths use the repo folder name `detection_test`.

| # | File | Primary path | Details |
|---|------|--------------|---------|
| 1 | `mobile_sam.pt` | `backend/weights/` | [MobileSAM](#mobilesam-legacy-player-segmentation) — `backend/scripts/install_mobile_sam.sh` |
| 2 | `jersey_number_b0.pt` | `detection_test/weights/` | [Jersey classifier](#jersey-number-classifier-efficientnet-b0) |
| 3 | `osnet_x1_0_soccernet.pth` | `detection_test/weights/` | [OSNet ReID](#osnet-reid-legacy--detection-pipeline) |
| 4 | `yolov8n_ball.pt` | `detection_test/weights/` then `backend/weights/` | [Ball YOLO](#ball-detector-yolov8) |
| 5 | `yolov8n_shuttle.pt` | `backend/weights/` | [Shuttle YOLO](#shuttle-detector-yolov8-badminton) — badminton rally stats only |

Person detector **`yolov8n.pt`**: Ultralytics auto-download on first analyze (`YOLO_WEIGHTS`).

**Verify files (bash/zsh, repo root):**

```bash
check() { test -f "$1" && echo "OK  $1" || echo "MISS $1"; }
check backend/weights/mobile_sam.pt
check detection_test/weights/jersey_number_b0.pt
check detection_test/weights/osnet_x1_0_soccernet.pth
check detection_test/weights/yolov8n_ball.pt || check backend/weights/yolov8n_ball.pt
check backend/weights/yolov8n_shuttle.pt
```

**Optional API check** (start backend first, e.g. `./scripts/dev_backend.sh` from repo root):

```bash
curl -s http://localhost:8000/health | python3 -m json.tool   # mobile_sam.status → ok; shuttle.status → ok (badminton)
```

| Missing | Effect |
|---------|--------|
| `mobile_sam.pt` | No masks / foot heatmap; bbox-only |
| `jersey_number_b0.pt` | EasyOCR jersey digits only |
| `osnet_x1_0_soccernet.pth` | Generic ImageNet OSNet |
| `yolov8n_ball.pt` | Ball event stats disabled |
| `yolov8n_shuttle.pt` | Badminton rally stats disabled (`badminton_stats_unavailable_reason: no_shuttle_weights`) |

Sandbox YOLO/ReID: [detection_test/WEIGHTS.md](../../detection_test/WEIGHTS.md).

---

## Ball detector (YOLOv8)

| File | Purpose |
|------|---------|
| **`yolov8n_ball.pt`** | Soccer ball detection for target-player event stats |

**Recommended path:**

```
detection_test/weights/yolov8n_ball.pt
```

Or `backend/weights/yolov8n_ball.pt`, or:

```bash
export BALL_WEIGHTS=/absolute/path/to/yolov8n_ball.pt
```

The weights file is **not** committed to git. Place it manually on each machine. If missing, ball events are disabled (`events_unavailable_reason: no_ball_weights`) and the rest of the analyze pipeline still runs.

---

## Shuttle detector (YOLOv8, badminton)

| File | Purpose |
|------|---------|
| **`yolov8n_shuttle.pt`** | Shuttlecock detection for badminton rally stats (rally wins, win rate, duration, winners) |

**Path:**

```
backend/weights/yolov8n_shuttle.pt
```

Or:

```bash
export SHUTTLE_WEIGHTS=/absolute/path/to/yolov8n_shuttle.pt
```

Not committed to git. If missing, rally stats are disabled (`badminton_stats_unavailable_reason: no_shuttle_weights`) and badminton movement stats / heatmap still run.

| Variable | Default | Description |
|----------|---------|-------------|
| `SHUTTLE_CONF` | `0.25` | Detection confidence threshold |
| `SHUTTLE_IOU` | `0.5` | NMS IoU |
| `SHUTTLE_LIVE_FRAMES` | `2` | Consecutive shuttle detections to start a rally (IDLE → LIVE) |
| `SHUTTLE_GAP_FRAMES` | `15` | Frames without a shuttle to end a rally (LIVE → END) |
| `SHUTTLE_TOUCH_RADIUS_PX` | `120` | Shuttle-to-locked-foot distance counting as a touch |
| `SHUTTLE_TOUCH_RECENT_FRAMES` | `30` | Touch recency window for the winners metric |

| Variable | Default | Description |
|----------|---------|-------------|
| `BALL_CONF` | `0.25` | Detection confidence threshold |
| `BALL_IOU` | `0.5` | NMS IoU |
| `BALL_KALMAN_GAP_MAX_FRAMES` | `22` | Max predicted frames before Kalman reset |
| `ENABLE_BALL_EVENTS` | `1` | Set `0` to skip ball detection entirely |

## Jersey number classifier (EfficientNet-B0)

| File | Purpose |
|------|---------|
| **`jersey_number_b0.pt`** | Primary jersey digit classifier (1–99 + unknown) |
| **`jersey_number_b0.json`** | Optional class-name sidecar |

**Recommended path:**

```
detection_test/weights/jersey_number_b0.pt
```

Or `backend/weights/jersey_number_b0.pt`, or `export JERSEY_WEIGHTS=/path/to/checkpoint.pt`.

If weights are missing, the API falls back to **EasyOCR** digits-only on jersey crops.

## OSNet ReID (legacy + detection pipeline)

| File | Purpose |
|------|---------|
| **`osnet_x1_0_soccernet.pth`** | SoccerNet-trained OSNet-x1_0 appearance embeddings |

**Recommended path:**

```
detection_test/weights/osnet_x1_0_soccernet.pth
```

Alternative (this folder):

```
backend/weights/osnet_x1_0_soccernet.pth
```

Or set:

```bash
export REID_WEIGHTS=/absolute/path/to/osnet_x1_0_soccernet.pth
```

### Dependencies

OSNet uses [torchreid](https://github.com/KaiyangZhou/deep-person-reid). Install from the detection test tree:

```bash
cd detection_test
./scripts/install_torchreid.sh   # if present
pip install torch torchvision
```

## MobileSAM (legacy player segmentation)

| File | Purpose |
|------|---------|
| **`mobile_sam.pt`** | MobileSAM checkpoint for per-frame mask on locked player |

**Recommended path:**

```
backend/weights/mobile_sam.pt
```

Download from [MobileSAM weights](https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt).

Install Python package:

```bash
pip install "git+https://github.com/ChaoningZhang/MobileSAM.git"
```

Or clone the repo and set `MOBILE_SAM_REPO=/path/to/MobileSAM`.

### Environment

| Variable | Default | Description |
|----------|---------|-------------|
| `SAM_ENABLED` | `1` | Set `0` to disable segmentation (bbox-only) |
| `SAM_WEIGHTS` | `backend/weights/mobile_sam.pt` | Checkpoint path |
| `SAM_DEVICE` | `auto` | `mps`, `cuda`, `cpu`, or `auto` |
| `SAM_WARMUP_FRAMES` | `3` | Consecutive frames on locked track before SAM starts |
| `REID_FALLBACK_THRESH` | `0.65` | ReID cosine min when lock track is missing |
| `REID_FALLBACK_MARGIN` | `0.10` | Min gap vs runner-up ReID score for fallback |
| `REID_FALLBACK_MAX_CENTER_FRAC` | `0.12` | Max bbox center jump (diag fraction) for fallback |
| `SAM_MASK_SCALE` | `0.5` | Downscale masks before RLE in API (0.125–1.0) |
| `TRACKER_TYPE` | `bytetrack` | Tracker backend: `bytetrack` or `botsort` (GMC for camera motion) |
| `TRACK_BUFFER` | `30`/`120` | Frames a lost track survives (default 120 under `botsort`) |
| `GMC_METHOD` | `sparseOptFlow` | BoT-SORT motion comp: `sparseOptFlow`/`orb`/`ecc`/`none` |

| Variable | Default | Description |
|----------|---------|-------------|
| `REID_WEIGHTS` | see paths above | Checkpoint path |
| `REID_ENABLED` | `1` | Set `0` to disable ReID |
| `REID_LOCK_THRESHOLD` | `0.65` | Cosine similarity to lock by appearance |
| `REID_MIN_SAMPLES` | `3` | Embeddings per track before ReID lock |
| `REID_PROTOTYPE_MIN_CONF` | `0.40` | Min jersey OCR conf to seed appearance prototype |
| `REID_GPU` | `0` | Set `1` for CUDA |
