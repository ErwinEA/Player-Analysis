# Model weights (backend)

Place trained checkpoints here or under `detetction_test/weights/` (preferred for shared tooling).

## Happy path checklist (full `POST /api/analyze`)

Use this on a **new machine** after `pip install -r backend/requirements.txt`. None of the files below are in git (see `.gitignore`).

| # | File | Put it here (first match wins) | Powers | How to get it |
|---|------|--------------------------------|--------|----------------|
| 1 | **`mobile_sam.pt`** | `backend/weights/mobile_sam.pt` | Player masks, `foot_x`/`foot_y`, mask-tight OCR/ReID | **Public download** — run `backend/scripts/install_mobile_sam.sh` (installs `mobile_sam` + `timm` and curls the checkpoint), or [MobileSAM weights](https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt) |
| 2 | **`jersey_number_b0.pt`** | `detetction_test/weights/jersey_number_b0.pt` (or `backend/weights/`) | Fast jersey digit reads (1–99); optional `jersey_number_b0.json` sidecar | **Train** on SoccerNet jersey-2023 — [detetction_test/training/jersey_number/README.md](../../detetction_test/training/jersey_number/README.md) (`SOCCERNET_PASS` + `train_jersey.py`). Without it: EasyOCR-only (slower, weaker lock). |
| 3 | **`osnet_x1_0_soccernet.pth`** | `detetction_test/weights/osnet_x1_0_soccernet.pth` (or `backend/weights/`) | Appearance lock / ReID fallback when jersey OCR is ambiguous | **Train** on SoccerNet-ReID — [detetction_test/training/reid_osnet/README.md](../../detetction_test/training/reid_osnet/README.md). Without it: generic ImageNet OSNet (weaker on broadcast footage). Alias: `osnet_soccer_reid.pth`. |
| 4 | **`yolov8n_ball.pt`** | `detetction_test/weights/yolov8n_ball.pt` then `backend/weights/yolov8n_ball.pt` | Ball detection → possession / pass / shot stats | **No public URL in this repo** — train a YOLOv8n single-class ball model (Ultralytics) on your labeled clips, or copy a teammate’s `.pt`. Without it: analyze still runs; ball events show `no_ball_weights`. |

**Also required (auto):** person detector **`yolov8n.pt`** — Ultralytics downloads on first analyze unless `YOLO_WEIGHTS` points elsewhere (e.g. repo-root `yolov8m.pt`).

**Python extras (not weights):**

| Component | Install |
|-----------|---------|
| MobileSAM | `pip install "git+https://github.com/ChaoningZhang/MobileSAM.git"` + `timm` — or `backend/scripts/install_mobile_sam.sh` |
| OSNet / torchreid | `detetction_test/scripts/install_torchreid.sh` (from repo root) |

**Verify from repo root:**

```bash
check() { test -f "$1" && echo "OK  $1" || echo "MISS $1"; }
check backend/weights/mobile_sam.pt
check detetction_test/weights/jersey_number_b0.pt
check detetction_test/weights/osnet_x1_0_soccernet.pth
check detetction_test/weights/yolov8n_ball.pt || check backend/weights/yolov8n_ball.pt
curl -s http://localhost:8000/health | python3 -m json.tool   # mobile_sam.status should be ok
```

**Degraded mode if something is missing:**

| Missing | Effect |
|---------|--------|
| `mobile_sam.pt` | No masks / foot heatmap points; bbox-only rows |
| `jersey_number_b0.pt` | EasyOCR jersey digits only |
| `osnet_x1_0_soccernet.pth` | Generic pretrained OSNet embeddings |
| `yolov8n_ball.pt` | Ball event stats disabled |

See also [detetction_test/WEIGHTS.md](../../detetction_test/WEIGHTS.md) for the sandbox detection pipeline (`yolo_soccer.pt`, etc.).

---

## Ball detector (YOLOv8)

| File | Purpose |
|------|---------|
| **`yolov8n_ball.pt`** | Soccer ball detection for target-player event stats |

**Recommended path:**

```
detetction_test/weights/yolov8n_ball.pt
```

Or `backend/weights/yolov8n_ball.pt`, or:

```bash
export BALL_WEIGHTS=/absolute/path/to/yolov8n_ball.pt
```

The weights file is **not** committed to git. Place it manually on each machine. If missing, ball events are disabled (`events_unavailable_reason: no_ball_weights`) and the rest of the analyze pipeline still runs.

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
detetction_test/weights/jersey_number_b0.pt
```

Or `backend/weights/jersey_number_b0.pt`, or `export JERSEY_WEIGHTS=/path/to/checkpoint.pt`.

If weights are missing, the API falls back to **EasyOCR** digits-only on jersey crops.

## OSNet ReID (legacy + detection pipeline)

| File | Purpose |
|------|---------|
| **`osnet_x1_0_soccernet.pth`** | SoccerNet-trained OSNet-x1_0 appearance embeddings |

**Recommended path:**

```
detetction_test/weights/osnet_x1_0_soccernet.pth
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
cd detetction_test
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
