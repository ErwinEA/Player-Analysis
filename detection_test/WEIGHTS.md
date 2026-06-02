# Model weights contract

Place trained artifacts in `detection_test/weights/` (gitignored). Scripts resolve paths via [weights_config.py](weights_config.py) or env vars.

## Files

| File | Source | Used by |
|------|--------|---------|
| `yolo_soccer.pt` | You train (SoccerNet YOLO) | `run_detect.py` — person detection |
| `osnet_x1_0_soccernet.pth` | OSNet training (`training/reid_osnet/`) | `run_detect.py` — DeepSORT ReID embedder |
| `jersey_number_b0.pt` | [training/jersey_number/](training/jersey_number/) on SoccerNet jersey-2023 | `run_identify.py` — jersey number |
| `jersey_number_b0.json` | Optional class index map | `jersey_number.py` |
| `yolov8n_ball.pt` | You train (ball YOLO) | `run_detect.py --detect-ball` |

Alias: `osnet_soccer_reid.pth` is accepted as an alternative name for the OSNet checkpoint.

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `YOLO_WEIGHTS` | `weights/yolo_soccer.pt` or `../yolov8m.pt` | YOLO person detector |
| `YOLO_CLASS_FILTER` | `auto` | `coco` → class 0 only; `none` → all classes; `auto` → no filter for soccer weights |
| `REID_WEIGHTS` | `weights/osnet_x1_0_soccernet.pth` | OSNet checkpoint for DeepSORT |
| `JERSEY_WEIGHTS` | `weights/jersey_number_b0.pt` | EfficientNet jersey classifier |
| `JERSEY_CLS_MIN_CONF` | `0.75` | Min classifier confidence before OCR fallback |
| `BALL_WEIGHTS` | `weights/yolov8n_ball.pt` | Ball detector |
| `SOCCERNET_DATA` | `detection_test/data/soccernet` | SoccerNet download + crops root |

## Formats

### YOLO (`yolo_soccer.pt`)

- Ultralytics YOLOv8 `.pt` export.
- Single-class person model: set `YOLO_CLASS_FILTER=none` or use auto mode.
- COCO person fine-tune: use `YOLO_CLASS_FILTER=coco`.

### OSNet (`osnet_x1_0_soccernet.pth`)

- TorchReID `osnet_x1_0` full `state_dict` from best validation epoch.
- Input: BGR crop resized to **256×128**, ImageNet normalization (torchreid default).
- Output: **512-d** L2-normalized embedding (via `reid_extractor.py` or DeepSORT `embedder=torchreid`).

### Jersey classifier (`jersey_number_b0.pt` + optional `.json`)

- PyTorch checkpoint with keys: `model_state_dict`, optional `class_names`.
- Classes: integers `1`–`99` plus `"unknown"` for unreadable backs.
- Input: BGR jersey-back crop (see `number_crop` in `ocr_profile.py`), 224×224, ImageNet norm.
- Sidecar JSON example:

```json
{
  "class_names": ["unknown", "1", "2", "..."],
  "input_size": 224
}
```

### Ball YOLO (`yolov8n_ball.pt`)

- Ultralytics YOLOv8 single-class ball detector.
- Used only when `--detect-ball` is passed; output in `detections.json` → `balls[]`.

## Fallback behavior

| Component | If weight missing |
|-----------|-------------------|
| YOLO | `../yolov8m.pt` (COCO person) |
| ReID | DeepSORT default Market-1501 OSNet via torchreid, or `--embedder mobilenet` |
| Jersey number | EasyOCR digits-only fallback |
| Ball | `--detect-ball` skipped with warning if no weights |

## OSNet training outputs

After `training/reid_osnet/train_osnet.py`:

- `weights/osnet_x1_0_soccernet.pth` — best checkpoint
- `training/reid_osnet/outputs/reid_metrics.json` — Rank-1, mAP
- `weights/osnet_x1_0_soccernet.onnx` — optional ONNX export
