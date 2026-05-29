# OSNet fine-tuning on SoccerNet-ReID

Scripts for downloading SoccerNet ReID data, training OSNet-x1_0, and evaluating Rank-1 / mAP.

## Prerequisites

```bash
cd detetction_test
source test/bin/activate
pip install -r requirements.txt
../../scripts/install_torchreid.sh
pip install -r training/reid_osnet/requirements-train.txt
```

SoccerNet ReID download (public share password):

```bash
export SOCCERNET_PASS=SoccerNet   # default for reid zips; use NDA password for broadcast videos only
export SOCCERNET_DATA="/path/to/detetction_test/data/soccernet"  # optional
```

## Download (correct API)

**Do not** use `downloadGames(files=["tracking"])` — that creates empty game folders. Task data is downloaded as zips via `downloadDataTask`.

### ReID only (recommended)

```bash
cd training/reid_osnet

python download_soccernet.py --tasks reid --split train --unpack
```

Expected layout after download + unpack:

```
data/soccernet/reid/train.zip
data/soccernet/reid/train/train/<league>/<season>/<game>/<camera>/*.png
```

Training reads PNGs directly from `reid/train/train/` (no extra copy step).

### Optional: tracking + broadcast video (large)

```bash
python download_soccernet.py --tasks tracking --split train --unpack
python download_soccernet.py --tasks tracking --split train --unpack \
  --videos 720p --video-split train
python extract_crops.py
```

### Verify download

```bash
find "$SOCCERNET_DATA" \( -name "*.zip" -o -name "*.png" \) | head -20
```

## Training pipeline

```bash
cd training/reid_osnet

# 1. Download + unpack (see above)
python download_soccernet.py --tasks reid --split train --unpack

# 2. Train (CUDA recommended; tqdm progress bars, use --no-progress for logs/CI)
python train_osnet.py

# 3. Evaluate (camera-disjoint query vs gallery on val split)
python eval_osnet.py

# 4. Optional ONNX export (uses train_manifest.json for num_classes)
python export_onnx.py
```

Optional cache (not required):

```bash
python prepare_reid_crops.py   # symlinks to player_crops/{person_uid}/{jersey}/
```

Outputs:

- `../../weights/osnet_x1_0_soccernet.pth` — best checkpoint
- `outputs/train_manifest.json` — num_classes and training metadata
- `outputs/reid_metrics.json` — Rank-1, mAP
- `../../weights/osnet_x1_0_soccernet.onnx` — optional feature extractor

## Use with DeepSORT

After training:

```bash
python run_detect.py video.mp4 --embedder torchreid --reid-weights weights/osnet_x1_0_soccernet.pth
```

If weights are missing, use `--embedder mobilenet` (default when no ReID weights present).
