# Jersey Number Classifier (EfficientNet-B0)

Train a 1–99 + `unknown` jersey digit classifier on **SoccerNet jersey-2023** tracklet thumbnails.

## Prerequisites

```bash
cd detetction_test
source test2/bin/activate   # or your venv
pip install -r training/jersey_number/requirements-train.txt
export SOCCERNET_PASS='your_nda_password'
```

## 1. Download tracklet data

```bash
cd training/jersey_number
python download_jersey.py --unpack
```

Uses:

```python
downloader.downloadDataTask(task="jersey-2023", split=["train", "test", "challenge"])
```

Data lands under `detetction_test/data/soccernet/jersey-2023/`.

**Note:** Training reads images directly from `train.zip` / `test.zip` — no full unpack required (saves disk space).

## 2. Build labeled manifest

Each tracklet folder `{player_id}/` is labeled from the split ground-truth JSON:

`{image_path → jersey_number}` (integer 1–99, or `-1` → `unknown`).

```bash
python prepare_manifest.py
```

Samples **1,000–2,000** train thumbnails (stratified by jersey class) into `outputs/jersey_manifest.json`.

## 3. Train

```bash
python train_jersey.py
# GPU: python train_jersey.py --device cuda
```

Exports:

- `detetction_test/weights/jersey_number_b0.pt`
- `detetction_test/weights/jersey_number_b0.json`

## 4. Evaluate

```bash
python eval_jersey.py
```

## Runtime (sandbox)

`detetction_test/jersey_number.py` loads the exported weights automatically when present; otherwise falls back to EasyOCR.

Used by `run_identify.py` via `JerseyProfileReader`.

## Config

Edit `config.yaml`:

| Key | Default | Description |
|-----|---------|-------------|
| `model` | `efficientnet_b0` | `efficientnet_b0` or `resnet18` |
| `max_train_samples` | `2000` | Cap train thumbnails |
| `min_train_samples` | `1000` | Target minimum |
| `epochs` | `25` | Max epochs (usually stops earlier via early stopping) |
| `early_stop_patience` | `3` | Stop if `val_loss` does not improve for this many epochs |
| `JERSEY_CLS_MIN_CONF` | `0.75` | Runtime inference threshold (env) |
