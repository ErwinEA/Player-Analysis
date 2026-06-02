#!/usr/bin/env python3
"""Evaluate trained jersey classifier on SoccerNet jersey-2023 test tracklets."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import load_config, manifest_path, save_json
from soccernet_jersey_dataset import CLASS_NAMES, JerseyNumberDataset


def _build_model(name: str, num_classes: int) -> nn.Module:
    from torchvision import models

    if name == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def _tracklet_predictions(
    entries: list[dict],
    model: nn.Module,
    class_names: list[str],
    input_size: int,
    device: torch.device,
) -> dict[str, tuple[str, float]]:
    ds = JerseyNumberDataset(entries, input_size=input_size, augment=False)
    num_workers = 0 if sys.platform == "darwin" else 2
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=num_workers)

    logits_sum: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = defaultdict(int)
    idx = 0
    model.eval()
    with torch.no_grad():
        for images, _ in tqdm(loader, desc="eval"):
            images = images.to(device)
            logits = model(images)
            for row in logits:
                entry = entries[idx]
                player_id = entry["player_id"]
                if player_id not in logits_sum:
                    logits_sum[player_id] = row.clone()
                else:
                    logits_sum[player_id] += row
                counts[player_id] += 1
                idx += 1

    out: dict[str, tuple[str, float]] = {}
    for player_id, summed in logits_sum.items():
        avg = summed / max(counts[player_id], 1)
        probs = torch.softmax(avg, dim=0)
        conf, pred_idx = float(probs.max().item()), int(probs.argmax().item())
        out[player_id] = (class_names[pred_idx], conf)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate jersey number classifier.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = Path(cfg["output_dir"])
    manifest_file = manifest_path(output_dir)
    if not manifest_file.is_file():
        print(f"Missing manifest: {manifest_file}. Run prepare_manifest.py first.", file=sys.stderr)
        sys.exit(1)

    manifest = json.loads(manifest_file.read_text())
    test_entries = manifest.get("test", [])
    if not test_entries:
        print("No test entries in manifest.", file=sys.stderr)
        sys.exit(1)

    weights_path = args.weights or Path(cfg["weights_out"])
    if not weights_path.is_file():
        print(f"Weights not found: {weights_path}", file=sys.stderr)
        sys.exit(1)

    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    class_names = list(ckpt.get("class_names", CLASS_NAMES))
    model_name = ckpt.get("model", cfg.get("model", "efficientnet_b0"))
    input_size = int(ckpt.get("input_size", cfg.get("input_size", 224)))

    device_str = args.device or cfg.get("device", "cpu")
    if device_str == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif device_str == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        if device_str in {"cuda", "mps"}:
            print(f"Requested device '{device_str}' unavailable — using cpu.")
        device = torch.device("cpu")

    model = _build_model(model_name, len(class_names))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    preds = _tracklet_predictions(test_entries, model, class_names, input_size, device)

    gt_by_player: dict[str, str] = {}
    for entry in test_entries:
        gt_by_player[entry["player_id"]] = entry["label"]

    correct = 0
    total = 0
    per_class_total: dict[str, int] = defaultdict(int)
    per_class_correct: dict[str, int] = defaultdict(int)

    for player_id, gt_label in gt_by_player.items():
        pred_label, _conf = preds.get(player_id, ("unknown", 0.0))
        total += 1
        per_class_total[gt_label] += 1
        if pred_label == gt_label:
            correct += 1
            per_class_correct[gt_label] += 1

    accuracy = correct / max(total, 1)
    metrics = {
        "tracklet_accuracy": accuracy,
        "tracklets": total,
        "frame_accuracy_note": "Primary metric is tracklet-level (one vote per player_id).",
        "per_class_tracklet_accuracy": {
            k: per_class_correct.get(k, 0) / max(per_class_total.get(k, 1), 1)
            for k in sorted(per_class_total.keys(), key=lambda x: (x == "unknown", x))
        },
    }
    out_path = output_dir / "eval_metrics.json"
    save_json(out_path, metrics)
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
