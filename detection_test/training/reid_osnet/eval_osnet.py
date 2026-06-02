#!/usr/bin/env python3
"""Evaluate OSNet ReID: Rank-1 accuracy and mAP with camera-disjoint query/gallery."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from common import load_config, load_train_manifest
from soccernet_reid_dataset import (
    SoccerNetReIDDataset,
    discover_training_samples,
    split_query_gallery,
    stratified_split,
)


def _extract_features(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    feats: list[np.ndarray] = []
    labels: list[int] = []
    with torch.no_grad():
        for images, lbls, _cam in loader:
            images = images.to(device)
            out = model(images)
            if isinstance(out, tuple):
                _, f = out
            else:
                f = out
            f = torch.nn.functional.normalize(f, p=2, dim=1)
            feats.append(f.cpu().numpy())
            labels.extend(lbls.numpy().tolist() if hasattr(lbls, "numpy") else lbls)
    return np.vstack(feats), np.asarray(labels, dtype=np.int64)


def compute_cmc_map(
    query_feats: np.ndarray,
    query_labels: np.ndarray,
    gallery_feats: np.ndarray,
    gallery_labels: np.ndarray,
) -> tuple[float, float]:
    """Rank-1 and mAP with cosine similarity on L2-normalized features."""
    if len(query_feats) == 0 or len(gallery_feats) == 0:
        return 0.0, 0.0

    sim = query_feats @ gallery_feats.T
    rank1_hits = 0
    aps: list[float] = []

    for i in range(len(query_feats)):
        order = np.argsort(-sim[i])
        matches = gallery_labels[order] == query_labels[i]
        if matches.any() and matches[0]:
            rank1_hits += 1
        rel = matches.astype(np.float32)
        if rel.sum() == 0:
            aps.append(0.0)
            continue
        cum_precision = np.cumsum(rel) / (np.arange(len(rel)) + 1)
        ap = float((cum_precision * rel).sum() / rel.sum())
        aps.append(ap)

    rank1 = rank1_hits / len(query_feats)
    mAP = float(np.mean(aps)) if aps else 0.0
    return rank1, mAP


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate OSNet ReID Rank-1 and mAP.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--crops-dir", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    crops_dir = Path(args.crops_dir or cfg["crops_dir"])
    data_root = Path(cfg["data_root"])
    weights = Path(args.weights or cfg["weights_out"])
    output_dir = Path(cfg["output_dir"])

    manifest = load_train_manifest(output_dir)
    if manifest:
        num_classes = int(manifest["num_classes"])
    else:
        samples_all = discover_training_samples(crops_dir, data_root)
        num_classes = len({s.player_id for s in samples_all})
        if num_classes < 1:
            print("No training data found and no train_manifest.json.", file=sys.stderr)
            sys.exit(1)

    samples = discover_training_samples(crops_dir, data_root)
    if not samples:
        print(f"No images under {data_root / 'reid/train/train'}", file=sys.stderr)
        sys.exit(1)

    _train, val_samples = stratified_split(samples, cfg["train_split"])
    query_samples, gallery_samples = split_query_gallery(val_samples, seed=cfg.get("seed", 42))

    if not query_samples or not gallery_samples:
        print("Query/gallery split empty; need multi-camera identities in val set.", file=sys.stderr)
        sys.exit(1)

    query_ds = SoccerNetReIDDataset(
        query_samples, cfg["input_height"], cfg["input_width"], train=False
    )
    gallery_ds = SoccerNetReIDDataset(
        gallery_samples, cfg["input_height"], cfg["input_width"], train=False
    )
    query_loader = DataLoader(query_ds, batch_size=cfg["batch_size"], shuffle=False)
    gallery_loader = DataLoader(gallery_ds, batch_size=cfg["batch_size"], shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from torchreid.models import build_model

    model = build_model(cfg["model_name"], num_classes=num_classes, pretrained=False)
    if not weights.is_file():
        print(f"Weights not found: {weights}", file=sys.stderr)
        sys.exit(1)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    model = model.to(device)

    query_feats, query_labels = _extract_features(model, query_loader, device)
    gallery_feats, gallery_labels = _extract_features(model, gallery_loader, device)
    rank1, mAP = compute_cmc_map(query_feats, query_labels, gallery_feats, gallery_labels)

    metrics = {
        "rank1": round(rank1, 4),
        "mAP": round(mAP, 4),
        "num_query": len(query_samples),
        "num_gallery": len(gallery_samples),
        "num_val": len(val_samples),
        "num_classes": num_classes,
        "eval_protocol": "camera_disjoint_query_gallery",
        "weights": str(weights),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "reid_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2))

    print(f"Rank-1: {rank1:.2%}  mAP: {mAP:.2%}")
    print(f"Query: {len(query_samples)}  Gallery: {len(gallery_samples)}")
    print(f"Wrote {out_path}")
    if rank1 < cfg.get("rank1_min_signoff", 0.70):
        print(
            f"Warning: Rank-1 {rank1:.2%} below sign-off threshold "
            f"{cfg.get('rank1_min_signoff', 0.70):.0%}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
