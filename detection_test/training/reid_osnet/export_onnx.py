#!/usr/bin/env python3
"""Export trained OSNet to ONNX (L2-normalized feature vectors for DeepSORT)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

from common import load_config, load_train_manifest
from soccernet_reid_dataset import discover_training_samples


class FeatureExportWrapper(nn.Module):
    """Export backbone features only (inference), not classification logits."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        if isinstance(out, tuple):
            features = out[1]
        else:
            features = out
        return torch.nn.functional.normalize(features, p=2, dim=1)


def _resolve_num_classes(cfg: dict, crops_dir: Path, data_root: Path) -> int:
    manifest = load_train_manifest(Path(cfg["output_dir"]))
    if manifest and "num_classes" in manifest:
        return int(manifest["num_classes"])

    samples = discover_training_samples(crops_dir, data_root)
    if samples:
        return len({s.player_id for s in samples})

    print(
        "No train_manifest.json and no samples found; cannot infer num_classes.",
        file=sys.stderr,
    )
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export OSNet checkpoint to ONNX.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    weights = Path(args.weights or cfg["weights_out"])
    if not weights.is_file():
        print(f"Weights not found: {weights}", file=sys.stderr)
        sys.exit(1)

    crops_dir = Path(cfg["crops_dir"])
    data_root = Path(cfg["data_root"])
    num_classes = _resolve_num_classes(cfg, crops_dir, data_root)

    from torchreid.models import build_model

    model = build_model(cfg["model_name"], num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    export_model = FeatureExportWrapper(model)
    export_model.eval()

    h, w = cfg["input_height"], cfg["input_width"]
    dummy = torch.randn(1, 3, h, w)
    out_path = args.output or weights.with_suffix(".onnx")

    torch.onnx.export(
        export_model,
        dummy,
        str(out_path),
        input_names=["input"],
        output_names=["features"],
        dynamic_axes={"input": {0: "batch"}, "features": {0: "batch"}},
        opset_version=17,
    )
    print(f"Exported {out_path} (num_classes={num_classes}, feature dim only)")


if __name__ == "__main__":
    main()
