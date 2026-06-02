#!/usr/bin/env python3
"""Train EfficientNet-B0 (or ResNet-18) jersey number classifier on SoccerNet jersey-2023."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common import jersey_task_dir, load_config, manifest_path, save_json
from soccernet_jersey_dataset import (
    CLASS_NAMES,
    JerseyNumberDataset,
    build_manifest,
    config_fingerprint,
)


def _build_model(name: str, num_classes: int, pretrained: bool) -> nn.Module:
    from torchvision import models

    if name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    raise ValueError(f"Unknown model: {name}. Use efficientnet_b0 or resnet18.")


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device_str == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if device_str in {"cuda", "mps"}:
        print(f"Requested device '{device_str}' unavailable — using cpu.", flush=True)
    return torch.device("cpu")


def _run_epoch(model, loader, criterion, device, optimizer=None, desc: str | None = None) -> tuple[float, float]:
    from tqdm import tqdm

    train = optimizer is not None
    model.train() if train else model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    batches = tqdm(loader, desc=desc, leave=False) if desc else loader
    for images, targets in batches:
        images = images.to(device)
        targets = targets.to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = criterion(logits, targets)
            if train:
                loss.backward()
                optimizer.step()
        total_loss += float(loss.item()) * images.size(0)
        preds = logits.argmax(dim=1)
        correct += int((preds == targets).sum().item())
        total += images.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def _load_or_build_manifest(manifest_file: Path, task_dir: Path, cfg: dict, force: bool) -> dict:
    fp = config_fingerprint(cfg)
    if manifest_file.is_file() and not force:
        manifest = json.loads(manifest_file.read_text())
        if manifest.get("config_fingerprint") == fp:
            return manifest
        print("Config changed — rebuilding manifest.")
    manifest = build_manifest(task_dir, cfg)
    manifest["config_fingerprint"] = fp
    save_json(manifest_file, manifest)
    print(f"Built manifest: {manifest_file}")
    print(f"Counts: {manifest['counts']}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Train jersey number classifier.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--force-manifest", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_root = args.data_root or Path(cfg["data_root"])
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    task_dir = jersey_task_dir(data_root)

    manifest_file = manifest_path(output_dir)
    manifest = _load_or_build_manifest(manifest_file, task_dir, cfg, args.force_manifest)

    train_entries = manifest["train"]
    val_entries = manifest["val"]
    if len(train_entries) < 50:
        print(f"Too few train samples ({len(train_entries)}). Download/unpack jersey-2023 first.", file=sys.stderr)
        sys.exit(1)

    device_str = args.device or cfg.get("device", "cpu")
    device = _resolve_device(device_str)
    pin_memory = device.type == "cuda"
    num_workers = 0 if sys.platform == "darwin" else int(cfg.get("num_workers", 4))

    input_size = int(cfg.get("input_size", 224))
    train_ds = JerseyNumberDataset(train_entries, input_size=input_size, augment=True)
    val_ds = JerseyNumberDataset(val_entries, input_size=input_size, augment=False)

    batch_size = int(cfg.get("batch_size", 32))
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )

    model_name = cfg.get("model", "efficientnet_b0")
    num_classes = len(CLASS_NAMES)
    model = _build_model(model_name, num_classes, bool(cfg.get("pretrained", True))).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("lr", 3e-4)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(cfg.get("scheduler_step_size", 8)),
        gamma=float(cfg.get("scheduler_gamma", 0.1)),
    )

    epochs = args.epochs or int(cfg.get("epochs", 25))
    patience = int(cfg.get("early_stop_patience", 3))
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    stale = 0
    best_path = output_dir / "best_jersey_b0.pt"
    weights_out = Path(cfg["weights_out"])
    meta_out = Path(cfg["meta_out"])
    history: list[dict] = []

    def _save_best(epoch: int, va_loss: float, va_acc: float) -> None:
        ckpt = {
            "model_state_dict": model.state_dict(),
            "class_names": CLASS_NAMES,
            "model": model_name,
            "input_size": input_size,
            "val_loss": va_loss,
            "val_accuracy": va_acc,
            "epoch": epoch,
        }
        torch.save(ckpt, best_path)
        weights_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, weights_out)
        save_json(
            meta_out,
            {
                "class_names": CLASS_NAMES,
                "input_size": input_size,
                "model": model_name,
                "best_val_loss": round(va_loss, 4),
                "best_val_accuracy": round(va_acc, 4),
                "best_epoch": epoch,
            },
        )

    print(
        f"Training {model_name} on {len(train_entries)} samples, val={len(val_entries)}, "
        f"device={device}, early_stop_patience={patience} (val_loss)",
        flush=True,
    )

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = _run_epoch(
            model, train_loader, criterion, device, optimizer, desc=f"train {epoch:02d}"
        )
        va_loss, va_acc = _run_epoch(model, val_loader, criterion, device, desc=f"val {epoch:02d}")
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": round(tr_loss, 4),
            "train_acc": round(tr_acc, 4),
            "val_loss": round(va_loss, 4),
            "val_acc": round(va_acc, 4),
        }
        history.append(row)
        print(
            f"epoch {epoch:02d}  train loss={tr_loss:.4f} acc={tr_acc:.3f}  "
            f"val loss={va_loss:.4f} acc={va_acc:.3f}",
            flush=True,
        )

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_val_acc = va_acc
            best_epoch = epoch
            stale = 0
            _save_best(epoch, va_loss, va_acc)
            print(f"  -> new best val_loss={va_loss:.4f} (saved)", flush=True)
        else:
            stale += 1
            print(f"  -> no val_loss improvement ({stale}/{patience})", flush=True)
            if stale >= patience:
                print(
                    f"Early stop at epoch {epoch}: val_loss stalled for {patience} epochs. "
                    f"Best epoch={best_epoch}, val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.3f}",
                    flush=True,
                )
                break

    metrics = {
        "best_val_loss": round(best_val_loss, 4),
        "best_val_accuracy": round(best_val_acc, 4),
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "early_stop_patience": patience,
        "train_samples": len(train_entries),
        "val_samples": len(val_entries),
        "history": history,
    }
    save_json(output_dir / "train_metrics.json", metrics)

    min_acc = float(cfg.get("min_val_accuracy", 0.5))
    if best_val_acc < min_acc:
        print(
            f"Warning: best val accuracy {best_val_acc:.3f} below min_val_accuracy={min_acc}",
            file=sys.stderr,
        )

    if not best_path.is_file():
        print("No checkpoint saved.", file=sys.stderr)
        sys.exit(1)

    ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    ckpt["metrics"] = metrics
    torch.save(ckpt, weights_out)
    print(f"Exported weights -> {weights_out}")
    print(f"Exported meta    -> {meta_out}")


if __name__ == "__main__":
    main()
