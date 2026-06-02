#!/usr/bin/env python3
"""Fine-tune OSNet-x1_0 on SoccerNet-ReID crops (CE + 0.5 * Triplet)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import load_config, save_train_manifest
from soccernet_reid_dataset import (
    SoccerNetReIDDataset,
    discover_training_samples,
    stratified_split,
)


class TripletLoss(nn.Module):
    def __init__(self, margin: float = 0.3) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        n = features.size(0)
        if n < 2:
            return features.sum() * 0.0
        dist = torch.cdist(features, features, p=2)
        loss = features.new_zeros(())
        count = 0
        for i in range(n):
            pos_mask = labels == labels[i]
            neg_mask = labels != labels[i]
            if pos_mask.sum() <= 1 or neg_mask.sum() == 0:
                continue
            pos_dist = dist[i][pos_mask].max()
            neg_dist = dist[i][neg_mask].min()
            loss = loss + torch.relu(pos_dist - neg_dist + self.margin)
            count += 1
        if count == 0:
            return features.sum() * 0.0
        return loss / count


def _build_model(model_name: str, num_classes: int, pretrained: bool):
    from torchreid.models import build_model

    # triplet loss mode: train forward returns (logits, embedding) for CE + triplet
    model = build_model(
        model_name, num_classes=num_classes, pretrained=pretrained, loss="triplet"
    )
    # #region agent log
    import json, time

    with open(
        "/Users/erwinelgy/projects/player analysis/.cursor/debug-bfa96a.log", "a"
    ) as _f:
        _f.write(
            json.dumps(
                {
                    "sessionId": "bfa96a",
                    "hypothesisId": "A",
                    "location": "train_osnet.py:_build_model",
                    "message": "model built",
                    "data": {"loss": getattr(model, "loss", None), "model_name": model_name},
                    "timestamp": int(time.time() * 1000),
                    "runId": "post-fix",
                }
            )
            + "\n"
        )
    # #endregion
    return model


def _forward_train(model, images):
    out = model(images)
    # #region agent log
    import json, time

    with open(
        "/Users/erwinelgy/projects/player analysis/.cursor/debug-bfa96a.log", "a"
    ) as _f:
        _f.write(
            json.dumps(
                {
                    "sessionId": "bfa96a",
                    "hypothesisId": "A",
                    "location": "train_osnet.py:_forward_train",
                    "message": "forward output",
                    "data": {
                        "training": model.training,
                        "loss": getattr(model, "loss", None),
                        "out_type": type(out).__name__,
                        "is_tuple": isinstance(out, tuple),
                        "tuple_len": len(out) if isinstance(out, tuple) else None,
                    },
                    "timestamp": int(time.time() * 1000),
                    "runId": "post-fix",
                }
            )
            + "\n"
        )
    # #endregion
    if isinstance(out, tuple) and len(out) >= 2:
        return out[0], out[1]
    raise RuntimeError("Expected (logits, features) tuple from torchreid model in train mode")


def train_one_epoch(
    model,
    loader,
    ce_loss,
    triplet_loss,
    optimizer,
    device,
    triplet_weight: float,
    epoch: int,
    total_epochs: int,
    scaler=None,
    show_progress: bool = True,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    batch_iter = loader
    if show_progress:
        batch_iter = tqdm(
            loader,
            desc=f"Train {epoch}/{total_epochs}",
            leave=False,
            unit="batch",
            position=1,
        )

    for images, labels, _cam in batch_iter:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits, features = _forward_train(model, images)
                loss = ce_loss(logits, labels) + triplet_weight * triplet_loss(features, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, features = _forward_train(model, images)
            loss = ce_loss(logits, labels) + triplet_weight * triplet_loss(features, labels)
            loss.backward()
            optimizer.step()

        batch_loss = float(loss.item())
        batch_acc = float((logits.argmax(dim=1) == labels).float().mean().item())
        total_loss += batch_loss * labels.size(0)
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += labels.size(0)

        if show_progress and hasattr(batch_iter, "set_postfix"):
            batch_iter.set_postfix(loss=f"{batch_loss:.4f}", acc=f"{batch_acc:.3f}")

    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    epoch: int,
    total_epochs: int,
    show_progress: bool = True,
) -> tuple[float, float]:
    model.eval()
    correct = 0
    total = 0

    batch_iter = loader
    if show_progress:
        batch_iter = tqdm(
            loader,
            desc=f"Val   {epoch}/{total_epochs}",
            leave=False,
            unit="batch",
            position=2,
        )

    for images, labels, _cam in batch_iter:
        images = images.to(device)
        labels = labels.to(device)
        feats = model(images)
        if isinstance(feats, tuple):
            logits = feats[0]
        else:
            logits = model.classifier(feats)
        preds = logits.argmax(dim=1)
        batch_acc = float((preds == labels).float().mean().item())
        correct += int((preds == labels).sum().item())
        total += labels.size(0)

        if show_progress and hasattr(batch_iter, "set_postfix"):
            batch_iter.set_postfix(acc=f"{batch_acc:.3f}")

    return correct / max(total, 1), total


def main() -> None:
    parser = argparse.ArgumentParser(description="Train OSNet-x1_0 on SoccerNet-ReID crops.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--crops-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars (logs / CI)",
    )
    args = parser.parse_args()

    show_progress = not args.no_progress

    cfg = load_config(args.config)
    crops_dir = Path(args.crops_dir or cfg["crops_dir"])
    data_root = Path(cfg["data_root"])

    samples = discover_training_samples(
        crops_dir, data_root, show_progress=show_progress
    )
    if len(samples) < 10:
        print(
            f"Need training images under {crops_dir} or {data_root / 'reid/train/train'} "
            f"(found {len(samples)}).\n"
            "Run: python download_soccernet.py --tasks reid --split train --unpack",
            file=sys.stderr,
        )
        sys.exit(1)

    train_samples, val_samples = stratified_split(samples, cfg["train_split"])
    num_classes = len({s.player_id for s in samples})
    print(f"Samples: train={len(train_samples)} val={len(val_samples)} classes={num_classes}")

    train_ds = SoccerNetReIDDataset(
        train_samples, cfg["input_height"], cfg["input_width"], train=True
    )
    val_ds = SoccerNetReIDDataset(
        val_samples, cfg["input_height"], cfg["input_width"], train=False
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
    )

    device_name = args.device or cfg["device"]
    if device_name == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable; falling back to CPU (smoke test only).")
        device_name = "cpu"
    device = torch.device(device_name)

    model = _build_model(cfg["model_name"], num_classes, cfg["pretrained"]).to(device)
    ce_loss = nn.CrossEntropyLoss()
    triplet_loss = TripletLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg["scheduler_step_size"], gamma=cfg["scheduler_gamma"]
    )
    scaler = torch.cuda.amp.GradScaler() if cfg.get("amp") and device.type == "cuda" else None

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_out = Path(cfg["weights_out"])
    weights_out.parent.mkdir(parents=True, exist_ok=True)

    best_val = 0.0
    plateau = 0
    epochs = args.epochs or cfg["epochs"]
    history: list[dict] = []

    epoch_iter = range(1, epochs + 1)
    epoch_bar: tqdm | None = None
    if show_progress:
        epoch_bar = tqdm(
            epoch_iter,
            desc="Epochs",
            unit="epoch",
            position=0,
            dynamic_ncols=True,
        )
        epoch_iter = epoch_bar

    for epoch in epoch_iter:
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            ce_loss,
            triplet_loss,
            optimizer,
            device,
            cfg["triplet_weight"],
            epoch,
            epochs,
            scaler,
            show_progress=show_progress,
        )
        val_acc, _ = evaluate(
            model, val_loader, device, epoch, epochs, show_progress=show_progress
        )
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_acc": round(val_acc, 4),
            "lr": lr,
        }
        history.append(row)

        epoch_summary = (
            f"epoch {epoch}/{epochs}: loss={train_loss:.4f} "
            f"train_acc={train_acc:.3f} val_acc={val_acc:.3f} lr={lr:.2e}"
        )
        if epoch_bar is not None:
            epoch_bar.set_postfix(
                loss=f"{train_loss:.3f}",
                train_acc=f"{train_acc:.3f}",
                val_acc=f"{val_acc:.3f}",
            )
            tqdm.write(epoch_summary)
        else:
            print(epoch_summary)

        if val_acc > best_val:
            best_val = val_acc
            plateau = 0
            torch.save(model.state_dict(), weights_out)
            msg = f"  saved best -> {weights_out}"
            if epoch_bar is not None:
                tqdm.write(msg)
            else:
                print(msg)
        else:
            plateau += 1
            if plateau >= cfg["early_stop_patience"]:
                if epoch_bar is not None:
                    epoch_bar.set_postfix_str("early stop")
                if epoch_bar is not None:
                    tqdm.write("Early stopping.")
                else:
                    print("Early stopping.")
                break

    (output_dir / "train_history.json").write_text(json.dumps(history, indent=2))
    save_train_manifest(
        output_dir,
        {
            "num_classes": num_classes,
            "num_samples": len(samples),
            "num_train": len(train_samples),
            "num_val": len(val_samples),
            "model_name": cfg["model_name"],
            "data_root": str(data_root),
            "weights_out": str(weights_out),
            "best_val_acc": round(best_val, 4),
        },
    )
    print(f"Training complete. Best val_acc={best_val:.4f}")


if __name__ == "__main__":
    main()
