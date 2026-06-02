"""Discover and load SoccerNet jersey-2023 tracklet thumbnails + labels."""

from __future__ import annotations

import hashlib
import io
import json
import random
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

UNKNOWN_LABEL = "unknown"
CLASS_NAMES = [UNKNOWN_LABEL] + [str(n) for n in range(1, 100)]


@dataclass(frozen=True)
class JerseySample:
    path: str
    player_id: str
    jersey_number: int
    split: str
    label: str
    zip_path: str | None = None
    zip_member: str | None = None


def _jersey_to_label(number: int) -> str:
    if number < 0 or number > 99 or number == 0:
        return UNKNOWN_LABEL
    return str(number)


def _load_gt_from_zip(zf: zipfile.ZipFile, split: str) -> dict[str, int]:
    candidates = [
        f"{split}/train_gt.json",
        f"{split}/test_gt.json",
        f"{split}/gt.json",
        f"train/train_gt.json" if split == "train" else f"test/test_gt.json",
    ]
    for member in candidates:
        try:
            raw = zf.read(member)
            gt_raw = json.loads(raw.decode("utf-8"))
            return {str(k): int(v) for k, v in gt_raw.items()}
        except KeyError:
            continue
    for name in zf.namelist():
        if name.endswith(".json") and split in name and "gt" in name.lower():
            gt_raw = json.loads(zf.read(name).decode("utf-8"))
            return {str(k): int(v) for k, v in gt_raw.items()}
    return {}


def _find_image_prefix(zf: zipfile.ZipFile, split: str) -> str:
    prefix_candidates = [
        f"{split}/images/",
        "train/images/" if split == "train" else "test/images/",
    ]
    for cand in prefix_candidates:
        for name in zf.namelist():
            if name.startswith(cand):
                return cand
    for name in zf.namelist():
        if "/images/" in name:
            idx = name.index("/images/") + 1
            return name[: idx + len("images/") - 1]
    raise FileNotFoundError(f"No images/ folder found in zip for split {split}")


def _index_players_in_zip(
    zf: zipfile.ZipFile,
    prefix: str,
    max_per_player: int,
    seed: int = 42,
) -> dict[str, list[str]]:
    by_player: dict[str, list[str]] = defaultdict(list)
    for name in zf.namelist():
        if not name.startswith(prefix) or name.endswith("/"):
            continue
        if not name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            continue
        player_id = name[len(prefix) :].split("/")[0]
        by_player[player_id].append(name)

    rng = random.Random(seed)
    capped: dict[str, list[str]] = {}
    for player_id, names in by_player.items():
        if len(names) > max_per_player:
            names = names[:]
            rng.shuffle(names)
            names = names[:max_per_player]
        capped[player_id] = names
    return capped


def _discover_from_zip(
    zip_path: Path,
    split: str,
    labels: dict[str, int],
    max_frames: int | None = None,
    max_per_player: int = 30,
    seed: int = 42,
) -> list[JerseySample]:
    samples: list[JerseySample] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        prefix = _find_image_prefix(zf, split)
        by_player = _index_players_in_zip(zf, prefix, max_per_player=max_per_player, seed=seed)

    rng = random.Random(seed)
    player_ids = list(by_player.keys())
    rng.shuffle(player_ids)

    for player_id in player_ids:
        jersey_number = labels.get(player_id, -1)
        label = _jersey_to_label(jersey_number)
        for name in by_player[player_id]:
            samples.append(
                JerseySample(
                    path=f"zip:{zip_path}::{name}",
                    player_id=player_id,
                    jersey_number=jersey_number,
                    split=split,
                    label=label,
                    zip_path=str(zip_path.resolve()),
                    zip_member=name,
                )
            )
            if max_frames is not None and len(samples) >= max_frames:
                return samples
    return samples


def _load_labels_for_zip(zip_path: Path, split: str) -> dict[str, int]:
    with zipfile.ZipFile(zip_path, "r") as zf:
        labels = _load_gt_from_zip(zf, split)
    if not labels:
        raise FileNotFoundError(f"No ground-truth JSON found inside {zip_path}")
    return labels


def _find_gt_json(split_dir: Path, task_dir: Path, split: str) -> Path | None:
    candidates = [
        split_dir / "train_gt.json",
        split_dir / "test_gt.json",
        split_dir / "gt.json",
        split_dir / "labels.json",
        task_dir / f"{split}_gt.json",
        task_dir / f"{split}.json",
    ]
    for path in candidates:
        if path.is_file():
            return path
    for path in sorted(split_dir.glob("*gt*.json")):
        if path.is_file():
            return path
    for path in sorted(split_dir.glob("*.json")):
        if path.name in {"manifest.json", "jersey_manifest.json"}:
            continue
        if path.is_file():
            return path
    return None


def _image_root(split_dir: Path) -> Path | None:
    for name in ("image", "images", "img"):
        root = split_dir / name
        if root.is_dir():
            return root
    return None


def discover_split(task_dir: Path, split: str) -> tuple[dict[str, int], Path | None, Path | None]:
    split_dir = task_dir / split
    if not split_dir.is_dir():
        return {}, None, None

    gt_path = _find_gt_json(split_dir, task_dir, split)
    if gt_path is None or not gt_path.is_file():
        return {}, _image_root(split_dir), None

    gt_raw = json.loads(gt_path.read_text(encoding="utf-8"))
    labels = {str(k): int(v) for k, v in gt_raw.items()}
    return labels, _image_root(split_dir), gt_path


def discover_samples(
    task_dir: Path,
    splits: list[str] | None = None,
    max_train_scan: int | None = None,
    max_test_scan: int | None = None,
    seed: int = 42,
) -> list[JerseySample]:
    splits = splits or ["train", "test"]
    samples: list[JerseySample] = []

    for split in splits:
        zip_path = task_dir / f"{split}.zip"
        if zip_path.is_file():
            labels = _load_labels_for_zip(zip_path, split)
            if split == "train":
                cap = max_train_scan
            else:
                cap = max_test_scan or 5000
            samples.extend(
                _discover_from_zip(
                    zip_path,
                    split,
                    labels,
                    max_frames=cap,
                    max_per_player=30,
                    seed=seed,
                )
            )
            continue

        labels, image_root, gt_path = discover_split(task_dir, split)
        if image_root is None or not image_root.is_dir():
            continue
        if gt_path is None or not labels:
            raise FileNotFoundError(
                f"Missing ground-truth for split '{split}' under {task_dir}. "
                f"Expected {split}.zip or unpacked {split}/ folder."
            )
        for player_dir in sorted(p for p in image_root.iterdir() if p.is_dir()):
            player_id = player_dir.name
            jersey_number = labels.get(player_id, -1)
            label = _jersey_to_label(jersey_number)
            for img_path in sorted(player_dir.glob("*")):
                if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                    continue
                samples.append(
                    JerseySample(
                        path=str(img_path.resolve()),
                        player_id=player_id,
                        jersey_number=jersey_number,
                        split=split,
                        label=label,
                    )
                )
    if not samples:
        raise FileNotFoundError(f"No jersey samples found under {task_dir}")
    return samples


def _group_by_player(samples: list[JerseySample]) -> dict[str, list[JerseySample]]:
    grouped: dict[str, list[JerseySample]] = defaultdict(list)
    for s in samples:
        grouped[s.player_id].append(s)
    return grouped


def sample_train_subset(
    samples: list[JerseySample],
    min_samples: int,
    max_samples: int,
    seed: int = 42,
) -> list[JerseySample]:
    train = [s for s in samples if s.split == "train"]
    by_player = _group_by_player(train)
    rng = random.Random(seed)

    players_by_label: dict[str, list[str]] = defaultdict(list)
    for player_id, frames in by_player.items():
        players_by_label[frames[0].label].append(player_id)

    chosen_players: list[str] = []
    labels = sorted(players_by_label.keys(), key=lambda x: (x == UNKNOWN_LABEL, x))
    per_label_players = max(1, max_samples // max(len(labels), 1) // 8)
    for label in labels:
        pool = players_by_label[label][:]
        rng.shuffle(pool)
        chosen_players.extend(pool[:per_label_players])

    chosen_frames: list[JerseySample] = []
    for player_id in chosen_players:
        chosen_frames.extend(by_player[player_id])

    if len(chosen_frames) < min_samples:
        remaining_players = [p for p in by_player if p not in chosen_players]
        rng.shuffle(remaining_players)
        for player_id in remaining_players:
            chosen_frames.extend(by_player[player_id])
            if len(chosen_frames) >= min_samples:
                break

    if len(chosen_frames) > max_samples:
        rng.shuffle(chosen_frames)
        chosen_frames = chosen_frames[:max_samples]
    return chosen_frames


def stratified_split_by_player(
    samples: list[JerseySample],
    train_frac: float,
    seed: int = 42,
) -> tuple[list[JerseySample], list[JerseySample]]:
    rng = random.Random(seed)
    by_player = _group_by_player(samples)
    players_by_label: dict[str, list[str]] = defaultdict(list)
    for player_id, frames in by_player.items():
        players_by_label[frames[0].label].append(player_id)

    train_players: set[str] = set()
    val_players: set[str] = set()
    for player_ids in players_by_label.values():
        rng.shuffle(player_ids)
        if len(player_ids) == 1:
            train_players.add(player_ids[0])
            continue
        cut = max(1, int(len(player_ids) * train_frac))
        cut = min(cut, len(player_ids) - 1)
        train_players.update(player_ids[:cut])
        val_players.update(player_ids[cut:])

    train_out = [s for s in samples if s.player_id in train_players]
    val_out = [s for s in samples if s.player_id in val_players]
    return train_out, val_out


def validate_label_distribution(samples: list[JerseySample], context: str) -> None:
    if not samples:
        raise ValueError(f"No samples for {context}")
    unknown = sum(1 for s in samples if s.label == UNKNOWN_LABEL)
    ratio = unknown / len(samples)
    if ratio > 0.95:
        raise ValueError(
            f"{context}: {ratio:.1%} samples labeled unknown — ground truth likely missing."
        )
    known_labels = {s.label for s in samples if s.label != UNKNOWN_LABEL}
    if not known_labels:
        raise ValueError(f"{context}: no known jersey labels (1-99) found.")


def config_fingerprint(cfg: dict) -> str:
    payload = {
        "max_train_samples": cfg.get("max_train_samples"),
        "min_train_samples": cfg.get("min_train_samples"),
        "train_val_split": cfg.get("train_val_split"),
        "seed": cfg.get("seed", 42),
    }
    raw = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def build_manifest(task_dir: Path, cfg: dict) -> dict:
    max_train = int(cfg.get("max_train_samples", 2000))
    # Scan enough frames to sample diverse tracklets without listing entire 700k+ train zip.
    scan_cap = max(max_train * 4, 8000)
    all_samples = discover_samples(
        task_dir,
        splits=["train", "test"],
        max_train_scan=scan_cap,
        max_test_scan=5000,
        seed=int(cfg.get("seed", 42)),
    )
    train_all = [s for s in all_samples if s.split == "train"]
    test_samples = [s for s in all_samples if s.split == "test"]
    validate_label_distribution(train_all + test_samples, "all splits")

    train_pool = sample_train_subset(
        all_samples,
        min_samples=int(cfg.get("min_train_samples", 1000)),
        max_samples=int(cfg.get("max_train_samples", 2000)),
        seed=int(cfg.get("seed", 42)),
    )
    validate_label_distribution(train_pool, "train pool")

    train_samples, val_samples = stratified_split_by_player(
        train_pool, float(cfg.get("train_val_split", 0.85)), seed=int(cfg.get("seed", 42))
    )

    def _serialize(items: list[JerseySample]) -> list[dict]:
        return [
            {
                "path": s.path,
                "player_id": s.player_id,
                "jersey_number": s.jersey_number,
                "label": s.label,
                "split": s.split,
                "zip_path": s.zip_path,
                "zip_member": s.zip_member,
            }
            for s in items
        ]

    label_hist: dict[str, int] = defaultdict(int)
    for s in train_samples:
        label_hist[s.label] += 1

    return {
        "task_dir": str(task_dir.resolve()),
        "class_names": CLASS_NAMES,
        "config_fingerprint": config_fingerprint(cfg),
        "source": "zip" if (task_dir / "train.zip").is_file() else "unpacked",
        "label_histogram_train": dict(sorted(label_hist.items(), key=lambda x: (x[0] == UNKNOWN_LABEL, x[0]))),
        "counts": {
            "train_all_scanned": len(train_all),
            "train_sampled": len(train_samples),
            "val": len(val_samples),
            "test": len(test_samples),
            "train_players": len({s.player_id for s in train_samples}),
            "val_players": len({s.player_id for s in val_samples}),
        },
        "train": _serialize(train_samples),
        "val": _serialize(val_samples),
        "test": _serialize(test_samples),
    }


class _ZipImageCache:
    """Keep ZipFile handles open to avoid reopening on every sample."""

    def __init__(self) -> None:
        self._handles: dict[str, zipfile.ZipFile] = {}

    def read(self, zip_path: str, zip_member: str) -> bytes:
        zf = self._handles.get(zip_path)
        if zf is None:
            zf = zipfile.ZipFile(zip_path, "r")
            self._handles[zip_path] = zf
        return zf.read(zip_member)

    def close(self) -> None:
        for zf in self._handles.values():
            zf.close()
        self._handles.clear()


def _open_image(entry: dict, zip_cache: _ZipImageCache | None = None) -> Image.Image:
    zip_path = entry.get("zip_path")
    zip_member = entry.get("zip_member")
    if zip_path and zip_member:
        if zip_cache is not None:
            data = zip_cache.read(zip_path, zip_member)
        else:
            with zipfile.ZipFile(zip_path, "r") as zf:
                data = zf.read(zip_member)
        return Image.open(io.BytesIO(data)).convert("RGB")
    return Image.open(entry["path"]).convert("RGB")


class JerseyNumberDataset(Dataset):
    def __init__(
        self,
        entries: list[dict],
        input_size: int = 224,
        augment: bool = False,
    ) -> None:
        self.entries = entries
        self._zip_cache = _ZipImageCache()
        self.class_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        if augment:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((input_size, input_size)),
                    transforms.ColorJitter(0.15, 0.15, 0.15, 0.05),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((input_size, input_size)),
                    transforms.ToTensor(),
                    normalize,
                ]
            )

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        import torch

        entry = self.entries[idx]
        img = _open_image(entry, self._zip_cache)
        label_name = entry["label"]
        target = self.class_to_idx.get(label_name, 0)
        return self.transform(img), torch.tensor(target, dtype=torch.long)

    def __del__(self) -> None:
        self._zip_cache.close()
