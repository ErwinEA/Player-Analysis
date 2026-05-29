"""PyTorch dataset for SoccerNet-ReID player crops."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


@dataclass(frozen=True)
class CropSample:
    path: Path
    player_id: int
    camera_id: str
    jersey: str


IMAGE_GLOBS = ("*.jpg", "*.jpeg", "*.png", "*.bmp")


def parse_reid_filename(filename: str) -> dict | None:
    """Parse SoccerNet-ReID filename (8 hyphen-separated fields before .png)."""
    stem = filename.rsplit(".", 1)[0]
    splits = stem.split("-")
    if len(splits) != 8:
        return None
    return {
        "bbox_idx": int(splits[0]),
        "action_idx": int(splits[1]),
        "person_uid": splits[2],
        "frame_idx": int(splits[3]),
        "clazz": splits[4],
        "jersey": splits[5],
        "uai": splits[6],
        "shape": splits[7],
    }


def discover_soccernet_reid_train(
    train_root: Path,
    players_only: bool = True,
    show_progress: bool = False,
) -> list[CropSample]:
    """Discover samples from unpacked reid/train/ tree (*/*/*/*/*.png)."""
    if not train_root.is_dir():
        return []

    uid_to_label: dict[str, int] = {}
    samples: list[CropSample] = []

    image_paths = train_root.glob("*/*/*/*/*.png")
    if show_progress:
        from tqdm import tqdm

        image_paths = tqdm(
            image_paths,
            desc="Indexing ReID images",
            unit="img",
        )

    for img_path in image_paths:
        info = parse_reid_filename(img_path.name)
        if info is None:
            continue
        if players_only and not info["clazz"].startswith("Player"):
            continue

        uid = info["person_uid"]
        if uid not in uid_to_label:
            uid_to_label[uid] = len(uid_to_label)
        label = uid_to_label[uid]
        camera_id = img_path.parent.name
        samples.append(
            CropSample(
                path=img_path,
                player_id=label,
                camera_id=camera_id,
                jersey=info["jersey"],
            )
        )
    return samples


def discover_training_samples(
    crops_dir: Path,
    data_root: Path | None = None,
    show_progress: bool = False,
) -> list[CropSample]:
    """Find training images: official reid/train/ tree first, then player_crops/ layout."""
    if data_root is not None:
        for rel in ("reid/train/train", "reid/train"):
            reid_train = data_root / rel
            samples = discover_soccernet_reid_train(
                reid_train, show_progress=show_progress
            )
            if len(samples) >= 10:
                return samples

    return discover_crops(crops_dir)


def discover_crops(crops_dir: Path) -> list[CropSample]:
    """Discover player_crops/{person_uid}/{jersey}/* layout (png/jpg)."""
    samples: list[CropSample] = []
    if not crops_dir.is_dir():
        return samples

    player_dirs = sorted(
        p for p in crops_dir.iterdir() if p.is_dir() and p.name != "prepare_summary.json"
    )
    player_to_label = {d.name: idx for idx, d in enumerate(player_dirs)}

    for player_dir in player_dirs:
        label = player_to_label[player_dir.name]
        for jersey_dir in player_dir.iterdir():
            if not jersey_dir.is_dir():
                continue
            for pattern in IMAGE_GLOBS:
                for img_path in jersey_dir.glob(pattern):
                    stem = img_path.stem
                    camera_id = stem.split("_f")[0] if "_f" in stem else jersey_dir.name
                    samples.append(
                        CropSample(
                            path=img_path,
                            player_id=label,
                            camera_id=camera_id,
                            jersey=jersey_dir.name,
                        )
                    )
    return samples


def split_query_gallery(
    samples: list[CropSample],
    seed: int = 42,
) -> tuple[list[CropSample], list[CropSample]]:
    """Hold out one camera view per identity for query; remaining views go to gallery."""
    by_player: dict[int, list[CropSample]] = {}
    for s in samples:
        by_player.setdefault(s.player_id, []).append(s)

    rng = random.Random(seed)
    query: list[CropSample] = []
    gallery: list[CropSample] = []

    for player_samples in by_player.values():
        cameras = sorted({s.camera_id for s in player_samples})
        if len(cameras) >= 2:
            query_cam = cameras[0]
            for s in player_samples:
                if s.camera_id == query_cam:
                    query.append(s)
                else:
                    gallery.append(s)
        else:
            rng.shuffle(player_samples)
            n_query = max(1, len(player_samples) // 2)
            query.extend(player_samples[:n_query])
            gallery.extend(player_samples[n_query:])

    return query, gallery


def stratified_split(
    samples: list[CropSample],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[list[CropSample], list[CropSample]]:
    by_player: dict[int, list[CropSample]] = {}
    for s in samples:
        by_player.setdefault(s.player_id, []).append(s)

    rng = random.Random(seed)
    train: list[CropSample] = []
    test: list[CropSample] = []
    for player_samples in by_player.values():
        rng.shuffle(player_samples)
        n_train = max(1, int(len(player_samples) * train_ratio))
        train.extend(player_samples[:n_train])
        test.extend(player_samples[n_train:])
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def build_transforms(height: int, width: int, train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((height, width)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((height, width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class SoccerNetReIDDataset(Dataset):
    """Returns (image_tensor, player_id, camera_id)."""

    def __init__(
        self,
        samples: list[CropSample],
        height: int = 256,
        width: int = 128,
        train: bool = True,
    ) -> None:
        self.samples = samples
        self.transform = build_transforms(height, width, train=train)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        sample = self.samples[idx]
        img = cv2.imread(str(sample.path))
        if img is None:
            img = np.zeros((128, 64, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = self.transform(img)
        return tensor, sample.player_id, sample.camera_id


class PKSampler(torch.utils.data.Sampler):
    """P identities x K images per batch for triplet-friendly batches."""

    def __init__(
        self,
        samples: list[CropSample],
        batch_size: int,
        num_instances: int = 4,
    ) -> None:
        self.samples = samples
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids = batch_size // num_instances
        by_pid: dict[int, list[int]] = {}
        for i, s in enumerate(samples):
            by_pid.setdefault(s.player_id, []).append(i)
        self.index_by_pid = by_pid
        self.pids = list(by_pid.keys())

    def __iter__(self):
        rng = random.Random()
        while True:
            rng.shuffle(self.pids)
            for start in range(0, len(self.pids), self.num_pids):
                selected = self.pids[start : start + self.num_pids]
                if len(selected) < self.num_pids:
                    break
                batch: list[int] = []
                for pid in selected:
                    indices = self.index_by_pid[pid]
                    if len(indices) >= self.num_instances:
                        batch.extend(rng.sample(indices, self.num_instances))
                    else:
                        batch.extend(rng.choices(indices, k=self.num_instances))
                yield batch

    def __len__(self) -> int:
        return max(1, len(self.pids) // self.num_pids)
