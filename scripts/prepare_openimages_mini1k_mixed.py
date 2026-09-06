#!/usr/bin/env python3
"""Build a non-copying mixed train index from OI Mini-1K and the custom train split."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--custom-dataset", default="laptop_mouse_yolov8")
    parser.add_argument("--output-dataset", default="openimages_mini1k_mixed")
    parser.add_argument("--expected-custom-images", type=int, default=163)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def images_under(path: Path) -> list[Path]:
    if not path.is_dir():
        raise FileNotFoundError(path)
    images = sorted(p.resolve() for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise RuntimeError(f"no images found: {path}")
    return images


def label_for(image: Path) -> Path:
    parts = list(image.parts)
    try:
        index = len(parts) - 1 - parts[::-1].index("images")
    except ValueError as exc:
        raise RuntimeError(f"image path has no images component: {image}") from exc
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def dataset_stats(images: list[Path]) -> dict:
    boxes = Counter()
    groups = Counter()
    for image in images:
        label = label_for(image)
        if not label.is_file():
            raise FileNotFoundError(f"missing label for {image}: {label}")
        present: set[int] = set()
        for line_number, line in enumerate(label.read_text(encoding="utf-8-sig").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f"{label}:{line_number}: expected 5 fields")
            class_id = int(fields[0])
            if class_id not in (0, 1):
                raise ValueError(f"{label}:{line_number}: unexpected class {class_id}")
            values = [float(value) for value in fields[1:]]
            if not all(0.0 <= value <= 1.0 for value in values) or values[2] <= 0 or values[3] <= 0:
                raise ValueError(f"{label}:{line_number}: invalid normalized box")
            boxes[str(class_id)] += 1
            present.add(class_id)
        if present == {0}:
            groups["laptop_only"] += 1
        elif present == {1}:
            groups["mouse_only"] += 1
        elif present == {0, 1}:
            groups["both"] += 1
        else:
            groups["negative"] += 1
    return {"images": len(images), "boxes": dict(boxes), "groups": dict(groups)}


def build_linked_train_tree(output: Path, sources: list[tuple[str, list[Path]]]) -> list[Path]:
    image_dir = output / "train" / "images"
    label_dir = output / "train" / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    wanted_images: set[str] = set()
    wanted_labels: set[str] = set()
    linked_images: list[Path] = []

    def link(source: Path, destination: Path) -> None:
        if destination.is_symlink():
            if destination.resolve() == source.resolve():
                return
            destination.unlink()
        elif destination.exists():
            raise RuntimeError(f"refusing to replace non-symlink: {destination}")
        destination.symlink_to(source)

    for prefix, images in sources:
        for image in images:
            stem = f"{prefix}__{image.stem}"
            linked_image = image_dir / f"{stem}{image.suffix.lower()}"
            linked_label = label_dir / f"{stem}.txt"
            link(image, linked_image)
            link(label_for(image), linked_label)
            wanted_images.add(linked_image.name)
            wanted_labels.add(linked_label.name)
            linked_images.append(linked_image.resolve())

    for directory, wanted in ((image_dir, wanted_images), (label_dir, wanted_labels)):
        for path in directory.iterdir():
            if path.name not in wanted:
                if not path.is_symlink():
                    raise RuntimeError(f"unexpected non-symlink in mixed tree: {path}")
                path.unlink()
    return linked_images


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    datasets = repo / "datasets"
    oi_root = datasets / "openimages_mini1k"
    custom_root = datasets / args.custom_dataset
    output = datasets / args.output_dataset

    if (oi_root / ".ready").read_text(encoding="utf-8").strip() != "1000":
        raise RuntimeError("Open Images Mini-1K is not ready")

    oi_images = images_under(oi_root / "train" / "images")
    custom_images = images_under(custom_root / "train" / "images")
    valid_images = images_under(custom_root / "valid" / "images")
    test_images = images_under(custom_root / "test" / "images")
    if len(oi_images) != 1000 or len(custom_images) != args.expected_custom_images:
        raise RuntimeError(
            f"expected 1000+{args.expected_custom_images} train images, "
            f"got {len(oi_images)}+{len(custom_images)}"
        )

    train_images = oi_images + custom_images
    train_hashes: dict[str, Path] = {}
    for completed, image in enumerate(train_images, 1):
        digest = sha256(image)
        if digest in train_hashes:
            raise RuntimeError(f"duplicate train image: {train_hashes[digest]} and {image}")
        train_hashes[digest] = image
        if completed % 250 == 0:
            print(f"hashed train images: {completed}/{len(train_images)}", flush=True)

    for split, images in (("valid", valid_images), ("test", test_images)):
        for image in images:
            digest = sha256(image)
            if digest in train_hashes:
                raise RuntimeError(f"train/{split} leakage: {train_hashes[digest]} and {image}")

    output.mkdir(parents=True, exist_ok=True)
    linked_images = build_linked_train_tree(
        output, [("oi", oi_images), ("custom", custom_images)]
    )
    (output / "train.txt").write_text(
        "\n".join(str(path) for path in linked_images) + "\n", encoding="utf-8"
    )
    (output / "data.yaml").write_text(
        f"path: {output}\n"
        "train: train/images\n"
        f"val: {custom_root / 'valid' / 'images'}\n"
        f"test: {custom_root / 'test' / 'images'}\n"
        "names:\n"
        "- laptop\n"
        "- mouse\n"
        "nc: 2\n",
        encoding="utf-8",
    )

    oi_stats = dataset_stats(oi_images)
    custom_stats = dataset_stats(custom_images)
    mixed_stats = dataset_stats(train_images)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "Open Images Mini-1K + custom train split",
        "training_images": len(train_images),
        "sources": {"openimages_mini1k": oi_stats, "custom_train": custom_stats},
        "mixed": mixed_stats,
        "validation": {"source": "custom valid only", "images": len(valid_images)},
        "test": {"source": "custom test only", "images": len(test_images)},
        "exact_hash_leakage": 0,
        "copied_images": 0,
        "linked_images": len(linked_images),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / ".ready").write_text(f"{len(train_images)}\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    print(f"MIXED DATASET READY: {len(train_images)} train images", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
