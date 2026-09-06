#!/usr/bin/env python3
"""Rebuild the 232-image custom dataset with capture-session-isolated splits.

The original dataset is read-only. Files are copied into a new dataset so the
old train/valid/test layout remains untouched and auditable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TIMESTAMP_RE = re.compile(r"IMG_(\d{8})_(\d{6})")
MAX_WITHIN_GROUP_GAP_SECONDS = 60

# Deterministic assignment for the nine capture groups in the current dataset.
# Positive capture sessions are never split across train/valid/test.
GROUP_ASSIGNMENTS = {
    "20260825_094637": "test",
    "20260825_095115": "valid",
    "20260825_105223": "test",
    "20260825_112251": "test",
    "20260825_115253": "test",
    "20260826_144428": "train",
    "20260830_105657": "test",
    "20260831_093301": "valid",
    "20260831_093833": "valid",
}
EXPECTED_SPLIT_COUNTS = {"train": 112, "valid": 91, "test": 29}


@dataclass(frozen=True)
class Item:
    image: Path
    label: Path
    old_split: str
    captured_at: datetime
    laptop_boxes: int
    mouse_boxes: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("datasets/laptop_mouse_yolov8"),
        help="source dataset, relative paths are resolved under --repo",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/laptop_mouse_yolov8_session_split"),
        help="new dataset, relative paths are resolved under --repo",
    )
    return parser.parse_args()


def resolve_under_repo(path: Path, repo: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_timestamp(name: str) -> datetime:
    match = TIMESTAMP_RE.search(name)
    if not match:
        raise ValueError(f"filename has no capture timestamp: {name}")
    return datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S")


def read_box_counts(label: Path) -> tuple[int, int]:
    counts = Counter()
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
        if not all(0 <= value <= 1 for value in values) or values[2] <= 0 or values[3] <= 0:
            raise ValueError(f"{label}:{line_number}: invalid YOLO box")
        counts[class_id] += 1
    return counts[0], counts[1]


def load_items(source: Path) -> list[Item]:
    items: list[Item] = []
    names: set[str] = set()
    hashes: dict[str, Path] = {}
    for split in ("train", "valid", "test"):
        image_dir = source / split / "images"
        label_dir = source / split / "labels"
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(f"missing source split: {source / split}")
        for image in sorted(image_dir.iterdir()):
            if not image.is_file() or image.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            if image.name in names:
                raise RuntimeError(f"duplicate filename across old splits: {image.name}")
            names.add(image.name)
            label = label_dir / f"{image.stem}.txt"
            if not label.is_file():
                raise FileNotFoundError(f"missing label for {image}")
            laptop, mouse = read_box_counts(label)
            digest = sha256(image)
            if digest in hashes:
                raise RuntimeError(f"exact duplicate images: {hashes[digest]} and {image}")
            hashes[digest] = image
            items.append(Item(image, label, split, parse_timestamp(image.name), laptop, mouse))
    if len(items) != 232:
        raise RuntimeError(f"expected 232 source images, found {len(items)}")
    return sorted(items, key=lambda item: (item.captured_at, item.image.name))


def group_items(items: list[Item]) -> list[list[Item]]:
    groups: list[list[Item]] = []
    for item in items:
        if (
            not groups
            or (item.captured_at - groups[-1][-1].captured_at).total_seconds()
            > MAX_WITHIN_GROUP_GAP_SECONDS
        ):
            groups.append([item])
        else:
            groups[-1].append(item)
    observed = {group[0].captured_at.strftime("%Y%m%d_%H%M%S") for group in groups}
    if observed != set(GROUP_ASSIGNMENTS):
        raise RuntimeError(
            "capture groups changed; refusing an unaudited split: "
            f"observed={sorted(observed)}"
        )
    return groups


def copy_verified(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256(source) != sha256(destination):
            raise RuntimeError(f"existing destination differs: {destination}")
        return
    shutil.copy2(source, destination)


def stats(items: list[Item]) -> dict:
    laptop_images = sum(item.laptop_boxes > 0 for item in items)
    mouse_images = sum(item.mouse_boxes > 0 for item in items)
    return {
        "images": len(items),
        "laptop_boxes": sum(item.laptop_boxes for item in items),
        "mouse_boxes": sum(item.mouse_boxes for item in items),
        "laptop_images": laptop_images,
        "mouse_images": mouse_images,
        "both_images": sum(item.laptop_boxes > 0 and item.mouse_boxes > 0 for item in items),
        "empty_images": sum(item.laptop_boxes + item.mouse_boxes == 0 for item in items),
    }


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    source = resolve_under_repo(args.source, repo)
    output = resolve_under_repo(args.output, repo)
    if source == output:
        raise RuntimeError("source and output must be different")

    items = load_items(source)
    groups = group_items(items)
    assigned: dict[str, list[Item]] = {"train": [], "valid": [], "test": []}
    item_group: dict[str, str] = {}
    group_manifest: list[dict] = []
    for group in groups:
        group_key = group[0].captured_at.strftime("%Y%m%d_%H%M%S")
        split = GROUP_ASSIGNMENTS[group_key]
        assigned[split].extend(group)
        for item in group:
            item_group[item.image.name] = group_key
        group_manifest.append(
            {
                "group": group_key,
                "start": group[0].captured_at.isoformat(),
                "end": group[-1].captured_at.isoformat(),
                "assigned_split": split,
                **stats(group),
            }
        )

    actual_counts = {split: len(values) for split, values in assigned.items()}
    if actual_counts != EXPECTED_SPLIT_COUNTS:
        raise RuntimeError(f"unexpected split counts: {actual_counts}")

    rows: list[dict] = []
    for split, split_items in assigned.items():
        for item in split_items:
            image_destination = output / split / "images" / item.image.name
            label_destination = output / split / "labels" / item.label.name
            copy_verified(item.image, image_destination)
            copy_verified(item.label, label_destination)
            rows.append(
                {
                    "image": item.image.name,
                    "captured_at": item.captured_at.isoformat(),
                    "capture_group": item_group[item.image.name],
                    "old_split": item.old_split,
                    "new_split": split,
                    "laptop_boxes": item.laptop_boxes,
                    "mouse_boxes": item.mouse_boxes,
                    "sha256": sha256(item.image),
                }
            )

    expected_names = {split: {item.image.name for item in values} for split, values in assigned.items()}
    for split in ("train", "valid", "test"):
        actual_names = {
            path.name
            for path in (output / split / "images").iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        }
        if actual_names != expected_names[split]:
            raise RuntimeError(f"unexpected files in output {split}")

    (output / "data.yaml").write_text(
        "names:\n"
        "- laptop\n"
        "- mouse\n"
        "nc: 2\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n",
        encoding="utf-8",
    )
    with (output / "split_assignment.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["new_split"], row["captured_at"])))

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "output": str(output),
        "method": "capture-session grouping from filename timestamps",
        "max_within_group_gap_seconds": MAX_WITHIN_GROUP_GAP_SECONDS,
        "source_images": len(items),
        "splits": {split: stats(values) for split, values in assigned.items()},
        "groups": group_manifest,
        "exact_duplicate_images": 0,
        "source_modified": False,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / ".ready").write_text("232\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    print("SESSION SPLIT READY: train=112 valid=91 test=29", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
