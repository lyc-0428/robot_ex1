#!/usr/bin/env python3
"""Build an exactly-1000-image Open Images V7 YOLO training set.

The generated train split is Open-Images-only. Existing project validation and
test splits are referenced by data.yaml but are never copied into train.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import sys
import threading
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image


LAPTOP = "/m/01c648"
COMPUTER_MOUSE = "/m/020lf"
ANIMAL_MOUSE = "/m/04rmv"
TARGET_BITS = {LAPTOP: 1, COMPUTER_MOUSE: 2}
TARGET_TO_YOLO = {LAPTOP: 0, COMPUTER_MOUSE: 1}

NEGATIVE_CLASSES = {
    "computer_keyboard": "/m/01m2v",
    "computer_monitor": "/m/02522",
    "remote_control": "/m/0qjjc",
    "mobile_phone": "/m/050k8",
    "book": "/m/0bt_c3",
    "desk": "/m/01y9k5",
    "office_supplies": "/m/02rdsp",
}

QUOTAS = {
    # Only 65 unique train images have usable bboxes for both target classes.
    # Keep 20 of them as deterministic download replacements.
    "laptop_only": 328,
    "computer_mouse_only": 327,
    "both": 45,
    "negative:computer_keyboard": 43,
    "negative:computer_monitor": 43,
    "negative:remote_control": 43,
    "negative:mobile_phone": 43,
    "negative:book": 43,
    "negative:desk": 43,
    "negative:office_supplies": 42,
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
S3_TEMPLATE = "https://open-images-dataset.s3.amazonaws.com/train/{image_id}.jpg"


@dataclass(frozen=True)
class Record:
    image_id: str
    group: str
    boxes: tuple[tuple[int, float, float, float, float], ...]

    def as_json(self) -> dict:
        return {
            "image_id": self.image_id,
            "group": self.group,
            "boxes": [list(box) for box in self.boxes],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--download-retries", type=int, default=5)
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def stable_shuffle(values: Iterable[str], seed: int, group: str) -> list[str]:
    result = sorted(set(values))
    random.Random(f"{seed}:{group}").shuffle(result)
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def valid_box(row: list[str], index: dict[str, int]) -> bool:
    if float(row[index["Confidence"]]) < 0.5:
        return False
    if row[index["IsGroupOf"]] == "1" or row[index["IsDepiction"]] == "1":
        return False
    xmin = float(row[index["XMin"]])
    xmax = float(row[index["XMax"]])
    ymin = float(row[index["YMin"]])
    ymax = float(row[index["YMax"]])
    return 0 <= xmin < xmax <= 1 and 0 <= ymin < ymax <= 1 and (xmax - xmin) > 0.002 and (ymax - ymin) > 0.002


def scan_bbox(path: Path) -> tuple[
    dict[str, int],
    dict[str, list[tuple[int, float, float, float, float]]],
    dict[str, set[str]],
    set[str],
]:
    relevant = set(TARGET_BITS) | set(NEGATIVE_CLASSES.values()) | {ANIMAL_MOUSE}
    negative_id_to_name = {value: name for name, value in NEGATIVE_CLASSES.items()}
    target_presence: dict[str, int] = defaultdict(int)
    target_boxes: dict[str, list[tuple[int, float, float, float, float]]] = defaultdict(list)
    negative_candidates: dict[str, set[str]] = {name: set() for name in NEGATIVE_CLASSES}
    animal_bbox: set[str] = set()

    log(f"Scanning bbox metadata: {path}")
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        index = {name: position for position, name in enumerate(header)}
        for line_number, row in enumerate(reader, 2):
            if line_number % 5_000_000 == 0:
                log(f"bbox rows scanned: {line_number:,}")
            label = row[index["LabelName"]]
            if label not in relevant:
                continue
            image_id = row[index["ImageID"]]
            if label in TARGET_BITS:
                target_presence[image_id] |= TARGET_BITS[label]
                if valid_box(row, index):
                    xmin = float(row[index["XMin"]])
                    xmax = float(row[index["XMax"]])
                    ymin = float(row[index["YMin"]])
                    ymax = float(row[index["YMax"]])
                    target_boxes[image_id].append(
                        (TARGET_TO_YOLO[label], xmin, ymin, xmax, ymax)
                    )
            elif label == ANIMAL_MOUSE:
                animal_bbox.add(image_id)
            elif valid_box(row, index):
                negative_candidates[negative_id_to_name[label]].add(image_id)
    log(
        "bbox scan complete: "
        f"target_images={len(target_presence):,}, animal_mouse={len(animal_bbox):,}"
    )
    return target_presence, target_boxes, negative_candidates, animal_bbox


def scan_image_labels(path: Path) -> tuple[dict[str, int], set[str]]:
    target_label_mask: dict[str, int] = defaultdict(int)
    animal_positive: set[str] = set()
    relevant = set(TARGET_BITS) | {ANIMAL_MOUSE}
    log(f"Scanning image-level metadata: {path}")
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        index = {name: position for position, name in enumerate(header)}
        for line_number, row in enumerate(reader, 2):
            if line_number % 10_000_000 == 0:
                log(f"image-label rows scanned: {line_number:,}")
            label = row[index["LabelName"]]
            if label not in relevant or float(row[index["Confidence"]]) < 0.5:
                continue
            image_id = row[index["ImageID"]]
            if label == ANIMAL_MOUSE:
                animal_positive.add(image_id)
            else:
                target_label_mask[image_id] |= TARGET_BITS[label]
    log(
        "image-label scan complete: "
        f"target_positive={len(target_label_mask):,}, animal_mouse={len(animal_positive):,}"
    )
    return target_label_mask, animal_positive


def usable_class_mask(boxes: list[tuple[int, float, float, float, float]]) -> int:
    mask = 0
    for class_id, *_ in boxes:
        mask |= 1 if class_id == 0 else 2
    return mask


def select_plan(
    target_presence: dict[str, int],
    target_boxes: dict[str, list[tuple[int, float, float, float, float]]],
    negative_candidates: dict[str, set[str]],
    animal_excluded: set[str],
    target_label_mask: dict[str, int],
    seed: int,
) -> tuple[dict[str, list[Record]], dict[str, list[Record]], dict[str, int]]:
    pools: dict[str, list[str]] = {}
    pools["laptop_only"] = [
        image_id
        for image_id, boxes in target_boxes.items()
        if image_id not in animal_excluded
        and target_presence.get(image_id) == 1
        and usable_class_mask(boxes) == 1
        and not (target_label_mask.get(image_id, 0) & 2)
    ]
    pools["computer_mouse_only"] = [
        image_id
        for image_id, boxes in target_boxes.items()
        if image_id not in animal_excluded
        and target_presence.get(image_id) == 2
        and usable_class_mask(boxes) == 2
        and not (target_label_mask.get(image_id, 0) & 1)
    ]
    pools["both"] = [
        image_id
        for image_id, boxes in target_boxes.items()
        if image_id not in animal_excluded
        and target_presence.get(image_id) == 3
        and usable_class_mask(boxes) == 3
    ]

    target_or_label = set(target_presence) | set(target_label_mask)
    for name, candidates in negative_candidates.items():
        pools[f"negative:{name}"] = [
            image_id
            for image_id in candidates
            if image_id not in animal_excluded and image_id not in target_or_label
        ]

    primary: dict[str, list[Record]] = {}
    backups: dict[str, list[Record]] = {}
    used: set[str] = set()
    candidate_counts = {group: len(set(values)) for group, values in pools.items()}

    for group, quota in QUOTAS.items():
        candidates = [
            image_id
            for image_id in stable_shuffle(pools[group], seed, group)
            if image_id not in used
        ]
        if len(candidates) < quota:
            raise RuntimeError(f"{group}: need {quota}, only {len(candidates)} unique candidates")
        chosen_ids = candidates[:quota]
        used.update(chosen_ids)
        primary[group] = [
            Record(image_id, group, tuple(target_boxes.get(image_id, ())))
            for image_id in chosen_ids
        ]

    for group, quota in QUOTAS.items():
        candidates = [
            image_id
            for image_id in stable_shuffle(pools[group], seed + 1, f"backup:{group}")
            if image_id not in used
        ]
        backup_count = min(len(candidates), max(100, quota // 2))
        chosen_ids = candidates[:backup_count]
        used.update(chosen_ids)
        backups[group] = [
            Record(image_id, group, tuple(target_boxes.get(image_id, ())))
            for image_id in chosen_ids
        ]
    return primary, backups, candidate_counts


def validate_image(path: Path) -> tuple[int, int, str]:
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        width, height = image.size
        if width < 32 or height < 32:
            raise ValueError(f"image too small: {width}x{height}")
    return width, height, sha256(path)


def download_one(record: Record, image_dir: Path, retries: int) -> dict:
    destination = image_dir / f"{record.image_id}.jpg"
    if destination.is_file():
        try:
            width, height, digest = validate_image(destination)
            return {"ok": True, "record": record, "path": destination, "sha256": digest, "width": width, "height": height}
        except Exception:
            destination.unlink(missing_ok=True)

    error = "unknown error"
    for attempt in range(1, retries + 1):
        temporary = destination.with_suffix(
            f".part.{os.getpid()}.{threading.get_ident()}"
        )
        try:
            request = urllib.request.Request(
                S3_TEMPLATE.format(image_id=record.image_id),
                headers={"User-Agent": "robot-ex1-openimages-mini1k/1.0"},
            )
            with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
            os.replace(temporary, destination)
            width, height, digest = validate_image(destination)
            return {"ok": True, "record": record, "path": destination, "sha256": digest, "width": width, "height": height}
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            temporary.unlink(missing_ok=True)
            destination.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(min(2 ** (attempt - 1), 10))
    return {"ok": False, "record": record, "error": error}


def existing_dataset_hashes(dataset_root: Path) -> set[str]:
    hashes: set[str] = set()
    for path in dataset_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            try:
                hashes.add(sha256(path))
            except OSError:
                pass
    log(f"Existing dataset hashes loaded: {len(hashes):,}")
    return hashes


def download_records(
    records: list[Record], image_dir: Path, workers: int, retries: int
) -> list[dict]:
    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(download_one, record, image_dir, retries): record
            for record in records
        }
        for completed, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results[result["record"].image_id] = result
            if completed % 50 == 0 or completed == len(futures):
                log(f"download batch progress: {completed}/{len(futures)}")
    return [results[record.image_id] for record in records]


def finalize_downloads(
    primary: dict[str, list[Record]],
    backups: dict[str, list[Record]],
    image_dir: Path,
    existing_hashes: set[str],
    workers: int,
    retries: int,
) -> tuple[list[dict], list[dict]]:
    accepted: list[dict] = []
    failures: list[dict] = []
    seen_hashes = set(existing_hashes)

    for group, quota in QUOTAS.items():
        group_accepted: list[dict] = []

        def consume(results: list[dict]) -> None:
            for result in results:
                record: Record = result["record"]
                if not result["ok"]:
                    failures.append({"image_id": record.image_id, "group": group, "reason": result["error"]})
                    continue
                if result["sha256"] in seen_hashes:
                    Path(result["path"]).unlink(missing_ok=True)
                    failures.append({"image_id": record.image_id, "group": group, "reason": "duplicate_sha256"})
                    continue
                if len(group_accepted) >= quota:
                    Path(result["path"]).unlink(missing_ok=True)
                    continue
                seen_hashes.add(result["sha256"])
                group_accepted.append(result)

        log(f"Downloading primary group {group}: {quota}")
        consume(download_records(primary[group], image_dir, workers, retries))
        cursor = 0
        while len(group_accepted) < quota and cursor < len(backups[group]):
            deficit = quota - len(group_accepted)
            batch = backups[group][cursor : cursor + max(16, deficit * 2)]
            cursor += len(batch)
            log(f"{group}: replacing {deficit} failed/duplicate images")
            consume(download_records(batch, image_dir, workers, retries))
        if len(group_accepted) != quota:
            raise RuntimeError(f"{group}: downloaded {len(group_accepted)}, required {quota}")
        accepted.extend(group_accepted)
        log(f"Group complete {group}: {len(group_accepted)}/{quota}")
    return accepted, failures


def yolo_lines(record: Record) -> str:
    lines: list[str] = []
    for class_id, xmin, ymin, xmax, ymax in record.boxes:
        xcenter = (xmin + xmax) / 2
        ycenter = (ymin + ymax) / 2
        width = xmax - xmin
        height = ymax - ymin
        lines.append(
            f"{class_id} {xcenter:.6f} {ycenter:.6f} {width:.6f} {height:.6f}"
        )
    return "\n".join(lines) + ("\n" if lines else "")


def write_dataset(
    repo: Path,
    dataset_root: Path,
    accepted: list[dict],
    failures: list[dict],
    candidate_counts: dict[str, int],
    primary: dict[str, list[Record]],
    backups: dict[str, list[Record]],
    metadata_paths: list[Path],
) -> None:
    image_dir = dataset_root / "train" / "images"
    label_dir = dataset_root / "train" / "labels"
    label_dir.mkdir(parents=True, exist_ok=True)
    accepted_ids = {item["record"].image_id for item in accepted}

    for path in image_dir.glob("*"):
        if path.is_file() and path.stem not in accepted_ids:
            path.unlink()
    for path in label_dir.glob("*.txt"):
        if path.stem not in accepted_ids:
            path.unlink()

    rows: list[dict] = []
    bbox_counts = Counter()
    for item in sorted(accepted, key=lambda value: value["record"].image_id):
        record: Record = item["record"]
        label_path = label_dir / f"{record.image_id}.txt"
        label_path.write_text(yolo_lines(record), encoding="utf-8")
        for box in record.boxes:
            bbox_counts[str(box[0])] += 1
        rows.append(
            {
                "image_id": record.image_id,
                "group": record.group,
                "sha256": item["sha256"],
                "width": item["width"],
                "height": item["height"],
                "bbox_count": len(record.boxes),
            }
        )

    data_yaml = dataset_root / "data.yaml"
    data_yaml.write_text(
        "names:\n"
        "- laptop\n"
        "- mouse\n"
        "nc: 2\n"
        "train: train/images\n"
        "val: ../laptop_mouse_yolov8/valid/images\n"
        "test: ../laptop_mouse_yolov8/test/images\n",
        encoding="utf-8",
    )

    with (dataset_root / "selected_images.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "dataset": "Open Images V7 Mini-1K, train only",
        "seed": 42,
        "total_images": len(rows),
        "quotas": QUOTAS,
        "actual_groups": dict(Counter(row["group"] for row in rows)),
        "target_bbox_counts": dict(bbox_counts),
        "candidate_counts": candidate_counts,
        "metadata": {str(path.relative_to(repo)): path.stat().st_size for path in metadata_paths},
        "existing_validation_and_test_only": "datasets/laptop_mouse_yolov8/{valid,test}",
        "training_images_from_existing_dataset": 0,
    }
    (dataset_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (dataset_root / "selection_plan.json").write_text(
        json.dumps(
            {
                "primary": {group: [record.as_json() for record in records] for group, records in primary.items()},
                "backups": {group: [record.as_json() for record in records] for group, records in backups.items()},
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (dataset_root / "download_failures.json").write_text(
        json.dumps(failures, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    image_count = sum(1 for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    label_count = sum(1 for path in label_dir.glob("*.txt"))
    empty_labels = sum(1 for path in label_dir.glob("*.txt") if path.stat().st_size == 0)
    if image_count != 1000 or label_count != 1000 or empty_labels != 300:
        raise RuntimeError(
            f"final validation failed: images={image_count}, labels={label_count}, empty={empty_labels}"
        )
    (dataset_root / ".ready").write_text("1000\n", encoding="utf-8")
    log(f"DATASET READY: images={image_count}, labels={label_count}, hard_negatives={empty_labels}")


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    metadata_root = repo / ".cache" / "openimages-v7"
    bbox_path = metadata_root / "oidv6-train-annotations-bbox.csv"
    label_path = metadata_root / "oidv7-train-annotations-human-imagelabels.csv"
    class_path = metadata_root / "oidv7-class-descriptions-boxable.csv"
    metadata_paths = [bbox_path, label_path, class_path]
    for path in metadata_paths:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)

    dataset_root = repo / "datasets" / "openimages_mini1k"
    image_dir = dataset_root / "train" / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    target_presence, target_boxes, negative_candidates, animal_bbox = scan_bbox(bbox_path)
    target_label_mask, animal_labels = scan_image_labels(label_path)
    animal_excluded = animal_bbox | animal_labels
    primary, backups, candidate_counts = select_plan(
        target_presence,
        target_boxes,
        negative_candidates,
        animal_excluded,
        target_label_mask,
        args.seed,
    )
    log("Candidate counts: " + json.dumps(candidate_counts, sort_keys=True))
    plan_preview = dataset_root / "selection_plan_pre_download.json"
    plan_preview.write_text(
        json.dumps(
            {
                "quotas": QUOTAS,
                "candidate_counts": candidate_counts,
                "primary": {group: [record.as_json() for record in records] for group, records in primary.items()},
                "backups": {group: [record.as_json() for record in records] for group, records in backups.items()},
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    existing_hashes = existing_dataset_hashes(repo / "datasets" / "laptop_mouse_yolov8")
    accepted, failures = finalize_downloads(
        primary,
        backups,
        image_dir,
        existing_hashes,
        max(1, args.workers),
        max(1, args.download_retries),
    )
    write_dataset(
        repo,
        dataset_root,
        accepted,
        failures,
        candidate_counts,
        primary,
        backups,
        metadata_paths,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
