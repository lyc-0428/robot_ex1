#!/usr/bin/env python3
"""Evaluate a fine-tuned YOLOv8 laptop/mouse model at fixed confidences."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import ultralytics
import yaml
from ultralytics import YOLO


MODEL_TO_DATASET = {0: 0, 1: 1}
DATASET_NAMES = {0: "laptop", 1: "mouse"}
DEFAULT_THRESHOLDS = "0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50,0.60,0.70,0.80,0.90"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument("--thresholds", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--output", type=Path, default=Path("reports/validation_finetuned_yolov8s")
    )
    parser.add_argument(
        "--cache", type=Path, default=Path(".cache/finetuned_evaluation")
    )
    return parser.parse_args()


def parse_thresholds(value: str) -> list[float]:
    thresholds = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    if not thresholds or thresholds[0] <= 0 or thresholds[-1] >= 1:
        raise ValueError("thresholds must be between 0 and 1")
    return thresholds


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def link_or_copy(source: Path, destination: Path) -> None:
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def build_finetuned_dataset(
    dataset_root: Path, split: str, cache_root: Path, model_names: dict[int, str]
) -> Path:
    source_images = dataset_root / split / "images"
    source_labels = dataset_root / split / "labels"
    if not source_images.is_dir() or not source_labels.is_dir():
        raise FileNotFoundError(f"missing split directories under {dataset_root / split}")

    mapped_images = cache_root / split / "images"
    mapped_labels = cache_root / split / "labels"
    mapped_images.mkdir(parents=True, exist_ok=True)
    mapped_labels.mkdir(parents=True, exist_ok=True)

    images = sorted(
        path
        for path in source_images.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise RuntimeError(f"no images found in {source_images}")

    for image in images:
        label = source_labels / f"{image.stem}.txt"
        if not label.is_file():
            raise FileNotFoundError(f"missing label for {image.name}")
        link_or_copy(image, mapped_images / image.name)

        mapped_lines: list[str] = []
        for line_number, line in enumerate(label.read_text(encoding="utf-8-sig").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f"{label}:{line_number}: expected 5 fields")
            dataset_class = int(fields[0])
            if dataset_class not in DATASET_NAMES:
                raise ValueError(f"{label}:{line_number}: invalid class {dataset_class}")
            model_class = dataset_class
            mapped_lines.append(" ".join((str(model_class), *fields[1:])))

        text = "\n".join(mapped_lines)
        if text:
            text += "\n"
        (mapped_labels / label.name).write_text(text, encoding="utf-8")

    for cache_file in cache_root.rglob("*.cache"):
        cache_file.unlink()

    yaml_path = cache_root / f"data_{split}_finetuned_ids.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "path": str(cache_root.resolve()),
                "train": f"{split}/images",
                "val": f"{split}/images",
                "test": f"{split}/images",
                "nc": len(model_names),
                "names": model_names,
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return yaml_path


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def class_counts(matrix: np.ndarray, class_id: int) -> dict[str, float | int]:
    true_positive = int(round(float(matrix[class_id, class_id])))
    false_positive = int(round(float(matrix[class_id, :].sum() - matrix[class_id, class_id])))
    false_negative = int(round(float(matrix[:, class_id].sum() - matrix[class_id, class_id])))
    precision = safe_ratio(true_positive, true_positive + false_positive)
    recall = safe_ratio(true_positive, true_positive + false_negative)
    f1 = safe_ratio(2 * precision * recall, precision + recall)
    accuracy = safe_ratio(true_positive, true_positive + false_positive + false_negative)
    return {
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "detection_accuracy": accuracy,
    }


def evaluate_thresholds(
    model: YOLO, data_yaml: Path, args: argparse.Namespace, thresholds: list[float]
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    validation_runs = args.cache / "runs"

    for threshold in thresholds:
        print(f"\n=== confidence={threshold:.2f} ===", flush=True)
        metrics = model.val(
            data=str(data_yaml),
            split="val",
            conf=threshold,
            iou=args.iou,
            classes=list(MODEL_TO_DATASET),
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            plots=True,
            save=False,
            save_json=False,
            project=str(validation_runs.resolve()),
            name=f"{args.split}_conf_{threshold:.2f}",
            exist_ok=True,
            verbose=False,
        )

        confusion = getattr(metrics, "confusion_matrix", None)
        if confusion is None:
            raise RuntimeError("Ultralytics did not return a confusion matrix")
        matrix = np.asarray(confusion.matrix, dtype=float)

        laptop = class_counts(matrix, 0)
        mouse = class_counts(matrix, 1)
        total_tp = int(laptop["tp"] + mouse["tp"])
        total_fp = int(laptop["fp"] + mouse["fp"])
        total_fn = int(laptop["fn"] + mouse["fn"])
        precision = safe_ratio(total_tp, total_tp + total_fp)
        recall = safe_ratio(total_tp, total_tp + total_fn)
        f1 = safe_ratio(2 * precision * recall, precision + recall)
        accuracy = safe_ratio(total_tp, total_tp + total_fp + total_fn)

        row: dict[str, float | int] = {
            "confidence": threshold,
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "detection_accuracy": accuracy,
            "map50": float(metrics.box.map50),
            "map50_95": float(metrics.box.map),
        }
        for name, values in (("laptop", laptop), ("mouse", mouse)):
            for key, value in values.items():
                row[f"{name}_{key}"] = value
        rows.append(row)
        print(
            f"TP={total_tp} FP={total_fp} FN={total_fn} "
            f"P={precision:.4f} R={recall:.4f} F1={f1:.4f} accuracy={accuracy:.4f}",
            flush=True,
        )
    return rows


def write_outputs(
    rows: list[dict[str, float | int]], args: argparse.Namespace, model_path: Path
) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / f"{args.split}_confidence_sweep.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    best = max(rows, key=lambda row: (float(row["f1"]), float(row["detection_accuracy"])))
    metadata = {
        "model": str(model_path.resolve()),
        "model_sha256": sha256(model_path),
        "dataset": str(args.dataset.resolve()),
        "split": args.split,
        "matching_iou": args.iou,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "model_to_dataset_class_mapping": {"0": 0, "1": 1},
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "ultralytics": ultralytics.__version__,
        "best_by_f1": best,
    }
    (args.output / f"{args.split}_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    x = [float(row["confidence"]) for row in rows]
    figure, axis = plt.subplots(figsize=(9, 6))
    for key, label in (
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1"),
        ("detection_accuracy", "TP / (TP + FP + FN)"),
    ):
        axis.plot(x, [float(row[key]) for row in rows], marker="o", label=label)
    axis.axvline(float(best["confidence"]), color="black", linestyle="--", alpha=0.5)
    axis.set(xlabel="Confidence threshold", ylabel="Metric", ylim=(0, 1.02))
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(args.output / f"{args.split}_confidence_sweep.png", dpi=180)
    plt.close(figure)

    table_lines = [
        "| Confidence | TP | FP | FN | Precision | Recall | F1 | Detection accuracy |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        table_lines.append(
            "| {confidence:.2f} | {tp} | {fp} | {fn} | {precision:.4f} | "
            "{recall:.4f} | {f1:.4f} | {detection_accuracy:.4f} |".format(**row)
        )
    markdown_table = "\n".join(table_lines)
    readme = f"""# Fine-tuned YOLOv8s confidence sweep

This evaluates the fine-tuned laptop/mouse model on the `{args.split}` split.

The model output classes are `0=laptop` and `1=mouse`, matching the dataset
class IDs directly. Detections are matched by
class with an IoU threshold of {args.iou:.2f}. Detection accuracy is defined as
`TP / (TP + FP + FN)`; it is not image-classification accuracy.

Best confidence in this evaluation by micro F1: **{float(best['confidence']):.2f}**
with F1 **{float(best['f1']):.4f}**, precision **{float(best['precision']):.4f}**,
and recall **{float(best['recall']):.4f}**.

{markdown_table}

The test split is intentionally not used to select the confidence threshold.
"""
    (args.output / "README.md").write_text(readme, encoding="utf-8")
    print(f"\nBest threshold: {float(best['confidence']):.2f}")
    print(f"Wrote {csv_path}")


def main() -> int:
    args = parse_args()
    args.model = args.model.resolve()
    args.dataset = args.dataset.resolve()
    args.output = args.output.resolve()
    args.cache = args.cache.resolve()
    thresholds = parse_thresholds(args.thresholds)

    model = YOLO(str(args.model))
    names = dict(model.names)
    if names.get(0) != "laptop" or names.get(1) != "mouse":
        raise RuntimeError("model does not have the expected fine-tuned classes")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    data_yaml = build_finetuned_dataset(args.dataset, args.split, args.cache, names)
    rows = evaluate_thresholds(model, data_yaml, args, thresholds)
    write_outputs(rows, args, args.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
