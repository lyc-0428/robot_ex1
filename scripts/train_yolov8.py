#!/usr/bin/env python3
"""Run or validate the reproducible YOLOv8 fine-tuning configuration."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch
import ultralytics
import yaml
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/yolov8s_finetune.yaml")
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="validate without starting training"
    )
    parser.add_argument(
        "--resume", action="store_true", help="resume from the run's last.pt"
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_names(names: list[str] | dict[int, str]) -> list[str]:
    if isinstance(names, list):
        return names
    return [names[key] for key in sorted(names, key=int)]


def load_configuration(config_path: Path, root: Path) -> dict:
    if not config_path.is_absolute():
        config_path = root / config_path
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("training configuration must be a YAML mapping")

    required = {"model", "data", "project", "name", "epochs", "device"}
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"missing configuration keys: {missing}")

    for key in ("model", "data", "project"):
        value = Path(config[key])
        if not value.is_absolute():
            value = root / value
        config[key] = str(value.resolve())
    config["_config_path"] = str(config_path)
    return config


def validate_inputs(config: dict) -> None:
    model_path = Path(config["model"])
    data_path = Path(config["data"])
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    names = normalize_names(data["names"])
    if names != ["laptop", "mouse"] or int(data["nc"]) != 2:
        raise ValueError(f"unexpected dataset classes: {names}")
    for split_key in ("train", "val", "test"):
        split_path = (data_path.parent / data[split_key]).resolve()
        if not split_path.is_dir():
            raise FileNotFoundError(f"{split_key}: {split_path}")


def main() -> int:
    args = parse_args()
    if args.dry_run and args.resume:
        raise ValueError("--dry-run and --resume cannot be used together")

    root = Path(__file__).resolve().parents[1]
    config = load_configuration(args.config, root)
    config_path = Path(config.pop("_config_path"))
    validate_inputs(config)

    model_path = Path(config["model"])
    data_path = Path(config["data"])
    project_path = Path(config["project"])
    output_dir = project_path / str(config["name"])

    print("Repository:", root)
    print("Configuration:", config_path)
    print("PyTorch:", torch.__version__)
    print("CUDA runtime:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))
    print("Resolved training arguments:")
    print(json.dumps(config, indent=2, ensure_ascii=False))

    if args.dry_run:
        model = YOLO(str(model_path))
        if model.names.get(63) != "laptop" or model.names.get(64) != "mouse":
            raise RuntimeError("unexpected pretrained COCO class mapping")
        print("DRY RUN PASSED: no training was started")
        return 0

    project_path.mkdir(parents=True, exist_ok=True)
    if args.resume:
        last_path = output_dir / "weights" / "last.pt"
        if not last_path.is_file():
            raise FileNotFoundError(f"resume checkpoint not found: {last_path}")
        model = YOLO(str(last_path))
        model.train(resume=True)
    else:
        if output_dir.exists():
            raise FileExistsError(
                f"run directory already exists: {output_dir}; use --resume or change name"
            )
        model = YOLO(str(model_path))
        model.train(**config)

    best_path = output_dir / "weights" / "best.pt"
    if not best_path.is_file():
        raise FileNotFoundError(f"training completed without best.pt: {best_path}")

    final_dir = root / "models" / "finetuned"
    final_dir.mkdir(parents=True, exist_ok=True)
    final_path = final_dir / "yolov8s_laptop_mouse_best.pt"
    shutil.copy2(best_path, final_path)

    final_model = YOLO(str(final_path))
    final_names = normalize_names(final_model.names)
    if final_names != ["laptop", "mouse"]:
        raise RuntimeError(f"unexpected fine-tuned classes: {final_names}")

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pretrained_model": str(model_path.relative_to(root)),
        "pretrained_sha256": sha256(model_path),
        "finetuned_model": str(final_path.relative_to(root)),
        "finetuned_sha256": sha256(final_path),
        "training_config": str(config_path.relative_to(root)),
        "dataset_yaml": str(data_path.relative_to(root)),
        "classes": final_names,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "ultralytics": ultralytics.__version__,
        "run_directory": str(output_dir.relative_to(root)),
    }
    manifest_path = root / "models" / "model_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("Fine-tuned model:", final_path)
    print("Manifest:", manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
