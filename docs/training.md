# YOLOv8 fine-tuning

## Models

- Initial model: `models/pretrained/yolov8s_coco.pt`
- Final model: `models/finetuned/yolov8s_laptop_mouse_best.pt`

Model binaries are kept outside normal Git history and will be distributed
using Git LFS or a GitHub Release.

## Validate configuration

```bash
python scripts/train_yolov8.py --dry-run
```

## Train

```bash
python scripts/train_yolov8.py
```

## Resume an interrupted run

```bash
python scripts/train_yolov8.py --resume
```

Training outputs are written to `runs/yolov8s/finetune`. The best checkpoint
is copied to the final model path, and `models/model_manifest.json` records
the model hashes and environment versions.
