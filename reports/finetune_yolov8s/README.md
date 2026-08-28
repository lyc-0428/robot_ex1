# YOLOv8s fine-tuning report

## Training setup

- Dataset: `datasets/laptop_mouse_yolov8`
- Classes: `laptop`, `mouse`
- Initial model: `models/pretrained/yolov8s_coco.pt`
- Configuration: `configs/yolov8s_finetune.yaml`
- Maximum epochs: 120
- Best epoch: 81
- Early stopping: epoch 106, patience 25
- Training exit code: 0
- GPU: NVIDIA GeForce RTX 4090

## Validation results

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| All | 0.965 | 0.887 | 0.916 | 0.865 |
| laptop | 0.964 | 0.866 | 0.872 | 0.829 |
| mouse | 0.965 | 0.907 | 0.959 | 0.901 |

These are validation-set results. The independent test split has not yet
been evaluated.

## Model artifacts

- Pretrained: `models/pretrained/yolov8s_coco.pt`
- Fine-tuned: `models/finetuned/yolov8s_laptop_mouse_best.pt`
- Pretrained SHA-256: `268e5bb54c640c96c3510224833bc2eeacab4135c6deb41502156e39986b562d`
- Fine-tuned SHA-256: `13f6c4266f818815ff85306ac5d5dae752a502a46335bc4fe560ef4c4f99f421`

Periodic training checkpoints remain local under
`runs/yolov8s/finetune/weights/` and are intentionally excluded from Git.
