# YOLOv8s smoke test

This report records the one-epoch GPU smoke test before formal fine-tuning.

## Setup

- Initial weights: COCO-pretrained YOLOv8s
- Dataset: laptop and mouse, YOLOv8 format
- Training images: 144
- Validation images: 41
- Test images: 20
- Input size: 512
- Batch size: 16
- Epochs: 1
- Device: NVIDIA GeForce RTX 4090
- Seed: 42

## Validation result

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| all | 41 | 85 | 0.856 | 0.683 | 0.744 | 0.699 |
| laptop | 24 | 31 | 0.741 | 0.739 | 0.678 | 0.664 |
| mouse | 32 | 54 | 0.971 | 0.628 | 0.810 | 0.735 |

The smoke test confirms that the dataset, CUDA environment, model loading,
training loop, validation loop, and artifact generation work correctly.

Smoke-test weights are intentionally excluded because they are disposable.
The original pretrained model and final fine-tuned model will be distributed
separately as versioned model artifacts.
