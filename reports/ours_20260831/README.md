# `ours` YOLOv8s training report

## Result

- Dataset commit: `e405c57` (`Add hard-negative dataset samples`)
- Initial model: `models/pretrained/yolov8s_coco.pt`
- Final model: `models/ours/yolov8s_ours.pt`
- Final model SHA-256: `591442fa211f1d752401f675b924a740e1eddd5de50d4228cc4e73d2f377b87e`
- Classes: `laptop`, `mouse`
- GPU: NVIDIA GeForce RTX 4090
- Training: 120 epochs, best epoch 96, 512 px, batch 16, AdamW

## Updated dataset

| Split | Images | Empty-label hard negatives |
|---|---:|---:|
| Train | 163 | 21 |
| Validation | 45 | 4 |
| Test | 24 | 5 |

The newly pushed camera-domain hard-negative images are included. Existing
Ultralytics photometric augmentation is retained; source images are not
permanently filtered to a single camera appearance because deployment uses
automatic exposure and automatic white balance.

## Best-checkpoint validation

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| All | 0.970 | 0.890 | 0.905 | 0.858 |
| laptop | 0.963 | 0.836 | 0.853 | 0.814 |
| mouse | 0.978 | 0.944 | 0.956 | 0.902 |

## Deployment-threshold comparison on the updated test split

Confidence is fixed at `0.80`, matching the ROS 2 detector default.

| Model | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Previous fine-tuned model | 31 | 1 | 2 | 0.9688 | 0.9394 | 0.9538 |
| `ours` | 30 | 0 | 3 | 1.0000 | 0.9091 | 0.9524 |

The updated model removes the observed test false positive at the deployment
threshold, with one additional false negative and essentially unchanged F1.
The full validation confidence sweep peaks at confidence `0.15`, but the
deployment threshold remains `0.80` because it is more appropriate for the
project's false-positive-sensitive camera use case and produced zero test
false positives.

## Reproducibility

- Training configuration: `configs/yolov8s_ours.yaml`
- Training log: `reports/ours_20260831/training.log`
- Training run: `runs/yolov8s/ours_20260831`
- Model manifest: `models/ours/manifest.json`
- Validation confidence sweep: `reports/ours_20260831/validation_confidence`
- Test confidence results: `reports/ours_20260831/test_confidence`
