# Pretrained YOLOv8s confidence sweep

This is the pre-fine-tuning baseline on the `test` split.

The COCO output classes `63=laptop` and `64=mouse` are remapped to dataset
classes `0=laptop` and `1=mouse` before evaluation. Detections are matched by
class with an IoU threshold of 0.50. Detection accuracy is defined as
`TP / (TP + FP + FN)`; it is not image-classification accuracy.

Best validation confidence by micro F1: **0.15**
with F1 **0.7619**, precision **0.8000**,
and recall **0.7273**.

| Confidence | TP | FP | FN | Precision | Recall | F1 | Detection accuracy |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.15 | 24 | 6 | 9 | 0.8000 | 0.7273 | 0.7619 | 0.6154 |

The test split is intentionally not used to select the confidence threshold.
