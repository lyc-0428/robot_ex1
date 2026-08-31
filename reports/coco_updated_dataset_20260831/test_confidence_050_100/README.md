# Pretrained YOLOv8s confidence sweep

This is the pre-fine-tuning baseline on the `test` split.

The COCO output classes `63=laptop` and `64=mouse` are remapped to dataset
classes `0=laptop` and `1=mouse` before evaluation. Detections are matched by
class with an IoU threshold of 0.50. Detection accuracy is defined as
`TP / (TP + FP + FN)`; it is not image-classification accuracy.

Best validation confidence by micro F1: **0.50**
with F1 **0.7931**, precision **0.9200**,
and recall **0.6970**.

| Confidence | TP | FP | FN | Precision | Recall | F1 | Detection accuracy |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 23 | 2 | 10 | 0.9200 | 0.6970 | 0.7931 | 0.6571 |
| 0.55 | 22 | 2 | 11 | 0.9167 | 0.6667 | 0.7719 | 0.6286 |
| 0.60 | 21 | 2 | 12 | 0.9130 | 0.6364 | 0.7500 | 0.6000 |
| 0.65 | 21 | 0 | 12 | 1.0000 | 0.6364 | 0.7778 | 0.6364 |
| 0.70 | 18 | 0 | 15 | 1.0000 | 0.5455 | 0.7059 | 0.5455 |
| 0.75 | 18 | 0 | 15 | 1.0000 | 0.5455 | 0.7059 | 0.5455 |
| 0.80 | 14 | 0 | 19 | 1.0000 | 0.4242 | 0.5957 | 0.4242 |
| 0.85 | 14 | 0 | 19 | 1.0000 | 0.4242 | 0.5957 | 0.4242 |
| 0.90 | 9 | 0 | 24 | 1.0000 | 0.2727 | 0.4286 | 0.2727 |
| 0.95 | 5 | 0 | 28 | 1.0000 | 0.1515 | 0.2632 | 0.1515 |
| 1.00 | 0 | 0 | 33 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

The test split is intentionally not used to select the confidence threshold.
