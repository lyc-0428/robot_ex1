# Pretrained YOLOv8s confidence sweep

This is the pre-fine-tuning baseline on the `valid` split.

The COCO output classes `63=laptop` and `64=mouse` are remapped to dataset
classes `0=laptop` and `1=mouse` before evaluation. Detections are matched by
class with an IoU threshold of 0.50. Detection accuracy is defined as
`TP / (TP + FP + FN)`; it is not image-classification accuracy.

Best validation confidence by micro F1: **0.15**
with F1 **0.7067**, precision **0.8154**,
and recall **0.6235**.

| Confidence | TP | FP | FN | Precision | Recall | F1 | Detection accuracy |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 54 | 32 | 31 | 0.6279 | 0.6353 | 0.6316 | 0.4615 |
| 0.10 | 53 | 17 | 32 | 0.7571 | 0.6235 | 0.6839 | 0.5196 |
| 0.15 | 53 | 12 | 32 | 0.8154 | 0.6235 | 0.7067 | 0.5464 |
| 0.20 | 50 | 11 | 35 | 0.8197 | 0.5882 | 0.6849 | 0.5208 |
| 0.25 | 48 | 9 | 37 | 0.8421 | 0.5647 | 0.6761 | 0.5106 |
| 0.30 | 48 | 7 | 37 | 0.8727 | 0.5647 | 0.6857 | 0.5217 |
| 0.40 | 44 | 6 | 41 | 0.8800 | 0.5176 | 0.6519 | 0.4835 |
| 0.50 | 40 | 4 | 45 | 0.9091 | 0.4706 | 0.6202 | 0.4494 |
| 0.60 | 39 | 3 | 46 | 0.9286 | 0.4588 | 0.6142 | 0.4432 |
| 0.70 | 32 | 3 | 53 | 0.9143 | 0.3765 | 0.5333 | 0.3636 |
| 0.80 | 26 | 2 | 59 | 0.9286 | 0.3059 | 0.4602 | 0.2989 |
| 0.90 | 20 | 2 | 65 | 0.9091 | 0.2353 | 0.3738 | 0.2299 |

The test split is intentionally not used to select the confidence threshold.
