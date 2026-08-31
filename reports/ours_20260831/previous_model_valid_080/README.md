# Fine-tuned YOLOv8s confidence sweep

This evaluates the fine-tuned laptop/mouse model on the `valid` split.

The model output classes are `0=laptop` and `1=mouse`, matching the dataset
class IDs directly. Detections are matched by
class with an IoU threshold of 0.50. Detection accuracy is defined as
`TP / (TP + FP + FN)`; it is not image-classification accuracy.

Best confidence in this evaluation by micro F1: **0.80**
with F1 **0.8961**, precision **1.0000**,
and recall **0.8118**.

| Confidence | TP | FP | FN | Precision | Recall | F1 | Detection accuracy |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.80 | 69 | 0 | 16 | 1.0000 | 0.8118 | 0.8961 | 0.8118 |

The test split is intentionally not used to select the confidence threshold.
