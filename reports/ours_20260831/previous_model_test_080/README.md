# Fine-tuned YOLOv8s confidence sweep

This evaluates the fine-tuned laptop/mouse model on the `test` split.

The model output classes are `0=laptop` and `1=mouse`, matching the dataset
class IDs directly. Detections are matched by
class with an IoU threshold of 0.50. Detection accuracy is defined as
`TP / (TP + FP + FN)`; it is not image-classification accuracy.

Best confidence in this evaluation by micro F1: **0.80**
with F1 **0.9538**, precision **0.9688**,
and recall **0.9394**.

| Confidence | TP | FP | FN | Precision | Recall | F1 | Detection accuracy |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.80 | 31 | 1 | 2 | 0.9688 | 0.9394 | 0.9538 | 0.9118 |

The test split is intentionally not used to select the confidence threshold.
