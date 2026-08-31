# Fine-tuned YOLOv8s confidence sweep

This evaluates the fine-tuned laptop/mouse model on the `test` split.

The model output classes are `0=laptop` and `1=mouse`, matching the dataset
class IDs directly. Detections are matched by
class with an IoU threshold of 0.50. Detection accuracy is defined as
`TP / (TP + FP + FN)`; it is not image-classification accuracy.

Best confidence in this evaluation by micro F1: **0.75**
with F1 **0.9688**, precision **1.0000**,
and recall **0.9394**.

| Confidence | TP | FP | FN | Precision | Recall | F1 | Detection accuracy |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 31 | 3 | 2 | 0.9118 | 0.9394 | 0.9254 | 0.8611 |
| 0.55 | 31 | 3 | 2 | 0.9118 | 0.9394 | 0.9254 | 0.8611 |
| 0.60 | 31 | 3 | 2 | 0.9118 | 0.9394 | 0.9254 | 0.8611 |
| 0.65 | 31 | 3 | 2 | 0.9118 | 0.9394 | 0.9254 | 0.8611 |
| 0.70 | 31 | 1 | 2 | 0.9688 | 0.9394 | 0.9538 | 0.9118 |
| 0.75 | 31 | 0 | 2 | 1.0000 | 0.9394 | 0.9688 | 0.9394 |
| 0.80 | 30 | 0 | 3 | 1.0000 | 0.9091 | 0.9524 | 0.9091 |
| 0.85 | 30 | 0 | 3 | 1.0000 | 0.9091 | 0.9524 | 0.9091 |
| 0.90 | 26 | 0 | 7 | 1.0000 | 0.7879 | 0.8814 | 0.7879 |
| 0.95 | 14 | 0 | 19 | 1.0000 | 0.4242 | 0.5957 | 0.4242 |
| 1.00 | 0 | 0 | 33 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

The test split is intentionally not used to select the confidence threshold.
