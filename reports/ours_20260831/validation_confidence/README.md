# Fine-tuned YOLOv8s confidence sweep

This evaluates the fine-tuned laptop/mouse model on the `valid` split.

The model output classes are `0=laptop` and `1=mouse`, matching the dataset
class IDs directly. Detections are matched by
class with an IoU threshold of 0.50. Detection accuracy is defined as
`TP / (TP + FP + FN)`; it is not image-classification accuracy.

Best confidence in this evaluation by micro F1: **0.15**
with F1 **0.9390**, precision **0.9747**,
and recall **0.9059**.

| Confidence | TP | FP | FN | Precision | Recall | F1 | Detection accuracy |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 77 | 4 | 8 | 0.9506 | 0.9059 | 0.9277 | 0.8652 |
| 0.10 | 77 | 3 | 8 | 0.9625 | 0.9059 | 0.9333 | 0.8750 |
| 0.15 | 77 | 2 | 8 | 0.9747 | 0.9059 | 0.9390 | 0.8851 |
| 0.20 | 75 | 2 | 10 | 0.9740 | 0.8824 | 0.9259 | 0.8621 |
| 0.25 | 74 | 1 | 11 | 0.9867 | 0.8706 | 0.9250 | 0.8605 |
| 0.30 | 74 | 1 | 11 | 0.9867 | 0.8706 | 0.9250 | 0.8605 |
| 0.40 | 73 | 0 | 12 | 1.0000 | 0.8588 | 0.9241 | 0.8588 |
| 0.50 | 73 | 0 | 12 | 1.0000 | 0.8588 | 0.9241 | 0.8588 |
| 0.60 | 71 | 0 | 14 | 1.0000 | 0.8353 | 0.9103 | 0.8353 |
| 0.70 | 71 | 0 | 14 | 1.0000 | 0.8353 | 0.9103 | 0.8353 |
| 0.80 | 68 | 0 | 17 | 1.0000 | 0.8000 | 0.8889 | 0.8000 |
| 0.90 | 68 | 0 | 17 | 1.0000 | 0.8000 | 0.8889 | 0.8000 |

The test split is intentionally not used to select the confidence threshold.
