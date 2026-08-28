# Fine-tuned YOLOv8s confidence sweep

This evaluates the fine-tuned laptop/mouse model on the `valid` split.

The model output classes are `0=laptop` and `1=mouse`, matching the dataset
class IDs directly. Detections are matched by
class with an IoU threshold of 0.50. Detection accuracy is defined as
`TP / (TP + FP + FN)`; it is not image-classification accuracy.

Best confidence in this evaluation by micro F1: **0.20**
with F1 **0.9325**, precision **0.9744**,
and recall **0.8941**.

| Confidence | TP | FP | FN | Precision | Recall | F1 | Detection accuracy |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 77 | 9 | 8 | 0.8953 | 0.9059 | 0.9006 | 0.8191 |
| 0.10 | 77 | 4 | 8 | 0.9506 | 0.9059 | 0.9277 | 0.8652 |
| 0.15 | 76 | 3 | 9 | 0.9620 | 0.8941 | 0.9268 | 0.8636 |
| 0.20 | 76 | 2 | 9 | 0.9744 | 0.8941 | 0.9325 | 0.8736 |
| 0.25 | 75 | 2 | 10 | 0.9740 | 0.8824 | 0.9259 | 0.8621 |
| 0.30 | 75 | 2 | 10 | 0.9740 | 0.8824 | 0.9259 | 0.8621 |
| 0.40 | 75 | 2 | 10 | 0.9740 | 0.8824 | 0.9259 | 0.8621 |
| 0.50 | 74 | 2 | 11 | 0.9737 | 0.8706 | 0.9193 | 0.8506 |
| 0.60 | 73 | 0 | 12 | 1.0000 | 0.8588 | 0.9241 | 0.8588 |
| 0.70 | 71 | 0 | 14 | 1.0000 | 0.8353 | 0.9103 | 0.8353 |
| 0.80 | 69 | 0 | 16 | 1.0000 | 0.8118 | 0.8961 | 0.8118 |
| 0.90 | 62 | 0 | 23 | 1.0000 | 0.7294 | 0.8435 | 0.7294 |

The test split is intentionally not used to select the confidence threshold.
