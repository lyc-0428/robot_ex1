# YOLOv8s final model comparison

## Evaluation protocol

The confidence threshold for each model was selected only on the validation
split:

- COCO-pretrained YOLOv8s: `0.15`
- Fine-tuned YOLOv8s: `0.20`

These thresholds were then locked before evaluating the independent test
split. The test split contains 20 images and 33 annotated objects: 8 laptops
and 25 mice. Detection TP, FP and FN use class-aware matching at IoU 0.50.

## Overall test results

| Model | Confidence | TP | FP | FN | Precision | Recall | F1 | Detection accuracy | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Pretrained | 0.15 | 24 | 6 | 9 | 0.8000 | 0.7273 | 0.7619 | 0.6154 | 0.8611 | 0.8198 |
| Fine-tuned | 0.20 | 31 | 3 | 2 | 0.9118 | 0.9394 | 0.9254 | 0.8611 | 0.9110 | 0.8529 |

## Absolute improvement after fine-tuning

| Metric | Improvement |
|---|---:|
| Precision | +0.1118 |
| Recall | +0.2121 |
| F1 | +0.1635 |
| Detection accuracy | +0.2457 |
| mAP50 | +0.0499 |
| mAP50-95 | +0.0331 |

## Per-class test results

| Model | Class | TP | FP | FN | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Pretrained | laptop | 7 | 4 | 1 | 0.6364 | 0.8750 | 0.7368 |
| Pretrained | mouse | 17 | 2 | 8 | 0.8947 | 0.6800 | 0.7727 |
| Fine-tuned | laptop | 7 | 1 | 1 | 0.8750 | 0.8750 | 0.8750 |
| Fine-tuned | mouse | 24 | 2 | 1 | 0.9231 | 0.9600 | 0.9412 |

Detection accuracy is `TP / (TP + FP + FN)`, not image-classification
accuracy. The mAP values shown here are produced after applying each locked
operating confidence threshold; they are directly comparable under this
protocol, but are distinct from a full low-confidence PR-curve evaluation.

Because the test set is small, these results should be confirmed with more
independent images before deployment.
