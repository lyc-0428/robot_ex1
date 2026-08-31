
Laptop and Mouse Detection - v2 YOLOv8 baseline v2
==============================

This dataset was exported via roboflow.com on August 27, 2026 at 9:09 AM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 232 images.
Laptop-and-Mouse-Objects are annotated in YOLOv8 format.

On August 31, 2026, 27 target-free hard-negative images were added to reduce
false-positive detections. They are represented by empty YOLO label files and
were kept in capture-sequence groups when assigned to train (19), validation
(4), and test (4) splits.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 512x512 (Stretch)

No image augmentation techniques were applied.


