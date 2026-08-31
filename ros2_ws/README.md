# ROS 2 Jetson deployment

The `robot_vision` package subscribes to `/camera/image_raw`, runs the
fine-tuned laptop/mouse YOLOv8 model, and publishes:

- `/detections` (`std_msgs/msg/String`): JSON detections with class,
  confidence, box coordinates and processing FPS.
- `/detections/image` (`sensor_msgs/msg/Image`): annotated video frames.

The default detection confidence is `0.80`.

## Build

Copy or link `src/robot_vision` into `~/robot_ws/src`, then run:

```bash
cd ~/robot_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select robot_vision
source install/setup.bash
```

## Run

Start the camera publisher first, then run:

```bash
ros2 run robot_vision yolo_detector
```

Override the model path when the repository is stored elsewhere:

```bash
ros2 run robot_vision yolo_detector --ros-args \
  -p model_path:=/absolute/path/to/yolov8s_laptop_mouse_best.pt
```

Display the annotated stream on the Jetson desktop:

```bash
ros2 run image_view image_view --ros-args -r image:=/detections/image
```
