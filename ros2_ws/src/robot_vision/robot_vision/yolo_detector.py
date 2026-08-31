"""Publish YOLOv8 detections and annotated camera frames over ROS 2."""

import json
import time
from pathlib import Path

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from ultralytics import YOLO


class YoloDetector(Node):
    """Run GPU inference on the latest ROS camera frame."""

    def __init__(self):
        super().__init__("yolo_detector")

        default_model = (
            Path.home()
            / "Team21"
            / "lyc"
            / "robot_ex1-main"
            / "models"
            / "finetuned"
            / "yolov8s_laptop_mouse_best.pt"
        )
        self.declare_parameter("model_path", str(default_model))
        self.declare_parameter("confidence", 0.80)
        self.declare_parameter("image_size", 512)
        self.declare_parameter("device", 0)

        self.model_path = self.get_parameter("model_path").value
        self.confidence = self.get_parameter("confidence").value
        self.image_size = self.get_parameter("image_size").value
        self.device = self.get_parameter("device").value

        self.get_logger().info(f"Loading model: {self.model_path}")
        self.model = YOLO(self.model_path)
        self.bridge = CvBridge()

        camera_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.image_subscriber = self.create_subscription(
            Image,
            "/camera/image_raw",
            self.image_callback,
            camera_qos,
        )
        self.image_publisher = self.create_publisher(
            Image, "/detections/image", 10
        )
        self.detection_publisher = self.create_publisher(
            String, "/detections", 10
        )

        self.smoothed_fps = 0.0
        self.frame_count = 0

        self.get_logger().info("YOLO detector is ready")
        self.get_logger().info("Subscribing: /camera/image_raw")
        self.get_logger().info("Publishing: /detections")
        self.get_logger().info("Publishing: /detections/image")

    def image_callback(self, image_message):
        """Detect objects in one camera frame and publish the result."""
        start_time = time.perf_counter()

        try:
            frame = self.bridge.imgmsg_to_cv2(
                image_message, desired_encoding="bgr8"
            )
            result = self.model.predict(
                source=frame,
                imgsz=self.image_size,
                conf=self.confidence,
                device=self.device,
                verbose=False,
            )[0]

            detections = []
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()

                for box, score, class_number in zip(
                    boxes, scores, classes
                ):
                    class_id = int(class_number)
                    x1, y1, x2, y2 = [float(value) for value in box]
                    detections.append(
                        {
                            "class_id": class_id,
                            "class_name": result.names[class_id],
                            "confidence": round(float(score), 4),
                            "bbox": {
                                "x1": round(x1, 1),
                                "y1": round(y1, 1),
                                "x2": round(x2, 1),
                                "y2": round(y2, 1),
                            },
                        }
                    )

            elapsed = time.perf_counter() - start_time
            current_fps = 1.0 / elapsed if elapsed > 0 else 0.0
            if self.smoothed_fps == 0.0:
                self.smoothed_fps = current_fps
            else:
                self.smoothed_fps = (
                    0.9 * self.smoothed_fps + 0.1 * current_fps
                )

            annotated_frame = result.plot()
            cv2.putText(
                annotated_frame,
                f"FPS: {self.smoothed_fps:.2f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )

            detection_message = String()
            detection_message.data = json.dumps(
                {
                    "frame": self.frame_count,
                    "fps": round(self.smoothed_fps, 2),
                    "count": len(detections),
                    "detections": detections,
                },
                ensure_ascii=False,
            )
            self.detection_publisher.publish(detection_message)

            output_image = self.bridge.cv2_to_imgmsg(
                annotated_frame, encoding="bgr8"
            )
            output_image.header = image_message.header
            self.image_publisher.publish(output_image)

            self.frame_count += 1
            if self.frame_count % 30 == 0:
                self.get_logger().info(
                    f"frame={self.frame_count}, "
                    f"objects={len(detections)}, "
                    f"fps={self.smoothed_fps:.2f}"
                )
        except Exception as error:  # keep the live stream alive on bad frames
            self.get_logger().error(f"Detection failed: {error}")


def main(args=None):
    """Start the ROS 2 detector node."""
    rclpy.init(args=args)
    node = YoloDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
