#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import numpy as np

COLOUR_RANGES = [
    ("orange", np.array([0, 120, 70]), np.array([25, 255, 255]),
               np.array([160, 120, 70]), np.array([180, 255, 255])),
    ("red",    np.array([0, 100, 70]), np.array([10, 255, 255]),
               np.array([170, 100, 70]), np.array([180, 255, 255])),
    ("yellow", np.array([20, 100, 70]), np.array([35, 255, 255]), None, None),
    ("green",  np.array([36, 60, 40]), np.array([85, 255, 255]), None, None),
    ("blue",   np.array([90, 60, 40]), np.array([130, 255, 255]), None, None),
]

def detect_colour(hsv_roi):
    best_colour = "unknown"
    best_count = 50
    for name, lo1, hi1, lo2, hi2 in COLOUR_RANGES:
        mask = cv2.inRange(hsv_roi, lo1, hi1)
        if lo2 is not None:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv_roi, lo2, hi2))
        count = cv2.countNonZero(mask)
        if count > best_count:
            best_count = count
            best_colour = name
        if best_colour == "red" or best_colour == "yellow":
            print("Area of interest found")
    return best_colour

def detect_shape(contour):
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0: return "unknown"
    approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
    sides = len(approx)
    area = cv2.contourArea(contour)
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
    if sides == 3: return "triangle"
    elif sides == 4:
        x, y, w, h = cv2.boundingRect(approx)
        return "square" if 0.85 <= w/h <= 1.15 else "rectangle"
    elif circularity > 0.75: return "circle"
    else: return f"poly({sides})"

def is_cone_shaped(contour):
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
    sides = len(approx)
    x, y, w, h = cv2.boundingRect(contour)
    aspect = h / float(w) if w > 0 else 0
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    solidity = cv2.contourArea(contour) / hull_area if hull_area > 0 else 0
    return sides <= 5 and aspect > 1.1 and solidity > 0.6

class ConeDetector(Node):
    def __init__(self):
        super().__init__('cone_detector')
        self.bridge = CvBridge()
        self.latest_depth = None

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.create_subscription(Image, '/oak/rgb/image_raw', self.rgb_callback, qos)
        self.create_subscription(Image, '/oak/stereo/image_raw', self.depth_callback, qos)

        self.mask_pub = self.create_publisher(Image, '/cone_mask', 10)
        self.get_logger().info("OAK-D Cone Detector Started")

    def depth_callback(self, depth_msg):
        if depth_msg.encoding == '16UC1':
            raw_data = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")
            self.latest_depth = raw_data.astype(np.float32) / 1000.0
        else:
            self.latest_depth = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")

    def rgb_callback(self, rgb_msg):
        
        bgr_frame = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)

        mask1 = cv2.inRange(hsv, np.array([0, 120, 70]), np.array([25, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([160, 120, 70]), np.array([180, 255, 255]))
        cone_mask = cv2.bitwise_or(mask1, mask2)

        contours, _ = cv2.findContours(cone_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        display_frame = bgr_frame.copy()

        for c in contours:
            if cv2.contourArea(c) < 300: continue
            M = cv2.moments(c)
            if M["m00"] == 0: continue
            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

            dist = None
            if self.latest_depth is not None:
                if cy < self.latest_depth.shape[0] and cx < self.latest_depth.shape[1]:
                    d = self.latest_depth[cy, cx]
                    if not np.isnan(d) and d > 0: dist = d

            if is_cone_shaped(c):
                label = f"CONE {dist:.2f}m" if dist else "CONE"
                cv2.drawContours(display_frame, [c], -1, (0, 165, 255), 2)
                cv2.putText(display_frame, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

        annotated_msg = self.bridge.cv2_to_imgmsg(display_frame, "bgr8")
        self.mask_pub.publish(annotated_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ConeDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
