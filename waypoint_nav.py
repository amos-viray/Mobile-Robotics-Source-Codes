#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
import cv2
from cv_bridge import CvBridge
import numpy as np
import math
import os
from datetime import datetime

CONE_ARRIVAL_DISTANCE = 2.0 

class WaypointManager(Node):
    def __init__(self):
        super().__init__('waypoint_manager')
        self.bridge = CvBridge()

  
        self.current_waypoint_index = 0
        self.state = 'IDLE'  
        self.automated_mode = True
        self.latest_depth = None
        self.cone_distance = None
        self.cone_cx = None
        self.latest_color_frame = None  
        self.journey_log = []


        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        self.create_subscription(Image, '/camera/image', self.rgb_callback, 10)
        self.create_subscription(Image, '/camera/depth_image', self.depth_callback, 10)
        self.create_subscription(Bool, '/automated_mode', self.mode_callback, 10)

        self.create_timer(0.5, self.state_machine)

        self.start_timer = self.create_timer(5.0, self.start_navigation)

        self.get_logger().info("Waypoint Manager Started — auto-starting in 5 seconds...")

    def start_navigation(self):
        """Fires once after startup delay to send the first Nav2 goal."""
        self.start_timer.cancel()  
        if self.automated_mode:
            self.get_logger().info("Auto-start: sending first waypoint goal")
            x, y = WAYPOINTS[self.current_waypoint_index]
            self.send_nav_goal(x, y)
            self.state = 'NAVIGATING'

    def depth_callback(self, msg):
        self.latest_depth = self.bridge.imgmsg_to_cv2(msg, "32FC1")

    def rgb_callback(self, rgb_msg):
        color_frame = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        self.latest_color_frame = color_frame.copy() 

        hsv = cv2.cvtColor(color_frame, cv2.COLOR_BGR2HSV)
        h, w = hsv.shape[:2]

        mask1 = cv2.inRange(hsv, np.array([0,   120, 70]), np.array([25,  255, 255]))
        mask2 = cv2.inRange(hsv, np.array([160, 120, 70]), np.array([180, 255, 255]))
        mask  = cv2.bitwise_or(mask1, mask2)
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        self.cone_distance = None
        self.cone_cx = None

        for c in contours:
            if cv2.contourArea(c) > 200:
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    if self.latest_depth is not None:
                        dist = self.latest_depth[cy, cx]
                        if not np.isnan(dist) and not np.isinf(dist):
                            self.cone_distance = dist
                            self.cone_cx = cx
                            cv2.circle(color_frame, (cx, cy), 5, (0, 255, 0), -1)
                            cv2.putText(color_frame, f"{dist:.2f}m", (cx+10, cy),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Detection Feed", color_frame)
        cv2.waitKey(1)

    def mode_callback(self, msg):
        was_automated = self.automated_mode
        self.automated_mode = msg.data

        if self.automated_mode and not was_automated and self.state == 'IDLE':
            self.get_logger().info("Automated mode enabled — sending first waypoint")
            x, y = WAYPOINTS[self.current_waypoint_index]
            self.send_nav_goal(x, y)
            self.state = 'NAVIGATING'

    def state_machine(self):
        if not self.automated_mode:
            return  

        if self.state == 'IDLE':
            pass  

        elif self.state == 'NAVIGATING':
            self.handle_navigating()

        elif self.state == 'CHECKING_CONE':
            self.handle_checking_cone()

        elif self.state == 'ROTATING':
            self.handle_rotating()

        elif self.state == 'DONE':
            self.handle_done()

    def handle_navigating(self):
        if self.cone_distance is not None and self.cone_distance < CONE_ARRIVAL_DISTANCE:
            self.get_logger().info(f"Cone detected at {self.cone_distance:.2f}m — waypoint reached!")
            self.state = 'CHECKING_CONE'

    def handle_checking_cone(self):

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("/home/ws/photos", exist_ok=True)
        photo_path = f"/home/ws/photos/waypoint_{self.current_waypoint_index}_{timestamp}.jpg"

        if self.latest_color_frame is not None:
            cv2.imwrite(photo_path, self.latest_color_frame)
            self.get_logger().info(f"Photo saved: {photo_path}")
        else:
            self.get_logger().warn("No camera frame available for photo")

        self.journey_log.append({
            'waypoint': self.current_waypoint_index,
            'cone_distance': self.cone_distance,
            'photo': photo_path,
        })

        if self.cone_cx is not None and self.cone_cx > 320:
            self.get_logger().info("Cone is on the right — moving to next waypoint")
            self.advance_to_next_waypoint()
        else:
            self.get_logger().info("Cone is on the left — rotating until it's on the right...")
            self.state = 'ROTATING'

    def handle_rotating(self):
        if self.cone_cx is not None and self.cone_cx > 320:
            self.get_logger().info("Cone now on right after rotation")
            self.advance_to_next_waypoint()

    def handle_done(self):
        self.get_logger().info("=== JOURNEY COMPLETE ===")
        self.print_journey_summary()

    def advance_to_next_waypoint(self):
        self.current_waypoint_index += 1

        if self.current_waypoint_index >= len(WAYPOINTS):
            self.get_logger().info("All waypoints visited — returning to start")
            self.send_nav_goal(0.0, 0.0)
            self.state = 'DONE'
        else:
            x, y = WAYPOINTS[self.current_waypoint_index]
            self.get_logger().info(f"Heading to waypoint {self.current_waypoint_index}: ({x}, {y})")
            self.send_nav_goal(x, y)
            self.state = 'NAVIGATING'

    def send_nav_goal(self, x, y):
        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y
        goal.pose.pose.orientation.w = 1.0

        self.nav_client.wait_for_server()
        self.nav_client.send_goal_async(goal)
        self.get_logger().info(f"Nav2 goal sent: ({x}, {y})")

    def print_journey_summary(self):
        self.get_logger().info("--- Journey Summary ---")
        for entry in self.journey_log:
            self.get_logger().info(
                f"Waypoint {entry['waypoint']}: "
                f"cone at {entry['cone_distance']:.2f}m, "
                f"photo: {entry['photo']}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = WaypointManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()