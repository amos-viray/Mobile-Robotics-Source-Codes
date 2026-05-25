import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist

class PioneerController(Node):
    def __init__(self):
        super().__init__('pioneer_controller')
        
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(Joy, '/joy', self.joy_callback, 10)

        # PS4 Mapping Constants
        self.BTN_X = 0
        self.BTN_CIRCLE = 1
        self.AXIS_L2 = 2 
        self.AXIS_R2 = 5 
        self.AXIS_LEFT_STICK_Y = 1
        self.AXIS_LEFT_STICK_X = 0

        self.mode = "STANDBY"
        self.get_logger().info("\033[94m[CONTROLLER] Node Started - Current State: STANDBY\033[0m")

    def joy_callback(self, msg):
        # 1. Check Inputs
        l2_pressed = msg.axes[self.AXIS_L2] < 0.0
        r2_pressed = msg.axes[self.AXIS_R2] < 0.0
        dead_man_active = l2_pressed or r2_pressed
        
        circle_held = msg.buttons[self.BTN_CIRCLE] == 1
        x_pressed = msg.buttons[self.BTN_X] == 1

        # --- DEBUG PRINT FOR INPUTS (Throttled) ---
        if dead_man_active and not circle_held:
            self.get_logger().info("--- Dead Man (L2/R2) SQUEEZED ---", throttle_duration_sec=2.0)

        # --- LOGIC GATE ---

        # PRIORITY 1: MANUAL DRIVE (Hold Circle)
        if circle_held:
            if self.mode != "MANUAL":
                self.get_logger().info("\033[92m[STATE CHANGE] --> MANUAL MODE (Circle Held)\033[0m")
            self.mode = "MANUAL"
            self.send_manual_commands(msg)

        # PRIORITY 2: AUTO DRIVE (Dead Man + X)
        elif dead_man_active and x_pressed:
            if self.mode != "AUTO":
                self.get_logger().info("\033[93m[STATE CHANGE] --> AUTO MODE (Pedal + X Active)\033[0m")
            self.mode = "AUTO"
            # Autonomous node takes over here
            pass 

        # PRIORITY 3: SAFETY STOP (Default)
        else:
            if self.mode != "STANDBY":
                self.get_logger().warn("\033[91m[STATE CHANGE] --> STANDBY (Safety Stop Applied)\033[0m")
            self.mode = "STANDBY"
            self.stop_robot()

    def send_manual_commands(self, joy_msg):
        twist = Twist()
        twist.linear.x = joy_msg.axes[self.AXIS_LEFT_STICK_Y] * 0.5
        twist.angular.z = joy_msg.axes[self.AXIS_LEFT_STICK_X] * 1.0
        self.cmd_pub.publish(twist)

    def stop_robot(self):
        self.cmd_pub.publish(Twist())

def main(args=None):
    rclpy.init(args=args)
    node = PioneerController()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
