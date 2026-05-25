import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from nav2_msgs.action import FollowWaypoints
from geometry_msgs.msg import PoseStamped
from rclpy.action import ActionClient
from std_msgs.msg import Header
import math
import json
import os
import time
import threading


RUNS_DIR           = '/home/ws/recorded_runs'
MIN_TRAVEL_DIST    = 0.3     # metres between saved poses
STOP_TIMEOUT       = 3.0     # seconds stationary before run ends
MIN_RUN_POSES      = 3       # discard runs shorter than this
LINEAR_THRESHOLD   = 0.02    # m/s — below this is "stationary"


class PathRecorderNode(Node):

    def __init__(self):
        super().__init__('path_recorder')
        os.makedirs(RUNS_DIR, exist_ok=True)

        self._recording    = False
        self._current_run  = []
        self._last_pose    = None
        self._last_moved   = None
        self._run_start    = None
        self._lock         = threading.Lock()

        self._action_client = ActionClient(self, FollowWaypoints, '/follow_waypoints')

        self.create_subscription(Odometry, '/odometry/filtered', self._odom_cb, 10)
        self.create_timer(0.5, self._check_stop_timer)

        self.get_logger().info(
            f'Path recorder started. Saving runs to {RUNS_DIR}. '
            f'Auto-records when robot moves, stops after {STOP_TIMEOUT}s stationary.'
        )

    # ------------------------------------------------------------------ #
    #  Odometry callback — core recording logic
    # ------------------------------------------------------------------ #
    def _odom_cb(self, msg: Odometry):
        x   = msg.pose.pose.position.x
        y   = msg.pose.pose.position.y
        yaw = self._yaw_from_quat(msg.pose.pose.orientation)
        vx  = abs(msg.twist.twist.linear.x)
        vy  = abs(msg.twist.twist.linear.y)
        moving = (vx + vy) > LINEAR_THRESHOLD

        with self._lock:
            if moving:
                self._last_moved = time.time()

                if not self._recording:
                    # Start a new run
                    self._recording   = True
                    self._current_run = [{'x': round(x, 4),
                                          'y': round(y, 4),
                                          'yaw': round(yaw, 4)}]
                    self._last_pose   = (x, y)
                    self._run_start   = time.time()
                    self.get_logger().info('Recording started.')
                else:
                    # Append pose if we've travelled far enough
                    if self._last_pose is not None:
                        dx = x - self._last_pose[0]
                        dy = y - self._last_pose[1]
                        if math.hypot(dx, dy) >= MIN_TRAVEL_DIST:
                            self._current_run.append({'x': round(x, 4),
                                                      'y': round(y, 4),
                                                      'yaw': round(yaw, 4)})
                            self._last_pose = (x, y)

    # ------------------------------------------------------------------ #
    #  Timer — end run after robot has been stationary for STOP_TIMEOUT s
    # ------------------------------------------------------------------ #
    def _check_stop_timer(self):
        with self._lock:
            if not self._recording:
                return
            if self._last_moved is None:
                return
            elapsed = time.time() - self._last_moved
            if elapsed >= STOP_TIMEOUT:
                self._save_run()

    # ------------------------------------------------------------------ #
    #  Save run to JSON
    # ------------------------------------------------------------------ #
    def _save_run(self):
        """Must be called with self._lock held."""
        poses = self._current_run
        self._recording   = False
        self._current_run = []
        self._last_pose   = None

        if len(poses) < MIN_RUN_POSES:
            self.get_logger().info(
                f'Run discarded — only {len(poses)} poses (min {MIN_RUN_POSES}).'
            )
            return

        duration = round(time.time() - self._run_start, 1) if self._run_start else 0
        ts       = time.strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(RUNS_DIR, f'run_{ts}.json')

        run_data = {
            'timestamp':    time.strftime('%Y-%m-%d %H:%M:%S'),
            'duration_secs': duration,
            'pose_count':   len(poses),
            'poses':        poses,
        }

        with open(filename, 'w') as f:
            json.dump(run_data, f, indent=2)

        self.get_logger().info(
            f'Run saved: {filename} ({len(poses)} poses, {duration}s)'
        )

    # ------------------------------------------------------------------ #
    #  Replay a saved run by sending poses to Nav2 /follow_waypoints
    # ------------------------------------------------------------------ #
    def replay_run(self, filename: str):
        try:
            with open(filename, 'r') as f:
                run_data = json.load(f)
        except Exception as e:
            self.get_logger().error(f'Failed to load run file {filename}: {e}')
            return False

        poses = run_data.get('poses', [])
        if not poses:
            self.get_logger().error('Run file has no poses.')
            return False

        self.get_logger().info(
            f'Replaying {len(poses)} poses from {os.path.basename(filename)}'
        )

        if not self._action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('/follow_waypoints action server not available.')
            return False

        goal = FollowWaypoints.Goal()
        for p in poses:
            ps               = PoseStamped()
            ps.header        = Header()
            ps.header.frame_id = 'map'
            ps.header.stamp  = self.get_clock().now().to_msg()
            ps.pose.position.x  = float(p['x'])
            ps.pose.position.y  = float(p['y'])
            ps.pose.position.z  = 0.0
            # Convert yaw back to quaternion (z/w only for 2D)
            yaw = float(p.get('yaw', 0.0))
            ps.pose.orientation.z = math.sin(yaw / 2.0)
            ps.pose.orientation.w = math.cos(yaw / 2.0)
            goal.poses.append(ps)

        self._action_client.send_goal_async(goal)
        return True

    # ------------------------------------------------------------------ #
    #  Utility
    # ------------------------------------------------------------------ #
    @staticmethod
    def _yaw_from_quat(q) -> float:
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny, cosy)


def main(args=None):
    rclpy.init(args=args)
    node = PathRecorderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
