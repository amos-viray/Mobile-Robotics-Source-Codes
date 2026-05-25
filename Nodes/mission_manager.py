#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import FollowWaypoints
 
import json
import subprocess
import threading
import time
 
STATE_IDLE       = 'idle'
STATE_EXPLORING  = 'exploring'
STATE_NAVIGATING = 'navigating'
 
 
def run_cmd(cmd: list, timeout: float = 10.0) -> tuple[bool, str]:
    """Run a CLI command, return (success, output)."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout)
        return result.returncode == 0, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return False, 'timeout'
    except Exception as e:
        return False, str(e)
 
 
class MissionManagerNode(Node):
 
    def __init__(self):
        super().__init__('mission_manager')
 
        self._state     = STATE_IDLE
        self._waypoints = []
        self._lock      = threading.Lock()
 
        self._explore_goal_handle = None
        self._nav_goal_handle     = None
 
        cb = ReentrantCallbackGroup()
 
        # ── Publishers ──
        self._phase_pub     = self.create_publisher(String, '/robot_phase',    10)
        self._state_pub     = self.create_publisher(String, '/robot_state',    10)
        self._action_pub    = self.create_publisher(String, '/robot_action',   10)
        self._recording_pub = self.create_publisher(String, '/recording_cmd',  10)
 
        # ── Subscribers ──
        self.create_subscription(String, '/robot_phase_cmd',
                                 self._phase_cmd_cb, 10, callback_group=cb)
        self.create_subscription(String, '/waypoint_cmd',
                                 self._waypoint_cmd_cb, 10, callback_group=cb)
 
        # ── Action clients ──
        self._nav_client = ActionClient(
            self, FollowWaypoints, '/follow_waypoints', callback_group=cb)
 
        # Status timer
        self.create_timer(1.0, self._status_timer_cb)
 
        self._publish_all(STATE_IDLE, 'Idle', 'Waiting for command')
        self.get_logger().info('Mission manager started — state: IDLE')
 
    # ── Command callbacks ──────────────────────────────────────────────────
 
    def _phase_cmd_cb(self, msg: String):
        cmd = msg.data.strip().lower()
        self.get_logger().info(f'Phase command: {cmd}')
        if cmd == 'start_exploration':
            threading.Thread(
                target=self._run_exploration, daemon=True).start()
        elif cmd == 'start_navigation':
            threading.Thread(
                target=self._cmd_start_navigation, daemon=True).start()
        elif cmd == 'stop':
            threading.Thread(
                target=self._cmd_stop, daemon=True).start()
 
    def _waypoint_cmd_cb(self, msg: String):
        try:
            data = json.loads(msg.data)
        except Exception:
            return
        if data.get('action') == 'set':
            with self._lock:
                self._waypoints = [
                    {'x': float(p['x']), 'y': float(p['y'])}
                    for p in data.get('poses', [])
                ]
            self.get_logger().info(
                f'Waypoints set: {len(self._waypoints)}')
            self._publish_action(
                f'{len(self._waypoints)} waypoints loaded')
        elif data.get('action') == 'clear':
            with self._lock:
                self._waypoints = []
            self._publish_action('Waypoints cleared')
 
    # ── Exploration ────────────────────────────────────────────────────────
 
    def _run_exploration(self):
        with self._lock:
            if self._state != STATE_IDLE:
                self.get_logger().warn(
                    f'Cannot explore from state {self._state}')
                return
            self._state = STATE_EXPLORING
 
        self._publish_all(STATE_EXPLORING, 'Exploring',
                          'Configuring exploration server...')
 
        # ── Step 1: lifecycle via CLI (no executor conflict) ──
        self._publish_action('Configuring roadmap_exploration_server...')
        ok, out = run_cmd([
            'ros2', 'lifecycle', 'set',
            '/roadmap_exploration_server', 'configure'], timeout=10.0)
        self.get_logger().info(f'Configure: ok={ok} {out.strip()}')
        time.sleep(1.0)
 
        self._publish_action('Activating roadmap_exploration_server...')
        ok, out = run_cmd([
            'ros2', 'lifecycle', 'set',
            '/roadmap_exploration_server', 'activate'], timeout=10.0)
        self.get_logger().info(f'Activate: ok={ok} {out.strip()}')
 
        if not ok:
            self.get_logger().error('Explorer failed to activate — aborting')
            with self._lock:
                self._state = STATE_IDLE
            self._publish_all(STATE_IDLE, 'Idle', 'Exploration failed to start')
            return
 
        time.sleep(1.0)
 
        # ── Step 2: send goal via CLI (most reliable approach) ──
        self._publish_action('Sending exploration goal...')
        ok, out = run_cmd([
            'ros2', 'action', 'send_goal',
            '/roadmap_explorer',
            'roadmap_explorer_msgs/action/Explore',
            '{exploration_bringup_mode: 0}'],
            timeout=3600.0)
        self.get_logger().info(f'Explore goal result: ok={ok} {out[:200]}')
 
        with self._lock:
            current = self._state
 
        if current == STATE_EXPLORING:
            with self._lock:
                self._state = STATE_IDLE
            self._publish_all(STATE_IDLE, 'Idle',
                              'Exploration complete')
 
        # Deactivate explorer
        run_cmd(['ros2', 'lifecycle', 'set',
                 '/roadmap_exploration_server', 'deactivate'])
 
    # ── Navigation ─────────────────────────────────────────────────────────
 
    def _cmd_start_navigation(self):
        with self._lock:
            if self._state != STATE_IDLE:
                self.get_logger().warn(
                    f'Cannot navigate from state {self._state}')
                return
            if not self._waypoints:
                self._publish_action(
                    'No waypoints loaded — add waypoints first')
                return
            self._state = STATE_NAVIGATING
            waypoints = list(self._waypoints)
 
        self._publish_all(STATE_NAVIGATING, 'Navigating',
                          f'Navigating to {len(waypoints)} waypoints...')
 
        self._publish_action('Waiting for /follow_waypoints server...')
        # Retry for up to 30 seconds — Nav2 lifecycle takes time to activate
        server_ready = False
        for attempt in range(6):
            if self._nav_client.wait_for_server(timeout_sec=5.0):
                server_ready = True
                break
            self.get_logger().warn(
                f'Waiting for /follow_waypoints... attempt {attempt+1}/6')
            self._publish_action(
                f'Waiting for Nav2... ({attempt+1}/6)')
            # On attempt 3, try manually activating Nav2 nodes
            if attempt == 2:
                self.get_logger().info(
                    'Attempting manual Nav2 lifecycle activation...')
                self._publish_action('Activating Nav2 nodes...')
                for node in ['controller_server', 'planner_server',
                             'behavior_server', 'bt_navigator',
                             'waypoint_follower']:
                    # Try configure first (may already be configured)
                    run_cmd(['ros2', 'lifecycle', 'set', f'/{node}',
                             'configure'], timeout=5.0)
                    run_cmd(['ros2', 'lifecycle', 'set', f'/{node}',
                             'activate'], timeout=5.0)
                time.sleep(2.0)
 
        if not server_ready:
            self.get_logger().error('/follow_waypoints not available after 30s')
            with self._lock:
                self._state = STATE_IDLE
            self._publish_all(STATE_IDLE, 'Idle',
                              'Nav2 not available — try again in a moment')
            return
 
        goal = FollowWaypoints.Goal()
        for i, wp in enumerate(waypoints):
            ps = PoseStamped()
            ps.header.frame_id    = 'map'
            # Each waypoint gets a unique timestamp — same stamp confuses Nav2
            ps.header.stamp       = self.get_clock().now().to_msg()
            ps.pose.position.x    = float(wp['x'])
            ps.pose.position.y    = float(wp['y'])
            ps.pose.position.z    = 0.0
            ps.pose.orientation.w = 1.0
            goal.poses.append(ps)
            # Small sleep so timestamps are actually different
            time.sleep(0.01)
 
        self._publish_action(
            f'Navigating — {len(waypoints)} waypoints queued')
 
        # Start recording the full navigation session
        rec_msg = String(); rec_msg.data = 'start'
        self._recording_pub.publish(rec_msg)
 
        # Send goal with callback — no spin_until_future_complete
        event = threading.Event()
        result_container = [None]
 
        def goal_response_cb(future):
            handle = future.result()
            if not handle.accepted:
                self.get_logger().error('Nav goal rejected')
                result_container[0] = 'rejected'
                event.set()
                return
            self._nav_goal_handle = handle
            result_future = handle.get_result_async()
            result_future.add_done_callback(result_cb)
 
        def result_cb(future):
            result_container[0] = future.result()
            event.set()
 
        def feedback_cb(feedback_msg):
            idx   = feedback_msg.feedback.current_waypoint
            wp    = waypoints[idx] if idx < len(waypoints) else None
            coord = f'({wp["x"]:.1f}, {wp["y"]:.1f})' if wp else ''
            self._publish_action(
                f'Navigating — waypoint {idx+1}/{len(waypoints)} {coord}')
 
        send_future = self._nav_client.send_goal_async(
            goal, feedback_callback=feedback_cb)
        send_future.add_done_callback(goal_response_cb)
 
        # Wait for completion (blocking this thread only, not the executor)
        event.wait(timeout=3600.0)
 
        with self._lock:
            current = self._state
 
        if current == STATE_NAVIGATING:
            # Stop recording
            rec_msg = String(); rec_msg.data = 'stop'
            self._recording_pub.publish(rec_msg)
            result = result_container[0]
            missed_indices = []
            if result and result != 'rejected':
                try:
                    missed_indices = list(result.result.missed_waypoints)
                except Exception:
                    pass
            missed  = len(missed_indices)
            reached = len(waypoints) - missed
            self._nav_goal_handle = None
            with self._lock:
                self._state = STATE_IDLE
            if missed == 0:
                msg = f'Navigation complete — all {reached} waypoints reached'
            else:
                missed_str = ', '.join(str(i+1) for i in missed_indices)
                msg = (f'Navigation complete — {reached}/{len(waypoints)} reached. '
                       f'Missed waypoints: {missed_str}')
            self.get_logger().info(msg)
            self._publish_all(STATE_IDLE, 'Idle', msg)
 
    # ── Stop ───────────────────────────────────────────────────────────────
 
    def _cmd_stop(self):
        with self._lock:
            prev = self._state
            self._state = STATE_IDLE
 
        self.get_logger().info(f'Stop: {prev} → IDLE')
 
        if self._nav_goal_handle:
            try:
                self._nav_goal_handle.cancel_goal_async()
            except Exception:
                pass
            self._nav_goal_handle = None
            # Stop recording if navigation was active
            rec_msg = String(); rec_msg.data = 'stop'
            self._recording_pub.publish(rec_msg)
 
        if prev == STATE_EXPLORING:
            run_cmd(['ros2', 'lifecycle', 'set',
                     '/roadmap_exploration_server', 'deactivate'])
 
        self._publish_all(STATE_IDLE, 'Idle', 'Stopped — waiting for command')
 
    # ── Helpers ────────────────────────────────────────────────────────────
 
    def _status_timer_cb(self):
        with self._lock:
            state = self._state
        msg = String(); msg.data = state
        self._phase_pub.publish(msg)
 
    def _publish_all(self, phase: str, state: str, action: str):
        with self._lock:
            self._state = phase
        p = String(); p.data = phase;  self._phase_pub.publish(p)
        s = String(); s.data = state;  self._state_pub.publish(s)
        a = String(); a.data = action; self._action_pub.publish(a)
 
    def _publish_action(self, action: str):
        a = String(); a.data = action
        self._action_pub.publish(a)
 
 
def main(args=None):
    rclpy.init(args=args)
    node = MissionManagerNode()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
 
 
if __name__ == '__main__':
    main()
 