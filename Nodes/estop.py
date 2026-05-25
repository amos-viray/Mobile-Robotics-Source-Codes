import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from collections import deque
import time
import os
import math
 
 
class EStopNode(Node):
 
    ESTOP_DISTANCE     = 1.0
    BUFFER_DURATION    = 5.0
    REPORT_DIR         = '/home/ws/incident_reports'
 
    SCAN_HISTORY_SIZE  = 8
    MOVEMENT_THRESHOLD = 0.15   # increased from 0.08 — less sensitive to rotation
    MOVING_HITS_NEEDED = 5      # increased from 3 — more evidence required
    APPROACH_THRESHOLD = 0.05   # object must be getting closer per scan
 
    def __init__(self):
        super().__init__('estop_node')
 
        self._scan_buffer  = deque()
        self._vel_buffer   = deque()
        self._scan_history = deque(maxlen=self.SCAN_HISTORY_SIZE)
        self._moving_hits  = 0
        self._estopped     = False
        self._report_saved = False
 
        self._cmd_pub    = self.create_publisher(Twist,  '/cmd_vel',      1)
        self._status_pub = self.create_publisher(String, '/estop_status', 10)
 
        self.create_subscription(Twist,     '/cmd_vel_in', self._vel_in_cb, 1)
        self.create_subscription(LaserScan, '/scan',       self._scan_cb,   10)
        self.create_subscription(Twist,     '/cmd_vel',    self._vel_cb,    10)
 
        self.create_timer(0.05, self._estop_timer_cb)
        self._publish_status(False)
        os.makedirs(self.REPORT_DIR, exist_ok=True)
        self.get_logger().info(
            'E-Stop node started. Gating /cmd_vel_in -> /cmd_vel. '
            f'threshold={self.MOVEMENT_THRESHOLD}m, '
            f'hits={self.MOVING_HITS_NEEDED}, '
            f'approach={self.APPROACH_THRESHOLD}m/scan'
        )
 
    def _vel_in_cb(self, msg: Twist):
        if not self._estopped:
            self._cmd_pub.publish(msg)
 
    def _estop_timer_cb(self):
        if self._estopped:
            self._cmd_pub.publish(Twist())
 
    def _publish_status(self, active: bool):
        msg = String()
        msg.data = 'active' if active else 'clear'
        self._status_pub.publish(msg)
 
    def _scan_cb(self, msg: LaserScan):
        now = self.get_clock().now().nanoseconds * 1e-9
        self._scan_buffer.append((now, msg))
        self._prune_buffer(self._scan_buffer, now)
 
        min_range = self._min_valid_range(msg)
 
        if min_range < self.ESTOP_DISTANCE:
            if self._is_moving_and_approaching(msg):
                self._moving_hits += 1
                self.get_logger().debug(
                    f'Moving+approaching at {min_range:.3f}m '
                    f'(hit {self._moving_hits}/{self.MOVING_HITS_NEEDED})'
                )
                if self._moving_hits >= self.MOVING_HITS_NEEDED and not self._estopped:
                    self.get_logger().warn(
                        f'MOVING OBSTACLE at {min_range:.3f}m — EMERGENCY STOP TRIGGERED'
                    )
                    self._estopped     = True
                    self._report_saved = False
                    self._publish_status(True)
                    self._trigger_estop(min_range, now)
            else:
                if self._moving_hits > 0:
                    self.get_logger().info(
                        f'Object at {min_range:.3f}m not confirmed moving — '
                        f'ignoring (hits reset from {self._moving_hits})'
                    )
                self._moving_hits = 0
        else:
            if self._estopped:
                self.get_logger().info('Obstacle cleared. E-stop resolved.')
                self._publish_status(False)
            self._estopped     = False
            self._report_saved = False
            self._moving_hits  = 0
 
        self._scan_history.append((now, msg))
 
    def _is_moving_and_approaching(self, current_scan: LaserScan) -> bool:
        """
        Two conditions must BOTH be true to trigger:
        1. Shift: closest range has changed by > MOVEMENT_THRESHOLD vs oldest scan
        2. Approach: range is consistently decreasing (object getting closer)
 
        Using global minimum range (not angle-specific) makes this
        rotation-invariant. A stationary cone seen during robot rotation
        shifts in angle but its minimum range does not consistently decrease.
        A genuinely approaching object will show both shift AND decreasing range.
        """
        if len(self._scan_history) < 2:
            return False
 
        _, oldest_scan = self._scan_history[0]
        _, recent_scan = self._scan_history[-1]
 
        current_min = self._min_valid_range(current_scan)
        oldest_min  = self._min_valid_range(oldest_scan)
        recent_min  = self._min_valid_range(recent_scan)
 
        if float('inf') in (current_min, oldest_min, recent_min):
            return False
 
        shift       = abs(current_min - oldest_min)
        shifted     = shift > self.MOVEMENT_THRESHOLD
        approaching = (oldest_min - recent_min) > self.APPROACH_THRESHOLD
 
        self.get_logger().debug(
            f'current={current_min:.3f} oldest={oldest_min:.3f} '
            f'recent={recent_min:.3f} shift={shift:.3f} '
            f'shifted={shifted} approaching={approaching}'
        )
 
        return shifted and approaching
 
    def _trigger_estop(self, trigger_distance: float, trigger_time: float):
        for _ in range(10):
            self._cmd_pub.publish(Twist())
        if not self._report_saved:
            self._save_incident_report(trigger_distance, trigger_time)
            self._report_saved = True
 
    def _vel_cb(self, msg: Twist):
        now = self.get_clock().now().nanoseconds * 1e-9
        self._vel_buffer.append((now, msg))
        self._prune_buffer(self._vel_buffer, now)
 
    def _save_incident_report(self, trigger_distance: float, trigger_time: float):
        timestamp_str = time.strftime('%Y%m%d_%H%M%S', time.localtime(trigger_time))
        filename = os.path.join(self.REPORT_DIR, f'incident_{timestamp_str}.txt')
        lines = []
        lines.append('=' * 60)
        lines.append('PIONEER BOT — EMERGENCY STOP INCIDENT REPORT')
        lines.append('=' * 60)
        lines.append(f'Trigger time (sim):   {trigger_time:.3f}s')
        lines.append(f'Trigger distance:     {trigger_distance:.4f}m')
        lines.append(f'Trigger cause:        Moving obstacle detected')
        lines.append(f'Report window:        last {self.BUFFER_DURATION}s (0.5s chunks)')
        lines.append('')
        bin_size = 0.5
        num_bins = int(self.BUFFER_DURATION / bin_size)
        bins     = [-self.BUFFER_DURATION + i * bin_size for i in range(num_bins + 1)]
        def get_bin(ts):
            return round((ts - trigger_time) / bin_size) * bin_size
        lines.append('-' * 60)
        lines.append('VELOCITY HISTORY (/cmd_vel) — one sample per 0.5s chunk')
        lines.append('-' * 60)
        lines.append(f'{"Chunk (s)":<14} {"Lin.X":>8} {"Lin.Y":>8} {"Lin.Z":>8} '
                     f'{"Ang.X":>8} {"Ang.Y":>8} {"Ang.Z":>8}')
        vel_bins = {}
        for ts, twist in self._vel_buffer:
            vel_bins[get_bin(ts)] = (ts, twist)
        for b in bins:
            if b in vel_bins:
                _, twist = vel_bins[b]
                lines.append(
                    f'{b:<14.1f} '
                    f'{twist.linear.x:>8.4f} {twist.linear.y:>8.4f} {twist.linear.z:>8.4f} '
                    f'{twist.angular.x:>8.4f} {twist.angular.y:>8.4f} {twist.angular.z:>8.4f}'
                )
            else:
                lines.append(f'{b:<14.1f} {"---":>8} {"---":>8} {"---":>8} '
                             f'{"---":>8} {"---":>8} {"---":>8}')
        lines.append('')
        lines.append('-' * 60)
        lines.append('LIDAR HISTORY (/scan) — one sample per 0.5s chunk')
        lines.append('-' * 60)
        lines.append(f'{"Chunk (s)":<14} {"Min (m)":>10} {"Max (m)":>10} '
                     f'{"Fwd (m)":>10} {"Num rays":>10}')
        scan_bins = {}
        for ts, scan in self._scan_buffer:
            scan_bins[get_bin(ts)] = (ts, scan)
        for b in bins:
            if b in scan_bins:
                _, scan = scan_bins[b]
                valid = [r for r in scan.ranges
                         if not math.isnan(r) and not math.isinf(r) and r > 0.0]
                min_r = min(valid) if valid else float('nan')
                max_r = max(valid) if valid else float('nan')
                mid   = len(scan.ranges) // 2
                fwd   = scan.ranges[mid] if scan.ranges else float('nan')
                lines.append(f'{b:<14.1f} {min_r:>10.4f} {max_r:>10.4f} '
                             f'{fwd:>10.4f} {len(scan.ranges):>10}')
            else:
                lines.append(f'{b:<14.1f} {"---":>10} {"---":>10} '
                             f'{"---":>10} {"---":>10}')
        lines.append('')
        lines.append('=' * 60)
        lines.append('END OF REPORT')
        lines.append('=' * 60)
        with open(filename, 'w') as f:
            f.write('\n'.join(lines))
        self.get_logger().info(f'Incident report saved to: {filename}')
 
    def _min_valid_range(self, msg: LaserScan) -> float:
        valid = [
            r for r in msg.ranges
            if not math.isnan(r) and not math.isinf(r)
            and msg.range_min < r < msg.range_max
        ]
        return min(valid) if valid else float('inf')
 
    def _prune_buffer(self, buf: deque, now: float):
        cutoff = now - self.BUFFER_DURATION
        while buf and buf[0][0] < cutoff:
            buf.popleft()
 
 
def main(args=None):
    rclpy.init(args=args)
    node = EStopNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
 
 
if __name__ == '__main__':
    main()