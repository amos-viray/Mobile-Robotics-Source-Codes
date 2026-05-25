#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import String
 
import cv2
from cv_bridge import CvBridge
import numpy as np
import math
import json
import os
import time
 
 
# ── HSV colour ranges ──────────────────────────────────────────────────────
COLOUR_RANGES = [
    ("red",    np.array([0,   120,  60]), np.array([10,  255, 255]),
               np.array([165, 120,  60]), np.array([180, 255, 255])),
    ("yellow", np.array([18,  100,  80]), np.array([35,  255, 255]),
               None, None),
]
 
# ── Detection tuning ───────────────────────────────────────────────────────
MIN_BLOB_AREA        = 200      # px²
MAX_BLOB_AREA        = 80000    # px²
MIN_DISTANCE         = 0.15     # m
MAX_DISTANCE         = 8.0      # m
MIN_CONFIDENCE       = 0.35
 
# ── Image-space tracker tuning ─────────────────────────────────────────────
TRACK_PIXEL_RADIUS   = 80       # px — max centroid shift to match same track
TRACK_DEPTH_DIFF     = 0.6      # m  — max depth difference to match same track
TRACK_MAX_MISS       = 6        # frames a track can go unseen before dropped
TRACK_CONFIRM_HITS   = 6        # consecutive hits needed to confirm a track
TRACK_MIN_VARIANCE   = 999.0    # no min variance requirement (simplify)
 
# ── Global dedup ───────────────────────────────────────────────────────────
GLOBAL_DEDUP_RADIUS  = 1.8      # m — confirmed cones within this = same cone
 
PHOTO_DIR            = '/home/ws/photos/objects'
 
 
class Track:
    """Single image-space track for one colour blob."""
    _id_counter = 0
 
    def __init__(self, colour, px, py, depth, confidence):
        Track._id_counter += 1
        self.id          = Track._id_counter
        self.colour      = colour
        self.hits        = 1
        self.miss        = 0
        self.confirmed   = False
        # Running sums for averaging
        self.px_sum      = px
        self.py_sum      = py
        self.depth_sum   = depth
        self.conf_sum    = confidence
        # Latest values
        self.px          = px
        self.py          = py
        self.depth       = depth
        # Store best frame for photo
        self.best_bgr     = None
        self.best_contour = None
 
    @property
    def avg_px(self):    return self.px_sum    / self.hits
    @property
    def avg_py(self):    return self.py_sum    / self.hits
    @property
    def avg_depth(self): return self.depth_sum / self.hits
    @property
    def avg_conf(self):  return self.conf_sum  / self.hits
 
    def update(self, px, py, depth, confidence, bgr, contour):
        self.hits      += 1
        self.miss       = 0
        self.px_sum    += px
        self.py_sum    += py
        self.depth_sum += depth
        self.conf_sum  += confidence
        self.px         = px
        self.py         = py
        self.depth      = depth
        # Keep the frame with the largest contour as the photo
        if (self.best_contour is None or
                cv2.contourArea(contour) > cv2.contourArea(self.best_contour)):
            self.best_bgr     = bgr.copy()
            self.best_contour = contour.copy()
 
 
class ObjectDetectorNode(Node):
 
    def __init__(self):
        super().__init__('object_detector')
        os.makedirs(PHOTO_DIR, exist_ok=True)
 
        self.bridge        = CvBridge()
        self._latest_depth = None
        self._camera_info  = None
        self._robot_pose   = None
 
        self._tracks       = []   # active image-space tracks
        self._confirmed    = []   # confirmed cones {colour, x, y, ...}
 
        best_effort_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
 
        self.create_subscription(Image,      '/camera/image',
                                 self._rgb_cb,   best_effort_qos)
        self.create_subscription(Image,      '/camera/depth_image',
                                 self._depth_cb, best_effort_qos)
        self.create_subscription(CameraInfo, '/camera/camera_info',
                                 self._info_cb,  best_effort_qos)
 
        from geometry_msgs.msg import PoseWithCovarianceStamped
        self.create_subscription(PoseWithCovarianceStamped, '/pose',
                                 self._pose_cb, 10)
 
        self._det_pub   = self.create_publisher(String, '/detected_objects', 10)
        self._debug_pub = self.create_publisher(Image,  '/object_detector/debug', 10)
 
        self.get_logger().info(
            'Object detector started (image-space tracker). '
            f'confirm={TRACK_CONFIRM_HITS} hits, '
            f'global_dedup={GLOBAL_DEDUP_RADIUS}m'
        )
 
    # ── Callbacks ─────────────────────────────────────────────────────────
 
    def _pose_cb(self, msg):
        x   = msg.pose.pose.position.x
        y   = msg.pose.pose.position.y
        yaw = self._yaw_from_quat(msg.pose.pose.orientation)
        self._robot_pose = (x, y, yaw)
 
    def _info_cb(self, msg: CameraInfo):
        if self._camera_info is not None:
            return
        K = msg.k
        self._camera_info = {
            'fx': K[0], 'fy': K[4],
            'cx': K[2], 'cy': K[5],
            'width': msg.width, 'height': msg.height,
        }
        self.get_logger().info(
            f'Camera intrinsics: fx={K[0]:.1f} fy={K[4]:.1f} '
            f'cx={K[2]:.1f} cy={K[5]:.1f} ({msg.width}×{msg.height})'
        )
 
    def _depth_cb(self, msg: Image):
        try:
            if msg.encoding == '16UC1':
                raw = self.bridge.imgmsg_to_cv2(msg, '16UC1')
                self._latest_depth = raw.astype(np.float32) / 1000.0
            else:
                self._latest_depth = self.bridge.imgmsg_to_cv2(msg, '32FC1')
        except Exception as e:
            self.get_logger().debug(f'Depth decode: {e}')
 
    # ── Main RGB pipeline ─────────────────────────────────────────────────
 
    def _rgb_cb(self, msg: Image):
        if self._robot_pose is None:
            return
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().debug(f'RGB decode: {e}')
            return
 
        hsv   = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        debug = bgr.copy()
 
        # ── Step 1: detect blobs this frame ───────────────────────────
        detections = []   # list of (colour, px, py, depth, confidence, contour)
 
        for colour, lo1, hi1, lo2, hi2 in COLOUR_RANGES:
            mask = cv2.inRange(hsv, lo1, hi1)
            if lo2 is not None:
                mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo2, hi2))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
            mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
 
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
            for contour in contours:
                area = cv2.contourArea(contour)
                if not (MIN_BLOB_AREA < area < MAX_BLOB_AREA):
                    continue
                M = cv2.moments(contour)
                if M['m00'] == 0:
                    continue
                px = int(M['m10'] / M['m00'])
                py = int(M['m01'] / M['m00'])
 
                depth = self._get_depth_neighborhood(px, py, radius=6)
                if depth is None or not (MIN_DISTANCE < depth < MAX_DISTANCE):
                    continue
 
                hull_area  = cv2.contourArea(cv2.convexHull(contour))
                solidity   = area / hull_area if hull_area > 0 else 0
                area_conf  = min(area / 2000.0, 1.0)
                confidence = float(0.5 * solidity + 0.5 * area_conf)
                if confidence < MIN_CONFIDENCE:
                    continue
 
                detections.append((colour, px, py, depth, confidence, contour))
 
        # ── Step 2: match detections to existing tracks ────────────────
        matched_tracks = set()
 
        for colour, px, py, depth, confidence, contour in detections:
            best_track = None
            best_score = float('inf')
 
            for track in self._tracks:
                if track.colour != colour:
                    continue
                pixel_dist = math.hypot(track.px - px, track.py - py)
                depth_diff = abs(track.depth - depth)
                # Must be within pixel AND depth thresholds
                if pixel_dist > TRACK_PIXEL_RADIUS:
                    continue
                if depth_diff > TRACK_DEPTH_DIFF:
                    continue
                # Score = weighted combination
                score = pixel_dist + depth_diff * 50
                if score < best_score:
                    best_score = score
                    best_track = track
 
            if best_track is not None:
                best_track.update(px, py, depth, confidence, bgr, contour)
                matched_tracks.add(best_track.id)
            else:
                # New track
                t = Track(colour, px, py, depth, confidence)
                t.best_bgr     = bgr.copy()
                t.best_contour = contour.copy()
                self._tracks.append(t)
                matched_tracks.add(t.id)
 
        # ── Step 3: age out unmatched tracks ──────────────────────────
        for track in self._tracks:
            if track.id not in matched_tracks:
                track.miss += 1
        self._tracks = [t for t in self._tracks if t.miss <= TRACK_MAX_MISS]
 
        # ── Step 4: check if any track is ready to confirm ────────────
        for track in self._tracks:
            if track.confirmed:
                continue
            if track.hits < TRACK_CONFIRM_HITS:
                continue
 
            # Project averaged pixel + depth to map
            map_xy = self._project_to_map(
                track.avg_px, track.avg_py, track.avg_depth)
            if map_xy is None:
                continue
            map_x, map_y = map_xy
 
            # Global dedup
            if self._is_confirmed_globally(track.colour, map_x, map_y):
                track.confirmed = True   # mark so we don't keep checking
                continue
 
            # Confirm it
            track.confirmed = True
            det = {
                'label':        f'{track.colour} cone',
                'colour':       track.colour,
                'x':            round(map_x, 3),
                'y':            round(map_y, 3),
                'distance':     round(track.avg_depth, 2),
                'confidence':   round(track.avg_conf, 2),
                'observations': track.hits,
            }
            self._confirmed.append(det)
            self._publish_detection(det)
            if track.best_bgr is not None and track.best_contour is not None:
                self._save_photo(track.best_bgr, track.best_contour, track.colour)
 
            self.get_logger().info(
                f'Confirmed {track.colour} cone at map '
                f'({map_x:.2f}, {map_y:.2f}) '
                f'depth={track.avg_depth:.2f}m '
                f'conf={track.avg_conf:.2f} '
                f'hits={track.hits}'
            )
 
        # ── Step 5: draw debug ────────────────────────────────────────
        for track in self._tracks:
            c = (0, 0, 220) if track.colour == 'red' else (0, 200, 220)
            cx, cy = int(track.px), int(track.py)
            if track.confirmed:
                cv2.circle(debug, (cx, cy), 8, c, 2)
                cv2.putText(debug, f'OK {track.colour}',
                            (cx, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)
            else:
                cv2.circle(debug, (cx, cy), 5, c, 1)
                cv2.putText(debug,
                            f'{track.colour} [{track.hits}/{TRACK_CONFIRM_HITS}]',
                            (cx, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1)
            cv2.putText(debug, f'd={track.depth:.2f}m',
                        (cx, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, c, 1)
 
        try:
            self._debug_pub.publish(self.bridge.cv2_to_imgmsg(debug, 'bgr8'))
        except Exception:
            pass
 
    # ── Helpers ───────────────────────────────────────────────────────────
 
    def _get_depth_neighborhood(self, px: int, py: int, radius: int = 6):
        if self._latest_depth is None:
            return None
        h, w = self._latest_depth.shape[:2]
        y1 = max(0, py - radius); y2 = min(h, py + radius + 1)
        x1 = max(0, px - radius); x2 = min(w, px + radius + 1)
        patch = self._latest_depth[y1:y2, x1:x2]
        valid = patch[
            (patch > MIN_DISTANCE) & (patch < MAX_DISTANCE) & ~np.isnan(patch)]
        return float(np.median(valid)) if len(valid) > 0 else None
 
    def _project_to_map(self, px, py, depth):
        if self._camera_info is None or self._robot_pose is None:
            return None
        ci = self._camera_info
        fx, fy, cx, cy = ci['fx'], ci['fy'], ci['cx'], ci['cy']
        cam_x = (px - cx) * depth / fx
        cam_z = depth
        robot_x_local =  cam_z
        robot_y_local = -cam_x
        rx, ry, ryaw = self._robot_pose
        map_x = rx + math.cos(ryaw) * robot_x_local - math.sin(ryaw) * robot_y_local
        map_y = ry + math.sin(ryaw) * robot_x_local + math.cos(ryaw) * robot_y_local
        return map_x, map_y
 
    def _is_confirmed_globally(self, colour: str, map_x: float, map_y: float) -> bool:
        for det in self._confirmed:
            if det['colour'] != colour:
                continue
            if math.hypot(det['x'] - map_x, det['y'] - map_y) < GLOBAL_DEDUP_RADIUS:
                return True
        return False
 
    def _publish_detection(self, det: dict):
        msg = String()
        msg.data = json.dumps(det)
        self._det_pub.publish(msg)
 
    def _save_photo(self, bgr: np.ndarray, contour, colour: str):
        x, y, w, h = cv2.boundingRect(contour)
        pad = 10
        x1 = max(0, x - pad);        y1 = max(0, y - pad)
        x2 = min(bgr.shape[1], x + w + pad); y2 = min(bgr.shape[0], y + h + pad)
        crop = bgr[y1:y2, x1:x2].copy()
 
        # Darken background outside contour
        mask = np.zeros(crop.shape[:2], dtype=np.uint8)
        shifted = contour - np.array([x1, y1])
        cv2.drawContours(mask, [shifted], -1, 255, cv2.FILLED)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask   = cv2.dilate(mask, kernel)
        bg     = cv2.bitwise_not(mask)
        crop[bg > 0] = (crop[bg > 0] * 0.35).astype(np.uint8)
 
        border = (0, 0, 220) if colour == 'red' else (0, 200, 220)
        cv2.drawContours(crop, [shifted], -1, border, 2)
        cv2.putText(crop, f'{colour.upper()} CONE',
                    (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, border, 1)
 
        ts    = time.strftime('%Y%m%d_%H%M%S')
        fname = os.path.join(PHOTO_DIR, f'{colour}_cone_{ts}.jpg')
        cv2.imwrite(fname, crop)
        self.get_logger().debug(f'Photo saved: {fname}')
 
    @staticmethod
    def _yaw_from_quat(q) -> float:
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny, cosy)
 
 
def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
 
 
if __name__ == '__main__':
    main()
 