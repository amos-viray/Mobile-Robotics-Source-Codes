import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path, Odometry
from std_msgs.msg import String
from sensor_msgs.msg import Image

import threading
import base64
import json
import math
import numpy as np
from flask import Flask, Response
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='threading')

state = {
    'phase': 'mapping',
    'robot_state': 'Initialising',
    'intended_action': 'Starting up...',
    'position': {'x': 0.0, 'y': 0.0, 'yaw': 0.0},
    'map': None,
    'map_info': None,
    'markers': [],
    'path': [],
    'estop': False,
}
state_lock = threading.Lock()

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Pioneer 3-AT Mission Control</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.2/socket.io.min.js"></script>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #f0f2f5; color: #1a1a1a; font-family: 'Segoe UI', sans-serif; height: 100vh; display: flex; flex-direction: column; overflow: hidden; }
  header { background: #1a1a2e; color: white; padding: 12px 20px; display: flex; align-items: center; justify-content: space-between; flex-shrink: 0; }
  header h1 { font-size: 1rem; font-weight: 600; letter-spacing: 1px; }
  .header-right { display: flex; align-items: center; gap: 14px; }
  .conn-indicator { display: flex; align-items: center; gap: 6px; font-size: 0.75rem; color: #aaa; }
  .conn-dot { width: 8px; height: 8px; border-radius: 50%; background: #e74c3c; }
  .conn-dot.connected { background: #2ecc71; }
  .phase-tag { background: #2ecc71; color: #1a1a2e; font-size: 0.72rem; font-weight: 700; padding: 3px 10px; border-radius: 3px; text-transform: uppercase; letter-spacing: 1px; }
  .phase-tag.waypoint { background: #f39c12; }
  #estop-banner { display: none; background: #e74c3c; color: white; text-align: center; padding: 10px; font-size: 1rem; font-weight: 700; letter-spacing: 2px; flex-shrink: 0; animation: flash 0.7s infinite alternate; }
  @keyframes flash { from { background: #e74c3c; } to { background: #c0392b; } }
  .main { display: grid; grid-template-columns: 220px 1fr 220px; flex: 1; overflow: hidden; }
  .panel { background: white; border-right: 1px solid #ddd; padding: 14px; overflow-y: auto; display: flex; flex-direction: column; gap: 14px; }
  .panel.right { border-right: none; border-left: 1px solid #ddd; }
  .section-title { font-size: 0.65rem; text-transform: uppercase; letter-spacing: 1.5px; color: #888; font-weight: 600; margin-bottom: 6px; padding-bottom: 4px; border-bottom: 1px solid #eee; }
  .status-block { display: flex; flex-direction: column; gap: 10px; }
  .status-label { font-size: 0.62rem; color: #999; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 2px; }
  .status-value { font-size: 0.88rem; font-weight: 600; color: #1a1a2e; background: #f7f8fa; border: 1px solid #e8e8e8; border-radius: 4px; padding: 5px 8px; }
  .status-value.estop-state { background: #fdecea; border-color: #e74c3c; color: #e74c3c; }
  .status-value.ok { color: #27ae60; }
  .status-value.warn { color: #e67e22; }
  .coord-row { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 6px; }
  .coord-box { background: #f7f8fa; border: 1px solid #e8e8e8; border-radius: 4px; padding: 6px 4px; text-align: center; }
  .coord-box .lbl { font-size: 0.55rem; color: #aaa; text-transform: uppercase; }
  .coord-box .val { font-size: 0.9rem; font-weight: 700; color: #1a1a2e; }
  .camera-box { background: #f7f8fa; border: 1px solid #e8e8e8; border-radius: 4px; min-height: 110px; display: flex; align-items: center; justify-content: center; overflow: hidden; color: #bbb; font-size: 0.75rem; }
  .camera-box img { width: 100%; border-radius: 4px; }
  .centre { display: flex; flex-direction: column; background: #f7f8fa; overflow: hidden; }
  .map-toolbar { background: white; border-bottom: 1px solid #ddd; padding: 8px 14px; display: flex; align-items: center; justify-content: space-between; flex-shrink: 0; }
  .map-toolbar-title { font-size: 0.72rem; font-weight: 600; color: #555; text-transform: uppercase; letter-spacing: 1px; }
  .legend { display: flex; gap: 12px; }
  .legend-item { display: flex; align-items: center; gap: 5px; font-size: 0.65rem; color: #888; }
  .legend-dot { width: 8px; height: 8px; border-radius: 50%; }
  .map-wrapper { flex: 1; display: flex; align-items: center; justify-content: center; overflow: hidden; padding: 10px; position: relative; }
  #map-canvas { border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.1); display: block; }
  .marker-card { background: #f7f8fa; border: 1px solid #e8e8e8; border-radius: 4px; padding: 8px; margin-bottom: 8px; }
  .marker-card .mc-label { font-size: 0.78rem; font-weight: 700; color: #1a1a2e; margin-bottom: 2px; }
  .marker-card .mc-pos { font-size: 0.65rem; color: #999; margin-bottom: 5px; }
  .marker-card img { width: 100%; border-radius: 3px; }
  .no-data { font-size: 0.75rem; color: #bbb; }
</style>
</head>
<body>
<header>
  <h1>Pioneer 3-AT &mdash; Mission Control</h1>
  <div class="header-right">
    <div class="conn-indicator">
      <div class="conn-dot" id="conn-dot"></div>
      <span id="conn-label">Connecting...</span>
    </div>
    <div class="phase-tag" id="phase-tag">Phase 1: Mapping</div>
  </div>
</header>
<div id="estop-banner">&#9940; EMERGENCY STOP &mdash; MOVING OBSTACLE DETECTED</div>
<div class="main">
  <div class="panel">
    <div>
      <div class="section-title">Robot Status</div>
      <div class="status-block">
        <div class="status-item">
          <div class="status-label">State</div>
          <div class="status-value ok" id="robot-state">Initialising</div>
        </div>
        <div class="status-item">
          <div class="status-label">Intended Action</div>
          <div class="status-value" id="robot-action">Starting up...</div>
        </div>
        <div class="status-item">
          <div class="status-label">Phase</div>
          <div class="status-value" id="robot-phase">Mapping</div>
        </div>
      </div>
    </div>
    <div>
      <div class="section-title">Position</div>
      <div class="coord-row">
        <div class="coord-box"><div class="lbl">X (m)</div><div class="val" id="pos-x">0</div></div>
        <div class="coord-box"><div class="lbl">Y (m)</div><div class="val" id="pos-y">0</div></div>
        <div class="coord-box"><div class="lbl">Yaw</div><div class="val" id="pos-yaw">0&deg;</div></div>
      </div>
    </div>
    <div>
      <div class="section-title">Camera Feed</div>
      <div class="camera-box" id="camera-feed">No feed yet</div>
    </div>
  </div>
  <div class="centre">
    <div class="map-toolbar">
      <span class="map-toolbar-title" id="map-title">Phase 1 &mdash; Live Map</span>
      <div class="legend">
        <div class="legend-item"><div class="legend-dot" style="background:#3498db"></div>Robot</div>
        <div class="legend-item"><div class="legend-dot" style="background:#2ecc71"></div>Marker</div>
        <div class="legend-item"><div class="legend-dot" style="background:#f39c12"></div>Path</div>
        <div class="legend-item"><div class="legend-dot" style="background:#e74c3c"></div>Goal</div>
      </div>
    </div>
    <div class="map-wrapper" id="map-wrapper">
      <canvas id="map-canvas"></canvas>
    </div>
  </div>
  <div class="panel right">
    <div>
      <div class="section-title" id="right-title">Detected Markers</div>
      <div id="right-content"><p class="no-data">No markers detected yet.</p></div>
    </div>
  </div>
</div>
<script>
const socket  = io();
const canvas  = document.getElementById('map-canvas');
const ctx     = canvas.getContext('2d');
const wrapper = document.getElementById('map-wrapper');

let currentPhase = 'mapping';
let mapImg       = null;
let mapInfo      = null;
let robotPos     = { x: 0, y: 0, yaw: 0 };
let markers      = [];
let path         = [];

socket.on('connect',    () => { document.getElementById('conn-dot').classList.add('connected');    document.getElementById('conn-label').textContent = 'Connected'; });
socket.on('disconnect', () => { document.getElementById('conn-dot').classList.remove('connected'); document.getElementById('conn-label').textContent = 'Disconnected'; });

socket.on('full_state', d => {
  if (d.phase)           setPhase(d.phase);
  if (d.robot_state)     setRobotState(d.robot_state);
  if (d.intended_action) document.getElementById('robot-action').textContent = d.intended_action;
  if (d.position)        updatePos(d.position);
  if (d.map)             loadMap(d.map, d.map_info);
  if (d.markers)         { markers = d.markers; renderMarkers(); }
  if (d.path)            { path = d.path; }
  if (d.estop)           setEstop(d.estop);
  draw();
});

socket.on('map_update',      d => { loadMap(d.map, d.map_info); });
socket.on('position_update', d => { updatePos(d); draw(); });
socket.on('path_update',     d => { path = d.path; draw(); updatePathPanel(); });
socket.on('state_update',    d => { setRobotState(d.robot_state); });
socket.on('action_update',   d => { document.getElementById('robot-action').textContent = d.intended_action; });
socket.on('phase_update',    d => { setPhase(d.phase); });
socket.on('estop_update',    d => { setEstop(d.estop); });
socket.on('markers_update',  d => { markers = d.markers; renderMarkers(); draw(); });
socket.on('camera_update',   d => {
  const f = document.getElementById('camera-feed');
  f.innerHTML = '<img src="data:image/jpeg;base64,' + d.image + '" alt="cam">';
});

function loadMap(b64, info) {
  mapInfo = info;
  const img = new Image();
  img.onload = () => { mapImg = img; fitCanvas(img.width, img.height); draw(); };
  img.src = 'data:image/png;base64,' + b64;
}

function fitCanvas(imgW, imgH) {
  const r = wrapper.getBoundingClientRect();
  const scale = Math.min((r.width - 20) / imgW, (r.height - 20) / imgH, 1);
  canvas.width  = Math.round(imgW * scale);
  canvas.height = Math.round(imgH * scale);
}

window.addEventListener('resize', () => { if (mapImg) { fitCanvas(mapImg.width, mapImg.height); draw(); } });

function w2c(wx, wy) {
  if (!mapImg || !mapInfo) return { cx: canvas.width / 2, cy: canvas.height / 2 };
  const scaleX = canvas.width  / mapImg.width;
  const scaleY = canvas.height / mapImg.height;
  const px = (wx - mapInfo.origin_x) / mapInfo.resolution;
  const py = mapImg.height - (wy - mapInfo.origin_y) / mapInfo.resolution;
  return { cx: px * scaleX, cy: py * scaleY };
}

function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (mapImg) {
    ctx.drawImage(mapImg, 0, 0, canvas.width, canvas.height);
  } else {
    ctx.fillStyle = '#e8eaed'; ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = '#bbb'; ctx.font = '14px Segoe UI'; ctx.textAlign = 'center';
    ctx.fillText('Waiting for map...', canvas.width / 2, canvas.height / 2);
  }
  if (path.length > 1) {
    ctx.beginPath(); ctx.strokeStyle = '#f39c12'; ctx.lineWidth = 2;
    const p0 = w2c(path[0].x, path[0].y); ctx.moveTo(p0.cx, p0.cy);
    for (let i = 1; i < path.length; i++) { const p = w2c(path[i].x, path[i].y); ctx.lineTo(p.cx, p.cy); }
    ctx.stroke();
    const last = w2c(path[path.length-1].x, path[path.length-1].y);
    ctx.beginPath(); ctx.arc(last.cx, last.cy, 6, 0, 2*Math.PI); ctx.fillStyle = '#e74c3c'; ctx.fill();
  }
  markers.forEach(m => {
    const {cx, cy} = w2c(m.x, m.y);
    ctx.beginPath(); ctx.arc(cx, cy, 6, 0, 2*Math.PI); ctx.fillStyle = '#2ecc71'; ctx.fill();
    ctx.fillStyle = '#1a1a2e'; ctx.font = 'bold 9px Segoe UI'; ctx.textAlign = 'center';
    ctx.fillText(m.label || '?', cx, cy - 10);
  });
  const {cx, cy} = w2c(robotPos.x, robotPos.y);
  const yr = -robotPos.yaw * Math.PI / 180;
  ctx.save(); ctx.translate(cx, cy); ctx.rotate(yr);
  ctx.beginPath(); ctx.moveTo(0, -12); ctx.lineTo(8, 8); ctx.lineTo(-8, 8); ctx.closePath();
  ctx.fillStyle = '#3498db'; ctx.fill(); ctx.restore();
}

function updatePos(p) {
  robotPos = p;
  document.getElementById('pos-x').textContent   = p.x;
  document.getElementById('pos-y').textContent   = p.y;
  document.getElementById('pos-yaw').textContent = p.yaw + '\u00b0';
}

function setRobotState(s) {
  const el = document.getElementById('robot-state');
  el.textContent = s; el.className = 'status-value';
  if (s === 'EMERGENCY STOP')                                    el.classList.add('estop-state');
  else if (['Exploring','Navigating','Mapping','Active'].includes(s)) el.classList.add('ok');
  else if (['Paused','Waiting'].includes(s))                     el.classList.add('warn');
}

function setEstop(active) {
  document.getElementById('estop-banner').style.display = active ? 'block' : 'none';
  if (active) {
    setRobotState('EMERGENCY STOP');
    document.getElementById('robot-action').textContent = 'Halted \u2014 moving obstacle within 1m';
  }
}

function setPhase(phase) {
  currentPhase = phase;
  const tag = document.getElementById('phase-tag');
  const phaseEl = document.getElementById('robot-phase');
  if (phase === 'mapping') {
    tag.textContent = 'Phase 1: Mapping'; tag.className = 'phase-tag';
    document.getElementById('map-title').textContent   = 'Phase 1 \u2014 Live Map';
    document.getElementById('right-title').textContent = 'Detected Markers';
    phaseEl.textContent = 'Mapping';
    renderMarkers();
  } else {
    tag.textContent = 'Phase 2: Waypoint Drive'; tag.className = 'phase-tag waypoint';
    document.getElementById('map-title').textContent   = 'Phase 2 \u2014 Planned Path';
    document.getElementById('right-title').textContent = 'Waypoint Path';
    phaseEl.textContent = 'Waypoint Driving';
    updatePathPanel();
  }
  draw();
}

function renderMarkers() {
  const el = document.getElementById('right-content');
  if (!markers.length) { el.innerHTML = '<p class="no-data">No markers detected yet.</p>'; return; }
  el.innerHTML = markers.map(m => `
    <div class="marker-card">
      <div class="mc-label">${m.label || 'Unknown'}</div>
      <div class="mc-pos">X: ${(m.x||0).toFixed(2)}m &nbsp; Y: ${(m.y||0).toFixed(2)}m</div>
      ${m.photo_b64 ? '<img src="data:image/jpeg;base64,'+m.photo_b64+'" alt="photo">' : '<span style="font-size:0.7rem;color:#bbb">No photo</span>'}
    </div>`).join('');
}

function updatePathPanel() {
  if (currentPhase !== 'waypoint') return;
  const el = document.getElementById('right-content');
  if (!path.length) { el.innerHTML = '<p class="no-data">No path planned yet.</p>'; return; }
  el.innerHTML = '<p style="font-size:0.72rem;color:#999;margin-bottom:8px">'+path.length+' points in planned path</p>'
    + path.map((p,i) => `
      <div class="marker-card">
        <div class="mc-label">Point ${i+1}</div>
        <div class="mc-pos">X: ${p.x.toFixed(2)}m &nbsp; Y: ${p.y.toFixed(2)}m</div>
      </div>`).join('');
}

canvas.width = 600; canvas.height = 600; draw();
</script>
</body>
</html>"""


def occupancy_grid_to_png_b64(msg):
    try:
        from PIL import Image as PILImage
        import io
        width  = msg.info.width
        height = msg.info.height
        data   = np.array(msg.data, dtype=np.int8).reshape((height, width))
        img    = np.zeros((height, width, 3), dtype=np.uint8)
        img[data == -1] = [200, 200, 200]
        img[data == 0]  = [245, 245, 245]
        img[data > 50]  = [50,  50,  50]
        img = np.flipud(img)
        pil_img = PILImage.fromarray(img, 'RGB')
        buf = io.BytesIO()
        pil_img.save(buf, format='PNG')
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception:
        return None


def ros_image_to_b64(msg):
    try:
        from PIL import Image as PILImage
        import io
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
        if msg.encoding == 'bgr8':
            arr = arr[:, :, ::-1]
        pil_img = PILImage.fromarray(arr, 'RGB')
        buf = io.BytesIO()
        pil_img.save(buf, format='JPEG', quality=75)
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception:
        return None


def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class UINode(Node):
    def __init__(self):
        super().__init__('ui_node')
        self._last_map_push   = 0.0
        self._last_image_push = 0.0

        self.create_subscription(OccupancyGrid, '/map',              self.map_cb,      10)
        self.create_subscription(Odometry,      '/odom',             self.odom_cb,     10)
        self.create_subscription(Path,          '/plan',             self.path_cb,     10)
        self.create_subscription(String,        '/robot_state',      self.state_cb,    10)
        self.create_subscription(String,        '/robot_action',     self.action_cb,   10)
        self.create_subscription(String,        '/robot_phase',      self.phase_cb,    10)
        self.create_subscription(String,        '/estop_status',     self.estop_cb,    10)
        self.create_subscription(String,        '/detected_objects', self.detected_cb, 10)
        self.create_subscription(Image,         '/camera/image',     self.image_cb,    10)

        self.get_logger().info('UI node running — open http://localhost:5000 in your browser.')

    def odom_cb(self, msg):
        yaw = yaw_from_quaternion(msg.pose.pose.orientation)
        pos = {
            'x':   round(msg.pose.pose.position.x, 3),
            'y':   round(msg.pose.pose.position.y, 3),
            'yaw': round(math.degrees(yaw), 1)
        }
        with state_lock:
            state['position'] = pos
        socketio.emit('position_update', pos)

    def map_cb(self, msg):
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self._last_map_push < 2.0:
            return
        self._last_map_push = now
        png_b64 = occupancy_grid_to_png_b64(msg)
        if png_b64:
            with state_lock:
                state['map'] = png_b64
                state['map_info'] = {
                    'resolution': msg.info.resolution,
                    'origin_x':   msg.info.origin.position.x,
                    'origin_y':   msg.info.origin.position.y,
                    'width':      msg.info.width,
                    'height':     msg.info.height,
                }
            socketio.emit('map_update', {'map': png_b64, 'map_info': state['map_info']})

    def path_cb(self, msg):
        pts = [{'x': p.pose.position.x, 'y': p.pose.position.y} for p in msg.poses]
        with state_lock:
            state['path'] = pts
        socketio.emit('path_update', {'path': pts})

    def state_cb(self, msg):
        with state_lock:
            state['robot_state'] = msg.data
        socketio.emit('state_update', {'robot_state': msg.data})

    def action_cb(self, msg):
        with state_lock:
            state['intended_action'] = msg.data
        socketio.emit('action_update', {'intended_action': msg.data})

    def phase_cb(self, msg):
        with state_lock:
            state['phase'] = msg.data
        socketio.emit('phase_update', {'phase': msg.data})

    def estop_cb(self, msg):
        active = msg.data.lower() == 'active'
        with state_lock:
            state['estop'] = active
        socketio.emit('estop_update', {'estop': active})

    def detected_cb(self, msg):
        try:
            obj = json.loads(msg.data)
            with state_lock:
                obj['photo_b64'] = state.get('_latest_image_b64')
                exists = any(
                    m['label'] == obj.get('label') and
                    abs(m['x'] - obj.get('x', 0)) < 0.5
                    for m in state['markers']
                )
                if not exists:
                    state['markers'].append(obj)
                markers = list(state['markers'])
            socketio.emit('markers_update', {'markers': markers})
        except Exception:
            pass

    def image_cb(self, msg):
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self._last_image_push < 0.5:
            return
        self._last_image_push = now
        b64 = ros_image_to_b64(msg)
        if b64:
            with state_lock:
                state['_latest_image_b64'] = b64
            socketio.emit('camera_update', {'image': b64})


@app.route('/')
def index():
    return Response(HTML, mimetype='text/html')

@app.route('/state')
def get_state():
    with state_lock:
        return json.dumps({k: v for k, v in state.items() if not k.startswith('_')})

@socketio.on('connect')
def on_connect():
    with state_lock:
        s = {k: v for k, v in state.items() if not k.startswith('_')}
    socketio.emit('full_state', s)


def ros_spin(node):
    rclpy.spin(node)

def main(args=None):
    try:
        from PIL import Image
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'Pillow', '--break-system-packages'])

    rclpy.init(args=args)
    node = UINode()
    ros_thread = threading.Thread(target=ros_spin, args=(node,), daemon=True)
    ros_thread.start()
    socketio.run(app, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
