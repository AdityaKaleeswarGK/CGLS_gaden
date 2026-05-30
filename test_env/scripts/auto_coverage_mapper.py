#!/usr/bin/env python3
"""
Adaptive Coverage Mapper – Two-phase BCD sweep with CUSUM spike detection.

Phase 1: Coarse BCD sweep (wide step) surveys the whole environment.
          A CUSUM change-point detector monitors gas readings in real-time.
          When it fires, the (x,y) region is marked as a hotspot.

Phase 2: For each detected hotspot, a tight local grid is generated and
          swept for dense sampling near the gas source.

The CUSUM detector is self-calibrating — it learns baseline mean and
variance from the first N samples, so no hardcoded ppm thresholds are
needed for spike detection.
"""
import os
import csv
import time
import math
import threading
from datetime import datetime
import yaml
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from nav_msgs.msg import OccupancyGrid
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import LaserScan
from olfaction_msgs.msg import GasSensor
from action_msgs.msg import GoalStatus
from std_srvs.srv import Empty
try:
    import tf2_ros
    from tf2_ros import TransformException, Buffer, TransformListener
except ImportError:
    pass


# ────────────────────────────────────────────────────────────
#  CUSUM Change-Point Detector
# ────────────────────────────────────────────────────────────
class CUSUMDetector:
    """
    Cumulative Sum (CUSUM) algorithm for online change-point detection.

    Math:
        S_n = max(0, S_{n-1} + x_n - mu_0 - drift)

        When S_n > threshold_h → declare a spike.

    Self-calibrating: collects `warmup` samples to estimate baseline
    mean (mu_0) and stddev (sigma). Then:
        drift = k * sigma      (sensitivity: lower k = more sensitive)
        threshold_h = h * sigma (decision boundary)

    O(1) per sample, no windowing, no hardcoded ppm values.
    """

    def __init__(self, warmup: int = 50, k: float = 0.5, h: float = 4.0):
        self.warmup = warmup
        self.k_factor = k
        self.h_factor = h

        self._samples = []
        self._calibrated = False
        self._mu = 0.0
        self._sigma = 1.0
        self._drift = 0.0
        self._threshold = float('inf')
        self._S = 0.0  # cumulative sum

    def update(self, value: float) -> bool:
        """Feed a new sample. Returns True if a spike is active."""
        if not self._calibrated:
            self._samples.append(value)
            if len(self._samples) >= self.warmup:
                self._calibrate()
            return False

        self._S = max(0.0, self._S + value - self._mu - self._drift)
        if self._S >= self._threshold:
            return True
        return False

    def reset_alarm(self):
        """Call after handling a spike to reset the cumulative sum."""
        self._S = 0.0

    def is_calibrated(self) -> bool:
        return self._calibrated

    @property
    def current_S(self) -> float:
        return self._S

    @property
    def threshold(self) -> float:
        return self._threshold

    def _calibrate(self):
        arr = np.array(self._samples)
        self._mu = float(np.mean(arr))
        self._sigma = max(float(np.std(arr)), 0.01)  # floor to avoid div/0
        self._drift = self.k_factor * self._sigma
        # Threshold: at least 1.0 ppm absolute (prevents triggering on noise
        # when baseline is zero, e.g. robot starts far from gas source)
        self._threshold = max(self.h_factor * self._sigma, 1.0)
        self._calibrated = True
        self._samples = []  # free memory


class AutoCoverageMapper(Node):
    def __init__(self):
        super().__init__('auto_coverage_mapper')
        self.set_parameters([rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)])

        self.declare_parameter('namespace', 'PioneerP3DX')
        self.declare_parameter('sensor_topics', [
            '/gas1/Sensor_reading',
            '/gas2/Sensor_reading',
            '/gas3/Sensor_reading',
            '/gas4/Sensor_reading',
        ])
        # Adaptive sweep parameters
        self.declare_parameter('coarse_step', 0.5)        # Phase 1: dense sweep
        self.declare_parameter('fine_step', 0.4)          # Phase 2: tight sweep in hotspots
        self.declare_parameter('hotspot_pad', 1.5)         # meters padding around hotspot center
        self.declare_parameter('safe_distance', 0.3)
        self.declare_parameter('nav_timeout', 30.0)
        self.declare_parameter('wp_tolerance', 0.5)
        # CUSUM tuning
        self.declare_parameter('cusum_warmup', 50)         # samples before calibration
        self.declare_parameter('cusum_k', 0.75)            # drift sensitivity (higher = less sensitive)
        self.declare_parameter('cusum_h', 5.0)             # decision threshold (higher = fewer triggers)
        # Hotspot validation gates — spike must pass ALL of these to become a fine-sweep region
        self.declare_parameter('min_spike_samples', 15)    # spike must last at least N samples
        self.declare_parameter('min_spike_peak_sigma', 3.0)  # peak must be > mean + N*sigma of baseline
        self.declare_parameter('max_hotspots', 3)          # cap total fine-sweep regions
        # Slowdown
        self.declare_parameter('gas_slowdown_speed', 0.15)
        # Output control
        self.declare_parameter('generate_plots', False)
        # Scenario config path (for README generation)
        self.declare_parameter('scenario_path', '')

        self.ns = self.get_parameter('namespace').value
        self.topics = self.get_parameter('sensor_topics').value
        self.coarse_step = self.get_parameter('coarse_step').value
        self.fine_step = self.get_parameter('fine_step').value
        self.hotspot_pad = self.get_parameter('hotspot_pad').value
        self.safe_distance = self.get_parameter('safe_distance').value
        self.nav_timeout = self.get_parameter('nav_timeout').value
        self.wp_tolerance = self.get_parameter('wp_tolerance').value
        self.gas_slowdown_speed = self.get_parameter('gas_slowdown_speed').value
        self.generate_plots = self.get_parameter('generate_plots').value
        self.scenario_path = self.get_parameter('scenario_path').value
        self.min_spike_samples = self.get_parameter('min_spike_samples').value
        self.min_spike_peak_sigma = self.get_parameter('min_spike_peak_sigma').value
        self.max_hotspots = self.get_parameter('max_hotspots').value

        # Sensor type mapping: topic → gas name for CSV gas_type column
        self._sensor_type_map = {
            '/gas1/Sensor_reading': 'ethanol',
            '/gas2/Sensor_reading': 'methane',
            '/gas3/Sensor_reading': 'hydrogen',
            '/gas4/Sensor_reading': 'propanol',
        }

        # CUSUM detector — one per sensor topic (independent calibration & thresholds)
        cusum_warmup = self.get_parameter('cusum_warmup').value
        cusum_k = self.get_parameter('cusum_k').value
        cusum_h = self.get_parameter('cusum_h').value
        self._cusum_params = (cusum_warmup, cusum_k, cusum_h)

        self._cusums = {}              # topic → CUSUMDetector
        self._cusum_spiking = {}       # topic → bool
        self._spike_positions = {}     # topic → [(x, y, ppm), ...] during active spike
        for t in self.topics:
            self._cusums[t] = CUSUMDetector(warmup=cusum_warmup, k=cusum_k, h=cusum_h)
            self._cusum_spiking[t] = False
            self._spike_positions[t] = []

        self.cbg = ReentrantCallbackGroup()
        self.nav_client = ActionClient(self, NavigateToPose, f'/{self.ns}/navigate_to_pose', callback_group=self.cbg)
        self.cmd_vel_pub = self.create_publisher(Twist, f'/{self.ns}/cmd_vel', 10)

        self.all_waypoints = []        # coarse path (for plotting)
        self.refinement_waypoints = [] # fine path (for plotting)
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.start_x = 0.0
        self.start_y = 0.0
        self._running = True
        self._any_cusum_spiking = False  # True if ANY sensor is currently spiking

        # Hotspot bounding boxes: {topic: [(cx, cy, peak_ppm), ...]}
        self._hotspots_per_topic = {t: [] for t in self.topics}
        # Merged hotspots for fine sweep (union of all topics)
        self._hotspots = []

        self.clear_local = self.create_client(Empty, f'/{self.ns}/local_costmap/clear_entirely_local_costmap')
        self.clear_global = self.create_client(Empty, f'/{self.ns}/global_costmap/clear_entirely_global_costmap')

        # Laser scan for smart recovery — find open directions
        self._latest_scan = None
        self.create_subscription(
            LaserScan, f'/{self.ns}/laser_scan',
            self._laser_cb, 10, callback_group=self.cbg
        )

        self.dataset = {}
        for t in self.topics:
            self.dataset[t] = {'x': [], 'y': [], 'ppm': [], 'timestamp': []}
            self.create_subscription(GasSensor, t, lambda msg, t=t: self.sensor_cb(msg, t), 10, callback_group=self.cbg)

        lqos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL, reliability=QoSReliabilityPolicy.RELIABLE)
        map_topic = f'/{self.ns}/map'
        self.get_logger().info(f"Subscribing to map on: {map_topic}")
        self.create_subscription(OccupancyGrid, map_topic, self.map_cb, lqos, callback_group=self.cbg)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.saved_map_msg = None
        self.map_data = None
        self.map_info = None
        self.path_generated = False
        self.path_timer = self.create_timer(1.0, self.generate_path, callback_group=self.cbg)

        # ── Data persistence ──
        # Each run gets its own timestamped folder
        self._run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._run_dir = os.path.expanduser(f'~/gaden_results/auto_coverage/{self._run_id}')
        self._data_dir = self._run_dir
        os.makedirs(self._run_dir, exist_ok=True)
        self._csv_path = os.path.join(self._run_dir, f'readings.csv')
        self._csv_file = open(self._csv_path, 'w', newline='')
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(['timestamp', 'x', 'y', 'gas_type', 'ppm', 'phase'])
        self._csv_file.flush()
        self._current_phase = 'coarse'
        self._sample_count = 0

        # Live concentration grid publisher (OccupancyGrid-style, 0-100 scaled)
        self._conc_grid_res = 0.2  # 20cm cells
        self._conc_grid = None     # initialized after map received
        self._conc_count = None    # sample count per cell (for running average)
        self._conc_pub = self.create_publisher(
            OccupancyGrid, f'/{self.ns}/concentration_grid',
            QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                       reliability=QoSReliabilityPolicy.RELIABLE)
        )
        self._conc_publish_timer = None  # started after map received

        self.get_logger().info(
            f"Adaptive Coverage Mapper initialized. "
            f"Coarse={self.coarse_step}m, Fine={self.fine_step}m, "
            f"CUSUM k={cusum_k}, h={cusum_h}. "
            f"CSV log: {self._csv_path}"
        )

    # ────────────────────────────────────────────────────────────
    #  Callbacks
    # ────────────────────────────────────────────────────────────
    def _update_robot_pose(self):
        try:
            t = self.tf_buffer.lookup_transform('map', f'{self.ns}_base_link', rclpy.time.Time())
            self.robot_x = t.transform.translation.x
            self.robot_y = t.transform.translation.y
            return True
        except Exception:
            return False

    def sensor_cb(self, msg, topic):
        if not self._running or self._csv_file.closed:
            return
        if self._update_robot_pose():
            now = self.get_clock().now().nanoseconds / 1e9
            self.dataset[topic]['x'].append(self.robot_x)
            self.dataset[topic]['y'].append(self.robot_y)
            self.dataset[topic]['ppm'].append(msg.raw)
            self.dataset[topic]['timestamp'].append(now)

            # Stream to CSV (real-time persistence)
            gas_type = self._sensor_type_map.get(topic, topic)
            self._csv_writer.writerow([
                f'{now:.3f}',
                f'{self.robot_x:.4f}', f'{self.robot_y:.4f}',
                gas_type, f'{msg.raw:.4f}', self._current_phase
            ])
            self._sample_count += 1
            if self._sample_count % 10 == 0:
                self._csv_file.flush()  # frequent flush — survives Ctrl+C

            # Update live concentration grid
            self._update_conc_grid(self.robot_x, self.robot_y, msg.raw)

            # Feed this topic's own CUSUM
            cusum = self._cusums[topic]
            spiking = cusum.update(msg.raw)

            if spiking and not self._cusum_spiking[topic]:
                self._cusum_spiking[topic] = True
                self._spike_positions[topic] = []
                self.get_logger().info(
                    f"CUSUM SPIKE [{topic}] at ({self.robot_x:.2f}, {self.robot_y:.2f}) "
                    f"ppm={msg.raw:.2f}, S={cusum.current_S:.2f}/{cusum.threshold:.2f}"
                )
            if spiking:
                self._spike_positions[topic].append((self.robot_x, self.robot_y, msg.raw))
            elif self._cusum_spiking[topic]:
                # Spike ended for this topic
                self._finalize_hotspot(topic)
                self._cusum_spiking[topic] = False
                cusum.reset_alarm()

            # Update aggregate flag (used for slowdown during navigation)
            self._any_cusum_spiking = any(self._cusum_spiking.values())

    def _finalize_hotspot(self, topic):
        """Convert accumulated spike samples into a hotspot — only if it passes validation."""
        positions = self._spike_positions[topic]
        if not positions:
            return
        arr = np.array(positions)  # (N, 3): x, y, ppm
        num_samples = len(arr)
        peak_ppm = float(arr[:, 2].max())

        # ── Gate 1: Minimum duration (not a transient blip) ──
        if num_samples < self.min_spike_samples:
            self.get_logger().info(
                f"Spike REJECTED [{topic}]: only {num_samples} samples "
                f"(need {self.min_spike_samples}). Transient noise."
            )
            self._spike_positions[topic] = []
            return

        # ── Gate 2: Minimum absolute concentration ──
        min_abs_ppm = 2.0  # reject hotspots below 2 ppm peak — clearly not a source
        if peak_ppm < min_abs_ppm:
            self.get_logger().info(
                f"Spike REJECTED [{topic}]: peak {peak_ppm:.2f} ppm < "
                f"minimum absolute threshold {min_abs_ppm} ppm. Background dispersion."
            )
            self._spike_positions[topic] = []
            return

        # ── Gate 3: Peak must be significantly above baseline ──
        cusum = self._cusums[topic]
        if cusum.is_calibrated():
            baseline_threshold = cusum._mu + self.min_spike_peak_sigma * cusum._sigma
            if peak_ppm < baseline_threshold:
                self.get_logger().info(
                    f"Spike REJECTED [{topic}]: peak {peak_ppm:.2f} ppm < "
                    f"baseline threshold {baseline_threshold:.2f} ppm "
                    f"(mean={cusum._mu:.2f} + {self.min_spike_peak_sigma}σ={cusum._sigma:.2f})"
                )
                self._spike_positions[topic] = []
                return

        # ── Passed validation — compute weighted centroid ──
        weights = arr[:, 2]
        total_w = weights.sum()
        if total_w > 0:
            cx = float(np.average(arr[:, 0], weights=weights))
            cy = float(np.average(arr[:, 1], weights=weights))
        else:
            cx = float(np.mean(arr[:, 0]))
            cy = float(np.mean(arr[:, 1]))

        # Merge with existing hotspot for this topic if overlapping
        topic_hotspots = self._hotspots_per_topic[topic]
        for i, (hx, hy, hp) in enumerate(topic_hotspots):
            if math.hypot(cx - hx, cy - hy) < self.hotspot_pad:
                if peak_ppm > hp:
                    topic_hotspots[i] = (cx, cy, peak_ppm)
                self.get_logger().info(
                    f"Hotspot merged [{topic}] at ({cx:.2f}, {cy:.2f}), peak={peak_ppm:.1f}"
                )
                self._spike_positions[topic] = []
                return

        topic_hotspots.append((cx, cy, peak_ppm))
        self.get_logger().info(
            f"CONFIRMED HOTSPOT [{topic}] #{len(topic_hotspots)} at ({cx:.2f}, {cy:.2f}), "
            f"peak={peak_ppm:.1f} ppm, {num_samples} samples"
        )
        self._spike_positions[topic] = []

    def _merge_all_hotspots(self):
        """Merge hotspots from all topics into one list for fine sweep.
        Overlapping hotspots from different gases get combined.
        Capped at max_hotspots, keeping the highest-peak ones."""
        self._hotspots = []
        for topic, topic_hotspots in self._hotspots_per_topic.items():
            for cx, cy, peak_ppm in topic_hotspots:
                merged = False
                for i, (hx, hy, hp) in enumerate(self._hotspots):
                    if math.hypot(cx - hx, cy - hy) < self.hotspot_pad:
                        if peak_ppm > hp:
                            self._hotspots[i] = (cx, cy, peak_ppm)
                        merged = True
                        break
                if not merged:
                    self._hotspots.append((cx, cy, peak_ppm))

        # Cap at max_hotspots — keep the strongest ones
        if len(self._hotspots) > self.max_hotspots:
            self._hotspots.sort(key=lambda h: h[2], reverse=True)
            dropped = len(self._hotspots) - self.max_hotspots
            self._hotspots = self._hotspots[:self.max_hotspots]
            self.get_logger().info(f"Capped hotspots: dropped {dropped} weakest regions")

        self.get_logger().info(
            f"Merged hotspots from {len(self.topics)} topics → "
            f"{len(self._hotspots)} regions for fine sweep"
        )

    def map_cb(self, msg: OccupancyGrid):
        self.saved_map_msg = msg
        # Initialize live concentration grid once we know map bounds
        if self._conc_grid is None:
            self._init_conc_grid(msg)

    # ────────────────────────────────────────────────────────────
    #  Live Concentration Grid (published as OccupancyGrid for RViz)
    # ────────────────────────────────────────────────────────────
    def _init_conc_grid(self, map_msg):
        """Create an empty concentration grid matching the map bounds."""
        res = self._conc_grid_res
        self._conc_ox = map_msg.info.origin.position.x
        self._conc_oy = map_msg.info.origin.position.y
        map_w = map_msg.info.width * map_msg.info.resolution
        map_h = map_msg.info.height * map_msg.info.resolution
        self._conc_w = int(math.ceil(map_w / res))
        self._conc_h = int(math.ceil(map_h / res))
        self._conc_grid = np.zeros((self._conc_h, self._conc_w), dtype=np.float64)
        self._conc_count = np.zeros((self._conc_h, self._conc_w), dtype=np.int32)
        self._conc_max_ppm = 1.0  # auto-scales as readings come in
        # Publish at 1 Hz
        self._conc_publish_timer = self.create_timer(1.0, self._publish_conc_grid, callback_group=self.cbg)
        self.get_logger().info(
            f"Concentration grid initialized: {self._conc_w}x{self._conc_h} cells "
            f"at {res}m resolution. Publishing on /{self.ns}/concentration_grid"
        )

    def _update_conc_grid(self, x, y, ppm):
        """Update running average concentration for the cell at (x, y)."""
        if self._conc_grid is None:
            return
        c = int((x - self._conc_ox) / self._conc_grid_res)
        r = int((y - self._conc_oy) / self._conc_grid_res)
        if 0 <= r < self._conc_h and 0 <= c < self._conc_w:
            # Incremental mean: new_mean = old_mean + (value - old_mean) / (count + 1)
            self._conc_count[r, c] += 1
            n = self._conc_count[r, c]
            self._conc_grid[r, c] += (ppm - self._conc_grid[r, c]) / n
            self._conc_max_ppm = max(self._conc_max_ppm, self._conc_grid[r, c])

    def _publish_conc_grid(self):
        """Publish concentration grid as OccupancyGrid (0-100 scaled)."""
        if self._conc_grid is None:
            return
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.info.resolution = self._conc_grid_res
        msg.info.width = self._conc_w
        msg.info.height = self._conc_h
        msg.info.origin.position.x = self._conc_ox
        msg.info.origin.position.y = self._conc_oy

        # Scale to 0-100: unvisited cells = -1 (unknown), visited = 0-100
        scaled = np.full((self._conc_h, self._conc_w), -1, dtype=np.int8)
        visited = self._conc_count > 0
        if self._conc_max_ppm > 0:
            scaled[visited] = np.clip(
                (self._conc_grid[visited] / self._conc_max_ppm * 100).astype(np.int8),
                0, 100
            )
        msg.data = scaled.flatten().tolist()
        self._conc_pub.publish(msg)

    # ────────────────────────────────────────────────────────────
    #  BCD Path Generation (parameterized by step size + optional bbox)
    # ────────────────────────────────────────────────────────────
    def _generate_bcd_waypoints(self, step, bbox=None):
        """
        Generate BCD lawnmower waypoints at the given step size.
        bbox: optional (xmin, ymin, xmax, ymax) to restrict the sweep area.
        Returns list of (x, y) waypoints.
        """
        msg = self.saved_map_msg
        res = msg.info.resolution
        ox = msg.info.origin.position.x
        oy = msg.info.origin.position.y
        w = msg.info.width
        h = msg.info.height
        data = self.map_data

        if bbox is not None:
            bx_min, by_min, bx_max, by_max = bbox
        else:
            bx_min, by_min = ox, oy
            bx_max, by_max = ox + w * res, oy + h * res

        endpoint_retract = 0.50
        laps = []
        cx = bx_min + step / 2.0
        col_idx = 0
        lap_id = 0

        while cx <= bx_max:
            ys = np.arange(by_min + step / 2.0, by_max, step)
            c_start = None
            c_end = None

            for cy in ys:
                r = int((cy - oy) / res)
                c = int((cx - ox) / res)
                if 0 <= r < h and 0 <= c < w:
                    s_cells = int(self.safe_distance / res)
                    rmin, rmax = max(0, r - s_cells), min(h - 1, r + s_cells)
                    cmin, cmax = max(0, c - s_cells), min(w - 1, c + s_cells)
                    chunk = data[rmin:rmax + 1, cmin:cmax + 1]
                    free = chunk.size > 0 and np.all((chunk >= 0) & (chunk < 40))

                    if free:
                        if c_start is None:
                            c_start = cy
                        c_end = cy
                    else:
                        if c_start is not None:
                            y1 = c_start + endpoint_retract
                            y2 = c_end - endpoint_retract
                            if y1 > y2:
                                y1 = y2 = (c_start + c_end) / 2.0
                            laps.append({'id': lap_id, 'x': cx, 'y1': y1, 'y2': y2, 'col': col_idx})
                            lap_id += 1
                            c_start = None

            if c_start is not None:
                y1 = c_start + endpoint_retract
                y2 = c_end - endpoint_retract
                if y1 > y2:
                    y1 = y2 = (c_start + c_end) / 2.0
                laps.append({'id': lap_id, 'x': cx, 'y1': y1, 'y2': y2, 'col': col_idx})
                lap_id += 1

            cx += step
            col_idx += 1

        if not laps:
            return []

        # Build adjacency
        lap_dict = {l['id']: l for l in laps}
        adjacency = {l['id']: [] for l in laps}
        for a in laps:
            for b in laps:
                if abs(a['col'] - b['col']) == 1:
                    if max(a['y1'], b['y1']) <= min(a['y2'], b['y2']) + (step / 2.0):
                        adjacency[a['id']].append(b['id'])

        # Route via adjacency-first DFS
        unvisited = set(l['id'] for l in laps)
        self._update_robot_pose()
        rx, ry = self.robot_x, self.robot_y

        def dist_to_lap(lap, r_x, r_y):
            dx = lap['x'] - r_x
            dy = 0 if lap['y1'] <= r_y <= lap['y2'] else min(abs(lap['y1'] - r_y), abs(lap['y2'] - r_y))
            return dx * dx + dy * dy

        current_lap_id = min(laps, key=lambda l: dist_to_lap(l, rx, ry))['id']
        curr_lap = lap_dict[current_lap_id]
        if abs(curr_lap['y1'] - ry) < abs(curr_lap['y2'] - ry):
            y_start, y_end = curr_lap['y1'], curr_lap['y2']
        else:
            y_start, y_end = curr_lap['y2'], curr_lap['y1']

        routed = []
        while unvisited:
            unvisited.discard(current_lap_id)
            lap = lap_dict[current_lap_id]
            if abs(lap['y1'] - lap['y2']) > 0.05:
                routed.append((lap['x'], y_start))
                routed.append((lap['x'], y_end))
            else:
                routed.append((lap['x'], y_start))
            last_x, last_y = lap['x'], y_end
            if not unvisited:
                break

            adj_unvisited = [nid for nid in adjacency[current_lap_id] if nid in unvisited]
            if adj_unvisited:
                best_adj, best_dist = None, float('inf')
                best_ys, best_ye = None, None
                for nid in adj_unvisited:
                    n_lap = lap_dict[nid]
                    d1, d2 = abs(n_lap['y1'] - last_y), abs(n_lap['y2'] - last_y)
                    if d1 < d2:
                        dist, ns, ne = d1, n_lap['y1'], n_lap['y2']
                    else:
                        dist, ns, ne = d2, n_lap['y2'], n_lap['y1']
                    if dist < best_dist:
                        best_dist, best_adj, best_ys, best_ye = dist, nid, ns, ne
                current_lap_id, y_start, y_end = best_adj, best_ys, best_ye
            else:
                best_lap, best_dist = None, float('inf')
                best_ys, best_ye = None, None
                for nid in unvisited:
                    n_lap = lap_dict[nid]
                    d1 = (n_lap['x'] - last_x) ** 2 + (n_lap['y1'] - last_y) ** 2
                    d2 = (n_lap['x'] - last_x) ** 2 + (n_lap['y2'] - last_y) ** 2
                    if d1 < d2:
                        dist, ns, ne = d1, n_lap['y1'], n_lap['y2']
                    else:
                        dist, ns, ne = d2, n_lap['y2'], n_lap['y1']
                    if dist < best_dist:
                        best_dist, best_lap, best_ys, best_ye = dist, nid, ns, ne
                current_lap_id, y_start, y_end = best_lap, best_ys, best_ye

        return routed

    def generate_path(self):
        if self.saved_map_msg is None or self.path_generated:
            return
        if not self._update_robot_pose():
            self.get_logger().info("Waiting for TF coordinates to connect to the map...")
            return

        rx, ry = self.robot_x, self.robot_y
        self.start_x, self.start_y = rx, ry
        self.get_logger().info(f"TF Linked! Robot is at ({rx:.2f}, {ry:.2f})")
        self.path_generated = True
        self.path_timer.cancel()

        msg = self.saved_map_msg
        self.get_logger().info("Map received! Generating COARSE BCD sweep...")

        res = msg.info.resolution
        w = msg.info.width
        h = msg.info.height
        data = np.array(msg.data).reshape((h, w))
        self.map_data = data
        self.map_info = msg.info

        # Phase 1: coarse sweep
        self.all_waypoints = self._generate_bcd_waypoints(self.coarse_step)
        self.get_logger().info(
            f"Coarse sweep: {len(self.all_waypoints)} waypoints at {self.coarse_step}m step. "
            f"CUSUM will detect hotspots for Phase 2 refinement."
        )

        nav_thread = threading.Thread(target=self._run_navigation, daemon=True)
        nav_thread.start()

    # ────────────────────────────────────────────────────────────
    #  Navigation execution
    # ────────────────────────────────────────────────────────────
    def _run_navigation(self):
        self.nav_client.wait_for_server()

        # ── Phase 1: Coarse sweep ──
        self._current_phase = 'coarse'
        self.get_logger().info("═══ PHASE 1: Coarse coverage sweep ═══")
        coarse_skipped = self._execute_waypoints(self.all_waypoints, phase_label="Coarse")

        # Finalize any in-progress spikes across all topics
        for topic in self.topics:
            if self._cusum_spiking[topic]:
                self._finalize_hotspot(topic)
                self._cusum_spiking[topic] = False

        # Merge hotspots from all sensor topics into unified fine-sweep regions
        self._merge_all_hotspots()

        # ── Phase 2: Refine hotspots ──
        if self._hotspots:
            self._current_phase = 'fine'
            self.get_logger().info(
                f"═══ PHASE 2: Refining {len(self._hotspots)} hotspot(s) "
                f"at {self.fine_step}m resolution ═══"
            )
            for i, (hx, hy, hp) in enumerate(self._hotspots):
                bbox = (
                    hx - self.hotspot_pad,
                    hy - self.hotspot_pad,
                    hx + self.hotspot_pad,
                    hy + self.hotspot_pad,
                )
                fine_wps = self._generate_bcd_waypoints(self.fine_step, bbox=bbox)
                self.refinement_waypoints.extend(fine_wps)
                self.get_logger().info(
                    f"Hotspot #{i + 1} ({hx:.2f}, {hy:.2f}, peak={hp:.1f} ppm): "
                    f"{len(fine_wps)} fine waypoints in "
                    f"[{bbox[0]:.1f},{bbox[1]:.1f}]→[{bbox[2]:.1f},{bbox[3]:.1f}]"
                )
                self._execute_waypoints(fine_wps, phase_label=f"Refine#{i + 1}")
        else:
            self.get_logger().info("No hotspots detected — skipping Phase 2 refinement.")

        total_coarse = len(self.all_waypoints)
        total_fine = len(self.refinement_waypoints)

        # Calculate coverage percentage
        coverage_pct = self._compute_coverage()

        self.get_logger().info(
            f"Coverage complete! Coarse: {total_coarse} wp, Fine: {total_fine} wp, "
            f"Hotspots: {len(self._hotspots)}, Map coverage: {coverage_pct:.1f}%. "
            f"Saving data..."
        )
        self._save_data()
        self._save_readme()
        self._save_occupancy_grid()
        if self.generate_plots:
            self.save_plots()
        else:
            self.get_logger().info("Plot generation skipped (generate_plots=False).")
        rclpy.shutdown()

    def _execute_waypoints(self, waypoints, phase_label=""):
        """Navigate through a list of waypoints. Returns number skipped."""
        total = len(waypoints)
        skipped = 0
        max_consecutive_failures = 2
        consecutive_failures = 0
        last_good_x, last_good_y = self.robot_x, self.robot_y

        for idx, (wx, wy) in enumerate(waypoints):
            if not rclpy.ok() or not self._running:
                break

            if not self._is_waypoint_reachable(wx, wy):
                skipped += 1
                continue

            self.get_logger().info(f"[{phase_label}] Waypoint {idx + 1}/{total}: ({wx:.2f}, {wy:.2f})")
            success = self._navigate_to(wx, wy)

            if success:
                consecutive_failures = 0
                last_good_x, last_good_y = wx, wy
            else:
                skipped += 1
                consecutive_failures += 1
                self.get_logger().warning(
                    f"[{phase_label}] Waypoint {idx + 1} failed. "
                    f"({consecutive_failures} consecutive failures)"
                )
                self._clear_costmaps()
                self._force_backup()

                if consecutive_failures >= max_consecutive_failures:
                    self.get_logger().warning(
                        f"{max_consecutive_failures} consecutive failures. Retreating to last good pose..."
                    )
                    self._clear_costmaps()
                    retreat_ok = self._navigate_to(last_good_x, last_good_y)
                    self._clear_costmaps()
                    if not retreat_ok:
                        # Retreat also failed — aggressive recovery
                        self.get_logger().warning("Retreat failed too. Aggressive backup...")
                        self._force_backup()
                    consecutive_failures = 0

        self.get_logger().info(f"[{phase_label}] Done: {total - skipped}/{total} waypoints reached.")
        return skipped

    # ────────────────────────────────────────────────────────────
    #  Laser scan callback & helpers
    # ────────────────────────────────────────────────────────────
    def _laser_cb(self, msg: LaserScan):
        self._latest_scan = msg

    def _find_best_escape_angle(self):
        """Analyze laser scan to find the most open direction.
        Returns (angle_rad, distance) relative to robot front, or None."""
        scan = self._latest_scan
        if scan is None:
            return None

        ranges = np.array(scan.ranges)
        # Replace inf/nan with max range
        valid = np.isfinite(ranges)
        ranges[~valid] = scan.range_max

        n = len(ranges)
        if n == 0:
            return None

        # Build angle array
        angles = np.array([scan.angle_min + i * scan.angle_increment for i in range(n)])

        # Use a sliding window to find the widest open sector
        # Window = ~60 degrees worth of rays
        window = max(1, int(1.05 / abs(scan.angle_increment)))  # ~60 deg
        half_w = window // 2

        best_avg = 0.0
        best_idx = n // 2  # default: straight ahead

        for i in range(half_w, n - half_w):
            sector = ranges[max(0, i - half_w):min(n, i + half_w + 1)]
            avg = np.mean(sector)
            # Prefer directions where the minimum is also high (no close walls)
            min_in_sector = np.min(sector)
            score = avg * 0.6 + min_in_sector * 0.4
            if score > best_avg:
                best_avg = score
                best_idx = i

        return (angles[best_idx], best_avg)

    def _is_front_blocked(self, threshold=0.35):
        """Check if the robot's front is blocked using laser scan."""
        scan = self._latest_scan
        if scan is None:
            return False

        ranges = np.array(scan.ranges)
        n = len(ranges)
        if n == 0:
            return False

        # Check rays within ±30 degrees of front
        front_arc = 0.52  # ~30 degrees
        center_start = max(0, int((-front_arc - scan.angle_min) / scan.angle_increment))
        center_end = min(n, int((front_arc - scan.angle_min) / scan.angle_increment))

        if center_start >= center_end:
            return False

        front_ranges = ranges[center_start:center_end]
        valid = front_ranges[np.isfinite(front_ranges)]
        if len(valid) == 0:
            return False

        return float(np.min(valid)) < threshold

    # ────────────────────────────────────────────────────────────
    #  Recovery
    # ────────────────────────────────────────────────────────────
    def _force_backup(self, duration=2.5, speed=-0.35):
        """Laser-guided escape: find the most open direction and drive toward it."""
        self._clear_costmaps()
        twist = Twist()
        origin_x, origin_y = self.robot_x, self.robot_y

        # ── Strategy 1: Laser-guided escape ──
        escape = self._find_best_escape_angle()
        if escape is not None:
            best_angle, best_dist = escape
            self.get_logger().info(
                f"[Recovery] Laser escape: best open direction at {math.degrees(best_angle):.0f}° "
                f"(clearance {best_dist:.2f}m)"
            )

            # First reverse away from any close obstacle
            if self._is_front_blocked():
                twist.linear.x = speed
                twist.angular.z = 0.0
                t0 = time.time()
                while time.time() - t0 < 1.5 and rclpy.ok() and self._running:
                    self.cmd_vel_pub.publish(twist)
                    time.sleep(0.05)
                self.cmd_vel_pub.publish(Twist())
                time.sleep(0.1)

            # Rotate toward the open direction
            spin_speed = 1.5 if best_angle >= 0 else -1.5
            spin_time = abs(best_angle) / 1.5
            twist.linear.x = 0.0
            twist.angular.z = spin_speed
            t0 = time.time()
            while time.time() - t0 < spin_time and rclpy.ok() and self._running:
                self.cmd_vel_pub.publish(twist)
                time.sleep(0.05)
            self.cmd_vel_pub.publish(Twist())
            time.sleep(0.1)

            # Drive forward into open space
            drive_time = min(3.0, max(1.0, best_dist * 2.0))
            twist.angular.z = 0.0
            twist.linear.x = 0.35
            t0 = time.time()
            start_x, start_y = self.robot_x, self.robot_y
            while time.time() - t0 < drive_time and rclpy.ok() and self._running:
                # Stop if we're about to hit something
                if self._is_front_blocked(0.3):
                    break
                self.cmd_vel_pub.publish(twist)
                time.sleep(0.05)
            self.cmd_vel_pub.publish(Twist())
            time.sleep(0.2)

            self._update_robot_pose()
            moved = math.hypot(self.robot_x - origin_x, self.robot_y - origin_y)
            if moved > 0.2:
                self.get_logger().info(f"[Recovery] Laser escape succeeded, moved {moved:.2f}m")
                return

        # ── Strategy 2: Blind multi-angle escape (fallback) ──
        self.get_logger().info("[Recovery] Laser escape insufficient, trying blind maneuvers...")
        escape_maneuvers = [
            (1.0, 1.57),    # 90 deg right
            (-1.0, 1.57),   # 90 deg left
            (1.0, 3.14),    # 180 deg
            (-1.0, 3.14),   # 180 deg other way
        ]

        for spin_dir, spin_duration in escape_maneuvers:
            if not rclpy.ok() or not self._running:
                return

            # Spin
            twist.linear.x = 0.0
            twist.angular.z = spin_dir * 1.5
            t0 = time.time()
            while time.time() - t0 < spin_duration and rclpy.ok() and self._running:
                self.cmd_vel_pub.publish(twist)
                time.sleep(0.05)
            self.cmd_vel_pub.publish(Twist())
            time.sleep(0.1)

            # Check if this direction is open, then drive
            if not self._is_front_blocked(0.4):
                twist.angular.z = 0.0
                twist.linear.x = 0.35
                t0 = time.time()
                start_x, start_y = self.robot_x, self.robot_y
                while time.time() - t0 < 2.5 and rclpy.ok() and self._running:
                    if self._is_front_blocked(0.3):
                        break
                    self.cmd_vel_pub.publish(twist)
                    time.sleep(0.05)
                self.cmd_vel_pub.publish(Twist())
                time.sleep(0.2)

                self._update_robot_pose()
                moved = math.hypot(self.robot_x - origin_x, self.robot_y - origin_y)
                if moved > 0.2:
                    self.get_logger().info(f"[Recovery] Blind escape succeeded, moved {moved:.2f}m")
                    return
            else:
                # Front blocked after spin — try reverse instead
                twist.angular.z = 0.0
                twist.linear.x = speed
                t0 = time.time()
                while time.time() - t0 < 2.0 and rclpy.ok() and self._running:
                    self.cmd_vel_pub.publish(twist)
                    time.sleep(0.05)
                self.cmd_vel_pub.publish(Twist())
                time.sleep(0.2)

                self._update_robot_pose()
                moved = math.hypot(self.robot_x - origin_x, self.robot_y - origin_y)
                if moved > 0.2:
                    return

        self.get_logger().warning("[Recovery] All escape attempts failed.")

    def _is_waypoint_reachable(self, x, y):
        if self.map_data is None or self.map_info is None:
            return True
        res = self.map_info.resolution
        ox = self.map_info.origin.position.x
        oy = self.map_info.origin.position.y
        h, w = self.map_data.shape
        # Check a larger area around the waypoint (robot radius + margin)
        check_radius = 0.35
        steps = [-check_radius, -check_radius / 2, 0, check_radius / 2, check_radius]
        for dx in steps:
            for dy in steps:
                c = int((x + dx - ox) / res)
                r = int((y + dy - oy) / res)
                if not (0 <= r < h and 0 <= c < w):
                    return False
                val = self.map_data[r, c]
                if val < 0 or val >= 50:
                    return False
        return True

    def _clear_costmaps(self):
        try:
            if self.clear_local.wait_for_service(timeout_sec=2.0):
                self.clear_local.call_async(Empty.Request())
            if self.clear_global.wait_for_service(timeout_sec=2.0):
                self.clear_global.call_async(Empty.Request())
            time.sleep(1.0)
        except Exception:
            pass

    def _navigate_to(self, x: float, y: float) -> bool:
        self._update_robot_pose()
        dist = math.hypot(self.robot_x - x, self.robot_y - y)
        if dist <= self.wp_tolerance:
            return True

        adaptive_timeout = max(20.0, min(60.0, dist * 12.0))

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = PoseStamped()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp.sec = 0
        goal_msg.pose.header.stamp.nanosec = 0
        goal_msg.pose.pose.position.x = float(x)
        goal_msg.pose.pose.position.y = float(y)

        yaw = math.atan2(y - self.robot_y, x - self.robot_x)
        goal_msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal_msg.pose.pose.orientation.w = math.cos(yaw / 2.0)

        future = self.nav_client.send_goal_async(goal_msg)
        accept_event = threading.Event()
        future.add_done_callback(lambda f: accept_event.set())
        if not accept_event.wait(timeout=10.0):
            return False

        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            return False

        result_future = goal_handle.get_result_async()
        result_event = threading.Event()
        result_future.add_done_callback(lambda f: result_event.set())

        t0 = time.time()
        last_progress_x, last_progress_y = self.robot_x, self.robot_y
        last_progress_time = t0
        stuck_timeout = 8.0  # seconds without meaningful movement = stuck
        while rclpy.ok() and self._running:
            if result_event.wait(timeout=0.2):
                break
            if time.time() - t0 > adaptive_timeout:
                try:
                    goal_handle.cancel_goal_async()
                except Exception:
                    pass
                result_event.wait(timeout=2.0)
                break

            # Early stuck detection: if robot hasn't moved 0.1m in stuck_timeout seconds, abort
            self._update_robot_pose()
            moved_since = math.hypot(self.robot_x - last_progress_x, self.robot_y - last_progress_y)
            if moved_since > 0.1:
                last_progress_x, last_progress_y = self.robot_x, self.robot_y
                last_progress_time = time.time()
            elif time.time() - last_progress_time > stuck_timeout:
                self.get_logger().warning(
                    f"[Nav] Stuck detected — no progress for {stuck_timeout:.0f}s. Cancelling goal."
                )
                try:
                    goal_handle.cancel_goal_async()
                except Exception:
                    pass
                result_event.wait(timeout=2.0)
                break

            # Reactive slowdown when any sensor's CUSUM is spiking
            if self._any_cusum_spiking:
                twist = Twist()
                twist.linear.x = self.gas_slowdown_speed
                yaw_to_goal = math.atan2(y - self.robot_y, x - self.robot_x)
                twist.angular.z = yaw_to_goal * 0.5
                self.cmd_vel_pub.publish(twist)

        self._update_robot_pose()
        dist = math.hypot(self.robot_x - x, self.robot_y - y)
        if dist <= self.wp_tolerance:
            return True

        if result_future.done():
            result = result_future.result()
            status = result.status if result is not None else GoalStatus.STATUS_UNKNOWN
            return status == GoalStatus.STATUS_SUCCEEDED

        return False

    # ────────────────────────────────────────────────────────────
    #  Coverage Calculation
    # ────────────────────────────────────────────────────────────
    def _compute_coverage(self):
        """Compute % of free map cells visited by the robot."""
        if self.map_data is None or self.map_info is None:
            return 0.0

        res = self.map_info.resolution
        ox = self.map_info.origin.position.x
        oy = self.map_info.origin.position.y
        h, w = self.map_data.shape

        # Free cells: occupancy value 0 (definitely free)
        free_mask = (self.map_data >= 0) & (self.map_data < 40)
        total_free = int(np.sum(free_mask))
        if total_free == 0:
            return 0.0

        # Mark cells visited by the robot (within robot radius of any trajectory point)
        visit_radius = 0.3  # meters
        visit_cells = int(visit_radius / res)
        visited = np.zeros_like(free_mask)

        for topic_data in self.dataset.values():
            for rx, ry in zip(topic_data['x'], topic_data['y']):
                c = int((rx - ox) / res)
                r = int((ry - oy) / res)
                rmin = max(0, r - visit_cells)
                rmax = min(h, r + visit_cells + 1)
                cmin = max(0, c - visit_cells)
                cmax = min(w, c + visit_cells + 1)
                visited[rmin:rmax, cmin:cmax] = True

        visited_free = int(np.sum(free_mask & visited))
        pct = 100.0 * visited_free / total_free

        self._coverage_pct = pct
        self._coverage_visited = visited_free
        self._coverage_total = total_free
        return pct

    # ────────────────────────────────────────────────────────────
    #  README & Occupancy Grid
    # ────────────────────────────────────────────────────────────
    def _save_readme(self):
        """Generate a README.md summarizing the run configuration."""
        lines = ['# Coverage Run Configuration\n']
        lines.append(f'**Run ID:** {self._run_id}\n')
        lines.append(f'**Robot Start Position:** ({self.start_x:.2f}, {self.start_y:.2f})\n')
        lines.append(f'**Namespace:** {self.ns}\n')
        lines.append(f'**Coarse Step:** {self.coarse_step}m | **Fine Step:** {self.fine_step}m\n')

        # Sensor configuration
        lines.append('\n## Sensors\n')
        lines.append('| Sensor | Topic | Type |')
        lines.append('|--------|-------|------|')
        for t in self.topics:
            stype = self._sensor_type_map.get(t, 'unknown')
            lines.append(f'| {stype} | `{t}` | {stype} |')

        # Try to read scenario config files for gas source info
        if self.scenario_path and os.path.isdir(self.scenario_path):
            # Read environment config
            config_path = os.path.join(self.scenario_path, 'config.yaml')
            if os.path.isfile(config_path):
                try:
                    with open(config_path) as f:
                        env_cfg = yaml.safe_load(f)
                    lines.append('\n## Environment Configuration\n')
                    lines.append(f'- **Config file:** `{config_path}`')
                    if 'cell_size' in env_cfg:
                        lines.append(f'- **Cell size:** {env_cfg["cell_size"]}m')
                    if 'empty_point' in env_cfg:
                        lines.append(f'- **Empty reference point:** {env_cfg["empty_point"]}')
                    if 'uniformWind' in env_cfg:
                        lines.append(f'- **Uniform wind:** {env_cfg["uniformWind"]}')
                    if 'unprocessed_wind_files' in env_cfg:
                        lines.append(f'- **Wind files:** {env_cfg["unprocessed_wind_files"]}')
                    if 'models' in env_cfg:
                        lines.append(f'- **CAD models:** {env_cfg["models"]}')
                except Exception as e:
                    lines.append(f'\n*Could not read config.yaml: {e}*')

            # Read simulation configs for gas source positions
            sim_dir = os.path.join(self.scenario_path, 'simulations')
            if os.path.isdir(sim_dir):
                lines.append('\n## Gas Sources\n')
                lines.append('| Simulation | Position | Gas Type | Growth Gamma | Filaments/sec |')
                lines.append('|------------|----------|----------|--------------|---------------|')
                for sim_name in sorted(os.listdir(sim_dir)):
                    sim_yaml = os.path.join(sim_dir, sim_name, 'sim.yaml')
                    if os.path.isfile(sim_yaml):
                        try:
                            with open(sim_yaml) as f:
                                sim_cfg = yaml.safe_load(f)
                            src = sim_cfg.get('source', {})
                            pos = src.get('position', 'N/A')
                            gas_type = sim_cfg.get('gasType', 'N/A')  # note: currently 0 for all
                            gamma = sim_cfg.get('filamentGrowthGamma', 'N/A')
                            fils = sim_cfg.get('numFilaments_sec', 'N/A')
                            lines.append(f'| {sim_name} | {pos} | {gas_type} | {gamma} | {fils} |')
                        except Exception:
                            pass

            # Read scene config
            scenes_dir = os.path.join(self.scenario_path, 'scenes')
            if os.path.isdir(scenes_dir):
                for scene_file in sorted(os.listdir(scenes_dir)):
                    if scene_file.endswith('.yaml'):
                        scene_path = os.path.join(scenes_dir, scene_file)
                        try:
                            with open(scene_path) as f:
                                scene_cfg = yaml.safe_load(f)
                            lines.append(f'\n## Scene: {scene_file}\n')
                            lines.append(f'- **Playback initial iteration:** {scene_cfg.get("playback_initial_iteration", "N/A")}')
                            sims = scene_cfg.get('simulations', [])
                            if sims:
                                lines.append(f'- **Active simulations:** {len(sims)}')
                                for s in sims:
                                    lines.append(f'  - {s.get("sim", "?")} (color: {s.get("gas_color", "N/A")})')
                        except Exception:
                            pass

            # Read BasicSimScene for robot config
            bss_path = os.path.join(self.scenario_path, 'BasicSimScene.yaml')
            if os.path.isfile(bss_path):
                try:
                    with open(bss_path) as f:
                        bss = yaml.safe_load(f)
                    lines.append('\n## Robot Configuration (BasicSimScene)\n')
                    for robot in bss.get('robots', []):
                        lines.append(f'- **Name:** {robot.get("name", "N/A")}')
                        lines.append(f'- **Radius:** {robot.get("radius", "N/A")}')
                        lines.append(f'- **Position:** {robot.get("position", "N/A")}')
                        lines.append(f'- **Angle:** {robot.get("angle", "N/A")} rad')
                except Exception:
                    pass
        else:
            lines.append('\n*No scenario_path provided — set the `scenario_path` parameter '
                         'to auto-populate gas source and environment details.*\n')

        # Coverage results
        if hasattr(self, '_coverage_pct'):
            lines.append(f'\n## Coverage Results\n')
            lines.append(f'- **Map coverage:** {self._coverage_pct:.1f}%')
            lines.append(f'- **Visited cells:** {self._coverage_visited}')
            lines.append(f'- **Total free cells:** {self._coverage_total}')

        # Hotspots detected
        if self._hotspots:
            lines.append(f'\n## Detected Hotspots\n')
            lines.append('| # | X | Y | Peak PPM |')
            lines.append('|---|---|---|----------|')
            for i, (hx, hy, hp) in enumerate(self._hotspots):
                lines.append(f'| {i+1} | {hx:.2f} | {hy:.2f} | {hp:.1f} |')

        lines.append(f'\n## Output Files\n')
        lines.append(f'- `readings.csv` — Full sensor readings (timestamp, x, y, gas_type, ppm, phase)')
        lines.append(f'- `occupancy_grid.pgm` — Occupancy grid of the environment')
        lines.append(f'- `run_data.npz` — NumPy archive with all data arrays')
        lines.append(f'- `README.md` — This file')

        readme_path = os.path.join(self._run_dir, 'README.md')
        with open(readme_path, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        print(f"Saved README: {readme_path}")

    def _save_occupancy_grid(self):
        """Save the occupancy grid as a PGM image file."""
        if self.map_data is None or self.map_info is None:
            self.get_logger().warning("No occupancy grid available to save.")
            return

        info = self.map_info
        h, w = self.map_data.shape

        # Save as PGM (standard ROS map format)
        pgm_path = os.path.join(self._run_dir, 'occupancy_grid.pgm')
        raw = np.array(self.map_data, dtype=np.int16).reshape((h, w))
        # Convert: free(0)=254, occupied(100)=0, unknown(-1)=205
        img = np.full((h, w), 205, dtype=np.uint8)
        img[raw == 0] = 254
        img[(raw > 0) & (raw < 50)] = 254 - (raw[(raw > 0) & (raw < 50)] * 2.54).astype(np.uint8)
        img[raw >= 50] = 0
        # Flip vertically (PGM origin is top-left, ROS is bottom-left)
        img = np.flipud(img)

        with open(pgm_path, 'wb') as f:
            f.write(f'P5\n{w} {h}\n255\n'.encode())
            f.write(img.tobytes())

        # Save companion YAML (so the grid can be loaded by map_server)
        yaml_path = os.path.join(self._run_dir, 'occupancy_grid.yaml')
        map_yaml = {
            'image': 'occupancy_grid.pgm',
            'resolution': float(info.resolution),
            'origin': [float(info.origin.position.x),
                       float(info.origin.position.y), 0.0],
            'negate': 0,
            'occupied_thresh': 0.65,
            'free_thresh': 0.196,
        }
        with open(yaml_path, 'w') as f:
            yaml.dump(map_yaml, f, default_flow_style=False)

        print(f"Saved occupancy grid: {pgm_path} ({w}x{h})")
        print(f"Saved occupancy YAML: {yaml_path}")

    # ────────────────────────────────────────────────────────────
    #  Data Persistence
    # ────────────────────────────────────────────────────────────
    def _save_data(self):
        """Save all collected data as NPZ archive for future analysis."""
        npz_path = os.path.join(self._run_dir, 'run_data.npz')

        save_dict = {}
        for t, d in self.dataset.items():
            clean = t.replace('/', '_').strip('_')
            save_dict[f'{clean}_x'] = np.array(d['x'])
            save_dict[f'{clean}_y'] = np.array(d['y'])
            save_dict[f'{clean}_ppm'] = np.array(d['ppm'])
            save_dict[f'{clean}_timestamp'] = np.array(d['timestamp'])

            # Run source localization and save posterior
            xs, ys_arr, ppm_arr = np.array(d['x']), np.array(d['y']), np.array(d['ppm'])
            if len(xs) > 10:
                posterior, gx, gy, bx, by, cr = self._estimate_source(xs, ys_arr, ppm_arr)
                save_dict[f'{clean}_posterior'] = posterior
                save_dict[f'{clean}_posterior_gx'] = gx
                save_dict[f'{clean}_posterior_gy'] = gy
                save_dict[f'{clean}_source_xy'] = np.array([bx, by])
                save_dict[f'{clean}_source_confidence'] = np.array(cr)

        # Save hotspots
        save_dict['hotspots'] = np.array(self._hotspots) if self._hotspots else np.array([])
        save_dict['coarse_waypoints'] = np.array(self.all_waypoints)
        save_dict['fine_waypoints'] = np.array(self.refinement_waypoints) if self.refinement_waypoints else np.array([])

        # Save concentration grid
        if self._conc_grid is not None:
            save_dict['conc_grid'] = self._conc_grid
            save_dict['conc_count'] = self._conc_count
            save_dict['conc_origin'] = np.array([self._conc_ox, self._conc_oy])
            save_dict['conc_resolution'] = np.array(self._conc_grid_res)

        # Coverage statistics
        if hasattr(self, '_coverage_pct'):
            save_dict['coverage_pct'] = np.array(self._coverage_pct)
            save_dict['coverage_visited_cells'] = np.array(self._coverage_visited)
            save_dict['coverage_total_free_cells'] = np.array(self._coverage_total)

        np.savez_compressed(npz_path, **save_dict)
        print(f"Saved NPZ archive: {npz_path}")
        print(f"  Contains: {list(save_dict.keys())}")
        print(f"  Reload with: data = np.load('{npz_path}')")

        # Close CSV
        self._csv_file.flush()
        self._csv_file.close()
        print(f"CSV log finalized: {self._csv_path} ({self._sample_count} samples)")

    # ────────────────────────────────────────────────────────────
    #  Bayesian Source Localization (isotropic diffusion model)
    # ────────────────────────────────────────────────────────────
    def _estimate_source(self, xs, ys, ppm, grid_res=0.08):
        """
        Bayesian source localization using weighted kernel density.

        Strategy:
        1. Filter to positive readings only (zeros carry no source info).
        2. For each candidate source cell, compute a score based on how well
           the ppm-weighted distance pattern matches a monotonic decay from
           that point. Uses a Gaussian kernel exp(-r²/2L²) with scale L
           estimated from the data spread.
        3. The score is tempered to avoid posterior collapse.

        Returns:
            posterior, grid_x, grid_y, best_x, best_y, confidence_radius
        """
        # Filter to positive readings — zeros tell us nothing about source location
        pos_mask = ppm > 0.1
        if pos_mask.sum() < 5:
            # Not enough data — return uniform posterior
            margin = 0.5
            cx = np.arange(xs.min() - margin, xs.max() + margin, grid_res)
            cy = np.arange(ys.min() - margin, ys.max() + margin, grid_res)
            gx, gy = np.meshgrid(cx, cy)
            posterior = np.ones(gx.shape) / gx.size
            return posterior, gx, gy, float(np.mean(xs)), float(np.mean(ys)), 999.0

        xp, yp, pp = xs[pos_mask], ys[pos_mask], ppm[pos_mask]

        margin = 1.0
        cx = np.arange(xs.min() - margin, xs.max() + margin, grid_res)
        cy = np.arange(ys.min() - margin, ys.max() + margin, grid_res)
        gx, gy = np.meshgrid(cx, cy)

        # Characteristic length scale — based on spread of high-ppm readings
        high_mask = pp > np.percentile(pp, 70)
        if high_mask.sum() > 3:
            L = max(np.std(xp[high_mask]) + np.std(yp[high_mask]), 0.5)
        else:
            L = 2.0

        # Weight each measurement by ppm² — high readings are much more informative
        weights = pp ** 2
        weights /= weights.sum()

        log_posterior = np.zeros(gx.shape)

        for i in range(gx.shape[0]):
            for j in range(gx.shape[1]):
                sx, sy = gx[i, j], gy[i, j]
                r2 = (xp - sx) ** 2 + (yp - sy) ** 2

                # Score 1: Weighted kernel density — high ppm near candidate = good
                kernel = np.exp(-r2 / (2 * L ** 2))
                score_density = np.sum(weights * kernel)

                # Score 2: Correlation between ppm and proximity
                # If this is the source, closer readings should have higher ppm
                proximity = np.exp(-r2 / (2 * (L * 1.5) ** 2))
                score_corr = np.corrcoef(pp, proximity)[0, 1] if len(pp) > 2 else 0.0
                score_corr = max(score_corr, 0.0)  # negative correlation = not source

                log_posterior[i, j] = np.log(score_density + 1e-20) + 2.0 * score_corr

        # Temper the posterior to avoid collapse (effective temperature)
        log_posterior -= np.max(log_posterior)
        temperature = 0.5  # higher = smoother posterior
        posterior = np.exp(log_posterior / temperature)
        posterior /= posterior.sum()

        # MAP estimate
        peak_idx = np.argmax(posterior)
        peak_r, peak_c = np.unravel_index(peak_idx, posterior.shape)
        best_x = float(gx[peak_r, peak_c])
        best_y = float(gy[peak_r, peak_c])

        # Confidence radius (sqrt of weighted variance around MAP)
        dx = gx - best_x
        dy = gy - best_y
        r2 = dx ** 2 + dy ** 2
        confidence_radius = float(np.sqrt(np.sum(posterior * r2)))

        return posterior, gx, gy, best_x, best_y, confidence_radius

    # ────────────────────────────────────────────────────────────
    #  Plotting — viridis theme, research-ready
    # ────────────────────────────────────────────────────────────
    def save_plots(self):
        from scipy.interpolate import griddata
        from scipy.spatial import cKDTree
        from scipy.ndimage import gaussian_filter
        from matplotlib.colors import LogNorm, PowerNorm
        from matplotlib_scalebar.scalebar import ScaleBar

        CMAP = 'viridis'

        plt.rcParams.update({
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'savefig.facecolor': 'white',
            'font.family': 'serif',
            'font.size': 11,
            'axes.labelsize': 12,
            'axes.titlesize': 13,
        })

        out = self._run_dir
        os.makedirs(out, exist_ok=True)
        print(f"Generating charts into {out}...")

        # Occupancy map — solid black walls, white free space
        occ_rgba = None
        occ_extent = None
        if self.map_data is not None and self.map_info is not None:
            info = self.map_info
            w, h = info.width, info.height
            raw = np.array(self.map_data, dtype=float).reshape((h, w))
            # RGBA: walls=black opaque, free=transparent, unknown=transparent
            occ_rgba = np.zeros((h, w, 4), dtype=float)
            wall_mask = raw >= 50
            occ_rgba[wall_mask] = [0, 0, 0, 1.0]  # solid black walls
            ox = info.origin.position.x
            oy = info.origin.position.y
            occ_extent = [ox, ox + w * info.resolution,
                          oy, oy + h * info.resolution]

        for t, d in self.dataset.items():
            if len(d['x']) == 0:
                continue
            xs, ys, ppm = np.array(d['x']), np.array(d['y']), np.array(d['ppm'])
            clean = t.replace('/', '_').strip('_')

            ppm_max = max(np.max(ppm), 1.0)
            ppm_floor = max(np.min(ppm[ppm > 0]), 0.1) if np.any(ppm > 0) else 0.1
            ppm_mean = float(np.mean(ppm))
            ppm_std = float(np.std(ppm))

            # ── Interpolation grid ──
            margin = 0.3
            grid_res = 0.04
            xg = np.arange(xs.min() - margin, xs.max() + margin, grid_res)
            yg = np.arange(ys.min() - margin, ys.max() + margin, grid_res)
            xi, yi = np.meshgrid(xg, yg)

            zi = griddata((xs, ys), ppm, (xi, yi), method='cubic')
            zi_near = griddata((xs, ys), ppm, (xi, yi), method='nearest')
            zi[np.isnan(zi)] = zi_near[np.isnan(zi)]
            zi = np.clip(zi, 0, None)

            # Coverage mask
            tree = cKDTree(np.column_stack([xs, ys]))
            dists, _ = tree.query(np.column_stack([xi.ravel(), yi.ravel()]))
            cmask = dists.reshape(xi.shape) > 0.6

            # Gaussian smoothing
            zi_smooth = gaussian_filter(zi, sigma=3.0)

            # Alpha channel: smooth fade at coverage boundary (no gray fringe)
            cmap_obj = plt.get_cmap(CMAP)
            alpha_raw = np.ones(cmask.shape, dtype=float)
            alpha_raw[cmask] = 0.0
            alpha_ch = gaussian_filter(alpha_raw, sigma=1.5)
            alpha_ch = np.clip(alpha_ch, 0, 1)
            alpha_ch = np.where(alpha_ch > 0.5, 1.0, alpha_ch * 2)

            # Pre-compute RGBA heatmap images
            norm_log = LogNorm(vmin=ppm_floor, vmax=ppm_max)
            norm_pow = PowerNorm(gamma=0.3, vmin=0, vmax=ppm_max)

            zi_for_log = zi_smooth.copy()
            zi_for_log[zi_for_log < ppm_floor] = ppm_floor
            rgba_log = cmap_obj(norm_log(zi_for_log))
            rgba_log[..., 3] = alpha_ch

            rgba_pow = cmap_obj(norm_pow(zi_smooth))
            rgba_pow[..., 3] = alpha_ch

            # Contour data (fill outside with 0 before smoothing)
            zi_cont = zi.copy()
            zi_cont[cmask] = 0
            zi_cont_smooth = gaussian_filter(zi_cont, sigma=2.0)

            # Contour levels
            n_lvl = 18
            if ppm_max > ppm_floor * 2:
                levels = np.concatenate([[0], np.geomspace(ppm_floor, ppm_max * 1.05, n_lvl)])
            else:
                levels = np.linspace(0, ppm_max * 1.05, n_lvl + 1)

            img_extent = [xg[0], xg[-1], yg[0], yg[-1]]

            peak_idx = np.nanargmax(np.where(cmask, -1, zi))
            pr, pc = np.unravel_index(peak_idx, zi.shape)
            peak_x, peak_y = float(xi[pr, pc]), float(yi[pr, pc])

            stats_text = (f"Stats (ppm):\n"
                          f"Min: {np.min(ppm):.2f}\n"
                          f"Max: {ppm_max:.2f}\n"
                          f"Mean: {ppm_mean:.2f}\n"
                          f"Std: {ppm_std:.2f}")

            def _walls(ax):
                """Overlay solid black walls from occupancy grid."""
                if occ_rgba is not None:
                    ax.imshow(occ_rgba, extent=occ_extent, origin='lower',
                              interpolation='nearest', zorder=5)

            def _scalebar(ax):
                """Add a 1m scale bar."""
                ax.add_artist(ScaleBar(1.0, 'm', location='lower left',
                                       length_fraction=0.15, box_alpha=0.7,
                                       font_properties={'size': 10}))

            def _stats(ax, text):
                """Add stats box."""
                ax.text(0.02, 0.98, text, transform=ax.transAxes,
                        fontsize=9, verticalalignment='top',
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                                  edgecolor='gray', alpha=0.85),
                        family='monospace', zorder=6)

            # ── 1) Coverage path ──
            fig, ax = plt.subplots(figsize=(10, 8))
            sc = ax.scatter(xs, ys, c=ppm, cmap=CMAP,
                            norm=PowerNorm(gamma=0.4, vmin=0, vmax=ppm_max),
                            alpha=0.8, s=8, zorder=2, edgecolors='none')
            _walls(ax)
            plt.colorbar(sc, ax=ax, label='Concentration (ppm)', shrink=0.82)
            ax.plot(peak_x, peak_y, 'r*', markersize=14, zorder=6,
                    markeredgecolor='black', markeredgewidth=0.5,
                    label=f'Peak: {ppm_max:.1f} ppm')
            _stats(ax, stats_text)
            _scalebar(ax)
            ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
            ax.set_title('Coverage Path — Gas Concentration')
            ax.legend(loc='upper right', fontsize=9)
            ax.set_aspect('equal')
            fig.tight_layout()
            fig.savefig(os.path.join(out, f'{clean}_scatter.png'), dpi=250)
            plt.close(fig)

            # ── 2) Heatmap — log scale with contour overlay ──
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.imshow(rgba_log, extent=img_extent, origin='lower',
                      interpolation='bilinear', zorder=1, aspect='auto')
            ax.contour(xi, yi, zi_cont_smooth, levels=levels, colors='#555555',
                       linewidths=0.35, alpha=0.45, zorder=2)
            _walls(ax)
            sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm_log)
            plt.colorbar(sm, ax=ax, label='Concentration (ppm)', shrink=0.82)
            ax.plot(peak_x, peak_y, 'r*', markersize=14, zorder=6,
                    markeredgecolor='white', markeredgewidth=0.8,
                    label=f'Peak: {ppm_max:.1f} ppm')
            _stats(ax, stats_text); _scalebar(ax)
            ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
            ax.set_title('Gas Concentration Map (log scale)')
            ax.legend(loc='upper right', fontsize=9)
            ax.set_aspect('equal')
            fig.tight_layout()
            fig.savefig(os.path.join(out, f'{clean}_heatmap_log.png'), dpi=250)
            plt.close(fig)

            # ── 3) Heatmap — enhanced contrast with contour overlay ──
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.imshow(rgba_pow, extent=img_extent, origin='lower',
                      interpolation='bilinear', zorder=1, aspect='auto')
            ax.contour(xi, yi, zi_cont_smooth, levels=levels, colors='#555555',
                       linewidths=0.35, alpha=0.45, zorder=2)
            _walls(ax)
            sm2 = plt.cm.ScalarMappable(cmap=CMAP, norm=norm_pow)
            plt.colorbar(sm2, ax=ax, label='Concentration (ppm)', shrink=0.82)
            ax.plot(peak_x, peak_y, 'r*', markersize=14, zorder=6,
                    markeredgecolor='white', markeredgewidth=0.8)
            _stats(ax, stats_text); _scalebar(ax)
            ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
            ax.set_title('Gas Concentration Map (enhanced contrast)')
            ax.set_aspect('equal')
            fig.tight_layout()
            fig.savefig(os.path.join(out, f'{clean}_heatmap.png'), dpi=250)
            plt.close(fig)

            # ── 4) Filled contour ──
            fig, ax = plt.subplots(figsize=(10, 8))
            cs = ax.contourf(xi, yi, zi_cont_smooth, levels=levels, cmap=CMAP, alpha=0.95)
            ax.contour(xi, yi, zi_cont_smooth, levels=levels, colors='#333333',
                       linewidths=0.35, alpha=0.55)
            _walls(ax)
            plt.colorbar(cs, ax=ax, label='Concentration (ppm)', shrink=0.82)
            ax.plot(peak_x, peak_y, 'r*', markersize=14, zorder=6,
                    markeredgecolor='white', markeredgewidth=0.8)
            _stats(ax, stats_text)
            _scalebar(ax)
            ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
            ax.set_title('Gas Concentration Contours')
            ax.set_aspect('equal')
            fig.tight_layout()
            fig.savefig(os.path.join(out, f'{clean}_contour.png'), dpi=250)
            plt.close(fig)

            # ── 5) Distribution comparison: power vs log ──
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
            ax1.imshow(rgba_pow, extent=img_extent, origin='lower',
                       interpolation='bilinear', zorder=1, aspect='auto')
            ax1.contour(xi, yi, zi_cont_smooth, levels=levels, colors='#444444',
                        linewidths=0.3, alpha=0.4, zorder=2)
            _walls(ax1)
            plt.colorbar(plt.cm.ScalarMappable(cmap=CMAP, norm=norm_pow),
                         ax=ax1, label='ppm (power)', shrink=0.82)
            ax1.set_title('Power Scale'); ax1.set_xlabel('X (m)'); ax1.set_ylabel('Y (m)')
            ax1.set_aspect('equal')

            ax2.imshow(rgba_log, extent=img_extent, origin='lower',
                       interpolation='bilinear', zorder=1, aspect='auto')
            ax2.contour(xi, yi, zi_cont_smooth, levels=levels, colors='#444444',
                        linewidths=0.3, alpha=0.4, zorder=2)
            _walls(ax2)
            plt.colorbar(plt.cm.ScalarMappable(cmap=CMAP, norm=norm_log),
                         ax=ax2, label='ppm (log scale)', shrink=0.82)
            ax2.set_title('Log Scale'); ax2.set_xlabel('X (m)'); ax2.set_ylabel('Y (m)')
            ax2.set_aspect('equal')

            fig.suptitle('Gas Distribution Comparison', fontsize=14, fontweight='bold')
            fig.tight_layout()
            fig.savefig(os.path.join(out, f'{clean}_distribution.png'), dpi=250)
            plt.close(fig)

            # ── 6) Gradient field ──
            zi_grad = zi_smooth.copy(); zi_grad[cmask] = 0
            grad_y, grad_x = np.gradient(zi_grad, grid_res, grid_res)
            mag = np.sqrt(grad_x**2 + grad_y**2) + 1e-10
            gxn, gyn = grad_x / mag, grad_y / mag
            skip = max(1, len(xi) // 22)
            arr_mask = ~cmask[::skip, ::skip]

            rgba_grad = rgba_pow.copy(); rgba_grad[..., 3] *= 0.5
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.imshow(rgba_grad, extent=img_extent, origin='lower',
                      interpolation='bilinear', zorder=1, aspect='auto')
            gu = gxn[::skip, ::skip].copy(); gv = gyn[::skip, ::skip].copy()
            gu[~arr_mask] = np.nan; gv[~arr_mask] = np.nan
            ax.quiver(xi[::skip, ::skip], yi[::skip, ::skip], gu, gv,
                      mag[::skip, ::skip], cmap='coolwarm', alpha=0.8,
                      scale=35, width=0.003, zorder=3)
            _walls(ax)
            ax.plot(peak_x, peak_y, 'r*', markersize=14, zorder=6,
                    markeredgecolor='white', markeredgewidth=0.8)
            _scalebar(ax)
            ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
            ax.set_title('Concentration Gradient Field')
            ax.set_aspect('equal')
            fig.tight_layout()
            fig.savefig(os.path.join(out, f'{clean}_gradient.png'), dpi=250)
            plt.close(fig)

            # ── 7) Concentration time-series ──
            fig, ax = plt.subplots(figsize=(14, 5))
            ax.plot(ppm, linewidth=0.7, color='#7b2cbf', alpha=0.9)
            ax.fill_between(range(len(ppm)), ppm, alpha=0.2, color='#9d4edd')
            ax.set_ylabel('Concentration (ppm)')
            ax.set_xlabel('Sample index')
            ax.set_title('Gas Concentration Along Path')
            ax.grid(True, alpha=0.15)
            fig.tight_layout()
            fig.savefig(os.path.join(out, f'{clean}_concentration.png'), dpi=250)
            plt.close(fig)

            # ── 8) CUSUM detection ──
            cusum_trace = []
            cw, ck, ch = self._cusum_params
            cusum_plot = CUSUMDetector(warmup=cw, k=ck, h=ch)
            spike_regions = []
            in_spike = False
            for i, val in enumerate(ppm):
                fired = cusum_plot.update(val)
                cusum_trace.append(cusum_plot.current_S if cusum_plot.is_calibrated() else 0.0)
                if fired and not in_spike:
                    in_spike = True
                    spike_regions.append([i, i])
                elif fired:
                    spike_regions[-1][1] = i
                elif in_spike:
                    in_spike = False
                    cusum_plot.reset_alarm()

            fig, ax = plt.subplots(figsize=(14, 4))
            ax.plot(cusum_trace, linewidth=0.8, color='#2c3e50', label='CUSUM S(n)')
            if cusum_plot.is_calibrated():
                ax.axhline(y=cusum_plot.threshold, color='#c0392b', linestyle='--',
                            alpha=0.7, linewidth=1.2, label=f'Threshold = {cusum_plot.threshold:.2f}')
            for s, e in spike_regions:
                ax.axvspan(s, e, alpha=0.12, color='#e74c3c')
            ax.set_xlabel('Sample index')
            ax.set_ylabel('CUSUM S(n)')
            ax.set_title('CUSUM Change-Point Detection')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.15)
            fig.tight_layout()
            fig.savefig(os.path.join(out, f'{clean}_cusum.png'), dpi=250)
            plt.close(fig)

            # ── 9) Dashboard (2x2) ──
            fig, axes = plt.subplots(2, 2, figsize=(16, 14))

            axes[0, 0].scatter(xs, ys, c=ppm, cmap=CMAP,
                               norm=PowerNorm(gamma=0.4, vmin=0, vmax=ppm_max),
                               s=5, alpha=0.8, zorder=2, edgecolors='none')
            _walls(axes[0, 0])
            axes[0, 0].set_title('Coverage Path'); axes[0, 0].set_aspect('equal')

            axes[0, 1].imshow(rgba_log, extent=img_extent, origin='lower',
                              interpolation='bilinear', zorder=1, aspect='auto')
            axes[0, 1].contour(xi, yi, zi_cont_smooth, levels=levels, colors='#444444',
                               linewidths=0.3, alpha=0.4, zorder=2)
            _walls(axes[0, 1])
            axes[0, 1].set_title('Heatmap (log scale)'); axes[0, 1].set_aspect('equal')

            axes[1, 0].contourf(xi, yi, zi_cont_smooth, levels=levels, cmap=CMAP, alpha=0.95)
            axes[1, 0].contour(xi, yi, zi_cont_smooth, levels=levels, colors='#333333',
                               linewidths=0.25, alpha=0.4)
            _walls(axes[1, 0])
            axes[1, 0].set_title('Contours'); axes[1, 0].set_aspect('equal')

            axes[1, 1].imshow(rgba_grad, extent=img_extent, origin='lower',
                              interpolation='bilinear', aspect='auto')
            axes[1, 1].quiver(xi[::skip, ::skip], yi[::skip, ::skip], gu, gv,
                              mag[::skip, ::skip], cmap='coolwarm', alpha=0.8,
                              scale=35, width=0.003)
            _walls(axes[1, 1])
            axes[1, 1].set_title('Gradient Field'); axes[1, 1].set_aspect('equal')

            fig.suptitle('Gas Mapping Dashboard', fontsize=15, fontweight='bold')
            fig.tight_layout()
            fig.savefig(os.path.join(out, f'{clean}_dashboard.png'), dpi=250)
            plt.close(fig)

            # ── 10) Source Localization ──
            print(f"Running source localization for {t}...")
            posterior, gx, gy, best_x, best_y, conf_r = self._estimate_source(xs, ys, ppm)

            fig, ax = plt.subplots(figsize=(10, 8))
            post_max = max(np.max(posterior), 1e-10)
            post_smooth = gaussian_filter(posterior, sigma=2.0)
            norm_post = PowerNorm(gamma=0.4, vmin=0, vmax=post_max)
            rgba_post = cmap_obj(norm_post(post_smooth))
            src_extent = [gx[0, 0], gx[0, -1], gy[0, 0], gy[-1, 0]]
            ax.imshow(rgba_post, extent=src_extent, origin='lower',
                      interpolation='bilinear', zorder=1, aspect='auto')
            _walls(ax)
            plt.colorbar(plt.cm.ScalarMappable(cmap=CMAP, norm=norm_post),
                         ax=ax, label='P(source)', shrink=0.82)

            ax.plot(best_x, best_y, 'r*', markersize=18, markeredgecolor='white',
                    markeredgewidth=1.0, label=f'Estimate: ({best_x:.2f}, {best_y:.2f})', zorder=6)
            theta = np.linspace(0, 2 * np.pi, 100)
            ax.plot(best_x + conf_r * np.cos(theta), best_y + conf_r * np.sin(theta),
                    'w--', linewidth=1.5, alpha=0.8, label=f'1\u03c3: {conf_r:.2f}m', zorder=6)

            max_ppm_idx = np.argmax(ppm)
            ax.plot(xs[max_ppm_idx], ys[max_ppm_idx], 'w^', markersize=10,
                    markeredgecolor='black', markeredgewidth=0.5,
                    label='Max reading', zorder=6)

            _scalebar(ax)
            ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
            ax.set_title('Source Localization (Bayesian)')
            ax.legend(loc='upper right', fontsize=9)
            ax.set_aspect('equal')
            fig.tight_layout()
            fig.savefig(os.path.join(out, f'{clean}_source_localization.png'), dpi=250)
            plt.close(fig)

            print(
                f"  Source estimate: ({best_x:.2f}, {best_y:.2f}) \u00b1 {conf_r:.2f}m\n"
                f"  Max-reading at ({xs[max_ppm_idx]:.2f}, {ys[max_ppm_idx]:.2f})"
            )

            print(f"Saved 10 plots for {t} -> {out}")


def main(args=None):
    rclpy.init(args=args)
    node = AutoCoverageMapper()
    exe = MultiThreadedExecutor()
    exe.add_node(node)
    try:
        exe.spin()
    except KeyboardInterrupt:
        node._running = False
        print(f"\n[Ctrl+C] Saving data to {node._run_dir} ...")
        try:
            node._save_data()
        except Exception as e:
            print(f"  Warning: NPZ save failed: {e}")
        try:
            node._save_readme()
        except Exception as e:
            print(f"  Warning: README save failed: {e}")
        try:
            node._save_occupancy_grid()
        except Exception as e:
            print(f"  Warning: Occupancy grid save failed: {e}")
        if node.generate_plots:
            try:
                node.save_plots()
            except Exception as e:
                print(f"  Warning: Plot save failed: {e}")
        print(f"[Ctrl+C] Done. Run folder: {node._run_dir}")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
