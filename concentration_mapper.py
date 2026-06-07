#!/usr/bin/env python3
"""
Gas Concentration Mapper — standalone node that records gas sensor readings
along the robot's path and generates comprehensive visualization outputs:

  1. Scatter plot (raw readings)
  2. Interpolated heatmap (cubic + nearest fill)
  3. Contour map with iso-concentration lines
  4. Concentration gradient (arrows pointing toward source)
  5. Time-series along path
  6. 2x2 summary dashboard
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node
from olfaction_msgs.msg import GasSensor
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener


class GasMapperNode(Node):
    def __init__(self):
        super().__init__('gas_concentration_mapper')
        self.set_parameters([rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)])

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.declare_parameter('sensor_topics', ['/fake_pid/Sensor_reading'])
        self.declare_parameter('base_frame', 'PioneerP3DX_base_link')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('grid_resolution', 0.1)  # heatmap cell size in meters

        self.base_frame = self.get_parameter('base_frame').value
        self.map_frame = self.get_parameter('map_frame').value
        self.grid_res = self.get_parameter('grid_resolution').value

        topics = self.get_parameter('sensor_topics').value

        self.data_history = {}
        for topic in topics:
            self.data_history[topic] = {'x': [], 'y': [], 'ppm': []}
            self.create_subscription(GasSensor, topic, lambda msg, t=topic: self.sensor_callback(msg, t), 10)

        self.get_logger().info("Gas Concentration Mapper initialized. Tracking coordinates and sensor PPM.")

    def sensor_callback(self, msg, topic):
        try:
            t = self.tf_buffer.lookup_transform(self.map_frame, self.base_frame, rclpy.time.Time())
            x = t.transform.translation.x
            y = t.transform.translation.y
            self.data_history[topic]['x'].append(x)
            self.data_history[topic]['y'].append(y)
            self.data_history[topic]['ppm'].append(msg.raw)
        except TransformException as ex:
            if not hasattr(self, '_warned_tf'):
                self.get_logger().error(f"TF Missing! Can't get robot location: {ex}")
                self._warned_tf = True

    def _estimate_source(self, xs, ys, ppm, grid_res=0.15):
        """
        Bayesian source localization using isotropic diffusion model.
        No wind needed — uses C(r) ∝ 1/r decay from candidate source.
        Marginalizes out the unknown release rate analytically.
        Returns (posterior, grid_x, grid_y, best_x, best_y, confidence_radius).
        """
        margin = 0.3
        cx = np.arange(xs.min() - margin, xs.max() + margin, grid_res)
        cy = np.arange(ys.min() - margin, ys.max() + margin, grid_res)
        gx, gy = np.meshgrid(cx, cy)

        eps = 0.15
        baseline_mask = ppm < np.percentile(ppm, 50)
        if baseline_mask.sum() > 5:
            sigma = max(np.std(ppm[baseline_mask]), 0.01)
        else:
            sigma = max(np.std(ppm) * 0.3, 0.01)

        log_posterior = np.zeros(gx.shape)

        for i in range(gx.shape[0]):
            for j in range(gx.shape[1]):
                sx, sy = gx[i, j], gy[i, j]
                r = np.sqrt((xs - sx) ** 2 + (ys - sy) ** 2) + eps
                f = 1.0 / r

                ff = np.dot(f, f)
                if ff < 1e-12:
                    log_posterior[i, j] = -1e10
                    continue
                alpha = max(np.dot(f, ppm) / ff, 0.0)

                residual = ppm - alpha * f
                log_posterior[i, j] = -0.5 * np.sum(residual ** 2) / (sigma ** 2)

        log_posterior -= np.max(log_posterior)
        posterior = np.exp(log_posterior)
        posterior /= posterior.sum()

        peak_idx = np.argmax(posterior)
        peak_r, peak_c = np.unravel_index(peak_idx, posterior.shape)
        best_x = gx[peak_r, peak_c]
        best_y = gy[peak_r, peak_c]

        dx = gx - best_x
        dy = gy - best_y
        confidence_radius = np.sqrt(np.sum(posterior * (dx ** 2 + dy ** 2)))

        return posterior, gx, gy, best_x, best_y, confidence_radius

    def save_plots(self):
        from scipy.interpolate import griddata

        output_dir = os.path.expanduser('~/gaden_results/concentration_maps')
        os.makedirs(output_dir, exist_ok=True)
        print(f"Generating concentration maps into {output_dir}...")

        for topic, data in self.data_history.items():
            if len(data['x']) == 0:
                print(f"No data for {topic}, skipping.")
                continue

            xs = np.array(data['x'])
            ys = np.array(data['y'])
            ppm = np.array(data['ppm'])
            clean = topic.replace('/', '_').strip('_')

            # Build interpolated grid
            margin = 0.2
            xi = np.arange(xs.min() - margin, xs.max() + margin, self.grid_res)
            yi = np.arange(ys.min() - margin, ys.max() + margin, self.grid_res)
            xi, yi = np.meshgrid(xi, yi)

            zi = griddata((xs, ys), ppm, (xi, yi), method='cubic')
            zi_nearest = griddata((xs, ys), ppm, (xi, yi), method='nearest')
            mask = np.isnan(zi)
            zi[mask] = zi_nearest[mask]
            zi = np.clip(zi, 0, None)  # concentration can't be negative

            peak_idx = np.nanargmax(zi)
            peak_r, peak_c = np.unravel_index(peak_idx, zi.shape)
            peak_x, peak_y = xi[peak_r, peak_c], yi[peak_r, peak_c]
            peak_val = np.nanmax(zi)

            # ── 1) Scatter ──
            fig, ax = plt.subplots(figsize=(10, 8))
            sc = ax.scatter(xs, ys, c=ppm, cmap='viridis', s=8, alpha=0.7)
            plt.colorbar(sc, ax=ax, label='Concentration (ppm)')
            ax.set_title(f'Raw Concentration Scatter: {topic}')
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_aspect('equal')
            fig.savefig(os.path.join(output_dir, f'{clean}_scatter.png'), dpi=150)
            plt.close(fig)

            # ── 2) Heatmap ──
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.pcolormesh(xi, yi, zi, cmap='viridis', shading='auto')
            plt.colorbar(im, ax=ax, label='Concentration (ppm)')
            ax.plot(peak_x, peak_y, 'c*', markersize=15, label=f'Peak: {peak_val:.1f} ppm')
            ax.set_title(f'Concentration Heatmap: {topic}')
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.legend()
            ax.set_aspect('equal')
            fig.savefig(os.path.join(output_dir, f'{clean}_heatmap.png'), dpi=150)
            plt.close(fig)

            # ── 3) Contour ──
            fig, ax = plt.subplots(figsize=(10, 8))
            levels = np.linspace(0, peak_val * 1.05, 15)
            cs = ax.contourf(xi, yi, zi, levels=levels, cmap='viridis', alpha=0.85)
            ax.contour(xi, yi, zi, levels=levels, colors='k', linewidths=0.3, alpha=0.5)
            plt.colorbar(cs, ax=ax, label='Concentration (ppm)')
            ax.plot(peak_x, peak_y, 'b*', markersize=15, label=f'Peak: {peak_val:.1f} ppm')
            ax.set_title(f'Concentration Contour Map: {topic}')
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.legend()
            ax.set_aspect('equal')
            fig.savefig(os.path.join(output_dir, f'{clean}_contour.png'), dpi=150)
            plt.close(fig)

            # ── 4) Gradient (diffusion direction) ──
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.pcolormesh(xi, yi, zi, cmap='viridis', shading='auto', alpha=0.4)
            grad_y, grad_x = np.gradient(zi, self.grid_res, self.grid_res)
            skip = max(1, len(xi) // 20)
            ax.quiver(
                xi[::skip, ::skip], yi[::skip, ::skip],
                grad_x[::skip, ::skip], grad_y[::skip, ::skip],
                color='navy', alpha=0.7
            )
            ax.plot(peak_x, peak_y, 'c*', markersize=15, label='Estimated source')
            ax.set_title(f'Concentration Gradient: {topic}')
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.legend()
            ax.set_aspect('equal')
            fig.savefig(os.path.join(output_dir, f'{clean}_gradient.png'), dpi=150)
            plt.close(fig)

            # ── 5) Time-series ──
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(ppm, linewidth=0.8, color='darkorange')
            ax.fill_between(range(len(ppm)), ppm, alpha=0.3, color='orange')
            ax.set_title(f'Concentration Along Path: {topic}')
            ax.set_xlabel('Sample index (time-ordered)')
            ax.set_ylabel('Concentration (ppm)')
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, f'{clean}_timeseries.png'), dpi=150)
            plt.close(fig)

            # ── 6) Dashboard (2x2) ──
            fig, axes = plt.subplots(2, 2, figsize=(16, 14))

            sc = axes[0, 0].scatter(xs, ys, c=ppm, cmap='viridis', s=5, alpha=0.7)
            plt.colorbar(sc, ax=axes[0, 0], label='ppm')
            axes[0, 0].set_title('Coverage Scatter')
            axes[0, 0].set_aspect('equal')

            im = axes[0, 1].pcolormesh(xi, yi, zi, cmap='viridis', shading='auto')
            plt.colorbar(im, ax=axes[0, 1], label='ppm')
            axes[0, 1].plot(peak_x, peak_y, 'c*', markersize=12)
            axes[0, 1].set_title('Concentration Heatmap')
            axes[0, 1].set_aspect('equal')

            cs = axes[1, 0].contourf(xi, yi, zi, levels=levels, cmap='viridis', alpha=0.85)
            axes[1, 0].contour(xi, yi, zi, levels=levels, colors='k', linewidths=0.3, alpha=0.5)
            plt.colorbar(cs, ax=axes[1, 0], label='ppm')
            axes[1, 0].set_title('Contour Map')
            axes[1, 0].set_aspect('equal')

            axes[1, 1].plot(ppm, linewidth=0.8, color='darkorange')
            axes[1, 1].fill_between(range(len(ppm)), ppm, alpha=0.3, color='orange')
            axes[1, 1].set_title('Concentration Along Path')
            axes[1, 1].set_xlabel('Sample index')
            axes[1, 1].set_ylabel('ppm')

            fig.suptitle(f'Gas Mapping Dashboard: {topic}', fontsize=14, fontweight='bold')
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, f'{clean}_dashboard.png'), dpi=150)
            plt.close(fig)

            # ── 7) Bayesian Source Localization ──
            print(f"Running Bayesian source localization for {topic}...")
            posterior, bgx, bgy, best_x, best_y, conf_r = self._estimate_source(xs, ys, ppm)

            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.pcolormesh(bgx, bgy, posterior, cmap='viridis', shading='auto')
            plt.colorbar(im, ax=ax, label='P(source here)')
            ax.plot(best_x, best_y, 'r*', markersize=18, markeredgecolor='white', markeredgewidth=1.0,
                    label=f'MAP estimate: ({best_x:.2f}, {best_y:.2f})')
            theta = np.linspace(0, 2 * np.pi, 100)
            ax.plot(best_x + conf_r * np.cos(theta), best_y + conf_r * np.sin(theta),
                    'r--', linewidth=1.5, alpha=0.7, label=f'1σ radius: {conf_r:.2f}m')
            max_ppm_idx = np.argmax(ppm)
            ax.plot(xs[max_ppm_idx], ys[max_ppm_idx], 'w^', markersize=10, markeredgecolor='black',
                    label=f'Max reading: ({xs[max_ppm_idx]:.2f}, {ys[max_ppm_idx]:.2f})')
            ax.set_title(f'Bayesian Source Localization: {topic}')
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.legend(loc='upper left', fontsize=9)
            ax.set_aspect('equal')
            fig.savefig(os.path.join(output_dir, f'{clean}_source_localization.png'), dpi=150)
            plt.close(fig)

            print(f"Saved 7 plots for {topic} -> {output_dir}")
            print(
                f"  Bayesian source estimate: ({best_x:.2f}, {best_y:.2f}) ± {conf_r:.2f}m\n"
                f"  (Max-reading at ({xs[max_ppm_idx]:.2f}, {ys[max_ppm_idx]:.2f}) — "
                f"Bayesian uses ALL readings including zeros)"
            )


def main(args=None):
    rclpy.init(args=args)
    mapper = GasMapperNode()
    try:
        rclpy.spin(mapper)
    except KeyboardInterrupt:
        pass
    finally:
        mapper.save_plots()
        mapper.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
