#!/usr/bin/env python3
"""
make_synthetic_run.py — Fabricate a run_data.npz (+ occupancy_grid.pgm/.yaml)
that matches the schema written by auto_coverage_mapper._save_data, so the
offline renderers (plot_from_npz.py, risk_map_prototype.py) and gas_viz can be
exercised end-to-end WITHOUT ROS/GADEN.

It simulates a lawnmower sweep of the 10x6 empty room through two Gaussian
plumes at the real source positions:
    gas1 = ethanol  @ [1, 4.95]  (strong, ~30 ppm peak, y-elongated line plume)
    gas2 = methane  @ [5, 3.0]   (weak,   ~2.2 ppm peak, near-isotropic)

This validates plumbing + visuals + estimator math. It is NOT a GADEN-accuracy
test — real-plume validation happens in the ROS sim.

Usage:
    python3 make_synthetic_run.py [out_dir]
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gas_viz  # noqa: E402

ROOM_W, ROOM_H = 10.0, 6.0
SOURCES = {
    'gas1_Sensor_reading': dict(name='ethanol', pos=(1.0, 4.95),
                                peak=30.0, sx=0.9, sy=1.6),   # line-ish, tall
    'gas2_Sensor_reading': dict(name='methane', pos=(5.0, 3.0),
                                peak=2.2, sx=0.8, sy=0.8),    # weak, isotropic
}
BASELINE = {'gas1_Sensor_reading': (0.05, 0.08),    # (mu, sigma) ppm
            'gas2_Sensor_reading': (0.02, 0.04)}


def lawnmower(step=0.5, ds=0.06):
    """Dense boustrophedon path covering the room interior."""
    xs, ys = [], []
    x = 0.6
    up = True
    while x <= ROOM_W - 0.6:
        col = np.arange(0.6, ROOM_H - 0.6, ds)
        if not up:
            col = col[::-1]
        xs.extend([x] * len(col))
        ys.extend(col.tolist())
        # connector along x at the row end
        x_next = min(x + step, ROOM_W - 0.6)
        link = np.arange(x, x_next, ds)
        xs.extend(link.tolist())
        ys.extend([col[-1]] * len(link))
        x = x_next + 1e-9
        up = not up
    return np.array(xs), np.array(ys)


def plume(xs, ys, src, rng):
    cx, cy = src['pos']
    g = src['peak'] * np.exp(-((xs - cx) ** 2 / (2 * src['sx'] ** 2)
                              + (ys - cy) ** 2 / (2 * src['sy'] ** 2)))
    # intermittent turbulent dropouts + sensor noise, clipped ≥ 0
    g *= rng.uniform(0.6, 1.0, size=g.shape)
    g += rng.normal(0, 0.03 * src['peak'] + 0.02, size=g.shape)
    return np.clip(g, 0, None)


def save_occupancy(out_dir, res=0.05):
    """Perimeter-wall occupancy grid in the PGM/YAML format gas_viz.load_walls
    expects (origin at [0,0], occupied=0, free=254, saved flipud)."""
    w = int(round(ROOM_W / res))
    h = int(round(ROOM_H / res))
    occ = np.zeros((h, w), dtype=np.int16)          # 0 = free
    b = max(1, int(round(0.1 / res)))
    occ[:b, :] = 100
    occ[-b:, :] = 100
    occ[:, :b] = 100
    occ[:, -b:] = 100
    img = np.full((h, w), 205, dtype=np.uint8)
    img[occ == 0] = 254
    img[occ >= 50] = 0
    img = np.flipud(img)
    with open(os.path.join(out_dir, 'occupancy_grid.pgm'), 'wb') as f:
        f.write(f'P5\n{w} {h}\n255\n'.encode())
        f.write(img.tobytes())
    with open(os.path.join(out_dir, 'occupancy_grid.yaml'), 'w') as f:
        f.write("image: occupancy_grid.pgm\n"
                f"resolution: {res}\norigin: [0.0, 0.0, 0.0]\n"
                "negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n")


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(42)
    xs, ys = lawnmower()
    t = np.arange(len(xs)) * 0.3                     # ~3 Hz

    save_dict = {}
    for key, src in SOURCES.items():
        ppm = plume(xs, ys, src, rng)
        save_dict[f'{key}_x'] = xs
        save_dict[f'{key}_y'] = ys
        save_dict[f'{key}_ppm'] = ppm
        save_dict[f'{key}_timestamp'] = t
        mu, sigma = BASELINE[key]
        save_dict[f'{key}_baseline_mu'] = np.array(mu)
        save_dict[f'{key}_baseline_sigma'] = np.array(sigma)

        r = gas_viz.estimate_source(xs, ys, ppm, timestamps=t,
                                    baseline_sigma=sigma)
        save_dict[f'{key}_posterior'] = r['posterior']
        save_dict[f'{key}_posterior_gx'] = r['gx']
        save_dict[f'{key}_posterior_gy'] = r['gy']
        save_dict[f'{key}_source_xy'] = np.array([r['src_x'], r['src_y']])
        save_dict[f'{key}_map_xy'] = np.array([r['best_x'], r['best_y']])
        save_dict[f'{key}_plume_axis'] = np.array(r['axis'])
        save_dict[f'{key}_sigma_major_minor'] = np.array(
            [r['sigma_major'], r['sigma_minor']])
        save_dict[f'{key}_directional'] = np.array(r['directional'])
        save_dict[f'{key}_low_confidence'] = np.array(r['low_confidence'])
        save_dict[f'{key}_peak_ppm'] = np.array(r['peak_ppm'])
        save_dict[f'{key}_method'] = np.array(r['method'])
        print(f"{src['name']:8s}: peak {r['peak_ppm']:5.2f} ppm  "
              f"MAP=({r['best_x']:.2f},{r['best_y']:.2f})  true={src['pos']}  "
              f"HPD50={r['hpd50']:.2f} m²  method={r['method']}")

    save_dict['coverage_pct'] = np.array(92.0)
    save_dict['hotspots'] = np.array([[s['pos'][0], s['pos'][1], s['peak']]
                                      for s in SOURCES.values()])
    np.savez_compressed(os.path.join(out_dir, 'run_data.npz'), **save_dict)
    save_occupancy(out_dir)
    print(f"\nWrote synthetic run -> {out_dir}")
    print("  run_data.npz, occupancy_grid.pgm, occupancy_grid.yaml")


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '/tmp/synthetic_run')
