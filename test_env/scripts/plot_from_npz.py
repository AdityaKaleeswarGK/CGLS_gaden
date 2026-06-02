#!/usr/bin/env python3
"""
Offline renderer for auto_coverage_mapper runs.

Regenerates the probabilistic source-localization figures and the
predicted-vs-actual benchmark from a finished run's ``run_data.npz`` — no need
to re-run the mission or have ROS running. Useful when a run was completed with
``generate_plots:=false`` (the posteriors are still saved in the NPZ).

Usage:
    python3 plot_from_npz.py <run_dir> [scenario_path]

    <run_dir>        folder containing run_data.npz (and optionally
                     occupancy_grid.pgm / .yaml for the wall overlay)
    [scenario_path]  scenario config dir holding simulations/*/sim.yaml, used to
                     overlay ground-truth sources and compute localization error.
"""
import os
import sys
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.patches import Ellipse
from scipy.ndimage import gaussian_filter

try:
    import yaml
except ImportError:
    yaml = None

CMAP = 'viridis'


def parse_true_sources(scenario_path):
    """Read ground-truth gas source positions from simulations/*/sim.yaml."""
    sources = []
    if not (scenario_path and yaml and os.path.isdir(scenario_path)):
        return sources
    sim_dir = os.path.join(scenario_path, 'simulations')
    if not os.path.isdir(sim_dir):
        return sources
    for name in sorted(os.listdir(sim_dir)):
        p = os.path.join(sim_dir, name, 'sim.yaml')
        if not os.path.isfile(p):
            continue
        try:
            with open(p) as f:
                cfg = yaml.safe_load(f)
            pos = (cfg.get('source', {}) or {}).get('position')
            if pos and len(pos) >= 2:
                sources.append({'name': name, 'x': float(pos[0]), 'y': float(pos[1])})
        except Exception:
            pass
    return sources


def load_walls(run_dir):
    """Return (rgba, extent) for a black-walls overlay, or (None, None)."""
    pgm = os.path.join(run_dir, 'occupancy_grid.pgm')
    yml = os.path.join(run_dir, 'occupancy_grid.yaml')
    if not (os.path.isfile(pgm) and os.path.isfile(yml) and yaml):
        return None, None
    try:
        with open(yml) as f:
            meta = yaml.safe_load(f)
        res = float(meta['resolution'])
        ox, oy = float(meta['origin'][0]), float(meta['origin'][1])
        with open(pgm, 'rb') as f:
            assert f.readline().strip() == b'P5'
            w, h = map(int, f.readline().split())
            f.readline()  # maxval
            data = np.frombuffer(f.read(), dtype=np.uint8).reshape((h, w))
        data = np.flipud(data)  # undo the flip applied on save → map frame
        rgba = np.zeros((h, w, 4), dtype=float)
        rgba[data <= 50] = [0, 0, 0, 1.0]   # occupied → solid black
        extent = [ox, ox + w * res, oy, oy + h * res]
        return rgba, extent
    except Exception as e:
        print(f"  (wall overlay skipped: {e})")
        return None, None


def _recompute_estimator():
    """Lazily import the node's _estimate_source so we can re-run the (possibly
    updated) localization algorithm on an old run's raw readings."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import auto_coverage_mapper as M
    obj = M.AutoCoverageMapper.__new__(M.AutoCoverageMapper)
    return obj._estimate_source


def render(run_dir, scenario_path=None, recompute=False):
    npz = os.path.join(run_dir, 'run_data.npz')
    if not os.path.isfile(npz):
        sys.exit(f"run_data.npz not found in {run_dir}")
    z = dict(np.load(npz))
    if recompute:
        est = _recompute_estimator()
        gwind = tuple(z['wind_upwind_xy']) if 'wind_upwind_xy' in z else None
        gcons = float(z['wind_consistency']) if 'wind_consistency' in z else 0.0
        print(f"Recomputing source estimates with the current algorithm"
              f"{' (wind-aware)' if gwind is not None else ''}...")
        for k in [k for k in z if k.endswith('_posterior') and 'gx' not in k and 'gy' not in k]:
            g = k[:-len('_posterior')]
            # prefer this gas's own local wind; fall back to the global mean
            wind = tuple(z[f'{g}_wind_upwind_xy']) if f'{g}_wind_upwind_xy' in z else gwind
            wcons = float(z[f'{g}_wind_consistency']) if f'{g}_wind_consistency' in z else gcons
            r = est(z[f'{g}_x'], z[f'{g}_y'], z[f'{g}_ppm'], timestamps=z.get(f'{g}_timestamp'),
                    wind=wind, wind_consistency=wcons)
            z[f'{g}_posterior'] = r['posterior']
            z[f'{g}_posterior_gx'] = r['gx']; z[f'{g}_posterior_gy'] = r['gy']
            z[f'{g}_source_xy'] = np.array([r['src_x'], r['src_y']])
            z[f'{g}_map_xy'] = np.array([r['best_x'], r['best_y']])
            z[f'{g}_plume_axis'] = np.array(r['axis'])
            z[f'{g}_sigma_major_minor'] = np.array([r['sigma_major'], r['sigma_minor']])
            z[f'{g}_directional'] = np.array(r['directional'])
            z[f'{g}_low_confidence'] = np.array(r['low_confidence'])
            z[f'{g}_peak_ppm'] = np.array(r['peak_ppm'])
            z[f'{g}_method'] = np.array(r['method'])
    occ_rgba, occ_extent = load_walls(run_dir)
    true_sources = parse_true_sources(scenario_path)
    cmap_obj = plt.get_cmap(CMAP)

    # Discover gas channels that carry a posterior
    gases = sorted({k[:-len('_posterior')] for k in z
                    if k.endswith('_posterior') and not k.endswith('_gx')
                    and not k.endswith('_gy')})

    per_gas = {}  # clean -> dict for benchmark
    for g in gases:
        ppm = z[f'{g}_ppm']
        if (ppm > 0.1).sum() < 5:
            print(f"{g}: <5 positive samples — skipped.")
            continue
        xs, ys = z[f'{g}_x'], z[f'{g}_y']
        post = z[f'{g}_posterior']
        gx, gy = z[f'{g}_posterior_gx'], z[f'{g}_posterior_gy']
        sx, sy = z[f'{g}_source_xy']
        smaj, smin = z[f'{g}_sigma_major_minor']
        ux, uy = z[f'{g}_plume_axis']
        # Prefer the saved flags; fall back to a shape heuristic for old NPZs.
        directional = bool(z[f'{g}_directional']) if f'{g}_directional' in z else bool(smaj > 1.3 * smin)
        low_conf = bool(z[f'{g}_low_confidence']) if f'{g}_low_confidence' in z else False
        peak_ppm = float(z[f'{g}_peak_ppm']) if f'{g}_peak_ppm' in z else float('nan')

        # Match sensor to its own source: launch pairs gasN ↔ simN. Fall back
        # to nearest only if no simN match (avoids a wrong-but-closer source
        # reporting a misleadingly small error).
        entry = {'src_x': float(sx), 'src_y': float(sy)}
        if true_sources:
            gas_label = g.split('_')[0]
            sim_name = gas_label.replace('gas', 'sim') if gas_label.startswith('gas') else None
            tgt = next((s for s in true_sources if s['name'] == sim_name), None)
            if tgt is None:
                tgt = min(true_sources, key=lambda s: math.hypot(sx - s['x'], sy - s['y']))
            entry.update(error_m=float(math.hypot(sx - tgt['x'], sy - tgt['y'])),
                         true_x=tgt['x'], true_y=tgt['y'], true_name=tgt['name'])
        per_gas[g] = entry

        # ── figure ──
        fig, ax = plt.subplots(figsize=(10, 8))
        post_smooth = gaussian_filter(post, sigma=2.0)
        norm_post = PowerNorm(gamma=0.4, vmin=0, vmax=max(post.max(), 1e-10))
        rgba_post = cmap_obj(norm_post(post_smooth))
        ax.imshow(rgba_post, extent=[gx[0, 0], gx[0, -1], gy[0, 0], gy[-1, 0]],
                  origin='lower', interpolation='bilinear', zorder=1, aspect='auto')
        if occ_rgba is not None:
            ax.imshow(occ_rgba, extent=occ_extent, origin='lower',
                      interpolation='nearest', zorder=5)
        plt.colorbar(plt.cm.ScalarMappable(cmap=CMAP, norm=norm_post),
                     ax=ax, label='P(source)', shrink=0.82)

        # 50% credible search region — the primary "look here" deliverable
        flat = post.ravel(); order = np.argsort(flat)[::-1]
        csum = np.cumsum(flat[order]); csum /= csum[-1]
        thr50 = flat[order[min(np.searchsorted(csum, 0.50), len(order) - 1)]]
        try:
            ax.contour(gx, gy, post, levels=[thr50], colors=['#ff3b3b'],
                       linewidths=2.0, zorder=6)
            ax.plot([], [], color='#ff3b3b', lw=2.0, label='50% search region')
        except Exception:
            pass

        ang = math.degrees(math.atan2(uy, ux))
        ax.add_patch(Ellipse((sx, sy), width=2 * smaj, height=2 * smin, angle=ang,
                             fill=False, edgecolor='white', linestyle='--',
                             linewidth=1.5, alpha=0.85, zorder=6))
        alen = max(smaj, 0.5)
        if directional:
            ax.annotate('', xy=(sx + ux * alen, sy + uy * alen),
                        xytext=(sx - ux * alen, sy - uy * alen),
                        arrowprops=dict(arrowstyle='-|>', color='#00e5ff',
                                        lw=2.0, alpha=0.9), zorder=6)
        else:
            ax.plot([sx - ux * alen, sx + ux * alen], [sy - uy * alen, sy + uy * alen],
                    color='#00e5ff', lw=1.8, alpha=0.8, zorder=6,
                    label='Plume axis (direction unknown)')

        ax.plot(sx, sy, 'r*', markersize=18, markeredgecolor='white',
                markeredgewidth=1.0, zorder=7, label=f'Best guess: ({sx:.2f}, {sy:.2f})')
        mi = int(np.argmax(ppm))
        ax.plot(xs[mi], ys[mi], 'w^', markersize=10, markeredgecolor='black',
                markeredgewidth=0.5, label='Max reading', zorder=6)
        for s in true_sources:
            ax.plot(s['x'], s['y'], 'P', markersize=13, color='#ff1493',
                    markeredgecolor='white', markeredgewidth=0.8, zorder=8)
        err_txt = ''
        if 'error_m' in entry:
            ax.plot([sx, entry['true_x']], [sy, entry['true_y']], color='#ff1493',
                    linestyle=':', linewidth=1.3, alpha=0.9, zorder=7)
            err_txt = f"  |  error = {entry['error_m']:.2f} m"
            ax.plot([], [], 'P', color='#ff1493',
                    label=f"True source ({entry['true_x']:.2f}, {entry['true_y']:.2f})")

        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
        meth = str(z[f'{g}_method']) if f'{g}_method' in z else 'wind-free'
        title = f'Source Region ({meth}) — P(source) + 50% search region' + err_txt
        if low_conf:
            title += f"\n[!] LOW CONFIDENCE — peak only {peak_ppm:.2f} ppm (gas barely detected)"
        ax.set_title(title)
        ax.legend(loc='upper right', fontsize=8)
        ax.set_aspect('equal')
        fig.tight_layout()
        out = os.path.join(run_dir, f'{g}_source_localization.png')
        fig.savefig(out, dpi=250); plt.close(fig)
        print(f"Saved {out}" + (f"  (error {entry['error_m']:.2f} m)" if 'error_m' in entry else ''))

    # ── benchmark figure ──
    errs = {k: v['error_m'] for k, v in per_gas.items() if 'error_m' in v}
    if true_sources and errs:
        fig, (axm, axb) = plt.subplots(1, 2, figsize=(18, 8),
                                       gridspec_kw={'width_ratios': [1.4, 1]})
        if occ_rgba is not None:
            axm.imshow(occ_rgba, extent=occ_extent, origin='lower',
                       interpolation='nearest', zorder=5)
        for s in true_sources:
            axm.plot(s['x'], s['y'], 'P', markersize=15, color='#ff1493',
                     markeredgecolor='white', markeredgewidth=1.0, zorder=8)
            axm.text(s['x'], s['y'], f"  {s['name']}", color='#ff1493', fontsize=8, zorder=8)
        axm.plot([], [], 'P', color='#ff1493', label='True source')
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, max(len(per_gas), 1)))
        for (g, e), col in zip(per_gas.items(), colors):
            if 'error_m' not in e:
                continue
            axm.plot(e['src_x'], e['src_y'], '*', markersize=16, color=col,
                     markeredgecolor='black', markeredgewidth=0.6, zorder=7,
                     label=f"{g} (err {e['error_m']:.2f}m)")
            axm.plot([e['src_x'], e['true_x']], [e['src_y'], e['true_y']],
                     color=col, linestyle=':', linewidth=1.4, alpha=0.9, zorder=6)
        axm.set_xlabel('X (m)'); axm.set_ylabel('Y (m)')
        axm.set_title('Predicted vs. Actual Source Locations')
        axm.legend(loc='upper right', fontsize=8); axm.set_aspect('equal')

        names = list(errs); vals = [errs[n] for n in names]
        bars = axb.bar(range(len(names)), vals, color=colors[:len(names)],
                       edgecolor='black', linewidth=0.6)
        mean_err = float(np.mean(vals))
        axb.axhline(mean_err, color='#c0392b', linestyle='--', linewidth=1.3,
                    label=f'Mean = {mean_err:.2f} m')
        for b, v in zip(bars, vals):
            axb.text(b.get_x() + b.get_width() / 2, v, f'{v:.2f}',
                     ha='center', va='bottom', fontsize=9)
        axb.set_xticks(range(len(names)))
        axb.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
        axb.set_ylabel('Localization error (m)')
        axb.set_title('Localization Error per Gas')
        axb.legend(fontsize=9); axb.grid(True, axis='y', alpha=0.2)
        fig.suptitle('Source Localization Benchmark', fontsize=15, fontweight='bold')
        fig.tight_layout()
        out = os.path.join(run_dir, 'benchmark_localization.png')
        fig.savefig(out, dpi=250); plt.close(fig)
        print(f"Saved {out}  (mean error {mean_err:.2f} m)")
    else:
        print("Benchmark skipped (no scenario_path / true sources given).")


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if a != '--recompute']
    recompute = '--recompute' in sys.argv
    if not args:
        sys.exit(__doc__)
    render(args[0], args[1] if len(args) > 1 else None, recompute=recompute)
