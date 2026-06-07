#!/usr/bin/env python3
"""
Offline renderer for auto_coverage_mapper runs.

Regenerates, from a finished run's ``run_data.npz`` (no ROS, no re-run):
  * ``<gas>_concentration_intensity.png`` — rich per-gas concentration heatmap
    showing WHERE the gas peaks (data-relative bands, peak region, coverage
    hatching, walls). The "middle ground" map.
  * ``<gas>_source_localization.png`` — probabilistic source REGION with
    50% / 90% credible (HPD) contours.
  * ``benchmark_localization.png`` — predicted vs. ground-truth sources.

All heavy lifting lives in ``gas_viz`` (shared with the live node), so this is a
thin orchestrator. ``--recompute`` re-runs the *current* estimator on the raw
readings — useful after the algorithm changes — and now works without rclpy
because the estimator is ROS-free.

Usage:
    python3 plot_from_npz.py <run_dir> [scenario_path] [--recompute]
"""
import os
import sys
import math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gas_viz  # noqa: E402

try:
    import yaml
except ImportError:
    yaml = None


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


def _result_from_npz(z, g):
    """Reconstruct an estimate_source-style result dict from saved arrays."""
    post = z[f'{g}_posterior']
    gx, gy = z[f'{g}_posterior_gx'], z[f'{g}_posterior_gy']
    sx, sy = z[f'{g}_source_xy']
    smaj, smin = z[f'{g}_sigma_major_minor']
    ux, uy = z[f'{g}_plume_axis']
    directional = bool(z[f'{g}_directional']) if f'{g}_directional' in z else bool(smaj > 1.3 * smin)
    low_conf = bool(z[f'{g}_low_confidence']) if f'{g}_low_confidence' in z else False
    peak_ppm = float(z[f'{g}_peak_ppm']) if f'{g}_peak_ppm' in z else float('nan')
    method = str(z[f'{g}_method']) if f'{g}_method' in z else 'bayes-1/r'
    return {
        'posterior': post, 'gx': gx, 'gy': gy,
        'src_x': float(sx), 'src_y': float(sy),
        'sigma_major': float(smaj), 'sigma_minor': float(smin),
        'axis': (float(ux), float(uy)), 'directional': directional,
        'method': method, 'low_confidence': low_conf, 'peak_ppm': peak_ppm,
        'candidates': gas_viz.top_candidates(post, gx, gy, n=3),
    }


def render(run_dir, scenario_path=None, recompute=False):
    npz = os.path.join(run_dir, 'run_data.npz')
    if not os.path.isfile(npz):
        sys.exit(f"run_data.npz not found in {run_dir}")
    z = dict(np.load(npz, allow_pickle=True))

    walls = gas_viz.load_walls(run_dir)
    true_sources = parse_true_sources(scenario_path)

    # Discover gas channels that carry a posterior (or, after recompute, ppm).
    gases = sorted({k[:-len('_posterior')] for k in z
                    if k.endswith('_posterior') and not k.endswith('_gx')
                    and not k.endswith('_gy')})
    if recompute:
        gases = sorted({k[:-len('_ppm')] for k in z if k.endswith('_ppm')
                        and not k.endswith('peak_ppm')})
        gwind = tuple(z['wind_upwind_xy']) if 'wind_upwind_xy' in z else None
        gcons = float(z['wind_consistency']) if 'wind_consistency' in z else 0.0
        print("Recomputing source estimates with the current estimator...")

    per_gas = {}
    for g in gases:
        if f'{g}_ppm' not in z:
            continue
        ppm = z[f'{g}_ppm']
        if (ppm > 0.1).sum() < 5:
            print(f"{g}: <5 positive samples — skipped.")
            continue
        xs, ys = z[f'{g}_x'], z[f'{g}_y']
        bmu = float(z[f'{g}_baseline_mu']) if f'{g}_baseline_mu' in z else 0.0
        bsig = float(z[f'{g}_baseline_sigma']) if f'{g}_baseline_sigma' in z else None

        if recompute:
            wind = tuple(z[f'{g}_wind_upwind_xy']) if f'{g}_wind_upwind_xy' in z else gwind
            wcons = float(z[f'{g}_wind_consistency']) if f'{g}_wind_consistency' in z else gcons
            r = gas_viz.estimate_source(xs, ys, ppm, timestamps=z.get(f'{g}_timestamp'),
                                        wind=wind, wind_consistency=wcons,
                                        baseline_sigma=bsig)
        else:
            r = _result_from_npz(z, g)
        sx, sy = r['src_x'], r['src_y']

        # Match sensor to its own source (launch pairs gasN ↔ simN); fall back to
        # nearest only when there is no simN match.
        entry = {'src_x': sx, 'src_y': sy}
        if true_sources:
            gas_label = g.split('_')[0]
            sim_name = gas_label.replace('gas', 'sim') if gas_label.startswith('gas') else None
            tgt = next((s for s in true_sources if s['name'] == sim_name), None)
            if tgt is None:
                tgt = min(true_sources, key=lambda s: math.hypot(sx - s['x'], sy - s['y']))
            entry.update(error_m=float(math.hypot(sx - tgt['x'], sy - tgt['y'])),
                         true_x=tgt['x'], true_y=tgt['y'], true_name=tgt['name'])
        per_gas[g] = entry

        # Per-gas figures show only THIS gas's matched ground-truth source.
        matched = ([{'name': entry['true_name'], 'x': entry['true_x'],
                     'y': entry['true_y']}] if 'true_x' in entry else None)
        gas_viz.concentration_intensity_map(
            os.path.join(run_dir, f'{g}_concentration_intensity.png'),
            g, xs, ys, ppm, baseline_mu=bmu, baseline_sigma=bsig,
            walls=walls, true_sources=matched)
        gas_viz.source_posterior_map(
            os.path.join(run_dir, f'{g}_source_localization.png'),
            g, xs, ys, ppm, r, walls=walls, true_sources=matched, entry=entry)
        print(f"Saved {g}: intensity + source maps"
              + (f"  (error {entry['error_m']:.2f} m)" if 'error_m' in entry else ''))

    _benchmark_figure(run_dir, per_gas, true_sources, walls)


def _benchmark_figure(run_dir, per_gas, true_sources, walls):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    errs = {k: v['error_m'] for k, v in per_gas.items() if 'error_m' in v}
    if not (true_sources and errs):
        print("Benchmark skipped (no scenario_path / true sources given).")
        return
    occ_rgba, occ_extent = walls if walls else (None, None)

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


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if a != '--recompute']
    recompute = '--recompute' in sys.argv
    if not args:
        sys.exit(__doc__)
    render(args[0], args[1] if len(args) > 1 else None, recompute=recompute)
