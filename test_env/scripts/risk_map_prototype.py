#!/usr/bin/env python3
"""
PROTOTYPE — Gas situational-awareness ("risk exposure") map from a saved
auto_coverage run. Validates the idea on real data before integrating.

It produces ONE figure with the 4 things a surveillance end-user actually needs:
  1. Risk map: per-cell worst gas exposure in SAFE/ELEVATED/HAZARD bands,
     combined across all gases.
  2. Coverage-confidence overlay: explored vs. NEVER-VISITED (hatched) — so the
     user knows whether "clean" means clean or just unsurveyed.
  3. Trend per hotspot: comparing the first vs. second half of the run, is gas
     in this region RISING / falling / stable, plus first-detection time.
  4. Wind context: prevailing wind arrow + a downwind "spread" cone from the
     worst hotspot (where it's heading / who's downwind).

Usage: python3 risk_map_prototype.py <run_dir>
"""
import os, sys, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Polygon
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

# Shared wall-loader (and other helpers) so this companion view stays consistent
# with the per-gas intensity / source maps emitted by plot_from_npz.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gas_viz

# Per-gas exposure thresholds (ppm): [elevated, hazard].
# Illustrative bands so the demo is meaningful at the sim's ppm scale; in a real
# deployment these come from each gas's exposure limit (e.g. OSHA PEL/IDLH).
GAS_BANDS = {
    'gas1': (2.0, 10.0),   # ethanol
    'gas2': (1.0, 5.0),    # methane (low-ppm here)
    'gas3': (1.0, 5.0),    # hydrogen
    'gas4': (2.0, 10.0),   # propanol
}
DEFAULT_BANDS = (1.0, 5.0)


def load_walls(run_dir):
    return gas_viz.load_walls(run_dir)


def main(run_dir):
    z = np.load(os.path.join(run_dir, 'run_data.npz'))
    occ_rgba, occ_ext = load_walls(run_dir)
    gases = sorted({k.split('_Sensor')[0] for k in z.files if k.endswith('_ppm')})

    # union of all sample positions → common interpolation grid
    allx = np.concatenate([z[f'{g}_Sensor_reading_x'] for g in gases])
    ally = np.concatenate([z[f'{g}_Sensor_reading_y'] for g in gases])
    res = 0.1
    xg = np.arange(allx.min() - 0.3, allx.max() + 0.3, res)
    yg = np.arange(ally.min() - 0.3, ally.max() + 0.3, res)
    xi, yi = np.meshgrid(xg, yg)
    gridpts = np.column_stack([xi.ravel(), yi.ravel()])

    # ── 1) Combined risk level per cell (worst band across gases) ──
    risk = np.zeros(xi.shape)          # 0 safe, 1 elevated, 2 hazard
    worst_gas = np.full(xi.shape, '', dtype=object)
    for g in gases:
        x, y, p = (z[f'{g}_Sensor_reading_x'], z[f'{g}_Sensor_reading_y'],
                   z[f'{g}_Sensor_reading_ppm'])
        if (p > 0.1).sum() < 5:
            continue
        zi = griddata((x, y), p, (xi, yi), method='linear', fill_value=0.0)
        zi = np.clip(gaussian_filter(zi, 1.5), 0, None)
        elev, haz = GAS_BANDS.get(g, DEFAULT_BANDS)
        lvl = np.where(zi >= haz, 2, np.where(zi >= elev, 1, 0))
        upd = lvl > risk
        risk[upd] = lvl[upd]; worst_gas[upd] = g

    # ── 2) Coverage confidence: distance to nearest actual sample ──
    from scipy.spatial import cKDTree
    tree = cKDTree(np.column_stack([allx, ally]))
    dist, _ = tree.query(gridpts)
    unvisited = (dist.reshape(xi.shape) > 0.6)   # never got within 0.6 m

    # ── figure ──
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(20, 8.5),
                                   gridspec_kw={'width_ratios': [1.25, 1]})

    # LEFT: risk map + coverage + wind
    risk_cmap = ListedColormap(['#2ecc71', '#f1c40f', '#e74c3c'])  # green/amber/red
    rm = np.ma.masked_where(unvisited, risk)     # don't color unknown areas
    axL.pcolormesh(xi, yi, rm, cmap=risk_cmap, norm=BoundaryNorm([-.5,.5,1.5,2.5],3),
                   shading='auto', zorder=1)
    # unvisited = hatched grey "UNKNOWN"
    axL.contourf(xi, yi, unvisited.astype(float), levels=[0.5, 1.5],
                 colors='none', hatches=['xxx'], zorder=2)
    axL.contourf(xi, yi, unvisited.astype(float), levels=[0.5, 1.5],
                 colors=[(0.5, 0.5, 0.5, 0.25)], zorder=2)
    if occ_rgba is not None:
        axL.imshow(occ_rgba, extent=occ_ext, origin='lower',
                   interpolation='nearest', zorder=5)

    # wind arrow + downwind spread cone from worst hotspot
    dw = z['wind_downwind_xy'] if 'wind_downwind_xy' in z.files else None
    haz_cells = np.argwhere(risk >= 2)
    if dw is not None and len(haz_cells):
        # worst hotspot = highest-risk cell nearest the densest hazard cluster
        cy, cx = haz_cells[len(haz_cells)//2]
        hx, hy = xi[cy, cx], yi[cy, cx]
        d = np.array(dw) / (np.linalg.norm(dw) + 1e-9)
        ang = math.atan2(d[1], d[0]); spread = math.radians(35); L = 3.5
        p1 = (hx + L*math.cos(ang-spread), hy + L*math.sin(ang-spread))
        p2 = (hx + L*math.cos(ang+spread), hy + L*math.sin(ang+spread))
        axL.add_patch(Polygon([(hx, hy), p1, p2], closed=True,
                      facecolor='#e74c3c', alpha=0.18, edgecolor='#e74c3c',
                      linestyle='--', zorder=4, label='downwind spread'))
        # wind arrow (corner)
        ax0x = xg[0] + 0.8*(xg[-1]-xg[0]); ax0y = yg[0] + 0.92*(yg[-1]-yg[0])
        axL.annotate('', xy=(ax0x + d[0], ax0y + d[1]), xytext=(ax0x, ax0y),
                     arrowprops=dict(arrowstyle='-|>', color='#2c3e50', lw=2.5), zorder=7)
        axL.text(ax0x, ax0y+0.3, 'wind', color='#2c3e50', fontsize=10, ha='center', zorder=7)

    from matplotlib.patches import Patch
    legend = [Patch(facecolor='#2ecc71', label='Safe'),
              Patch(facecolor='#f1c40f', label='Elevated'),
              Patch(facecolor='#e74c3c', label='HAZARD'),
              Patch(facecolor=(0.5,0.5,0.5,0.4), hatch='xxx', label='Unknown (not surveyed)')]
    cov = float(z['coverage_pct']) if 'coverage_pct' in z.files else float('nan')
    axL.legend(handles=legend, loc='upper left', fontsize=9, framealpha=0.9)
    axL.set_title(f'Gas Risk Map  —  coverage {cov:.0f}%  (worst gas per cell)',
                  fontsize=13, fontweight='bold')
    axL.set_xlabel('X (m)'); axL.set_ylabel('Y (m)'); axL.set_aspect('equal')

    # RIGHT: per-gas situation report (trend + first detection + extent)
    axR.axis('off')
    rows = [['Gas', 'Peak ppm', 'Area\n(elev+)', 'First seen', 'Trend (1st→2nd half)']]
    for g in gases:
        p = z[f'{g}_Sensor_reading_ppm']; t = z[f'{g}_Sensor_reading_timestamp']
        if (p > 0.1).sum() < 5:
            continue
        t0 = t.min()
        elev = GAS_BANDS.get(g, DEFAULT_BANDS)[0]
        # area = fraction of visited grid above 'elevated' for this gas
        x, y = z[f'{g}_Sensor_reading_x'], z[f'{g}_Sensor_reading_y']
        zi = griddata((x, y), p, (xi, yi), method='linear', fill_value=0.0)
        area_m2 = float((~unvisited & (gaussian_filter(zi,1.5) >= elev)).sum()) * res*res
        # first detection above elevated
        above = np.where(p >= elev)[0]
        first = f'{t[above[0]]-t0:.0f}s' if len(above) else 'never'
        # trend: mean ppm first half vs second half (of detected samples)
        mid = t0 + (t.max()-t0)/2
        h1 = p[(t < mid) & (p > 0.1)]; h2 = p[(t >= mid) & (p > 0.1)]
        if len(h1) > 3 and len(h2) > 3:
            r = (h2.mean()+1e-6)/(h1.mean()+1e-6)
            trend = '↑ RISING' if r > 1.25 else ('↓ falling' if r < 0.8 else '→ stable')
            trend += f' ({h1.mean():.1f}→{h2.mean():.1f})'
        else:
            trend = 'n/a'
        rows.append([g, f'{p.max():.1f}', f'{area_m2:.1f} m²', first, trend])
    tbl = axR.table(cellText=rows[1:], colLabels=rows[0], loc='center', cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1, 2.4)
    for c in range(len(rows[0])):
        tbl[0, c].set_facecolor('#34495e'); tbl[0, c].set_text_props(color='white', fontweight='bold')
    axR.set_title('Situation Report', fontsize=13, fontweight='bold')

    fig.suptitle('Gas Surveillance — Situational Awareness', fontsize=16, fontweight='bold')
    fig.tight_layout()
    out = os.path.join(run_dir, 'risk_exposure_map.png')
    fig.savefig(out, dpi=200); plt.close(fig)
    print(f'Saved {out}')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1])
