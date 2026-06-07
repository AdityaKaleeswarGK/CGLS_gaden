#!/usr/bin/env python3
"""
gas_viz.py — Shared, ROS-free gas-mapping analysis & visualization.

Everything here is importable WITHOUT rclpy/ROS, so the same code runs live
(inside ``auto_coverage_mapper`` at mission end) and offline (``plot_from_npz``,
``risk_map_prototype`` and the synthetic-run harness). Only ``numpy`` is needed
at import time; ``matplotlib`` / ``scipy`` are imported lazily inside the
plotting helpers so :func:`estimate_source` stays maximally portable.

Three responsibilities:

  * :func:`estimate_source` — physically-grounded probabilistic source
    localization. Returns a true ``P(source | readings)`` posterior REGION via a
    ``1/r`` Bayesian inversion with analytic release-rate marginalization — not a
    measurement-density blob. Keeps the exact return-dict keys the rest of the
    pipeline already consumes, and adds 50% / 90% HPD areas.

  * :func:`concentration_intensity_map` — rich per-gas concentration heatmap with
    data-relative intensity bands (trace -> PEAK), a highlighted peak region,
    coverage hatching and a wall overlay. The "good middle ground" between a
    binary hazard map and the raw research plots.

  * :func:`source_posterior_map` — renders the posterior with 50% / 90% credible
    (HPD) regions: "look here", a region rather than a false pinpoint.
"""
import math
import numpy as np


# ════════════════════════════════════════════════════════════════════
#  Probabilistic source localization (1/r Bayesian inversion)
# ════════════════════════════════════════════════════════════════════
def baseline_sigma_estimate(ppm, baseline_sigma=None, floor=0.05):
    """Noise std for the likelihood. Prefer the CUSUM-learned pre-gas baseline
    sigma; otherwise fall back to the spread of the lowest-40% readings (a
    robust baseline proxy that is not dominated by plume samples)."""
    if baseline_sigma is not None and np.isfinite(baseline_sigma) and baseline_sigma > 0:
        return float(max(baseline_sigma, floor))
    ppm = np.asarray(ppm, float)
    if ppm.size:
        lo = ppm[ppm <= np.percentile(ppm, 40)]
        if lo.size >= 5:
            return float(max(np.std(lo), floor))
        return float(max(np.std(ppm) * 0.3, floor))
    return floor


def top_candidates(posterior, gx, gy, n=3, min_sep=0.8):
    """Up to N well-separated posterior maxima: [(x, y, prob), ...] high->low."""
    order = np.argsort(posterior.ravel())[::-1]
    cands = []
    for idx in order:
        r, c = np.unravel_index(idx, posterior.shape)
        x, y, p = float(gx[r, c]), float(gy[r, c]), float(posterior[r, c])
        if all(math.hypot(x - cx, y - cy) > min_sep for cx, cy, _ in cands):
            cands.append((x, y, p))
        if len(cands) >= n:
            break
    return cands


def estimate_source(xs, ys, ppm, timestamps=None, grid_res=0.10, wind=None,
                    wind_consistency=0.0, use_wind_shift=False,
                    baseline_sigma=None, eps=0.15, max_meas=1500,
                    rel_noise=0.0, eff_obs=20.0):
    """Probabilistic gas-source-REGION localization.

    The posterior is a genuine ``P(source = s | readings)`` map built from an
    isotropic steady-state diffusion forward model ``C(s, p) = A / (||s-p|| +
    eps)``. For every candidate cell ``s`` the release rate ``A`` is marginalized
    analytically (non-negative least squares), and the Gaussian residual
    likelihood is evaluated against ALL readings — including near-zero ones,
    whose absence of gas constrains where the source can be. This peaks at the
    cell that best explains the observed field, not merely where the robot
    happened to measure the most gas.

    When a reliable measured wind is supplied (``use_wind_shift`` and
    ``wind_consistency >= 0.4``), the forward model advects: gas released at
    ``s`` is transported downwind before measurement, which moves posterior mass
    upwind toward the true source.

    A ppm^2-weighted PCA still recovers the plume axis / spread for the drawn
    ellipse and (when confident) a directional arrow.

    Returns a dict with (at least) the keys the rest of the pipeline consumes:
        posterior, gx, gy, best_x, best_y, src_x, src_y, conf_radius, axis,
        directional, method, low_confidence, peak_ppm, sigma_major, sigma_minor,
        candidates, grid_res, centroid, hpd50, hpd90
    """
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    ppm = np.asarray(ppm, float)

    margin = 1.0
    cx = np.arange(xs.min() - margin, xs.max() + margin, grid_res)
    cy = np.arange(ys.min() - margin, ys.max() + margin, grid_res)
    gx, gy = np.meshgrid(cx, cy)

    pos_mask = ppm > 0.1
    if pos_mask.sum() < 5:
        # Not enough signal — uniform posterior, fall back to mean position.
        posterior = np.ones(gx.shape) / gx.size
        mx, my = float(np.mean(xs)), float(np.mean(ys))
        return {
            'posterior': posterior, 'gx': gx, 'gy': gy,
            'best_x': mx, 'best_y': my, 'src_x': mx, 'src_y': my,
            'conf_radius': 999.0, 'axis': (1.0, 0.0), 'directional': False,
            'method': 'insufficient-signal',
            'low_confidence': True,
            'peak_ppm': float(ppm.max()) if ppm.size else 0.0,
            'sigma_major': 0.0, 'sigma_minor': 0.0, 'centroid': (mx, my),
            'candidates': [(mx, my, 1.0)], 'grid_res': grid_res,
            'hpd50': 0.0, 'hpd90': 0.0,
        }

    xp, yp, pp = xs[pos_mask], ys[pos_mask], ppm[pos_mask]

    # ── ppm²-weighted centroid + covariance (PCA → plume axis, for drawing) ──
    w = pp.astype(np.float64) ** 2
    w /= w.sum()
    mu_x = float(np.sum(w * xp))
    mu_y = float(np.sum(w * yp))
    dx = xp - mu_x
    dy = yp - mu_y
    cxx = float(np.sum(w * dx * dx))
    cyy = float(np.sum(w * dy * dy))
    cxy = float(np.sum(w * dx * dy))
    evals, evecs = np.linalg.eigh(np.array([[cxx, cxy], [cxy, cyy]]))
    e_major = evecs[:, 1]
    sigma_minor = float(np.sqrt(max(evals[0], 1e-4)))
    sigma_major = float(np.sqrt(max(evals[1], 1e-4)))

    # Which end of the axis is the source? Skew of the along-axis projection
    # (heavy low-ppm tail sits downwind) + spatial-extent asymmetry. Only trust
    # a direction when both cues agree AND the plume is clearly elongated.
    proj = dx * e_major[0] + dy * e_major[1]
    m3 = float(np.sum(w * proj ** 3))
    sign_skew = -np.sign(m3) if m3 != 0 else 0.0
    pp_plus, pp_minus = proj[proj > 0], proj[proj < 0]
    ext_plus = np.percentile(pp_plus, 90) if pp_plus.size else 0.0
    ext_minus = np.percentile(-pp_minus, 90) if pp_minus.size else 0.0
    sign_ext = -np.sign(ext_plus - ext_minus)
    elongated = sigma_major > 1.3 * sigma_minor
    peak_ppm = float(pp.max())
    strong_signal = peak_ppm >= 3.0 and len(pp) >= 30
    low_confidence = peak_ppm < 2.0

    use_wind = (use_wind_shift and wind is not None
                and wind_consistency >= 0.4 and not low_confidence)
    if use_wind:
        u = np.array(wind, dtype=float)
        u /= max(np.linalg.norm(u), 1e-9)      # measured upwind direction → source
        shift_confident = True
        method = 'wind-bayes'
    elif sign_skew != 0 and sign_skew == sign_ext and elongated and strong_signal:
        u = e_major * sign_skew                 # inferred upwind direction → source
        shift_confident = True
        method = 'bayes-1/r'
    else:
        u = e_major                             # orientation only, no sign
        shift_confident = False
        method = 'bayes-1/r'

    # ── 1/r Bayesian source-likelihood posterior over the grid ──
    # Spatially bin readings to one observation per ~bin_res cell. The robot
    # dwells unevenly along its path; without this, oversampled locations would
    # dominate the likelihood and collapse the posterior to a false pinpoint.
    # Binning gives each surveyed location equal weight (a genuine spatial
    # observation) and keeps the inversion cheap.
    bin_res = 0.20
    bx = np.round((xs - xs.min()) / bin_res).astype(np.int64)
    by = np.round((ys - ys.min()) / bin_res).astype(np.int64)
    key = bx * 1_000_003 + by
    _, inv = np.unique(key, return_inverse=True)
    cnt = np.bincount(inv).astype(np.float64)
    sx_all = np.bincount(inv, weights=xs) / cnt
    sy_all = np.bincount(inv, weights=ys) / cnt
    c_all = np.bincount(inv, weights=ppm) / cnt
    N = sx_all.size
    if N > max_meas:
        # Stratified subsample: always keep the most informative (high-ppm)
        # cells, fill the remainder at random (reproducibly).
        order = np.argsort(c_all)[::-1]
        keep = order[:max_meas // 2]
        rest = order[max_meas // 2:]
        rng = np.random.default_rng(0)
        keep = np.concatenate(
            [keep, rng.choice(rest, max_meas - keep.size, replace=False)])
        sx_all, sy_all, c_all = sx_all[keep], sy_all[keep], c_all[keep]

    sigma_n = baseline_sigma_estimate(ppm, baseline_sigma)
    # Homoscedastic by default (rel_noise=0): a single noise scale keeps the MAP
    # driven by the informative high-ppm points. rel_noise>0 optionally adds a
    # concentration-proportional term for model mismatch, but must be small —
    # too much suppresses the signal and the estimate drifts to empty space.
    sigma_i = sigma_n + rel_noise * c_all
    inv_var = 1.0 / (sigma_i * sigma_i)

    gfx = gx.ravel()
    gfy = gy.ravel()
    px = sx_all[None, :]
    py = sy_all[None, :]
    c_obs = c_all[None, :]
    # Advection offset: gas released at s is carried downwind (−u) by tau before
    # being measured, so evaluate the model at s − u*tau. Only when wind-trusted.
    tau = (0.8 * sigma_major) if use_wind else 0.0

    chi2 = np.empty(gfx.size, dtype=np.float64)
    chunk = 400
    for i0 in range(0, gfx.size, chunk):
        i1 = min(i0 + chunk, gfx.size)
        sxc = gfx[i0:i1, None] - u[0] * tau
        syc = gfy[i0:i1, None] - u[1] * tau
        r = np.sqrt((px - sxc) ** 2 + (py - syc) ** 2) + eps
        f = 1.0 / r
        ff = np.sum(f * f, axis=1)
        fc = np.sum(f * c_obs, axis=1)
        alpha = np.clip(fc / np.maximum(ff, 1e-12), 0.0, None)   # release rate ≥ 0
        resid = c_obs - alpha[:, None] * f
        chi2[i0:i1] = np.sum(resid * resid * inv_var[None, :], axis=1)

    # Empirical-Bayes noise: scale the likelihood by the best-fit residual level
    # (chi2_min) instead of an absolute sensor sigma. This makes the posterior
    # shape depend only on the RELATIVE goodness-of-fit across candidates and on
    # eff_obs — scale-invariant to signal amplitude, sample count and baseline
    # noise. When the 1/r model fits poorly (real plume mismatch) chi2_min is
    # large and the region honestly widens; eff_obs sets the credible-region
    # size (≈ an effective number of independent observations).
    chi2_min = max(float(chi2.min()), 1e-12)
    logL = -0.5 * eff_obs * (chi2 / chi2_min)
    logL -= logL.max()
    post = np.exp(logL)
    psum = post.sum()
    posterior = ((post / psum).reshape(gx.shape) if psum > 0
                 else np.ones_like(gx, dtype=float) / gx.size)

    pr, pc = np.unravel_index(np.argmax(posterior), posterior.shape)
    best_x, best_y = float(gx[pr, pc]), float(gy[pr, pc])
    # The posterior MAP is now a physically meaningful source estimate.
    src_x, src_y = best_x, best_y

    # HPD region areas (smallest area holding 50% / 90% of posterior mass).
    flat = posterior.ravel()
    order = np.argsort(flat)[::-1]
    csum = np.cumsum(flat[order])
    cell_area = grid_res * grid_res

    def _hpd_area(frac):
        k = int(np.searchsorted(csum, frac))
        return float((k + 1) * cell_area)

    cands = top_candidates(posterior, gx, gy, n=3)

    return {
        'posterior': posterior, 'gx': gx, 'gy': gy,
        'best_x': best_x, 'best_y': best_y,
        'src_x': src_x, 'src_y': src_y,
        'conf_radius': float(np.sqrt(max(sigma_major * sigma_minor, 1e-4))),
        'axis': (float(u[0]), float(u[1])),
        'directional': bool(shift_confident), 'method': method,
        'low_confidence': bool(low_confidence), 'peak_ppm': peak_ppm,
        'sigma_major': sigma_major, 'sigma_minor': sigma_minor,
        'centroid': (mu_x, mu_y),
        'candidates': cands, 'grid_res': grid_res,
        'hpd50': _hpd_area(0.50), 'hpd90': _hpd_area(0.90),
    }


# ════════════════════════════════════════════════════════════════════
#  Shared grid / coverage / wall helpers (lazy scipy / matplotlib)
# ════════════════════════════════════════════════════════════════════
def interp_grid(xs, ys, ppm, margin=0.3, res=0.04):
    """Interpolate scattered (x, y, ppm) onto a regular grid (cubic, nearest
    fill, clipped ≥ 0). Returns (xi, yi, zi)."""
    from scipy.interpolate import griddata
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    ppm = np.asarray(ppm, float)
    xg = np.arange(xs.min() - margin, xs.max() + margin, res)
    yg = np.arange(ys.min() - margin, ys.max() + margin, res)
    xi, yi = np.meshgrid(xg, yg)
    zi = griddata((xs, ys), ppm, (xi, yi), method='cubic')
    zin = griddata((xs, ys), ppm, (xi, yi), method='nearest')
    nan = np.isnan(zi)
    zi[nan] = zin[nan]
    return xi, yi, np.clip(zi, 0, None)


def coverage_mask(xs, ys, xi, yi, radius=0.6):
    """Boolean grid mask: True where no sample came within ``radius`` metres
    (i.e. unsurveyed — 'unknown', not 'clean')."""
    from scipy.spatial import cKDTree
    tree = cKDTree(np.column_stack([np.asarray(xs, float), np.asarray(ys, float)]))
    d, _ = tree.query(np.column_stack([xi.ravel(), yi.ravel()]))
    return (d.reshape(xi.shape) > radius)


def load_walls(run_dir):
    """Return (rgba, extent) for a solid-black wall overlay from a run's
    occupancy_grid.pgm/.yaml, or (None, None) if unavailable."""
    import os
    pgm = os.path.join(run_dir, 'occupancy_grid.pgm')
    yml = os.path.join(run_dir, 'occupancy_grid.yaml')
    try:
        import yaml
    except ImportError:
        return None, None
    if not (os.path.isfile(pgm) and os.path.isfile(yml)):
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
        data = np.flipud(data)  # undo the save-time flip → map frame
        rgba = np.zeros((h, w, 4), dtype=float)
        rgba[data <= 50] = [0, 0, 0, 1.0]   # occupied → solid black
        extent = [ox, ox + w * res, oy, oy + h * res]
        return rgba, extent
    except Exception as e:
        print(f"  (wall overlay skipped: {e})")
        return None, None


def _scalebar(ax):
    """Add a 1 m scale bar if matplotlib-scalebar is available (else no-op)."""
    try:
        from matplotlib_scalebar.scalebar import ScaleBar
        ax.add_artist(ScaleBar(1.0, 'm', location='lower left',
                               length_fraction=0.15, box_alpha=0.7,
                               font_properties={'size': 10}))
    except Exception:
        pass


def _draw_true_sources(ax, true_sources, label=True):
    for s in (true_sources or []):
        ax.plot(s['x'], s['y'], 'P', markersize=13, color='#ff1493',
                markeredgecolor='white', markeredgewidth=0.9, zorder=8)
    if true_sources and label:
        ax.plot([], [], 'P', color='#ff1493', markeredgecolor='white',
                label='True source')


# ════════════════════════════════════════════════════════════════════
#  Deliverable #1 — per-gas concentration INTENSITY map (middle ground)
# ════════════════════════════════════════════════════════════════════
def concentration_intensity_map(out_path, label, xs, ys, ppm, baseline_mu=0.0,
                                baseline_sigma=None, walls=(None, None),
                                true_sources=None, coverage_radius=0.6,
                                title=None):
    """Rich per-gas concentration heatmap that shows WHERE the gas peaks.

    A continuous peak-emphasizing heatmap (turbo, gamma-compressed) + labelled
    data-relative iso-bands (trace / low / moderate / high / PEAK) + a
    highlighted PEAK contour, with unsurveyed area hatched and walls overlaid.
    Returns a small stats dict.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import PowerNorm
    from scipy.ndimage import gaussian_filter

    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    ppm = np.asarray(ppm, float)
    occ_rgba, occ_extent = walls if walls else (None, None)

    xi, yi, zi = interp_grid(xs, ys, ppm)
    zi = gaussian_filter(zi, 1.5)
    unvis = coverage_mask(xs, ys, xi, yi, coverage_radius)
    res = float(xi[0, 1] - xi[0, 0])

    peak = max(float(np.nanmax(zi)), 1e-6)
    sig = (baseline_sigma if (baseline_sigma and baseline_sigma > 0)
           else max(float(np.std(ppm)) * 0.3, 0.05))
    trace = float(baseline_mu) + 3.0 * sig          # detectable above noise

    surveyed = zi[~unvis]
    pos = surveyed[surveyed > trace]
    if pos.size > 20:
        lo, mid, hi = np.percentile(pos, [40, 70, 90])
    else:
        lo, mid, hi = trace * 2, peak * 0.4, peak * 0.7
    edges = np.maximum.accumulate(
        np.array([0.0, trace, lo, mid, hi, peak * 1.001]))
    band_labels = ['trace', 'low', 'moderate', 'high', 'PEAK']

    fig, ax = plt.subplots(figsize=(11, 7.5))

    # Continuous heatmap, peak-emphasizing. Surveyed-but-clean cells fade to a
    # faint floor so peak regions stand out; unsurveyed cells go transparent
    # (then hatched). This is what gives "where does it peak" at a glance.
    cmap = plt.get_cmap('turbo')
    norm = PowerNorm(gamma=0.45, vmin=0, vmax=peak)
    alpha_cov = np.clip(gaussian_filter(np.where(unvis, 0.0, 1.0), 1.5), 0, 1)
    conc_vis = np.clip((zi - trace) / max(edges[3] - trace, 1e-6), 0, 1)
    alpha = alpha_cov * (0.12 + 0.88 * conc_vis)
    rgba = cmap(norm(zi))
    rgba[..., 3] = alpha
    extent = [xi[0, 0], xi[0, -1], yi[0, 0], yi[-1, 0]]
    ax.imshow(rgba, extent=extent, origin='lower', interpolation='bilinear',
              zorder=1, aspect='auto')

    # Thin iso-concentration lines for low / moderate / high (no inline labels —
    # the quantitative band scale lives on the colourbar to avoid clutter).
    # contour() requires STRICTLY increasing levels; np.maximum.accumulate can
    # leave equal adjacent edges when the gas field is flat/sparse, so dedupe.
    band_levels = np.unique([e for e in edges[2:-1] if e > 0])
    if band_levels.size:
        ax.contour(xi, yi, zi, levels=band_levels, colors='#333333',
                   linewidths=0.5, alpha=0.4, zorder=3)
    # Highlighted PEAK region (top band) — "more gas here".
    if edges[-2] > edges[1]:
        ax.contour(xi, yi, zi, levels=[edges[-2]], colors=['#ff2d2d'],
                   linewidths=2.2, zorder=4)
        ax.plot([], [], color='#ff2d2d', lw=2.2, label='peak region')

    # Unsurveyed → hatched grey "unknown".
    try:
        ax.contourf(xi, yi, unvis.astype(float), levels=[0.5, 1.5],
                    colors='none', hatches=['xxx'], zorder=2)
        ax.contourf(xi, yi, unvis.astype(float), levels=[0.5, 1.5],
                    colors=[(0.5, 0.5, 0.5, 0.22)], zorder=2)
    except Exception:
        pass

    if occ_rgba is not None:
        ax.imshow(occ_rgba, extent=occ_extent, origin='lower',
                  interpolation='nearest', zorder=5)

    # Peak marker + ppm²-centroid ("center of mass of gas").
    masked = np.where(unvis, -np.inf, zi)
    pr, pc = np.unravel_index(np.argmax(masked), zi.shape)
    peak_x, peak_y = float(xi[pr, pc]), float(yi[pr, pc])
    ax.plot(peak_x, peak_y, '*', markersize=17, color='white',
            markeredgecolor='black', markeredgewidth=0.8, zorder=7,
            label=f'peak {peak:.2g} ppm')
    w2 = np.clip(ppm, 0, None) ** 2
    if w2.sum() > 0:
        cmx = float(np.sum(w2 * xs) / w2.sum())
        cmy = float(np.sum(w2 * ys) / w2.sum())
        ax.plot(cmx, cmy, 'o', markersize=11, markerfacecolor='none',
                markeredgecolor='white', markeredgewidth=1.6, zorder=6,
                label='concentration centroid')

    _draw_true_sources(ax, true_sources)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(sm, ax=ax, shrink=0.85, label='Concentration (ppm)')
    # The colourbar carries the quantitative intensity-band scale.
    tick_e, tick_l = [], []
    for e, lb in zip(edges[1:], band_labels):
        if e > 0:
            tick_e.append(float(e))
            tick_l.append(f'{lb}  {e:.2g}')
    if tick_e:
        cbar.set_ticks(tick_e)
        cbar.set_ticklabels(tick_l, fontsize=7)

    area_mod = float((~unvis & (zi >= edges[3])).sum()) * res * res
    pct_seen = 100.0 * float((~unvis).sum()) / unvis.size
    info = (f"peak: {peak:.2g} ppm\n"
            f"area ≥ moderate: {area_mod:.1f} m²\n"
            f"surveyed: {pct_seen:.0f}% of frame")
    ax.text(0.02, 0.98, info, transform=ax.transAxes, fontsize=9,
            va='top', family='monospace', zorder=8,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='gray', alpha=0.85))

    _scalebar(ax)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(title or f'{label} — Concentration Intensity (where the gas peaks)')
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return {'peak_ppm': peak, 'area_moderate_m2': area_mod,
            'pct_surveyed': pct_seen, 'band_edges': edges.tolist()}


# ════════════════════════════════════════════════════════════════════
#  Deliverable #2 — per-gas probabilistic source map (50% / 90% HPD)
# ════════════════════════════════════════════════════════════════════
def source_posterior_map(out_path, label, xs, ys, ppm, r, walls=(None, None),
                         true_sources=None, entry=None):
    """Render a single gas's source posterior as a probability REGION.

    ``r`` is the dict returned by :func:`estimate_source`. ``entry`` (optional)
    may carry benchmark fields ``error_m`` / ``true_x`` / ``true_y`` to annotate
    localization error.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import PowerNorm
    from matplotlib.patches import Ellipse
    from scipy.ndimage import gaussian_filter

    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    ppm = np.asarray(ppm, float)
    occ_rgba, occ_extent = walls if walls else (None, None)

    post = r['posterior']
    gx, gy = r['gx'], r['gy']
    src_x, src_y = r['src_x'], r['src_y']
    smaj, smin = r['sigma_major'], r['sigma_minor']
    ux, uy = r['axis']

    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.get_cmap('viridis')
    post_smooth = gaussian_filter(post, sigma=2.0)
    norm = PowerNorm(gamma=0.4, vmin=0, vmax=max(post.max(), 1e-10))
    rgba = cmap(norm(post_smooth))
    extent = [gx[0, 0], gx[0, -1], gy[0, 0], gy[-1, 0]]
    ax.imshow(rgba, extent=extent, origin='lower', interpolation='bilinear',
              zorder=1, aspect='auto')
    if occ_rgba is not None:
        ax.imshow(occ_rgba, extent=occ_extent, origin='lower',
                  interpolation='nearest', zorder=5)
    plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax,
                 label='P(source)', shrink=0.82)

    # 50% / 90% highest-posterior-density credible regions.
    flat = post.ravel()
    order = np.argsort(flat)[::-1]
    csum = np.cumsum(flat[order])
    csum = csum / csum[-1]

    def _thr(frac):
        k = int(np.searchsorted(csum, frac))
        return flat[order[min(k, len(order) - 1)]]

    for frac, col, lw in ((0.90, '#ffd166', 1.6), (0.50, '#ff3b3b', 2.2)):
        try:
            ax.contour(gx, gy, post, levels=[_thr(frac)], colors=[col],
                       linewidths=lw, zorder=6)
            ax.plot([], [], color=col, lw=lw, label=f'{int(frac*100)}% region')
        except Exception:
            pass

    # Plume axis: directional arrow if confident, else an undirected line.
    ang = math.degrees(math.atan2(uy, ux))
    ax.add_patch(Ellipse((src_x, src_y), width=2 * smaj, height=2 * smin,
                         angle=ang, fill=False, edgecolor='white',
                         linestyle='--', linewidth=1.4, alpha=0.85, zorder=6))
    alen = max(smaj, 0.5)
    if r.get('directional'):
        ax.annotate('', xy=(src_x + ux * alen, src_y + uy * alen),
                    xytext=(src_x - ux * alen, src_y - uy * alen),
                    arrowprops=dict(arrowstyle='-|>', color='#00e5ff',
                                    lw=2.0, alpha=0.9), zorder=6)
    else:
        ax.plot([src_x - ux * alen, src_x + ux * alen],
                [src_y - uy * alen, src_y + uy * alen],
                color='#00e5ff', lw=1.8, alpha=0.8, zorder=6,
                label='plume axis (direction unknown)')

    ax.plot(src_x, src_y, 'r*', markersize=18, markeredgecolor='white',
            markeredgewidth=1.0, zorder=7,
            label=f'best guess ({src_x:.2f}, {src_y:.2f})')
    for rank, (cxv, cyv, _) in enumerate(r.get('candidates', [])[:3], 1):
        ax.plot(cxv, cyv, 'o', markersize=9, markerfacecolor='none',
                markeredgecolor='yellow', markeredgewidth=1.5, zorder=6)
        ax.text(cxv, cyv, f' #{rank}', color='yellow', fontsize=9,
                fontweight='bold', zorder=7)
    if ppm.size:
        mi = int(np.argmax(ppm))
        ax.plot(xs[mi], ys[mi], 'w^', markersize=10, markeredgecolor='black',
                markeredgewidth=0.5, label='max reading', zorder=6)

    _draw_true_sources(ax, true_sources)
    err_txt = ''
    if entry and 'error_m' in entry:
        ax.plot([src_x, entry['true_x']], [src_y, entry['true_y']],
                color='#ff1493', linestyle=':', linewidth=1.3, alpha=0.9, zorder=7)
        err_txt = f"  |  error = {entry['error_m']:.2f} m"

    _scalebar(ax)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    meth = r.get('method', 'bayes-1/r')
    title = f'{label} — Source Region ({meth}): P(source) + 50/90% credible' + err_txt
    if r.get('low_confidence'):
        title += (f"\n[!] LOW CONFIDENCE — peak only {r.get('peak_ppm', 0):.2f} ppm "
                  f"(gas barely detected)")
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=8)
    ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
