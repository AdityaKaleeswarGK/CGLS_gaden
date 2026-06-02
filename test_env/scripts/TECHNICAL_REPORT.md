# Technical Report: Auto Coverage & Values Capturing with GADEN

**Package:** `test_env/scripts`
**Files covered:** `auto_coverage_mapper.py`, `concentration_mapper.py`
**Simulation backend:** GADEN filament-based gas dispersion (ROS 2 Humble)
**Date:** 2026-04-06

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [GADEN Gas Dispersion — What the Mappers Are Reading](#2-gaden-gas-dispersion--what-the-mappers-are-reading)
3. [Values Capturing — `concentration_mapper.py`](#3-values-capturing--concentration_mapperpy)
4. [Auto Coverage Mapper — `auto_coverage_mapper.py`](#4-auto-coverage-mapper--auto_coverage_mapperpy)
   - 4.1 [Phase 1: Coarse BCD Lawnmower Sweep](#41-phase-1-coarse-bcd-lawnmower-sweep)
   - 4.2 [Online CUSUM Change-Point Detector](#42-online-cusum-change-point-detector)
   - 4.3 [Hotspot Validation & Merging](#43-hotspot-validation--merging)
   - 4.4 [Phase 2: Fine Refinement](#44-phase-2-fine-refinement)
   - 4.5 [Reactive Velocity Reduction](#45-reactive-velocity-reduction)
   - 4.6 [Live Concentration Grid](#46-live-concentration-grid)
5. [Bayesian Source Localization](#5-bayesian-source-localization)
6. [Data Persistence & Outputs](#6-data-persistence--outputs)
7. [ROS Integration](#7-ros-integration)
8. [Configuration Parameters](#8-configuration-parameters)
9. [Assessment & Known Limitations](#9-assessment--known-limitations)

---

## 1. System Overview

Two gas-mapping pipelines run on top of the GADEN simulation stack:

| Node | Script | Mode | Purpose |
|------|--------|------|---------|
| `auto_coverage_mapper` | `auto_coverage_mapper.py` | Active — controls robot | Two-phase adaptive sweep with online spike detection |
| `concentration_mapper` | `concentration_mapper.py` | Passive — records only | Log concentration during any externally driven mission |

Both sit downstream of `gaden_player`, which replays pre-simulated filament snapshots and exposes them as ROS services. The simulated sensor node (`fake_gas_sensor`) queries those services and publishes `olfaction_msgs/GasSensor` messages, which both mappers consume.

```
gaden_player  ──/odor_value srv──►  fake_gas_sensor  ──/fake_pid/Sensor_reading──►  mappers
              ──/wind_value srv──►  simulated_anemometer
```

---

## 2. GADEN Gas Dispersion — What the Mappers Are Reading

### 2.1 Filament Dispersion Model

GADEN represents the gas cloud as a set of discrete *filaments*, each carrying a fixed molecular quantity Q. Their state evolves under advection and turbulent diffusion:

**Advection (Euler step with turbulent noise):**

```
r_f(t + dt) = r_f(t) + u(r_f, t) * dt + xi
```

where `u` is the interpolated wind field and `xi ~ N(0, sigma_noise^2)` with `sigma_noise = 0.1 m`.

**Filament width growth:**

```
sigma_f(t) = sigma_0 + gamma * sqrt(t)
```

Default values: `sigma_0 = 1.5 m`, `gamma = 10.0`.

**Concentration at sensor position p (superposition of 3D Gaussians):**

```
C(p) = sum_f  Q / (sigma_f * sqrt(2*pi))^3  *  exp( -||p - r_f||^2 / (2 * sigma_f^2) )
```

This is the field that the mappers attempt to reconstruct from sparse point measurements.

### 2.2 Simulated Sensor Transducer

`fake_gas_sensor` applies a nonlinear MOX transfer function before publishing:

```
RS/R0 = A * [ppm]^B        (sensor-specific A, B from datasheet)
```

Example: TGS2620 + Ethanol → `A = 62.32`, `B = -0.7155`.

A first-order temporal lag is also applied:

```
y(t) = y(t-1) + (dt / tau) * (y_inf - y(t-1))
```

where `tau` is the rise/decay time constant (e.g. TGS2600 ethanol: `tau_rise = 4.8 s`, `tau_decay = 18.75 s`). The PID model skips the power-law and applies only a chemical correction factor. Both models introduce realistic lag that affects where a gas encounter is spatially attributed.

---

## 3. Values Capturing — `concentration_mapper.py`

This node is entirely passive. It does not issue navigation commands; it only listens.

### 3.1 Data Recording

At every `GasSensor` message callback:
1. Look up robot pose via TF: `map → base_link` (or configured `base_frame`).
2. Append `(x, y, ppm)` to an in-memory list per sensor topic.
3. TF lookup errors are caught and logged once only.

```python
# concentration_mapper.py lines 53-64
stamp = rclpy.time.Time()
trans = self.tf_buffer.lookup_transform('map', self.base_frame, stamp)
x = trans.transform.translation.x
y = trans.transform.translation.y
self.data_history[topic].append((x, y, msg.raw))
```

Data accumulates in RAM until the node is stopped (`Ctrl+C` or `SIGINT`), at which point all output is generated.

### 3.2 Visualization (7 Plots)

All figures saved to `~/gaden_results/concentration_maps/`:

| # | Plot | Method |
|---|------|--------|
| 1 | Scatter (raw readings) | `matplotlib.scatter`, coloured by ppm |
| 2 | Interpolated heatmap | Cubic spline + nearest-fill (see §3.3) |
| 3 | Contour map | 15 iso-concentration levels |
| 4 | Gradient field | `quiver` of `np.gradient(heatmap)` |
| 5 | Time-series | ppm vs. sample index |
| 6 | 2×2 Dashboard | Summary of plots 1–4 |
| 7 | Bayesian source | Posterior heatmap + MAP point + confidence circle |

### 3.3 Interpolation Scheme

```python
z = griddata((xs, ys), ppms, (xi, yi), method='cubic')
mask = np.isnan(z)
z[mask] = griddata((xs, ys), ppms, (xi[mask], yi[mask]), method='nearest')
z = np.clip(z, 0, None)
```

Cubic interpolation gives a C1-smooth estimate inside the convex hull of measurement points. Nearest-neighbor fills the exterior to avoid blank edges. The positivity clip enforces the physical constraint `C >= 0`.

---

## 4. Auto Coverage Mapper — `auto_coverage_mapper.py`

This node simultaneously controls the robot and records data. An online detection algorithm shapes which regions receive dense sampling.

### 4.1 Phase 1: Coarse BCD Lawnmower Sweep

A **Boustrophedon Cell Decomposition (BCD)** path is computed over the free space from the Nav2 OccupancyGrid.

**Grid construction:**
- Slice the map into vertical columns spaced `coarse_step` apart (default `0.5 m`).
- Within each column, find contiguous obstacle-free segments (occupancy below threshold, plus a `safe_distance = 0.6 m` exclusion buffer around occupied cells).
- Each free segment becomes one *lap* (a vertical line the robot will traverse).

**Routing (DFS nearest-adjacent):**
1. Start from the lap geometrically closest to the robot's current pose.
2. Among unvisited laps, prefer those in an adjacent column first (`|col_a - col_b| <= 1`).
3. Fall back to global nearest-neighbor if no adjacent unvisited lap exists.
4. For each lap, choose the entry end closest to the robot's current heading.

This produces a near-continuous, backtrack-minimising path without requiring a full TSP solver.

### 4.2 Online CUSUM Change-Point Detector

The detection backbone is an **upper one-sided CUSUM (Cumulative Sum Control Chart)**, one instance per sensor topic, with self-calibrating baseline.

#### Self-calibration

The first `cusum_warmup = 50` samples are consumed silently to estimate the environmental baseline:

```
mu_0 = mean(x[0..N])
sigma = std(x[0..N])
```

No hardcoded PPM threshold is used. All decision boundaries are relative to this baseline.

#### Running statistic

At each new sample `x_n`:

```
S_n = max(0,  S_{n-1}  +  x_n - mu_0 - k * sigma)
```

where `k = 0.75` is the *allowance* (the dead zone below which drift is ignored).

#### Decision rule

```
if S_n > h * sigma:
    SPIKE DETECTED
    S_n <- 0   (reset)
```

with `h = 5.0`.

#### Statistical properties

- **False alarm rate:** Average run length to false alarm under the null (no change) scales as approximately `exp(2h)`. At `h = 5` this is on the order of `exp(10) ≈ 22,000` samples — extremely conservative.
- **Detection delay:** Expected samples to detect a true step change of size `delta` above baseline is approximately `h / delta`.
- **Dead zone:** Drift up to `k * sigma = 0.75 * sigma` is deliberately ignored, preventing noise triggering.

### 4.3 Hotspot Validation & Merging

Raw CUSUM alarms are refined through three successive gates before Phase 2 is committed to them.

**Gate 1 — Duration:**

```
len(spike_samples) >= min_spike_samples   (default: 15)
```

Rejects transient single-sample exceedances that accumulate past the CUSUM threshold.

**Gate 2 — Peak amplitude:**

```
max(spike_ppms) > mu_0 + min_spike_peak_sigma * sigma   (default: 3.0)
```

A 3-sigma significance test on the peak, independent of the cumulative test. Both gates must pass simultaneously.

**Gate 3 — Spatial merging:**

Validated hotspots from all topics are pooled. Pairs within `hotspot_pad = 1.5 m` of each other are merged. The merged location is the **PPM-weighted centroid**:

```
x_c = sum(ppm_i * x_i) / sum(ppm_i)
y_c = sum(ppm_i * y_i) / sum(ppm_i)
```

Using concentration as weights causes the centroid to gravitate toward the highest-concentration sub-region, which geometrically is closest to the true source. The merged hotspot retains the maximum peak PPM of all contributing spikes.

The top `max_hotspots = 3` by peak PPM are retained for Phase 2.

### 4.4 Phase 2: Fine Refinement

For each retained hotspot centroid `(x_c, y_c)`, a fine BCD path is generated within the bounding box:

```
[x_c - hotspot_pad,  x_c + hotspot_pad]  x  [y_c - hotspot_pad,  y_c + hotspot_pad]
```

at `fine_step = 0.25 m` column spacing — half the coarse spacing, giving 4× the spatial density.

The same navigation + CUSUM pipeline runs over this grid. This is a rudimentary form of **informative path planning**: the coarse sweep identifies where interesting structure is; the fine sweep invests sampling budget there.

### 4.5 Reactive Velocity Reduction

During navigation, the CUSUM state is monitored continuously. When any sensor is currently in an active spiking state:

- The Nav2 goal is cancelled.
- Raw `cmd_vel` commands are published at `gas_slowdown_speed = 0.15 m/s` toward the current waypoint.

**Motivation:** MOX sensors have rise times of 5–20 seconds. Slowing down increases dwell time in high-concentration zones, allowing the sensor to approach its equilibrium response and producing denser spatial samples near the hotspot.

**Navigation recovery:** Three consecutive navigation failures trigger a recovery sequence: costmap clear → spin (π/2 rad) → reverse (`-0.3 m/s`) → retry.

### 4.6 Live Concentration Grid

A 2D occupancy grid at `0.2 m` resolution is maintained as a **running mean per cell** using Welford's algorithm:

```
count[u,v]  += 1
n = count[u,v]
grid[u,v]   += (ppm - grid[u,v]) / n
```

This is numerically stable, requires O(1) memory per cell, and computes the exact mean without storing all historical values. Unvisited cells are marked `-1`. The grid is published as `OccupancyGrid` (0–100 scale, normalised by peak observed PPM) at 1 Hz on `/PioneerP3DX/concentration_grid` for live RViz feedback.

---

## 5. Bayesian Source Localization

The estimator lives in the shared, ROS-free module `gas_viz.estimate_source` — used live by `auto_coverage_mapper` at mission end and offline by `plot_from_npz` / the synthetic-run harness, so the *identical* algorithm runs in both places. It runs once over all collected `(x, y, ppm)` data per gas and returns a probability **region** (a posterior over source location) plus ranked candidates and 50% / 90% credible (HPD) areas — deliberately *not* a single false pinpoint.

### 5.1 Forward Model

The source localization assumes an **isotropic steady-state diffusion model** (no wind advection):

```
C_pred(s, p)  =  A / ( ||s - p|| + eps )
```

where:
- `s` = candidate source location (grid cell)
- `p` = measurement position
- `A` = unknown release rate (marginalized analytically)
- `eps = 0.15 m` = regularization preventing singularity at `r = 0`

This corresponds to the Green's function of the 3D Laplace equation `∇²C = 0` outside the source — valid under steady-state, isotropic, no-wind assumptions.

### 5.2 Analytical Marginalization of Release Rate A

For each candidate source location `s`, define the unit-scaled decay vector:

```
f_i  =  1 / ( ||s - p_i|| + eps )
```

The optimal release rate scale (least-squares in A) is:

```
alpha*  =  (f · C_obs) / (f · f)      [dot products over all measurements]
alpha*  =  max(alpha*, 0)              [non-negativity: source can't emit negatively]
```

### 5.3 Log Posterior and Normalization

```
log P(s | C_obs)  =  -1/(2 * sigma_n^2)  *  sum_i [ C_obs_i - alpha* * f_i ]^2
```

where `sigma_n` is the noise standard deviation, estimated as the 50th percentile (median) of all observed PPM readings — a robust estimator that is not dominated by high-concentration outliers.

Normalize over all grid cells:

```
P(s | C_obs)  =  exp( log P(s) ) / sum_{s'} exp( log P(s') )
```

**MAP estimate:**

```
s_hat  =  argmax_s  P(s | C_obs)
```

**Posterior spread (confidence radius):**

```
sigma_r  =  sqrt( sum_s  P(s | C_obs) * ||s - s_hat||^2 )
```

This is the posterior root-mean-square distance from the MAP — a single scalar summarising how concentrated the belief is spatially.

The posterior grid is computed at `0.10 m` resolution over the full map extent. This is an `O(N * M)` batch computation (N grid cells × M measurements), executed once at mission end.

### 5.4 Calibration improvements (current implementation)

Four changes make the posterior an honest *region* rather than an over-confident dot, and remove three biases noted in earlier versions:

1. **All readings, including near-zero.** The likelihood uses every sample, not only positive ones: the *absence* of gas constrains where the source can be (a candidate that predicts gas where none was measured is penalised).

2. **Spatial binning.** Readings are binned to one observation per `0.20 m` cell before the inversion. The robot dwells unevenly along its path; without this, oversampled locations would dominate the likelihood and collapse the posterior to a false pinpoint. Each surveyed location now contributes once.

3. **Empirical-Bayes, scale-invariant noise.** Instead of an absolute sensor `sigma_n`, the likelihood is scaled by the best-fit residual level `chi2_min` (an MLE of the effective noise). The posterior *shape* then depends only on the relative goodness-of-fit across candidates — invariant to signal amplitude, sample count and baseline noise. When the `1/r` model fits poorly (real-plume mismatch) `chi2_min` is large and the region honestly widens. The CUSUM-learned baseline `sigma` (exported per gas in the NPZ) provides a floor, replacing the old "median of all readings" estimator that over-counted plume samples.

4. **Power likelihood (`eff_obs`).** The exponent is weighted by an *effective number of independent spatial observations* (`eff_obs ≈ 20`) rather than the raw, autocorrelated sample count, so the credible-region size reflects genuine uncertainty rather than survey density. `eff_obs` is the single knob for region size.

**Optional wind advection.** When a measured wind is supplied and consistent (`wind_consistency ≥ 0.4`, `use_wind_shift=True`), the forward model is evaluated at `s − û·tau`: gas released at `s` is transported downwind before measurement, which moves posterior mass *upwind* toward the true source. Off by default (turbulent indoor fields make a single mean-wind vector unreliable); the wind-free `1/r` inversion is the default.

---

## 6. Data Persistence & Outputs

### 6.1 CSV (Streaming, Fault-Tolerant)

```
~/gaden_results/auto_coverage/data/readings_YYYYMMDD_HHMMSS.csv
```

Columns: `timestamp_sec, topic, x, y, ppm, phase`

Written at every sensor callback. Flushed to disk every 100 samples. If the node crashes, all previously written data is preserved.

### 6.2 NPZ Archive (End of Mission)

```
~/gaden_results/auto_coverage/data/run_YYYYMMDD_HHMMSS.npz
```

Contains per-topic NumPy arrays `(x, y, ppm, timestamps)`, Bayesian posterior grids, source MAP estimate, confidence radius, live concentration grid, and waypoint lists. Load with `np.load(path, allow_pickle=True)`.

### 6.3 Visualization Plots (7 per Sensor Topic)

| # | Filename suffix | Content |
|---|-----------------|---------|
| 1 | `_scatter.png` | Raw readings coloured by PPM, sweep path overlay |
| 2 | `_heatmap.png` | Cubic-interpolated concentration field |
| 3 | `_contour.png` | 15 iso-concentration levels |
| 4 | `_gradient.png` | Quiver arrows of `∇C` |
| 5 | `_timeseries.png` | PPM vs. sample index + CUSUM value overlay |
| 6 | `_dashboard.png` | 2×2 summary of plots 1–4 |
| 7 | `_bayesian.png` | Posterior heatmap + MAP point + confidence circle |

Concentration mapper outputs go to `~/gaden_results/concentration_maps/`.

#### Decision-ready per-gas maps (`gas_viz`, emitted live and by `plot_from_npz`)

| Filename suffix | Content |
|-----------------|---------|
| `_concentration_intensity.png` | **Where the gas peaks** — turbo heatmap with data-relative intensity bands (trace → PEAK) auto-scaled per gas, a highlighted peak region, unsurveyed area hatched, walls overlaid. The "good middle ground" between a binary hazard map and the raw research plots. |
| `_source_localization.png` | **Probabilistic source REGION** — `P(source)` posterior with 50% / 90% credible (HPD) contours, ranked candidates and plume axis. A region, not a pinpoint. |
| `benchmark_localization.png` | Predicted vs. ground-truth sources + per-gas error bars. |

Regenerate offline from a finished run with `python3 plot_from_npz.py <run_dir> <scenario_path> [--recompute]` (ROS-free). `make_synthetic_run.py` fabricates a schema-compatible `run_data.npz` for testing the renderers without GADEN. The combined cross-gas hazard overview + situation report remains in `risk_map_prototype.py`.

---

## 7. ROS Integration

### 7.1 Topic Summary

| Topic | Type | Publisher | Subscriber | Purpose |
|-------|------|-----------|------------|---------|
| `/fake_pid/Sensor_reading` | `olfaction_msgs/GasSensor` | `fake_gas_sensor` | both mappers | Gas concentration readings |
| `/PioneerP3DX/concentration_grid` | `nav_msgs/OccupancyGrid` | `auto_coverage_mapper` | RViz | Live 20 cm concentration map |
| `/cmd_vel` | `geometry_msgs/Twist` | `auto_coverage_mapper` | `diff_drive_controller` | Direct velocity during reactive slowdown |
| `/map` | `nav_msgs/OccupancyGrid` | `map_server` | `auto_coverage_mapper` | Free-space for BCD path generation |

### 7.2 Service & Action Summary

| Interface | Type | Server | Client | Purpose |
|-----------|------|--------|--------|---------|
| `/odor_value` | `GasPosition` srv | `gaden_player` | `fake_gas_sensor` | Query gas concentration at position |
| `/wind_value` | `WindPosition` srv | `gaden_player` | `simulated_anemometer` | Query wind velocity at position |
| `/PioneerP3DX/navigate_to_pose` | `NavigateToPose` action | `nav2` | `auto_coverage_mapper` | Send waypoint goals |
| `/{ns}/clear_entirely_*_costmap` | `Empty` srv | `nav2` | `auto_coverage_mapper` | Reset costmaps on recovery |

### 7.3 TF Tree

```
map
 ├─ PioneerP3DX_base_link          (robot body)
 │   ├─ PioneerP3DX_pid_frame      (gas sensor, 0.5 m height)
 │   └─ PioneerP3DX_anemometer_frame (wind sensor, 0.5 m height)
 └─ base_link                      (Nav2 compatibility alias)
```

Both mappers look up `map → base_link` (or `map → PioneerP3DX_base_link`) at each sensor callback to associate a world-frame position with each reading.

---

## 8. Configuration Parameters

### Auto Coverage Mapper

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `namespace` | `'PioneerP3DX'` | — | Robot namespace prefix |
| `sensor_topics` | `['/fake_pid/Sensor_reading']` | — | Gas sensor topic list |
| `coarse_step` | `0.5` | m | Phase 1 column spacing |
| `fine_step` | `0.25` | m | Phase 2 column spacing |
| `hotspot_pad` | `1.5` | m | Half-width of fine-sweep bounding box |
| `safe_distance` | `0.6` | m | Obstacle exclusion buffer |
| `nav_timeout` | `30.0` | s | Per-waypoint navigation timeout |
| `wp_tolerance` | `0.5` | m | Goal-reached distance threshold |
| `gas_slowdown_speed` | `0.15` | m/s | Velocity while CUSUM is spiking |
| `cusum_warmup` | `50` | samples | Calibration window |
| `cusum_k` | `0.75` | — | CUSUM allowance (dead zone factor) |
| `cusum_h` | `5.0` | — | CUSUM decision threshold |
| `min_spike_samples` | `15` | samples | Minimum spike duration |
| `min_spike_peak_sigma` | `3.0` | σ | Peak significance threshold |
| `max_hotspots` | `3` | — | Maximum Phase 2 refinement regions |

### Concentration Mapper

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sensor_topics` | `['/fake_pid/Sensor_reading']` | Gas sensor topic list |
| `base_frame` | `'PioneerP3DX_base_link'` | Robot TF frame |
| `map_frame` | `'map'` | World TF frame |
| `grid_resolution` | `0.1` | Heatmap cell size (m) |

---

## 9. Assessment & Known Limitations

### What Works Well

- **CUSUM self-calibration** avoids hardcoded PPM thresholds and adapts to any simulation baseline automatically.
- **Two-phase strategy** (coarse detect → fine refine) is an efficient use of sampling budget without requiring a prior map of source locations.
- **PPM-weighted centroid** correctly exploits the monotonic relationship between concentration and proximity; higher readings pull the centroid toward the source.
- **Welford running mean** for the live grid is exact, numerically stable, and O(1) memory per cell.
- **CSV streaming with periodic flush** provides fault-tolerant data capture.

### Known Limitations

**1. Source localization model mismatch** *(partially addressed)*

The forward model assumes `C ∝ 1/r` (isotropic steady-state diffusion); GADEN produces anisotropic, time-varying, wind-advected plumes, so the wind-free posterior can still bias downwind. Two mitigations are now in place (§5.4): the optional wind-advection offset `C_pred(s, p) ≈ A / (||s − û·tau − p|| + eps)` when a consistent wind is measured, and the empirical-Bayes noise scaling that *widens* the credible region precisely when model mismatch is large (so the reported region stays honest rather than confidently wrong).

**2. Sensor lag vs. spatial stamping**

MOX sensors have time constants of 5–20 seconds. At `0.3 m/s` with `tau_decay = 18 s`, the 63%-rise-time position error is approximately `0.3 * 18 = 5.4 m` — larger than the hotspot merge radius of `1.5 m`. Readings are stamped at the robot's *current* position when the message is received, not where the gas encounter actually occurred. The reactive slowdown partially compensates this but only after the CUSUM has already fired.

**3. Noise variance estimation** *(addressed)*

Earlier versions estimated `sigma_n` from the median of *all* observations, including plume samples. The current estimator (§5.4) instead floors the noise at the CUSUM-learned **pre-gas baseline** `sigma` (exported per gas in the NPZ) and scales the likelihood by the empirical best-fit residual `chi2_min`, so the posterior shape no longer depends on that biased global statistic.

**4. Phase 1 exit condition**

Phase 1 terminates when all coarse waypoints are exhausted, regardless of whether any spike has been detected. If the robot exits the plume before completing the full sweep (e.g., narrow corridor environment), no hotspots will be found and Phase 2 will not execute. A time-windowed early-exit condition — trigger Phase 2 as soon as `N` consecutive waypoints pass with no spike after the last detection — would make the system more robust.

**5. RAM accumulation in concentration_mapper**

`concentration_mapper.py` stores all `(x, y, ppm)` triplets in unbounded Python lists. For long missions or high sensor rates this can become significant. A bounded circular buffer or periodic flush to disk would be safer.

**6. No wind field integration**

The wind data published by `simulated_anemometer` is not used by either mapper. Incorporating upwind navigation during Phase 2 (chemotaxis) would accelerate source localization significantly, especially in corridor environments where `1/r` isotropic posterior is poorly conditioned.
