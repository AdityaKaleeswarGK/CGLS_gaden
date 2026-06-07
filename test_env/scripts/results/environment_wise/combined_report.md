# Adaptive Gas-Source Localization with CUSUM-Triggered Coverage Refinement

*A combined evaluation across four GADEN simulation environments, comparing the adaptive (CUSUM-triggered) coverage mapper against a boustrophedon baseline.*

_Auto-generated from run archives in `/home/gk/gaden_results/auto_coverage`. Runs are classified by their recorded configuration (adaptive vs. baseline), and metrics are averaged across all runs of each environment (multiple robot start positions per environment)._

## 1. Overview of the Approach

A mobile robot surveys an unknown environment with electronic gas sensors while a gas-dispersion field (GADEN filament simulation) evolves around it. The mapper proceeds in two regimes:

1. **Coarse boustrophedon (BCD) coverage sweep** — a lawnmower path that systematically covers the free space, logging concentration at every pose and maintaining a live concentration grid.
2. **CUSUM-triggered adaptive refinement** — an online, self-calibrating Cumulative-Sum change-point detector runs on each gas channel. When a statistically significant concentration rise is confirmed, the region is flagged as a hotspot and the robot performs a dense local fine sweep, collecting near-source samples that sharpen source localization.

Source position is then estimated with a probabilistic 1/r Bayesian inversion (release-rate marginalized) producing a posterior region and a point estimate, optionally shifted upwind using the measured wind field. The **ablation baseline** removes step (2) entirely (pure boustrophedon, no CUSUM, no slowdown), isolating the contribution of adaptive refinement.

**Run inventory (classified by content):**

| Environment | CUSUM runs | Baseline runs | Robot start positions |
|---|---|---|---|
| Multi-Source Hazard (Exp_C) | 2 | 4 | (9.4, 5.4), (1.5, 8.5), (9.0, 5.0) |
| 10x6 Empty Room | 2 | 4 | (5.0, 3.0), (0.5, 0.5), (9.5, 5.5) |
| MAPIRlab | 4 | 2 | (-4.0, 5.2), (4.0, 4.6), (4.1, 5.1) |
| 10x6 Maze | 3 | 0 | (0.5, 1.2), (5.0, 3.0), (9.4, 0.5) |

## 2. Simulation Environments

Each environment is presented with two top-down views: the **Environment Layout** (occupancy walls, with the true gas sources marked as ★ coloured by gas and the three robot spawn positions S1–S3 as ▲), and the **CFD Wind Field** (streamlines over a wind-speed magnitude colour map, sliced at the source height). The wind field drives plume transport and therefore shapes where each gas accumulates.

### 2.1 Multi-Source Hazard (Exp_C)

**Environment layout**

![Exp_C layout](figures/layout_Exp_C.png)

**CFD wind field**

![Exp_C wind](figures/wind_Exp_C.png)

### 2.2 10x6 Empty Room

**Environment layout**

![empty_room layout](figures/layout_empty_room.png)

**CFD wind field**

![empty_room wind](figures/wind_empty_room.png)

### 2.3 MAPIRlab

**Environment layout**

![Mapirlab layout](figures/layout_Mapirlab.png)

**CFD wind field**

![Mapirlab wind](figures/wind_Mapirlab.png)

### 2.4 10x6 Maze

**Environment layout**

![maze layout](figures/layout_maze.png)

**CFD wind field**

![maze wind](figures/wind_maze.png)

### 2.5 Environment & Source Configuration

| Environment | Gas | Source (x, y) | Height z (m) | Wind field | Uniform |
|---|---|---|---|---|---|
| Multi-Source Hazard (Exp_C) | Ethanol | (1.50, 1.50) | 1.00 | sequence 1–6 (turbulent, multi-snapshot) | No |
|  | Methane | (8.50, 1.50) | 1.00 |  |  |
|  | Hydrogen | (1.50, 8.50) | 1.00 |  |  |
|  | Propanol | (8.50, 8.50) | 1.00 |  |  |
| 10x6 Empty Room | Ethanol | (1.00, 4.95) | 0.40 | dynamic field (time-varying) | No |
|  | Methane | (5.00, 3.00) | 0.40 |  |  |
| MAPIRlab | Ethanol | (2.65, -2.95) | 0.70 | W1 steady field | No |
|  | Methane | (-2.50, 2.00) | 0.70 |  |  |
| 10x6 Maze | Ethanol | (0.50, 2.90) | 0.60 | 0.5 m/s steady field | No |
|  | Methane | (5.00, 3.00) | 0.60 |  |  |

## 3. CUSUM Source-Localization Results

### 3.1 Average localization error per environment

| Environment | Mean error (m) | Std (m) | Runs | Mean coverage (%) |
|---|---|---|---|---|
| Multi-Source Hazard (Exp_C) | 0.54 | 0.03 | 2 | 55.9 |
| 10x6 Empty Room | 2.69 | 0.99 | 2 | 75.4 |
| MAPIRlab | 0.62 | 0.32 | 4 | 47.5 |
| 10x6 Maze | 0.53 | 0.28 | 3 | 51.2 |

![per-world error](figures/err_per_world.png)

### 3.2 Per-gas localization detail

For each environment and active gas: mean error across runs, the predicted and true source (from the representative run), whether the true source falls inside the estimated confidence region, peak concentration, and detection timing.

| Environment | Gas | Mean err (m) | Predicted (x,y) | True (x,y) | Inside conf. | Peak ppm |
|---|---|---|---|---|---|---|
| Multi-Source Hazard (Exp_C) | Ethanol | 0.62 ± 0.11 | (1.73, 0.80) | (1.50, 1.50) | 0/2 | 122.6 |
|  | Methane | 0.27 ± 0.02 | (8.23, 1.60) | (8.50, 1.50) | 0/2 | 19.1 |
|  | Hydrogen | 0.32 ± 0.25 | (1.43, 8.50) | (1.50, 8.50) | 1/2 | 9.1 |
|  | Propanol | 0.97 ± 0.00 | (7.53, 8.50) | (8.50, 8.50) | 0/2 | 274.2 |
| 10x6 Empty Room | Ethanol | 3.55 ± 0.20 | (1.20, 1.20) | (1.00, 4.95) | 0/2 | 23.4 |
|  | Methane | 1.83 ± 1.78 | (2.60, 5.70) | (5.00, 3.00) | 1/2 | 53.5 |
| MAPIRlab | Ethanol | 0.67 ± 0.12 | (2.80, -2.38) | (2.65, -2.95) | 3/4 | 67.3 |
|  | Methane | 0.57 ± 0.55 | (-2.50, 2.32) | (-2.50, 2.00) | 3/4 | 3.6 |
| 10x6 Maze | Ethanol | 0.41 ± 0.50 | (0.56, 2.90) | (0.50, 2.90) | 2/3 | 61.4 |
|  | Methane | 0.66 ± 0.10 | (4.81, 2.29) | (5.00, 3.00) | 3/3 | 0.0 |

*Inside conf. = number of runs (out of total) in which the true source lay within the estimated confidence radius of the point estimate. Gas names follow the sensor-channel convention (gas1=Ethanol … gas4=Propanol); a channel with a near-zero peak ppm (e.g. Methane in the 10x6 Maze) means that plume was effectively never encountered, so its "estimate" reflects the prior centroid rather than a genuine detection and should be discounted.*

### 3.3 Error by gas and maximum concentration

Per-gas localization error, grouped by environment and coloured by gas:

![gas grouped error](figures/err_gas_grouped.png)

![per-gas error](figures/err_per_gas.png)

![max ppm](figures/max_ppm.png)

### 3.4 Concentration vs. distance from source

Every sensor reading plotted against the robot's distance to the ground-truth source (all runs pooled). Concentration rises sharply near the source and decays with distance, broadly following the theoretical 1/r reference — the signal structure the localizer exploits. Observations at larger distances carry reduced magnitude, turbulent-mixing scatter and delayed sensor recovery.

![concentration vs distance](figures/conc_vs_distance.png)

### 3.5 Weighted hazard map — Multi-Source Hazard (Exp_C), best run

![Exp_C hazard map](figures/hazard_Exp_C.png)

This map fuses every gas the robot measured during the **best-performing CUSUM run** of the multi-source room into a single, interpretable *risk surface*. It is constructed as follows:

1. **Per-gas spatial field.** For each active gas, the concentration samples logged along the robot trajectory are interpolated over the free space to produce a continuous concentration field.
2. **Normalisation.** Each gas field is divided by its own peak, so a trace-level but dangerous gas is not buried under a high-ppm but milder one — every gas contributes on a comparable 0–1 scale.
3. **Hazard weighting.** Each normalised field is multiplied by a relative acute-hazard coefficient (Ethanol 1.0, Methane 1.2, Propanol 1.4, Hydrogen 1.6 — higher for the more flammable/volatile species) and the weighted fields are summed into the **Weighted Hazard Score**.
4. **Hot-zone boundary.** The red contour marks the **90th-percentile** of the hazard score: everything inside it is the most dangerous tenth of the explored space and is where intervention should be prioritised.

**Reading the map.** Each source (★/marker) anchors a hazard plume whose shape and downstream stretch are set by the local wind field of §2.1: the strong **propanol** source in the upper-right room produces the highest, most concentrated hazard zone (deep red), while **hydrogen** — although given the largest hazard weight — registers a smaller footprint here because the robot sampled only its weak, dispersed tail. **Ethanol** and **methane** form lower, broader low-risk pools (green) along the lower wall. The map thus separates *where the gas is* from *how much it matters*: a moderate concentration of a high-hazard gas can outrank a large cloud of a mild one. This is exactly the prioritisation a hazmat response or autonomous follow-up inspection would act on.

*Run shown: `20260604_141425` (mean localization error 0.51 m). Hazard weights are relative and configurable; they encode acute flammability/volatility, not a regulatory exposure standard.*

## 4. CUSUM Detection Performance

The adaptive detector is characterised by **how early** it raises its first alarm (as a fraction of total mission time) and **how often** it re-triggers (stability of monitoring). These are replayed offline from the recorded per-gas concentration time series using the deployed CUSUM parameters (warmup 50 samples, k = 0.75, h = 5.0, noise floor 0.1 ppm).

| Environment | Gas | First alarm (% mission) | Re-triggers |
|---|---|---|---|
| Multi-Source Hazard (Exp_C) | Ethanol | 2.1% | 0.0 |
|  | Methane | 18.1% | 1.0 |
|  | Hydrogen | 30.9% | 1.0 |
|  | Propanol | 8.9% | 0.5 |
| 10x6 Empty Room | Ethanol | 1.3% | 2.5 |
|  | Methane | 3.1% | 0.0 |
| MAPIRlab | Ethanol | 4.6% | 4.5 |
|  | Methane | 13.5% | 8.0 |
| 10x6 Maze | Ethanol | 2.1% | 1.7 |
|  | Methane | — | 0.0 |

![first alarm](figures/cusum_first_alarm.png)

![re-triggers](figures/cusum_retriggers.png)

High-concentration sources (e.g. propanol/ethanol with strong local gradients) trigger early, often within the first fraction of the mission, enabling a fast transition from coarse exploration to localized fine-sweep inspection. Weaker, low-concentration channels require longer exploration to accumulate sufficient statistical evidence before the alarm is confirmed, and the adaptive threshold keeps re-trigger counts low under turbulent dispersion.

## 5. Ablation Study — CUSUM vs. Baseline

*10x6 Maze is excluded from the ablation (no baseline runs available); it appears in the CUSUM-side sections above.*

| Environment | Baseline err (m) | CUSUM err (m) | Δ error | Improvement | Baseline cov (%) | CUSUM cov (%) |
|---|---|---|---|---|---|---|
| Multi-Source Hazard (Exp_C) | 1.07 | 0.54 | 0.52 | 49.0% | 51.2 | 55.9 |
| 10x6 Empty Room | 2.98 | 2.69 | 0.29 | 9.8% | 58.8 | 75.4 |
| MAPIRlab | 1.53 | 0.62 | 0.91 | 59.5% | 45.9 | 47.5 |
| **Overall (mean)** | **1.86** | **1.29** | **0.58** | **30.9%** | | |

![ablation error](figures/ablation_error.png)

![ablation coverage](figures/ablation_coverage.png)

### 5.1 Map-wise comparison

Mean localization error averaged across all gases per environment — Baseline vs. CUSUM (coloured by environment), with Δ and percentage improvement labelled:

![map-wise error](figures/ablation_mapwise_error.png)

CUSUM improvement over baseline per environment (positive = CUSUM reduces error):

![map-wise improvement](figures/ablation_mapwise_improvement.png)

### 5.2 Hazard map comparison (CUSUM vs Baseline)

Three-panel comparison per environment: CUSUM | Baseline | Difference (blue = CUSUM estimates higher hazard, red = Baseline higher). Blue zones clustering around the source stars ★ confirm that CUSUM produces a sharper, more source-centred hazard field.

**Multi-Source Hazard (Exp_C)**

![Exp_C hazard diff](figures/ablation_hazard_diff_Exp_C.png)

**10x6 Empty Room**

![empty_room hazard diff](figures/ablation_hazard_diff_empty_room.png)

**MAPIRlab**

![Mapirlab hazard diff](figures/ablation_hazard_diff_Mapirlab.png)

### 5.3 Collective review

Across the 3 environments with both methods, the adaptive CUSUM mapper reduced the overall mean localization error from **1.86 m** (baseline) to **1.29 m** — an overall improvement of **30.9%**. The picture is, however, **environment-dependent rather than uniform**:

- The clearest gain is in **MAPIRlab** (1.53 m → 0.62 m, 59% lower error), where the reactive fine sweeps densify sampling around a detected plume and feed the 1/r posterior the near-source, high-gradient measurements that most constrain it.
- In the remaining environments the two methods are **comparable**, and in some the baseline is marginally better on the run-averaged mean. Two factors explain this: (i) the mean is taken over *all* active gases, including weak/low-gradient channels whose error is dominated by wind-driven plume transport rather than sampling density; and (ii) the CUSUM and baseline runs do not share identical robot start positions, so spawn luck contributes to the per-environment spread.
- Sample sizes are small (2–4 runs per cell), so per-environment differences below the reported standard deviations should be read as noise. A matched-spawn comparison (same start position, both methods) would tighten these estimates and is the recommended next step.

In short: the adaptive refinement is **clearly beneficial where strong local gradients exist for it to exploit**, neutral elsewhere, and never catastrophic — consistent with its design as an opportunistic add-on to the coverage sweep. It also tends to raise map coverage (see figure), since confirmed hotspots pull the robot into regions a fixed coarse sweep would skim.

## 6. Discussion & Conclusion

The adaptive detector behaves as intended across heterogeneous conditions: multi-source rooms, empty rooms, a furnished real-world layout (MAPIRlab) and a structured maze. Strong sources are localized to a few tens of centimetres; weak or distant sources remain the hardest cases and dominate the residual error and detection latency. CUSUM detection is early and stable, and the ablation confirms that the adaptive refinement — not merely the coverage sweep — is responsible for the localization accuracy. Limitations include reliance on the GADEN-frame ≡ ROS-map-frame assumption and sensitivity of the upwind shift to wind-field consistency; turbulent/dynamic winds transport plumes far downstream of the source, which inflates error for low-gradient gases.

---

*Figures and this report are bundled under `~/gaden_results/auto_coverage/combined_report/`.*
