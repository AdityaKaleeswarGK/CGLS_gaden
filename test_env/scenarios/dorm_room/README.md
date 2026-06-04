# dorm_room — fire / CO₂ dispersion with a pseudo-YOLO no-go zone

A small dorm room (~5.4 × 7.0 m) used to test **auto-coverage + gas mapping**
when there is a hazard the robot must **detect and avoid** while still mapping
the gas that spreads around it.

Scenario story: a **fire** (the red dot on the floor-plan) is releasing **CO₂**.
A vision model (YOLO, simulated here) flags the fire at high confidence; the
robot must **skip that region** (you don't drive a robot into a fire) and
efficiently cover the rest of the room, mapping how the CO₂ spreads — the way
smoke from a room fire drifts toward an **open window**.

GADEN cannot simulate the fire object itself, so it is modelled two ways at
once: a **CO₂ point source** at the fire (for the gas map) **and** a known
**no-go location** the coverage planner carves around (for the avoidance).

**Why CO₂, and why subtle?** A real fire emits mostly water vapour and CO₂,
plus carbon monoxide (CO) and visible smoke/soot — CO₂ is a *major* product
but the plume is a mixture, not pure CO₂. So we model a **controlled, subtle**
CO₂ release (low peak concentration + low emission rate) rather than a flood,
and lean on diffusion so it still **spreads through the room** instead of
forming one tight jet. (Smoke — gasType 13 — would instead *rise*; see Tuning.)

```
 +Y (7.0) +-[ wardrobe ]-=== window ===---------------------+
          |                                                 |
          |                                                 |  CO2 vents up to
          |                                                 |  the window AND
          |                   * fire / CO2  (3.80, 1.80)    |  spreads through
          |                 ^ start (2.70, 0.60)            |  the rest of the room
    (0,0) +-------------------------------------------------+ +X (5.4)
```

| Element            | Location / value                                  |
|--------------------|---------------------------------------------------|
| Room (outer)       | x ∈ [0, 5.4], y ∈ [0, 7.0], height 2.4 m          |
| Wardrobe (obstacle)| x ∈ [0.2, 1.6], y ∈ [5.6, 6.8] (top-left)         |
| Open window (vent) | top wall, x ∈ [1.9, 2.9] — GADEN **outlet**        |
| Robot start        | (2.70, 0.60), facing +Y                           |
| Fire / CO₂ source  | point (3.80, 1.80, 0.50), `gasType: 11` (CO₂)     |
| YOLO no-go         | centre (3.80, 1.80), radius 0.6 m (carved 0.9 m)  |
| Wind               | feeble dorm draft (-0.02, 0.06, 0) m/s ≈ 0.06     |

CO₂ has specific gravity **1.52** (heavier than air → sinks and pools low), so
the low gas sensor (z = 0.5 m) detects it well. The feeble but steady dorm
draft carries the plume from the fire **dominantly up toward the open window**
(venting) without being a strong jet, while the larger filament growth and
turbulent noise let it **also spread to the rest of the room**. The gas-source
localizer can
re-find the fire from the surrounding readings even though the robot never
drives onto it.

## Run it

```bash
# 0) (once, after adding/editing the scenario) build so the files land in the install share
cd ~/ros2_ws && colcon build --symlink-install --packages-select test_env && source install/setup.bash

# 1) preprocess: voxelise the CAD into occupancy + wind grid
ros2 launch test_env gaden_preproc_launch.py scenario:=dorm_room configuration:=config1

# 2) generate the CO2 dispersal (filament simulator)
ros2 launch test_env gaden_sim_launch.py scenario:=dorm_room configuration:=config1 simulation:=sim1

# 3) bring up the robot + nav + the CO2 sensor (gas1 retargeted to CO2) + RViz
ros2 launch test_env main_simbot_launch.py scenario:=dorm_room configuration:=config1 \
     simulation:=sim1 num_sensors:=1 gas1_target:=carbonDioxide

# 4) run the auto-coverage mapper with the pseudo-YOLO fire no-go enabled
SHARE=$(ros2 pkg prefix test_env)/share/test_env
ros2 run test_env auto_coverage_mapper --ros-args \
  -p scenario_path:=$SHARE/scenarios/dorm_room/environment_configurations/config1 \
  -p generate_plots:=true \
  -p sensor_topics:=['/gas1/Sensor_reading'] \
  -p use_wind_shift:=true \
  -p nogo_skip_enabled:=true -p nogo_x:=3.8 -p nogo_y:=1.8 -p nogo_radius:=0.6 -p nogo_label:=fire \
  -p gas_labels:=['/gas1/Sensor_reading=carbonDioxide']
```

Results land in `~/gaden_results/auto_coverage/<timestamp>/` (plots, `readings.csv`,
`yolo_events.csv`, `run_data.npz`, `README.md`). The coverage/heatmap plots show
the fire no-go as a red hatched circle.

## Tuning

- **Geometry** — edit dimensions in `test_env/scripts/make_dorm_geometry.py`
  and re-run it (`python3 test_env/scripts/make_dorm_geometry.py`), then redo
  step 1–2.
- **Source position / height** — `simulations/sim1/sim.yaml` → `source.position`.
  It sits toward the bottom of the room (3.8, 1.8); CO₂ sinks, so the low z = 0.5
  source is detected easily — raise z to model a higher fire.
- **CO₂ amount & spread** — also in `sim1/sim.yaml`: `filamentPPMcenter` (10) and
  `numFilaments_sec` (5) set *how much* CO₂ (lower = subtler); `filamentGrowthGamma`
  (20) and `filamentNoise_std` (0.07) set *how widely* it spreads (higher = reaches
  more of the room — including slowly backward/below the source — at lower peaks). To model **smoke** that rises instead, set
  `gasType: 13` (specific gravity 0.89) and raise the sensor height.
  **Any of these changes requires re-running step 2** (the dispersal is precomputed).
- **Window / draft** — move the window gap in `make_dorm_geometry.py` and the
  draft vector in `wind_simulations/draft/wind_0.csv` (each line is one
  timestep's `Ux, Uy, Uz`, shared by all cells; `uniformWind: true`).
- **No-go size / detection** — `nogo_radius`, `yolo_detect_range`,
  `yolo_confidence` are mapper params (no rebuild needed).
