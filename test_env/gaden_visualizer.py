#!/usr/bin/env python3
"""
GADEN Wind Field Visualizer
============================
Produces two publication-ready figures (white background) for any scenario:
  1. <scenario>_<wind>_world.png  — Environment layout only (STL outlines)
  2. <scenario>_<wind>_wind.png   — Environment + wind direction arrows

Setup:
    Place this script at:
        /home/gk/ros2_ws/src/gaden/test_env/gaden_visualizer.py

    Install dependencies (once):
        pip install trimesh matplotlib numpy pyyaml

Usage:
    python3 gaden_visualizer.py --scenario MAPIRlab --wind W1
    python3 gaden_visualizer.py --scenario 10x6_maze --wind 1ms
    python3 gaden_visualizer.py --scenario 10x6_empty_room --wind dynamic
    python3 gaden_visualizer.py --scenario MAPIRlab --wind W1 --z 1.5
    python3 gaden_visualizer.py --scenario MAPIRlab --wind W1 --output ~/Desktop

Output:
    Two PNGs saved to gaden_viz_output/ (or --output folder):
        MAPIRlab_W1_world.png
        MAPIRlab_W1_wind.png
"""

import argparse
import os
import sys
import glob
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import trimesh

# ─────────────────────────────────────────────────────────────
# PATH — script lives in test_env/, scenarios/ is next to it
# ─────────────────────────────────────────────────────────────
SCENARIOS_BASE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "scenarios"
)

# ─────────────────────────────────────────────────────────────
# STL DISPLAY STYLES  (white background, paper-friendly)
# ─────────────────────────────────────────────────────────────
STL_STYLES = {
    "walls":     {"color": "#111111", "alpha": 0.95, "lw": 2.2},
    "inner":     {"color": "#111111", "alpha": 0.70, "lw": 1.0},
    "tables":    {"color": "#111111", "alpha": 0.80, "lw": 1.2},
    "wardrobes": {"color": "#111111", "alpha": 0.80, "lw": 1.2},
    "doors":     {"color": "#111111", "alpha": 0.85, "lw": 1.4},
    "windows":   {"color": "#111111", "alpha": 0.85, "lw": 1.4},
    "default":   {"color": "#111111", "alpha": 0.70, "lw": 1.0},
}

LEGEND_PATCHES = [
    mpatches.Patch(color="#111111", label="Walls"),
    mpatches.Patch(color="#444444", label="Inner Structure"),
    mpatches.Patch(color="#7a5500", label="Tables"),
    mpatches.Patch(color="#3d2b1a", label="Wardrobes"),
    mpatches.Patch(color="#cc6600", label="Doors"),
    mpatches.Patch(color="#1a7acc", label="Windows"),
    mpatches.Patch(color="red",     label="Gas Source"),
]

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def load_config(scenario_path):
    """Read config.yaml and sim.yaml → (cfg dict, cell_size, source pos)."""
    cfg_path = os.path.join(scenario_path,
                            "environment_configurations", "config1", "config.yaml")
    sim_path = os.path.join(scenario_path,
                            "environment_configurations", "config1",
                            "simulations", "sim1", "sim.yaml")
    cfg = {}
    cell_size = 0.1
    source = None

    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f) or {}
        cell_size = cfg.get("cell_size", 0.1)
    else:
        print(f"  [warn] config.yaml not found at {cfg_path}")

    if os.path.exists(sim_path):
        with open(sim_path) as f:
            sim = yaml.safe_load(f) or {}
        pos = sim.get("source", {}).get("position", None)
        if pos:
            source = pos

    return cfg, cell_size, source


def find_stl_files(scenario_path, cfg):
    """Resolve STL paths from config.yaml, fallback to scanning cad_models/."""
    cad_dir = os.path.join(scenario_path, "cad_models")
    stl_files = []

    models = cfg.get("models", []) + cfg.get("outlets_models", [])
    for m in models:
        if isinstance(m, str) and not m.startswith("!"):
            resolved = os.path.normpath(
                os.path.join(scenario_path,
                             "environment_configurations", "config1", m)
            )
            if os.path.exists(resolved):
                stl_files.append(resolved)

    if not stl_files and os.path.isdir(cad_dir):
        stl_files = sorted(glob.glob(os.path.join(cad_dir, "*.stl")))

    return stl_files


def find_wind_csv(scenario_path, wind_name):
    """Find wind CSV file(s) for the given wind sub-folder name."""
    wind_base = os.path.join(scenario_path, "wind_simulations")
    wind_dir  = os.path.join(wind_base, wind_name)

    if not os.path.isdir(wind_dir):
        available = os.listdir(wind_base)
        matches = [d for d in available if wind_name.lower() in d.lower()]
        if matches:
            wind_dir = os.path.join(wind_base, matches[0])
        else:
            print(f"\n[ERROR] Wind scenario '{wind_name}' not found.")
            print(f"  Available: {available}")
            sys.exit(1)

    csvs = sorted(glob.glob(os.path.join(wind_dir, "wind_at_cell_centers_*.csv")))
    if not csvs:
        csvs = sorted(glob.glob(os.path.join(wind_dir, "*.csv")))
    if not csvs:
        print(f"[ERROR] No CSV files found in {wind_dir}")
        sys.exit(1)

    return csvs


def load_wind(csv_path):
    """Load wind CSV → DataFrame with columns U,V,W,X,Y,Z,speed."""
    df = pd.read_csv(csv_path)
    rename = {}
    for c in df.columns:
        cl = c.lower()
        if "u [m/s]:0" in cl: rename[c] = "U"
        elif "u [m/s]:1" in cl: rename[c] = "V"
        elif "u [m/s]:2" in cl: rename[c] = "W"
        elif "points:0"  in cl: rename[c] = "X"
        elif "points:1"  in cl: rename[c] = "Y"
        elif "points:2"  in cl: rename[c] = "Z"
    df = df.rename(columns=rename)
    df["speed"] = np.sqrt(df["U"]**2 + df["V"]**2)
    return df


def get_stl_style(stl_path):
    fname = os.path.basename(stl_path).lower()
    for key in STL_STYLES:
        if key in fname:
            return STL_STYLES[key]
    return STL_STYLES["default"]


def draw_stl_outlines(ax, stl_files, z_height):
    """Draw 2D cross-section outlines of all STL files at given Z height."""
    for stl_path in stl_files:
        style = get_stl_style(stl_path)
        try:
            mesh = trimesh.load(stl_path, force="mesh")
            section = mesh.section(
                plane_origin=[0, 0, z_height],
                plane_normal=[0, 0, 1]
            )
            if section is None:
                continue
            # to_2D() returns vertices in a local projected frame with a
            # translation offset — apply the transform to recover world XY.
            path2d, transform = section.to_2D()
            ox, oy = transform[0, 3], transform[1, 3]
            for entity in path2d.entities:
                pts = path2d.vertices[entity.points]
                wx = pts[:, 0] * transform[0, 0] + pts[:, 1] * transform[0, 1] + ox
                wy = pts[:, 0] * transform[1, 0] + pts[:, 1] * transform[1, 1] + oy
                ax.plot(wx, wy,
                        color=style["color"],
                        alpha=style["alpha"],
                        linewidth=style["lw"],
                        solid_capstyle="round",
                        zorder=5)
        except Exception as e:
            print(f"  [warn] Could not section {os.path.basename(stl_path)}: {e}")


def apply_paper_style(ax, title):
    """Clean white-background style for publication figures."""
    ax.set_facecolor("white")
    ax.set_xlabel("X (m)", fontsize=12, color="black")
    ax.set_ylabel("Y (m)", fontsize=12, color="black")
    ax.set_title(title, fontsize=13, color="black", pad=12)
    ax.tick_params(colors="black", labelsize=10)
    ax.set_aspect("equal")
    ax.grid(True, color="#e0e0e0", linewidth=0.6, zorder=0)
    for sp in ax.spines.values():
        sp.set_edgecolor("#aaaaaa")


# ─────────────────────────────────────────────────────────────
# FIGURE 1 — World Only
# ─────────────────────────────────────────────────────────────

def plot_world(scenario_name, wind_name, stl_files, source, z_height, output_dir):
    print("  [1/2] Environment layout ...")

    fig, ax = plt.subplots(figsize=(9, 9), facecolor="white")
    draw_stl_outlines(ax, stl_files, z_height)

    apply_paper_style(ax,
        f"{scenario_name}  —  Environment Layout\n"
        f"Top-Down View  (z = {z_height} m)"
    )

    plt.tight_layout()
    out = os.path.join(output_dir, f"{scenario_name}_{wind_name}_world.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"       → {out}")
    return out


# ─────────────────────────────────────────────────────────────
# FIGURE 2 — CFD Wind Field (streamlines + contour fill)
# ─────────────────────────────────────────────────────────────

def plot_wind(scenario_name, wind_name, csv_path, stl_files,
              cell_size, source, z_height, output_dir):
    print("  [2/2] CFD wind field ...")

    from scipy.interpolate import griddata  # pip install scipy

    df     = load_wind(csv_path)
    sliced = df[(df["Z"] >= z_height - 0.12) & (df["Z"] <= z_height + 0.12)].copy()

    # Interpolate scattered points onto a dense regular grid
    xi = np.linspace(sliced["X"].min(), sliced["X"].max(), 300)
    yi = np.linspace(sliced["Y"].min(), sliced["Y"].max(), 300)
    Xi, Yi = np.meshgrid(xi, yi)
    pts = sliced[["X", "Y"]].values
    U_grid = griddata(pts, sliced["U"].values, (Xi, Yi), method="linear")
    V_grid = griddata(pts, sliced["V"].values, (Xi, Yi), method="linear")
    S_grid = np.sqrt(U_grid**2 + V_grid**2)

    vmin = np.nanpercentile(S_grid, 5)
    vmax = np.nanpercentile(S_grid, 95)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.jet

    fig, ax = plt.subplots(figsize=(10, 10), facecolor="white")
    ax.set_facecolor("white")

    # 1. Contour fill — velocity magnitude background
    cf = ax.contourf(Xi, Yi, S_grid, levels=60,
                     cmap=cmap, alpha=0.55,
                     vmin=vmin, vmax=vmax, zorder=1)

    # 2. Streamlines — line width scales with speed
    lw_field = 1.2 + 2.0 * (S_grid - vmin) / (vmax - vmin + 1e-9)
    ax.streamplot(Xi, Yi, U_grid, V_grid,
                  color=S_grid,
                  cmap=cmap,
                  linewidth=lw_field,
                  density=2.0,
                  arrowsize=1.2,
                  arrowstyle="->",
                  norm=norm,
                  zorder=2)

    # 3. STL outlines on top
    draw_stl_outlines(ax, stl_files, z_height)

    # 4. Colorbar
    cbar = fig.colorbar(cf, ax=ax, pad=0.02, fraction=0.035)
    cbar.set_label("Velocity Magnitude (m/s)", fontsize=11, color="black")
    cbar.ax.tick_params(colors="black", labelsize=9)
    cbar.outline.set_edgecolor("#aaaaaa")

    csv_name = os.path.basename(csv_path)
    apply_paper_style(ax,
        f"{scenario_name}  —  Wind Flow Field: {wind_name}\n"
        f"Top-Down View  (z = {z_height} m)"
    )
    ax.set_xlim(sliced["X"].min() - 0.1, sliced["X"].max() + 0.1)
    ax.set_ylim(sliced["Y"].min() - 0.1, sliced["Y"].max() + 0.1)
    ax.grid(False)
    plt.tight_layout()
    out = os.path.join(output_dir, f"{scenario_name}_{wind_name}_wind.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"       → {out}")
    return out


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GADEN Visualizer — world layout + wind field (white background)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 gaden_visualizer.py --scenario MAPIRlab --wind W1
  python3 gaden_visualizer.py --scenario 10x6_maze --wind 1ms
  python3 gaden_visualizer.py --scenario 10x6_empty_room --wind dynamic
  python3 gaden_visualizer.py --scenario MAPIRlab --wind W2 --z 1.5

  Output always saved to:
    /home/gk/ros2_ws/src/gaden/test_env/visualization_output/
        """
    )
    parser.add_argument("--scenario", required=True,
                        help="Scenario folder name  (e.g. MAPIRlab, 10x6_maze)")
    parser.add_argument("--wind",     required=True,
                        help="Wind sub-folder name  (e.g. W1, dynamic, 1ms)")
    parser.add_argument("--z",        type=float, default=1.0,
                        help="Z slice height in metres — height above floor for the wind slice (default: 1.0 = robot/sensor height)")
    parser.add_argument("--output",   default=None,
                        help="Override output folder (default: visualization_output/ inside test_env)")
    parser.add_argument("--base",     default=SCENARIOS_BASE,
                        help=f"Override scenarios base path")
    args = parser.parse_args()

    scenario_path = os.path.join(args.base, args.scenario)
    if not os.path.isdir(scenario_path):
        print(f"\n[ERROR] Scenario not found: {scenario_path}")
        print(f"  Folders inside {args.base}:")
        for d in sorted(os.listdir(args.base)):
            print(f"    {d}")
        sys.exit(1)

    output_dir = args.output or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "visualization_output"
    )
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  GADEN Visualizer")
    print(f"  Scenario  : {args.scenario}")
    print(f"  Wind      : {args.wind}")
    print(f"  Z height  : {args.z} m")
    print(f"  Output    : {output_dir}")
    print(f"{'='*55}\n")

    cfg, cell_size, source = load_config(scenario_path)
    print(f"  Cell size : {cell_size} m")
    print(f"  Source    : {source}\n")

    stl_files = find_stl_files(scenario_path, cfg)
    if not stl_files:
        print("  [warn] No STL files found — world figure will be empty.")
    else:
        print(f"  STL files ({len(stl_files)}):")
        for s in stl_files:
            print(f"    {os.path.basename(s)}")
        print()

    csv_files = find_wind_csv(scenario_path, args.wind)
    csv_path  = csv_files[0]
    print(f"  Wind CSV  : {os.path.basename(csv_path)}")
    if len(csv_files) > 1:
        print(f"  ({len(csv_files)} CSVs found — using first one)\n")

    print()
    plot_world(args.scenario, args.wind, stl_files, source, args.z, output_dir)
    plot_wind(args.scenario, args.wind, csv_path, stl_files,
              cell_size, source, args.z, output_dir)

    print(f"\n✅  Done — both figures in: {output_dir}\n")


if __name__ == "__main__":
    main()
