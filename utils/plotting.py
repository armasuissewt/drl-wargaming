# plotting.py
#
# Examples of tree plotting
#
# Author: Stefano Toniolo, Giacomo Del Rio

import itertools
import math
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image

from aerialsim.patriot_unit import Patriot
from aerialsim_plotter.aerialsim_plotter import AERIAL_COLORS as COLORS, ColorRGBA, Drawable, SamBattery, Waypoint, \
    PolyLine, AerialsimPlotter, BackgroundMesh, TopLeftMessage, Airplane
from environments.flytozurich.flytozurich_env_sim import FtzRealState, UnitInfo
from environments.flytozurich.wrappers.flytozurich_sim_wrappers import SimFtzObsRewAdapterEnv
from mcts_common.mcts_policy_torch import TorchMctsPolicy
from simulator.map_limits import MapLimits

# Plotting variables
paths_outline = ColorRGBA(0.98, 0.964, 0.509, 1)

DEG_TO_RAD = math.pi / 180


def plot_value_map_dir(policy: TorchMctsPolicy, sample_rs, obs_enc: str, n_cells: int, out_file: Path):
    map_limits = MapLimits(left_lon=5.1, bottom_lat=45.4, right_lon=11.1, top_lat=47.9)
    n_dots = 100
    headings = [0, 90, 180, 270]

    # Create latitude/longitude grid in which calculate the value
    lons = np.linspace(map_limits.left_lon, map_limits.right_lon, n_dots + 1)[:-1] \
           + map_limits.longitude_extent() / (n_dots * 2)
    lats = np.linspace(map_limits.bottom_lat, map_limits.top_lat, n_dots + 1)[:-1] \
           + map_limits.latitude_extent() / (n_dots * 2)

    # Make a list of abstract states
    states: List[List] = []
    for heading in headings:
        head_s = []
        for lat, lon in itertools.product(lats, lons):
            rs = FtzRealState(airplane=UnitInfo(0, lat, lon, 500, heading, 0, ''),
                              sam=None, target=None, missiles=[], elapsed_time=0)
            head_s.append(SimFtzObsRewAdapterEnv.abstract_state_inner(rs, obs_enc=obs_enc, n_cells=n_cells))
        states.append(head_s)
    states_array = [np.array(x) for x in states]

    # Evaluate abstract states
    values = []
    for s in states_array:
        with torch.no_grad():
            _, v = policy.net(torch.tensor(s, dtype=torch.float))
            values.append(v)

    # Prepare drawables for visual recognition
    objects: List[Drawable] = [
        SamBattery(sample_rs.sam.lat, sample_rs.sam.lon, sample_rs.sam.heading,
                   missile_range=Patriot.missile_range, radar_range=Patriot.radar_range,
                   radar_amplitude_deg=Patriot.radar_width,
                   edge_color=COLORS['blue_outline'], fill_color=COLORS['blue_fill'], zorder=2),
        Waypoint(sample_rs.target.lat, sample_rs.target.lon, COLORS['yellow_outline'],
                 COLORS['yellow_fill'], zorder=2)
    ]

    # Plot a map for each direction
    global_min, global_max = min([v.min() for v in values]), max([v.max() for v in values])
    file_names = []
    for i, heading in enumerate(headings):
        file_names.append(str(out_file) + f"_{heading}.png")
        v = values[i]
        v_min, v_max = v.min(), v.max()
        mesh = BackgroundMesh(lons, lats, torch.reshape(v, (n_dots, n_dots)),
                              cmap='copper', vmin=global_min, vmax=max(global_min + 0.1, global_max))
        tree_plotter = AerialsimPlotter(map_extents=map_limits, dpi=200, background_mesh=mesh)
        tree_plotter.to_png(file_names[-1],
                            objects + [
                                Airplane(sample_rs.airplane.lat, sample_rs.airplane.lon, heading,
                                         COLORS['red_outline'], COLORS['red_fill'], zorder=2),
                                TopLeftMessage(f"D={heading}, Min_v={v_min:.4f}, Max_v={v_max:.4f}")])

    # Combine the plots in one image
    images = [Image.open(x) for x in file_names[:4]]
    width, height = images[0].size
    composed_im = Image.new('RGB', (width * 2, height * 2))
    composed_im.paste(images[0], (0, 0))
    composed_im.paste(images[1], (width, 0))
    composed_im.paste(images[3], (0, height))
    composed_im.paste(images[2], (width, height))
    composed_im.save(str(out_file))

    # Delete single maps
    for i in file_names:
        Path(i).unlink()


def plot_episode(trajectory: List[FtzRealState], out_file: Path):
    objects: List[Drawable] = []

    # Add battery and target
    rs: FtzRealState = trajectory[0]
    objects.append(SamBattery(rs.sam.lat, rs.sam.lon, rs.sam.heading,
                              missile_range=Patriot.missile_range,
                              radar_range=Patriot.radar_range, radar_amplitude_deg=Patriot.radar_width,
                              edge_color=COLORS['blue_outline'], fill_color=COLORS['blue_fill'], zorder=2))
    objects.append(Waypoint(rs.target.lat, rs.target.lon, COLORS['yellow_outline'], COLORS['yellow_fill'], zorder=2))

    # Add airplane trajectories
    path = [(x.airplane.lat, x.airplane.lon) for x in trajectory if x.airplane]
    objects.append(PolyLine(path, line_width=2, dash=None, edge_color=paths_outline, zorder=1))
    objects.append(
        Airplane(rs.airplane.lat, rs.airplane.lon, rs.airplane.heading, COLORS['red_outline'], COLORS['red_fill'],
                 zorder=2))

    # Plot
    map_limits = MapLimits(4, 45, 12, 48.5)
    tree_plotter = AerialsimPlotter(map_extents=map_limits, dpi=400)
    tree_plotter.to_png(str(out_file), objects)
