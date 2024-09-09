# destpat_plotter.py
#
# Plotter for the Destpat gym environment
#
# Author: Giacomo Del Rio
# Creation date: 27 Feb 2023

import itertools
import subprocess
import tempfile
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Union, List, Optional, Dict, Tuple

import numpy as np
import torch

from PIL import Image

from aerialsim.patriot_unit import Patriot
from aerialsim.rafale_unit import Rafale
from environments.destpat.destpat_abstractions import DestpatRealState, UnitInfo, ContactInfo, ContactType, UnitState, \
    DestpatAbstractions, Side
from aerialsim_plotter.aerialsim_plotter import AerialsimPlotter, ColorRGBA, TopLeftMessage, Airplane, StatusMessage, \
    Drawable, SamBattery, Arc, Text, Missile, PolyLine, BackgroundMesh
from environments.destpat.destpat_env_cmo import CmoDestpatEnv
from environments.destpat.destpat_env_sim import SimDestpatEnv
from mcts_common.mcts_policy_torch import TorchMctsPolicy


class DestpatPlotter:

    def __init__(self, map_ext=None):
        self.ffmpeg_exe = r"C:\Program Files\ffmpeg-05-2022\bin\ffmpeg.exe"
        self.codec = "libx264"

        self.map_ext = CmoDestpatEnv.map_limits if map_ext is None else map_ext
        self.red_outline = ColorRGBA(0.8, 0.2, 0.2, 1)
        self.red_fill = ColorRGBA(0.8, 0.2, 0.2, 0.2)
        self.blue_outline = ColorRGBA(0.3, 0.6, 0.9, 0.5)
        self.blue_fill = ColorRGBA(0.3, 0.6, 0.9, 0.2)
        self.yellow_outline = ColorRGBA(0.8, 0.8, 0.2, 1)
        self.yellow_fill = ColorRGBA(0.8, 0.8, 0.2, 0.2)
        self.paths_outline = ColorRGBA(0.8, 0.8, 0.8, 1)

        self.plt = AerialsimPlotter(self.map_ext, dpi=200)

    def plot_state(self, rs: DestpatRealState, filename: Union[str, Path]):
        objects = self._make_state_objects(rs)
        self.plt.to_png(filename, objects)

    def plot_episode(self, rs_l: List[DestpatRealState], filename: Union[str, Path]):
        objects = []

        past_positions: Dict[str, Tuple[float, float]] = {}

        rs_prev: Optional[DestpatRealState] = None
        for rs in rs_l:
            if rs.att_air_1 is None:
                if rs_prev is not None and rs_prev.att_air_1 is not None:
                    objects.append(Text(rs_prev.att_air_1.lat, rs_prev.att_air_1.lon, "KILL", self.red_outline))
            else:
                objects.append(Arc(rs.att_air_1.lat, rs.att_air_1.lon, 500, 0, 360, fill_color=self.red_outline))

            if rs.att_air_2 is None:
                if rs_prev is not None and rs_prev.att_air_2 is not None:
                    objects.append(Text(rs_prev.att_air_2.lat, rs_prev.att_air_2.lon, "KILL", self.red_outline))
            else:
                objects.append(Arc(rs.att_air_2.lat, rs.att_air_2.lon, 500, 0, 360, fill_color=self.red_outline))

            for m in rs.missiles.values():
                if m.unit_id in past_positions:
                    prev_lat, prev_lon = past_positions[m.unit_id]
                    objects.append(
                        PolyLine([(prev_lat, prev_lon), (m.lat, m.lon)], line_width=0.5, edge_color=self.red_outline))
                past_positions[m.unit_id] = m.lat, m.lon

            for c in rs.contacts.values():
                if c.type in [ContactType.UNKNOWN, ContactType.MOBILE]:
                    continue

                if c.type == ContactType.AIR:
                    objects.append(Arc(c.lat, c.lon, 500, 0, 360, fill_color=self.yellow_outline))

                if c.type == ContactType.MISSILE and c.cont_id in past_positions:
                    prev_lat, prev_lon = past_positions[c.cont_id]
                    objects.append(
                        PolyLine([(prev_lat, prev_lon), (c.lat, c.lon)], line_width=0.5,
                                 edge_color=self.yellow_outline))
                past_positions[c.cont_id] = c.lat, c.lon

            rs_prev = rs

        objects.extend(self._make_state_objects(rs_l[-1]))
        self.plt.to_png(filename, objects)

    def plot_rollouts(self, rs_l: List[List[DestpatRealState]], filename: Union[str, Path]):
        objects = []

        # Add trajectories
        for traj in rs_l:
            path = [(x.att_air_1.lat, x.att_air_1.lon) for x in traj if x.att_air_1]
            objects.append(PolyLine(path, line_width=0.8, dash=None, edge_color=self.paths_outline, zorder=1))

            # Add bomb releases
            prev_state = traj[0]
            for step in traj[1:]:
                if prev_state.att_air_1 is not None and step.att_air_1 is not None:
                    if prev_state.att_air_1.a2l > step.att_air_1.a2l:
                        objects.append(Arc(prev_state.att_air_1.lat, prev_state.att_air_1.lon, 1000, 0, 360,
                                           fill_color=self.red_outline, zorder=2))
                    if prev_state.att_air_1.a2a > step.att_air_1.a2a:
                        objects.append(Arc(prev_state.att_air_1.lat, prev_state.att_air_1.lon, 1000, 0, 360,
                                           fill_color=self.yellow_outline, zorder=2))
                prev_state = step

        objects.append(TopLeftMessage(f"N={len(rs_l)}"))

        state_objects = self._make_state_objects(rs_l[0][0])
        objects.extend(state_objects[1:])
        self.plt.to_png(filename, objects)

    def plot_movie(self, rs_l: List[DestpatRealState], filename: Union[str, Path], fps: int = 1, traces: bool = True):
        with tempfile.TemporaryDirectory() as tmp_dir:
            frames = [i for i in range(len(rs_l))] + [len(rs_l) - 1]  # Double last frame
            for i, f in enumerate(frames):
                if traces:
                    self.plot_episode(rs_l[:f + 1], Path(tmp_dir) / f"frame_{i + 1}.png")
                else:
                    self.plot_state(rs_l[f], Path(tmp_dir) / f"frame_{i + 1}.png")

            cmd_line = [f"{self.ffmpeg_exe}", "-y", "-r", f"{fps}", "-start_number", "1",
                        "-i", f"{tmp_dir}\\frame_%d.png", "-vcodec", f"{self.codec}",
                        f"{filename}"]
            subprocess.run(cmd_line, check=True, capture_output=True)

    def plot_policy_sparse_1(self, policy: TorchMctsPolicy, sample_rs: DestpatRealState, obs_enc: str, n_cells: int,
                             out_file: Path):
        map_limits = self.map_ext
        n_dots = 50
        headings = [0, 90, 180, 270]

        # Create latitude/longitude grid in which calculate the value
        lons = np.linspace(map_limits.left_lon, map_limits.right_lon, n_dots + 1)[:-1] \
               + map_limits.longitude_extent() / (n_dots * 2)
        lats = np.linspace(map_limits.bottom_lat, map_limits.top_lat, n_dots + 1)[:-1] \
               + map_limits.latitude_extent() / (n_dots * 2)

        # Make a list of abstract states
        img_states: List[List[np.ndarray]] = []
        for sam_damaged, a2l_qty in [(False, 1), (False, 0), (True, 0)]:
            states: List[List] = []
            for heading in headings:
                head_s = []
                for lat, lon in itertools.product(lats, lons):
                    rs = DestpatRealState(
                        att_air_1=UnitInfo("ID0", Side.RED, lat, lon, 1000, heading, 500, a2a=0, a2l=a2l_qty, state="Ok"),
                        att_air_2=None,
                        def_air_1=None,
                        def_sam=sample_rs.def_sam, contacts=[], missiles=[], current_time=datetime.now(),
                        initial_time=datetime.now())
                    head_s.append(SimDestpatEnv.abstract_state(obs_enc, n_cells, [rs], bad_firing=False))
                states.append(head_s)
            states_array: List[np.ndarray] = [np.array(x) for x in states]
            img_states.append(states_array)

        # Evaluate abstract states
        img_values: List[torch.tensor] = []
        for states_array in img_states:
            values = []
            for s in states_array:
                with torch.no_grad():
                    _, v = policy.net(torch.tensor(s, dtype=torch.float))
                    values.append(v)
            values_avg = torch.stack(values).mean(dim=0)
            img_values.append(values_avg)

        # Prepare drawables for visual recognition
        objects: List[Drawable] = [
            SamBattery(sample_rs.def_sam.lat, sample_rs.def_sam.lon, sample_rs.def_sam.heading,
                       missile_range=Patriot.missile_range, radar_range=Patriot.radar_range,
                       radar_amplitude_deg=Patriot.radar_width,
                       edge_color=self.blue_outline, fill_color=self.blue_fill, zorder=1)
        ]

        # Plot a map for each case
        global_min, global_max = min([v.min() for v in img_values]), max([v.max() for v in img_values])
        file_names = []
        for case, values_avg in zip(["Ok-1", "Ok-0", "Dam-0"], img_values):
            file_names.append(str(out_file) + f"_{case}.png")
            mesh = BackgroundMesh(lons, lats, torch.reshape(values_avg, (n_dots, n_dots)),
                                  cmap='copper', vmin=global_min, vmax=max(global_min + 0.1, global_max))
            tree_plotter = AerialsimPlotter(map_extents=map_limits, dpi=200, background_mesh=mesh)
            tree_plotter.to_png(file_names[-1],
                                objects + [
                                    TopLeftMessage(f"Case={case}, Min_v={global_min:.4f}, Max_v={global_max:.4f}")])

        # Combine the plots in one image
        images = [Image.open(x) for x in file_names]
        width, height = images[0].size
        composed_im = Image.new('RGB', (width * 2, height * 2))
        composed_im.paste(images[0], (0, 0))
        composed_im.paste(images[1], (width, 0))
        # composed_im.paste(images[3], (0, height))
        composed_im.paste(images[2], (width, height))
        composed_im.save(str(out_file))

        # Delete single maps
        for i in file_names:
            Path(i).unlink()

    def plot_abstract_state_sparse_3(self, sample_rs: DestpatRealState, state: np.ndarray, n_cells: int,
                                     out_file: Path):
        map_limits = self.map_ext
        n_dots = n_cells

        # Create latitude/longitude grid for cmap
        lons = np.linspace(map_limits.left_lon, map_limits.right_lon, n_dots + 1)[:-1] \
               + map_limits.longitude_extent() / (n_dots * 2)
        lats = np.linspace(map_limits.bottom_lat, map_limits.top_lat, n_dots + 1)[:-1] \
               + map_limits.latitude_extent() / (n_dots * 2)

        # Prepare drawables for visual recognition
        objects: List[Drawable] = [
            SamBattery(sample_rs.def_sam.lat, sample_rs.def_sam.lon, sample_rs.def_sam.heading,
                       missile_range=Patriot.missile_range, radar_range=Patriot.radar_range,
                       radar_amplitude_deg=Patriot.radar_width,
                       edge_color=self.blue_outline, fill_color=self.blue_fill, zorder=1)
        ]

        # Plot a map for each layer
        if len(state.shape) == 1:
            state = state.reshape((4, n_cells, n_cells))
        layers = ["Airplane", "Missile", "Sam", "Flags"]
        file_names = []
        for i in range(state.shape[0]):
            file_names.append(str(out_file) + f"_{layers[i]}.png")
            mesh = BackgroundMesh(lons, lats, state[i, :, :], cmap='copper', vmin=-0.1, vmax=1.1)
            tree_plotter = AerialsimPlotter(map_extents=map_limits, dpi=200, background_mesh=mesh)
            tree_plotter.to_png(file_names[-1],
                                objects + [TopLeftMessage(f"{layers[i]}")])

        # Combine the plots in one image
        images = [Image.open(x) for x in file_names]
        width, height = images[0].size
        composed_im = Image.new('RGB', (width * 2, height * 2))
        composed_im.paste(images[0], (0, 0))
        composed_im.paste(images[1], (width, 0))
        composed_im.paste(images[2], (0, height))
        composed_im.paste(images[3], (width, height))
        composed_im.save(str(out_file))

        # Delete single maps
        for i in file_names:
            Path(i).unlink()

    def plot_policy_full(self, policy: TorchMctsPolicy, sample_rs: DestpatRealState, da: DestpatAbstractions,
                         n_dots: int, out_file: Path):
        map_limits = self.map_ext
        scenarios = [
            ("Ok-1", False, 1),  # SAM not damaged, 1 missile(s) remaining
            ("Ok-0", False, 0),  # SAM not damaged, 0 missile(s) remaining
        ]

        # Create latitude/longitude grid in which calculate the value
        lons = np.linspace(map_limits.left_lon,
                           map_limits.right_lon, n_dots + 1)[:-1] + map_limits.longitude_extent() / (n_dots * 2)
        lats = np.linspace(map_limits.bottom_lat,
                           map_limits.top_lat, n_dots + 1)[:-1] + map_limits.latitude_extent() / (n_dots * 2)

        # Make a list of observations for each scenario
        scenarios_obs: Dict[str, List[np.ndarray]] = {}
        for scenario_name, sam_damaged, a2l_qty in scenarios:
            observations: List[np.ndarray] = []
            for lat, lon in itertools.product(lats, lons):
                rs = DestpatRealState(
                    att_air_1=UnitInfo("ID0", Side.RED, lat, lon, 1000, 90, 500, a2a=0, a2l=a2l_qty,
                                       state=UnitState.OK),
                    att_air_2=None,
                    def_air_1=None,
                    def_sam=UnitInfo("ID1", Side.BLUE, sample_rs.def_sam.lat, sample_rs.def_sam.lon,
                                     sample_rs.def_sam.alt, sample_rs.def_sam.heading, speed=0, a2a=0, a2l=0,
                                     state=UnitState.DAMAGED if sam_damaged else UnitState.OK),
                    contacts=OrderedDict(),
                    missiles={},
                    current_time=datetime.now(),
                    initial_time=datetime.now(),
                    events={})
                observations.append(da.abstract_state([rs]))
            scenarios_obs[scenario_name] = observations

        # Evaluate observations for each scenario
        scenarios_values: Dict[str, torch.tensor] = {}
        for scenario_name, _, _ in scenarios:
            with torch.no_grad():
                _, v = policy.net(torch.tensor(np.stack(scenarios_obs[scenario_name]), dtype=torch.float))
                scenarios_values[scenario_name] = v

        # Prepare drawables for visual recognition
        objects: List[Drawable] = [
            SamBattery(sample_rs.def_sam.lat, sample_rs.def_sam.lon, sample_rs.def_sam.heading,
                       missile_range=Patriot.missile_range, radar_range=Patriot.radar_range,
                       radar_amplitude_deg=Patriot.radar_width,
                       edge_color=self.blue_outline, fill_color=self.blue_fill, zorder=1)
        ]

        # Plot a map for each scenario
        global_min = min([v.min() for v in scenarios_values.values()])
        global_max = max([v.max() for v in scenarios_values.values()])
        file_names = []
        for scenario_name, _, _ in scenarios:
            file_names.append(str(out_file) + f"_{scenario_name}.png")
            mesh = BackgroundMesh(lons, lats, torch.reshape(scenarios_values[scenario_name], (n_dots, n_dots)),
                                  cmap='copper', vmin=global_min, vmax=global_max)
            as_plotter = AerialsimPlotter(map_extents=map_limits, dpi=200, background_mesh=mesh)
            as_plotter.to_png(file_names[-1],
                              objects + [
                                  TopLeftMessage(
                                      f"Scenario={scenario_name}, "
                                      f"Min_v={scenarios_values[scenario_name].min():.4f}, "
                                      f"Max_v={scenarios_values[scenario_name].max():.4f}")])

        # Combine the plots in one image
        images = [Image.open(x) for x in file_names]
        width, height = images[0].size
        composed_im = Image.new('RGB', (width * 2, height))
        composed_im.paste(images[0], (0, 0))
        composed_im.paste(images[1], (width, 0))
        composed_im.save(str(out_file))

        # Delete single maps
        for i in file_names:
            Path(i).unlink()

    @staticmethod
    def plot_abstract_state_full(o: np.ndarray, out_file: Path, dpi=400):
        n_layers = o.shape[0]
        n_cells = o.shape[1]

        fig, axs = plt.subplots(ncols=n_layers)
        if not isinstance(axs, list):
            axs = [axs]
        for i in range(n_layers):
            axs[i].imshow(o[i], interpolation='none', vmin=-1, vmax=1, cmap='RdYlGn')

        for i in range(n_layers):
            for side in ['top', 'bottom', 'left', 'right']:
                axs[i].spines[side].set(lw=0.4)
            axs[i].set_xlim([-1, n_cells])
            axs[i].set_ylim(n_cells, -1)
            axs[i].set_yticks(np.arange(0, n_cells + 1, step=5, ))
            axs[i].set_yticks(np.linspace(-0.5, n_cells - 0.5, n_cells + 1, endpoint=True), minor=True)
            axs[i].grid(True, which='minor', linestyle='-', linewidth=0.1)
            axs[i].set_xticks(np.arange(0, n_cells + 1, step=5, ))
            axs[i].set_xticks(np.linspace(-0.5, n_cells - 0.5, n_cells + 1, endpoint=True), minor=True)
            axs[i].tick_params(which='major', labelsize=3, width=0.4, pad=1)
            axs[i].tick_params(which='minor', labelsize=2, width=0.0)
        plt.savefig(out_file, bbox_inches='tight', dpi=dpi)
        plt.close()

    def _make_state_objects(self, rs: DestpatRealState) -> List[Drawable]:
        status_message = ""
        objects = [TopLeftMessage(rs.current_time.strftime("%Y %b %d %H:%M:%S"))]

        if rs.att_air_1:
            objects.extend(
                self._make_airplane(rs.att_air_1,
                                    text="ed" if rs.att_air_1.state == UnitState.ENGAGED_DEFENSIVE else None))

        if rs.att_air_2:
            objects.extend(
                self._make_airplane(rs.att_air_2,
                                    text="ed" if rs.att_air_2.state == UnitState.ENGAGED_DEFENSIVE else None))

        if rs.def_air_1:
            objects.extend(self._make_airplane(rs.def_air_1))

        if rs.def_sam:
            objects.extend(self._make_sam_battery(rs.def_sam,
                                                  text="Damaged" if rs.def_sam.state == UnitState.DAMAGED else None))

        for c in rs.contacts.values():
            if c.lat != 0 and c.lon != 0:
                objects.extend(self._make_contact(c))

        for u in rs.missiles.values():
            if u.lat != 0 and u.lon != 0:
                objects.extend(self._make_missile(u))

        if len(status_message) > 0:
            objects.append(StatusMessage(status_message))
        return objects

    def _make_contact(self, c: ContactInfo) -> List[Drawable]:
        objects = [Arc(c.lat, c.lon, 2_500, 0, 360, line_width=1, edge_color=self.yellow_outline, fill_color=None)]
        if c.type == ContactType.UNKNOWN:
            objects.append(Text(c.lat, c.lon, "?", text_color=self.yellow_outline))
        elif c.type == ContactType.MOBILE:
            objects.append(Text(c.lat, c.lon, "P", text_color=self.yellow_outline))
        elif c.type == ContactType.AIR:
            objects.append(Text(c.lat, c.lon, "A", text_color=self.yellow_outline))
        elif c.type == ContactType.MISSILE:
            objects[0].radius = 1_600
            objects.append(Text(c.lat, c.lon, "m", text_color=self.yellow_outline))
        return objects

    def _make_missile(self, ui: UnitInfo) -> List[Drawable]:
        c_out, c_fill = self._get_color(ui.side)
        return [Missile(ui.lat, ui.lon, ui.heading, edge_color=c_out, fill_color=c_fill, zorder=1)]

    def _make_airplane(self, ui: UnitInfo, text: Optional[str] = None) -> List[Drawable]:
        c_out, c_fill = self._get_color(ui.side)
        return [Airplane(ui.lat, ui.lon, ui.heading, edge_color=c_out, fill_color=c_fill,
                         radar=(Rafale.radar_range, Rafale.radar_width), info_text=text, zorder=1)]

    def _make_sam_battery(self, ui: UnitInfo, text: Optional[str] = None) -> List[Drawable]:
        c_out, c_fill = self._get_color(ui.side)
        return [SamBattery(ui.lat, ui.lon, ui.heading,
                           missile_range=Patriot.missile_range, radar_range=Patriot.radar_range,
                           radar_amplitude_deg=Patriot.radar_width,
                           edge_color=c_out, fill_color=c_fill, info_text=text, zorder=0)]

    def _get_color(self, side: Side):
        return (self.red_outline, self.red_fill) if side == Side.RED else (self.blue_outline, self.blue_fill)
