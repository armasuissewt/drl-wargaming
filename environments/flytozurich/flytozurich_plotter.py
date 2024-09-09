# flytozurich_plotter.py
#
# Plotter for the [Sim|Cmo]FlyToZurichEnv gym environment
#
# Author: Giacomo Del Rio
# Creation date: 10 November 2021

from collections import defaultdict
from pathlib import Path
from typing import Union, Optional, Dict, Tuple, List, Callable

import numpy as np

from aerialsim.patriot_unit import Patriot
from aerialsim_plotter.aerialsim_plotter import AERIAL_COLORS as COLORS
from aerialsim_plotter.aerialsim_plotter import TopLeftMessage, AerialsimPlotter, PolyLine, Airplane, Drawable, \
    SamBattery, Waypoint, Missile, StatusMessage, ColorRGBA, Rect
from environments.flytozurich.flytozurich_env_cmo import CmoFlyToZurichEnv
from environments.flytozurich.flytozurich_env_sim import FtzRealState
from simulator.map_limits import MapLimits


class FlyToZurichEnvPlotter:
    def __init__(self, map_ext: MapLimits = None):
        self.ffmpeg_exe = r"C:\Program Files\ffmpeg-05-2022\bin\ffmpeg.exe"
        self.codec = "libx264"

        self.map_ext = CmoFlyToZurichEnv.map_limits if map_ext is None else map_ext
        self.red_outline = ColorRGBA(0.8, 0.2, 0.2, 1)
        self.red_fill = ColorRGBA(0.8, 0.2, 0.2, 0.2)
        self.blue_outline = ColorRGBA(0.3, 0.6, 0.9, 0.5)
        self.blue_fill = ColorRGBA(0.3, 0.6, 0.9, 0.2)
        self.yellow_outline = ColorRGBA(0.8, 0.8, 0.2, 1)
        self.yellow_fill = ColorRGBA(0.8, 0.8, 0.2, 0.2)
        self.paths_outline = ColorRGBA(0.8, 0.8, 0.8, 1)

        self.plt = AerialsimPlotter(self.map_ext, dpi=200)

    def plot_state(self, rs: FtzRealState, filename: Union[str, Path],
                   paths: Optional[Dict[str, List[Tuple[float, float]]]] = None):
        status_message = ""
        objects = [TopLeftMessage(f"Seconds: {rs.elapsed_time}")]

        # Append paths
        if paths:
            if rs.airplane:
                objects.append(
                    PolyLine(paths[rs.airplane.id], line_width=1, dash=(2, 2), edge_color=COLORS['red_outline']))
            for m in rs.missiles:
                objects.append(PolyLine(paths[m.id], line_width=1, dash=(2, 2), edge_color=COLORS['blue_outline']))

        # Append units
        if rs.airplane:
            objects.append(Airplane(rs.airplane.lat, rs.airplane.lon, rs.airplane.heading,
                                    edge_color=COLORS['red_outline'], fill_color=COLORS['red_fill'],
                                    info_text="Rafale"))
            if rs.airplane.state == "EngagedDefensive":
                status_message += "Aircraft engaged"

        if rs.sam:
            objects.append(SamBattery(rs.sam.lat, rs.sam.lon, rs.sam.heading, missile_range=Patriot.missile_range,
                                      radar_range=Patriot.radar_range, radar_amplitude_deg=Patriot.radar_width,
                                      edge_color=COLORS['blue_outline'], fill_color=COLORS['blue_fill'],
                                      info_text="Patriot"))
        if rs.target:
            objects.append(Waypoint(rs.target.lat, rs.target.lon, edge_color=COLORS['yellow_outline'],
                                    fill_color=COLORS['yellow_fill'], info_text="Target"))
        for m in rs.missiles:
            objects.append(
                Missile(m.lat, m.lon, m.heading, edge_color=COLORS['blue_outline'], fill_color=COLORS['blue_fill'],
                        info_text=None, zorder=0))
        objects.append(StatusMessage(status_message))

        self.plt.to_png(filename, objects)

    def plot_episode(self, trajectory: List[FtzRealState], out_file: Path):
        objects: List[Drawable] = []

        # Add battery and target
        rs: FtzRealState = trajectory[0]
        objects.append(SamBattery(rs.sam.lat, rs.sam.lon, rs.sam.heading,
                                  missile_range=Patriot.missile_range,
                                  radar_range=Patriot.radar_range, radar_amplitude_deg=Patriot.radar_width,
                                  edge_color=self.blue_outline, fill_color=self.blue_fill, zorder=2))
        objects.append(
            Waypoint(rs.target.lat, rs.target.lon, self.yellow_outline, self.yellow_fill, zorder=2))

        # Add airplane trajectory
        path = [(x.airplane.lat, x.airplane.lon) for x in trajectory if x.airplane]
        objects.append(PolyLine(path, line_width=1, dash=None, edge_color=self.red_outline, zorder=1))
        objects.append(
            Airplane(rs.airplane.lat, rs.airplane.lon, rs.airplane.heading, self.red_outline, self.red_fill, zorder=2))

        # Add missiles trajectories
        missile_paths = defaultdict(list)
        for rs in trajectory:
            for m in rs.missiles:
                missile_paths[m.id].append((m.lat, m.lon, m.heading))
        for _, path in missile_paths.items():
            objects.append(PolyLine([(x[0], x[1]) for x in path], line_width=1, dash=(2, 2),
                                    edge_color=self.blue_outline, zorder=1))
            objects.append(Missile(path[-1][0], path[-1][1], path[-1][2], edge_color=self.blue_outline,
                                   fill_color=self.blue_fill, zorder=2))

        # Plot
        self.plt.to_png(out_file, objects)

    def plot_airplane_paths(self, paths: List[List[Tuple[float, float]]], filename: Union[str, Path]):
        objects: List[Drawable] = []

        # Append paths
        for p in paths:
            objects.append(
                PolyLine(p, line_width=1, dash=(2, 2), edge_color=ColorRGBA(1, 1, 1, .1)))

        objects.append(StatusMessage("Airplane paths"))

        self.plt.to_png(filename, objects)

    def plot_heat_map(self, fn: Callable[[float, float], float], subdiv: int, filename: Union[str, Path]):
        # Get values
        step_lon = (self.map_ext.right_lon - self.map_ext.left_lon) / subdiv
        step_lat = (self.map_ext.top_lat - self.map_ext.bottom_lat) / (subdiv // 2)
        buffer = np.empty(shape=(subdiv, subdiv // 2))
        for lon_id, lon in enumerate(
                np.linspace(self.map_ext.left_lon, self.map_ext.right_lon, num=subdiv, endpoint=False)):
            for lat_id, lat in enumerate(
                    np.linspace(self.map_ext.bottom_lat, self.map_ext.top_lat, num=subdiv // 2, endpoint=False)):
                buffer[lon_id, lat_id] = fn(lat + step_lat / 2.0, lon + step_lon / 2.0)

        # Normalize values
        buffer = (buffer - np.min(buffer)) / (np.max(buffer) - np.min(buffer))

        # Print values
        objects: List[Drawable] = []
        for lon_id, lon in enumerate(
                np.linspace(self.map_ext.left_lon, self.map_ext.right_lon, num=subdiv, endpoint=False)):
            for lat_id, lat in enumerate(
                    np.linspace(self.map_ext.bottom_lat, self.map_ext.top_lat, num=subdiv // 2, endpoint=False)):
                color = ColorRGBA(buffer[lon_id, lat_id], buffer[lon_id, lat_id], buffer[lon_id, lat_id], 0.9)
                objects.append(Rect(lon, lat, lon + step_lon, lat + step_lat, edge_color=ColorRGBA(1, 1, 1, 0),
                                    fill_color=color))

        self.plt.to_png(filename, objects)
