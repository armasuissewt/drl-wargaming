# aerialsim_plotter_test.py
#
# Test for the AerialsimPlotter
#
# Author: Giacomo Del Rio
# Creation date: 17 May 2023

from datetime import datetime

from aerialsim_plotter.aerialsim_plotter import PlotConfig, AerialsimPlotter, StatusMessage, \
    TopLeftMessage, Airplane, SamBattery, Missile, Waypoint, PolyLine, Arc, AERIAL_COLORS, SmallMissile
from simulator.map_limits import MapLimits

map_extents = MapLimits(4, 45, 12, 48.5)

plt_cfg = PlotConfig()
plt = AerialsimPlotter(map_extents, dpi=200, config=plt_cfg)

objects = [
    StatusMessage("Time to go"),
    TopLeftMessage(datetime.now().strftime("%Y %b %d %H:%M:%S")),
    Airplane(47.3099374631706, 5.2946515109602, 86, edge_color=AERIAL_COLORS['red_outline'],
             fill_color=AERIAL_COLORS['red_fill'], info_text="Rafale", zorder=0),
    SamBattery(47.1335734412755, 7.06062200381671, 245, missile_range=111, radar_range=140,
               radar_amplitude_deg=140, edge_color=AERIAL_COLORS['blue_outline'], fill_color=AERIAL_COLORS['blue_fill'],
               info_text="Patriot", zorder=0),
    Missile(47.0, 6.0, 300, edge_color=AERIAL_COLORS['blue_outline'], fill_color=AERIAL_COLORS['blue_fill'],
            info_text="missile", zorder=0),
    SmallMissile(47.5, 6.5, 100, edge_color=AERIAL_COLORS['red_outline'], fill_color=AERIAL_COLORS['red_fill'],
                 info_text="small missile", zorder=0),
    Waypoint(47.374167, 8.648056, edge_color=AERIAL_COLORS['yellow_outline'],
             fill_color=AERIAL_COLORS['yellow_fill'], info_text="Target", zorder=0),
    PolyLine([(47, 8), (47.5, 9), (47.6, 10)], line_width=2, dash=(2, 2), edge_color=AERIAL_COLORS['red_outline'],
             zorder=0),
    Arc(47.3099374631706, 5.2946515109602, 14_000, 60, 114, line_width=1, dash=None,
        edge_color=AERIAL_COLORS['red_outline'], fill_color=None, zorder=0),
    Arc(47.3099374631706, 5.2946515109602, 14_000, -50, 50, line_width=1, dash=None, edge_color=None,
        fill_color=AERIAL_COLORS['red_outline'], zorder=0)
]
plt.to_png("sample_out.png", objects)
