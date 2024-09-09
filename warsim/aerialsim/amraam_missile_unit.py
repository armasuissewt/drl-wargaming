# amraam_missile_unit.py
#
# A air-to-air AMRAAM missile unit
#
# Author: Giacomo Del Rio
# Creation date: 19 May 2023

from typing import List

import numpy as np
from scipy.interpolate import interp1d

from aerialsim.aerialsim_base import ASUnit
from aerialsim.missile_base_unit import MissileBase
from simulator.dte_simulator import Pose
from simulator.units_of_measure import Seconds, MetersPerSecond, Meters, DegreesPerSecond


class AmraamMissile(MissileBase):
    max_heading_change: DegreesPerSecond = 20
    max_altitude_change: MetersPerSecond = 300
    speed_profile_time: List[Seconds] = np.array([0, 27, 240])
    speed_profile_speed: List[MetersPerSecond] = np.array([900, 1_372, 350])
    hit_range: Meters = 500
    hit_max_altitude_delta: Meters = 200

    def __init__(self, pose: Pose, side: int, airborn_time: Seconds, target: ASUnit):
        super().__init__(pose, side, airborn_time, AmraamMissile.speed_profile_speed[-1], target,
                         AmraamMissile._make_speed_profile(), AmraamMissile.max_heading_change,
                         AmraamMissile.max_altitude_change, AmraamMissile.hit_range,
                         AmraamMissile.hit_max_altitude_delta)

    @staticmethod
    def _make_speed_profile() -> interp1d:
        return interp1d(AmraamMissile.speed_profile_time, AmraamMissile.speed_profile_speed, kind='quadratic',
                        assume_sorted=True, bounds_error=False,
                        fill_value=(AmraamMissile.speed_profile_speed[0], AmraamMissile.speed_profile_speed[-1]))

    def __str__(self):
        return f"Amraam[{self.id}]"
