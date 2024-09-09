# pac3_missile_unit.py
#
# A patriot PAC-3 missile unit
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


class Pac3Missile(MissileBase):
    max_heading_change: DegreesPerSecond = 20
    max_altitude_change: MetersPerSecond = 300
    speed_profile_time: List[Seconds] = np.array([0, 16, 36, 96, 156, 216, 276, 336])
    speed_profile_speed: List[MetersPerSecond] = np.array(
        [150 * 0.514444, 2300 * 0.514444, 2300 * 0.514444, 1238 * 0.514444, 724 * 0.514444, 510 * 0.514444,
         383 * 0.514444, 350 * 0.514444])  # Max speed was 2650
    hit_range: Meters = 1_000
    hit_max_altitude_delta: Meters = 500

    def __init__(self, pose: Pose, side: int, airborn_time: Seconds, target: ASUnit):
        super().__init__(pose, side, airborn_time, Pac3Missile.speed_profile_speed[-1], target,
                         Pac3Missile._make_speed_profile(), Pac3Missile.max_heading_change,
                         Pac3Missile.max_altitude_change, Pac3Missile.hit_range,
                         Pac3Missile.hit_max_altitude_delta)

    @staticmethod
    def _make_speed_profile() -> interp1d:
        return interp1d(Pac3Missile.speed_profile_time, Pac3Missile.speed_profile_speed, kind='quadratic',
                        assume_sorted=True, bounds_error=False,
                        fill_value=(Pac3Missile.speed_profile_speed[0], Pac3Missile.speed_profile_speed[-1]))

    def __str__(self):
        return f"Pac3[{self.id}]"
