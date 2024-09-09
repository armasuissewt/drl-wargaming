# paveway_missile.py
#
# A Paveway missile unit
#
# Author: Giacomo Del Rio
# Creation date: 14 May 2023

from typing import List

import numpy as np
from scipy.interpolate import interp1d

from aerialsim.aerialsim_base import UnitDestroyedEvent, ASUnit, UnitDamagedEvent
from aerialsim.missile_base_unit import MissileBase
from simulator.dte_simulator import Pose, Event, DteSimulator
from simulator.units_of_measure import Seconds, MetersPerSecond, Meters, DegreesPerSecond


class PavewayMissile(MissileBase):
    """ This kind of missile has a GPS guidance system that allows a fire-and-forget behavior. """

    lifetime: Seconds = 120
    max_heading_change: DegreesPerSecond = 10
    max_altitude_change: MetersPerSecond = 500
    speed_profile_time: List[Seconds] = np.array([0])
    hit_range: Meters = 500
    hit_max_altitude_delta: Meters = 300
    damage: float = 10

    def __init__(self, pose: Pose, side: int, airborn_time: Seconds, initial_speed: MetersPerSecond, target: ASUnit):
        super().__init__(pose, side, airborn_time, PavewayMissile.lifetime, target,
                         PavewayMissile._make_speed_profile(initial_speed * 1.2), PavewayMissile.max_heading_change,
                         PavewayMissile.max_altitude_change, PavewayMissile.hit_range,
                         PavewayMissile.hit_max_altitude_delta)

    @staticmethod
    def _make_speed_profile(initial_speed: MetersPerSecond) -> interp1d:
        return interp1d(PavewayMissile.speed_profile_time, np.array([initial_speed]), kind='linear',
                        assume_sorted=True, bounds_error=False, fill_value=(initial_speed, initial_speed))

    def update(self, time_elapsed: Seconds, sim: DteSimulator) -> List[Event]:
        self.set_heading(self.bearing_to(self.target))
        target_dst = self.distance_to(self.target)
        delta_alt = self.pose.alt - self.target.pose.alt
        time_to_target = target_dst / self.speed
        if time_to_target > 5:
            self.set_altitude(self.pose.alt - delta_alt / time_to_target * sim.tick_secs)
        else:
            self.set_altitude(self.target.pose.alt)

        return super().update(time_elapsed, sim)

    def target_hit_action(self, sim: DteSimulator, target_distance: Meters) -> List[Event]:
        sim.remove_unit(self.id)
        self.target.health -= PavewayMissile.damage
        if self.target.health <= 0:
            sim.remove_unit(self.target.id)
            return [UnitDestroyedEvent(self, self.target)]
        else:
            return [UnitDamagedEvent(self, self.target)]

    def __str__(self):
        return f"Paveway[{self.id}]"
