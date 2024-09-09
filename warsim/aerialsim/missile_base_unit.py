# missile_base_unit.py
#
# Base class for all the missile units
#
# Author: Giacomo Del Rio
# Creation date: 24 May 2023

from typing import List

from scipy.interpolate import interp1d

from aerialsim.aerialsim_base import UnitDestroyedEvent, ASUnit
from simulator.angles import signed_heading_diff
from simulator.dte_simulator import Pose, DteSimulator, Event
from simulator.geodesics import geodetic_direct
from simulator.units_of_measure import Seconds, MetersPerSecond, Degree, Meters, DegreesPerSecond


class MissileBase(ASUnit):
    health: float = 10

    def __init__(self, pose: Pose, side: int, airborn_time: Seconds, lifetime: Seconds, target: ASUnit,
                 speed_profile: interp1d, max_heading_change: DegreesPerSecond, max_altitude_change: MetersPerSecond,
                 hit_range: Meters, hit_max_altitude_delta: Meters):
        super().__init__(pose, side, MissileBase.health)
        self.airborn_time: Seconds = airborn_time
        self.lifetime: Seconds = lifetime
        self.target: ASUnit = target
        self.speed_profile = speed_profile
        self.max_heading_change = max_heading_change
        self.max_altitude_change = max_altitude_change
        self.hit_range = hit_range
        self.hit_max_altitude_delta = hit_max_altitude_delta

        self.speed: MetersPerSecond = self.speed_profile(0)
        self.new_heading: Degree = pose.head
        self.new_altitude: Meters = pose.alt

    def set_heading(self, new_heading: Degree):
        assert 0 <= new_heading < 360
        self.new_heading = new_heading

    def set_altitude(self, new_altitude: Meters):
        assert 0 <= new_altitude
        self.new_altitude = new_altitude

    def update(self, time_elapsed: Seconds, sim: DteSimulator) -> List[Event]:
        # Check if the target has been hit
        target_distance = self.distance_to(self.target)
        if target_distance <= self.hit_range and sim.unit_exists(self.target.id):
            if abs(self.pose.alt - self.target.pose.alt) <= self.hit_max_altitude_delta:
                return self.target_hit_action(sim, target_distance)

        # Check if eol is arrived
        life_time = sim.time_elapsed - self.airborn_time
        if life_time >= self.lifetime:
            sim.remove_unit(self.id)
            return []

        # Update heading
        if self.pose.head != self.new_heading:
            delta = signed_heading_diff(self.pose.head, self.new_heading)
            max_deg = self.max_heading_change * time_elapsed
            if abs(delta) <= max_deg:
                self.pose.head = self.new_heading
            else:
                self.pose.head += max_deg if delta >= 0 else -max_deg
                self.pose.head %= 360

        # Update altitude
        if self.pose.alt != self.new_altitude:
            delta = self.new_altitude - self.pose.alt
            max_delta = self.max_altitude_change * time_elapsed
            if abs(delta) <= max_delta:
                self.pose.alt = self.new_altitude
            else:
                self.pose.alt += max_delta if delta >= 0 else -max_delta

        # Update position
        self.pose.lat, self.pose.lon = geodetic_direct(self.pose.lat, self.pose.lon, self.pose.head,
                                                       self.speed * time_elapsed)

        # Update speed
        self.speed = self.speed_profile(life_time)
        return []

    def target_hit_action(self, sim: DteSimulator, target_distance: Meters) -> List[Event]:
        sim.remove_unit(self.id)
        sim.remove_unit(self.target.id)
        return [UnitDestroyedEvent(self, self.target)]
