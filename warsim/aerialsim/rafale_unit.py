# rafale_unit.py
#
# A Rafale airplane unit
#
# Author: Giacomo Del Rio
# Creation date: 19 May 2023

from typing import List, Dict

from aerialsim.aerialsim_base import UnitDestroyedEvent, ASUnit, units_in_sector, UnitDetected, point_in_sector, \
    MissileFired
from aerialsim.amraam_missile_unit import AmraamMissile
from aerialsim.paveway_missile import PavewayMissile
from simulator.angles import signed_heading_diff, Degree
from simulator.dte_simulator import DteSimulator, Pose, Event
from simulator.geodesics import geodetic_direct
from simulator.units_of_measure import MetersPerSecond, Meters, Seconds, DegreesPerSecond


class Rafale(ASUnit):
    health: float = 10
    max_heading_change: DegreesPerSecond = 15  # Turn rate (max 27)
    max_altitude_change: MetersPerSecond = 200  # Climb rate (max 304)
    min_speed: MetersPerSecond = 120 * 0.514444
    max_speed: MetersPerSecond = 920 * 0.514444
    max_speed_delta: MetersPerSecond = 34 * 0.514444
    cannon_range: Meters = 1_800
    cannon_width: Degree = 10
    cannon_max_time: Seconds = 120
    cannon_burst_time: Seconds = 5
    cannon_max_altitude_delta: Meters = 500
    cannon_hit_prob_per_burst: float = 0.8
    radar_width: Degree = 120
    radar_range: Meters = 100_000

    def __init__(self, pose: Pose, side: int, speed: MetersPerSecond, n_amraam: int, n_paveway: int = 0):
        super().__init__(pose, side, Rafale.health)
        self.speed = speed
        self.new_speed = speed
        self.new_heading = pose.head
        self.new_altitude = pose.alt

        self.n_amraam = n_amraam
        self.flying_amraams: Dict[ASUnit, ASUnit] = {}  # amraam -> aircraft
        self.amraams_to_fire: List[ASUnit] = []  # amraams to be fired at next update()

        self.n_paveway = n_paveway
        self.flying_paveway: Dict[ASUnit, ASUnit] = {}  # paveway -> aircraft
        self.paveway_to_fire: List[ASUnit] = []  # paveway to be fired at next update()

        self.cannon_remain_secs: Seconds = Rafale.cannon_max_time
        self.cannon_current_burst_secs: Seconds = 0.0
        self.detected_units: List[ASUnit] = []

    def set_heading(self, new_heading: Degree):
        assert 0 <= new_heading < 360
        self.new_heading = new_heading

    def set_altitude(self, new_altitude: Meters):
        assert 0 <= new_altitude
        self.new_altitude = new_altitude

    def set_speed(self, new_speed: MetersPerSecond):
        if new_speed > Rafale.max_speed or new_speed < Rafale.min_speed:
            raise Exception(f"Rafale.set_speed Speed must be in [{Rafale.min_speed}m/s, {Rafale.max_speed}m/s]"
                            f", got {new_speed}")
        self.new_speed = new_speed

    def fire_cannon(self):
        self.cannon_current_burst_secs = min(self.cannon_remain_secs, Rafale.cannon_burst_time)

    def fire_amraam(self, target_unit: ASUnit):
        self.amraams_to_fire.append(target_unit)

    def fire_paveway(self, target_unit: ASUnit):
        self.paveway_to_fire.append(target_unit)

    def update(self, time_elapsed: Seconds, sim: DteSimulator) -> List[Event]:
        # Update heading
        if self.pose.head != self.new_heading:
            delta = signed_heading_diff(self.pose.head, self.new_heading)
            max_deg = Rafale.max_heading_change * time_elapsed
            if abs(delta) <= max_deg:
                self.pose.head = self.new_heading
            else:
                self.pose.head += max_deg if delta >= 0 else -max_deg
                self.pose.head %= 360

        # Update altitude
        if self.pose.alt != self.new_altitude:
            delta = self.new_altitude - self.pose.alt
            max_delta = Rafale.max_altitude_change * time_elapsed
            if abs(delta) <= max_delta:
                self.pose.alt = self.new_altitude
            else:
                self.pose.alt += max_delta if delta >= 0 else -max_delta

        # Update position
        self.pose.lat, self.pose.lon = geodetic_direct(self.pose.lat, self.pose.lon, self.pose.head,
                                                       self.speed * time_elapsed)

        # Update speed
        if self.speed != self.new_speed:
            delta = self.new_speed - self.speed
            max_delta = Rafale.max_speed_delta * time_elapsed
            if abs(delta) <= max_delta:
                self.speed = self.new_speed
            else:
                self.speed += max_delta if delta >= 0 else -max_delta

        # Update cannon
        events = []
        if self.cannon_current_burst_secs > 0:
            for unit in list(sim.active_units.values()):  # Copy values to avoid "dictionary changed size during iter."
                if unit.id != self.id and abs(self.pose.alt - unit.pose.alt) <= Rafale.cannon_max_altitude_delta:
                    p_kill = Rafale.cannon_hit_prob_per_burst * \
                             min(self.cannon_current_burst_secs, time_elapsed) / Rafale.cannon_burst_time
                    if sim.rnd_gen.random() < p_kill:
                        if point_in_sector(unit.pose.lat, unit.pose.lon, self.pose.lat, self.pose.lon,
                                           Rafale.cannon_range, self.pose.head, Rafale.cannon_width):
                            sim.remove_unit(unit.id)
                            events.append(UnitDestroyedEvent(self, unit))

            self.cannon_current_burst_secs = max(self.cannon_current_burst_secs - time_elapsed, 0.0)
            self.cannon_remain_secs = max(self.cannon_remain_secs - time_elapsed, 0.0)

        # Update radar
        det_units = units_in_sector(sim, self, Rafale.radar_range, self.pose.head, Rafale.radar_width, 0, self.side)
        for unit, dst, angle in det_units:
            if unit not in self.detected_units:
                events.append(UnitDetected(self, unit))
        self.detected_units = [u[0] for u in det_units]

        # Fire amraams
        if self.n_amraam > 0:
            for tgt in self.amraams_to_fire:
                if tgt in self.detected_units:
                    amraam = AmraamMissile(self.pose.copy(), self.side, sim.time_elapsed, tgt)
                    sim.add_unit(amraam)
                    events.append(MissileFired(self, amraam, tgt))
                    self.flying_amraams[amraam] = tgt
                    self.n_amraam -= 1
        self.amraams_to_fire = []

        # Fire paveways
        if self.n_paveway > 0:
            for tgt in self.paveway_to_fire:
                if tgt in self.detected_units:
                    paveway = PavewayMissile(self.pose.copy(), self.side, sim.time_elapsed, self.speed, tgt)
                    sim.add_unit(paveway)
                    events.append(MissileFired(self, paveway, tgt))
                    self.flying_paveway[paveway] = tgt
                    self.n_paveway -= 1
        self.paveway_to_fire = []

        # Remove dead amraams
        for m in list(self.flying_amraams):
            if not sim.unit_exists(m.id):
                del self.flying_amraams[m]

        # Remove dead paveways
        for m in list(self.flying_paveway):
            if not sim.unit_exists(m.id):
                del self.flying_paveway[m]

        # amraams guidance
        for missile, target in self.flying_amraams.items():
            if target in self.detected_units:
                assert isinstance(missile, AmraamMissile)
                missile.set_heading(missile.bearing_to(target))
                missile.set_altitude(target.pose.alt)

        return events

    def __str__(self):
        return f"Rafale[{self.id}]"
