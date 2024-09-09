# patriot_unit.py
#
# A Patriot SAM unit
#
# Author: Giacomo Del Rio
# Creation date: 19 May 2023

from typing import List, Dict

from aerialsim.aerialsim_base import UnitDetected, MissileFired, ASUnit, units_in_sector
from aerialsim.pac3_missile_unit import Pac3Missile
from simulator.dte_simulator import Unit, Pose, DteSimulator, Event
from simulator.units_of_measure import Seconds, Meters, Degree


class Patriot(ASUnit):
    health: float = 20
    radar_width: Degree = 140
    radar_range: Meters = 140_000
    radar_min_distance: Meters = 500
    min_detection_altitude: Meters = 300
    missile_range: Meters = 111_000

    def __init__(self, pose: Pose, side: int):
        super().__init__(pose, side, Patriot.health)
        self.detected_aircrafts: List[Unit] = []
        self.flying_missiles: Dict[Unit, Unit] = {}  # missile -> aircraft

    def update(self, time_elapsed: Seconds, sim: DteSimulator) -> List[Event]:
        # Units detection
        events = []
        det_units = units_in_sector(sim, self, Patriot.radar_range, self.pose.head + Patriot.radar_width / 2,
                                    Patriot.radar_width, Patriot.min_detection_altitude, self.side)
        for unit, dst, angle in det_units:
            if dst < Patriot.radar_min_distance:
                continue  # Skip units too close

            if unit not in self.detected_aircrafts:
                events.append(UnitDetected(self, unit))
            # Decide if firing
            if dst <= Patriot.missile_range and unit not in self.flying_missiles.values():
                pac3 = Pac3Missile(self.pose.copy(), self.side, sim.time_elapsed, unit)
                sim.add_unit(pac3)
                events.append(MissileFired(self, pac3, unit))
                self.flying_missiles[pac3] = unit
        self.detected_aircrafts = [u[0] for u in det_units if u[1] > Patriot.radar_min_distance]

        # Remove dead missiles
        for m in list(self.flying_missiles):
            if not sim.unit_exists(m.id):
                del self.flying_missiles[m]

        # Missile guidance
        for missile, aircraft in self.flying_missiles.items():
            if aircraft in self.detected_aircrafts:
                assert isinstance(missile, Pac3Missile)
                missile.set_heading(missile.bearing_to(aircraft))
                missile.set_altitude(aircraft.pose.alt)

        return events

    def __str__(self):
        return f"Patriot[{self.id}]"
