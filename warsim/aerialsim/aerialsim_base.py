# aerialsim_base.py
#
# The events that can be raised in an aerial simulation
#
# Author: Giacomo Del Rio
# Creation date: 17 May 2023

from typing import List, Tuple, Optional

from simulator.angles import signed_heading_diff
from simulator.dte_simulator import Event, Unit, Pose, DteSimulator
from simulator.geodesics import geodetic_distance, geodetic_bearing
from simulator.units_of_measure import Meters, Degree, Latitude, Longitude


# --- Units base class
class ASUnit(Unit):
    def __init__(self, pose: Pose, side: int, health: float):
        """ Base attributes for an Aerial Simulation Unit (ASUnit)

        :param pose: the pose of the unit in the space
        :param side: the side (we can have as many sides as we want)
        :param health: a number that summarize the health status of a unit. When 0, the unit is destroyed.
        """
        super().__init__(pose)
        self.side: int = side
        self.health: float = health


# --- Events
class UnitDetected(Event):
    def __init__(self, origin: ASUnit, detected_unit: ASUnit):
        super().__init__("UnitDetected", origin)
        self.detected_unit = detected_unit

    def __str__(self):
        return super().__str__() + f"({self.origin} -> {self.detected_unit})"


class MissileFired(Event):
    def __init__(self, origin: ASUnit, missile: ASUnit, target: ASUnit):
        super().__init__("MissileFired", origin)
        self.missile_unit = missile
        self.target_unit = target

    def __str__(self):
        return super().__str__() + f"({self.origin} -> {self.missile_unit} -> {self.target_unit})"


class UnitDestroyedEvent(Event):
    def __init__(self, origin: ASUnit, unit_destroyed: ASUnit):
        super().__init__("UnitDestroyedEvent", origin)
        self.unit_destroyed = unit_destroyed

    def __str__(self):
        return super().__str__() + f"({self.origin} -> {self.unit_destroyed})"


class UnitDamagedEvent(Event):
    def __init__(self, origin: ASUnit, unit_damaged: ASUnit):
        super().__init__("UnitDamagedEvent", origin)
        self.unit_damaged = unit_damaged

    def __str__(self):
        return super().__str__() + f"({self.origin} -> {self.unit_damaged})"


# --- Functions
def point_in_sector(point_lat: Latitude, point_lon: Longitude, center_lat: Latitude, center_lon: Longitude,
                    sector_range: Meters, sector_heading: Degree, sector_width: Degree) -> \
        Optional[Tuple[Meters, Degree]]:
    """ Check if (point_lat, point_lon) is contained in the sector centered in (center_lat, center_lon) with
        radius sector_range, headed towards sector_heading and an amplitude of sector_width.

        :return: if the point is in the sector, a tuple with the distance and the bearing, otherwise None
    """
    point_distance: Meters = geodetic_distance(center_lat, center_lon, point_lat, point_lon)
    if point_distance <= sector_range:
        point_bearing: Degree = geodetic_bearing(center_lat, center_lon, point_lat, point_lon)
        angle_delta: Degree = abs(signed_heading_diff(sector_heading, point_bearing))
        if angle_delta <= sector_width / 2.0:
            return point_distance, point_bearing
    return None


def units_in_sector(sim: DteSimulator, source: ASUnit,
                    sector_range: Meters, sector_heading: Degree, sector_width: Degree,
                    min_altitude: Meters = 0, skip_side: Optional[int] = None) -> List[Tuple[ASUnit, Meters, Degree]]:
    """ Returns all the active units in sim that are contained in the sector centered at source with
        radius sector_range, headed towards sector_heading and an amplitude of sector_width.
        The units are filtered to have a min_altitude and, optionally, skip the side of skip_side
    """
    res: List[Tuple[ASUnit, Meters, Degree]] = []
    for unit in sim.active_units.values():
        assert isinstance(unit, ASUnit)
        if unit != source and unit.pose.alt >= min_altitude and (skip_side is None or unit.side != skip_side):
            pt_in_sect = point_in_sector(unit.pose.lat, unit.pose.lon, source.pose.lat, source.pose.lon, sector_range,
                                         sector_heading, sector_width)
            if pt_in_sect is not None:
                res.append((unit, pt_in_sect[0], pt_in_sect[1]))
    return res
