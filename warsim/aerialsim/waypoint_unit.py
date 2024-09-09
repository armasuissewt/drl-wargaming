# waypoint_unit.py
#
# A static Waypoint unit
#
# Author: Giacomo Del Rio
# Creation date: 19 May 2023


from typing import List, Optional

from aerialsim.aerialsim_base import ASUnit
from simulator.dte_simulator import Pose, Event, DteSimulator
from simulator.units_of_measure import Seconds


class Waypoint(ASUnit):

    def __init__(self, pose: Pose, side: int, text: Optional[str] = None):
        super().__init__(pose, side, health=10)
        self.text = text

    def update(self, time_elapsed: Seconds, sim: DteSimulator) -> List[Event]:
        return []

    def __str__(self):
        return f"Waypoint[{self.id}]"
