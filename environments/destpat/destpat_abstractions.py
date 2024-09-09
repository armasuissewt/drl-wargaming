# destpat_abstractions.py
#
# Abstraction functions for the Destpat environment
#
# Author: Giacomo Del Rio
# Creation date: 24 Jan 2024

import pickle
from collections import defaultdict
from datetime import datetime
from enum import Enum
from typing import Optional, List, Tuple, Dict, Literal, OrderedDict

import numpy as np
from skimage import draw

from simulator.map_limits import MapLimits


# ------------- REAL STATE DEFINITIONS -------------

class Side(Enum):
    RED = 1
    BLUE = 2
    GREEN = 3
    NEUTRAL = 4


class UnitState(Enum):
    OK = 1
    DAMAGED = 2
    ENGAGED_DEFENSIVE = 3


class UnitInfo:
    def __init__(self, unit_id: str, side: Side, lat: float, lon: float, alt: float, heading: float, speed: float,
                 a2a: int, a2l: int, state: UnitState):
        self.unit_id: str = unit_id
        self.side: Side = side
        self.lat: float = lat
        self.lon: float = lon
        self.alt: float = alt
        self.heading: float = heading
        self.speed: float = speed
        self.a2a: int = a2a  # Number of Air-to-air missiles available
        self.a2l: int = a2l  # Number of Air-to-land missiles available
        self.state: UnitState = state

    def __str__(self):
        return f"[{self.unit_id}, {self.side}](lat={self.lat:.3f}, lon={self.lon:.3f}, alt={self.alt}, " \
               f"head={self.heading}, speed={self.speed}, a2a={self.a2a}, a2l={self.a2l}, state={self.state})"


class ContactType(Enum):
    UNKNOWN = 1
    MOBILE = 2
    MOBILE_DAM = 3
    AIR = 4
    MISSILE = 5


class ContactInfo:
    def __init__(self, cont_id: str, side: Side, lat: float, lon: float, contact_type: ContactType):
        self.cont_id: str = cont_id
        self.side: Side = side
        self.lat: float = lat
        self.lon: float = lon
        self.type: ContactType = contact_type

    def __str__(self):
        return f"{self.type}[{self.cont_id}, {self.side}](lat={self.lat:.3f}, lon={self.lon:.3f})"


class DestpatEvent:
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.args = {**kwargs}

        if name not in ['ATT_1_BAD_FIRE', 'ATT_1_FIRED', 'ATT_1_DESTROYED', 'DEF_1_DESTROYED', 'SAM_DAMAGED',
                        'SAM_DESTROYED']:
            raise RuntimeError(f"Unknown event {name}")

    def __str__(self):
        args_str = ', '.join([f'{k}={1}' for k, v in self.args.items()])
        return f"Event[{self.name}]({args_str})"


class DestpatRealState:
    def __init__(self, att_air_1: Optional[UnitInfo], att_air_2: Optional[UnitInfo], def_air_1: Optional[UnitInfo],
                 def_sam: Optional[UnitInfo], contacts: OrderedDict[str, ContactInfo],
                 missiles: Dict[str, UnitInfo], current_time: datetime, initial_time: datetime,
                 events: Dict[str, DestpatEvent]):
        self.att_air_1: Optional[UnitInfo] = att_air_1
        self.att_air_2: Optional[UnitInfo] = att_air_2
        self.def_air_1: Optional[UnitInfo] = def_air_1
        self.def_sam: Optional[UnitInfo] = def_sam
        self.contacts: OrderedDict[str, ContactInfo] = contacts  # ID -> Contact
        self.missiles: Dict[str, UnitInfo] = missiles  # ID -> UnitInfo
        self.current_time: datetime = current_time
        self.initial_time: datetime = initial_time
        self.events: Dict[str, DestpatEvent] = events

    def __str__(self):
        res = f"ATT-Air-1{self.att_air_1 if self.att_air_1 else ' - Killed'}\n"
        res += f"ATT-Air-2{self.att_air_2 if self.att_air_2 else ' - Killed'}\n"
        res += f"DEF-Air-1{self.def_air_1 if self.def_air_1 else ' - Killed'}\n"
        res += f"DEF-Sam{self.def_sam if self.def_sam else ' - Killed'}\n"
        for i, c in enumerate(self.contacts):
            res += f"Contact[{i}]{self.contacts[c]}\n"
        for i, m in enumerate(self.missiles):
            res += f"Missile[{i}]{self.missiles[m]}\n"
        res += f"Initial time: {self.initial_time}\n"
        res += f"Current time: {self.current_time}\n"
        res += f"Events: {self.events.keys()}"
        return res


# ------------- ABSTRACTION FUNCTIONS -------------

class DestpatAbstractions:

    def __init__(self, n_cells: int, obs_enc: str, rew_sig: str, max_steps: int, map_limits: MapLimits,
                 traces_len: int = 0, max_altitude: float = 25_000):
        self.n_cells = n_cells
        self.obs_enc = obs_enc
        self.rew_sig = rew_sig
        self.max_steps = max_steps
        self.map_limits = map_limits
        self.traces_len = traces_len
        self.max_altitude = max_altitude

        if self.obs_enc not in ['full', 'full_chunky']:
            raise RuntimeError(f"Unsupported observation encoding {obs_enc}")

        if self.rew_sig not in ['naive', 'penalties']:
            raise RuntimeError(f"Unsupported reward signal {rew_sig}")

    def get_obs_shape(self) -> Tuple:
        if self.obs_enc == 'full':
            return 1, self.n_cells, self.n_cells
        elif self.obs_enc == 'full_chunky':
            return self.n_cells, self.n_cells, 1

    def get_obs_min_max_val(self) -> Tuple[float, float]:
        if self.obs_enc in ['full', 'full_chunky']:
            return -1.01, 1.01

    def get_obs_dtype(self) -> np.number:
        if self.obs_enc in ['full', 'full_chunky']:
            return np.float32

    def get_reward_range(self) -> Tuple[float, float]:
        if self.rew_sig == 'naive':
            return -5.0, 8.0
        elif self.rew_sig == 'penalties':
            return -5.0, 8.0

    def compute_reward(self, rs: DestpatRealState, curr_step: int) -> Tuple[float, bool, bool]:
        if self.rew_sig == "naive":
            if rs.att_air_1 is None:
                return -5, True, False
            elif not self.map_limits.in_boundary(rs.att_air_1.lat, rs.att_air_1.lon):
                return -5, True, False
            elif curr_step >= self.max_steps:
                return 0.0, False, True  # Truncated
            elif rs.def_sam is None or rs.def_sam.state == UnitState.DAMAGED:
                return 8.0, True, False
            else:
                return 0.0, False, False
        elif self.rew_sig == "penalties":
            if rs.att_air_1 is None:
                return -5, True, False  # Attacker killed
            elif not self.map_limits.in_boundary(rs.att_air_1.lat, rs.att_air_1.lon):
                return -5, True, False  # Attacker out of map
            elif curr_step >= self.max_steps:
                return 0.0, False, True  # Truncated
            elif rs.def_sam is None or rs.def_sam.state == UnitState.DAMAGED:
                return 8.0, True, False  # SAM destroyed or damaged
            elif "ATT_1_BAD_FIRE" in rs.events:
                return -5.0, False, False  # Bad firing
            elif "ATT_1_FIRED" in rs.events:
                return -1.0, False, False  # Firing
            elif "DEF_1_DESTROYED" in rs.events:
                return 2.0, False, False  # Defender airplane destroyed
            else:
                return 0.0, False, False

    def abstract_state(self, rs_hist: List[DestpatRealState]) -> np.array:
        if self.obs_enc == 'full':
            return self.abstract_state_full(rs_hist)
        elif self.obs_enc == 'full_chunky':
            return self.abstract_state_full_chunky(rs_hist)

    def abstract_state_full(self, rs_hist: List[DestpatRealState]) -> np.array:
        o = np.zeros(shape=(1, self.n_cells, self.n_cells), dtype=np.float32)

        # Plane 0 -- Contacts
        contact_traces = self.get_contacts_traces(rs_hist)
        for c_id, t in contact_traces.items():
            contact_type = rs_hist[-1].contacts[c_id].type
            self.add_unit_trace(o, 0, t, contact_type, negative=True)

        # Plane 0 -- Attacker missiles
        missiles_traces = self.get_missiles_traces(rs_hist)
        for t in missiles_traces.values():
            self.add_unit_trace(o, 0, t, ContactType.MISSILE)

        # Plane 0 -- Attacker airplane
        att_air_1_trace = self.get_unit_trace(rs_hist, 'att_air_1')
        self.add_unit_trace(o, 0, att_air_1_trace, ContactType.AIR)

        # Plane 0 -- Ammunition
        if rs_hist[-1].att_air_1 is not None:
            if rs_hist[-1].att_air_1.a2l > 0:
                o[0, 0:3, 0:3] = 0.5
            if rs_hist[-1].att_air_1.a2a > 0:
                o[0, 0:3, self.n_cells - 3:self.n_cells] = 0.5

        # Plane 0 -- Bad firing
        if "ATT_1_BAD_FIRE" in rs_hist[-1].events:
            if rs_hist[-1].events['ATT_1_BAD_FIRE'].args['mtype'] == 'a2l':
                o[0, 0:3, 5] = 0.25
                o[0, 1, 5:8] = 0.25
            else:
                o[0, 0:3, self.n_cells - 6] = 0.25
                o[0, 1, self.n_cells - 8:self.n_cells - 5] = 0.25

        return o

    def abstract_state_full_chunky(self, rs_hist: List[DestpatRealState]) -> np.array:
        o = self.abstract_state_full(rs_hist)
        return np.einsum('kij->ijk', o)

    def get_unit_trace(self, rs_hist: List[DestpatRealState],
                       unit: Literal['att_air_1', 'def_sam', 'def_air_1']) -> List[Tuple[int, int, float]]:
        if len(rs_hist) == 0 or getattr(rs_hist[-1], unit) is None:
            return []

        trace = []
        begin_step = max(0, len(rs_hist) - self.traces_len)
        for i in range(begin_step, len(rs_hist), 1):
            ui: UnitInfo = getattr(rs_hist[i], unit)
            if ui is not None:
                trace.append(self._position_to_xy(ui.lat, ui.lon) + (ui.alt,))
        return trace

    def get_missiles_traces(self, rs_hist: List[DestpatRealState]) -> Dict[str, List[Tuple[int, int, float]]]:
        if len(rs_hist) == 0 or len(rs_hist[-1].missiles) == 0:
            return {}

        begin_step = max(0, len(rs_hist) - self.traces_len)

        traces = defaultdict(list)
        for m_id in rs_hist[-1].missiles:
            for i in range(begin_step, len(rs_hist), 1):
                ui: UnitInfo = rs_hist[i].missiles[m_id] if m_id in rs_hist[i].missiles else None
                if ui is not None:
                    traces[m_id].append(self._position_to_xy(ui.lat, ui.lon) + (ui.alt,))

        return traces

    def get_contacts_traces(self, rs_hist: List[DestpatRealState]) -> Dict[str, List[Tuple[int, int, float]]]:
        if len(rs_hist) == 0 or len(rs_hist[-1].contacts) == 0:
            return {}

        begin_step = max(0, len(rs_hist) - self.traces_len)

        traces = defaultdict(list)
        for c_id in rs_hist[-1].contacts:
            for i in range(begin_step, len(rs_hist), 1):
                ci: ContactInfo = rs_hist[i].contacts[c_id] if c_id in rs_hist[i].contacts else None
                if ci is not None:
                    traces[c_id].append(self._position_to_xy(ci.lat, ci.lon) + (0,))

        return traces

    def add_unit_trace(self, o: np.ndarray, plane: int, trace: List[Tuple[int, int, float]],
                       head_shape: ContactType, explosion: bool = False, negative: bool = False):
        """
        :param o: the observation array
        :param plane: the layer of o used to draw the trace [0.3]
        :param trace: a list of positions
        :param head_shape: the "shape" of the head
        :param explosion: if True, draw an explosion box at the head of the trace
        :param negative: if True, draw with negative values instead of positive
        """
        trace_color = -0.3 if negative else 0.3

        def safe_write(a, idx, _r, _c, v, no_overwrite=False):
            if 0 <= _r < self.n_cells and 0 <= _c < self.n_cells:
                if no_overwrite:
                    if a[idx, _r, _c] == 0:
                        a[idx, _r, _c] = v
                else:
                    a[idx, _r, _c] = v

        if len(trace) == 0:
            return

        for i, (r, c, h) in enumerate(trace):
            safe_write(o, plane, r, c, trace_color)
            if i + 1 < len(trace):
                rr, cc = draw.line(r, c, trace[i + 1][0], trace[i + 1][1])
                for ri, ci in zip(rr, cc):
                    safe_write(o, plane, ri, ci, trace_color, no_overwrite=True)

        head_r, head_c, head_color = trace[-1]
        head_color = self.altitude_to_intensity(head_color) * (-1 if negative else 1)
        if explosion:
            head_color = -1 if negative else 1
            safe_write(o, plane, head_r, head_c, head_color)
            safe_write(o, plane, head_r - 1, head_c - 1, head_color)
            safe_write(o, plane, head_r - 1, head_c + 1, head_color)
            safe_write(o, plane, head_r + 1, head_c - 1, head_color)
            safe_write(o, plane, head_r + 1, head_c + 1, head_color)
            safe_write(o, plane, head_r - 1, head_c, head_color)
            safe_write(o, plane, head_r + 1, head_c, head_color)
            safe_write(o, plane, head_r, head_c - 1, head_color)
            safe_write(o, plane, head_r, head_c + 1, head_color)
        else:
            if head_shape == ContactType.UNKNOWN:
                safe_write(o, plane, head_r, head_c, head_color)
            elif head_shape == ContactType.MOBILE:
                safe_write(o, plane, head_r, head_c, head_color)
                safe_write(o, plane, head_r - 1, head_c - 1, head_color)
                safe_write(o, plane, head_r - 1, head_c + 1, head_color)
                safe_write(o, plane, head_r + 1, head_c - 1, head_color)
                safe_write(o, plane, head_r + 1, head_c + 1, head_color)
            elif head_shape == ContactType.MOBILE_DAM:
                safe_write(o, plane, head_r, head_c, head_color)
                safe_write(o, plane, head_r - 1, head_c - 1, head_color)
                safe_write(o, plane, head_r + 1, head_c + 1, head_color)
            elif head_shape == ContactType.AIR:
                safe_write(o, plane, head_r - 1, head_c, head_color)
                safe_write(o, plane, head_r + 1, head_c, head_color)
                safe_write(o, plane, head_r, head_c - 1, head_color)
                safe_write(o, plane, head_r, head_c + 1, head_color)
            elif head_shape == ContactType.MISSILE:
                safe_write(o, plane, head_r, head_c, head_color)
                safe_write(o, plane, head_r + 1, head_c, head_color)
                safe_write(o, plane, head_r, head_c + 1, head_color)
                safe_write(o, plane, head_r + 1, head_c + 1, head_color)
            else:
                raise RuntimeError(f"Unknown head shape {head_shape}")

    def altitude_to_intensity(self, altitude: float) -> float:
        return (altitude / self.max_altitude * 0.3) + 0.5

    def _position_to_xy(self, latitude: float, longitude: float):
        lat_rel, lon_rel = self.map_limits.relative_position(latitude, longitude)
        lat_idx = min(int(lat_rel * (self.n_cells - 1)), self.n_cells - 1)
        lon_idx = min(int(lon_rel * (self.n_cells - 1)), self.n_cells - 1)
        return (self.n_cells - 1) - lat_idx, lon_idx
