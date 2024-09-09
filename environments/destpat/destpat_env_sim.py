# destpat_env_sim.py
#
# Gym environment based on python simulator
#
# Author: Giacomo Del Rio
# Creation date: 21 June 2023

import copy
from collections import OrderedDict
from datetime import datetime
from typing import Optional, List, Tuple, Dict, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.spaces import Box

from aerialsim.aerialsim_base import ASUnit, MissileFired, UnitDestroyedEvent, UnitDamagedEvent
from aerialsim.amraam_missile_unit import AmraamMissile
from aerialsim.pac3_missile_unit import Pac3Missile
from aerialsim.patriot_unit import Patriot
from aerialsim.paveway_missile import PavewayMissile
from aerialsim.rafale_unit import Rafale
from environments.destpat.destpat_abstractions import DestpatRealState, UnitInfo, ContactInfo, DestpatAbstractions, \
    DestpatEvent, UnitState, ContactType, Side
from simulator.angles import sum_angles
from simulator.dte_simulator import DteSimulator, Pose
from simulator.map_limits import MapLimits
from simulator.units_of_measure import Seconds


class SimDestpatEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    """ For a description of the environment see CmoDestpatEnv class """
    metadata = {'render.modes': ['ansi', 'real_state']}

    # Simulations variables
    initial_time = datetime(2021, 7, 24, 15, 00, 00)
    sim_tick_secs = 0.2
    map_limits = MapLimits(6.55, 45.74, 10.87, 47.83)

    def __init__(self, max_steps: int, timestep: Seconds, obs_enc: str = 'full', rew_sig: str = 'penalties',
                 n_cells: Optional[int] = 50, traces_len=5, worker_id: int = 0, render_mode: Optional[str] = None):
        super().__init__()
        self.max_steps = max_steps
        self.timestep: Seconds = timestep
        self.obs_enc = obs_enc
        self.rew_sig = rew_sig
        self.n_cells = n_cells
        self.traces_len = traces_len
        self.worker_id = worker_id
        self.render_mode = render_mode

        self.af = DestpatAbstractions(n_cells, obs_enc, rew_sig, max_steps, SimDestpatEnv.map_limits, traces_len)

        # --- Gym environment attributes
        o_low, o_high = self.af.get_obs_min_max_val()
        self.observation_space = Box(shape=self.af.get_obs_shape(), low=o_low, high=o_high,
                                     dtype=self.af.get_obs_dtype())
        self.reward_range = self.af.get_reward_range()
        self.action_space = spaces.Discrete(5)

        # --- Setup simulation
        self.simulator = DteSimulator(tick_time=SimDestpatEnv.sim_tick_secs)
        self.att_1_id: Optional[int] = None
        self.patriot_id: Optional[int] = None
        self.def_1_id: Optional[int] = None
        self._setup_scenario()

        # --- Define simulation state
        self.episode_real_states: List[DestpatRealState] = []
        self.current_abstract_state: Optional[np.array] = None
        self._initial_state_sim: DteSimulator = copy.deepcopy(self.simulator)
        self._contacts_ages: Dict[int, float] = {}
        self._current_step: int = 0

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        self.simulator = copy.deepcopy(self._initial_state_sim)
        self.episode_real_states = [self._read_state_from_simulation({})]
        self.current_abstract_state = self.af.abstract_state(self.episode_real_states)
        self._contacts_ages = {}
        self._current_step = 0
        return self.current_abstract_state, {'real_state': self.episode_real_states[-1]}

    def step(self, action: int) -> Tuple[np.array, float, bool, bool, Dict]:
        self._issue_actions_to_simulator(action)
        events = self._run_simulation()
        if action in [3, 4] and "ATT_1_FIRED" not in events:
            events['ATT_1_BAD_FIRE'] = DestpatEvent('ATT_1_BAD_FIRE', mtype='a2a' if action == 4 else 'a2l')

        # Read back the state from the simulator and save it
        self.episode_real_states.append(self._read_state_from_simulation(events))
        self.current_abstract_state = self.af.abstract_state(self.episode_real_states)
        reward, terminated, truncated = self.af.compute_reward(self.episode_real_states[-1], self._current_step)
        return self.current_abstract_state, reward, terminated, truncated, {'real_state': self.episode_real_states[-1]}

    def render(self):
        if self.render_mode == 'ansi':
            return str(self.episode_real_states[-1])
        elif self.render_mode == 'real_state':
            return self.episode_real_states[-1]

    def _issue_actions_to_simulator(self, action):
        attacker = self.simulator.get_unit(self.att_1_id)
        assert isinstance(attacker, Rafale)
        if action == 0:
            attacker.set_heading(sum_angles(attacker.pose.head, -15))
        elif action == 1:
            pass
        elif action == 2:
            attacker.set_heading(sum_angles(attacker.pose.head, 15))
        elif action == 3:
            if self.simulator.unit_exists(self.patriot_id):
                attacker.fire_paveway(self.simulator.get_unit(self.patriot_id))  # noqa
        elif action == 4:
            if self.simulator.unit_exists(self.def_1_id):
                attacker.fire_amraam(self.simulator.get_unit(self.def_1_id))  # noqa
        else:
            raise RuntimeError(f"Action {action} is out of range")

    def _run_simulation(self) -> Dict[str, DestpatEvent]:
        events = {}
        start_time: Seconds = self.simulator.time_elapsed
        while self.simulator.time_elapsed - start_time < self.timestep:
            self._control_blue_units()
            sim_events = self.simulator.do_tick()
            for e in sim_events:
                if isinstance(e, MissileFired) and e.target_unit.id in [self.def_1_id, self.patriot_id]:
                    events["ATT_1_FIRED"] = DestpatEvent('ATT_1_FIRED')
                if isinstance(e, UnitDestroyedEvent) and e.unit_destroyed.id == self.att_1_id:
                    events["ATT_1_DESTROYED"] = DestpatEvent('ATT_1_DESTROYED', lat=e.unit_destroyed.pose.lat,
                                                             lon=e.unit_destroyed.pose.lon)
                if isinstance(e, UnitDamagedEvent) and e.unit_damaged.id == self.patriot_id:
                    events["SAM_DAMAGED"] = DestpatEvent('SAM_DAMAGED')
                if isinstance(e, UnitDestroyedEvent) and e.unit_destroyed.id == self.patriot_id:
                    events["SAM_DESTROYED"] = DestpatEvent('SAM_DESTROYED', lat=e.unit_destroyed.pose.lat,
                                                           lon=e.unit_destroyed.pose.lon)
                if isinstance(e, UnitDestroyedEvent) and e.unit_destroyed.id == self.def_1_id:
                    events["DEF_1_DESTROYED"] = DestpatEvent('DEF_1_DESTROYED', lat=e.unit_destroyed.pose.lat,
                                                             lon=e.unit_destroyed.pose.lon)

            # Early-exit logic
            if not self.simulator.unit_exists(self.att_1_id):
                break  # Attacker destroyed
            if not self.simulator.unit_exists(self.patriot_id):
                break  # Patriot destroyed

        self._current_step += 1
        return events

    def _control_blue_units(self):
        try:
            att_1 = self.simulator.get_unit(self.att_1_id)
            def_air = self.simulator.get_unit(self.def_1_id)
            patriot = self.simulator.get_unit(self.patriot_id)
            assert isinstance(att_1, Rafale)
            assert isinstance(def_air, Rafale)
        except KeyError:
            return

        if att_1 in def_air.detected_units:
            def_air.set_heading(def_air.bearing_to(att_1))
            if def_air.distance_to(att_1) < 70_000 and def_air.n_amraam > 0:
                def_air.fire_amraam(att_1)
        else:
            def_air.set_heading(sum_angles(def_air.bearing_to(patriot), 90.0))

    def _read_state_from_simulation(self, events: Dict[str, DestpatEvent]) -> DestpatRealState:
        # Units
        att_1_info = self._get_unit_info(self.att_1_id)
        patriot_info = self._get_unit_info(self.patriot_id)
        def_1_info = self._get_unit_info(self.def_1_id)

        # Contacts
        contacts: List[(int, ContactInfo)] = []
        if att_1_info is not None:
            att_1 = self.simulator.get_unit(self.att_1_id)
            assert isinstance(att_1, Rafale)

            for u in att_1.detected_units:
                if u.side == 1:
                    if u.id not in self._contacts_ages:
                        self._contacts_ages[u.id] = self.simulator.time_elapsed

                    if isinstance(u, Patriot):
                        contact_type = ContactType.MOBILE
                        if u.id == self.patriot_id and u.health < 20:
                            contact_type = ContactType.MOBILE_DAM
                    elif isinstance(u, Rafale):
                        contact_type = ContactType.AIR
                    elif isinstance(u, AmraamMissile) or isinstance(u, Pac3Missile) or isinstance(u, PavewayMissile):
                        contact_type = ContactType.MISSILE
                    else:
                        contact_type = ContactType.UNKNOWN
                    contacts.append((self._contacts_ages[u.id],
                                     ContactInfo(str(u.id), Side.BLUE, u.pose.lat, u.pose.lon, contact_type)))
            contacts.sort(key=lambda x: x[0])
        contacts_dict = OrderedDict()
        for c in contacts:
            contacts_dict[c[1].cont_id] = c[1]

        # Missiles
        missiles: Dict[str, UnitInfo] = {}
        for unit_id, u in self.simulator.active_units.items():
            assert isinstance(u, ASUnit)
            if u.side == 0 and (isinstance(u, AmraamMissile) or isinstance(u, PavewayMissile)):
                missiles[str(unit_id)] = UnitInfo(str(unit_id), Side.RED, u.pose.lat, u.pose.lon, u.pose.alt,
                                                  u.pose.head, u.speed, 0, 0, UnitState.OK)

        return DestpatRealState(att_air_1=att_1_info, att_air_2=None, def_air_1=def_1_info, def_sam=patriot_info,
                                contacts=contacts_dict, missiles=missiles, current_time=datetime.now(),
                                initial_time=SimDestpatEnv.initial_time, events=events)

    def _setup_scenario(self):
        att_1_unit = Rafale(Pose(46.79, 7.93, heading=84.0, altitude=10972.8),
                            side=0, speed=400, n_amraam=1, n_paveway=1)
        patriot_unit = Patriot(Pose(46.83, 9.75, heading=40.0, altitude=0), side=1)
        def_1_unit = Rafale(Pose(46.7, 9.5, heading=183.0, altitude=10972.8),
                            side=1, speed=400, n_amraam=1, n_paveway=0)

        self.att_1_id = self.simulator.add_unit(att_1_unit)
        self.patriot_id = self.simulator.add_unit(patriot_unit)
        self.def_1_id = self.simulator.add_unit(def_1_unit)

    def _get_unit_info(self, unit_id: int) -> Optional[UnitInfo]:
        try:
            unit = self.simulator.get_unit(unit_id)
            assert isinstance(unit, ASUnit)
            speed = unit.speed if hasattr(unit, 'speed') else 0
            a2a, a2l = 0, 0
            if isinstance(unit, Rafale):
                a2a = unit.n_amraam
                a2l = unit.n_paveway
            state = UnitState.OK
            if isinstance(unit, Patriot) and unit.health < 20:
                state = UnitState.DAMAGED

            return UnitInfo(str(unit.id), Side.RED if unit.side == 0 else Side.BLUE, unit.pose.lat, unit.pose.lon,
                            unit.pose.alt, unit.pose.head, speed, a2a=a2a, a2l=a2l, state=state)
        except KeyError:
            return None
