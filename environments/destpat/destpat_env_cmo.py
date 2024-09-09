# destpat_env_cmo.py
#
# Gym environment based on CMO.
#
# There are two F-35 that should destroy a Patriot SAM unit. The SAM is protected by another F-35.
#
# Author: Giacomo Del Rio
# Creation date: 24 Jan 2023

import atexit
import math
import sys
import time
from collections import OrderedDict
from datetime import datetime
from typing import Optional, Tuple, Dict, Union, List

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.spaces import Box

from cmoclient.cmo_connector import CMOConnector, CMOConnectorException
from cmoclient.cmo_instance import CMOInstance
from environments.destpat.destpat_abstractions import DestpatRealState, UnitInfo, ContactInfo, DestpatAbstractions, \
    Side, UnitState, ContactType, DestpatEvent
from simulator.angles import sum_angles
from simulator.geodesics import geodetic_direct
from simulator.map_limits import MapLimits


class CmoDestpatEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    """
        Description:
            There are two F-35 that should destroy a Patriot SAM unit. The SAM is protected by another F-35.
        Observation:
            Type: Box(36) **All values will be normalized to [0, 1]**
            Num  Observation                      Real Min                Real Max
            --------------------------------------------------------------------------------
            0    attacker #1 latitude              map min                map max
            1    attacker #1 longitude             map min                map max
            2    attacker #1 altitude              0                      16000
            3    attacker #1 heading               0                      360
            4    attacker #1 speed (knots)         0                      1000
            5    attacker #1 AMRAAM qty.           0                      2
            6    attacker #1 Paveway qty.          0                      2
            7    attacker #1 state                 0=Killed, 1=Ok, 2=EngagedDefensive
            -----------------------------------------------------------------------------
            8    attacker #2 latitude              map min                map max
            9    attacker #2 longitude             map min                map max
            10   attacker #2 altitude              0                      16000
            11   attacker #2 heading               0                      360
            12   attacker #2 speed (knots)         0                      1000
            13   attacker #2 AMRAAM qty.           0                      2
            14   attacker #2 JSOW qty.             0                      2
            15   attacker #2 state                 0=Killed, 1=Ok, 2=EngagedDefensive
            --------------------------------------------------------------------------------
            16   contact #1 est. latitude          map min                map max
            17   contact #1 est. longitude         map min                map max
            18   contact #1 est. kind              0=Unknown, 1=Mobile Unit, 2=Air, 3=Missile
            19   contact #2 est. latitude          map min                map max
            20   contact #2 est. longitude         map min                map max
            21   contact #2 est. kind              0=Unknown, 1=Mobile Unit, 2=Air, 3=Missile
            22   contact #3 est. latitude          map min                map max
            23   contact #3 est. longitude         map min                map max
            24   contact #3 est. kind              0=Unknown, 1=Mobile Unit, 2=Air, 3=Missile
            25   contact #4 est. latitude          map min                map max
            26   contact #4 est. longitude         map min                map max
            27   contact #4 est. kind              0=Unknown, 1=Mobile Unit, 2=Air, 3=Missile
            28   contact #5 est. latitude          map min                map max
            29   contact #5 est. longitude         map min                map max
            30   contact #5 est. kind              0=Unknown, 1=Mobile Unit, 2=Air, 3=Missile
            31   contact #6 est. latitude          map min                map max
            32   contact #6 est. longitude         map min                map max
            33   contact #6 est. kind              0=Unknown, 1=Mobile Unit, 2=Air, 3=Missile
            --------------------------------------------------------------------------------
            34   Patriot SAM is damaged            0                      1
            35   time elapsed (seconds)            0                      1800
            --------------------------------------------------------------------------------
        Actions:
            Type: Discrete(42)
            Num   Action
            --------------------------------------------------------------------------------
            0     No action
            --------------------------------------------------------------------------------
            1     (1 + 0 ) attacker #1 Turn 45deg CCW
            2     (1 + 1 ) attacker #1 Turn 45deg CW
            3     (1 + 2 ) attacker #1 Set altitude LOW
            4     (1 + 3 ) attacker #1 Set altitude MID
            5     (1 + 4 ) attacker #1 Set altitude HIGH
            6     (1 + 5 ) attacker #1 Set speed LOW
            7     (1 + 6 ) attacker #1 Set speed MID
            8     (1 + 7 ) attacker #1 Set speed HIGH
            9     (1 + 8 ) attacker #1 Fire Air-to-Air at contact #1
            10    (1 + 9 ) attacker #1 Fire Air-to-Air at contact #2
            11    (1 + 10) attacker #1 Fire Air-to-Air at contact #3
            12    (1 + 11) attacker #1 Fire Air-to-Air at contact #4
            13    (1 + 12) attacker #1 Fire Air-to-Air at contact #5
            14    (1 + 13) attacker #1 Fire Air-to-Air at contact #6
            15    (1 + 14) attacker #1 Fire Air-to-Land at contact #1
            16    (1 + 15) attacker #1 Fire Air-to-Land at contact #2
            17    (1 + 16) attacker #1 Fire Air-to-Land at contact #3
            18    (1 + 17) attacker #1 Fire Air-to-Land at contact #4
            19    (1 + 18) attacker #1 Fire Air-to-Land at contact #5
            20    (1 + 19) attacker #1 Fire Air-to-Land at contact #6
            --------------------------------------------------------------------------------
            21    (21 + 0 ) attacker #2 Turn 45deg CCW
            22    (21 + 1 ) attacker #2 Turn 45deg CW
            23    (21 + 2 ) attacker #2 Set altitude LOW
            24    (21 + 3 ) attacker #2 Set altitude MID
            25    (21 + 4 ) attacker #2 Set altitude HIGH
            26    (21 + 5 ) attacker #2 Set speed LOW
            27    (21 + 6 ) attacker #2 Set speed MID
            28    (21 + 7 ) attacker #2 Set speed HIGH
            29    (21 + 8 ) attacker #2 Fire Air-to-Air at contact #1
            30    (21 + 9 ) attacker #2 Fire Air-to-Air at contact #2
            31    (21 + 10) attacker #2 Fire Air-to-Air at contact #3
            32    (21 + 11) attacker #2 Fire Air-to-Air at contact #4
            33    (21 + 12) attacker #2 Fire Air-to-Air at contact #5
            34    (21 + 13) attacker #2 Fire Air-to-Air at contact #6
            35    (21 + 14) attacker #2 Fire Air-to-Land at contact #1
            36    (21 + 15) attacker #2 Fire Air-to-Land at contact #2
            37    (21 + 16) attacker #2 Fire Air-to-Land at contact #3
            38    (21 + 17) attacker #2 Fire Air-to-Land at contact #4
            39    (21 + 18) attacker #2 Fire Air-to-Land at contact #5
            40    (21 + 19) attacker #2 Fire Air-to-Land at contact #6
            --------------------------------------------------------------------------------
        Episode termination/truncation and rewards:
            - Patriot is damaged or destroyed (win):    8   End
            - One attacker airplane is killed:         -5
            - Attacker both killed:                    -5   End
            - Defender airplane is killed:              2
            - Maximum timesteps elapsed:               -5   End
            - Airplane goes out of map:                -5   End
    """
    metadata = {'render.modes': ['ansi', 'real_state']}

    # Simulations variables
    scenario_file = r"scenarios\destroy_patriot.scen"
    lua_helpers_file = r"cmoclient\py_helpers.lua"
    att_side_guid = 'CXB7LT-0HMNQOBFVME9H'
    def_side_guid = 'CXB7LT-0HMNQOBFVME9J'
    att_a1_guid = 'CXB7LT-0HMNQOBG0PL7A'
    att_a1_a2a_guid = 'CXB7LT-0HMNQOBG0PL83'
    att_a1_a2l_guid = 'CXB7LT-0HMNQOBG0PL84'
    att_a2_guid = 'CXB7LT-0HMNT69B3PM5E'
    att_a2_a2a_guid = 'CXB7LT-0HMNT69B3PM65'
    att_a2_a2l_guid = 'CXB7LT-0HMNT69B3PM64'
    def_a1_guid = 'CXB7LT-0HMNQOBFVMEQB'
    def_sam_guid = 'CXB7LT-0HMNQOBG05AVM'
    map_limits = MapLimits(7.5, 45.76, 10.50, 47.81)

    def __init__(self, max_steps: int, timestep: int, tick_hms: Tuple[int, int, int], obs_enc: str = 'full',
                 rew_sig: str = 'penalties', n_cells: Optional[int] = 50, traces_len=5, run_instance: bool = True,
                 max_cmo_error_retry: int = 2, worker_id: int = 0, render_mode: Optional[str] = None):
        """ Builds a new CmoDestpatEnv

        :param max_steps: maximum number of step() calls for an episode before truncation
        :param timestep: duration, in seconds, of a single step() in CMO simulation
        :param tick_hms: time span of a single tick in CMO simulation in (hours, minutes, seconds)
            (a single step contains timestep/(ticks[0] * 3600 + ticks[1] * 60 + ticks[0]) ticks)
        :param obs_enc: encoding of the observations
        :param rew_sig: reward signal
        :param n_cells: number of subdivisions for observation encoding
        :param traces_len: length of traces of the units
        :param run_instance: if True, launch a new instance of CMO, otherwise connect to an existing one
        :param max_cmo_error_retry: number of retries before an exception
        :param worker_id: id of the environment.
            For multiprocessing each environment must have an unique, >=0, worker_id
        :param render_mode: specifies the output of the render() method
        """
        super().__init__()
        self.max_steps = max_steps
        self.timestep = timestep
        self.tick = tick_hms
        self.obs_enc = obs_enc
        self.rew_sig = rew_sig
        self.n_cells = n_cells
        self.traces_len = traces_len
        self.worker_id = worker_id
        self.render_mode = render_mode

        self.af = DestpatAbstractions(n_cells, obs_enc, rew_sig, max_steps, CmoDestpatEnv.map_limits, traces_len)

        # Gym environment attributes
        o_low, o_high = self.af.get_obs_min_max_val()
        self.observation_space = Box(shape=self.af.get_obs_shape(), low=o_low, high=o_high,
                                     dtype=self.af.get_obs_dtype())
        self.reward_range = self.af.get_reward_range()
        self.action_space = spaces.Discrete(5)

        # Define CMO interface
        self.max_cmo_error_retry = max_cmo_error_retry
        self.run_instance = run_instance
        self.cmo_port: int = CMOInstance.CMO_DEFAULT_PORT + self.worker_id
        self.cmo_instance: Optional[CMOInstance] = None
        self.cmo_conn: Optional[CMOConnector] = None
        self._reset_cmo()

        # Define simulation state
        self.episode_real_states: List[DestpatRealState] = []
        self.current_abstract_state: Optional[np.array] = None
        self._initial_state_xml: str = self.cmo_conn.export_scenario_to_xml()
        self._initial_time: datetime = self.cmo_conn.current_time()
        self._next_contact_age: int = 0
        self._contacts_ages: Dict[str, int] = {}
        self._max_contacts: int = 6
        self._ui_time_compression: int = 15
        self._current_step: int = 0

        # Safety cleanup code
        if run_instance:
            atexit.register(lambda x: x.kill(), self.cmo_instance)

    def close(self):
        if self.run_instance:
            CMOInstance.kill_cmo_instance_for_port(self.cmo_port)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """ Resets the environment to an initial state and returns an initial observation.
        :return:
            - the initial observation
            - info dictionary with a 'real_state' key
        """
        super().reset(seed=seed)

        self.episode_real_states = []
        for _ in range(self.max_cmo_error_retry):
            try:
                self.cmo_conn.import_scenario_from_xml(self._initial_state_xml)
                self.episode_real_states.append(self._read_state_from_simulation())
                break
            except CMOConnectorException as e:
                print(f"reset() Got exception from CMO. Restarting. {e}", file=sys.stderr)
                self._reset_cmo()
        if len(self.episode_real_states) == 0:
            raise RuntimeError(f"Can't restart CMO after {self.max_cmo_error_retry} attempts.")
        self.current_abstract_state = self.af.abstract_state(self.episode_real_states)
        self._current_step = 0
        self._contacts_ages = {}
        return self.current_abstract_state, {'real_state': self.episode_real_states[-1]}

    def step(self, action: int) -> Tuple[np.array, float, bool, bool, Dict]:
        """ Run one timestep of the environment's dynamics. When end of episode is reached, you are responsible for
            calling `reset()` to reset this environment's state.
        :param action: the action issued
        :return:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        try:
            self._issue_actions_to_simulator(action)
            self._run_simulation()

            # Read back the state from the simulator
            self.episode_real_states.append(self._read_state_from_simulation())
            dynamic_events = CmoDestpatEnv.detect_dynamic_events(self.episode_real_states[-2],
                                                                 self.episode_real_states[-1], action)
            self.episode_real_states[-1].events.update(dynamic_events)
            self.current_abstract_state = self.af.abstract_state(self.episode_real_states)
        except CMOConnectorException as e:
            raise e

        reward, terminated, truncated = self.af.compute_reward(self.episode_real_states[-1], self._current_step)
        return self.current_abstract_state, reward, terminated, truncated, {'real_state': self.episode_real_states[-1]}

    def render(self):
        if self.render_mode == 'ansi':
            return str(self.episode_real_states[-1])
        elif self.render_mode == 'real_state':
            return self.episode_real_states[-1]

    def save(self) -> Dict:
        """ Save the current state of the simulation to a dictionary """
        return {
            'episode_real_states': self.episode_real_states,
            'current_abstract_state': self.current_abstract_state,
            '_current_step': self._current_step,
            '_next_contact_age': self._next_contact_age,
            '_contacts_ages': self._contacts_ages,
            'cmo_state': self.cmo_conn.export_scenario_to_xml()
        }

    def restore(self, state_dict: Dict) -> None:
        """ Restore the state of the simulation from a dictionary """
        self.episode_real_states = state_dict['episode_real_states']
        self.current_abstract_state = state_dict['current_abstract_state']
        self._current_step = state_dict['_current_step']
        self._next_contact_age = state_dict['_next_contact_age']
        self._contacts_ages = state_dict['_contacts_ages']
        self.cmo_conn.import_scenario_from_xml(state_dict['cmo_state'])

    def _issue_actions_to_simulator(self, action: int) -> None:
        unit = self.episode_real_states[-1].att_air_1
        if unit is not None:
            if 0 <= action <= 2:  # Direction
                if action == 0:
                    new_heading = sum_angles(unit.heading, -45)
                elif action == 1:
                    new_heading = unit.heading  # Go straight
                else:
                    new_heading = sum_angles(unit.heading, 45)
                waypoint = geodetic_direct(unit.lat, unit.lon, new_heading,
                                           self._steering_waypoint_distance_m(unit.speed))
                self.cmo_conn.set_course(unit.unit_id, waypoint)

            elif action == 3:  # Fire A2L
                for c_id, c in self.episode_real_states[-1].contacts.items():
                    if c.type in [ContactType.MOBILE, ContactType.MOBILE_DAM]:
                        weapon_id = 2061 if unit.unit_id == "CXB7LT-0HMNQOBG0PL7A" else 826
                        self.cmo_conn.attack_contact(unit.unit_id, c_id, 1, None, weapon_id, 2)

            elif action == 4:  # Fire A2A
                for c_id, c in self.episode_real_states[-1].contacts.items():
                    if c.type == ContactType.AIR:
                        weapon_id = 51
                        self.cmo_conn.attack_contact(unit.unit_id, c_id, 1, None, weapon_id, 2)

            else:
                raise RuntimeError(f"Action {action} is out of range")

    def _run_simulation(self):
        tick_secs = self.tick[0] * 3600.0 + self.tick[1] * 60.0 + self.tick[2]
        n_steps = math.floor(self.timestep / tick_secs)

        for i in range(n_steps):
            if not self.run_instance:
                self.cmo_conn.set_time_compression(self._ui_time_compression)
            self.cmo_conn.run_period(self.tick)
            # Put here optional Early-exit logic

        self._current_step += 1

    def _steering_waypoint_distance_m(self, unit_speed_knots: float) -> float:
        speed_kmh = unit_speed_knots * 1.852
        time_h = self.timestep / 3600.0
        return math.ceil(speed_kmh * time_h) * 1_000.0 + 10_000.0  # +10km  for some slack

    def _reset_cmo(self):
        if self.run_instance:
            CMOInstance.kill_cmo_instance_for_port(self.cmo_port)

            scenario = CmoDestpatEnv.scenario_file
            self.cmo_instance: CMOInstance = CMOInstance(scenario, self.cmo_port,
                                                         startup_time_secs=20, autoexit=False)
            self.cmo_instance.run()
        self.cmo_conn: CMOConnector = CMOConnector(address=('localhost', self.cmo_port))
        self.cmo_conn.connect()

        # After run_script() we need to reconnect to avoid the '<eof> expected near ...' error
        self.cmo_conn.run_script(CmoDestpatEnv.lua_helpers_file)
        self.cmo_conn.disconnect()
        time.sleep(0.5)
        self.cmo_conn.connect()

        self._set_patriot_destroyed_event()

    def _set_patriot_destroyed_event(self):
        self.cmo_conn.set_trigger('add', 'UnitDamaged', 'Patriot damaged',
                                  target_filter={'SpecificUnitID': CmoDestpatEnv.def_sam_guid},
                                  params="DamagePercent=20")
        self.cmo_conn.set_action('add', 'Points', 'Patriot damaged increase pt',
                                 params="SideId='ATTACKER', PointChange=50")
        self.cmo_conn.set_event('New event', "mode='add'")
        self.cmo_conn.set_event_trigger('New event', "mode='add', name='Patriot damaged'")
        self.cmo_conn.set_event_action('New event', "mode='add', name='Patriot damaged increase pt'")

    def _get_contact_age(self, contact_id: str) -> int:
        if contact_id not in self._contacts_ages:
            self._contacts_ages[contact_id] = self._next_contact_age
            self._next_contact_age += 1
        return self._contacts_ages[contact_id]

    def _read_attacker_state(self, unit_guid: str, a2a_wpn_guid: str) -> Optional[UnitInfo]:
        unit = self.cmo_conn.get_unit(unit_guid)
        if unit:
            weap = self.cmo_conn.get_loadout_weapons(unit_guid)
            if weap[0]['wpn_guid'] == a2a_wpn_guid:
                a2a = weap[0]['wpn_current']
                a2l = weap[1]['wpn_current']
            else:
                a2a = weap[1]['wpn_current']
                a2l = weap[0]['wpn_current']
            unit_state = CmoDestpatEnv.convert_cmo_unit_state(unit["unitstate"])
            return UnitInfo(unit_guid, Side.RED, unit['latitude'], unit['longitude'],
                            unit['altitude'], unit['heading'], unit['speed'], a2a, a2l, unit_state)
        else:
            return None

    def _read_state_from_simulation(self) -> DestpatRealState:
        att_a1 = self._read_attacker_state(CmoDestpatEnv.att_a1_guid, CmoDestpatEnv.att_a1_a2a_guid)
        att_a2 = None  # self._read_attacker_state(CmoDestpatEnv.att_a2_guid, CmoDestpatEnv.att_a2_a2a_guid)
        unit = self.cmo_conn.get_unit(CmoDestpatEnv.def_a1_guid)
        def_a1 = UnitInfo(CmoDestpatEnv.def_a1_guid, Side.BLUE, unit['latitude'], unit['longitude'],
                          unit['altitude'], unit['heading'], unit['speed'], 0, 0,
                          CmoDestpatEnv.convert_cmo_unit_state(unit["unitstate"])) if unit else None
        unit = self.cmo_conn.get_unit(CmoDestpatEnv.def_sam_guid)
        def_sam = UnitInfo(CmoDestpatEnv.def_sam_guid, Side.BLUE, unit['latitude'], unit['longitude'],
                           unit['altitude'], unit['heading'], unit['speed'], 0, 0,
                           CmoDestpatEnv.convert_cmo_unit_state(unit["unitstate"])) if unit else None

        # AIM-120D AMRAAM P3I.4, AGM-154C JSOW [BROACH], GBU-49/B Paveway II GPS/LGB [Mk82]
        missiles: Dict[str, UnitInfo] = {}
        for unit in self.cmo_conn.get_side_units(CmoDestpatEnv.att_side_guid):
            if unit['name'][:8] in ['AIM-120D', 'AGM-154C', 'GBU-49/B']:
                m = self.cmo_conn.get_unit(unit['guid'])
                missiles[unit['guid']] = UnitInfo(unit['guid'], Side.RED, m['latitude'], m['longitude'],
                                                  m['altitude'], m['heading'], m['speed'], 0, 0, UnitState.OK)

        contacts_list: List[Tuple[int, ContactInfo]] = []
        for c in self.cmo_conn.get_contacts("ATTACKER"):
            if len(contacts_list) >= self._max_contacts:
                break
            lat = float(self.cmo_conn.get_contact_attr("ATTACKER", c['guid'], 'latitude', as_float=True))
            lon = float(self.cmo_conn.get_contact_attr("ATTACKER", c['guid'], 'longitude', as_float=True))
            contact_type = CmoDestpatEnv.convert_cmo_contact_type(c['type'])
            contacts_list.append(
                (self._get_contact_age(c['guid']), ContactInfo(c['guid'], Side.BLUE, lat, lon, contact_type)))
        contacts_list.sort(key=lambda x: x[0])
        contacts = OrderedDict()
        for c in contacts_list:
            contacts[c[1].cont_id] = c[1]

        score = self.cmo_conn.get_score('ATTACKER')
        mission_completed = score >= 50
        sim_time = self.cmo_conn.current_time()

        # Detect SAM_DAMAGED event (SAM_DESTROYED is subsumed)
        events: Dict[str, DestpatEvent] = {}
        if mission_completed:
            def_sam.state = UnitState.DAMAGED
            for _, c in contacts_list:
                if c.type == ContactType.MOBILE:
                    c.type = ContactType.MOBILE_DAM
            events['SAM_DAMAGED'] = DestpatEvent('SAM_DAMAGED')

        return DestpatRealState(att_a1, att_a2, def_a1, def_sam, contacts, missiles, sim_time,
                                self._initial_time, events)

    @staticmethod
    def detect_dynamic_events(prev_state: Optional[DestpatRealState], current_state: DestpatRealState,
                              action: int) -> Dict[str, DestpatEvent]:
        events: Dict[str, DestpatEvent] = {}

        # --- Detect ATT_1_BAD_FIRE and ATT_1_FIRED events
        if prev_state is not None and action in [3, 4]:
            bad_fire = True
            for m_id, _ in current_state.missiles.items():
                if m_id not in prev_state.missiles:
                    bad_fire = False
                    break
            if bad_fire:
                events['ATT_1_BAD_FIRE'] = DestpatEvent('ATT_1_BAD_FIRE', mtype='a2a' if action == 4 else 'a2l')
            else:
                events['ATT_1_FIRED'] = DestpatEvent('ATT_1_FIRED')

        # --- Detect ATT_1_DESTROYED event
        if prev_state is not None:
            if prev_state.att_air_1 is not None and current_state.att_air_1 is None:
                events['ATT_1_DESTROYED'] = DestpatEvent('ATT_1_DESTROYED')

        # --- Detect DEF_1_DESTROYED event
        if prev_state is not None:
            if prev_state.def_air_1 is not None and current_state.def_air_1 is None:
                events['DEF_1_DESTROYED'] = DestpatEvent('DEF_1_DESTROYED')

        return events

    @staticmethod
    def convert_cmo_unit_state(unit_state: str) -> UnitState:
        if unit_state in ['OnPlottedCourse', 'Unassigned', 'EngagedOffensive', 'OnPatrol', 'Tasked']:
            return UnitState.OK
        elif unit_state == "EngagedDefensive":
            return UnitState.ENGAGED_DEFENSIVE
        else:
            raise ValueError(f"Unknown unit state {unit_state}")

    @staticmethod
    def convert_cmo_contact_type(contact_type: str) -> ContactType:
        if contact_type == 'Mobile Unit':
            return ContactType.MOBILE
        elif contact_type == 'Air':
            return ContactType.AIR
        elif contact_type == 'Missile':
            return ContactType.MISSILE
        else:
            return ContactType.UNKNOWN
