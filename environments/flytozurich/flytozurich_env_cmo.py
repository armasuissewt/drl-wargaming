# flytozurich_env_cmo.py
#
# Gym environment for the simple scenario in which an airplane should arrive in Zurich without being
# destroyed by a SAM Patriot unit.
# CMO based
#
# Author: Giacomo Del Rio
# Creation date: 9 November 2021

import atexit
import math
import time
from datetime import datetime
from typing import Optional, Tuple, Dict, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.spaces import Box

from cmoclient.cmo_connector import CMOConnector, CMOConnectorException
from cmoclient.cmo_instance import CMOInstance
from environments.flytozurich.flytozurich_env_sim import FtzRealState, UnitInfo
from simulator.angles import sum_angles
from simulator.geodesics import geodetic_direct, geodetic_distance
from simulator.map_limits import MapLimits


class CmoFlyToZurichEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    """
        Description:
            An airplane is flying from Dijon to Zurich. There is a SAM battery in between. The airplane
            should arrive on Zurich without being destroyed. Actions allows to go straight, turn left or
            turn right.
        Observation:
            Type: Box(13)
            Num     Observation               Min                     Max
            0       airplane latitude         -90                     90
            1       airplane longitude        -180                    180
            2       airplane altitude         0                       Inf
            3       airplane heading          0                       360
            4       airplane speed (knots)    0                       Inf
            5       sam battery latitude      -90                     90
            6       sam battery longitude     -180                    180
            7       sam battery altitude      0                       Inf
            8       sam battery heading       0                       360
            9       target latitude           -90                     90
            10      target longitude          -180                    180
            11      airplane state            0=OnSupportMission/OnPlottedCourse, 1=EngagedDefensive, 2=Killed
            12      time elapsed (seconds)    0                       Inf
        Actions:
            Type: Discrete(3)
            Num   Action
            0     Turn 10deg CCW
            1     Do nothing (continue straight)
            2     Turn 10deg CW
        Episode termination/truncation and rewards:
            - Airplane engaged/killed by a missile:    -10
            - Airplane escapes the map boundaries:     -10
            - Airplane reaches the target:             +10
            - Maximum timesteps reached (truncation):  -0.01
            - Step:                                    -0.01
    """
    metadata = {'render.modes': ['ansi', 'real_state', 'human']}

    # Simulations variables
    scenario_file = r"scenarios\simple_scenario.scen"
    lua_helpers_file = r"cmoclient\cmoclient\py_helpers.lua"
    blue_side_guid = 'd539fdd4-e88c-436c-8b16-94a891dafa2d'
    airplane_guid = "c0d644fd-5be6-4c3e-b0ae-da30638e826f"
    defender_guid = "38ad0146-f62d-4d72-a54e-134292155432"
    target_refpoint_name = "RP-Target"
    target_reached_distance = 10_000
    map_limits = MapLimits(left_lon=5.1, bottom_lat=45.4, right_lon=11.1, top_lat=47.9)

    # map_limits = MapLimits(4, 45, 12, 48.5)  # Old

    def __init__(self, max_steps: int, timestep: int, tick_hms: Tuple[int, int, int], run_instance: bool = True,
                 max_cmo_error_retry: int = 10, worker_id: int = 0, render_mode: Optional[str] = None):
        """ Builds a new CmoFlyToZurichEnv

        :param max_steps: maximum number of step() calls for an episode before truncation
        :param timestep: duration, in seconds, of a single step() in CMO simulation
        :param tick_hms: time span of a single tick in CMO simulation in (hours, minutes, seconds)
            (a single step contains timestep/(ticks[0] * 3600 + ticks[1] * 60 + ticks[0]) ticks)
        :param run_instance: if True, launch a new instance of CMO, otherwise connect to an existing one
        :param max_cmo_error_retry: number of retries before an exception
        :param worker_id: id of the environment.
            For multiprocessing each environment must have an unique, >=0, worker_id
        :param render_mode: specifies the output of the render() method
        """
        super(CmoFlyToZurichEnv, self).__init__()
        self.max_steps = max_steps
        self.timestep = timestep
        self.tick = tick_hms
        self.worker_id = worker_id

        # Gym environment attributes
        self.reward_range = (-0.01, 10)
        self.action_space = spaces.Discrete(3)
        self.observation_space = Box(
            low=np.array([-90, -180, 0, 0, 0, -90, -180, 0, 0, -90, -180, 0, 0], dtype=np.float32),
            high=np.array([90, 180, np.Inf, 360, np.Inf, 90, 180, np.Inf, 360, 90, 180, 2, np.Inf], dtype=np.float32),
            dtype=np.float32)
        self.render_mode = render_mode

        # Define CMO interface
        self.max_cmo_error_retry = max_cmo_error_retry
        self.run_instance = run_instance
        self.cmo_port: int = CMOInstance.CMO_DEFAULT_PORT + self.worker_id
        self.cmo_instance: Optional[CMOInstance] = None
        self.cmo_conn: Optional[CMOConnector] = None
        self._reset_cmo()

        # Define simulation state
        self.current_real_state: Optional[FtzRealState] = None
        self.current_abstract_state: Optional[np.array] = None
        self._initial_state_xml: str = self.cmo_conn.export_scenario_to_xml()
        self._initial_time: datetime = self.cmo_conn.current_time()
        self._target_pose: UnitInfo = self._read_target_pose()
        self._current_step: int = 0
        self.last_distance_from_target = np.Inf
        self.target_reached = False

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

        for _ in range(self.max_cmo_error_retry):
            try:
                self.cmo_conn.import_scenario_from_xml(self._initial_state_xml)
                self.current_real_state = self._read_state_from_simulation()
                break
            except CMOConnectorException as e:
                print(f"reset() Got exception from CMO. Restarting. {e}")
                self._reset_cmo()
        self.current_abstract_state = self._abstract_state(self.current_real_state)
        self._current_step = 0
        self.last_distance_from_target = np.Inf
        self.target_reached = False
        return self.current_abstract_state, {'real_state': self.current_real_state}

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

            # Read back the state from the simulator and save it
            self.current_real_state = self._read_state_from_simulation()
            self.current_abstract_state = self._abstract_state(self.current_real_state)
        except CMOConnectorException as e:
            # print(f"step() Got exception from CMO. State:")
            # self.current_real_state.print()
            # print(f"step() Current action = {action}")
            raise e

        reward, terminated, truncated = self._compute_reward(self.current_real_state)
        return self.current_abstract_state, reward, terminated, truncated, {'real_state': self.current_real_state}

    def render(self):
        if self.render_mode == 'ansi':
            return str(self.current_real_state)
        elif self.render_mode == 'real_state':
            return self.current_real_state

    def save(self) -> Dict:
        """ Save the current state of the simulation to a dictionary """
        return {
            'current_real_state': self.current_real_state,
            'current_abstract_state': self.current_abstract_state,
            '_current_step': self._current_step,
            'last_distance_from_target': self.last_distance_from_target,
            'target_reached': self.target_reached,
            'cmo_state': self.cmo_conn.export_scenario_to_xml()
        }

    def restore(self, state_dict: Dict) -> None:
        """ Restore the state of the simulation from a dictionary """
        self.current_real_state = state_dict['current_real_state']
        self.current_abstract_state = state_dict['current_abstract_state']
        self._current_step = state_dict['_current_step']
        self.last_distance_from_target = state_dict['last_distance_from_target']
        self.target_reached = state_dict['target_reached']
        self.cmo_conn.import_scenario_from_xml(state_dict['cmo_state'])

    def _issue_actions_to_simulator(self, action):
        # Uses the strategy to set a virtual waypoint which is located at a fixed distance from
        # the actual airplane position but rotated of the desired angle from the current heading
        rafale = self.current_real_state.airplane
        new_heading = rafale.heading
        if action == 0:
            new_heading = sum_angles(rafale.heading, -10)
        elif action == 2:
            new_heading = sum_angles(rafale.heading, 10)
        waypoint = geodetic_direct(rafale.lat, rafale.lon, new_heading,
                                   self._steering_waypoint_distance_m(rafale.speed))
        self.cmo_conn.set_course(CmoFlyToZurichEnv.airplane_guid, waypoint)

    def _run_simulation(self):
        tick_secs = self.tick[0] * 3600.0 + self.tick[1] * 60.0 + self.tick[2]
        n_steps = math.floor(self.timestep / tick_secs)

        for i in range(n_steps):
            self.cmo_conn.run_period(self.tick)

            # Early-exit logic
            self.target_reached = self.cmo_conn.get_score('red') == 10
            ap = self.cmo_conn.get_unit(CmoFlyToZurichEnv.airplane_guid)
            if ap is None or self.target_reached:
                break

            self.last_distance_from_target = geodetic_distance(ap['latitude'], ap['longitude'],
                                                               self.current_real_state.target.lat,
                                                               self.current_real_state.target.lon)
        self._current_step += 1

    def _compute_reward(self, rs: FtzRealState) -> Tuple[float, bool, bool]:
        if self.target_reached:
            return 10.0, True, False
        elif rs.airplane is None:
            return -10, True, False
        elif not CmoFlyToZurichEnv.map_limits.in_boundary(rs.airplane.lat, rs.airplane.lon):
            return -10, True, False
        elif rs.airplane.state == "EngagedDefensive":
            return -10, True, False
        elif self._current_step >= self.max_steps:
            return -0.01, False, True  # Truncated
        else:
            return -0.01, False, False

    def _steering_waypoint_distance_m(self, unit_speed_knots: float) -> float:
        speed_kmh = unit_speed_knots * 1.852
        time_h = self.timestep / 3600.0
        return math.ceil(speed_kmh * time_h) * 1_000.0 + 10_000.0  # +10km  for some slack

    def _reset_cmo(self):
        if self.run_instance:
            CMOInstance.kill_cmo_instance_for_port(self.cmo_port)

            scenario = CmoFlyToZurichEnv.scenario_file
            self.cmo_instance: CMOInstance = CMOInstance(scenario, self.cmo_port,
                                                         startup_time_secs=20, autoexit=False)
            self.cmo_instance.run()
        self.cmo_conn: CMOConnector = CMOConnector(address=('localhost', self.cmo_port))
        self.cmo_conn.connect()

        # After run_script() we need to reconnect to avoid the '<eof> expected near ...' error
        self.cmo_conn.run_script(CmoFlyToZurichEnv.lua_helpers_file)
        self.cmo_conn.disconnect()
        time.sleep(0.5)
        self.cmo_conn.connect()

        self._set_target_reached_event()

    def _set_target_reached_event(self):
        self.cmo_conn.set_trigger('add', 'UnitEntersArea', 'Rafale on target',
                                  target_filter={'SpecificUnitID': "c0d644fd-5be6-4c3e-b0ae-da30638e826f"},
                                  params="area={'rp-56','rp-57','rp-58','rp-59'}")
        self.cmo_conn.set_action('add', 'Points', 'Rafale on target increase pt',
                                 params="SideId='Red', PointChange=10")
        self.cmo_conn.set_event('New event', "mode='add'")
        self.cmo_conn.set_event_trigger('New event', "mode='add', name='Rafale on target'")
        self.cmo_conn.set_event_action('New event', "mode='add', name='Rafale on target increase pt'")

    def _read_target_pose(self) -> UnitInfo:
        rp = self.cmo_conn.get_reference_points('RED', CmoFlyToZurichEnv.target_refpoint_name)
        return UnitInfo(id=rp[0]['guid'], lat=rp[0]['latitude'],
                        lon=rp[0]['longitude'], alt=400, heading=0, speed=0, state="Ok")

    def _read_state_from_simulation(self) -> FtzRealState:
        ap = self.cmo_conn.get_unit(CmoFlyToZurichEnv.airplane_guid)
        de = self.cmo_conn.get_unit(CmoFlyToZurichEnv.defender_guid)
        airplane = UnitInfo(CmoFlyToZurichEnv.airplane_guid, ap['latitude'], ap['longitude'], ap['altitude'],
                            ap['heading'], ap['speed'], ap["unitstate"]) if ap else None
        sam = UnitInfo(CmoFlyToZurichEnv.defender_guid, de['latitude'], de['longitude'], de['altitude'], de['heading'],
                       de['speed'], de["unitstate"]) if de else None
        missiles = []
        for unit in self.cmo_conn.get_side_units(CmoFlyToZurichEnv.blue_side_guid):
            if unit['name'].startswith('MIM-104F Patriot'):
                m = self.cmo_conn.get_unit(unit['guid'])
                missiles.append(UnitInfo(unit['guid'], m['latitude'], m['longitude'], m['altitude'],
                                         m['heading'], m['speed'], m["unitstate"]))
        sim_time = self.cmo_conn.current_time()
        return FtzRealState(airplane, sam, self._target_pose, missiles, (sim_time - self._initial_time).seconds)

    def _abstract_state(self, rs: FtzRealState) -> np.array:
        state = np.zeros(shape=(13,), dtype=float)
        if rs.airplane:
            state[0] = rs.airplane.lat
            state[1] = rs.airplane.lon
            state[2] = rs.airplane.alt
            state[3] = rs.airplane.heading
            state[4] = rs.airplane.speed
        if rs.sam:
            state[5] = rs.sam.lat
            state[6] = rs.sam.lon
            state[7] = rs.sam.alt
            state[8] = rs.sam.heading
        state[9] = rs.target.lat
        state[10] = rs.target.lon
        state[11] = 2 if rs.airplane is None else (
            0 if rs.airplane.state in ["OnSupportMission", "OnPlottedCourse"] else 1)
        state[12] = rs.elapsed_time
        return state
