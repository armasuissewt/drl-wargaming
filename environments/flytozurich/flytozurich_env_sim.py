# flytozurich_env_sim.py
#
# Gym environment for the scenario in which an airplane should arrive in Zurich without being
# destroyed by a SAM Patriot unit.
# Warsim based.
#
# Author: Giacomo Del Rio

import pickle
from collections import namedtuple
from typing import Optional, List, Tuple, Dict, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.spaces import Box

from aerialsim.patriot_unit import Patriot
from aerialsim.rafale_unit import Rafale
from aerialsim.waypoint_unit import Waypoint
from simulator.angles import sum_angles
from simulator.dte_simulator import DteSimulator, Pose
from simulator.geodesics import geodetic_distance
from simulator.map_limits import MapLimits
from simulator.units_of_measure import Seconds, Meters

# Immutable tuple-like class to store Unit info
UnitInfo = namedtuple('UnitInfo', ['id', 'lat', 'lon', 'alt', 'heading', 'speed', 'state'])


class FtzRealState:
    def __init__(self, airplane: Optional[UnitInfo], sam: Optional[UnitInfo], target: Optional[UnitInfo],
                 missiles: List[UnitInfo], elapsed_time: Seconds):
        self.airplane: Optional[UnitInfo] = airplane
        self.sam: Optional[UnitInfo] = sam
        self.target: Optional[UnitInfo] = target
        self.missiles: List[UnitInfo] = missiles
        self.elapsed_time: Seconds = elapsed_time

    def print(self):
        print(str(self))

    def __str__(self):
        out = ""
        if self.airplane:
            out += f"Airplane[{self.airplane.id}]: (lat={self.airplane.lat:.3f}, lon={self.airplane.lon:.3f}, " \
                   f"alt={self.airplane.alt}, head={self.airplane.heading:.1f}, speed={self.airplane.speed}, " \
                   f"state={self.airplane.state})\n"
        else:
            out += "Airplane: killed\n"
        if self.sam:
            out += f"SAM[{self.sam.id}]: (lat={self.sam.lat:.3f}, lon={self.sam.lon:.3f}, alt={self.sam.alt}, " \
                   f"head={self.sam.heading:.1f}, speed={self.sam.speed}, state={self.sam.state})\n"
        else:
            out += "SAM: killed\n"
        for m in self.missiles:
            out += f"  Missile[{m.id}]: (lat={m.lat:.3f}, lon={m.lon:.3f}, alt={m.alt}, head={m.heading:.1f}, " \
                   f"speed={m.speed}, state={m.state})\n"
        out += f"Target[{self.target.id}]: (lat={self.target.lat:.3f}, lon={self.target.lon:.3f})\n"
        out += f"Elapsed time: {self.elapsed_time}"
        return out


class SimFlyToZurichEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    """ For a description of the environment see CmoFlyToZurichEnv class """
    metadata = {'render.modes': ['ansi', 'real_state']}

    # Simulations variables
    sim_tick_secs = 1
    airplane_start_pose = UnitInfo(0, lat=46.715, lon=6.358, alt=10017.82, heading=86.06418, speed=480 * 0.514444,
                                   state="OnSupportMission")
    sam_start_pose = UnitInfo(0, lat=46.539, lon=8.124, alt=3865.0, heading=245, speed=0, state="Ok")
    target_pose = UnitInfo(0, lat=46.779, lon=9.615, alt=400, heading=0, speed=0, state="Ok")
    map_limits = MapLimits(left_lon=5.1, bottom_lat=45.4, right_lon=11.1, top_lat=47.9)
    rafale_missile_detection_range: Meters = 60_000
    target_reached_distance: Meters = 10_000

    def __init__(self, max_steps: int, timestep: int, worker_id: int = 0, render_mode: Optional[str] = None):
        super(SimFlyToZurichEnv, self).__init__()
        self.max_steps = max_steps
        self.timestep_secs = timestep
        self.worker_id = worker_id

        # Gym environment attributes
        self.reward_range = (-10, 10)
        self.action_space = spaces.Discrete(3)
        self.observation_space = Box(
            low=np.array([-90, -180, 0, 0, 0, -90, -180, 0, 0, -90, -180, 0, 0], dtype=np.float32),
            high=np.array([90, 180, np.Inf, 360, np.Inf, 90, 180, np.Inf, 360, 90, 180, 2, np.Inf], dtype=np.float32),
            dtype=np.float32)
        self.render_mode = render_mode

        # Setup simulation
        self.simulator = DteSimulator(tick_time=SimFlyToZurichEnv.sim_tick_secs)
        self.rafale_id: Optional[int] = None
        self.sam_id: Optional[int] = None
        self.target_id: Optional[int] = None
        self.__setup_scenario()

        # Define simulation state
        self.current_real_state: Optional[FtzRealState] = None
        self.current_abstract_state: Optional[np.array] = None
        self._initial_state_sim: bytes = pickle.dumps(self.simulator)
        self._target_pose: UnitInfo = self.__get_unit_info(self.target_id)
        self._current_step: int = 0
        self.last_distance_from_target: Meters = np.Inf

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        self.simulator = pickle.loads(self._initial_state_sim)
        self.current_real_state = self._read_state_from_simulation()
        self.current_abstract_state = self._abstract_state(self.current_real_state)
        self._current_step = 0
        self.last_distance_from_target = np.Inf
        return self.current_abstract_state, {'real_state': self.current_real_state}

    def step(self, action: int) -> Tuple[np.array, float, bool, bool, Dict]:
        self._issue_actions_to_simulator(action)
        self._run_simulation()

        # Read back the state from the simulator and save it
        self.current_real_state = self._read_state_from_simulation()
        self.current_abstract_state = self._abstract_state(self.current_real_state)

        reward, terminated, truncated = self._compute_reward(self.current_real_state)
        return self.current_abstract_state, reward, terminated, truncated, {'real_state': self.current_real_state}

    def render(self):
        if self.render_mode == 'ansi':
            return str(self.current_real_state)
        elif self.render_mode == 'real_state':
            return self.current_real_state

    def _issue_actions_to_simulator(self, action):
        rafale = self.simulator.get_unit(self.rafale_id)
        assert isinstance(rafale, Rafale)
        if action == 0:
            rafale.set_heading(sum_angles(rafale.pose.head, -10))
        elif action == 2:
            rafale.set_heading(sum_angles(rafale.pose.head, 10))

    def _run_simulation(self):
        start_time = self.simulator.time_elapsed
        while self.simulator.time_elapsed - start_time < self.timestep_secs:
            self.simulator.do_tick()

            # Early-exit logic
            try:
                rafale = self.simulator.get_unit(self.rafale_id)
                self.last_distance_from_target = geodetic_distance(rafale.pose.lat, rafale.pose.lon,
                                                                   SimFlyToZurichEnv.target_pose.lat,
                                                                   SimFlyToZurichEnv.target_pose.lon)
                if self.last_distance_from_target < SimFlyToZurichEnv.target_reached_distance:
                    break  # Target reached
            except KeyError:
                break  # Airplane destroyed

        self._current_step += 1

    def _compute_reward(self, rs: FtzRealState) -> Tuple[float, bool, bool]:
        if rs.airplane is None:
            return -10, True, False
        elif not SimFlyToZurichEnv.map_limits.in_boundary(rs.airplane.lat, rs.airplane.lon):
            return -10, True, False
        elif rs.airplane.state == "EngagedDefensive":
            return -10, True, False
        elif self._current_step >= self.max_steps:
            return -0.01, False, True  # Truncated
        elif self.last_distance_from_target < SimFlyToZurichEnv.target_reached_distance:
            return 10.0, True, False
        else:
            return -0.01, False, False

    def _read_state_from_simulation(self) -> FtzRealState:
        airplane = self.__get_airplane_info()
        sam = self.__get_unit_info(self.sam_id)
        missiles = self.__get_missiles(self.sam_id)
        target = self.__get_unit_info(self.target_id)
        return FtzRealState(airplane, sam, target, missiles, self.simulator.time_elapsed)

    def __get_airplane_info(self) -> Optional[UnitInfo]:
        try:
            rafale_state = "OnSupportMission"
            rafale = self.simulator.get_unit(self.rafale_id)
            assert isinstance(rafale, Rafale)
        except KeyError:
            return None

        try:
            sam = self.simulator.get_unit(self.sam_id)
            assert isinstance(sam, Patriot)
            for missile, aircraft in sam.flying_missiles.items():
                if rafale.distance_to(missile) < SimFlyToZurichEnv.rafale_missile_detection_range:
                    rafale_state = "EngagedDefensive"
            return UnitInfo(rafale.id, rafale.pose.lat, rafale.pose.lon, rafale.pose.alt, rafale.pose.head,
                            rafale.speed, rafale_state)
        except KeyError:
            return UnitInfo(rafale.id, rafale.pose.lat, rafale.pose.lon, rafale.pose.alt, rafale.pose.head,
                            rafale.speed, rafale_state)

    def __is_airplane_engaged(self) -> bool:
        rafale = self.simulator.get_unit(self.rafale_id)
        sam = self.simulator.get_unit(self.sam_id)
        assert isinstance(sam, Patriot)
        for missile, aircraft in sam.flying_missiles.items():
            if rafale.distance_to(missile) < SimFlyToZurichEnv.rafale_missile_detection_range:
                return True
        return False

    def __get_missiles(self, sam_id: int) -> List[UnitInfo]:
        res = []
        if self.simulator.unit_exists(sam_id):
            sam = self.simulator.get_unit(self.sam_id)
            assert isinstance(sam, Patriot)
            for missile, aircraft in sam.flying_missiles.items():
                res.append(UnitInfo(missile.id, missile.pose.lat, missile.pose.lon, missile.pose.alt,
                                    missile.pose.head, missile.speed, "Ok"))  # noqa
        return res

    def _abstract_state(self, rs: FtzRealState) -> np.array:
        state = np.zeros(shape=(13,), dtype=np.float32)
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

    def __setup_scenario(self):
        raf_unit = Rafale(Pose(SimFlyToZurichEnv.airplane_start_pose.lat, SimFlyToZurichEnv.airplane_start_pose.lon,
                               SimFlyToZurichEnv.airplane_start_pose.heading,
                               SimFlyToZurichEnv.airplane_start_pose.alt),
                          side=0, speed=SimFlyToZurichEnv.airplane_start_pose.speed, n_amraam=0, n_paveway=0)
        sam_unit = Patriot(Pose(SimFlyToZurichEnv.sam_start_pose.lat, SimFlyToZurichEnv.sam_start_pose.lon,
                                SimFlyToZurichEnv.sam_start_pose.heading, SimFlyToZurichEnv.sam_start_pose.alt), side=1)

        tgt_unit = Waypoint(Pose(SimFlyToZurichEnv.target_pose.lat, SimFlyToZurichEnv.target_pose.lon,
                                 SimFlyToZurichEnv.target_pose.heading, SimFlyToZurichEnv.target_pose.alt), side=2,
                            text="target")
        self.rafale_id = self.simulator.add_unit(raf_unit)
        self.sam_id = self.simulator.add_unit(sam_unit)
        self.target_id = self.simulator.add_unit(tgt_unit)

    def __get_unit_info(self, unit_id: int) -> Optional[UnitInfo]:
        try:
            unit = self.simulator.get_unit(unit_id)
            speed = unit.speed if hasattr(unit, 'speed') else None
            return UnitInfo(unit.id, unit.pose.lat, unit.pose.lon, unit.pose.alt, unit.pose.head, speed, "Ok")
        except KeyError:
            return None
