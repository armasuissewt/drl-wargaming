# flytozurich_sim_wrappers.py
#
# Wrappers for the SimFlyToZurichEnv environment
#
# Author: Giacomo Del Rio
# Creation date: 3 March 2022

import math
from typing import Tuple, Dict, Optional

import numpy as np
from gymnasium.spaces import Box

from environments.flytozurich.flytozurich_env_sim import SimFlyToZurichEnv, FtzRealState

DEG_TO_RAD = math.pi / 180


class SimFtzObsRewAdapterEnv(SimFlyToZurichEnv):
    """
    Description:
        Adapter on the SimFlyToZurichEnv environment. Can change the observation space and the reward signal
    Observation:
        Sparse_3: Box(1, n, n)
            Num              Observation                            Min                     Max
            1, x,   y        airplane position+sine                  0                       1
            1, x+1, y        airplane position+cosine                0                       1
        Sparse_2: Box(n * n)  [not good]
            Num              Observation                            Min                     Max
            0,...,(n*n) -1   airplane position+heading               0                       1
        Sparse_1: Box(n * n + 2)
            Num              Observation                            Min                     Max
            0,...,(n*n) -1   airplane position 1-hot encoded         0                       1
            (n*n)            airplane heading sine                   0                       1
            (n*n) + 1        airplane heading cosine                 0                       1
        Continuous: Box(4)
            Num     Observation                            Min                     Max
            0       airplane map relative latitude          0                       1
            1       airplane map relative longitude         0                       1
            2       airplane heading sine                   0                       1
            3       airplane heading cosine                 0                       1
    Reward:
        - Naive: +10 for target, -10 for die, small step penalty
        - Goodstep: +10 for target, -10 for die, +0.01 for getting closer to target, small step penalty
    """

    def __init__(self, max_steps: int, timestep: int, obs_enc: str = 'Sparse_1', rew_sig: str = 'Naive',
                 n_cells: Optional[int] = 100, worker_id: int = 0, render_mode: Optional[str] = None):
        super().__init__(max_steps, timestep, worker_id, render_mode)

        self.obs_enc = obs_enc
        self.rew_sig = rew_sig
        self.n_cells = n_cells
        self.prev_dst_from_target = np.NAN

        if self.obs_enc == 'Sparse_3':
            self.observation_space = Box(low=0.0, high=1.0, shape=(1, n_cells, n_cells), dtype=np.float32)
        elif self.obs_enc == 'Sparse_2':
            self.observation_space = Box(low=0.0, high=1.0, shape=(n_cells * n_cells,), dtype=np.float32)
        elif self.obs_enc == 'Sparse_1':
            self.observation_space = Box(low=0.0, high=1.0, shape=(n_cells * n_cells + 2,), dtype=np.float32)
        elif self.obs_enc == 'Continuous':
            self.observation_space = Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
        else:
            raise RuntimeError(f"Unknown encoding {self.obs_enc}")

        if self.rew_sig == 'Naive':
            self.reward_range = (-10, 10)
        elif self.rew_sig == 'Goodstep':
            self.reward_range = (-10, 10)
        else:
            raise RuntimeError(f"Unknown reward {self.rew_sig}")

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        self.prev_dst_from_target = np.NAN
        return super().reset(seed=seed, options=options)

    def step(self, action: int) -> Tuple[np.array, float, bool, bool, Dict]:
        self.prev_dst_from_target = self.last_distance_from_target
        return super().step(action)

    def _abstract_state(self, rs: FtzRealState) -> np.array:
        return self.abstract_state_inner(rs, self.obs_enc, self.n_cells)

    def _compute_reward(self, rs: FtzRealState) -> Tuple[float, bool, bool]:
        return self.compute_reward_inner(rs, self.rew_sig, self._current_step, self.max_steps,
                                         self.last_distance_from_target, self.prev_dst_from_target)

    @staticmethod
    def _position_to_xy(latitude: float, longitude: float, n_cells: int):
        lat_rel, lon_rel = SimFlyToZurichEnv.map_limits.relative_position(latitude, longitude)
        lat_idx = min(int(lat_rel * (n_cells - 1)), n_cells - 1)
        lon_idx = min(int(lon_rel * (n_cells - 1)), n_cells - 1)
        return lat_idx, lon_idx

    @staticmethod
    def abstract_state_inner(rs: FtzRealState, obs_enc: str, n_cells: Optional[int]) -> np.array:
        if obs_enc == 'Continuous':
            obs = np.zeros(shape=4, dtype=np.float32)
            if rs.airplane:
                obs[0], obs[1] = SimFlyToZurichEnv.map_limits.relative_position(rs.airplane.lat, rs.airplane.lon)
                obs[2] = (math.sin(rs.airplane.heading * DEG_TO_RAD) + 1.0) / 2.0
                obs[3] = (math.cos(rs.airplane.heading * DEG_TO_RAD) + 1.0) / 2.0
            return obs

        if obs_enc == 'Sparse_1':
            position = np.zeros(shape=(n_cells, n_cells), dtype=np.float32)
            heading = np.zeros(shape=(2,), dtype=np.float32)
            if rs.airplane:
                x, y = SimFtzObsRewAdapterEnv._position_to_xy(rs.airplane.lat, rs.airplane.lon, n_cells)
                position[x, y] = 1
                heading[0] = (math.sin(rs.airplane.heading * DEG_TO_RAD) + 1.0) / 2.0
                heading[1] = (math.cos(rs.airplane.heading * DEG_TO_RAD) + 1.0) / 2.0
            return np.concatenate([position.flatten(), heading])

        if obs_enc == 'Sparse_2':
            position_heading = np.zeros(shape=(n_cells, n_cells), dtype=np.float32)
            if rs.airplane:
                x, y = SimFtzObsRewAdapterEnv._position_to_xy(rs.airplane.lat, rs.airplane.lon, n_cells)
                position_heading[x, y] = rs.airplane.heading / 360.0
            return position_heading.flatten()

        if obs_enc == 'Sparse_3':
            position_heading = np.zeros(shape=(1, n_cells, n_cells), dtype=np.float32)
            if rs.airplane:
                x, y = SimFtzObsRewAdapterEnv._position_to_xy(rs.airplane.lat, rs.airplane.lon, n_cells)
                position_heading[0, x, y] = (math.sin(rs.airplane.heading * DEG_TO_RAD) + 1.0) / 2.0
                if x + 1 < n_cells:
                    position_heading[0, x + 1, y] = (math.cos(rs.airplane.heading * DEG_TO_RAD) + 1.0) / 2.0
            return position_heading

    @staticmethod
    def compute_reward_inner(rs: FtzRealState, rew_sig: str, curr_step: int, max_steps: int,
                             last_distance_from_target: float, prev_dst_from_target: float) -> Tuple[float, bool, bool]:
        if rew_sig == 'Naive':
            if rs.airplane is None:
                return -10, True, False
            elif not SimFlyToZurichEnv.map_limits.in_boundary(rs.airplane.lat, rs.airplane.lon):
                return -10, True, False
            elif rs.airplane.state == "EngagedDefensive":
                return -10, True, False
            elif curr_step >= max_steps:
                return -0.01, False, True  # Truncated
            elif last_distance_from_target < SimFlyToZurichEnv.target_reached_distance:
                return 10.0, True, False
            else:
                return -0.01, False, False
        elif rew_sig == 'Goodstep':
            if rs.airplane is None:
                return -10, True, False
            elif not SimFlyToZurichEnv.map_limits.in_boundary(rs.airplane.lat, rs.airplane.lon):
                return -10, True, False
            elif rs.airplane.state == "EngagedDefensive":
                return -10, True, False
            elif curr_step >= max_steps:
                return -0.01, False, True  # Truncated
            elif last_distance_from_target < SimFlyToZurichEnv.target_reached_distance:
                return 10.0, True, False
            else:
                if last_distance_from_target < prev_dst_from_target:
                    return 0.01, False, False
                else:
                    return -0.01, False, False
        else:
            raise RuntimeError(f"Unknown reward {rew_sig}")

    @staticmethod
    def _inverse_distance_to_target_reward_fn(distance: float) -> float:
        return (1.0 - (distance / 315.0)) / 2.0
