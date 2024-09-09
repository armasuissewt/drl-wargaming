# flytozurich_cmo_wrappers.py
#
# Wrappers for the CmoFlyToZurichEnv environment
#
# Author: Giacomo Del Rio
# Creation date: 4 May 2022

import math
from typing import Tuple, Dict, Optional

import numpy as np
from gymnasium.spaces import Box

from environments.flytozurich.flytozurich_env_cmo import CmoFlyToZurichEnv
from environments.flytozurich.flytozurich_env_sim import FtzRealState

DEG_TO_RAD = math.pi / 180


class CmoFtzObsRewAdapterEnv(CmoFlyToZurichEnv):
    """ See SimFtzObsRewAdapterEnv for full description """

    def __init__(self, max_steps: int, timestep: int, tick_hms: Tuple[int, int, int], run_instance: bool = True,
                 max_cmo_error_retry: int = 10, worker_id: int = 0, obs_enc: str = 'Sparse_1', rew_sig: str = 'Naive',
                 n_cells: Optional[int] = 100, render_mode: Optional[str] = None):
        super().__init__(max_steps, timestep, tick_hms, run_instance, max_cmo_error_retry, worker_id, render_mode)

        self.obs_enc = obs_enc
        self.rew_sig = rew_sig
        self.n_cells = n_cells
        self.prev_dst_from_target = np.NAN

        if self.obs_enc == 'Sparse_4':
            self.observation_space = Box(low=0.0, high=1.0, shape=(n_cells, n_cells, 1), dtype=np.float32)
        elif self.obs_enc == 'Sparse_3':
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

    def save(self) -> Dict:
        d = super().save()
        d['prev_dst_from_target'] = self.prev_dst_from_target
        return d

    def restore(self, state_dict: Dict) -> None:
        self.prev_dst_from_target = state_dict['prev_dst_from_target']
        super().restore(state_dict)

    def _abstract_state(self, rs: FtzRealState) -> np.array:
        return self.abstract_state_inner(rs, self.obs_enc, self.n_cells)

    def _compute_reward(self, rs: FtzRealState) -> Tuple[float, bool, bool]:
        return self.compute_reward_inner(rs, self.rew_sig, self._current_step, self.max_steps,
                                         self.last_distance_from_target, self.prev_dst_from_target, self.target_reached)

    @staticmethod
    def _position_to_xy(latitude: float, longitude: float, n_cells: int):
        lat_rel, lon_rel = CmoFlyToZurichEnv.map_limits.relative_position(latitude, longitude)
        lat_idx = min(int(lat_rel * (n_cells - 1)), n_cells - 1)
        lon_idx = min(int(lon_rel * (n_cells - 1)), n_cells - 1)
        return lat_idx, lon_idx

    @staticmethod
    def abstract_state_inner(rs: FtzRealState, obs_enc: str, n_cells: Optional[int]) -> np.array:
        if obs_enc == 'Continuous':
            obs = np.zeros(shape=4, dtype=np.float32)
            if rs.airplane:
                obs[0], obs[1] = CmoFlyToZurichEnv.map_limits.relative_position(rs.airplane.lat, rs.airplane.lon)
                obs[2] = (math.sin(rs.airplane.heading * DEG_TO_RAD) + 1.0) / 2.0
                obs[3] = (math.cos(rs.airplane.heading * DEG_TO_RAD) + 1.0) / 2.0
            return obs

        if obs_enc == 'Sparse_1':
            position = np.zeros(shape=(n_cells, n_cells), dtype=np.float32)
            heading = np.zeros(shape=(2,), dtype=np.float32)
            if rs.airplane:
                x, y = CmoFtzObsRewAdapterEnv._position_to_xy(rs.airplane.lat, rs.airplane.lon, n_cells)
                position[x, y] = 1
                heading[0] = (math.sin(rs.airplane.heading * DEG_TO_RAD) + 1.0) / 2.0
                heading[1] = (math.cos(rs.airplane.heading * DEG_TO_RAD) + 1.0) / 2.0
            return np.concatenate([position.flatten(), heading])

        if obs_enc == 'Sparse_2':
            position_heading = np.zeros(shape=(n_cells, n_cells), dtype=np.float32)
            if rs.airplane:
                x, y = CmoFtzObsRewAdapterEnv._position_to_xy(rs.airplane.lat, rs.airplane.lon, n_cells)
                position_heading[x, y] = rs.airplane.heading / 360.0
            return position_heading.flatten()

        if obs_enc == 'Sparse_3':
            position_heading = np.zeros(shape=(1, n_cells, n_cells), dtype=np.float32)
            if rs.airplane:
                x, y = CmoFtzObsRewAdapterEnv._position_to_xy(rs.airplane.lat, rs.airplane.lon, n_cells)
                position_heading[0, x, y] = (math.sin(rs.airplane.heading * DEG_TO_RAD) + 1.0) / 2.0
                if x + 1 < n_cells:
                    position_heading[0, x + 1, y] = (math.cos(rs.airplane.heading * DEG_TO_RAD) + 1.0) / 2.0
            return position_heading

        if obs_enc == 'Sparse_4':
            position_heading = np.zeros(shape=(n_cells, n_cells, 1), dtype=np.float32)
            if rs.airplane:
                x, y = CmoFtzObsRewAdapterEnv._position_to_xy(rs.airplane.lat, rs.airplane.lon, n_cells)
                position_heading[x, y, 0] = (math.sin(rs.airplane.heading * DEG_TO_RAD) + 1.0) / 2.0
                if x + 1 < n_cells:
                    position_heading[x + 1, y, 0] = (math.cos(rs.airplane.heading * DEG_TO_RAD) + 1.0) / 2.0
            return position_heading

    @staticmethod
    def compute_reward_inner(rs: FtzRealState, rew_sig: str, curr_step: int, max_steps: int,
                             last_distance_from_target: float, prev_dst_from_target: float, target_reached: bool) -> \
            Tuple[float, bool, bool]:
        if rew_sig == 'Naive':
            if rs.airplane is None:
                return -10, True, False
            elif not CmoFlyToZurichEnv.map_limits.in_boundary(rs.airplane.lat, rs.airplane.lon):
                return -10, True, False
            elif rs.airplane.state == "EngagedDefensive":
                return -10, True, False
            elif curr_step >= max_steps:
                return -0.01, False, True  # Truncated
            elif target_reached:
                return 10.0, True, False
            else:
                return -0.01, False, False
        elif rew_sig == 'Goodstep':
            if rs.airplane is None:
                return -10, True, False
            elif not CmoFlyToZurichEnv.map_limits.in_boundary(rs.airplane.lat, rs.airplane.lon):
                return -10, True, False
            elif rs.airplane.state == "EngagedDefensive":
                return -10, True, False
            elif curr_step >= max_steps:
                return -0.01, False, True  # Truncated
            elif target_reached:
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
