# mcts_rollout_worker.py
#
# A ray worker to execute MCTS episodes
#
# Author: Giacomo Del Rio
# Creation date: 5 Jun 2023

import logging
import pickle
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional

import numpy as np
import ray
import yaml

from mcts_common.mcts_policy_base import MctsPolicy
from mcts_common.mcts_utils import get_class, parse_interpolator, make_probability_vector, MinMax, setup_logger
from policy_parallel.shared_storage import MctsSharedStorage, RolloutSummary
from mcts_common.rollback_env import RollbackEnv
from single_thread.mcts_single import EpisodeStep, MCTSSingle
from single_thread.mcts_single_trainer import NetSample, EpisodeResult, MctsSingleTrainer


@ray.remote
class MctsRolloutWorker:

    def __init__(self, config: Dict, worker_id: int, shared_storage: MctsSharedStorage, out_dir: Path, is_eval: bool):
        # Config
        self.cfg: Dict = config
        self.worker_id: int = worker_id
        self.shared_storage = shared_storage
        self.render_dir = out_dir / "renders"
        self.logging_enabled = self.cfg['workers_logging']
        self.logger: Optional[logging.Logger] = None
        self.out_dir = out_dir
        self.is_eval = is_eval
        self.render_interval = self.cfg['render_interval']
        self.render_max_per_interval = self.cfg['render_max_per_interval']

        # Build environment
        self.safe_env: RollbackEnv = RollbackEnv(self.cfg['env_class'], self.cfg['env_config'],
                                                 self.cfg['env_restore_mode'], self.cfg['max_retry_on_error'],
                                                 worker_id=worker_id if self.cfg['env_worker_id_param'] else None)
        self.n_actions = self.safe_env.get_n_actions()

        # Build policy
        self.policy_class = get_class(self.cfg['policy_class'])
        self.policy: MctsPolicy = self.policy_class(learning_rate=self.cfg['learning_rate'],
                                                    obs_shape=self.safe_env.env.observation_space.shape,
                                                    n_actions=self.n_actions, vf_loss_coeff=self.cfg['vf_loss_coeff'],
                                                    **self.cfg['policy_config'])

        # Train variables
        self.epoch = 0
        self.interval = 0
        self.episode_in_current_interval = 0
        self.total_episodes = 0
        self.min_max = MinMax()
        self.step_temperature = parse_interpolator(self.cfg['step_temperature'], "Bad 'step_temperature' parameter.")
        self.train_temperature = parse_interpolator(self.cfg['train_temperature'], "Bad 'train_temperature' parameter.")
        self.rnd_gen = np.random.default_rng(self.cfg['random_seed'])

    def play(self):
        if self.logging_enabled:
            self.logger = setup_logger(self.out_dir / "workers_logs" / f"worker_{self.worker_id:03d}.log",
                                       to_console=False)
            self.safe_env.set_logger(self.logger)

        stop_flag = False
        while not stop_flag:
            try:
                weights, self.epoch = ray.get(self.shared_storage.get_policy_weights.remote(self.epoch))
                curr_interval = self.epoch - (self.epoch % self.render_interval) \
                    if self.render_interval > 0 else self.epoch
                if curr_interval > self.interval:
                    self.interval = curr_interval
                    self.episode_in_current_interval = 0
                if weights is not None:
                    self.policy.set_weights(weights)

                episode: EpisodeResult = self.safe_execute_episode()
                result = RolloutSummary(episode.samples, episode.reward, episode.time_elapsed,
                                        len(episode.trajectory) - 1,
                                        episode.n_rollouts, episode.n_rollouts_steps,
                                        (self.min_max.min, self.min_max.max),
                                        self.step_temperature.val(), self.train_temperature.val())
                stop_flag = ray.get(
                    self.shared_storage.update_on_episode.remote(self.worker_id, self.epoch, result, self.is_eval))
                self.step_temperature.update(self.epoch)
                self.train_temperature.update(self.epoch)
                self.episode_in_current_interval += 1
                self.total_episodes += 1

                if self.logging_enabled:
                    self.logger.info(f"Episode {self.total_episodes} completed: epoch={self.epoch}, "
                                     f"n_samples={len(episode.samples)}, reward={episode.reward}, "
                                     f"time={episode.time_elapsed}")
            except RuntimeError as e:
                if self.logging_enabled:
                    self.logger.error(f"Exception: {traceback.format_exc()}")
                if "The safe_execute_episode() failed with" in str(e):
                    print(f"WORKER[{self.worker_id}] Restarting in 5 second...", file=sys.stderr)
                    time.sleep(5)
                else:
                    raise e

    def execute_episode(self) -> EpisodeResult:
        trajectory: List[EpisodeStep] = []
        samples: List[NetSample] = []
        start_time = datetime.now()
        episode_reward = 0.0
        step = 0
        terminated, truncated = False, False
        current_obs, current_info = self.safe_env.reset_safe(self.cfg['env_reset_seed'])
        self.render(step)
        mcts = MCTSSingle(env=self.safe_env, policy=self.policy, root_obs=current_obs, root_info=current_info,
                          n_actions=self.n_actions, max_rollout_steps=self.cfg['max_rollout_steps'],
                          gamma=self.cfg['discount_factor'], uct_policy_c=self.cfg['uct_policy_c'],
                          uct_exploration_c=self.cfg['uct_exploration_c'],
                          add_exploration_noise=self.cfg['add_exploration_noise'],
                          exploration_noise_dirichlet_alpha=self.cfg['exploration_noise_dirichlet_alpha'],
                          exploration_noise_fraction=self.cfg['exploration_noise_fraction'],
                          random_rollout=self.cfg['random_rollout'], rollout_start_node=self.cfg['rollout_start_node'],
                          min_max=MinMax() if self.cfg['reset_minmax_each_episode'] else self.min_max)

        while not (terminated or truncated) and step < self.cfg['max_episode_steps']:
            mcts.build(n_nodes=self.cfg['num_expansions'])
            act_pref = mcts.root_action_preferences(self.cfg['step_criterion'])
            pi_step = make_probability_vector(act_pref, self.step_temperature.val())
            pi_train = make_probability_vector(act_pref, self.train_temperature.val())
            samples.append(NetSample(current_obs, pi_train, mcts.root.V))
            action = self.rnd_gen.choice(len(pi_step), p=pi_step)
            prev_obs = current_obs
            prev_info = current_info
            self.safe_env.save_checkpoint()
            current_obs, reward, terminated, truncated, current_info = self.safe_env.step_safe(action)
            trajectory.append(EpisodeStep(prev_obs, action, reward, False, False, prev_info))
            episode_reward += reward
            mcts.advance_root(action, self.safe_env, new_root_obs=current_obs, new_root_info=current_info)
            step += 1
            self.render(step)

        trajectory.append(EpisodeStep(current_obs, None, None, terminated, truncated, current_info))
        if self.cfg['value_estimation'] == 'returns':  # default 'tree'
            MctsSingleTrainer.set_value_estimations_to_returns(samples, trajectory, self.cfg['discount_factor'])

        episode_duration = datetime.now() - start_time
        return EpisodeResult(trajectory, samples, episode_reward, episode_duration, terminated,
                             mcts.total_rollouts, mcts.total_rollouts_steps)

    def render(self, step: int):
        if (self.render_interval > 0) and (self.worker_id <= 1) and \
                (self.episode_in_current_interval < self.render_max_per_interval):
            obj = self.safe_env.render()
            if obj is not None and not isinstance(obj, bool):
                name_suffix = f"{self.epoch:09}-{self.episode_in_current_interval:04}-{step:06}.pkl"
                if self.is_eval:
                    pickle.dump(obj, open(self.render_dir / f"evaluation_{name_suffix}", "wb"))
                else:
                    pickle.dump(obj, open(self.render_dir / f"rollout_{name_suffix}", "wb"))

    def safe_execute_episode(self) -> EpisodeResult:
        attempt = 0
        last_error = None
        while attempt < self.cfg['max_retry_on_error']:
            try:
                return self.execute_episode()
            except Exception as e:
                if self.logging_enabled:
                    self.logger.error(f"safe_execute_episode: {e}")
                print(f"WORKER[{self.worker_id}] safe_execute_episode: "
                      f"error {attempt + 1}/{self.cfg['max_retry_on_error']}. {e}",
                      file=sys.stderr)
                last_error = e
                attempt += 1
                self.safe_env.rebuild_env()
        raise RuntimeError(f"WORKER[{self.worker_id}] The safe_execute_episode() failed with {last_error}")

    def get_observation_space(self) -> Tuple:
        return self.safe_env.env.observation_space.shape

    def get_n_actions(self) -> int:
        return self.n_actions

    def load_checkpoint(self, checkpoint_dir: Union[Path, str]) -> None:
        """ Load state from a checkpoint directory.

        :param checkpoint_dir: the directory of a previously saved checkpoint
        """
        checkpoint_dir = Path(checkpoint_dir)
        trainer_state_file = checkpoint_dir / "trainer_state.yaml"
        with open(trainer_state_file) as f:
            state = yaml.safe_load(f)
            self.epoch = state["epoch"] - 1
            self.interval = self.epoch - (self.epoch % self.render_interval) if self.render_interval > 0 else self.epoch
            self.episode_in_current_interval = 0
            self.min_max.update(state['min_q'])
            self.min_max.update(state['max_q'])

        self.step_temperature.update(self.epoch)
        self.train_temperature.update(self.epoch)
