# shared_storage.py
#
# A ray object to share state among workers
#
# Author: Giacomo Del Rio
# Creation date: 5 Jun 2023

import csv
import lzma
import pickle
import tarfile
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from shutil import copyfile
from typing import Optional, Tuple, Dict, Any, List, Union

import numpy as np
import ray
import yaml
from torch.utils.tensorboard import SummaryWriter

from mcts_common.mcts_policy_base import MctsReplayBuffer
from mcts_common.mcts_utils import get_class, MinMax
from single_thread.mcts_single_trainer import NetSample


class RolloutSummary:
    """ Holds summary results of a rollout """

    def __init__(self, train_samples: List[NetSample], total_reward: float, episode_time: timedelta,
                 length: int, n_rollouts: int, n_rollouts_steps: int, min_max: Tuple[float, float],
                 step_temperature: float, train_temperature: float):
        self.train_samples: List[NetSample] = train_samples  # training samples collected
        self.total_reward: float = total_reward  # total episode reward
        self.episode_time: timedelta = episode_time  # time elapsed
        self.length: int = length  # number of steps of the episode
        self.n_rollouts: int = n_rollouts  # total number of rollouts performed
        self.n_rollouts_steps: int = n_rollouts_steps  # total number of rollouts steps performed
        self.min_max: Tuple[float, float] = min_max  # Minimum and maximum Q value of the worker
        self.step_temperature: float = step_temperature  # total number of rollouts steps performed
        self.train_temperature: float = train_temperature  # total number of rollouts steps performed


class TrainingStats:
    """ Holds summary statistics of the overall training process """

    def __init__(self, epoch: int, total_episodes: int, total_samples: int, total_rollouts: int,
                 total_rollouts_steps: int, replay_buffer_size: int, last_eval_reward: float,
                 workers_epoch: np.ndarray):
        self.epoch: int = epoch  # Number of times the policy has been trained
        self.total_episodes: int = total_episodes  # Number of played games by rollout workers
        self.total_samples: int = total_samples  # Total number of samples added to the replay buffer
        self.total_rollouts: int = total_rollouts  # Total rollouts
        self.total_rollouts_steps: int = total_rollouts_steps  # Total steps of all the rollouts
        self.replay_buffer_size: int = replay_buffer_size  # Size of the replay buffer
        self.last_eval_reward: float = last_eval_reward  # Reward of the last evaluation episode
        self.workers_epoch: np.ndarray = workers_epoch  # Last completed rollout for workers


class RolloutStats:
    """ Holds summary statistics of an episode (either training or eval) """

    def __init__(self, epoch: int, worker_id: int, total_reward: float, episode_length: float, episode_time: timedelta,
                 n_rollouts_steps: int, min_max: Tuple[float, float], step_temperature: float,
                 train_temperature: float):
        self.epoch: int = epoch  # Training epoch
        self.worker_id: int = worker_id  # Worker id
        self.total_reward: float = total_reward  # Sum of the rewards of the episode
        self.episode_length: float = episode_length  # Length of episode
        self.episode_time: timedelta = episode_time  # Running time of the episode
        self.n_rollouts_steps: int = n_rollouts_steps  # Number of rollout steps
        self.min_max: Tuple[float, float] = min_max  # Minimum and maximum Q value of the worker
        self.step_temperature: float = step_temperature  # Step temperature of episode
        self.train_temperature: float = train_temperature  # Train temperatures of episode


class PolicyStats:
    """ Holds summary statistics of policy training """

    def __init__(self, epoch: int, loss_p: float = 0, loss_v: float = 0):
        self.epoch: int = epoch  # Training epoch
        self.loss_p: float = loss_p  # Policy loss of the last training step
        self.loss_v: float = loss_v  # Value loss of the last training step


@ray.remote
class MctsSharedStorage:

    def __init__(self, config: Dict, out_dir: Path):
        # Config
        self.config: Dict = config

        # Policy weights
        self.policy_weights: Optional[ray.ObjectRef] = None
        self.epoch: int = 0  # Epoch is incremented each time the policy_weights are updated

        # Replay buffer
        self.replay_buffer_class = get_class(self.config['replay_buffer_class'])
        self.replay_buffer: Optional[MctsReplayBuffer] = None

        # Min-Max Q
        self.min_max: MinMax = MinMax()

        # Statistics
        self.policy_stats: Dict[int, PolicyStats] = {0: PolicyStats(0, 0, 0)}
        self.rollout_stats: Dict[int, List[RolloutStats]] = defaultdict(list)  # epoch -> rollouts
        self.eval_stats: Dict[int, List[RolloutStats]] = defaultdict(list)  # epoch -> rollouts
        self.totals_stats: Dict[int, List[int]] = defaultdict(list)  # epoch -> []
        self.workers_epoch: np.ndarray = np.zeros(shape=(self.config['num_rollout_workers'] + 1,))

        # Training statistics (summary)
        self.total_episodes: int = 0  # Number of played games by rollout workers
        self.total_samples: int = 0  # Total number of samples generated by rollout workers
        self.total_rollouts: int = 0  # Total rollouts
        self.total_rollouts_steps: int = 0  # Total steps of all the rollouts
        self.last_eval_reward: float = np.nan  # Last evaluation reward

        # Stop flag
        self.stop_flag: bool = False

        # Logging
        self.out_dir: Path = out_dir
        self.tb_writer: Optional[SummaryWriter] = None
        self.csv_episodes: Path = self.out_dir / "episodes.csv"
        self.csv_episodes_eval: Path = self.out_dir / "episodes_eval.csv"
        self.last_logged_epoch: int = -1

    def raise_stop_flag(self):
        self.stop_flag = True

    def clear_stop_flag(self):
        self.stop_flag = False

    def initialize_replay_buffer(self, obs_shape: Tuple, n_actions: int):
        self.replay_buffer = self.replay_buffer_class(obs_shape=obs_shape, n_actions=n_actions,
                                                      **self.config['replay_buffer_config'])

    def get_policy_weights(self, version: int) -> Tuple[Optional[ray.ObjectRef], int]:
        if self.epoch > version and self.policy_weights is not None:
            return self.policy_weights, self.epoch
        else:
            return None, self.epoch

    def set_policy_weights(self, weights: ray.ObjectRef, loss_p: float, loss_v: float) -> Tuple[int, bool]:
        self.policy_weights = weights
        self.epoch += 1
        self.policy_stats[self.epoch] = PolicyStats(self.epoch, loss_p, loss_v)
        if self.config['checkpoint_freq'] and (self.epoch % self.config['checkpoint_freq'] == 0):
            self.make_checkpoint()
        return self.epoch, self.stop_flag

    def get_training_batch(self) -> Tuple[int, Any]:
        if len(self.replay_buffer) >= self.config['replay_buffer_min_size']:
            return self.total_samples, self.replay_buffer.sample(self.config['train_batch_size'],
                                                                 random_sample=self.config['random_batch'])
        else:
            return self.total_samples, None

    def get_total_samples(self) -> int:
        return self.total_samples

    def update_on_episode(self, worker_id: int, epoch: int, episode: RolloutSummary, is_eval: bool) -> bool:
        assert worker_id > 0 or (is_eval and worker_id == 0)
        assert epoch <= self.epoch

        self.workers_epoch[worker_id] = epoch
        if is_eval:
            self.eval_stats[epoch].append(
                RolloutStats(epoch, worker_id, episode.total_reward, episode.length, episode.episode_time,
                             episode.n_rollouts_steps, episode.min_max, episode.step_temperature,
                             episode.train_temperature))
            self.last_eval_reward = episode.total_reward
        else:
            for sam in episode.train_samples:
                self.replay_buffer.push(np.copy(sam.s), np.copy(sam.p), sam.v)
            self.min_max.update(episode.min_max[0])
            self.min_max.update(episode.min_max[1])
            self.total_episodes += 1
            self.total_samples += len(episode.train_samples)
            self.total_rollouts += episode.n_rollouts
            self.total_rollouts_steps += episode.n_rollouts_steps
            self.rollout_stats[epoch].append(
                RolloutStats(epoch, worker_id, episode.total_reward, episode.length, episode.episode_time,
                             episode.n_rollouts_steps, episode.min_max, episode.step_temperature,
                             episode.train_temperature))

        return self.stop_flag

    def get_training_stats(self) -> TrainingStats:
        return TrainingStats(self.epoch, self.total_episodes, self.total_samples, self.total_rollouts,
                             self.total_rollouts_steps, len(self.replay_buffer), self.last_eval_reward,
                             self.workers_epoch)

    def log_stats(self, elapsed_seconds: float, is_last: bool = False):
        if not self.out_dir.exists():
            return

        if self.tb_writer is None:
            self.tb_writer = SummaryWriter(str(self.out_dir.absolute()))

        chunk = self.config['logging_aggregation_epochs']
        if is_last:
            max_loggable_epoch = int((self.epoch // chunk) * chunk)
        else:
            max_loggable_epoch = int((np.min(self.workers_epoch) // chunk) * chunk)

        for epoch in range(self.last_logged_epoch + chunk, max_loggable_epoch + 1, chunk):
            epoch_num_training_episodes = 0
            epoch_sum_of_training_total_reward = 0
            epoch_sum_of_training_episodes_len = 0
            epoch_sum_of_training_rollout_steps = 0
            epoch_sum_of_training_total_time = timedelta(0)
            epoch_sum_of_training_step_temperature = 0
            epoch_sum_of_training_train_temperature = 0
            epoch_training_min_max = MinMax()

            epoch_num_eval_episodes = 0
            epoch_sum_of_eval_total_reward = 0
            epoch_sum_of_eval_episodes_len = 0
            epoch_eval_min_max = MinMax()

            epoch_num_policy_stats = 0
            epoch_sum_of_value_loss = 0
            epoch_sum_of_policy_loss = 0

            for i in range(epoch - chunk + 1, epoch + 1):
                for rs in self.rollout_stats[i]:
                    assert i == rs.epoch
                    epoch_num_training_episodes += 1
                    epoch_sum_of_training_total_reward += rs.total_reward
                    epoch_sum_of_training_episodes_len += rs.episode_length
                    epoch_sum_of_training_rollout_steps += rs.n_rollouts_steps
                    epoch_sum_of_training_total_time += rs.episode_time
                    epoch_sum_of_training_step_temperature += rs.step_temperature
                    epoch_sum_of_training_train_temperature += rs.train_temperature
                    epoch_training_min_max.update(rs.min_max[0])
                    epoch_training_min_max.update(rs.min_max[1])
                for rs in self.eval_stats[i]:
                    assert i == rs.epoch
                    epoch_num_eval_episodes += 1
                    epoch_sum_of_eval_total_reward += rs.total_reward
                    epoch_sum_of_eval_episodes_len += rs.episode_length
                    epoch_eval_min_max.update(rs.min_max[0])
                    epoch_eval_min_max.update(rs.min_max[1])
                if i in self.policy_stats and i != 0:
                    assert i == self.policy_stats[i].epoch
                    epoch_num_policy_stats += 1
                    epoch_sum_of_value_loss += self.policy_stats[i].loss_v
                    epoch_sum_of_policy_loss += self.policy_stats[i].loss_p

            if epoch_num_training_episodes > 0:
                self.tb_writer.add_scalar("Train/Episodes per epoch", epoch_num_training_episodes, epoch)
                self.tb_writer.add_scalar("Train/Mean reward",
                                          epoch_sum_of_training_total_reward / epoch_num_training_episodes, epoch)
                self.tb_writer.add_scalar("Train/Mean episode length",
                                          epoch_sum_of_training_episodes_len / epoch_num_training_episodes, epoch)
                mean_episode_time = epoch_sum_of_training_total_time.total_seconds() / epoch_num_training_episodes
                self.tb_writer.add_scalar("Train/Seconds_per_episode", mean_episode_time, epoch)
                throughput = epoch_sum_of_training_rollout_steps / epoch_sum_of_training_total_time.total_seconds()
                self.tb_writer.add_scalar("Train/Steps_per_second", throughput, epoch)
                self.tb_writer.add_scalar("Train/Step temperature",
                                          epoch_sum_of_training_step_temperature / epoch_num_training_episodes, epoch)
                self.tb_writer.add_scalar("Train/Train temperature",
                                          epoch_sum_of_training_train_temperature / epoch_num_training_episodes, epoch)
                self.tb_writer.add_scalar("Train/Q-Min", epoch_training_min_max.min, epoch)
                self.tb_writer.add_scalar("Train/Q-Max", epoch_training_min_max.max, epoch)

            if epoch_num_eval_episodes > 0:
                self.tb_writer.add_scalar("Evaluation/Episodes per epoch", epoch_num_eval_episodes, epoch)
                self.tb_writer.add_scalar("Evaluation/Mean reward",
                                          epoch_sum_of_eval_total_reward / epoch_num_eval_episodes, epoch)
                self.tb_writer.add_scalar("Evaluation/Mean episode length",
                                          epoch_sum_of_eval_episodes_len / epoch_num_eval_episodes, epoch)
                # Do not log q-min and q-max for evaluation: should always be 'reset_minmax_each_episode' = True
                # self.tb_writer.add_scalar("Evaluation/Q-Min", epoch_eval_min_max.min, epoch)
                # self.tb_writer.add_scalar("Evaluation/Q-Max", epoch_eval_min_max.max, epoch)

            if epoch_num_policy_stats > 0:
                self.tb_writer.add_scalar("Network/Pi loss", epoch_sum_of_policy_loss / epoch_num_policy_stats, epoch)
                self.tb_writer.add_scalar("Network/V loss", epoch_sum_of_value_loss / epoch_num_policy_stats, epoch)

        self.tb_writer.add_scalar("Train/Episodes total", self.total_episodes, self.epoch)
        self.tb_writer.add_scalar("Train/Samples total", self.total_samples, self.epoch)
        self.tb_writer.add_scalar("Train/Seconds total", elapsed_seconds, self.epoch)
        self.tb_writer.add_scalar("Train/Steps total", self.total_rollouts_steps, self.epoch)
        self.tb_writer.add_scalar("Train/Replay buffer size", len(self.replay_buffer), self.epoch)

        # Write episodes csv
        with open(self.csv_episodes, 'a', newline='') as f:
            for epoch in range(self.last_logged_epoch + chunk, max_loggable_epoch + 1, chunk):
                for i in range(epoch - chunk + 1, epoch + 1):
                    for rs in self.rollout_stats[i]:
                        csv.writer(f).writerow(
                            [rs.epoch, rs.worker_id, rs.total_reward, rs.episode_length,
                             rs.episode_time.total_seconds(), rs.n_rollouts_steps, rs.min_max[0],
                             rs.min_max[1], rs.step_temperature, rs.train_temperature])

        with open(self.csv_episodes_eval, 'a', newline='') as f:
            for epoch in range(self.last_logged_epoch + chunk, max_loggable_epoch + 1, chunk):
                for i in range(epoch - chunk + 1, epoch + 1):
                    for rs in self.eval_stats[i]:
                        csv.writer(f).writerow(
                            [rs.epoch, rs.worker_id, rs.total_reward, rs.episode_length,
                             rs.episode_time.total_seconds(), rs.n_rollouts_steps, rs.min_max[0], rs.min_max[1],
                             rs.step_temperature, rs.train_temperature])

        self.last_logged_epoch = max_loggable_epoch

    def make_checkpoint(self) -> Path:
        """ Make a checkpoint saving the current trainer state.

            Output directory will be: self.out_dir / checkpoint_{self.epoch}

        :return: the checkpoint directory
        """
        # Create directory
        chk_dir = self.out_dir / f"checkpoint_{self.epoch:06}"
        chk_dir.mkdir(parents=False, exist_ok=False)

        # Store weights and replay buffer
        with lzma.open(chk_dir / "policy_net_weights.pkl.lzma", "wb") as f:
            pickle.dump(self.policy_weights, f)
        with lzma.open(chk_dir / "replay_buffer.pkl.lzma", "wb") as f:
            pickle.dump(self.replay_buffer, f)

        # Config
        copyfile(self.out_dir / 'config.yaml', chk_dir / 'config.yaml')

        # Store logs
        copyfile(self.out_dir / 'mcts.log', chk_dir / 'mcts.log')
        copyfile(self.out_dir / 'episodes.csv', chk_dir / 'episodes.csv')
        copyfile(self.out_dir / 'episodes_eval.csv', chk_dir / 'episodes_eval.csv')
        with tarfile.open(chk_dir / 'tensorboard_logs.tar.gz', "w:gz") as tar:
            for f in self.out_dir.glob("*tfevents*"):
                tar.add(f, arcname=f.name)

        # Trainer state
        with open(chk_dir / "trainer_state.yaml", mode="wt", encoding="utf-8") as file:
            yaml.dump({
                "epoch": self.epoch,
                "total_episodes": self.total_episodes,
                "total_samples": self.total_samples,
                "total_rollouts": self.total_rollouts,
                "total_rollouts_steps": self.total_rollouts_steps,
                "last_eval_reward": self.last_eval_reward,
                "min_q": self.min_max.min,
                "max_q": self.min_max.max,
            }, file)

        return chk_dir

    def load_checkpoint(self, checkpoint_dir: Union[Path, str], load_replay_buffer: bool = True) -> None:
        """ Load current trainer state from a checkpoint directory.

        :param checkpoint_dir: the directory of a previously saved checkpoint
        :param load_replay_buffer: if True, load also the content of the replay buffer
        """
        checkpoint_dir = Path(checkpoint_dir)
        policy_weights_file = sorted(checkpoint_dir.glob("policy_net_weights*"))[0]
        with lzma.open(policy_weights_file, "rb") as f:
            self.policy_weights = pickle.load(f)

        if load_replay_buffer:
            replay_buffer_file = checkpoint_dir / "replay_buffer.pkl.lzma"
            with lzma.open(replay_buffer_file, "rb") as f:
                self.replay_buffer = pickle.load(f)

        trainer_state_file = checkpoint_dir / "trainer_state.yaml"
        with open(trainer_state_file) as f:
            state = yaml.safe_load(f)
            self.epoch = state["epoch"]
            self.total_episodes = state["total_episodes"]
            self.total_samples = state["total_samples"]
            self.total_rollouts = state["total_rollouts"]
            self.total_rollouts_steps = state["total_rollouts_steps"]
            self.last_eval_reward = state["last_eval_reward"]
            self.min_max.update(state["min_q"])
            self.min_max.update(state["max_q"])
