# mcts_policy_parallel_trainer.py
#
# An MCTS based reinforcement learning agent with policy parallelization
#
# Author: Giacomo Del Rio
# Creation date: 5 Jun 2023

from __future__ import annotations

import csv
import logging
import tarfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from shutil import copyfile
from typing import Dict, Callable, Optional, Tuple, Union

import ray
import yaml

from mcts_common.mcts_utils import check_config, merge_config, setup_logger
from policy_parallel.mcts_rollout_worker import MctsRolloutWorker
from policy_parallel.mcts_trainer_worker import MctsTrainerWorker
from policy_parallel.shared_storage import MctsSharedStorage, TrainingStats


class MctsPolicyParallelTrainer:
    """
        An MctsPolicyParallelTrainer is responsible for optimizing a given policy.
    """

    def __init__(self, config: Dict, callbacks: Dict[str, Callable] = None, validate_params: bool = True):
        """ Builds a new MctsPolicyParallelTrainer

        :param config: configuration dictionary. For a list of required configurations, see validate_config()
        :param callbacks: callbacks dictionary. See MctsSingleTrainer class.
        """

        # Validate configuration
        self.cfg = config
        self.callbacks = callbacks if callbacks else {}
        if validate_params:
            check_config(self.cfg, 'MctsPolicyParallelTrainer')
            MctsPolicyParallelTrainer.validate_callbacks(self.callbacks)

        # Output
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if self.cfg['out_experiment_dir'] is None:
            env_name = self.cfg['env_class'].split('.')[-1]
            uid = uuid.uuid4().hex[-6:].lower()
            self.out_dir: Path = Path(self.cfg['out_base_dir']) / f"MCTS_{env_name}_{self.timestamp}_{uid}"
        elif self.cfg['out_experiment_dir'] == '<current_dir>':
            self.out_dir: Path = Path().resolve()
        else:
            self.out_dir: Path = Path(self.cfg['out_base_dir']) / self.cfg['out_experiment_dir']
        self.logger: Optional[logging.Logger] = None
        self.csv_episodes: Path = self.out_dir / "episodes.csv"
        self.csv_episodes_eval: Path = self.out_dir / "episodes_eval.csv"
        self.workers_logs_dir = self.out_dir / "workers_logs" if self.cfg['workers_logging'] else None

        # Create actors
        self.shared_storage = MctsSharedStorage.remote(config, self.out_dir)
        self.rollout_workers = []
        for i in range(self.cfg['num_rollout_workers']):
            self.rollout_workers.append(
                MctsRolloutWorker.remote(config, worker_id=i + 1, shared_storage=self.shared_storage,
                                         out_dir=self.out_dir, is_eval=False)
            )
        self.policy_trainer = MctsTrainerWorker.remote(config, self.shared_storage)
        self.cfg_eval = merge_config(self.cfg, self.cfg['evaluation_config'])
        self.eval_worker = MctsRolloutWorker.remote(self.cfg_eval, worker_id=0, shared_storage=self.shared_storage,
                                                    out_dir=self.out_dir, is_eval=True)

        # Initialize actors
        self.observation_space: Tuple = ray.get(self.rollout_workers[0].get_observation_space.remote())
        self.n_actions: int = ray.get(self.rollout_workers[0].get_n_actions.remote())
        ray.get(self.shared_storage.initialize_replay_buffer.remote(self.observation_space, self.n_actions))
        ray.get(self.policy_trainer.initialize_policy.remote(self.observation_space, self.n_actions))
        self.epoch: int = 0

    def _setup_output(self):
        """ Create the output folder and set up the output objects
        """
        if self.logger is not None:
            return  # Output already setup

        self.out_dir.mkdir(parents=False, exist_ok=True)
        if (self.cfg['render_interval'] > 0) or \
                ('render_interval' in self.cfg['evaluation_config'] and
                 self.cfg['evaluation_config']['render_interval'] > 0):
            (self.out_dir / 'renders').mkdir(parents=False, exist_ok=True)

        self.logger = setup_logger(self.out_dir / "mcts.log", to_console=self.cfg['verbose'])
        if not (self.out_dir / "config.yaml").exists():
            with open(self.out_dir / "config.yaml", mode="wt", encoding="utf-8") as file:
                yaml.dump(self.cfg, file)

        if self.workers_logs_dir is not None:
            self.workers_logs_dir.mkdir(parents=False, exist_ok=True)

        # CSV files
        if not self.csv_episodes.exists():
            with open(self.csv_episodes, 'w', newline='') as f:
                csv.writer(f).writerow(
                    ['epoch', 'worker_id', 'total_reward', 'episode_length', 'episode_time', 'n_rollouts_steps',
                     'min_q', 'max_q', 'step_temperature', 'train_temperature'])

        if not self.csv_episodes_eval.exists():
            with open(self.csv_episodes_eval, 'w', newline='') as f:
                csv.writer(f).writerow(
                    ['epoch', 'worker_id', 'total_reward', 'episode_length', 'episode_time', 'n_rollouts_steps',
                     'min_q', 'max_q', 'step_temperature', 'train_temperature'])

    def train(self) -> None:
        """ Train the policy for the number of epochs specified in config 'training_epochs'
        """
        self._setup_output()
        self.logger.info(f"Training MCTS on {self.cfg['env_class']}")
        if 'on_train_begin' in self.callbacks:
            self.callbacks['on_train_begin'](self, self.shared_storage, self.epoch)

        # Start the rollout workers and trainer
        for i in range(self.cfg['num_rollout_workers']):
            self.rollout_workers[i].play.remote()
        self.policy_trainer.train.remote()
        self.eval_worker.play.remote()

        # Log progress
        start_time = datetime.now()
        last_log_time = start_time
        while self.epoch <= self.cfg['training_epochs']:
            stats: TrainingStats = ray.get(self.shared_storage.get_training_stats.remote())
            if stats.epoch > self.epoch and 'on_epoch_end' in self.callbacks:
                self.callbacks['on_epoch_end'](self, self.shared_storage, stats.epoch)
            self.epoch = stats.epoch
            elapsed_time = datetime.now() - start_time
            self.logger.info(
                f"Epoch: {self.epoch}, "
                f"Episodes: {stats.total_episodes:,}, "
                f"Samples: {stats.total_samples:,}, "
                f"Replay buffer: {stats.replay_buffer_size:,}, "
                f"Total steps: {stats.total_rollouts_steps:,}, "
                f"Eval: {stats.last_eval_reward:04f}, "
                f"Time: {elapsed_time}")
            self.logger.info(
                f"Worker's epochs: {[int(f) for f in stats.workers_epoch]}")
            if (datetime.now() - last_log_time).total_seconds() >= self.cfg['logging_interval_secs']:
                self.shared_storage.log_stats.remote(elapsed_time.total_seconds())
                last_log_time = datetime.now()

            time.sleep(5)

        # Stop training
        self.shared_storage.raise_stop_flag.remote()
        elapsed_time = datetime.now() - start_time
        self.shared_storage.log_stats.remote(elapsed_time.total_seconds(), is_last=True)
        time.sleep(2)
        if 'on_train_end' in self.callbacks:
            self.callbacks['on_train_end'](self, self.shared_storage, self.epoch)

    @staticmethod
    def validate_callbacks(callbacks: Dict) -> None:
        """ Validate the callbacks' dictionary. If some keys are unknown a ValueError is raised

        :param callbacks: callbacks dictionary
        """
        available_keys = ['on_train_begin', 'on_train_end', 'on_epoch_end']
        for k in callbacks:
            if k not in available_keys:
                raise ValueError(f"Unknown parameter '{k}' in callbacks")

    def _load_checkpoint(self, checkpoint_dir: Union[Path, str], load_replay_buffer: bool = True) -> None:
        """ Load current trainer state from a checkpoint directory.

        :param checkpoint_dir: the directory of a previously saved checkpoint
        :param load_replay_buffer: if True, load also the content of the replay buffer
        """
        self.logger.info(f"Loading checkpoint from \"{checkpoint_dir}\"")
        checkpoint_dir = Path(checkpoint_dir)
        trainer_state_file = checkpoint_dir / "trainer_state.yaml"
        with open(trainer_state_file) as f:
            state = yaml.safe_load(f)
            self.epoch = state['epoch']

    @staticmethod
    def from_checkpoint(checkpoint_dir: Path | str, load_replay_buffer: bool = True,
                        callbacks: Dict[str, Callable] = None) -> MctsPolicyParallelTrainer:
        """ Build a new MctsPolicyParallelTrainer from a previously saved checkpoint.

            The 'out_experiment_dir' will be forced to None so that a new output folder is created.
            This is needed to avoid conflicts with new tensorboard files and checkpoints.

        :param checkpoint_dir: the directory of a previously saved checkpoint
        :param load_replay_buffer: if True, load also the content of the replay buffer
        :param callbacks: if present, the callbacks for the trainer
        :return: the newly built MctsPolicyParallelTrainer
        """
        checkpoint_dir = Path(checkpoint_dir)
        with open(checkpoint_dir / 'config.yaml') as f:
            config = yaml.safe_load(f)
            config['out_experiment_dir'] = None

        # Build trainer instance
        trainer = MctsPolicyParallelTrainer(config, callbacks=callbacks)

        # Copy files from checkpoint
        trainer.out_dir.mkdir(parents=False, exist_ok=True)
        copyfile(checkpoint_dir / 'config.yaml', trainer.out_dir / 'config.yaml')
        copyfile(checkpoint_dir / 'mcts.log', trainer.out_dir / 'mcts.log')
        copyfile(checkpoint_dir / 'episodes.csv', trainer.out_dir / 'episodes.csv')
        copyfile(checkpoint_dir / 'episodes_eval.csv', trainer.out_dir / 'episodes_eval.csv')
        with tarfile.open(checkpoint_dir / 'tensorboard_logs.tar.gz', "r:gz") as tar:
            tar.extractall(path=trainer.out_dir)

        # Load trainer state from checkpoint
        trainer._setup_output()
        trainer._load_checkpoint(checkpoint_dir, load_replay_buffer=load_replay_buffer)

        # Load state into workers
        trainer.shared_storage.load_checkpoint.remote(checkpoint_dir, load_replay_buffer)
        for i in range(trainer.cfg['num_rollout_workers']):
            trainer.rollout_workers[i].load_checkpoint.remote(checkpoint_dir)
        trainer.policy_trainer.load_checkpoint.remote(checkpoint_dir)
        trainer.eval_worker.load_checkpoint.remote(checkpoint_dir)

        return trainer
