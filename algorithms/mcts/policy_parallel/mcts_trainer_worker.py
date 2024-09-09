# mcts_trainer_worker.py
#
# A ray worker to train a MCTS policy
#
# Author: Giacomo Del Rio
# Creation date: 5 Jun 2023

import lzma
import pickle
import time
from pathlib import Path
from typing import Dict, Tuple, Optional, Union

import ray
import yaml

from mcts_common.mcts_policy_base import MctsPolicy
from mcts_common.mcts_utils import get_class
from policy_parallel.shared_storage import MctsSharedStorage


@ray.remote
class MctsTrainerWorker:

    def __init__(self, config: Dict, shared_storage: MctsSharedStorage):
        # Config
        self.cfg: Dict = config
        self.shared_storage = shared_storage

        # Build policy
        self.policy_class = get_class(self.cfg['policy_class'])
        self.policy: Optional[MctsPolicy] = None
        self.batch_size = self.cfg['train_batch_size']
        self.num_sgd_per_epoch = self.cfg['num_sgd_per_epoch']
        self.policy_train_max_rate = self.cfg['policy_train_max_rate']
        self.epoch = 0

    def initialize_policy(self, obs_shape: Tuple, n_actions: int):
        self.policy = self.policy_class(learning_rate=self.cfg['learning_rate'], obs_shape=obs_shape,
                                        n_actions=n_actions, vf_loss_coeff=self.cfg['vf_loss_coeff'],
                                        **self.cfg['policy_config'])

    def train(self):
        stop_flag = False
        while not stop_flag:
            loss_p, loss_v = 0, 0
            total_samples, batch = 0, None
            for _ in range(self.num_sgd_per_epoch):
                total_samples, batch = ray.get(self.shared_storage.get_training_batch.remote())
                if batch is None:
                    break
                loss_p, loss_v = self.policy.train(batch)

            if batch is None:
                time.sleep(0.5)
            else:
                self.epoch, stop_flag = ray.get(
                    self.shared_storage.set_policy_weights.remote(self.policy.get_weights(), loss_p, loss_v))

            if self.policy_train_max_rate is not None and total_samples > 0:
                num_trained_samples = self.epoch * self.batch_size * self.num_sgd_per_epoch
                while num_trained_samples / total_samples > self.policy_train_max_rate:
                    time.sleep(0.5)
                    total_samples = ray.get(self.shared_storage.get_total_samples.remote())

    def load_checkpoint(self, checkpoint_dir: Union[Path, str]) -> None:
        """ Load state from a checkpoint directory.

        :param checkpoint_dir: the directory of a previously saved checkpoint
        """
        checkpoint_dir = Path(checkpoint_dir)

        policy_weights_file = sorted(checkpoint_dir.glob("policy_net_weights*"))[0]
        with lzma.open(policy_weights_file, "rb") as f:
            weights = pickle.load(f)
            self.policy.set_weights(weights)

        trainer_state_file = checkpoint_dir / "trainer_state.yaml"
        with open(trainer_state_file) as f:
            state = yaml.safe_load(f)
            self.epoch = state["epoch"] - 1
