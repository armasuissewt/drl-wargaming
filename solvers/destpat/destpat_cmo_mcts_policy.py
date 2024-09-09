# destpat_cmo_mcts_policy.py
#
# Solve the CmoDestpatEnv problem with the MCTS policy-parallel algorithm
#
# Author: Giacomo Del Rio
# Creation date: 2024 Feb 19

import glob
import pickle
import time
from pathlib import Path
from typing import Optional, Dict

import ray
from ray.actor import ActorHandle

from environments.destpat.destpat_abstractions import DestpatAbstractions
from environments.destpat.destpat_env_cmo import DestpatRealState
from environments.destpat.destpat_plotter import DestpatPlotter
from environments.destpat.destpat_env_sim import SimDestpatEnv
from mcts_common.mcts_utils import get_render_file_info, get_class
from policy_parallel.mcts_policy_parallel_trainer import MctsPolicyParallelTrainer


@ray.remote
class DestpatAsyncPlotter:
    def __init__(self, config: Dict, epoch: int, plots_dir: Path, renders_dir: Path, shared_storage: ActorHandle):
        self.cfg: Dict = config
        self.epoch: int = epoch
        self.plots_dir: Path = plots_dir
        self.renders_dir: Path = renders_dir
        self.shared_storage: ActorHandle = shared_storage
        self.policy_class = get_class(self.cfg['policy_class'])
        n_cells = self.cfg['env_config']['n_cells']
        self.policy = self.policy_class(learning_rate=self.cfg['learning_rate'],
                                        obs_shape=(1, n_cells, n_cells),
                                        n_actions=5, vf_loss_coeff=self.cfg['vf_loss_coeff'],
                                        **self.cfg['policy_config'])
        self.abs_fn = DestpatAbstractions(n_cells, self.cfg['env_config']['obs_enc'], self.cfg['env_config']['rew_sig'],
                                          self.cfg['env_config']['max_steps'], SimDestpatEnv.map_limits,
                                          self.cfg['env_config']['traces_len'])
        self.last_plotted_policy_epoch = epoch
        self.policy_plot_interval = 10
        self.sample_real_state: Optional[DestpatRealState] = None
        self.plotter = DestpatPlotter(SimDestpatEnv.map_limits)

    def run(self):
        while True:
            self.plot_episodes(skip_last=True)
            self.plot_policy()
            time.sleep(2)

    def plot_policy(self):
        weights, self.epoch = ray.get(self.shared_storage.get_policy_weights.remote(self.epoch))
        if weights is not None:
            self.policy.set_weights(weights)

        if (self.epoch > self.last_plotted_policy_epoch + self.policy_plot_interval) and (
                self.sample_real_state is not None):
            self.plotter.plot_policy_full(self.policy, self.sample_real_state, self.abs_fn,
                                          n_dots=self.cfg['env_config']['n_cells'],
                                          out_file=self.plots_dir / f"value_{self.epoch}.png")
            self.last_plotted_policy_epoch = self.epoch

    def plot_episodes(self, skip_last: bool):
        eval_files = sorted([self.renders_dir / f for f in glob.glob("evaluation_*.pkl", root_dir=self.renders_dir)])
        if len(eval_files) > 0:
            self.plot_episodes_from_file_list(eval_files, self.plots_dir, delete_after_plot=True,
                                              skip_last=skip_last)
        rollout_files = sorted([self.renders_dir / f for f in glob.glob("rollout_*.pkl", root_dir=self.renders_dir)])
        if len(rollout_files) > 0:
            self.plot_episodes_from_file_list(rollout_files, self.plots_dir, delete_after_plot=True,
                                              skip_last=skip_last)

    def plot_episodes_from_file_list(self, render_files: list[Path], out_dir: Path, delete_after_plot: bool,
                                     skip_last: bool):
        current_prefix, current_epoch, current_in_epoch, _ = get_render_file_info(render_files[0])
        current_epoch_files = [render_files[0]]
        for f in render_files[1:]:
            f_prefix, f_epoch, f_in_epoch, _ = get_render_file_info(f)
            if f_prefix != current_prefix or f_epoch > current_epoch or f_in_epoch > current_in_epoch:
                out_file = out_dir / f"{current_prefix}_{current_epoch:06}-{current_in_epoch:04}.png"
                self.plot_episode(current_epoch_files, out_file)
                if delete_after_plot:
                    for r in current_epoch_files:
                        r.unlink()
                current_prefix = f_prefix
                current_epoch = f_epoch
                current_in_epoch = f_in_epoch
                current_epoch_files.clear()
            current_epoch_files.append(f)

        if current_epoch_files and not skip_last:
            out_file = out_dir / f"{current_prefix}_{current_epoch:06}-{current_in_epoch:04}.png"
            self.plot_episode(current_epoch_files, out_file)
            if delete_after_plot:
                for r in current_epoch_files:
                    r.unlink()

    def plot_episode(self, step_files: list[Path], out_file: Path):
        assert len(step_files) > 0
        trajectory = [pickle.load(open(f, "rb")) for f in step_files]
        self.plotter.plot_episode(trajectory, out_file)
        if self.sample_real_state is None:
            self.sample_real_state = trajectory[0]


class CallbackManager:
    def __init__(self):
        self.cfg: Optional[Dict] = None
        self.plots_dir: Optional[Path] = None
        self.renders_dir: Optional[Path] = None
        self.trainer: Optional[MctsPolicyParallelTrainer] = None
        self.plotter: Optional[ActorHandle] = None

    def on_train_begin(self, trainer: MctsPolicyParallelTrainer, shared_storage: ActorHandle, epoch: int):
        self.cfg = trainer.cfg
        self.plots_dir = trainer.out_dir / "plots"
        self.plots_dir.mkdir(parents=False, exist_ok=True)
        self.renders_dir = trainer.out_dir / "renders"
        self.trainer = trainer
        self.plotter = DestpatAsyncPlotter.remote(self.cfg, epoch, self.plots_dir, self.renders_dir, shared_storage)
        self.plotter.run.remote()

    def on_train_end(self, trainer: MctsPolicyParallelTrainer, shared_storage: ActorHandle, epoch: int):
        ray.kill(self.plotter)


def train_new():
    config = {
        # --- Environment
        'env_class': 'environments.destpat.destpat_env_cmo.CmoDestpatEnv',
        'env_config': {'max_steps': 60,
                       'timestep': 20,
                       'tick_hms': (0, 0, 20),
                       'obs_enc': 'full',
                       'rew_sig': 'penalties',
                       'n_cells': 100,
                       'traces_len': 3,
                       'render_mode': 'real_state'},
        'env_worker_id_param': True,
        'env_reset_seed': None,
        'env_restore_mode': 'save_restore',
        # --- Policy and replay buffer
        'policy_class': 'mcts_common.mcts_policy_torch.TorchMctsPolicy',
        'policy_config': {
            'net_type': 'convolutional',
            'conv_layers': [
                'Conv2d(50, (3, 3), (1, 1), (0, 0))',
                'ReLU()',
                'MaxPool2d((2, 2), (2, 2), (0, 0))',
                'Conv2d(100, (4, 4), (1, 1), (0, 0))',
                'ReLU()',
                'MaxPool2d((2, 2), (2, 2), (0, 0))'
            ],
            'dense_layers': [200, 100],
            'shared_layers': 0
        },
        'replay_buffer_class': 'mcts_common.mcts_policy_torch.TorchMctsReplayBuffer',
        'replay_buffer_config': {'max_size': 10_000},
        'replay_buffer_min_size': 200,
        # --- MCTS
        'discount_factor': 0.99,
        'rollout_start_node': 'leaf',
        'uct_policy_c': 1.0,
        'uct_exploration_c': 1.414,
        'step_temperature': 0.25,
        'train_temperature': 0.6,
        'num_expansions': 20,
        'max_rollout_steps': 0,
        'random_rollout': False,
        'step_criterion': 'qvalues',
        'value_estimation': 'tree',
        'add_exploration_noise': True,
        'exploration_noise_dirichlet_alpha': 0.2,
        'exploration_noise_fraction': 0.2,
        'reset_minmax_each_episode': True,
        'num_rollout_workers': 20,
        # --- Training
        'training_epochs': 2_000,
        'max_episode_steps': 1_000,
        'random_seed': None,
        'train_batch_size': 256,
        'random_batch': True,
        'learning_rate': 0.0001,
        'vf_loss_coeff': 1.0,
        'num_sgd_per_epoch': 20,
        'out_base_dir': 'out_destpat',
        'out_experiment_dir': None,
        'max_retry_on_error': 2,
        'checkpoint_freq': 10,
        'verbose': True,
        'logging_interval_secs': 30,
        'logging_aggregation_epochs': 5,
        'policy_train_max_rate': None,
        'render_interval': 1,
        'render_max_per_interval': 1,
        'workers_logging': True,
        # --- Evaluation
        'evaluation_config': {
            'reset_minmax_each_episode': True,
            'step_criterion': 'policy',
            'step_temperature': 0.0,
            'num_expansions': 0,
            'random_rollout': False,
            'max_rollout_steps': 0,
            'add_exploration_noise': False,
            'render_interval': 5
        },
        'custom': None
    }

    cb_manager = CallbackManager()
    trainer = MctsPolicyParallelTrainer(config,
                                        callbacks={'on_train_begin': cb_manager.on_train_begin,
                                                   'on_train_end': cb_manager.on_train_end})
    trainer.train()


def main():
    train_new()


if __name__ == "__main__":
    main()
