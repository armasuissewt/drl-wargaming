# flytozurich_cmo_mcts_policy.py
#
# Solve the FlyToZurich CMO problem with the MCTS policy-parallel algorithm
#
# Author: Giacomo Del Rio
# Creation date: 15 Jan 2024

import glob
import pickle
import time
from pathlib import Path
from typing import Optional, Dict

import ray
from ray.actor import ActorHandle

from environments.flytozurich.flytozurich_env_sim import FtzRealState
from mcts_common.mcts_utils import get_class, get_render_file_info
from policy_parallel.mcts_policy_parallel_trainer import MctsPolicyParallelTrainer
from utils.plotting import plot_episode, plot_value_map_dir


@ray.remote
class FtzAsyncPlotter:
    def __init__(self, config: Dict, epoch: int, plots_dir: Path, renders_dir: Path, shared_storage: ActorHandle):
        self.cfg: Dict = config
        self.epoch: int = epoch
        self.plots_dir: Path = plots_dir
        self.renders_dir: Path = renders_dir
        self.shared_storage: ActorHandle = shared_storage
        self.policy_class = get_class(self.cfg['policy_class'])
        n_cells = self.cfg['env_config']['n_cells']
        self.policy = self.policy_class(learning_rate=self.cfg['learning_rate'], obs_shape=(n_cells * n_cells + 2,),
                                        n_actions=3, vf_loss_coeff=self.cfg['vf_loss_coeff'],
                                        **self.cfg['policy_config'])
        self.last_plotted_policy_epoch = epoch
        self.policy_plot_interval = 10
        self.sample_real_state: Optional[FtzRealState] = None

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
            plot_value_map_dir(self.policy, self.sample_real_state,
                               self.cfg['env_config']['obs_enc'], self.cfg['env_config']['n_cells'],
                               self.plots_dir / f"value_{self.epoch}.png")
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
        plot_episode(trajectory, out_file)
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
        self.plotter = FtzAsyncPlotter.remote(self.cfg, epoch, self.plots_dir, self.renders_dir, shared_storage)
        self.plotter.run.remote()

    def on_train_end(self, trainer: MctsPolicyParallelTrainer, shared_storage: ActorHandle, epoch: int):
        ray.kill(self.plotter)


def train_new():
    config = {
        # --- Environment
        'env_class': 'environments.flytozurich.wrappers.flytozurich_cmo_wrappers.CmoFtzObsRewAdapterEnv',
        'env_config': {'max_steps': 150,
                       'timestep': 20,
                       'tick_hms': (0, 0, 20),
                       'obs_enc': 'Sparse_1',
                       'rew_sig': 'Naive',
                       'n_cells': 84,
                       'render_mode': 'real_state'},
        'env_worker_id_param': True,
        'env_reset_seed': None,
        'env_restore_mode': 'save_restore',  # save_restore, deepcopy
        # --- Policy and replay buffer
        'policy_class': 'mcts_common.mcts_policy_torch.TorchMctsPolicy',
        'policy_config': {
            'net_type': 'feedforward',
            'conv_layers': [],
            'dense_layers': [200, 100],
            'shared_layers': 0
        },
        'replay_buffer_class': 'mcts_common.mcts_policy_torch.TorchMctsReplayBuffer',
        'replay_buffer_config': {'max_size': 10_000},
        'replay_buffer_min_size': 10,
        # --- MCTS
        'discount_factor': 0.99,
        'rollout_start_node': 'leaf',  # 'root', 'leaf'
        'uct_policy_c': 1.0,
        'uct_exploration_c': 1.414,  # 1.414
        'step_temperature': 0.2,  # 'linear(0.2, 0.05, 1000)'
        'train_temperature': 0.5,
        'num_expansions': 50,
        'max_rollout_steps': 3,
        'random_rollout': False,
        'step_criterion': 'qvalues',  # counts, qvalues, policy
        'value_estimation': 'tree',  # tree, returns
        'add_exploration_noise': True,
        'exploration_noise_dirichlet_alpha': 0.25,
        'exploration_noise_fraction': 0.2,
        'reset_minmax_each_episode': True,
        'num_rollout_workers': 40,
        # --- Training
        'training_epochs': 10_000,
        'max_episode_steps': 1_000,
        'random_seed': None,
        'train_batch_size': 256,
        'random_batch': True,
        'learning_rate': 0.0001,  # 0.0001
        'vf_loss_coeff': 1.0,
        'num_sgd_per_epoch': 20,
        'out_base_dir': 'out_ftz',
        'out_experiment_dir': None,
        'max_retry_on_error': 2,
        'checkpoint_freq': 10,
        'verbose': True,
        'logging_interval_secs': 30,  # Do not set too low
        'logging_aggregation_epochs': 5,
        'policy_train_max_rate': 150,  # None or int
        'render_interval': 1,  # >0 for rendering
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


def train_resume():
    # --- Configuration
    chk_dir = Path(
        r'D:\Projects\CmoResearch\outnew\MCTS_SimFtzObsRewAdapterEnv_2023-06-20_18-12-48_76ee02\checkpoint_002000')

    # Create trainer
    cb_manager = CallbackManager()
    callbacks = {'on_train_begin': cb_manager.on_train_begin,
                 'on_train_end': cb_manager.on_train_end}
    trainer = MctsPolicyParallelTrainer.from_checkpoint(chk_dir, callbacks=callbacks)
    trainer.train()


def main():
    train_new()
    # train_resume()


if __name__ == "__main__":
    main()
