# rllib_learner.py
#
# Helper class to apply an RLLib algorithm to an environment
#
# The environment must fill the info dictionary with a 'real_state' item
#
# Author: Giacomo Del Rio
# Creation date: 11 Jan 2024

import lzma
import pickle
import shutil
import statistics
import sys
from copy import copy
from datetime import timedelta
from pathlib import Path
from typing import List, Union, Callable, Dict, Optional, Tuple, Any

import gymnasium as gym
import pandas as pd
from ray.rllib.algorithms import AlgorithmConfig, Algorithm
from ray.tune.logger import Logger, UnifiedLogger


class RllibLearner:

    def __init__(self, out_dir: Path, experiment_name: str, code_files: List[Union[Path, str]],
                 training_steps: int, eval_period: int, checkpoint_period: int,
                 env_creator: Callable[[Dict, bool], gym.Env], config: AlgorithmConfig,
                 plotter: Any):
        self.out_dir = out_dir / experiment_name
        self.experiment_name = experiment_name
        self.code_files = code_files
        self.training_steps = training_steps
        self.eval_period = eval_period
        self.checkpoint_period = checkpoint_period
        self.env_creator = env_creator
        self.config = config
        self.plotter = plotter
        self.evaluation_max_retry = 5

        self.eval_env: Optional[gym.Env] = None
        self.trainer: Algorithm | None = None
        self.progress_csv_path = self.out_dir / "progress.csv"
        self.progress_plot_path = self.out_dir / "progress.png"
        self.evaluation_csv_path = self.out_dir / "evaluation.csv"
        self.evaluation_plot_path = self.out_dir / "evaluation.png"
        self.traces_dir = self.out_dir / "eval_traces"
        self.plots_dir = self.out_dir / "plots"

    def learn(self):
        self.prepare_directories()
        self.eval_env = self.env_creator(self.config.environment().env_config, True)
        self.trainer = self.config.build(logger_creator=custom_logger_creator(self.out_dir))
        self.learn_loop()

    def learn_loop(self):
        for i in range(1, self.training_steps + 1, 1):
            result = self.trainer.train()
            elapsed = timedelta(seconds=result['time_total_s'])
            print(f"Step {i}/{self.training_steps} ({elapsed}):"
                  f"  Reward (Max/Avg/Min): {result['episode_reward_max']:.2f}/"
                  f"{result['episode_reward_mean']:.2f}/{result['episode_reward_min']:.2f}  "
                  f"Episodes: {result['episodes_total']}  "
                  f"Ep. len mean: {result['episode_len_mean']:.2f}")

            self.plot_training_stats()

            if i % self.eval_period == 0 or i == self.training_steps:
                self.evaluate_agent()

            if i % self.checkpoint_period == 0 or i == self.training_steps:
                self.make_checkpoint()

    def plot_training_stats(self):
        progress = pd.read_csv(self.progress_csv_path)
        self.plot_progress(progress, "Training", self.progress_plot_path)

    def evaluate_agent(self):
        # Run episodes
        episodes_total = self.trainer._episodes_total  # noqa
        eval_lst = []
        for i in range(self.trainer.config['evaluation_duration']):
            trace, reward = self.execute_episode()
            plot_path = self.plots_dir / f"eval_episode-{self.trainer.iteration:05}_{i:02}.png"
            self.plotter.plot_episode(trace, plot_path)
            eval_lst.append({'train_iter': self.trainer.iteration,
                             'train_episodes': episodes_total,
                             'train_time': self.trainer._time_total,  # noqa
                             'eval_run': i,
                             'reward': reward,
                             'steps': len(trace),
                             'trace': trace})

        traces_pkl = self.traces_dir / f"eval_traces-{self.trainer.iteration:05}.pkl.lzma"
        with lzma.open(traces_pkl, "wb") as _f:
            pickle.dump(eval_lst, _f)  # noqa

        # Make evaluation plots
        eval_csv = pd.read_csv(self.evaluation_csv_path) if self.evaluation_csv_path.exists() else None
        new_row = pd.DataFrame({
            'training_iteration': self.trainer.iteration,
            'episode_reward_max': max([d['reward'] for d in eval_lst]),
            'episode_reward_mean': statistics.mean([d['reward'] for d in eval_lst]),
            'episode_reward_min': min([d['reward'] for d in eval_lst]),
            'episode_len_mean': statistics.mean([d['steps'] for d in eval_lst]),
            'episodes_total': episodes_total
        }, index=[0])
        eval_csv = pd.concat([eval_csv, new_row], ignore_index=True) if eval_csv is not None else new_row
        eval_csv.to_csv(self.evaluation_csv_path, index=False)
        self.plot_progress(eval_csv, "Evaluation", self.evaluation_plot_path)

    def make_checkpoint(self):
        checkpoint_dir = self.out_dir / f"checkpoint_{self.trainer.iteration:05}"
        self.trainer.save(str(checkpoint_dir))
        print(f"Checkpoint saved at {checkpoint_dir}", )

    def execute_episode(self) -> Tuple[List, float]:
        for _ in range(self.evaluation_max_retry):
            try:
                trace: List = []
                s, info = self.eval_env.reset()
                trace.append(copy(info['real_state']))
                step = 0
                reward = 0
                terminated = truncated = False
                while not (terminated or truncated):
                    a = self.trainer.compute_single_action(observation=s,
                                                           explore=self.trainer.config['evaluation_config']['explore'])
                    s1, r, terminated, truncated, info = self.eval_env.step(a)
                    trace.append(copy(info['real_state']))
                    reward += r
                    s = s1
                    step += 1
                return trace, reward
            except Exception as e:
                print(f"evaluate_agent: {e}", file=sys.stderr)
                pass

        raise RuntimeError(f"More than {self.evaluation_max_retry} errors in a single evaluation.")

    def prepare_directories(self):
        self.out_dir.mkdir(parents=False, exist_ok=False)
        code_dir = self.out_dir / "code"
        code_dir.mkdir(parents=False, exist_ok=False)
        self.traces_dir.mkdir(exist_ok=False)
        self.plots_dir.mkdir(exist_ok=False)
        for f in self.code_files:
            shutil.copy2(Path(f), code_dir)

    @staticmethod
    def plot_progress(progress: pd.DataFrame, title: str, out_file: Path):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        _fig = plt.figure(figsize=(20, 10), dpi=300)
        gs = _fig.add_gridspec(ncols=1, nrows=2, height_ratios=[5, 5])
        ax = gs.subplots(sharex=True)

        ax[0].title.set_text('Reward')
        ax[0].plot(progress['episodes_total'], progress['episode_reward_max'], label='max', color="green", lw=1, ls=':')
        ax[0].plot(progress['episodes_total'], progress['episode_reward_mean'], label='mean', color="black", lw=1)
        ax[0].plot(progress['episodes_total'], progress['episode_reward_min'], label='min', color="red", lw=1, ls=':')
        ax[0].set_xlabel('Epochs')
        ax[0].set_ylabel('Reward')
        ax[0].legend(loc="best")

        ax[1].title.set_text('Mean episode length')
        ax[1].plot(progress['episodes_total'], progress['episode_len_mean'], color="red", lw=1)
        ax[1].set_xlabel('Epochs')
        ax[1].set_ylabel('Steps')

        _fig.suptitle(title)
        plt.savefig(out_file, bbox_inches='tight')
        plt.close()


def custom_logger_creator(log_dir: Path) -> Callable[[], Logger]:
    def logger_creator(config):
        log_dir.mkdir(exist_ok=True, parents=False)
        return UnifiedLogger(config, str(log_dir), loggers=None)

    return logger_creator  # noqa
