# flytozurich_sim_ppo.py
#
# Apply PPO solver to the SimFtzObsRewAdapterEnv environment
#
# Author: Giacomo Del Rio
# Creation date: 26 Feb 2024

from datetime import datetime
from pathlib import Path
from typing import Dict

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

from environments.flytozurich.flytozurich_plotter import FlyToZurichEnvPlotter
from environments.flytozurich.wrappers.flytozurich_sim_wrappers import SimFtzObsRewAdapterEnv
from utils.rllib_learner import RllibLearner


def env_creator(env_config: Dict, is_eval=False):
    return SimFtzObsRewAdapterEnv(**env_config)


def main():
    register_env("SimFtzObsRewAdapterEnv", env_creator)

    config = (
        PPOConfig()
        .environment(env="SimFtzObsRewAdapterEnv",
                     env_config={'max_steps': 150,
                                 'timestep': 20,
                                 'obs_enc': 'Sparse_1',
                                 'rew_sig': 'Naive',
                                 'n_cells': 100,
                                 'render_mode': 'real_state'},
                     disable_env_checking=True,
                     render_env=False)
        .rollouts(num_rollout_workers=10,
                  rollout_fragment_length='auto',
                  batch_mode="complete_episodes",
                  ignore_worker_failures=True
                  )
        .fault_tolerance(recreate_failed_workers=True)
        .framework("torch",
                   eager_tracing=True)
        .training(gamma=0.99,
                  model={"fcnet_hiddens": [256, 128],
                         "vf_share_layers": False},
                  train_batch_size=200,
                  sgd_minibatch_size=128,
                  num_sgd_iter=32,
                  vf_clip_param=12.0,
                  entropy_coeff_schedule=[[0, 0.1], [35_000, 0.01], [70_000, 0.0]],
                  )
        .evaluation(
            evaluation_interval=None,
            evaluation_num_workers=0,
            evaluation_duration=1,
            evaluation_duration_unit="episodes",
            evaluation_config={"explore": True}
        )
        .debugging(log_level="ERROR")
    )

    ray.init(address='auto')
    learner = RllibLearner(
        out_dir=Path(r"out_ftz"),
        experiment_name=f"FTZ-SIM_PPO_{datetime.now():%Y-%m-%d_%H-%M-%S}",
        code_files=[Path(__file__)],
        training_steps=4_000,
        eval_period=20,
        checkpoint_period=50,
        env_creator=env_creator,
        config=config,
        plotter=FlyToZurichEnvPlotter()
    )
    learner.learn()


if __name__ == '__main__':
    main()
