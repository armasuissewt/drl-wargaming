# destpat_cmo_dqn.py
#
# Apply DQN solver to the CmoDestpatEnv environment
#
# Author: Giacomo Del Rio
# Creation date: 20 Feb 2024

from datetime import datetime
from pathlib import Path
from typing import Dict

import ray
from ray.rllib.algorithms.dqn import DQNConfig
from ray.tune.registry import register_env

from environments.destpat.destpat_env_cmo import CmoDestpatEnv
from environments.destpat.destpat_plotter import DestpatPlotter
from utils.rllib_learner import RllibLearner


def env_creator(env_config: Dict, is_eval=False):
    env_id = 0 if is_eval else env_config.worker_index  # noqa
    return CmoDestpatEnv(**env_config, worker_id=env_id)


def main():
    register_env("CmoDestpatEnv", env_creator)

    config = (
        DQNConfig()
        .environment(env="CmoDestpatEnv",
                     env_config={'max_steps': 60,
                                 'timestep': 15,
                                 'tick_hms': (0, 0, 15),
                                 'obs_enc': 'full_chunky',
                                 'rew_sig': 'penalties',
                                 'n_cells': 84,
                                 'traces_len': 4,
                                 'render_mode': 'real_state'},
                     disable_env_checking=True,
                     render_env=False)
        .rollouts(num_rollout_workers=20,
                  rollout_fragment_length="auto",
                  batch_mode="complete_episodes"
                  )
        .fault_tolerance(recreate_failed_workers=True,
                         max_num_worker_restarts=1_000)
        .framework("torch",
                   eager_tracing=True)
        .training(gamma=0.99,
                  lr=.0005,
                  noisy=False,
                  train_batch_size=64,
                  target_network_update_freq=500,  # default 500
                  v_min=-80,
                  v_max=10,
                  n_step=5,
                  replay_buffer_config={
                      "type": "MultiAgentPrioritizedReplayBuffer",
                      "capacity": 500_000,
                      "prioritized_replay_alpha": 0.8,
                      "prioritized_replay_beta": 0.4,
                      "prioritized_replay_eps": 1e-6},
                  num_steps_sampled_before_learning_starts=100
                  )
        .exploration(exploration_config={
            'type': "EpsilonGreedy",
            'initial_epsilon': 0.8,
            'final_epsilon': 0.02,
            'epsilon_timesteps': 250_000}
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
        out_dir=Path(r"out_destpat"),
        experiment_name=f"DESTPAT-CMO_DQN_{datetime.now():%Y-%m-%d_%H-%M-%S}",
        code_files=[Path(__file__)],
        training_steps=1_500,
        eval_period=2,
        checkpoint_period=5,
        env_creator=env_creator,
        config=config,
        plotter=DestpatPlotter()
    )
    learner.learn()


if __name__ == '__main__':
    main()
