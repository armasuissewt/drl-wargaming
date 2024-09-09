# flytozurich_sim_dqn.py
#
# Apply DQN solver to the SimFtzObsRewAdapterEnv environment
#
# Author: Giacomo Del Rio
# Creation date: 26 Feb 2024

from datetime import datetime
from pathlib import Path
from typing import Dict

import ray
from ray.rllib.algorithms.dqn import DQNConfig
from ray.tune.registry import register_env

from environments.flytozurich.flytozurich_plotter import FlyToZurichEnvPlotter
from environments.flytozurich.wrappers.flytozurich_sim_wrappers import SimFtzObsRewAdapterEnv
from utils.rllib_learner import RllibLearner


def env_creator(env_config: Dict, is_eval=False):
    return SimFtzObsRewAdapterEnv(**env_config)


def main():
    register_env("SimFtzObsRewAdapterEnv", env_creator)

    config = (
        DQNConfig()
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
                  rollout_fragment_length="auto",
                  batch_mode="complete_episodes",
                  ignore_worker_failures=True,
                  recreate_failed_workers=True,
                  )
        .framework("torch",
                   eager_tracing=True)
        .training(gamma=0.99,
                  lr=.0005,
                  noisy=False,
                  train_batch_size=64,
                  target_network_update_freq=500,
                  v_min=-11,
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
            'initial_epsilon': 0.7,
            'final_epsilon': 0.01,
            'epsilon_timesteps': 500_000}
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
        experiment_name=f"DQN_SIM-FTZ_{datetime.now():%Y-%m-%d_%H-%M-%S}",
        code_files=[Path(__file__)],
        training_steps=2_000,
        eval_period=5,
        checkpoint_period=10,
        env_creator=env_creator,
        config=config,
        plotter=FlyToZurichEnvPlotter()
    )
    learner.learn()


if __name__ == '__main__':
    main()
