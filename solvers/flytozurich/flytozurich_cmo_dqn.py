# flytozurich_cmo_dqn.py
#
# Apply DQN solver to the CmoFlyToZurichEnv environment
#
# Author: Giacomo Del Rio
# Creation date: 12 Jan 2024

from datetime import datetime
from pathlib import Path
from typing import Dict

from ray.rllib.algorithms.dqn import DQNConfig
from ray.tune.registry import register_env

from environments.flytozurich.flytozurich_plotter import FlyToZurichEnvPlotter
from environments.flytozurich.wrappers.flytozurich_cmo_wrappers import CmoFtzObsRewAdapterEnv
from utils.rllib_learner import RllibLearner


def env_creator(env_config: Dict, is_eval=False):
    env_id = 0 if is_eval else env_config.worker_index  # noqa
    return CmoFtzObsRewAdapterEnv(**env_config, worker_id=env_id)


def main():
    register_env("CmoFtzObsRewAdapterEnv", env_creator)

    config = (
        DQNConfig()
        .environment(env="CmoFtzObsRewAdapterEnv",
                     env_config={
                         'max_steps': 80, 'timestep': 20, 'tick_hms': (0, 0, 20),
                         'obs_enc': 'Sparse_4', 'rew_sig': 'Naive',
                         'n_cells': 84, 'render_mode': 'real_state'
                     },
                     disable_env_checking=True,
                     render_env=False)
        .rollouts(num_rollout_workers=10,
                  rollout_fragment_length="auto",
                  batch_mode="complete_episodes",
                  ignore_worker_failures=True,
                  recreate_failed_workers=True
                  )
        .framework("torch",
                   eager_tracing=True)
        .training(gamma=0.99,
                  lr=.0005,  # default 0.0005
                  noisy=False,
                  # model={"fcnet_hiddens": [128, 64]},
                  train_batch_size=64,
                  target_network_update_freq=500,  # default 500
                  v_min=-11,
                  v_max=10,
                  n_step=5,
                  replay_buffer_config={
                      "type": "MultiAgentPrioritizedReplayBuffer",
                      "capacity": 5_000,
                      "prioritized_replay_alpha": 0.8,
                      "prioritized_replay_beta": 0.4,
                      "prioritized_replay_eps": 1e-6},
                  num_steps_sampled_before_learning_starts=100
                  )
        .exploration(exploration_config={
            'type': "EpsilonGreedy",
            'initial_epsilon': 0.9,  # default 1.0
            'final_epsilon': 0.01,  # default 0.02
            'epsilon_timesteps': 500_000}  # default 10_000
        )
        .evaluation(
            evaluation_interval=None,
            evaluation_num_workers=0,
            evaluation_duration=1,
            evaluation_duration_unit="episodes",
            evaluation_config={"explore": True}
        )
        .debugging(log_level="ERROR")
        # .reporting(min_sample_timesteps_per_iteration=1000)
    )

    learner = RllibLearner(
        out_dir=Path(r"out_ftz"),
        experiment_name=f"DQN_CMO-FTZ_{datetime.now():%Y-%m-%d_%H-%M-%S}",
        code_files=[Path(__file__)],
        training_steps=2_000,
        eval_period=2,
        checkpoint_period=5,
        env_creator=env_creator,
        config=config,
        plotter=FlyToZurichEnvPlotter()
    )
    learner.learn()


if __name__ == '__main__':
    main()
