# flytozurich_cmo_ppo.py
#
# Apply PPO solver to the CmoFlyToZurichEnv environment
#
# Author: Giacomo Del Rio
# Creation date: 18 May 2022

from datetime import datetime
from pathlib import Path
from typing import Dict

from ray.rllib.algorithms.ppo import PPOConfig
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
        PPOConfig()
        .environment(env="CmoFtzObsRewAdapterEnv",
                     env_config={
                         'max_steps': 80, 'timestep': 20, 'tick_hms': (0, 0, 20),
                         'obs_enc': 'Sparse_4', 'rew_sig': 'Naive',
                         'n_cells': 84, 'render_mode': 'real_state'
                     },
                     disable_env_checking=True,
                     render_env=False)
        .rollouts(num_rollout_workers=10,
                  rollout_fragment_length=50,
                  batch_mode="complete_episodes",
                  ignore_worker_failures=True,
                  recreate_failed_workers=True
                  )
        .framework("torch",
                   eager_tracing=True)
        .training(gamma=0.95,
                  # lr=0.001
                  # model={"fcnet_hiddens": [128, 64]},
                  train_batch_size=500,
                  sgd_minibatch_size=64,
                  num_sgd_iter=32,
                  vf_clip_param=10.0,
                  # entropy_coeff=0.1,
                  # "kl_coeff": 0.2,
                  # "kl_target ": 0.01,
                  # "entropy_coeff_schedule": None,
                  # "entropy_coeff": 0.0,
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

    learner = RllibLearner(
        out_dir=Path(r"out_ftz"),
        experiment_name=f"PPO_CMO-FTZ_{datetime.now():%Y-%m-%d_%H-%M-%S}",
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
