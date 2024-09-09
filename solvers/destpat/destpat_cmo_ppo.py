# destpat_cmo_ppo.py
#
# Apply PPO solver to the CmoDestpatEnv environment
#
# Author: Giacomo Del Rio
# Creation date: 20 Feb 2024

from datetime import datetime
from pathlib import Path
from typing import Dict

from ray.rllib.algorithms.ppo import PPOConfig
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
        PPOConfig()
        .environment(env="CmoDestpatEnv",
                     env_config={'max_steps': 60,
                                 'timestep': 20,
                                 'tick_hms': (0, 0, 20),
                                 'obs_enc': 'full_chunky',
                                 'rew_sig': 'penalties',
                                 'n_cells': 84,
                                 'traces_len': 3,
                                 'render_mode': 'real_state'},
                     disable_env_checking=True,
                     render_env=False)
        .rollouts(num_rollout_workers=40,
                  rollout_fragment_length='auto',
                  batch_mode="complete_episodes",
                  ignore_worker_failures=True,
                  recreate_failed_workers=True
                  )
        .framework("torch",
                   eager_tracing=True)
        .training(gamma=0.99,
                  train_batch_size=500,
                  sgd_minibatch_size=64,
                  num_sgd_iter=32,
                  vf_clip_param=10.0,
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
        out_dir=Path(r"out_destpat"),
        experiment_name=f"DESTPAT-CMO_PPO_{datetime.now():%Y-%m-%d_%H-%M-%S}",
        code_files=[Path(__file__)],
        training_steps=2_000,
        eval_period=2,
        checkpoint_period=5,
        env_creator=env_creator,
        config=config,
        plotter=DestpatPlotter()
    )
    learner.learn()


if __name__ == '__main__':
    main()
