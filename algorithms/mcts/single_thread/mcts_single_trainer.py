# mcts_single_trainer.py
#
# An MCTS based reinforcement learning agent
#
# Author: Giacomo Del Rio
# Creation date: 12 Apr 2023

from __future__ import annotations

import csv
import logging
import lzma
import pickle
import sys
import tarfile
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from shutil import copyfile
from typing import List, Dict, Callable, Union, Optional

import numpy as np
import yaml
from torch.utils.tensorboard import SummaryWriter

from mcts_common.mcts_policy_base import MctsPolicy
from mcts_common.mcts_utils import get_class, merge_config, setup_logger, check_config, parse_interpolator, MinMax, \
    make_probability_vector
from mcts_common.rollback_env import RollbackEnv
from single_thread.mcts_single import MCTSSingle, ObsType, EpisodeStep


class NetSample:
    """ A NetSample is one training sample for the policy neural network,
        composed by an observation, an actions probabilities estimate and a value estimate
    """

    def __init__(self, s: ObsType, p: np.ndarray, v: float):
        self.s = s
        self.p = p
        self.v = v


class EpisodeResult:
    """ Contains all the results of an episode """

    def __init__(self, trajectory: List[EpisodeStep], samples: List[NetSample], reward: float, time_elapsed: timedelta,
                 terminated: bool, n_rollouts: int, n_rollouts_steps: int):
        self.trajectory: List[EpisodeStep] = trajectory  # the episode trajectory
        self.samples: List[NetSample] = samples  # training samples collected
        self.reward: float = reward  # total episode reward
        self.time_elapsed: timedelta = time_elapsed  # time elapsed
        self.terminated: bool = terminated  # a flag that is true iif the episode is terminated
        self.n_rollouts: int = n_rollouts  # total number of rollouts performed
        self.n_rollouts_steps: int = n_rollouts_steps  # total number of rollouts steps performed


class EvaluationResult:
    """ Contains the results of an evaluation episode """

    def __init__(self, epoch: int, num_evaluations: int, mean_reward: float, mean_episode_len: float,
                 eval_time_s: float):
        self.epoch: int = epoch
        self.num_evaluations: int = num_evaluations
        self.mean_reward: float = mean_reward
        self.mean_episode_len: float = mean_episode_len
        self.eval_time_s: float = eval_time_s


class EpochResult:
    """ Contains all the results of a single epoch of training """

    def __init__(self, epoch: int, epoch_time_s: float, mean_reward: float, mean_episode_len: float, num_rollouts: int,
                 num_rollouts_steps: int, total_episodes: int, total_time_s: float, total_rollouts: int,
                 total_rollouts_steps: int, dataset_size: int, network_pi_loss: float, network_v_loss: float,
                 evaluation: Optional[EvaluationResult]):
        self.epoch: int = epoch
        self.epoch_time_s: float = epoch_time_s
        self.mean_reward: float = mean_reward
        self.mean_episode_len: float = mean_episode_len
        self.num_rollouts: int = num_rollouts
        self.num_rollouts_steps: int = num_rollouts_steps

        self.total_episodes: int = total_episodes
        self.total_time_s: float = total_time_s
        self.total_rollouts: int = total_rollouts
        self.total_rollouts_steps: int = total_rollouts_steps

        self.dataset_size: int = dataset_size
        self.network_pi_loss: float = network_pi_loss
        self.network_v_loss: float = network_v_loss

        self.evaluation: Optional[EvaluationResult] = evaluation


class MctsSingleTrainer:
    """
        An MctsSingleTrainer is responsible for optimizing a given policy.

        It executes a given number of training epochs, each of those is composed by 'episodes_per_epoch' episodes.
        At the end of each epoch the policy is trained.

        Since MCTS is a model-based algorithm, the environment must support some mechanism to save and restore
        the state of the environment. This allows the algorithm to simulate starting from any previously saved state.
        Two mechanisms are supported:
            - the use of deepcopy()
                With the use of deepcopy(), the MCTS algorithm itself will make a copy of the environment when needed.
                Most pure Python environments allow this method.

            - the save() and restore() method.
                Use this mechanism when the state of the environment may reside outside the Python managed code
                (i.e. an external simulator)
                To use this method, the environment must be augmented with two methods whose signatures are:
                    save() -> Any
                    restore(saved_obj: Any) -> None
                You can use the standalone check_env_save_restore() function to check the env for the above two methods.
        Use the configuration parameter 'env_restore_mode' to be 'deepcopy' or 'save_restore'.
    """

    def __init__(self, config: Dict, callbacks: Dict[str, Callable] = None, validate_params: bool = True):
        """ Builds a new MctsSingleTrainer

        :param config: configuration dictionary. For a list of required configurations, see validate_config()
        :param callbacks: callbacks dictionary. Available callbacks:
            'on_train_begin': called every time the train() method is called
                signature: (trainer: MctsTrainer, epoch: int)
            'on_epoch_end': called at the end of each epoch
                signature: (epoch: int)
            'on_episode_end': called at the end of each episode
                signature: (epoch: int, episode_in_epoch: int, trajectory: List[EpisodeStep], episode_time: timedelta)
            'on_tree_build': called each time an expansion of the tree has been performed.
                signature: (epoch: int, episode_in_epoch: int, step: int, mcts: MCTS)
            'on_rollout': called at the end of each MCTS rollout
                signature: (trajectory: List[EpisodeStep], last_obs_value: float)
            'on_evaluation_episode_end': called at the end of each episode during evaluation
                signature: (epoch: int, episode_in_eval: int, trajectory: List[EpisodeStep], episode_time: timedelta)
        :param validate_params: if true, validate the config and the callbacks dictionary
        """

        # Validate configuration
        self.cfg = config
        self.callbacks = callbacks if callbacks else {}
        if validate_params:
            check_config(self.cfg, 'MctsSingleTrainer')
            MctsSingleTrainer.validate_callbacks(self.callbacks)
        self.rnd_gen = np.random.default_rng(self.cfg['random_seed'])

        # Build environment
        self.safe_env: RollbackEnv = RollbackEnv(self.cfg['env_class'], self.cfg['env_config'],
                                                 self.cfg['env_restore_mode'], self.cfg['max_retry_on_error'],
                                                 worker_id=0 if self.cfg['env_worker_id_param'] else None)
        self.n_actions = self.safe_env.get_n_actions()

        # Evaluation configuration and environment
        self.cfg_eval = merge_config(self.cfg, self.cfg['evaluation_config'])
        if self.cfg['evaluation_interval'] > 0 and ('env_config' in self.cfg['evaluation_config']):
            self.safe_eval_env: RollbackEnv = RollbackEnv(self.cfg['env_class'], self.cfg['env_config'],
                                                          self.cfg['env_restore_mode'], self.cfg['max_retry_on_error'],
                                                          worker_id=1 if self.cfg['env_worker_id_param'] else None)
        else:
            self.safe_eval_env: RollbackEnv = self.safe_env

        # Build policy network and replay buffer
        self.policy_class = get_class(self.cfg['policy_class'])
        self.policy: MctsPolicy = self.policy_class(learning_rate=self.cfg['learning_rate'],
                                                    obs_shape=self.safe_env.env.observation_space.shape,
                                                    n_actions=self.n_actions, vf_loss_coeff=self.cfg['vf_loss_coeff'],
                                                    **self.cfg['policy_config'])
        self.replay_buffer_class = get_class(self.cfg['replay_buffer_class'])
        self.replay_buffer = self.replay_buffer_class(obs_shape=self.safe_env.env.observation_space.shape,
                                                      n_actions=self.n_actions, **self.cfg['replay_buffer_config'])

        # Train variables
        self.epoch = 1
        self.episode_in_epoch = 0
        self.total_episodes: int = 0
        self.total_training_time: timedelta = timedelta(0)
        self.total_rollouts: int = 0
        self.total_rollouts_steps: int = 0
        self.in_evaluation = False
        self.min_max = MinMax()

        # Changing params
        self.step_temperature = parse_interpolator(self.cfg['step_temperature'], "Bad 'step_temperature' parameter.")
        self.train_temperature = parse_interpolator(self.cfg['train_temperature'], "Bad 'train_temperature' parameter.")
        self.eval_step_temperature = parse_interpolator(self.cfg_eval['step_temperature'],
                                                        "Bad 'step_temperature' parameter in eval.")

        # Output
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if self.cfg['out_experiment_dir'] is None:
            env_name = self.cfg['env_class'].split('.')[-1]
            uid = uuid.uuid4().hex[-6:].lower()
            self.out_dir: Path = Path(self.cfg['out_base_dir']) / f"MCTS_{env_name}_{self.timestamp}_{uid}"
        elif self.cfg['out_experiment_dir'] == '<current_dir>':
            self.out_dir: Path = Path().resolve()
        else:
            self.out_dir: Path = Path(self.cfg['out_base_dir']) / self.cfg['out_experiment_dir']
        self.logger: Optional[logging.Logger] = None
        self.tb_writer: Optional[SummaryWriter] = None
        self.csv_file: Path = self.out_dir / "results.csv"
        self.csv_file_eval: Path = self.out_dir / "results_eval.csv"

    def _setup_output(self) -> None:
        """ Create the output folder and set up the output objects
        """
        if self.logger is not None:
            return  # Output already setup

        self.out_dir.mkdir(parents=False, exist_ok=True)
        self.logger = setup_logger(self.out_dir / "mcts.log", to_console=self.cfg['verbose'])
        self.tb_writer = SummaryWriter(str(self.out_dir.absolute()))
        if not (self.out_dir / "config.yaml").exists():
            with open(self.out_dir / "config.yaml", mode="wt", encoding="utf-8") as file:
                yaml.dump(self.cfg, file)

        if not self.csv_file.exists():
            with open(self.csv_file, 'w', newline='') as f:
                csv.writer(f).writerow(['epoch', 'episode', 'episode_time_s', 'episode_reward', 'episode_len',
                                        'num_rollouts', 'num_rollouts_steps', 'replay_buffer', 'total_steps',
                                        'step_temperature', 'train_temperature'])

        if not self.csv_file_eval.exists():
            with open(self.csv_file_eval, 'w', newline='') as f:
                csv.writer(f).writerow(['train_epochs', 'train_episodes', 'eval_episode', 'episode_time_s',
                                        'episode_reward', 'episode_len', 'total_steps'])

    def train(self) -> None:
        """ Train the policy for the number of epochs specified in config 'training_epochs'
        """
        self._setup_output()
        self.logger.info(f"Training MCTS on {self.cfg['env_class']}")
        if 'on_train_begin' in self.callbacks:
            self.callbacks['on_train_begin'](self, self.epoch)

        while self.epoch <= self.cfg['training_epochs']:
            self.train_single()

    def train_single(self) -> EpochResult:
        """ Train the policy for a single epoch.
            It ignores the maximum number of epochs specified in config 'training_epochs'

            :return: a summary of the epoch
        """
        self._setup_output()
        self.logger.info(f"#### Training epoch {self.epoch}/{self.cfg['training_epochs']} ####")

        # Executes episodes
        trajectory = []
        epoch_total_reward = 0.0
        epoch_time = timedelta(0)
        epoch_rollouts = 0
        epoch_rollouts_steps = 0
        for self.episode_in_epoch in range(1, self.cfg['episodes_per_epoch'] + 1):
            self.logger.info(f" - Running episode {self.episode_in_epoch}/{self.cfg['episodes_per_epoch']}: ")

            episode: EpisodeResult = self.safe_execute_episode(self.safe_env)
            for sam in episode.samples:
                self.replay_buffer.push(sam.s, sam.p, sam.v)
            epoch_rollouts += episode.n_rollouts
            epoch_rollouts_steps += episode.n_rollouts_steps
            epoch_time += episode.time_elapsed
            epoch_total_reward += episode.reward
            trajectory = episode.trajectory
            self.total_episodes += 1
            self.total_rollouts += episode.n_rollouts
            self.total_rollouts_steps += episode.n_rollouts_steps
            self.logger.info(f"      len={len(trajectory) - 1}, reward={episode.reward:.4f}, "
                             f"time={episode.time_elapsed}, terminated={episode.terminated}")
            with open(self.csv_file, 'a', newline='') as f:
                csv.writer(f).writerow([
                    self.epoch, self.total_episodes, episode.time_elapsed.total_seconds(), episode.reward,
                    len(trajectory) - 1, episode.n_rollouts, episode.n_rollouts_steps, len(self.replay_buffer),
                    self.total_rollouts_steps, self.step_temperature.val(), self.train_temperature.val()])

            if 'on_episode_end' in self.callbacks:
                self.callbacks['on_episode_end'](self.epoch, self.episode_in_epoch, trajectory,
                                                 episode.time_elapsed)
        self.total_training_time += epoch_time

        # Train the network
        loss_p, loss_v = self.train_network()

        # Logging
        mean_reward = round(epoch_total_reward / self.cfg['episodes_per_epoch'], ndigits=4)
        mean_actions = round(len(trajectory) - 1 / self.cfg['episodes_per_epoch'], ndigits=1)
        throughput = epoch_rollouts_steps / epoch_time.total_seconds()
        self.logger.info(f" - Summary: epoch time: {epoch_time}, mean reward: {mean_reward}, "
                         f"mean actions: {mean_actions}, dataset size: {len(self.replay_buffer)}, "
                         f"throughput: {throughput:.2f} steps/sec")
        self.tb_writer.add_scalar("MCTS/Mean reward", mean_reward, self.epoch)
        self.tb_writer.add_scalar("MCTS/Mean episode length", mean_actions, self.epoch)
        self.tb_writer.add_scalar("MCTS/Throughput", throughput, self.epoch)
        self.tb_writer.add_scalar("MCTS/Steps", self.total_rollouts_steps, self.epoch)
        self.tb_writer.add_scalar("MCTS/Replay buffer", len(self.replay_buffer), self.epoch)
        self.tb_writer.add_scalar("MCTS/Step temperature", self.step_temperature.val(), self.epoch)
        self.tb_writer.add_scalar("MCTS/Train temperature", self.train_temperature.val(), self.epoch)
        if len(self.replay_buffer) >= self.cfg['replay_buffer_min_size']:
            self.tb_writer.add_scalar("Network/Pi loss", loss_p, self.epoch)
            self.tb_writer.add_scalar("Network/V loss", loss_v, self.epoch)

        if self.cfg['checkpoint_freq'] and (self.epoch % self.cfg['checkpoint_freq'] == 0):
            self.make_checkpoint()

        if 'on_epoch_end' in self.callbacks:
            self.callbacks['on_epoch_end'](self.epoch)

        eval_res = None
        if (self.epoch % self.cfg['evaluation_interval']) == 0:
            eval_res = self.evaluate()

        self.epoch += 1
        self.step_temperature.update(self.epoch - 1)
        self.train_temperature.update(self.epoch - 1)
        self.eval_step_temperature.update(self.epoch - 1)

        return EpochResult(self.epoch, epoch_time.total_seconds(), mean_reward, mean_actions, epoch_rollouts,
                           epoch_rollouts_steps, self.total_episodes, self.total_training_time.total_seconds(),
                           self.total_rollouts, self.total_rollouts_steps, len(self.replay_buffer), loss_p, loss_v,
                           evaluation=eval_res)

    def train_network(self) -> (float, float):
        """ One epoch of training """
        loss_p, loss_v = 0, 0
        if len(self.replay_buffer) >= self.cfg['replay_buffer_min_size']:
            for _ in range(self.cfg['num_sgd_per_epoch']):
                batch = self.replay_buffer.sample(self.cfg['train_batch_size'], random_sample=self.cfg['random_batch'])
                loss_p, loss_v = self.policy.train(batch)
        return loss_p, loss_v

    def evaluate(self) -> EvaluationResult:
        """ Evaluate the trained policy """
        # Save config
        self.in_evaluation = True
        tmp_cfg = self.cfg
        self.cfg = self.cfg_eval
        tmp_step_temp = self.step_temperature
        self.step_temperature = self.eval_step_temperature

        # Evaluate
        self.logger.info(f"Evaluation")
        total_reward = 0.0
        total_steps = 0.0
        evaluation_time = timedelta(0)
        for i in range(1, self.cfg['evaluation_duration'] + 1):
            self.logger.info(f" - Running episode {i}/{self.cfg['evaluation_duration']}: ")

            episode = self.safe_execute_episode(self.safe_eval_env)
            evaluation_time += episode.time_elapsed
            total_reward += episode.reward
            total_steps += len(episode.trajectory) - 1
            trajectory = episode.trajectory
            self.logger.info(f"      len={len(trajectory) - 1}, reward={episode.reward:.4f}, "
                             f"time={episode.time_elapsed}, terminated={episode.terminated}")
            with open(self.csv_file_eval, 'a', newline='') as f:
                csv.writer(f).writerow([
                    self.epoch, (self.epoch - 1) * self.cfg['episodes_per_epoch'] + self.episode_in_epoch, i,
                    episode.time_elapsed.total_seconds(), episode.reward, len(trajectory) - 1,
                    self.total_rollouts_steps])

            if 'on_evaluation_episode_end' in self.callbacks:
                self.callbacks['on_evaluation_episode_end'](self.epoch, i, trajectory, episode.time_elapsed)

        # Logging
        mean_reward = round(total_reward / self.cfg['evaluation_duration'], ndigits=4)
        mean_actions = round(total_steps / self.cfg['evaluation_duration'], ndigits=1)
        self.logger.info(f" - Summary: evaluation time: {evaluation_time}, mean reward: {mean_reward}, "
                         f"mean actions: {mean_actions}")

        self.tb_writer.add_scalar("Evaluation/Mean reward", mean_reward, self.epoch)
        self.tb_writer.add_scalar("Evaluation/Mean episode length", mean_actions, self.epoch)

        # Restore config
        self.cfg = tmp_cfg
        self.step_temperature = tmp_step_temp
        self.in_evaluation = False

        return EvaluationResult(self.epoch, self.cfg['evaluation_duration'], mean_reward, mean_actions,
                                evaluation_time.total_seconds())

    def execute_episode(self, safe_env: RollbackEnv) -> EpisodeResult:
        """ Execute one episode

        :param safe_env: the environment
        :return: an EpisodeResult object
        """
        trajectory: List[EpisodeStep] = []
        samples: List[NetSample] = []
        start_time = datetime.now()
        episode_reward = 0.0
        step = 0
        terminated, truncated = False, False
        current_obs, current_info = safe_env.reset_safe(self.cfg['env_reset_seed'])
        mcts = MCTSSingle(env=safe_env, policy=self.policy, root_obs=current_obs, root_info=current_info,
                          n_actions=self.n_actions, max_rollout_steps=self.cfg['max_rollout_steps'],
                          gamma=self.cfg['discount_factor'], uct_policy_c=self.cfg['uct_policy_c'],
                          uct_exploration_c=self.cfg['uct_exploration_c'],
                          add_exploration_noise=self.cfg['add_exploration_noise'],
                          exploration_noise_dirichlet_alpha=self.cfg['exploration_noise_dirichlet_alpha'],
                          exploration_noise_fraction=self.cfg['exploration_noise_fraction'],
                          random_rollout=self.cfg['random_rollout'], rollout_start_node=self.cfg['rollout_start_node'],
                          min_max=MinMax() if self.cfg['reset_minmax_each_episode'] else self.min_max,
                          callbacks={k: self.callbacks[k] for k in ['on_rollout', 'on_debug'] if k in self.callbacks})

        while not (terminated or truncated) and step < self.cfg['max_episode_steps']:
            mcts.build(n_nodes=self.cfg['num_expansions'])
            if 'on_tree_build' in self.callbacks and self.cfg['num_expansions'] > 0:
                self.callbacks['on_tree_build'](epoch=self.epoch, episode_in_epoch=self.episode_in_epoch, step=step,
                                                mcts=mcts)

            act_pref = mcts.root_action_preferences(self.cfg['step_criterion'])
            pi_step = make_probability_vector(act_pref, self.step_temperature.val())
            pi_train = make_probability_vector(act_pref, self.train_temperature.val())
            samples.append(NetSample(current_obs, pi_train, mcts.root.V))
            action = self.rnd_gen.choice(len(pi_step), p=pi_step)
            # ------- debug
            # print(f"Step {step}: act_pref={act_pref}, pi_step={pi_step}, pi_train={pi_train}, action={action}, "
            #       f"s_temp={self.step_temperature.val()}, t_temp={self.train_temperature.val()}")
            # ------- debug
            prev_obs = current_obs
            prev_info = current_info
            safe_env.save_checkpoint()
            current_obs, reward, terminated, truncated, current_info = safe_env.step_safe(action)
            trajectory.append(EpisodeStep(prev_obs, action, reward, False, False, prev_info))
            episode_reward += reward
            mcts.advance_root(action, safe_env, new_root_obs=current_obs, new_root_info=current_info)
            step += 1

        trajectory.append(EpisodeStep(current_obs, None, None, terminated, truncated, current_info))
        if self.cfg['value_estimation'] == 'returns':  # default 'tree'
            self.set_value_estimations_to_returns(samples, trajectory, self.cfg['discount_factor'])

        episode_duration = datetime.now() - start_time
        return EpisodeResult(trajectory, samples, episode_reward, episode_duration, terminated,
                             mcts.total_rollouts, mcts.total_rollouts_steps)

    def safe_execute_episode(self, safe_env: RollbackEnv) -> EpisodeResult:
        """ Execute self.execute_episode() and return the result.
            In case of exceptions, restart the episode from the beginning for a maximum of
            self.env.max_retry_on_error times.
        """
        attempt = 0
        last_error = None
        while attempt < self.cfg['max_retry_on_error']:
            try:
                return self.execute_episode(safe_env)
            except Exception as e:
                print(f"safe_execute_episode: error {attempt + 1}/{self.cfg['max_retry_on_error']}. {e}",
                      file=sys.stderr)
                last_error = e
                attempt += 1
                safe_env.rebuild_env()
        raise RuntimeError(f"The safe_execute_episode() failed with {last_error}")

    @staticmethod
    def set_value_estimations_to_returns(samples: List[NetSample], trajectory: List[EpisodeStep],
                                         gamma: float) -> None:
        """ Modifies the values in samples so that each one is the return computed over the trajectory

        :param samples: a list of samples to be modified as computed by execute_episode()
        :param trajectory: the trajectory of the episode that generated samples
        :param gamma: the discount factor
        """
        samples[-1].v = trajectory[-2].r if trajectory[-1].terminated else samples[-1].v
        for i in range(len(samples) - 2, -1, -1):
            samples[i].v = trajectory[i].r + gamma * samples[i + 1].v

    def set_callbacks(self, callbacks: Dict[str, Callable]) -> None:
        self.callbacks = callbacks if callbacks else {}
        MctsSingleTrainer.validate_callbacks(self.callbacks)

    @staticmethod
    def validate_callbacks(callbacks: Dict) -> None:
        """ Validate the callbacks' dictionary. If some keys are unknown a ValueError is raised

        :param callbacks: callbacks dictionary
        """
        available_keys = ['on_train_begin', 'on_epoch_end', 'on_episode_end', 'on_tree_build', 'on_rollout',
                          'on_evaluation_episode_end', 'on_debug']
        for k in callbacks:
            if k not in available_keys:
                raise ValueError(f"Unknown parameter '{k}' in callbacks")

    def make_checkpoint(self, chk_dir: Optional[Path] = None) -> Path:
        """ Make a checkpoint saving the current trainer state.

            Output directory will be: self.out_dir / checkpoint_{self.epoch}

        :param chk_dir: optional checkpoint directory. If None, create a new one inside out_dir
        :return: the checkpoint directory
        """
        self.logger.info(f" - Make checkpoint")

        # Create directory
        if chk_dir is None:
            chk_dir = self.out_dir / f"checkpoint_{self.epoch:03}"
            chk_dir.mkdir(parents=False, exist_ok=False)

        # Store weights and replay buffer
        self.policy.save_weights(chk_dir / "policy_net_weights")
        with lzma.open(chk_dir / "replay_buffer.pkl.lzma", "wb") as f:
            pickle.dump(self.replay_buffer, f)  # noqa

        # Config
        copyfile(self.out_dir / 'config.yaml', chk_dir / 'config.yaml')

        # Store logs
        copyfile(self.out_dir / 'mcts.log', chk_dir / 'mcts.log')
        copyfile(self.out_dir / 'results.csv', chk_dir / 'results.csv')
        copyfile(self.out_dir / 'results_eval.csv', chk_dir / 'results_eval.csv')
        with tarfile.open(chk_dir / 'tensorboard_logs.tar.gz', "w:gz") as tar:
            for f in self.out_dir.glob("*tfevents*"):
                tar.add(f, arcname=f.name)

        # Trainer state
        with open(chk_dir / "trainer_state.yaml", mode="wt", encoding="utf-8") as file:
            yaml.dump({
                "total_episodes": self.total_episodes,
                "total_training_time": self.total_training_time.total_seconds(),
                "total_rollouts": self.total_rollouts,
                "total_rollouts_steps": self.total_rollouts_steps,
                "epoch": self.epoch,
                "min_q": self.min_max.min,
                "max_q": self.min_max.max
            }, file)

        return chk_dir

    def _load_checkpoint(self, checkpoint_dir: Union[Path, str], load_replay_buffer: bool = True) -> None:
        """ Load current trainer state from a checkpoint directory.

        :param checkpoint_dir: the directory of a previously saved checkpoint
        :param load_replay_buffer: if True, load also the content of the replay buffer
        """
        self.logger.info(f"Loading checkpoint from \"{checkpoint_dir}\"")
        checkpoint_dir = Path(checkpoint_dir)
        policy_weights_file = sorted(checkpoint_dir.glob("policy_net_weights*"))[0]
        try:
            self.policy.load_weights(policy_weights_file)
        except pickle.UnpicklingError:
            # Fallback to import weights from policy-parallel
            with lzma.open(policy_weights_file, "rb") as f:
                weights = pickle.load(f)
                self.policy.set_weights(weights)
        self.logger.info(f" - Loaded \"{policy_weights_file}\"")

        if load_replay_buffer:
            replay_buffer_file = checkpoint_dir / "replay_buffer.pkl.lzma"
            with lzma.open(replay_buffer_file, "rb") as f:
                self.replay_buffer = pickle.load(f)
            self.logger.info(f" - Loaded \"{replay_buffer_file}\", samples={self.replay_buffer.size}")

        trainer_state_file = checkpoint_dir / "trainer_state.yaml"
        with open(trainer_state_file) as f:
            state = yaml.safe_load(f)
            self.epoch = state['epoch'] + 1
            self.total_episodes = state["total_episodes"]
            self.total_training_time = timedelta(
                state["total_training_time"]) if 'total_training_time' in state else timedelta(0)
            self.total_rollouts = state["total_rollouts"]
            self.total_rollouts_steps = state["total_rollouts_steps"]
            self.min_max = MinMax()
            if 'min_q' in state and 'max_q' in state:
                self.min_max.update(state["min_q"])
                self.min_max.update(state["max_q"])
            self.logger.info(f" - Loaded \"{trainer_state_file}\" (epoch={self.epoch})")

        self.step_temperature.update(self.epoch - 1)
        self.train_temperature.update(self.epoch - 1)
        self.eval_step_temperature.update(self.epoch - 1)

    @staticmethod
    def from_checkpoint(checkpoint_dir: Path | str, load_replay_buffer: bool = True,
                        callbacks: Dict[str, Callable] = None) -> MctsSingleTrainer:
        """ Build a new MctsSingleTrainer from a previously saved checkpoint.

            The 'out_experiment_dir' will be forced to None so that a new output folder is created.
            This is needed to avoid conflicts with new tensorboard files and checkpoints.

        :param checkpoint_dir: the directory of a previously saved checkpoint
        :param load_replay_buffer: if True, load also the content of the replay buffer
        :param callbacks: if present, the callbacks for the trainer
        :return: the newly built MctsSingleTrainer
        """
        with open(checkpoint_dir / 'config.yaml') as f:
            config = yaml.safe_load(f)
            config['out_experiment_dir'] = None

            # To enable cross-compatibility, remove extra configs from parallel versions
            extra_configs = ['num_rollout_workers', 'reset_tree_each_step', 'logging_interval_secs',
                             'logging_aggregation_epochs', 'policy_train_max_rate', 'render_interval',
                             'render_max_per_interval']
            for c in extra_configs:
                if c in config:
                    del config[c]

            # To enable cross-compatibility, add missing configs from parallel versions
            if 'episodes_per_epoch' not in config:
                config['episodes_per_epoch'] = 1
            if 'evaluation_interval' not in config:
                config['evaluation_interval'] = 1
            if 'evaluation_duration' not in config:
                config['evaluation_duration'] = 1

        # Build trainer instance
        trainer = MctsSingleTrainer(config, callbacks=callbacks)

        # Copy files from checkpoint
        trainer.out_dir.mkdir(parents=False, exist_ok=True)
        copyfile(checkpoint_dir / 'config.yaml', trainer.out_dir / 'config.yaml')
        copyfile(checkpoint_dir / 'mcts.log', trainer.out_dir / 'mcts.log')
        copyfile(checkpoint_dir / 'results.csv', trainer.out_dir / 'results.csv')
        copyfile(checkpoint_dir / 'results_eval.csv', trainer.out_dir / 'results_eval.csv')
        with tarfile.open(checkpoint_dir / 'tensorboard_logs.tar.gz', "r:gz") as tar:
            tar.extractall(path=trainer.out_dir)

        # Load trainer state from checkpoint
        trainer._setup_output()
        trainer._load_checkpoint(checkpoint_dir, load_replay_buffer=load_replay_buffer)
        return trainer
