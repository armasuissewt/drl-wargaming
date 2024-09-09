# mcts_utils.py
#
# MCTS standalone utility functions
#
# Author: Giacomo Del Rio
# Creation date: 14 Apr 2023

from __future__ import annotations

import copy
import logging
import sys
from collections import namedtuple
from pathlib import Path
from typing import Dict, Union, TypeVar, Literal

import numpy as np

# Used to represent an observation from the environment
ObsType = TypeVar("ObsType")


class MinMax:
    """
        Holds a couple (max, min) which can be used to normalize values in between
    """

    def __init__(self):
        self.max = -float("inf")
        self.min = float("inf")

    def update(self, value: float) -> None:
        """
        Update the max/min with value

        :param value: a number
        """
        self.max = max(self.max, value)
        self.min = min(self.min, value)

    def normalize(self, value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Normalize value to [0, 1] using [min, max] as range
        NB: if values in value are beyond [min, max], there could be a value <0 or >1 in result

        :param value: value to be normalized
        :return: the normalized value
        """
        if self.max > self.min:
            return (value - self.min) / (self.max - self.min)
        return value


class ParamInterpolator:
    """ A ParamInterpolator is used to modify a parameter during the training.

        A typical example is to progressively reduce the temperature to a target value.
    """

    def update(self, epoch: int):
        """ Update the parameter for current epoch """
        raise NotImplementedError()

    def val(self) -> float:
        """ Return the actual parameter value """
        raise NotImplementedError()


class ConstantInterpolator(ParamInterpolator):
    """ A constant ParamInterpolator """

    def __init__(self, value: float):
        self.value = value

    def update(self, epoch: int):
        pass

    def val(self) -> float:
        return self.value


class LinearInterpolator(ParamInterpolator):
    """ A linear ParamInterpolator """

    def __init__(self, initial_value: float, final_value: float, final_epoch: int):
        self.initial_value = initial_value
        self.final_value = final_value
        self.final_epoch = final_epoch
        self.curr_value = initial_value

    def update(self, epoch: int):
        if epoch >= self.final_epoch:
            self.curr_value = self.final_value
        else:
            ratio = epoch / self.final_epoch
            self.curr_value = self.initial_value * (1 - ratio) + self.final_value * ratio

    def val(self) -> float:
        return self.curr_value

    @staticmethod
    def from_string(s: str) -> LinearInterpolator:
        s = s.replace(" ", "").lower()
        if s.startswith("linear"):
            s = s[6:]
            if s[0] != '(' or s[-1] != ')':
                raise RuntimeError(f"Bad LinearInterpolator specification {s}. Expected '(' and ')'.")
            params = s[1:-1].split(',')
            if len(params) != 3:
                raise RuntimeError(f"Bad LinearInterpolator specification {s}. Expected exactly 3 parameters.")
            try:
                initial_v = float(params[0])
                final_v = float(params[1])
                final_e = int(params[2])
            except Exception:
                raise RuntimeError(f"Bad LinearInterpolator specification {s}. "
                                   f"Expected (float, float, int) as parameters.")
            return LinearInterpolator(initial_v, final_v, final_e)
        else:
            raise RuntimeError(f"Bad LinearInterpolator specification {s}. Expected 'linear'.")


def parse_interpolator(s: str, error_msg_prefix: str = '') -> ParamInterpolator:
    """ Parse a concrete interpolator from a string description.

        Supported values:
            int:                 a simple integer become a ConstantInterpolator
            linear(x, y, z):     a LinearInterpolator with params x, y, z
    """
    try:
        val = float(s)
        return ConstantInterpolator(val)
    except ValueError:
        s = s.replace(" ", "").lower()
        if s.startswith('linear'):
            try:
                return LinearInterpolator.from_string(s)
            except Exception as e:
                raise RuntimeError(f"{error_msg_prefix} {e}.")
        else:
            raise RuntimeError(f"{error_msg_prefix} Unknown interpolator {s}.")


def make_probability_vector(action_preferences: np.ndarray, temp: float) -> np.ndarray:
    """ Normalize the action preferences, so they become a probability vector (>=0 and sum to 1).
    Probabilities are rescaled by the temperature parameter temp, so that:
        temp == 0: the best action has probability 1 and all the rest 0
        0 < temp < 1: the best action is enhanced
        temp == 1: the action probabilities are strictly proportional to the action preferences (no effect)
        1 < temp < inf: the actions are smoothed to uniform distribution
        temp == inf: uniform distribution for actions

    NB: due to rounding errors, temperature should not be too small (>= 0.05)

    :param action_preferences: an array with non-negative action preferences
    :param temp: temperature to adjust probabilities
    :return: a probability vector
    """
    if temp == 0:
        probs = np.zeros_like(action_preferences)
        probs[np.argmax(action_preferences)] = 1
    elif temp == np.inf or action_preferences.sum() == 0:
        probs = np.full_like(action_preferences, 1 / len(action_preferences))
    else:
        probs = action_preferences ** (1 / temp)
        sum_probs = probs.sum()
        if sum_probs == 0:  # rounding errors fallback
            probs = np.zeros_like(action_preferences)
            probs[np.argmax(action_preferences)] = 1
        else:
            probs /= sum_probs

    return probs


def setup_logger(out_file: Path, to_console: bool) -> logging.Logger:
    """ Builds a new logger that outputs to the given file and optionally to the console

    :param out_file: the output file
    :param to_console: if True, log also to the console
    :return: the logger
    """
    logger = logging.getLogger(out_file.name)
    logger.setLevel(logging.INFO)
    log_formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", datefmt='%Y-%m-%d %H:%M:%S')

    file_handler = logging.FileHandler(out_file, mode='a')
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    if to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_formatter)
        logger.addHandler(console_handler)

    return logger


def get_class(class_name: str) -> type:
    """ Load a class and return a reference to it """
    parts = class_name.split('.')
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m


RenderFileInfo = namedtuple('RenderFileInfo', ['prefix', 'epoch', 'episode', 'step'])


def get_render_file_info(f: Path) -> RenderFileInfo:
    """ Return information about a render file (a file generated by the .render() call an environment)

    :param f: the file to analyze
    :return: a tuple containing the information
    """
    prefix = f.name.split('_')[0]
    parts = f.name[:-4].split('_')[1]
    epoch, in_epoch, step = parts.split('-')
    return RenderFileInfo(prefix, int(epoch), int(in_epoch), int(step))


def merge_config(cfg: Dict, eval_cfg: Dict) -> Dict:
    """ Merge default and evaluation configuration dictionaries.
        A copy of cfg is returned where values form eval_cfg will override values of cfg.

    :param cfg: the default configuration dictionary
    :param eval_cfg: the evaluation configuration dictionary
    :return: the merged dictionary
    """
    tmp = copy.deepcopy(cfg)
    for k, v in eval_cfg.items():
        if k != 'evaluation_config':
            tmp[k] = v
    return tmp


def check_config(cfg: Dict,
                 trainer: Literal[
                     'MctsSingleTrainer', 'MctsRootParallelTrainer',
                     'MctsTreeParallelTrainer', 'MctsPolicyParallelTrainer']):
    """ Check the validity of a configuration dictionary for a given trainer.

        If an error is encountered in the configuration an exception is raised.

    :param cfg: the configuration dictionary
    :param trainer: the trainer class name
    """

    def checked(d, p):
        try:
            return d[p]
        except KeyError:
            raise RuntimeError(f"check_config() Missing required parameter '{p}'.")

    # ---------- Environment ----------
    param = 'env_class'
    try:
        _ = get_class(checked(cfg, param))
    except Exception as e:
        raise RuntimeError(f"check_config() Wrong parameter '{param}'={cfg[param]}. Expected modulename.classname. {e}")

    param = 'env_config'
    if not isinstance(checked(cfg, param), dict):
        raise RuntimeError(f"check_config() Wrong parameter '{param}'. "
                           f"Expected a dictionary. Use an empty dictionary {{}} for no parameters.")

    param = 'env_worker_id_param'
    if not isinstance(checked(cfg, param), bool):
        raise RuntimeError(f"check_config() Wrong parameter '{param}'. Expected a boolean.")

    param = 'env_reset_seed'
    if checked(cfg, param) is not None and not isinstance(checked(cfg, param), int):
        raise RuntimeError(f"check_config() Wrong parameter '{param}'. Expected an integer or None.")

    param = 'env_restore_mode'
    _ = checked(cfg, param)

    param = 'policy_class'
    _ = checked(cfg, param)

    param = 'policy_config'
    _ = checked(cfg, param)

    param = 'replay_buffer_class'
    _ = checked(cfg, param)

    param = 'replay_buffer_config'
    _ = checked(cfg, param)

    param = 'replay_buffer_min_size'
    _ = checked(cfg, param)

    param = 'discount_factor'
    _ = checked(cfg, param)

    param = 'rollout_start_node'
    _ = checked(cfg, param)

    param = 'uct_policy_c'
    _ = checked(cfg, param)

    param = 'uct_exploration_c'
    _ = checked(cfg, param)

    param = 'step_temperature'
    if not (isinstance(checked(cfg, param), str) or isinstance(checked(cfg, param), int) or isinstance(
            checked(cfg, param), float)):
        raise RuntimeError(f"check_config() Wrong parameter '{param}'. Expected a float or an interpolator "
                           f"specification (ex. linear(1.0, 0.1, 1000)).")

    param = 'train_temperature'
    if not (isinstance(checked(cfg, param), str) or isinstance(checked(cfg, param), int) or isinstance(
            checked(cfg, param), float)):
        raise RuntimeError(f"check_config() Wrong parameter '{param}'. Expected a float or an interpolator "
                           f"specification (ex. linear(1.0, 0.1, 1000)).")

    param = 'num_expansions'
    _ = checked(cfg, param)

    param = 'max_rollout_steps'
    _ = checked(cfg, param)

    param = 'random_rollout'
    _ = checked(cfg, param)

    param = 'step_criterion'
    _ = checked(cfg, param)

    param = 'value_estimation'
    _ = checked(cfg, param)

    param = 'add_exploration_noise'
    _ = checked(cfg, param)

    param = 'exploration_noise_dirichlet_alpha'
    _ = checked(cfg, param)

    param = 'exploration_noise_fraction'
    _ = checked(cfg, param)

    param = 'reset_minmax_each_episode'
    _ = checked(cfg, param)

    param = 'training_epochs'
    _ = checked(cfg, param)

    if trainer != 'MctsPolicyParallelTrainer':
        param = 'episodes_per_epoch'
        _ = checked(cfg, param)

    param = 'max_episode_steps'
    _ = checked(cfg, param)

    param = 'random_seed'
    _ = checked(cfg, param)

    param = 'train_batch_size'
    _ = checked(cfg, param)

    param = 'random_batch'
    _ = checked(cfg, param)

    param = 'learning_rate'
    _ = checked(cfg, param)

    param = 'vf_loss_coeff'
    _ = checked(cfg, param)

    param = 'num_sgd_per_epoch'
    _ = checked(cfg, param)

    param = 'out_base_dir'
    _ = checked(cfg, param)

    param = 'out_experiment_dir'
    _ = checked(cfg, param)

    param = 'max_retry_on_error'
    _ = checked(cfg, param)

    param = 'checkpoint_freq'
    _ = checked(cfg, param)

    param = 'verbose'
    _ = checked(cfg, param)

    if trainer != 'MctsPolicyParallelTrainer':
        param = 'evaluation_interval'
        _ = checked(cfg, param)

    if trainer != 'MctsPolicyParallelTrainer':
        param = 'evaluation_duration'
        _ = checked(cfg, param)

    param = 'evaluation_config'
    _ = checked(cfg, param)

    param = 'custom'
    _ = checked(cfg, param)

    if trainer in ['MctsTreeParallelTrainer', 'MctsRootParallelTrainer', 'MctsPolicyParallelTrainer']:
        param = 'num_rollout_workers'
        _ = checked(cfg, param)

    if trainer == 'MctsRootParallelTrainer':
        param = 'reset_tree_each_step'
        _ = checked(cfg, param)

    if trainer == 'MctsPolicyParallelTrainer':
        param = 'logging_interval_secs'
        _ = checked(cfg, param)

        param = 'logging_aggregation_epochs'
        _ = checked(cfg, param)

        param = 'policy_train_max_rate'
        _ = checked(cfg, param)

        param = 'render_interval'
        _ = checked(cfg, param)

        param = 'render_max_per_interval'
        _ = checked(cfg, param)

        param = 'workers_logging'
        _ = checked(cfg, param)

    common_keys = [
        'env_class', 'env_config', 'env_worker_id_param', 'env_reset_seed', 'max_retry_on_error',
        'policy_class', 'policy_config', 'replay_buffer_class', 'replay_buffer_config',
        'replay_buffer_min_size', 'training_epochs', 'episodes_per_epoch', 'max_episode_steps',
        'random_seed', 'train_batch_size', 'random_batch', 'learning_rate', 'vf_loss_coeff',
        'num_sgd_per_epoch', 'rollout_start_node', 'uct_policy_c', 'uct_exploration_c', 'discount_factor',
        'num_expansions', 'max_rollout_steps', 'random_rollout', 'step_criterion', 'value_estimation',
        'step_temperature', 'train_temperature', 'add_exploration_noise', 'exploration_noise_dirichlet_alpha',
        'exploration_noise_fraction', 'env_restore_mode', 'out_base_dir', 'out_experiment_dir', 'verbose',
        'custom', 'checkpoint_freq', 'evaluation_interval', 'evaluation_duration', 'evaluation_config',
        'reset_minmax_each_episode']
    root_keys_add = ['num_rollout_workers', 'reset_tree_each_step']
    tree_keys_add = ['num_rollout_workers']
    policy_keys_add = ['num_rollout_workers', 'logging_interval_secs', 'logging_aggregation_epochs',
                       'policy_train_max_rate', 'render_interval', 'render_max_per_interval', 'workers_logging']
    policy_keys_del = ['episodes_per_epoch', 'evaluation_interval', 'evaluation_duration']

    if trainer == 'MctsRootParallelTrainer':
        common_keys += root_keys_add
    elif trainer == 'MctsTreeParallelTrainer':
        common_keys += tree_keys_add
    elif trainer == 'MctsPolicyParallelTrainer':
        common_keys += policy_keys_add
        common_keys = [item for item in common_keys if item not in policy_keys_del]

    for k in cfg:
        if k not in common_keys:
            raise RuntimeError(
                f"check_config() Unexpected parameter '{k}'. Please, put extra parameter inside the 'custom' key.")
