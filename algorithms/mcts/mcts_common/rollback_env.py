# rollback_env.py
#
# The RollbackEnv class
#
# Author: Giacomo Del Rio
# Creation date: 7 May 2023

import copy
import inspect
import logging
import sys
from typing import Any, Tuple, Dict, Optional

import gymnasium as gym

from mcts_common.mcts_utils import get_class


class RollbackEnv:
    """ A RollbackEnv is gymnasium environment container that allows to save the internal state of an environment
        in a checkpoint and then to restore the checkpoint when needed.

        Two save/restore mechanisms are supported:
            - the use of deepcopy()
                With the use of deepcopy(), a deep copy of the environment is made for a checkpoint.
                Most pure Python environments allows this method.

            - the save() and restore() method.
                For this mechanism the gym environment must support two additional methods: save() and restore().
                It is needed when the state of the environment may reside outside the Python managed code
                (i.e. an external simulator)
                Signatures of save and restore methods must be:
                    save() -> Any
                    restore(saved_obj: Any) -> None
                You can use the standalone check_env_save_restore() function to check the env for the above two methods.

        Note: A RollbackEnv is NOT an environment wrapper.
    """

    def __init__(self, env_class: str, env_config: Dict, env_restore_mode: str, max_retry_on_error: int,
                 worker_id: Optional[int], logger: Optional[logging.Logger] = None):
        """ Build a new RollbackEnv that contains a new instance of the requested gymnasium environment.

        :param env_class: the full name of the environment class
        :param env_config: a dictionary with the parameters for building the environment
        :param env_restore_mode: Can be 'deepcopy' or 'save_restore'.
        :param max_retry_on_error: number of attempts in reset() and step() in case of exception
        :param worker_id: if not None, pass an additional 'worker_id' parameter to the environment's constructor
        :param logger: an optional logger to log errors instead of printing on stderr
        """
        self.env_class = get_class(env_class)
        self.env_config = {**env_config, **({'worker_id': worker_id} if worker_id else {})}
        self.env_restore_mode = env_restore_mode
        self.max_retry_on_error = max_retry_on_error
        self.logger = logger
        self.env: Optional[gym.Env] = self._build()
        self.checkpoint = None
        self.aligned_to_checkpoint = False

    def _log_err(self, message):
        if self.logger is not None:
            self.logger.error(message)
        else:
            print(message, file=sys.stderr)

    def _build(self) -> gym.Env:
        env: gym.Env = self.env_class(**self.env_config)
        if not isinstance(env.action_space, gym.spaces.Discrete):
            err_msg = f"MCTS agent supports only gym.spaces.Discrete action spaces. Got: {env.action_space}"
            self._log_err(err_msg)
            raise RuntimeError(err_msg)
        if self.env_restore_mode == 'save_restore':
            check_env_save_restore(env)
        self.aligned_to_checkpoint = False
        return env

    def set_logger(self, logger: logging.Logger):
        """ Replace the logger """
        self.logger = logger

    def reset(self, seed: Optional[int] = None) -> Tuple[Any, dict]:
        """ Forward the reset() to the environment """
        self.aligned_to_checkpoint = False
        return self.env.reset(seed=seed)

    def reset_safe(self, seed: Optional[int] = None) -> Tuple[Any, dict]:
        """ Execute the reset(seed) method of the contained environment.
            In the case the reset(seed) method raises an exception, it rebuilds the environment and tries again up
            to max_retry_on_error times.

        :return: the result of reset(seed)
        """
        attempt = 0
        last_error = None
        while attempt < self.max_retry_on_error:
            try:
                self.aligned_to_checkpoint = False
                return self.env.reset(seed=seed)
            except Exception as e:
                self._log_err(f"RollbackEnv: reset() error. {e}")
                last_error = e
                attempt += 1
                self.env.close()
                del self.env
                self.env = self._build()
        self._log_err(f"the reset() failed with {last_error}")
        raise RuntimeError(f"the reset() failed with {last_error}")

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, dict]:
        """ Forward the step() to the environment """
        self.aligned_to_checkpoint = False
        return self.env.step(action)

    def step_safe(self, action: Any) -> Tuple[Any, float, bool, bool, dict]:
        """ Execute the step(action) method of the contained environment.
            In the case the step(action) method raises an exception, it recreates the
            environment, restores the last checkpoint and try again up to max_retry_on_error times.

        :param action: the action to be executed in step()
        :return: the result of step(action)
        """
        attempt = 0
        last_error = None
        while attempt < self.max_retry_on_error:
            try:
                self.aligned_to_checkpoint = False
                return self.env.step(action)
            except Exception as e:
                self._log_err(f"RollbackEnv: step() error. {e}")
                last_error = e
                attempt += 1
                self.restore_checkpoint(rebuild_env=True)
        self._log_err(f"the step() failed with {last_error}")
        raise RuntimeError(f"the step() failed with {last_error}")

    def render(self) -> Any:
        """ Forward the render() to the environment """
        return self.env.render()

    def save_checkpoint(self) -> None:
        """ Save and store a checkpoint of the environment """
        if not self.aligned_to_checkpoint:
            self.checkpoint = self.get_env_state()
            self.aligned_to_checkpoint = True

    def set_checkpoint(self, state: Any) -> None:
        """ Set a new checkpoint for the environment """
        self.checkpoint = state
        self.aligned_to_checkpoint = False

    def restore_checkpoint(self, rebuild_env: bool) -> None:
        """ Restore a previously saved checkpoint of the environment """
        if not self.aligned_to_checkpoint:
            self.set_env_state(self.checkpoint, rebuild_env)
            self.aligned_to_checkpoint = True

    def get_env_state(self) -> Any:
        """ Get a copy of the internal environment's state """
        if self.env_restore_mode == 'deepcopy':
            return copy.deepcopy(self.env)
        else:
            return self.env.save()  # noqa

    def set_env_state(self, state: Any, rebuild_env: bool = False) -> None:
        """ Restore a previously saved copy of the environment's state """
        if self.env_restore_mode == 'deepcopy':
            self.env = copy.deepcopy(state)
        else:
            if rebuild_env:
                self.rebuild_env()
            self.env.restore(state)  # noqa
        self.aligned_to_checkpoint = False

    def set_env_state_safe(self, state: Any, rebuild_env: bool = False) -> None:
        """ Call set_env_state() up to max_retry_on_error times. Rebuild environment on each failure """
        attempt = 0
        last_error = None
        while attempt < self.max_retry_on_error:
            try:
                self.set_env_state(state, rebuild_env)
                return
            except Exception as e:
                self._log_err(f"RollbackEnv: set_env_state_safe() error. {e}")
                last_error = e
                attempt += 1
                self.rebuild_env()
        self._log_err(f"the set_env_state_safe() failed with {last_error}")
        raise RuntimeError(f"the set_env_state_safe() failed with {last_error}")

    def rebuild_env(self) -> None:
        """ Close and recreate the environment object """
        self.env.close()
        del self.env
        self.env = self._build()
        self.aligned_to_checkpoint = False

    def get_n_actions(self) -> int:
        """ Get the number of available actions """
        if self.env is not None:
            return self.env.action_space.n  # noqa
        else:
            raise RuntimeError(f"Environment not yet built!")


def check_env_save_restore(env: gym.Env) -> None:
    """ Check if env has the required save() and restore() method

    :param env: a gymnasium environment
    :return: if the environment has the wrong save() and restore() methods, raise an exception, otherwise return None
    """
    save_method = getattr(env, 'save', None)
    if save_method is not None:
        inspect.getfullargspec(save_method)
    else:
        raise ValueError(f"Environment doesn't have a save() method")

    restore_method = getattr(env, 'restore', None)
    if restore_method is not None:
        args = inspect.getfullargspec(restore_method)
        if len(args.args) != 2:
            raise ValueError("Method restore() of the environment must accept only 1 positional parameter")
    else:
        raise ValueError("Environment doesn't have a restore_method() method")
