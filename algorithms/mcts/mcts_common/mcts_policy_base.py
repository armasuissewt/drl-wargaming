# mcts_policy_base.py
#
# Base policy class for an MCTS agent
#
# Author: Giacomo Del Rio
# Creation date: 12 apr 2023

from pathlib import Path
from typing import Generic, Tuple, Any

import numpy as np

from mcts_common.mcts_utils import ObsType


class MctsPolicy(Generic[ObsType]):
    """ The Monte Carlo tree search policy class.
        It encapsulates a neural network to provide uniform access to prediction and training
        Naming conventions:
            p: vector of probabilities associated with each action
            v: expected return from a given state
    """

    def __init__(self, learning_rate: float, obs_shape: Tuple, n_actions: int, vf_loss_coeff: float):
        """
        :param learning_rate: the learning rate to train the network
        :param obs_shape: shape of observations from the environment
        :param n_actions: number of actions
        :param vf_loss_coeff: coefficient to scale the value loss
        """
        self.learning_rate = learning_rate
        self.obs_shape = obs_shape
        self.n_actions = n_actions
        self.vf_loss_coeff = vf_loss_coeff

    def train(self, batch: Any) -> Tuple[float, float]:
        """ Train the network for one epoch on the given batch

        :param batch: the batch of samples for training
        :return: the p and v loss
        """
        raise NotImplementedError()

    def predict(self, state: ObsType) -> Tuple[np.ndarray, float]:
        """ Predict the action probabilities and return for given state

        :param state: environment observation
        :return: the action probabilities (p) and value (v)
        """
        raise NotImplementedError()

    def predict_p(self, state: ObsType) -> np.ndarray:
        """ Predict the action probabilities for given state

        :param state: environment observation
        :return: the action probabilities (p)
        """
        return self.predict(state)[0]

    def predict_v(self, state: ObsType) -> float:
        """ Predict the return for given state

        :param state: environment observation
        :return: the return (v)
        """
        return self.predict(state)[1]

    def get_weights(self) -> Any:
        """ Get the weights of the network

        :return: the network weights
        """
        raise NotImplementedError()

    def set_weights(self, w: Any) -> None:
        """ Set the weights of the network

        :param w: the network weights
        """
        raise NotImplementedError()

    def load_weights(self, file: Path | str) -> None:
        """ Load the weights of the network from a file

        :param file: file to load the weights
        """
        raise NotImplementedError()

    def save_weights(self, file: Path) -> None:
        """ Save the weights of the network to a file

        :param file: file to write the weights. It's fine to add a suffix to the file name
        """
        raise NotImplementedError()


class MctsReplayBuffer(Generic[ObsType]):
    """ The Monte Carlo tree search replay buffer """

    def __init__(self, obs_shape: Tuple, n_actions: int, max_size: int):
        self.obs_shape = obs_shape
        self.n_actions = n_actions
        self.max_size = max_size
        self.size = 0

    def push(self, observation: ObsType, action_probs: np.ndarray, value: float) -> None:
        """ Push one sample in the replay buffer.
            If the buffer is full, the oldest element is replaced

        :param observation: the observation from the environment
        :param action_probs: the actions probability vector
        :param value: the state value
        """
        raise NotImplementedError()

    def sample(self, n_samples: int, random_sample: bool) -> Any:
        """ Returns a random sample of observations of length n_samples
            If there are fewer samples in the buffer than n_samples, returns the entire buffer

        :param n_samples: size of the sample
        :param random_sample: if true take a random sample, if false, take the last n_samples samples
        :return: a sample of the replay buffer
        """
        raise NotImplementedError()

    def __len__(self) -> int:
        """ Returns the actual size of the replay buffer

        :return: actual size of the replay buffer
        """
        return self.size

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """ Returns a sample at given index
            Output can have any type, not necessarily np.ndarray

        :param index: sample index in the dataset
        :return: a tuple with three elements:
          - the observation
          - the actions probability vector
          - the state value
        """
        raise NotImplementedError()
