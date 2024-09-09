# mcts_policy_torch.py
#
# pyTorch based policy class for the MCTS agent
#
# Author: Giacomo Del Rio
# Creation date: 21 Oct 2022

import copy
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from mcts_common.mcts_policy_base import MctsPolicy, ObsType, MctsReplayBuffer


class TorchFFNet(nn.Module):
    def __init__(self, input_shape: Tuple, n_actions: int, dense_layers: List[int], shared_layers: int):
        """ A feedforward neural network with two heads: one for action probabilities and the other for value.
            The heads may or may not share initial layers according to shared_layers parameter.

        :param input_shape: input shape
        :param n_actions: number of actions
        :param dense_layers: list of sizes for layers
        :param shared_layers: number of initial dense layers shared between policy and value heads.
            (0 <= shared_layers <= len(dense_layers))
        """
        super().__init__()

        # Check parameters
        if len(input_shape) > 1:
            raise ValueError(
                f"Feedforward neural network can work only with flattened observations. Got {input_shape}.")

        if len(dense_layers) < 1 or len(dense_layers) > 3:
            raise ValueError(f"Feedforward NN must have be between 1 and 3 layers. Got {dense_layers}.")

        if shared_layers < 0 or shared_layers > len(dense_layers):
            raise ValueError(f"shared_layers must be >= 0 and <= len(dense_layers). Got {shared_layers}.")

        # Build fully connected layers
        in_size = input_shape[0]
        d_layers_shared = []
        for layer_size in dense_layers:
            d_layers_shared.extend([nn.Linear(in_size, layer_size), nn.ReLU()])
            in_size = layer_size
        d_layers_p = d_layers_shared[shared_layers * 2:]
        d_layers_v = copy.deepcopy(d_layers_p)
        d_layers_shared = d_layers_shared[:shared_layers * 2]
        self.dense_net_shared = nn.Sequential(*d_layers_shared) if len(d_layers_shared) > 0 else None
        self.dense_net_p = nn.Sequential(*d_layers_p) if len(d_layers_p) > 0 else None
        self.dense_net_v = nn.Sequential(*d_layers_v) if len(d_layers_v) > 0 else None

        # Build output layers
        self.actions_output = nn.Linear(dense_layers[-1], n_actions)
        self.value_output = nn.Linear(dense_layers[-1], 1)

    def forward(self, x: torch.Tensor, avoid_softmax=False) -> Tuple[torch.Tensor, torch.Tensor]:
        shared_out = self.dense_net_shared(x) if self.dense_net_shared is not None else x
        p_out = self.dense_net_p(shared_out) if self.dense_net_p is not None else shared_out
        v_out = self.dense_net_v(shared_out) if self.dense_net_v is not None else shared_out
        if avoid_softmax:
            return self.actions_output(p_out), self.value_output(v_out)
        else:
            return torch.softmax(self.actions_output(p_out), dim=-1), self.value_output(v_out)

    def forward_p(self, x: torch.Tensor) -> torch.Tensor:
        shared_out = self.dense_net_shared(x) if self.dense_net_shared is not None else x
        p_out = self.dense_net_p(shared_out) if self.dense_net_p is not None else shared_out
        return torch.softmax(self.actions_output(p_out), dim=-1)

    def forward_v(self, x: torch.Tensor) -> torch.Tensor:
        shared_out = self.dense_net_shared(x) if self.dense_net_shared is not None else x
        v_out = self.dense_net_v(shared_out) if self.dense_net_v is not None else shared_out
        return self.value_output(v_out)


class TorchConvNet(nn.Module):
    def __init__(self, input_shape: Tuple, n_actions: int, conv_layers: List[str], dense_layers: List[int],
                 shared_layers: int):
        """ Provides a convolutional neural network with two heads: one for action probabilities and the other for value

        :param input_shape: input shape
        :param n_actions: number of actions
        :param conv_layers: a description of the convolutional layers of the network
        :param dense_layers: size of the dense layers in top of the convolutional ones
        :param shared_layers: number of dense layers shared between policy and value heads.
            (0 <= shared_layers <= len(dense_layers)). The convolutional layers are always shared.
        """
        super().__init__()

        # Check parameters
        if len(dense_layers) < 1 or len(dense_layers) > 3:
            raise ValueError(f"There must be 1 or 3 dense layers. Got {dense_layers}.")

        if len(input_shape) < 2 or len(input_shape) > 3:
            raise ValueError(f"Convolutional neural network can work only with "
                             f"(n_channels, x, y) or (x, y) observations. Got {input_shape}.")
        conv_in_size = 1 if len(input_shape) == 2 else input_shape[0]

        if shared_layers < 0 or shared_layers > len(dense_layers):
            raise ValueError(f"shared_layers must be >= 0 and <= len(dense_layers). Got {shared_layers}.")

        # Build convolutional layers
        c_layers = []
        in_channels = conv_in_size
        for lay in conv_layers:
            layer_name, layer_args = TorchLayerParser(lay).parse_layer()
            if layer_name == 'Conv2d':
                c_layers.append(nn.Conv2d(in_channels=in_channels, out_channels=int(layer_args[0]),
                                          kernel_size=layer_args[1], stride=layer_args[2], padding=layer_args[3]))
                in_channels = int(layer_args[0])
            elif layer_name == 'MaxPool2d':
                c_layers.append(nn.MaxPool2d(kernel_size=layer_args[0], stride=layer_args[1], padding=layer_args[2]))
            elif layer_name == 'ReLU':
                c_layers.append(nn.ReLU())
            elif layer_name == 'Tanh':
                c_layers.append(nn.Tanh())
        c_layers.append(nn.Flatten())
        self.conv_net = nn.Sequential(*c_layers)

        # Compute the input size of fully connected layers
        rnd_obs = torch.randn(size=((1, 1) if len(input_shape) == 2 else (1,)) + input_shape)
        dense_in_size = self.conv_net(rnd_obs).shape[1]

        # Build fully connected layers
        d_layers_shared = []
        for size in dense_layers:
            d_layers_shared.extend([nn.Linear(dense_in_size, size), nn.ReLU()])
            dense_in_size = size
        d_layers_p = d_layers_shared[shared_layers * 2:]
        d_layers_v = copy.deepcopy(d_layers_p)
        d_layers_shared = d_layers_shared[:shared_layers * 2]
        self.dense_net_shared = nn.Sequential(*d_layers_shared) if len(d_layers_shared) > 0 else None
        self.dense_net_p = nn.Sequential(*d_layers_p) if len(d_layers_p) > 0 else None
        self.dense_net_v = nn.Sequential(*d_layers_v) if len(d_layers_v) > 0 else None

        # Build output layers
        self.actions_output = nn.Linear(dense_layers[-1], n_actions)
        self.value_output = nn.Linear(dense_layers[-1], 1)

    def forward_p(self, x: torch.Tensor) -> torch.Tensor:
        conv_out = self.conv_net(x)
        shared_out = self.dense_net_shared(conv_out) if self.dense_net_shared is not None else conv_out
        p_out = self.dense_net_p(shared_out) if self.dense_net_p is not None else shared_out
        return torch.softmax(self.actions_output(p_out), dim=-1)

    def forward_v(self, x: torch.Tensor) -> torch.Tensor:
        conv_out = self.conv_net(x)
        shared_out = self.dense_net_shared(conv_out) if self.dense_net_shared is not None else conv_out
        v_out = self.dense_net_v(shared_out) if self.dense_net_v is not None else shared_out
        return self.value_output(v_out)

    def forward(self, x: torch.Tensor, avoid_softmax=False) -> Tuple[torch.Tensor, torch.Tensor]:
        conv_out = self.conv_net(x)
        shared_out = self.dense_net_shared(conv_out) if self.dense_net_shared is not None else conv_out
        p_out = self.dense_net_p(shared_out) if self.dense_net_p is not None else shared_out
        v_out = self.dense_net_v(shared_out) if self.dense_net_v is not None else shared_out
        if avoid_softmax:
            return self.actions_output(p_out), self.value_output(v_out)
        else:
            return torch.softmax(self.actions_output(p_out), dim=-1), self.value_output(v_out)


class TorchMctsPolicy(MctsPolicy[ObsType]):
    """ Torch based policy neural network
    """

    def __init__(self, learning_rate: float, obs_shape: Tuple, n_actions: int, vf_loss_coeff: float, net_type: str,
                 dense_layers: List[int], shared_layers: int, conv_layers: Optional[List[str]]):
        """ Create a new policy

        :param learning_rate: the learning rate to train the network
        :param obs_shape: shape of observations from the environment
        :param n_actions: number of actions
        :param vf_loss_coeff: coefficient to scale the value loss
        :param net_type: network type. Supported types are: "feedforward", "convolutional".
        :param dense_layers: number of dense layers for the "feedforward" network type. Min 1, max 3. In case of
            "convolutional" network type, these layers will be appended to the convolutional ones.
        :param shared_layers: number of dense layers shared between policy and value heads.
            (0 <= shared_layers <= len(dense_layers)). The convolutional layers are always shared.
        :param conv_layers: a description of the convolutional layers of the network
        """
        super().__init__(learning_rate, obs_shape, n_actions, vf_loss_coeff)

        self.shared_layers = shared_layers
        if net_type == 'feedforward':
            self.net = TorchFFNet(obs_shape, n_actions, dense_layers, shared_layers)
        elif net_type == 'convolutional':
            self.net = TorchConvNet(obs_shape, n_actions, conv_layers, dense_layers, shared_layers)
        else:
            raise ValueError(f"Unknown network type {net_type}")

        self.loss_reward = nn.MSELoss()
        self.loss_actions = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        self.retain_graph_during_learn = self.shared_layers >= 0 or net_type == 'convolutional'

    def train(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Tuple[float, float]:
        ps_hat, vs_hat = self.net.forward(batch[0], avoid_softmax=True)
        loss_p = self.loss_actions(ps_hat, batch[1])
        loss_v = self.loss_reward(vs_hat, batch[2].unsqueeze(1)) * self.vf_loss_coeff

        self.optimizer.zero_grad()
        loss_p.backward(retain_graph=self.retain_graph_during_learn)
        loss_v.backward()
        self.optimizer.step()
        return loss_p.item(), loss_v.item()

    def predict(self, state: ObsType) -> Tuple[np.ndarray, float]:
        with torch.no_grad():
            pi, v = self.net(torch.tensor(state, dtype=torch.float)[None, :])
            return pi.numpy()[0], v.item()

    def predict_p(self, state: ObsType) -> np.ndarray:
        with torch.no_grad():
            return self.net.forward_p(torch.tensor(state, dtype=torch.float)[None, :]).numpy()[0]

    def predict_v(self, state: ObsType) -> float:
        with torch.no_grad():
            return self.net.forward_v(torch.tensor(state, dtype=torch.float)[None, :]).item()

    def get_weights(self) -> Dict:
        return self.net.state_dict()

    def set_weights(self, w: Dict) -> None:
        self.net.load_state_dict(w)

    def load_weights(self, file: Path | str) -> None:
        self.net.load_state_dict(torch.load(file))

    def save_weights(self, file: Path) -> None:
        torch.save(self.net.state_dict(), file.with_suffix('.pt'))


class TorchMctsReplayBuffer(MctsReplayBuffer[ObsType]):
    """ pyTorch based Monte Carlo tree search replay buffer """

    def __init__(self, obs_shape: Tuple, n_actions: int, max_size: int):
        super().__init__(obs_shape, n_actions, max_size)
        self.head = 0
        self.states = torch.empty(size=(max_size,) + obs_shape, dtype=torch.float)
        self.probs = torch.empty(size=(max_size, n_actions), dtype=torch.float)
        self.vals = torch.empty(size=(max_size,), dtype=torch.float)

    def push(self, observation: ObsType, action_probs: np.ndarray, value: float) -> None:
        self.states[self.head] = torch.from_numpy(observation)
        self.probs[self.head] = torch.from_numpy(action_probs)
        self.vals[self.head] = value

        self.head = (self.head + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, n_samples: int, random_sample: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n = min(n_samples, self.size)
        if random_sample:
            idx = np.random.choice(self.size, size=n, replace=False)
            return self.states[idx], self.probs[idx], self.vals[idx]
        else:
            return self.states[-n:], self.probs[-n:], self.vals[-n:]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.states[index], self.probs[index], self.vals[index]


class TorchLayerParser:
    """ Utility class to parse pytorch layers definitions. Supported layers:
        - Conv2d(10, (2, 2), (2, 2), (0, 0)): num_filters, kernel_size, stride, padding
        - MaxPool2d((2, 2), (2, 2), (0, 0)): kernel_size, stride, padding
        - ReLU()
        - Tanh()
    """

    def __init__(self, s: str):
        self.s = s
        self.tk = self.Tokenizer(s)

    def parse_layer(self) -> Tuple[str, Tuple]:
        layer_name = self.tk.curr_token
        if layer_name in ['(', ')', ','] or layer_name is None:
            raise ValueError(f"Expected layer name at beginning of string. {self.s}")

        if layer_name not in ['Conv2d', 'ReLU', 'MaxPool2d', 'Tanh']:
            raise ValueError(f"unknown layer name '{layer_name}'.")
        self.tk.next()

        layer_args = self.parse_tuple()
        self._match(None)
        return layer_name, layer_args

    def parse_tuple(self) -> Tuple:
        self._match('(')
        args = self.parse_arg_list([])
        self._match(')')
        return tuple(args)

    def parse_arg_list(self, already_parsed: List) -> List:
        if self.tk.curr_token == ')':
            return already_parsed
        elif self.tk.curr_token == '(':
            already_parsed += [self.parse_tuple()]
        elif self.tk.curr_token == ',':
            self.tk.next()
        elif self.tk.curr_token is None:
            raise ValueError(f"Unexpected end of parameter list. {self.s}")
        else:
            already_parsed += [self._to_number(self.tk.curr_token)]
            self.tk.next()
        return self.parse_arg_list(already_parsed)

    @staticmethod
    def _to_number(i: str) -> Union[int, str]:
        try:
            return int(i)
        except ValueError:
            return i

    def _match(self, expected_token: Optional[str]):
        if expected_token is None and self.tk.curr_token is not None:
            raise ValueError(f"Expected End of string. {self.s}")
        elif self.tk.curr_token != expected_token:
            raise ValueError(f"Expected '{expected_token}'. {self.s}")
        self.tk.next()

    class Tokenizer:
        def __init__(self, s: str):
            self.s: str = s
            self.curr_token: Optional[str] = self.next()

        def next(self) -> Optional[str]:
            next_token = ""
            i = 0
            while i < len(self.s) and self.s[i] not in ['(', ')', ',']:
                if not self.s[i].isspace():
                    next_token += self.s[i]
                i += 1

            if i == 0 or len(next_token) == 0:
                if len(self.s) == 0:
                    self.curr_token = None
                else:
                    self.curr_token = self.s[i] if i < len(self.s) else None
                    i += 1
            else:
                self.curr_token = next_token

            self.s = self.s[i:]
            return self.curr_token
