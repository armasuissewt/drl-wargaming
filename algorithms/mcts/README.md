# MCTS

Implementation of the Monte Carlo Tree Search (MCTS) algorithm for the 
experiments in the paper:

Giacomo, Del Rio and Oleg, Szehr and Alessandro, Antonucci and Matthias, Sommer and Michael, Ruëgsegger (2024) **"Deep Reinforcement Learning in Wargaming"**

## Environments

This MCTS algorithm that can work with
any [Gymnasium](https://gymnasium.farama.org/index.html) environment that satisfies the
following two conditions:

1. the actions are discrete (`gymnasium.spaces.Discrete`)
2. the internal state of the environment can be saved/restored

The ability of saving and restoring the internal state of the environment can
be obtained in two ways:

- The environment is simple enough that a `copy.deepcopy` can do the job. In this
  case the environment can be used as-is and the MCTS algorithm will take care
  of making copies when needed.
- The environment itself implements the `save()` and `restore()` methods. This is
  usually needed when the environment is not pure Python code and needs extra
  processing to properly save and restore states.

## Usage

For usage examples see the [solvers](https://github.com/armasuissewt/drl-wargaming/tree/master/solvers) folder of the repository.

## Citation

Please cite this code as specified in the [README.md](https://github.com/armasuissewt/drl-wargaming/blob/master/README.md) file in the root of this repository.

## License

MIT License

Copyright (c) 2024 IDSIA

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.