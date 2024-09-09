# DRL for Wargaming

Code for the paper **"Deep Reinforcement Learning in Wargaming"**.

## Environment

The code has been tested in Windows with Python version `3.11`. The required packages are listed in `requirements.txt`.

## Running the examples

The learning scripts are located into the `solvers` folder. Before running the `cmo` experiments, that requires the Professional Edition of "Command: modern operations" to be installed, you have to checks the paths in:
 - `cmo_instance.py` (`CMO_EXECUTABLE` variable)
 - `destpat_env_cmo.py` (`scenario_file` and `lua_helpers_file` variables)
 - `flytozurich_env_cmo.py` (`scenario_file` and `lua_helpers_file` variables) 

to reflect the ones in your installation.

To run the `solvers` scripts, the following folders **must** be in you `PYTHONPATH` variable:
- `algorithms/mcts`
- `cmoclient`
- `warsim`

## Citation

Please use this bibtex if you want to cite the paper in your publications:

```text
@article{drlwargaming2024,
  author       = {Giacomo, Del Rio and Oleg, Szehr and Alessandro, Antonucci and Matthias, Sommer and Michael, Ru{\"e}gsegger},
  title        = {Deep Reinforcement Learning in Wargaming},
  year         = {2024},
  publisher    = {},
  journal      = {}
  volume       = {}
  number       = {}
  pages        = {}
  month        = {}
  doi          = {}
}
```

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