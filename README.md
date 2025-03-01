<p align="center">
  <img src="docs/source/mars-logo.svg" alt="marsopt logo" width="250">
</p>


# marsopt
**Mixed Adaptive Random Search for Optimization**

[![PyPI version](https://img.shields.io/pypi/v/marsopt.svg)](https://pypi.org/project/marsopt/)
[![License](https://img.shields.io/github/license/sibirbil/marsopt.svg)](LICENSE)
[![Python Versions](https://img.shields.io/pypi/pyversions/marsopt.svg)](https://pypi.org/project/marsopt/)

`marsopt` is a Python library designed to simplify and accelerate hyperparameter and black-box optimization. It supports **continuous**, **integer**, and **categorical** variables. Based on **Mixed Adaptive Random Search** (MARS) algorithm, `marsopt` dynamically balances exploration and exploitation through:
- **adaptive noise** for sampling,
- **elite selection** to guide the search toward promising regions,
- flexible handling of **log-scale** and **categorical** parameters,
- minimization **or** maximization of any user-defined objective.

MARS iteratively refines a population of “elite” solutions and generates new candidates by **perturbing** those elites with gradually decreasing noise. For a more detailed explanation of MARS, see our **[algorithm overview](https://marsopt.readthedocs.io/en/latest/algorithm.html)** in the documentation.

## Features
- **Mixed Variable Support**: Optimize integer, float (with optional log-scale), and categorical variables in the same study.  
- **Adaptive Sampling**: Early iterations explore widely, while later iterations exploit the best regions found, thanks to a built-in **cosine annealing** scheme for noise.  
- **Easy Setup**: Simply define an objective function, specify a variable search space, and run `study.optimize()`.  
- **Resume & Extend**: Continue a `Study` with more trials at any time without losing past information.  
- **Rich Tracking**: Inspect all trial details (objective values, parameters, times, and so on) for deeper analysis.  

## Installation
Install `marsopt` from PyPI:

```bash
pip install marsopt
```

## Getting Started

Below is a simplified example of how to use `marsopt` to tune a few common hyperparameters:

```python
from marsopt import Study, Trial
import numpy as np

def objective(trial: Trial) -> float:
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
    layers = trial.suggest_int("num_layers", 1, 5)
    optimizer = trial.suggest_categorical("optimizer", ["adam", "sgd", "rmsprop"])

    score = -5 * (np.log10(lr) + 3) ** 2  
    score += np.log1p(layers) * 10  
    score += {"adam": 15, "sgd": 5, "rmsprop": 20}[optimizer]

    return score 

# Run optimization
study = Study(direction="minimize", random_state=42) # Minimize the  score
study.optimize(objective, n_trials=50)
```

### Documentation
For more detailed information about the API and advanced usage, please refer to the full [documentation](https://marsopt.readthedocs.io/en/latest/).

## Contributing

Contributions are welcome! If you'd like to improve `marsopt` or suggest new features, feel free to fork the repository and submit a pull request.

## License
This project is licensed under the [MIT License](LICENSE).  

