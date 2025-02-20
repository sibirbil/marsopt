# marsopt
**Mixed Adaptive Random Search for Hyperparameter Optimization**

[![PyPI version](https://img.shields.io/pypi/v/marsopt.svg)](https://pypi.org/project/marsopt/)
[![License](https://img.shields.io/github/license/sibirbil/marsopt.svg)](LICENSE)
[![Python Versions](https://img.shields.io/pypi/pyversions/marsopt.svg)](https://pypi.org/project/marsopt/)

`marsopt` is a Python library designed to simplify and accelerate hyperparameter and black-box optimization and for mixed parameter spaces—supporting **continuous**, **integer**, and **categorical** parameters. Powered by **Mixed Adaptive Random Search**, `marsopt` dynamically balances exploration and exploitation through:
- **Adaptive noise** (cosine-annealed) for sampling,
- **Elite selection** to guide the search toward promising regions,
- Flexible handling of **log-scale** and **categorical** parameters,
- Minimization **or** maximization of any user-defined objective.

`marsopt` uses a Mixed Adaptive Random Search approach, iteratively refining a population of “elite” solutions and generating new candidates by **perturbing** those elites with gradually decreasing noise.

For a more detailed explanation, see the **[Algorithm Details](https://marsopt.readthedocs.io/en/latest/algorithm.html)** in the documentation.

---

## Features
- **Mixed Parameter Support**: Optimize integer, float (with optional log-scale), and categorical parameters in the same study.  
- **Adaptive Sampling**: Early iterations explore widely, while later iterations exploit the best regions found, thanks to a built-in **cosine annealing** scheme for noise.  
- **Easy Setup**: Simply define an objective function, specify a parameter search space, and run `study.optimize()`.  
- **Resume & Extend**: Continue a `Study` with more trials at any time without losing past information.  
- **Rich Tracking**: Inspect all trial details (objective values, parameters, times, etc.) for deeper analysis.  
- **Parameter Importance**: Quickly see which hyperparameters most correlate with objective outcomes (via Spearman correlation).

---

## Installation
Install `marsopt` from PyPI:

```bash
pip install marsopt
```

---

## Getting Started

### Quick Example

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

---

### Documentation
For more detailed information about the API and advanced usage, please refer to the full  [documentation](https://marsopt.readthedocs.io/en/latest/).

---

## Contributing

Contributions are welcome! If you'd like to improve `ruleopt` or suggest new features, feel free to fork the repository and submit a pull request.

---

## License
This project is licensed under the [MIT License](LICENSE).  

