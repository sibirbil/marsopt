# marsopt
**Mixed Adaptive Random Search for Hyperparameter Optimization**

[![PyPI version](https://img.shields.io/pypi/v/marsopt.svg)](https://pypi.org/project/marsopt/)
[![License](https://img.shields.io/github/license/yourusername/marsopt.svg)](LICENSE)
[![Python Versions](https://img.shields.io/pypi/pyversions/marsopt.svg)](https://pypi.org/project/marsopt/)

`marsopt` is a Python library designed to **simplify and accelerate hyperparameter optimization** (HPO) for mixed parameter spaces—supporting **continuous**, **integer**, and **categorical** parameters. Powered by **Mixed Adaptive Random Search**, `marsopt` dynamically balances exploration and exploitation through:
- **Adaptive noise** (cosine-annealed) for sampling,
- **Elite selection** to guide the search toward promising regions,
- Flexible handling of **log-scale** and **categorical** parameters,
- Minimization **or** maximization of any user-defined objective.


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

Below is a simplified example of how to use `marsopt` to tune a few common hyperparameters.

```python
import numpy as np
from marsopt import Study, Trial

def objective_function(trial: Trial) -> float:
    # 1) Suggest hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
    num_layers    = trial.suggest_int("num_layers", 1, 5)
    optimizer     = trial.suggest_categorical("optimizer", ["adam", "sgd", "rmsprop"])
    
    # 2) Simulate a "loss" (your real code would train a model & evaluate validation loss)
    #    We'll create a synthetic function for demonstration:
    simulated_loss = (np.log10(learning_rate) + 3)**2 - np.log1p(num_layers)*5.0
    if optimizer == "rmsprop":
        simulated_loss -= 5  # let's say 'rmsprop' performs slightly better

    return simulated_loss  # 'direction="minimize"' by default

# 3) Create a Study
study = Study(
    direction="minimize",
    random_state=42,
    verbose=True,
)

# 4) Run optimization
study.optimize(objective_function, n_trials=30)

# 5) Check best result
best_trial = study.best_trial
print("Best iteration:", best_trial["iteration"])
print("Best objective value:", best_trial["objective_value"])
print("Best parameters:", best_trial["params"])
```

1. **Import** and define an `objective_function` that receives a `Trial` object.  
2. **Suggest parameters** (float, int, categorical).  
3. **Return** a floating-point score or loss.  
4. **Initialize** a `Study` and call `study.optimize(...)`.  
5. **Inspect** the best trial found.

### Real-World Scenarios
In a real setting, instead of a synthetic function, you would:
- Configure a **model** (e.g., with PyTorch, TensorFlow, scikit-learn, XGBoost, etc.) using the parameters from `trial`.  
- Train and evaluate performance on a **validation set**.  
- Return the **validation loss** (or `-accuracy` if you prefer to minimize negative accuracy, or switch to `direction="maximize"` and just return accuracy).

---

## Basic Concepts

### The `Study` Class
A `Study` object encapsulates the entire hyperparameter optimization process:
- **`direction`**: `"minimize"` (e.g., for loss) or `"maximize"` (e.g., for accuracy).  
- **`n_init_points`**: Number of random trials before adaptive sampling begins. Defaults to ~\(\sqrt{n\_trials}\).  
- **`initial_noise`** and **`final_noise`**: Controls the range of exploration; noise is annealed from a higher value to a lower value over time.  
- **`random_state`**: Seed for reproducibility.  
- **`verbose`**: Print logs during each trial if `True`.

### The `Trial` Class
Each `Trial` represents a single evaluation of the objective. Inside `objective_function(trial)`:
- `trial.suggest_float(name, low, high, log=False)`  
- `trial.suggest_int(name, low, high)`  
- `trial.suggest_categorical(name, categories)`

Return a **float** representing the objective value (loss, metric, etc.).

### Objective Function
Your custom objective function must:
1. **Propose** values for each hyperparameter via `trial.suggest_...` methods.  
2. **Train / evaluate** your model (or run any process) using these hyperparameters.  
3. **Return** a single float. `marsopt` will record this value to guide future sampling.

---

## Accessing Results

### Trial History
After optimization finishes:
```python
for t in study.trials:
    print("Iteration:", t["iteration"])
    print("Objective:", t["objective_value"])
    print("Params:", t["parameters"])
    print("---")
```
Each trial entry includes:
- `iteration`  
- `objective_value`  
- `parameters` (dict of all hyperparameter suggestions)  
- `trial_time` (duration of that trial)

### Objective Values & Elapsed Times
To quickly fetch arrays:
```python
obj_values = study.objective_values
times      = study.elapsed_times
print("Objective values:", obj_values)
print("Elapsed times:", times)
```

---

## Parameter Importance
`marsopt` can compute parameter importances via **Spearman correlation** (requires `scipy`):
```python
importances = study.parameter_importance()
for param, imp in importances.items():
    print(f"{param}: {imp:.4f}")
```
Higher absolute correlation indicates a stronger relationship between a parameter and the objective outcome. This can help identify which hyperparameters matter most.

---

## Advanced Configuration

- **Controlling Noise**  
  - `initial_noise=0.2` (default)  
  - `final_noise=1/n_trials` or a small constant if desired  
  Noise is annealed using **cosine** scheduling from early exploration to late exploitation.

- **Increasing Trials**  
  If you want to refine further after an initial run:
  ```python
  study.optimize(objective_function, n_trials=50)  # add 50 more trials
  ```
  All information from previous trials is preserved.

- **Integer and Categorical Sampling**  
  - Integer parameters are handled by sampling from a continuous distribution and rounding probabilistically.  
  - Categorical parameters are internally represented with one-hot vectors; a softmax function with decreasing temperature guides the choice of categories over time.

---

## Algorithm Overview

`marsopt` uses a Mixed Adaptive Random Search approach, iteratively refining a population of “elite” solutions and generating new candidates by **perturbing** those elites with gradually decreasing noise. Key steps:

1. **Random Initialization**  
   - Sample `n_init_points` random trials to explore the space freely.

2. **Elite Selection**  
   - Identify the best-performing trials as “elites” in each iteration.

3. **Parameter Perturbation**  
   - For each parameter, sample around elite values with noise scaled by \(\eta(t)\).  
   - For categorical parameters, transform them into one-hot vectors, average them over elites, add noise, and apply softmax to decide the final category.

4. **Noise Annealing**  
   - \(\eta(t)\) is typically decreased using a **cosine schedule**, ensuring exploration early on and exploitation later.

5. **Objective Evaluation**  
   - Evaluate the objective on new candidate solutions.

6. **Repeat**  
   - Update the ranking of elite solutions, lower the noise, and continue until reaching the desired number of trials.

This process yields a powerful yet flexible method of handling diverse parameter types within a single optimization framework.

---

## Contributing
Contributions are welcome! If you have ideas, suggestions, or bug fixes:
1. **Open an Issue** describing the problem or feature request.  
2. **Fork** this repository, **commit** your changes, and **create a pull request**.  
3. Ensure all existing tests pass and add new tests if needed.

We appreciate community feedback to make `marsopt` more robust and feature-rich.

---

## License
This project is licensed under the [MIT License](LICENSE).  
You are free to use, modify, and distribute this code for both commercial and non-commercial purposes under the terms of the license.

---

**Happy optimizing with `marsopt`!** For questions or guidance, feel free to open an issue on our [GitHub repository](https://github.com/yourusername/marsopt).
