# marsopt
**Mixed Adaptive Random Search for Optimization**

`marsopt` is a Python library designed for optimization tasks involving **mixed parameter spaces**, supporting both numerical (integer, float, log-scale) and categorical parameters. It provides an **adaptive random search** mechanism, where noise injection and annealing strategies help balance exploration and exploitation through the optimization process. You can use `marsopt` for either **minimization** or **maximization** problems.

## Key Features
- **Mixed Parameter Spaces**: Easily handle integers, floats (with optional log-scaling), and categorical parameters in a single optimization run.
- **Adaptive Noise**: Dynamically adjust the noise level as iterations progress; start with higher noise for broad exploration and reduce it over time for fine-grained local search.
- **Automatic Initialization**: Begin with a set of random samples (defined by `n_init_points` or chosen automatically) to establish a good initial distribution of parameter values.
- **Flexible Objective**: Minimize or maximize any user-defined objective function.
- **Logging and Analysis**: Track objective values, trial durations, and parameter suggestions for analysis and reproducibility.

## Installation
```bash
pip install marsopt
```

## Getting Started

### Basic Example
Below is a minimal example demonstrating how to set up and run an optimization process using `marsopt`:

```python
import numpy as np
from marsopt import Study, Trial

# Define your objective function
def objective_function(trial: Trial) -> float:
    # Suggest a float parameter between 0.1 and 10, possibly on a log scale
    param_float = trial.suggest_float("learning_rate", 0.1, 10.0, log=True)
    
    # Suggest an integer parameter between 1 and 100
    param_int = trial.suggest_int("num_trees", 1, 100)
    
    # Suggest a categorical parameter
    param_cat = trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"])
    
    # Suppose our objective is a mock function (you can replace it with an actual model evaluation)
    # For demonstration, we'll just combine them in a random way:
    loss = (param_float * np.log(param_int + 1)) - len(param_cat)
    
    return loss  # by default, we minimize this loss

# Create a Study
study = Study(
    initial_noise=0.2,
    direction="minimize",
    random_state=42,
    verbose=True
)

# Run 50 optimization trials
study.optimize(objective_function, n_trials=50)

# Get the best trial
best = study.best_trial
print("Best iteration:", best["iteration"])
print("Best objective value:", best["objective_value"])
print("Best parameters:", best["params"])
```

### Explanation
1. **Create an objective function** (`objective_function`) that receives a `Trial` object.  
2. **Suggest parameters** within specified bounds:
   - `trial.suggest_float(...)` for continuous variables (can be log-scaled).
   - `trial.suggest_int(...)` for integer variables (can also be log-scaled).
   - `trial.suggest_categorical(...)` for categorical variables.
3. **Return a numeric value** representing the objective. By default, the `Study` tries to **minimize** this value.  
4. **Initialize a `Study`** object, setting `initial_noise`, `direction`, `random_state`, etc.  
5. **Run `optimize`** with `n_trials=50`. The study will iterate 50 times, each time sampling a new set of parameters and evaluating the objective.  
6. **Retrieve results** with `study.best_trial` or inspect the entire history using `study.trials`, `study.objective_values`, and `study.elapsed_times`.

## Advanced Usage
- **Updating an Existing Study**: If you already have some trials completed, you can rerun `optimize` with additional trials, and `marsopt` will continue from where it left off.
- **Logging**: By setting `verbose=True`, each trial’s progress will be printed out. You can also customize logging behavior by modifying the `OptimizationLogger`.

## Contributing
Contributions to `marsopt` are welcome! Feel free to open issues or pull requests to improve the library.
