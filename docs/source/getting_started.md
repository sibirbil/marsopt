Below is a detailed **Getting Started** guide for using the **marsopt** package to perform parameter optimization using the **Study** and **Trial** classes. This guide walks you through installation, basic usage, key features, and advanced tips. 

---

## 1. Introduction

`marsopt` is a Python library designed for easy and efficient hyperparameter optimization using a Mixed Adaptive Random Search approach. It supports numerical (both integer and float, optionally in log scale) and categorical parameters. The optimization process can be targeted towards **minimization** or **maximization** of the user-defined objective function.

In this guide, we will:

1. Install and import `marsopt`.
2. Define a custom objective function.
3. Set up a `Study`.
4. Run the optimization.
5. Analyze the results.

---

## 2. Installation

You can install `marsopt` (and its dependencies) using `pip`:

```bash
pip install marsopt
```

If you plan to use features like parameter importance (which requires `scipy`), ensure you have it installed:

```bash
pip install scipy
```

---

## 3. Basic Concepts

### 3.1. The **Study** Class

- A **Study** represents an optimization experiment.
- You initialize a **Study** by specifying key parameters such as:
  - `direction`: "minimize" or "maximize".
  - `n_init_points`: Number of purely random initial trials (defaults to `round(√n_trials)`, if not provided).
  - `initial_noise` and `final_noise`: Control the amount of noise (i.e., how aggressively or conservatively new parameter values are sampled).
  - `random_state`: For reproducibility.
  - `verbose`: Whether to print logs during optimization.

You then call the `.optimize()` method on the **Study** to run a number of trials (`n_trials`).

### 3.2. The **Trial** Class

- A **Trial** encapsulates a single evaluation of your objective function. 
- Within each **Trial**, you define hyperparameter suggestions via:
  - `suggest_float()`: for continuous parameters (e.g., learning rate).
  - `suggest_int()`: for integer-valued parameters (e.g., number of layers).
  - `suggest_categorical()`: for discrete string-based parameters (e.g., optimizer types).

### 3.3. Objective Function

- You must provide a callable `objective_function(trial)` that:
  1. Uses **suggest** methods to define parameters.
  2. Returns a single scalar metric (float) representing the performance to be optimized (the "objective" value).

---

## 4. Minimal Working Example

Below is a complete example showcasing how to use **marsopt** to optimize a synthetic machine learning model:

```python
import numpy as np
from marsopt import Study, Trial

def objective(trial: Trial) -> float:
    """
    Example objective function that simulates a machine learning model's performance.
    We'll optimize:
    - learning_rate (float)
    - num_layers (int)
    - optimizer_type (categorical)
    - dropout_rate (float)
    
    The function creates a synthetic response surface with some noise.
    """

    # 1) Suggest hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
    num_layers = trial.suggest_int("num_layers", 1, 5)
    optimizer = trial.suggest_categorical("optimizer", ["adam", "sgd", "rmsprop"])
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    
    # 2) Simulate model performance
    #    This is a synthetic function that "favors":
    #    - learning_rate around 0.001
    #    - higher number of layers (with diminishing returns)
    #    - 'rmsprop' as an example "best" (synthetic)
    #    - dropout around 0.3
    
    # Learning rate component (optimal around 0.001)
    lr_score = -3 * (np.log10(learning_rate) + 3) ** 2
    
    # Number of layers component (diminishing returns)
    layer_score = np.log1p(num_layers) * 20
    
    # Optimizer component (here we define some arbitrary preferences)
    optimizer_scores = {"adam": 10.0, "rmsprop": 50.0, "sgd": 10.0}
    opt_score = optimizer_scores[optimizer]
    
    # Dropout component (optimal around 0.3)
    dropout_score = -10 * (dropout_rate - 0.3) ** 2
    
    # Combine scores and add some noise (higher is "better")
    score = (lr_score + layer_score + opt_score + dropout_score + 10) / 4
    noise = np.random.normal(0, 0.1)
    
    # If 'direction' is "minimize", we return the negative of the score
    # so that higher 'score' leads to a lower 'objective'.
    return -(score + noise)

# 3) Create and run a Study
study = Study(
    direction="minimize",   # or "maximize"
    random_state=42,
    verbose=True,
)

n_trials = 100
study.optimize(objective, n_trials)

# 4) Retrieve best trial and more info
best = study.best_trial
print("Best trial iteration:", best["iteration"])
print("Best trial objective:", best["objective_value"])
print("Best trial parameters:", best["params"])
```

### Explanation

1. **Objective function**:
   - We define `objective(trial)` with a few parameter suggestions.
   - We compute a synthetic performance metric (`score`) and return it (inverted if we are minimizing).

2. **Study**:
   - We create a `Study` object with `direction="minimize"` (also accepts `"maximize"`).
   - We specify `random_state=42` for reproducibility.
   - `verbose=True` prints logs to track progress.

3. **Running optimization**:
   - We call `study.optimize(objective, n_trials)`, which runs 100 trials.

4. **Best trial**:
   - The `.best_trial` property gives a dictionary of the best trial’s iteration, objective value, execution time, and parameters.

---

## 5. Accessing Detailed Results

### 5.1. Trial History

```python
# Access the full list of trials conducted
all_trials = study.trials
for t in all_trials[:5]:  # print first 5
    print(f"Iteration: {t['iteration']}, Objective: {t['objective_value']}, "
          f"Params: {t['parameters']}")
```

Each element in `study.trials` is a dictionary containing:
- **iteration** (int)
- **objective_value** (float)
- **trial_time** (float) – the time spent in that trial
- **parameters** (dict) – a dict of all parameters suggested in that trial

### 5.2. Objective Values & Elapsed Times

```python
# NumPy arrays with all objective values and elapsed times
obj_values = study.objective_values
times = study.elapsed_times

print("Objective values:", obj_values)
print("Elapsed times:", times)
```

These arrays are in order of trial index.

---

## 6. Parameter Importance

`marsopt` includes a simple utility to measure **parameter importance** via Spearman correlation:

```python
importances = study.parameter_importance()
print("Parameter Importances:")
for param, importance in importances.items():
    print(f"{param}: {importance:.4f}")
```

It calculates absolute Spearman correlation between each parameter and the objective values, giving a rough sense of which parameters most influence the objective. (Requires `scipy`.)

---

## 7. Advanced Configuration

### 7.1. Controlling Noise

- **`initial_noise`** (float): The noise added to each parameter suggestion at the start of the search. 
  - Defaults to `0.2`.
- **`final_noise`** (float): The noise level as the search nears completion.
  - Defaults to `1 / n_trials` if not set.
- During the search, the noise transitions from `initial_noise` to `final_noise` following a cosine annealing schedule. This helps the search to explore widely at the beginning and refine towards the end.

### 7.2. Initial Random Points

- **`n_init_points`** (int): Number of random points before applying the adaptive strategy. 
  - By default, `n_init_points = round(√n_trials)`.

### 7.3. Adding Trials Incrementally

You can run additional trials beyond your initial `n_trials` by calling `.optimize()` again with more trials. `marsopt` will pick up where it left off:

```python
# Suppose we already ran 100 trials. Now we want 50 more:
study.optimize(objective, 50)
```

Your `Study` object will seamlessly extend its arrays and maintain state to continue optimization.

---

## 8. Common Errors and Troubleshooting

1. **Parameter Validation**: 
   - If you get a `TypeError` or `ValueError` when calling `suggest_float()`, `suggest_int()`, or `suggest_categorical()`, ensure:
     - `low < high` for numerical parameters.
     - `categories` has unique and valid entries for categorical parameters.
     - `log=True` parameters must have `low > 0`.

2. **Objective Function Not Returning float**:
   - Your `objective_function` **must** return a single `float`. Make sure your return type is correct.

3. **No Trials Conducted**:
   - Accessing properties like `best_trial` or `parameter_importance()` before any trials have run will raise errors. Call `study.optimize()` first.

---

## 9. Conclusion

`marsopt` offers a straightforward yet flexible interface for hyperparameter optimization, suitable for various tasks including Machine Learning model tuning. By combining random exploration and adaptive local search, it achieves a balance of exploration and exploitation, especially helpful in diverse search spaces.

**Key Takeaways**:
- Define an **objective function** that uses `Trial`'s suggest methods.
- Instantiate a **Study** with desired parameters.
- Call `.optimize()` to run a specified number of trials.
- Retrieve **best_trial**, track all trials with **trials**, or leverage **parameter_importance()**.

We hope this guide helps you quickly get started and effectively tune your models or processes with `marsopt`!

---

*For further details, bug reports, or feature requests, please visit the [GitHub repository](https://github.com/). If you encounter any issues or have suggestions for improvement, feel free to open an issue or submit a pull request.*