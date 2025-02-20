Below is an expanded version of the `getting_started.md` file, presented in **English**, with additional details and references to **real-world hyperparameter optimization** scenarios.

---

# Getting Started

## 1. Introduction

`marsopt` is a Python library designed to simplify and accelerate **hyperparameter optimization** tasks using a Mixed Adaptive Random Search approach. It effectively handles diverse hyperparameter types, including:

- **Numerical** (integer or float, optionally on a log scale),  
- **Categorical** (e.g., optimizer types, feature encoders, etc.).

`marsopt` provides an easy-to-use interface for **minimizing** or **maximizing** any user-defined objective (or loss) function, commonly found in **machine learning** or **deep learning** workflows. If you have ever tuned parameters like **learning rate**, **number of layers**, **dropout rates**, or **optimizer types**, you know how crucial and time-consuming this step can be. By leveraging adaptive sampling, `marsopt` can help you **explore** the parameter space broadly in the beginning and **exploit** promising areas in later iterations.

In this guide, we will:

1. Install and import `marsopt`.  
2. Define a custom objective function that represents a real-world training scenario.  
3. Set up and configure a `Study`.  
4. Run the optimization.  
5. Inspect results and interpret them for further improvements.  

---

## 2. Installation

Install `marsopt` using `pip`:

```bash
pip install marsopt
```

If you plan to use the parameter importance features, you should also ensure `scipy` is installed:

```bash
pip install scipy
```

---

## 3. Basic Concepts

### 3.1. The **Study** Class

A `Study` object encapsulates your entire hyperparameter optimization experiment. Key configuration options include:

- **`direction`**:  
  - `"minimize"` or `"maximize"`.  
  - If you have a loss function (like cross-entropy), you might want to **minimize** it.  
  - If you have a metric (like accuracy or F1 score), you might want to **maximize** it.

- **`n_init_points`**:  
  - The number of purely random initial trials (defaults to `round(√n_trials)` if not specified).  
  - These initial random trials help the optimizer gather a broad sense of the search space.

- **`initial_noise`** and **`final_noise`**:  
  - Control how much variability (i.e., "noise") is introduced when suggesting new parameter values.  
  - The noise typically decreases over time (cosine annealing), enabling exploration early on and fine-tuning later.

- **`random_state`**:  
  - Seed for reproducibility. Provide an integer so you can replicate results exactly.

- **`verbose`**:  
  - `True` prints logs after each trial; `False` runs silently.

Once configured, you call the **`.optimize()`** method to run a specified number of trials (`n_trials`).

### 3.2. The **Trial** Class

A `Trial` represents a **single** evaluation of your objective function. Inside the `objective_function(trial)`:

- You define how to **suggest** each hyperparameter:
  - `suggest_float(param_name, low, high, log=False)`  
  - `suggest_int(param_name, low, high)`  
  - `suggest_categorical(param_name, categories)`

You then **return** a **float** that indicates your objective value (or loss).  

### 3.3. Objective Function

- The objective function is the heart of your optimization workflow.  
- It must receive a `Trial` object and use that object’s **suggest** methods to propose values.  
- After configuring and running your model or simulation with those values, it **returns** a single floating-point metric (e.g., validation loss, or negative of a validation accuracy, etc.).

---

## 4. Minimal Working Example

Below is a simplified yet demonstrative example of how to use `marsopt` to optimize a set of **typical machine learning hyperparameters**—learning rate, number of layers, optimizer type, and dropout rate:

```python
import numpy as np
from marsopt import Study, Trial

def objective(trial: Trial) -> float:
    """
    Example objective function simulating a model's performance.
    We're optimizing:
      - learning_rate (float)
      - num_layers (int)
      - optimizer_type (categorical)
      - dropout_rate (float)
    
    This function returns a single float to be minimized.
    """

    # 1) Suggest hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
    num_layers = trial.suggest_int("num_layers", 1, 5)
    optimizer = trial.suggest_categorical("optimizer", ["adam", "sgd", "rmsprop"])
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    
    # 2) Simulate model performance (synthetic)
    #    In a real case, you would train a model here (e.g., PyTorch or scikit-learn)
    #    and compute a validation loss or metric.

    # Below is a hypothetical "true" score function:
    lr_score = -3 * (np.log10(learning_rate) + 3) ** 2  # best around 1e-3
    layer_score = np.log1p(num_layers) * 20             # more layers is better, but diminishing returns
    optimizer_scores = {"adam": 10.0, "rmsprop": 50.0, "sgd": 10.0}
    opt_score = optimizer_scores[optimizer]
    dropout_score = -10 * (dropout_rate - 0.3) ** 2     # best around 0.3

    # Combine them into a single score. We'll say "higher is better" for the moment,
    # but since 'direction' is set to 'minimize', we return the negative of it.
    raw_score = lr_score + layer_score + opt_score + dropout_score + 10
    noise = np.random.normal(0, 0.1)  # add random noise

    # Return negative (we want to minimize the objective)
    return -(raw_score + noise)

# 3) Create and run a Study
study = Study(
    direction="minimize",
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

### In a Real-World Scenario

- Instead of the synthetic function, you might build and train a **neural network** (e.g., using PyTorch, TensorFlow, or another framework), measure your **validation loss** or **metric**, and return that value.  
- If you want to **maximize accuracy**, you can simply return `-accuracy` when using `direction="minimize"`, or switch the study direction to `"maximize"` and return `accuracy` directly.

---

## 5. Accessing Detailed Results

### 5.1. Trial History

After the optimization completes, you can inspect the details of each trial:

```python
all_trials = study.trials
for t in all_trials[:5]:  # print first 5
    print(f"Iteration: {t['iteration']}, Objective: {t['objective_value']}, "
          f"Params: {t['parameters']}")
```

Each trial dictionary contains:
- **iteration**: The trial index (1-based or 0-based depending on implementation).  
- **objective_value**: The final metric or loss returned by your `objective` function.  
- **trial_time**: How long that trial took to run.  
- **parameters**: A dictionary of all hyperparameters suggested for that trial.

### 5.2. Objective Values & Elapsed Times

Sometimes you want arrays of all objective values to quickly visualize or analyze them:

```python
obj_values = study.objective_values
times = study.elapsed_times

print("Objective values:", obj_values)
print("Elapsed times:", times)
```

---

## 6. Parameter Importance

`marsopt` can provide a quick measure of parameter importance via **Spearman correlation**:

```python
importances = study.parameter_importance()
print("Parameter Importances:")
for param, importance in importances.items():
    print(f"{param}: {importance:.4f}")
```

- This calculates the absolute Spearman correlation coefficient between each parameter and the objective values.  
- It can offer a **rough** idea of which hyperparameters most strongly affect model performance. (Requires `scipy` to be installed.)

---

## 7. Advanced Configuration

### 7.1. Controlling Noise

- **`initial_noise`** (float): The initial sampling noise. Default is `0.2`.  
- **`final_noise`** (float): How much noise remains at the end of the search. Defaults to `1 / n_trials` if not set.  

Internally, a **cosine annealing** schedule adjusts noise from `initial_noise` down to `final_noise`, facilitating broad exploration early on and refinement later.

### 7.2. Initial Random Points

- **`n_init_points`** (int): Number of random points sampled before adaptive strategies kick in.  
  - Defaults to `round(√n_trials)` if unspecified.

### 7.3. Adding More Trials Later

If you decide 100 trials aren’t enough, you can resume with additional trials:

```python
# Add 50 more trials
study.optimize(objective, n_trials=50)
```

`marsopt` retains its internal state and continues from the previously explored space.

---

## 8. Common Errors and Troubleshooting

1. **Parameter Ranges**:  
   - Ensure for `suggest_float()` or `suggest_int()`, the `low` argument is strictly less than `high`.  
   - For `log=True`, make sure both `low` and `high` are > 0.

2. **Return Type**:  
   - The `objective` function **must** return a single `float`. Returning `None` or an array will cause errors.

3. **No Trials Conducted**:  
   - Calling methods like `study.best_trial` before any trials are run will raise an exception. Make sure to call `study.optimize()` first.

4. **Categorical Values**:  
   - `suggest_categorical()` requires a non-empty list of categories.  

---
