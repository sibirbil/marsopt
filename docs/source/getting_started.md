# Getting Started

**Mixed Adaptive Random Search** (MARS) is a method for **optimizing** any user-defined **black-box problems**, commonly found in **machine learning** or **deep learning** hyperparameter tuning workflows. MARS explores the space of variables broadly in the beginning and exploits promising areas in later iterations. Mathematically, MARS can be used to solve

$$
\min\{f(x) : x \in \mathcal{X}\},
$$

where $f$ is a real-valued function denoting the **objective function** and $\mathcal{X}$ is the **variable space**. MARS effectively handles diverse variable types including:
- **numerical** (integer or float, optionally on a log scale),  
- **categorical** (e.g., optimizer types, feature encoders, and so on).

To provide an easy-to-use interface for MARS, we have implemented a new Python library `marsopt` that we introduce in the subsequent part. Note that, we refer to the iterates of MARS interchangeably as **trials**, **solutions**, or **points** - these all reside in $\mathcal{X}$. 

## 1. Installation

Install `marsopt` using `pip`:

```bash
pip install marsopt
```

## 2. Basic Concepts

In this section, we will introduce the key components of `marsopt`. It is worth mentioning that our Python objects are named similarly to those found in the popular `optuna` package, making it easier for users to navigate and understand the structure.

### The **Study** Class

A `Study` object encapsulates your entire optimization experiment. Key configuration options include:

- **`direction`**:  
  - `"minimize"` or `"maximize"`.  
  - If you have a loss function (like cross-entropy), you might want to **minimize** it.  


- **`n_init_points`**:  
  - The number of purely random initial trials (defaults to `max(10, round(√n_trials))` if not specified).  
  - These initial random trials help the optimizer gather a broad sense of the search space.

- **`initial_noise`** and **`final_noise`**:  
  - Control how much variability (i.e., "noise") is introduced when suggesting new variable values.  
  - The noise decreases over time, enabling exploration early on and fine-tuning later.

- **`random_state`**:  
  - Seed for reproducibility. Provide an integer so you can replicate results exactly.

- **`verbose`**:  
  - `True` prints logs after each trial; `False` runs silently.

Once configured, you call the **`.optimize()`** method to run a specified number of trials (`n_trials`).

### The **Trial** Class

A `Trial` represents a **single** evaluation of your objective function. Inside the `objective_function(trial)`:

- You define how to **suggest** each variable:
  - `suggest_float(name, low, high, log=False)`  
  - `suggest_int(name, low, high, log=False)`  
  - `suggest_categorical(name, categories)`

You then **return** a **float or integer** that indicates your objective value.  

### Objective Function

- It must receive a `Trial` object and use that object’s **suggest** methods to propose values.  
- After configuring and running your model or simulation with those values, it must **return a single real numeric value**. NaN is not accepted; positive or negative infinity is allowed.

## 3. Minimal Working Example

Below is a simplified yet demonstrative example of how to use `marsopt` to optimize a set of **typical machine learning hyperparameters** - learning rate, number of layers, optimizer type, and dropout rate:

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

    return -score

# Run optimization
study = Study(direction="minimize", random_state=42)
study.optimize(objective, n_trials=50)
```
```
[I ...] Optimization started with 50 trials.
[I ...] Trial 1 finished with value: -7.249446 and variables: {'learning_rate': 0.020983, 'num_layers': 2, 'optimizer': sgd}. Best is trial 1 with value: -7.249446.
[I ...] Trial 2 finished with value: -8.678749 and variables: {'learning_rate': 0.037652, 'num_layers': 4, 'optimizer': sgd}. Best is trial 2 with value: -8.678749.
[I ...] Trial 3 finished with value: -7.42204 and variables: {'learning_rate': 0.084502, 'num_layers': 2, 'optimizer': adam}. Best is trial 2 with value: -8.678749.
...
...
[I ...] Trial 50 finished with value: -32.903512 and variables: {'learning_rate': 0.000885, 'num_layers': 5, 'optimizer': adam}. Best is trial 37 with value: -37.91758.
```

## 4. Accessing Detailed Results

Below we detail how one can collect information about the optimization process conducted by `marsopt`.  

### Trial History

After the optimization completes, you can inspect the details of each trial:

```python
study.trials
```

```python
[{'iteration': 1,
  'objective_value': -7.249445914023765,
  'trial_time': ...,
  'variables': {'learning_rate': 0.020983027299866144,
   'num_layers': 2,
   'optimizer': 'sgd'},
  'user_attrs': {}},
 {'iteration': 2,
  'objective_value': -8.6787492582556,
  'trial_time': ...,
  'variables': {'learning_rate': 0.03765249501831187,
   'num_layers': 4,
   'optimizer': 'sgd'},
  'user_attrs': {}},
  ...
 {'iteration': 50,
  'objective_value': -32.90351179940006,
  'trial_time': ...,
  'variables': {'learning_rate': 0.0008849700072462417,
   'num_layers': 5,
   'optimizer': 'adam'},
  'user_attrs': {}}]
```

Each trial dictionary contains:
- **iteration**: The trial index.  
- **objective_value**: The final metric or loss returned by your `objective` function.  
- **trial_time**: How long that trial took to run.  
- **variables**: A dictionary of all variables suggested for that trial.
- **user_attrs**: A dictionary of user-defined attributes added via `trial.add_attr()`.

Likewise, one can also inspect the **best trial**:

```python
study.best_trial
```

```python
{'iteration': 37,
 'objective_value': -37.91757992304764,
 'trial_time': ...,
 'variables': {'learning_rate': 0.0010039652381640435,
  'num_layers': 5,
  'optimizer': 'rmsprop'},
 'user_attrs': {}}
```

### Objective Values and Elapsed Times

Sometimes you want arrays of all objective function values to quickly visualize or analyze them:

```python
study.objective_values
```

```python
array([-7.24944591, -8.67874926, -7.42203965, ..., -32.9035118])
```

```python
study.elapsed_times
```

```python
array([...])  # execution times in seconds
```

## 5. Advanced Configuration

This section gives a few other parameters that users can adjust.

### Controlling Noise

- **`initial_noise`** (float): The initial sampling noise. Default is `0.33`.
- **`final_noise`** (float): How much noise remains at the end of the search. Defaults to `max(1e-7, min(1 / n_trials, initial_noise))` if not set.

Internally, a **cosine annealing** schedule adjusts noise from `initial_noise` down to `final_noise`, facilitating broad exploration early on and refinement later.

### Initial Random Points

- **`n_init_points`** (int): Number of random points sampled before adaptive strategies kick in.  Defaults to `max(10, round(√n_trials))` if unspecified.

### Epsilon-Greedy Exploration

- **`epsilon`** (float, default `1.0`): Controls a small dose of pure random exploration that is mixed into the adaptive phase. At each adaptive trial, with probability `epsilon / (t + 1)` MARS ignores the elite-guided sampler and draws a uniform random sample from the search space. The probability decays harmonically with the trial index, so exploration is strongest early on and fades over time. Set to a smaller value (or `0`) to reduce or disable random fallback.

### Elite Window

- **`elite_window`** (int, default `None`): If set, only the most recent `elite_window` completed trials are considered when forming the elite set (and the candidate pool used by the categorical good/bad scoring). Useful when the search space drifts, when older trials are no longer representative, or when you want the optimizer to “forget” early random exploration faster. If `None`, the full completed history is used.

### Adding More Trials Later

If you decide 50 trials aren’t enough, you can resume with additional trials:

```python
study.optimize(objective, n_trials=50)
```
```
[I ...] Trial 51 finished with value: -36.412249 and variables: {'learning_rate': 0.000283, 'num_layers': 5, 'optimizer': rmsprop}. Best is trial 37 with value: -37.91758.
[I ...] Trial 52 finished with value: -35.939487 and variables: {'learning_rate': 0.0015, 'num_layers': 4, 'optimizer': rmsprop}. Best is trial 37 with value: -37.91758.
...
[I ...] Trial 100 finished with value: -37.901111 and variables: {'learning_rate': 0.000876, 'num_layers': 5, 'optimizer': rmsprop}. Best is trial 37 with value: -37.91758.
```
`marsopt` retains its internal state and continues from the previously explored space.
