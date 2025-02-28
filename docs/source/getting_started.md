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
  - `suggest_float(param_name, low, high, log=False)`  
  - `suggest_int(param_name, low, high, log=False)`  
  - `suggest_categorical(param_name, categories)`

You then **return** a **float or integer** that indicates your objective value.  

### Objective Function

- It must receive a `Trial` object and use that object’s **suggest** methods to propose values.  
- After configuring and running your model or simulation with those values, it must **return a single numeric value**.

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

    return score 

# Run optimization
study = Study(direction="minimize") # Minimize the  score
study.optimize(objective, n_trials=50)
```
```
[I 2025-02-20 19:56:20, 16] Optimization started with 50 trials.
[I 2025-02-20 19:56:20, 17] Trial 1 finished with value: -32.841185 and variables: {'learning_rate': 0.001329, 'num_layers': 5, 'optimizer': adam}. Best is trial 0 with value: -32.841185.
[I 2025-02-20 19:56:20, 18] Trial 2 finished with value: -30.738093 and variables: {'learning_rate': 0.006174, 'num_layers': 3, 'optimizer': rmsprop}. Best is trial 0 with value: -32.841185.
[I 2025-02-20 19:56:20, 18] Trial 3 finished with value: -6.086478 and variables: {'learning_rate': 0.039676, 'num_layers': 3, 'optimizer': sgd}. Best is trial 0 with value: -32.841185.
...
...
[I 2025-02-20 19:56:20, 48] Trial 50 finished with value: -37.847763 and variables: {'learning_rate': 0.001313, 'num_layers': 5, 'optimizer': rmsprop}. Best is trial 34 with value: -37.917444.
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
  'objective_value': -32.84118472952258,
  'trial_time': 0.000260959001025185,
  'variables': {'learning_rate': 0.0013292918943162175,
   'num_layers': 5,
   'optimizer': 'adam'}},
 {'iteration': 2,
  'objective_value': -30.738093352759925,
  'trial_time': 0.0001224999978148844,
  'variables': {'learning_rate': 0.006173770394704574,
   'num_layers': 3,
   'optimizer': 'rmsprop'}},
  ...
 {'iteration': 50,
  'objective_value': -37.84776341066681,
  'trial_time': 0.00016675000006216578,
  'variables': {'learning_rate': 0.001312740598216683,
   'num_layers': 5,
   'optimizer': 'rmsprop'}}]
```

Each trial dictionary contains:
- **iteration**: The trial index.  
- **objective_value**: The final metric or loss returned by your `objective` function.  
- **trial_time**: How long that trial took to run.  
- **variables**: A dictionary of all variables suggested for that trial.

Likewise, one can also inspect the **best trial**:

```python
study.best_trial
```

```python
{'iteration': 35, 
 'objective_value': -37.917443575884434, 
 'trial_time': 0.0003397920008865185, 
 'params': {'learning_rate': 0.0010127390829420338, 
  'num_layers': 5, 
  'optimizer': 'rmsprop'}}
```

### Objective Values and Elapsed Times

Sometimes you want arrays of all objective function values to quickly visualize or analyze them:

```python
study.objective_values
```

```python
array([-32.84118473, -30.73809335,  -6.08647779, ..., -37.84776341])
```

```python
study.elapsed_times
```

```python
array([2.60959001e-04, 1.22499998e-04, 1.15458002e-04, ..., 1.66750000e-04])
```

## 5. Advanced Configuration

This section gives a few other parameters that users can adjust.

### Controlling Noise

- **`initial_noise`** (float): The initial sampling noise. Default is `0.2`.  
- **`final_noise`** (float): How much noise remains at the end of the search. Defaults to `2 / n_trials` if not set.  

Internally, a **cosine annealing** schedule adjusts noise from `initial_noise` down to `final_noise`, facilitating broad exploration early on and refinement later.

### Initial Random Points

- **`n_init_points`** (int): Number of random points sampled before adaptive strategies kick in.  Defaults to `max(10, round(√n_trials))` if unspecified.

### Adding More Trials Later

If you decide 50 trials aren’t enough, you can resume with additional trials:

```python
study.optimize(objective, n_trials=50)
```
```
[I 2025-02-20 20:17:46, 688] Trial 51 finished with value: -37.917177 and variables: {'learning_rate': 0.001021, 'num_layers': 5, 'optimizer': rmsprop}. Best is trial 34 with value: -37.917444.
[I 2025-02-20 20:17:46, 689] Trial 52 finished with value: -37.066078 and variables: {'learning_rate': 0.002586, 'num_layers': 5, 'optimizer': rmsprop}. Best is trial 34 with value: -37.917444.
...
[I 2025-02-20 20:17:46, 722] Trial 100 finished with value: -37.908955 and variables: {'learning_rate': 0.000909, 'num_layers': 5, 'optimizer': rmsprop}. Best is trial 93 with value: -37.917579.
```
`marsopt` retains its internal state and continues from the previously explored space.