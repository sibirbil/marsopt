# Getting Started

## 1. Introduction

`marsopt` is a Python library designed to simplify and accelerate **hyperparameter optimization** tasks using a Mixed Adaptive Random Search approach. It effectively handles diverse hyperparameter types, including:

- **Numerical** (integer or float, optionally on a log scale),  
- **Categorical** (e.g., optimizer types, feature encoders, etc.).

`marsopt` provides an easy-to-use interface for **minimizing** or **maximizing** any user-defined objective (or loss) function, commonly found in **machine learning** or **deep learning** workflows. If you have ever tuned parameters like **learning rate**, **number of layers**, **dropout rates**, or **optimizer types**, you know how crucial and time-consuming this step can be. By leveraging adaptive sampling, `marsopt` can help you **explore** the parameter space broadly in the beginning and **exploit** promising areas in later iterations.

---

## 2. Installation

Install `marsopt` using `pip`:

```bash
pip install marsopt
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

You then **return** a **float or integer** that indicates your objective value.  

### 3.3. Objective Function

- The objective function is the heart of your optimization workflow.  
- It must receive a `Trial` object and use that object’s **suggest** methods to propose values.  
- After configuring and running your model or simulation with those values, it **returns** a single floating-point metric (e.g., validation loss, or negative of a validation accuracy, etc.).

---

## 4. Minimal Working Example

Below is a simplified yet demonstrative example of how to use `marsopt` to optimize a set of **typical machine learning hyperparameters**—learning rate, number of layers, optimizer type, and dropout rate:

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
```
[I 2025-02-20 19:56:20, 16] Optimization started with 50 trials.
[I 2025-02-20 19:56:20, 17] Trial 1 finished with value: -32.841185 and parameters: {'learning_rate': 0.001329, 'num_layers': 5, 'optimizer': adam}. Best is trial 0 with value: -32.841185.
[I 2025-02-20 19:56:20, 18] Trial 2 finished with value: -30.738093 and parameters: {'learning_rate': 0.006174, 'num_layers': 3, 'optimizer': rmsprop}. Best is trial 0 with value: -32.841185.
[I 2025-02-20 19:56:20, 18] Trial 3 finished with value: -6.086478 and parameters: {'learning_rate': 0.039676, 'num_layers': 3, 'optimizer': sgd}. Best is trial 0 with value: -32.841185.
...
...
[I 2025-02-20 19:56:20, 48] Trial 50 finished with value: -37.847763 and parameters: {'learning_rate': 0.001313, 'num_layers': 5, 'optimizer': rmsprop}. Best is trial 34 with value: -37.917444.
```


---

## 5. Accessing Detailed Results

### 5.1. Trial History

After the optimization completes, you can inspect the details of each trial:

```python
study.trials
```

```python
[{'iteration': 1,
  'objective_value': -32.84118472952258,
  'trial_time': 0.000260959001025185,
  'parameters': {'learning_rate': 0.0013292918943162175,
   'num_layers': 5,
   'optimizer': 'adam'}},
 {'iteration': 2,
  'objective_value': -30.738093352759925,
  'trial_time': 0.0001224999978148844,
  'parameters': {'learning_rate': 0.006173770394704574,
   'num_layers': 3,
   'optimizer': 'rmsprop'}},
  ...
 {'iteration': 50,
  'objective_value': -37.84776341066681,
  'trial_time': 0.00016675000006216578,
  'parameters': {'learning_rate': 0.001312740598216683,
   'num_layers': 5,
   'optimizer': 'rmsprop'}}]
```

Each trial dictionary contains:
- **iteration**: The trial index.  
- **objective_value**: The final metric or loss returned by your `objective` function.  
- **trial_time**: How long that trial took to run.  
- **parameters**: A dictionary of all hyperparameters suggested for that trial.

####  5.1.1 Best Trial

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

### 5.2. Objective Values & Elapsed Times

Sometimes you want arrays of all objective values to quickly visualize or analyze them:

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

---

## 6. Parameter Importance

`marsopt` can provide a quick measure of parameter importance via **Spearman correlation**:

```python
study.parameter_importance()
```

```python
{'optimizer': 0.6924193652047251,
 'num_layers': 0.655940963922529,
 'learning_rate': 0.052292917166866744}
```

- This calculates the absolute Spearman correlation coefficient between each parameter and the objective values.  
- It can offer a **rough** idea of which hyperparameters most strongly affect model performance. Requires `scipy` to be installed.

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
study.optimize(objective, n_trials=50)
```
```
[I 2025-02-20 20:17:46, 688] Trial 51 finished with value: -37.917177 and parameters: {'learning_rate': 0.001021, 'num_layers': 5, 'optimizer': rmsprop}. Best is trial 34 with value: -37.917444.
[I 2025-02-20 20:17:46, 689] Trial 52 finished with value: -37.066078 and parameters: {'learning_rate': 0.002586, 'num_layers': 5, 'optimizer': rmsprop}. Best is trial 34 with value: -37.917444.
...
[I 2025-02-20 20:17:46, 722] Trial 100 finished with value: -37.908955 and parameters: {'learning_rate': 0.000909, 'num_layers': 5, 'optimizer': rmsprop}. Best is trial 93 with value: -37.917579.
```


`marsopt` retains its internal state and continues from the previously explored space.