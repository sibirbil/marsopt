# Performance Analysis

## Blackbox Optimization Experiments

For each problem in blackbox optimization, we conducted 30 runs with different random seeds to compare the average performance of different algorithms. The comparison included:
- CMA-ES
- Optuna
- Random Search
- MARS (our algorithm)

### Results at Different Iterations:

#### 100 Iterations
![Value comparison results for 100 trials](_static/performance/heatmap_values_100_trials.png)
![Frequency-based comparison for 100 trials](_static/performance/heatmap_frequency_100_trials.png)

The results show that MARS significantly outperformed other algorithms.

#### 1000 Iterations
![Value comparison results for 1000 trials](_static/performance/heatmap_values_1000_trials.png)
![Frequency-based comparison for 1000 trials](_static/performance/heatmap_frequency_1000_trials.png)

CMA-ES took the lead, though its results were very close to MARS's performance.

## LightGBM Hyperparameter Optimization

We compared two different objective functions for hyperparameter optimization using LightGBM on the California Housing dataset:

### Simple vs Complex Objective Differences:

#### Simple Objective
* Uses only GBDT (Gradient Boosting Decision Tree) as the boosting type
* Has a narrower parameter search space
* Optimizes basic parameters such as:
  - L1/L2 regularization
  - Learning rate
  - Number of leaves
  - Feature and bagging fractions
  - Minimum child samples

![Simple objective results on California Housing](_static/performance/hyperparameter_california_housing_simple.png)

#### Complex Objective
* Allows selection between GBDT and GOSS (Gradient-based One-Side Sampling) boosting types
* Features a wider parameter search space
* Includes additional parameters such as:
  - Top rate and other rate (for GOSS)
  - Maximum depth
  - Maximum bin
  - Additional tuning options for both boosting types

![Complex objective results on California Housing](_static/performance/hyperparameter_california_housing_complex.png)

### Performance
The results show that both objective functions performed slightly better than the standard Optuna optimization, with the complex objective providing marginally better results due to its more comprehensive parameter space and flexibility in boosting type selection.