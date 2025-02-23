import time
import numpy as np
import optuna
from marsopt import Study
import pandas as pd
from joblib import Parallel, delayed

# Define the objective function for benchmarking
def objective(trial, n_params):
    return np.sum([trial.suggest_float(f"{i}", 0, 1) for i in range(n_params)])

# Define benchmarking settings
iterations = [100, 250]
n_params_list = [5, 10, 15, 25]
n_seeds = 30  # Number of repetitions for averaging

# Function to benchmark an optimizer with multiple seeds
def benchmark_optimizer(n_params, iter_count):
    optuna.logging.disable_default_handler()
    row = {"n_params": n_params, "iterations": iter_count}

    marsopt_times = []
    optuna_tpe_times = []
    optuna_cma_times = []

    for seed in range(n_seeds):
        np.random.seed(seed)

        # Mars Optimizer
        start = time.perf_counter()
        study = Study(verbose=False)
        study.optimize(lambda trial: objective(trial, n_params), iter_count)
        marsopt_times.append(time.perf_counter() - start)

        # Optuna TPE
        start = time.perf_counter()
        optuna_study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler())
        optuna_study.optimize(lambda trial: objective(trial, n_params), iter_count)
        optuna_tpe_times.append(time.perf_counter() - start)

        # Optuna CMA-ES
        start = time.perf_counter()
        optuna_study = optuna.create_study(direction="minimize", sampler=optuna.samplers.CmaEsSampler())
        optuna_study.optimize(lambda trial: objective(trial, n_params), iter_count)
        optuna_cma_times.append(time.perf_counter() - start)

    # Store average runtimes
    row["MarsOpt_avg_time"] = np.mean(marsopt_times)
    row["Optuna-TPE_avg_time"] = np.mean(optuna_tpe_times)
    row["Optuna-CMA_avg_time"] = np.mean(optuna_cma_times)

    return row

# Run benchmarking in parallel
results = Parallel(n_jobs=-1)(
    delayed(benchmark_optimizer)(n_params, iter_count)
    for n_params in n_params_list for iter_count in iterations
)

# Convert results to a DataFrame and display
df_results = pd.DataFrame(results)