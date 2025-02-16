from concurrent.futures import ProcessPoolExecutor
import itertools
import numpy as np
import csv
import optuna
from marsopt import MARSOpt
import os
from bbox_test.bbox import *

optuna.logging.disable_default_handler()


class OptunaProblemWrapper:
    def __init__(self, problem, int_indices=None):
        self.problem = problem
        self.int_indices = int_indices if int_indices else []
        self.bounds = problem.bounds

    def __call__(self, trial):
        params = []
        for i, (lb, ub) in enumerate(self.bounds):
            if i in self.int_indices:
                param = trial.suggest_int(f"x{i}", int(lb), int(ub))
            else:
                param = trial.suggest_float(f"x{i}", lb, ub)
            params.append(param)
        return self.problem.do_evaluate(np.array(params))


def run_single_cma(args):
    test, max_iter, seed = args

    problem = eval(test["name"])(test["dim"])

    if test["res"] is not None:
        problem = Discretizer(problem, test["res"])

    wrapped_problem = OptunaProblemWrapper(
        problem, int_indices=test["int"] if test["int"] is not None else []
    )

    sampler = optuna.samplers.CmaEsSampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    study.optimize(
        wrapped_problem,
        n_trials=max_iter,
    )

    return study.best_value


def run_single_mars(args):
    test, max_iter, seed = args

    problem = eval(test["name"])(test["dim"])

    if test["res"] is not None:
        problem = Discretizer(problem, test["res"])

    wrapped_problem = OptunaProblemWrapper(
        problem, int_indices=test["int"] if test["int"] is not None else []
    )

    study = MARSOpt(
        random_state=seed,
        n_init_points=10,
    )

    study.optimize(
        wrapped_problem,
        n_trials=max_iter,
    )

    return study.best_value


def run_single_optuna(args):
    test, check_points, seed, method = args

    problem = eval(test["name"])(test["dim"])

    if test["res"] is not None:
        problem = Discretizer(problem, test["res"])

    if method == "random":
        sampler = optuna.samplers.RandomSampler(seed=seed)

    else:
        sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def objective(trial):
        params = []
        for i, (lb, ub) in enumerate(problem.bounds):
            if test["int"] is not None and i in test["int"]:
                params.append(trial.suggest_int(str(i), int(lb), int(ub)))
            else:
                params.append(trial.suggest_float(str(i), lb, ub))
        return problem.do_evaluate(np.array(params))

    study.optimize(objective, n_trials=max(check_points))

    # Calculate cumulative minimums
    trial_values = [t.value for t in study.trials]
    cumulative_min = np.minimum.accumulate(trial_values)

    # Get values at checkpoints
    results = {}
    for cp in check_points:
        if cp <= len(cumulative_min):
            results[cp] = cumulative_min[cp - 1]

    return results


def run_optimization(tests, method="mars", output_dir="./results"):
    optuna.logging.disable_default_handler()
    n_seeds = 30
    check_points = [50, 75, 100, 150, 200, 250, 500, 1000]

    os.makedirs(output_dir, exist_ok=True)

    for test in tests:
        if method == "mars" or method == "cma":
            output_save_path = f"{output_dir}/{test['name']}_{test['dim']}.csv"
            if method == "cma":
                run_func = run_single_cma

            else:
                run_func = run_single_mars
            configurations = list(
                itertools.product([test], check_points, range(n_seeds))
            )

        else:
            output_save_path = f"{output_dir}/{test['name']}_{test['dim']}.csv"
            run_func = run_single_optuna
            configurations = list(
                itertools.product(
                    [test],
                    [check_points],  # Pass entire check_points list
                    range(n_seeds),
                    [method],
                )
            )

        os.makedirs(os.path.dirname(output_save_path), exist_ok=True)
        print(f"Running {method} test: {test['name']} dim={test['dim']}")

        with open(output_save_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                ["function", "max_iter", "seed", "min_fx", "max_fx", "avg_fx", "std_fx"]
            )

            with ProcessPoolExecutor() as executor:
                results = list(executor.map(run_func, configurations))

            if method == "mars" or method == "cma":
                # Process MARS results as before
                grouped_results = {}
                for config, result in zip(configurations, results):
                    max_iter = config[1]
                    if max_iter not in grouped_results:
                        grouped_results[max_iter] = []
                    grouped_results[max_iter].append(result)
            else:
                # Process Optuna results from cumulative minimums
                grouped_results = {cp: [] for cp in check_points}
                for result in results:
                    for cp in check_points:
                        if cp in result:
                            grouped_results[cp].append(result[cp])

            for max_iter, results_list in grouped_results.items():
                writer.writerow(
                    [
                        test["name"],
                        max_iter,
                        n_seeds,
                        float(min(results_list)),
                        float(max(results_list)),
                        float(np.mean(results_list)),
                        float(np.std(results_list)),
                    ]
                )


if __name__ == "__main__":
    run_optimization(tests_for_nonparametric, method='mars', output_dir="./results/mars")

    run_optimization(tests_for_nonparametric, method='optuna', output_dir="./results/optuna")

    run_optimization(tests_for_nonparametric, method='random', output_dir="./results/random")

    run_optimization(tests_for_nonparametric, method="cma", output_dir="./results/cma")
