"""
MARS vs Academic-MARS benchmark comparison on BBOB test functions.

Compares two configurations:
  - mars_default:  Study() with all default settings (original MARS)
  - mars_academic:  Study(exploration_mode="epsilon", elite_weighting="log_rank", elite_window=50)

No external optimizers needed — pure marsopt comparison.
"""

from concurrent.futures import ProcessPoolExecutor
import itertools
import numpy as np
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from bbox import (
    tests_for_nonparametric,
    tests_for_auc,
    Discretizer,
)
from marsopt import Study


def make_objective(problem, int_indices):
    """Create a marsopt-compatible objective from a bbox problem."""
    bounds = problem.bounds

    def objective(trial):
        params = []
        for i, (lb, ub) in enumerate(bounds):
            if i in int_indices:
                params.append(trial.suggest_int(f"x{i}", int(lb), int(ub)))
            else:
                params.append(trial.suggest_float(f"x{i}", lb, ub))
        return problem.do_evaluate(np.array(params))

    return objective


def run_single(args):
    """Run a single (test, method, n_trials, seed) configuration."""
    test, method, n_trials, seed = args

    problem = eval(test["name"])(test["dim"])

    if test["res"] is not None:
        problem = Discretizer(problem, test["res"])

    int_indices = test["int"] if test["int"] is not None else []
    objective = make_objective(problem, int_indices)

    if method == "mars_default":
        study = Study(random_state=seed, verbose=False)
    elif method == "mars_academic":
        study = Study(
            random_state=seed,
            verbose=False,
            exploration_mode="epsilon",
            epsilon=1.0,
            elite_weighting="log_rank",
            elite_window=50,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    study.optimize(objective, n_trials=n_trials)
    return study.objective_values.tolist()


def run_benchmark(tests, methods, output_file, category):
    """Run all (test × method × checkpoint × seed) combinations."""
    n_seeds = 30
    check_points = [50, 75, 100, 150, 250, 500, 1000]

    with open(output_file, mode="a", newline="") as file:
        writer = csv.writer(file)

        for test in tests:
            for method in methods:
                print(
                    f"[{category}] {method} | {test['name']} "
                    f"dim={test['dim']} int={test['int']} res={test['res']}"
                )

                configs = list(
                    itertools.product(
                        [test], [method], check_points, range(n_seeds)
                    )
                )

                with ProcessPoolExecutor() as executor:
                    results = list(executor.map(run_single, configs))

                for (t, m, cp, seed), history in zip(configs, results):
                    writer.writerow(
                        [
                            category,
                            f"{t['name']}_{t['dim']}_{t['res']}_{t['int']}",
                            m,
                            cp,
                            seed,
                            history,
                        ]
                    )


if __name__ == "__main__":
    output_file = "optimization_results.csv"

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["problem_category", "problem_name", "method", "trials", "seed", "history"]
        )

    methods = ["mars_default", "mars_academic"]

    run_benchmark(tests_for_nonparametric, methods, output_file, "nonparametric")
    run_benchmark(tests_for_auc, methods, output_file, "auc")

    print(f"\nDone. Results saved to {output_file}")
