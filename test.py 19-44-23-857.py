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
    
    study.optimize(wrapped_problem, n_trials=max_iter)
    history = [t.value for t in study.trials]
    return history

def run_single_mars(args):
    test, max_iter, seed = args
    problem = eval(test["name"])(test["dim"])
    
    if test["res"] is not None:
        problem = Discretizer(problem, test["res"])
        
    wrapped_problem = OptunaProblemWrapper(
        problem, int_indices=test["int"] if test["int"] is not None else []
    )
    
    study = MARSOpt(random_state=seed, n_init_points=10)
    study.optimize(wrapped_problem, n_trials=max_iter)
    return study.objective_values.tolist()

def run_single_optuna(args):
    test, max_trials, seed, method = args
    problem = eval(test["name"])(test["dim"])
    max_iter = max(max_trials)  # max_trials is now the check_points list
    
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

    # Optimize for maximum trials needed
    study.optimize(objective, n_trials=max_iter)
    
    # Get full history
    history = [t.value for t in study.trials]
    
    # Create a dictionary to store histories for each checkpoint
    checkpoint_histories = {}
    for cp in max_trials:  # Using max_trials (check_points) passed in args
        if cp <= len(history):
            checkpoint_histories[cp] = history[:cp]
            
    return checkpoint_histories

def run_optimization(tests, method, output_file, problem_category):
    optuna.logging.disable_default_handler()
    n_seeds = 30
    check_points = [50, 75, 100, 150, 250, 500, 1000]

    with open(output_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # Write header if file is empty
        if os.path.getsize(output_file) == 0:
            writer.writerow(['problem_category', 'problem_name', 'method', 'trials', 'seed', 'history'])

        for test in tests:
            print(f"Running {method} test: {test['name']} dim={test['dim']}")

            if method == "mars":
                run_func = run_single_mars
                configurations = list(itertools.product([test], check_points, range(n_seeds)))
            elif method == "cma":
                run_func = run_single_cma
                configurations = list(itertools.product([test], check_points, range(n_seeds)))
            else:  # optuna or random
                run_func = run_single_optuna
                configurations = list(itertools.product([test], [check_points], range(n_seeds), [method]))

            with ProcessPoolExecutor() as executor:
                results = list(executor.map(run_func, configurations))

            # Write results
            for idx, (config, result) in enumerate(zip(configurations, results)):
                if method in ["mars", "cma"]:
                    max_iter = config[1]
                    seed = config[2]
                    writer.writerow([
                        problem_category,
                        f"{test['name']}_{test['dim']}_{test['res']}_{test['int']}",
                        method,
                        max_iter,
                        seed,
                        result
                    ])
                else:  # optuna or random
                    seed = config[2]
                    # Write a row for each checkpoint
                    for max_iter, history in result.items():
                        writer.writerow([
                            problem_category,
                            f"{test['name']}_{test['dim']}_{test['res']}_{test['int']}",
                            method,
                            max_iter,
                            seed,
                            history
                        ])

if __name__ == "__main__":
    output_file = "optimization_results.csv"
    
    # Create new file with header
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['problem_category', 'problem_name', 'method', 'trials', 'seed', 'history'])
    
    # Run optimizations for each category and method
    methods = ["mars", "optuna", "random", "cma"]
    
    for method in methods:
        run_optimization(tests_for_nonparametric, method, output_file, "tests_for_nonparametric")
        run_optimization(tests_for_auc, method, output_file, "tests_for_auc")