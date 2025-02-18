import optuna
from marsopt import MARSOpt
import lightgbm as lgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np
import json
from concurrent.futures import ProcessPoolExecutor

optuna.logging.disable_default_handler()

# Generate data
X, y = make_classification(
    n_samples=10000,
    n_features=25,
    n_informative=20,
    n_redundant=5,
    n_classes=2,
    random_state=42,
)

# Split the data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


def objective(trial):
    train_set = lgb.Dataset(X_train, label=y_train)
    valid_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

    param = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "random_state": 42,
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 255),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }

    callbacks = [lgb.callback.early_stopping(5, first_metric_only=True, verbose=False)]

    model = lgb.train(
        param,
        train_set,
        valid_sets=[valid_set],
        num_boost_round=200,
        callbacks=callbacks,
    )

    return model.best_score["valid_0"]["binary_logloss"]


def run_optuna_experiment(seed):
    print("optuna seed", seed)
    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=100)

    return {
        "best_value": study.best_trial.value,
        "trial_history": [trial.value for trial in study.trials],
    }


def run_marsopt_experiment(seed):
    print("mars seed", seed)
    study = MARSOpt(n_init_points=10, random_state=seed)
    study.optimize(objective, n_trials=100)

    return {
        "best_value": study.best_value,
        "trial_history": study.objective_values.tolist(),
    }


if __name__ == "__main__":
    n_runs = 30
    optuna_results = []
    marsopt_results = []

    # Run experiments in parallel
    with ProcessPoolExecutor() as executor:
        # Submit Optuna jobs
        optuna_futures = [
            executor.submit(run_optuna_experiment, i) for i in range(n_runs)
        ]
        # Submit MARSOpt jobs
        marsopt_futures = [
            executor.submit(run_marsopt_experiment, i) for i in range(n_runs)
        ]

        # Collect Optuna results
        for future in optuna_futures:
            result = future.result()
            optuna_results.append(result)

        # Collect MARSOpt results
        for future in marsopt_futures:
            result = future.result()
            marsopt_results.append(result)

    # Calculate statistics for both methods
    def calculate_stats(results):
        values = [r["best_value"] for r in results]
        return {
            "min_loss": float(min(values)),
            "max_loss": float(max(values)),
            "std_loss": float(np.std(values)),
            "avg_loss": float(np.mean(values)),
        }

    # Prepare results dictionary
    results_data = {
        "experiment": "optimization_comparison",
        "optuna": {
            "statistics": calculate_stats(optuna_results),
            "all_runs": [r["trial_history"] for r in optuna_results],
        },
        "marsopt": {
            "statistics": calculate_stats(marsopt_results),
            "all_runs": [r["trial_history"] for r in marsopt_results],
        },
    }

    # Save to JSON file
    with open("results_comparison_full_history2.json", "w") as f:
        json.dump(results_data, f, indent=2)
