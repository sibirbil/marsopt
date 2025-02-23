import optuna
from marsopt import Study
import lightgbm as lgb
from sklearn.datasets import make_classification, fetch_california_housing
from sklearn.model_selection import train_test_split
import json
import joblib
from joblib import Parallel, delayed


# 📌 Veri setlerini oluştur
def generate_data(task_type, n_samples=10000):
    if task_type == "classification":
        X, y = make_classification(
            n_samples=n_samples,
            n_features=25,
            n_informative=20,
            n_redundant=5,
            n_classes=2,
            random_state=42,
        )
    elif task_type == "regression":
        data = fetch_california_housing()
        X, y = data.data, data.target

    return train_test_split(X, y, test_size=0.2, random_state=42)


# 📌 Basit objective (GBDT, az parametre aralığı)
def objective_simple(trial, task_type, X_train, X_val, y_train, y_val):
    train_set = lgb.Dataset(X_train, label=y_train)
    valid_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

    param = {
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

    if task_type == "classification":
        param.update({"objective": "binary", "metric": "binary_logloss"})
        metric = "binary_logloss"
    else:
        param.update({"objective": "regression", "metric": "l2"})
        metric = "l2"

    callbacks = [lgb.callback.early_stopping(10, first_metric_only=True, verbose=False)]

    model = lgb.train(
        param,
        train_set,
        valid_sets=[valid_set],
        num_boost_round=200,
        callbacks=callbacks,
    )

    return model.best_score["valid_0"][metric]


# 📌 Karmaşık objective (GBDT/GOSS, geniş parametre aralığı)
def objective_complex(trial, task_type, X_train, X_val, y_train, y_val):
    train_set = lgb.Dataset(X_train, label=y_train)
    valid_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

    param = {
        "verbosity": -1,
        "random_state": 42,
    }

    if task_type == "classification":
        param.update({"objective": "binary", "metric": "binary_logloss"})
        metric = "binary_logloss"
    else:
        param.update({"objective": "regression", "metric": "l2"})
        metric = "l2"

    # Boosting tipi seçimi (GBDT / GOSS)
    boosting_type = trial.suggest_categorical("boosting_type", ["gbdt", "goss"])
    param["boosting_type"] = boosting_type

    if boosting_type == "goss":
        param.update(
            {
                "top_rate": trial.suggest_float("top_rate", 0.1, 0.3),
                "other_rate": trial.suggest_float("other_rate", 0.1, 0.3),
            }
        )
    else:
        param.update(
            {
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            }
        )

    param.update(
        {
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 2, 255),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "max_bin": trial.suggest_int("max_bin", 200, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        }
    )

    callbacks = [lgb.callback.early_stopping(10, first_metric_only=True, verbose=False)]

    model = lgb.train(
        param,
        train_set,
        valid_sets=[valid_set],
        num_boost_round=200,
        callbacks=callbacks,
    )

    return model.best_score["valid_0"][metric]


# 📌 Optuna & MARSOpt çalıştırma fonksiyonları
def run_experiment(
    optimizer, seed, task_type, objective_func, n_trials, X_train, X_val, y_train, y_val
):
    print(f"Running {optimizer} with seed {seed} on {task_type}")

    if optimizer == "optuna":
        optuna.logging.disable_default_handler()
        study = optuna.create_study(
            direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)
        )
        study.optimize(
            lambda trial: objective_func(
                trial, task_type, X_train, X_val, y_train, y_val
            ),
            n_trials=n_trials,
        )
        return {
            "best_value": study.best_trial.value,
            "trial_history": [trial.value for trial in study.trials],
        }

    elif optimizer == "marsopt":
        study = Study(random_state=seed, verbose=False)
        study.optimize(
            lambda trial: objective_func(
                trial, task_type, X_train, X_val, y_train, y_val
            ),
            n_trials=n_trials,
        )
        return {
            "best_value": study.best_trial["objective_value"],
            "trial_history": study.objective_values.tolist(),
        }


if __name__ == "__main__":
    optuna.logging.disable_default_handler()
    sample_sizes = [25000]
    task_types = ["regression"]
    n_trials_list = [
        100,
        #250,
        #500,
    ]  # MarsOpt için kullanılmayan liste artık kullanılacak
    seeds = list(range(30))

    results = {}

    def run_single_experiment(task, n_samples, obj_func, seed, n_trials_list):
        """Tek bir deneyi çalıştırır ve sonucunu döndürür."""
        X_train, X_val, y_train, y_val = generate_data(task, n_samples)

        obj_name = obj_func.__name__

        if task == "regression":
            key = f"california_housing_{obj_name}"
        else:
            key = f"{task}_{n_samples}_{obj_name}"

        optuna_result = run_experiment(
            "optuna",
            seed,
            task,
            obj_func,
            max(n_trials_list),
            X_train,
            X_val,
            y_train,
            y_val,
        )

        marsopt_results = {
            f"marsopt_{n_trials}": run_experiment(
                "marsopt",
                seed,
                task,
                obj_func,
                n_trials,
                X_train,
                X_val,
                y_train,
                y_val,
            )
            for n_trials in n_trials_list
        }

        return key, seed, {"optuna": optuna_result, **marsopt_results}

    # Tüm deneyleri işleme sokacak fonksiyon
    all_experiments = [
        (
            task,
            n_samples if task == "classification" else sample_sizes[0],
            obj_func,
            seed,
            n_trials_list,
        )
        for task in task_types
        for n_samples in sample_sizes
        for obj_func in [objective_complex]
        for seed in seeds
        if task == "classification"
        or n_samples == sample_sizes[0]  # Regresyon için sadece ilk sample size'ı al
    ]

    # Paralel işlem
    num_jobs = joblib.cpu_count() - 1  # Maksimum CPU kullanımı için
    experiment_results = Parallel(n_jobs=num_jobs)(
        delayed(run_single_experiment)(task, n_samples, obj_func, seed, n_trials)
        for task, n_samples, obj_func, seed, n_trials in all_experiments
    )

    # Sonuçları birleştir
    for key, seed, result in experiment_results:
        if key not in results:
            results[key] = {}
        results[key][seed] = result

    # Sonuçları JSON dosyasına kaydet
    with open("hyperparameter_results2.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Experiment results saved to results.json")
