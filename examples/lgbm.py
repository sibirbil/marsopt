from lightgbm import LGBMRegressor
from marsopt import Study, Trial
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_diabetes

# Load dataset
data = load_diabetes()
X, y = data.data, data.target
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Objective function for hyperparameter tuning
def objective(trial: Trial):
    params = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 20, 200),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
        "verbosisy":-1
    }

    model = LGBMRegressor(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return  mean_squared_error(y_valid, preds)

# Create MarsOpt study
study = Study(random_state=42, n_init_points=10)
study.optimize(objective, n_trials=3)

# Print best trial
best_trial = study.best_trial
print("Best trial:")
print(f"MSE: {best_trial['objective_value']}")
print("  Best parameters:")
for key, value in best_trial["parameters"].items():
    print(f"    {key}: {value}")
