import json
from  bbox_test.visualize import visualize_optimization_results
import pandas as pd
import pandas as pd
import numpy as np
def prepare_visualization_data(results_dict):
    rows = []
    
    for problem in results_dict.keys():
        problem_data = results_dict[problem]
        
        for seed, methods in problem_data.items():
            # Process Mars results
            mars_configs = {
                100: methods["marsopt_100"]["trial_history"],
                #249: methods["marsopt_250"]["trial_history"],
                #499: methods["marsopt_500"]["trial_history"]
            }
            
            # Adjust trials to match actual history length
            for expected_trials, history in mars_configs.items():
                actual_trials = len(history)
                rows.append({
                    'problem_category': 'hyperparameter',
                    'problem_name': problem,
                    'trials': actual_trials,  # Add 1 to match the length
                    'method': 'mars',
                    'history': str(history)
                })
            
            # Process Optuna results
            optuna_history = methods["optuna"]["trial_history"]
            for trials in [100]:
                if len(optuna_history) >= trials:
                    rows.append({
                        'problem_category': 'hyperparameter',
                        'problem_name': problem,
                        'trials': trials,
                        'method': 'optuna',
                        'history': str(optuna_history[:trials])
                    })
    
    return pd.DataFrame(rows)

# Prepare and visualize the data
with open("hyperparameter_results2.json", "r") as f:
    res = json.load(f)

results_df = prepare_visualization_data(res)
visualize_optimization_results(results_df, "hyperparamete2", detailed=True)