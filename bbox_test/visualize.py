import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

def visualize_optimization_results(results_df: pd.DataFrame, save_path: str = None, detailed: bool = False):
    """
    Visualize optimization results with subplots for different trial counts.
    Shows mean with ±1 standard deviation bands.
    
    Args:
        results_df (pd.DataFrame): DataFrame containing optimization results
        save_path (str, optional): Path to save the plots. If None, plots will be displayed
        detailed (bool, optional): If True, adds detailed final iterations view and adjusts y-axis
    """
    plt.style.use('default')
    sns.set_style("whitegrid")
    colors = sns.color_palette('deep')
    
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
    
    unique_categories = results_df['problem_category'].unique()
    
    for category in unique_categories:
        category_data = results_df[results_df['problem_category'] == category]
        unique_problems = category_data['problem_name'].unique()
        
        for problem in unique_problems:
            problem_data = category_data[category_data['problem_name'] == problem]
            unique_trials = sorted(problem_data['trials'].unique())
            n_trials = len(unique_trials)
            
            # Create single figure for all trials of the same problem
            if detailed:
                fig = plt.figure(figsize=(15, 10 * n_trials))  # Height scales with number of trials
                gs = fig.add_gridspec(n_trials * 3, 2, height_ratios=[2, 1, 1] * n_trials, width_ratios=[2, 1])
            else:
                fig = plt.figure(figsize=(15, 5 * n_trials))
                gs = fig.add_gridspec(n_trials, 2, width_ratios=[2, 1])
            
            for idx, trial_count in enumerate(unique_trials):
                trial_data = problem_data[problem_data['trials'] == trial_count]
                unique_methods = trial_data['method'].unique()
                
                if detailed:
                    # Main plot at 3*idx position
                    ax = fig.add_subplot(gs[3*idx:3*idx+1, 0])
                    table_ax = fig.add_subplot(gs[3*idx:3*idx+1, 1])
                    # Detail plot spans the next two rows
                    detail_ax = fig.add_subplot(gs[3*idx+1:3*idx+3, :])
                else:
                    ax = fig.add_subplot(gs[idx, 0])
                    table_ax = fig.add_subplot(gs[idx, 1])
                
                table_ax.axis('off')
                
                stats_data = []
                best_mean = float('inf')
                all_method_data = {}
                
                global_min = float('inf')
                global_max = float('-inf')
                
                # Process each method
                for method_idx, method in enumerate(unique_methods):
                    method_data = trial_data[trial_data['method'] == method]
                    
                    try:
                        all_cum_mins = []
                        final_values = []
                        
                        for history in method_data['history']:
                            history_array = np.array(eval(history))
                            if len(history_array) < trial_count:
                                continue
                            cum_min = np.minimum.accumulate(history_array[:trial_count])
                            all_cum_mins.append(cum_min)
                            final_values.append(cum_min[-1])
                        
                        if all_cum_mins:
                            all_cum_mins = np.vstack(all_cum_mins)
                            mean_cum_min = np.mean(all_cum_mins, axis=0)
                            std_cum_min = np.std(all_cum_mins, axis=0)
                            
                            global_min = min(global_min, np.min(mean_cum_min - std_cum_min))
                            global_max = max(global_max, np.max(mean_cum_min + std_cum_min))
                            
                            all_method_data[method] = {
                                'mean': mean_cum_min,
                                'std': std_cum_min,
                                'final_values': final_values
                            }
                            
                            iterations = np.arange(trial_count)
                            ax.fill_between(iterations, 
                                          mean_cum_min - std_cum_min,
                                          mean_cum_min + std_cum_min,
                                          color=colors[method_idx], 
                                          alpha=0.2)
                            ax.plot(iterations, mean_cum_min, 
                                  label=method.upper(),
                                  color=colors[method_idx], 
                                  linewidth=2.5)
                            
                            if detailed:
                                last_n = min(50, trial_count)
                                detail_ax.fill_between(iterations[-last_n:], 
                                                     mean_cum_min[-last_n:] - std_cum_min[-last_n:],
                                                     mean_cum_min[-last_n:] + std_cum_min[-last_n:],
                                                     color=colors[method_idx], 
                                                     alpha=0.2)
                                detail_ax.plot(iterations[-last_n:], mean_cum_min[-last_n:],
                                             label=method.upper(),
                                             color=colors[method_idx],
                                             linewidth=2.5)
                            
                            mean_val = np.mean(final_values)
                            stats_data.append({
                                'Method': method.upper(),
                                'Mean': f"{mean_val:.4f}",
                                'Std': f"±{np.std(final_values):.4f}",
                                'Min': f"{np.min(final_values):.4f}",
                                'Max': f"{np.max(final_values):.4f}",
                                '_mean_val': mean_val
                            })
                            best_mean = min(best_mean, mean_val)
                            
                    except Exception as e:
                        print(f"Error processing {method}: {str(e)}")
                        continue
                
                y_margin = (global_max - global_min) * 0.05
                y_min = global_min - y_margin
                y_max = global_max + y_margin
                
                ax.set_ylim(y_min, y_max)
                
                ax.grid(True, color='gray', alpha=0.15)
                ax.set_title(f'Trials: {trial_count}', fontsize=14, pad=20)
                ax.set_xlabel('Iteration', fontsize=12)
                ax.set_ylabel('Best Found Value', fontsize=12)
                ax.legend(title='Method', title_fontsize=12, fontsize=10)
                
                if detailed:
                    final_min = float('inf')
                    final_max = float('-inf')
                    last_n = min(50, trial_count)
                    
                    for method_info in all_method_data.values():
                        final_min = min(final_min, np.min(method_info['mean'][-last_n:] - method_info['std'][-last_n:]))
                        final_max = max(final_max, np.max(method_info['mean'][-last_n:] + method_info['std'][-last_n:]))
                    
                    detail_margin = (final_max - final_min) * 0.1
                    detail_ax.set_ylim(final_min - detail_margin, final_max + detail_margin)
                    
                    detail_ax.grid(True, color='gray', alpha=0.15)
                    detail_ax.set_title(f'Final Iterations Detail (Trials: {trial_count})', fontsize=14, pad=20)
                    detail_ax.set_xlabel('Iteration', fontsize=12)
                    detail_ax.set_ylabel('Best Found Value', fontsize=12)
                    detail_ax.legend(title='Method', title_fontsize=12, fontsize=10)
                
                stats_data.sort(key=lambda x: x['_mean_val'])
                table_data = []
                cell_colors = []
                
                for row in stats_data:
                    is_best = abs(row['_mean_val'] - best_mean) < 1e-10
                    row_data = [row['Method'], row['Mean'], row['Std'], row['Min'], row['Max']]
                    row_colors = ['#e6ffe6' if is_best else 'white'] * 5
                    table_data.append(row_data)
                    cell_colors.append(row_colors)
                
                table = table_ax.table(cellText=table_data,
                                     colLabels=['METHOD', 'MEAN', 'STD', 'MIN', 'MAX'],
                                     cellLoc='center',
                                     loc='center',
                                     cellColours=cell_colors)
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1.2, 1.5)
            
            fig.suptitle(f'{category}\n{problem}', fontsize=16, y=0.98)
            plt.tight_layout()
            
            if save_path:
                filename = f'{category}_{problem}.png'
                full_path = os.path.join(save_path, filename)
                plt.savefig(full_path, dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.show()

if __name__ == "__main__":
    results_df = pd.read_csv("optimization_results2.csv")
    visualize_optimization_results(results_df, save_path="optimization_plots", detailed=False)