import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

def visualize_optimization_results(results_df: pd.DataFrame, save_path: str = None):
    """
    Visualize optimization results with subplots for different trial counts.
    
    Args:
        results_df (pd.DataFrame): DataFrame containing optimization results
        save_path (str, optional): Path to save the plots. If None, plots will be displayed
    """
    # Set style
    plt.style.use('default')
    sns.set_style("whitegrid")
    colors = sns.color_palette('deep')
    
    # Get unique categories and problems
    unique_categories = results_df['problem_category'].unique()
    
    for category in unique_categories:
        category_data = results_df[results_df['problem_category'] == category]
        unique_problems = category_data['problem_name'].unique()
        
        for problem in unique_problems:
            problem_data = category_data[category_data['problem_name'] == problem]
            unique_trials = sorted(problem_data['trials'].unique())
            n_trials = len(unique_trials)
            
            # Create figure with subplots
            fig = plt.figure(figsize=(15, 5 * n_trials))
            gs = fig.add_gridspec(n_trials, 2, width_ratios=[2, 1])
            
            for idx, trial_count in enumerate(unique_trials):
                trial_data = problem_data[problem_data['trials'] == trial_count]
                unique_methods = trial_data['method'].unique()
                
                # Create plot area
                ax = fig.add_subplot(gs[idx, 0])
                # Create table area
                table_ax = fig.add_subplot(gs[idx, 1])
                table_ax.axis('off')
                
                # Dictionary to store statistical summary
                stats_data = []
                best_mean = float('inf')
                
                # Process each method
                for method_idx, method in enumerate(unique_methods):
                    method_data = trial_data[trial_data['method'] == method]
                    
                    try:
                        all_cum_mins = []
                        final_values = []
                        
                        for history in method_data['history']:
                            history_array = np.array(eval(history))  # Convert string to array
                            if len(history_array) < trial_count:
                                continue
                            cum_min = np.minimum.accumulate(history_array[:trial_count])
                            all_cum_mins.append(cum_min)
                            final_values.append(cum_min[-1])
                        
                        if all_cum_mins:
                            all_cum_mins = np.vstack(all_cum_mins)
                            mean_cum_min = np.mean(all_cum_mins, axis=0)
                            min_cum_min = np.min(all_cum_mins, axis=0)
                            max_cum_min = np.max(all_cum_mins, axis=0)
                            
                            # Plot
                            iterations = np.arange(trial_count)
                            ax.fill_between(iterations, min_cum_min, max_cum_min,
                                          color=colors[method_idx], alpha=0.2)
                            ax.plot(iterations, mean_cum_min, label=method.upper(),
                                  color=colors[method_idx], linewidth=2.5)
                            
                            # Store statistics
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
                
                # Style plot
                ax.grid(True, color='gray', alpha=0.15)
                ax.set_title(f'Trials: {trial_count}', fontsize=14, pad=20)
                ax.set_xlabel('Iteration', fontsize=12)
                ax.set_ylabel('Best Found Value', fontsize=12)
                ax.legend(title='Method', title_fontsize=12, fontsize=10)
                
                # Create performance table
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
            
            # Set overall title
            fig.suptitle(f'{category}\n{problem}', fontsize=16, y=0.98)
            plt.tight_layout()
            
            if save_path:
                filename = f'{category}_{problem}.png'
                full_path = os.path.join(save_path, filename)
                plt.savefig(full_path, dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.show()

# Example usage:
# results_df = pd.read_csv("optimization_results.csv")
# visualize_optimization_results(results_df, save_path=None)