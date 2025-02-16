import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# Updated color map and legend labels for the new comparison
COLOR_MAP = {
    'Trial': '#1f77b4',
    'Optuna': '#ff7f0e',
    'CMA': '#2ca02c',
    'Random': '#9467bd'
}

LEGEND_LABELS = {
    'Trial': 'Trial',
    'Optuna': 'Optuna (TPE)',
    'CMA': 'CMA-ES',
    'Random': 'Random'
}

plt.rcParams.update({
    'mathtext.default': 'regular',
    'font.size': 12
})

os.makedirs("./results/comparison", exist_ok=True)

for file in os.listdir("./results/mars"):
    if file.endswith(".csv"):
        problem_name = os.path.splitext(file)[0]
        
        # Load all data sources without checkpoint limitation
        trial_df = pd.read_csv(f"./results/mars/{file}")
        trial_df = trial_df.assign(param_group='Trial')
        
        optuna_df = pd.read_csv(f"./results/optuna/{file}")
        optuna_df = optuna_df.assign(param_group='Optuna')
        
        cma_df = pd.read_csv(f"./results/cma/{file}")
        # Rename max_iter to max_iter for consistency
        cma_df = cma_df.assign(param_group='CMA')
        
        random_df = pd.read_csv(f"./results/random/{file}")
        random_df = random_df.assign(param_group='Random')
        
        # Combine all data
        combined_df = pd.concat([
            trial_df,
            optuna_df,
            cma_df,
            random_df
        ], ignore_index=True)

        # Plotting setup
        plt.figure(figsize=(25, 13))
        width = 0.18
        spacing = 1.5
        
        max_iters = sorted(combined_df['max_iter'].unique())
        x_base = np.arange(len(max_iters)) * spacing
        
        ordered_groups = ['Trial', 'Optuna', 'CMA', 'Random']

        # Bar plots
        for cp_idx, max_iter in enumerate(max_iters):
            groups_data = combined_df[combined_df['max_iter'] == max_iter]
            
            for group_idx, group_name in enumerate(ordered_groups):
                group_data = groups_data[groups_data['param_group'] == group_name]
                if group_data.empty:
                    continue

                x = x_base[cp_idx] + (group_idx - len(ordered_groups)/2 + 0.5) * width
                
                avg = group_data['avg_fx'].values[0]
                std = group_data['std_fx'].values[0]
                min_val = group_data['min_fx'].values[0]
                max_val = group_data['max_fx'].values[0]

                plt.bar(
                    x, avg, width,
                    color=COLOR_MAP[group_name],
                    alpha=0.9,
                    edgecolor='black'
                )
                
                plt.errorbar(
                    x, avg, yerr=std,
                    fmt='none', ecolor='black',
                    capsize=5, elinewidth=1
                )
                
                plt.plot(
                    [x - width/2, x + width/2], [min_val]*2,
                    color=COLOR_MAP[group_name], lw=2, alpha=0.7
                )
                plt.plot(
                    [x - width/2, x + width/2], [max_val]*2,
                    color=COLOR_MAP[group_name], lw=2, alpha=0.7
                )

        # X-axis formatting with rotated labels for better readability
        plt.xticks(x_base, max_iters, rotation=45)
        plt.xlim(x_base[0]-spacing*0.5, x_base[-1]+spacing*0.5)
        plt.xlabel('Iterations', fontsize=14, labelpad=15)  # Changed from 'Check Points' to 'Iterations'

        pivot_table = combined_df.pivot(
            index='max_iter',
            columns='param_group',
            values='avg_fx'
        ).sort_index()
        pivot_table = pivot_table[ordered_groups]  # Sütunları ordered_groups sırasına göre düzenle

        # Sonra tablo oluşturma aynı şekilde devam eder
        cell_text = []
        for cp in max_iters:
            row = pivot_table.loc[cp]
            min_val = row.min()
            formatted_row = []
            for val in row:
                formatted_row.append(f'$\\bf{{{val:.5f}}}$' if val == min_val else f'{val:.5f}')
            cell_text.append(formatted_row)

        # Adjust table position and size based on number of checkpoints
        table_height = min(0.6, 0.8 * (10 / len(max_iters)))
        table = plt.table(
            cellText=cell_text,
            rowLabels=[f'Iter {cp}' for cp in max_iters],  # Changed from 'Check Point' to 'Iter'
            colLabels=[LEGEND_LABELS[g] for g in ordered_groups],
            cellLoc='center',
            loc='center right',
            bbox=[1.05, 0.2, 0.3, table_height]
        )
        table.set_fontsize(11)
        table.scale(1, 1.3)

        # Other formatting
        plt.title(f"{problem_name}\nPerformance Comparison", fontsize=16, pad=20)
        plt.ylabel('Objective Value (Lower is Better)', fontsize=14)
        plt.grid(axis='y', alpha=0.3)
        
        legend_elements = [Line2D([0], [0], color=COLOR_MAP[g], lw=4, label=LEGEND_LABELS[g]) 
                         for g in ordered_groups]
        plt.legend(
            handles=legend_elements,
            bbox_to_anchor=(1.05, 1.02),
            loc='upper left',
            title='Optimization Methods',
            fontsize=12
        )

        plt.tight_layout()
        plt.subplots_adjust(right=0.72)
        plt.savefig(f"./results/comparison/{problem_name}_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()