import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def analyze_optimization_results(csv_file):
    """
    Analysis of optimization results using raw average values.
    """
    df = pd.read_csv(csv_file)
    df["history"] = df["history"].apply(lambda x: np.array(eval(x), dtype=np.float64))

    def analyze_by_trials(group):
        histories = group["history"].values
        best_values = [np.round(np.nanmin(hist), decimals=5) for hist in histories]
        return pd.Series(
            {
                "mean_best": np.nanmean(best_values),
                "std_best": np.nanstd(best_values),
                "median_best": np.nanmedian(best_values),
                "min_best": np.nanmin(best_values),
                "max_best": np.nanmax(best_values),
            }
        )

    detailed_analysis = (
        df.groupby(["problem_category", "problem_name", "method", "trials"])
        .apply(analyze_by_trials, include_groups=False)
        .reset_index()
    )

    return detailed_analysis


def plot_heatmaps(detailed_analysis, save_dir="heatmap_plots", plot_type="values"):
    """
    Generate problem-specific heatmaps with different visualization options.
    """
    os.makedirs(save_dir, exist_ok=True)
    trial_counts = sorted(detailed_analysis["trials"].unique())

    # Mavi tonlarında renk skalası - en iyi (1. rank) koyu mavi, en kötü beyaz
    colors = ["#0D47A1", "#1E88E5", "#64B5F6", "#BBDEFB", "#FFFFFF"]

    def create_rank_based_colors(row):
        ranks = pd.Series(row).rank(method="min")
        return [colors[int(r) - 1] for r in ranks]

    for trials in trial_counts:
        trial_data = detailed_analysis[detailed_analysis["trials"] == trials]

        if plot_type == "frequency":
            plt.figure(figsize=(15, 8))

            # Problem ve method için pivot table oluştur
            pivot_data = pd.pivot_table(
                trial_data,
                values="mean_best",
                index="problem_name",
                columns="method",
                aggfunc="mean",
            )

            # Her method için rank frekanslarını hesapla
            method_rank_counts = {}
            for method in pivot_data.columns:
                rank_counts = []
                for idx in pivot_data.index:
                    row = pivot_data.loc[idx]
                    ranks = row.rank(method="min")
                    rank = int(ranks[method])
                    rank_counts.append(rank)
                rank_distribution = pd.Series(rank_counts).value_counts().sort_index()
                method_rank_counts[method] = rank_distribution

            # Bar plot için hazırlık
            methods = list(method_rank_counts.keys())
            bar_positions = np.arange(len(methods))
            bar_width = 0.2
            max_rank = 4
            bar_colors = ["#0D47A1", "#1E88E5", "#64B5F6", "#BBDEFB"]

            # Her rank için bar çiz
            for rank in range(1, max_rank + 1):
                frequencies = [
                    method_rank_counts[method].get(rank, 0) for method in methods
                ]
                bars = plt.bar(
                    bar_positions + (rank - 1) * bar_width,
                    frequencies,
                    bar_width,
                    label=f"Rank {rank}",
                    color=bar_colors[rank - 1],
                )
                
                # Add frequency labels on top of each bar
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    if height > 0:  # Only show label if bar has height
                        plt.text(
                            bar.get_x() + bar.get_width()/2,
                            height,
                            str(int(height)),
                            ha='center',
                            va='bottom',
                            fontsize=10
                        )

            plt.xlabel("Methods")
            plt.ylabel("Number of Problems")
            plt.title(f"Rank Distribution Across Problems - {trials} Trials")
            plt.xticks(bar_positions + bar_width * 1.5, methods, rotation=45)
            plt.legend()
            plt.tight_layout()

        else:
            pivot_data = pd.pivot_table(
                trial_data,
                values="mean_best",
                index="problem_name",
                columns="method",
                aggfunc="mean",
            )

            plt.figure(figsize=(14, len(pivot_data) * 0.6 + 2))

            color_matrix = []
            for idx in pivot_data.index:
                row_values = pivot_data.loc[idx].values
                row_colors = create_rank_based_colors(row_values)
                color_matrix.append(row_colors)

            color_matrix_rgb = np.array(
                [[plt.matplotlib.colors.to_rgb(c) for c in row] for row in color_matrix]
            )
            plt.imshow(color_matrix_rgb, aspect="auto")

            for i in range(len(pivot_data.index)):
                for j, method in enumerate(pivot_data.columns):
                    value = pivot_data.iloc[i, j]
                    if plot_type == "ranks":
                        ranks = pivot_data.iloc[i].rank(method="min")
                        display_value = (
                            f"{int(ranks[method])}" 
                        )
                    else:
                        display_value = f"{value:.5f}"

                    text_color = (
                        "white"
                        if color_matrix[i][j] in ["#0D47A1", "#1E88E5"]
                        else "black"
                    )
                    plt.text(
                        j,
                        i,
                        display_value,
                        ha="center",
                        va="center",
                        color=text_color,
                        fontsize=12,
                    )

            plt.title(
                f"Problem-Specific Performance {'Values' if plot_type == 'values' else 'Rankings'} - {trials} Trials",
                fontsize=14,
            )
            plt.xlabel("Methods", fontsize=12)
            plt.ylabel("Problems", fontsize=12)

            plt.xticks(
                range(len(pivot_data.columns)),
                pivot_data.columns,
                rotation=45,
                ha="right",
                fontsize=11,
            )
            plt.yticks(range(len(pivot_data.index)), pivot_data.index, fontsize=11)
            plt.grid(False)
            plt.tight_layout()

        save_path = os.path.join(save_dir, f"heatmap_{plot_type}_{trials}_trials.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved {plot_type} plot for {trials} trials to: {save_path}")


if __name__ == "__main__":
    detailed_results = analyze_optimization_results("optimization_results2.csv")

    # Mevcut plot'lar
    plot_heatmaps(detailed_results, save_dir="optimization_heatmaps2", plot_type="values")
    plot_heatmaps(detailed_results, save_dir="optimization_heatmaps2", plot_type="ranks")
    plot_heatmaps(detailed_results, save_dir="optimization_heatmaps2", plot_type="frequency")