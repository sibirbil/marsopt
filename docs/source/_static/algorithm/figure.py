import numpy as np
import matplotlib.pyplot as plt
import os

# Function to calculate adaptive parameters
def calculate_adaptive_params(n_trials, initial_noise=0.2, final_noise=None):
    if final_noise is None:
        final_noise = 2.0 / n_trials

    n_elites = np.zeros(n_trials)
    noise_levels = np.zeros(n_trials)
    cat_temps = np.zeros(n_trials)

    elite_scale = 2.0 * np.sqrt(n_trials)

    for iteration in range(n_trials):
        progress = (iteration + 1)  / n_trials
        n_elites[iteration] = max(1, round(elite_scale * progress * (1 - progress)))
        cos_anneal = (1 + np.cos(np.pi * progress)) * 0.5
        noise_levels[iteration] = final_noise + (initial_noise - final_noise) * cos_anneal
        cat_temps[iteration] = (0.1 + 0.9 * cos_anneal)

    return n_elites, noise_levels, cat_temps

# Calculate values for 100 iterations
n_trials = 100
n_elites_100, noise_100, cat_temps_100 = calculate_adaptive_params(n_trials)

# Directory to save plots
save_dir = os.getcwd()

# Filenames for the plots
filenames = [
    os.path.join(save_dir, "number_of_elites_100.png"),
    os.path.join(save_dir, "noise_level_100.png"),
    os.path.join(save_dir, "categorical_temperature_100.png")
]

# Function to create and save plots
def create_and_save_plot(x, y, title, ylabel, color, filename):
    fig, ax = plt.subplots()
    ax.plot(x, y, color=color, linewidth=2.5)

    # Styling
    ax.set_title(title, pad=20, fontsize=16, fontweight='bold')
    ax.set_xlabel('Trial', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    # Grid and limits
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlim(1, n_trials)  # Changed from (0, n_trials) to (1, n_trials)

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin - (ymax - ymin) * 0.1, ymax + (ymax - ymin) * 0.1)

    plt.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)  # Close the figure to free memory

# Creating and saving the plots with proper LaTeX notation in titles
# Changed range(1, n_trials) to range(1, n_trials+1) to include 100
create_and_save_plot(range(1, n_trials+1), n_elites_100, r'$n_{\mathrm{elite}}(t)$', 'Number of Elites', 'blue', filenames[0])
create_and_save_plot(range(1, n_trials+1), noise_100, r'$\eta(t)$', 'Noise Level', 'green', filenames[1])
create_and_save_plot(range(1, n_trials+1), cat_temps_100, r'$T_{\mathrm{cat}}(t)$', 'Categorical Temperature', 'red', filenames[2])