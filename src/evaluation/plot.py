import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def plot_f1_scores(f1_scores, line_values, eval_model_labels, gen_model_labels):
    n_eval_models = len(f1_scores)
    n_gen_models = len(f1_scores[0])

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(n_eval_models)
    bar_width = 0.1

    colors = ["tab:orange", "tab:green", "tab:red", "tab:purple"]

    # Plot bars
    for i in range(n_gen_models):
        x_offsets = x + (i - (n_gen_models - 1) / 2) * bar_width
        y_values = [group[i] for group in f1_scores]

        ax.bar(
            x_offsets,
            y_values,
            width=bar_width,
            label=gen_model_labels[i],
            color=colors[i],
            edgecolor="black",
            linewidth=1,
        )

    # Plot horizontal lines
    for g_idx, val in enumerate(line_values):
        group_center = x[g_idx]
        x_start = group_center - (n_gen_models / 2) * bar_width
        x_end = group_center + (n_gen_models / 2) * bar_width

        ax.hlines(y=val, xmin=x_start, xmax=x_end, color="tab:blue", linestyle="-", linewidth=1.5)

    ax.set_xticks(x)
    ax.set_xticklabels(eval_model_labels)
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0, 0.75)

    # Create a custom legend entry for the horizontal line
    line_element = Line2D(
        [0],
        [0],
        color="tab:blue",
        linestyle="-",
        linewidth=1.5,
    )

    # Get existing handles/labels (from the bars), then add our custom line
    handles, labels = ax.get_legend_handles_labels()
    handles = [line_element] + handles
    labels = ["Original Data"] + labels

    # Use the updated handles and labels for the figure legend
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.532, 0.98), ncol=n_gen_models + 1)

    plt.tight_layout()
    plt.savefig(os.path.join("results", "f1_scores_10eps.png"), dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    f1_scores_1eps = [
        [0.638, 0.102, 0.638, 0.639],
        [0.636, 0.187, 0.641, 0.632],
        [0.609, 0.404, 0.524, 0.447],
        [0.645, 0.151, 0.657, 0.560],
    ]

    f1_scores_5eps = [
        [0.653, 0.102, 0.641, 0.653],
        [0.646, 0.354, 0.646, 0.611],
        [0.640, 0.260, 0.638, 0.496],
        [0.652, 0.250, 0.648, 0.503],
    ]

    f1_scores_10eps = [
        [0.641, 0.107, 0.642, 0.618],
        [0.647, 0.082, 0.637, 0.645],
        [0.620, 0.162, 0.635, 0.573],
        [0.640, 0.603, 0.645, 0.410],
    ]

    f1_scores_training_data = [0.651, 0.647, 0.662, 0.663]
    eval_model_labels = ["Random Forest", "Logistic Regression", "Decision Tree", "Gradient Boost"]
    gen_model_labels = ["PATEGAN", "DPGAN", "PATECTGAN", "DPCTGAN"]

    plot_f1_scores(
        f1_scores=f1_scores_10eps,
        line_values=f1_scores_training_data,
        eval_model_labels=eval_model_labels,
        gen_model_labels=gen_model_labels,
    )
