from scipy.stats import linregress, ttest_ind
import matplotlib.pyplot as plt
import numpy as np
import os


metric_name_dict = {
    "div_monotonicity": "KL-divergence monotonicity",
    "local_minima": "Local Minima",
    "divergence": "Final KL-divergence",
    "energy": "Energy Value",
    "state_counts": "State Visitation",
    "gradient_mags": "Gradient Magnitude",
}


def get_plot_format_given_name(exp_name, give_linestyle=True):
    strength_to_alpha = {
        "zero": 0.8,
        "min": 0.6,
        "light": 0.7,
        "medium": 0.8,
        "heavy": 0.9,
        "max": 1.0,
    }
    strength_to_linestyle_pure = {
        "zero": "solid",
        "min": (0, (5, 12)),
        "light": (0, (5, 9)),
        "medium": (0, (5, 6)),
        "heavy": (0, (5, 3)),
        "max": (0, (5, 0)),
    }
    strength_to_linestyle_mixed = {
        "zero_zero": "solid",
        "zero_light": "solid",
        "zero_medium": "solid",
        "light_zero": "solid",
        "medium_zero": "solid",
        "min_min": (0, (5, 15)),
        "light_light": (0, (5, 9)),
        "medium_medium": (0, (5, 6)),
        "max_medium": "dashdot",
        "medium_max": "dashdot",
        "heavy_heavy": (0, (5, 3)),
        "max_heavy": "dotted",
        "heavy_max": "dotted",
        "zero_max": (0, (5, 0)),
        "max_zero": (0, (5, 0)),
        "max_max": (0, (5, 0)),
    }
    dose_list = ["zero", "min", "light", "medium", "heavy", "max"]
    color_baseline = "tab:gray"
    colors_1a = [
        "#2293b3",
        "#4ea9c2",
        "#64b3c9",
        "#7abed1",
        "#a7d4e1",
        "#bddee8",
    ]
    colors_1a.reverse()
    colors_2a = [
        "#df212b",
        "#e54d55",
        "#e8636a",
        "#ec7a80",
        "#f2a6aa",
        "#f5bcbf",
    ]
    colors_2a.reverse()
    colors_mixed = [
        "#5d41d1",
        "#7d67da",
        "#8d7ade",
        "#9e8de3",
        "#beb3ed",
        "#cec6f1",
    ]
    colors_mixed.reverse()

    exp_name_elems = exp_name.split("_")
    if len(exp_name_elems) == 4:
        if exp_name_elems[1] == "zero" and exp_name_elems[3] == "zero":
            color = color_baseline
        elif exp_name_elems[1] == "zero":
            dose_num = dose_list.index(exp_name_elems[3])
            color = colors_2a[dose_num]
        elif exp_name_elems[3] == "zero":
            dose_num = dose_list.index(exp_name_elems[1])
            color = colors_1a[dose_num]
        else:
            dose_num_1a = dose_list.index(exp_name_elems[1])
            dose_num_2a = dose_list.index(exp_name_elems[3])
            avg_dose_num = int((dose_num_1a + dose_num_2a) / 2)
            color = colors_mixed[avg_dose_num]
        alpha = np.mean(
            [
                strength_to_alpha[exp_name_elems[1]],
                strength_to_alpha[exp_name_elems[3]],
            ]
        )
        if give_linestyle:
            linestyle = strength_to_linestyle_mixed[
                f"{exp_name_elems[1]}_{exp_name_elems[3]}"
            ]
        else:
            linestyle = "solid"
    elif len(exp_name_elems) == 2:
        if exp_name_elems[0] == "2a":
            dose_num = dose_list.index(exp_name_elems[1])
            color = colors_2a[dose_num]
        elif exp_name_elems[0] == "1a":
            dose_num = dose_list.index(exp_name_elems[1])
            color = colors_1a[dose_num]
        else:
            str_1a = float(exp_name_elems[0]) * 2.0
            str_2a = float(exp_name_elems[1]) * 2.0
            # create new mixed color (RBG) based on
            # relative strength of 1a (blue) and 2a (red)
            str_1a_rgb = np.array([0, 0, str_1a])
            str_2a_rgb = np.array([str_2a, 0, 0])
            color = str_1a_rgb + str_2a_rgb
            alpha = np.mean([str_1a, str_2a])
            if give_linestyle:
                linestyle = strength_to_linestyle_mixed[
                    f"{exp_name_elems[1]}_{exp_name_elems[3]}"
                ]
            else:
                linestyle = "solid"
            return color, alpha, linestyle
        if exp_name_elems[1] == "zero":
            color = color_baseline
        alpha = strength_to_alpha[exp_name_elems[1]]
        if give_linestyle:
            linestyle = strength_to_linestyle_pure[exp_name_elems[1]]
        else:
            linestyle = "solid"
    else:
        raise ValueError("Unrecognized condition")
    return color, alpha, linestyle


def get_label_given_name(exp_name):
    exp_name_elems = exp_name.split("_")
    if len(exp_name_elems) == 4:
        if exp_name_elems[1] == "zero" and exp_name_elems[3] == "zero":
            return "zero"
        elif exp_name_elems[1] == "zero":
            return f"2a_{exp_name_elems[3]}"
        elif exp_name_elems[3] == "zero":
            return f"1a_{exp_name_elems[1]}"
        elif exp_name_elems[1] == exp_name_elems[3]:
            return f"mix_{exp_name_elems[1]}"
        else:
            return f"1a_{exp_name_elems[1]}\n2a_{exp_name_elems[3]}"
    elif len(exp_name_elems) == 2:
        return exp_name


def plot_metrics(
    metrics_dict,
    plt_name="all",
    minimal_timeseries=True,
    format="png",
    render_titles=True,
    exp_dict=None,
):
    # Combine plots into a single figure with subplots
    if minimal_timeseries:
        fig, axes = plt.subplots(1, 5, figsize=(12, 2.5))
    else:
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        fig.suptitle("Simulation Metrics", fontsize=24)
    # use helvetica font
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Helvetica"]

    # Create an empty list to store handles and labels for the legend
    handles, labels = [], []

    for exp_name, metrics in metrics_dict.items():
        color, alpha, linestyle = get_plot_format_given_name(exp_name)
        metrics, std_errors = metrics
        metrics_data = [
            (metrics["energy"], std_errors["energy"], "Energy Value"),
            (
                metrics["gradient_mags"],
                std_errors["gradient_mags"],
                "Gradient Magnitude",
            ),
            (metrics["local_minima"], std_errors["local_minima"], "Local Minima Count"),
            (metrics["state_counts"], std_errors["state_counts"], "State Count"),
            (metrics["divergence"], std_errors["divergence"], "Divergence"),
        ]
        if not minimal_timeseries:
            metrics_data.extend(
                [
                    (metrics["levels"], std_errors["levels"], "Drug Level"),
                    (metrics["entropy"], std_errors["entropy"], "Entropy"),
                    (
                        metrics["target_energy"],
                        std_errors["target_energy"],
                        "Target Energy",
                    ),
                ]
            )
        exp_name_elems = exp_name.split("_")
        if len(exp_name_elems) == 4:
            use_name = get_label_given_name(exp_name)
        else:
            use_name = exp_name
        for ax, (vals, std_err, title) in zip(axes.ravel(), metrics_data):
            (line,) = ax.plot(
                vals,
                label=use_name,
                alpha=alpha,
                linestyle=linestyle,
                color=color,
            )  # unpack the line object from ax.plot()
            ax.fill_between(
                range(len(vals)),
                vals - std_err,
                vals + std_err,
                alpha=(alpha * 0.2),
                color=color,
                linewidth=0.1,
            )
            # draw horizontal line at initial value
            ax.axhline(vals[0], color="gray", linestyle="--", linewidth=0.7)
            # get drug effect duration
            if exp_dict is not None:
                drug_steps = exp_dict[exp_name].drug_steps
            # draw vertical lines for drug application
            ax.axvline(drug_steps, color="gray", linestyle=(0, (1, 5)), linewidth=1.0)
            ax.set_xlabel("Steps (T)", fontsize=12)
            if render_titles:
                ax.set_title(title, fontsize=14)
            # remove upper and right spines
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # Collect handles and labels for the legend
        handles.append(line)
        labels.append(use_name)

    # Set legend to the right of the subplots
    bbox_offset = -0.2 if len(labels) > 7 else -0.15
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, bbox_offset),
        fontsize=13,
        title_fontsize=14,
        ncols=8,
        frameon=False,
    )

    figtext = plt_name.split("_")
    figtext = " ".join(figtext)
    fig.text(
        x=0,
        y=0.5,
        s=figtext,
        fontsize=14,
        rotation=90,
        horizontalalignment="center",
        verticalalignment="center",
        fontweight="bold",
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.87, left=0.045)  # Adjust title spacing

    # check if output directory exists and create if not
    if not os.path.exists("./output/figures/metrics/timeseries/"):
        os.makedirs("./output/figures/metrics/timeseries/")
    fig.savefig(
        f"./output/figures/metrics/timeseries/timeseries_{plt_name}.{format}",
        format=f"{format}",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def plot_bar(metrics_dict, keys, titles, plt_name="all", format="png"):
    # We use the first metric to sort all the metrics so that the
    # results appear in the same order on each bar chart.
    y_ref = []
    x_ref = []
    for i, (title, key) in enumerate(zip(titles, keys)):
        x = []
        y = []
        y_error = []
        colors = []
        for exp_name, metrics in metrics_dict.items():
            color, _, _ = get_plot_format_given_name(exp_name)
            metrics, std_errors = metrics
            x.append(get_label_given_name(exp_name))
            y.append(metrics[key][-1])
            if i == 0:
                x_ref.append(get_label_given_name(exp_name))
                y_ref.append(metrics[key][-1])
            y_error.append(std_errors[key][-1])
            colors.append(color)

        # sort by y reference value
        if i == 0:
            y_ref, x_ref = zip(*sorted(zip(y_ref, x_ref), key=lambda x: -x[0]))
        idx_order = [x_ref.index(e) for e in x]

        _, sorted_x = zip(*sorted(zip(idx_order.copy(), x), key=lambda x: x[0]))
        _, sorted_y = zip(*sorted(zip(idx_order.copy(), y), key=lambda x: x[0]))
        _, sorted_y_error = zip(
            *sorted(zip(idx_order.copy(), y_error), key=lambda x: x[0])
        )
        _, sorted_colors = zip(
            *sorted(zip(idx_order.copy(), colors), key=lambda x: x[0])
        )

        fig = plt.figure(figsize=(9, 5))
        plt.bar(sorted_x, sorted_y, color=sorted_colors)
        plt.title(title)
        plt.errorbar(
            sorted_x,
            sorted_y,
            yerr=sorted_y_error,
            alpha=0.5,
            capsize=6,
            color="black",
            fmt="o",
            markersize=3,
        )
        ax = fig.axes
        ax[0].spines["top"].set_visible(False)
        ax[0].spines["right"].set_visible(False)
        ax[0].tick_params(labelrotation=0)
        # find the max and min values for the y axis
        y_max = max(sorted_y) + max(sorted_y_error) * 1.5
        y_min = min(sorted_y) - max(sorted_y_error) * 1.5
        ax[0].set_ylim([y_min, y_max])
        # check if output directory exists and create if not
        if not os.path.exists("./output/figures/metrics/barcharts/"):
            os.makedirs("./output/figures/metrics/barcharts/")
        fig.savefig(
            f"./output/figures/metrics/barcharts/bar_{key}_{plt_name}.{format}",
            format=f"{format}",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(fig)


def zero_one_norm(matrix):
    """Normalize matrix to be between 0 and 1"""
    return (matrix - np.min(matrix)) / np.ptp(matrix)


def plot_results_matrix(metric_results, one_a_range, two_a_range, plt_name=""):
    # plot a matrix of the final energy values
    final_divergence = np.zeros((len(two_a_range), len(one_a_range)))
    div_monotonicity = np.zeros((len(two_a_range), len(one_a_range)))
    for i, two_a in enumerate(two_a_range):
        for j, one_a in enumerate(one_a_range):
            # calculate the final divergence
            div_item = metric_results[f"{one_a}_{two_a}"][0]["divergence"]
            final_divergence[i, j] = div_item[-1]
            # calculate the divergence monotonicity
            div_mono_item = metric_results[f"{one_a}_{two_a}"][0]["div_monotonicity"]
            div_monotonicity[i, j] = div_mono_item[-1]

    if not os.path.exists("./output/figures/metrics/matrix/"):
        os.makedirs("./output/figures/metrics/matrix/")

    final_divergence = zero_one_norm(final_divergence)
    div_monotonicity = zero_one_norm(div_monotonicity)
    metrics = ["div_monotonicity", "divergence", "weighted"]
    for metric in metrics:
        if metric == "div_monotonicity":
            cmap = "Greens_r"
            title = "KL-divergence monotonicity"
            weighted_score = div_monotonicity
        elif metric == "divergence":
            cmap = "Blues_r"
            title = "Final KL-divergence"
            weighted_score = final_divergence
        elif metric == "weighted":
            cmap = "GnBu_r"
            title = "Weighted score"
            weighted_score = 0.5 * final_divergence + 0.5 * div_monotonicity
        else:
            raise ValueError("Unrecognized metric")
        weighted_score = zero_one_norm(weighted_score)
        plot_result_matrix(
            weighted_score,
            one_a_range,
            two_a_range,
            title,
            cmap,
            plt_name,
        )


def plot_result_matrix(
    values, one_a_range, two_a_range, metric_name, cmap, plt_name=""
):
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Helvetica"]
    # ensure values are only to two decimal places
    values = np.round(values, 2)
    plt.imshow(values * 0.666 + 0.333, cmap=cmap, vmin=0, vmax=1)
    # set axis labels
    plt.xlabel("5-HT1a", fontsize=14)
    plt.ylabel("5-HT2a", fontsize=14)
    plt.title(f"{metric_name}", fontsize=14)
    # use values from lists to set ticks
    dose_names = ["zero", "min", "light", "medi", "heavy", "max"]
    plt.xticks(np.arange(len(one_a_range)), dose_names, fontsize=12)
    plt.yticks(np.arange(len(two_a_range)), dose_names, fontsize=12)
    # plot values in matrix
    for i in range(len(two_a_range)):
        for j in range(len(one_a_range)):
            plt.text(
                j, i, round(values[i, j], 3), ha="center", va="center", fontsize=12
            )
    # make spines grey
    plt.gca().spines["bottom"].set_color("white")
    plt.gca().spines["top"].set_color("white")
    plt.gca().spines["right"].set_color("white")
    plt.gca().spines["left"].set_color("white")
    plt.savefig(
        f"./output/figures/metrics/matrix/matrix_{metric_name}{plt_name}.pdf",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()


def get_experiment(all_results, exp_name, metric, summary_type, drug_steps):
    if summary_type == "final":
        return all_results[exp_name][metric][:, -1]
    elif summary_type == "avg":
        return all_results[exp_name][metric][:, :drug_steps].mean(axis=1)
    elif summary_type == "max":
        return all_results[exp_name][metric][:, :drug_steps].max(axis=1)
    elif summary_type == "min":
        return all_results[exp_name][metric][:, :drug_steps].min(axis=1)
    else:
        raise ValueError("Invalid summary_type")


def plot_correlation(
    all_results,
    experiment_dict,
    x_metric,
    y_metric,
    summary_type_x,
    summary_type_y,
    axs,
    y_label=True,
):
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Helvetica"]
    # use data from all experiments to generate a correlation matrix
    experiment_names = list(experiment_dict.keys())
    placebo_name = experiment_names[0]
    drug_steps = experiment_dict[placebo_name].drug_steps
    xs, ys = [], []
    for exp_name in experiment_names:
        # get length of dose response
        drug_steps = experiment_dict[exp_name].drug_steps
        x = get_experiment(all_results, exp_name, x_metric, summary_type_x, drug_steps)
        y = get_experiment(all_results, exp_name, y_metric, summary_type_y, drug_steps)
        # average over each experiment
        x = x.mean(keepdims=True)
        y = y.mean(keepdims=True)
        xs.append(x)
        ys.append(y)
        # get color based on experiment name
        color, alpha, _ = get_plot_format_given_name(exp_name, give_linestyle=False)
        # plot x and y
        size = np.log((alpha + 1)) * 500
        axs.scatter(x, y, label=exp_name, color=color, s=size, alpha=0.33)

    x = np.concatenate(xs)
    y = np.concatenate(ys)
    # run a linear regression to compare x and y
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    # report r^2 and p-value, using only three decimal places
    print(f"{x_metric} vs {y_metric}: r^2 = {r_value**2:.3f}, p = {p_value:.3f}")
    axs.set_xlabel(f"{metric_name_dict[x_metric]} ({summary_type_x})", fontsize=14)
    if y_label:
        axs.set_ylabel(f"{metric_name_dict[y_metric]}", fontsize=14)
    else:
        # remove y tick labels
        axs.set_yticklabels([])
        # remove y ticks
        axs.yaxis.set_ticks_position("none")
        # remove left spine
        axs.spines["left"].set_visible(False)
    # use latex for the title so that r^2 is formatted correctly
    if p_value < 0.001:
        title_text = f"$r^2 = {r_value**2:.3f}$, $p < 0.001$"
    else:
        title_text = f"$r^2 = {r_value**2:.3f}$, $p = {p_value:.3f}$"
    axs.set_title(title_text, y=0.95, fontsize=14)
    # add regression line
    axs.plot(x, slope * x + intercept, color="grey", alpha=0.5)
    # remove top and right spines
    axs.spines["top"].set_visible(False)
    axs.spines["right"].set_visible(False)
    # set x and y limits
    axs.set_ylim(y.min() * 0.9, y.max() * 1.1)
    return axs


def plot_multi_corrs(
    all_results,
    experiment_dict,
    x_metrics,
    y_metric,
    summary_type_xs,
    summary_type_y,
    plot_name,
):
    fig, axs = plt.subplots(1, 4, figsize=(12, 3.5), dpi=200)
    add_y_label = True
    for x_metric, summary_type_x, ax in zip(x_metrics, summary_type_xs, axs):
        _ = plot_correlation(
            all_results,
            experiment_dict,
            x_metric,
            y_metric,
            summary_type_x,
            summary_type_y,
            ax,
            y_label=add_y_label,
        )
        add_y_label = False
    fig.tight_layout()
    plt.show()
    # save figure as pdf
    if not os.path.exists("./output/figures/metrics/corrs/"):
        os.makedirs("./output/figures/metrics/corrs/")
    fig.savefig(
        f"./output/figures/metrics/corrs/{plot_name}_correlation.pdf",
        bbox_inches="tight",
    )


def ttests(all_results, experiment_dict, metric, summary_type):
    if not os.path.exists("./output/figures/metrics/"):
        os.makedirs("./output/figures/metrics/")

    experiment_names = list(experiment_dict.keys())
    placebo_name = experiment_names[0]
    drug_steps = experiment_dict[placebo_name].drug_steps
    placebo = get_experiment(
        all_results, placebo_name, metric, summary_type, drug_steps
    )
    for exp_name in experiment_names[1:]:
        # get length of dose response
        drug_steps = experiment_dict[exp_name].drug_steps
        treatment = get_experiment(
            all_results, exp_name, metric, summary_type, drug_steps
        )
        # run a t-test to compare placebo and treatment
        t_stat, p_val = ttest_ind(placebo, treatment)
        # report it as <test>: t(d) = <t_stat>, p = <p_val>
        # Use only three decimal places for t and p
        print(
            f"{exp_name} vs {placebo_name}: t({len(placebo) - 1}) = {t_stat:.3f}, p = {p_val:.3f}"
        )
