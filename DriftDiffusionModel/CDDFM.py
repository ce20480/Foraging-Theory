import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import os

# import torch
# import torch.optim as optim

plt.rcParams.update({"text.usetex": True, "font.size": 16})
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def record_threshold_data(df_walks, OLT, y_intercepts, gradients, total_steps):
    records = []

    for y_intercept in y_intercepts:
        for gradient in gradients:
            avg_time = calculate_average_meeting_time(
                df_walks, OLT, y_intercept, gradient, total_steps
            )
            records.append((y_intercept, gradient, avg_time))

    return pd.DataFrame(
        records, columns=["y_intercept", "gradient", "average_meeting_time"]
    )


def plot_3d_threshold_data(df_threshold_data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    xs = df_threshold_data["y_intercept"]
    ys = df_threshold_data["gradient"]
    zs = df_threshold_data["average_meeting_time"]

    ax.scatter(xs, ys, zs)

    ax.set_xlabel("Y-Intercept")
    ax.set_ylabel("Gradient")
    ax.set_zlabel("Average Meeting Time")

    plt.show()


def create_best_thresholds():
    # Assuming df_walks is a DataFrame where each row is a walk and each column is a timestep
    # Define the range for y-intercepts and gradients
    y_begin = -5
    y_end = 5
    grad_begin = -1
    grad_end = 1
    range_of_values = 100
    OLT = 50
    # y_intercepts = np.linspace(-10, 10, 100)  # Adjust these ranges and steps as needed
    # gradients = np.linspace(-1, 1, 100)  # Adjust these ranges and steps as needed
    y_intercepts = np.linspace(
        y_begin, y_end, range_of_values
    )  # Adjust these ranges and steps as needed
    gradients = np.linspace(
        grad_begin, grad_end, range_of_values
    )  # Adjust these ranges and steps as needed
    total_steps = 100  # or the maximum number of timesteps you have in your walks
    # best_threshold = find_best_threshold(
    #     pd_walks_data,
    #     OLT=OLT,
    #     y_intercepts=y_intercepts,
    #     gradients=gradients,
    #     total_steps=total_steps,
    # )

    # Record the threshold data
    df_threshold_data = record_threshold_data(
        pd_walks_data,
        OLT=OLT,
        y_intercepts=y_intercepts,
        gradients=gradients,
        total_steps=total_steps,
    )

    # how to conver dataframe to csv
    df_threshold_data.to_csv(
        f"data/optimal_threshold_data_{y_begin}_{y_end}_{grad_begin}_{grad_end}.csv",
        index=False,
    )

    # Plot the 3D data
    plot_3d_threshold_data(df_threshold_data)


def plot_random_walks_with_threshold_no_stopping(
    df_walks, initial_y_intercept, gradient, OLT, line_alpha=0.5
):
    """
    Plots all random walks with a linear threshold determined by an initial y-intercept and a gradient.

    Args:
    df_walks (DataFrame): DataFrame where each row is a random walk and each column is a timestep.
    initial_y_intercept (float): The starting value of the threshold at time zero.
    gradient (float): The rate of change of the threshold per timestep.
    OLT (int): Optimal Leaving Time, to be highlighted on the graph.
    line_alpha (float): Transparency of the random walk lines, default is 0.5.
    """
    # Calculate the number of timesteps based on the longest walk
    max_timesteps = df_walks.apply(lambda x: x.last_valid_index(), axis=1).max() + 1

    # Generate the threshold line values over the timesteps
    threshold_values = np.array(
        [initial_y_intercept + gradient * t for t in range(max_timesteps)]
    )

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot each random walk
    for index, walk in df_walks.iterrows():
        valid_data = walk.dropna().values
        ax.plot(valid_data, color="black", alpha=line_alpha)

    # Plot the threshold line
    ax.plot(threshold_values, label="Threshold", color="red", linestyle="--")

    # Highlight the Optimal Leaving Time
    ax.axvline(
        x=OLT,
        color="blue",
        linestyle="-",
        linewidth=2,
        label="Optimal Leaving Time (OLT)",
    )

    ax.set_title("Random Walks with Linear Threshold")
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Walk Value")
    ax.legend()
    plt.show()


def plot_random_walks_with_threshold(
    df_walks,
    initial_y_intercept,
    gradient,
    OLT,
    num_walks=None,
    line_alpha=0.5,
    threshold_timesteps=None,
):
    """
    Plots specified number of random walks with a linear threshold, each walk stops at the point it meets the threshold.
    The threshold is plotted for a specified number of timesteps.

    Args:
    df_walks (DataFrame): DataFrame where each row is a random walk and each column is a timestep.
    initial_y_intercept (float): The starting value of the threshold at time zero.
    gradient (float): The rate of change of the threshold per timestep.
    OLT (int): Optimal Leaving Time, to be highlighted on the graph.
    num_walks (int, optional): Number of random walks to plot. If None, plots all walks.
    line_alpha (float): Transparency of the random walk lines, default is 0.5.
    threshold_timesteps (int, optional): Number of timesteps for which the threshold is plotted. If None, uses the max available timesteps.
    """
    # Ensure all data are floats to avoid type errors
    df_walks = df_walks.apply(pd.to_numeric, errors="coerce")

    # Calculate the number of timesteps based on the longest walk
    max_timesteps = df_walks.apply(lambda x: x.last_valid_index(), axis=1).max()
    if max_timesteps is None:
        raise ValueError("No valid data in any walks.")
    max_timesteps = int(max_timesteps) + 1

    # Adjust the number of threshold timesteps if specified
    if threshold_timesteps is not None and threshold_timesteps > 0:
        max_timesteps = min(max_timesteps, threshold_timesteps)

    # Generate the threshold line values over the timesteps
    threshold_values = np.array(
        [initial_y_intercept + gradient * t for t in range(max_timesteps)]
    )

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Select random walks if num_walks is specified and less than total walks
    if num_walks is not None and num_walks < len(df_walks):
        sampled_indices = np.random.choice(
            df_walks.index, size=num_walks, replace=False
        )
        selected_walks = df_walks.loc[sampled_indices]
    else:
        selected_walks = df_walks

    # Plot each random walk up to the threshold
    for index, walk in selected_walks.iterrows():
        valid_data = walk.dropna().values
        # Find the first index where walk meets or exceeds the threshold
        stop_index = np.where(valid_data >= threshold_values[: len(valid_data)])[0]
        if stop_index.size > 0:
            stop_index = (
                stop_index[0] + 1
            )  # Include the point where it crosses the threshold
        else:
            stop_index = len(
                valid_data
            )  # If it never meets the threshold, plot all data

        ax.plot(
            range(stop_index), valid_data[:stop_index], color="black", alpha=line_alpha
        )

    # Plot the threshold line
    ax.plot(threshold_values, label="Threshold", color="red", linestyle="--")

    # Highlight the Optimal Leaving Time
    ax.axvline(
        x=OLT,
        color="blue",
        linestyle="-",
        linewidth=2,
        label="Optimal Leaving Time (OLT)",
    )

    ax.set_title("Random Walks Meeting Linear Threshold")
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Walk Value")
    ax.legend()
    plt.show()


def n_plot_walks_meeting_threshold(
    df_walks,
    initial_y_intercept,
    gradient,
    OLT,
    max_timesteps=None,
    alpha=0.2,
    iter_max=10,
):
    """
    Plots all random walks up to where they meet the threshold or up to a specified timestep.
    Plots the threshold, highlights the Optimal Leaving Time (OLT), and allows for adjustable transparency in walk lines.

    Args:
    df_walks (DataFrame): DataFrame where each row is a random walk and each column is a timestep.
    initial_y_intercept (float): Initial value of the threshold.
    gradient (float): Rate of change of the threshold per timestep.
    OLT (int): Optimal Leaving Time, to be highlighted on the graph.
    max_timesteps (int, optional): Maximum number of timesteps to plot. If None, uses the maximum from the walks.
    alpha (float, optional): Transparency level of the walk lines, default is 0.5.
    iter_max (int, optional): Maximum number of iterations to plot the walks, default is 100.
    """
    # Convert all data to numeric, coercing errors to NaN, then drop rows that are all NaN
    df_walks = df_walks.dropna(how="all")

    # Calculate maximum timesteps if not provided
    if max_timesteps is None:
        max_timesteps = (
            int(df_walks.apply(lambda x: x.last_valid_index(), axis=1).max()) + 1
        )

    timesteps = np.arange(max_timesteps)
    threshold_values = initial_y_intercept + gradient * timesteps

    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    iter = 0

    # Plot each walk up to the point it meets the threshold or max_timesteps
    while iter < iter_max:
        for index, walk in df_walks.iterrows():
            walk_data = walk.dropna().values
            # Determine the shortest length to plot
            length_to_plot = min(len(walk_data), max_timesteps)
            walk_data = walk_data[:length_to_plot]
            threshold_subarray = threshold_values[:length_to_plot]

            # Find where the walk meets or exceeds the threshold
            meets_threshold_indices = np.where(walk_data >= threshold_subarray)[0]
            if len(meets_threshold_indices) > 0:
                plot_up_to = meets_threshold_indices[0]  # Include the meeting point
                iter += 1
            else:
                plot_up_to = (
                    length_to_plot  # Plot up to the available data if no meeting point
                )

            # Plot the walk
            ax.plot(
                range(plot_up_to),
                walk_data[:plot_up_to],
                color="black",
                alpha=alpha,
                label=f"Walks" if index == 0 else "",
            )

    # Plot the threshold line
    ax.plot(
        timesteps[:max_timesteps],
        threshold_values[:max_timesteps],
        "r--",
        label="Threshold",
    )

    # Highlight the OLT
    ax.axvline(
        x=OLT,
        color="blue",
        linestyle="-",
        linewidth=2,
        label="Optimal Leaving Time (OLT)",
    )

    # Set plot titles and labels
    ax.set_title("Random Walks Meeting Threshold")
    ax.set_xlabel("Time Steps(seconds)")
    ax.set_ylabel("Walk Value")
    ax.legend()

    plt.show()


def m_plot_walks_meeting_threshold(
    df_walks,
    y_intercept,
    gradient,
    OLT,
    num_walks_to_plot=None,
    timesteps=None,
    line_alpha=0.5,
):
    """
    Plots the random walks up to the point they meet a linear threshold or up to a specified timestep.

    Args:
    df_walks (DataFrame): DataFrame where each row represents a random walk.
    y_intercept (float): The initial value of the threshold.
    gradient (float): The rate of change of the threshold.
    OLT (int): Optimal Leaving Time, highlighted on the plot.
    num_walks_to_plot (int, optional): Maximum number of walks to plot. If None, plots all.
    timesteps (int, optional): Number of timesteps to include in the plot. If None, uses the maximum available.
    line_alpha (float): Transparency of the walk lines.
    """
    if num_walks_to_plot is not None:
        df_walks = df_walks.sample(n=min(num_walks_to_plot, len(df_walks)))

    if timesteps is None:
        timesteps = (
            int(df_walks.apply(lambda x: x.last_valid_index(), axis=1).max()) + 1
        )

    threshold_values = y_intercept + gradient * np.arange(timesteps)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Iterate through each walk
    for index, walk in df_walks.iterrows():
        valid_data = walk.dropna().values
        min_length = min(len(valid_data), timesteps)
        threshold_to_compare = threshold_values[:min_length]

        stop_index = np.where(valid_data[:min_length] >= threshold_to_compare)[0]
        if stop_index.size > 0:
            stop_index = stop_index[0]
        else:
            stop_index = min_length

        ax.plot(
            np.arange(stop_index),
            valid_data[:stop_index],
            color="blue",
            alpha=line_alpha,
            label=f"Walks" if index == 0 else "",
        )
    print(stop_index)

    # Plot the threshold
    ax.plot(
        threshold_values, "-", color="orange", alpha=line_alpha, label="Leave Threshold"
    )

    # Highlight the Optimal Leaving Time (OLT)
    ax.axvline(
        x=OLT,
        color="red",
        linestyle="--",
        linewidth=2,
        label="Optimal Leaving Time (OLT)",
    )

    ax.set_title("Example of Combined Drift Diffusion Foraging Model")
    ax.set_xlabel("Time Step(s)")
    ax.set_ylabel("Evidence")
    ax.legend()

    plt.show()


def nonlinear_threshold(t, y_intercept, decay_rate, OLT):
    """
    Non-linear threshold function that decreases over time and stabilizes post-OLT.

    Args:
    t (int or ndarray): Time steps.
    y_intercept (float): Initial threshold value at t=0.
    decay_rate (float): Rate at which the threshold decays.
    OLT (int): Optimal Leaving Time, the time after which the threshold change stabilizes.
    """
    return y_intercept * np.exp(-decay_rate * np.minimum(t, OLT))


def plot_walks_meeting_nonlinear_threshold(
    df_walks,
    y_intercept,
    decay_rate,
    OLT,
    non_linear_threshold=nonlinear_threshold,
    num_walks_to_plot=None,
    timesteps=None,
    line_alpha=0.5,
):
    """
    Plots the random walks up to the point they meet a non-linear threshold or up to a specified timestep.

    Args:
    df_walks (DataFrame): DataFrame where each row represents a random walk.
    y_intercept (float): The initial value of the threshold.
    decay_rate (float): Decay rate of the threshold.
    OLT (int): Optimal Leaving Time, highlighted on the plot.
    num_walks_to_plot (int, optional): Maximum number of walks to plot. If None, plots all.
    timesteps (int, optional): Number of timesteps to include in the plot. If None, uses the maximum available.
    line_alpha (float): Transparency of the walk lines.
    """
    if num_walks_to_plot is not None:
        df_walks = df_walks.sample(n=min(num_walks_to_plot, len(df_walks)))

    if timesteps is None:
        timesteps = (
            int(df_walks.apply(lambda x: x.last_valid_index(), axis=1).max()) + 1
        )

    time_steps = np.arange(timesteps)
    threshold_values = non_linear_threshold(time_steps, y_intercept, decay_rate, OLT)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Iterate through each walk
    for index, walk in df_walks.iterrows():
        valid_data = walk.dropna().values
        min_length = min(len(valid_data), timesteps)
        threshold_to_compare = threshold_values[:min_length]

        stop_index = np.where(valid_data[:min_length] >= threshold_to_compare)[0]
        if stop_index.size > 0:
            stop_index = stop_index[0]
        else:
            stop_index = min_length

        ax.plot(
            np.arange(stop_index),
            valid_data[:stop_index],
            color="black",
            alpha=line_alpha,
        )

    # Plot the threshold
    ax.plot(threshold_values, "-", color="orange", label="Non-linear Threshold")

    # Highlight the Optimal Leaving Time (OLT)
    ax.axvline(
        x=OLT,
        color="red",
        linestyle="--",
        linewidth=2,
        label="Optimal Leaving Time (OLT)",
    )

    ax.set_title("Random Walks Meeting the Non-linear Threshold")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Position")
    ax.legend()

    plt.show()


def any_plot_walks_meeting_nonlinear_threshold(
    df_walks,
    non_linear_func,
    params,
    OLT,
    num_walks_to_plot=None,
    timesteps=None,
    line_alpha=0.5,
):
    """
    Plots the random walks up to the point they meet a non-linear threshold or up to a specified timestep,
    using a non-linear function provided by the user.

    Args:
    df_walks (DataFrame): DataFrame where each row represents a random walk.
    non_linear_func (function): Non-linear function that defines the threshold.
    params (dict): Parameters for the non-linear function.
    OLT (int): Optimal Leaving Time, highlighted on the plot.
    num_walks_to_plot (int, optional): Maximum number of walks to plot. If None, plots all.
    timesteps (int, optional): Number of timesteps to include in the plot. If None, uses the maximum available.
    line_alpha (float): Transparency of the walk lines.
    """
    if num_walks_to_plot is not None:
        df_walks = df_walks.sample(n=min(num_walks_to_plot, len(df_walks)))

    if timesteps is None:
        timesteps = (
            int(df_walks.apply(lambda x: x.last_valid_index(), axis=1).max()) + 1
        )

    time_steps = np.arange(timesteps)
    threshold_values = non_linear_func(time_steps, OLT, **params)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Iterate through each walk
    for index, walk in df_walks.iterrows():
        valid_data = walk.dropna().values
        min_length = min(len(valid_data), timesteps)
        threshold_to_compare = threshold_values[:min_length]

        stop_index = np.where(valid_data[:min_length] >= threshold_to_compare)[0]
        if stop_index.size > 0:
            stop_index = stop_index[0]
        else:
            stop_index = min_length

        ax.plot(
            np.arange(stop_index),
            valid_data[:stop_index],
            color="black",
            alpha=line_alpha,
        )

    # Plot the threshold
    ax.plot(threshold_values, "--", color="orange", label="Non-linear Threshold")

    # Highlight the Optimal Leaving Time (OLT)
    ax.axvline(
        x=OLT,
        color="red",
        linestyle="--",
        linewidth=2,
        label="Optimal Leaving Time (OLT)",
    )

    ax.set_title("Random Walks Meeting the Non-linear Threshold")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Position")
    ax.legend()

    plt.show()


def exponential_decay(t, OLT, y_intercept, decay_rate):
    """Example non-linear threshold function"""
    return y_intercept * np.exp(-decay_rate * np.minimum(t, OLT))


if __name__ == "__main__":
    # how to get path of file in a folder in the same directory as the script using os
    path_to_random_walks = os.path.join((os.getcwd()), "data/random_walks.csv")
    path_to_error_data = os.path.join(
        (os.getcwd()), "data/random_walks_with_step_1000_bounds_0.0001_5.csv"
    )
    pd_walks_data = pd.read_csv(path_to_random_walks)
    pd_error_data = pd.read_csv(path_to_error_data)

    # Plot all random walks with a linear threshold without stopping at the threshold
    # plot_random_walks_with_threshold(
    #     pd_walks_data, initial_y_intercept=5, gradient=-0.1, OLT=50
    # )

    # Plot all random walks with a linear threshold with stopping at the threshold
    # plot_random_walks_with_threshold(
    #     pd_walks_data,
    #     initial_y_intercept=5,
    #     gradient=-0.1,
    #     OLT=50,
    #     num_walks=10,
    #     threshold_timesteps=100,
    # )
    # y_intercept = 0.353535354
    # gradient = 0.01010101
    # y_intercept = 4.994949  # linear
    # gradient = -0.090909  # linear
    y_intercept = 3.45  # linear
    gradient = -0.09  # linear
    # y_intercept = 4.545455  # non linear
    # gradient = -0.080808  # non linear
    # n_plot_walks_meeting_threshold(
    #     pd_walks_data,
    #     initial_y_intercept=0.353535354,
    #     gradient=0.01010101,
    #     max_timesteps=100,
    #     OLT=50,
    # )

    m_plot_walks_meeting_threshold(
        pd_walks_data,
        y_intercept=y_intercept,
        gradient=gradient,
        OLT=38.83,  # 50, 24.7572815532, 38.834951456
        timesteps=100,
        # num_walks_to_plot=1,
    )

    # plot_walks_meeting_nonlinear_threshold(
    #     pd_walks_data,
    #     y_intercept=2,
    #     decay_rate=0.05,
    #     OLT=50,
    #     timesteps=100,
    #     num_walks_to_plot=10,
    # )

    # any_plot_walks_meeting_nonlinear_threshold(
    #     pd_walks_data,
    #     exponential_decay,
    #     {"y_intercept": 10, "decay_rate": 0.1},
    #     OLT=50,
    #     timesteps=100,
    #     num_walks_to_plot=100,
    # )

    # path_to_average_times = os.path.join(
    #     (os.getcwd()), "data/optimal_threshold_data.csv"
    # )
    # df_threshold_average_times = pd.read_csv(path_to_average_times)
    # plot_3d_threshold_data(df_threshold_average_times)
