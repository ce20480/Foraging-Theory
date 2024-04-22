import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def find_meeting_times_final(data, m, c, max_timesteps=None):
    """
    Calculate the meeting times between the quadratic threshold y = mt^2 + c and each series in the data.

    Args:
    data (DataFrame): The dataset containing multiple series of data points.
    m (float): Coefficient of t^2 in the threshold equation.
    c (float): Constant term in the threshold equation.
    max_timesteps (int, optional): Maximum number of timesteps to calculate the meeting times for.

    Returns:
    meeting_times (dict): Dictionary of meeting times for each series where keys are series indices.
    """
    meeting_times = {}
    if max_timesteps is None:
        max_timesteps = data.shape[0]
    time_steps = np.arange(max_timesteps)

    # Calculate the quadratic threshold for each timestep
    threshold_values = m * time_steps**2 + c

    # Calculate meeting times for each series
    for column in data.columns:
        walk = data[column].dropna()
        max_calculate_timesteps = min(len(walk), max_timesteps)
        meet_time = np.where(
            walk[:max_calculate_timesteps] >= threshold_values[:max_calculate_timesteps]
        )[0]
        if meet_time.size > 0:
            meeting_times[column] = meet_time[0]
        else:
            meeting_times[column] = None

    return meeting_times


def calculate_rewards(walk, m, c, reward_coeff, reward_decay, max_timesteps):
    """Calculates accumulated rewards for a walk given the quadratic threshold parameters and reward function."""
    threshold = m * np.arange(len(walk)) ** 2 + c
    meet_time = np.where(walk >= threshold)[0]
    if meet_time.size > 0:
        meet_time = meet_time[0] if meet_time[0] < max_timesteps else max_timesteps
        times = np.arange(meet_time + 1)
        rewards = reward_coeff * np.exp(-reward_decay * times)
        return np.sum(rewards)
    return 0


def plot_total_time_vs_average_reward_with_OLT(data, thresholds, reward_settings):
    """
    Plots the total meeting time against the average reward for each random walk and marks the theoretical total OLT reward.
    Adjusted for quadratic threshold.
    """
    num_walks = data.shape[1]
    total_meeting_times = np.zeros(num_walks)
    total_rewards = np.zeros(num_walks)
    total_OLT = sum(x[2] for x in thresholds)
    total_OLT_reward = 0

    for (m, c, OLT), (reward_coeff, reward_decay) in zip(thresholds, reward_settings):
        olt_times = np.arange(OLT + 1)
        olt_reward = np.sum(reward_coeff * np.exp(-reward_decay * olt_times))
        total_OLT_reward += olt_reward
        for i in range(num_walks):
            walk = data.iloc[:, i].dropna()
            threshold = m * np.arange(len(walk)) ** 2 + c
            meet_time = np.where(walk >= threshold)[0]
            if meet_time.size > 0:
                meet_time = meet_time[0]
                times = np.arange(meet_time + 1)
                rewards = reward_coeff * np.exp(-reward_decay * times)
                total_rewards[i] += np.sum(rewards)
                total_meeting_times[i] += meet_time

    average_rewards = total_rewards / len(thresholds)
    plt.figure(figsize=(12, 8))
    plt.scatter(
        total_meeting_times, average_rewards, color="blue", label="Average Reward"
    )
    plt.scatter(
        total_OLT,
        total_OLT_reward / len(thresholds),
        color="red",
        s=100,
        label="Total OLT Reward",
        marker="X",
    )
    plt.title("Total Meeting Time vs Average Reward and Total OLT Reward")
    plt.xlabel("Total Meeting Time")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.show()


def m_plot_walks_meeting_threshold(
    ax, df_walks, y_intercept, gradient, OLT, timesteps, line_alpha=0.5
):
    """
    Plots the random walks up to the point they meet a quadratic threshold y = mt^2 + c on a given Axes object.

    Args:
    ax (matplotlib.axes.Axes): The axes object where the plot will be drawn.
    df_walks (DataFrame): DataFrame where each row represents a random walk.
    y_intercept (float): The initial value of the threshold.
    gradient (float): Coefficient of t^2 in the threshold equation.
    OLT (int): Optimal Leaving Time, highlighted on the plot.
    timesteps (int): Number of timesteps to include in the plot.
    line_alpha (float): Transparency of the walk lines.
    """
    threshold_values = y_intercept + gradient * np.arange(timesteps) ** 2

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
            label=f"Walk {index}" if index == 0 else "",
        )

    # Plot the threshold
    ax.plot(
        np.arange(timesteps),
        threshold_values,
        "-",
        color="orange",
        alpha=line_alpha,
        label="Leave Threshold",
    )

    # Highlight the Optimal Leaving Time (OLT)
    ax.axvline(
        x=OLT,
        color="red",
        linestyle="--",
        linewidth=2,
        label="Optimal Leaving Time (OLT)",
    )

    ax.set_title("Walks Meeting Quadratic Threshold")
    ax.set_xlabel("Time Step(s)")
    ax.set_ylabel("Walk Value")
    ax.legend()


def plot_combined_walks_and_histogram(data, thresholds):
    """
    Plot multiple scenarios with random walks meeting quadratic thresholds and a histogram of total meeting times.

    Args:
    data (DataFrame): DataFrame containing multiple series of random walks, each column is a walk.
    thresholds (list of tuples): Each tuple contains (m, c, OLT) for each run.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    max_timesteps = 100
    total_meeting_times = []

    for i, (m, c, OLT) in enumerate(thresholds):
        m_plot_walks_meeting_threshold(axes[i], data, c, m, OLT, max_timesteps)
        total_meeting_times.extend(
            data.apply(
                lambda x: (
                    np.where(x >= m * np.arange(len(x)) ** 2 + c)[0][0]
                    if np.any(x >= m * np.arange(len(x)) ** 2 + c)
                    else max_timesteps
                ),
                axis=0,
            )
        )

    # Filter total_meeting_times to consider only up to max_timesteps for histogram
    filtered_total_times = [
        time if time < max_timesteps else max_timesteps for time in total_meeting_times
    ]

    # Plotting the total meeting times histogram
    axes[-1].hist(filtered_total_times, bins=20, color="green", alpha=0.7)
    total_OLT = sum(x[2] for x in thresholds)
    axes[-1].axvline(
        x=total_OLT, color="red", linestyle="--", linewidth=2, label="Total OLT"
    )
    axes[-1].set_title("Total Meeting Times Across Runs")
    axes[-1].set_xlabel("Total Meeting Time (Timestep)")
    axes[-1].set_ylabel("Frequency")
    axes[-1].legend()
    axes[-1].grid(True)

    plt.tight_layout()
    plt.show()


def plot_walks_and_histogram_with_different_thresholds(
    data, thresholds_line, thresholds_hist
):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    max_timesteps = 100

    # Plotting line plots using the first set of thresholds
    for i, (m, c, OLT) in enumerate(thresholds_line):
        m_plot_walks_meeting_threshold(axes[i], data, c, m, OLT, max_timesteps)

    # Calculate and plot the histogram using the second set of thresholds
    total_meeting_times = []
    for m, c, _ in thresholds_hist:
        current_meeting_times = data.apply(
            lambda x: (
                np.where(x >= m * np.arange(len(x)) ** 2 + c)[0][0]
                if np.any(x >= m * np.arange(len(x)) ** 2 + c)
                else max_timesteps
            ),
            axis=0,
        )
        total_meeting_times.extend(current_meeting_times)
        print(
            f"Threshold: {m}, {c}, Meeting times: {current_meeting_times[:5]}"
        )  # Debug print

    filtered_total_times = [
        time if time < max_timesteps else max_timesteps for time in total_meeting_times
    ]

    # Histogram of total meeting times
    axes[-1].hist(filtered_total_times, bins=20, color="green", alpha=0.7)
    total_OLT = sum(x[2] for x in thresholds_hist)
    axes[-1].axvline(
        x=total_OLT, color="red", linestyle="--", linewidth=2, label="Total OLT"
    )
    axes[-1].set_title("Total Meeting Times Across Runs")
    axes[-1].set_xlabel("Total Meeting Time (Timestep)")
    axes[-1].set_ylabel("Frequency")
    axes[-1].legend()
    axes[-1].grid(True)

    plt.tight_layout()
    plt.show()


# Re-run this function with any changes to the thresholds_hist list to observe changes.

if __name__ == "__main__":
    # how to get path of file in a folder in the same directory as the script using os
    path_to_random_walks = os.path.join((os.getcwd()), "data/random_walks.csv")
    path_to_error_data = os.path.join(
        (os.getcwd()), "data/random_walks_with_step_1000_bounds_0.0001_5.csv"
    )
    pd_walks_data = pd.read_csv(path_to_random_walks)
    pd_error_data = pd.read_csv(path_to_error_data)

    # y_intercept = 0.353535354
    # gradient = 0.01010101
    c = 8.8  # playing around optimised
    m = -0.090909  # playing around optimised
    # c = 4.994949  # linear
    # m = -0.090909  # linear
    # c = 4.994949  # linear
    # m = -0.090909  # linear
    # c = 4.545455  # non linear
    # m = -0.080808  # non linear
    max_timesteps = 100
    OLT = 50

    # Define threshold parameters for each run
    # thresholds = [
    #     (-0.02, 1, 50),  # First run parameters
    #     (-0.08, 2, 25),  # Second run parameters
    #     (-0.09, 3, 39),  # Third run parameters
    # ]
    # thresholds = [
    #     (-0.090909, 8.8, 50),  # First run parameters
    #     (-0.030303, 5.4, 25),  # Second run parameters
    #     (-0.020202, 5.5, 39),  # Third run parameters
    # ]

    thresholds = [  # Non_linear threshold
        (-0.0030303, 8.8, 50),  # First run parameters
        (-0.0030303, 5.4, 25),  # Second run parameters
        (-0.0020202, 5.5, 39),  # Third run parameters
    ]
    reward_settings = [
        (57.5, 0.11),  # Reward settings for OLT = 50
        (32.5, 0.11),  # Reward settings for OLT ≈ 24.757
        (45, 0.11),  # Reward settings for OLT ≈ 38.835
    ]

    # Define the threshold parameters for each scenario
    thresholds_line = [
        (-0.0030909, 4.994949, 50),
        (-0.0030808, 3.994949, 25),
        (-0.0020202, 3.45, 39),
    ]

    thresholds_hist = [(-0.090909, 8.8, 50), (-0.030303, 5.4, 25), (-0.020202, 5.5, 39)]

    # plot_total_time_vs_average_reward_with_OLT(
    #     pd_walks_data, thresholds, reward_settings
    # )
    # plot_combined_walks_and_histogram(pd_walks_data, thresholds)
    plot_walks_and_histogram_with_different_thresholds(
        pd_walks_data, thresholds_line, thresholds_hist
    )
