import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def find_meeting_times_final(data, m, c, max_timesteps=None):
    """
    Calculate the meeting times between the threshold y = mt + c and each series in the data,
    up to a maximum number of timesteps.

    Args:
    data (DataFrame): The dataset containing multiple series of data points.
    m (float): Gradient of the threshold line.
    c (float): Y-intercept of the threshold line.
    max_timesteps (int, optional): Maximum number of timesteps to calculate the meeting times for.

    Returns:
    meeting_times (dict): Dictionary of meeting times for each series where keys are series indices.
    """
    meeting_times = {}
    if max_timesteps is None:
        max_timesteps = data.shape[0]  # Use all available timesteps if not specified
    time_steps = np.arange(max_timesteps)

    # Threshold for each timestep
    threshold_values = m * time_steps + c

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
            meeting_times[column] = None  # No meeting time if threshold is not met

    return meeting_times


def plot_meeting_times_histogram(meeting_times, OLT):
    """
    Plots a histogram of the timesteps at which random walks meet the threshold, and marks the Optimal Leaving Time (OLT).

    Args:
    meeting_times (dict): Dictionary of meeting times for each random walk, where values are the timesteps.
    OLT (float): The Optimal Leaving Time to highlight on the histogram.
    """
    # Filter out None values and convert to list
    times = [time for time in meeting_times.values() if time is not None]

    plt.figure(figsize=(10, 5))
    plt.hist(times, bins=20, color="blue", alpha=0.7)
    plt.axvline(
        x=OLT,
        color="red",
        label="Optimal Leaving Time (OLT)",
        linestyle="--",
        linewidth=2,
    )
    plt.title("Histogram of Meeting Times")
    plt.xlabel("Meeting Time (s)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.show()


def hist_plot_model_runs_and_total_times(data, thresholds, max_timesteps=100):
    """
    Plot the results of multiple threshold runs and the total meeting times histogram.

    Args:
    data (DataFrame): The dataset containing multiple series of random walks.
    thresholds (list of tuples): Each tuple contains (m, c, OLT) for each run.
    max_timesteps (int): Maximum number of timesteps to consider for each run.
    """
    num_walks = data.shape[1]
    total_meeting_times = np.zeros(num_walks)
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
    axes = axes.flatten()

    # Process each run
    for i, (m, c, OLT) in enumerate(thresholds):
        meeting_times = find_meeting_times_final(data, m, c, max_timesteps)
        times = [time for time in meeting_times.values() if time is not None]

        # Plot histogram for this run
        axes[i].hist(times, bins=20, color="blue", alpha=0.7)
        axes[i].axvline(x=OLT, color="red", linestyle="--", linewidth=2, label="OLT")
        axes[i].set_title(f"Run {i+1}: m={m}, c={c}")
        axes[i].set_xlabel("Meeting Time (Timestep)")
        axes[i].set_ylabel("Frequency")
        axes[i].legend()
        axes[i].grid(True)

        # Update total meeting times
        total_meeting_times += np.array(
            [time if time is not None else 0 for time in meeting_times.values()]
        )

    # Total OLT is the sum of individual OLTs
    total_OLT = sum([olt for _, _, olt in thresholds])

    # Plot total meeting times histogram
    axes[3].hist(total_meeting_times, bins=20, color="green", alpha=0.7)
    axes[3].axvline(
        x=total_OLT, color="red", linestyle="--", linewidth=2, label="Total OLT"
    )
    axes[3].set_title("Total Meeting Times Across Runs")
    axes[3].set_xlabel("Total Meeting Time (Timestep)")
    axes[3].set_ylabel("Frequency")
    axes[3].legend()
    axes[3].grid(True)

    plt.tight_layout()
    plt.show()


def m_plot_walks_meeting_threshold(
    ax, df_walks, y_intercept, gradient, OLT, timesteps, line_alpha=0.5
):
    """
    Plots the random walks up to the point they meet a linear threshold or up to a specified timestep on a given Axes object.

    Args:
    ax (matplotlib.axes.Axes): The axes object where the plot will be drawn.
    df_walks (DataFrame): DataFrame where each row represents a random walk.
    y_intercept (float): The initial value of the threshold.
    gradient (float): The rate of change of the threshold.
    OLT (int): Optimal Leaving Time, highlighted on the plot.
    timesteps (int): Number of timesteps to include in the plot.
    line_alpha (float): Transparency of the walk lines.
    """
    threshold_values = y_intercept + gradient * np.arange(timesteps)

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
            label=f"Walk" if index == 0 else "",
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

    ax.set_title("Walks Meeting Threshold")
    ax.set_xlabel("Time Step(s)")
    ax.set_ylabel("Walk Value")
    ax.legend()


def plot_combined_walks_and_histogram(data, thresholds):
    """
    Plot multiple scenarios with random walks meeting thresholds and a histogram of total meeting times.

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
                    np.where(x >= m * np.arange(len(x)) + c)[0][0]
                    if np.any(x >= m * np.arange(len(x)) + c)
                    else max_timesteps
                ),
                axis=0,
            )
        )

    # Filter total_meeting_times to consider only up to max_timesteps for histogram
    filtered_total_times = [
        time * 4.5 if time < max_timesteps else max_timesteps
        for time in total_meeting_times
    ]

    # Plotting the total meeting times histogram
    axes[-1].hist(filtered_total_times, bins=20, color="green", alpha=0.7)
    total_OLT = sum(x[2] for x in thresholds)
    axes[-1].axvline(
        x=total_OLT, color="red", linestyle="--", linewidth=2, label="Total OLT"
    )
    axes[-1].set_title("Total Meeting Times Across Runs")
    axes[-1].set_xlabel("Total Meeting Time (s)")
    axes[-1].set_ylabel("Frequency")
    axes[-1].legend()
    axes[-1].grid(True)

    plt.tight_layout()
    # plt.savefig("Combined_Walks_and_Histogram_6.png")
    plt.show()


def calculate_rewards(walk, m, c, reward_coeff, reward_decay, max_timesteps):
    """Calculates accumulated rewards for a walk given the threshold parameters and reward function."""
    threshold = m * np.arange(len(walk)) + c
    meet_time = np.where(walk >= threshold)[0]
    if meet_time.size > 0:
        meet_time = meet_time[0] if meet_time[0] < max_timesteps else max_timesteps
        times = np.arange(meet_time + 1)
        rewards = reward_coeff * np.exp(-reward_decay * times)
        return np.sum(rewards)
    return 0


def plot_accumulated_rewards(data, thresholds, reward_settings):
    """Plot accumulated rewards across multiple threshold settings."""
    accumulated_rewards = np.zeros(data.shape[1])

    # Calculate rewards for each setting
    for (m, c, OLT), (reward_coeff, reward_decay) in zip(thresholds, reward_settings):
        for i in range(data.shape[1]):
            walk = data.iloc[:, i].dropna()
            accumulated_rewards[i] += calculate_rewards(
                walk, m, c, reward_coeff, reward_decay, 100
            )

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.bar(range(data.shape[1]), accumulated_rewards, color="skyblue")
    plt.title("Total Accumulated Rewards Across All Runs")
    plt.xlabel("Random Walk Index")
    plt.ylabel("Accumulated Reward")
    plt.show()


def plot_total_time_vs_average_reward(data, thresholds, reward_settings):
    """
    Plots the total meeting time against the average reward for each random walk.

    Args:
    data (DataFrame): DataFrame containing multiple series of random walks, each column is a walk.
    thresholds (list of tuples): Each tuple contains (m, c, OLT) for each run.
    reward_settings (list of tuples): Each tuple contains (reward_coeff, reward_decay) corresponding to each threshold.
    """
    num_walks = data.shape[1]
    total_meeting_times = np.zeros(num_walks)
    total_rewards = np.zeros(num_walks)

    # Calculate total meeting times and rewards for each walk
    for (m, c, _), (reward_coeff, reward_decay) in zip(thresholds, reward_settings):
        for i in range(num_walks):
            walk = data.iloc[:, i].dropna()
            threshold = m * np.arange(len(walk)) + c
            meet_time = np.where(walk >= threshold)[0]
            if meet_time.size > 0:
                meet_time = meet_time[0]
                times = np.arange(meet_time + 1)
                rewards = reward_coeff * np.exp(-reward_decay * times)
                total_rewards[i] += np.sum(rewards)
                total_meeting_times[i] += meet_time

    # Calculate average rewards
    average_rewards = total_rewards / len(thresholds)

    # Plotting the total meeting times vs average rewards
    plt.figure(figsize=(10, 6))
    plt.scatter(total_meeting_times, average_rewards, color="blue")
    plt.title("Total Meeting Time vs Average Reward")
    plt.xlabel("Total Meeting Time(s)")
    plt.ylabel("Average Reward")
    plt.grid(True)
    plt.show()


def plot_total_time_vs_average_reward_with_OLT(data, thresholds, reward_settings):
    """
    Plots the total meeting time against the average reward for each random walk and marks the theoretical total OLT reward.

    Args:
    data (DataFrame): DataFrame containing multiple series of random walks, each column is a walk.
    thresholds (list of tuples): Each tuple contains (m, c, OLT) for each run.
    reward_settings (list of tuples): Each tuple contains (reward_coeff, reward_decay) corresponding to each threshold.
    """
    num_walks = data.shape[1]
    total_meeting_times = np.zeros(num_walks)
    total_rewards = np.zeros(num_walks)
    total_OLT = sum(x[2] for x in thresholds)
    total_OLT_reward = 0

    # Calculate total meeting times and rewards for each walk
    for (m, c, OLT), (reward_coeff, reward_decay) in zip(thresholds, reward_settings):
        olt_times = np.arange(OLT + 1)
        olt_reward = np.sum(reward_coeff * np.exp(-reward_decay * olt_times))
        total_OLT_reward += olt_reward

        for i in range(num_walks):
            walk = data.iloc[:, i].dropna()
            threshold = m * np.arange(len(walk)) + c
            meet_time = np.where(walk >= threshold)[0]
            if meet_time.size > 0:
                meet_time = meet_time[0]
                times = np.arange(meet_time + 1)
                rewards = reward_coeff * np.exp(-reward_decay * times)
                total_rewards[i] += np.sum(rewards)
                total_meeting_times[i] += meet_time

    # Calculate average rewards
    average_rewards = total_rewards / len(thresholds)

    # Plotting the total meeting times vs average rewards and total OLT reward
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
    plt.title("Total Meeting Time(s) vs Average Reward and Total OLT Reward")
    plt.xlabel("Total Meeting Time(s)")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig("Total_Time_vs_Average_Reward_2.png")
    # plt.show()


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
                np.where(x >= m * np.arange(len(x)) + c)[0][0]
                if np.any(x >= m * np.arange(len(x)) + c)
                else max_timesteps
            ),
            axis=0,
        )
        total_meeting_times.extend(current_meeting_times)
        # print(
        #     f"Threshold: {m}, {c}, Meeting times: {current_meeting_times[:5]}"
        # )  # Debug print

    filtered_total_times = [
        4.8 * time if time < max_timesteps else max_timesteps
        for time in total_meeting_times
    ]

    # Histogram of total meeting times
    axes[-1].hist(filtered_total_times, bins=20, color="green", alpha=0.7)
    total_OLT = sum(x[2] for x in thresholds_hist)
    axes[-1].axvline(
        x=total_OLT, color="red", linestyle="--", linewidth=2, label="Total OLT"
    )
    axes[-1].set_title("Total Meeting Times Across Runs")
    axes[-1].set_xlabel("Total Meeting Time (s)")
    axes[-1].set_ylabel("Frequency")
    axes[-1].legend()
    axes[-1].grid(True)

    plt.tight_layout()
    # plt.show()
    plt.savefig("Combined_Walks_and_Histogram_7.png")


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
    # c = 8.8  # playing around optimised
    # m = -0.090909  # playing around optimised
    c = 4.994949  # linear
    m = -0.090909  # linear
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
    thresholds = [
        (-0.090909, 4.994949, 50),  # First run parameters
        (-0.180808, 3.994949, 25),  # Second run parameters
        (-0.090202, 3.45, 39),  # Third run parameters
    ]

    # thresholds = [  # Non_linear threshold
    #     (-0.090909, 8.8, 50),  # First run parameters
    #     (-0.030303, 5.4, 25),  # Second run parameters
    #     (-0.020202, 5.5, 39),  # Third run parameters
    # ]
    reward_settings = [
        (57.5, 0.11),  # Reward settings for OLT = 50
        (32.5, 0.11),  # Reward settings for OLT ≈ 24.757
        (45, 0.11),  # Reward settings for OLT ≈ 38.835
    ]

    # Define the threshold parameters for each scenario
    thresholds_line = [
        (-0.090909, 4.994949, 50),
        (-0.180808, 3.994949, 25),
        (-0.090202, 3.45, 39),
    ]

    thresholds_hist = [(-0.090909, 8.8, 50), (-0.06, 6.4, 25), (-0.020202, 4, 39)]

    # plot_walks_and_histogram_with_different_thresholds(
    #     pd_walks_data, thresholds_line, thresholds_hist
    # )

    # Assuming `data` is your DataFrame loaded with random walks
    # plot_accumulated_rewards(pd_walks_data, thresholds, reward_settings)

    # plot_total_time_vs_average_reward(pd_walks_data, thresholds, reward_settings)
    # plot_total_time_vs_average_reward_with_OLT(
    #     pd_walks_data, thresholds, reward_settings
    # )

    # plot_combined_walks_and_histogram(pd_walks_data, thresholds)

    # Example usage
    # Assuming `data` is your DataFrame loaded with random walks
    # hist_plot_model_runs_and_total_times(pd_walks_data, thresholds)

    # Calculate meeting times with controlled timesteps
    meeting_times = find_meeting_times_final(pd_walks_data, m, c, max_timesteps)
    #
    # Plot the histogram of meeting times
    plot_meeting_times_histogram(meeting_times, OLT=50)

    # m_plot_walks_meeting_threshold(
    #     pd_walks_data,
    #     y_intercept=y_intercept,
    #     gradient=gradient,
    #     OLT=50,  # 50, 24.7572815532, 38.834951456
    #     timesteps=100,
    #     num_walks_to_plot=1,
    # )
