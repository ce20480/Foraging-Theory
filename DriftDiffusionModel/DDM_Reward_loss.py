import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def dynamic_threshold_linear_extended(
    t, OLT, initial_bound_distance, end_y, total_steps
):
    """
    Calculate a dynamic threshold that linearly decreases towards 'end_y'.
    This linear change happens across the entire time frame up to 'total_steps'.
    """
    if t <= total_steps:
        # Linear decrease from initial_bound_distance to end_y across total_steps
        return (
            initial_bound_distance
            + ((end_y - initial_bound_distance) / total_steps) * t
        )
    else:
        # Beyond total_steps, the threshold stays at end_y
        return end_y


def visualize_walks_with_linear_threshold_extended(
    df_walks, OLT, initial_bound_distance, end_y, total_steps
):
    fig, ax = plt.subplots(figsize=(12, 7))

    # Calculate the dynamic threshold across all steps, extending beyond OLT
    thresholds = [
        dynamic_threshold_linear_extended(
            t, OLT, initial_bound_distance, end_y, total_steps
        )
        for t in range(total_steps + 1)
    ]

    # Iterate through walks and plot those meeting the threshold
    for index, row in df_walks.iterrows():
        walk = row.dropna().values
        walk_length = min(len(walk), total_steps)
        hit_threshold = False
        for i in range(walk_length):
            if walk[i] > thresholds[i]:
                # Plot up to the point where the walk hits the threshold
                ax.plot(range(i), walk[:i], label=f"Walk {index}")
                hit_threshold = True
                break
        if not hit_threshold:
            # If the walk does not hit the threshold, plot the entire walk
            ax.plot(
                range(walk_length), walk[:walk_length], label=f"Walk {index}", alpha=0.5
            )

    # Plot the dynamic threshold
    ax.plot(thresholds, "--", color="orange", label="Dynamic Linear Threshold")

    # Highlight the OLT
    ax.axvline(
        x=OLT,
        color="red",
        linestyle="-",
        linewidth=2,
        label="Optimal Leaving Time (OLT)",
    )

    ax.set_title("Walks Meeting Linear Threshold")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Position")
    plt.tight_layout()
    plt.show()


def calculate_reward_loss_simple(walks, thresholds, OLT, reward_per_step):
    """
    Calculate the reward loss for walks not meeting the threshold at OLT.

    :param walks: DataFrame of walks.
    :param thresholds: List of threshold values over time.
    :param OLT: Optimal Leaving Time.
    :param reward_per_step: Reward obtained per time step.
    :return: DataFrame with reward loss information.
    """
    reward_losses = []

    for index, row in walks.iterrows():
        walk = row.dropna().values
        time_of_exit = next(
            (i for i, pos in enumerate(walk[:OLT]) if pos > thresholds[i]), OLT
        )
        optimal_reward = OLT * reward_per_step
        actual_reward = time_of_exit * reward_per_step
        reward_loss = optimal_reward - actual_reward
        reward_losses.append(
            {"walk_id": index, "reward_loss": reward_loss, "time_of_exit": time_of_exit}
        )

    return pd.DataFrame(reward_losses)


def calculate_reward_loss(walks, thresholds, OLT, reward_per_step):
    """
    Calculate the reward loss for walks not meeting the threshold at OLT.

    :param walks: DataFrame of walks.
    :param thresholds: List of threshold values over time.
    :param OLT: Optimal Leaving Time.
    :param reward_per_step: Reward obtained per time step.
    :return: DataFrame with reward loss information.
    """
    reward_losses = []

    for index, row in walks.iterrows():
        walk = row.dropna().values
        time_of_exit = next(
            (i for i, pos in enumerate(walk[:OLT]) if pos > thresholds[i]), OLT
        )
        optimal_reward = OLT * reward_per_step
        actual_reward = time_of_exit * reward_per_step[index]
        reward_loss = optimal_reward - actual_reward
        reward_losses.append(
            {"walk_id": index, "reward_loss": reward_loss, "time_of_exit": time_of_exit}
        )

    return pd.DataFrame(reward_losses)


def generate_linear_thresholds(OLT, start_threshold, end_threshold, total_steps):
    """
    Generate an array of threshold values that linearly interpolate between
    start_threshold and end_threshold, reaching end_threshold at OLT, and then
    linearly decreasing or increasing towards the next threshold.

    :param OLT: Optimal Leaving Time - the time at which the threshold reaches end_threshold.
    :param start_threshold: The starting threshold value at time 0.
    :param end_threshold: The threshold value at OLT.
    :param total_steps: Total number of steps in the simulation.
    :return: A list of threshold values for each time step.
    """
    thresholds = []
    # Linearly decrease the threshold from start_threshold to end_threshold up to OLT
    for t in range(OLT):
        threshold_t = start_threshold + (end_threshold - start_threshold) * (t / OLT)
        thresholds.append(threshold_t)

    # After OLT, you can choose to keep the threshold constant or adjust it further.
    # Here, it is kept constant at end_threshold for simplicity.
    for t in range(OLT, total_steps):
        thresholds.append(end_threshold)

    return thresholds


def calculate_reward_loss_adjusted(
    walks, thresholds, OLT, reward_function, total_steps
):
    """
    Adjusted function to calculate the reward loss for walks not meeting the threshold at OLT.

    :param walks: DataFrame or similar structure with walks data.
    :param thresholds: Array-like structure with threshold values over time.
    :param OLT: Optimal Leaving Time.
    :param reward_function: Function to calculate reward at a given time step.
    :param total_steps: Total number of time steps considered.
    :return: DataFrame with adjusted reward loss information.
    """
    reward_losses = []

    # Loop through each walk
    for index, walk in walks.iterrows():
        walk_data = walk.dropna().values
        time_of_exit = None

        # Find the time step where the walk first exceeds the threshold
        for t in range(min(len(walk_data), total_steps)):
            if walk_data[t] > thresholds[t]:
                time_of_exit = t
                break

        # If the walk never exceeds the threshold, use the last observed time step
        if time_of_exit is None:
            time_of_exit = min(len(walk_data), total_steps) - 1

        # Calculate the actual and optimal rewards
        actual_reward = sum([reward_function(t) for t in range(1, time_of_exit + 1)])
        optimal_reward = sum([reward_function(t) for t in range(1, OLT + 1)])

        # Calculate reward loss
        reward_loss = optimal_reward - actual_reward

        reward_losses.append(
            {"walk_id": index, "reward_loss": reward_loss, "time_of_exit": time_of_exit}
        )

    return pd.DataFrame(reward_losses)


# Define a simple linear reward function for demonstration purposes
def reward_function(t, type="diminishing"):
    match type:
        case "diminishing":
            return 1 - 0.01 * t  # Example: diminishing reward per time step
        case "exponential":
            return 0.065 * np.exp(
                -0.11 * t
            )  # Example: exponential reward per time step
        case "constant":
            return 1  # constant reward per time step


if __name__ == "__main__":
    pd_walks_data = pd.read_csv("../data/random_walks.csv")
    pd_error_data = pd.read_csv(
        "../data/random_walks_with_step_1000_bounds_0.0001_5.csv"
    )
    # visualise linear threshold that goes negative and beyond OLT
    # visualize_walks_with_linear_threshold_extended(
    #     pd_walks_data, OLT=50, initial_bound_distance=5, end_y=-5, total_steps=100
    # )

    # Example usage
    # thresholds = np.linspace(5, -5, pd_walks_data.index.size)
    # time = np.linspace(0, pd_walks_data.index.size, pd_walks_data.index.size)
    # A = 10  # Amplitude of reward
    # Lambda = 0.11  # Decay rate of reward
    # reward_per_step = 1  # Simplified reward per time step
    # reward_loss_df = calculate_reward_loss_simple(
    #     pd_walks_data, thresholds, OLT=50, reward_per_step=reward_per_step
    # )
    # reward_per_step = [
    #     A * np.exp(-Lambda * t) for t in time
    # ]  # exponential reward per time step
    # reward_loss_df = calculate_reward_loss(
    #     pd_walks_data, thresholds, OLT=50, reward_per_step=reward_per_step
    # )
    # print(reward_loss_df.head())

    # Example usage:
    OLT = 50  # Optimal Leaving Time
    start_threshold = 5  # Starting threshold
    end_threshold = -5  # Threshold at OLT
    total_steps = pd_walks_data.index.size  # Total duration of the simulation

    # Generate the dynamic thresholds
    thresholds = generate_linear_thresholds(
        OLT, start_threshold, end_threshold, total_steps
    )

    # Now you can pass these thresholds to the adjusted function
    reward_loss_df = calculate_reward_loss_adjusted(
        pd_walks_data, thresholds, OLT, reward_function, total_steps
    )

    print(reward_loss_df.head())
