import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def dynamic_threshold_around_OLT(
    current_time, OLT, initial_bound_distance, min_bound_distance
):
    """
    Calculate dynamic thresholds that decrease as current_time approaches OLT
    and then stabilize or change behavior post-OLT.
    """
    if current_time < OLT:
        # Example: Linear decrease towards OLT
        progress_ratio = current_time / OLT
        distance = initial_bound_distance - (
            (initial_bound_distance - min_bound_distance) * progress_ratio
        )
    else:
        # Keep threshold constant after OLT or implement another rule
        distance = min_bound_distance  # This keeps the threshold constant after OLT
    upper_bound = starting_position + distance
    lower_bound = starting_position - distance
    return upper_bound, lower_bound


def visualize_walks_meeting_threshold_before_OLT_old(
    df_walks, OLT, initial_bound_distance, min_bound_distance
):
    fig, ax = plt.subplots(figsize=(12, 7))
    threshold_meet_before_OLT = []

    # Calculate dynamic thresholds up to the OLT
    upper_thresholds = [
        dynamic_threshold_around_OLT(
            t, OLT, initial_bound_distance, min_bound_distance
        )[0]
        for t in range(OLT)
    ]
    lower_thresholds = [
        dynamic_threshold_around_OLT(
            t, OLT, initial_bound_distance, min_bound_distance
        )[1]
        for t in range(OLT)
    ]

    # Determine which walks meet the threshold before the OLT
    for index, row in df_walks.iterrows():
        walk = row.dropna().values[:OLT]
        if any(walk > upper_thresholds) or any(walk < lower_thresholds):
            threshold_meet_before_OLT.append(index)

    # Plot walks that meet the condition
    for index in threshold_meet_before_OLT:
        walk = df_walks.loc[index].dropna().values[:OLT]
        ax.plot(range(OLT), walk, label=f"Walk {index}")

    # Plot dynamic thresholds
    ax.plot(range(OLT), upper_thresholds, "--", color="grey", label="Upper Threshold")
    ax.plot(range(OLT), lower_thresholds, "--", color="grey", label="Lower Threshold")

    # Highlight the Optimal Leaving Time (OLT)
    ax.axvline(
        x=OLT - 1,
        color="red",
        linestyle="-",
        linewidth=2,
        label="Optimal Leaving Time (OLT)",
    )

    ax.set_title("Walks Meeting Threshold Before OLT")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Position")
    # ax.legend(loc="upper right")
    plt.tight_layout()


def visualize_walks_meeting_upper_threshold_before_OLT(
    df_walks, OLT, initial_bound_distance, min_bound_distance
):
    fig, ax = plt.subplots(figsize=(12, 7))

    # Calculate dynamic upper threshold up to the OLT
    upper_thresholds = [
        dynamic_threshold_around_OLT(
            t, OLT, initial_bound_distance, min_bound_distance
        )[0]
        for t in range(OLT)
    ]

    # Track walks that meet the upper threshold before the OLT and their stopping points
    walks_meeting_threshold = []
    stop_points = []

    for index, row in df_walks.iterrows():
        walk = row.dropna().values
        stop_point = OLT  # Default to OLT if threshold not met
        for i, position in enumerate(walk):
            if i >= OLT:  # Only consider up to OLT
                break
            if position > upper_thresholds[i]:  # Threshold met
                stop_point = i
                walks_meeting_threshold.append(index)
                stop_points.append(stop_point)
                break

    # Plot walks that meet the upper threshold condition
    for index, stop_point in zip(walks_meeting_threshold, stop_points):
        walk = (
            df_walks.loc[index].dropna().values[:stop_point]
        )  # Include the point where threshold is met
        ax.plot(range(len(walk)), walk, label=f"Walk {index}")

    # Plot dynamic upper threshold
    ax.plot(range(OLT), upper_thresholds, "--", color="orange", label="Upper Threshold")

    # Highlight the Optimal Leaving Time (OLT)
    ax.axvline(
        x=OLT,
        color="red",
        linestyle="-",
        linewidth=2,
        label="Optimal Leaving Time (OLT)",
    )

    ax.set_title("Walks Meeting Upper Threshold Before OLT")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Position")
    # ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def dynamic_threshold_non_linear(t, OLT, initial_bound_distance, min_bound_distance):
    """
    Calculate a non-linearly decreasing dynamic threshold as t approaches OLT.
    The threshold decreases more rapidly as it gets closer to OLT.
    """
    if t >= OLT:
        return min_bound_distance
    else:
        # Non-linear decrease: e.g., exponential decay towards the OLT
        # Adjust the rate of decay with the exponent to fit the desired urgency
        urgency_factor = (
            5  # Adjust this factor to control how quickly the threshold decreases
        )
        time_ratio = 1 - ((t / OLT) ** urgency_factor)
        distance = (
            min_bound_distance
            + (initial_bound_distance - min_bound_distance) * time_ratio
        )
        return distance


def visualize_walks_meeting_non_linear_threshold_before_OLT(
    df_walks, OLT, initial_bound_distance, min_bound_distance, num_walks=10
):
    fig, ax = plt.subplots(figsize=(12, 7))

    upper_thresholds = [
        dynamic_threshold_non_linear(t, OLT, initial_bound_distance, min_bound_distance)
        for t in range(OLT)
    ]

    # Choose a subset of walks to visualize
    sample_indices = np.random.choice(df_walks.index, size=num_walks, replace=False)

    for index in sample_indices:
        walk = (
            df_walks.loc[index].dropna().values[:OLT]
        )  # Ensure walk is limited to OLT for comparison
        # Determine the first point where the walk exceeds the threshold, if at all
        for i, pos in enumerate(walk):
            if pos > upper_thresholds[i]:
                stop_point = i  # Stop at first point where threshold is exceeded
                break
        else:
            stop_point = (
                len(walk) - 1
            )  # Use the last point if threshold is never exceeded

        # Correct the plotting range to align x and y dimensions
        ax.plot(range(stop_point), walk[:stop_point], label=f"Walk {index}")

    ax.plot(
        range(OLT),
        upper_thresholds,
        "--",
        color="orange",
        label="Dynamic Upper Threshold",
    )

    ax.axvline(
        x=OLT - 1,
        color="red",
        linestyle="-",
        linewidth=2,
        label="Optimal Leaving Time (OLT)",
    )

    ax.set_title("Walks Meeting Non-linear Upper Threshold Before OLT")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Position")
    # ax.legend()
    plt.tight_layout()
    plt.show()


def visualize_walks_hitting_upper_threshold_before_OLT(
    df_walks, OLT, initial_bound_distance, min_bound_distance, num_walks=100
):
    fig, ax = plt.subplots(figsize=(12, 7))

    # Define the dynamic upper threshold across the time steps up to OLT
    upper_thresholds = [
        dynamic_threshold_non_linear(t, OLT, initial_bound_distance, min_bound_distance)
        for t in range(OLT)
    ]

    # Iterate over walks and visualize only those hitting the threshold before OLT
    for index, row in df_walks.iterrows():
        walk = (
            row.dropna().values
        )  # Get the walk data, dropping NA for incomplete walks
        walk_length = min(len(walk), OLT)  # Limit the walk to OLT for comparison
        for i in range(walk_length):
            if (
                walk[i] > upper_thresholds[i]
            ):  # Check if the walk hits the upper threshold
                # Plot the walk up to the point it hits the threshold
                ax.plot(range(i), walk[:i], label=f"Walk {index}")
                break  # Stop after visualizing up to the threshold hit

    # Plot the dynamic upper threshold
    ax.plot(
        range(OLT),
        upper_thresholds,
        "--",
        color="orange",
        label="Dynamic Upper Threshold",
    )

    # Highlight the OLT
    ax.axvline(
        x=OLT,
        color="red",
        linestyle="-",
        linewidth=2,
        label="Optimal Leaving Time (OLT)",
    )

    ax.set_title("Walks Hitting Upper Threshold Before OLT")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Position")
    # ax.legend()  # Optional: Comment out if the legend is too crowded
    plt.tight_layout()
    plt.show()
