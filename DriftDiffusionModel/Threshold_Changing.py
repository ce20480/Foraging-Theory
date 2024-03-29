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


# Now trying to extend threshold to exist for negative thresholds as well
def dynamic_threshold_extended(t, OLT, initial_bound_distance, end_y, total_steps):
    """
    Calculate a dynamic threshold that decreases towards 'end_y' after OLT.
    The decrease is linear from the initial bound distance to 'end_y'.
    """
    if t < OLT:
        # Up to OLT, decrease non-linearly as before
        time_ratio = 1 - ((t / OLT) ** 5)  # Example non-linearity
        return initial_bound_distance + (end_y - initial_bound_distance) * time_ratio
    else:
        # Beyond OLT, linearly decrease to 'end_y'
        extended_ratio = (t - OLT) / (total_steps - OLT)
        return end_y + (initial_bound_distance - end_y) * (1 - extended_ratio)


def visualize_walks_with_extended_threshold(
    df_walks, OLT, initial_bound_distance, end_y, total_steps
):
    fig, ax = plt.subplots(figsize=(12, 7))

    # Calculate the dynamic threshold across all steps, extending beyond OLT
    thresholds = [
        dynamic_threshold_extended(t, OLT, initial_bound_distance, end_y, total_steps)
        for t in range(total_steps)
    ]

    walks_meeting_threshold = []

    # Determine termination point for each walk based on the extended threshold
    for index, row in df_walks.iterrows():
        walk = row.dropna().values
        for i, pos in enumerate(walk):
            if i < len(thresholds) and pos > thresholds[i]:
                walks_meeting_threshold.append(
                    (index, i)
                )  # Record walk index and termination point
                ax.plot(range(i + 1), walk[: i + 1], label=f"Walk {index}")
                break

    # Plot the dynamic threshold
    ax.plot(thresholds, "--", color="orange", label="Dynamic Upper Threshold")

    # Highlight the OLT
    ax.axvline(
        x=OLT,
        color="red",
        linestyle="-",
        linewidth=2,
        label="Optimal Leaving Time (OLT)",
    )

    ax.set_title("Walks Meeting Extended Threshold Before and After OLT")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Position")
    plt.tight_layout()
    plt.show()


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


if __name__ == "__main__":
    pd_walks_data = pd.read_csv("../data/random_walks.csv")
    pd_error_data = pd.read_csv(
        "../data/random_walks_with_step_1000_bounds_0.0001_5.csv"
    )
    # # Visualize non-linear threshold up to 0
    # visualize_walks_hitting_upper_threshold_before_OLT(
    #     pd_walks_data,
    #     OLT=50,
    #     initial_bound_distance=1,
    #     min_bound_distance=0.1,
    #     num_walks=10000,
    # )
    # Visualize non-linear threshold with negative values as well
    # visualize_walks_with_extended_threshold(
    #     pd_walks_data, OLT=50, initial_bound_distance=5, end_y=-5, total_steps=100
    # )
    visualize_walks_with_linear_threshold_extended(
        pd_walks_data, OLT=50, initial_bound_distance=5, end_y=-5, total_steps=100
    )
