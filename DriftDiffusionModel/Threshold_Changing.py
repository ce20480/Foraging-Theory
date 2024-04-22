import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import os

# import torch
# import torch.optim as optim

plt.rcParams.update({"text.usetex": True, "font.size": 16})
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


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


def calculate_average_meeting_time(df_walks, OLT, y_intercept, gradient, total_steps):
    meeting_times = []
    for _, walk in df_walks.iterrows():
        for t, position in enumerate(walk.dropna()):
            threshold = y_intercept + gradient * t
            if position >= threshold:
                meeting_times.append(t)
                break
            if t >= total_steps:
                meeting_times.append(t)  # If the walk never meets the threshold
                break

    average_meeting_time = np.mean(meeting_times) if meeting_times else None
    return average_meeting_time


def calculate_average_diff_meeting_time(
    df_walks, OLT, y_intercept, gradient, total_steps
):
    meeting_times = []
    for _, walk in df_walks.iterrows():
        for t, position in enumerate(walk.dropna()):
            threshold = y_intercept + gradient * t
            if position >= threshold:
                t = abs(t - OLT)  # Calculate the difference from the OLT
                meeting_times.append(t)
                break
            if t >= total_steps:
                t = abs(t - OLT)  # Calculate the difference from the OLT
                meeting_times.append(t)  # If the walk never meets the threshold
                break

    average_meeting_time = np.mean(meeting_times) if meeting_times else None
    return average_meeting_time


def non_linear_calculate_average_diff_meeting_time(
    df_walks, OLT, y_intercept, gradient, total_steps
):
    meeting_times = []
    for _, walk in df_walks.iterrows():
        for t, position in enumerate(walk.dropna()):
            threshold = y_intercept + gradient * (t ^ 2)
            if position >= threshold:
                t = abs(t - OLT)  # Calculate the difference from the OLT
                meeting_times.append(t)
                break
            if t >= total_steps:
                t = abs(t - OLT)  # Calculate the difference from the OLT
                meeting_times.append(t)  # If the walk never meets the threshold
                break

    average_meeting_time = np.mean(meeting_times) if meeting_times else None
    return average_meeting_time


def find_best_threshold(df_walks, OLT, y_intercepts, gradients, total_steps):
    best_params = {
        "y_intercept": None,
        "gradient": None,
        "average_meeting_time": float("inf"),
    }

    for y_intercept in y_intercepts:
        for gradient in gradients:
            avg_time = calculate_average_meeting_time(
                df_walks, OLT, y_intercept, gradient, total_steps
            )
            if avg_time and abs(avg_time - OLT) < abs(
                best_params["average_meeting_time"] - OLT
            ):
                best_params["y_intercept"] = y_intercept
                best_params["gradient"] = gradient
                best_params["average_meeting_time"] = avg_time

    return best_params


def record_threshold_data(df_walks, OLT, y_intercepts, gradients, total_steps):
    records = []

    for y_intercept in y_intercepts:
        for gradient in gradients:
            avg_time = non_linear_calculate_average_diff_meeting_time(
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
    ax.set_zlabel("Temporal Difference(s)")

    # plt.savefig("3D_threshold_data_2.png")
    plt.show()


def create_best_thresholds():
    # Assuming df_walks is a DataFrame where each row is a walk and each column is a timestep
    # Define the range for y-intercepts and gradients
    y_begin = 0  # used to be -5
    y_end = 5
    grad_begin = -0.002  # used to be between -1 and 1
    grad_end = 0
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
    # plot_3d_threshold_data(df_threshold_data)


if __name__ == "__main__":
    # how to get path of file in a folder in the same directory as the script using os
    path_to_random_walks = os.path.join((os.getcwd()), "data/random_walks.csv")
    path_to_error_data = os.path.join(
        (os.getcwd()), "data/random_walks_with_step_1000_bounds_0.0001_5.csv"
    )
    pd_walks_data = pd.read_csv(path_to_random_walks)
    pd_error_data = pd.read_csv(
        path_to_error_data
    )  # This is a data with error in the bounds

    # Visualize non-linear threshold up to 0
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
    # visualize_walks_with_linear_threshold_extended(
    #     pd_walks_data, OLT=50, initial_bound_distance=5, end_y=-5, total_steps=100
    # )

    # create_best_thresholds()
    #
    path_to_average_times = os.path.join(
        (os.getcwd()),
        "data/optimal_threshold_data_0_5_-0.002_0.csv",
        # (os.getcwd()),
        # "data/optimal_threshold_data_0_5_-2.0_0.csv",
    )
    df_threshold_average_times = pd.read_csv(path_to_average_times)
    # find the second smallest element in the average meeting time column and its row
    print(
        df_threshold_average_times.loc[
            df_threshold_average_times["average_meeting_time"].nsmallest(3).index[1]
        ]
    )
    print(
        df_threshold_average_times.loc[
            df_threshold_average_times["average_meeting_time"].idxmin()
        ]
    )
    print(df_threshold_average_times["average_meeting_time"].min())
    plot_3d_threshold_data(df_threshold_average_times)
