import numpy as np
import matplotlib.pyplot as plt


def plot_RR_for_environments_with_patches(RRE, RRE_patch, maxRRE, patch_start_R):
    env_n, _, patch_n = RRE_patch.shape
    for env_type in range(env_n):
        plt.figure(figsize=(10, 7))
        plt.plot(RRE[env_type, :], label=f"Environment {env_type+1} RRE")
        for p_type in range(patch_n):
            plt.plot(RRE_patch[env_type, :, p_type], "--", label=f"Patch {p_type+1}")
            # Calculate OLT for each patch using maxRRE for the current environment
            OLT = optimal_lt(patch_start_R[p_type], maxRRE[env_type])
        plt.axhline(y=maxRRE[env_type], color="red", linestyle="--", label="Max RRE")
        plt.xlabel("Patch Residence Time (t)")
        plt.ylabel("RR")
        plt.title(f"RR for Environment {env_type+1} (Background RR)")
        plt.legend()
        plt.show()

    return OLT


def plot_RR_for_all_patches(RR):
    plt.figure(figsize=(10, 7))
    for i in range(RR.shape[0]):
        plt.plot(RR[i, :], label=f"Patch {i+1} RR")
    plt.xlabel("Patch Residence Time (t)")
    plt.ylabel("RR")
    plt.title("RR for All Patch Types")
    plt.legend()
    plt.show()


def plot_overlap_RRE_and_instant_R(RRE, instant_R, maxRRE):
    """
    Plots the overlap of Reward Rate for Environments (RRE) and instantaneous reward rates,
    and draws horizontal lines to indicate maximum RRE values.

    Parameters:
    - RRE: A 2D array where each row represents an environment's RRE over time.
    - instant_R: A 2D array where each row represents an instantaneous reward rate for a patch type over time.
    - maxRRE: An array of maximum RRE values for each environment.
    """
    plt.figure(figsize=(10, 7))

    # Plot RRE for each environment
    for i in range(RRE.shape[0]):
        plt.plot(RRE[i, :], "--", label=f"Environment {i+1} RRE")
        plt.xlabel("Patch Residence Time (t)")
        plt.ylabel("RR")
        plt.title("ARR And Instantaneous RR For LeHeron Task To Find OLT")

    # Overlay instantaneous reward rates
    for i in range(instant_R.shape[0]):
        plt.plot(instant_R[i, :], "-", color="Black", label=f"Instant R Patch {i+1}")
        plt.xlabel("Patch Residence Time (t)")
        plt.ylabel("Instantaneous Reward Rate")

    # Draw horizontal lines to indicate RRE leave points
    colors = [
        "red",
        # "green",
        # "blue",
        # "cyan",
        # "magenta",
        # "orange",
        # "yellow",
    ]  # Extend or loop through colors as needed
    for i, max_rre in enumerate(maxRRE):
        color = colors[
            i % len(colors)
        ]  # Cycle through colors if there are more lines than defined colors
        plt.axhline(
            y=max_rre,
            color=color,
            linestyle="-",
            label=f"Max RRE Env {i+1}",
            alpha=1 / (i + 1),
        )

    plt.legend()
    plt.savefig("Overlap_RRE_Instant_R.png")


def plot_instant_R_with_maxRRE(instant_R, maxRRE):
    """
    Plots the foreground instantaneous reward for all patches and draws horizontal lines
    to indicate the maximum RRE values for each environment.

    Parameters:
    - instant_R: A 2D array where each row represents an instantaneous reward rate for a patch type over time.
    - maxRRE: An array of maximum RRE values for each environment.
    """
    plt.figure(figsize=(10, 7))

    # Plot the instantaneous reward for each patch type
    for i in range(instant_R.shape[0]):
        plt.plot(instant_R[i, :], label=f"Instant Reward Patch {i+1}")
    plt.xlabel("Patch Residence Time (t)")
    plt.ylabel("Instantaneous Reward Rate")
    plt.title("Instantaneous Reward for All Patches")

    # Draw horizontal lines to indicate max RRE for each environment
    colors = [
        "red",
        "green",
        "blue",
        "cyan",
        "magenta",
        "yellow",
    ]  # Extend or loop through colors as needed
    for i, max_rre in enumerate(maxRRE):
        color = colors[
            i % len(colors)
        ]  # Cycle through colors if there are more lines than defined colors
        plt.axhline(y=max_rre, color=color, linestyle="-", label=f"Max RRE Env {i+1}")

    plt.legend()
    plt.show()


def calculate_OLT_for_patches_across_environments(instant_R, maxRRE, reso):
    """
    Calculate the Optimal Leaving Time (OLT) for each patch across environments, using the
    instantaneous reward rate (instant_R) and comparing it against the maximum RRE of each environment.

    Parameters:
    - instant_R: A 2D numpy array with shape (patch_n, reso_len) for the instantaneous reward rate of patches.
    - maxRRE: A 1D numpy array with the maximum RRE for each environment.
    - reso: A numpy array representing the time points resolution.

    Returns:
    - OLT: A dictionary with keys as (environment index, patch index) and values as OLT times.
    """
    OLT = {}
    patch_n, reso_len = instant_R.shape
    env_n = len(maxRRE)  # Assuming one maxRRE value per environment

    for env_index in range(env_n):
        for patch_index in range(patch_n):
            # Find the first index where instant_R for a patch falls below the maxRRE of the environment
            below_max_indices = np.where(instant_R[patch_index, :] < maxRRE[env_index])[
                0
            ]

            if below_max_indices.size > 0:
                # If there's at least one point where instant_R is below maxRRE, record the time
                OLT[(env_index, patch_index)] = reso[below_max_indices[0]]
            else:
                # If instant_R never falls below maxRRE within the observed timeframe
                OLT[(env_index, patch_index)] = None

    return OLT


if __name__ == "__main__":
    # Setting patch reward rate calculations
    # Decay value, how fast the resources (energy) depletes in a patch.
    a = 0.075  # Decay rate for resources in a patch

    # Travel time to another patch.
    travel_t = 6  # Time it takes to travel between patches

    # Starting resources in patches: low, medium, and high quality.
    patch_start_R = np.array([32.5, 45, 57.5])

    # Environment setups: rich and poor.
    env = np.array([[0.2, 0.3, 0.5], [0.5, 0.3, 0.2]])
    env_n = env.shape[0]  # Number of environment types

    # Simulation parameters
    measurements = 10000
    time = 50
    reso = np.linspace(1, time, measurements)

    # Assuming patch_start_R, reso, travel_t are defined
    patch_n = len(patch_start_R)
    reso_len = len(reso)
    travel_t = 6  # Defined as per the MATLAB code

    # Initialize arrays
    E = np.zeros((patch_n, reso_len))
    total_E = np.zeros((patch_n, reso_len))
    instant_R = np.zeros((patch_n, reso_len))
    RR = np.zeros((patch_n, reso_len))

    # Patch gain = energy you gain in patch at single timepoint t
    def patch_gain(t, A):
        return A * np.exp(-0.11 * t)

    # Total energy gained so far up to time point t
    def patch_gain_total(t, A):
        r = np.exp(-0.11)
        r_t = np.exp(-0.11 * t)
        return A * (1 - r_t) / (1 - r)

    # Foreground reward rate = derivative of patch gain
    def der_patch_gain(t, A):
        r = np.exp(-0.11)
        r_t = np.exp(-0.11 * t)  # r^t
        return (A * r_t * np.log(r)) / (r - 1)

    # Environmental reward rate
    def reward_rate(p, t, g, d):
        return (p * g) / (d + p * t)

    # Derivative of the reward rate
    def der_reward_rate(p1, p2, a1, a2, t, A, tr):
        r = np.exp(-0.11)
        r_t = np.exp(-0.11 * t)  # r^t
        return ((a2 * p2 + a1 * p1) * (r_t * (np.log(r) * (t + tr) - 1) + 1)) / (
            (r - 1) * (t + tr) ** 2
        )

    # Optimal leaving time based on a given function
    def optimal_lt(A, y):
        r = np.exp(-0.11)
        return np.log(y * (r - 1) / (A * np.log(r))) / np.log(r)

    # Calculation loop
    for type in range(patch_n):
        for t in range(reso_len):
            E[type, t] = patch_gain(reso[t], patch_start_R[type])
            total_E[type, t] = patch_gain_total(reso[t], patch_start_R[type])
            instant_R[type, t] = der_patch_gain(reso[t], patch_start_R[type])
            RR[type, t] = total_E[type, t] / (travel_t + reso[t])

    """
    Now onto calculating MVT for LeHeron rich and poor environments
    """

    # Initialize arrays for Reward Rate in Environments (RRE) and patches
    env_n = len(env)
    patch_n = len(patch_start_R)
    RRE = np.zeros((env_n, len(reso)))
    RRE_patch = np.zeros((env_n, len(reso), patch_n))

    # Calculate RRE and RRE_patch
    for env_type in range(env_n):
        for p_type in range(patch_n):
            for t_idx, t in enumerate(reso):
                # Use pre-calculated total_E for the current patch type and time point
                current_total_E = total_E[p_type, t_idx]

                # Update RRE and RRE_patch using pre-calculated total energy
                RRE[env_type, t_idx] += (
                    env[env_type, p_type] * current_total_E / (travel_t + t)
                )
                RRE_patch[env_type, t_idx, p_type] = (
                    env[env_type, p_type] * current_total_E / (travel_t + t)
                )

    maxRRE = np.max(RRE, axis=1)
    # _ = plot_RR_for_environments_with_patches(RRE, RRE_patch, maxRRE, patch_start_R)
    # print(_)
    # plot_RR_for_all_patches(RR)
    plot_overlap_RRE_and_instant_R(RRE, instant_R, maxRRE)
    # plot_instant_R_with_maxRRE(instant_R, maxRRE)

    # Assuming RRE and RRE_patch are defined according to your data structure
    # OLT = calculate_OLT_for_patches_across_environments(instant_R, maxRRE, reso)
    #
    # # Print the calculated OLTs
    # for key, olt_time in OLT.items():
    #     if olt_time is not None:
    #         print(
    #             f"Environment {key[0] + 1}, Patch {key[1] + 1}: OLT is at time {olt_time}"
    #         )
    #     else:
    #         print(
    #             f"Environment {key[0] + 1}, Patch {key[1] + 1}: No OLT found within given resolution"
    #         )
