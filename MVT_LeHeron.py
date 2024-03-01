# import numpy as np
# import matplotlib.pyplot as plt
#
# # Code to solve LeHeron marginal value theorem problem
# # Assumed predefined variables and functions based on your MATLAB code
# env_n = 2  # Number of environments, e.g., rich and poor
# patch_n = 3  # Number of patch types
# reso = np.linspace(1, 50, 10000)  # Resolution from 1 to 50 with 10000 measurements
# env = np.array([[0.2, 0.3, 0.5], [0.5, 0.3, 0.2]])  # Environment probabilities
# patch_start_R = np.array([32.5, 45, 57.5])  # Starting rewards for patches
# travel_t = 6  # Travel time between patches
#
#
# # Define Python functions corresponding to MATLAB functions
# def patch_gain(t, A):
#     return A * np.exp(-0.11 * t)
#
#
# def patch_gain_total(t, A):
#     return A * (1 - np.exp(-0.11 * t)) / (1 - np.exp(-0.11))
#
#
# def der_patch_gain(t, A):
#     return (A * np.exp(-0.11 * t) * np.log(np.exp(-0.11))) / (np.exp(-0.11) - 1)
#
#
# def reward_rate(p, t, g, d):
#     return (p * g) / (d + p * t)
#
#
# def optimal_lt(A, y):
#     return np.log(y * (np.exp(-0.11) - 1) / (A * np.log(np.exp(-0.11)))) / np.log(
#         np.exp(-0.11)
#     )
#
#
# # Initialize arrays for storing calculated values
# RRE = np.zeros((env_n, len(reso)))
# RRE_patch = np.zeros((env_n, len(reso), patch_n))
# total_E = np.zeros((patch_n, len(reso)))  # Placeholder for total energy gained
#
# # Calculate total energy gained for simplicity in this context
# for i, A in enumerate(patch_start_R):
#     total_E[i, :] = patch_gain_total(reso, A)
#
# # Calculate RRE for both environments
# for env_type in range(env_n):
#     for p_type in range(patch_n):
#         for t in range(len(reso)):
#             RRE[env_type, t] += env[env_type, p_type] * (
#                 total_E[p_type, t] / (travel_t + reso[t])
#             )
#             RRE_patch[env_type, t, p_type] = env[env_type, p_type] * (
#                 total_E[p_type, t] / (travel_t + reso[t])
#             )
#
# # Assuming patch_start_R and reso are defined as before
# instant_R = np.zeros((len(patch_start_R), len(reso)))
#
# for i, A in enumerate(patch_start_R):
#     instant_R[i, :] = der_patch_gain(reso, A)
#
# # Calculate the maximum RRE to determine the optimal leaving time
# maxRRE = RRE.max(axis=1)
#
#
# # Assuming necessary variables and functions (patch_gain, patch_gain_total, etc.) are defined as before
#
#
# def plot_environment_reward_rate(RRE, maxRRE, reso):
#     """
#     Plots the Reward Rate for Environments.
#
#     Parameters:
#     - RRE: Reward Rate for Environments array.
#     - maxRRE: Maximum RRE values for determining optimal leaving time.
#     - reso: Resolution array for time.
#     """
#     plt.figure(figsize=(10, 6))
#     for i in range(RRE.shape[0]):
#         plt.plot(reso, RRE[i, :], label=f"Environment {i+1} RRE")
#         plt.axhline(y=maxRRE[i], color="gray", linestyle="--", label=f"Max RRE {i+1}")
#     plt.xlabel("Patch Residence Time (t)")
#     plt.ylabel("Reward Rate (RR)")
#     plt.title("Reward Rate for Each Environment")
#     plt.legend()
#     plt.show()
#
#
# def plot_patch_reward_rate(RRE_patch, env, reso):
#     """
#     Plots the Reward Rate for each Patch within environments.
#
#     Parameters:
#     - RRE_patch: Reward Rate for each Patch within environments array.
#     - env: Environment probabilities array.
#     - reso: Resolution array for time.
#     """
#     for env_type in range(RRE_patch.shape[0]):
#         plt.figure(figsize=(10, 6))
#         for p_type in range(RRE_patch.shape[2]):
#             plt.plot(
#                 reso,
#                 RRE_patch[env_type, :, p_type],
#                 label=f"Patch {p_type+1} in Environment {env_type+1}",
#             )
#         plt.xlabel("Patch Residence Time (t)")
#         plt.ylabel("Reward Rate (RR)")
#         plt.title(f"Reward Rate for Patches in Environment {env_type+1}")
#         plt.legend()
#         plt.show()
#
#
# def plot_instantaneous_reward(instant_R, reso):
#     """
#     Plots the instantaneous reward for all patches.
#
#     Parameters:
#     - instant_R: Instantaneous reward rate array for each patch type.
#     - reso: Resolution array for time.
#     """
#     plt.figure(figsize=(10, 6))
#     for i in range(instant_R.shape[0]):
#         plt.plot(reso, instant_R[i, :], label=f"Instantaneous Reward for Patch {i+1}")
#     plt.xlabel("Patch Residence Time (t)")
#     plt.ylabel("Instantaneous Reward Rate")
#     plt.title("Instantaneous Reward Rate for All Patches")
#     plt.legend()
#     plt.show()
#
#
# # Example usage:
# # Ensure RRE, maxRRE, RRE_patch, and instant_R are calculated before calling these functions
# plot_environment_reward_rate(RRE, maxRRE, reso)
# plot_patch_reward_rate(RRE_patch, env, reso)
# plot_instantaneous_reward(instant_R, reso)
"""






# Calculating RRE and RRE_patch
for env_type in range(env_n):
    for p_type in range(patch_n):
        for t_idx, t in enumerate(reso):
            total_E = patch_gain_total(t, patch_start_R[p_type])
            RRE[env_type, t_idx] += env[env_type, p_type] * total_E / (travel_t + t)
            RRE_patch[env_type, t_idx, p_type] = (
                env[env_type, p_type] * total_E / (travel_t + t)
            )

# Finding max RRE to identify the optimal leaving time
maxRRE = np.max(RRE, axis=1)

# Calculating OLT for each environment based on RRE
# Identifying the index (time) where RRE matches max RRE for each environment
OLT_indices = [np.where(RRE[i] == maxRRE[i])[0][0] for i in range(env_n)]
OLT_times = reso[OLT_indices]  # Converting indices to actual times

# Since Python doesn't use 1-based indexing like MATLAB, adjust OLT_indices if necessary
print("Optimal Leaving Times (OLT) for environments:", OLT_times)

# Plotting RRE for all environments and patches within those environments
for i in range(env_n):
    plt.figure()
    plt.plot(reso, RRE[i, :], label="Environment RRE")
    for p_type in range(patch_n):
        plt.plot(reso, RRE_patch[i, :, p_type], "--", label=f"Patch {p_type+1}")
        # OLT calculation for each patch type within the environment (assuming optimal_lt returns a scalar)
        OLT2 = optimal_lt(patch_start_R[p_type], maxRRE[i])
    plt.axhline(y=maxRRE[i], color="red", linestyle="--")  # Marking max RRE
    plt.xlabel("Patch Residence Time (t)")
    plt.ylabel("RR")
    plt.title("RR for Environment (Background RR)")
    plt.legend()
    plt.show()

# Assuming RR is calculated similarly to RRE but for each patch type individually
RR = np.random.rand(patch_n, len(reso))  # Example RR data for each patch type

plt.figure()
for i in range(patch_n):
    plt.plot(reso, RR[i, :], label=f"Patch {i+1} RR")
plt.xlabel("Patch Residence Time (t)")
plt.ylabel("RR")
plt.title("RR for All Patch Types")
plt.legend()
plt.show()

plt.figure()
for i in range(env_n):
    plt.plot(reso, RRE[i, :], label=f"Environment {i+1} RRE")
for i in range(patch_n):
    plt.plot(reso, instant_R[i, :], "--", label=f"Instantaneous Reward Patch {i+1}")
plt.xlabel("Patch Residence Time (t)")
plt.ylabel("RR / Instantaneous Reward")
plt.title("Overlap of RRE and Instantaneous Rewards")
plt.legend()
plt.show()

# Plotting RRE for each environment and marking the max RRE with a horizontal line
for i in range(env_n):
    plt.figure(figsize=(10, 6))  # Create a new figure for each environment
    plt.plot(reso, RRE[i, :], label=f"Environment {i+1} RRE")  # Plot RRE
    plt.axhline(
        y=maxRRE[i], color="red", linestyle="--", label="Max RRE"
    )  # Mark max RRE
    plt.xlabel("Patch Residence Time (t)")
    plt.ylabel("Reward Rate (RR)")
    plt.title(f"RR for Environment {i+1} (Background RR)")
    plt.legend()
    plt.show()
"""
import numpy as np
import matplotlib.pyplot as plt

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


# Calculate max RRE for each environment to determine OLT
maxRRE = np.max(RRE, axis=1)
OLT_indices = np.argmax(RRE, axis=1)  # Indices of max RRE for each environment
OLT_times = reso[OLT_indices]  # Convert indices to actual times for OLT

print("Optimal Leaving Times (OLT) for environments:", OLT_times)


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
        plt.plot(RRE[i, :], label=f"Environment {i+1} RRE")
        plt.xlabel("Patch Residence Time (t)")
        plt.ylabel("RR")
        plt.title("RR for Environment (Background RR)")

    # Overlay instantaneous reward rates
    for i in range(instant_R.shape[0]):
        plt.plot(instant_R[i, :], "--", label=f"Instant R Patch {i+1}")
        plt.xlabel("Patch Residence Time (t)")
        plt.ylabel("Foreground Reward Rate")

    # Draw horizontal lines to indicate RRE leave points
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
        plt.axhline(y=max_rre, color=color, linestyle="--", label=f"Max RRE Env {i+1}")

    plt.legend()
    plt.show()


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
    plt.ylabel("Foreground")
    plt.title("Foreground Instantaneous Reward for All Patches")

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
        plt.axhline(y=max_rre, color=color, linestyle="--", label=f"Max RRE Env {i+1}")

    plt.legend()
    plt.show()


# _ = plot_RR_for_environments_with_patches(RRE, RRE_patch, maxRRE, patch_start_R)
# print(_)
# plot_RR_for_all_patches(RR)
# plot_overlap_RRE_and_instant_R(RRE, instant_R, maxRRE)
# plot_instant_R_with_maxRRE(instant_R, maxRRE)


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


# Assuming RRE and RRE_patch are defined according to your data structure
OLT = calculate_OLT_for_patches_across_environments(instant_R, maxRRE, reso)

# Print the calculated OLTs
for key, olt_time in OLT.items():
    if olt_time is not None:
        print(
            f"Environment {key[0] + 1}, Patch {key[1] + 1}: OLT is at time {olt_time}"
        )
    else:
        print(
            f"Environment {key[0] + 1}, Patch {key[1] + 1}: No OLT found within given resolution"
        )
