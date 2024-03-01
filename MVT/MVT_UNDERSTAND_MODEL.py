import numpy as np
import matplotlib.pyplot as plt


# Function to calculate energy gained at a single time point in a patch
def patch_gain(t, A):
    return A * np.exp(-0.11 * t)


# Function to calculate total energy gained up to time point t
def patch_gain_total(t, A):
    r_t = np.exp(-0.11 * t)
    return A * (1 - r_t) / (1 - np.exp(-0.11))


# Function to calculate derivative of patch gain (instantaneous reward rate)
def der_patch_gain(t, A):
    r_t = np.exp(-0.11 * t)
    return (A * r_t * np.log(np.exp(-0.11))) / (np.exp(-0.11) - 1)


if __name__ == "__main__":
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

    # Calculation loop
    for type in range(patch_n):
        for t in range(reso_len):
            E[type, t] = patch_gain(reso[t], patch_start_R[type])
            total_E[type, t] = patch_gain_total(reso[t], patch_start_R[type])
            instant_R[type, t] = der_patch_gain(reso[t], patch_start_R[type])
            RR[type, t] = total_E[type, t] / (travel_t + reso[t])

    # Plotting Reward Rates for each patch type
    plt.figure(figsize=(10, 6))
    for i in range(patch_n):
        plt.plot(reso, RR[i, :], label=f"Patch Type {i+1}")
        plt.axhline(y=np.max(RR[i, :]), color="grey", linestyle="--")
    plt.xlabel("Patch Residence Time (t)")
    plt.ylabel("Reward Rate (RR)")
    plt.title("Reward Rate for Each Patch Type")
    plt.legend()
    plt.show()

    # Calculate and plot Optimal Leaving Time (OLT) - finding the max RR index for each patch type
    OLT = np.argmax(RR, axis=1)

    # Plotting Instantaneous Reward for all patches along with RR
    plt.figure(figsize=(10, 6))
    for i in range(patch_n):
        plt.plot(
            reso, instant_R[i, :], label=f"Instantaneous Reward for Patch Type {i+1}"
        )
        plt.plot(reso, RR[i, :], label=f"Reward Rate for Patch Type {i+1}")
        plt.axvline(x=reso[OLT[i]], color="grey", linestyle="--")
    plt.xlabel("Patch Residence Time (t)")
    plt.ylabel("Reward / Reward Rate")
    plt.title("Instantaneous Reward and Reward Rate for Each Patch Type")
    plt.legend()
    plt.show()
