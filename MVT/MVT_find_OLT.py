import numpy as np
import matplotlib.pyplot as plt

# activate latex
plt.rcParams.update({"text.usetex": True, "font.size": 16})


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
    # return (A * r_t * (-0.11)) / (np.exp(-0.11) - 1)
    return A * r_t * (-0.11)


if __name__ == "__main__":
    # Decay value, how fast the resources (energy) depletes in a patch.
    a = 0.075  # Decay rate for resources in a patch

    # Travel time to another patch.
    travel_t = 6  # Time it takes to travel between patches

    # Starting resources in patches: low, medium, and high quality.
    patch_start_R = np.array([32.5, 45, 57.5])

    # Environment setups: rich and poor.
    env = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    env_n = env.shape[0]

    # Simulation parameters
    measurements = 10000
    time = 50
    reso = np.linspace(1, time, measurements)
    t_diff = reso[1] - reso[0]

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
            # the denominator is the total time spent in the patch
            # and that is why reward rate reaches a maximum and then starts decreasing

    # Plotting Reward Rates for each patch type
    plt.figure(figsize=(10, 6))
    for i in range(patch_n):
        plt.plot(reso, RR[i, :], label=f"Patch Type {i+1}")
        plt.axhline(
            y=np.max(RR[i, :]),
            color="grey",
            linestyle="--",
            label=f"Max RR for Patch Types" if i == 0 else None,
        )
    plt.xlabel("Patch Residence Time (t)")
    plt.ylabel("Average Reward Rate (RR)")
    plt.title("Average Reward Rate for Each Environment")
    plt.legend()
    plt.savefig("Reward_Rate_Patch_Type.png")

    # Calculate and plot Optimal Leaving Time (OLT) - finding the max RR index for each patch type
    OLT = np.argmax(RR, axis=1)
    # print(OLT*t_diff)

    # Plotting Instantaneous Reward for all patches along with RR
    plt.figure(figsize=(10, 6))
    for i in range(patch_n):
        plt.plot(
            reso,
            E[i, :],
            label=f"Instantaneous Reward Rates For Each Patch Type" if i == 0 else None,
        )
        plt.plot(
            reso,
            RR[i, :],
            label=f"Average Reward Rate for Patch Type" if i == 0 else None,
        )
        plt.axvline(
            x=reso[OLT[i] - 75],
            color="grey",
            linestyle="--",
            label="OLT" if i == 0 else None,
        )
        plt.axhline(
            y=np.max(RR[i, :]),
            color="grey",
            linestyle="--",
            label=f"Max RR for Patch Types" if i == 0 else None,
        )

    plt.xlabel("Patch Residence Time (t)")
    plt.ylabel("Reward / Reward Rate")
    plt.title(
        "Instantaneous Reward and Average Reward Rate for Each Patch Type To Find OLT"
    )
    plt.legend()
    plt.savefig("InstantAndARROLT.png")
