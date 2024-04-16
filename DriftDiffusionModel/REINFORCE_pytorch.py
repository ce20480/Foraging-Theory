import tensorflow as tf
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import os


class Policy(torch.nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.params = torch.nn.Parameter(
            torch.tensor([1.0, -0.01])
        )  # y_value, gradient

    def forward(self, t, OLT):
        y_value, gradient = self.params
        if t < OLT:
            return y_value + gradient * t
        return y_value + gradient * OLT  # Example logic for threshold


def run_simulation(df_walks, OLT, policy):
    times = []
    for index, row in df_walks.iterrows():
        walk = row.dropna().values
        for t, pos in enumerate(walk):
            threshold = policy.forward(t, OLT)
            if pos > threshold:  # Walk meets or exceeds the threshold
                times.append(t)
                break
        else:
            # If threshold is never met, you might consider the full length or handle differently
            times.append(len(walk))

    average_time = sum(times) / len(times)
    return times, average_time


def compute_reward(average_time, OLT):
    # Reward is higher when average time is closer to OLT
    # Example: negative exponential of the absolute difference
    return -abs(average_time - OLT)


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

    OLT = 50

    policy = Policy()
    optimizer = optim.Adam(policy.parameters(), lr=0.01)

    # Example training loop
    for episode in range(1000):
        optimizer.zero_grad()
        # Simulate data or use real data here
        times, average_time = run_simulation(pd_walks_data, OLT, policy)
        reward = compute_reward(average_time, OLT)
        (-reward).backward()  # Maximize reward by minimizing negative reward
        optimizer.step()

    print("Learned parameters:", policy.params)
