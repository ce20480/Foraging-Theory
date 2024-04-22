import numpy as np
import pandas as pd
import os

# Load the data
path_to_random_walks = os.path.join((os.getcwd()), "data/random_walks.csv")
random_walks = pd.read_csv(path_to_random_walks)  # Adjust the path as necessary

# Constants
OLT = 50  # Optimal Leaving Time
alpha = 0.0001  # Learning rate
epsilon = 0.01  # Small value for finite difference approximation


# Function to simulate the interaction and find the crossing time
def simulate_interaction(random_walks, m, c):
    num_steps = random_walks.shape[1]
    time_steps = np.arange(num_steps)
    threshold = m * np.square(time_steps) + c
    crossing_times = np.full(random_walks.shape[0], np.nan)
    for i in range(random_walks.shape[0]):
        cross_indices = np.where(random_walks.iloc[i] >= threshold)[0]
        if len(cross_indices) > 0:
            crossing_times[i] = cross_indices[0]
    return crossing_times


# Reward function
def calculate_rewards(crossing_times, OLT):
    valid_times = crossing_times[~np.isnan(crossing_times)]
    rewards = -np.abs(valid_times - OLT)
    return np.mean(rewards)


# Gradient estimation using finite differences
def estimate_gradient(random_walks, m, c, epsilon):
    current_reward = calculate_rewards(simulate_interaction(random_walks, m, c), OLT)
    reward_m = calculate_rewards(
        simulate_interaction(random_walks, m + epsilon, c), OLT
    )
    reward_c = calculate_rewards(
        simulate_interaction(random_walks, m, c + epsilon), OLT
    )
    grad_m = (reward_m - current_reward) / epsilon
    grad_c = (reward_c - current_reward) / epsilon
    return grad_m, grad_c


# Initial parameters
m = -0.1  # Start with some arbitrary values
c = 3.0

# Training loop
num_iterations = 100  # Set the number of iterations as needed
for iteration in range(num_iterations):
    grad_m, grad_c = estimate_gradient(random_walks, m, c, epsilon)
    m += alpha * grad_m
    c += alpha * grad_c
    if iteration % 10 == 0:
        reward = calculate_rewards(simulate_interaction(random_walks, m, c), OLT)
        print(f"Iteration {iteration}: m = {m:.4f}, c = {c:.4f}, Reward = {reward:.4f}")

print(f"Final Parameters - m: {m:.4f}, c: {c:.4f}")
