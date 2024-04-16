import tensorflow as tf
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import os

plt.rcParams.update({"text.usetex": True, "font.size": 16})


class ThresholdPolicy(tf.keras.Model):
    def __init__(self, OLT, y_value=0.5, gradient=-0.1):
        super(ThresholdPolicy, self).__init__()
        self.y_value = tf.Variable(initial_value=y_value, trainable=True)
        self.gradient = tf.Variable(initial_value=gradient, trainable=True)
        self.OLT = tf.cast(OLT, tf.float32)  # Cast OLT to float

    def call(self, t):
        """Calculate the threshold at time t."""
        t_float = tf.cast(t, tf.float32)
        return self.y_value + self.gradient * tf.minimum(t_float, self.OLT)


def simulate_walks(walks_data, model):
    leaving_times = []
    for walk in walks_data:
        for t, position in enumerate(walk):
            threshold = model(t)
            if position > threshold:
                leaving_times.append(t)
                break
    return np.array(leaving_times)


def compute_loss(leaving_times, OLT):
    average_leaving_time = tf.reduce_mean(leaving_times)
    return tf.square(average_leaving_time - OLT)


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

    # Example usage
    model = ThresholdPolicy(OLT=50, y_value=0.5, gradient=-0.1)
    optimizer = tf.optimizers.Adam(learning_rate=0.01)

    for epoch in range(100):  # Number of iterations to refine the threshold parameters
        with tf.GradientTape() as tape:
            leaving_times = simulate_walks(pd_walks_data, model)
            loss = compute_loss(leaving_times, model.OLT)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        print(f"Epoch {epoch}, Loss: {loss.numpy()}")
