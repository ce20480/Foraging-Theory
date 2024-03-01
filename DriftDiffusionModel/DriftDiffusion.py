# This is my python implementation of the drift-diffusion model

# Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ast
from scipy import stats
from scipy.special import erf

# from scipy.stats import lognorm, norm, probplot

"""
random imports from copilot
# import matplotlib.animation as animation
# import matplotlib.cm as cm
# import matplotlib.colors as colors
# import matplotlib.colorbar as colorbar
# import matplotlib.patches as patches
# import matplotlib.path as path
# import matplotlib.lines as lines
# import matplotlib.patheffects as path_effects
# import matplotlib.gridspec as gridspec
# import matplotlib.ticker as ticker
# import matplotlib.pylab as pylab
# import matplotlib as mpl
# import scipy.stats as stats
# import scipy.optimize as optimize
"""


def drift_diffusion_model(
    starting_position, drift_variable, variance, Upper_Bound, Lower_Bound, plot=True
):
    """
    This function runs the drift-diffusion model
    Inputs:
      starting_position: the starting position of the model
      drift_variable: the drift variable of the model
      variance: the variance of the model
      Upper_Bound: the upper bound of the model
      Lower_Bound: the lower bound of the model
      plot: a boolean variable that determines whether or not to plot the model
    Outputs:
      position: the final position of the model
      time: the time it took for the model to reach its final position
    """
    x_position = starting_position
    mean = drift_variable
    if plot:
        x_walk = np.array([])
        x_walk = np.append(x_walk, x_position)
        while x_position < Upper_Bound and x_position > Lower_Bound:
            x_position += np.random.normal(mean, variance)
            x_walk = np.append(x_walk, x_position)
        time_length = len(x_walk) - 1
        time = np.linspace(0, time_length, len(x_walk))
        return time, x_walk
    else:
        while x_position < Upper_Bound and x_position > Lower_Bound:
            x_position += np.random.normal(mean, variance)
        t = len(x_position) - 1
        return t, x_position


# @title Helper Functions


def simulate_and_plot_SPRT_fixedtime(mu, sigma, stop_time, num_sample, verbose=True):
    """Simulate and plot a SPRT for a fixed amount of time given a std.

    Args:
      mu (float): absolute mean value of the symmetric observation distributions
      sigma (float): Standard deviation of the observations.
      stop_time (int): Number of steps to run before stopping.
      num_sample (int): The number of samples to plot.
    """

    evidence_history_list = []
    if verbose:
        print("Trial\tTotal_Evidence\tDecision")
    for i in range(num_sample):
        evidence_history, decision, Mvec = simulate_SPRT_fixedtime(mu, sigma, stop_time)
        if verbose:
            print("{}\t{:f}\t{}".format(i, evidence_history[-1], decision))
        evidence_history_list.append(evidence_history)

    fig, ax = plt.subplots()
    maxlen_evidence = np.max(list(map(len, evidence_history_list)))
    ax.plot(np.zeros(maxlen_evidence), "--", c="red", alpha=1.0)
    for evidences in evidence_history_list:
        ax.plot(np.arange(len(evidences)), evidences)
        ax.set_xlabel("Time")
        ax.set_ylabel("Accumulated log likelihood ratio")
        ax.set_title(
            "Log likelihood ratio trajectories under the fixed-time " + "stopping rule"
        )

    plt.show(fig)


def simulate_and_plot_SPRT_fixedthreshold(mu, sigma, num_sample, alpha, verbose=True):
    """Simulate and plot a SPRT for a fixed amount of times given a std.

    Args:
      mu (float): absolute mean value of the symmetric observation distributions
      sigma (float): Standard deviation of the observations.
      num_sample (int): The number of samples to plot.
      alpha (float): Threshold for making a decision.
    """
    # calculate evidence threshold from error rate
    threshold = threshold_from_errorrate(alpha)

    # run simulation
    evidence_history_list = []
    if verbose:
        print("Trial\tTime\tAccumulated Evidence\tDecision")
    for i in range(num_sample):
        evidence_history, decision, Mvec = simulate_SPRT_threshold(mu, sigma, threshold)
        if verbose:
            print(
                "{}\t{}\t{:f}\t{}".format(i, len(Mvec), evidence_history[-1], decision)
            )
        evidence_history_list.append(evidence_history)

    fig, ax = plt.subplots()
    maxlen_evidence = np.max(list(map(len, evidence_history_list)))
    ax.plot(np.repeat(threshold, maxlen_evidence + 1), c="red")
    ax.plot(-np.repeat(threshold, maxlen_evidence + 1), c="red")
    ax.plot(np.zeros(maxlen_evidence + 1), "--", c="red", alpha=0.5)

    for evidences in evidence_history_list:
        ax.plot(np.arange(len(evidences) + 1), np.concatenate([[0], evidences]))

    ax.set_xlabel("Time")
    ax.set_ylabel("Accumulated log likelihood ratio")
    ax.set_title("Log likelihood ratio trajectories under the threshold rule")

    plt.show(fig)


def simulate_and_plot_accuracy_vs_threshold(mu, sigma, threshold_list, num_sample):
    """Simulate and plot a SPRT for a set of thresholds given a std.

    Args:
      mu (float): absolute mean value of the symmetric observation distributions
      sigma (float): Standard deviation of the observations.
      alpha_list (float): List of thresholds for making a decision.
      num_sample (int): The number of samples to plot.
    """
    accuracies, decision_speeds = simulate_accuracy_vs_threshold(
        mu, sigma, threshold_list, num_sample
    )

    # Plotting
    fig, ax = plt.subplots()
    ax.plot(decision_speeds, accuracies, linestyle="--", marker="o")
    ax.plot([np.amin(decision_speeds), np.amax(decision_speeds)], [0.5, 0.5], c="red")
    ax.set_xlabel("Average Decision speed")
    ax.set_ylabel("Average Accuracy")
    ax.set_title("Speed/Accuracy Tradeoff")
    ax.set_ylim(0.45, 1.05)

    plt.show(fig)


def threshold_from_errorrate(alpha):
    """Calculate log likelihood ratio threshold from desired error rate `alpha`

    Args:
      alpha (float): in (0,1), the desired error rate

    Return:
      threshold: corresponding evidence threshold
    """
    threshold = np.log((1.0 - alpha) / alpha)
    return threshold


def simulate_SPRT_fixedtime(mu, sigma, stop_time, true_dist=1):
    """Simulate a Sequential Probability Ratio Test with fixed time stopping
        rule. Two observation models are 1D Gaussian distributions N(1,sigma^2) and
        N(-1,sigma^2).

        Args:
          mu (float): absolute mean value of the symmetric observation distributions
          sigma (float): Standard deviation of observation models
          stop_time (int): Number of samples to take before stopping
          true_dist (1 or -1): Which state is the true state.

        Returns:
          evidence_history (numpy vector): the history of cumulated evidence given
                                            generated data
          decision (int): 1 for s = 1, -1 for s = -1
          Mvec (numpy vector): the generated sequences of measurement data in this trial
    run below to see if this works
    # Set random seed
    # np.random.seed(100)

    # Set model parameters
    # mu = 0.2
    # sigma = 3.5  # standard deviation for p+ and p-
    # num_sample = 10  # number of simulations to run
    # stop_time = 150  # number of steps before stopping

    # Simulate and visualize
    # simulate_and_plot_SPRT_fixedtime(mu, sigma, stop_time, num_sample)
    """

    # Set means of observation distributions
    assert mu > 0, "Mu should be > 0"
    mu_pos = mu
    mu_neg = -mu

    # Make observation distributions
    p_pos = stats.norm(loc=mu_pos, scale=sigma)
    p_neg = stats.norm(loc=mu_neg, scale=sigma)

    # Generate a random sequence of measurements
    if true_dist == 1:
        Mvec = p_pos.rvs(size=stop_time)
    else:
        Mvec = p_neg.rvs(size=stop_time)

    # Calculate log likelihood ratio for each measurement (delta_t)
    # ll_ratio_vec = log_likelihood_ratio(Mvec, p_neg, p_pos)

    # STEP 1: Calculate accumulated evidence (S) given a time series of evidence (hint: np.cumsum)
    # evidence_history = np.cumsum(ll_ratio_vec)
    #
    # # STEP 2: Make decision based on the sign of the evidence at the final time.
    # decision = np.sign(evidence_history[-1])
    #
    # return evidence_history, decision, Mvec


def create_random_walks_csv(
    starting_position,
    drift_variable,
    variance,
    Upper_Bound,
    Lower_Bound,
    num_walks=1000,
    store_csv=True,
):
    """
    This function creates a csv file with num_walks(base=1000) random walks
    Inputs:
        starting_position: the starting position of the model
        drift_variable: the drift variable of the model
        variance: the variance of the model
        Upper_Bound: the upper bound of the model
        Lower_Bound: the lower bound of the model
        num_walks: the number of random walks to create
        store_csv: a boolean variable that determines whether or not to store the csv
    Outputs:
        pd_data: a pandas dataframe of the
        csv file with num_walks random walks if store_csv is True
    """
    walks = []
    for i in range(num_walks):
        _, x_walk = drift_diffusion_model(
            starting_position, drift_variable, variance, Upper_Bound, Lower_Bound
        )
        walks.append(x_walk)
    pd_data = pd.DataFrame(walks)
    if store_csv:
        pd_data.to_csv("random_walks.csv", sep=",", index=False)
    return pd_data


def check_ddm_func():
    """
    Checks the drift_diffusion_model function
    """
    time, x_walk = drift_diffusion_model(
        starting_position, drift_variable, variance, Upper_Bound, Lower_Bound, plot
    )
    print(pd.DataFrame(time))
    print(pd.DataFrame(x_walk))
    pd_data = pd.DataFrame({"time": time, "x_walk": x_walk})
    print(pd_data)


def check_walk_save_func():
    """
    Checks the create_random_walks_csv function
    """
    pd_walks_data = create_random_walks_csv(
        starting_position, drift_variable, variance, Upper_Bound, Lower_Bound, 2
    )
    print(pd_walks_data)


def create_random_walks(num_walks, store_csv):
    """
    Creates a csv file with num_walks random walks
    Inputs:
        num_walks: the number of random walks to create
        store_csv: a boolean variable that determines whether or not to store the csv
    Outputs:
        pd_data: a pandas dataframe of the csv file with num_walks random walks
    """
    create_random_walks_csv(
        starting_position,
        drift_variable,
        variance,
        Upper_Bound,
        Lower_Bound,
        num_walks,
        store_csv,
    )


# def threshold_error(pd_walks_data, threshold):
#     """
#     Finds the error_rate at a certain threshold for the random walks
#     Inputs:
#         pd_walks_data: a pandas dataframe of the csv file with num_walks random walks
#         threshold: the threshold to find the error rate at
#     Outputs:
#         error_rate: the error rate at the threshold
#     """
#     iterations = len(pd_walks_data)
#     error = np.zeros(iterations)
#     for i in range(iterations):
#         final_walk_value = pd_walks_data.iloc[i].dropna().iloc[-1]
#         if final_walk_value > threshold:
#             error[i] = 1
#         elif final_walk_value < -threshold:
#             error[i] = 0
#         else:
#             raise ValueError("The threshold is too small")
#     correct_rate = np.sum(error) * 100 / iterations
#     error_rate = 100 - correct_rate
#     return error_rate


# def new_threshold_error(pd_walks_data, threshold, correct_threshold="higher"):
#     """
#     Finds the error_rate at a certain threshold for the random walks
#     Inputs:
#         pd_walks_data: a pandas dataframe of the csv file with num_walks random walks
#         threshold: the threshold to find the error rate at
#         correct_threshold: a string that determines whether the threshold is higher or lower
#     Outputs:
#         error_rate: the error rate at the threshold
#     """
#     if correct_threshold == "higher":
#         pd_correct = pd_walks_data[pd_walks_data > threshold]
#     elif correct_threshold == "lower":
#         pd_correct = pd_walks_data[pd_walks_data < -threshold]
#     else:
#         raise ValueError("Please choose correct threshold either higher or lower")
#     iterations = len(pd_walks_data)
#     correct_num = pd_correct.any(axis=1).apply(int).sum()
#     correct_rate = correct_num * 100 / iterations
#     error_rate = 100 - correct_rate
#     return error_rate


# def new_threshold_time(pd_walks_data, threshold):
#     """
#     Finds the error_rate at a certain threshold for the random walks
#     Inputs:
#         pd_walks_data: a pandas dataframe of the csv file with num_walks random walks
#         threshold: the threshold to find the error rate at
#     Outputs:
#         error_rate: the error rate at the threshold
#     """
#     pd_higher = pd_walks_data[pd_walks_data > threshold]
#     pd_lower = pd_walks_data[pd_walks_data < -threshold]
#     iterations = len(pd_walks_data)
#     x_higher = np.zeros(iterations)
#     x_lower = np.zeros(iterations)
#     time_higher = np.zeros(iterations)
#     time_lower = np.zeros(iterations)
#     for i in range(iterations):
#         if pd_higher.iloc[i].dropna().tolist() != []:
#             x_higher[i] = pd_higher.iloc[i].dropna().index[0]
#         elif pd_lower.iloc[i].dropna().tolist() != []:
#             x_lower[i] = pd_lower.iloc[i].dropna().index[0]
#         else:
#             raise ValueError("The threshold is too small")
#     return x_higher, x_lower, time_higher, time_lower


def find_first_occurrence(row, threshold):
    """
    Finds the first occurrence of a value in a row that is greater than the threshold. Used with apply on a dataframe
    Inputs:
        row: a row of a dataframe
        threshold: the threshold to find the first occurence of a value greater than it
    Outputs:
        (idx, value): a tuple of the index and value of the first occurrence of a value greather than the threshold
    """
    for idx, value in enumerate(row):
        if value > threshold or value < -threshold:
            return (idx, value)
    # If no such value is found, return a default value (e.g., 0)
    return 0


def create_threshold_meet_pd(pd_walks_data, Bound):
    """
    Finds the error of the random walks
    Inputs:
        pd_walks_data: a pandas dataframe of the csv file with num_walks random walks
        Bound: the threshold to find the error rate at
    Outputs:
        pd_walks_data: a pandas dataframe of the csv file with num_walks random walks
    """
    column_name = f"{Bound}"
    df = pd.DataFrame()
    df[column_name] = pd_walks_data.apply(
        find_first_occurrence, threshold=Bound, axis=1
    )
    return df


def find_error_and_time(df, Bound, correct_threshold="higher"):
    col = df[f"{Bound}"]
    # Split the column into two columns
    value_elements = col.apply(lambda x: x[1])
    index_elements = col.apply(lambda x: x[0])
    # Check which elements are positive (greater than 0)
    if correct_threshold == "higher":
        error_elements = value_elements < 0
        # Count the number of error elements
        error_count = error_elements.sum()
    elif correct_threshold == "lower":
        error_elements = value_elements > 0
        # Count the number of error elements
        error_count = error_elements.sum()
    else:
        raise ValueError("Please choose correct threshold either higher or lower")
    # Calculate the proportion of errors
    error_rate = error_count * 100 / len(col)
    average_time = index_elements.mean()
    return error_rate, average_time


def find_x_trajectory(df, pd_walks_data, Bound, correct_threshold=None):
    """
    Finds the x trajectory of the random walks
    Inputs:
        df: a pandas dataframe of the csv file with num_walks random walks
        pd_walks_data: a pandas dataframe of the csv file with num_walks random walks
        Bound: the threshold to find the error rate at
        correct_threshold: a string that determines whether the threshold is higher or lower
    Outputs:
        x_walk_df: a pandas dataframe of the x trajectory of the random walks
    """
    col = df[f"{Bound}"]
    index_elements = col.apply(lambda x: x[0])
    value_elements = col.apply(lambda x: x[1])
    if correct_threshold != None:
        if correct_threshold == "higher":
            correct_elements = value_elements > 0
        elif correct_threshold == "lower":
            correct_elements = value_elements < 0
        else:
            raise ValueError("Please choose correct threshold either higher or lower")
        pd_correct = pd_walks_data[correct_elements]
        x_correct_indexes = index_elements[correct_elements]
        x_correct_df = pd_correct.apply(
            lambda row: pd.Series(row.iloc[: x_correct_indexes[row.name] + 1].tolist()),
            axis=1,
        )
        pd_incorrect = pd_walks_data[~correct_elements]
        x_incorrect_indexes = index_elements[~correct_elements]
        x_incorrect_df = pd_incorrect.apply(
            lambda row: pd.Series(
                row.iloc[: x_incorrect_indexes[row.name] + 1].tolist()
            ),
            axis=1,
        )
    # Create a dataframe of the x trajectory of the random walks
    x_walk_df = pd_walks_data.apply(
        lambda row: pd.Series(row.iloc[: index_elements[row.name] + 1].tolist()), axis=1
    )
    if correct_threshold != None:
        times_correct_df = index_elements[correct_elements]
        times_incorrect_df = index_elements[~correct_elements]
        return (
            x_walk_df,
            x_correct_df,
            x_incorrect_df,
            times_correct_df,
            times_incorrect_df,
        )
    else:
        return x_walk_df


def plot_x_trajectory(dfs, num_rows):
    """
    Plots the x trajectory of the random walks
    Inputs:
        dfs: a list of pandas dataframes of the x trajectory of the random walks
        num_rows: the number of rows to plot
    Outputs:
        None
    """
    # x_walk_df = find_x_trajectory(pd_indexes, pd_walks_data, Bound)
    if isinstance(dfs, list):
        correct_dfs = dfs[0].head(num_rows)
        incorrect_dfs = dfs[1].head(num_rows)
        if len(dfs) > 2:
            correct_times_df = dfs[2].head(num_rows)
            incorrect_times_df = dfs[3].head(num_rows)
            # make a histogram of the times and create a subplot for it where correct is on top and incorrect is on bottom with line graph in the middle
            fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(8, 8))
            plt.subplots_adjust(
                wspace=0.5, hspace=0.5
            )  # Adjust the space between subplots
            blue_solid = (0.2, 0.2, 1.0, 1)
            blue = (0.2, 0.2, 1.0, 0.2)
            red = (1.0, 0.2, 0.2, 0.2)
            red_solid = (1.0, 0.2, 0.2, 1)

            # Fit a normal distribution to the data
            # mu_correct, std_correct = norm.fit(correct_times_df.values)
            # xmin_correct, xmax_correct = axs[0].get_xlim()
            # x_correct = np.linspace(xmin_correct, xmax_correct, 100)
            # p_correct = norm.pdf(x_correct, mu_correct, std_correct)

            # Fit lognormal distribution to the data
            # shape_correct, loc_correct, scale_correct = lognorm.fit(
            #     correct_times_df.values
            # )
            # p_correct = lognorm.pdf(
            #     x_correct, shape_correct, loc=loc_correct, scale=scale_correct
            # )
            # axs[0].plot(x_correct, p_correct)

            axs[0].hist(correct_times_df.values, bins=100, color=blue_solid)
            axs[0].locator_params(axis="x", nbins=10)  # Adjust nbins as needed
            axs[0].set_title("Correct Times")
            axs[0].set_xlabel("Reaction Time")
            axs[0].set_ylabel("Frequency")

            # Fit a lognormal distribution to the data
            # shape_incorrect, loc_incorrect, scale_incorrect = lognorm.fit(
            #     incorrect_times_df.values
            # )
            # xmin_incorrect, xmax_incorrect = axs[2].get_xlim()
            # x_incorrect = np.linspace(xmin_incorrect, xmax_incorrect, 100)
            # Fit lognormal distribution to the data
            # p_incorrect = lognorm.pdf(
            #     x_incorrect, shape_incorrect, loc=loc_incorrect, scale=scale_incorrect
            # )

            # Fit a normal distribution to the data
            # mu_incorrect, std_incorrect = norm.fit(incorrect_times_df.values)
            # xmin_incorrect, xmax_incorrect = axs[0].get_xlim()
            # p_incorrect = norm.pdf(x_incorrect, mu_incorrect, std_incorrect)
            # axs[2].plot(x_incorrect, p_incorrect)

            axs[2].hist(incorrect_times_df.values, bins=100, color=red)
            axs[2].locator_params(axis="x", nbins=10)  # Adjust nbins as needed
            axs[2].set_title("Incorrect Times")
            axs[2].set_xlabel("Reaction Time")
            axs[2].set_ylabel("Frequency")

            axs[1].locator_params(axis="x", nbins=10)  # Adjust nbins as needed
            axs[1].set_xlabel("Reaction Time")
            axs[1].set_ylabel("Value")
            # Plot each correct row
            for idx, row in correct_dfs.iterrows():
                axs[1].plot(row.index, row.values, color=blue_solid)
            # Plot each incorrect row
            for idx, row in incorrect_dfs.iterrows():
                axs[1].plot(row.index, row.values, color=red)
        else:
            # Plot each correct row
            for idx, row in correct_dfs.iterrows():
                plt.plot(row.index, row.values, color="blue")
            # Plot each incorrect row
            for idx, row in incorrect_dfs.iterrows():
                plt.plot(row.index, row.values, color="red")
    else:
        # df = x_walk_df.iloc[:num_rows, :]
        df = dfs.head(num_rows)
        # Plot each row
        for idx, row in df.iterrows():
            plt.plot(row.index, row.values)
        # Customize the plot
    plt.show()


def check_subplot_graph(pd_walks_data, Bound):
    pd_indexes = create_threshold_meet_pd(pd_walks_data, Bound)
    (
        x_walk_df,
        x_correct_df,
        x_incorrect_df,
        times_correct_df,
        times_incorrect_df,
    ) = find_x_trajectory(pd_indexes, pd_walks_data, Bound, correct_threshold="higher")
    dfs = [x_correct_df, x_incorrect_df, times_correct_df, times_incorrect_df]
    plot_x_trajectory(dfs, 1000)


def save_bounds_csv(pd_walks_data, num_steps, bound_lower_limit, bound_higher_limit):
    """
    Saves the csv file of the random walks with the bounds
    Inputs:
        pd_walks_data: a pandas dataframe of the random walks
        num_steps: the number of steps in the random walks
        bound_lower_limit: the lower limit of the bounds
        bound_higher_limit: the higher limit of the bounds
    Outputs:
        None
    """
    if bound_lower_limit <= 0:
        print("Please choose a bound lower limit greater than 0")
        bound_lower_limit = 0.001
    limit = np.linspace(bound_lower_limit, bound_higher_limit, num_steps)
    dfs_to_concat = []
    for Bound in limit:
        pd_index = create_threshold_meet_pd(pd_walks_data, Bound)

        # Dynamically create a new column with the name based on the variable Bound
        column_name = f"{Bound}"

        # Create a new DataFrame or Series for each iteration
        new_df = pd.DataFrame(pd_index, columns=[column_name])

        # Append to the list
        dfs_to_concat.append(new_df)

    pd_indexes = pd.concat(dfs_to_concat, axis=1)
    # save the csv file
    pd_indexes.to_csv(
        f"random_walks_with_step_{num_steps}_bounds_{bound_lower_limit}_{bound_higher_limit}.csv"
    )
    # return pd_indexes


def error_time_threshold_graph(pd_indexes, correct_threshold="higher"):
    bounds = pd_indexes.columns[1:]
    errors = np.zeros(len(bounds))
    times = np.zeros(len(bounds))
    for idx, bound in enumerate(bounds):
        pd_indexes[bound] = pd_indexes[bound].apply(ast.literal_eval)
        error_rate, average_time = find_error_and_time(
            pd_indexes, bound, correct_threshold=correct_threshold
        )
        errors[idx] = error_rate
        times[idx] = average_time
    return errors, times


def create_speed_accuracy_graph(pd_error_data):
    errors, times = error_time_threshold_graph(pd_error_data)
    plt.figure(figsize=(6, 6))
    plt.plot(times, errors)
    plt.xlabel("Reaction Time")
    plt.ylabel("Error Rate")
    plt.title("Error Rate vs Reaction Time taken to reach threshold")
    plt.locator_params(axis="x", nbins=10)  # Adjust nbins as needed
    plt.show()


def find_error_bound(pd_walks_data, Bound):
    pd_indexes = create_threshold_meet_pd(pd_walks_data, Bound)
    error_rate, average_time = find_error_and_time(
        pd_indexes, Bound, correct_threshold="higher"
    )
    print(f"error rate: {error_rate}%")
    print(f"average time: {round(average_time, 2)}")


if __name__ == "__main__":
    # np.random.seed(0)
    starting_position = 0
    drift_variable = 0.01
    variance = 0.1
    Upper_Bound = 5
    Lower_Bound = -5
    Bound = 1  # the bound for the random walks but do slightly higher then intended threshold
    plot = True
    num_walks = 10000
    store_csv = True
    num_steps = 1000
    bound_lower_limit = 0.0001
    bound_higher_limit = 5
    # check_ddm_func()
    # check_walk_save_func()
    # create_random_walks(num_walks, store_csv) # now stored 10,000 walks with drift variable 0.01 and thresholds max at 5 and min at -5
    # save_bounds_csv(pd_walks_data, num_steps, bound_lower_limit, bound_higher_limit) with bounds 0.0001 to 5 with 1000 steps
    pd_walks_data = pd.read_csv("../data/random_walks.csv")
    pd_error_data = pd.read_csv(
        "../data/random_walks_with_step_1000_bounds_0.0001_5.csv"
    )
    # find_error_bound(
    #     pd_walks_data, Bound
    # )  # dont use pd_error_data because the column names do not have integer values
    # create_speed_accuracy_graph(pd_error_data)
    check_subplot_graph(pd_walks_data, Bound)
