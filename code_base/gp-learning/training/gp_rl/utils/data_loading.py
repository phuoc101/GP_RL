from loguru import logger
import pickle
import numpy as np
import torch


def load_data(file):
    with open(file, "rb") as f:
        return pickle.load(f)


def save_data(file, data):
    with open(file, "wb") as f:
        pickle.dump(data, f)


def get_Xy(joint, data, num_inputs):
    j = -1
    if joint == "boom":
        j = 0
    elif joint == "bucket":
        j = 1
    elif joint == "telescope":
        j = 2
    if j != -1:
        if num_inputs == 3:
            X = data["X1_xvu"][j, :, :]
            y = np.vstack((data["Y1"][j, :], data["Y2"][j, :])).T
        elif num_inputs == 2:
            X = data["X1_xu"][j, :, :]
            y = np.vstack(data["Y1"][j, :])
        return X, y
    else:
        raise AttributeError("Invalid joint value")


def load_training_data(
    data_path, num_inputs=3, joint="boom", output_torch=True, normalize=False
):
    """
    call this to load <training_data>.pkl that was saved in processing
    - normalizes data
    - converts to torch tensors
    - returns mean, std, lower and upper bounds on data
    numInputs == 2 gives inputs as (pos, u)
    numInputs == 3 gives inputs as (pos, vel, u)
    """
    training_data = load_data(file=data_path)
    # Get X, Y based on num of inputs (to GP)
    X, y = get_Xy(joint=joint, data=training_data, num_inputs=num_inputs)
    logger.debug(f"Shape X_train: {X.shape}, Shape y_train: {y.shape}")
    # normalize X, Y
    mean_states, std_states = get_mean_std(X)
    # mean_states = np.zeros(shape=std_states.shape)  # for simplicity
    if normalize:
        normalize_with_mean_std(
            X1=X, y1=y, mean_states=mean_states, std_states=std_states
        )
    else:
        mean_states = np.zeros(shape=mean_states.shape)
        std_states = np.ones(shape=std_states.shape)
    # these are "Normalized" bounds
    x_lb = np.min(X, 0)
    x_ub = np.max(X, 0)
    # multiply inputs U
    # X[:, -1] = X[:, -1] * 5

    if output_torch:
        X = torch.Tensor(X).contiguous()
        y = torch.Tensor(y).contiguous()  # for some reason this is needed
    return X, y, mean_states, std_states, x_lb, x_ub


def load_test_data(
    data_path, joint="boom", num_inputs=3, output_torch=True, normalize=False
):
    """
    call this to load <test_data>.pkl that was saved in processing
    - normalizes data
    - converts to torch tensors
    numInputs == 2 gives inputs as (pos, u)
    numInputs == 3 gives inputs as (pos, vel, u)
    """
    test_data = load_data(file=data_path)
    # Get X, Y based on num of inputs (to GP)
    X, y = get_Xy(joint=joint, data=test_data, num_inputs=num_inputs)

    logger.debug(f"Shape X_test: {X.shape}, Shape y_test: {y.shape}")
    # multiply inputs U
    # X[:, -1] = X[:, -1] * 5

    if output_torch:
        X = torch.Tensor(X).contiguous()
        y = torch.Tensor(y).contiguous()  # for some reason this is needed
    return X, y


def get_mean_std(X1):
    """normalize the data from its own mean, std - use for first collected data
        X1: Input data

    Returns:
        The mean and standard deviation of the input data
    """
    mean_states = np.mean(X1[:, :-1], 0)
    std_states = np.std(X1[:, :-1], 0)
    return mean_states, std_states


# normalize the data given mean and std, use for new data
def normalize_with_mean_std(X1, y1, mean_states, std_states):
    """Normalize the data with given mean and standard deviation
        X1 : Input data
        y1 : Output data
        mean_states : mean of input data
        std_states : standard deviation of input

    Returns:
        Normalized input and output
    """
    X = np.zeros(X1.shape)
    X[:, :-1] = np.divide(X1[:, :-1] - mean_states, std_states)
    X[:, -1] = X1[:, -1]  # control inputs are not normalised
    y = np.divide(y1, std_states)
    return X, y
