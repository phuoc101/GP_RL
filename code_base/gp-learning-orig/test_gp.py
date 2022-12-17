import pickle
import numpy as np
import logging
import matplotlib.pyplot as plt


def getMS_XY(X1):
    # normalize the data from its own mean, std - use for first collected data
    mean_states = np.mean(X1[:, :-1], 0)
    std_states = np.std(X1[:, :-1], 0)
    return mean_states, std_states


def normalize_XY_fromMS(X1, Y1, mean_states, std_states):
    X = np.zeros(X1.shape)
    X[:, :-1] = np.divide(X1[:, :-1] - mean_states, std_states)
    X[:, -1] = X1[:, -1]  # control inputs are not normalised
    Y = np.divide(Y1, std_states)
    return X, Y


def load_data(file):  # load data from Avant experiments
    with open(file, "rb") as f:
        return pickle.load(f)


def load_trainingData(dir_data, numInputs=3, normalize_=True):
    # numInputs == 2 gives inputs as (pos, u)
    # numInputs == 3 gives inputs as (pos, vel, u)
    """call this to load *TrainingData.pkl that was saved in processing
    *normalizes data
    *converts to torch tensors
    *returns mean, std, lower and upper bounds on data"""
    training_data = load_data(f"{dir_data}")
    # get X, Y based on the number of inputs (to GP) that's desired
    if numInputs == 3:
        X = training_data["X1_xvu"]
        Y = np.vstack((training_data["Y1"], training_data["Y2"])).T
    else:
        X = training_data["X1_xu"]
        Y = training_data["Y1"]
    # normalize X, Y
    mean_states, std_states = getMS_XY(X)
    mean_states = 0 * mean_states  # for simplicity
    if normalize_:
        X, Y = normalize_XY_fromMS(X, Y, mean_states, std_states)
    else:
        mean_states = mean_states * 0
        std_states = std_states * 0 + 1
    xlb = np.min(X, axis=0)  # these are "Normalized" bounds
    xub = np.max(X, axis=0)
    # multiply inputs U
    X[:, -1] = X[:, -1] * 5

    return X, Y, mean_states, std_states, xlb, xub


def main():
    logging.basicConfig(format="[%(levelname)s]: %(message)s", level=logging.INFO)
    # t, data, u = load_data_numpy("./data/avant_TrainingData.pkl")
    X, Y, mean_states, std_states, xlb, xub = load_trainingData(
        "./data/avant_TrainingData.pkl"
    )
    # X: (pos, vel, u)
    logging.info(f"X shape: {X.shape}")
    # Y: (diff between continuous pos, vel)
    logging.info(f"Y shape: {Y.shape}")
    fig, ax = plt.subplots(2)
    ax[0].plot(range(X.shape[0]), X, label=["pos", "vel", "u"])
    ax[0].legend()
    ax[1].plot(range(Y.shape[0]), Y, label=["y1", "y2"])
    ax[1].legend()
    plt.show()


if __name__ == "__main__":
    main()
