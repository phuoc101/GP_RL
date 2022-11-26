import pickle
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np


def visualize_data(data, filename):
    logger.debug(f"Position readings count: {len(data['X1_xvu'][:,0])}")
    logger.debug(f"Velocity readings count: {len(data['X1_xvu'][:,1])}")
    logger.debug(f"Input cmd count: {len(data['X1_xvu'][:,2])}")
    title = f"Plots with data from bag {filename}"
    fig, ax = plt.subplots(2, 2)
    fig.suptitle(title)
    ax[0, 0].plot(data["timestamp"], data["X1_xvu"][:, 0], label="boom position (rad)")
    ax[0, 0].plot(data["timestamp"], data["X1_xvu"][:, 1], label="boom velocity")
    ax[0, 0].plot(data["timestamp"], data["X1_xvu"][:, 2], label="input u")
    ax[0, 1].plot(data["timestamp"], data["Y1"], label="d_boom_pos (rad)")
    ax[1, 0].plot(data["timestamp"], data["Y2"], label="d_boom_vel")
    ax[1, 1].plot(np.diff(data["timestamp"]), label="timestmp_diff (sec)")
    for a in ax:
        for x in a:
            x.legend()
    plt.legend()
    plt.show()


def load_data(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
        f.close()
    return data


def main():
    filename = "./out_10Hz.pkl"
    data = load_data(filename)
    visualize_data(data, filename)


if __name__ == "__main__":
    main()
