import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from cfg import configs
from models.GPModel import GPModel
from utils.data_loading import load_test_data


def main(opts):
    config = configs.get_train_config()
    config = {**config, "GP_training_iter": opts.gp_train_iter, "verbose": opts.verbose}
    model = GPModel(**config)
    model.initialize_model(
        path_model=opts.gpmodel,
        path_train_data=opts.train,
    )
    pred_mean, pred_conf = model.eval(path_test_data=opts.test)
    X_test, y_test = load_test_data(data_path=opts.test)
    X_test = X_test.numpy()
    real_pos = X_test[:, 0]
    real_vel = X_test[:, 1]
    pos_init = X_test[0, 0]
    vel_init = X_test[0, 1]
    pos_pred_mean_delta = pred_mean[:, 0]
    vel_pred_mean_delta = pred_mean[:, 1]
    pos_pred_mean = np.ones(len(pos_pred_mean_delta)) * pos_init
    vel_pred_mean = np.ones(len(pos_pred_mean_delta)) * vel_init
    for i, delta in enumerate(pos_pred_mean_delta):
        if i > 0:
            pos_pred_mean[i] = pos_pred_mean[i - 1] + delta
    for i, delta in enumerate(vel_pred_mean_delta):
        if i > 0:
            vel_pred_mean[i] = vel_pred_mean[i - 1] + delta
    plt.title(f"GP model prediction on {opts.test}")
    plt.plot(real_pos, label="real_pos")
    plt.plot(pos_pred_mean, label="pred_pos")
    plt.plot(real_vel, label="real_vel")
    plt.plot(vel_pred_mean, label="pred_vel")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--train",
        type=str,
        default="../data/avant_TrainingData.pkl",
        help="Path to training data",
    )
    parser.add_argument(
        "-T",
        "--test",
        type=str,
        default="../data/avant_TestData.pkl",
        help="Path to test data",
    )
    parser.add_argument(
        "-gp",
        "--gpmodel",
        type=str,
        default="./results/GPmodel.pkl",
        help="Path to training data",
    )
    parser.add_argument(
        "--gp-train-iter",
        type=int,
        default=500,
        help="Maximum train iterations for GP model",
    )
    parser.add_argument(
        "-v,",
        "--verbose",
        type=int,
        default=1,
        help="Path to test data",
    )
    opts = parser.parse_args()
    main(opts)
