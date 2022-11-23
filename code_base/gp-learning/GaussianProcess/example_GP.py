import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from cfg import configs
from models.GPModel import GPModel
from utils.data_loading import load_test_data
import os
from pathlib import Path

# * make the paths absolute instead of relative, to avoid headaches
dir_path = Path(os.path.dirname(os.path.abspath(__file__)))


def main(opts):
    config = configs.get_train_config()
    config = {**config, "GP_training_iter": opts.gp_train_iter, "verbose": opts.verbose}
    config["force_train"] = opts.force_train
    model = GPModel(**config)
    model.initialize_model(
        path_model=opts.gpmodel,
        path_train_data=opts.train,
    )
    pred_mean, pred_conf = model.eval(path_test_data=opts.test)
    # X_test is (pos, vel, u)
    X_test, y_test = load_test_data(data_path=opts.test)
    X_test = X_test.numpy()
    real_pos = X_test[:, 0]
    # real_vel = X_test[:, 1]
    # * GP predictions should be compared with labels (y), not inputs (X) because the
    # inputs skip certain processings like normalization and scalings
    real_dx = y_test[:, 0]  # label of dx
    real_dv = y_test[:, 1]
    pos_init = X_test[0, 0]
    pos_pred_mean_delta = pred_mean[:, 0]
    pos_pred_mean = np.ones(len(pos_pred_mean_delta)) * pos_init
    # this is dead reckoning, it's not going to be a useful as an evaluation tool if
    # it's done for long horizons because it is open-loop and model errors will grow.
    for i, delta in enumerate(pos_pred_mean_delta):
        if i > 0:
            pos_pred_mean[i] = pos_pred_mean[i - 1] + delta
    fig, ax = plt.subplots(3, 1)
    ax[0].set_title(f"GP model prediction on {opts.test}")
    ax[0].plot(real_pos, label="real_pos")
    ax[0].plot(pos_pred_mean, label="integrated_pos")
    ax[1].plot(real_dx, label="real_dx")
    ax[1].plot(pred_mean[:, 0], label="pred_dx")
    # ax[1].plot(real_vel, label="real velocity")
    ax[2].plot(real_dv, label="real_dv")
    ax[2].plot(pred_mean[:, 1], label="pred_dv")

    ax[0].legend()
    ax[1].legend()
    ax[2].legend()

    plt.show()


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        type=str,
        default=dir_path / "../data/avant_TrainingData.pkl",
        help="Path to training data",
    )
    parser.add_argument(
        "-gp",
        "--gpmodel",
        type=str,
        default=dir_path / "./results/GPmodel.pkl",
        help="Path to training data",
    )
    parser.add_argument(
        "--test",
        type=str,
        default=dir_path / "../data/avant_TestData.pkl",
        help="Path to test data",
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
    parser.add_argument(
        "--force-train",
        action="store_true",
        help="Force retraining",
    )
    opts = parser.parse_args()
    main(opts)
