import sys
import argparse
import matplotlib.pyplot as plt
from loguru import logger

from cfg import configs
from models.GPModel import GPModel


def main(opts):
    config = configs.get_train_config()
    config = {**config, "GP_training_iter": opts.gp_train_iter, "verbose": opts.verbose}
    model = GPModel(**config)
    model.initialize_model(
        path_model=opts.gpmodel,
        path_train_data=opts.train,
    )
    pred_mean, pred_conf = model.eval(path_test_data=opts.test)
    plt.title(f"GP model prediction on {opts.test}")
    plt.plot(pred_mean[:, 0], label="y1")
    plt.plot(pred_mean[:, 1], label="y2")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        type=str,
        default="../data/avant_TrainingData.pkl",
        help="Path to training data",
    )
    parser.add_argument(
        "-gp",
        "--gpmodel",
        type=str,
        default="./results/GPmodel.pkl",
        help="Path to training data",
    )
    parser.add_argument(
        "--test",
        type=str,
        default="../data/avant_TestData.pkl",
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
    opts = parser.parse_args()
    main(opts)
